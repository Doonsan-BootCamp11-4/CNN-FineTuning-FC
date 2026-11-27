"""Standard PyTorch training loop for frozen-backbone FC transfer learning."""

import json
from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from fcft.accel import (
    CompressedFeatureBank,
    FusedLinearHeadSession,
    ResidualAdapterHead,
    auto_tune_resilient as _core_auto_tune_resilient,
    fit_balanced_ridge_head,
    fit_closed_form_head,
    fit_dual_anchor_prototypes,
    is_accel_available,
    multihead_linear_forward,
)
from fcft.data import build_loaders
from fcft.losses import UnifiedClassificationLoss, apply_cutmix, compute_class_weights, compute_log_priors
from fcft.model import (
    CosineClassifierHead,
    DualAnchorHybridHead,
    FeatureCenterNormLayer,
    FeatureNormLayer,
    MLPClassifierHead,
    SpectralNormLayer,
    SubCenterHead,
    build_head,
    build_model,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _freeze_bn_stats(model: nn.Module) -> None:
    """Keep BatchNorm layers in frozen backbone locked in eval mode."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters())
            if not params or not any(p.requires_grad for p in params):
                module.eval()


class ModelEMA:
    """Exponential moving average tracker over trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = {}


class ModelSWA:
    """Stochastic Weight Averaging accumulator over late-epoch trainable weights."""

    def __init__(self, model: nn.Module):
        self.acc = {n: torch.zeros_like(p.detach()) for n, p in model.named_parameters() if p.requires_grad}
        self.count = 0
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.acc:
                self.acc[n].add_(p.detach())
        self.count += 1

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        if self.count <= 0:
            return
        self._backup = {}
        inv = 1.0 / float(self.count)
        for n, p in model.named_parameters():
            if n in self.acc:
                self._backup[n] = p.detach().clone()
                p.copy_(self.acc[n] * inv)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = {}


def _step_batch_loss(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: UnifiedClassificationLoss,
    num_classes: int,
    mixup_alpha: float,
    label_smoothing: float,
    cutmix_alpha: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    _ = num_classes, label_smoothing
    return criterion.forward_batch(forward_fn, x, y, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha)


def _train(
    model: nn.Module,
    loader,
    criterion: UnifiedClassificationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 12,
    mixup_alpha: float = 0.0,
    label_smoothing: float = 0.0,
    ema: Optional[ModelEMA] = None,
    cutmix_alpha: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    criterion.train()
    _freeze_bn_stats(model)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, batch_ok = _step_batch_loss(model, x, y, criterion, num_classes, mixup_alpha, label_smoothing, cutmix_alpha=cutmix_alpha)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        correct += batch_ok
        total += bsz
    denom = max(1, total)
    return total_loss / denom, correct / denom


def _is_better_val(score: float, val_loss: float, best_score: float, best_loss: float) -> bool:
    """Compare checkpoints by primary metric with lower val_loss tie-breaking."""
    return (score > best_score + 1e-12) or (abs(score - best_score) <= 1e-12 and val_loss < best_loss)


def _macro_f1_from_cm(cm: torch.Tensor) -> float:
    """Compute unweighted Macro F1 from an integer (C, C) confusion matrix."""
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    denom = (2.0 * tp + fp + fn).clamp(min=1e-12)
    f1 = torch.where((2.0 * tp + fp + fn) > 0.0, (2.0 * tp) / denom, torch.zeros_like(tp))
    return float(f1.mean().item())


def _eval(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bsz = y.size(0)
            total_loss += float(loss.item()) * bsz
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += bsz
    denom = max(1, total)
    return total_loss / denom, correct / denom


def run_once_epoch(
    model: nn.Module,
    loaders: Dict[str, object],
    criterion: UnifiedClassificationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 12,
    mixup_alpha: float = 0.0,
    label_smoothing: float = 0.0,
    ema: Optional[ModelEMA] = None,
    cutmix_alpha: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Run one train epoch followed by one validation epoch (using EMA if enabled)."""
    train_loss, train_acc = _train(model, loaders["train"], criterion, optimizer, device, num_classes, mixup_alpha, label_smoothing, ema, cutmix_alpha=cutmix_alpha)
    if ema is not None:
        ema.apply_to(model)
    val_loss, val_acc = _eval(model, loaders["val"], criterion, device)
    if ema is not None:
        ema.restore(model)
    return train_loss, train_acc, val_loss, val_acc


def _extract_backbone_features(model: nn.Module, loader, device: torch.device, tta: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_fc = model.fc
    model.fc = nn.Identity()
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            xd = x.to(device)
            feat = model(xd)
            if tta:
                feat = 0.5 * (feat + model(torch.flip(xd, dims=[3])))
            xs.append(feat.cpu())
            ys.append(y.cpu())
    model.fc = orig_fc
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _has_center_norm(head_mod: nn.Module) -> bool:
    return isinstance(head_mod, FeatureCenterNormLayer) or (
        isinstance(head_mod, nn.Sequential) and any(isinstance(sub, FeatureCenterNormLayer) for sub in head_mod)
    )


def _fit_head_center_stats(head_mod: nn.Module, features: torch.Tensor) -> None:
    """Fit FeatureCenterNormLayer train-only mean vector before forward or ridge warm start."""
    if isinstance(head_mod, FeatureCenterNormLayer):
        head_mod.fit_stats(features)
        return
    if isinstance(head_mod, nn.Sequential):
        for sub in head_mod:
            if isinstance(sub, FeatureCenterNormLayer):
                sub.fit_stats(features)


@torch.no_grad()
def _calibrate_ridge_temperature(linear: nn.Linear, features: torch.Tensor, targets: torch.Tensor) -> None:
    """Scale closed-form MSE ridge weights uniformly for CrossEntropy temperature without changing argmax."""
    if features.numel() == 0 or linear.weight.numel() == 0:
        return
    f_c = features.detach().to(device="cpu", dtype=torch.float32)
    y_c = targets.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    w_c = linear.weight.detach().to(device="cpu", dtype=torch.float32)
    b_c = linear.bias.detach().to(device="cpu", dtype=torch.float32) if linear.bias is not None else None
    logits = F.linear(f_c, w_c, b_c)
    best_s = 1.0
    best_loss = float(F.cross_entropy(logits, y_c).item())
    for cand in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        ce = float(F.cross_entropy(logits * cand, y_c).item())
        if ce < best_loss:
            best_loss = ce
            best_s = cand
    if best_s != 1.0:
        linear.weight.mul_(best_s)
        if linear.bias is not None:
            linear.bias.mul_(best_s)


@torch.no_grad()
def _calibrate_cosine_scale(
    cos_logits: torch.Tensor,
    targets: torch.Tensor,
    scale_param: nn.Parameter,
) -> None:
    """Tune cosine classifier temperature across candidate scales using CrossEntropy."""
    if cos_logits.numel() == 0:
        return
    y_c = targets.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    u_c = cos_logits.detach().to(device="cpu", dtype=torch.float32)
    cur_s = float(scale_param.detach().item())
    best_s = cur_s
    best_ce = float(F.cross_entropy(u_c * cur_s, y_c).item())
    for cand in (8.0, 12.0, 16.0, 24.0, 32.0):
        ce = float(F.cross_entropy(u_c * cand, y_c).item())
        if ce < best_ce:
            best_ce, best_s = ce, cand
    scale_param.fill_(best_s)


def _unwrap_head_core(head_mod: nn.Module) -> nn.Module:
    if isinstance(head_mod, nn.Sequential):
        for sub in head_mod:
            if not isinstance(sub, (FeatureNormLayer, FeatureCenterNormLayer, SpectralNormLayer, nn.Dropout)):
                return sub
    return head_mod


def _apply_head_lr_scale(opt: torch.optim.Optimizer, head_mod: nn.Module, head_lr_scale: float) -> None:
    """Scale optimizer LR on 1-shot initialized linear/anchor head parameters relative to adapter parameters."""
    scale = float(head_lr_scale)
    if scale < 0.0:
        raise ValueError(f"head_lr_scale must be non-negative, got {head_lr_scale!r}")
    if scale == 1.0:
        return
    core = _unwrap_head_core(head_mod)
    if isinstance(core, ResidualAdapterHead):
        head_ids = {id(p) for p in core.fc.parameters() if p.requires_grad}
    elif isinstance(core, DualAnchorHybridHead):
        head_ids = {id(core.anchors)} if core.anchors.requires_grad else set()
    elif isinstance(core, (CosineClassifierHead, SubCenterHead)):
        head_ids = {id(core.weight)} if core.weight.requires_grad else set()
    else:
        return
    if not head_ids:
        return
    adapter_groups, head_groups = [], []
    for grp in opt.param_groups:
        ad_p = [p for p in grp["params"] if id(p) not in head_ids]
        hd_p = [p for p in grp["params"] if id(p) in head_ids]
        if ad_p:
            adapter_groups.append({**grp, "params": ad_p})
        if hd_p:
            head_groups.append({**grp, "params": hd_p, "lr": float(grp["lr"]) * scale})
    if adapter_groups and head_groups:
        opt.param_groups = adapter_groups + head_groups


def _warm_start_head(head_mod: nn.Module, features: torch.Tensor, targets: torch.Tensor, ridge: float) -> None:
    if not is_accel_available():
        return
    f_in, target_mod = features, head_mod
    if isinstance(target_mod, nn.Sequential):
        for idx, sub in enumerate(target_mod):
            if isinstance(sub, (FeatureNormLayer, FeatureCenterNormLayer, SpectralNormLayer)):
                with torch.no_grad():
                    f_in = sub(f_in)
            elif isinstance(sub, nn.Dropout):
                continue
            else:
                target_mod = sub if idx == len(target_mod) - 1 else target_mod[idx]
                break
    if isinstance(target_mod, DualAnchorHybridHead):
        anchors = fit_dual_anchor_prototypes(f_in, targets, int(target_mod.anchors.shape[0]), harm_beta=0.25)
        with torch.no_grad():
            target_mod.anchors.copy_(anchors.to(target_mod.anchors))
        return
    if isinstance(target_mod, CosineClassifierHead):
        tmp = nn.Linear(target_mod.weight.shape[1], target_mod.weight.shape[0])
        fit_closed_form_head(tmp, f_in, targets, ridge=float(ridge))
        with torch.no_grad():
            w_n = F.normalize(tmp.weight.to(target_mod.weight), p=2, dim=-1)
            target_mod.weight.copy_(w_n)
            x_n = F.normalize(f_in.to(target_mod.weight), p=2, dim=-1)
            _calibrate_cosine_scale(F.linear(x_n, w_n), targets, target_mod.scale)
        return
    if isinstance(target_mod, SubCenterHead):
        n_cls, k_c, d_in = target_mod.num_classes, target_mod.num_centers, int(target_mod.weight.shape[1])
        tmp = nn.Linear(d_in, n_cls)
        fit_closed_form_head(tmp, f_in, targets, ridge=float(ridge))
        with torch.no_grad():
            base_w = F.normalize(tmp.weight.to(target_mod.weight), p=2, dim=-1)
            cur_view = target_mod.weight.view(n_cls, k_c, d_in)
            cur_view[:, 0, :].copy_(base_w)
            for k_i in range(1, k_c):
                cur_view[:, k_i, :].copy_(F.normalize(base_w + 0.05 * cur_view[:, k_i, :], p=2, dim=-1))
            cos_all = (F.linear(F.normalize(f_in.to(target_mod.weight), p=2, dim=-1), F.normalize(target_mod.weight, p=2, dim=-1)))
            _calibrate_cosine_scale(cos_all.view(f_in.shape[0], n_cls, k_c).max(dim=-1).values, targets, target_mod.scale)
        return
    if isinstance(target_mod, ResidualAdapterHead):
        with torch.no_grad():
            f_in = target_mod.encode(f_in.to(target_mod.fc.weight.device)).cpu()
        target_mod = target_mod.fc
    linear = target_mod if isinstance(target_mod, nn.Linear) else getattr(target_mod, "fc", None)
    if isinstance(linear, nn.Linear):
        fit_balanced_ridge_head(linear, f_in, targets, ridge=float(ridge))
        _calibrate_ridge_temperature(linear, f_in, targets)


def _extract_warm_start_payload(head_mod: nn.Module) -> Optional[dict]:
    core = _unwrap_head_core(head_mod)
    if isinstance(core, DualAnchorHybridHead):
        return {"kind": "dual_anchor", "anchors": core.anchors.detach().cpu().clone()}
    if isinstance(core, (CosineClassifierHead, SubCenterHead)):
        return {
            "kind": "cosine" if isinstance(core, CosineClassifierHead) else "subcenter",
            "weight": core.weight.detach().cpu().clone(),
            "scale": core.scale.detach().cpu().clone(),
        }
    linear = core if isinstance(core, nn.Linear) else (core.fc if isinstance(core, ResidualAdapterHead) else None)
    if isinstance(linear, nn.Linear):
        return {
            "kind": "linear",
            "weight": linear.weight.detach().cpu().clone(),
            "bias": linear.bias.detach().cpu().clone() if linear.bias is not None else None,
        }
    return None


def _apply_warm_start_payload(head_mod: nn.Module, payload: Optional[dict]) -> bool:
    if not payload:
        return False
    core = _unwrap_head_core(head_mod)
    kind = payload.get("kind")
    with torch.no_grad():
        if kind == "dual_anchor" and isinstance(core, DualAnchorHybridHead):
            core.anchors.copy_(payload["anchors"].to(core.anchors))
            return True
        if kind in ("cosine", "subcenter") and isinstance(core, (CosineClassifierHead, SubCenterHead)):
            core.weight.copy_(payload["weight"].to(core.weight))
            if payload.get("scale") is not None:
                core.scale.copy_(payload["scale"].to(core.scale))
            return True
        linear = core if isinstance(core, nn.Linear) else (core.fc if isinstance(core, ResidualAdapterHead) else None)
        if kind == "linear" and isinstance(linear, nn.Linear):
            linear.weight.copy_(payload["weight"].to(linear.weight))
            if linear.bias is not None and payload.get("bias") is not None:
                linear.bias.copy_(payload["bias"].to(linear.bias))
            return True
    return False


def compute_warm_start_state(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    num_classes: int,
    head_type: str = "linear",
    feature_norm: bool = False,
    feature_norm_mode: Optional[str] = None,
    feature_dropout: float = 0.0,
    head_kwargs: Optional[dict] = None,
    warm_start_ridge: float = 0.5,
) -> Optional[dict]:
    """Compute a reusable 1-shot head warm-start payload for identical cached pipelines."""
    if not is_accel_available():
        return None
    h_mod = build_head(
        int(x_tr.shape[1]), int(num_classes), head_type=head_type, feature_norm=feature_norm,
        feature_norm_mode=feature_norm_mode, feature_dropout=feature_dropout, **dict(head_kwargs or {}),
    )
    _fit_head_center_stats(h_mod, x_tr)
    _warm_start_head(h_mod, x_tr, y_tr, float(warm_start_ridge))
    return _extract_warm_start_payload(h_mod)


def _resolve_criterion(
    targets: Optional[torch.Tensor],
    num_classes: int,
    label_smoothing: float,
    focal_gamma: float,
    class_weight_mode: Optional[str],
    device: torch.device,
    logit_adj_tau: float = 0.0,
    logit_margin: float = 0.0,
    loss_cap_tau: float = 0.0,
    loss_cap_alpha: float = 0.0,
    loss_cap_mode: str = "fixed",
    loss_cap_param: float = 1.75,
) -> UnifiedClassificationLoss:
    cw = None
    if class_weight_mode is not None and targets is not None:
        cw = compute_class_weights(targets, num_classes, mode=class_weight_mode)
    lp = None
    if targets is not None and float(logit_adj_tau) != 0.0:
        lp = compute_log_priors(targets, num_classes)
    return UnifiedClassificationLoss(
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        class_weights=cw,
        logit_adj_tau=float(logit_adj_tau),
        log_priors=lp,
        logit_margin=float(logit_margin),
        loss_cap_tau=float(loss_cap_tau),
        loss_cap_alpha=float(loss_cap_alpha),
        loss_cap_mode=str(loss_cap_mode),
        loss_cap_param=float(loss_cap_param),
    ).to(device)


def _prepare_descriptor_features(model: nn.Module, loader, device: torch.device, return_assembled: bool = False):
    """Fit descriptor channel statistics once from train loader and optionally return assembled features."""
    bbs, descs, ys = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            bb_f, d_f = model.extract_raw_parts(x.to(device))
            if bb_f is not None:
                bbs.append(bb_f.cpu())
            descs.append(d_f.cpu())
            ys.append(y.cpu())
    desc_all = torch.cat(descs, dim=0)
    model.standardizer.fit_stats(desc_all)
    if not return_assembled:
        return None, None
    bb_all = torch.cat(bbs, dim=0).to(device) if bbs else None
    feats = model.assemble_features(bb_all, desc_all.to(device)).cpu()
    return feats, torch.cat(ys, dim=0)


def _save_checkpoint(
    best_path: Path,
    model: nn.Module,
    config: dict,
    classes: List[str],
    best_val_acc: float,
    epoch: int,
    head_type: str,
    feature_norm: bool,
    head_kwargs: dict,
    pool_mode: str = "avg",
    feature_mode: str = "backbone",
    desc_weight: float = 1.0,
    feature_norm_mode: Optional[str] = None,
    feature_dropout: float = 0.0,
) -> None:
    safe_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": safe_cfg,
            "classes": list(classes),
            "best_val_acc": best_val_acc,
            "epoch": epoch,
            "head_type": head_type,
            "feature_norm": bool(feature_norm),
            "feature_norm_mode": feature_norm_mode,
            "feature_dropout": float(feature_dropout),
            "pool_mode": pool_mode,
            "feature_mode": feature_mode,
            "desc_weight": float(desc_weight),
            "head_kwargs": dict(head_kwargs),
        },
        best_path,
    )


def _step_scheduler(sched, val_loss: float) -> None:
    if sched is None:
        return
    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        sched.step(val_loss)
    else:
        sched.step()


def _maybe_promote_swa(
    swa: Optional[ModelSWA],
    eval_fn: Callable[[], float],
    model: nn.Module,
    best_val_acc: float,
    best_epoch: int,
    tot_ep: int,
    save_fn: Callable[[float, int], None],
) -> Tuple[float, int]:
    if swa is None or swa.count <= 0:
        return best_val_acc, best_epoch
    swa.apply_to(model)
    swa_acc = float(eval_fn())
    if swa_acc > best_val_acc:
        best_val_acc, best_epoch = swa_acc, tot_ep
        save_fn(best_val_acc, tot_ep)
    swa.restore(model)
    return best_val_acc, best_epoch


def _init_training_head(model: nn.Module, train_loader, device: torch.device, warm_start_ridge: Optional[float]) -> None:
    need_center = _has_center_norm(model.fc)
    need_feats = (warm_start_ridge is not None) or need_center
    if hasattr(model, "extract_raw_parts"):
        feats, ys = _prepare_descriptor_features(model, train_loader, device, return_assembled=need_feats)
        if feats is not None:
            _fit_head_center_stats(model.fc, feats)
            if warm_start_ridge is not None:
                _warm_start_head(model.fc, feats, ys, float(warm_start_ridge))
    elif need_feats:
        feats, ys = _extract_backbone_features(model, train_loader, device)
        _fit_head_center_stats(model.fc, feats)
        if warm_start_ridge is not None:
            _warm_start_head(model.fc, feats, ys, float(warm_start_ridge))


def _run_training_loop(
    model: nn.Module, loaders: dict, criterion, opt: torch.optim.Optimizer, sched,
    device: torch.device, config: dict, ema: Optional[ModelEMA], swa: Optional[ModelSWA],
    best_path: Path, history_path: Path, meta: tuple,
) -> Tuple[List[dict], float, int]:
    num_cls, tot_ep, mixup_a, cutmix_a, smooth, head_t, fn, h_kw, pool_m, feat_m, desc_w, fn_m, f_drop, swa_ep = meta
    classes, save_en = loaders["train"].dataset.classes, bool(config.get("save_ckpt", True))
    history, best_val_acc, best_val_loss, best_epoch = [], -1.0, float("inf"), 0
    try:
        for epoch in range(1, tot_ep + 1):
            epoch_lr = float(opt.param_groups[0]["lr"])
            tr_l, tr_a, va_l, va_a = run_once_epoch(model, loaders, criterion, opt, device, num_cls, mixup_a, smooth, ema, cutmix_alpha=cutmix_a)
            _step_scheduler(sched, va_l)
            if swa is not None and epoch > max(0, tot_ep - swa_ep):
                swa.update(model)
            if _is_better_val(va_a, va_l, best_val_acc, best_val_loss):
                best_val_acc, best_val_loss, best_epoch = va_a, va_l, epoch
                if save_en:
                    if ema is not None:
                        ema.apply_to(model)
                    _save_checkpoint(best_path, model, config, classes, best_val_acc, epoch, head_t, fn, h_kw, pool_m, feat_m, desc_w, fn_m, f_drop)
                    if ema is not None:
                        ema.restore(model)
            history.append({"epoch": epoch, "lr": epoch_lr, "train_loss": tr_l, "train_acc": tr_a, "val_loss": va_l, "val_acc": va_a})
    finally:
        if history:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_val_acc, best_epoch = _maybe_promote_swa(
        swa, lambda: _eval(model, loaders["val"], criterion, device)[1],
        model, best_val_acc, best_epoch, tot_ep,
        lambda acc, ep: _save_checkpoint(best_path, model, config, classes, acc, ep, head_t, fn, h_kw, pool_m, feat_m, desc_w, fn_m, f_drop) if save_en else None,
    )
    return history, best_val_acc, best_epoch


def run(
    config: dict, optimizer: Optional[Callable] = None, scheduler: Optional[Callable] = None,
    train_transform=None, head_type: str = "linear", feature_norm: bool = False,
    pool_mode: str = "avg", head_kwargs: Optional[dict] = None, label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0, ema_decay: float = 0.0, warm_start_ridge: Optional[float] = None,
    class_weight_mode: Optional[str] = None, focal_gamma: float = 0.0, feature_mode: str = "backbone",
    desc_weight: float = 1.0, cutmix_alpha: float = 0.0, feature_norm_mode: Optional[str] = None,
    feature_dropout: float = 0.0, logit_adj_tau: float = 0.0, logit_margin: float = 0.0, swa_epochs: int = 0,
    head_lr_scale: float = 1.0, loss_cap_tau: float = 0.0, loss_cap_alpha: float = 0.0,
    loss_cap_mode: str = "fixed", loss_cap_param: float = 1.75,
) -> dict:
    """Train FC head on frozen backbone/descriptors, saving best.pt and epoch-wise history.json."""
    _set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config["out_dir"]) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, history_path = run_dir / "best.pt", run_dir / "history.json"
    h_kw, num_cls, tot_ep = dict(head_kwargs or {}), int(config["num_classes"]), int(config.get("epochs", 15))
    loaders = build_loaders(config, train_transform=train_transform)
    model = build_model(
        config.get("model", "resnet18"), num_cls, head_type=head_type, feature_norm=feature_norm,
        pool_mode=pool_mode, feature_mode=feature_mode, desc_weight=desc_weight,
        feature_norm_mode=feature_norm_mode, feature_dropout=feature_dropout, **h_kw,
    ).to(device)
    _init_training_head(model, loaders["train"], device, warm_start_ridge)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optimizer(params) if optimizer is not None else torch.optim.Adam(params, lr=1e-3)
    _apply_head_lr_scale(opt, model.fc, head_lr_scale)
    all_targets = torch.tensor([t for _, t in loaders["train"].dataset.samples], dtype=torch.long)
    criterion = _resolve_criterion(all_targets, num_cls, label_smoothing, focal_gamma, class_weight_mode, device, logit_adj_tau=logit_adj_tau, logit_margin=logit_margin, loss_cap_tau=loss_cap_tau, loss_cap_alpha=loss_cap_alpha, loss_cap_mode=loss_cap_mode, loss_cap_param=loss_cap_param)
    meta = (num_cls, tot_ep, mixup_alpha, cutmix_alpha, label_smoothing, head_type, feature_norm, h_kw, pool_mode, feature_mode, desc_weight, feature_norm_mode, feature_dropout, int(swa_epochs))
    history, best_val_acc, best_epoch = _run_training_loop(
        model, loaders, criterion, opt, scheduler(opt) if scheduler is not None else None,
        device, config, ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None,
        ModelSWA(model) if int(swa_epochs) > 0 else None, best_path, history_path, meta,
    )
    return {"name": config["name"], "best_val_acc": best_val_acc, "best_epoch": best_epoch, "final_lr": float(opt.param_groups[0]["lr"]), "history": history, "best_ckpt": str(best_path), "history_path": str(history_path)}


def _train_cached_epoch(
    head: nn.Module, n_train: int, batch_size: int, device: torch.device, criterion: UnifiedClassificationLoss,
    optimizer: torch.optim.Optimizer, num_classes: int, mixup_alpha: float, label_smoothing: float,
    ema: Optional[ModelEMA], model: nn.Module, bank: Optional[CompressedFeatureBank],
    x_tr: torch.Tensor, y_tr: torch.Tensor, cutmix_alpha: float = 0.0, feature_noise_std: float = 0.0,
) -> Tuple[float, float]:
    head.train()
    criterion.train()
    perm = torch.randperm(n_train, device=(x_tr.device if bank is None else torch.device("cpu")))
    tot_loss, tot_ok = 0.0, 0
    for start in range(0, n_train, batch_size):
        idx = perm[start:start + batch_size]
        bx = bank.get_tensor_batch(idx, device=device, dtype=torch.float32) if bank is not None else x_tr[idx].to(device)
        if feature_noise_std > 0.0:
            bx = bx + torch.randn_like(bx) * float(feature_noise_std)
        by = y_tr[idx].to(device)
        optimizer.zero_grad()
        loss, batch_ok = _step_batch_loss(head, bx, by, criterion, num_classes, mixup_alpha, label_smoothing, cutmix_alpha=cutmix_alpha)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        tot_loss += float(loss.item()) * by.size(0)
        tot_ok += batch_ok
    denom = max(1, n_train)
    return tot_loss / denom, tot_ok / denom


@torch.no_grad()
def _eval_cached(
    head: nn.Module, x_va: torch.Tensor, y_va: torch.Tensor, batch_size: int, device: torch.device, criterion: nn.Module,
    num_classes: Optional[int] = None, return_f1: bool = False,
):
    head.eval()
    criterion.eval()
    n_val = int(x_va.shape[0])
    tot_loss, tot_ok = 0.0, 0
    cm = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long) if (return_f1 and num_classes) else None
    for start in range(0, n_val, batch_size):
        bx = x_va[start:start + batch_size].to(device)
        by = y_va[start:start + batch_size].to(device)
        logits = head(bx)
        loss = criterion(logits, by)
        preds = logits.argmax(dim=1)
        tot_loss += float(loss.item()) * by.size(0)
        tot_ok += int((preds == by).sum().item())
        if cm is not None:
            idx_flat = by.cpu() * int(num_classes) + preds.cpu()
            cm.view(-1).index_add_(0, idx_flat, torch.ones_like(idx_flat))
    denom = max(1, n_val)
    if cm is not None:
        return tot_loss / denom, tot_ok / denom, _macro_f1_from_cm(cm)
    return tot_loss / denom, tot_ok / denom


_AUTO_COMPRESS_BYTES = 256 * 1024 * 1024


def _build_feature_bank(x_tr: torch.Tensor, compress_cache) -> Tuple[Optional[CompressedFeatureBank], int]:
    raw_bytes = int(x_tr.numel() * x_tr.element_size())
    if not compress_cache or not is_accel_available():
        return None, raw_bytes
    q_mode = str(compress_cache) if isinstance(compress_cache, str) else "dyadic"
    if q_mode == "none":
        return None, raw_bytes
    if q_mode == "auto":
        if raw_bytes <= _AUTO_COMPRESS_BYTES:
            return None, raw_bytes
        q_mode = "dyadic"
    bank = CompressedFeatureBank(max_entries=max(1, int(x_tr.shape[0])), quant_mode=q_mode)
    bank.put_batch(list(range(int(x_tr.shape[0]))), x_tr)
    return bank, bank.memory_bytes()


def _extract_cached_train(model: nn.Module, loader, device: torch.device, tta: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model, "extract_raw_parts"):
        return _extract_backbone_features(model, loader, device, tta=tta)
    bbs, descs, ys = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            xd = x.to(device)
            bb_f, d_f = model.extract_raw_parts(xd)
            if bb_f is not None:
                bbs.append(bb_f.cpu())
            descs.append(d_f.cpu())
            ys.append(y.cpu())
            if tta:
                bb_flip, d_flip = model.extract_raw_parts(torch.flip(xd, dims=[3]))
                if bb_flip is not None:
                    bbs.append(bb_flip.cpu())
                descs.append(d_flip.cpu())
                ys.append(y.cpu())
    desc_all = torch.cat(descs, dim=0)
    model.standardizer.fit_stats(desc_all)
    bb_all = torch.cat(bbs, dim=0).to(device) if bbs else None
    feats = model.assemble_features(bb_all, desc_all.to(device)).cpu()
    return feats, torch.cat(ys, dim=0)


def _resolve_cached_features(
    model: nn.Module,
    config: dict,
    train_transform,
    device: torch.device,
    num_cls: int,
    tta_extract: bool,
    precomputed_features,
    precomputed_classes: Optional[List[str]],
    precomputed_standardizer_state: Optional[dict] = None,
):
    if precomputed_standardizer_state is not None and hasattr(model, "standardizer"):
        model.standardizer.load_state_dict(precomputed_standardizer_state)
    if precomputed_features is not None:
        x_tr, y_tr, x_va, y_va = precomputed_features
        cls_names = list(precomputed_classes) if precomputed_classes is not None else [str(i) for i in range(num_cls)]
        return x_tr, y_tr, x_va, y_va, cls_names
    loaders = build_loaders(config, train_transform=train_transform)
    cls_names = list(loaders["train"].dataset.classes)
    x_tr, y_tr = _extract_cached_train(model, loaders["train"], device, tta=tta_extract)
    val_tta = False if hasattr(model, "extract_raw_parts") else tta_extract
    x_va, y_va = _extract_backbone_features(model, loaders["val"], device, tta=val_tta)
    return x_tr, y_tr, x_va, y_va, cls_names


def _split_norm_eval_head(head_mod: nn.Module) -> Tuple[Optional[nn.Module], nn.Module]:
    if isinstance(head_mod, nn.Sequential) and len(head_mod) >= 2 and isinstance(
        head_mod[0], (FeatureNormLayer, FeatureCenterNormLayer, SpectralNormLayer)
    ):
        children = list(head_mod.children())
        rest = children[1] if len(children) == 2 else nn.Sequential(*children[1:])
        return children[0], rest
    return None, head_mod


def _can_use_fused_linear_session(
    train_head: nn.Module,
    compute_dev: torch.device,
    bank: Optional[CompressedFeatureBank],
    opt: torch.optim.Optimizer,
    criterion,
    mixup_a: float,
    cutmix_a: float,
    ema: Optional[ModelEMA],
    swa: Optional[ModelSWA],
) -> bool:
    if compute_dev.type != "cpu" or not is_accel_available() or bank is not None:
        return False
    if not isinstance(train_head, nn.Linear) or train_head.bias is None:
        return False
    if mixup_a != 0.0 or cutmix_a != 0.0 or ema is not None or swa is not None:
        return False
    if not isinstance(opt, (torch.optim.AdamW, torch.optim.Adam)):
        return False
    for grp in opt.param_groups:
        if tuple(grp.get("betas", (0.9, 0.999))) != (0.9, 0.999) or float(grp.get("eps", 1e-8)) != 1e-8:
            return False
        if bool(grp.get("amsgrad", False)) or bool(grp.get("maximize", False)):
            return False
    if len(opt.param_groups) == 1:
        if float(opt.param_groups[0].get("weight_decay", 0.0)) != 0.0:
            return False
    elif len(opt.param_groups) == 2 and isinstance(opt, torch.optim.AdamW):
        g0, g1 = opt.param_groups[0], opt.param_groups[1]
        if len(g0["params"]) != 1 or len(g1["params"]) != 1:
            return False
        if g0["params"][0] is not train_head.weight or g1["params"][0] is not train_head.bias:
            return False
        if float(g1.get("weight_decay", 0.0)) != 0.0 or abs(float(g0["lr"]) - float(g1["lr"])) > 1e-12:
            return False
    else:
        return False
    if not isinstance(criterion, UnifiedClassificationLoss):
        return False
    if criterion.focal_gamma != 0.0 or criterion.logit_adj_tau != 0.0 or criterion.logit_margin != 0.0:
        return False
    return criterion.cap_gate is None or criterion.loss_cap_mode == "fixed"


def _run_cached_loop(
    model: nn.Module, config: dict, opt: torch.optim.Optimizer, sched, criterion,
    ema: Optional[ModelEMA], swa: Optional[ModelSWA], bank: Optional[CompressedFeatureBank],
    x_tr: torch.Tensor, y_tr: torch.Tensor, x_va: torch.Tensor, y_va: torch.Tensor,
    cls_names: List[str], best_path: Path, history_path: Path, meta: tuple,
) -> Tuple[List[dict], float, int]:
    bsz, tot_ep, num_cls, mixup_a, cutmix_a, smooth, head_t, fn, h_kw, pool_m, feat_m, desc_w, fn_m, f_drop = meta
    save_en = bool(config.get("save_ckpt", True))
    compute_dev = opt.param_groups[0]["params"][0].device
    norm_mod, eval_head = _split_norm_eval_head(model.fc)
    if norm_mod is not None:
        with torch.no_grad():
            x_va_eval = norm_mod(x_va.to(compute_dev))
    else:
        x_va_eval = x_va.to(compute_dev) if (compute_dev.type == "cuda" and x_va.numel() * 4 <= _AUTO_COMPRESS_BYTES) else x_va
    y_va_eval = y_va.to(compute_dev) if compute_dev.type == "cuda" else y_va

    train_head = model.fc
    x_tr_use, y_tr_use = x_tr, y_tr
    if bank is None:
        if norm_mod is not None and mixup_a == 0.0 and cutmix_a == 0.0 and f_drop == 0.0 and isinstance(model.fc, nn.Sequential) and len(model.fc) == 2:
            with torch.no_grad():
                x_tr_use = norm_mod(x_tr.to(compute_dev))
            train_head = eval_head
        elif compute_dev.type == "cuda" and x_tr.numel() * 4 <= _AUTO_COMPRESS_BYTES:
            x_tr_use = x_tr.to(compute_dev)
        y_tr_use = y_tr.to(compute_dev) if compute_dev.type == "cuda" else y_tr

    fused_sess = None
    if _can_use_fused_linear_session(train_head, compute_dev, bank, opt, criterion, mixup_a, cutmix_a, ema, swa):
        fused_sess = FusedLinearHeadSession(
            int(train_head.in_features),
            int(train_head.out_features),
            max_batch_size=bsz,
            class_weights=criterion.class_weights,
        )
        fused_sess.sync_from(train_head)

    use_f1 = str(config.get("ckpt_metric_eff", "val_acc")).lower() == "macro_f1"
    f_noise = float(config.get("feature_noise_std_eff", 0.0))
    history, best_score, best_val_acc, best_val_loss, best_epoch = [], -1.0, -1.0, float("inf"), 0
    try:
        for epoch in range(1, tot_ep + 1):
            epoch_lr = float(opt.param_groups[0]["lr"])
            if fused_sess is not None:
                tr_l, tr_a = fused_sess.run_epoch(
                    x_tr_use, y_tr_use, bsz,
                    lr=epoch_lr,
                    weight_decay=(float(opt.param_groups[0].get("weight_decay", 0.0)) if isinstance(opt, torch.optim.AdamW) else 0.0),
                    label_smoothing=criterion.label_smoothing,
                    tau=(float(criterion.loss_cap_tau) if criterion.cap_gate is not None else 0.0),
                    alpha=(float(criterion.loss_cap_alpha) if criterion.cap_gate is not None else 0.0),
                )
                fused_sess.sync_to(train_head)
            else:
                tr_l, tr_a = _train_cached_epoch(
                    train_head, int(x_tr_use.shape[0]), bsz, compute_dev, criterion, opt, num_cls, mixup_a, smooth,
                    ema, model, bank, x_tr_use, y_tr_use, cutmix_alpha=cutmix_a, feature_noise_std=f_noise,
                )
            if swa is not None and epoch > max(0, tot_ep - int(config.get("swa_epochs_eff", 0))):
                swa.update(model)
            if ema is not None:
                ema.apply_to(model)
            if use_f1:
                va_l, va_a, va_f1 = _eval_cached(eval_head, x_va_eval, y_va_eval, bsz, compute_dev, criterion, num_classes=num_cls, return_f1=True)
                cur_score = va_f1
            else:
                va_l, va_a = _eval_cached(eval_head, x_va_eval, y_va_eval, bsz, compute_dev, criterion)
                cur_score = va_a
            _step_scheduler(sched, va_l)
            if va_a > best_val_acc:
                best_val_acc = va_a
            if _is_better_val(cur_score, va_l, best_score, best_val_loss):
                best_score, best_val_loss, best_epoch = cur_score, va_l, epoch
                if save_en:
                    _save_checkpoint(best_path, model, config, cls_names, va_a, epoch, head_t, fn, h_kw, pool_m, feat_m, desc_w, fn_m, f_drop)
            if ema is not None:
                ema.restore(model)
            row = {"epoch": epoch, "lr": epoch_lr, "train_loss": tr_l, "train_acc": tr_a, "val_loss": va_l, "val_acc": va_a}
            if use_f1:
                row["val_macro_f1"] = va_f1
            history.append(row)
    finally:
        if history:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_val_acc, best_epoch = _maybe_promote_swa(
        swa, lambda: _eval_cached(eval_head, x_va_eval, y_va_eval, bsz, compute_dev, criterion)[1],
        model, best_val_acc, best_epoch, tot_ep,
        lambda acc, ep: _save_checkpoint(best_path, model, config, cls_names, acc, ep, head_t, fn, h_kw, pool_m, feat_m, desc_w, fn_m, f_drop) if save_en else None,
    )
    return history, best_val_acc, best_epoch


def _batched_layer_norm(x_k: torch.Tensor, norms: List[nn.LayerNorm]) -> torch.Tensor:
    mean = x_k.mean(dim=-1, keepdim=True)
    var = (x_k - mean).pow(2).mean(dim=-1, keepdim=True)
    x_hat = (x_k - mean) * torch.rsqrt(var + float(norms[0].eps))
    w = torch.stack([n.weight for n in norms], dim=0).unsqueeze(1)
    b = torch.stack([n.bias for n in norms], dim=0).unsqueeze(1)
    return x_hat * w + b


def _batched_linear_bmm(x_k: torch.Tensor, linears: List[nn.Linear]) -> torch.Tensor:
    w = torch.stack([m.weight for m in linears], dim=0)
    out = torch.bmm(x_k, w.transpose(1, 2))
    if linears[0].bias is not None:
        b = torch.stack([m.bias for m in linears], dim=0).unsqueeze(1)
        out = out + b
    return out


def _batched_activation(h_k: torch.Tensor, acts: List[nn.Module], same_act: bool) -> torch.Tensor:
    if same_act:
        return acts[0](h_k)
    return torch.stack([act(h_k[i]) for i, act in enumerate(acts)], dim=0)


def _batched_dropout(h_k: torch.Tensor, drops: List[nn.Module], training: bool, same_drop: bool) -> torch.Tensor:
    if not training or all(isinstance(d, nn.Identity) for d in drops):
        return h_k
    if same_drop:
        return drops[0](h_k)
    return torch.stack([d(h_k[i]) for i, d in enumerate(drops)], dim=0)


def _co_train_forward_cores(
    head_type: str,
    cores: List[nn.Module],
    x_in: torch.Tensor,
    training: bool,
    same_act: bool,
    same_drop: bool,
) -> torch.Tensor:
    k_cnt = len(cores)
    if head_type == "linear":
        if not training and not x_in.is_cuda and not torch.is_grad_enabled() and is_accel_available():
            w_bank = torch.stack([c.weight for c in cores], dim=0)
            b_bank = torch.stack([c.bias for c in cores], dim=0) if cores[0].bias is not None else None
            return multihead_linear_forward(x_in, w_bank, b_bank)
        return _batched_linear_bmm(x_in.unsqueeze(0).expand(k_cnt, -1, -1), cores)
    x_k = x_in.unsqueeze(0).expand(k_cnt, -1, -1)
    if head_type in ("mlp", "residual"):
        seqs = [c.net for c in cores] if head_type == "mlp" else [c.blocks[0] for c in cores]
        h = _batched_layer_norm(x_k, [s[0] for s in seqs])
        h = _batched_linear_bmm(h, [s[1] for s in seqs])
        h = _batched_activation(h, [s[2] for s in seqs], same_act)
        h = _batched_dropout(h, [s[3] for s in seqs], training, same_drop)
        h = _batched_linear_bmm(h, [s[4] for s in seqs])
        if head_type == "mlp":
            return h
        out = x_k + h
        if isinstance(cores[0].out_norm, nn.LayerNorm):
            out = _batched_layer_norm(out, [c.out_norm for c in cores])
        return _batched_linear_bmm(out, [c.fc for c in cores])


def _is_co_trainable_core(head_type: str, core: nn.Module) -> bool:
    if head_type == "linear":
        return isinstance(core, nn.Linear)
    if head_type == "mlp":
        return isinstance(core, MLPClassifierHead) and len(core.net) == 5
    if head_type == "residual":
        return isinstance(core, ResidualAdapterHead) and len(core.blocks) == 1
    return False


def co_train_heads_cached(
    specs: List[dict],
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    cls_names: List[str],
) -> List[dict]:
    """Co-train K compatible FC heads simultaneously over shared batches via torch.bmm."""
    if not specs:
        return []
    first_cfg = specs[0]["config"]
    first_kw = dict(specs[0].get("train_kwargs") or {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(first_cfg.get("seed", 42))
    num_cls = int(first_cfg["num_classes"])
    bsz = int(first_cfg.get("batch_size", 32))
    tot_ep = int(first_cfg.get("epochs", 15))
    head_t = str(first_kw.get("head_type", "linear"))
    fn = bool(first_kw.get("feature_norm", False))
    fn_m = first_kw.get("feature_norm_mode", None)
    f_drop = float(first_kw.get("feature_dropout", 0.0))
    pool_m = str(first_kw.get("pool_mode", "avg"))
    feat_m = str(first_kw.get("feature_mode", "backbone"))
    desc_w = float(first_kw.get("desc_weight", 1.0))
    mixup_a = float(first_kw.get("mixup_alpha", 0.0))
    cutmix_a = float(first_kw.get("cutmix_alpha", 0.0))
    f_noise = float(first_kw.get("feature_noise_std", 0.0))
    warm_ridge = first_kw.get("warm_start_ridge", None)

    models, opts, scheds, crits, emas, swas, cores = [], [], [], [], [], [], []
    run_dirs, h_kws, smooths, swa_eps = [], [], [], []
    prev_threads = torch.get_num_threads()
    if device.type == "cpu" and prev_threads > 2:
        torch.set_num_threads(2)
    try:
        for sp in specs:
            cfg = sp["config"]
            tk = dict(sp.get("train_kwargs") or {})
            h_kw = dict(tk.get("head_kwargs") or {})
            run_dir = Path(cfg["out_dir"]) / cfg["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            _set_seed(seed)
            mod = build_model(
                cfg.get("model", "resnet18"), num_cls, head_type=head_t, feature_norm=fn,
                pool_mode=pool_m, feature_mode=feat_m, desc_weight=desc_w,
                feature_norm_mode=fn_m, feature_dropout=f_drop, **h_kw,
            ).to(device)
            std_st = sp.get("precomputed_standardizer_state")
            if std_st is not None and hasattr(mod, "standardizer"):
                mod.standardizer.load_state_dict(std_st)
            _fit_head_center_stats(mod.fc, x_tr)
            ws_pay = tk.get("precomputed_warm_start")
            if warm_ridge is not None:
                if not (ws_pay is not None and _apply_warm_start_payload(mod.fc, ws_pay)):
                    _warm_start_head(mod.fc, x_tr, y_tr, float(warm_ridge))
            params = [p for p in mod.fc.parameters() if p.requires_grad]
            opt_fn, sched_fn = sp.get("optimizer"), sp.get("scheduler")
            opt = opt_fn(params) if opt_fn is not None else torch.optim.Adam(params, lr=1e-3)
            _apply_head_lr_scale(opt, mod.fc, float(tk.get("head_lr_scale", 1.0)))
            sched = sched_fn(opt) if sched_fn is not None else None
            ls = float(tk.get("label_smoothing", 0.0))
            crit = _resolve_criterion(
                y_tr, num_cls, ls, float(tk.get("focal_gamma", 0.0)), tk.get("class_weight_mode"),
                device, logit_adj_tau=float(tk.get("logit_adj_tau", 0.0)),
                logit_margin=float(tk.get("logit_margin", 0.0)),
                loss_cap_tau=float(tk.get("loss_cap_tau", 0.0)),
                loss_cap_alpha=float(tk.get("loss_cap_alpha", 0.0)),
                loss_cap_mode=str(tk.get("loss_cap_mode", "fixed")),
                loss_cap_param=float(tk.get("loss_cap_param", 1.75)),
            )
            ema_d = float(tk.get("ema_decay", 0.0))
            swa_e = int(tk.get("swa_epochs", 0))
            core = _unwrap_head_core(mod.fc)
            if not _is_co_trainable_core(head_t, core):
                raise ValueError(f"Unsupported co-train core for head_type={head_t!r}")
            models.append(mod)
            cores.append(core)
            opts.append(opt)
            scheds.append(sched)
            crits.append(crit)
            emas.append(ModelEMA(mod, decay=ema_d) if ema_d > 0.0 else None)
            swas.append(ModelSWA(mod) if swa_e > 0 else None)
            run_dirs.append(run_dir)
            h_kws.append(h_kw)
            smooths.append(ls)
            swa_eps.append(swa_e)

        norm_mod, _ = _split_norm_eval_head(models[0].fc)
        if head_t in ("mlp", "residual"):
            seqs0 = [c.net for c in cores] if head_t == "mlp" else [c.blocks[0] for c in cores]
            acts = [s[2] for s in seqs0]
            drops = [s[3] for s in seqs0]
        else:
            acts, drops = [], []
        same_act = bool(acts) and all(
            type(a) is type(acts[0]) and type(a) in (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU) and repr(a) == repr(acts[0])
            for a in acts
        )
        same_drop = (not drops) or all(type(d) is type(drops[0]) and getattr(d, "p", 0.0) == getattr(drops[0], "p", 0.0) for d in drops)

        x_va_dev = x_va.to(device) if (norm_mod is not None or (device.type == "cuda" and x_va.numel() * 4 <= _AUTO_COMPRESS_BYTES)) else x_va
        y_va_dev = y_va.to(device) if device.type == "cuda" else y_va
        if norm_mod is not None:
            with torch.no_grad():
                x_va_eval = norm_mod(x_va_dev)
        else:
            x_va_eval = x_va_dev

        pre_norm_train = norm_mod is not None and mixup_a == 0.0 and cutmix_a == 0.0 and f_drop == 0.0
        if pre_norm_train:
            with torch.no_grad():
                x_tr_use = norm_mod(x_tr.to(device))
        elif device.type == "cuda" and x_tr.numel() * 4 <= _AUTO_COMPRESS_BYTES:
            x_tr_use = x_tr.to(device)
        else:
            x_tr_use = x_tr
        y_tr_use = y_tr.to(device) if device.type == "cuda" else y_tr

        k_cnt = len(specs)
        n_tr = int(x_tr_use.shape[0])
        n_va = int(x_va_eval.shape[0])
        histories: List[List[dict]] = [[] for _ in range(k_cnt)]
        best_accs = [-1.0] * k_cnt
        best_losses = [float("inf")] * k_cnt
        best_eps = [0] * k_cnt
        _set_seed(seed)

        def _eval_all() -> Tuple[List[float], List[float]]:
            for m in models:
                m.fc.eval()
            for cr in crits:
                cr.eval()
            t_loss = [0.0] * k_cnt
            t_ok = [0] * k_cnt
            with torch.no_grad():
                for start in range(0, n_va, bsz):
                    bx = x_va_eval[start:start + bsz].to(device)
                    by = y_va_dev[start:start + bsz].to(device)
                    logits_k = _co_train_forward_cores(head_t, cores, bx, False, same_act, same_drop)
                    row_n = int(by.size(0))
                    for idx_k in range(k_cnt):
                        lg = logits_k[idx_k]
                        t_loss[idx_k] += float(crits[idx_k](lg, by).item()) * row_n
                        t_ok[idx_k] += int((lg.argmax(dim=1) == by).sum().item())
            den = max(1, n_va)
            return [v / den for v in t_loss], [v / den for v in t_ok]

        for epoch in range(1, tot_ep + 1):
            ep_lrs = [float(o.param_groups[0]["lr"]) for o in opts]
            for m in models:
                m.fc.train()
            for cr in crits:
                cr.train()
            perm = torch.randperm(n_tr, device=x_tr_use.device)
            tr_losses = [0.0] * k_cnt
            tr_oks = [0] * k_cnt
            for start in range(0, n_tr, bsz):
                idx = perm[start:start + bsz]
                bx = x_tr_use[idx].to(device)
                if f_noise > 0.0:
                    bx = bx + torch.randn_like(bx) * f_noise
                by = y_tr_use[idx].to(device)
                row_n = int(by.size(0))

                use_cut = cutmix_a > 0.0 and (mixup_a <= 0.0 or float(torch.rand(()).item()) < 0.5)
                if use_cut:
                    y_eye = F.one_hot(by.long(), num_cls).float()
                    x_in, y_mix_oh, target_cmp = apply_cutmix(bx, y_eye, by, cutmix_a)
                    y_softs = [
                        (y_mix_oh * (1.0 - smooths[i]) + (smooths[i] / float(num_cls))) if smooths[i] > 0.0 else y_mix_oh
                        for i in range(k_cnt)
                    ]
                    lam = None
                elif mixup_a > 0.0:
                    lam = float(torch._sample_dirichlet(torch.tensor([mixup_a, mixup_a], dtype=torch.float32))[0].item())
                    mix_p = torch.randperm(row_n, device=device)
                    x_in = lam * bx + (1.0 - lam) * bx[mix_p]
                    target_cmp = by if lam >= 0.5 else by[mix_p]
                    y_softs = []
                    for i in range(k_cnt):
                        y_oh = crits[i]._smooth_targets(by)
                        y_softs.append(lam * y_oh + (1.0 - lam) * y_oh[mix_p])
                else:
                    x_in = bx
                    target_cmp = by
                    y_softs = [crits[i]._smooth_targets(by) for i in range(k_cnt)]

                if not pre_norm_train and norm_mod is not None:
                    with torch.no_grad():
                        x_in = norm_mod(x_in)
                if f_drop > 0.0:
                    x_in = F.dropout(x_in, p=f_drop, training=True)

                for opt in opts:
                    opt.zero_grad()
                logits_k = _co_train_forward_cores(head_t, cores, x_in, True, same_act, same_drop)
                step_losses = []
                for idx_k in range(k_cnt):
                    lk = crits[idx_k]._compute_train_loss(logits_k[idx_k], y_softs[idx_k])
                    step_losses.append(lk)
                    tr_losses[idx_k] += float(lk.detach().item()) * row_n
                    tr_oks[idx_k] += int((logits_k[idx_k].detach().argmax(dim=1) == target_cmp).sum().item())
                torch.stack(step_losses).sum().backward()
                for idx_k in range(k_cnt):
                    opts[idx_k].step()
                    if emas[idx_k] is not None:
                        emas[idx_k].update(models[idx_k])

            for idx_k in range(k_cnt):
                if swas[idx_k] is not None and epoch > max(0, tot_ep - swa_eps[idx_k]):
                    swas[idx_k].update(models[idx_k])
                if emas[idx_k] is not None:
                    emas[idx_k].apply_to(models[idx_k])
            va_losses, va_accs = _eval_all()
            den_tr = max(1, n_tr)
            for idx_k in range(k_cnt):
                cfg_k = specs[idx_k]["config"]
                save_en = bool(cfg_k.get("save_ckpt", True))
                _step_scheduler(scheds[idx_k], va_losses[idx_k])
                if _is_better_val(va_accs[idx_k], va_losses[idx_k], best_accs[idx_k], best_losses[idx_k]):
                    best_accs[idx_k], best_losses[idx_k], best_eps[idx_k] = va_accs[idx_k], va_losses[idx_k], epoch
                    if save_en:
                        _save_checkpoint(
                            run_dirs[idx_k] / "best.pt", models[idx_k], cfg_k, cls_names,
                            best_accs[idx_k], epoch, head_t, fn, h_kws[idx_k], pool_m, feat_m, desc_w, fn_m, f_drop,
                        )
                if emas[idx_k] is not None:
                    emas[idx_k].restore(models[idx_k])
                row = {
                    "epoch": epoch, "lr": ep_lrs[idx_k],
                    "train_loss": tr_losses[idx_k] / den_tr, "train_acc": tr_oks[idx_k] / den_tr,
                    "val_loss": va_losses[idx_k], "val_acc": va_accs[idx_k],
                }
                histories[idx_k].append(row)

        for idx_k in range(k_cnt):
            if swas[idx_k] is not None and swas[idx_k].count > 0:
                cfg_k = specs[idx_k]["config"]
                save_en = bool(cfg_k.get("save_ckpt", True))
                eval_mod = _unwrap_head_core(models[idx_k].fc)
                best_accs[idx_k], best_eps[idx_k] = _maybe_promote_swa(
                    swas[idx_k], lambda em=eval_mod, cr=crits[idx_k]: _eval_cached(em, x_va_eval, y_va_dev, bsz, device, cr)[1],
                    models[idx_k], best_accs[idx_k], best_eps[idx_k], tot_ep,
                    lambda acc, ep, k=idx_k, c=cfg_k, sv=save_en: _save_checkpoint(
                        run_dirs[k] / "best.pt", models[k], c, cls_names, acc, ep, head_t, fn, h_kws[k], pool_m, feat_m, desc_w, fn_m, f_drop,
                    ) if sv else None,
                )
    finally:
        for idx_k in range(k_cnt):
            if histories[idx_k]:
                (run_dirs[idx_k] / "history.json").write_text(json.dumps(histories[idx_k], indent=2), encoding="utf-8")
        if torch.get_num_threads() != prev_threads:
            torch.set_num_threads(prev_threads)

    raw_b = int(x_tr.numel() * x_tr.element_size())
    return [
        {
            "name": specs[idx_k]["config"]["name"],
            "best_val_acc": best_accs[idx_k],
            "best_epoch": best_eps[idx_k],
            "final_lr": float(opts[idx_k].param_groups[0]["lr"]),
            "history": histories[idx_k],
            "best_ckpt": str(run_dirs[idx_k] / "best.pt"),
            "history_path": str(run_dirs[idx_k] / "history.json"),
            "cache_memory_bytes": raw_b,
            "raw_feature_bytes": raw_b,
        }
        for idx_k in range(k_cnt)
    ]


def run_cached(
    config: dict, optimizer: Optional[Callable] = None, scheduler: Optional[Callable] = None,
    train_transform=None, head_type: str = "linear", feature_norm: bool = False,
    pool_mode: str = "avg", head_kwargs: Optional[dict] = None, label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0, ema_decay: float = 0.0, warm_start_ridge: Optional[float] = None,
    tta_extract: bool = False, compress_cache=True, class_weight_mode: Optional[str] = None,
    focal_gamma: float = 0.0, feature_mode: str = "backbone", desc_weight: float = 1.0,
    precomputed_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    precomputed_classes: Optional[List[str]] = None, precomputed_standardizer_state: Optional[dict] = None,
    cutmix_alpha: float = 0.0, feature_norm_mode: Optional[str] = None, feature_dropout: float = 0.0,
    logit_adj_tau: float = 0.0, logit_margin: float = 0.0, swa_epochs: int = 0,
    head_lr_scale: float = 1.0, loss_cap_tau: float = 0.0, loss_cap_alpha: float = 0.0,
    loss_cap_mode: str = "fixed", loss_cap_param: float = 1.75, auto_tune_resilient: bool = False,
    precomputed_warm_start: Optional[dict] = None, feature_noise_std: float = 0.0, ckpt_metric: str = "val_acc",
) -> dict:
    """Extract frozen features once and train only the FC head on cached embeddings."""
    _set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config["out_dir"]) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, history_path = run_dir / "best.pt", run_dir / "history.json"
    h_kw, num_cls, bsz, tot_ep = dict(head_kwargs or {}), int(config["num_classes"]), int(config.get("batch_size", 32)), int(config.get("epochs", 15))
    model = build_model(
        config.get("model", "resnet18"), num_cls, head_type=head_type, feature_norm=feature_norm,
        pool_mode=pool_mode, feature_mode=feature_mode, desc_weight=desc_weight,
        feature_norm_mode=feature_norm_mode, feature_dropout=feature_dropout, **h_kw,
    ).to(device)
    x_tr, y_tr, x_va, y_va, cls_names = _resolve_cached_features(
        model, config, train_transform, device, num_cls, tta_extract,
        precomputed_features, precomputed_classes, precomputed_standardizer_state,
    )
    tuned_info: Optional[dict] = None
    if auto_tune_resilient and is_accel_available():
        in_dim = int(x_tr.shape[1])

        def _probe_head(xs: torch.Tensor, ys: torch.Tensor, learnable_leak: bool = False, channel_leak: bool = False):
            kw_probe = {**h_kw, "learnable_leak": bool(learnable_leak), "channel_leak": bool(channel_leak)}
            h_mod = build_head(
                in_dim, num_cls, head_type=head_type, feature_norm=feature_norm,
                feature_norm_mode=feature_norm_mode, feature_dropout=feature_dropout, **kw_probe,
            )
            _fit_head_center_stats(h_mod, xs)
            if warm_start_ridge is not None:
                _warm_start_head(h_mod, xs, ys, float(warm_start_ridge))
            return h_mod

        def _probe_loss(ys, loss_cap_mode="fixed", loss_cap_tau=0.8, loss_cap_alpha=0.5, loss_cap_param=1.75, class_weight_mode=None, label_smoothing=0.05, device=device):
            return _resolve_criterion(
                ys, num_cls, label_smoothing, focal_gamma, class_weight_mode, device,
                logit_adj_tau=logit_adj_tau, logit_margin=logit_margin,
                loss_cap_tau=loss_cap_tau, loss_cap_alpha=loss_cap_alpha,
                loss_cap_mode=loss_cap_mode, loss_cap_param=loss_cap_param,
            )

        tuned_info = _core_auto_tune_resilient(
            x_tr, y_tr, x_va, y_va, num_cls, _probe_head, _probe_loss,
            probe_epochs=max(2, min(5, tot_ep // 2 if tot_ep >= 4 else tot_ep)),
            batch_size=bsz, seed=int(config.get("seed", 42)), label_smoothing=float(label_smoothing),
        )
        best_hp = tuned_info["best"]
        loss_cap_mode = best_hp["loss_cap_mode"]
        loss_cap_tau = best_hp["loss_cap_tau"]
        loss_cap_alpha = best_hp["loss_cap_alpha"]
        loss_cap_param = best_hp["loss_cap_param"]
        class_weight_mode = best_hp["class_weight_mode"]
        h_kw = {**h_kw, "learnable_leak": best_hp["learnable_leak"], "channel_leak": best_hp["channel_leak"]}
        model.fc = build_head(
            in_dim, num_cls, head_type=head_type, feature_norm=feature_norm,
            feature_norm_mode=feature_norm_mode, feature_dropout=feature_dropout, **h_kw,
        ).to(device)
        _set_seed(int(config.get("seed", 42)))

    raw_bytes = int(x_tr.numel() * x_tr.element_size())
    bank, cache_bytes = _build_feature_bank(x_tr, compress_cache)
    _fit_head_center_stats(model.fc, x_tr)
    if warm_start_ridge is not None:
        if not (precomputed_warm_start is not None and not auto_tune_resilient and _apply_warm_start_payload(model.fc, precomputed_warm_start)):
            _warm_start_head(model.fc, x_tr, y_tr, float(warm_start_ridge))
    params = [p for p in model.fc.parameters() if p.requires_grad]
    opt = optimizer(params) if optimizer is not None else torch.optim.Adam(params, lr=1e-3)
    _apply_head_lr_scale(opt, model.fc, head_lr_scale)
    criterion = _resolve_criterion(y_tr, num_cls, label_smoothing, focal_gamma, class_weight_mode, device, logit_adj_tau=logit_adj_tau, logit_margin=logit_margin, loss_cap_tau=loss_cap_tau, loss_cap_alpha=loss_cap_alpha, loss_cap_mode=loss_cap_mode, loss_cap_param=loss_cap_param)
    loop_cfg = {**config, "swa_epochs_eff": int(swa_epochs), "feature_noise_std_eff": float(feature_noise_std), "ckpt_metric_eff": str(ckpt_metric)}
    meta = (bsz, tot_ep, num_cls, mixup_alpha, cutmix_alpha, label_smoothing, head_type, feature_norm, h_kw, pool_mode, feature_mode, desc_weight, feature_norm_mode, feature_dropout)
    prev_threads = torch.get_num_threads()
    if device.type == "cpu" and prev_threads > 2:
        torch.set_num_threads(2)
    try:
        history, best_val_acc, best_epoch = _run_cached_loop(
            model, loop_cfg, opt, scheduler(opt) if scheduler is not None else None, criterion,
            ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None,
            ModelSWA(model) if int(swa_epochs) > 0 else None,
            bank, x_tr, y_tr, x_va, y_va, cls_names, best_path, history_path, meta,
        )
    finally:
        if torch.get_num_threads() != prev_threads:
            torch.set_num_threads(prev_threads)
    out = {"name": config["name"], "best_val_acc": best_val_acc, "best_epoch": best_epoch, "final_lr": float(opt.param_groups[0]["lr"]), "history": history, "best_ckpt": str(best_path), "history_path": str(history_path), "cache_memory_bytes": cache_bytes, "raw_feature_bytes": raw_bytes}
    if tuned_info is not None:
        out["auto_tuned"] = tuned_info
    return out
