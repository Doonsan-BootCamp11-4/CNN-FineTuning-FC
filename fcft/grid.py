"""Declarative Grid Search builder, picklable factories, and Markdown comparison formatting."""

import itertools
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch

from fcft.accel import ACTIVATION_CHOICES, CurvatureBalancedOptimizer, cosine_decay_factor, is_accel_available
from fcft.data import build_aug_transform


LOSS_PRESETS: Dict[str, Dict[str, Any]] = {
    "ce": {"label_smoothing": 0.0, "focal_gamma": 0.0, "class_weight_mode": None, "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "ce_ls0.05": {"label_smoothing": 0.05, "focal_gamma": 0.0, "class_weight_mode": None, "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "ce_ls0.1": {"label_smoothing": 0.1, "focal_gamma": 0.0, "class_weight_mode": None, "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "focal_g1.5": {"label_smoothing": 0.0, "focal_gamma": 1.5, "class_weight_mode": None, "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "focal_g2.0_ls": {"label_smoothing": 0.05, "focal_gamma": 2.0, "class_weight_mode": None, "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "cw_inv_sqrt": {"label_smoothing": 0.05, "focal_gamma": 0.0, "class_weight_mode": "inv_sqrt", "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "cw_effective": {"label_smoothing": 0.05, "focal_gamma": 0.0, "class_weight_mode": "effective", "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "logit_adj_tau0.5": {"label_smoothing": 0.05, "focal_gamma": 0.0, "class_weight_mode": None, "logit_adj_tau": 0.5, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "logit_adj_margin": {"label_smoothing": 0.05, "focal_gamma": 0.0, "class_weight_mode": None, "logit_adj_tau": 0.5, "logit_margin": 0.1, "loss_cap_mode": "fixed", "loss_cap_tau": 0.0, "loss_cap_alpha": 0.0, "loss_cap_param": 0.0},
    "resilient_ce": {"label_smoothing": 0.05, "focal_gamma": 0.0, "class_weight_mode": None, "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 1.0, "loss_cap_alpha": 0.5, "loss_cap_param": 0.0},
    "resilient_cw": {"label_smoothing": 0.05, "focal_gamma": 0.0, "class_weight_mode": "effective", "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "fixed", "loss_cap_tau": 0.8, "loss_cap_alpha": 0.5, "loss_cap_param": 0.0},
    "resilient_auto": {"label_smoothing": 0.05, "focal_gamma": 0.0, "class_weight_mode": "effective", "logit_adj_tau": 0.0, "logit_margin": 0.0, "loss_cap_mode": "mad", "loss_cap_tau": 0.8, "loss_cap_alpha": 0.5, "loss_cap_param": 1.75, "learnable_leak": True},
}
_LOSS_ALIASES = {"ce_ls": "ce_ls0.05", "focal": "focal_g1.5", "logit_adj": "logit_adj_tau0.5", "resilient": "resilient_ce"}


def resolve_loss_preset(loss_mode: Optional[str]) -> Dict[str, Any]:
    """Resolve a named loss preset into UnifiedClassificationLoss keyword arguments."""
    if loss_mode is None:
        return {}
    key = _LOSS_ALIASES.get(str(loss_mode), str(loss_mode))
    if key not in LOSS_PRESETS:
        raise ValueError(f"Unsupported loss_mode: {loss_mode!r}")
    return {k: v for k, v in LOSS_PRESETS[key].items() if k not in ("learnable_leak", "channel_leak")}


class OptimizerFactory:
    """Picklable optimizer factory compatible with Windows/Linux spawn multiprocessing."""

    def __init__(
        self,
        mode: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        no_bias_decay: bool = True,
    ):
        self.mode = str(mode)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.no_bias_decay = bool(no_bias_decay)

    def __call__(self, params) -> torch.optim.Optimizer:
        req = [p for p in params if getattr(p, "requires_grad", True)]
        if self.mode == "calibrated":
            return CurvatureBalancedOptimizer(req, lr=self.lr, weight_decay=self.weight_decay)
        if self.mode == "sgd":
            return torch.optim.SGD(req, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
        if self.mode == "adam":
            return torch.optim.Adam(req, lr=self.lr, weight_decay=self.weight_decay)
        if self.mode == "rmsprop":
            return torch.optim.RMSprop(req, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        if self.mode == "nadam":
            return torch.optim.NAdam(req, lr=self.lr, weight_decay=self.weight_decay)
        if self.mode == "radam":
            return torch.optim.RAdam(req, lr=self.lr, weight_decay=self.weight_decay)
        if self.mode != "adamw":
            raise ValueError(f"Unsupported optimizer_mode: {self.mode!r}")
        if not self.no_bias_decay:
            return torch.optim.AdamW(req, lr=self.lr, weight_decay=self.weight_decay)
        decay = [p for p in req if p.ndim >= 2]
        no_decay = [p for p in req if p.ndim < 2]
        groups = ([{"params": decay, "weight_decay": self.weight_decay}] if decay else []) + ([{"params": no_decay, "weight_decay": 0.0}] if no_decay else [])
        return torch.optim.AdamW(groups if groups else req, lr=self.lr)


class _CosineWarmupFn:
    """Picklable LR multiplier callable using compiled cosine decay."""

    def __init__(self, total_epochs: int, min_ratio: float, warmup_epochs: int):
        self.total = max(1, int(total_epochs))
        self.warm = max(0, min(self.total - 1, int(warmup_epochs)))
        self.floor = float(min_ratio)

    def __call__(self, epoch_idx: int) -> float:
        if self.warm > 0 and epoch_idx < self.warm:
            return float(epoch_idx + 1) / float(self.warm + 1)
        prog = min(1.0, float(max(0, epoch_idx - self.warm)) / float(max(1, self.total - self.warm)))
        return self.floor + (1.0 - self.floor) * cosine_decay_factor(prog)


class SchedulerFactory:
    """Picklable LR scheduler factory compatible with Windows/Linux spawn multiprocessing."""

    def __init__(
        self,
        kind: str = "cosine",
        total_epochs: int = 15,
        warmup_epochs: int = 0,
        step_size: int = 5,
        gamma: float = 0.5,
        milestones: Tuple[int, ...] = (5, 10),
        min_ratio: float = 0.01,
        eta_min: float = 1e-6,
        patience: int = 2,
        start_factor: float = 0.1,
    ):
        self.kind = str(kind)
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = int(warmup_epochs)
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.milestones = tuple(int(m) for m in milestones)
        self.min_ratio = float(min_ratio)
        self.eta_min = float(eta_min)
        self.patience = int(patience)
        self.start_factor = float(start_factor)

    def __call__(self, optimizer: torch.optim.Optimizer):
        if self.kind in ("cosine", "warmup_cosine"):
            warm = max(1, self.warmup_epochs) if self.kind == "warmup_cosine" else self.warmup_epochs
            if is_accel_available():
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_CosineWarmupFn(self.total_epochs, self.min_ratio, warm))
            if warm <= 0:
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_epochs, eta_min=self.eta_min)
            wu = max(1, min(warm, max(1, self.total_epochs - 1)))
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.start_factor, total_iters=wu),
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.total_epochs - wu), eta_min=self.eta_min),
                ],
                milestones=[wu],
            )
        if self.kind == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, self.step_size), gamma=self.gamma)
        if self.kind == "multistep":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(self.milestones), gamma=self.gamma)
        if self.kind == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=self.gamma, patience=self.patience, min_lr=self.eta_min)
        raise ValueError(f"Unsupported scheduler: {self.kind!r}")


def _optimizer_factory(mode: str = "adamw", lr: float = 1e-3, weight_decay: float = 1e-2) -> Callable:
    return OptimizerFactory(mode=mode, lr=lr, weight_decay=weight_decay, no_bias_decay=True)


def exact_cosine_scheduler(total_epochs: int = 15, min_ratio: float = 0.01, warmup_epochs: int = 0) -> Callable:
    """Create LR scheduler backed by linear warmup and compiled high-precision cosine decay."""
    return SchedulerFactory(kind="cosine", total_epochs=total_epochs, warmup_epochs=warmup_epochs, min_ratio=min_ratio, eta_min=min_ratio * 1e-3)


def _sched_factory(kind: str, total_epochs: int, warmup_epochs: int = 0) -> Optional[Callable]:
    if kind in (None, "none"):
        return None
    if kind in ("cosine", "warmup_cosine"):
        wu = max(1, int(warmup_epochs)) if kind == "warmup_cosine" else 0
        return SchedulerFactory(kind=kind, total_epochs=total_epochs, warmup_epochs=wu)
    if kind == "step":
        return SchedulerFactory(kind="step", total_epochs=total_epochs, step_size=max(1, total_epochs // 3), gamma=0.5)
    raise ValueError(f"Unsupported grid scheduler: {kind!r}")


def _mix_alphas(mix_mode: str) -> Dict[str, float]:
    if mix_mode in (None, "none"):
        return {"mixup_alpha": 0.0, "cutmix_alpha": 0.0}
    if mix_mode == "mixup":
        return {"mixup_alpha": 0.2, "cutmix_alpha": 0.0}
    if mix_mode == "cutmix":
        return {"mixup_alpha": 0.0, "cutmix_alpha": 1.0}
    if mix_mode == "both":
        return {"mixup_alpha": 0.2, "cutmix_alpha": 1.0}
    raise ValueError(f"Unsupported mix_mode: {mix_mode!r}")


def _resolve_norm_mode(combo: dict, defaults: dict) -> str:
    mode = combo.get("feature_norm_mode", defaults.get("feature_norm_mode", None))
    if mode is not None:
        return str(mode)
    return "power_l2" if bool(combo.get("feature_norm", defaults.get("feature_norm", True))) else "none"


def _build_train_kwargs(combo: dict, defaults: dict, fnorm_mode: str, alphas: Dict[str, float]) -> Dict[str, Any]:
    pool = combo.get("pool_mode", defaults.get("pool_mode", "avg"))
    head = combo.get("head_type", defaults.get("head_type", "linear"))
    raw_loss_m = combo.get("loss_mode", defaults.get("loss_mode", None))
    preset_full = LOSS_PRESETS.get(_LOSS_ALIASES.get(str(raw_loss_m), str(raw_loss_m)), {}) if raw_loss_m is not None else {}
    lp = resolve_loss_preset(raw_loss_m)
    cw = combo.get("class_weight_mode", lp.get("class_weight_mode", defaults.get("class_weight_mode", None)))
    la_fallback = lp.get("logit_adj_tau", 0.0 if cw is not None else float(defaults.get("logit_adj_tau", 0.0)))
    la_tau = float(combo["logit_adj_tau"]) if "logit_adj_tau" in combo else float(la_fallback)
    kw: Dict[str, Any] = {
        "pool_mode": pool,
        "head_type": head,
        "feature_mode": combo.get("feature_mode", defaults.get("feature_mode", "backbone")),
        "feature_norm": fnorm_mode != "none",
        "feature_norm_mode": fnorm_mode,
        "feature_dropout": float(combo.get("feature_dropout", defaults.get("feature_dropout", 0.0))),
        "label_smoothing": float(combo.get("label_smoothing", lp.get("label_smoothing", defaults.get("label_smoothing", 0.0)))),
        "focal_gamma": float(combo.get("focal_gamma", lp.get("focal_gamma", defaults.get("focal_gamma", 0.0)))),
        "logit_adj_tau": la_tau,
        "logit_margin": float(combo.get("logit_margin", lp.get("logit_margin", defaults.get("logit_margin", 0.0)))),
        "loss_cap_mode": str(combo.get("loss_cap_mode", lp.get("loss_cap_mode", defaults.get("loss_cap_mode", "fixed")))),
        "loss_cap_tau": float(combo.get("loss_cap_tau", lp.get("loss_cap_tau", defaults.get("loss_cap_tau", 0.0)))),
        "loss_cap_alpha": float(combo.get("loss_cap_alpha", lp.get("loss_cap_alpha", defaults.get("loss_cap_alpha", 0.0)))),
        "loss_cap_param": float(combo.get("loss_cap_param", lp.get("loss_cap_param", defaults.get("loss_cap_param", 1.75)))),
        "swa_epochs": int(combo.get("swa_epochs", defaults.get("swa_epochs", 0))),
        "class_weight_mode": cw,
        "ema_decay": float(combo.get("ema_decay", defaults.get("ema_decay", 0.99))),
        "warm_start_ridge": combo.get("warm_start_ridge", defaults.get("warm_start_ridge", 10.0)),
        "head_lr_scale": float(combo.get("head_lr_scale", defaults.get("head_lr_scale", 1.0))),
        "mixup_alpha": alphas["mixup_alpha"],
        "cutmix_alpha": alphas["cutmix_alpha"],
    }
    trainer_k = combo.get("trainer", defaults.get("trainer", "cached"))
    if trainer_k == "cached":
        kw["tta_extract"] = bool(combo.get("tta_extract", defaults.get("tta_extract", True)))
        kw["compress_cache"] = combo.get("compress_cache", defaults.get("compress_cache", "auto"))
        kw["feature_noise_std"] = float(combo.get("feature_noise_std", defaults.get("feature_noise_std", 0.0)))
        kw["ckpt_metric"] = str(combo.get("ckpt_metric", defaults.get("ckpt_metric", "val_acc")))
        if "auto_tune_resilient" in combo or "auto_tune_resilient" in defaults:
            kw["auto_tune_resilient"] = bool(combo.get("auto_tune_resilient", defaults.get("auto_tune_resilient", False)))
    h_kw = dict(combo.get("head_kwargs") or defaults.get("head_kwargs") or {})
    if head == "subcenter":
        h_kw.setdefault("num_centers", int(combo.get("num_centers", 3)))
    act_name = combo.get("activation", defaults.get("activation", None))
    if act_name is not None:
        act_low = str(act_name).lower()
        if act_low not in ACTIVATION_CHOICES and act_low not in ("swish", "smooth_learn"):
            raise ValueError(f"Unsupported activation: {act_name!r}")
        if head in ("residual", "mlp", "dual_anchor"):
            h_kw["activation"] = act_low
    for leak_k in ("learnable_leak", "channel_leak"):
        if leak_k in combo or leak_k in preset_full or leak_k in defaults:
            if head in ("residual", "mlp", "dual_anchor"):
                h_kw.setdefault(leak_k, bool(combo.get(leak_k, preset_full.get(leak_k, defaults.get(leak_k, False)))))
    if h_kw:
        kw["head_kwargs"] = h_kw
    return kw


def _build_grid_entry(idx: int, combo: dict, base_cfg: dict, defaults: dict) -> dict:
    fnorm_mode = _resolve_norm_mode(combo, defaults)
    mix = combo.get("mix_mode", defaults.get("mix_mode", "none"))
    train_kwargs = _build_train_kwargs(combo, defaults, fnorm_mode, _mix_alphas(mix))
    pool, head = train_kwargs["pool_mode"], train_kwargs["head_type"]
    cw, la_tau = train_kwargs["class_weight_mode"], train_kwargs["logit_adj_tau"]
    opt_m = str(combo.get("optimizer_mode", defaults.get("optimizer_mode", "adamw")))
    lr_val = float(combo.get("lr", defaults.get("lr", 2e-3)))
    wd_val = float(combo.get("weight_decay", defaults.get("weight_decay", 1e-2)))
    sched_k = combo.get("scheduler", defaults.get("scheduler", "cosine"))
    warm_ep = int(combo.get("warmup_epochs", defaults.get("warmup_epochs", 2)))
    aug_m = combo.get("aug_mode", defaults.get("aug_mode", None))
    eval_rm = str(combo.get("eval_resize_mode", defaults.get("eval_resize_mode", "squash")))
    tta_m = combo.get("eval_tta_mode", defaults.get("eval_tta_mode", "flip2"))
    use_aug_b = bool(combo.get("use_aug_bank", defaults.get("use_aug_bank", False)))
    aug_rep = int(combo.get("aug_repeats", defaults.get("aug_repeats", 3 if use_aug_b else 0)))
    trainer_k = combo.get("trainer", defaults.get("trainer", "cached"))
    bal_tag = str(cw) if cw is not None else (f"la{int(round(la_tau * 10))}" if la_tau > 0.0 else "std")
    tag = f"grid_{idx:03d}_{pool}_{head}_{opt_m}_{mix}_{bal_tag}"
    params_view = {
        "pool": pool, "head": head, "norm": fnorm_mode, "opt": opt_m, "mix": mix,
        "aug": f"{aug_rep}x({aug_m or 'rrc_jitter'})" if use_aug_b else "-", "resize": eval_rm,
        "smooth": train_kwargs["label_smoothing"], "focal": train_kwargs["focal_gamma"],
        "la": la_tau, "cw": str(cw or "-"), "swa": train_kwargs["swa_epochs"],
    }
    return {
        "config": {**base_cfg, "name": tag, "eval_resize_mode": eval_rm},
        "trainer": trainer_k,
        "grid_params": params_view,
        "optimizer": _optimizer_factory(mode=opt_m, lr=lr_val, weight_decay=wd_val),
        "scheduler": _sched_factory(sched_k, int(base_cfg["epochs"]), warmup_epochs=warm_ep),
        "train_transform": build_aug_transform(aug_m),
        "train_kwargs": train_kwargs,
        "eval_tta_mode": tta_m,
        "aug_mode": aug_m,
        "use_aug_bank": use_aug_b,
        "aug_repeats": aug_rep,
    }


def expand_grid(
    grid_spec: Dict[str, Sequence[Any]],
    base_config: Optional[dict] = None,
    defaults: Optional[dict] = None,
    max_runs: Optional[int] = None,
    seed: int = 42,
) -> List[dict]:
    """Expand a parameter dictionary into runnable grid search experiment entries."""
    if base_config is None:
        from configs.axes import BASE

        base_cfg = dict(BASE)
    else:
        base_cfg = dict(base_config)
    dfl = dict(defaults or {})
    keys = list(grid_spec.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid_spec[k] for k in keys))]
    if max_runs is not None and len(combos) > int(max_runs):
        rng = random.Random(int(seed))
        rng.shuffle(combos)
        combos = combos[:int(max_runs)]
    return [_build_grid_entry(i + 1, c, base_cfg, dfl) for i, c in enumerate(combos)]


_CUSTOM_BASE_KEYS = frozenset(
    {"split_csv", "data_root", "out_dir", "num_classes", "epochs", "batch_size", "num_workers", "seed", "model", "eval_resize_mode"}
)
_CUSTOM_COMBO_KEYS = frozenset(
    {
        "trainer", "pool_mode", "head_type", "activation", "feature_mode", "feature_norm", "feature_norm_mode",
        "feature_dropout", "loss_mode", "label_smoothing", "focal_gamma", "class_weight_mode", "logit_adj_tau",
        "logit_margin", "loss_cap_mode", "loss_cap_tau", "loss_cap_alpha", "loss_cap_param", "learnable_leak",
        "channel_leak", "optimizer_mode", "lr", "weight_decay", "scheduler", "warmup_epochs", "mix_mode",
        "aug_mode", "use_aug_bank", "aug_repeats", "tta_extract", "compress_cache", "auto_tune_resilient",
        "swa_epochs", "ema_decay", "warm_start_ridge", "head_lr_scale", "feature_noise_std", "ckpt_metric",
        "head_kwargs", "num_centers", "eval_resize_mode", "eval_tta_mode",
    }
)


def build_custom_entry(
    choices: Optional[Dict[str, Any]] = None,
    *,
    base_config: Optional[dict] = None,
    defaults: Optional[dict] = None,
    name: Optional[str] = None,
) -> dict:
    """Build one runnable experiment entry from user-selected technique options."""
    if base_config is None:
        from configs.axes import BASE

        base_cfg = dict(BASE)
    else:
        base_cfg = dict(base_config)
    if defaults is None:
        from configs.grid import DEFAULTS

        dfl = dict(DEFAULTS)
    else:
        dfl = dict(defaults)

    raw = dict(choices or {})
    unknown = sorted(set(raw) - (_CUSTOM_BASE_KEYS | _CUSTOM_COMBO_KEYS))
    if unknown:
        raise ValueError(f"Unsupported custom option key(s): {', '.join(unknown)}")

    combo: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in _CUSTOM_BASE_KEYS and v is not None:
            base_cfg[k] = v
        if k in _CUSTOM_COMBO_KEYS:
            combo[k] = v

    entry = _build_grid_entry(1, combo, base_cfg, dfl)
    if name:
        entry["config"]["name"] = str(name)
    elif entry["config"]["name"].startswith("grid_001_"):
        entry["config"]["name"] = f"single_{entry['config']['name'][9:]}"
    return entry


def run_single_experiment(
    choices: Optional[Dict[str, Any]] = None,
    *,
    base_config: Optional[dict] = None,
    defaults: Optional[dict] = None,
    name: Optional[str] = None,
    eval_test: bool = True,
    tta_mode: Optional[str] = None,
    use_disk_cache: bool = True,
    cache_dir: Optional[str] = None,
    bundle: Optional[dict] = None,
    bundle_path: Optional[str] = None,
) -> dict:
    """Train one custom technique combination and optionally run single-shot Test evaluation."""
    from fcft.data import assemble_from_bundle, extract_shared_feature_bundle
    from fcft.evaluate import evaluate
    from fcft.model import build_model
    from fcft.train import run, run_cached

    entry = build_custom_entry(choices, base_config=base_config, defaults=defaults, name=name)
    cfg = dict(entry["config"])
    extra = dict(entry.get("train_kwargs") or {})
    trainer_k = entry.get("trainer", "cached")
    active_bundle = bundle

    if trainer_k == "cached":
        pool_m, feat_m = extra.get("pool_mode", "avg"), extra.get("feature_mode", "backbone")
        if active_bundle is None:
            use_aug = bool(entry.get("use_aug_bank", False))
            aug_rep = int(entry.get("aug_repeats", 3)) if use_aug else 0
            t_pools = [pool_m] if feat_m != "descriptor" else []
            active_bundle = extract_shared_feature_bundle(
                cfg,
                aug_transform=entry.get("train_transform") if use_aug else None,
                aug_repeats=aug_rep,
                include_flip=True,
                include_descriptors=(feat_m != "backbone"),
                include_test=bool(eval_test),
                use_disk_cache=use_disk_cache,
                cache_dir=cache_dir,
                aug_mode=str(entry.get("aug_mode") or "rrc_jitter"),
                eval_resize_mode=str(cfg.get("eval_resize_mode", "squash")),
                target_pools=t_pools,
                bundle_path=bundle_path,
            )
        tmp_mod = None
        if feat_m != "backbone":
            tmp_mod = build_model(
                cfg.get("model", "resnet18"),
                int(cfg["num_classes"]),
                head_type=extra.get("head_type", "linear"),
                feature_norm=bool(extra.get("feature_norm", False)),
                pool_mode=pool_m,
                feature_mode=feat_m,
                **dict(extra.get("head_kwargs") or {}),
            )
        pre = assemble_from_bundle(
            active_bundle,
            tmp_mod,
            pool_mode=pool_m,
            feature_mode=feat_m,
            tta_extract=bool(extra.get("tta_extract", False)),
            use_aug_bank=bool(entry.get("use_aug_bank", False)),
        )
        std_state = tmp_mod.standardizer.state_dict() if (tmp_mod is not None and hasattr(tmp_mod, "standardizer")) else None
        result = run_cached(
            cfg,
            optimizer=entry.get("optimizer"),
            scheduler=entry.get("scheduler"),
            train_transform=entry.get("train_transform"),
            precomputed_features=pre,
            precomputed_classes=active_bundle["classes"],
            precomputed_standardizer_state=std_state,
            **extra,
        )
    else:
        result = run(
            cfg,
            optimizer=entry.get("optimizer"),
            scheduler=entry.get("scheduler"),
            train_transform=entry.get("train_transform"),
            **extra,
        )

    result["grid_params"] = dict(entry.get("grid_params") or {})
    if eval_test:
        tta = str(tta_mode or entry.get("eval_tta_mode", "flip2"))
        metrics = evaluate(cfg, tta_mode=tta, precomputed_test=active_bundle if trainer_k == "cached" else None)
        result["test_accuracy"] = float(metrics["accuracy"])
        result["test_macro_f1"] = float(metrics["macro_avg"]["f1-score"])
        result["metrics"] = metrics
    return result


def format_grid_table(records: Sequence[dict], param_map: Optional[Dict[str, dict]] = None) -> str:
    """Render ranked Grid Search results with key hyperparameter columns."""
    pmap = param_map or {}
    lines = [
        "| Rank | Experiment | Pool | Head | Opt | Mix/CutMix | Weight / LA | Aug | SWA | Best Val Acc |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for rank, r in enumerate(records, start=1):
        p = pmap.get(r["name"], {})
        bal = f"{p.get('cw', '-')} / {p.get('la', 0.0)}"
        lines.append(
            f"| {rank} | `{r['name']}` | {p.get('pool', '-')} | {p.get('head', '-')} "
            f"| {p.get('opt', '-')} | {p.get('mix', '-')} | {bal} | {p.get('aug', '-')} | {p.get('swa', 0)} | {r['best_val_acc']:.4f} |"
        )
    return "\n".join(lines) + "\n"
