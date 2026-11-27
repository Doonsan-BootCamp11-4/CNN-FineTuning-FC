"""Class-imbalance weighting, focal modulation, Mixup, and CutMix classification loss."""

from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from fcft.accel import AdaptiveBoundedLossGate, calibrated_log_softmax, exact_log_priors, is_accel_available


def compute_log_priors(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute natural-base log class priors ln(count_c / N) without raw Python math."""
    counts = torch.bincount(targets.detach().long().cpu().reshape(-1), minlength=int(num_classes)).clamp(min=1)
    if is_accel_available():
        return exact_log_priors(counts).to(device=targets.device)
    probs = counts.float() / counts.float().sum().clamp(min=1.0)
    return torch.log(probs).to(device=targets.device)


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    mode: str = "effective",
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute normalized per-class weights to counter class imbalance."""
    counts = torch.bincount(targets.detach().long().cpu().reshape(-1), minlength=int(num_classes)).float().clamp(min=1.0)
    if mode == "effective":
        b = float(beta)
        eff = (1.0 - torch.pow(torch.tensor(b, dtype=torch.float32), counts)).clamp(min=1e-12)
        weights = (1.0 - b) / eff
    elif mode == "inv_sqrt":
        weights = 1.0 / torch.sqrt(counts)
    else:
        raise ValueError(f"Unsupported class weight mode: {mode!r}")
    return weights * (float(num_classes) / weights.sum().clamp(min=1e-12))


def apply_cutmix(
    x: torch.Tensor,
    y_soft: torch.Tensor,
    y_hard: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply spatial box CutMix on 4D images or feature-block CutMix on 2D embeddings."""
    lam = float(torch._sample_dirichlet(torch.tensor([alpha, alpha], dtype=torch.float32))[0].item())
    bsz = int(x.size(0))
    perm = torch.randperm(bsz, device=x.device)
    x_mix = x.clone()
    if x.ndim == 4:
        _, _, h, w = x.shape
        cut_ratio = max(0.0, 1.0 - lam) ** 0.5
        cw = max(1, min(w, int(round(w * cut_ratio))))
        ch = max(1, min(h, int(round(h * cut_ratio))))
        cx = int(torch.randint(0, w, (1,), device=x.device).item())
        cy = int(torch.randint(0, h, (1,), device=x.device).item())
        x1, x2 = max(0, cx - cw // 2), min(w, cx + cw - cw // 2)
        y1, y2 = max(0, cy - ch // 2), min(h, cy + ch - ch // 2)
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam_ex = 1.0 - float((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
    else:
        dim = int(x.shape[1])
        cut_d = max(1, min(dim - 1, int(round((1.0 - lam) * dim)))) if dim > 1 else 0
        s0 = int(torch.randint(0, max(1, dim - cut_d + 1), (1,), device=x.device).item())
        if cut_d > 0:
            x_mix[:, s0:s0 + cut_d] = x[perm, s0:s0 + cut_d]
        lam_ex = 1.0 - float(cut_d) / float(max(1, dim))
    y_mix = lam_ex * y_soft + (1.0 - lam_ex) * y_soft[perm]
    target_cmp = y_hard if lam_ex >= 0.5 else y_hard[perm]
    return x_mix, y_mix, target_cmp


class UnifiedClassificationLoss(nn.Module):
    """Cross-entropy loss supporting smoothing, focal gamma, logit adjustment, margin, weights, Mixup, and CutMix."""

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        logit_adj_tau: float = 0.0,
        log_priors: Optional[torch.Tensor] = None,
        logit_margin: float = 0.0,
        loss_cap_tau: float = 0.0,
        loss_cap_alpha: float = 0.0,
        loss_cap_mode: str = "fixed",
        loss_cap_param: float = 1.75,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.logit_adj_tau = float(logit_adj_tau)
        self.logit_margin = float(logit_margin)
        self.loss_cap_tau = float(loss_cap_tau)
        self.loss_cap_alpha = float(loss_cap_alpha)
        self.loss_cap_mode = str(loss_cap_mode).lower()
        self.loss_cap_param = float(loss_cap_param)
        if self.loss_cap_mode in ("mad", "quantile") or self.loss_cap_tau > 0.0:
            self.cap_gate: Optional[AdaptiveBoundedLossGate] = AdaptiveBoundedLossGate(
                mode=self.loss_cap_mode,
                tau=self.loss_cap_tau,
                alpha=self.loss_cap_alpha,
                param=self.loss_cap_param,
            )
        else:
            self.cap_gate = None
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.detach().float().clone())
        else:
            self.class_weights = None
        if log_priors is not None:
            self.register_buffer("log_priors", log_priors.detach().float().clone())
        else:
            self.log_priors = None

    def _smooth_targets(self, y: torch.Tensor) -> torch.Tensor:
        y_oh = F.one_hot(y.long(), self.num_classes).float()
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            y_oh = y_oh * (1.0 - eps) + (eps / float(self.num_classes))
        return y_oh

    def loss_from_soft_targets(self, logits: torch.Tensor, y_soft: torch.Tensor, apply_prior: bool = True) -> torch.Tensor:
        eff_logits = logits
        if apply_prior and self.log_priors is not None and self.logit_adj_tau != 0.0:
            eff_logits = eff_logits + self.logit_adj_tau * self.log_priors.unsqueeze(0)
        use_native = is_accel_available()
        log_probs = calibrated_log_softmax(eff_logits) if use_native else F.log_softmax(eff_logits, dim=-1)
        per_cls = -y_soft * log_probs
        if self.focal_gamma > 0.0:
            probs = log_probs.exp()
            per_cls = per_cls * (1.0 - probs).clamp(min=1e-7).pow(self.focal_gamma)
        if self.class_weights is not None:
            per_cls = per_cls * self.class_weights.unsqueeze(0)
        sample_loss = per_cls.sum(dim=-1)
        if self.cap_gate is not None:
            sample_loss = self.cap_gate(sample_loss)
        return sample_loss.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_from_soft_targets(logits, self._smooth_targets(targets), apply_prior=False)

    def _compute_train_loss(self, logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
        adj = logits - (self.logit_margin * y_soft) if self.logit_margin > 0.0 else logits
        return self.loss_from_soft_targets(adj, y_soft, apply_prior=True)

    def forward_batch(
        self,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
    ) -> Tuple[torch.Tensor, int]:
        y_oh = self._smooth_targets(y)
        use_cut = cutmix_alpha > 0.0 and (mixup_alpha <= 0.0 or bool((torch.rand(()) < 0.5).item()))
        if use_cut:
            x_mix, y_mix, target_cmp = apply_cutmix(x, y_oh, y, float(cutmix_alpha))
            logits = forward_fn(x_mix)
            return self._compute_train_loss(logits, y_mix), int((logits.argmax(dim=1) == target_cmp).sum().item())
        if mixup_alpha <= 0.0:
            logits = forward_fn(x)
            return self._compute_train_loss(logits, y_oh), int((logits.argmax(dim=1) == y).sum().item())
        lam = torch._sample_dirichlet(torch.tensor([mixup_alpha, mixup_alpha], dtype=torch.float32))[0].to(device=x.device)
        perm = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1.0 - lam) * x[perm]
        y_mix = lam * y_oh + (1.0 - lam) * y_oh[perm]
        logits = forward_fn(x_mix)
        loss = self._compute_train_loss(logits, y_mix)
        target_cmp = torch.where(lam >= 0.5, y, y[perm])
        return loss, int((logits.argmax(dim=1) == target_cmp).sum().item())
