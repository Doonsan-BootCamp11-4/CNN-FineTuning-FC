"""Shared base configuration and single-axis experiment definitions (LR, Scheduler, Transform, Loss, Activation)."""

from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision import transforms

from configs.base import BASE
from fcft.data import IMAGENET_MEAN, IMAGENET_STD


def make_adam(lr: float, weight_decay: float = 0.0) -> Callable:
    from fcft.grid import OptimizerFactory

    return OptimizerFactory(mode="adam", lr=lr, weight_decay=weight_decay, no_bias_decay=False)


def make_adamw(lr: float, weight_decay: float = 1e-4, no_bias_decay: bool = False) -> Callable:
    from fcft.grid import OptimizerFactory

    return OptimizerFactory(mode="adamw", lr=lr, weight_decay=weight_decay, no_bias_decay=no_bias_decay)


def make_sgd(lr: float, momentum: float = 0.9, weight_decay: float = 1e-4, nesterov: bool = True) -> Callable:
    from fcft.grid import OptimizerFactory

    return OptimizerFactory(mode="sgd", lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)


def make_rmsprop(lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 1e-4) -> Callable:
    from fcft.grid import OptimizerFactory

    return OptimizerFactory(mode="rmsprop", lr=lr, weight_decay=weight_decay, momentum=momentum)


def make_nadam(lr: float = 1e-3, weight_decay: float = 1e-4) -> Callable:
    from fcft.grid import OptimizerFactory

    return OptimizerFactory(mode="nadam", lr=lr, weight_decay=weight_decay)


def make_radam(lr: float = 1e-3, weight_decay: float = 1e-4) -> Callable:
    from fcft.grid import OptimizerFactory

    return OptimizerFactory(mode="radam", lr=lr, weight_decay=weight_decay)


def make_step_lr(step_size: int = 5, gamma: float = 0.1) -> Callable:
    from fcft.grid import SchedulerFactory

    return SchedulerFactory(kind="step", step_size=step_size, gamma=gamma)


def make_multistep_lr(milestones=(5, 10), gamma: float = 0.2) -> Callable:
    from fcft.grid import SchedulerFactory

    return SchedulerFactory(kind="multistep", milestones=tuple(milestones), gamma=gamma)


def make_cosine_lr(t_max: Optional[int] = None, eta_min: float = 1e-6) -> Callable:
    from fcft.grid import SchedulerFactory

    total = int(t_max if t_max is not None else BASE["epochs"])
    return SchedulerFactory(kind="cosine", total_epochs=total, warmup_epochs=0, eta_min=eta_min)


def make_warmup_cosine_lr(
    warmup_epochs: int = 3, total_epochs: Optional[int] = None, start_factor: float = 0.1, eta_min: float = 1e-6,
) -> Callable:
    from fcft.grid import SchedulerFactory

    total = int(total_epochs if total_epochs is not None else BASE["epochs"])
    return SchedulerFactory(kind="warmup_cosine", total_epochs=total, warmup_epochs=warmup_epochs, start_factor=start_factor, eta_min=eta_min)


def make_plateau_lr(factor: float = 0.5, patience: int = 2, min_lr: float = 1e-6) -> Callable:
    from fcft.grid import SchedulerFactory

    return SchedulerFactory(kind="plateau", gamma=factor, patience=patience, eta_min=min_lr)


def build_train_transform(*extra_ops) -> transforms.Compose:
    """Compose 224x224 ImageNet-normalized transform with optional augmentations."""
    return transforms.Compose([transforms.Resize((224, 224)), *extra_ops, transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _std_entry(name: str, optimizer=None, scheduler=None, train_transform=None) -> dict:
    return {"config": {**BASE, "name": name}, "optimizer": optimizer, "scheduler": scheduler, "train_transform": train_transform}


LR_EXPERIMENTS: List[dict] = [
    _std_entry("lr_adam_3e-4", optimizer=make_adam(lr=3e-4)),
    _std_entry("lr_adam_1e-3", optimizer=make_adam(lr=1e-3)),
    _std_entry("lr_adamw_1e-4_wd1e-4", optimizer=make_adamw(lr=1e-4, weight_decay=1e-4)),
    _std_entry("lr_adamw_3e-4_wd1e-4", optimizer=make_adamw(lr=3e-4, weight_decay=1e-4)),
    _std_entry("lr_adamw_1e-3_wd1e-4", optimizer=make_adamw(lr=1e-3, weight_decay=1e-4)),
    _std_entry("lr_adamw_3e-3_wd1e-2", optimizer=make_adamw(lr=3e-3, weight_decay=1e-2)),
    _std_entry("lr_adamw_1e-2_wd1e-2", optimizer=make_adamw(lr=1e-2, weight_decay=1e-2)),
    _std_entry("lr_adamw_1e-3_wd5e-2", optimizer=make_adamw(lr=1e-3, weight_decay=5e-2)),
    _std_entry("lr_adamw_1e-3_nobias_wd1e-2", optimizer=make_adamw(lr=1e-3, weight_decay=1e-2, no_bias_decay=True)),
    _std_entry("lr_sgd_1e-2_nesterov", optimizer=make_sgd(lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)),
    _std_entry("lr_sgd_5e-2_momentum", optimizer=make_sgd(lr=5e-2, momentum=0.9, weight_decay=1e-4, nesterov=False)),
    _std_entry("lr_sgd_5e-2_nesterov", optimizer=make_sgd(lr=5e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)),
    _std_entry("lr_sgd_1e-1_nesterov", optimizer=make_sgd(lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=True)),
    _std_entry("lr_rmsprop_1e-3_m0.9", optimizer=make_rmsprop(lr=1e-3, momentum=0.9, weight_decay=1e-4)),
    _std_entry("lr_nadam_1e-3_wd1e-4", optimizer=make_nadam(lr=1e-3, weight_decay=1e-4)),
    _std_entry("lr_radam_1e-3_wd1e-4", optimizer=make_radam(lr=1e-3, weight_decay=1e-4)),
]

SCHEDULER_EXPERIMENTS: List[dict] = [
    _std_entry("sched_none"),
    _std_entry("sched_steplr_s5_g0.1", scheduler=make_step_lr(step_size=5, gamma=0.1)),
    _std_entry("sched_multistep_5_10", scheduler=make_multistep_lr(milestones=(5, 10), gamma=0.2)),
    _std_entry("sched_cosine", scheduler=make_cosine_lr(t_max=BASE["epochs"], eta_min=1e-6)),
    _std_entry("sched_warmup_cosine", scheduler=make_warmup_cosine_lr(warmup_epochs=3, total_epochs=BASE["epochs"], eta_min=1e-6)),
    _std_entry("sched_plateau_f0.5_p2", scheduler=make_plateau_lr(factor=0.5, patience=2, min_lr=1e-6)),
]

TRANSFORM_EXPERIMENTS: List[dict] = [
    _std_entry("tf_baseline", train_transform=build_train_transform()),
    _std_entry("tf_hflip", train_transform=build_train_transform(transforms.RandomHorizontalFlip(p=0.5))),
    _std_entry("tf_flip_rot15", train_transform=build_train_transform(transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(15))),
    _std_entry("tf_color_jitter", train_transform=build_train_transform(transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))),
    _std_entry("tf_resize256_crop224", train_transform=transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])),
    _std_entry("tf_trivial_augment", train_transform=build_train_transform(transforms.TrivialAugmentWide())),
]

LOSS_SPECS: List[Tuple[str, str, str]] = [
    ("loss_ce", "ce", "표준 CrossEntropy (규제·가중치 미적용 기준선)"),
    ("loss_ce_ls0.05", "ce_ls0.05", "CrossEntropy + LabelSmoothing(0.05)"),
    ("loss_ce_ls0.1", "ce_ls0.1", "CrossEntropy + LabelSmoothing(0.10)"),
    ("loss_focal_g1.5", "focal_g1.5", "Focal Loss (gamma=1.5, 어려운 샘플 집중)"),
    ("loss_focal_g2.0_ls", "focal_g2.0_ls", "Focal Loss (gamma=2.0) + LabelSmoothing(0.05)"),
    ("loss_cw_inv_sqrt", "cw_inv_sqrt", "역제곱근 클래스 가중치(inv_sqrt) + LabelSmoothing(0.05)"),
    ("loss_cw_effective", "cw_effective", "Effective Number 클래스 가중치 + LabelSmoothing(0.05)"),
    ("loss_logit_adj_tau0.5", "logit_adj_tau0.5", "Logit-Adjusted CE (tau=0.5, 장꼬리 불균형 보정) + LS(0.05)"),
    ("loss_logit_adj_margin", "logit_adj_margin", "Logit-Adjusted CE (tau=0.5) + Margin(0.1) + LS(0.05)"),
    ("loss_resilient_ce", "resilient_ce", "상한 포화 CrossEntropy (노이즈·이상치 기울기 억제, τ=1.0, α=0.5) + LS(0.05)"),
    ("loss_resilient_cw", "resilient_cw", "상한 포화 CrossEntropy + Effective 클래스 가중치 (τ=0.8, α=0.5) + LS(0.05)"),
    ("loss_resilient_auto", "resilient_auto", "분포 자기조절 포화(MAD) + 학습형 SmoothActivation(learn_leak) + CW(eff)"),
]

ACTIVATION_SPECS: List[Tuple[str, str, str]] = [
    ("act_relu", "relu", "표준 ReLU 기준선 (MLP 256D 은닉층)"),
    ("act_leaky_relu", "leaky_relu", "LeakyReLU (음수 기울기 0.0625)"),
    ("act_elu", "elu", "ELU (지수 선형 유닛)"),
    ("act_gelu", "gelu", "GELU (가우시안 오차 선형 유닛)"),
    ("act_silu", "silu", "SiLU / Swish (x * sigmoid(x))"),
    ("act_mish", "mish", "Mish (x * tanh(softplus(x)))"),
    ("act_smooth", "smooth", "SmoothActivation (연속 비대칭 게이트)"),
]


def _build_loss_experiments() -> List[dict]:
    from fcft.grid import OptimizerFactory, SchedulerFactory, resolve_loss_preset

    ep = int(BASE["epochs"])
    out = []
    for name, loss_mode, desc in LOSS_SPECS:
        extra_tk = dict(resolve_loss_preset(loss_mode))
        if loss_mode == "resilient_auto":
            extra_tk["head_kwargs"] = {"learnable_leak": True}
        out.append({
            "config": {**BASE, "name": name}, "trainer": "cached", "description": desc,
            "optimizer": OptimizerFactory("adamw", lr=2e-3, weight_decay=1e-2, no_bias_decay=True),
            "scheduler": SchedulerFactory("warmup_cosine", total_epochs=ep, warmup_epochs=2),
            "train_transform": None,
            "train_kwargs": {
                "pool_mode": "norm_multiscale", "head_type": "residual", "feature_mode": "backbone",
                "feature_norm": True, "feature_norm_mode": "center_power_l2", "warm_start_ridge": 0.5,
                "head_lr_scale": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0, "ema_decay": 0.99,
                "swa_epochs": 4, "tta_extract": True, "compress_cache": "auto", **extra_tk,
            },
            "eval_tta_mode": "flip2", "use_aug_bank": True, "aug_repeats": 3, "loss_mode": loss_mode,
        })
    return out


def _build_activation_experiments() -> List[dict]:
    from fcft.grid import OptimizerFactory, SchedulerFactory, resolve_loss_preset

    ep = int(BASE["epochs"])
    out = []
    for name, act, desc in ACTIVATION_SPECS:
        out.append({
            "config": {**BASE, "name": name}, "trainer": "cached", "description": desc,
            "optimizer": OptimizerFactory("adamw", lr=2e-3, weight_decay=1e-2, no_bias_decay=True),
            "scheduler": SchedulerFactory("warmup_cosine", total_epochs=ep, warmup_epochs=2),
            "train_transform": None,
            "train_kwargs": {
                "pool_mode": "norm_multiscale", "head_type": "mlp", "head_kwargs": {"activation": act},
                "feature_mode": "backbone", "feature_norm": True, "feature_norm_mode": "center_power_l2",
                "warm_start_ridge": 0.5, "mixup_alpha": 0.2, "cutmix_alpha": 1.0, "ema_decay": 0.99,
                "swa_epochs": 4, "tta_extract": True, "compress_cache": "auto", **resolve_loss_preset("logit_adj_tau0.5"),
            },
            "eval_tta_mode": "flip2", "use_aug_bank": True, "aug_repeats": 3, "activation": act,
        })
    return out


LOSS_EXPERIMENTS: List[dict] = _build_loss_experiments()
ACTIVATION_EXPERIMENTS: List[dict] = _build_activation_experiments()
LOSS_DESCRIPTIONS: Dict[str, str] = {n: d for n, _, d in LOSS_SPECS}
ACTIVATION_DESCRIPTIONS: Dict[str, str] = {n: d for n, _, d in ACTIVATION_SPECS}
