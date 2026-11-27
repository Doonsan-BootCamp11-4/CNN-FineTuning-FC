"""Declarative Grid Search matrix of proven frozen-backbone FC techniques for teammates."""

from typing import Any, Dict, List, Sequence

from configs.axes import BASE
from fcft.grid import expand_grid

# 백본(ResNet18)은 100% 동결 상태로 유지하고, FC 분류 헤드에서만 실측 성능이 검증된 정예 기법들을 조합합니다.
# 아래 SUPPORTED_OPTIONS를 참고해 리스트 값만 추가/수정하시면 바로 그리드 서치로 비교하실 수 있습니다.
SUPPORTED_OPTIONS: Dict[str, Sequence[Any]] = {
    "pool_mode": ["avg", "norm_multiscale"],
    "head_type": ["residual", "mlp", "linear"],
    "activation": ["smooth", "smooth_learn", "relu", "leaky_relu", "elu", "gelu", "silu", "mish"],
    "feature_norm_mode": ["power_l2", "center_power_l2", "center_std", "center_std_power", "l2"],
    "loss_mode": ["ce", "ce_ls0.05", "ce_ls0.1", "focal_g1.5", "focal_g2.0_ls", "cw_inv_sqrt", "cw_effective", "logit_adj_tau0.5", "logit_adj_margin", "resilient_ce", "resilient_cw", "resilient_auto"],
    "optimizer_mode": ["adamw", "calibrated", "sgd"],
    "head_lr_scale": [1.0, 0.2, 0.0],
    "mix_mode": ["none", "mixup", "cutmix", "both"],
    "class_weight_mode": [None, "inv_sqrt", "effective"],
    "scheduler": ["cosine", "warmup_cosine", "step"],
    "logit_adj_tau": [0.0, 0.5],
    "swa_epochs": [0, 4],
    "eval_resize_mode": ["squash", "center256"],
    "aug_mode": ["rrc_jitter", "resize256_crop224", "trivial", "randaugment", "erasing", "affine"],
    "feature_noise_std": [0.0, 0.02],
    "ckpt_metric": ["val_acc", "macro_f1"],
    "model": ["resnet18", "resnet34", "resnet50"],
}

GRID: Dict[str, Sequence[Any]] = {
    "pool_mode": ["norm_multiscale"],
    "head_type": ["residual", "mlp"],
    "mix_mode": ["cutmix", "both"],
    "optimizer_mode": ["adamw", "calibrated"],
    "class_weight_mode": [None, "inv_sqrt"],
}

DEFAULTS: Dict[str, Any] = {
    "trainer": "cached",
    "feature_mode": "backbone",
    "feature_norm_mode": "power_l2",
    "eval_resize_mode": "squash",
    "aug_mode": "rrc_jitter",
    "label_smoothing": 0.05,
    "focal_gamma": 0.0,
    "logit_adj_tau": 0.5,
    "feature_dropout": 0.0,
    "lr": 2e-3,
    "weight_decay": 1e-2,
    "scheduler": "warmup_cosine",
    "warmup_epochs": 2,
    "swa_epochs": 4,
    "ema_decay": 0.99,
    "warm_start_ridge": 0.5,
    "head_lr_scale": 1.0,
    "tta_extract": True,
    "compress_cache": "auto",
    "use_aug_bank": True,
    "aug_repeats": 3,
    "eval_tta_mode": "flip2",
    "feature_noise_std": 0.0,
    "ckpt_metric": "val_acc",
}

EXPERIMENTS: List[dict] = expand_grid(GRID, base_config=BASE, defaults=DEFAULTS)
