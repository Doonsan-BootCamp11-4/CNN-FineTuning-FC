"""Beginner-friendly incremental Before-vs-After comparison presets for CNN-FineTuning-FC."""

from typing import Callable, List, Optional
from torchvision import transforms

from configs.axes import BASE
from fcft.grid import OptimizerFactory, SchedulerFactory

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _rich_train_transform() -> transforms.Compose:
    """Standard beginner-friendly augmentation pipeline matching 224x224 ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _adamw_nobias(lr: float = 1e-3, weight_decay: float = 1e-2) -> Callable:
    """Create a picklable AdamW factory applying weight decay only to 2D+ weight tensors."""
    return OptimizerFactory(mode="adamw", lr=lr, weight_decay=weight_decay, no_bias_decay=True)


def _cosine_scheduler(t_max: int = 15, eta_min: float = 1e-5) -> Callable:
    """Create a picklable CosineAnnealingLR factory over `t_max` epochs."""
    return SchedulerFactory(kind="cosine", total_epochs=t_max, warmup_epochs=0, eta_min=eta_min)


def _make_entry(
    name: str,
    description: str,
    trainer: str = "standard",
    optimizer: Optional[Callable] = None,
    scheduler: Optional[Callable] = None,
    train_transform=None,
    train_kwargs: Optional[dict] = None,
    eval_tta_mode: str = "none",
) -> dict:
    """Build one validated comparison preset entry with BASE config isolation."""
    return {
        "config": {**BASE, "name": name},
        "trainer": trainer,
        "description": description,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_transform": train_transform,
        "train_kwargs": dict(train_kwargs or {}),
        "eval_tta_mode": eval_tta_mode,
    }


EXPERIMENTS: List[dict] = [
    _make_entry(
        name="compare_0_baseline",
        description="Before 기준선: ResNet18(AvgPool) + Linear FC + Adam(1e-3), 증강·규제 미적용",
        trainer="cached",
        train_kwargs={"compress_cache": False},
        eval_tta_mode="none",
    ),
    _make_entry(
        name="compare_1_aug_cosine",
        description="+데이터증강·스케줄러: Flip/Rot15/Jitter + AdamW(wd=1e-2) + CosineLR + LabelSmoothing(0.05)",
        trainer="standard",
        optimizer=_adamw_nobias(lr=1e-3, weight_decay=1e-2),
        scheduler=_cosine_scheduler(t_max=int(BASE["epochs"])),
        train_transform=_rich_train_transform(),
        train_kwargs={"label_smoothing": 0.05},
        eval_tta_mode="none",
    ),
    _make_entry(
        name="compare_2_norm_cosinehead",
        description="+피처정규화·코사인헤드·균형가중치: AvgMax(1024D) + CosineHead(lr=3e-3) + WarmStart + InvSqrt",
        trainer="cached",
        optimizer=_adamw_nobias(lr=3e-3, weight_decay=1e-2),
        scheduler=_cosine_scheduler(t_max=int(BASE["epochs"])),
        train_kwargs={
            "pool_mode": "avgmax",
            "head_type": "cosine",
            "feature_norm": True,
            "label_smoothing": 0.05,
            "warm_start_ridge": 10.0,
            "class_weight_mode": "inv_sqrt",
            "tta_extract": True,
            "compress_cache": True,
        },
        eval_tta_mode="flip2",
    ),
    _make_entry(
        name="compare_3_mixup_ema",
        description="+과적합억제: Mixup(0.2) + ModelEMA(0.99) 추가 (AvgMax + CosineHead + WarmStart)",
        trainer="cached",
        optimizer=_adamw_nobias(lr=3e-3, weight_decay=1e-2),
        scheduler=_cosine_scheduler(t_max=int(BASE["epochs"])),
        train_kwargs={
            "pool_mode": "avgmax",
            "head_type": "cosine",
            "feature_norm": True,
            "label_smoothing": 0.05,
            "mixup_alpha": 0.2,
            "ema_decay": 0.99,
            "warm_start_ridge": 10.0,
            "class_weight_mode": "inv_sqrt",
            "tta_extract": True,
            "compress_cache": True,
        },
        eval_tta_mode="flip2",
    ),
    _make_entry(
        name="compare_4_cached_residual",
        description="+고속캐시·잔차헤드·초기해: GeMPool + ResidualAdapter + Ridge Warm-Start + 압축피처 캐시",
        trainer="cached",
        optimizer=_adamw_nobias(lr=1e-3, weight_decay=1e-2),
        scheduler=_cosine_scheduler(t_max=int(BASE["epochs"])),
        train_kwargs={
            "pool_mode": "gem",
            "head_type": "residual",
            "feature_norm": True,
            "label_smoothing": 0.05,
            "mixup_alpha": 0.2,
            "ema_decay": 0.99,
            "warm_start_ridge": 10.0,
            "class_weight_mode": "effective",
            "tta_extract": True,
            "compress_cache": True,
        },
        eval_tta_mode="flip2",
    ),
    _make_entry(
        name="compare_5_multiscale_rich",
        description="+다중스케일·서브센터·Focal·4뷰TTA: Multiscale(768D) + SubCenter(K=3, lr=3e-3) + Focal(1.5) + 4뷰 TTA",
        trainer="cached",
        optimizer=_adamw_nobias(lr=3e-3, weight_decay=1e-2),
        scheduler=_cosine_scheduler(t_max=int(BASE["epochs"])),
        train_kwargs={
            "pool_mode": "multiscale",
            "head_type": "subcenter",
            "head_kwargs": {"num_centers": 3},
            "feature_norm": True,
            "label_smoothing": 0.05,
            "mixup_alpha": 0.2,
            "ema_decay": 0.99,
            "warm_start_ridge": 10.0,
            "class_weight_mode": "inv_sqrt",
            "focal_gamma": 1.5,
            "tta_extract": True,
            "compress_cache": True,
        },
        eval_tta_mode="multiview4",
    ),
    _make_entry(
        name="compare_6_pure_descriptor",
        description="+백본프리 서술자: CNN 없이 Dual-Scale 서브픽셀 서술자(128D) + Flip증강 + ResidualAdapter + Ridge",
        trainer="cached",
        optimizer=_adamw_nobias(lr=3e-3, weight_decay=1e-2),
        scheduler=_cosine_scheduler(t_max=int(BASE["epochs"])),
        train_kwargs={
            "feature_mode": "descriptor",
            "head_type": "residual",
            "feature_norm": True,
            "label_smoothing": 0.05,
            "ema_decay": 0.99,
            "warm_start_ridge": 5.0,
            "class_weight_mode": "inv_sqrt",
            "tta_extract": True,
            "compress_cache": True,
        },
        eval_tta_mode="flip2",
    ),
    _make_entry(
        name="compare_7_hybrid_fusion",
        description="+하이브리드 융합: ResNet18(AvgMax 1024D) + 서브픽셀 서술자(128D) + SubCenter(K=3) + Focal(1.5) + 4뷰 TTA",
        trainer="cached",
        optimizer=_adamw_nobias(lr=3e-3, weight_decay=1e-2),
        scheduler=_cosine_scheduler(t_max=int(BASE["epochs"])),
        train_kwargs={
            "feature_mode": "hybrid",
            "pool_mode": "avgmax",
            "head_type": "subcenter",
            "head_kwargs": {"num_centers": 3},
            "feature_norm": True,
            "label_smoothing": 0.05,
            "mixup_alpha": 0.2,
            "ema_decay": 0.99,
            "warm_start_ridge": 10.0,
            "class_weight_mode": "inv_sqrt",
            "focal_gamma": 1.5,
            "tta_extract": True,
            "compress_cache": True,
        },
        eval_tta_mode="multiview4",
    ),
]
