"""Public package interface for frozen-backbone ResNet18 FC fine-tuning pipeline."""

# 1. Protected Acceleration Core API (binary + vault backed)
from fcft.accel import (
    ACTIVATION_CHOICES,
    CompressedFeatureBank,
    CurvatureBalancedOptimizer,
    FusedLinearHeadSession,
    ResidualAdapterHead,
    SmoothActivation,
    build_activation,
    fit_balanced_ridge_head,
    fit_closed_form_head,
    is_accel_available,
    multihead_linear_forward,
    smooth_activation,
)

# 2. Dataset, Split Validation & Shared Disk Feature Cache
from fcft.data import (
    EVAL_TRANSFORM,
    GarbageSplitDataset,
    MultiPoolResNet18,
    assemble_from_bundle,
    build_loaders,
    extract_shared_feature_bundle,
    inspect_split,
    predict_logits_from_test_bundle,
)

# 2.5. Declarative Grid & Single Custom Runner
from fcft.grid import build_custom_entry, expand_grid, run_single_experiment

# 3. Evaluation, TTA & Multi-Checkpoint Ensemble
from fcft.evaluate import evaluate, evaluate_ensemble, load_checkpoint_model, predict_logits

# 4. Loss Functions & Imbalance Weighting
from fcft.losses import UnifiedClassificationLoss, compute_class_weights

# 5. Frozen Backbone, Normalization & Classifier Heads
from fcft.model import (
    AvgMaxPool2d,
    CosineClassifierHead,
    DualAnchorHybridHead,
    FeatureCenterNormLayer,
    FeatureNormLayer,
    GeMPool2d,
    MLPClassifierHead,
    MultiScaleResNet18,
    SpectralNormLayer,
    SubCenterHead,
    build_backbone,
    build_head,
    build_model,
    fuse_backbone_conv_bn,
)

# 6. Summary Tables, CSV Export & Comparison Plots
from fcft.report import (
    collect_summaries,
    format_markdown_table,
    plot_comparison_curves,
    save_summary_csv,
    select_best_by_axis,
)

# 7. Standard & Cached FC Trainers
from fcft.train import ModelEMA, ModelSWA, co_train_heads_cached, run, run_cached, run_once_epoch

__all__ = [
    "ACTIVATION_CHOICES",
    "AvgMaxPool2d",
    "CompressedFeatureBank",
    "CosineClassifierHead",
    "CurvatureBalancedOptimizer",
    "DualAnchorHybridHead",
    "EVAL_TRANSFORM",
    "FeatureCenterNormLayer",
    "FeatureNormLayer",
    "FusedLinearHeadSession",
    "GarbageSplitDataset",
    "GeMPool2d",
    "MLPClassifierHead",
    "ModelEMA",
    "ModelSWA",
    "MultiPoolResNet18",
    "MultiScaleResNet18",
    "ResidualAdapterHead",
    "SmoothActivation",
    "SpectralNormLayer",
    "SubCenterHead",
    "UnifiedClassificationLoss",
    "assemble_from_bundle",
    "build_activation",
    "build_backbone",
    "build_custom_entry",
    "build_head",
    "build_loaders",
    "build_model",
    "co_train_heads_cached",
    "collect_summaries",
    "compute_class_weights",
    "evaluate",
    "evaluate_ensemble",
    "expand_grid",
    "extract_shared_feature_bundle",
    "fit_balanced_ridge_head",
    "fit_closed_form_head",
    "format_markdown_table",
    "fuse_backbone_conv_bn",
    "inspect_split",
    "is_accel_available",
    "load_checkpoint_model",
    "multihead_linear_forward",
    "plot_comparison_curves",
    "predict_logits",
    "predict_logits_from_test_bundle",
    "run",
    "run_cached",
    "run_once_epoch",
    "run_single_experiment",
    "save_summary_csv",
    "select_best_by_axis",
    "smooth_activation",
]
