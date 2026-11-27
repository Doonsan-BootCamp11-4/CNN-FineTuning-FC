"""Shared base configuration for frozen-backbone ResNet18 FC transfer learning."""

from typing import Any, Dict

BASE: Dict[str, Any] = {
    "model": "resnet18",
    "split_csv": "../CNN-FineTuning-Dataset-Split/splits/garbage12_602020_seed42.csv",
    "data_root": "/kaggle/input/garbage-classification/garbage_classification",
    "out_dir": "outputs",
    "num_classes": 12,
    "epochs": 15,
    "batch_size": 32,
    "num_workers": 2,
    "seed": 42,
}
