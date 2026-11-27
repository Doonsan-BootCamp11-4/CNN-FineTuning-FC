"""Shared pytest fixtures creating a temporary 12-image stratified fake dataset."""

import csv
import sys
from pathlib import Path
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.axes import BASE  # noqa: E402


@pytest.fixture
def fake_dataset(tmp_path: Path) -> dict:
    """Create 12 synthetic images (3 classes x 4, including 1 grayscale) + split CSV and config."""
    data_root = tmp_path / "images"
    out_dir = tmp_path / "outputs"
    csv_path = tmp_path / "split.csv"

    # Unsorted order intentionally verifies alphabetical class sorting in build_loaders.
    classes = ["paper", "cardboard", "glass"]
    splits_per_class = ["train", "train", "val", "test"]
    rows = []

    for cls_idx, cls_name in enumerate(classes):
        cls_dir = data_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j, split in enumerate(splits_per_class):
            img_name = f"{cls_name}_{j}.png"
            img_path = cls_dir / img_name
            if cls_name == "glass" and j == 0:
                img = Image.new("L", (32, 32), color=80 + cls_idx * 40)
            else:
                img = Image.new("RGB", (32, 32), color=(40 * (cls_idx + 1), 30 * (j + 1), 120))
            img.save(img_path)
            rows.append({"path": f"{cls_name}/{img_name}", "label": cls_name, "split": split})

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    config = {
        **BASE,
        "name": "test_run",
        "split_csv": str(csv_path),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "num_classes": 3,
        "num_workers": 0,
        "epochs": 1,
        "batch_size": 2,
        "seed": 42,
    }
    return {
        "config": config,
        "csv_path": csv_path,
        "data_root": data_root,
        "out_dir": out_dir,
        "classes_sorted": sorted(classes),
        "grayscale_rel": "glass/glass_0.png",
    }
