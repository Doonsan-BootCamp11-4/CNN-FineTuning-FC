import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from fcft.data import build_loaders
from fcft.model import build_model

# pyplot 대신 Figure를 직접 만들어 저장만 한다 (화면 표시 backend가 필요 없음).


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    """loader 전체에 대해 (정답 목록, 예측 목록)을 돌려준다."""
    model.eval()
    
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    return y_true, y_pred


def _save_confusion_matrix(cm: list[list[int]], classes: list[str], path: Path) -> None:
    fig = Figure(figsize=(9, 9))
    ax = fig.subplots()
    ConfusionMatrixDisplay(np.array(cm), display_labels=classes).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(path)


def plot_history(out_dir: Path) -> None:
    """history.json → curves.png (왼쪽 loss, 오른쪽 acc, 각각 train/val)."""
    history = json.loads((out_dir / "history.json").read_text())
    epochs = [h["epoch"] for h in history]

    fig = Figure(figsize=(10, 4))
    for ax, key in zip(fig.subplots(1, 2), ("loss", "acc")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="val")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png")


def evaluate(config: dict) -> dict:
    """<out_dir>/<name>/best.pt로 test를 평가한다.

    저장: metrics.json(accuracy, 클래스별 precision/recall/F1, confusion matrix),
          confusion_matrix.png, curves.png
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["out_dir"]) / config["name"]
    loader = build_loaders(config)["test"]
    classes = loader.dataset.classes

    model = build_model(config["model"], config["num_classes"]).to(device)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))

    y_true, y_pred = _predict(model, loader, device)

    labels = list(range(len(classes)))  # test에 없는 클래스가 있어도 행·열을 모두 만든다
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    metrics = {"accuracy": report["accuracy"], "per_class": report, "confusion_matrix": cm}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _save_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")
    plot_history(out_dir)
    return metrics
