"""Test evaluation, JSON metrics export, TTA inference, confusion matrix, and curves."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from fcft.accel import calibrated_softmax, is_accel_available
from fcft.data import build_loaders, predict_logits_from_test_bundle
from fcft.model import model_from_checkpoint


def _compute_metrics(y_true: List[int], y_pred: List[int], classes: List[str]) -> dict:
    num_classes = len(classes)
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    total = len(y_true)
    correct = sum(cm[i][i] for i in range(num_classes))
    accuracy = float(correct / total) if total > 0 else 0.0

    per_class: Dict[str, dict] = {}
    p_sum = r_sum = f_sum = 0.0
    wp_sum = wr_sum = wf_sum = 0.0

    for i, cls_name in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes)) - tp
        fn = sum(cm[i][c] for c in range(num_classes)) - tp
        sup = tp + fn
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / sup) if sup > 0 else 0.0
        f1 = float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cls_name] = {
            "precision": prec,
            "recall": rec,
            "f1-score": f1,
            "support": int(sup),
        }
        p_sum += prec
        r_sum += rec
        f_sum += f1
        wp_sum += prec * sup
        wr_sum += rec * sup
        wf_sum += f1 * sup

    top_confusions = [
        {"true": classes[i], "pred": classes[j], "count": int(cm[i][j])}
        for i in range(num_classes)
        for j in range(num_classes)
        if i != j and cm[i][j] > 0
    ]
    top_confusions.sort(key=lambda d: (-d["count"], d["true"], d["pred"]))

    c_div = max(1, num_classes)
    t_div = max(1, total)
    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_avg": {
            "precision": p_sum / c_div,
            "recall": r_sum / c_div,
            "f1-score": f_sum / c_div,
            "support": int(total),
        },
        "weighted_avg": {
            "precision": wp_sum / t_div,
            "recall": wr_sum / t_div,
            "f1-score": wf_sum / t_div,
            "support": int(total),
        },
        "confusion_matrix": cm,
        "top_confusions": top_confusions[:10],
    }


def _save_confusion_matrix(cm: List[List[int]], classes: List[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(classes) * 0.8), max(4, len(classes) * 0.7)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = list(range(len(classes)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_curves(history_path: Path, out_path: Path) -> None:
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    epochs = [row["epoch"] for row in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if epochs:
        ax1.plot(epochs, [r["train_loss"] for r in history], label="train_loss")
        ax1.plot(epochs, [r["val_loss"] for r in history], label="val_loss")
        ax2.plot(epochs, [r["train_acc"] for r in history], label="train_acc")
        ax2.plot(epochs, [r["val_acc"] for r in history], label="val_acc")
        ax1.legend()
        ax2.legend()
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def predict_logits(model, x: torch.Tensor, tta_mode: str = "none") -> torch.Tensor:
    """Compute logits with optional 2-view flip or 4-view crop+flip TTA."""
    if tta_mode == "none":
        return model(x)
    if tta_mode == "flip2":
        return 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
    if tta_mode == "multiview4":
        h, w = int(x.shape[2]), int(x.shape[3])
        mh, mw = max(1, h // 16), max(1, w // 16)
        zoom = F.interpolate(x[:, :, mh:h - mh, mw:w - mw], size=(h, w), mode="bilinear", align_corners=False)
        return 0.25 * (model(x) + model(torch.flip(x, dims=[3])) + model(zoom) + model(torch.flip(zoom, dims=[3])))
    raise ValueError(f"Unsupported tta_mode: {tta_mode!r}")


def evaluate(
    config: dict,
    tta: bool = False,
    tta_mode: Optional[str] = None,
    precomputed_test: Optional[dict] = None,
) -> dict:
    """Evaluate best checkpoint on test split (or cached test bundle) and export metrics + plots."""
    run_dir = Path(config["out_dir"]) / config["name"]
    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoint(ckpt, config, device)
    mode = tta_mode if tta_mode is not None else ("flip2" if tta else "none")

    if precomputed_test is not None and "test" in precomputed_test:
        classes = list(precomputed_test["classes"])
        y_true = [int(v) for v in precomputed_test["y_test"].tolist()]
        logits = predict_logits_from_test_bundle(
            model,
            precomputed_test["test"],
            pool_mode=ckpt.get("pool_mode", "avg"),
            feature_mode=ckpt.get("feature_mode", "backbone"),
            tta_mode=mode,
            device=device,
        )
        y_pred = logits.argmax(dim=1).cpu().tolist()
    else:
        eval_cfg = {
            **config,
            "eval_resize_mode": ckpt.get("config", {}).get(
                "eval_resize_mode", config.get("eval_resize_mode", "squash")
            ),
        }
        loaders = build_loaders(eval_cfg)
        test_loader = loaders["test"]
        classes = list(test_loader.dataset.classes)
        y_true = []
        y_pred = []
        with torch.inference_mode():
            for x, y in test_loader:
                logits = predict_logits(model, x.to(device), tta_mode=mode)
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                y_true.extend(y.tolist())

    metrics = _compute_metrics(y_true, y_pred, classes)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_confusion_matrix(metrics["confusion_matrix"], classes, run_dir / "confusion_matrix.png")
    _save_curves(run_dir / "history.json", run_dir / "curves.png")
    return metrics


def load_checkpoint_model(config: dict, device: torch.device) -> nn.Module:
    """Reconstruct a trained model from `<out_dir>/<name>/best.pt`."""
    ckpt_path = Path(config["out_dir"]) / config["name"] / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return model_from_checkpoint(ckpt, config, device)


def _normalize_weights(num_models: int, weights: Optional[Sequence[float]]) -> List[float]:
    if weights is None:
        return [1.0 / float(num_models)] * num_models
    if len(weights) != num_models:
        raise ValueError(f"Expected {num_models} ensemble weights, got {len(weights)}.")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("Sum of ensemble weights must be positive.")
    return [float(w) / total for w in weights]


def _eval_raw_ensemble_grouped(
    models: Sequence[nn.Module],
    ckpts: Sequence[dict],
    configs: Sequence[dict],
    ref_cfg: dict,
    w_norm: Sequence[float],
    tta_mode: str,
    temp: float,
    device: torch.device,
) -> Tuple[List[str], List[int], List[int]]:
    resize_modes = [
        ck.get("config", {}).get("eval_resize_mode", cfg.get("eval_resize_mode", "squash"))
        for ck, cfg in zip(ckpts, configs)
    ]
    classes, y_true, prob_accum = [], [], None
    for rm in dict.fromkeys(resize_modes):
        test_loader = build_loaders({**ref_cfg, "eval_resize_mode": rm})["test"]
        if not classes:
            classes = list(test_loader.dataset.classes)
        rm_indices = [i for i, m in enumerate(resize_modes) if m == rm]
        rm_probs, record_y = [], (not y_true)
        for x, y in test_loader:
            xd, batch_sum = x.to(device), None
            for i_m in rm_indices:
                lg = predict_logits(models[i_m], xd, tta_mode=tta_mode)
                pr = (calibrated_softmax(lg, temperature=temp) if is_accel_available() else F.softmax(lg / temp, dim=-1)) * w_norm[i_m]
                batch_sum = pr if batch_sum is None else (batch_sum + pr)
            rm_probs.append(batch_sum.cpu())
            if record_y:
                y_true.extend(y.tolist())
        rm_cat = torch.cat(rm_probs, dim=0)
        prob_accum = rm_cat if prob_accum is None else (prob_accum + rm_cat)
    return classes, y_true, prob_accum.argmax(dim=1).tolist()


@torch.inference_mode()
def evaluate_ensemble(
    configs: Sequence[dict],
    out_name: str = "ensemble",
    out_dir: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    tta_mode: str = "flip2",
    temperature: float = 1.0,
    precomputed_test: Optional[dict] = None,
) -> dict:
    """Blend calibrated softmax probabilities from multiple trained FC checkpoints."""
    if not configs:
        raise ValueError("configs must contain at least one trained experiment config.")
    w_norm = _normalize_weights(len(configs), weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_cfg = configs[0]
    ckpts = [torch.load(Path(cfg["out_dir"]) / cfg["name"] / "best.pt", map_location="cpu", weights_only=True) for cfg in configs]
    models = [model_from_checkpoint(ck, cfg, device) for ck, cfg in zip(ckpts, configs)]
    temp = max(1e-4, float(temperature))

    if precomputed_test is not None and "test" in precomputed_test:
        classes = list(precomputed_test["classes"])
        y_true = [int(v) for v in precomputed_test["y_test"].tolist()]
        prob_sum = None
        for mod, ck, wt in zip(models, ckpts, w_norm):
            lg = predict_logits_from_test_bundle(
                mod, precomputed_test["test"], pool_mode=ck.get("pool_mode", "avg"),
                feature_mode=ck.get("feature_mode", "backbone"), tta_mode=tta_mode, device=device,
            )
            probs = (calibrated_softmax(lg, temperature=temp) if is_accel_available() else F.softmax(lg / temp, dim=-1)) * wt
            prob_sum = probs if prob_sum is None else (prob_sum + probs)
        y_pred = prob_sum.argmax(dim=1).cpu().tolist()
    else:
        classes, y_true, y_pred = _eval_raw_ensemble_grouped(models, ckpts, configs, ref_cfg, w_norm, tta_mode, temp, device)

    metrics = _compute_metrics(y_true, y_pred, classes)
    save_dir = Path(out_dir if out_dir is not None else ref_cfg["out_dir"]) / out_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_confusion_matrix(metrics["confusion_matrix"], classes, save_dir / "confusion_matrix.png")
    return metrics
