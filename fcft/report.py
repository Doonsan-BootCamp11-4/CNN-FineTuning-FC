"""Experiment summary tables, CSV export, and comparison plots for team reports."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_one_summary(exp_dir: Path) -> Optional[dict]:
    hist_path = exp_dir / "history.json"
    if not hist_path.is_file():
        return None
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    if not history:
        return None
    best_row = max(history, key=lambda r: (float(r["val_acc"]), -float(r["val_loss"])))
    last_row = history[-1]
    metrics_path = exp_dir / "metrics.json"
    test_acc, macro_f1 = None, None
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_acc = float(metrics.get("accuracy", 0.0))
        macro_f1 = float(metrics.get("macro_avg", {}).get("f1-score", 0.0))
    return {
        "name": exp_dir.name,
        "epochs_done": len(history),
        "best_val_acc": float(best_row["val_acc"]),
        "best_epoch": int(best_row["epoch"]),
        "final_val_loss": float(last_row["val_loss"]),
        "final_lr": float(last_row["lr"]),
        "test_acc": test_acc,
        "macro_f1": macro_f1,
    }


def collect_summaries(out_dir: str | Path, names: Optional[Sequence[str]] = None) -> List[dict]:
    """Collect and rank completed experiments under `out_dir` by best validation accuracy."""
    root = Path(out_dir)
    if not root.exists():
        return []
    dirs = [root / n for n in names] if names is not None else sorted(p for p in root.iterdir() if p.is_dir())
    records = []
    for d in dirs:
        rec = _load_one_summary(d)
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: (-r["best_val_acc"], r["final_val_loss"], r["name"]))
    return records


def format_markdown_table(records: Sequence[dict]) -> str:
    """Format experiment summary records as a GitHub-flavored Markdown table."""
    lines = [
        "| Experiment | Epochs | Best Val Acc | Best Epoch | Final Val Loss | Final LR | Test Acc | Macro F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        t_acc = f"{r['test_acc']:.4f}" if r["test_acc"] is not None else "-"
        m_f1 = f"{r['macro_f1']:.4f}" if r["macro_f1"] is not None else "-"
        lines.append(
            f"| `{r['name']}` | {r['epochs_done']} | {r['best_val_acc']:.4f} | {r['best_epoch']} "
            f"| {r['final_val_loss']:.4f} | {r['final_lr']:.6g} | {t_acc} | {m_f1} |"
        )
    return "\n".join(lines) + "\n"


def format_before_after_table(
    records: Sequence[dict],
    baseline_name: str = "compare_0_baseline",
    descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Format Before-vs-After progression rows with %p validation accuracy gain over baseline."""
    desc_map = descriptions or {}
    base_rec = next((r for r in records if r["name"] == baseline_name), None)
    base_acc = float(base_rec["best_val_acc"]) if base_rec is not None else None
    lines = [
        "| Preset | Epochs | Best Val Acc | Δ Val Acc (%p) | Final Val Loss | What Changed |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in records:
        if base_acc is None:
            delta_str = "-"
        elif r["name"] == baseline_name:
            delta_str = "0.00%p"
        else:
            delta_pp = (float(r["best_val_acc"]) - base_acc) * 100.0
            delta_str = f"{delta_pp:+.2f}%p"
        note = desc_map.get(r["name"], "-").replace("|", "/")
        lines.append(
            f"| `{r['name']}` | {r['epochs_done']} | {r['best_val_acc']:.4f} | {delta_str} "
            f"| {r['final_val_loss']:.4f} | {note} |"
        )
    return "\n".join(lines) + "\n"


def save_summary_csv(records: Sequence[dict], csv_path: str | Path) -> Path:
    """Export experiment summary records to CSV."""
    dst = Path(csv_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "epochs_done", "best_val_acc", "best_epoch", "final_val_loss", "final_lr", "test_acc", "macro_f1"]
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fields})
    return dst


def plot_comparison_curves(
    out_dir: str | Path,
    names: Sequence[str],
    out_png: str | Path,
    title: str = "Experiment Comparison",
) -> Path:
    """Plot validation accuracy and loss curves across multiple experiments on one figure."""
    root = Path(out_dir)
    dst = Path(out_png)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False
    for name in names:
        hist_path = root / name / "history.json"
        if not hist_path.is_file():
            continue
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        if not history:
            continue
        epochs = [r["epoch"] for r in history]
        ax_acc.plot(epochs, [r["val_acc"] for r in history], marker="o", markersize=3, label=name)
        ax_loss.plot(epochs, [r["val_loss"] for r in history], marker="o", markersize=3, label=name)
        plotted = True
    ax_acc.set_title(f"{title} — Val Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_loss.set_title(f"{title} — Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    if plotted:
        ax_acc.legend(fontsize=8)
        ax_loss.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    return dst


def select_best_by_axis(out_dir: str | Path) -> Dict[str, Optional[dict]]:
    """Return the highest best_val_acc record for each experiment axis prefix."""
    records = collect_summaries(out_dir)
    prefixes = {
        "lr": "lr_",
        "scheduler": "sched_",
        "transform": "tf_",
        "loss": "loss_",
        "activation": "act_",
        "compare": "compare_",
        "grid": "grid_",
    }
    best: Dict[str, Optional[dict]] = {}
    for axis, pref in prefixes.items():
        cands = [r for r in records if r["name"].startswith(pref)]
        best[axis] = cands[0] if cands else None
    return best
