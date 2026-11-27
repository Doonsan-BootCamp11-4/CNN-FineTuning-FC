"""Stratified K-Fold cross-validation keeping test split strictly isolated."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import torch

from fcft.accel import extract_dual_descriptors
from fcft.data import MultiPoolResNet18, _save_bundle_atomic, build_eval_transform, GarbageSplitDataset
from fcft.model import build_backbone, build_model
from fcft.train import run, run_cached


def build_stratified_folds(
    split_csv: str | Path,
    num_classes: int,
    k_folds: int = 5,
    seed: int = 42,
    include_val: bool = True,
) -> Tuple[List[str], List[Tuple[str, int]], List[Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]]]:
    """Partition train(+val) rows into stratified K folds while keeping test untouched."""
    k = int(k_folds)
    if k < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}.")
    with Path(split_csv).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    classes = sorted({row["label"] for row in rows})
    if len(classes) != int(num_classes):
        raise ValueError(f"Expected {num_classes} classes, found {len(classes)}.")
    lbl_idx = {c: i for i, c in enumerate(classes)}

    pool_by_cls: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    test_samples: List[Tuple[str, int]] = []
    allowed = {"train", "val"} if include_val else {"train"}
    for r in rows:
        sp = r["split"].strip()
        item = (r["path"], lbl_idx[r["label"]])
        if sp in allowed:
            pool_by_cls[item[1]].append(item)
        elif sp == "test":
            test_samples.append(item)

    buckets: List[List[Tuple[str, int]]] = [[] for _ in range(k)]
    for cls_i in sorted(pool_by_cls.keys()):
        items = list(pool_by_cls[cls_i])
        random.Random(int(seed) + cls_i).shuffle(items)
        for pos, item in enumerate(items):
            buckets[pos % k].append(item)

    folds: List[Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]] = []
    for val_i in range(k):
        val_s = list(buckets[val_i])
        tr_s = [it for f_i in range(k) if f_i != val_i for it in buckets[f_i]]
        folds.append((tr_s, val_s))
    return classes, test_samples, folds


def _write_fold_csv(csv_path: Path, tr_s: List[Tuple[str, int]], va_s: List[Tuple[str, int]], te_s: List[Tuple[str, int]], classes: List[str]) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for rel, idx in tr_s:
            w.writerow([rel, classes[idx], "train"])
        for rel, idx in va_s:
            w.writerow([rel, classes[idx], "val"])
        for rel, idx in te_s:
            w.writerow([rel, classes[idx], "test"])
    return csv_path


def _fold_bundle_features(
    bundle: dict,
    tr_idx: List[int],
    va_idx: List[int],
    config: dict,
    pool_mode: str,
    feature_mode: str,
    desc_weight: float,
    tta_extract: bool,
):
    tr_map, y_all = bundle["pool"], bundle["y_pool"]
    idx_t = torch.tensor(tr_idx, dtype=torch.long)
    idx_v = torch.tensor(va_idx, dtype=torch.long)
    y_tr, y_va = y_all[idx_t], y_all[idx_v]
    if feature_mode == "backbone":
        x_tr = 0.5 * (tr_map[pool_mode][idx_t] + tr_map[f"{pool_mode}_flip"][idx_t]) if tta_extract else tr_map[pool_mode][idx_t]
        x_va = 0.5 * (tr_map[pool_mode][idx_v] + tr_map[f"{pool_mode}_flip"][idx_v]) if tta_extract else tr_map[pool_mode][idx_v]
        return (x_tr, y_tr, x_va, y_va), None

    mod = build_model(config.get("model", "resnet18"), int(config["num_classes"]), pool_mode=pool_mode, feature_mode=feature_mode, desc_weight=desc_weight)
    d_tr = torch.cat([tr_map["desc"][idx_t], tr_map["desc_flip"][idx_t]], dim=0) if tta_extract else tr_map["desc"][idx_t]
    y_out = torch.cat([y_tr, y_tr], dim=0) if tta_extract else y_tr
    mod.standardizer.fit_stats(d_tr)
    bb_tr = (torch.cat([tr_map[pool_mode][idx_t], tr_map[f"{pool_mode}_flip"][idx_t]], dim=0) if tta_extract else tr_map[pool_mode][idx_t]) if feature_mode == "hybrid" else None
    bb_va = tr_map[pool_mode][idx_v] if feature_mode == "hybrid" else None
    x_tr = mod.assemble_features(bb_tr, d_tr)
    x_va = mod.assemble_features(bb_va, tr_map["desc"][idx_v])
    return (x_tr, y_out, x_va, y_va), mod.standardizer.state_dict()


@torch.inference_mode()
def extract_kfold_pool_bundle(
    config: dict,
    pool_samples: List[Tuple[str, int]],
    classes: List[str],
    mp_backbone=None,
    include_desc: bool = True,
    use_disk_cache: bool = True,
    cache_dir: Optional[str | Path] = None,
    target_pools: Optional[Sequence[str]] = None,
) -> dict:
    """Extract pool features once across all train+val samples using channels_last and large batches."""
    eval_rm = str(config.get("eval_resize_mode", "squash"))
    pool_tag = "all" if target_pools is None else ",".join(sorted(set(target_pools)))
    h = hashlib.sha256()
    h.update(f"kf|{config.get('model', 'resnet18')}|{len(pool_samples)}|{include_desc}|{eval_rm}|{pool_tag}|{','.join(classes)}".encode("utf-8"))
    for p, lbl in pool_samples[:32]:
        h.update(f"|{p}:{lbl}".encode("utf-8"))
    c_root = Path(cache_dir) if cache_dir is not None else (Path(config.get("out_dir", "outputs")) / "_feature_cache")
    c_path = c_root / f"kfold_{h.hexdigest()[:16]}.pt"
    if use_disk_cache and c_path.is_file():
        return torch.load(c_path, map_location="cpu", weights_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mp_backbone is None:
        bb_avg, _ = build_backbone(config.get("model", "resnet18"), pool_mode="avg", fuse_conv_bn=(device.type == "cuda"))
        mp_backbone = MultiPoolResNet18(bb_avg, target_pools=target_pools)
    mp_backbone = mp_backbone.to(device)
    if device.type == "cpu":
        mp_backbone = mp_backbone.to(memory_format=torch.channels_last)

    ds = GarbageSplitDataset(pool_samples, Path(config["data_root"]), classes, build_eval_transform(eval_rm))
    bsz = max(128, int(config.get("batch_size", 32)))
    nw = int(config.get("num_workers", 2))
    pool_loader = torch.utils.data.DataLoader(
        ds, batch_size=bsz, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0),
    )
    orig_c, flip_c, do_c, df_c, ys = [], [], [], [], []
    for x, y in pool_loader:
        xd = x.to(device)
        cur_b = int(xd.shape[0])
        cat_in = torch.cat([xd, torch.flip(xd, dims=[3])], dim=0)
        if device.type == "cpu":
            cat_in = cat_in.contiguous(memory_format=torch.channels_last)
        both = mp_backbone(cat_in)
        orig_c.append({k: v[:cur_b] for k, v in both.items()})
        flip_c.append({k: v[cur_b:] for k, v in both.items()})
        ys.append(y.cpu())
        if include_desc:
            desc_both = extract_dual_descriptors(cat_in.contiguous() if device.type == "cpu" else cat_in, denormalize=True).cpu()
            do_c.append(desc_both[:cur_b])
            df_c.append(desc_both[cur_b:])
    pool = {k: torch.cat([c[k] for c in orig_c], dim=0) for k in orig_c[0]}
    for k in orig_c[0]:
        pool[f"{k}_flip"] = torch.cat([c[k] for c in flip_c], dim=0)
    if include_desc:
        pool["desc"] = torch.cat(do_c, dim=0)
        pool["desc_flip"] = torch.cat(df_c, dim=0)
    bundle = {"pool": pool, "y_pool": torch.cat(ys, dim=0), "classes": classes}
    if use_disk_cache:
        _save_bundle_atomic(c_path, bundle)
    return bundle


def run_kfold(
    config: dict,
    k_folds: int = 5,
    trainer: str = "cached",
    include_val: bool = True,
    optimizer: Optional[Callable] = None,
    scheduler: Optional[Callable] = None,
    train_transform=None,
    pool_bundle: Optional[dict] = None,
    **train_kwargs,
) -> dict:
    """Run stratified K-Fold CV and aggregate mean/std validation accuracy across folds."""
    classes, test_samples, folds = build_stratified_folds(
        config["split_csv"], int(config["num_classes"]), k_folds=k_folds, seed=int(config.get("seed", 42)), include_val=include_val,
    )
    pool_samples = [it for it in folds[0][0]] + [it for it in folds[0][1]]
    pos_map = {it: idx for idx, it in enumerate(pool_samples)}

    root_dir = Path(config["out_dir"]) / config["name"]
    root_dir.mkdir(parents=True, exist_ok=True)
    fold_accs: List[float] = []
    fold_res: List[dict] = []
    runner = run_cached if trainer == "cached" else run

    for f_i, (tr_s, va_s) in enumerate(folds):
        fold_name = f"{config['name']}_f{f_i}"
        fold_csv = _write_fold_csv(Path(config["out_dir"]) / fold_name / "split.csv", tr_s, va_s, test_samples, classes)
        f_cfg = {**config, "name": fold_name, "split_csv": str(fold_csv)}
        extra = dict(train_kwargs)
        if trainer == "cached" and pool_bundle is not None:
            tr_idx = [pos_map[it] for it in tr_s]
            va_idx = [pos_map[it] for it in va_s]
            pre, std_st = _fold_bundle_features(
                pool_bundle, tr_idx, va_idx, f_cfg,
                extra.get("pool_mode", "avg"), extra.get("feature_mode", "backbone"),
                float(extra.get("desc_weight", 1.0)), bool(extra.get("tta_extract", False)),
            )
            extra["precomputed_features"] = pre
            extra["precomputed_classes"] = classes
            extra["precomputed_standardizer_state"] = std_st
        res = runner(f_cfg, optimizer=optimizer, scheduler=scheduler, train_transform=train_transform, **extra)
        fold_res.append(res)
        fold_accs.append(float(res["best_val_acc"]))

    mean_acc = float(sum(fold_accs) / len(fold_accs))
    var = sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs)
    std_acc = float(var ** 0.5)
    best_fold = max(range(len(fold_accs)), key=lambda i: fold_accs[i])
    summary = {
        "name": config["name"],
        "k_folds": int(k_folds),
        "fold_val_accs": fold_accs,
        "mean_val_acc": mean_acc,
        "std_val_acc": std_acc,
        "best_fold": best_fold,
        "best_fold_ckpt": fold_res[best_fold]["best_ckpt"],
    }
    (root_dir / "kfold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
