"""Dataset builders, split validation, and single-pass feature bundle caching."""

from collections import Counter, defaultdict
import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from fcft.accel import extract_dual_descriptors, is_accel_available
from fcft.model import build_backbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(eval_resize_mode: str = "squash") -> transforms.Compose:
    """Build deterministic evaluation transform supporting squash or aspect-preserving center crop."""
    if eval_resize_mode == "squash":
        resize_ops = [transforms.Resize((224, 224))]
    elif eval_resize_mode == "center256":
        resize_ops = [transforms.Resize(256), transforms.CenterCrop(224)]
    else:
        raise ValueError(f"Unsupported eval_resize_mode: {eval_resize_mode!r}")
    return transforms.Compose([*resize_ops, transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def build_aug_transform(aug_mode: Optional[str]) -> Optional[transforms.Compose]:
    """Build standard preprocessing/augmentation pipeline for both image training and aug-bank caching."""
    if aug_mode in (None, "none"):
        return None
    norm = [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    presets = {
        "rrc_jitter": [transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.15, 0.15, 0.15), *norm],
        "resize256_crop224": [transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), *norm],
        "standard": [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(15), transforms.ColorJitter(0.15, 0.15, 0.15), *norm],
        "trivial": [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.TrivialAugmentWide(), *norm],
        "randaugment": [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandAugment(num_ops=2, magnitude=7), *norm],
        "erasing": [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.15, 0.15, 0.15), *norm, transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))],
        "affine": [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)), transforms.ColorJitter(0.15, 0.15, 0.15), *norm],
    }
    if aug_mode not in presets:
        raise ValueError(f"Unsupported aug_mode: {aug_mode!r}")
    return transforms.Compose(presets[aug_mode])


EVAL_TRANSFORM = build_eval_transform("squash")


def inspect_split(
    split_csv: str | Path,
    data_root: Optional[str | Path] = None,
    check_files: bool = False,
    expected_classes: Optional[int] = None,
) -> dict:
    """Validate split CSV columns, per-class split counts, and optional file existence."""
    csv_path = Path(split_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "label", "split"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Split CSV missing required columns {sorted(required)}: {reader.fieldnames}")
        rows = list(reader)

    if not rows:
        raise ValueError("Split CSV contains 0 data rows.")

    split_counts: Counter = Counter()
    class_split_counts: Dict[str, Counter] = defaultdict(Counter)
    missing_files: List[str] = []
    root = Path(data_root) if data_root is not None else None

    for row in rows:
        split = row["split"].strip()
        label = row["label"].strip()
        rel_path = row["path"].strip()
        split_counts[split] += 1
        class_split_counts[label][split] += 1
        if check_files and root is not None and not (root / rel_path).is_file():
            missing_files.append(rel_path)

    classes = sorted(class_split_counts.keys())
    if expected_classes is not None and len(classes) != int(expected_classes):
        raise ValueError(f"Expected {expected_classes} classes, found {len(classes)}: {classes}")

    return {
        "ok": len(missing_files) == 0,
        "total": len(rows),
        "num_classes": len(classes),
        "classes": classes,
        "split_counts": dict(split_counts),
        "class_split_counts": {c: dict(class_split_counts[c]) for c in classes},
        "missing_files": missing_files,
    }


def format_split_report(rep: dict) -> str:
    """Format split validation summary into a Markdown table."""
    lines = [
        f"Total samples: {rep['total']} | Classes: {rep['num_classes']} | Splits: {rep['split_counts']}",
        "",
        "| Class | Train | Val | Test | Total |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for cls in rep["classes"]:
        sc = rep["class_split_counts"][cls]
        tr, va, te = sc.get("train", 0), sc.get("val", 0), sc.get("test", 0)
        lines.append(f"| `{cls}` | {tr} | {va} | {te} | {tr + va + te} |")
    if rep["missing_files"]:
        lines.append(f"\nMissing files ({len(rep['missing_files'])}): {rep['missing_files'][:5]}")
    return "\n".join(lines)


class GarbageSplitDataset(Dataset):
    """Image classification dataset backed by a split CSV and root directory."""

    def __init__(self, samples: List[Tuple[str, int]], data_root: Path, classes: List[str], transform):
        self.samples = samples
        self.data_root = Path(data_root)
        self.classes = list(classes)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path, target = self.samples[idx]
        img_path = self.data_root / rel_path
        with Image.open(img_path) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
        tensor = self.transform(rgb) if self.transform is not None else rgb
        return tensor, target


def loaders_from_samples(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    data_root: str | Path,
    classes: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    seed: int = 42,
    eval_transform=None,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders directly from split sample lists."""
    ev_tf = eval_transform if eval_transform is not None else EVAL_TRANSFORM
    tr_tf = train_transform if train_transform is not None else ev_tf
    datasets = {
        "train": GarbageSplitDataset(train_samples, Path(data_root), classes, tr_tf),
        "val": GarbageSplitDataset(val_samples, Path(data_root), classes, ev_tf),
        "test": GarbageSplitDataset(test_samples, Path(data_root), classes, ev_tf),
    }
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    pin = bool(torch.cuda.is_available())
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=gen, pin_memory=pin),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
    }


def build_loaders(config: dict, train_transform=None) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders from split_csv with sorted class indices."""
    split_csv = Path(config["split_csv"])
    data_root = Path(config["data_root"])
    num_classes = int(config["num_classes"])
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 4))
    seed = int(config.get("seed", 42))

    with split_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    classes = sorted({row["label"] for row in rows})
    if len(classes) != num_classes:
        raise ValueError(f"Expected {num_classes} classes in CSV, found {len(classes)}: {classes}")
    label_to_idx = {label: i for i, label in enumerate(classes)}

    grouped: Dict[str, List[Tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row["split"].strip()
        if split in grouped:
            grouped[split].append((row["path"], label_to_idx[row["label"]]))

    ev_tf = build_eval_transform(str(config.get("eval_resize_mode", "squash")))
    return loaders_from_samples(
        grouped["train"], grouped["val"], grouped["test"], data_root, classes,
        batch_size=batch_size, num_workers=num_workers, train_transform=train_transform, seed=seed, eval_transform=ev_tf,
    )


class MultiPoolResNet18(nn.Module):
    """Extract avg, avgmax, gem, multiscale, and norm_multiscale features in one backbone pass."""

    def __init__(self, base_avg_backbone: nn.Module, target_pools: Optional[Sequence[str]] = None):
        super().__init__()
        self.stem = nn.Sequential(base_avg_backbone.conv1, base_avg_backbone.bn1, base_avg_backbone.relu, base_avg_backbone.maxpool)
        self.layer1 = base_avg_backbone.layer1
        self.layer2 = base_avg_backbone.layer2
        self.layer3 = base_avg_backbone.layer3
        self.layer4 = base_avg_backbone.layer4
        self.target_pools = set(target_pools) if target_pools is not None else None
        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pools = self.target_pools
        f3 = self.layer3(self.layer2(self.layer1(self.stem(x))))
        f4 = self.layer4(f3)
        out: Dict[str, torch.Tensor] = {}
        need_avg = pools is None or bool(pools & {"avg", "avgmax", "multiscale", "norm_multiscale"})
        p_avg = F.adaptive_avg_pool2d(f4, (1, 1)).flatten(1) if need_avg else None
        if pools is None or "avg" in pools:
            out["avg"] = p_avg.cpu()
        if pools is None or "avgmax" in pools:
            out["avgmax"] = torch.cat([p_avg, F.adaptive_max_pool2d(f4, (1, 1)).flatten(1)], dim=1).cpu()
        if pools is None or "gem" in pools:
            out["gem"] = f4.clamp(min=1e-6).pow(3.0).mean(dim=(-2, -1), keepdim=True).pow(1.0 / 3.0).flatten(1).cpu()
        if pools is None or "multiscale" in pools or "norm_multiscale" in pools:
            p_ms3 = F.adaptive_avg_pool2d(f3, (1, 1)).flatten(1)
            if pools is None or "multiscale" in pools:
                out["multiscale"] = torch.cat([p_ms3, p_avg], dim=1).cpu()
            if pools is None or "norm_multiscale" in pools:
                out["norm_multiscale"] = torch.cat([F.normalize(p_ms3, p=2, dim=-1), F.normalize(p_avg, p=2, dim=-1)], dim=1).cpu()
        return out


def _merge_chunks(chunks: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = list(chunks[0].keys())
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def _zoom_crop(x: torch.Tensor) -> torch.Tensor:
    h, w = int(x.shape[2]), int(x.shape[3])
    mh, mw = max(1, h // 16), max(1, w // 16)
    return F.interpolate(x[:, :, mh:h - mh, mw:w - mw], size=(h, w), mode="bilinear", align_corners=False)


def _split_views(feats: Dict[str, torch.Tensor], bsz: int, n_views: int) -> List[Dict[str, torch.Tensor]]:
    return [{k: v[i * bsz:(i + 1) * bsz] for k, v in feats.items()} for i in range(n_views)]


def _build_views(xd: torch.Tensor, include_flip: bool, include_zoom: bool) -> List[torch.Tensor]:
    views = [xd] + ([torch.flip(xd, dims=[3])] if include_flip else [])
    if include_zoom:
        zm = _zoom_crop(xd)
        views.extend([zm, torch.flip(zm, dims=[3])])
    return views


def _merge_view_outputs(
    view_names: List[str],
    chunks_by_view: Dict[str, List[Dict[str, torch.Tensor]]],
    desc_by_view: Dict[str, List[torch.Tensor]],
    include_desc: bool,
) -> Dict[str, torch.Tensor]:
    out = _merge_chunks(chunks_by_view["orig"])
    for vn in view_names[1:]:
        for k, v in _merge_chunks(chunks_by_view[vn]).items():
            out[f"{k}_{vn}"] = v
    if include_desc:
        out["desc"] = torch.cat(desc_by_view["orig"], dim=0)
        for vn in view_names[1:]:
            out[f"desc_{vn}"] = torch.cat(desc_by_view[vn], dim=0)
    return out


@torch.inference_mode()
def _extract_loader_bundle(
    mp_bb: MultiPoolResNet18,
    loader,
    device: torch.device,
    include_flip: bool,
    include_desc: bool,
    include_zoom: bool = False,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    view_names = ["orig"] + (["flip"] if include_flip else []) + (["zoom", "zoom_flip"] if include_zoom else [])
    chunks_by_view: Dict[str, List[Dict[str, torch.Tensor]]] = {vn: [] for vn in view_names}
    desc_by_view: Dict[str, List[torch.Tensor]] = {vn: [] for vn in view_names}
    ys: List[torch.Tensor] = []

    for x, y in loader:
        xd = x.to(device)
        views = _build_views(xd, include_flip, include_zoom)
        cat_in = torch.cat(views, dim=0)
        if device.type == "cpu":
            cat_in = cat_in.contiguous(memory_format=torch.channels_last)
        fused = mp_bb(cat_in)
        for vn, part in zip(view_names, _split_views(fused, int(xd.shape[0]), len(views))):
            chunks_by_view[vn].append(part)
        for vn, vx in (zip(view_names, views) if include_desc else ()):
            desc_by_view[vn].append(extract_dual_descriptors(vx, denormalize=True).cpu())
        ys.append(y.cpu())
    return _merge_view_outputs(view_names, chunks_by_view, desc_by_view, include_desc), torch.cat(ys, dim=0)


@torch.inference_mode()
def _extract_aug_bank(
    mp_bb: MultiPoolResNet18,
    config: dict,
    train_transform,
    repeats: int,
    device: torch.device,
    aug_mode: str = "rrc_jitter",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    tf = train_transform if train_transform is not None else build_aug_transform(aug_mode or "rrc_jitter")
    ext_cfg = {**config, "shuffle_train": False, "batch_size": max(128, int(config.get("batch_size", 32)))}
    base_ld = build_loaders(ext_cfg, train_transform=tf)["train"]
    nw = int(ext_cfg.get("num_workers", 2))
    aug_ld = torch.utils.data.DataLoader(
        base_ld.dataset,
        batch_size=int(ext_cfg["batch_size"]),
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    chunks_by_pool: Dict[str, List[torch.Tensor]] = {}
    ys = []
    for _ in range(repeats):
        for x, y in aug_ld:
            xb = x.to(device, memory_format=torch.channels_last) if device.type == "cpu" else x.to(device)
            out = mp_bb(xb)
            for k, v in out.items():
                chunks_by_pool.setdefault(k, []).append(v)
            ys.append(y.cpu())
    del aug_ld
    return {k: torch.cat(vs, dim=0) for k, vs in chunks_by_pool.items()}, torch.cat(ys, dim=0)


def read_sorted_classes(split_csv: str | Path) -> List[str]:
    """Read sorted unique class labels from split CSV without loading images."""
    with Path(split_csv).open("r", newline="", encoding="utf-8") as f:
        return sorted({row["label"] for row in csv.DictReader(f)})


def _bundle_cache_path(
    config: dict,
    aug_repeats: int,
    include_flip: bool,
    include_desc: bool,
    cache_dir: Optional[str | Path],
    aug_mode: str = "rrc_jitter",
    eval_resize_mode: str = "squash",
    target_pools: Optional[Sequence[str]] = None,
) -> Path:
    h = hashlib.sha256()
    sp = Path(config["split_csv"])
    if sp.is_file():
        h.update(sp.read_bytes())
    else:
        h.update(Path(config["data_root"]).name.encode("utf-8"))
    pool_tag = "all" if target_pools is None else ",".join(sorted(set(target_pools)))
    h.update(
        f"|{config.get('model', 'resnet18')}|{aug_repeats}"
        f"|{include_flip}|{include_desc}|{aug_mode}|{eval_resize_mode}|{pool_tag}".encode("utf-8")
    )
    root = Path(cache_dir) if cache_dir is not None else (Path(config.get("out_dir", "outputs")) / "_feature_cache")
    return root / f"bundle_{h.hexdigest()[:16]}.pt"


def _can_write_dir(dir_path: Path) -> bool:
    cur = dir_path
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return os.access(cur, os.W_OK)


def _save_bundle_atomic(c_path: Path, bundle: dict) -> bool:
    if not _can_write_dir(c_path.parent):
        return False
    c_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = c_path.with_suffix(".tmp")
    torch.save(bundle, tmp_p)
    tmp_p.replace(c_path)
    return True


def _new_extract_runner(
    config: dict, eval_rm: str, target_pools: Optional[Sequence[str]],
) -> Tuple[torch.device, MultiPoolResNet18, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb_avg, _ = build_backbone(config.get("model", "resnet18"), pool_mode="avg", fuse_conv_bn=(device.type == "cuda"))
    mp_bb = MultiPoolResNet18(bb_avg, target_pools=target_pools).to(device)
    if device.type == "cpu":
        mp_bb = mp_bb.to(memory_format=torch.channels_last)
    extract_cfg = {**config, "eval_resize_mode": eval_rm, "batch_size": max(128, int(config.get("batch_size", 32)))}
    return device, mp_bb, extract_cfg


def extract_shared_feature_bundle(
    config: dict,
    aug_transform=None,
    aug_repeats: int = 0,
    include_flip: bool = True,
    include_descriptors: Optional[bool] = None,
    include_test: bool = False,
    use_disk_cache: bool = True,
    cache_dir: Optional[str | Path] = None,
    aug_mode: str = "rrc_jitter",
    eval_resize_mode: Optional[str] = None,
    target_pools: Optional[Sequence[str]] = None,
    bundle_path: Optional[str | Path] = None,
) -> dict:
    """Extract or load a portable feature bundle (`bundle_path` supports mounted read-only files)."""
    use_desc = is_accel_available() if include_descriptors is None else bool(include_descriptors)
    eff_eval_rm = str(eval_resize_mode or config.get("eval_resize_mode", "squash"))
    eff_aug_m = str(aug_mode or "rrc_jitter")
    c_path = (
        Path(bundle_path)
        if bundle_path is not None
        else _bundle_cache_path(
            config, int(aug_repeats), bool(include_flip), use_desc, cache_dir,
            aug_mode=eff_aug_m, eval_resize_mode=eff_eval_rm, target_pools=target_pools,
        )
    )
    if (use_disk_cache or bundle_path is not None) and c_path.is_file():
        cached = torch.load(c_path, map_location="cpu", weights_only=True)
        missing_p = [p for p in (target_pools or ()) if p not in cached.get("train", {})]
        if missing_p:
            raise ValueError(
                f"Mounted bundle {c_path} is missing requested pool(s): {missing_p}. "
                f"Available: {sorted(cached.get('train', {}).keys())}"
            )
        if not include_test or "test" in cached:
            return cached
        device, mp_bb, extract_cfg = _new_extract_runner(config, eff_eval_rm, target_pools)
        test_loader = build_loaders(extract_cfg, train_transform=None)["test"]
        te_map, y_te = _extract_loader_bundle(mp_bb, test_loader, device, True, use_desc, include_zoom=True)
        cached["test"], cached["y_test"] = te_map, y_te
        if use_disk_cache:
            _save_bundle_atomic(c_path, cached)
        return cached

    device, mp_bb, extract_cfg = _new_extract_runner(config, eff_eval_rm, target_pools)
    loaders = build_loaders(extract_cfg, train_transform=None)
    classes = list(loaders["train"].dataset.classes)
    tr_map, y_tr = _extract_loader_bundle(mp_bb, loaders["train"], device, include_flip, use_desc)
    va_map, y_va = _extract_loader_bundle(mp_bb, loaders["val"], device, include_flip, use_desc)
    bundle: Dict[str, object] = {"classes": classes, "train": tr_map, "y_train": y_tr, "val": va_map, "y_val": y_va}
    if include_test:
        te_map, y_te = _extract_loader_bundle(mp_bb, loaders["test"], device, True, use_desc, include_zoom=True)
        bundle["test"], bundle["y_test"] = te_map, y_te
    if aug_repeats > 0:
        aug_x, aug_y = _extract_aug_bank(mp_bb, extract_cfg, aug_transform, int(aug_repeats), device, aug_mode=eff_aug_m)
        bundle["train_aug"], bundle["y_train_aug"] = aug_x, aug_y
    if use_disk_cache or bundle_path is not None:
        _save_bundle_atomic(c_path, bundle)
    return bundle


def assemble_from_bundle(
    bundle: dict,
    model: nn.Module,
    pool_mode: str = "avg",
    feature_mode: str = "backbone",
    tta_extract: bool = False,
    use_aug_bank: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (x_tr, y_tr, x_va, y_va) tensors for a preset directly from a pre-extracted bundle."""
    tr, va = bundle["train"], bundle["val"]
    y_tr, y_va = bundle["y_train"], bundle["y_val"]
    if feature_mode == "backbone":
        x_tr = 0.5 * (tr[pool_mode] + tr[f"{pool_mode}_flip"]) if tta_extract else tr[pool_mode]
        x_va = 0.5 * (va[pool_mode] + va[f"{pool_mode}_flip"]) if tta_extract else va[pool_mode]
        if use_aug_bank and "train_aug" in bundle and pool_mode in bundle["train_aug"]:
            x_tr = torch.cat([x_tr, bundle["train_aug"][pool_mode]], dim=0)
            y_tr = torch.cat([y_tr, bundle["y_train_aug"]], dim=0)
        return x_tr, y_tr, x_va, y_va

    dev = model.standardizer.mean.device
    d_tr = torch.cat([tr["desc"], tr["desc_flip"]], dim=0) if tta_extract else tr["desc"]
    y_out = torch.cat([y_tr, y_tr], dim=0) if tta_extract else y_tr
    model.standardizer.fit_stats(d_tr.to(dev))
    bb_tr = (torch.cat([tr[pool_mode], tr[f"{pool_mode}_flip"]], dim=0) if tta_extract else tr[pool_mode]) if feature_mode == "hybrid" else None
    bb_va = va[pool_mode] if feature_mode == "hybrid" else None
    x_tr = model.assemble_features(bb_tr.to(dev) if bb_tr is not None else None, d_tr.to(dev)).cpu()
    x_va = model.assemble_features(bb_va.to(dev) if bb_va is not None else None, va["desc"].to(dev)).cpu()
    return x_tr, y_out, x_va, y_va


@torch.inference_mode()
def predict_logits_from_test_bundle(
    model: nn.Module,
    test_map: dict,
    pool_mode: str = "avg",
    feature_mode: str = "backbone",
    tta_mode: str = "none",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute head logits directly from pre-extracted test bundle views matching predict_logits."""
    suffixes = {"none": [""], "flip2": ["", "_flip"], "multiview4": ["", "_flip", "_zoom", "_zoom_flip"]}
    if tta_mode not in suffixes:
        raise ValueError(f"Unsupported tta_mode: {tta_mode!r}")
    views = suffixes[tta_mode]
    logit_acc = None
    for sfx in views:
        if feature_mode == "backbone":
            feat = test_map[f"{pool_mode}{sfx}"].to(device)
        elif feature_mode == "descriptor":
            feat = model.assemble_features(None, test_map[f"desc{sfx}"].to(device))
        else:
            feat = model.assemble_features(test_map[f"{pool_mode}{sfx}"].to(device), test_map[f"desc{sfx}"].to(device))
        lg = model.fc(feat)
        logit_acc = lg if logit_acc is None else (logit_acc + lg)
    return logit_acc / float(len(views))
