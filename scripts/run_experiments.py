"""CLI runner for 3-axis FC fine-tuning experiments with resume support & auto-summary."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Tuple
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import axes as axes_cfg  # noqa: E402
from configs import compare as compare_cfg  # noqa: E402
from configs import grid as grid_cfg  # noqa: E402
from fcft.accel import auto_tune_resilient, is_accel_available  # noqa: E402
from fcft.data import (  # noqa: E402
    MultiPoolResNet18,
    assemble_from_bundle,
    extract_shared_feature_bundle,
    format_split_report,
    inspect_split,
)
from fcft.grid import format_grid_table, run_single_experiment  # noqa: E402
from fcft.kfold import build_stratified_folds, extract_kfold_pool_bundle, run_kfold  # noqa: E402
from fcft.losses import UnifiedClassificationLoss, compute_class_weights  # noqa: E402
from fcft.model import build_backbone, build_head, build_model  # noqa: E402
from fcft.report import (  # noqa: E402
    collect_summaries,
    format_before_after_table,
    format_markdown_table,
    plot_comparison_curves,
    save_summary_csv,
    select_best_by_axis,
)
from fcft.train import (  # noqa: E402
    _fit_head_center_stats,
    _warm_start_head,
    co_train_heads_cached,
    compute_warm_start_state,
    run,
    run_cached,
)


def _clone_entry(entry: dict) -> dict:
    out = {**entry, "config": dict(entry["config"])}
    if "train_kwargs" in entry and isinstance(entry["train_kwargs"], dict):
        tk = dict(entry["train_kwargs"])
        if isinstance(tk.get("head_kwargs"), dict):
            tk["head_kwargs"] = dict(tk["head_kwargs"])
        out["train_kwargs"] = tk
    return out


def select_experiments(axis: str = "all") -> Dict[str, List[dict]]:
    """Return experiment entries grouped by requested axis."""
    mapping = {
        "lr": [_clone_entry(e) for e in axes_cfg.LR_EXPERIMENTS],
        "scheduler": [_clone_entry(e) for e in axes_cfg.SCHEDULER_EXPERIMENTS],
        "transform": [_clone_entry(e) for e in axes_cfg.TRANSFORM_EXPERIMENTS],
    }
    if axis == "all":
        return mapping
    extra_axes = {
        "compare": [_clone_entry(e) for e in compare_cfg.EXPERIMENTS],
        "grid": [_clone_entry(e) for e in grid_cfg.EXPERIMENTS],
        "loss": [_clone_entry(e) for e in axes_cfg.LOSS_EXPERIMENTS],
        "activation": [_clone_entry(e) for e in axes_cfg.ACTIVATION_EXPERIMENTS],
    }
    if axis in extra_axes:
        return {axis: extra_axes[axis]}
    if axis not in mapping:
        raise ValueError(f"Unknown axis: {axis!r}")
    return {axis: mapping[axis]}


def _is_done(out_dir: Path, name: str, target_epochs: int) -> bool:
    exp_dir = out_dir / name
    ckpt = exp_dir / "best.pt"
    hist = exp_dir / "history.json"
    if not (ckpt.is_file() and hist.is_file()):
        return False
    rows = json.loads(hist.read_text(encoding="utf-8"))
    return len(rows) >= int(target_epochs)


def _override_config(cfg: dict, args_dict: dict) -> dict:
    updated = dict(cfg)
    for k in ("split_csv", "data_root", "out_dir", "num_classes", "epochs", "batch_size", "num_workers"):
        if args_dict.get(k) is not None:
            updated[k] = args_dict[k]
    return updated


def _write_reports(out_dir: Path, axis_groups: Dict[str, List[dict]]) -> List[dict]:
    all_names = [e["config"]["name"] for exps in axis_groups.values() for e in exps]
    records = collect_summaries(out_dir, names=all_names)
    md = format_markdown_table(records)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    save_summary_csv(records, out_dir / "summary.csv")
    for ax_name, exps in axis_groups.items():
        ax_names = [e["config"]["name"] for e in exps]
        plot_comparison_curves(out_dir, ax_names, out_dir / f"comparison_{ax_name}.png", title=f"Axis: {ax_name}")
    for ax_name, baseline_name, fname in (
        ("compare", "compare_0_baseline", "compare_before_after.md"),
        ("loss", "loss_ce", "loss_comparison.md"),
        ("activation", "act_relu", "activation_comparison.md"),
    ):
        if ax_name in axis_groups:
            by_name = {r["name"]: r for r in records}
            ordered = [by_name[e["config"]["name"]] for e in axis_groups[ax_name] if e["config"]["name"] in by_name]
            desc_map = {e["config"]["name"]: e.get("description", "") for e in axis_groups[ax_name]}
            (out_dir / fname).write_text(
                format_before_after_table(ordered, baseline_name=baseline_name, descriptions=desc_map),
                encoding="utf-8",
            )
    if "grid" in axis_groups:
        param_map = {e["config"]["name"]: e.get("grid_params", {}) for e in axis_groups["grid"]}
        grid_recs = [r for r in records if r["name"] in param_map]
        (out_dir / "grid_summary.md").write_text(format_grid_table(grid_recs, param_map=param_map), encoding="utf-8")
    best = select_best_by_axis(out_dir)
    (out_dir / "best_by_axis.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return records


def _build_entry_model(cfg: dict, extra: dict):
    return build_model(
        cfg.get("model", "resnet18"),
        int(cfg["num_classes"]),
        head_type=extra.get("head_type", "linear"),
        feature_norm=bool(extra.get("feature_norm", False)),
        pool_mode=extra.get("pool_mode", "avg"),
        feature_mode=extra.get("feature_mode", "backbone"),
        desc_weight=float(extra.get("desc_weight", 1.0)),
        **dict(extra.get("head_kwargs") or {}),
    )


def _run_entry(cfg: dict, entry: dict, bundle: Optional[dict] = None, fast_cpu: bool = False) -> dict:
    extra = dict(entry.get("train_kwargs") or {})
    pool_m = extra.get("pool_mode", "avg")
    feat_m = extra.get("feature_mode", "backbone")
    can_use_bundle = bundle is not None and (
        entry.get("trainer") == "cached" or (fast_cpu and feat_m == "backbone" and pool_m == "avg")
    )
    if can_use_bundle:
        use_aug = bool(entry.get("use_aug_bank", False) or (fast_cpu and entry.get("train_transform") is not None))
        tmp_mod = None if feat_m == "backbone" else _build_entry_model(cfg, extra)
        pre = assemble_from_bundle(
            bundle,
            tmp_mod,
            pool_mode=pool_m,
            feature_mode=feat_m,
            tta_extract=bool(extra.get("tta_extract", False)),
            use_aug_bank=use_aug,
        )
        std_state = tmp_mod.standardizer.state_dict() if (tmp_mod is not None and hasattr(tmp_mod, "standardizer")) else None
        return run_cached(
            cfg,
            optimizer=entry.get("optimizer"),
            scheduler=entry.get("scheduler"),
            train_transform=entry.get("train_transform"),
            precomputed_features=pre,
            precomputed_classes=bundle["classes"],
            precomputed_standardizer_state=std_state,
            **extra,
        )
    runner = run_cached if entry.get("trainer") == "cached" else run
    return runner(
        cfg,
        optimizer=entry.get("optimizer"),
        scheduler=entry.get("scheduler"),
        train_transform=entry.get("train_transform"),
        **extra,
    )


def _entry_bundle_key(cfg: dict, entry: dict, fast_cpu: bool) -> Tuple[str, str, bool]:
    eval_rm = str(cfg.get("eval_resize_mode", "squash"))
    use_aug = bool(entry.get("use_aug_bank", False) or (fast_cpu and entry.get("train_transform") is not None))
    aug_m = str(entry.get("aug_mode") or "rrc_jitter") if use_aug else "_none"
    return eval_rm, aug_m, use_aug


def _pick_bundle(
    bundles: Dict[Tuple[str, str], dict],
    cfg: dict,
    entry: dict,
    fast_cpu: bool,
) -> Optional[dict]:
    if not bundles:
        return None
    eval_rm, aug_m, use_aug = _entry_bundle_key(cfg, entry, fast_cpu)
    if use_aug:
        return bundles.get((eval_rm, aug_m))
    if (eval_rm, "_none") in bundles:
        return bundles[(eval_rm, "_none")]
    for (k_rm, _), bdl in bundles.items():
        if k_rm == eval_rm:
            return bdl
    return None


def _collect_bundle_specs(
    groups: Dict[str, List[dict]], overrides: dict, limit: Optional[int], skip_existing: bool, fast_cpu: bool,
) -> Tuple[Dict[Tuple[str, str], dict], set, set]:
    specs: Dict[Tuple[str, str], dict] = {}
    aug_rms, desc_rms = set(), set()
    for exps in groups.values():
        for entry in (exps[:limit] if limit is not None else exps):
            cfg = _override_config(entry["config"], overrides)
            if skip_existing and _is_done(Path(cfg["out_dir"]), cfg["name"], int(cfg["epochs"])):
                continue
            eval_rm, aug_m, use_aug = _entry_bundle_key(cfg, entry, fast_cpu)
            if use_aug:
                aug_rms.add(eval_rm)
            cur = specs.setdefault((eval_rm, aug_m), {"cfg": cfg, "aug_tf": None, "reps": 0, "pools": set()})
            tk = entry.get("train_kwargs") or {}
            feat_m = tk.get("feature_mode", "backbone")
            if feat_m != "descriptor":
                cur["pools"].add(str(tk.get("pool_mode", "avg")))
            if use_aug:
                cur["reps"] = max(cur["reps"], int(entry.get("aug_repeats", 3)))
                if fast_cpu and cur["aug_tf"] is None and entry.get("train_transform") is not None:
                    cur["aug_tf"] = entry["train_transform"]
            if feat_m in ("descriptor", "hybrid"):
                desc_rms.add(eval_rm)
    return specs, aug_rms, desc_rms


def _maybe_extract_bundles(
    groups: Dict[str, List[dict]],
    overrides: dict,
    limit: Optional[int],
    skip_existing: bool,
    shared_cache: bool,
    fast_cpu: bool,
    use_disk_cache: bool = True,
    cache_dir: Optional[str] = None,
    bundle_path: Optional[str] = None,
) -> Dict[Tuple[str, str], dict]:
    if not (shared_cache or fast_cpu or bundle_path is not None):
        return {}
    specs, aug_rms, desc_rms = _collect_bundle_specs(groups, overrides, limit, skip_existing, fast_cpu)
    pools_by_rm: Dict[str, set] = {}
    for (eval_rm, _), sp in specs.items():
        pools_by_rm.setdefault(eval_rm, set()).update(sp["pools"])
    bundles: Dict[Tuple[str, str], dict] = {}
    for (eval_rm, aug_m), sp in specs.items():
        if aug_m == "_none" and eval_rm in aug_rms:
            continue
        t_pools = sorted(pools_by_rm.get(eval_rm) or sp["pools"])
        bundles[(eval_rm, aug_m)] = extract_shared_feature_bundle(
            sp["cfg"],
            aug_transform=sp["aug_tf"],
            aug_repeats=sp["reps"],
            include_flip=True,
            include_descriptors=(eval_rm in desc_rms),
            include_test=False,
            use_disk_cache=use_disk_cache,
            cache_dir=cache_dir,
            aug_mode=("rrc_jitter" if aug_m == "_none" else aug_m),
            eval_resize_mode=eval_rm,
            target_pools=t_pools,
            bundle_path=bundle_path,
        )
    return bundles


def export_shared_bundle(
    axis: str = "grid",
    overrides: Optional[dict] = None,
    export_path: Optional[str] = None,
    fast_cpu: bool = False,
) -> Path:
    """Extract and export a single portable bundle (.pt) with train, val, test, and aug_bank."""
    groups = select_experiments(axis)
    ov = dict(overrides or {})
    specs, aug_rms, desc_rms = _collect_bundle_specs(groups, ov, limit=None, skip_existing=False, fast_cpu=fast_cpu)
    eval_rms = sorted({rm for (rm, _) in specs})
    eval_rm = eval_rms[0] if eval_rms else "squash"
    aug_tf, max_reps, aug_mode, pools, desc_need = None, 0, "rrc_jitter", set(), (eval_rm in desc_rms)
    base_cfg = _override_config(axes_cfg.BASE, ov)
    for (rm, am), sp in specs.items():
        if rm != eval_rm:
            continue
        base_cfg = sp["cfg"]
        pools.update(sp["pools"])
        if sp["reps"] > max_reps:
            max_reps, aug_tf = sp["reps"], sp["aug_tf"]
            aug_mode = "rrc_jitter" if am == "_none" else am
        elif rm in aug_rms and am != "_none":
            aug_mode = am
    out_p = Path(export_path) if (export_path and export_path != "__AUTO__") else (
        Path(base_cfg["out_dir"]) / "_feature_cache" / f"fc_cache_{axis}.pt"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    extract_shared_feature_bundle(
        base_cfg,
        aug_transform=aug_tf,
        aug_repeats=max_reps,
        include_flip=True,
        include_descriptors=desc_need,
        include_test=True,
        use_disk_cache=True,
        aug_mode=aug_mode,
        eval_resize_mode=eval_rm,
        target_pools=sorted(pools) if pools else None,
        bundle_path=out_p,
    )
    mb = float(out_p.stat().st_size) / (1024.0 * 1024.0)
    print(f"Exported portable bundle: {out_p} ({mb:.2f} MB)")
    print(f"Mount via: python scripts/run_experiments.py --axis {axis} --shared-cache --cache-file {out_p}")
    return out_p


def _run_kfold_batch(
    groups: Dict[str, List[dict]],
    overrides: dict,
    k_folds: int,
    limit: Optional[int],
    shared_cache: bool,
    use_disk_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    entries = [e for exps in groups.values() for e in (exps[:limit] if limit is not None else exps)]
    if not entries:
        return []
    first_cfg = _override_config(entries[0]["config"], overrides)
    out_dir = Path(first_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_bundle = None
    if shared_cache:
        classes, _, folds = build_stratified_folds(first_cfg["split_csv"], int(first_cfg["num_classes"]), k_folds=k_folds, seed=int(first_cfg.get("seed", 42)))
        pool_samples = list(folds[0][0]) + list(folds[0][1])
        need_desc = any((e.get("train_kwargs") or {}).get("feature_mode", "backbone") in ("descriptor", "hybrid") for e in entries)
        need_pools = sorted({
            str((e.get("train_kwargs") or {}).get("pool_mode", "avg"))
            for e in entries if (e.get("train_kwargs") or {}).get("feature_mode", "backbone") != "descriptor"
        })
        bb_avg, _ = build_backbone(first_cfg.get("model", "resnet18"), pool_mode="avg")
        pool_bundle = extract_kfold_pool_bundle(
            first_cfg, pool_samples, classes, MultiPoolResNet18(bb_avg, target_pools=need_pools),
            include_desc=need_desc, use_disk_cache=use_disk_cache, cache_dir=cache_dir, target_pools=need_pools,
        )
    results = []
    for entry in entries:
        cfg = _override_config(entry["config"], overrides)
        extra = dict(entry.get("train_kwargs") or {})
        results.append(run_kfold(
            cfg, k_folds=k_folds, trainer=entry.get("trainer", "cached"), optimizer=entry.get("optimizer"),
            scheduler=entry.get("scheduler"), train_transform=entry.get("train_transform"), pool_bundle=pool_bundle, **extra,
        ))
    results.sort(key=lambda r: -float(r["mean_val_acc"]))
    lines = ["| Rank | Experiment | K-Folds | Mean Val Acc | Std |", "| :---: | :--- | :---: | :---: | :---: |"]
    for rk, r in enumerate(results, 1):
        lines.append(f"| {rk} | `{r['name']}` | {r['k_folds']} | {r['mean_val_acc']:.4f} | {r['std_val_acc']:.4f} |")
    (out_dir / "kfold_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


_WORKER_CTX = None


def _worker_init(ctx_data=None) -> None:
    torch.set_num_threads(1)
    if ctx_data is not None:
        global _WORKER_CTX
        _WORKER_CTX = ctx_data


def _worker_run(idx: int) -> dict:
    jobs, bundles, fast_cpu = _WORKER_CTX
    cfg, entry = jobs[idx]
    return _run_entry(cfg, entry, bundle=_pick_bundle(bundles, cfg, entry, fast_cpu), fast_cpu=fast_cpu)


def _resolve_worker_count(jobs: List[Tuple[dict, dict]], bundles: Dict[Tuple[str, str], dict], workers: Optional[int]) -> int:
    if torch.cuda.is_available() or len(jobs) <= 1:
        return 1
    if not all(e.get("trainer") == "cached" or e.get("train_kwargs", {}).get("feature_mode", "backbone") == "backbone" for _, e in jobs):
        return 1
    if workers is not None:
        return max(1, min(int(workers), len(jobs)))
    n_rows = max((int(b["y_train"].shape[0]) for b in bundles.values()), default=0)
    n_cpu = os.cpu_count() or 1
    if n_rows >= 64 and len(jobs) >= 4 and n_cpu >= 4:
        return min(8, n_cpu, len(jobs))
    return 1


def _run_jobs(
    jobs: List[Tuple[dict, dict]],
    bundles: Dict[Tuple[str, str], dict],
    fast_cpu: bool,
    workers: Optional[int] = None,
) -> List[dict]:
    if not jobs:
        return []
    eff_w = _resolve_worker_count(jobs, bundles, workers)
    if eff_w <= 1:
        return [_run_entry(cfg, entry, bundle=_pick_bundle(bundles, cfg, entry, fast_cpu), fast_cpu=fast_cpu) for cfg, entry in jobs]
    prev_pp = os.environ.get("PYTHONPATH")
    proj_str = str(PROJECT_ROOT)
    if not prev_pp or proj_str not in prev_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{proj_str}{os.pathsep}{prev_pp}" if prev_pp else proj_str
    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=eff_w, initializer=_worker_init, initargs=((jobs, bundles, fast_cpu),)) as pool:
            return pool.map(_worker_run, range(len(jobs)), chunksize=1)
    finally:
        if prev_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prev_pp


def _halving_select_entries(
    entries: List[dict],
    overrides: dict,
    bundles: Dict[Tuple[str, str], dict],
    fast_cpu: bool,
    halving_factor: int = 3,
    workers: Optional[int] = None,
    co_train: bool = False,
) -> List[dict]:
    if len(entries) <= 3:
        return entries
    sample_cfg = _override_config(entries[0]["config"], overrides)
    screen_ep = max(2, min(4, int(sample_cfg["epochs"]) // 3))
    screen_dir = Path(sample_cfg["out_dir"]) / "_screen"
    jobs = [({**_override_config(e["config"], overrides), "out_dir": str(screen_dir), "epochs": screen_ep, "save_ckpt": False}, e) for e in entries]
    jobs = _attach_shared_warm_starts(jobs, bundles, fast_cpu)
    if co_train and bundles:
        results = _run_jobs_co_train(jobs, bundles, fast_cpu, workers=workers)
    else:
        results = _run_jobs(jobs, bundles, fast_cpu, workers=workers)
    scored = sorted(zip((float(r["best_val_acc"]) for r in results), entries), key=lambda pair: -pair[0])
    shutil.rmtree(screen_dir, ignore_errors=True)
    keep_n = max(2, (len(entries) + max(2, int(halving_factor)) - 1) // max(2, int(halving_factor)))
    return [pair[1] for pair in scored[:keep_n]]


def _apply_auto_tune_resilient(
    groups: Dict[str, List[dict]],
    overrides: dict,
    bundles: Dict[Tuple[str, str], dict],
    use_disk_cache: bool,
    cache_dir: Optional[str],
) -> Optional[dict]:
    if not is_accel_available():
        return None
    all_entries = [e for exps in groups.values() for e in exps]
    if not all_entries:
        return None
    ref_entry = next(
        (
            e for e in all_entries
            if e.get("loss_mode") == "resilient_auto" or (e.get("train_kwargs") or {}).get("loss_cap_mode") in ("mad", "quantile")
        ),
        all_entries[0],
    )
    ref_cfg = _override_config(ref_entry["config"], overrides)
    extra = dict(ref_entry.get("train_kwargs") or {})
    pool_m = extra.get("pool_mode", "norm_multiscale")
    feat_m = extra.get("feature_mode", "backbone")
    bdl = _pick_bundle(bundles, ref_cfg, ref_entry, fast_cpu=False)
    if bdl is None:
        bdl = extract_shared_feature_bundle(
            ref_cfg, aug_repeats=int(ref_entry.get("aug_repeats", 0)), include_flip=True,
            include_descriptors=(feat_m in ("descriptor", "hybrid")), include_test=False,
            use_disk_cache=use_disk_cache, cache_dir=cache_dir, target_pools=[pool_m],
        )
    pre = assemble_from_bundle(bdl, None, pool_mode=pool_m, feature_mode=feat_m, tta_extract=bool(extra.get("tta_extract", True)), use_aug_bank=bool(ref_entry.get("use_aug_bank", False)))
    x_tr, y_tr, x_va, y_va = pre
    in_dim, num_cls = int(x_tr.shape[1]), int(ref_cfg["num_classes"])
    head_t = extra.get("head_type", "residual")
    fn_en, fn_m = bool(extra.get("feature_norm", True)), extra.get("feature_norm_mode", "center_power_l2")
    f_drop = float(extra.get("feature_dropout", 0.0))
    ridge_v = extra.get("warm_start_ridge", None)
    base_hkw = dict(extra.get("head_kwargs") or {})

    def _p_head(xs: torch.Tensor, ys: torch.Tensor, learnable_leak: bool = False, channel_leak: bool = False):
        h_mod = build_head(
            in_dim, num_cls, head_type=head_t, feature_norm=fn_en, feature_norm_mode=fn_m,
            feature_dropout=f_drop, **{**base_hkw, "learnable_leak": learnable_leak, "channel_leak": channel_leak},
        )
        _fit_head_center_stats(h_mod, xs)
        if ridge_v is not None:
            _warm_start_head(h_mod, xs, ys, float(ridge_v))
        return h_mod

    def _p_loss(ys, loss_cap_mode="fixed", loss_cap_tau=0.8, loss_cap_alpha=0.5, loss_cap_param=1.75, class_weight_mode=None, label_smoothing=0.05, device=torch.device("cpu")):
        cw = compute_class_weights(ys, num_cls, mode=class_weight_mode) if (class_weight_mode is not None and ys is not None) else None
        return UnifiedClassificationLoss(num_classes=num_cls, label_smoothing=label_smoothing, class_weights=cw, loss_cap_mode=loss_cap_mode, loss_cap_tau=loss_cap_tau, loss_cap_alpha=loss_cap_alpha, loss_cap_param=loss_cap_param).to(device)

    tuned = auto_tune_resilient(x_tr, y_tr, x_va, y_va, num_cls, _p_head, _p_loss, batch_size=int(ref_cfg.get("batch_size", 64)), seed=int(ref_cfg.get("seed", 42)))
    out_dir = Path(ref_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tuned_resilient.json").write_text(json.dumps(tuned, indent=2, ensure_ascii=False), encoding="utf-8")
    best_hp = tuned["best"]
    for exps in groups.values():
        for idx_e, entry in enumerate(exps):
            tk = dict(entry.get("train_kwargs") or {})
            if entry.get("loss_mode") == "resilient_auto" or ("grid" in groups and idx_e == 0) or tk.get("loss_cap_mode") in ("mad", "quantile"):
                tk.update({k: best_hp[k] for k in ("loss_cap_mode", "loss_cap_tau", "loss_cap_alpha", "loss_cap_param", "class_weight_mode")})
                h_kw = dict(tk.get("head_kwargs") or {})
                h_kw.update({"learnable_leak": best_hp["learnable_leak"], "channel_leak": best_hp["channel_leak"]})
                tk["head_kwargs"] = h_kw
                entry["train_kwargs"] = tk
    return tuned


def _attach_shared_warm_starts(
    jobs: List[Tuple[dict, dict]],
    bundles: Dict[Tuple[str, str], dict],
    fast_cpu: bool,
) -> List[Tuple[dict, dict]]:
    if not bundles or len(jobs) < 2 or not is_accel_available():
        return jobs
    counts: Dict[tuple, int] = {}
    keys: List[Optional[tuple]] = []
    for cfg, entry in jobs:
        extra = entry.get("train_kwargs") or {}
        bdl = _pick_bundle(bundles, cfg, entry, fast_cpu)
        head_t = extra.get("head_type", "linear")
        ridge = extra.get("warm_start_ridge", None)
        if (
            bdl is None or ridge is None or extra.get("precomputed_warm_start") is not None
            or extra.get("auto_tune_resilient", False) or head_t not in ("linear", "residual", "cosine", "dual_anchor")
        ):
            keys.append(None)
            continue
        h_kw = extra.get("head_kwargs") or {}
        use_aug = bool(entry.get("use_aug_bank", False) or (fast_cpu and entry.get("train_transform") is not None))
        k = (
            id(bdl), extra.get("pool_mode", "avg"), extra.get("feature_mode", "backbone"),
            bool(extra.get("feature_norm", False)), extra.get("feature_norm_mode", None),
            float(extra.get("feature_dropout", 0.0)), head_t, float(ridge),
            bool(extra.get("tta_extract", False)), use_aug,
            bool(h_kw.get("out_norm", True)), float(h_kw.get("scale", 16.0)),
        )
        counts[k] = counts.get(k, 0) + 1
        keys.append(k)
    cache: Dict[tuple, Optional[dict]] = {}
    out_jobs: List[Tuple[dict, dict]] = []
    for (cfg, entry), k in zip(jobs, keys):
        if k is None or counts.get(k, 0) < 2:
            out_jobs.append((cfg, entry))
            continue
        if k not in cache:
            extra = dict(entry.get("train_kwargs") or {})
            bdl = _pick_bundle(bundles, cfg, entry, fast_cpu)
            feat_m = extra.get("feature_mode", "backbone")
            tmp_mod = None if feat_m == "backbone" else _build_entry_model(cfg, extra)
            pre = assemble_from_bundle(
                bdl, tmp_mod, pool_mode=extra.get("pool_mode", "avg"),
                feature_mode=feat_m, tta_extract=bool(extra.get("tta_extract", False)), use_aug_bank=k[9],
            )
            cache[k] = compute_warm_start_state(
                pre[0], pre[1], int(cfg["num_classes"]), head_type=k[6],
                feature_norm=k[3], feature_norm_mode=k[4], feature_dropout=k[5],
                head_kwargs=extra.get("head_kwargs"), warm_start_ridge=k[7],
            )
        ws = cache[k]
        if ws is None:
            out_jobs.append((cfg, entry))
        else:
            out_jobs.append((cfg, {**entry, "train_kwargs": {**dict(entry.get("train_kwargs") or {}), "precomputed_warm_start": ws}}))
    return out_jobs


def _co_train_group_key(cfg: dict, entry: dict, bundle: Optional[dict], fast_cpu: bool) -> Optional[tuple]:
    if bundle is None:
        return None
    extra = dict(entry.get("train_kwargs") or {})
    pool_m = extra.get("pool_mode", "avg")
    feat_m = extra.get("feature_mode", "backbone")
    can_use = entry.get("trainer") == "cached" or (fast_cpu and feat_m == "backbone" and pool_m == "avg")
    if not can_use or bool(extra.get("auto_tune_resilient", False)):
        return None
    head_t = str(extra.get("head_type", "linear"))
    h_kw = dict(extra.get("head_kwargs") or {})
    if head_t not in ("linear", "mlp", "residual"):
        return None
    if head_t == "residual" and int(h_kw.get("depth", 1)) != 1:
        return None
    use_aug = bool(entry.get("use_aug_bank", False) or (fast_cpu and entry.get("train_transform") is not None))
    return (
        id(bundle),
        pool_m,
        feat_m,
        float(extra.get("desc_weight", 1.0)),
        bool(extra.get("tta_extract", False)),
        use_aug,
        head_t,
        bool(extra.get("feature_norm", False)),
        extra.get("feature_norm_mode", None),
        float(extra.get("feature_dropout", 0.0)),
        float(h_kw.get("curv_strength", 1.0)),
        bool(h_kw.get("bias", True)),
        int(h_kw.get("hidden_dim", 256)),
        int(h_kw.get("depth", 1)),
        bool(h_kw.get("out_norm", True)),
        str(cfg.get("model", "resnet18")),
        int(cfg.get("seed", 42)),
        int(cfg["num_classes"]),
        int(cfg.get("batch_size", 32)),
        int(cfg.get("epochs", 15)),
        float(extra.get("mixup_alpha", 0.0)),
        float(extra.get("cutmix_alpha", 0.0)),
        extra.get("warm_start_ridge", None),
    )


_CO_TRAIN_MAX_GROUP = 16


def _run_jobs_co_train(
    jobs: List[Tuple[dict, dict]],
    bundles: Dict[Tuple[str, str], dict],
    fast_cpu: bool,
    workers: Optional[int] = None,
) -> List[dict]:
    groups: Dict[tuple, List[int]] = {}
    leftover_idx: List[int] = []
    job_bundles: List[Optional[dict]] = []
    for idx, (cfg, entry) in enumerate(jobs):
        bdl = _pick_bundle(bundles, cfg, entry, fast_cpu)
        job_bundles.append(bdl)
        g_key = _co_train_group_key(cfg, entry, bdl, fast_cpu)
        if g_key is None:
            leftover_idx.append(idx)
        else:
            groups.setdefault(g_key, []).append(idx)

    out: List[Optional[dict]] = [None] * len(jobs)
    for idxs in groups.values():
        if len(idxs) < 2:
            leftover_idx.extend(idxs)
            continue
        bdl = job_bundles[idxs[0]]
        cfg0, ent0 = jobs[idxs[0]]
        ex0 = dict(ent0.get("train_kwargs") or {})
        feat_m = ex0.get("feature_mode", "backbone")
        pool_m = ex0.get("pool_mode", "avg")
        use_aug = bool(ent0.get("use_aug_bank", False) or (fast_cpu and ent0.get("train_transform") is not None))
        tmp_mod = None if feat_m == "backbone" else _build_entry_model(cfg0, ex0)
        pre = assemble_from_bundle(
            bdl, tmp_mod, pool_mode=pool_m, feature_mode=feat_m,
            tta_extract=bool(ex0.get("tta_extract", False)), use_aug_bank=use_aug,
        )
        std_state = tmp_mod.standardizer.state_dict() if (tmp_mod is not None and hasattr(tmp_mod, "standardizer")) else None
        for start in range(0, len(idxs), _CO_TRAIN_MAX_GROUP):
            sub_idxs = idxs[start:start + _CO_TRAIN_MAX_GROUP]
            if len(sub_idxs) < 2:
                leftover_idx.extend(sub_idxs)
                continue
            specs = [
                {
                    "config": jobs[i][0],
                    "train_kwargs": dict(jobs[i][1].get("train_kwargs") or {}),
                    "optimizer": jobs[i][1].get("optimizer"),
                    "scheduler": jobs[i][1].get("scheduler"),
                    "precomputed_standardizer_state": std_state,
                }
                for i in sub_idxs
            ]
            res_list = co_train_heads_cached(specs, *pre, bdl["classes"])
            for i, res in zip(sub_idxs, res_list):
                out[i] = res

    if leftover_idx:
        leftover_idx.sort()
        rem_res = _run_jobs([jobs[i] for i in leftover_idx], bundles, fast_cpu, workers=workers)
        for i, res in zip(leftover_idx, rem_res):
            out[i] = res
    return [r for r in out if r is not None]


def run_batch(
    axis: str = "all",
    split_csv: Optional[str] = None,
    data_root: Optional[str] = None,
    out_dir: Optional[str] = None,
    num_classes: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    skip_existing: bool = True,
    summarize: bool = True,
    limit: Optional[int] = None,
    shared_cache: bool = False,
    fast_cpu: bool = False,
    k_folds: int = 1,
    max_runs: Optional[int] = None,
    use_disk_cache: bool = True,
    cache_dir: Optional[str] = None,
    halving: bool = False,
    halving_factor: int = 3,
    workers: Optional[int] = None,
    auto_tune_resilient_flag: bool = False,
    co_train: bool = False,
    bundle_path: Optional[str] = None,
) -> List[dict]:
    """Execute experiment axes or grid search with shared disk cache, halving, or K-Fold CV."""
    groups = select_experiments(axis)
    eff_limit = max_runs if max_runs is not None else limit
    overrides = {
        "split_csv": split_csv, "data_root": data_root, "out_dir": out_dir,
        "num_classes": num_classes, "epochs": epochs, "batch_size": batch_size, "num_workers": num_workers,
    }
    bundles = _maybe_extract_bundles(
        groups, overrides, eff_limit, skip_existing, (shared_cache or co_train), fast_cpu,
        use_disk_cache=use_disk_cache, cache_dir=cache_dir, bundle_path=bundle_path,
    )
    if auto_tune_resilient_flag:
        _apply_auto_tune_resilient(groups, overrides, bundles, use_disk_cache=use_disk_cache, cache_dir=cache_dir)
    if halving:
        for ax_k, exps in list(groups.items()):
            sliced = exps[:eff_limit] if eff_limit is not None else exps
            groups[ax_k] = _halving_select_entries(sliced, overrides, bundles, fast_cpu, halving_factor=halving_factor, workers=workers, co_train=co_train)
            eff_limit = None
    if int(k_folds) >= 2:
        return _run_kfold_batch(groups, overrides, int(k_folds), eff_limit, shared_cache, use_disk_cache=use_disk_cache, cache_dir=cache_dir)
    jobs: List[Tuple[dict, dict]] = []
    resolved_out: Optional[Path] = None
    for _, exps in groups.items():
        for entry in (exps[:eff_limit] if eff_limit is not None else exps):
            cfg = _override_config(entry["config"], overrides)
            resolved_out = Path(cfg["out_dir"])
            if skip_existing and _is_done(resolved_out, cfg["name"], int(cfg["epochs"])):
                continue
            jobs.append((cfg, entry))
    jobs = _attach_shared_warm_starts(jobs, bundles, fast_cpu)
    if co_train and bundles:
        results = _run_jobs_co_train(jobs, bundles, fast_cpu, workers=workers)
    else:
        results = _run_jobs(jobs, bundles, fast_cpu, workers=workers)
    if summarize and resolved_out is not None:
        _write_reports(resolved_out, groups)
    return results


def _parse_cli_value(raw: str):
    """Parse CLI --set value into None, bool, int, float, or str."""
    low = raw.strip().lower()
    if low in ("none", "null"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _format_options_table() -> str:
    """Format SUPPORTED_OPTIONS for `--list-options`."""
    fallback = {"pool_mode": "avg", "head_type": "linear", "activation": "smooth", "optimizer_mode": "adamw", "mix_mode": "none", "class_weight_mode": None}
    lines = ["사용 가능한 기법 옵션 (--single / configs/grid.py):"]
    for key, vals in grid_cfg.SUPPORTED_OPTIONS.items():
        dfl = grid_cfg.DEFAULTS[key] if key in grid_cfg.DEFAULTS else axes_cfg.BASE.get(key, fallback.get(key, "-"))
        v_str = ", ".join("None" if v is None else str(v) for v in vals)
        lines.append(f"  - {key} (기본={dfl}): {v_str}")
    return "\n".join(lines)


def _collect_single_choices(args: argparse.Namespace) -> dict:
    """Collect user-selected technique flags and `--set key=val` pairs for `--single`."""
    flag_map = {
        "pool": "pool_mode", "head": "head_type", "activation": "activation", "norm": "feature_norm_mode",
        "loss": "loss_mode", "opt": "optimizer_mode", "scheduler": "scheduler", "mix": "mix_mode",
        "aug": "aug_mode", "cw": "class_weight_mode", "swa": "swa_epochs", "lr": "lr",
        "head_lr_scale": "head_lr_scale", "warm_start": "warm_start_ridge",
        "eval_resize": "eval_resize_mode", "tta": "eval_tta_mode", "trainer": "trainer",
    }
    choices = {}
    for attr in ("split_csv", "data_root", "out_dir", "num_classes", "epochs", "batch_size", "num_workers"):
        val = getattr(args, attr, None)
        if val is not None:
            choices[attr] = val
    for attr, dest in flag_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            choices[dest] = None if (attr == "cw" and str(val).lower() == "none") else val
    for item in (getattr(args, "set", None) or []):
        if "=" not in item:
            raise ValueError(f"--set must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        choices[k.strip()] = _parse_cli_value(v)
    return choices


def _print_single_summary(res: dict) -> None:
    p = res.get("grid_params") or {}
    lines = [
        f"=== Single Run: {res['name']} ===",
        f"  Options  : pool={p.get('pool', '-')}, head={p.get('head', '-')}, norm={p.get('norm', '-')}, opt={p.get('opt', '-')}, mix={p.get('mix', '-')}, aug={p.get('aug', '-')}, cw={p.get('cw', '-')}, swa={p.get('swa', 0)}",
        f"  Best Val : {float(res['best_val_acc']):.4f}",
    ]
    if "test_accuracy" in res:
        lines.append(f"  Test Acc : {float(res['test_accuracy']):.4f} | Macro F1: {float(res['test_macro_f1']):.4f}")
    lines.append(f"  Artifact : {res.get('best_ckpt', '-')}")
    print("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run axis, Grid Search, or single custom FC experiments.")
    parser.add_argument("--axis", choices=("all", "lr", "scheduler", "transform", "loss", "activation", "compare", "grid"), default="all")
    parser.add_argument("--single", action="store_true", help="Run one custom technique combination and evaluate it.")
    parser.add_argument("--list-options", action="store_true", help="List available single/grid technique options.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--pool", default=None)
    parser.add_argument("--head", default=None)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--norm", default=None)
    parser.add_argument("--loss", default=None)
    parser.add_argument("--opt", default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--mix", default=None)
    parser.add_argument("--aug", default=None)
    parser.add_argument("--cw", default=None)
    parser.add_argument("--swa", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--head-lr-scale", type=float, default=None)
    parser.add_argument("--warm-start", type=float, default=None)
    parser.add_argument("--eval-resize", default=None)
    parser.add_argument("--tta", choices=("none", "flip2", "multiview4"), default=None)
    parser.add_argument("--trainer", choices=("cached", "standard"), default=None)
    parser.add_argument("--set", action="append", default=[], help="Extra key=value overrides (repeatable).")
    parser.add_argument("--no-test", action="store_true", help="Skip final Test evaluation in --single mode.")
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--shared-cache", action="store_true")
    parser.add_argument("--fast-cpu", action="store_true")
    parser.add_argument("--k-folds", type=int, default=1)
    parser.add_argument("--no-disk-cache", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-file", default=None, help="Path to a pre-extracted portable .pt bundle (read-only mount supported).")
    parser.add_argument("--export-cache", nargs="?", const="__AUTO__", default=None, help="Extract and save a portable .pt bundle (with test+aug) for mounting/sharing.")
    parser.add_argument("--halving", action="store_true")
    parser.add_argument("--halving-factor", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--auto-tune-resilient", action="store_true")
    parser.add_argument("--co-train", action="store_true")
    parser.add_argument("--check-split", action="store_true")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--expected-classes", type=int, default=axes_cfg.BASE["num_classes"])
    args = parser.parse_args(argv)

    if args.list_options:
        print(_format_options_table())
        return 0

    if args.check_split:
        rep = inspect_split(
            args.split_csv or axes_cfg.BASE["split_csv"],
            data_root=args.data_root or axes_cfg.BASE["data_root"],
            check_files=args.check_files,
            expected_classes=args.expected_classes,
        )
        print(format_split_report(rep))
        return 0 if rep["ok"] else 1

    if args.export_cache is not None:
        overrides = {
            "split_csv": args.split_csv, "data_root": args.data_root, "out_dir": args.out_dir,
            "num_classes": args.num_classes, "epochs": args.epochs, "batch_size": args.batch_size, "num_workers": args.num_workers,
        }
        export_shared_bundle(axis=args.axis, overrides=overrides, export_path=args.export_cache, fast_cpu=args.fast_cpu)
        return 0

    if args.single:
        res = run_single_experiment(
            _collect_single_choices(args),
            name=args.name,
            eval_test=not args.no_test,
            tta_mode=args.tta,
            use_disk_cache=not args.no_disk_cache,
            cache_dir=args.cache_dir,
            bundle_path=args.cache_file,
        )
        _print_single_summary(res)
        return 0

    run_batch(
        axis=args.axis, split_csv=args.split_csv, data_root=args.data_root, out_dir=args.out_dir,
        num_classes=args.num_classes, epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, skip_existing=not args.no_skip, summarize=not args.no_summarize,
        limit=args.limit, shared_cache=args.shared_cache, fast_cpu=args.fast_cpu, k_folds=args.k_folds,
        max_runs=args.max_runs, use_disk_cache=not args.no_disk_cache, cache_dir=args.cache_dir,
        halving=args.halving, halving_factor=args.halving_factor, workers=args.workers,
        auto_tune_resilient_flag=args.auto_tune_resilient, co_train=args.co_train, bundle_path=args.cache_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
