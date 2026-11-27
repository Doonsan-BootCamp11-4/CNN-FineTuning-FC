"""Tests for experiment axes, compare presets, Grid Search, K-Fold CV, spawn workers, and reports."""

import json
from pathlib import Path
import pickle
from PIL import Image
import torch
from torchvision import transforms

from configs import axes as axes_cfg
from configs import compare as compare_cfg
from configs import grid as grid_cfg
from fcft.evaluate import evaluate, evaluate_ensemble
from fcft.grid import build_custom_entry, expand_grid, run_single_experiment
from fcft.kfold import build_stratified_folds, run_kfold
from fcft.model import build_model
from fcft.report import collect_summaries, format_before_after_table, format_markdown_table, plot_comparison_curves, save_summary_csv, select_best_by_axis
from fcft.train import _set_seed, run
from scripts.run_experiments import main as cli_main, run_batch, select_experiments


# 확인: 모든 실험 축 계약, picklable 팩토리 직렬화(spawn 호환), 16개 LR 실험 전수 E2E 학습을 검증 (계약 / 직렬화 / E2E)
def test_axes_contracts_picklable_factories_and_lr_e2e(fake_dataset):
    std_entries = axes_cfg.LR_EXPERIMENTS + axes_cfg.SCHEDULER_EXPERIMENTS + axes_cfg.TRANSFORM_EXPERIMENTS
    assert len(std_entries) >= 28 and len(axes_cfg.LR_EXPERIMENTS) == 16
    seen = set()
    for entry in std_entries:
        assert set(entry.keys()) == {"config", "optimizer", "scheduler", "train_transform"}
        cfg = entry["config"]
        assert cfg["name"] not in seen
        seen.add(cfg["name"])
        json.dumps(cfg)
        pickle.dumps(entry)

    sample_img = Image.new("RGB", (64, 48), color=(120, 180, 90))
    for entry in axes_cfg.TRANSFORM_EXPERIMENTS:
        assert entry["train_transform"](sample_img).shape == (3, 224, 224)

    _set_seed(int(fake_dataset["config"]["seed"]))
    ref = build_model(name="resnet18", num_classes=fake_dataset["config"]["num_classes"])
    ref_conv1, ref_bn1, ref_fc_w = ref.conv1.weight.detach().clone(), ref.bn1.running_mean.detach().clone(), ref.fc.weight.detach().clone()

    for entry in axes_cfg.LR_EXPERIMENTS:
        cfg = {**entry["config"], "split_csv": fake_dataset["config"]["split_csv"], "data_root": fake_dataset["config"]["data_root"], "out_dir": fake_dataset["config"]["out_dir"], "num_classes": 3, "num_workers": 0, "epochs": 1, "batch_size": 4}
        res = run(cfg, optimizer=entry["optimizer"])
        assert torch.isfinite(torch.tensor([res["history"][0]["train_loss"], res["history"][0]["val_loss"]])).all().item()
        sd = torch.load(Path(cfg["out_dir"]) / cfg["name"] / "best.pt", map_location="cpu", weights_only=True)["model_state_dict"]
        assert torch.equal(sd["conv1.weight"], ref_conv1) and torch.equal(sd["bn1.running_mean"], ref_bn1) and not torch.equal(sd["fc.weight"], ref_fc_w)


# 확인: compare(8단계), loss(12종), activation(7종) 축 실행·비교표·플롯·skip_existing·Top-2 앙상블을 검증 (정상 동작 / E2E)
def test_compare_loss_activation_axes_and_reports(fake_dataset):
    assert len(compare_cfg.EXPERIMENTS) == 8 and len(axes_cfg.LOSS_EXPERIMENTS) == 12 and len(axes_cfg.ACTIVATION_EXPERIMENTS) == 7
    assert select_experiments("compare") == {"compare": compare_cfg.EXPERIMENTS}
    assert select_experiments("loss") == {"loss": axes_cfg.LOSS_EXPERIMENTS}
    assert select_experiments("activation") == {"activation": axes_cfg.ACTIVATION_EXPERIMENTS}
    pickle.dumps(compare_cfg.EXPERIMENTS[1])
    pickle.dumps(axes_cfg.LOSS_EXPERIMENTS[0])

    base_cfg = fake_dataset["config"]
    out_cmp = Path(base_cfg["out_dir"]) / "cmp"
    r_cmp = run_batch(axis="compare", split_csv=base_cfg["split_csv"], data_root=base_cfg["data_root"], out_dir=str(out_cmp), num_classes=3, epochs=1, batch_size=2, num_workers=0, skip_existing=True)
    assert len(r_cmp) == 8 and (out_cmp / "compare_before_after.md").is_file() and (out_cmp / "comparison_compare.png").is_file()
    assert len(run_batch(axis="compare", split_csv=base_cfg["split_csv"], data_root=base_cfg["data_root"], out_dir=str(out_cmp), num_classes=3, epochs=1, batch_size=2, num_workers=0, skip_existing=True)) == 0

    ranked = collect_summaries(out_cmp, names=[e["config"]["name"] for e in compare_cfg.EXPERIMENTS])
    assert "0.00%p" in format_before_after_table(ranked)
    m_ens = evaluate_ensemble([{**base_cfg, "out_dir": str(out_cmp), "name": r["name"]} for r in ranked[:2]], out_name="cmp_ens", tta_mode="multiview4")
    assert sum(sum(row) for row in m_ens["confusion_matrix"]) == 3

    out_loss = Path(base_cfg["out_dir"]) / "axis_loss"
    run_batch(axis="loss", split_csv=base_cfg["split_csv"], data_root=base_cfg["data_root"], out_dir=str(out_loss), num_classes=3, epochs=1, batch_size=2, num_workers=0, limit=3, shared_cache=True, auto_tune_resilient_flag=True)
    assert (out_loss / "loss_comparison.md").is_file() and (out_loss / "tuned_resilient.json").is_file() and select_best_by_axis(out_loss)["loss"] is not None

    out_act = Path(base_cfg["out_dir"]) / "axis_act"
    run_batch(axis="activation", split_csv=base_cfg["split_csv"], data_root=base_cfg["data_root"], out_dir=str(out_act), num_classes=3, epochs=1, batch_size=2, num_workers=0, limit=4, shared_cache=True)
    assert (out_act / "activation_comparison.md").is_file() and select_best_by_axis(out_act)["activation"] is not None
    assert 0.0 <= evaluate({**base_cfg, "out_dir": str(out_act), "name": "act_gelu"}, tta_mode="flip2")["accuracy"] <= 1.0


# 확인: Grid 16조합, 확장 옵션(loss_mode/activation/hlr), halving, K-Fold 비오염 및 리포트 유틸을 검증 (Grid / K-Fold)
def test_grid_halving_kfold_and_report_helpers(fake_dataset):
    assert len(grid_cfg.EXPERIMENTS) == 16
    pickle.dumps(grid_cfg.EXPERIMENTS[0])
    custom = expand_grid(
        {"loss_mode": ["resilient_auto"], "activation": ["smooth_learn"], "feature_norm_mode": ["center_std_power"], "head_lr_scale": [0.2]},
        base_config=fake_dataset["config"], defaults={"head_type": "residual", "trainer": "cached"},
    )
    tk = custom[0]["train_kwargs"]
    assert tk["loss_cap_mode"] == "mad" and tk["loss_cap_alpha"] == 0.5 and tk["class_weight_mode"] == "effective"
    assert tk["head_kwargs"]["activation"] == "smooth_learn" and tk["head_kwargs"]["learnable_leak"] is True and tk["head_lr_scale"] == 0.2

    base = fake_dataset["config"]
    classes, test_samples, folds = build_stratified_folds(base["split_csv"], num_classes=3, k_folds=3, seed=42)
    test_paths = {p for p, _ in test_samples}
    assert all(not (test_paths & {p for p, _ in tr_s}) and not (test_paths & {p for p, _ in va_s}) for tr_s, va_s in folds)
    kf = run_kfold({**base, "name": "kf_test", "epochs": 1}, k_folds=3, trainer="cached", cutmix_alpha=1.0)
    assert len(kf["fold_val_accs"]) == 3 and Path(kf["best_fold_ckpt"]).is_file()

    out_halv = Path(base["out_dir"]) / "grid_halving"
    r_h = run_batch(axis="grid", split_csv=base["split_csv"], data_root=base["data_root"], out_dir=str(out_halv), num_classes=3, epochs=2, batch_size=2, num_workers=0, max_runs=6, shared_cache=True, halving=True, halving_factor=3)
    assert len(r_h) == 2 and not (out_halv / "_screen").exists() and (out_halv / "grid_summary.md").is_file()

    recs = collect_summaries(out_halv)
    assert "| `grid_" in format_markdown_table(recs)
    assert save_summary_csv(recs, out_halv / "custom.csv").is_file()
    assert plot_comparison_curves(out_halv, [recs[0]["name"]], out_halv / "cmp.png").is_file()


# 확인: spawn 멀티워커(workers=2)가 fork 경고 없이 순차(workers=1)와 동일한 정확도·산출물을 내는지 검증 (병렬 정합성 / Windows·Linux 호환)
def test_parallel_spawn_workers_parity_without_fork_warning(fake_dataset, recwarn):
    base = fake_dataset["config"]
    out_seq, out_par = Path(base["out_dir"]) / "grid_seq", Path(base["out_dir"]) / "grid_par"
    r_seq = run_batch(axis="grid", split_csv=base["split_csv"], data_root=base["data_root"], out_dir=str(out_seq), num_classes=3, epochs=2, batch_size=2, num_workers=0, max_runs=4, shared_cache=True, workers=1)
    r_par = run_batch(axis="grid", split_csv=base["split_csv"], data_root=base["data_root"], out_dir=str(out_par), num_classes=3, epochs=2, batch_size=2, num_workers=0, max_runs=4, shared_cache=True, workers=2)
    assert len(r_seq) == 4 and len(r_par) == 4
    for rs, rp in zip(r_seq, r_par):
        assert rs["name"] == rp["name"] and abs(float(rs["best_val_acc"]) - float(rp["best_val_acc"])) < 1e-4
    fork_warns = [w for w in recwarn if "fork" in str(w.message).lower()]
    assert not fork_warns


# 확인: --co-train 다중 헤드 bmm 병렬 학습이 loss/activation 축에서 체크포인트·리포트·평가 호환성을 유지하는지 검증 (Multi-Head bmm Co-Training)
def test_multihead_co_train_loss_and_activation_axes(fake_dataset):
    base = fake_dataset["config"]
    out_loss = Path(base["out_dir"]) / "co_loss"
    r_loss = run_batch(
        axis="loss", split_csv=base["split_csv"], data_root=base["data_root"],
        out_dir=str(out_loss), num_classes=3, epochs=2, batch_size=2, num_workers=0,
        limit=4, shared_cache=True, co_train=True,
    )
    assert len(r_loss) == 4 and (out_loss / "loss_comparison.md").is_file()
    for row in r_loss:
        assert Path(row["best_ckpt"]).is_file() and len(row["history"]) == 2
        m = evaluate({**base, "out_dir": str(out_loss), "name": row["name"]}, tta_mode="none")
        assert 0.0 <= m["accuracy"] <= 1.0

    out_act = Path(base["out_dir"]) / "co_act"
    r_act = run_batch(
        axis="activation", split_csv=base["split_csv"], data_root=base["data_root"],
        out_dir=str(out_act), num_classes=3, epochs=2, batch_size=2, num_workers=0,
        limit=3, shared_cache=True, co_train=True,
    )
    assert len(r_act) == 3 and (out_act / "activation_comparison.md").is_file()
    m_act = evaluate({**base, "out_dir": str(out_act), "name": r_act[0]["name"]}, tta_mode="flip2")
    assert 0.0 <= m_act["accuracy"] <= 1.0

    from fcft.losses import UnifiedClassificationLoss
    from fcft.train import _eval_cached
    crit_mad = UnifiedClassificationLoss(num_classes=3, loss_cap_mode="mad", loss_cap_tau=0.8, loss_cap_alpha=0.5, loss_cap_param=1.75)
    crit_mad.train()
    _ = crit_mad(torch.randn(8, 3) * 3.0, torch.randint(0, 3, (8,)))
    tau_before = float(crit_mad.cap_gate.ema_tau.item())
    _eval_cached(torch.nn.Linear(8, 3), torch.randn(10, 8), torch.randint(0, 3, (10,)), 5, torch.device("cpu"), crit_mad)
    assert float(crit_mad.cap_gate.ema_tau.item()) == tau_before and not crit_mad.training

    out_co_halv = Path(base["out_dir"]) / "co_halv"
    r_ch = run_batch(
        axis="loss", split_csv=base["split_csv"], data_root=base["data_root"],
        out_dir=str(out_co_halv), num_classes=3, epochs=2, batch_size=2, num_workers=0,
        max_runs=6, shared_cache=True, halving=True, halving_factor=3, co_train=True,
    )
    assert len(r_ch) == 2 and not (out_co_halv / "_screen").exists()

    orig_loss_kw = dict(axes_cfg.LOSS_EXPERIMENTS[-1]["train_kwargs"])
    out_at = Path(base["out_dir"]) / "co_at_immut"
    run_batch(
        axis="loss", split_csv=base["split_csv"], data_root=base["data_root"],
        out_dir=str(out_at), num_classes=3, epochs=2, batch_size=2, num_workers=0,
        limit=2, shared_cache=True, co_train=True, auto_tune_resilient_flag=True,
    )
    assert axes_cfg.LOSS_EXPERIMENTS[-1]["train_kwargs"] == orig_loss_kw and (out_at / "tuned_resilient.json").is_file()


# 확인: 직접 기법 선택 단일 학습·평가 API(build_custom_entry/run_single_experiment) 및 CLI(--list-options/--single)를 검증 (단일 선택 실행 / E2E)
def test_single_custom_experiment_api_and_cli(fake_dataset, capsys):
    import pytest

    base = fake_dataset["config"]
    with pytest.raises(ValueError):
        build_custom_entry({"unknown_flag_xyz": 1}, base_config=base)

    custom_entry = build_custom_entry(
        {"head_type": "mlp", "activation": "gelu", "loss_mode": "focal_g2.0_ls", "mix_mode": "both"},
        base_config=base,
    )
    assert custom_entry["config"]["name"].startswith("single_")
    pickle.dumps(custom_entry)

    out_api = Path(base["out_dir"]) / "single_api"
    res_api = run_single_experiment(
        {
            "out_dir": str(out_api), "epochs": 1, "batch_size": 2, "num_workers": 0,
            "head_type": "mlp", "activation": "gelu", "loss_mode": "focal_g2.0_ls",
            "mix_mode": "both", "use_aug_bank": False, "swa_epochs": 0,
        },
        base_config=base,
        name="my_mlp_focal",
        eval_test=True,
        tta_mode="flip2",
        use_disk_cache=False,
    )
    assert res_api["name"] == "my_mlp_focal" and 0.0 <= res_api["test_accuracy"] <= 1.0
    assert (out_api / "my_mlp_focal" / "metrics.json").is_file()
    assert (out_api / "my_mlp_focal" / "confusion_matrix.png").is_file()

    res_std = run_single_experiment(
        {"out_dir": str(out_api), "epochs": 1, "batch_size": 2, "num_workers": 0, "trainer": "standard", "head_type": "linear"},
        base_config=base,
        name="my_std_lin",
        eval_test=False,
    )
    assert res_std["name"] == "my_std_lin" and "test_accuracy" not in res_std

    assert cli_main(["--list-options"]) == 0
    assert "head_type" in capsys.readouterr().out

    out_cli = Path(base["out_dir"]) / "single_cli"
    rc = cli_main([
        "--single", "--name", "cli_res_cw",
        "--split-csv", base["split_csv"], "--data-root", base["data_root"], "--out-dir", str(out_cli),
        "--num-classes", "3", "--epochs", "1", "--batch-size", "2", "--num-workers", "0",
        "--head", "residual", "--loss", "cw_inv_sqrt", "--mix", "cutmix", "--activation", "silu",
        "--tta", "none", "--set", "use_aug_bank=false", "--set", "swa_epochs=0", "--no-disk-cache",
    ])
    out_txt = capsys.readouterr().out
    assert rc == 0 and "Single Run: cli_res_cw" in out_txt and "Test Acc" in out_txt
    assert (out_cli / "cli_res_cw" / "metrics.json").is_file()
