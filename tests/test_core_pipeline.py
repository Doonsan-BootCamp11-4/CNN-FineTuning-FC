"""Tests for model assembly, training loops, losses, feature norms, two-speed LR, and evaluation."""

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from fcft.evaluate import evaluate, evaluate_ensemble
from fcft.losses import UnifiedClassificationLoss, apply_cutmix, compute_class_weights
from fcft.model import (
    CosineClassifierHead,
    DualAnchorHybridHead,
    FeatureCenterNormLayer,
    FeatureNormLayer,
    MLPClassifierHead,
    SpectralNormLayer,
    SubCenterHead,
    build_backbone,
    build_model,
)
import fcft.train as train_mod
from fcft.train import _apply_head_lr_scale, run, run_cached


# 확인: 모델 동결·출력 shape·풀링 차원·미지원 인자 예외 및 4D/2D CutMix·UnifiedLoss 계산을 검증 (정상 동작 / 예외)
def test_model_pools_heads_and_unified_loss():
    model = build_model(name="resnet18", num_classes=12)
    assert set(n for n, p in model.named_parameters() if p.requires_grad) == {"fc.weight", "fc.bias"}
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 12)

    x = torch.randn(2, 3, 224, 224)
    for mode, dim in {"avg": 512, "avgmax": 1024, "gem": 512, "multiscale": 768, "norm_multiscale": 768}.items():
        bb, out_dim = build_backbone("resnet18", pool_mode=mode)
        assert out_dim == dim and bb(x).shape == (2, dim)

    for head_t in ("residual", "mlp", "dual_anchor"):
        assert build_model("resnet18", 3, head_type=head_t, activation="gelu")(x).shape == (2, 3)

    with pytest.raises(ValueError):
        build_model(name="unknown_cnn", num_classes=12)
    with pytest.raises(ValueError):
        build_model(name="resnet18", num_classes=12, head_type="unknown_head")
    with pytest.raises(ValueError):
        build_backbone("resnet18", pool_mode="invalid_pool")
    with pytest.raises(ValueError):
        build_model(num_classes=4, feature_mode="invalid_mode")

    y_h = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    xm4, ym4, _ = apply_cutmix(torch.rand(4, 3, 32, 32), F.one_hot(y_h, num_classes=3).float(), y_h, alpha=1.0)
    assert xm4.shape == (4, 3, 32, 32) and torch.allclose(ym4.sum(dim=-1), torch.ones(4), atol=1e-5)

    w_eff = compute_class_weights(torch.tensor([0, 0, 0, 0, 1, 1, 2], dtype=torch.long), num_classes=3, mode="effective", beta=0.9)
    crit = UnifiedClassificationLoss(num_classes=3, label_smoothing=0.05, focal_gamma=2.0, class_weights=w_eff, logit_margin=0.1)
    loss, ok = crit.forward_batch(SubCenterHead(16, 3, 3), torch.randn(7, 16), torch.tensor([0, 0, 0, 0, 1, 1, 2]), mixup_alpha=0.3, cutmix_alpha=0.5)
    assert torch.isfinite(loss).item() and 0 <= ok <= 7
    crit_res = UnifiedClassificationLoss(num_classes=3, label_smoothing=0.05, class_weights=w_eff, loss_cap_tau=0.8, loss_cap_alpha=0.5)
    logits_r = torch.randn(7, 3, requires_grad=True)
    loss_r = crit_res(logits_r, torch.tensor([0, 0, 0, 0, 1, 1, 2]))
    loss_r.backward()
    assert torch.isfinite(loss_r).item() and torch.isfinite(logits_r.grad).all().item()


# 확인: 동결 BN 유지, 커스텀 옵티마이저·ReduceLROnPlateau, 학습 중단 시 history 보존, evaluate 산출물을 검증 (요구사항 / 예외)
def test_standard_train_scheduler_interrupt_and_evaluate(fake_dataset, monkeypatch):
    cfg = fake_dataset["config"]
    ref = build_model(name=cfg["model"], num_classes=cfg["num_classes"])
    ref_conv1, ref_bn1 = ref.conv1.weight.detach().clone(), ref.bn1.running_mean.detach().clone()

    train_mod.run(cfg)
    run_dir = Path(cfg["out_dir"]) / cfg["name"]
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert torch.equal(ckpt["model_state_dict"]["conv1.weight"], ref_conv1)
    assert torch.equal(ckpt["model_state_dict"]["bn1.running_mean"], ref_bn1)

    m = evaluate(cfg)
    assert 0.0 <= m["accuracy"] <= 1.0 and sum(sum(r) for r in m["confusion_matrix"]) == 3
    assert set(m["per_class"].keys()) == set(fake_dataset["classes_sorted"])
    assert (run_dir / "metrics.json").is_file() and (run_dir / "confusion_matrix.png").is_file() and (run_dir / "curves.png").is_file()

    cfg_opt = {**cfg, "name": "opt_sched", "epochs": 2}
    res_opt = run(cfg_opt, optimizer=lambda p: torch.optim.SGD(p, lr=0.5), scheduler=lambda o: torch.optim.lr_scheduler.StepLR(o, 1, 0.1))
    assert res_opt["history"][0]["lr"] == pytest.approx(0.5) and res_opt["history"][1]["lr"] == pytest.approx(0.05)

    res_plat = run({**cfg, "name": "plateau", "epochs": 3}, optimizer=lambda p: torch.optim.SGD(p, lr=0.1), scheduler=lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5, patience=0))
    assert len(res_plat["history"]) == 3 and res_plat["final_lr"] <= 0.1

    orig_ep, calls = train_mod.run_once_epoch, {"n": 0}

    def fail_second(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("interrupted")
        return orig_ep(*a, **kw)

    monkeypatch.setattr(train_mod, "run_once_epoch", fail_second)
    with pytest.raises(RuntimeError):
        train_mod.run({**cfg, "name": "interrupt_run", "epochs": 2})
    assert len(json.loads((Path(cfg["out_dir"]) / "interrupt_run" / "history.json").read_text(encoding="utf-8"))) == 1


# 확인: Cosine/Residual/SubCenter/DualAnchor/MLP 헤드, Mixup·EMA·SWA, Descriptor/Hybrid TTA 비붕괴 및 앙상블을 검증 (정상 동작 / 회귀)
def test_rich_heads_descriptor_hybrid_cached_and_ensemble(fake_dataset):
    base = fake_dataset["config"]
    c_cos = {**base, "name": "rich_cos", "epochs": 1}
    run_cached(c_cos, head_type="cosine", feature_norm=True, pool_mode="multiscale", warm_start_ridge=10.0, class_weight_mode="effective", focal_gamma=1.5)
    m_cos = evaluate(c_cos, tta_mode="multiview4")

    c_res = {**base, "name": "rich_res_ex", "epochs": 2}
    r_res = run_cached(c_res, head_type="residual", feature_norm=True, label_smoothing=0.05, mixup_alpha=0.2, ema_decay=0.9, warm_start_ridge=5.0, compress_cache="exact")
    assert r_res["cache_memory_bytes"] > 0

    c_desc = {**base, "name": "desc_tta", "epochs": 2}
    r_desc = run_cached(c_desc, feature_mode="descriptor", head_type="residual", feature_norm=True, warm_start_ridge=5.0, tta_extract=True, compress_cache=True)
    assert r_desc["raw_feature_bytes"] == 12 * 128 * 4

    c_hyb = {**base, "name": "hyb_sub", "epochs": 2}
    run_cached(c_hyb, feature_mode="hybrid", pool_mode="avgmax", head_type="subcenter", head_kwargs={"num_centers": 3}, feature_norm=True, warm_start_ridge=10.0, tta_extract=True, compress_cache=True)
    m_hyb = evaluate(c_hyb, tta_mode="multiview4")

    c_da = {**base, "name": "da_std", "epochs": 1}
    m_da_mod = build_model("resnet18", 3, head_type="dual_anchor", pool_mode="norm_multiscale", feature_norm_mode="spectral", feature_dropout=0.1)
    assert isinstance(m_da_mod.fc[-1], DualAnchorHybridHead) and isinstance(m_da_mod.fc[0], SpectralNormLayer)
    run(c_da, head_type="dual_anchor", pool_mode="norm_multiscale", feature_norm_mode="spectral", logit_adj_tau=0.5, swa_epochs=1)

    ens = evaluate_ensemble([c_cos, c_da], out_name="rich_ens", weights=[0.6, 0.4], tta_mode="multiview4", temperature=1.2)
    assert 0.0 <= m_cos["accuracy"] <= 1.0 and 0.0 <= m_hyb["accuracy"] <= 1.0 and sum(sum(r) for r in ens["confusion_matrix"]) == 3


# 확인: 1-Pass 채널 정규화(center_power_l2/l2/center_std/center_std_power), 체크포인트 호환, 2속도 LR 분리를 검증 (정규화 / 역전파)
def test_onepass_feature_norms_and_two_speed_head_lr(fake_dataset):
    c_pow = FeatureCenterNormLayer(8, scale_std=False, power=True)
    assert "inv_std" not in dict(c_pow.named_buffers())
    c_pow.fit_stats(torch.tensor([[2.0] * 8, [4.0] * 8]))
    assert torch.allclose(c_pow.mean, torch.full((1, 8), 3.0))

    c_std = FeatureCenterNormLayer(4, scale_std=True, power=False)
    c_std.fit_stats(torch.tensor([[10.0, 0.0, 2.0, -4.0], [10.0, 4.0, 6.0, 4.0]]))
    assert torch.allclose(c_std.inv_std[0, 1:], torch.tensor([0.5, 0.5, 0.25]), atol=1e-4)
    assert torch.allclose(c_std(torch.tensor([[10.0, 4.0, 6.0, 4.0]])).norm(dim=-1), torch.ones(1), atol=1e-5)

    for nm in ("l2", "power_l2", "center_power_l2", "center_std", "center_std_power"):
        assert isinstance(build_model("resnet18", 3, head_type="residual", feature_norm_mode=nm).fc[0], (FeatureNormLayer, FeatureCenterNormLayer))

    m_res = build_model("resnet18", 3, head_type="residual", feature_norm_mode="center_std")
    req = [p for p in m_res.fc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in req if p.ndim >= 2], "weight_decay": 1e-2}, {"params": [p for p in req if p.ndim < 2], "weight_decay": 0.0}], lr=2e-3)
    with pytest.raises(ValueError):
        _apply_head_lr_scale(opt, m_res.fc, -0.1)
    _apply_head_lr_scale(opt, m_res.fc, 0.2)
    assert abs(opt.param_groups[0]["lr"] - 2e-3) < 1e-12 and any(abs(g["lr"] - 4e-4) < 1e-12 for g in opt.param_groups)

    cfg = {**fake_dataset["config"], "name": "cstd_hlr0", "epochs": 2}
    res = run_cached(cfg, head_type="residual", feature_norm_mode="center_std", warm_start_ridge=0.5, head_lr_scale=0.0)
    assert len(res["history"]) == 2 and 0.0 <= evaluate(cfg, tta_mode="flip2")["accuracy"] <= 1.0

    cfg_auto = {**fake_dataset["config"], "name": "resilient_auto_run", "epochs": 2}
    res_auto = run_cached(
        cfg_auto, head_type="residual", feature_norm_mode="center_power_l2",
        loss_cap_mode="mad", loss_cap_alpha=0.5, loss_cap_param=1.75,
        head_kwargs={"learnable_leak": True, "channel_leak": True}, auto_tune_resilient=True,
    )
    assert "auto_tuned" in res_auto and len(res_auto["auto_tuned"]["trials"]) >= 7
    assert 0.0 <= evaluate(cfg_auto, tta_mode="flip2")["accuracy"] <= 1.0

    c_auto = {**fake_dataset["config"], "name": "auto_raw_pre_norm", "epochs": 2}
    r_auto = run_cached(c_auto, head_type="residual", feature_norm_mode="center_power_l2", warm_start_ridge=0.5, compress_cache="auto")
    assert r_auto["cache_memory_bytes"] == r_auto["raw_feature_bytes"] > 0
    ws_payload = train_mod.compute_warm_start_state(
        torch.randn(12, 512), torch.tensor([0, 1, 2] * 4, dtype=torch.long),
        3, head_type="residual", feature_norm_mode="center_power_l2", warm_start_ridge=0.5,
    )
    assert ws_payload is not None and ws_payload["kind"] == "linear"
    c_ws = {**fake_dataset["config"], "name": "auto_ws_reuse", "epochs": 2}
    r_ws = run_cached(c_ws, head_type="residual", feature_norm_mode="center_power_l2", warm_start_ridge=0.5, compress_cache="auto", precomputed_warm_start=ws_payload)
    assert len(r_ws["history"]) == 2


# 확인: ResNet34/50 백본 확장, feature_noise_std·Macro-F1 체크포인트, K-Fold target_pools 고속 추출을 검증 (병목 해소 회귀)
def test_bottleneck_fixes_backbones_noise_f1_and_target_pools(fake_dataset):
    from fcft.kfold import build_stratified_folds, extract_kfold_pool_bundle

    x_s = torch.randn(1, 3, 224, 224)
    bb34, d34 = build_backbone("resnet34", pool_mode="norm_multiscale", pretrained=False)
    bb50, d50 = build_backbone("resnet50", pool_mode="norm_multiscale", pretrained=False)
    assert d34 == 768 and bb34(x_s).shape == (1, 768)
    assert d50 == 3072 and bb50(x_s).shape == (1, 3072)

    base = fake_dataset["config"]
    cfg_f1 = {**base, "name": "noise_f1_ckpt", "epochs": 2}
    res_f1 = run_cached(
        cfg_f1, head_type="residual", pool_mode="norm_multiscale", feature_norm_mode="power_l2",
        warm_start_ridge=0.5, cutmix_alpha=1.0, feature_noise_std=0.02, ckpt_metric="macro_f1",
    )
    assert "val_macro_f1" in res_f1["history"][0] and 0.0 <= evaluate(cfg_f1, tta_mode="flip2")["accuracy"] <= 1.0

    classes, _, folds = build_stratified_folds(base["split_csv"], num_classes=3, k_folds=2, seed=42)
    pool_samples = list(folds[0][0]) + list(folds[0][1])
    kb = extract_kfold_pool_bundle(
        base, pool_samples, classes, include_desc=False, use_disk_cache=False, target_pools=["norm_multiscale"],
    )
    assert set(kb["pool"].keys()) == {"norm_multiscale", "norm_multiscale_flip"}
