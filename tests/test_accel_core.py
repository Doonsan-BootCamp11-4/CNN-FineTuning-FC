"""Tests for compiled native kernels (.so/.dll), thread safety, 1-shot solvers, and quantized feature banks."""

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import subprocess
import pytest
import torch
import torch.nn as nn

from fcft.accel import (
    ACTIVATION_CHOICES,
    AdaptiveBoundedLossGate,
    CompressedFeatureBank,
    CurvatureBalancedOptimizer,
    FusedLinearHeadSession,
    ResidualAdapterHead,
    SmoothActivation,
    auto_tune_resilient,
    bounded_loss,
    build_activation,
    calibrated_constant,
    calibrated_log_softmax,
    calibrated_softmax,
    cosine_decay_factor,
    exact_log_priors,
    extract_dual_descriptors,
    extract_patch_descriptors,
    fit_balanced_ridge_head,
    fit_closed_form_head,
    fit_dual_anchor_prototypes,
    is_accel_available,
    multihead_linear_forward,
    smooth_activation,
    spectral_power_norm,
)
from fcft.model import build_backbone, fuse_backbone_conv_bn
from fcft.train import run, run_cached


# 확인: Linux(.so)/Windows(.dll)/core.vault 존재, 단일 export(fc_exec), 평문 소스·수식 비노출 및 pickle 호환을 검증 (요구사항 / 이식성)
def test_opaque_binary_loaded_and_stripped():
    assert is_accel_available()
    proj_root = Path(__file__).resolve().parent.parent
    bin_dir = proj_root / "fcft" / "bin"
    so_path, dll_path, vault_path = bin_dir / "libfc_accel.so", bin_dir / "fc_accel.dll", bin_dir / "core.vault"
    assert so_path.is_file() and dll_path.is_file() and vault_path.is_file()
    assert not list(proj_root.rglob("*.cpp")) and not list(proj_root.rglob("*.hpp"))

    nm_out = subprocess.check_output(["nm", "-D", str(so_path)], text=True)
    assert [ln.split()[-1] for ln in nm_out.splitlines() if " T " in ln] == ["fc_exec"]
    assert dll_path.read_bytes()[:2] == b"MZ" and vault_path.read_bytes()[:4] == b"FCV1"

    accel_src = (proj_root / "fcft" / "accel.py").read_text(encoding="utf-8")
    for hidden_tok in ("fc_exec", "3.75", "0.0625", "1/15"):
        assert hidden_tok not in accel_src

    assert ResidualAdapterHead.__module__ == "fcft.accel"
    restored = pickle.loads(pickle.dumps(ResidualAdapterHead(in_dim=16, num_classes=3, hidden_dim=8)))
    assert restored(torch.randn(2, 16)).shape == (2, 3)


# 확인: SmoothActivation·build_activation·ResidualAdapterHead 순전파·역전파 및 CurvatureBalancedOptimizer 스텝을 검증 (정상 동작 / 경계)
def test_activations_residual_head_and_optimizer_step():
    x = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=torch.float32, requires_grad=True)
    y = smooth_activation(x, leak=0.0625)
    y.sum().backward()
    assert torch.isfinite(y).all().item() and torch.isfinite(x.grad).all().item()

    x_a = torch.randn(3, 16, requires_grad=True)
    for act_name in (*ACTIVATION_CHOICES, "swish"):
        assert torch.isfinite(build_activation(act_name)(x_a)).all().item()
    with pytest.raises(ValueError):
        build_activation("bad_act")

    head = ResidualAdapterHead(in_dim=32, num_classes=4, hidden_dim=16, depth=1)
    opt = CurvatureBalancedOptimizer(head.parameters(), lr=1e-2, weight_decay=1e-4)
    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(head(torch.randn(8, 32)), torch.randint(0, 4, (8,)))
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()


# 확인: 1-Shot Ridge/Balanced Ridge 솔버 정확도 및 CurvatureBalancedOptimizer 실학습 호환성을 검증 (정상 동작)
def test_closed_form_head_fit_and_optimizer_with_train_run(fake_dataset):
    torch.manual_seed(7)
    feats = torch.randn(96, 64)
    targets = (feats @ torch.randn(64, 6)).argmax(dim=1)
    head = nn.Linear(64, 6)
    info = fit_closed_form_head(head, feats, targets, ridge=1.0)
    assert info["num_samples"] == 96
    assert float((head(feats).argmax(dim=1) == targets).float().mean().item()) > 0.90

    noisy = targets.clone()
    noisy[:8] = (targets[:8] + 1) % 6
    head_bal = nn.Linear(64, 6)
    fit_balanced_ridge_head(head_bal, feats, noisy, ridge=0.5, outlier_gate=True)
    assert float((head_bal(feats[8:]).argmax(dim=1) == targets[8:]).float().mean().item()) > 0.85

    cfg = {**fake_dataset["config"], "name": "accel_opt_run"}
    res = run(cfg, optimizer=lambda params: CurvatureBalancedOptimizer(params, lr=1e-3, weight_decay=1e-4))
    assert len(res["history"]) == 1


# 확인: CompressedFeatureBank의 dyadic 2^k 압축, 0-ULP exact 복원 및 CurvatureBalancedOptimizer 다중 스텝을 검증 (무결성 / 경계)
def test_compressed_feature_bank_and_multi_step_curvature():
    torch.manual_seed(13)
    bank = CompressedFeatureBank(max_entries=16, quant_mode="dyadic")
    feats = torch.randn(16, 256, dtype=torch.float32)
    keys = [f"img_{i}" for i in range(16)]
    bank.put_batch(keys, feats)
    assert bank.memory_bytes() < (feats.numel() * 4) * 0.35
    for sc in bank._scales.tolist():
        assert sc > 0.0 and float(torch.frexp(torch.tensor(sc))[0].item()) == 0.5

    rec = bank.get_batch(list(reversed(keys)))
    assert rec is not None and rec.shape == (16, 256)
    assert (rec - feats.flip(0)).abs().max().item() < 0.08

    feats_ext = feats.clone()
    feats_ext[0, 0], feats_ext[0, 1], feats_ext[1, 3] = 1.2e5, 1.0e-4, -0.0
    ex_bank = CompressedFeatureBank(max_entries=16, quant_mode="exact")
    ex_bank.put_batch(list(range(16)), feats_ext)
    rec_ex = ex_bank.get_tensor_batch(torch.tensor([1, 0, 15], dtype=torch.int32))
    assert rec_ex is not None and torch.equal(rec_ex.view(torch.int32), feats_ext[[1, 0, 15]].view(torch.int32))

    w = torch.nn.Parameter(torch.randn(16, 4))
    opt = CurvatureBalancedOptimizer([w], lr=1e-2, weight_decay=1e-2)
    x_in = torch.randn(8, 16)
    for _ in range(5):
        opt.zero_grad()
        (x_in @ w).pow(2).mean().backward()
        opt.step()
    assert torch.isfinite(w).all().item()


# 확인: 서술자(op_id=7) 변별력과 다중 스레드 동시 호출 무결성 및 컴파일 상수·Softmax·SpectralNorm·DualAnchor를 검증 (동시성 / 정밀도)
def test_descriptors_multithread_and_compiled_ops():
    xs = torch.linspace(0.0, 1.0, 32).view(1, 1, 1, 32).expand(1, 3, 32, 32)
    p_feats = extract_patch_descriptors(torch.cat([(0.4 + 0.1 * xs).clamp(0, 1), (0.5 + 0.35 * torch.sin(xs * 22.0)).clamp(0, 1)], dim=0))
    assert p_feats.shape == (2, 64) and not torch.allclose(p_feats[0], p_feats[1])

    torch.manual_seed(99)
    batch_imgs = torch.rand(4, 3, 224, 224)
    ref_desc = extract_dual_descriptors(batch_imgs, denormalize=False)
    with ThreadPoolExecutor(max_workers=6) as pool:
        outs = list(pool.map(lambda _: extract_dual_descriptors(batch_imgs, denormalize=False), range(12)))
    for out in outs:
        assert out.shape == (4, 128) and torch.isfinite(out).all().item() and torch.allclose(out, ref_desc, atol=1e-6)

    assert abs(calibrated_constant("pi") - float(torch.acos(torch.tensor(-1.0, dtype=torch.float64)).item())) < 1e-12
    assert abs(calibrated_constant("e") - float(torch.exp(torch.tensor(1.0, dtype=torch.float64)).item())) < 1e-12
    phi_v = calibrated_constant("phi")
    assert abs(phi_v * phi_v - phi_v - 1.0) < 1e-12 and abs(cosine_decay_factor(0.5) - 0.5) < 1e-12

    logits = torch.randn(4, 12, dtype=torch.float32, requires_grad=True)
    probs = calibrated_softmax(logits, temperature=1.2)
    assert torch.allclose(probs, torch.softmax(logits / 1.2, dim=-1), atol=1e-5)
    probs.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all().item()
    lp = calibrated_log_softmax(logits.detach().requires_grad_(True))
    assert torch.allclose(lp, torch.log_softmax(logits.detach(), dim=-1), atol=1e-5)

    assert torch.allclose(exact_log_priors(torch.tensor([10, 30, 60], dtype=torch.int64)).exp().sum(), torch.tensor(1.0), atol=1e-5)
    xn = torch.randn(4, 32, requires_grad=True)
    yn = spectral_power_norm(xn, curv_strength=1.0)
    assert torch.allclose(yn.norm(dim=-1), torch.ones(4), atol=1e-5)
    yn.sum().backward()
    assert xn.grad is not None and torch.isfinite(xn.grad).all().item()

    anc = fit_dual_anchor_prototypes(torch.rand(9, 16), torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long), num_classes=3, harm_beta=0.25)
    assert anc.shape == (3, 16) and torch.allclose(anc.norm(dim=-1), torch.ones(3), atol=1e-5)

    xs_loss = torch.tensor([0.2, 0.8, 2.8], dtype=torch.float32, requires_grad=True)
    ys_loss = bounded_loss(xs_loss, tau=0.8, alpha=0.5)
    assert torch.allclose(ys_loss[:2], xs_loss[:2], atol=1e-6) and float(ys_loss[2].item()) == pytest.approx(1.8, abs=1e-5)
    ys_loss.sum().backward()
    assert torch.allclose(xs_loss.grad, torch.tensor([1.0, 1.0, 0.25]), atol=1e-5)

    act_learn = SmoothActivation(leak=0.0625, learnable=True, num_channels=4)
    xl = torch.tensor([[-2.0, -1.0, 0.5, 2.0], [-0.5, 1.0, -3.0, 0.0]], dtype=torch.float32, requires_grad=True)
    yl = act_learn(xl)
    assert yl.shape == xl.shape and torch.isfinite(yl).all().item()
    (yl.pow(2).sum()).backward()
    assert xl.grad is not None and act_learn.raw_leak.grad is not None
    assert torch.isfinite(act_learn.raw_leak.grad).all().item() and act_learn.raw_leak.grad.abs().sum().item() > 0.0
    assert isinstance(build_activation("smooth_learn", leak=0.0625), SmoothActivation)

    gate_mad = AdaptiveBoundedLossGate(mode="mad", tau=0.8, alpha=0.5, param=1.5)
    gate_mad.train()
    hi_losses = torch.tensor([2.5, 3.0, 3.5, 4.0, 5.0], dtype=torch.float32, requires_grad=True)
    capped = gate_mad(hi_losses)
    assert float(gate_mad.ema_tau.item()) > 2.0 and float(capped[-1].item()) < float(hi_losses[-1].item())
    capped.mean().backward()
    assert hi_losses.grad is not None and torch.isfinite(hi_losses.grad).all().item()

    gate_q = AdaptiveBoundedLossGate(mode="quantile", tau=0.8, alpha=0.5, param=0.75)
    gate_q.train()
    q_out = gate_q(torch.tensor([0.1, 0.2, 0.4, 0.8, 2.0], dtype=torch.float32))
    assert float(gate_q.ema_tau.item()) >= 0.4 and torch.isfinite(q_out).all().item()

    xt_p = torch.randn(36, 16)
    yt_p = (xt_p[:, :3]).argmax(dim=1)
    xv_p = torch.randn(12, 16)
    yv_p = (xv_p[:, :3]).argmax(dim=1)
    tuned = auto_tune_resilient(
        xt_p, yt_p, xv_p, yv_p, 3,
        lambda xs, ys, learnable_leak=False, channel_leak=False: ResidualAdapterHead(16, 3, hidden_dim=8, learnable_leak=learnable_leak, channel_leak=channel_leak),
        lambda ys, loss_cap_mode="fixed", loss_cap_tau=0.8, loss_cap_alpha=0.5, loss_cap_param=1.75, class_weight_mode=None, label_smoothing=0.05, device=torch.device("cpu"): nn.CrossEntropyLoss(),
        probe_epochs=2, batch_size=12, refine_bilevel=False,
    )
    assert "loss_cap_mode" in tuned["best"] and len(tuned["trials"]) == 7


# 확인: Zero-Heap C++ op_id=27/28 (FusedLinearHeadSession/multihead_linear_forward) 및 Conv-BN 폴딩 정합성을 검증 (Zero-Heap / Folding)
def test_zero_heap_fused_session_multihead_and_conv_bn_folding(tmp_path):
    torch.manual_seed(7)
    x_sep = torch.randn(30, 16)
    y_sep = x_sep[:, :3].argmax(dim=1)
    lin_ref = nn.Linear(16, 3)
    sess = FusedLinearHeadSession(16, 3, max_batch_size=30)
    sess.sync_from(lin_ref)
    l0, _ = sess.step(x_sep, y_sep, lr=0.1, weight_decay=1e-4, label_smoothing=0.05)
    for _ in range(18):
        l1, _ = sess.step(x_sep, y_sep, lr=0.1, weight_decay=1e-4, label_smoothing=0.05)
    sess.sync_to(lin_ref)
    assert l1 < l0 and (lin_ref(x_sep).argmax(dim=1) == y_sep).float().mean().item() >= 0.85

    heads = [nn.Linear(16, 3) for _ in range(4)]
    w_bank = torch.stack([h.weight.detach() for h in heads], dim=0)
    b_bank = torch.stack([h.bias.detach() for h in heads], dim=0)
    out_mh = multihead_linear_forward(x_sep, w_bank, b_bank)
    out_ref = torch.stack([h(x_sep) for h in heads], dim=0)
    assert out_mh.shape == (4, 30, 3) and torch.allclose(out_mh, out_ref, atol=1e-5)

    x_512 = torch.randn(30, 512)
    res_rc = run_cached(
        {"out_dir": str(tmp_path), "name": "fused_lin_rc", "num_classes": 3, "epochs": 3, "batch_size": 10, "seed": 42},
        optimizer=lambda p: torch.optim.AdamW(p, lr=5e-2, weight_decay=1e-3),
        head_type="linear",
        compress_cache=False,
        precomputed_features=(x_512, y_sep, x_512[:12], y_sep[:12]),
        precomputed_classes=["c0", "c1", "c2"],
    )
    assert Path(res_rc["best_ckpt"]).is_file() and len(res_rc["history"]) == 3

    bb_raw, _ = build_backbone("resnet18", fuse_conv_bn=False)
    bb_raw.eval()
    bb_fused = fuse_backbone_conv_bn(copy.deepcopy(bb_raw))
    img = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(bb_raw(img), bb_fused(img), atol=1e-4)
    assert isinstance(bb_fused.bn1, nn.Identity)

    with pytest.raises(RuntimeError):
        sess.step(x_sep[:4], torch.tensor([0, 1, 9, 0], dtype=torch.long), lr=0.05)

    import fcft
    from fcft.grid import OptimizerFactory
    from fcft.losses import UnifiedClassificationLoss
    from fcft.train import _can_use_fused_linear_session
    for sym in ("FusedLinearHeadSession", "multihead_linear_forward", "fuse_backbone_conv_bn", "co_train_heads_cached"):
        assert hasattr(fcft, sym)
    lin512 = nn.Linear(512, 3)
    crit_fix = UnifiedClassificationLoss(num_classes=3)
    opt_nbd = OptimizerFactory("adamw", lr=1e-2, weight_decay=1e-2, no_bias_decay=True)(lin512.parameters())
    opt_bd = OptimizerFactory("adamw", lr=1e-2, weight_decay=1e-2, no_bias_decay=False)(lin512.parameters())
    opt_eps = torch.optim.AdamW(lin512.parameters(), lr=1e-2, weight_decay=0.0, eps=1e-5)
    assert _can_use_fused_linear_session(lin512, torch.device("cpu"), None, opt_nbd, crit_fix, 0.0, 0.0, None, None) is True
    assert _can_use_fused_linear_session(lin512, torch.device("cpu"), None, opt_bd, crit_fix, 0.0, 0.0, None, None) is False
    assert _can_use_fused_linear_session(lin512, torch.device("cpu"), None, opt_eps, crit_fix, 0.0, 0.0, None, None) is False

    lin_ep = nn.Linear(16, 3)
    sess_ep = FusedLinearHeadSession(16, 3, max_batch_size=10)
    sess_ep.sync_from(lin_ep)
    l_e0, _ = sess_ep.run_epoch(x_sep, y_sep, batch_size=10, lr=0.1, weight_decay=1e-4, label_smoothing=0.05)
    for _ in range(5):
        l_e1, acc_e1 = sess_ep.run_epoch(x_sep, y_sep, batch_size=10, lr=0.1, weight_decay=1e-4, label_smoothing=0.05)
    sess_ep.sync_to(lin_ep)
    assert l_e1 < l_e0 and acc_e1 >= 0.85 and (lin_ep(x_sep).argmax(dim=1) == y_sep).float().mean().item() >= 0.85
