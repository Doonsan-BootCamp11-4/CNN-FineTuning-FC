"""Tests for dataset loading, EXIF/transforms, split inspection CLI, and shared disk bundle cache."""

import csv
from pathlib import Path
from PIL import Image
import pytest
import torch
from torchvision import transforms

from fcft.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    GarbageSplitDataset,
    MultiPoolResNet18,
    assemble_from_bundle,
    build_aug_transform,
    build_eval_transform,
    build_loaders,
    extract_shared_feature_bundle,
    inspect_split,
)
from fcft.evaluate import evaluate, evaluate_ensemble
from fcft.model import build_backbone
from scripts.run_experiments import main as cli_main, run_batch


# 확인: DataLoader 분할·클래스 정렬·흑백 RGB 변환·학습 전용 transform 격리·클래스 수 예외를 검증 (정상 동작 / 경계)
def test_loaders_grayscale_transform_isolation_and_class_guard(fake_dataset):
    loaders = build_loaders(fake_dataset["config"])
    assert set(loaders.keys()) == {"train", "val", "test"}
    assert loaders["train"].dataset.classes == ["cardboard", "glass", "paper"]
    assert len(loaders["train"].dataset) == 6 and len(loaders["val"].dataset) == 3 and len(loaders["test"].dataset) == 3
    x, y = next(iter(loaders["train"]))
    assert x.shape == (2, 3, 224, 224) and y.shape == (2,)

    train_ds = loaders["train"].dataset
    gray_idx = next(i for i, (rel, _) in enumerate(train_ds.samples) if rel == fake_dataset["grayscale_rel"])
    assert train_ds[gray_idx][0].shape == (3, 224, 224)

    sentinel_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Lambda(lambda t: torch.full_like(t, 7.0))])
    loaders_tf = build_loaders(fake_dataset["config"], train_transform=sentinel_tf)
    xt, _ = next(iter(loaders_tf["train"]))
    xv, _ = next(iter(loaders_tf["val"]))
    _ = IMAGENET_MEAN, IMAGENET_STD
    assert torch.allclose(xt, torch.full_like(xt, 7.0)) and xv.abs().max().item() < 5.0

    with pytest.raises(ValueError):
        build_loaders({**fake_dataset["config"], "num_classes": 12})


# 확인: EXIF 회전 자동 보정, center256 리사이즈, aug_mode 프리셋 및 inspect_split/--check-split CLI를 검증 (전처리 / CLI)
def test_exif_preproc_presets_and_check_split_cli(tmp_path, fake_dataset):
    img_p = tmp_path / "rotated.jpg"
    im = Image.new("RGB", (40, 20), (200, 30, 30))
    exif = im.getexif()
    exif[0x0112] = 6
    im.save(img_p, format="JPEG", exif=exif.tobytes())
    assert GarbageSplitDataset([("rotated.jpg", 0)], tmp_path, ["cardboard"], transform=transforms.ToTensor())[0][0].shape == (3, 40, 20)

    ev_c256 = build_eval_transform("center256")
    assert isinstance(ev_c256.transforms[0], transforms.Resize) and isinstance(ev_c256.transforms[1], transforms.CenterCrop)
    assert next(iter(build_loaders({**fake_dataset["config"], "eval_resize_mode": "center256"})["val"]))[0].shape[1:] == (3, 224, 224)
    assert build_aug_transform("none") is None
    for m in ("rrc_jitter", "resize256_crop224", "trivial", "randaugment", "erasing", "affine"):
        assert build_aug_transform(m)(Image.new("RGB", (64, 48), (10, 20, 30))).shape == (3, 224, 224)
    with pytest.raises(ValueError):
        build_eval_transform("invalid_resize")
    with pytest.raises(ValueError):
        build_aug_transform("unknown_aug")

    rep = inspect_split(fake_dataset["csv_path"], data_root=fake_dataset["data_root"], check_files=True, expected_classes=3)
    assert rep["ok"] is True and rep["split_counts"] == {"train": 6, "val": 3, "test": 3}
    assert cli_main(["--check-split", "--split-csv", str(fake_dataset["csv_path"]), "--data-root", str(fake_dataset["data_root"]), "--expected-classes", "3", "--check-files"]) == 0

    broken_csv = tmp_path / "missing_file.csv"
    with broken_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        w.writeheader()
        w.writerow({"path": "glass/does_not_exist.png", "label": "glass", "split": "train"})
    assert inspect_split(broken_csv, data_root=fake_dataset["data_root"], check_files=True)["ok"] is False


# 확인: MultiPoolResNet18 단일 패스 정확성, target_pools/channels_last, lazy test 캐시, 디스크 캐시 격리를 검증 (수치·캐시 정합성)
def test_multipool_bundle_cache_isolation_and_eval_parity(fake_dataset):
    torch.manual_seed(0)
    x = torch.randn(3, 3, 224, 224)
    bb_avg, _ = build_backbone("resnet18", "avg")
    mp = MultiPoolResNet18(bb_avg)
    with torch.no_grad():
        bundled = mp(x)
        for mode in ("avg", "avgmax", "gem", "multiscale"):
            ref = build_backbone("resnet18", mode)[0](x)
            assert torch.allclose(bundled[mode], ref, atol=1e-5, rtol=1e-5)

    mp_nms = MultiPoolResNet18(bb_avg, target_pools=["norm_multiscale"]).to(memory_format=torch.channels_last)
    out_nms = mp_nms(x[:2].contiguous(memory_format=torch.channels_last))
    assert set(out_nms.keys()) == {"norm_multiscale"} and torch.allclose(out_nms["norm_multiscale"], bundled["norm_multiscale"][:2], atol=1e-5, rtol=1e-5)

    base_cfg = fake_dataset["config"]
    iso_dir = Path(base_cfg["out_dir"]) / "iso_cache"
    b1 = extract_shared_feature_bundle(base_cfg, aug_repeats=1, include_flip=False, include_descriptors=False, use_disk_cache=True, cache_dir=iso_dir, aug_mode="rrc_jitter", eval_resize_mode="squash")
    b2 = extract_shared_feature_bundle(base_cfg, aug_repeats=1, include_flip=False, include_descriptors=False, use_disk_cache=True, cache_dir=iso_dir, aug_mode="trivial", eval_resize_mode="center256")
    assert len(list(iso_dir.glob("bundle_*.pt"))) == 2 and not torch.allclose(b1["train_aug"]["avg"], b2["train_aug"]["avg"])

    lazy_dir = Path(base_cfg["out_dir"]) / "lazy_cache"
    bl1 = extract_shared_feature_bundle(base_cfg, aug_repeats=1, include_flip=True, include_descriptors=False, include_test=False, use_disk_cache=True, cache_dir=lazy_dir)
    bl2 = extract_shared_feature_bundle(base_cfg, aug_repeats=1, include_flip=True, include_descriptors=False, include_test=True, use_disk_cache=True, cache_dir=lazy_dir)
    assert "test" not in bl1 and "test" in bl2 and len(list(lazy_dir.glob("bundle_*.pt"))) == 1


# 확인: fast_cpu 공유 캐시 E2E와 precomputed_test 평가·앙상블이 이미지 직접 평가와 오차 없이 일치하는지 검증 (정합 / E2E)
def test_fast_cpu_shared_cache_and_precomputed_eval_parity(fake_dataset):
    base_cfg = fake_dataset["config"]
    out_dir = Path(base_cfg["out_dir"]) / "cache_parity"
    cache_dir = Path(base_cfg["out_dir"]) / "disk_cache"
    results = run_batch(
        axis="compare", split_csv=base_cfg["split_csv"], data_root=base_cfg["data_root"],
        out_dir=str(out_dir), num_classes=base_cfg["num_classes"], epochs=1, batch_size=2,
        num_workers=0, limit=3, shared_cache=True, fast_cpu=True, use_disk_cache=True, cache_dir=str(cache_dir),
    )
    assert len(results) == 3 and list(cache_dir.glob("bundle_*.pt"))
    bundle = extract_shared_feature_bundle(base_cfg, include_flip=True, include_descriptors=False, include_test=True, use_disk_cache=True, cache_dir=cache_dir)
    c0 = {**base_cfg, "out_dir": str(out_dir), "name": "compare_0_baseline"}
    c2 = {**base_cfg, "out_dir": str(out_dir), "name": "compare_2_norm_cosinehead"}
    for tta_m in ("none", "flip2", "multiview4"):
        raw_m, cache_m = evaluate(c2, tta_mode=tta_m), evaluate(c2, tta_mode=tta_m, precomputed_test=bundle)
        assert raw_m["confusion_matrix"] == cache_m["confusion_matrix"]
    ens_raw = evaluate_ensemble([c0, c2], out_name="ens_raw", out_dir=str(out_dir), tta_mode="multiview4")
    ens_cache = evaluate_ensemble([c0, c2], out_name="ens_cache", out_dir=str(out_dir), tta_mode="multiview4", precomputed_test=bundle)
    assert ens_raw["confusion_matrix"] == ens_cache["confusion_matrix"]

    b_aug = extract_shared_feature_bundle(base_cfg, aug_repeats=2, include_flip=True, include_descriptors=False, use_disk_cache=True, cache_dir=cache_dir)
    x_tr_aug, y_tr_aug, x_va_aug, _ = assemble_from_bundle(b_aug, None, pool_mode="norm_multiscale", tta_extract=True, use_aug_bank=True)
    assert x_tr_aug.shape[0] == b_aug["y_train"].shape[0] * 3 and y_tr_aug.shape[0] == x_tr_aug.shape[0] and x_va_aug.shape[1] == 768


# 확인: OS/경로 무관 캐시 키 공유, 읽기전용 마운트(bundle_path), CLI --export-cache -> --cache-file E2E를 검증 (포터블 캐시)
def test_portable_bundle_export_and_readonly_mount(fake_dataset, tmp_path):
    import os
    import shutil
    from fcft.data import _bundle_cache_path

    base_cfg = fake_dataset["config"]
    alt_root = tmp_path / "alt_images"
    shutil.copytree(base_cfg["data_root"], alt_root)
    p1 = _bundle_cache_path(base_cfg, 1, True, False, tmp_path / "c1", target_pools=["norm_multiscale"])
    p2 = _bundle_cache_path({**base_cfg, "data_root": str(alt_root)}, 1, True, False, tmp_path / "c1", target_pools=["norm_multiscale"])
    assert p1 == p2

    ro_dir = tmp_path / "ro_mount"
    ro_dir.mkdir(parents=True, exist_ok=True)
    exported = ro_dir / "fc_grid_portable.pt"
    rc_exp = cli_main([
        "--axis", "grid", "--export-cache", str(exported),
        "--split-csv", base_cfg["split_csv"], "--data-root", base_cfg["data_root"],
        "--out-dir", str(tmp_path / "exp_out"), "--num-classes", "3", "--batch-size", "2", "--num-workers", "0",
    ])
    assert rc_exp == 0 and exported.is_file()

    # 읽기 전용 마운트(Kaggle /kaggle/input 시뮬레이션): 쓰기 시도 없이 정상 로드 + 단일/그리드 실행 통과
    os.chmod(ro_dir, 0o555)
    try:
        loaded = extract_shared_feature_bundle(
            {**base_cfg, "data_root": str(alt_root)},
            aug_repeats=3, include_flip=True, include_descriptors=False, include_test=True,
            target_pools=["norm_multiscale"], bundle_path=exported,
        )
        assert "test" in loaded and "norm_multiscale" in loaded["train"]

        out_mnt = tmp_path / "mounted_single"
        rc_single = cli_main([
            "--single", "--name", "mounted_res",
            "--split-csv", base_cfg["split_csv"], "--data-root", str(alt_root), "--out-dir", str(out_mnt),
            "--num-classes", "3", "--epochs", "1", "--batch-size", "2", "--num-workers", "0",
            "--pool", "norm_multiscale", "--head", "residual", "--mix", "cutmix", "--tta", "flip2",
            "--set", "swa_epochs=0", "--cache-file", str(exported),
        ])
        assert rc_single == 0 and (out_mnt / "mounted_res" / "metrics.json").is_file()
    finally:
        os.chmod(ro_dir, 0o755)
