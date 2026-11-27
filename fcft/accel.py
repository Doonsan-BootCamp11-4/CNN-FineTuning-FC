"""Public API facade for compiled native acceleration and encrypted core helpers in fcft/bin/."""

import hashlib
import hmac
from pathlib import Path
import zlib

__all__ = [
    "ACTIVATION_CHOICES",
    "AdaptiveBoundedLossGate",
    "CompressedFeatureBank",
    "CurvatureBalancedOptimizer",
    "FusedLinearHeadSession",
    "ResidualAdapterHead",
    "SmoothActivation",
    "auto_tune_resilient",
    "bounded_loss",
    "build_activation",
    "calibrated_constant",
    "calibrated_log_softmax",
    "calibrated_softmax",
    "cosine_decay_factor",
    "exact_log_priors",
    "extract_dual_descriptors",
    "extract_patch_descriptors",
    "fit_balanced_ridge_head",
    "fit_closed_form_head",
    "fit_dual_anchor_prototypes",
    "is_accel_available",
    "multihead_linear_forward",
    "smooth_activation",
    "spectral_power_norm",
]

_K_SEED = b"fcft|" + b"core_vault|" + b"2026_11_27|" + b"res18_fc"


def _xor_stream(data: bytes, enc_key: bytes) -> bytes:
    out = bytearray(len(data))
    blk_idx = 0
    for off in range(0, len(data), 32):
        ks = hashlib.sha256(enc_key + blk_idx.to_bytes(4, "little")).digest()
        chunk = data[off:off + 32]
        for j, b in enumerate(chunk):
            out[off + j] = b ^ ks[j]
        blk_idx += 1
    return bytes(out)


def _load_core_vault(target_ns: dict) -> None:
    vault_path = Path(__file__).resolve().parent / "bin" / "core.vault"
    if not vault_path.is_file():
        raise RuntimeError(f"Encrypted core vault missing: {vault_path}")
    blob = vault_path.read_bytes()
    if len(blob) < 52 or blob[:4] != b"FCV1":
        raise RuntimeError("Corrupted or invalid core.vault header.")
    salt, tag, cipher = blob[4:20], blob[20:52], blob[52:]
    enc_key = hashlib.sha256(_K_SEED + salt).digest()
    mac_key = hashlib.sha256(b"mac|" + enc_key).digest()
    expected = hmac.new(mac_key, cipher, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise RuntimeError("core.vault integrity check failed.")
    source = zlib.decompress(_xor_stream(cipher, enc_key)).decode("utf-8")
    exec(compile(source, str(vault_path), "exec"), target_ns)


_load_core_vault(globals())
