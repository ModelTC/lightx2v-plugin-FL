"""Smoke tests that run without any real accelerator.

They verify the plugin's *wiring* — registration, registry keys, torch fallback
math — not kernel performance. Run with: ``pytest -q`` (CPU is fine).
"""

import torch


def test_device_registered():
    import lightx2v_fl  # noqa: F401  (auto-registers)
    from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

    assert "flagos" in PLATFORM_DEVICE_REGISTER
    assert "fl" in PLATFORM_DEVICE_REGISTER


def test_op_keys_registered():
    import lightx2v_fl  # noqa: F401
    from lightx2v.utils.registry_factory import (
        ATTN_WEIGHT_REGISTER,
        LN_WEIGHT_REGISTER,
        MM_WEIGHT_REGISTER,
        RMS_WEIGHT_REGISTER,
        ROPE_REGISTER,
    )

    assert "flagos_flash_attn" in ATTN_WEIGHT_REGISTER
    assert "flagos" in MM_WEIGHT_REGISTER
    assert "flagos-fp8" in MM_WEIGHT_REGISTER
    assert "flagos-int8" in MM_WEIGHT_REGISTER
    assert "flagos_rms_norm" in RMS_WEIGHT_REGISTER
    assert "flagos_layer_norm" in LN_WEIGHT_REGISTER
    assert "flagos_rope" in ROPE_REGISTER


def test_attn_fallback_matches_sdpa():
    import lightx2v_fl  # noqa: F401
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

    torch.manual_seed(0)
    s, h, d = 16, 4, 32
    q, k, v = (torch.randn(s, h, d) for _ in range(3))
    attn = ATTN_WEIGHT_REGISTER["flagos_flash_attn"]()
    out = attn.apply(q, k, v)
    assert out.shape == (s, h * d)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
    ).permute(1, 0, 2).reshape(s, h * d)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


def test_rms_norm_fallback():
    import lightx2v_fl  # noqa: F401
    from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER

    x = torch.randn(8, 64)
    rms = RMS_WEIGHT_REGISTER["flagos_rms_norm"]("w", eps=1e-6)
    rms.weight = torch.ones(64)
    out = rms.apply(x)
    assert out.shape == x.shape
