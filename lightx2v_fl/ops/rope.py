"""Rotary position embedding for FlagOS.

LightX2V passes ``cos_sin_cache`` as a complex tensor (real=cos, imag=sin),
interleaved layout — matching the convention in the iluvatar/enflame rope impls.
FlagGems exposes ``apply_rotary_pos_emb`` / ``rotary_embedding``; we probe for it
and fall back to a torch implementation otherwise.
"""

from __future__ import annotations

import torch

from lightx2v.utils.registry_factory import ROPE_REGISTER
from lightx2v_platform.ops.rope.rope_template import RopeTemplate

try:
    import flag_gems

    _GEMS_ROPE = getattr(flag_gems, "apply_rotary_pos_emb", None) or getattr(
        flag_gems, "rotary_embedding", None
    )
except Exception:  # pragma: no cover
    flag_gems = None
    _GEMS_ROPE = None


def _apply_rotary_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved rotary application in pure torch.

    x: [..., D] with D even; cos/sin: [..., D/2] broadcastable to x's leading dims.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot_even = x1 * cos - x2 * sin
    rot_odd = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


@ROPE_REGISTER("flagos_rope")
class FlagOSRope(RopeTemplate):
    def apply(self, xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache: torch.Tensor):
        if torch.is_complex(cos_sin_cache):
            cos = cos_sin_cache.real.contiguous()
            sin = cos_sin_cache.imag.contiguous()
        else:
            half = cos_sin_cache.shape[-1] // 2
            cos = cos_sin_cache[..., :half].contiguous()
            sin = cos_sin_cache[..., half:].contiguous()

        if _GEMS_ROPE is not None:
            try:
                xq_o, xk_o = _GEMS_ROPE(xq, xk, cos, sin)
                return xq_o.to(self.infer_dtype), xk_o.to(self.infer_dtype)
            except Exception:
                pass

        xq_o = _apply_rotary_ref(xq, cos, sin)
        xk_o = _apply_rotary_ref(xk, cos, sin)
        return xq_o.to(self.infer_dtype), xk_o.to(self.infer_dtype)
