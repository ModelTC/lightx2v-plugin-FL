"""Flash-attention op backed by FlagGems, with a torch SDPA fallback.

Layout follows the WAN varlen convention used across lightx2v_platform:
    q / k / v : [S, H, D]  (or [B, S, H, D])  where S = total tokens
    output    : [S, H*D]
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.ops.attn.template import AttnWeightTemplate

from lightx2v_fl.dispatch import dispatch_op


def _sdp(
    q4d: torch.Tensor, k4d: torch.Tensor, v4d: torch.Tensor, config=None
) -> torch.Tensor:
    """Dense attention on [B, L, H, D] inputs -> [B, L, H, D]."""
    q_t = q4d.permute(0, 2, 1, 3).contiguous()
    k_t = k4d.permute(0, 2, 1, 3).contiguous()
    v_t = v4d.permute(0, 2, 1, 3).contiguous()
    out = dispatch_op(
        "attention",
        q_t,
        k_t,
        v_t,
        torch_impl=_torch_sdp,
        config=config,
    )
    return out.permute(0, 2, 1, 3)


def _torch_sdp(q, k, v, **parameters):
    return F.scaled_dot_product_attention(q, k, v, **parameters)


@ATTN_WEIGHT_REGISTER("flagos_flash_attn")
class FlagOSFlashAttnWeight(AttnWeightTemplate):
    """FlagOS attention. Registered as ``flagos_flash_attn``."""

    _fl_dispatch_operator = "attention"

    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError("Q/K/V must share a 3D packed or 4D batched layout")
        if k.shape[:-1] != v.shape[:-1] or q.shape[-1] != k.shape[-1]:
            raise ValueError("Incompatible Q/K/V shapes")
        if q.shape[-2] != k.shape[-2]:
            raise ValueError("Q/K/V must have matching head counts")
        width = q.shape[-2] * v.shape[-1]
        if q.ndim == 4 and (q.shape[0] != k.shape[0]):
            raise ValueError("Q/K/V must have matching batch sizes")
        if cu_seqlens_q is None:
            if cu_seqlens_kv is not None:
                raise ValueError("cu_seqlens_kv requires cu_seqlens_q")
            # Preserve the batch dimension, including unequal Q and KV lengths.
            inputs = (
                (q, k, v) if q.ndim == 4 else tuple(x.unsqueeze(0) for x in (q, k, v))
            )
            return _sdp(*inputs, self.config).reshape(-1, width)

        batch_shape = (q.shape[:2], k.shape[:2]) if q.ndim == 4 else None
        q, k, v = (x.reshape(-1, x.shape[-2], x.shape[-1]) for x in (q, k, v))
        ckv = cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens_q

        def boundaries(prefix, total, maximum, name):
            if (
                prefix.ndim != 1
                or prefix.numel() < 2
                or prefix.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    f"{name} must be a 1D integer cumulative-length tensor"
                )
            # This eager varlen fallback already needs host slicing. Copy once,
            # rather than synchronizing for every boundary. Not graph-safe.
            points = prefix.tolist()
            lengths = [b - a for a, b in itertools.pairwise(points)]
            if points[0] != 0 or points[-1] != total or any(n < 0 for n in lengths):
                raise ValueError(
                    f"{name} must monotonically cover all tokens from zero"
                )
            if maximum is not None and max(lengths) > maximum:
                raise ValueError(f"{name} exceeds max_seqlen")
            return points

        cq = boundaries(cu_seqlens_q, q.shape[0], max_seqlen_q, "cu_seqlens_q")
        ck = boundaries(ckv, k.shape[0], max_seqlen_kv, "cu_seqlens_kv")
        if len(cq) != len(ck):
            raise ValueError("Q and KV must describe the same number of sequences")
        if batch_shape is not None:
            # A dense batch must not be repartitioned across physical rows.
            for points, (batch, length) in zip((cq, ck), batch_shape):
                if points != [i * length for i in range(batch + 1)]:
                    raise ValueError(
                        "4D cumulative lengths must match dense batch boundaries"
                    )
            return _sdp(
                q.reshape(*batch_shape[0], *q.shape[-2:]),
                k.reshape(*batch_shape[1], *k.shape[-2:]),
                v.reshape(*batch_shape[1], *v.shape[-2:]),
                self.config,
            ).reshape(-1, width)

        outputs = []
        for qs, qe, ks, ke in zip(cq, cq[1:], ck, ck[1:]):
            if qs == qe:
                continue
            if ks == ke:
                # SDPA's empty-key result is zero; avoid backend-dependent kernels.
                outputs.append(v.new_zeros((qe - qs, width)))
                continue
            xi = _sdp(
                q[qs:qe].unsqueeze(0),
                k[ks:ke].unsqueeze(0),
                v[ks:ke].unsqueeze(0),
                self.config,
            )
            outputs.append(xi.reshape(qe - qs, width))
        return torch.cat(outputs, dim=0) if outputs else v.new_empty((0, width))
