"""Pre-execution adapters to public FlagGems and FlagOmni APIs; no numerical kernels."""

from __future__ import annotations

import inspect
from functools import lru_cache

from loguru import logger


@lru_cache(maxsize=1)
def load_flaggems():
    try:
        import flag_gems
    except Exception as exc:  # noqa: BLE001 - optional vendor import, before compute
        logger.debug(f"[lightx2v-fl] FlagGems unavailable for explicit dispatch: {exc}")
        return None
    return flag_gems


def bind_call(fn, args, kwargs):
    """Return a zero-argument call only if its signature accepts the contract."""
    if not callable(fn):
        return None
    try:
        try:
            signature = _signature(fn)
        except TypeError:
            signature = inspect.signature(fn)
        signature.bind(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    return lambda: fn(*args, **kwargs)


@lru_cache(maxsize=128)
def _signature(fn):
    return inspect.signature(fn)


@lru_cache(maxsize=1)
def load_flagomni():
    try:
        from flag_omni import lightx2v
    except ModuleNotFoundError as exc:
        # Missing optional package permits pre-execution selection only. A
        # broken installed dependency must not silently select another backend.
        if exc.name not in ("flag_omni", "flag_omni.lightx2v"):
            raise
        return None
    return lightx2v


def flagomni_call(op_name, args, kwargs):
    from lightx2v_fl.models.h3_operators import H3_OPERATOR_APIS

    public_name = (
        "sage_attention"
        if op_name == "sage_attention"
        else H3_OPERATOR_APIS.get(op_name)
    )
    if public_name is None:
        return None
    library = load_flagomni()
    if library is None:
        return None
    return bind_call(getattr(library, public_name, None), args, kwargs)


def flaggems_call(gems, op_name, args, kwargs):
    if gems is None:
        return None
    if op_name == "sage_attention":
        return bind_call(getattr(gems, "sage_attention", None), args, kwargs)
    if op_name.startswith("h3_"):
        return None
    if op_name == "rms_norm":
        x, weight = args
        fn = getattr(gems, "rms_norm", None)
        # Recognize both public signatures without executing either as a probe.
        return bind_call(fn, (x, (x.shape[-1],), weight), kwargs) or bind_call(
            fn, (x, weight), kwargs
        )
    if op_name == "layer_norm":
        x, weight, bias = args
        if weight is None:
            return None  # Matches the existing non-affine torch path.
        return bind_call(
            getattr(gems, "layer_norm", None),
            (x, (x.shape[-1],), weight, bias),
            kwargs,
        )
    # Only dense SDPA-compatible APIs accept this BHSD -> BHSD contract.
    # Varlen APIs require metadata; flash_attention_forward can return tuples.
    attention = getattr(getattr(gems, "ops", None), "attention", None)
    functions = (
        attention,
        getattr(attention, "scaled_dot_product_attention", None),
        getattr(gems, "scaled_dot_product_attention", None),
    )
    for fn in functions:
        call = bind_call(fn, args, kwargs)
        if call is not None:
            return call
    return None


def flash_attn_call(op_name, args, kwargs):
    """Explicit vendor FlashAttention adapter, BHSD in/out; never a Sage alias.

    The vendor image owns the flash_attn package. Missing packages permit
    selection fallback; installed-package errors and execution errors propagate.
    """
    if op_name != "attention" or len(args) != 3:
        return None
    try:
        from flash_attn import flash_attn_func
    except ModuleNotFoundError as exc:
        if exc.name == "flash_attn":
            return None
        raise
    q, k, v = args
    if any(x.ndim != 4 for x in args):
        return None
    # FlashAttention uses bottom-right causal alignment; SDPA uses top-left.
    if kwargs.get("is_causal", False) and q.shape[-2] != k.shape[-2]:
        return None
    parameters = {
        "dropout_p": kwargs.get("dropout_p", 0.0),
        "softmax_scale": kwargs.get("scale"),
        "causal": kwargs.get("is_causal", False),
        "return_attn_probs": False,
    }

    # Layout adaptation is performed only after this backend is selected.
    def run():
        result = flash_attn_func(
            *(x.transpose(1, 2).contiguous() for x in (q, k, v)), **parameters
        )
        return result.transpose(1, 2)

    return run
