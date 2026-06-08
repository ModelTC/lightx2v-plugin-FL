"""Optional global aten-level patching with FlagGems.

LightX2V routes its heavy linear/attn/norm ops through weight templates (which
we override in ``lightx2v_fl.ops``). But a model still issues many *generic*
torch ops — softmax, elementwise, reductions — directly. ``flag_gems.enable()``
lowers those to FlagGems Triton kernels at the aten dispatch level, giving
cross-chip coverage for the long tail too.

This is opt-in (LIGHTX2V_FL_GLOBAL_GEMS=1) because it changes global torch
behaviour for the whole process. The template path is the primary, always-on
mechanism; this stacks on top.
"""

from __future__ import annotations

from loguru import logger

_ENABLED = False


def enable_global_flaggems() -> None:
    global _ENABLED
    if _ENABLED:
        return
    try:
        import flag_gems
    except Exception as exc:
        logger.warning(f"[lightx2v-fl] cannot enable global FlagGems: {exc}")
        return

    # flag_gems.enable() registers all (or selected) ops into the aten library.
    # An optional comma-separated unused list lets callers exclude ops that
    # regress for their model/chip.
    import os

    unused_env = os.getenv("LIGHTX2V_FL_GEMS_UNUSED", "").strip()
    unused = [s for s in unused_env.split(",") if s] or None
    flag_gems.enable(unused=unused)
    _ENABLED = True
    logger.info("[lightx2v-fl] global FlagGems aten patching enabled.")
