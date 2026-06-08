"""
lightx2v-plugin-fl
==================

A pluggable LightX2V backend built on the FlagOS unified multi-chip stack:

  * **FlagGems** — a Triton operator library that auto-detects the underlying
    vendor (NVIDIA / Ascend / Cambricon / MetaX / MUSA / Kunlun / Iluvatar / ...)
    and runs one set of kernels across all of them.
  * **FlagCX**   — a cross-chip collective communication library exposed as a
    PyTorch ``ProcessGroup`` backend.

Unlike the per-chip directories under ``lightx2v_platform/ops/<chip>/`` — each of
which hand-writes a kernel set for one vendor — ``flagos`` is a *meta-platform*:
a single ``PLATFORM=flagos`` backend that covers every chip FlagOS supports.

Activation
----------
Preferred (once the upstream entry-point hook lands — see
``docs/upstream-entrypoint-hook.md``)::

    pip install lightx2v-plugin-fl
    PLATFORM=flagos python lightx2v/infer.py ...

Zero-upstream-change fallback — import this package *before* ``lightx2v`` so its
registrations land before ``lightx2v.utils.registry_factory`` snapshots the
platform registries::

    import lightx2v_fl              # noqa: F401  (calls register())
    import lightx2v

``register()`` is idempotent and is the single entry point that wires every
device / op / communicator into LightX2V's registries.
"""

from __future__ import annotations

import os

from loguru import logger

_REGISTERED = False


def register() -> None:
    """Wire the FlagOS backend into LightX2V's platform registries.

    Idempotent: safe to call from the entry point, from an explicit
    ``import lightx2v_fl``, and from tests.

    Critical timing note
    --------------------
    ``lightx2v/utils/registry_factory.py`` does, at import time::

        ATTN_WEIGHT_REGISTER.merge(PLATFORM_ATTN_WEIGHT_REGISTER)
        MM_WEIGHT_REGISTER.merge(PLATFORM_MM_WEIGHT_REGISTER)
        ...

    ``merge`` is a one-shot *snapshot copy*. Anything registered into the
    ``PLATFORM_*`` tables *after* that import is invisible to the framework.

    To be robust regardless of import order, our op modules register into the
    *final* ``lightx2v.utils.registry_factory`` tables directly (not the
    ``PLATFORM_*`` staging tables). The device, which is consumed via the
    staging table before merge, is registered eagerly here.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # 1. Device must be registered into PLATFORM_DEVICE_REGISTER before
    #    set_ai_device() looks it up.
    from . import device  # noqa: F401  (registration side effect)

    # 2. Operators register into the *final* lightx2v registries (see docstring).
    from .ops import register_ops

    register_ops()

    # 3. Optional: globally patch generic torch aten ops with FlagGems Triton
    #    kernels (softmax, elementwise, ...) that don't go through LightX2V's
    #    weight-template path. Off by default; opt in with LIGHTX2V_FL_GLOBAL_GEMS=1.
    if os.getenv("LIGHTX2V_FL_GLOBAL_GEMS", "0").lower() in ("1", "true"):
        from .enable import enable_global_flaggems

        enable_global_flaggems()

    _REGISTERED = True
    logger.info("[lightx2v-fl] FlagOS backend registered (PLATFORM=flagos).")


# Importing the package is enough to activate it in the fallback path.
# Guarded so a failed optional import never breaks `import lightx2v`.
if os.getenv("LIGHTX2V_FL_AUTO_REGISTER", "1").lower() in ("1", "true"):
    try:
        register()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[lightx2v-fl] auto-register skipped: {exc}")
