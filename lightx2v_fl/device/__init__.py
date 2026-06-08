"""
FlagOS meta-device for LightX2V.

Registers a single ``flagos`` platform that, instead of binding to one fixed
vendor, delegates physical-device detection to FlagGems' ``DeviceDetector`` and
collective-comm init to FlagCX. One backend, many chips.
"""

from __future__ import annotations

import os

import torch
from loguru import logger

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def _detect_torch_device() -> str:
    """Return the torch device string for the underlying FlagOS chip.

    Priority:
      1. FlagGems' own vendor detection (authoritative in the FlagOS stack).
      2. A direct probe of common torch device modules (works even if FlagGems
         is not yet importable, e.g. during a dry capability check).
      3. Fall back to "cuda".
    """
    # 1. Ask FlagGems — it already ran DeviceDetector at import time.
    try:
        import flag_gems

        dev = getattr(flag_gems, "device", None)
        if dev:
            return str(dev)
    except Exception:  # pragma: no cover - FlagGems optional
        pass

    # 2. Probe torch device backends directly.
    for attr, name in (
        ("cuda", "cuda"),   # NVIDIA / AMD ROCm / Iluvatar / MetaX (cuda-compat)
        ("mlu", "mlu"),     # Cambricon
        ("npu", "npu"),     # Ascend
        ("musa", "musa"),   # Moore Threads
        ("xpu", "xpu"),     # Intel / Kunlun
    ):
        mod = getattr(torch, attr, None)
        if mod is not None:
            try:
                if mod.is_available():
                    return name
            except Exception:  # pragma: no cover
                continue

    return "cuda"


@PLATFORM_DEVICE_REGISTER("flagos")
class FlagOSDevice:
    """Meta-device backed by FlagGems (compute) + FlagCX (comm)."""

    name = "flagos"

    @staticmethod
    def get_device() -> str:
        return _detect_torch_device()

    @staticmethod
    def is_available() -> bool:
        # Available iff FlagGems imports AND a concrete torch device is present.
        try:
            import flag_gems  # noqa: F401
        except Exception:
            logger.warning("[lightx2v-fl] flag_gems not importable; flagos unavailable.")
            return False
        dev = _detect_torch_device()
        mod = getattr(torch, dev, None)
        try:
            return mod is not None and mod.is_available()
        except Exception:
            return dev == "cuda"

    @staticmethod
    def init_device_env() -> None:
        dev = _detect_torch_device()
        logger.info(f"[lightx2v-fl] FlagOS platform initialising on torch device '{dev}'.")
        try:
            import flag_gems

            logger.info(
                f"[lightx2v-fl] FlagGems vendor='{getattr(flag_gems, 'vendor_name', '?')}' "
                f"device='{getattr(flag_gems, 'device', dev)}'."
            )
        except Exception:  # pragma: no cover
            logger.warning("[lightx2v-fl] FlagGems not available; ops will use torch fallback.")

    @staticmethod
    def init_parallel_env() -> None:
        """Initialise distributed comm via FlagCX (falls back to vendor CCL)."""
        import torch.distributed as dist

        dev = _detect_torch_device()
        rank = int(os.environ.get("RANK", "0"))

        backend = _select_dist_backend(dev)
        logger.info(f"[lightx2v-fl] init_process_group(backend='{backend}', rank={rank}).")
        dist.init_process_group(backend=backend)

        # Pin this rank to its device.
        mod = getattr(torch, dev, None)
        if mod is not None and hasattr(mod, "set_device"):
            mod.set_device(rank % mod.device_count())


def _select_dist_backend(dev: str) -> str:
    """Choose the torch.distributed backend string.

    FlagCX integrates as a torch ProcessGroup backend named ``flagcx`` and is
    enabled by ``import flagcx``. We use the heterogeneous form
    ``cpu:gloo,<dev>:flagcx`` so CPU-side collectives stay on gloo. If FlagCX is
    not installed we fall back to the vendor-native CCL that torch ships.
    """
    if os.getenv("LIGHTX2V_FL_DISABLE_FLAGCX", "0").lower() in ("1", "true"):
        return _vendor_native_backend(dev)
    try:
        import flagcx  # noqa: F401  (registers the 'flagcx' torch backend)

        return f"cpu:gloo,{dev}:flagcx"
    except Exception:
        logger.warning("[lightx2v-fl] flagcx not importable; using vendor-native CCL.")
        return _vendor_native_backend(dev)


def _vendor_native_backend(dev: str) -> str:
    return {
        "cuda": "nccl",
        "npu": "hccl",
        "mlu": "cncl",
        "musa": "mccl",
        "xpu": "ccl",
    }.get(dev, "nccl")


# Register an alias so PLATFORM=flagos and PLATFORM=fl both work.
PLATFORM_DEVICE_REGISTER._dict.setdefault("fl", FlagOSDevice)
