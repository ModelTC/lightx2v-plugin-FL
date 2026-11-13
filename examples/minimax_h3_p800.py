"""Explicit P800 native BF16 H3 launcher; forwards arguments to minimax_h3.py.

Use with the device-matched vendor image and a validated BF16 config. This is
an opt-in synchronous communication baseline, not a Sage or performance recipe.
"""

import os
import runpy
import sys
from pathlib import Path


def main():
    if "--enable-fusions" in sys.argv[1:]:
        raise RuntimeError("P800 native baseline does not enable optional H3 fusions")
    # Must precede importing torch/vendor libraries. Only this explicit launcher
    # sets these platform-specific defaults; other platforms are unaffected.
    for name, value in {
        "XPU_EVENT_KL3_ENABLE": "0",
        "TRITONXPU_BUDGET_TILING": "0",
        "LIGHTX2V_FL_GLOBAL_GEMS": "0",
        "LIGHTX2V_FL_DISABLE_FLAGCX": "1",
        "PLATFORM": "flagos",
    }.items():
        if name in os.environ and os.environ[name] != value:
            raise RuntimeError(f"P800 baseline requires {name}={value}")
        os.environ[name] = value

    import torch
    import torch.distributed as dist

    # Bind before FlagGems/host imports or process-group creation.
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    import flag_gems

    # Delay plugin registration until the wrapper is installed.
    os.environ["LIGHTX2V_FL_AUTO_REGISTER"] = "0"
    import lightx2v_fl
    from lightx2v_fl.device.p800_collectives import synchronized_all_to_all

    with synchronized_all_to_all(dist, torch.cuda, vendor_name=flag_gems.vendor_name):
        lightx2v_fl.register()
        runpy.run_path(
            str(Path(__file__).with_name("minimax_h3.py")), run_name="__main__"
        )


if __name__ == "__main__":
    main()
