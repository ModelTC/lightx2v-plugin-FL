"""CPU-only contract tests, no vendor runtime required."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "p800_collectives",
    Path(__file__).parents[1] / "lightx2v_fl/device/p800_collectives.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sync_and_restore_on_error():
    events = []

    def original(
        out,
        inp,
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
        async_op=False,
    ):
        events.append("submit")
        return "result"

    dist = SimpleNamespace(all_to_all_single=original)
    device = SimpleNamespace(synchronize=lambda: events.append("sync"))
    with pytest.raises(ValueError, match="caller"):
        with module.synchronized_all_to_all(dist, device, vendor_name="kunlunxin"):
            assert dist.all_to_all_single(None, None) == "result"
            with pytest.raises(RuntimeError, match="async_op"):
                dist.all_to_all_single(None, None, None, None, None, True)
            with pytest.raises(RuntimeError, match="async_op"):
                dist.all_to_all_single(None, None, async_op=True)
            raise ValueError("caller")
    assert events == ["submit", "sync"]
    assert dist.all_to_all_single is original


def test_other_vendor_rejected_before_patching():
    original = lambda: None
    dist = SimpleNamespace(all_to_all_single=original)
    with pytest.raises(RuntimeError, match="kunlunxin"):
        with module.synchronized_all_to_all(dist, None, vendor_name="nvidia"):
            pytest.fail("entered")
    assert dist.all_to_all_single is original


def test_device_binding_precedes_process_group(monkeypatch):
    import sys
    import torch
    import torch.distributed as dist

    class Registry:
        _dict = {}

        def __call__(self, name):
            return lambda cls: cls

    monkeypatch.setitem(
        sys.modules,
        "lightx2v_platform.registry_factory",
        SimpleNamespace(PLATFORM_DEVICE_REGISTER=Registry()),
    )
    device_spec = importlib.util.spec_from_file_location(
        "device_under_test",
        Path(__file__).parents[1] / "lightx2v_fl/device/__init__.py",
    )
    device_module = importlib.util.module_from_spec(device_spec)
    device_spec.loader.exec_module(device_module)
    events = []
    monkeypatch.setenv("RANK", "17")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(device_module, "_detect_torch_device", lambda: "cuda")
    monkeypatch.setattr(device_module, "_select_dist_backend", lambda dev: "nccl")
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda rank: events.append(("device", rank))
    )
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda **kwargs: events.append(("pg", kwargs["backend"])),
    )
    device_module.FlagOSDevice.init_parallel_env()
    assert events == [("device", 1), ("pg", "nccl")]
