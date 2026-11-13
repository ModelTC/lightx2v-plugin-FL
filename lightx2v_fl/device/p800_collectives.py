"""Explicit correctness-first workaround for the validated P800 vendor stack.

This serializes compute/communication and is not a general performance policy.
Only use in an exclusive inference process, before host modules alias collectives.
"""

from contextlib import contextmanager
from functools import wraps
from inspect import signature


@contextmanager
def synchronized_all_to_all(dist, device, *, vendor_name):
    """Restore the owned wrapper on exit; reject async before submitting work."""
    if vendor_name != "kunlunxin":
        raise RuntimeError("P800 collective workaround requires vendor kunlunxin")
    original = dist.all_to_all_single
    contract = signature(original)

    @wraps(original)
    def synchronized(*args, **kwargs):
        bound = contract.bind(*args, **kwargs)
        if bound.arguments.get("async_op", False):
            raise RuntimeError("P800 synchronized baseline does not support async_op")
        result = original(*args, **kwargs)
        device.synchronize()
        return result

    dist.all_to_all_single = synchronized
    try:
        yield
    finally:
        if dist.all_to_all_single is not synchronized:
            raise RuntimeError("P800 collective wrapper ownership changed")
        dist.all_to_all_single = original
