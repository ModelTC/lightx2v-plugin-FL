"""Cross-repository public-API checks; no accelerator or host model claim."""

import importlib
from types import SimpleNamespace

import pytest
import torch

pytest_plugins = ["test_dispatch_callers"]


def test_model_policy_mapping_is_owned_by_plugin():
    from lightx2v_fl.models.h3_operators import H3_OPERATOR_APIS

    assert len(H3_OPERATOR_APIS) == 7
    assert all(name.startswith("h3_") for name in H3_OPERATOR_APIS)
    assert all(not name.startswith("h3_") for name in H3_OPERATOR_APIS.values())


@pytest.mark.parametrize("legacy_alias", [False, True])
def test_h3_routes_to_flagomni_and_preserves_policy(callers, monkeypatch, legacy_alias):
    dispatch, _, _ = callers
    calls = []

    def packed_swiglu(x, *, tier):
        calls.append((x, tier))
        return x

    monkeypatch.setattr(
        dispatch.backends,
        "load_flagomni",
        lambda: SimpleNamespace(packed_swiglu=packed_swiglu),
    )
    x = torch.ones(1, 4)
    config = {"fl_dispatch": {"h3_swiglu": {"parameters": {"tier": "baseline"}}}}
    if legacy_alias:
        config["fl_dispatch"]["h3_swiglu"]["order"] = ["lightx2v_gems"]
    assert dispatch.dispatch_op("h3_swiglu", x, config=config, torch_impl=None) is x
    assert calls == [(x, "baseline")]


def test_actual_flagomni_api_binding_and_cpu_rejection(callers, monkeypatch):
    dispatch, _, _ = callers
    try:
        library = importlib.import_module("flag_omni.lightx2v")
    except ModuleNotFoundError as error:
        if error.name == "flag_omni":
            pytest.skip("Install the platform-matched shared FlagOmni dependency for cross-repo ABI test")
        raise
    from lightx2v_fl.models.h3_operators import H3_OPERATOR_APIS

    assert not any(name.startswith("h3_") for name in library.__all__)
    for public_name in H3_OPERATOR_APIS.values():
        assert callable(getattr(library, public_name))
    monkeypatch.setattr(dispatch.backends, "load_flagomni", lambda: library)
    with pytest.raises(NotImplementedError, match="CUDA or MUSA"):
        dispatch.dispatch_op("h3_swiglu", torch.empty(1, 4), torch_impl=None)
    with pytest.raises(NotImplementedError, match="CUDA or MUSA"):
        dispatch.dispatch_op(
            "sage_attention",
            *(torch.empty(1, 17, 2, 128) for _ in range(3)),
            torch_impl=None,
        )


def test_shipped_h3_example_selects_flagomni():
    import json
    from pathlib import Path

    from lightx2v_fl.dispatch.config import resolve_policy

    config = json.loads(
        (Path(__file__).parents[1] / "examples/minimax_h3_sm90.json").read_text()
    )
    for op in ("sage_attention", "h3_rms_norm", "h3_swiglu"):
        assert resolve_policy(op, config=config)[0] == ("flagomni",)


def test_sage_example_uses_automatic_backend_selection():
    import json
    from pathlib import Path

    config = json.loads(
        (Path(__file__).parents[1] / "examples/minimax_h3_sm90.json").read_text()
    )
    params = config["fl_dispatch"]["sage_attention"]["parameters"]
    assert params == {"pv_dtype": "fp8"}
