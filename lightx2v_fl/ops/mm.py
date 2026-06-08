"""Matrix-multiply ops for FlagOS.

Three registry keys:
    "flagos"      — bf16/fp16 linear (FlagGems addmm/mm, torch fallback)
    "flagos-fp8"  — fp8 per-channel symmetric weight, dequant→compute
    "flagos-int8" — int8 per-channel symmetric weight, dequant→compute

The quant variants reuse MMWeightQuantTemplate's loaders verbatim
(load_fp8_perchannel_sym / load_int8_perchannel_sym), so weight files produced
for the other platforms load unchanged. Only ``apply()`` differs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.ops.mm.template import MMWeightQuantTemplate, MMWeightTemplate

try:
    import flag_gems  # noqa: F401

    _HAS_GEMS = True
except Exception:  # pragma: no cover
    _HAS_GEMS = False


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """F.linear, dispatched through FlagGems when its global patch is active.

    When LIGHTX2V_FL_GLOBAL_GEMS=1 has been set, torch.nn.functional.linear is
    already lowered to FlagGems' addmm/mm at the aten level, so we just call it.
    Otherwise this is plain torch — correct on every backend.
    """
    return F.linear(x, weight, bias)


@MM_WEIGHT_REGISTER("flagos")
class FlagOSMmWeight(MMWeightTemplate):
    """Unquantised bf16/fp16 linear."""

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, create_cpu_buffer=False,
                 lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer,
                         lazy_load, lazy_load_file, is_post_adapter)

    def load(self, weight_dict):
        if self.create_cuda_buffer:
            self._load_cuda_buffer(weight_dict)
        elif self.create_cpu_buffer:
            self._load_cpu_pin_buffer()
        else:
            self._load_default_tensors(weight_dict)

    def _load_default_tensors(self, weight_dict):
        if self.lazy_load:
            self.weight, self.bias = None, None
            return
        device = weight_dict[self.weight_name].device
        if device.type == "cpu":
            self.pin_weight = self._pin(weight_dict[self.weight_name])
            self.pin_bias = (
                self._pin(weight_dict[self.bias_name])
                if self.bias_name is not None and self.bias_name in weight_dict
                else None
            )
            del weight_dict[self.weight_name]
        else:
            self.weight = weight_dict[self.weight_name]
            self.bias = weight_dict[self.bias_name] if self.bias_name and self.bias_name in weight_dict else None

    def _load_cuda_buffer(self, weight_dict):
        self.weight_cuda_buffer = weight_dict[self.weight_name].to(AI_DEVICE)
        if self.bias_name is not None and self.bias_name in weight_dict:
            self.bias_cuda_buffer = weight_dict[self.bias_name].to(AI_DEVICE)

    def _load_cpu_pin_buffer(self):
        from safetensors import safe_open

        with safe_open(self.lazy_load_file, framework="pt", device="cpu") as f:
            self.pin_weight = self._pin(f.get_tensor(self.weight_name))

    @staticmethod
    def _pin(tensor):
        if tensor is None:
            return None
        pin = torch.empty(tensor.shape, pin_memory=True, dtype=tensor.dtype)
        pin.copy_(tensor)
        return pin

    def apply(self, input_tensor):
        if hasattr(self, "weight_cuda_buffer"):
            weight = self.weight_cuda_buffer
            bias = getattr(self, "bias_cuda_buffer", None)
        elif hasattr(self, "weight") and self.weight is not None:
            weight, bias = self.weight, self.bias
        else:
            weight = self.pin_weight.to(AI_DEVICE)
            bias = self.pin_bias.to(AI_DEVICE) if getattr(self, "pin_bias", None) is not None else None
        return _linear(input_tensor, weight, bias)


class _FlagOSQuantMm(MMWeightQuantTemplate):
    """Shared dequant→linear path for fp8 / int8. Storage is quantised (memory
    saving); compute is done in ``infer_dtype`` after per-channel dequant."""

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, create_cpu_buffer=False,
                 lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer,
                         lazy_load, lazy_load_file, is_post_adapter)
        self.weight_need_transpose = False  # handled in apply()
        self.infer_dtype = torch.float16

    def load(self, weight_dict):
        # load_func is set by the subclass; reuse the template's quant loaders.
        self.load_func(weight_dict)

    def apply(self, input_tensor):
        if input_tensor.dtype != self.infer_dtype:
            input_tensor = input_tensor.to(self.infer_dtype)

        weight = self.weight
        scale = self.weight_scale
        w = weight.to(self.infer_dtype)
        if scale.dim() == 1:
            w = w * scale.to(self.infer_dtype).unsqueeze(0)
        else:
            w = w * scale.to(self.infer_dtype).t()
        w = w.t()  # → [out, in] for F.linear

        bias = self.bias.to(self.infer_dtype) if getattr(self, "bias", None) is not None else None
        return _linear(input_tensor, w, bias)


@MM_WEIGHT_REGISTER("flagos-fp8")
class FlagOSFp8MmWeight(_FlagOSQuantMm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_scale_name = self.weight_name.removesuffix(".weight") + ".weight_scale"
        self.load_func = self.load_fp8_perchannel_sym


@MM_WEIGHT_REGISTER("flagos-int8")
class FlagOSInt8MmWeight(_FlagOSQuantMm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_scale_name = self.weight_name.removesuffix(".weight") + ".weight_scale"
        self.load_func = self.load_int8_perchannel_sym
