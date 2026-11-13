# Bringing up another accelerator platform

This guide is for platform integration engineers. FlagGems device support is a
starting point, not an end-to-end H3 support guarantee. Validate the combination
of hardware, driver, vendor PyTorch, compiler, operator library and communication
runtime before enabling an optimized path. SageAttention tuning is not required
for the initial BF16 bring-up.

## Ownership and dispatch

* **LightX2V** owns model execution, parallelism and request handling.
* **This plugin** owns model adaptation, registry keys, policy and call wiring.
* **FlagGems** owns ordinary operators and optional ATen replacement.
* **FlagOmni** owns model-independent fused operators. Vendor/architecture
  implementations belong under its runtime backend, not model-specific APIs.
* **FlagCX or vendor CCL** owns collectives. Validate it separately from compute.

Do not identify hardware from `torch.cuda` alone. Kunlun Xtorch and other vendor
stacks expose CUDA-compatible APIs. Use FlagGems vendor identification and the
compiler target for specialized implementations. SM90/FP8 is not a generic CUDA
capability. An unsupported operation should select an explicitly supported path
**before** execution; never catch numerical/compiler errors and silently retry.

The public attention entry is `sage_attention`; backend-specific public names
are not needed. For initial bring-up, use a verified vendor BF16 FlashAttention
implementation instead of requiring Sage. Preserve packed `cu_seqlens` isolation,
layout, dtype, scale, mask and return-value semantics when writing its adapter.
The generic `flagos_flash_attn` key currently dispatches dense SDPA-compatible
APIs; its packed fallback loops over samples and reads boundaries on the host.
It is not a fused varlen or fullgraph implementation. SDPA may allocate quadratic
memory: do not use a long-sequence run to discover whether Flash was selected.

## Image and dependency preparation

There is no universal binary image. Start with the vendor's matching driver,
PyTorch and compiler image; do not replace vendor PyTorch with a stock CUDA wheel.
Record image **digest**, driver/runtime, Python/PyTorch, FlagTree/Triton,
FlagGems/FlagOmni/plugin/LightX2V commits, attention and collective libraries.
A locally cached image ID is not a distributable image reference. Obtain the
vendor's registry/archive and verify its digest before installation.

```bash
# Inside the device-matched image, with the vendor torch/compiler already working:
git clone https://github.com/flagos-ai/lightx2v-plugin-FL.git
cd lightx2v-plugin-FL
# Check out the reviewed PR/release commit, not an unrelated moving branch.
python -m pip install --no-deps -e .
python -m pip install --no-deps -r requirements-h3-operators.txt
python -m pip check
python -c 'import torch, triton, flag_gems; print(torch.__version__, triton.__version__, flag_gems.vendor_name, flag_gems.device)'
```

`--no-deps` preserves the vendor stack, but means **you** must install compatible
missing Python dependencies reported by `pip check`. The H3 requirements pin
source revisions, not a complete vendor environment. Install the matching
LightX2V and `lightx2v_platform` host packages; the H3 integration was developed
against LightX2V `c58c16d607c49bf6021fecc6238ed177752d7e87`.
Plugin registration must occur before the host takes registry snapshots; see
[entry-point integration](upstream-entrypoint-hook.md).

### Recorded P800 base image

The isolated P800 baseline used this registry image (registry access required):

```text
harbor.baai.ac.cn/plugin/kunlunxin001-gems5.4.0-tree0.7.0rc2-triton3.6.0-cx0.13.0-plugin0.3.0-vllm0.24.0-cp310-pt29-x64-v5.0.21.47:202610110945
```

Verified repository digest:
`sha256:1ed2c722cbbf091e7c36f342d2ade1e77f8258164139ce4caf618cbbbb9d3abb`.
This identifies the **base image**, not a self-contained H3 release image.
The run additionally installed the host, plugin/operator sources and Python
media dependencies in an isolated environment. The base contains FlagTree
0.7.0rc2/Triton 3.6/Python 3.10/vendor PyTorch 2.9/XRT 5.0.21.47;
FlagTree 0.7.1+xpu3.6 was tested as a separate overlay. Do not overwrite the
vendor stack or infer that the image's bundled FlagGems matches this PR's pin.
Only authorized users should pull the image; credentials are not part of the
recipe. Other platforms need their own matching vendor image, not this image.

## Start conservatively

For a supported generic host model:

```bash
PLATFORM=flagos LIGHTX2V_FL_GLOBAL_GEMS=0 LIGHTX2V_FL_DISABLE_FLAGCX=1 \
  python lightx2v/infer.py --model_cls wan2.1 --model_path /models/wan \
  --config_json lightx2v_fl/configs/wan_t2v_flagos.json
```

For H3, follow [the H3 launcher](minimax-h3.md), including model/conditioning
checks. `examples/minimax_h3_sm90.json` is a specialized Sage
and RMS configuration, **not** a ready-to-run generic BF16 configuration. The
launcher accepts validated native attention/RMS registry keys and keeps host
operations unchanged unless `--enable-fusions` is supplied. A new
platform adapter must supply validated attention/RMS registry keys and matching
host configuration before claiming H3 support. Do not merely change FP8 to BF16
in that JSON. Initially omit `--enable-fusions` and keep
unsupported FFN/rotary fusions on the host path. Vendor Flash adapters and scoped
communication workarounds from isolated experiments must be ported and tested
before they become a supported recipe in this repository.

Ordinary explicit dispatch policy is configurable independently of ATen patching:

```json
{"fl_dispatch": {"attention": {"order": ["torch"], "parameters": {}},
                 "rms_norm": {"order": ["flaggems", "torch"], "parameters": {}}}}
```

This selects implementations, not a promise that torch SDPA chooses Flash.
Never use this as an unbounded-memory fallback for long H3 sequences. H3-specific
operators currently require FlagOmni and have no automatic torch retry.

Enable global FlagGems replacement only after explicit operator validation.
Defaults exclude `copy`, `copy_`, `zero`, `zero_`, `is_same_size`. A user blacklist
**replaces** defaults; use `LIGHTX2V_FL_GEMS_BLACKLIST_APPEND` to extend them.

## Evidence and remaining platform work

| Platform | Existing evidence | What this does not establish |
|---|---|---|
| NVIDIA SM90 | Specialized H3 configuration and operator integration | Other NVIDIA architectures or all precision modes |
| Moore Threads S5000 | Separate vendor-stack H3 end-to-end and raw-QKV Sage validation | Stock compiler compatibility or unchanged output quality; ported artifacts must be matched to commits |
| Kunlun P800 | Isolated 8-card BF16 vendor Flash H3 functional run, 352×640, 22 frames, one step | Production quality/performance, generic PR installation or INT8 Sage acceptance |
| Other FlagGems platforms | Common plugin interfaces | H3, fused operators, Flash or multi-card readiness |

The P800 baseline used Xtorch 2.9, vendor FlashAttention
`2.4.2+d2a78a4`, and native CCL. Repeated real compute/communication required
`XPU_EVENT_KL3_ENABLE=0` and explicit synchronization after all-to-all in that
isolated implementation. This is a platform-specific tested configuration, not
a universal performance recommendation or a claim that FlagCX fails generally.
FlagTree `0.7.1+xpu3.6` was tested separately for Sage; it does not make the
experimental Sage path accepted. Do not copy an unverified Sage prototype into
the default platform selection.

## Acceptance ladder

1. Run CPU policy/ABI tests: `python -m pytest -q tests`. These are not GPU tests.
2. Check one device: vendor identity, dtype support, small GEMM/RMS and the exact
   selected Flash implementation. Compare against an independent high-precision
   reference, including odd lengths and unequal Q/KV lengths.
3. Test two packed samples with very different values: changing sample B must
   not change sample A. Validate padding, empty segments and output layout.
4. Run one small full-model case with fixed weights, conditioning and seed.
   Verify finite outputs, decoded frames/audio and peak device/host memory.
5. Validate multi-card collectives, then repeated **compute plus communication**
   with real layouts and streams. A stand-alone all-to-all pass is insufficient.
   Bind each rank's device before process-group initialization.
6. Test graph capture/replay with changed inputs separately. Never call an eager
   result a graph pass. Check all ranks, completion codes and resource cleanup.
7. Increase frames/resolution/steps only after the bounded run passes. Measure
   cold compilation, warmed network, VAE and complete request time separately.

For new operators, implement numerical code in FlagGems/FlagOmni and only wiring
in this plugin. Add capability selection and contract tests, pin the tested
revision, and publish a reproducible image/build recipe plus model launch config.
Do not commit weights, credentials, machine-local paths or experiment JSON dumps.

## Explicit native Flash route (no Sage dependency)

The `flash_attn` dispatch backend adapts the vendor image's public
`flash_attn.flash_attn_func` to the plugin's dense BHSD contract. It uses the same
public entry verified in the isolated P800 baseline, without exposing a fake
Sage implementation name. Select it explicitly; it is not auto-selected merely
because a package is installed:

```json
{"attn_type": "flagos_flash_attn", "rms_type": "flagos_rms_norm",
 "fl_dispatch": {"attention": {"order": ["flash_attn"], "parameters": {}}}}
```

Merge these fields into the vendor host's complete H3 config, then use the H3
launcher **without** `--enable-fusions`. The generic RMS path reuses FlagGems or
the torch reference; revalidate H3 rounding/accuracy before comparing final
quality. This route has adapter contract tests, not a new full-model GPU result.
For packed inputs the existing per-sequence adapter retains isolation, but is
not fullgraph-safe. No automatic SDPA fallback is included in this example.

### Bounded BF16 launcher configuration

`examples/minimax_h3_bf16.json` is the concrete one-step/22-frame/64×64 bring-up
configuration, with native Flash explicitly selected and no ATen/fusion opt-in.
It deliberately requires FlagGems RMS rather than silently selecting a different
rounding path. Confirm the image supports the remaining host RoPE/FFN/VAE
operations; this configuration is a test candidate until the target passes.

```bash
# Run from this plugin checkout, with the vendor host installed/importable.
# MODEL_DIR contains the verified MiniMax/MiniMax-H3 checkpoint.
# CONDITIONING_FILE and its SHA are for this exact PROMPT (see minimax-h3.md).
PLATFORM=flagos LIGHTX2V_FL_GLOBAL_GEMS=0 LIGHTX2V_FL_DISABLE_FLAGCX=1 \
python examples/minimax_h3.py --model-path "$MODEL_DIR" \
  --config examples/minimax_h3_bf16.json --seed 42 --prompt "$PROMPT" \
  --conditioning "$CONDITIONING_FILE" --conditioning-sha256 "$CONDITIONING_SHA256" \
  --output /outputs/h3-bf16-smoke.mp4
```

The launcher imports `lightx2v_fl` before creating the pipeline, so this command
activates plugin registries even without the optional upstream entry-point hook.
It binds the runner dispatch policy to owned attention/RMS objects for inference
and restores their original configuration afterwards.

For SP8, the host config uses
`"parallel": {"seq_p_size": 8, "seq_p_attn_type": "ulysses"}` and the launcher
runs under `torchrun --standalone --nproc_per_node=8`. Do not launch it before
single-card acceptance and communication/stream checks. On the tested P800
stack, also set `XPU_EVENT_KL3_ENABLE=0` and
`LIGHTX2V_FL_DISABLE_FLAGCX=1`. The earlier passing isolation added explicit
synchronization after all-to-all; that host-side change is **not supplied by
this PR**, so these environment variables alone are not a complete SP8 fix.
Use the vendor/host implementation that provides the validated synchronization
before attempting this step. The new public launcher has not inherited an
8-card acceptance result merely by adopting the same Flash public function.
