# lightx2v-plugin-FL

A **pluggable** [LightX2V](https://github.com/ModelTC/LightX2V) backend built on
the [FlagOS](https://github.com/FlagOpen) unified multi-chip stack:

- **[FlagGems](https://github.com/FlagOpen/FlagGems)** — a Triton operator
  library that auto-detects the underlying vendor (NVIDIA / Ascend / Cambricon /
  MetaX / MUSA / Kunlun / Iluvatar / …) and runs *one* set of kernels across all
  of them.
- **[FlagCX](https://github.com/FlagOpen/FlagCX)** — a cross-chip collective
  communication library that plugs into PyTorch as a `flagcx` ProcessGroup
  backend.

## Why a `flagos` meta-platform?

LightX2V's `lightx2v_platform` already abstracts domestic chips, but each chip
lives in its own `ops/<attn|mm|norm|rope>/<chip>/` directory with a hand-written
kernel set. That is N adaptations for N chips.

`flagos` is different: it is **one** `PLATFORM=flagos` backend that covers *every*
chip FlagOS supports, by delegating compute to FlagGems and communication to
FlagCX. Add a chip to FlagOS → it works in LightX2V through this plugin, no new
LightX2V code.

```
            ┌──────────────────────────────────────────────┐
            │            LightX2V (host framework)         │
            │   registry_factory: MM/ATTN/RMS/ROPE keys    │
            └────────────────┬─────────────────────────────┘
                             │ "flagos*" registry keys
            ┌────────────────▼─────────────────────────────┐
            │          lightx2v-plugin-FL (this repo)      │
            │  device/  ops/{attn,mm,norm,rope}  enable.py │
            └──────────┬────────────────────────┬──────────┘
               compute │                   comm │
            ┌──────────▼─────────┐   ┌──────────▼──────────┐
            │      FlagGems      │   │        FlagCX       │
            │  (Triton kernels)  │   │  (ProcessGroup PG)  │
            └────────┬───────────┘   └──────────┬──────────┘
   NVIDIA · Ascend · Cambricon · MetaX · MUSA · Kunlun · Iluvatar · …
```

## Install

```bash
# 1. the plugin (this repo)
pip install -e .

# 2. FlagGems for your chip (example: NVIDIA)
pip install "flag_gems[nvidia]"      # or [ascend], [cambricon], ...

# 3. (optional, multi-card) FlagCX torch plugin for your chip
#    see https://github.com/FlagOpen/FlagCX
```

`flag_gems` and `flagcx` are **not** hard dependencies — the plugin degrades to
torch reference implementations when they are absent, so `import` never breaks.

## Use

```bash
PLATFORM=flagos python lightx2v/infer.py \
    --model_cls wan2.1 \
    --model_path /path/to/wan \
    --config_json lightx2v_fl/configs/wan_t2v_flagos.json
```

The config selects the FlagOS ops by registry key:

| Config field        | Value                  |
|---------------------|------------------------|
| `self_attn_1_type`  | `flagos_flash_attn`    |
| `cross_attn_*_type` | `flagos_flash_attn`    |
| `rms_norm_type`     | `flagos_rms_norm`      |
| `layer_norm_type`   | `flagos_layer_norm`    |
| `rope_type`         | `flagos_rope`          |
| `dit_quant_scheme`  | `flagos` / `flagos-fp8` / `flagos-int8` |

### Optional: global FlagGems aten patching

Lower *generic* torch ops (softmax, elementwise, …) to FlagGems too:

```bash
LIGHTX2V_FL_GLOBAL_GEMS=1 PLATFORM=flagos python lightx2v/infer.py ...
```

### Knobs

| Env var | Default | Effect |
|---|---|---|
| `LIGHTX2V_FL_AUTO_REGISTER` | `1` | Register on `import lightx2v_fl`. |
| `LIGHTX2V_FL_GLOBAL_GEMS` | `0` | `flag_gems.enable()` at aten level. |
| `LIGHTX2V_FL_GEMS_UNUSED` | – | Comma-list of aten ops to exclude from global patch. |
| `LIGHTX2V_FL_DISABLE_FLAGCX` | `0` | Use vendor-native CCL instead of FlagCX. |

## Activation: how the plugin is discovered

There are two paths; pick based on whether the upstream hook has landed.

### Preferred — entry point (needs the upstream hook)

`pyproject.toml` exposes:

```toml
[project.entry-points."lightx2v.platform_plugins"]
flagos = "lightx2v_fl:register"
```

Once LightX2V scans this group inside `set_ai_device()` (see
[`docs/upstream-entrypoint-hook.md`](docs/upstream-entrypoint-hook.md)),
`pip install lightx2v-plugin-fl` is all that's required.

### Fallback — import order (works today, zero upstream change)

`import lightx2v_fl` **before** `import lightx2v`. The plugin registers its ops
into the *final* `lightx2v.utils.registry_factory` tables, so it is immune to the
one-shot `merge()` snapshot that copies `PLATFORM_*` registries at framework
import time.

> **The snapshot gotcha.** `registry_factory.py` runs
> `ATTN_WEIGHT_REGISTER.merge(PLATFORM_ATTN_WEIGHT_REGISTER)` at import. `merge`
> is a copy, not a live view — anything registered into `PLATFORM_*` *after* that
> line is invisible. This plugin sidesteps it by registering into the final
> tables directly. The upstream hook fixes it properly by running plugin
> registration *before* the merge.

## Tests

```bash
pytest -q        # CPU-only smoke tests: wiring + torch-fallback correctness
```

## Status

MVP skeleton. The op `apply()` paths call FlagGems where its API is available and
fall back to torch otherwise; per-chip kernel selection and numerical alignment
(esp. fp8/int8 quant gemm) need validation on real hardware. See repo issues /
roadmap.

## License

Apache-2.0
