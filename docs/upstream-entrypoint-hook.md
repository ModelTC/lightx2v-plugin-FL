# Proposal: an entry-point hook for out-of-tree platform plugins

**Target:** `ModelTC/LightX2V`
**Scope:** ~10 lines in `lightx2v_platform/set_ai_device.py` (+ docs)
**Goal:** let a third-party pip package register a new `PLATFORM` backend
(device + ops) **without editing any LightX2V source file** — the same
out-of-tree (OOT) plugin model vLLM uses for hardware backends.

---

## 1. Motivation

Today, adding a chip backend to `lightx2v_platform` requires editing in-tree
files:

- `lightx2v_platform/base/__init__.py` — import the new `Device` class.
- `lightx2v_platform/ops/__init__.py` — add an `elif PLATFORM == "<chip>":`
  branch that imports the chip's op modules.

This means every backend must be upstreamed (or the user must patch the repo) to
be usable. For ecosystem partners who want to ship a backend as an installable
package — e.g. a FlagOS backend that covers many chips at once via FlagGems /
FlagCX — there is no clean seam.

vLLM solved the identical problem with entry-point plugins
(`vllm.platform_plugins`). This proposal adds the equivalent seam to LightX2V.

## 2. The timing constraint (why a naive `import` is not enough)

`lightx2v/utils/registry_factory.py` builds the framework-facing registries by
**copying** the platform staging registries at import time:

```python
# lightx2v/utils/registry_factory.py  (current)
ATTN_WEIGHT_REGISTER.merge(PLATFORM_ATTN_WEIGHT_REGISTER)
MM_WEIGHT_REGISTER.merge(PLATFORM_MM_WEIGHT_REGISTER)
RMS_WEIGHT_REGISTER.merge(PLATFORM_RMS_WEIGHT_REGISTER)
LN_WEIGHT_REGISTER.merge(PLATFORM_LAYERNORM_WEIGHT_REGISTER)
ROPE_REGISTER.merge(PLATFORM_ROPE_REGISTER)
```

`merge()` is a one-shot snapshot, not a live view. So a plugin's registrations
are only picked up if they run **before** this module is imported. The current
load order is:

```
import lightx2v
 └─ lightx2v/__init__.py:  import lightx2v_platform.set_ai_device
      ├─ from lightx2v_platform import *          # devices register
      ├─ set_ai_device()                          # init device for PLATFORM
      └─ from lightx2v_platform.ops import *       # in-tree op branches register
 └─ ... later: registry_factory imported → merge() snapshots PLATFORM_* tables
```

The right seam is therefore **inside `set_ai_device()`**, after the in-tree op
import and before any `registry_factory` import — exactly where in-tree branches
already populate `PLATFORM_*`.

## 3. Proposed change

Add an entry-point scan to `lightx2v_platform/set_ai_device.py`:

```python
# lightx2v_platform/set_ai_device.py
import os

from loguru import logger

from lightx2v_platform import *


def _load_platform_plugins():
    """Discover out-of-tree platform backends via entry points.

    Third-party packages register under the 'lightx2v.platform_plugins' group.
    Each entry point is a zero-arg callable that registers its Device class into
    PLATFORM_DEVICE_REGISTER and its ops into the PLATFORM_* op registries.

    Runs here — after in-tree ops import, before registry_factory's merge() — so
    plugin registrations are included in the framework registries.
    """
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover  (py<3.8, not supported anyway)
        return

    try:
        eps = entry_points(group="lightx2v.platform_plugins")
    except TypeError:
        # importlib.metadata < 3.10 returns a dict-like object.
        eps = entry_points().get("lightx2v.platform_plugins", [])

    for ep in eps:
        try:
            ep.load()()
            logger.info(f"Loaded LightX2V platform plugin: {ep.name}")
        except Exception as e:
            logger.warning(f"Failed to load platform plugin '{ep.name}': {e}")


def set_ai_device():
    platform = os.getenv("PLATFORM", "cuda")
    _load_platform_plugins()        # <-- NEW: before device lookup / ops import
    init_ai_device(platform)
    check_ai_device(platform)


set_ai_device()
from lightx2v_platform.ops import *  # noqa: E402
```

Note `_load_platform_plugins()` is called **before** `init_ai_device(platform)`
so a plugin-provided device (e.g. `flagos`) is present in
`PLATFORM_DEVICE_REGISTER` when the lookup happens.

### Plugin side (no further upstream change)

```toml
# a third-party package's pyproject.toml
[project.entry-points."lightx2v.platform_plugins"]
flagos = "lightx2v_fl:register"
```

```python
# lightx2v_fl/__init__.py
def register():
    from . import device     # @PLATFORM_DEVICE_REGISTER("flagos")
    from .ops import register_ops
    register_ops()            # registers into PLATFORM_* op tables
```

Then:

```bash
pip install lightx2v-plugin-fl
PLATFORM=flagos python lightx2v/infer.py ...
```

## 4. Why this is safe

- **No behaviour change when no plugins are installed.** `entry_points(group=…)`
  returns empty; the loop is a no-op. In-tree platforms are untouched.
- **Failures are isolated.** A broken plugin logs a warning and is skipped; it
  cannot crash `set_ai_device()`.
- **No new dependency.** `importlib.metadata` is stdlib (3.8+).
- **Mirrors a proven design.** This is the same mechanism vLLM uses for OOT
  hardware backends, so the pattern is familiar to the ecosystem.

## 5. Optional follow-up (nice-to-have, not required)

Make the op registries live instead of snapshotted, removing the ordering
constraint entirely. E.g. have `registry_factory` reference the `PLATFORM_*`
tables via a chained lookup rather than `merge()`-copy at import. This is a
larger change and is **not** needed for the plugin model — the hook above is
sufficient — but it would let plugins register at any time.

## 6. Test plan

1. **No-plugin regression:** existing platforms (`cuda`, `ascend_npu`, …) behave
   identically; assert `set_ai_device()` output unchanged.
2. **Plugin discovery:** install a stub package exposing a dummy
   `lightx2v.platform_plugins` entry point; assert its device key appears in
   `PLATFORM_DEVICE_REGISTER` and its op keys reach
   `lightx2v.utils.registry_factory` after `import lightx2v`.
3. **Failure isolation:** a plugin whose `register()` raises only logs a warning;
   `set_ai_device()` still completes for the configured `PLATFORM`.

---

### Appendix — current vs proposed `set_ai_device.py` (diff)

```diff
 import os

+from loguru import logger
+
 from lightx2v_platform import *


+def _load_platform_plugins():
+    try:
+        from importlib.metadata import entry_points
+    except Exception:
+        return
+    try:
+        eps = entry_points(group="lightx2v.platform_plugins")
+    except TypeError:
+        eps = entry_points().get("lightx2v.platform_plugins", [])
+    for ep in eps:
+        try:
+            ep.load()()
+            logger.info(f"Loaded LightX2V platform plugin: {ep.name}")
+        except Exception as e:
+            logger.warning(f"Failed to load platform plugin '{ep.name}': {e}")
+
+
 def set_ai_device():
     platform = os.getenv("PLATFORM", "cuda")
+    _load_platform_plugins()
     init_ai_device(platform)
     check_ai_device(platform)


 set_ai_device()
 from lightx2v_platform.ops import *  # noqa: E402
```
