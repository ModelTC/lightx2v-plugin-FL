# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""H3 policy names mapped to the public, model-independent FlagOmni APIs.

Model configuration, weight staging, patching and lifecycle stay in this plugin.
FlagOmni owns only numerical operator contracts and backend implementation.
"""

H3_OPERATOR_APIS = {
    "h3_rms_norm": "rms_norm",
    "h3_swiglu": "packed_swiglu",
    "h3_vae_qk_norm_rope": "qk_norm_rope",
    "h3_vae_residual": "scaled_residual",
    "h3_vae_residual_norm": "residual_rms_norm",
    "h3_reference_blend": "reference_blend",
    "h3_rotary_tables": "rotary_tables",
}
