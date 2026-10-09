"""From-scratch speculative decoding for Qwen3.8 (mlx-lm ``qwen3_5``) in MLX.

Modules:
  gdn       rollback-capable gated-delta-net layers and their cache
  model     SpecModel: target forward with hidden states, rollback ``commit``
  drafters  the Drafter interface, n-gram and draft-model drafters, MTP plan
  decode    greedy baseline and the speculative loop
  bench     benchmark harness, verify-cost curve
"""

from .decode import GenResult, greedy_generate, speculative_generate
from .drafters import DraftModelDrafter, Drafter, HybridDrafter, NgramDrafter, NoDrafter
from .model import ForwardOutput, SpecModel
