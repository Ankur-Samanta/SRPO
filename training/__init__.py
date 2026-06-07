"""Thought-level GRPO training on VERL.

Importing this module registers agent loops with VERL's agent loop registry.
"""

from training.thought_agent_loop import ThoughtAgentLoop
from training.thought_ics_agent_loop import ThoughtICSAgentLoop
from training.srpo_agent_loop import SRPOAgentLoop
from training.srpo_2x4_agent_loop import SRPO2x4AgentLoop
from training.srpo_mask_variants import SRPONoMaskAgentLoop
from training import srpo_loss  # noqa: F401 -- registers "srpo" policy loss (used by SRPO/RRPO)
from training import srpo_clip_loss   # noqa: F401 -- registers "srpo_clip" policy loss (clipped-surrogate ablation)

# Patch compute_data_metrics in ray_trainer's module namespace so ICS stats
# (ics_triggered, ics_iterations, ics_corrected, per-iteration accuracy) are
# emitted to wandb for any run that uses ThoughtICSAgentLoop. No-op on non-ICS
# runs since compute_ics_metrics returns {} when ICS keys are absent.
import verl.trainer.ppo.ray_trainer as _ray_trainer_mod
from training.ics_metrics import compute_ics_metrics as _compute_ics_metrics

_orig_compute_data_metrics = _ray_trainer_mod.compute_data_metrics

def _patched_compute_data_metrics(batch, **kwargs):
    result = _orig_compute_data_metrics(batch, **kwargs)
    result.update(_compute_ics_metrics(batch))
    return result

_ray_trainer_mod.compute_data_metrics = _patched_compute_data_metrics

__all__ = ["ThoughtAgentLoop", "ThoughtICSAgentLoop", "SRPOAgentLoop", "SRPO2x4AgentLoop", "SRPONoMaskAgentLoop"]
