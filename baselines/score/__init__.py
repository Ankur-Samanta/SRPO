"""SCoRe (Self-Correction via Reinforcement Learning) baseline for SRPO.

Implements the SCoRe method from:
    Kumar et al. "Training Language Models to Self-Correct via Reinforcement
    Learning." ICLR 2025. arXiv:2409.12917.

This module registers:
    - "score_agent" agent loop with VERL
    - "score" advantage estimator with VERL (REINFORCE with shaped reward)
    - Patches compute_advantage to apply SCoRe's shaped reward

Key differences from TGRPO/SRPO:
    - Multi-turn self-correction: generates y1, if wrong appends "try again"
      and generates y2 from scratch (no error localization)
    - REINFORCE policy gradient (no PPO clipping, no GRPO group-relative
      advantages)
    - Shaped reward: R(y2) + alpha * (R(y2) - R(y1)) rewards self-correction
    - Two-stage training: Stage 1 initializes correction, Stage 2 trains both
"""

from baselines.score.score_agent_loop import SCoReAgentLoop
from baselines.score import score_advantage  # noqa: F401 -- registers "score" estimator
from baselines.score.score_scoring_hook import patch_compute_advantage

# Apply monkey-patch so verl's compute_advantage passes SCoRe metadata
# to the advantage estimator for shaped reward computation.
patch_compute_advantage()

__all__ = ["SCoReAgentLoop"]
