"""Critique-GRPO baseline for SCPO.

Implements the Critique-GRPO method from:
    Zhang et al. "Critique-GRPO: Advancing LLM Reasoning with Natural
    Language and Numerical Feedback." arXiv:2506.03106.

This module registers:
    - "critique_grpo_agent" agent loop with VERL
    - "critique_grpo" advantage estimator with VERL (Dr.GRPO, no std norm)
    - "critique_grpo" policy loss with VERL (on/off-policy shaping)
    - Patches compute_advantage to handle refinement token masking

Key differences from TGRPO/SCGRPO:
    - Generates N responses, critiques incorrect ones ("solution is incorrect"),
      generates refinements, mixes 1 refinement + (N-1) originals
    - Off-policy refinement tokens use shaping function p/(p+gamma)
    - GRPO advantages without std normalization (Dr.GRPO style)
    - No error localization — critique is at the whole-response level
"""

from baselines.critique_grpo.critique_agent_loop import CritiqueGRPOAgentLoop
from baselines.critique_grpo import critique_advantage  # noqa: F401
from baselines.critique_grpo import critique_loss  # noqa: F401
from baselines.critique_grpo.critique_scoring_hook import patch_compute_advantage

patch_compute_advantage()

__all__ = ["CritiqueGRPOAgentLoop"]
