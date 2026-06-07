"""Shared-prefix masking variant of SRPO.

Buffer layout and advantage computation are identical to the parent class
— only the suffix_start_idx metadata (and hence the loss mask) differs.

                            mask_correction_prefix
  srpo_agent (default)            True
  srpo_nomask_agent                   False    ← no shared-prefix masking
"""

from training.srpo_agent_loop import SRPOAgentLoop


class SRPONoMaskAgentLoop(SRPOAgentLoop):
    """SRPO with no shared-prefix masking (G2 counterfactuals get full-response gradient)."""

    _buffers: dict = {}
    mask_correction_prefix = False
