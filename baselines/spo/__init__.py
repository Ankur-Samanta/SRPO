"""SPO-Tree baseline for SCPO.

Faithful implementation of SPO-Tree (arXiv:2505.23564, §5) — the tree-based
segment-level variant reported in the paper.

Algorithm (paper §3–§5):
    1. One-shot generation with token-level logprobs.
    2. Adaptive cutpoint-based partition:
           𝒰 = {t : π_θ_old(y_t|s_t) < ρ}
           K = ⌈|𝒰| / interval⌉  segments
           boundaries solve min Σ |𝒰 ∩ [t_k, t_{k+1})|² (even-cutpoint split).
    3. MC value estimation V̂(s_{t_k}) at each of the K-1 interior boundaries
       (N rollouts each). V(s_0) is reused from the group-mean outcome of the
       sibling rollouts; V(s_T) is the trajectory's own terminal reward.
    4. Eq. 2 segment advantages A_k = V(t_k) − V(t_{k-1}).
    5. Eq. 3 probability mask M_t = 𝕀[π_θ_old < ρ] and Z_s = Σ M_t
       normalization (applied via per-sequence advantage scaling so verl's
       default token-mean aggregator reproduces the paper's loss exactly).

Prompt format is kept from TGRPO (thought-based) per project convention; this
is a generation-side convention and is orthogonal to SPO segmentation.

Registers:
    - "spo_tree_agent" agent loop
    - "spo_tree" advantage estimator
    - monkey-patch on verl.trainer.ppo.ray_trainer.compute_advantage
"""

from baselines.spo.spo_tree_agent_loop import SPOTreeAgentLoop
from baselines.spo import spo_advantage  # noqa: F401 -- registers "spo_tree"
from baselines.spo.spo_scoring_hook import patch_compute_advantage

# Bridge verl's compute_advantage to the SPO-Tree scoring path.
patch_compute_advantage()

__all__ = ["SPOTreeAgentLoop"]
