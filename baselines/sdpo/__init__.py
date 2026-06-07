"""SDPO baseline for SCPO.

Implements the paper-faithful ``sdpo_full`` mode, selected via
``actor.policy_loss.loss_mode``:

    "sdpo_full"  -- full-logit KL distillation (paper-faithful primary mode).
                    Requires patched _forward_micro_batch with all_logps/topk
                    support and an EMA teacher built from ref_module_fsdp.
                    Config: baselines/sdpo/config/sdpo_full_math500.yaml

Reference:
    Hübotter et al. "Reinforcement Learning via Self-Distillation." arXiv:2601.20802.
    Code: https://github.com/lasgroup/SDPO
"""

# Teacher-prompt setup (shared) + full-logit distillation patches.
from baselines.sdpo.sdpo_patch_trainer import patch_fit_for_sdpo
from baselines.sdpo.sdpo_forward import patch_forward_micro_batch
from baselines.sdpo.sdpo_full_patch_actor import patch_update_policy_full
from baselines.sdpo.sdpo_patch_worker import patch_worker_init_model

# Order matters: forward-patch + worker-patch must happen before update_policy
# patches read teacher_module / call patched _forward_micro_batch.
patch_forward_micro_batch()
patch_worker_init_model()
patch_update_policy_full()     # "sdpo_full"
patch_fit_for_sdpo()

__all__: list[str] = []
