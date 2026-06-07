"""SDPO core: self-distillation loss and config.

Verbatim from lasgroup/SDPO (arXiv:2601.20802), with minimal
adaptation for standalone import.

This file contains:
    - SelfDistillationConfig: dataclass for SDPO hyperparameters
    - compute_self_distillation_loss: the core KL distillation loss
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import agg_loss


# --------------------------------------------------------------------------- #
# Config (from verl/workers/config/actor.py in SDPO repo)
# --------------------------------------------------------------------------- #

@dataclass
class SelfDistillationConfig:
    full_logit_distillation: bool = True
    alpha: float = 0.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False

    def get(self, key, default=None):
        return getattr(self, key, default)


# --------------------------------------------------------------------------- #
# Loss (from verl/trainer/ppo/core_algos.py in SDPO repo)
# --------------------------------------------------------------------------- #

def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:

    metrics = {}

    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    if self_distillation_config.full_logit_distillation:
        use_topk = self_distillation_config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError(
                    "top-k distillation requires student_topk_log_probs "
                    "and teacher_topk_log_probs."
                )

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)
                tail_log = torch.log(-torch.expm1(log_s))
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if self_distillation_config.distillation_add_tail:
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(
                    student_distill_log_probs
                )
                teacher_distill_log_probs = renorm_topk_log_probs(
                    teacher_distill_log_probs
                )
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError(
                    "full_logit_distillation requires student_all_log_probs "
                    "and teacher_all_log_probs."
                )
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        if self_distillation_config.alpha == 0.0:
            kl_loss = F.kl_div(
                student_distill_log_probs,
                teacher_distill_log_probs,
                reduction="none",
                log_target=True,
            )
        elif self_distillation_config.alpha == 1.0:
            kl_loss = F.kl_div(
                teacher_distill_log_probs,
                student_distill_log_probs,
                reduction="none",
                log_target=True,
            )
        else:
            alpha = torch.tensor(
                self_distillation_config.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_distill_log_probs + torch.log(1 - alpha),
                        teacher_distill_log_probs + torch.log(alpha),
                    ]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(
                mixture_log_probs,
                teacher_distill_log_probs,
                reduction="none",
                log_target=True,
            )
            kl_student = F.kl_div(
                mixture_log_probs,
                student_distill_log_probs,
                reduction="none",
                log_target=True,
            )
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)

        per_token_loss = kl_loss.sum(-1)
    else:
        assert self_distillation_config.alpha == 1.0, (
            "Only reverse KL is supported for non-full-logit distillation"
        )
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    is_clip = self_distillation_config.is_clip
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError(
                "old_log_probs is required for distillation IS ratio."
            )

        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=loss_mask.sum().clamp(min=1.0),
    )
    return loss, metrics
