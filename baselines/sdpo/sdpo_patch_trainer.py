"""Patch verl's RayPPOTrainer for SDPO self-distillation batch building.

Monkey-patches the training loop to call _maybe_build_self_distillation_batch
after reward computation and before advantage computation. This builds
teacher prompts (question + successful solution) and tokenizes them so
the actor can do a teacher forward pass during update_policy.

Verbatim logic from lasgroup/SDPO ray_trainer.py, adapted as a patch.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from baselines.sdpo.sdpo_core import SelfDistillationConfig

logger = logging.getLogger(__name__)



# Module-level stash so the actor patch can find the config
_sdpo_algorithm_config = None


def _build_sdpo_config(trainer_config):
    """Extract SelfDistillationConfig from trainer config.

    Checks algorithm.self_distillation (preferred location, avoids
    Hydra ActorConfig validation), then actor.self_distillation as fallback.
    """
    global _sdpo_algorithm_config
    sd_raw = trainer_config.algorithm.get("self_distillation", None)
    if sd_raw is None:
        sd_raw = trainer_config.actor_rollout_ref.actor.get("self_distillation", None)
    if sd_raw is None:
        return None
    cfg = SelfDistillationConfig()
    if hasattr(sd_raw, "items"):
        items = sd_raw.items()
    elif hasattr(sd_raw, "__iter__"):
        items = []
        for k in dir(sd_raw):
            if not k.startswith("_") and hasattr(cfg, k):
                items.append((k, getattr(sd_raw, k)))
    else:
        items = []
    for k, v in items:
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # Stash so the actor patch can find it
    _sdpo_algorithm_config = cfg
    return cfg


def _remove_thinking_trace(text: str) -> str:
    """Remove <think>...</think> tags and their content from text."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def _collect_feedback(
    include_environment_feedback: bool,
    reward_extra_infos_dict: Optional[dict[str, Any]],
    batch_size: int,
) -> list[Any]:
    feedback_list: list[Any] = [None] * batch_size
    if include_environment_feedback and reward_extra_infos_dict is not None:
        raw_feedback = reward_extra_infos_dict.get("feedback", [])
        for i in range(min(len(raw_feedback), batch_size)):
            if (raw_feedback[i]
                    and isinstance(raw_feedback[i], str)
                    and raw_feedback[i].strip()):
                feedback_list[i] = raw_feedback[i]
    return feedback_list


def _collect_solutions_by_uid(
    batch, reward_tensor, success_reward_threshold
) -> dict[Any, list[int]]:
    seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
    uids = batch.non_tensor_batch["uid"]
    success_by_uid: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        if seq_scores[idx] >= success_reward_threshold:
            success_by_uid[uid].append(idx)
    return success_by_uid


def _get_solution(
    idx: int,
    success_by_uid: dict[Any, list[int]],
    uids,
    response_texts: list[str],
    dont_reprompt_on_self_success: bool = False,
    remove_thinking_from_demonstration: bool = False,
) -> Optional[str]:
    uid = uids[idx]
    solution_idxs = success_by_uid[uid]
    if dont_reprompt_on_self_success:
        solution_idxs = [j for j in solution_idxs if j != idx]
    if len(solution_idxs) == 0:
        return None
    solution_idx = solution_idxs[0]
    solution_str = response_texts[solution_idx]
    if remove_thinking_from_demonstration:
        solution_str = _remove_thinking_trace(solution_str)
    return solution_str


def maybe_build_self_distillation_batch(
    batch,
    reward_tensor: torch.Tensor,
    tokenizer,
    sd_cfg: SelfDistillationConfig,
    reward_extra_infos_dict: Optional[dict[str, list]] = None,
    chat_template_kwargs: Optional[dict] = None,
):
    """Build teacher prompt inputs for SDPO self-distillation.

    Finds successful responses, builds teacher prompts that include the
    successful solution as context, tokenizes them, and returns tensors
    to be unioned into the training batch.

    Returns None if SDPO is not configured or no data to process.
    """
    from verl.protocol import DataProto
    from verl.utils.model import compute_position_id_with_mask

    device = batch.batch["input_ids"].device
    response_mask = batch.batch["response_mask"]
    responses = batch.batch["responses"]
    batch_size = batch.batch.batch_size[0]

    # Decode responses
    response_texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in responses
    ]

    # Extract prompt text from raw_prompt messages
    prompt_texts = []
    for msgs in batch.non_tensor_batch["raw_prompt"]:
        if isinstance(msgs, list) and len(msgs) > 0:
            last_msg = msgs[-1]
            if isinstance(last_msg, dict):
                prompt_texts.append(last_msg.get("content", ""))
            else:
                prompt_texts.append(str(last_msg))
        else:
            prompt_texts.append(str(msgs))

    # Collect feedback (environment errors, etc.)
    feedback_list = _collect_feedback(
        include_environment_feedback=sd_cfg.include_environment_feedback,
        reward_extra_infos_dict=reward_extra_infos_dict,
        batch_size=batch_size,
    )

    # Find successful solutions grouped by uid
    success_by_uid = _collect_solutions_by_uid(
        batch, reward_tensor, sd_cfg.success_reward_threshold
    )

    # Get solution string for each sample
    solution_strs = [
        _get_solution(
            i,
            success_by_uid,
            batch.non_tensor_batch["uid"],
            response_texts,
            sd_cfg.dont_reprompt_on_self_success,
            sd_cfg.get("remove_thinking_from_demonstration", False),
        )
        for i in range(batch_size)
    ]

    def _build_teacher_message(i: int) -> list[dict]:
        """Build the teacher prompt message for sample i."""
        raw_prompt = batch.non_tensor_batch["raw_prompt"][i]
        if isinstance(raw_prompt, list):
            system_messages = raw_prompt[:-1]
        else:
            system_messages = []

        has_solution = solution_strs[i] is not None
        has_feedback = feedback_list[i] is not None
        feedback_only_without_solution = sd_cfg.get(
            "environment_feedback_only_without_solution", False
        )
        use_feedback = has_feedback and (
            not feedback_only_without_solution or not has_solution
        )

        solution_section = ""
        if has_solution:
            solution_section = sd_cfg.solution_template.format(
                successful_previous_attempt=solution_strs[i]
            )

        feedback_section = ""
        if use_feedback:
            feedback_section = sd_cfg.feedback_template.format(
                feedback_raw=feedback_list[i]
            )

        if use_feedback or has_solution:
            reprompt_text = sd_cfg.reprompt_template.format(
                prompt=prompt_texts[i],
                solution=solution_section,
                feedback=feedback_section,
            )
        else:
            reprompt_text = prompt_texts[i]

        return system_messages + [{"role": "user", "content": reprompt_text}]

    messages = [_build_teacher_message(i) for i in range(batch_size)]

    # Tokenize teacher prompts
    template_kwargs = {}
    if chat_template_kwargs:
        template_kwargs = dict(chat_template_kwargs)

    # Try to use the tokenizer's chat template. Thread enable_thinking if the
    # tokenizer/template supports it (parity with upstream SDPO which passes it
    # through for Qwen3-style thinking models).
    enable_thinking = template_kwargs.pop("enable_thinking", None)
    apply_kwargs = dict(template_kwargs)
    if enable_thinking is not None:
        apply_kwargs["enable_thinking"] = enable_thinking
    try:
        teacher_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            max_length=sd_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
            **apply_kwargs,
        )
    except Exception:
        # Fallback: encode concatenated text
        teacher_input_ids_list = []
        for msg_list in messages:
            text = " ".join(m.get("content", "") for m in msg_list)
            ids = tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=sd_cfg.max_reprompt_len,
                truncation=True,
            )
            teacher_input_ids_list.append(
                torch.tensor(ids, dtype=torch.long)
            )
        # Pad to same length
        max_len = max(len(ids) for ids in teacher_input_ids_list)
        padded_ids = []
        padded_mask = []
        pad_id = tokenizer.pad_token_id or 0
        for ids in teacher_input_ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            padded_mask.append(
                torch.cat([
                    torch.ones(len(ids), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )
        teacher_prompt = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_mask),
        }

    # Concatenate teacher prompt with response tokens
    teacher_input_ids = torch.cat(
        [teacher_prompt["input_ids"].to(device), responses], dim=1
    )
    teacher_attention_mask = torch.cat(
        [teacher_prompt["attention_mask"].to(device), response_mask], dim=1
    )
    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

    # Build self_distillation_mask: True if sample has solution or feedback
    feedback_only_without_solution = sd_cfg.get(
        "environment_feedback_only_without_solution", False
    )
    feedback_used = [
        feedback_list[i] is not None
        and (not feedback_only_without_solution or solution_strs[i] is None)
        for i in range(batch_size)
    ]
    self_distillation_mask = torch.tensor(
        [
            solution_strs[i] is not None or feedback_used[i]
            for i in range(batch_size)
        ],
        dtype=torch.float32,
        device=device,
    )

    # Metrics
    uids = set(batch.non_tensor_batch["uid"])
    num_with_solution = sum(1 for s in solution_strs if s is not None)
    num_with_feedback_used = sum(1 for f in feedback_used if f)
    sdpo_metrics = {
        "self_distillation/success_group_fraction": (
            len([uid for uid in uids if len(success_by_uid[uid]) > 0])
            / max(len(uids), 1)
        ),
        "self_distillation/success_sample_fraction": (
            num_with_solution / batch_size
        ),
        "self_distillation/feedback_used_fraction": (
            num_with_feedback_used / batch_size
        ),
        "self_distillation/reprompt_sample_fraction": (
            self_distillation_mask.float().mean().item()
        ),
    }

    return (
        DataProto.from_dict(
            tensors={
                "teacher_input_ids": teacher_input_ids,
                "teacher_attention_mask": teacher_attention_mask,
                "teacher_position_ids": teacher_position_ids,
                "self_distillation_mask": self_distillation_mask,
            }
        ),
        sdpo_metrics,
    )


def patch_fit_for_sdpo():
    """Monkey-patch RayPPOTrainer.fit to inject self-distillation batch building.

    Injects a call to maybe_build_self_distillation_batch right after
    reward computation (after extract_reward) and before advantage
    computation (before compute_advantage).

    We do this by patching the compute_advantage function to first
    build the self-distillation batch if SDPO is configured.
    """
    import verl.trainer.ppo.ray_trainer as ray_trainer_module

    original_compute_advantage = ray_trainer_module.compute_advantage

    if getattr(original_compute_advantage, "_sdpo_trainer_patched", False):
        return

    # We store the tokenizer and config as module-level state, set during
    # the first call from a trainer that has SDPO configured.
    _sdpo_state = {"tokenizer": None, "config": None, "sd_cfg": None}

    def patched_compute_advantage(data, adv_estimator, **kwargs):
        """Wrapper that builds self-distillation batch before computing advantages."""
        config = kwargs.get("config", None)

        # Check if SDPO is configured by looking for self_distillation data
        # that was already added to the batch (via the separate patch below)
        # This function just passes through to the original
        return original_compute_advantage(data, adv_estimator, **kwargs)

    patched_compute_advantage._sdpo_trainer_patched = True
    ray_trainer_module.compute_advantage = patched_compute_advantage

    # The main injection: patch _update_actor to build the batch before updating
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if not hasattr(RayPPOTrainer, "_sdpo_original_update_actor"):
        original_update_actor = RayPPOTrainer._update_actor

        def patched_update_actor(self, batch):
            """Build self-distillation batch before actor update.

            Activates for both loss_mode='sdpo' (token-KL) and 'sdpo_full'
            (full-logit / top-k KL). Both need the same teacher-prompt batch.
            """
            loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get(
                "loss_mode", "vanilla"
            )
            if loss_mode not in ("sdpo", "sdpo_full"):
                return original_update_actor(self, batch)

            # Build self-distillation batch if not already present
            if "teacher_input_ids" not in batch.batch.keys():
                sd_cfg = _build_sdpo_config(self.config)
                if sd_cfg is not None:
                    # Stash config in meta_info so actor patch can read it
                    batch.meta_info["sdpo_config"] = sd_cfg
                    # Get reward tensor from batch
                    reward_tensor = batch.batch.get(
                        "token_level_rewards",
                        batch.batch.get("token_level_scores", None),
                    )
                    if reward_tensor is not None:
                        chat_template_kwargs = getattr(
                            self.config.data, "apply_chat_template_kwargs", None
                        )
                        if chat_template_kwargs and hasattr(chat_template_kwargs, "__iter__"):
                            chat_template_kwargs = dict(chat_template_kwargs)
                        else:
                            chat_template_kwargs = None

                        result = maybe_build_self_distillation_batch(
                            batch=batch,
                            reward_tensor=reward_tensor,
                            tokenizer=self.tokenizer,
                            sd_cfg=sd_cfg,
                            chat_template_kwargs=chat_template_kwargs,
                        )
                        if result is not None:
                            sd_batch, sd_metrics = result
                            batch = batch.union(sd_batch)
                            # Log metrics
                            for k, v in sd_metrics.items():
                                logger.info(f"  {k}: {v:.4f}")

            return original_update_actor(self, batch)

        RayPPOTrainer._sdpo_original_update_actor = original_update_actor
        RayPPOTrainer._update_actor = patched_update_actor
        logger.info("SDPO: patched RayPPOTrainer._update_actor")

    logger.info("SDPO: patched compute_advantage (trainer)")
