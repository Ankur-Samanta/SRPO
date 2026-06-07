"""SDPO-full: patch DataParallelPPOActor.update_policy for full-logit KL distillation.

Activates when ``actor.policy_loss.loss_mode == "sdpo_full"``. Faithful port of
lasgroup/SDPO verl/workers/actor/dp_actor.py::update_policy SDPO branch:

    1. Student forward returns log_probs + either all_logps (full vocab) or
       topk_logps + topk_indices.
    2. Teacher forward receives student's topk_indices (for aligned top-k KL) and
       the teacher nn.Module via ``module=`` (no actor_module swap hack).
    3. Loss = compute_self_distillation_loss with full signature.
    4. After ``_optimizer_step``, call ``self._update_teacher()`` for EMA.

The SDPO-light variant (``loss_mode == "sdpo"``) in sdpo_patch_actor.py is
unchanged and remains available as a token-KL approximation.
"""

import logging
from typing import Optional

import torch

from baselines.sdpo.sdpo_core import SelfDistillationConfig, compute_self_distillation_loss

logger = logging.getLogger(__name__)


def _build_sdpo_config(source) -> Optional[SelfDistillationConfig]:
    if source is None:
        return None
    if isinstance(source, SelfDistillationConfig):
        return source
    cfg = SelfDistillationConfig()
    if hasattr(source, "items"):
        items = source.items()
    elif hasattr(source, "__iter__"):
        items = [(k, getattr(source, k)) for k in dir(source)
                 if not k.startswith("_") and hasattr(cfg, k)]
    else:
        items = []
    for k, v in items:
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def patch_update_policy_full():
    """Monkey-patch DataParallelPPOActor.update_policy to handle loss_mode='sdpo_full'."""
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
    from verl.utils.device import get_device_id
    from verl.utils.py_functional import append_to_dict

    original_update_policy = DataParallelPPOActor.update_policy
    if getattr(original_update_policy, "_sdpo_full_patched", False):
        return

    def patched_update_policy(self, data):
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "sdpo_full":
            return original_update_policy(self, data)

        # ---- SDPO-full path ----
        sd_cfg = _build_sdpo_config(data.meta_info.get("sdpo_config", None))
        if sd_cfg is None:
            sd_cfg = _build_sdpo_config(self.config.get("self_distillation", None))
        if sd_cfg is None:
            logger.warning("sdpo_full loss_mode but no self_distillation config; falling back to vanilla")
            return original_update_policy(self, data)

        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses", "response_mask", "input_ids", "attention_mask",
            "position_ids", "old_log_probs", "advantages",
        ]
        for k in [
            "teacher_input_ids", "teacher_attention_mask",
            "teacher_position_ids", "self_distillation_mask",
        ]:
            if k in data.batch.keys():
                select_keys.append(k)
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        non_tensor_select_keys = []
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("multi_modal_inputs")

        data = data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=non_tensor_select_keys,
        )

        # SDPO-full: decide which distillation flavor we're in.
        #   full_logit_distillation + distillation_topk is None → full vocab KL
        #   full_logit_distillation + distillation_topk int     → top-k approx
        #   full_logit_distillation=False                       → reverse-KL (sdpo-light)
        if not sd_cfg.full_logit_distillation:
            raise ValueError(
                "loss_mode='sdpo_full' requires full_logit_distillation=True. "
                "Use loss_mode='sdpo' for the token-level reverse-KL variant."
            )
        return_all_logps = sd_cfg.distillation_topk is None
        distill_topk = sd_cfg.distillation_topk

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {"actor/pg_loss": 0.0, "actor/kl_loss": 0.0}

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    from verl.workers.actor.dp_actor import prepare_dynamic_batch
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}

                    model_inputs = {
                        **micro_batch.batch,
                        **micro_batch.non_tensor_batch,
                        "pad_token_id": pad_token_id,
                    }
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or entropy_coeff != 0

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = (
                            response_mask.shape[0] / self.config.ppo_mini_batch_size
                        )
                    else:
                        loss_scale_factor = 1 / gradient_accumulation

                    # --- Student forward ---
                    student_outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                    )
                    log_prob = student_outputs["log_probs"]
                    entropy = student_outputs["entropys"] if calculate_entropy else None
                    student_all_logps = student_outputs.get("all_logps")
                    student_topk_logps = student_outputs.get("topk_logps")
                    student_topk_indices = student_outputs.get("topk_indices")

                    if on_policy:
                        old_log_prob = log_prob.detach()

                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    self_distillation_mask = model_inputs.get("self_distillation_mask", None)
                    has_teacher_data = "teacher_input_ids" in model_inputs

                    if has_teacher_data and self_distillation_mask is not None:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        teacher_model = getattr(self, "teacher_module", None) or self.actor_module

                        with torch.no_grad():
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps")
                        teacher_topk_logps = teacher_outputs.get("topk_logps")

                        pg_loss, pg_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=sd_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                        )
                        pg_metrics["self_distillation/empty_target_batch"] = (
                            self_distillation_mask.sum().item() == 0
                        )
                    else:
                        # No teacher data this batch: fall back to vanilla policy loss
                        policy_loss_fn = get_policy_loss_fn("vanilla")
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=model_inputs["advantages"],
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )

                    micro_batch_metrics.update(pg_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor

                    loss = policy_loss * loss_scale_factor
                    loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()

        # EMA teacher update (no-op if teacher_regularization != "ema" or update_rate == 0)
        if hasattr(self, "_update_teacher"):
            self._update_teacher()

        return metrics

    patched_update_policy._sdpo_full_patched = True
    DataParallelPPOActor.update_policy = patched_update_policy
    logger.info("SDPO-full: patched DataParallelPPOActor.update_policy (loss_mode='sdpo_full')")
