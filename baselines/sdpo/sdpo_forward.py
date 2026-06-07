"""SDPO-full: patched _forward_micro_batch with full-logit distillation support.

Verbatim port of lasgroup/SDPO verl/workers/actor/dp_actor.py::_forward_micro_batch,
which extends base verl's forward with four new parameters:

    return_all_logps : bool       -- return log_softmax over full vocab (bsz, T, V)
    distill_topk     : int | None -- return top-k log-probs (bsz, T, K)
    topk_indices     : tensor     -- teacher reuses student's indices (bsz, T, K)
    module           : nn.Module  -- run forward through an alternate module (teacher)

Structure matches base verl's _forward_micro_batch exactly; only SDPO additions are
spliced in.  The function is monkey-patched onto DataParallelPPOActor when the
baselines.sdpo package is imported (see __init__.py).

Reference:
    https://github.com/lasgroup/SDPO/blob/main/verl/workers/actor/dp_actor.py
"""

from typing import Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.torch_functional import logprobs_from_logits


def _chunked_logsumexp(x: torch.Tensor, chunk_size: int = 8192, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    # Drop-in replacement for torch.logsumexp(x, dim=dim, keepdim=keepdim) that
    # avoids materializing the full (..., V).exp() intermediate. Mathematically
    # identical via logsumexp(x) = m + log(sum(exp(x - m))). Used to dodge OOM
    # on Qwen-14B where V=151936 makes the in-place exp peak the budget.
    n = x.size(dim)
    if n <= chunk_size:
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)
    m = x.amax(dim=dim, keepdim=True)
    s = torch.zeros_like(m)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = x.narrow(dim, start, end - start)
        s = s + (chunk - m).exp().sum(dim=dim, keepdim=True)
    out = m + s.log()
    return out if keepdim else out.squeeze(dim)
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)


def patched_forward_micro_batch(
    self,
    micro_batch: dict[str, torch.Tensor],
    temperature: float,
    calculate_entropy: bool = False,
    return_all_logps: bool = False,
    distill_topk: Optional[int] = None,
    topk_indices: Optional[torch.Tensor] = None,
    module: Optional[torch.nn.Module] = None,
) -> dict[str, torch.Tensor]:
    """
    Returns:
        dict[str, torch.Tensor]:
            log_probs: (bs, response_len)
            if calculate_entropy is True:
                entropys: (bs, response_len)
            if calculate_sum_pi_squared is True:
                sum_pi_squared: (bs, response_len)
            if return_all_logps and not use_topk:
                all_logps: (bs, response_len, vocab)
            if distill_topk or topk_indices is set:
                topk_logps: (bs, response_len, k)
                topk_indices: (bs, response_len, k)  -- only when student (caller didn't pass)
    """
    calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
    sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)

    use_topk = distill_topk is not None or topk_indices is not None
    compute_all_logps = return_all_logps and not use_topk
    return_topk_indices = use_topk and topk_indices is None
    if (return_all_logps or use_topk) and self.use_fused_kernels:
        raise ValueError("SDPO logit distillation requires disabling fused kernels.")

    model = module if module is not None else self.actor_module

    # PrefixGrouper path for shared-prefix optimization
    if self.use_prefix_grouper:
        can_use_pg = (
            not self.use_remove_padding
            and not self.use_ulysses_sp
            and not self.use_fused_kernels
            and not self.use_dynamic_bsz
            and not return_all_logps
            and not use_topk
        )
        if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
            from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

            return forward_micro_batch_with_prefix_grouper(
                micro_batch=micro_batch,
                model=model,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
                device_name=self.device_name,
                param_dtype=self.param_dtype,
                use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
            )

    response_length = micro_batch["responses"].size(-1)
    multi_modal_inputs = {}
    if "multi_modal_inputs" in micro_batch.keys():
        from verl.utils.model import extract_multi_modal_inputs

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

    with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        entropy = None
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)

        if self.use_remove_padding:
            input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            is_mask_all_zero = attention_mask.sum() == 0
            if is_mask_all_zero:
                input_ids_rmpad = torch.zeros(
                    (1, self.ulysses_sequence_parallel_size),
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                if position_ids.dim() == 3:
                    position_ids_rmpad = torch.zeros(
                        (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                        device=position_ids.device,
                        dtype=position_ids.dtype,
                    )
                else:
                    position_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=position_ids.device,
                        dtype=position_ids.dtype,
                    )

            if "image_bound" in multi_modal_inputs:
                from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                    input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                )

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            if self.use_ulysses_sp:
                is_vlm_model = hasattr(
                    getattr(model, "module", model).config, "vision_config"
                )
                if is_vlm_model:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                else:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    position_ids_rmpad=None,
                    sp_size=self.ulysses_sequence_parallel_size,
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

            extra_args = {}
            if self.use_fused_kernels:
                extra_args["temperature"] = temperature
                extra_args["return_dict"] = True

            output = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                **extra_args,
            )

            if self.use_fused_kernels:
                log_probs = output.log_probs.squeeze(0)
                entropy_rmpad = output.entropy.squeeze(0)
                all_logps_rmpad = None
                topk_logps_rmpad = None
                topk_indices_rmpad = None
            else:
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)

                # --- SDPO: full-logit branch (exclusive with topk) ---
                all_logps_rmpad = (
                    torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None
                )

                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                if calculate_entropy:
                    entropy_rmpad = (
                        self.compute_entropy_from_logits(logits_rmpad)
                        if not self.config.entropy_checkpointing
                        else torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )
                    )

                # --- SDPO: top-k branch ---
                topk_logps_rmpad = None
                topk_indices_rmpad = None
                if use_topk:
                    if topk_indices is None:
                        topk = min(distill_topk, logits_rmpad.shape[-1])
                        topk_logits_rmpad, topk_indices_rmpad = torch.topk(
                            logits_rmpad, topk, dim=-1
                        )
                    else:
                        # Teacher path: reuse student's (bsz, response_len, k) indices.
                        # Build (bsz, seqlen, k) with zeros in the prompt region, then unpad
                        # using teacher's `indices`.
                        topk = topk_indices.size(-1)
                        full_topk_indices = torch.zeros(
                            batch_size,
                            seqlen,
                            topk,
                            device=topk_indices.device,
                            dtype=topk_indices.dtype,
                        )
                        full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                        topk_indices_rmpad = index_first_axis(
                            rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                        )
                        if self.use_ulysses_sp:
                            topk_indices_rmpad = slice_input_tensor(
                                topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                            ).squeeze(0)
                        topk_logits_rmpad = torch.gather(
                            logits_rmpad, dim=-1, index=topk_indices_rmpad
                        )
                    logsumexp_rmpad = _chunked_logsumexp(logits_rmpad, chunk_size=8192, dim=-1, keepdim=True)
                    topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                if calculate_sum_pi_squared:
                    sum_pi_squared_rmpad = (
                        self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                        if not sum_pi_squared_checkpointing
                        else torch.utils.checkpoint.checkpoint(
                            self.calculate_sum_pi_squared_from_logits, logits_rmpad
                        )
                    )

            # gather if sp > 1
            if self.use_ulysses_sp:
                log_probs = gather_outputs_and_unpad(
                    log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size,
                )
                if calculate_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size,
                    )
                if compute_all_logps:
                    all_logps_rmpad = gather_outputs_and_unpad(
                        all_logps_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size,
                    )
                if use_topk:
                    topk_logps_rmpad = gather_outputs_and_unpad(
                        topk_logps_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size,
                    )
                    if return_topk_indices:
                        topk_indices_rmpad = gather_outputs_and_unpad(
                            topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size,
                        )
                if calculate_sum_pi_squared:
                    sum_pi_squared_rmpad = gather_outputs_and_unpad(
                        sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size,
                    )

            if is_mask_all_zero:
                log_probs = log_probs[:0]
                if calculate_entropy:
                    entropy_rmpad = entropy_rmpad[:0]
                if compute_all_logps:
                    all_logps_rmpad = all_logps_rmpad[:0]
                if use_topk:
                    topk_logps_rmpad = topk_logps_rmpad[:0]
                    if return_topk_indices:
                        topk_indices_rmpad = topk_indices_rmpad[:0]

            # pad back to (bsz, seqlen, *)
            if calculate_entropy:
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1),
                    indices=indices, batch=batch_size, seqlen=seqlen,
                )
            if calculate_sum_pi_squared:
                full_sum_pi_squared = pad_input(
                    hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                    indices=indices, batch=batch_size, seqlen=seqlen,
                )
            if compute_all_logps:
                full_all_logps = pad_input(
                    hidden_states=all_logps_rmpad,
                    indices=indices, batch=batch_size, seqlen=seqlen,
                )
            if use_topk:
                full_topk_logps = pad_input(
                    hidden_states=topk_logps_rmpad,
                    indices=indices, batch=batch_size, seqlen=seqlen,
                )
                if return_topk_indices:
                    full_topk_indices = pad_input(
                        hidden_states=topk_indices_rmpad,
                        indices=indices, batch=batch_size, seqlen=seqlen,
                    )
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1),
                indices=indices, batch=batch_size, seqlen=seqlen,
            )

            # slice to response portion
            if calculate_entropy:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
            if calculate_sum_pi_squared:
                sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
            if compute_all_logps:
                all_logps = full_all_logps[:, -response_length - 1 : -1, :]
            if use_topk:
                topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                if return_topk_indices:
                    topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

        else:  # not using rmpad
            extra_args = {}
            if self.use_fused_kernels:
                extra_args["temperature"] = temperature
                extra_args["return_dict"] = True

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                **extra_args,
            )

            if self.use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]
            else:
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if compute_all_logps:
                    all_logps = torch.log_softmax(logits, dim=-1)
                if use_topk:
                    if topk_indices is None:
                        topk = min(distill_topk, logits.size(-1))
                        topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                    else:
                        topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                    logsumexp = _chunked_logsumexp(logits, chunk_size=8192, dim=-1, keepdim=True)
                    topk_logps = topk_logits - logsumexp
                if calculate_entropy:
                    if not self.config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                if calculate_sum_pi_squared:
                    sum_pi_squared = (
                        self.calculate_sum_pi_squared_from_logits(logits)
                        if not sum_pi_squared_checkpointing
                        else torch.utils.checkpoint.checkpoint(
                            self.calculate_sum_pi_squared_from_logits, logits
                        )
                    )

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropy
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        if compute_all_logps:
            outputs["all_logps"] = all_logps
        if use_topk:
            outputs["topk_logps"] = topk_logps
            if return_topk_indices:
                outputs["topk_indices"] = topk_indices
        return outputs


def patch_forward_micro_batch():
    """Monkey-patch DataParallelPPOActor._forward_micro_batch with SDPO-full version."""
    import logging

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    logger = logging.getLogger(__name__)
    original = DataParallelPPOActor._forward_micro_batch
    if getattr(original, "_sdpo_full_patched", False):
        return

    patched_forward_micro_batch._sdpo_full_patched = True
    DataParallelPPOActor._forward_micro_batch = patched_forward_micro_batch
    logger.info("SDPO-full: patched DataParallelPPOActor._forward_micro_batch")
