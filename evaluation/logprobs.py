"""Log probability computation utilities adapted from TRL."""

import torch
import torch.nn.functional as F
from typing import List
from transformers import PreTrainedModel, PreTrainedTokenizer


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient log_softmax -> gather operation. Copied from TRL.

    Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py

    This function is equivalent to:
        logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

    Args:
        logits: Logits tensor of shape (..., num_classes)
        index: Index tensor of shape (...), specifying positions to gather

    Returns:
        Gathered log probabilities with the same shape as index
    """
    if logits.dtype in [torch.float32, torch.float64]:
        # More memory efficient approach for float32/64
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # Compute logsumexp along the vocab dimension (last dim) for all positions at once
        logsumexp_values = torch.logsumexp(logits, dim=-1)  # [batch_size, seq_len]
        per_token_logps = selected_logits - logsumexp_values
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def compute_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    completions: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute log probabilities of completions given prompts.

    This is a simple loop-based implementation suitable for small batches.
    For larger batches, use compute_logprobs_from_batch with proper batching.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        completions: List of completion strings
        device: Device to run on

    Returns:
        logprobs: Sum of log probs for each completion [batch_size]
    """
    batch_logprobs = []

    for prompt, completion in zip(prompts, completions):
        # Tokenize prompt and completion separately to avoid BPE issues
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        completion_ids = tokenizer.encode(completion, add_special_tokens=False, return_tensors="pt")

        prompt_len = prompt_ids.shape[1]
        completion_len = completion_ids.shape[1]

        # Handle empty completion - log probability of empty sequence is undefined
        # Return -inf to signal this is an invalid/degenerate case
        if completion_len == 0:
            batch_logprobs.append(torch.tensor(float('-inf'), device=device, dtype=torch.float32))
            continue

        # Concatenate prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)
        seq_len = input_ids.shape[1]

        # Create attention mask (all 1s since no padding)
        attention_mask = torch.ones_like(input_ids)

        # Create loss mask: 0 for prompt, 1 for completion
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[:, prompt_len:] = 1

        # Forward pass
        if model.training:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        # Compute log probs for completion tokens only
        # For autoregressive LM: logits[t] predicts token[t+1]
        # So we want logits[prompt_len-1:seq_len-1] to predict tokens[prompt_len:seq_len]
        shift_logits = logits[:, prompt_len - 1:-1, :]  # [1, completion_len, vocab]
        shift_labels = input_ids[:, prompt_len:]  # [1, completion_len]

        # Use selective_log_softmax for memory efficiency
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)

        # Sum to get sequence-level log prob
        total_logprob = per_token_logps.sum()
        batch_logprobs.append(total_logprob)

    return torch.stack(batch_logprobs)


def compute_logprobs_batched(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Batched log prob computation with masking. Uses -100 convention for ignored tokens.

    Args:
        model: The language model
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        labels: [batch_size, seq_len] with -100 for tokens to ignore

    Returns:
        logprobs: [batch_size] sum of log probs per example
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Use selective_log_softmax
    # Need to handle -100 labels by temporarily replacing them
    valid_mask = (shift_labels != -100)
    shift_labels_safe = shift_labels.clone()
    shift_labels_safe[~valid_mask] = 0

    per_token_logps = selective_log_softmax(shift_logits, shift_labels_safe)

    # Mask out ignored tokens
    per_token_logps = per_token_logps * valid_mask.float()

    # Sum per example
    return per_token_logps.sum(dim=-1)
