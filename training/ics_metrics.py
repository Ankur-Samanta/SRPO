"""Aggregate ICS stats from non_tensor_batch into wandb-loggable metrics.

Works for any method that uses ThoughtICSAgentLoop to fill the rollout buffer
(thought_ics_agent, srpo_agent, etc.).

Injected via a monkey-patch of compute_data_metrics in ray_trainer's module
namespace — see __init__.py. No-op when ICS keys are absent (non-ICS runs).
"""

import numpy as np


def compute_ics_metrics(batch) -> dict:
    """Read ICS extra_fields from non_tensor_batch and return metric dict.

    In the batch, n=rollout_n trajectories exist per prompt:
    - ics_triggered / ics_iterations / ics_corrected: same value repeated n
      times per prompt (all slots share the prompt-level stat), so a plain
      mean across all batch elements equals the prompt-level mean.
    - ics_iter_oracle_correct: only slot 0 has a non-None list; slots 1..n-1
      are None. Filtering out None gives one list per prompt.
    """
    metrics = {}
    ntb = batch.non_tensor_batch

    if "ics_triggered" not in ntb:
        return metrics

    triggered = np.array([
        bool(x) if x is not None else False
        for x in ntb["ics_triggered"]
    ])
    metrics["ics/triggered_rate"] = float(triggered.mean())

    if "ics_iterations" in ntb:
        iterations = np.array([
            int(x) if x is not None else 0
            for x in ntb["ics_iterations"]
        ])
        metrics["ics/iterations_mean"] = float(iterations.mean())
        if triggered.any():
            metrics["ics/iterations_mean_when_triggered"] = float(
                iterations[triggered].mean()
            )

    if "ics_corrected" in ntb:
        corrected = np.array([
            bool(x) if x is not None else False
            for x in ntb["ics_corrected"]
        ])
        metrics["ics/corrected_rate"] = float(corrected.mean())
        if triggered.any():
            metrics["ics/correction_success_rate"] = float(
                corrected[triggered].mean()
            )

    if "ics_iter_oracle_correct" in ntb:
        # One list per prompt (slot 0 only); None for all other slots
        valid = [
            x for x in ntb["ics_iter_oracle_correct"]
            if x is not None and hasattr(x, "__len__") and len(x) > 0
        ]
        if valid:
            max_iter = max(len(v) for v in valid)
            for k in range(max_iter):
                vals = [float(v[k]) for v in valid if len(v) > k]
                if vals:
                    label = "fresh" if k == 0 else f"correction_{k}"
                    metrics[f"ics/iter_accuracy/{label}"] = float(np.mean(vals))
            # Overall accuracy across all iterations pooled
            all_vals = [float(x) for v in valid for x in v]
            metrics["ics/iter_accuracy/overall"] = float(np.mean(all_vals))

    # --- Localization metrics ---
    # ics_error_steps is a list per trajectory (same value repeated n times per
    # prompt). Flatten across all entries to get mean localized step.
    if "ics_error_steps" in ntb:
        all_steps = [
            step
            for entry in ntb["ics_error_steps"]
            if entry is not None and hasattr(entry, "__len__")
            for step in entry
        ]
        if all_steps:
            steps_arr = np.array(all_steps, dtype=float)
            metrics["ics/localization/mean_error_step"] = float(steps_arr.mean())
            # Fraction of localizations that pointed to step 1 (forces full regen)
            metrics["ics/localization/step1_rate"] = float((steps_arr == 1).mean())

        # Relative error position: error_step / num_thoughts of the source chain.
        # Fresh chain slots (suffix_start_idx == 0) are the chains being localized,
        # so their num_thoughts gives a clean denominator. We pair per-trajectory.
        if "num_thoughts" in ntb and "suffix_start_idx" in ntb:
            rel_positions = []
            for i, entry in enumerate(ntb["ics_error_steps"]):
                if (entry is None or not hasattr(entry, "__len__") or len(entry) == 0):
                    continue
                si = ntb["suffix_start_idx"][i]
                nt = ntb["num_thoughts"][i]
                if si is not None and int(si) == 0 and nt is not None and int(nt) > 0:
                    # fresh chain: use its num_thoughts as denominator
                    for step in entry:
                        rel_positions.append(float(step) / float(nt))
            if rel_positions:
                metrics["ics/localization/mean_relative_error_step"] = float(
                    np.mean(rel_positions)
                )

    # --- Fresh chain vs correction accuracy ---
    # Uses srpo_reward and suffix_start_idx which are set for all ICS runs.
    # suffix_start_idx == 0 → fresh chain (full response is suffix)
    # suffix_start_idx  > 0 → ICS correction (prefix was kept, only suffix is new)
    if "srpo_reward" in ntb and "suffix_start_idx" in ntb:
        rewards = np.array([
            float(x) if x is not None else 0.0
            for x in ntb["srpo_reward"]
        ])
        suffix_starts = np.array([
            int(x) if x is not None else 0
            for x in ntb["suffix_start_idx"]
        ])
        is_fresh = suffix_starts == 0
        is_correction = suffix_starts > 0

        if is_fresh.any():
            metrics["ics/accuracy/fresh_chains"] = float(rewards[is_fresh].mean())
        if is_correction.any():
            metrics["ics/accuracy/corrections"] = float(rewards[is_correction].mean())
        if is_fresh.any() and is_correction.any():
            metrics["ics/accuracy/correction_improvement"] = (
                metrics["ics/accuracy/corrections"] - metrics["ics/accuracy/fresh_chains"]
            )

        # --- Per-prompt correct count distribution ---
        # Infer rollout_n from ics_iter_oracle_correct (non-None only on slot 0,
        # one per prompt). This lets us group trajectories by prompt.
        n_total = len(rewards)
        n_prompts = sum(
            1 for x in ntb.get("ics_iter_oracle_correct", [])
            if x is not None
        )
        if n_prompts > 0 and n_total % n_prompts == 0:
            rollout_n = n_total // n_prompts
            per_prompt_rewards = rewards.reshape(n_prompts, rollout_n)
            correct_counts = per_prompt_rewards.sum(axis=1)  # (n_prompts,)

            metrics["training/prompts/mean_correct_per_prompt"] = float(correct_counts.mean())
            metrics["training/prompts/all_wrong_rate"] = float((correct_counts == 0).mean())
            metrics["training/prompts/all_correct_rate"] = float((correct_counts == rollout_n).mean())
            # Prompts with mixed outcomes — the only ones with actual learning signal
            metrics["training/prompts/mixed_rate"] = float(
                ((correct_counts > 0) & (correct_counts < rollout_n)).mean()
            )

    return metrics
