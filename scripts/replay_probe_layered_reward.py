"""Re-score existing probe rollouts with the layered retro_score.

Loads outputs/probe/retrosynthesis_qwen14b_k8_t07/records.json (the 50-prompt,
8-rollout probe at T=0.7) and re-scores every rollout under the new layered
reward. Reports reward distribution, pass@8 by reward threshold, and a
side-by-side comparison vs. the original exact-match outcomes.

Usage:
    python scripts/replay_probe_layered_reward.py \\
        [--records outputs/probe/retrosynthesis_qwen14b_k8_t07/records.json] \\
        [--templates ~/data/rlhf/retrosynthesis_uspto50k/templates.json]
"""

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


def load_scorer():
    spec = importlib.util.spec_from_file_location(
        "reward_scorers",
        str(Path(__file__).resolve().parents[1] / "training" / "reward_scorers.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--records",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "probe" / "retrosynthesis_qwen14b_k8_t07" / "records.json"),
    )
    ap.add_argument(
        "--templates",
        default=str(Path.home() / "data" / "rlhf" / "retrosynthesis_uspto50k" / "templates.json"),
    )
    args = ap.parse_args()

    scorer = load_scorer()
    recs = json.load(open(args.records))
    use_templates = Path(args.templates).exists()
    print(f"[replay] {len(recs)} prompts, {sum(len(r['rollouts']) for r in recs)} total rollouts")
    print(f"[replay] templates: {'using ' + args.templates if use_templates else 'NOT FOUND, Tanimoto-only fallback'}")

    bucket_names = [
        "0.0 (gated)",
        "(0.0, 0.10]",
        "(0.10, 0.20]",
        "(0.20, 0.30]",
        "(0.30, 0.50]",
        "0.8 template",
        "1.0 exact",
    ]
    bucket_counts = [0] * len(bucket_names)
    rewards_per_prompt: list[list[float]] = []

    n_pass1_old = n_pass8_old = 0   # original exact-match metrics
    n_any_pos_per_prompt = 0
    max_per_prompt: list[float] = []

    for r in recs:
        target = r.get("product")
        gold = r.get("gold_canonical") or r.get("reactants_gold")
        prompt_rewards = []
        any_pos = False
        for ro in r["rollouts"]:
            completion = ro["completion"]
            extra = {"product_smiles": target}
            if use_templates:
                extra["template_path"] = args.templates
            score = scorer.retro_score(completion, gold, extra)
            prompt_rewards.append(score)

            if score == 0.0:
                bucket_counts[0] += 1
            elif score == 1.0:
                bucket_counts[6] += 1
            elif abs(score - 0.8) < 1e-9:
                bucket_counts[5] += 1
            elif score <= 0.10:
                bucket_counts[1] += 1
            elif score <= 0.20:
                bucket_counts[2] += 1
            elif score <= 0.30:
                bucket_counts[3] += 1
            else:
                bucket_counts[4] += 1

            if score > 0:
                any_pos = True
            if ro.get("exact_match"):
                n_pass1_old += 1

        if r.get("any_exact"):
            n_pass8_old += 1
        if any_pos:
            n_any_pos_per_prompt += 1
        rewards_per_prompt.append(prompt_rewards)
        max_per_prompt.append(max(prompt_rewards))

    n_prompts = len(recs)
    total = sum(bucket_counts)
    print("\n=== reward bucket distribution (per rollout) ===")
    for name, count in zip(bucket_names, bucket_counts):
        pct = 100 * count / total if total else 0
        print(f"  {name:25s}: {count:4d} / {total}  ({pct:5.1f}%)")

    print("\n=== per-prompt summary ===")
    print(f"  prompts                  : {n_prompts}")
    print(f"  any positive reward (k=8): {n_any_pos_per_prompt} ({100*n_any_pos_per_prompt/n_prompts:.1f}%)")
    print(f"  any exact match (old)    : {n_pass8_old} ({100*n_pass8_old/n_prompts:.1f}%)")
    print(f"  pass@1 exact (old)       : {n_pass1_old} / {total}  ({100*n_pass1_old/total:.2f}%)")

    mean_max = sum(max_per_prompt) / n_prompts
    mean_avg = sum(sum(p)/len(p) for p in rewards_per_prompt) / n_prompts
    print(f"\n  mean of max-per-prompt   : {mean_max:.3f}  (best-rollout reward)")
    print(f"  mean of mean-per-prompt  : {mean_avg:.3f}  (avg-rollout reward; what GRPO sees)")

    # Show a few prompts where layered reward changes outcomes
    print("\n=== prompts where layered reward gives nonzero but exact-match was 0 ===")
    interesting = []
    for r, rewards in zip(recs, rewards_per_prompt):
        had_exact = any(ro.get("exact_match") for ro in r["rollouts"])
        max_r = max(rewards)
        if not had_exact and max_r > 0:
            interesting.append((max_r, r, rewards))
    interesting.sort(key=lambda x: -x[0])
    for max_r, r, rewards in interesting[:5]:
        print(f"\nprompt {r['id']} (class={r['rxn_class']}) max_reward={max_r:.2f}")
        print(f"  product : {r['product']}")
        print(f"  gold    : {r['gold_canonical']}")
        for ro, sc in zip(r["rollouts"], rewards):
            if sc > 0:
                print(f"    [r={sc:.2f}] {ro['pred_canonical']}")


if __name__ == "__main__":
    main()
