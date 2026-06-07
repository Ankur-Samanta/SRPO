"""Analyze SRPO self-localization stats from wandb.

Uses the actual logged keys (actor/srpo_*, critic/rewards/*, etc.) to answer:
  1. Is self-localization working as intended?
  2. Are we seeing successful corrections?
  3. Would 2x4 shared-prefix groups be better than the current 4+4 hybrid?

Usage:
  python training/scripts/analyze_srpo_stats.py
  python training/scripts/analyze_srpo_stats.py --run numina_oly_olmo7b_srpo_rand
  python training/scripts/analyze_srpo_stats.py --compare
"""

import argparse
import sys

import numpy as np

try:
    import wandb
except ImportError:
    sys.exit("wandb not installed. pip install wandb")

WANDB_PROJECT = "srpo"
WANDB_ENTITY = None

METRICS = [
    # SRPO-specific
    "actor/srpo_correction_frac",     # fraction of 8 slots that are corrections (>0 → G2 triggered)
    "actor/srpo_suffix_tokens_mean",  # mean tokens in the learnable suffix (small → prefix reuse; 0 → no G2)
    "actor/reset_advantage_mean",
    "actor/reset_advantage_std",
    # Reward / signal quality
    "critic/rewards/mean",
    "critic/rewards/group_correct_mean",   # mean correct per group of 8
    "critic/rewards/group_std_mean",       # mean within-group reward std (signal richness)
    "critic/rewards/degenerate_group_frac",# frac groups with all-same rewards (zero signal)
    "critic/pass_at_k/1",
    "critic/pass_at_k/4",
    "critic/pass_at_k/8",
    # Response length (for normalising suffix tokens)
    "response_length/mean",
    # Val performance
    "val-core/numinamath_olympiads/reward/mean@1",
    "val-aux/numinamath_olympiads/math_correct/mean@1",
]


def fetch_history(run_name):
    api = wandb.Api()
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    runs = [r for r in api.runs(path) if r.name == run_name]
    if not runs:
        runs = [r for r in api.runs(path)
                if r.config.get("trainer", {}).get("experiment_name") == run_name]
    if not runs:
        sys.exit(f"No run '{run_name}' in project '{WANDB_PROJECT}'")
    run = sorted(runs, key=lambda r: r.created_at)[-1]
    print(f"  {run_name}  (id={run.id}, state={run.state}, steps={run.summary.get('_step','?')})")
    history = list(run.scan_history(keys=METRICS))
    return run, history


def series(history, key):
    return [row[key] for row in history if key in row and row[key] is not None]


def fmt(arr, pct=False):
    if len(arr) == 0:
        return "n/a"
    a = np.array(arr)
    s = 100 if pct else 1
    u = "%" if pct else ""
    return f"mean={a.mean()*s:.1f}{u}  final={a[-1]*s:.1f}{u}  [min={a.min()*s:.1f}{u}, max={a.max()*s:.1f}{u}]  n={len(a)}"


def analyze_run(run_name):
    run, hist = fetch_history(run_name)

    print(f"\n{'='*68}")
    print(f"  {run_name}")
    print(f"{'='*68}\n")

    # ------------------------------------------------------------------
    # Derived: infer Group-2 trigger rate from srpo_correction_frac
    # In 4+1+3 design: when G2 triggered → 3/8 slots are corrections
    # → srpo_correction_frac ≈ 0.375 × triggered_rate
    # ------------------------------------------------------------------
    corr_frac = series(hist, "actor/srpo_correction_frac")
    suffix_tok = series(hist, "actor/srpo_suffix_tokens_mean")
    resp_len   = series(hist, "response_length/mean")

    triggered_rate_est = None
    if corr_frac:
        triggered_rate_est = np.array(corr_frac) / (3/8)

    print("── 1. SELF-LOCALIZATION HEALTH ─────────────────────────────────")
    print(f"  srpo_correction_frac  : {fmt(corr_frac)}")
    if triggered_rate_est is not None:
        print(f"  → implied trigger rate: {fmt(list(triggered_rate_est), pct=True)}")
        print(f"    (correction_frac / 0.375; when G2 fires, 3 of 8 slots are corrections)")

    if suffix_tok and resp_len:
        # suffix_tokens_mean is averaged over ALL 8 slots; fresh slots have suffix_start=0
        # so the suffix IS the full response. Corrections have suffix_start>0 → shorter suffix.
        # Proxy for prefix reuse: for correction slots specifically,
        # avg suffix tokens ≈ (total suffix * 8) / 3 - contribution of fresh slots
        # Simpler: compare suffix_tok to resp_len — if they're similar, corrections start near token 0
        mean_suffix = np.mean(suffix_tok)
        mean_resp   = np.mean(resp_len)
        # The 4 fresh slots always have suffix = full response; the 3 correction slots have suffix < resp
        # slot-weighted: mean_suffix ≈ (4/8)*resp + (3/8)*correction_suffix + (1/8)*0 [parent=0]
        # → correction_suffix ≈ (mean_suffix - 0.5*resp_len) / 0.375
        tr_mean = float(triggered_rate_est.mean()) if triggered_rate_est is not None else 0.375
        # When G2 not triggered all 8 are fresh → suffix = resp; weight correction slots by trigger rate
        # Approximate: correction_suffix = (mean_suffix - (1 - 0.375*tr_mean)*mean_resp) / (0.375*tr_mean)
        denom = 0.375 * tr_mean
        if denom > 0.01:
            correction_suffix_est = (mean_suffix - (1 - denom) * mean_resp) / denom
            prefix_frac = 1 - correction_suffix_est / mean_resp
            print(f"\n  srpo_suffix_tokens_mean : {fmt(suffix_tok)}")
            print(f"  response_length/mean    : {fmt(resp_len)}")
            print(f"  → est. correction suffix tokens : {correction_suffix_est:.0f} of {mean_resp:.0f} total")
            print(f"  → est. prefix reuse frac        : {prefix_frac*100:.0f}%  "
                  f"(fraction of response tokens kept as prefix)")
            if prefix_frac < 0.1:
                print(f"  ⚠  Very little prefix reuse — localization is pointing near step 1.")
                print(f"     Corrections are essentially fresh regenerations from a failed chain.")
            elif prefix_frac > 0.3:
                print(f"  ✓  Non-trivial prefix reuse ({prefix_frac*100:.0f}%) — localization finding mid-chain errors.")
            else:
                print(f"  ~  Modest prefix reuse ({prefix_frac*100:.0f}%).")
    elif suffix_tok:
        print(f"\n  srpo_suffix_tokens_mean : {fmt(suffix_tok)}")

    # ------------------------------------------------------------------
    # 2. Signal quality / correction success
    # ------------------------------------------------------------------
    print("\n── 2. TRAINING SIGNAL QUALITY ──────────────────────────────────")
    print(f"  rewards/mean          : {fmt(series(hist, 'critic/rewards/mean'))}")
    print(f"  group_correct_mean    : {fmt(series(hist, 'critic/rewards/group_correct_mean'))}")
    print(f"    (mean # correct out of 8 per prompt)")
    print(f"  group_std_mean        : {fmt(series(hist, 'critic/rewards/group_std_mean'))}")
    print(f"    (within-group reward std; higher = more signal)")
    print(f"  degenerate_group_frac : {fmt(series(hist, 'critic/rewards/degenerate_group_frac'), pct=True)}")
    print(f"    (all-same reward = zero gradient; lower is better)")
    print(f"  reset_advantage_mean   : {fmt(series(hist, 'actor/reset_advantage_mean'))}")
    print(f"  reset_advantage_std    : {fmt(series(hist, 'actor/reset_advantage_std'))}")
    print(f"  pass@1                : {fmt(series(hist, 'critic/pass_at_k/1'), pct=True)}")
    print(f"  pass@4                : {fmt(series(hist, 'critic/pass_at_k/4'), pct=True)}")
    print(f"  pass@8                : {fmt(series(hist, 'critic/pass_at_k/8'), pct=True)}")

    gcm = series(hist, "critic/rewards/group_correct_mean")
    degen = series(hist, "critic/rewards/degenerate_group_frac")
    gsm = series(hist, "critic/rewards/group_std_mean")
    reward_mean = series(hist, "critic/rewards/mean")
    adv_mean = series(hist, "actor/reset_advantage_mean")
    pa1 = series(hist, "critic/pass_at_k/1")
    pa8 = series(hist, "critic/pass_at_k/8")

    if pa1 and pa8:
        gap = np.array(pa8) - np.array(pa1[:len(pa8)])
        print(f"\n  pass@8 - pass@1 gap     : {fmt(list(gap), pct=True)}")
        print(f"    (gap > 0 means the model can solve it given more attempts — correction opportunity)")

    if degen:
        d_final = degen[-1]
        print(f"\n  Interpretation:")
        if d_final > 0.4:
            print(f"  ⚠  degenerate_group_frac={d_final*100:.0f}% — many prompts have all-same rewards.")
            print(f"     Either all 8 rollouts solve it (easy) or all fail (too hard). Low learning signal.")
        else:
            print(f"  ✓  degenerate_group_frac={d_final*100:.0f}% — most prompts have mixed outcomes. Good signal.")

    # ------------------------------------------------------------------
    # 3. 4+4 hybrid vs 2×4 shared-prefix
    # ------------------------------------------------------------------
    print("\n── 3. HYBRID 4+4 vs 2×4 SHARED-PREFIX ─────────────────────────")

    if triggered_rate_est is not None and pa1 and pa8:
        tr = float(triggered_rate_est.mean())
        fa = float(np.mean(pa1))       # fresh chain accuracy ≈ pass@1
        pa8_val = float(np.mean(pa8))  # upper bound on what corrections can achieve

        # Estimate correction accuracy:
        # Group 2 has 1 failed parent + 3 corrections from a localized reset.
        # If corrections were independent Bernoulli(p_corr), and correction_success_rate
        # is the probability ≥1 correction succeeds:
        # We can bound p_corr from group_correct_mean and correction_frac.
        # Simple estimate: corrections have accuracy ≈ pass@1 * (1 + prefix_boost)
        # Without direct per-group data, use pass@1 as lower bound, pass@8 as upper.

        print(f"  Estimated trigger rate  : {tr*100:.0f}%")
        print(f"  pass@1 (≈ fresh acc)    : {fa*100:.0f}%")
        print(f"  pass@8 (≈ upper bound)  : {pa8_val*100:.0f}%")

        # Mixed rate analysis
        # Group 1 (4 fresh Bernoulli(fa)): P(mixed) = 1 - fa^4 - (1-fa)^4
        p_g1_mixed = 1 - fa**4 - (1-fa)**4
        # Group 2 (4 slots: 1 forced-fail + 3 Bernoulli(p_corr)):
        # parent reward = 0. corrections reward ~ p_corr.
        # G2 is degenerate only if all 3 corrections also fail, or all 3 succeed.
        # Use pass@1 as conservative p_corr, pass@8 as optimistic.
        for label, p_corr in [("conservative (p_corr=pass@1)", fa),
                               ("optimistic   (p_corr=pass@8)", pa8_val)]:
            # G2 rewards: [0, X1, X2, X3] where Xi ~ Bernoulli(p_corr)
            # G2 mixed if not all [0,0,0,0] and not all [0,1,1,1] (not possible since parent=0)
            # Actually G2 degenerate if all 4 are equal: either all 0 (p_corr=0 for corrections)
            # p_g2_all_wrong (all 4 zero) = (1-p_corr)^3
            p_g2_degen = (1-p_corr)**3   # parent always 0; all corrections fail
            p_g2_mixed = 1 - p_g2_degen  # at least one correction succeeds

            # Effective G2 signal rate = triggered AND G2 mixed
            eff_g2 = tr * p_g2_mixed

            # Under 2x4: need 2 failures, each with 3 corrections
            # P(both triggered) ≈ tr^2 (independent)
            p_2x4_both = tr ** 2
            eff_2x4 = p_2x4_both * p_g2_mixed ** 2

            print(f"\n  [{label}]")
            print(f"    Current 4+4 hybrid:")
            print(f"      G1 signal rate (mixed 4 fresh)  : {p_g1_mixed*100:.0f}%  (unconditional)")
            print(f"      G2 trigger rate                 : {tr*100:.0f}%")
            print(f"      G2 mixed | triggered            : {p_g2_mixed*100:.0f}%")
            print(f"      G2 effective signal rate        : {eff_g2*100:.0f}%  (triggered AND mixed)")
            print(f"      Both G1+G2 active               : {tr * p_g1_mixed * p_g2_mixed*100:.0f}%")
            print(f"    Hypothetical 2×4 shared-prefix:")
            print(f"      Need 2 failures → P(both triggered) ≈ {p_2x4_both*100:.0f}%")
            print(f"      Both groups mixed               : {eff_2x4*100:.0f}%")
            print(f"      vs current G2-only signal       : {eff_g2*100:.0f}%")
            if p_2x4_both < tr * 0.7:
                print(f"      ⚠  2×4 cuts effective trigger rate significantly.")
                print(f"         You'd lose the reliable G1 fresh signal in exchange.")
            else:
                print(f"      ~  2×4 may be viable if correction accuracy >> fresh accuracy.")

    # ------------------------------------------------------------------
    # Val performance
    # ------------------------------------------------------------------
    print("\n── 4. VALIDATION PERFORMANCE ───────────────────────────────────")
    val_reward = series(hist, "val-core/numinamath_olympiads/reward/mean@1")
    val_correct = series(hist, "val-aux/numinamath_olympiads/math_correct/mean@1")
    print(f"  val reward@1  : {fmt(val_reward)}")
    print(f"  val correct@1 : {fmt(val_correct, pct=True)}")
    print()

    return {
        "triggered_rate": float(triggered_rate_est.mean()) if triggered_rate_est is not None else None,
        "corr_frac": float(np.mean(corr_frac)) if corr_frac else None,
        "suffix_tok": float(np.mean(suffix_tok)) if suffix_tok else None,
        "resp_len": float(np.mean(resp_len)) if resp_len else None,
        "reward_mean": float(np.mean(reward_mean)) if reward_mean else None,
        "adv_mean": float(np.mean(adv_mean)) if adv_mean else None,
        "degen": float(np.mean(degen)) if degen else None,
        "group_std": float(np.mean(gsm)) if gsm else None,
        "pa1": float(np.mean(pa1)) if pa1 else None,
        "pa8": float(np.mean(pa8)) if pa8 else None,
        "val_correct": float(val_correct[-1]) if val_correct else None,
    }


def compare(run_a, run_b):
    print(f"\nComparing runs:")
    stats_a = analyze_run(run_a)
    stats_b = analyze_run(run_b)

    print(f"\n{'='*68}")
    print(f"  COMPARISON: {run_a}  vs  {run_b}")
    print(f"{'='*68}")
    fields = [
        ("triggered_rate",  True,  "Group-2 trigger rate"),
        ("corr_frac",       True,  "correction_frac"),
        ("suffix_tok",      False, "mean suffix tokens"),
        ("resp_len",        False, "mean response length"),
        ("reward_mean",      False, "rewards/mean"),
        ("adv_mean",        False, "reset_advantage_mean"),
        ("degen",           True,  "degenerate_group_frac"),
        ("group_std",       False, "group_std_mean"),
        ("pa1",             True,  "pass@1"),
        ("pa8",             True,  "pass@8"),
        ("val_correct",     True,  "val correct@1"),
    ]
    for key, pct, label in fields:
        a = stats_a.get(key)
        b = stats_b.get(key)
        if a is None and b is None:
            continue
        s = 100 if pct else 1
        u = "%" if pct else ""
        a_s = f"{a*s:.1f}{u}" if a is not None else "n/a"
        b_s = f"{b*s:.1f}{u}" if b is not None else "n/a"
        delta = ""
        if a is not None and b is not None:
            d = (b - a) * s
            delta = f"  (Δ={d:+.1f}{u})"
        print(f"  {label:<32} {run_a.split('_')[-1]:<8} {a_s:<10} {run_b.split('_')[-1]:<8} {b_s:<10}{delta}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="numina_oly_olmo7b_srpo")
    global WANDB_PROJECT, WANDB_ENTITY
    parser.add_argument("--project", default="srpo")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Compare self-loc vs rand for the default model")
    parser.add_argument("--model", default="olmo7b", help="model prefix for --compare")
    args = parser.parse_args()
    WANDB_PROJECT = args.project
    WANDB_ENTITY = args.entity

    if args.compare:
        compare(
            f"numina_oly_{args.model}_srpo",
            f"numina_oly_{args.model}_srpo_rand",
        )
    else:
        analyze_run(args.run)


if __name__ == "__main__":
    main()
