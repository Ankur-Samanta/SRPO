# Credit-Assignment Analysis

End-to-end pipeline for quantifying and visualizing how gradient signal
distributes across thoughts during SCGRPO / TGRPO training. Purpose: test the
hypothesis that SCGRPO's contrastive rollouts (shared ICS prefixes + divergent
suffixes) produce more differentiated per-thought credit than TGRPO.

All instrumentation is **opt-in via env vars** and **cannot break training**
(gated, wrapped in try/except, empty-dict / no-op when disabled).

## Three instrumentation hooks

| Hook | File | Fires at | Output |
|---|---|---|---|
| Branch structure dump | `training/branch_logger.py` | ICS coordinator (slot 0), once per prompt | JSONL: `{question, ics_stats, rollouts:[{slot, decoded_thoughts, thought_boundaries, segment_lengths, segment_fp, oracle_correct, ...}]}` |
| Credit aggregate metrics | `training/credit_metrics.py` | Policy loss, every microbatch | Scalars in wandb: `credit/thought_loss_gini`, `credit/top10pct_mass`, `credit/thought_loss_abs_mean/std`, `credit/n_valid_thoughts`, `credit/thoughts_per_seq_mean/std`, `credit/tokens_per_thought_mean`, `credit/adv_abs_mean`, `credit/adv_nonzero_frac` |
| Per-thought loss dump | `training/loss_dumper.py` | Policy loss, every Nth microbatch | JSONL: one record per rollout — `{fp, segment_lengths, thought_losses, advantage, microbatch_idx, row_idx}` |

The two JSONL dumps are joined offline by `segment_fp` (sha1 over the
token-count-per-thought tuple) to produce a per-edge gradient-overlay
visualization.

## Env vars

Configured automatically in `batch_scripts/submit_thought_scgrpo.sh` and
`batch_scripts/submit_thought_grpo.sh`. Values shown are the defaults.

```bash
# SCGRPO only
SCGRPO_BRANCH_DUMP_DIR=${SCPO_DIR}/logs/scgrpo_branches
SCGRPO_BRANCH_DUMP_EVERY=20        # dump every Nth prompt
SCGRPO_BRANCH_DUMP_MAX=0           # 0 = unlimited

# SCGRPO + TGRPO
SCGRPO_CREDIT_METRICS=1            # wandb aggregate metrics

SCGRPO_LOSS_DUMP_DIR=${SCPO_DIR}/logs/{scgrpo,tgrpo}_loss
SCGRPO_LOSS_DUMP_EVERY=50          # dump every Nth microbatch
SCGRPO_LOSS_DUMP_MAX=0
```

To **disable** any hook: unset its `_DIR` env var (or for credit metrics,
`SCGRPO_CREDIT_METRICS=0`). When disabled, the hooks are effectively free
(one env var check + early return).

## Live analysis: wandb metrics

During training, watch these in wandb to test the hypothesis:

| Metric | Expected for SCGRPO | Expected for TGRPO |
|---|---|---|
| `credit/thought_loss_gini` | higher (more concentrated) | lower |
| `credit/top10pct_mass` | higher | lower |
| `credit/thought_loss_abs_std` | higher (more differentiated) | lower |
| `credit/adv_abs_mean` | similar | similar |

If SCGRPO ≈ TGRPO on these, the "contrastive signal differentiates credit"
hypothesis is weak.

## Offline analysis: branching + gradient overlay

Run after a training job has produced some dumps. Located at
`scripts/visualize_branches.py`.

```bash
# Structural viz only (no loss overlay): edges colored by correctness,
# thickness ∝ rollout count passing through.
python scripts/visualize_branches.py logs/scgrpo_branches --latest --out out.png

# Gradient-overlay viz: requires a matched loss dump dir.
# Edges colored by sign (blue=reinforced, red=penalized), thickness ∝ |loss|.
python scripts/visualize_branches.py \
    logs/scgrpo_branches/branches_pid12345_TS.jsonl \
    --loss-dumps logs/scgrpo_loss \
    --index -1 \
    --out gradient_overlay.png
```

Output prints a join report:
```
loss join: matched=5/8, ambiguous=2, missed=1
```

- **matched**: 1:1 unique fingerprint match → edge loss definite
- **ambiguous**: multiple rollouts or loss records share the fp → edge drawn
  dashed; assignment is best-effort
- **missed**: no matching loss record found (throttling gap, or rollout got
  dropped before making it to training)

## Viz interpretation

For a group of 8 rollouts rendered as a trie:

- **Trunk (shared prefix) with thin edges** = signals cancelled under GRPO
  zero-sum advantages → the credit-assignment property working as designed
- **Thick blue suffix edges** = thoughts that made the trajectory correct →
  reinforced strongly
- **Thick red suffix edges** = thoughts that made the trajectory wrong →
  penalized strongly
- **Dashed edges** = fingerprint collision; reliability uncertain
- **Gray dotted edges** = no matching loss data (not part of the microbatch
  that got dumped, due to `SCGRPO_LOSS_DUMP_EVERY` throttling)

## Fingerprint collision notes

`segment_fp = sha1(tuple(token_count_per_thought))`. Two rollouts collide iff
they have the same number of thoughts with the same token count in each
corresponding position. In practice, collision rates observed should be <1%
for olympiad-style problems (long varied rollouts). If the join report shows
>5% ambiguous, interpret the viz with care or tighten the dump cadence.

## File layout

```
training/
├── branch_logger.py         # rollout-time dump
├── credit_metrics.py        # wandb aggregate metrics
├── loss_dumper.py           # loss-time per-thought dump
├── thought_grpo_loss.py     # hosts the loss hooks
└── thought_ics_agent_loop.py # hosts the rollout hook

scripts/
└── visualize_branches.py    # offline join + tree viz

logs/
├── scgrpo_branches/         # branch dumps (SCGRPO only — requires ICS)
├── scgrpo_loss/             # loss dumps from SCGRPO runs
└── tgrpo_loss/              # loss dumps from TGRPO runs
```
