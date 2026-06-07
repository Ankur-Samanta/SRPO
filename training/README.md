# `training/` — Thought-level GRPO on VERL

Implementation of the training methods in this repo, built on top of
[VERL](https://github.com/volcengine/verl)'s GRPO pipeline. This document is the
developer map for the `training/` package; for the method, results, and the
quickstart that drives these jobs through `batch_scripts/`, see the
[top-level README](../README.md).

Two training methods share this infrastructure:

1. **Thought-GRPO** — the baseline. N independent thought-by-thought chains per
   prompt, one outcome reward each, standard group-normalized GRPO advantage.
2. **SRPO / RRPO** — the reset methods. Under a fixed rollout budget, split the N
   slots into a *base* group (fresh i.i.d. chains) and a *reset* group
   (counterfactual continuations resampled from an intermediate state). **SRPO**
   self-localizes the first erroneous thought and resets there; **RRPO** resets at
   a random thought. The gradient on the shared prefix is masked, so learning
   concentrates on the suffix after the reset point.

Reasoning is generated **thought-by-thought**: each action is one reasoning step
delimited by `</thought>`, with `\boxed{answer}` as the terminal signal. These
boundaries are what make localization and resetting tractable.

---

## Why VERL

- **Hybrid engine** — model weights are shared between vLLM (inference) and FSDP
  (training); the vLLM KV cache is released during the actor update, giving the
  full GPU to training.
- **Async generation** — all N rollouts for a prompt run concurrently as async
  tasks, which is what lets the reset methods coordinate a shared rollout buffer.
- **Agent-loop API** — `AgentLoopBase` supports multi-step generation with stop
  tokens, growing prompts, and prefix caching across steps.

---

## How generation works (the Thought MDP)

Every method subclasses the same base loop (`thought_agent_loop.py`):

1. Extract the question from the raw prompt and format it with a template
   (`prompt_templates.py`); tokenize with `tokenizer.encode()` (not
   `apply_chat_template`).
2. Loop up to `max_thoughts` (default 20): vLLM `generate(stop=["</thought>"],
   include_stop_str_in_output=True)`, accumulate token ids + log-probs, extend
   the prompt for the next thought (same `request_id` → KV-cache reuse), and stop
   when `\boxed{` appears.
3. Safety-truncate at the last *complete* thought boundary if the response would
   exceed `response_length`.
4. Return an `AgentLoopOutput` (prompt_ids, response_ids, response_mask, logprobs).

`include_stop_str_in_output=True` keeps the `</thought>` delimiter in the
response so thought boundaries survive into the training batch.

### Thought-GRPO baseline

`thought_agent` runs N of these chains independently. Reward is terminal
(correct/incorrect); advantage is GRPO group-normalized within a prompt's N
chains (shared `uid`).

### SRPO / RRPO

`srpo_agent` / `rrpo_agent` (`srpo_agent_loop.py`, built on the ICS machinery in
`thought_ics_agent_loop.py`) fill the N-slot rollout buffer from a coordinator:

```
SRPOAgentLoop.run() — N concurrent calls per prompt:
  Slot 0 (coordinator):
    Group 1: generate 4 fresh i.i.d. chains
    Group 2: for a failed chain, localize the first erroneous thought
             (a CoT call → \boxed{step}; SRPO), or pick a random step (RRPO);
             backtrack to that thought boundary; resample a continuation from
             the shared prefix
    → buffer holds 4 base + 4 reset trajectories
  Slots 1..N-1: await the coordinator, take buffer[my_slot]
  All N slots return an AgentLoopOutput; the trainer sees N trajectories
```

The reset group shares its prefix with the chain it branched from; the **`srpo`
policy loss** (`srpo_loss.py`) masks the gradient on that shared prefix so only
the suffix tokens after the reset point are trained. SRPO vs RRPO is a single
config flag (`random_localization`), so both map to the same class.

---

## Agent-loop registry

Agent loops are registered by name in
[`config/thought_agent_config.yaml`](config/thought_agent_config.yaml)
(`name:` → `_target_:` + per-loop params). Select one with
`actor_rollout_ref.rollout.agent.default_agent_loop=<name>`:

| Registry name | Class | Role |
|---|---|---|
| `thought_agent` | `ThoughtAgentLoop` | Baseline — N independent chains |
| `thought_ics_agent` | `ThoughtICSAgentLoop` | Iterative self-correction base class |
| `srpo_agent` / `rrpo_agent` | `SRPOAgentLoop` | SRPO (self-loc) / RRPO (random); 4 base + 4 reset |
| `srpo_2x4_agent` / `rrpo_2x4_agent` | `SRPO2x4AgentLoop` | Sampling ablation: 2 groups of 4 |
| `srpo_1x8_agent` / `rrpo_1x8_agent` | `SRPO1x8AgentLoop` | Sampling ablation: 1 prefix + 8 resets |
| `srpo_nomask_agent` / `rrpo_nomask_agent` | `SRPONoMaskAgentLoop` | No shared-prefix masking ablation |

`__init__.py` imports these classes (so Hydra `_target_` can resolve them),
registers the `srpo` / `srpo_clip` **policy losses**, and monkey-patches
`compute_data_metrics` so ICS stats (`ics_triggered`, `ics_iterations`,
`ics_corrected`, per-iteration accuracy) reach wandb on any reset run (a no-op on
baseline runs).

---

## Policy losses

Registered via VERL's `register_policy_loss` and selected with
`actor_rollout_ref.actor.policy_loss.loss_mode=<name>`:

| `loss_mode` | File | Description |
|---|---|---|
| `srpo` | `srpo_loss.py` | Two-group GRPO with pre-computed advantages + suffix-only gradients (shared prefix masked) |
| `srpo_clip` | `srpo_clip_loss.py` | Clipped-surrogate ablation: standard PPO/GRPO clipped ratio with SRPO advantages + suffix mask |

The baseline uses VERL's default (vanilla GRPO) loss.

---

## File structure

```
training/
├── __init__.py                  # registers agent loops + policy losses; patches ICS metrics
├── thought_agent_loop.py        # base thought-by-thought loop (Thought-GRPO baseline)
├── thought_ics_agent_loop.py    # iterative self-correction (localize → backtrack → regenerate)
├── srpo_agent_loop.py           # SRPO / RRPO: 4 fresh + 4 reset counterfactuals
├── srpo_2x4_agent_loop.py       # ablation: 2 groups of 4
├── srpo_1x8_agent_loop.py       # ablation: 1 prefix + 8 resets
├── srpo_mask_variants.py        # ablation: no shared-prefix masking
├── srpo_loss.py                 # "srpo" policy loss (two-group, suffix mask)
├── srpo_clip_loss.py            # "srpo_clip" policy loss (clipped surrogate)
├── prompt_templates.py          # thought-by-thought prompt + localization templates
├── reward_fn.py                 # VERL reward entrypoint, routed by data_source
├── reward_scorers.py            # non-math scorers (MC, code, sciknoweval, ifeval, yes/no)
├── branch_logger.py             # opt-in reset-tree dumps (SRPO_BRANCH_DUMP_* env vars)
├── ics_metrics.py               # ICS rollout stats → wandb
├── best_checkpoint_patch.py     # keep best + last LoRA checkpoints only
├── config/                      # Hydra configs (see below)
├── scripts/                     # launch, data prep, experiment archiving
└── tests/
```

---

## Configs

Hydra configs live in [`config/`](config/). The `_math500` suffix is just the
base config's default experiment; the `batch_scripts/` submit scripts override
the model, data, experiment name, and project at launch.

| Config | Used by | Purpose |
|---|---|---|
| `srpo_math500.yaml` | `launch_srpo_training.sh` | SRPO/RRPO base (`loss_mode=srpo`, `default_agent_loop=srpo_agent`) |
| `srpo_clip_math500.yaml` | `launch_srpo_training.sh` | Clipped-surrogate variant base (`loss_mode=srpo_clip`) |
| `thought_grpo_math500.yaml` | `launch_training.sh` | Thought-GRPO baseline base |
| `thought_agent_config.yaml` | all runs | Agent-loop registry (`name` → `_target_` + params) |
| `thought_agent_config_ics_eval.yaml` | eval | Agent-loop params for ICS-style evaluation |

Each base config points `actor_rollout_ref.rollout.agent.agent_loop_config_path`
at `thought_agent_config.yaml`. Override any value on the CLI, e.g.
`actor_rollout_ref.actor.optim.lr=1e-6`.

---

## Running

The canonical entry points are the `batch_scripts/` submit scripts (SLURM or
`--local`); see the [top-level README](../README.md#usage). Those wrap two launch
scripts here:

```bash
# SRPO / RRPO and their ablations  (config-name=srpo_math500)
bash training/scripts/launch_srpo_training.sh \
    actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
    data.train_files=$HOME/data/rlhf/numinamath_olympiads/train.parquet \
    data.val_files=$HOME/data/rlhf/numinamath_olympiads/test.parquet \
    trainer.experiment_name=numina_oly_olmo7b_srpo

# Thought-GRPO baseline  (config-name=thought_grpo_math500)
bash training/scripts/launch_training.sh \
    actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
    trainer.experiment_name=numina_oly_olmo7b_thought ...
```

Both invoke `scripts/main_ppo_patched.py` — VERL's PPO entrypoint with this
repo's patches (`best_checkpoint_patch`, ICS metrics) applied via
`main_ppo_wrapper.py` — and then `scripts/save_experiment.py` archives the
config, metrics, and final checkpoint under `experiments/`.

### GPU overrides

The submit scripts set these; apply the same when launching directly.

- **7–8B (OLMo-3-7B), TP=2 / 2 GPUs:** lower micro-batches
  (`ppo_micro_batch_size_per_gpu=2`, `log_prob_micro_batch_size_per_gpu=8`).
- **14B (Qwen-2.5-14B), TP=4 / 4 GPUs:** `gpu_memory_utilization=0.35`,
  `ppo_micro_batch_size_per_gpu=1`, `log_prob_micro_batch_size_per_gpu=4`.

---

## Rewards

`reward_fn.py` is the VERL reward entrypoint; it routes by `data_source`:

- **Math** (`numinamath_olympiads`, `math_level5`, `aime`, …) — extract the last
  `\boxed{}` and verify with VERL's math checker.
- **Code** (`livecodebench*`) — extract the code block and run it against the
  LiveCodeBench test cases (`reward_scorers.code_score`).
- **Multiple-choice** (`gpqa`, `csqa`, `mathqa`, `mmlu_pro`) — letter extraction
  and match.
- **SciKnowEval / IFEval / yes-no** — handled by the corresponding scorers in
  `reward_scorers.py`.

An optional small `_format_reward` (counts `</thought>` delimiters) can be added
on top of the base correctness reward.

---

## Datasets

Prepare train/test parquet splits under `~/data/rlhf/`:

```bash
python training/scripts/prepare_datasets.py \
    --datasets numinamath_olympiads livecodebench_medium
```

`prepare_datasets.py` covers the math, code, and MC sources used here; the other
`scripts/prepare_*.py` build auxiliary splits (MATH Level 5, reasoning-eval
benchmarks, the numina+sciknow mix).

---

## Instrumentation

- **`branch_logger.py`** — opt-in. Set `SRPO_BRANCH_DUMP_DIR` (and optionally
  `SRPO_BRANCH_DUMP_EVERY`, `SRPO_BRANCH_DUMP_MAX`) to dump the per-prompt reset
  tree as JSONL for offline analysis. No-op when unset; never affects training.
- **`ics_metrics.py`** — aggregates reset/correction stats into wandb-loggable
  metrics (wired in automatically via `__init__.py`).
- **`scripts/analyze_srpo_stats.py`** — post-hoc analysis of the dumped stats.

---

## Environment

- **Conda env**: `srpo` (Python 3.12). Key deps: torch, vLLM, VERL — installed
  via pip; see [`requirements.txt`](../requirements.txt) /
  [`environment.yml`](../environment.yml) at the repo root.
- **Critical**: `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` is needed
  for vLLM's CUDA libraries (the launch scripts set this).
- **Tracking**: runs log to wandb under `trainer.project_name`; set
  `trainer.logger=["console"]` to disable. Checkpoints are LoRA adapters under
  `checkpoints/${project_name}/${experiment_name}/` (best + last only, via
  `best_checkpoint_patch.py`).
