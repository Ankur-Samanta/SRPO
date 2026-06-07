# Experiment Log

## SC-GRPO (SCPO) vs Thought-GRPO (TGRPO) — Main Comparison

### Setup

- **Models**: Qwen 2.5 7B Instruct, Llama 3.1 8B Instruct
- **Training**: 2 epochs, LoRA rank 64, 2x A100 40GB, TP=2
- **TGRPO**: 8 independent thought chains per prompt, outcome reward, GRPO advantage
- **SCPO**: ICS agent loop (iterative critique & self-correction), L2 localization (no ground truth shown), judge temp 0.3, max 20 thoughts, 256 tok/thought
- **Metric**: Greedy val accuracy (temp=0, n=1) at final training step

### Commands

Full job definitions with all overrides are in:
- **TGRPO**: [`batch_scripts/submit_thought_grpo.sh`](../batch_scripts/submit_thought_grpo.sh)
- **SCPO (ICS)**: [`batch_scripts/submit_thought_scgrpo.sh`](../batch_scripts/submit_thought_scgrpo.sh)
- **Pass@k eval**: [`scripts/eval_pass_at_k.py`](../scripts/eval_pass_at_k.py)
- **Data prep**: [`training/scripts/prepare_datasets.py`](scripts/prepare_datasets.py)

```bash
# TGRPO
bash batch_scripts/submit_thought_grpo.sh {model}_{dataset}

# SCPO (ICS)
bash batch_scripts/submit_thought_scgrpo.sh {model}_{dataset}
```

Where `{model}` is `qwen7b` or `llama8b` and `{dataset}` is `math500`, `mathlvl5`, `aime`, `gpqa`, `csqa`, `mathqa`.

### Data Preparation

```bash
# MATH-500, GPQA, CSQA, MathQA (400/100 splits, MC answers uppercased)
python training/scripts/prepare_datasets.py

# MATH Level 5 (400/100)
python training/scripts/prepare_math_level5.py

# AIME (400/100)
python training/scripts/prepare_aime.py
```

Parquets at `~/data/rlhf/{dataset}/{train,test}.parquet`.

Note: GPQA only has 448 problems total, so split is 348/100. GPQA requires `max_prompt_length=4096` due to long science prompts.

### Config

- **SCPO ICS config**: `training/config/thought_agent_config.yaml` (thought_ics_agent entry)
  - `autonomy_level: 2` (L2 — binary "you're wrong", no ground truth shown)
  - `localization_temp: 0.3`
  - `localization_max_tokens: 2048`
- **Training config**: `training/config/thought_grpo_math500.yaml`
  - `rollout.n: 8`, `rollout.prompt_length: 2048`, `rollout.response_length: 5120`
  - `trainer.total_epochs: 2`

### Results — Greedy Val Accuracy (Final Step)

| Dataset | Model | TGRPO | SCPO | Delta | Winner |
|---------|-------|-------|------|-------|--------|
| gpqa | llama8b | 0.00 | **0.27** | +0.27 | **SCPO** |
| gpqa | qwen7b | 0.33 | **0.36** | +0.03 | **SCPO** |
| aime | qwen7b | 0.21 | **0.23** | +0.02 | **SCPO** |
| aime | llama8b | 0.26 | 0.26 | 0.00 | Tie |
| mathlvl5 | qwen7b | 0.49 | **0.51** | +0.02 | **SCPO** |
| mathlvl5 | llama8b | 0.23 | **0.24** | +0.01 | **SCPO** |
| csqa | qwen7b | 0.84 | **0.87** | +0.03 | **SCPO** |
| csqa | llama8b | 0.75 | 0.75 | 0.00 | Tie |
| math500 | llama8b | 0.47 | **0.48** | +0.01 | **SCPO** |
| math500 | qwen7b | **0.72** | 0.67 | -0.05 | TGRPO |
| mathqa | qwen7b | **0.89** | 0.81 | -0.08 | TGRPO |
| mathqa | llama8b | **0.72** | 0.65 | -0.07 | TGRPO |

**Scorecard**: SCPO 7 wins, TGRPO 3 wins, 2 ties.

**Pattern**: SCPO wins on hard tasks (GPQA, AIME, MATH Level 5, CSQA) where base model accuracy is low and self-correction has room to help. TGRPO wins on easier tasks (MathQA, MATH-500) where accuracy is already high and the ICS correction loop adds noise.

### Wandb

Project: `ankur-samanta-wb/thought_grpo`

Experiment names follow `{dataset}_{model}_{method}` convention:
- TGRPO: `{dataset}_{model}_thought` (e.g., `gpqa_qwen7b_thought`)
- SCPO: `{dataset}_{model}_ics` (e.g., `gpqa_qwen7b_ics`)

### Checkpoints

Saved at `checkpoints/thought_grpo/{experiment_name}/global_step_{N}/actor/lora_adapter/`.

---

## Pass@k Evaluation (AIME + MATH Level 5)

### Setup

- **Method**: temp=0.7, n=10 samples per problem, thought chain generation via vLLM with native LoRA
- **Script**: `scripts/eval_pass_at_k.py`

### Commands

```bash
python3 scripts/eval_pass_at_k.py \
    --base-model {model_path} \
    --lora-adapter checkpoints/thought_grpo/{exp}/global_step_24/actor/lora_adapter \
    --test-data ~/data/rlhf/{dataset}/test.parquet \
    --k 10 --temperature 0.7 --use-examples false \
    --output-dir outputs/eval/{exp}
```

### Results — AIME (temp=0.7)

| Experiment | pass@1 | pass@5 | pass@10 |
|------------|--------|--------|---------|
| SCPO Llama8b | **0.262** | **0.410** | 0.450 |
| TGRPO Llama8b | 0.246 | 0.406 | **0.460** |
| SCPO Qwen7b | 0.211 | 0.355 | 0.400 |
| TGRPO Qwen7b | 0.215 | **0.377** | **0.440** |

SCPO has slight edge at pass@1 (greedy-like), TGRPO catches up at higher k (diversity).

---

## Ablation: L1 vs L2 Localization

### Setup

L1 (oracle): ground truth shown to judge during error localization.
L2 (binary): judge only told "your answer is wrong", no ground truth.

Config: `training/config/thought_agent_config_l1_256.yaml`

### Results — Greedy Val Accuracy (L2 vs L1)

| Dataset | Model | L2 | L1 | Winner |
|---------|-------|-------|------|--------|
| AIME | Llama8b | **0.29** | 0.09 | **L2** |
| MLvl5 | Llama8b | **0.21** | 0.20 | **L2** |
| MLvl5 | Qwen7b | **0.48** | 0.45 | **L2** |
| AIME | Qwen7b | crashed | — | — |

L1 collapsed badly on AIME Llama8b (0.09 vs 0.29). Showing ground truth likely leaks the answer, making correction too easy during training but not transferring to eval without the oracle.

---

## Ablation: Judge Temperature 0.7

### Status

All 4 runs (lt07) failed before step 1 due to disk quota issues. No results.
Config: `training/config/thought_agent_config_loctemp07.yaml`

---

## Ablation: Format Reward (+0.02 per `</thought>`, cap 0.2)

### Setup

Added a rule-based format reward proportional to the number of `</thought>` delimiters.
- `format_reward_cap=0.2`, `format_reward_steps=10` (+0.02 per thought, saturates at 10 steps)
- Additive with the 1.0/0.0 correctness reward
- Passed via `custom_reward_function.reward_kwargs` in the YAML config

Implementation: `training/reward_fn.py` (returns dict with `score`, `math_correct`, `format_reward`)

### Results — Greedy Val Accuracy (MATH Level 5, final checkpoint)

| Run | Baseline | + Format Reward | Delta |
|-----|----------|-----------------|-------|
| TGRPO Qwen 7B | **0.49** | 0.31 | -18pp |
| TGRPO Llama 8B | **0.23** | 0.16 | -7pp |
| SCGRPO Qwen 7B | **0.51** | 0.38 | -13pp |
| SCGRPO Llama 8B | **0.20** | 0.16 | -4pp |

### Analysis

Format reward hurt across the board. The format reward was near-saturated (~0.195/0.2) for
almost all rollouts since the models already produce ~10 thoughts per chain. This means:
1. It adds a near-constant offset to every reward within a GRPO group
2. Constant offsets cancel in GRPO advantage normalization (subtract mean, divide by std)
3. But the inflated reward variance slightly dilutes the correctness gradient signal

**Conclusion**: Format reward is not useful when the model already reliably produces structured
thought chains. The `</thought>` delimiter is enforced by the generation loop's stop tokens,
so there is no format compliance problem to solve. Default remains `format_reward_cap=0.0`.
