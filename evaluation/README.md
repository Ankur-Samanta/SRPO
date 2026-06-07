# SCPO — Thought-Level GRPO with Process Rewards

Thought-level GRPO training for math reasoning. Trains models to generate step-by-step thought chains (`</thought>` delimited) using best-of-K selection guided by a rubric-based process reward model (ThoughtEvaluator).

Built on TRL's `GRPOTrainer` — we override `_generate_and_score_completions()` to do thought-level rollouts while TRL handles loss computation, checkpointing, and optimization.

## Quick Start

```bash
# Train Qwen 7B on MATH-500 with frozen evaluator (1 GPU)
CUDA_VISIBLE_DEVICES=6 python -m evaluation.scripts.train \
    --thought-level-grpo --algorithm grpo \
    --model Qwen/Qwen2.5-7B-Instruct --dataset math500 \
    --num-prompts 500 --num-steps 100 --batch-size 4 \
    --freeze-evaluator \
    --num-generations 4 --lr 1e-5 --logging-steps 1 \
    --save-steps 999 --seed 42 --output-dir experiments
```

## Key Files

| File | Description |
|------|-------------|
| `evaluation/thought_grpo_trainer.py` | ThoughtGRPOTrainer — core training loop |
| `evaluation/thought_evaluator.py` | ThoughtEvaluator — rubric-based process reward scoring |
| `evaluation/thought_mdp.py` | Depth-synchronized batched thought chain generation |
| `evaluation/holistic_grpo_trainer.py` | HolisticGRPOTrainer — full-rollout baseline (ablation) |
| `evaluation/holistic_evaluator.py` | HolisticEvaluator — trajectory-level rubric scoring |
| `evaluation/evaluate.py` | Post-training evaluation (greedy thought chains + answer extraction) |
| `evaluation/config.py` | SCPOConfig dataclass |
| `evaluation/constants.py` | Prompt templates, thought delimiter |
| `evaluation/scripts/train.py` | CLI entry point |
| `scripts/eval_base_models.py` | Evaluate base or trained model on test set |
| `scripts/analyze_trajectories.py` | Run ThoughtEvaluator on saved trajectories, compare base vs trained |

---

## Process Reward Evaluation Pipelines

The process reward pipelines let you evaluate *thought quality* — not just final-answer accuracy. The ThoughtEvaluator scores each thought step on three rubric dimensions (forward_progress, substantiveness, coherence) using an LLM-as-judge.

### Pipeline 1: Train with process rewards

```bash
# Standard (policy evaluates itself)
CUDA_VISIBLE_DEVICES=6 python -m evaluation.scripts.train \
    --thought-level-grpo --algorithm grpo \
    --model Qwen/Qwen2.5-7B-Instruct --dataset math500 \
    --num-prompts 500 --num-steps 100 --batch-size 4 \
    --num-generations 4 --lr 1e-5 --seed 42 --output-dir experiments

# Frozen evaluator (base model evaluates — prevents reward hacking)
# Same as above but add --freeze-evaluator
CUDA_VISIBLE_DEVICES=6 python -m evaluation.scripts.train \
    --thought-level-grpo --algorithm grpo \
    --model Qwen/Qwen2.5-7B-Instruct --dataset math500 \
    --num-prompts 500 --num-steps 100 --batch-size 4 \
    --freeze-evaluator \
    --num-generations 4 --lr 1e-5 --seed 42 --output-dir experiments
```

Training produces:
- `experiments/thought_grpo_<model>/checkpoint-final/` — LoRA adapter
- `experiments/thought_grpo_<model>/eval_results.json` — test accuracy
- `experiments/thought_grpo_<model>/eval_results_detailed.json` — per-problem results with full thought chains

### Holistic GRPO Baseline (ablation)

The holistic baseline isolates the contribution of iterative thought-by-thought generation. It uses the **exact same** prompt template and rubric dimensions, but generates full rollouts in one pass and evaluates each complete trajectory with a single model call. This tests whether the thought-by-thought decomposition actually matters.

Key differences from thought-level GRPO:
- **1 vLLM generate call** total (not 1 per step) — no `stop=[delimiter]`, model generates freely
- **1 evaluator call** total (not 1 per step) — scores all thoughts in each trajectory at once
- **2 weight swaps** total (not 2 per step) — if `--freeze-evaluator` is used
- No best-of-K selection or thought history accumulation
- Group = prompt (K rollouts per prompt), not (prompt, step)

```bash
CUDA_VISIBLE_DEVICES=6 python -m evaluation.scripts.train \
    --holistic-grpo --algorithm grpo \
    --model Qwen/Qwen2.5-7B-Instruct --dataset math500 \
    --num-prompts 500 --num-steps 100 --batch-size 4 \
    --freeze-evaluator \
    --num-generations 4 --lr 1e-5 --seed 42 --output-dir experiments
```

Produces:
- `experiments/holistic_grpo_<model>/checkpoint-final/` — LoRA adapter
- `experiments/holistic_grpo_<model>/eval_results.json` — test accuracy
- Metrics: `holistic/avg_thoughts_per_trajectory` shows whether the model actually produces `</thought>` delimiters when generating freely (key diagnostic)

`--holistic-grpo` is mutually exclusive with `--thought-level-grpo`, `--mc`, and `--sc`.

### Pipeline 2: Evaluate base model on test set

Generates greedy thought chains (no evaluator in the loop) and checks final `\boxed{}` answers.

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/eval_base_models.py Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=7 python scripts/eval_base_models.py meta-llama/Llama-3.1-8B-Instruct
```

Produces:
- `experiments/base_eval_<model>/eval_results.json` — test accuracy
- `experiments/base_eval_<model>/eval_results_detailed.json` — per-problem results with thought chains

### Pipeline 3: Evaluate trained checkpoint on test set

Same greedy evaluation but loads a LoRA checkpoint and merges it first.

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/eval_base_models.py Qwen/Qwen2.5-7B-Instruct \
    --checkpoint experiments/thought_grpo_qwen2.5-7b-instruct
```

### Pipeline 4: Compare base vs trained thought quality with ThoughtEvaluator

This is the key analysis script. Takes the saved `eval_results_detailed.json` from pipelines 2 and 3 (which contain full thought chains), runs the ThoughtEvaluator on every thought step in both, and produces a head-to-head comparison.

```bash
# Use base model as evaluator
CUDA_VISIBLE_DEVICES=6 python scripts/analyze_trajectories.py \
    --base-results experiments/base_eval_qwen2_5-7b-instruct/eval_results_detailed.json \
    --trained-results experiments/thought_grpo_qwen2.5-7b-instruct/eval_results_detailed.json \
    --evaluator-model Qwen/Qwen2.5-7B-Instruct \
    --output-dir experiments/trajectory_analysis_qwen

# Use the trained checkpoint as evaluator instead
CUDA_VISIBLE_DEVICES=6 python scripts/analyze_trajectories.py \
    --base-results experiments/base_eval_qwen2_5-7b-instruct/eval_results_detailed.json \
    --trained-results experiments/thought_grpo_qwen2.5-7b-instruct/eval_results_detailed.json \
    --evaluator-model Qwen/Qwen2.5-7B-Instruct \
    --evaluator-checkpoint experiments/thought_grpo_qwen2.5-7b-instruct \
    --output-dir experiments/trajectory_analysis_qwen_trained_eval
```

Produces:
- `comparison.json` — problems categorized as stayed_correct, stayed_incorrect, improved, regressed
- `base_scored.json` — per-problem ThoughtEvaluator scores for base model chains
- `trained_scored.json` — per-problem ThoughtEvaluator scores for trained model chains
- `improved_detail.json` — full thought chains + eval scores for problems that went incorrect -> correct
- `regressed_detail.json` — same for problems that went correct -> incorrect
- Console output with per-category dimension breakdowns (composite, forward_progress, substantiveness, coherence)

---

## ThoughtEvaluator Scoring

The ThoughtEvaluator (`evaluation/thought_evaluator.py`) scores each thought on three dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| forward_progress | 0.4 | Does this thought advance toward solving the problem? |
| substantiveness | 0.4 | Is this thought substantive (calculations, reasoning) vs filler? |
| coherence | 0.2 | Does this thought follow logically from the previous context? |

Composite score = weighted sum of dimension scores (range 0-1).

### Freeze Evaluator

`--freeze-evaluator` swaps vLLM weights to base model (no LoRA) before scoring, and back to policy weights after. This prevents reward hacking where the policy learns to generate thoughts that game its own evaluator rather than genuinely improving reasoning.

Key implementation details:
- Weight swaps happen once per step (step-major loop), not once per problem
- `reset_prefix_cache()` called after each swap to invalidate stale KV entries
- Only works with vLLM (warns at init if vLLM not available)

---

## Typical Experiment Workflow

```bash
# 1. Train (with frozen evaluator)
CUDA_VISIBLE_DEVICES=6 python -m evaluation.scripts.train \
    --thought-level-grpo --algorithm grpo \
    --model Qwen/Qwen2.5-7B-Instruct --dataset math500 \
    --num-prompts 500 --num-steps 100 --batch-size 4 \
    --freeze-evaluator --num-generations 4 --lr 1e-5 \
    --seed 42 --data-seed 42 --train-ratio 0.8 --val-ratio 0.0 --test-ratio 0.2 \
    --output-dir experiments 2>&1 | tee experiments/train.log

# 2. Evaluate base model (if not already done)
CUDA_VISIBLE_DEVICES=6 python scripts/eval_base_models.py Qwen/Qwen2.5-7B-Instruct

# 3. Compare thought quality
CUDA_VISIBLE_DEVICES=6 python scripts/analyze_trajectories.py \
    --base-results experiments/base_eval_qwen2_5-7b-instruct/eval_results_detailed.json \
    --trained-results experiments/thought_grpo_qwen2.5-7b-instruct/eval_results_detailed.json \
    --evaluator-model Qwen/Qwen2.5-7B-Instruct \
    --output-dir experiments/trajectory_analysis_qwen
```

## Data Splits

All experiments use the same deterministic split (seed=42):
- 500 problems from MATH-500
- 400 train / 0 val / 100 test (80/0/20)
- `--data-seed 42` ensures identical splits across runs
