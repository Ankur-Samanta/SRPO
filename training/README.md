# Thought-Level GRPO on VERL

Thought-by-thought rollout generation inside VERL's GRPO framework. Instead of
generating a single long completion per prompt, the model reasons in discrete
steps delimited by `</thought>`, with `\boxed{answer}` as the terminal signal.

Three training methods share this infrastructure:

1. **Thought GRPO** (Phase 1): N independent thought chains per prompt, outcome reward
2. **SC-GRPO** (Phase 2): Self-correction via ICS (iterative correction search)
3. **Process GRPO** (Phase 3): Best-of-K per step with rubric scoring

## Why VERL (not TRL)

- **Hybrid engine**: model weights shared between vLLM (inference) and HF (training).
  vLLM KV cache released during actor update, giving full GPU to training.
- **Async vLLM generation**: all N rollouts run concurrently as async tasks.
- **Agent loop API**: `AgentLoopBase` supports multi-step generation with stop tokens,
  growing prompts, and prefix caching.

## Architecture

### Phase 1: Thought GRPO (N Independent Chains)

Each prompt gets N independent thought-by-thought chains. Terminal reward (correct/incorrect).
Standard GRPO advantage (group-normalize within same prompt).

```
RayPPOTrainer.fit()
  for each training step:
    batch = DataLoader(32 prompts)
    batch.repeat(n=8)  -->  256 samples (each prompt x 8 chains)

    generate_sequences(batch):
      AgentLoopManager dispatches to AgentLoopWorker (Ray actor)
      for each sample (async, concurrent):
        ThoughtAgentLoop.run():
          1. Extract question from raw_prompt
          2. Build prompt: template.format(question=question)
          3. Tokenize with tokenizer.encode() (NOT apply_chat_template)
          4. Loop up to max_thoughts (20):
             - Pre-generation check: if remaining response_length < max_tokens_per_thought, stop
             - vLLM generate(stop=["</thought>"], include_stop_str_in_output=True)
             - accumulate token_ids + log_probs
             - extend prompt for next thought (prefix caching reuses KV)
             - if "\boxed{" in decoded text: break
          5. Safety truncation: if response exceeds response_length, truncate at
             last complete thought boundary (not mid-thought)
          6. Return AgentLoopOutput(prompt_ids, response_ids, response_mask, logprobs)

    reward_fn(batch):
      NaiveRewardManager decodes response, extracts last \boxed{},
      compares to ground_truth via is_equiv() --> 0.0 or 1.0

    GRPO advantage:
      group by uid (8 chains per prompt share same uid)
      advantage = (reward - group_mean) / (group_std + eps)

    actor PPO update:
      response_mask-weighted clipped policy gradient loss (vanilla mode)
```

### Phase 2: SC-GRPO (Iterative Self-Correction)

Instead of N independent chains, the ICS agent's coordinator (slot 0) fills the
ENTIRE N-slot rollout buffer by alternating fresh chain generation and ICS correction.
When a fresh chain is wrong, the coordinator triggers ICS: localize the error step,
backtrack, regenerate from the prefix. All trajectories (fresh and ICS corrections)
count as rollouts.

```
ThoughtICSAgentLoop.run() -- N concurrent calls per prompt:

  Slot 0 (coordinator) -- _fill_rollout_buffer():
    buffer = []
    while len(buffer) < N:
      1. Generate fresh chain (thought-by-thought, stop at </thought>)
      2. Append to buffer
      3. If buffer full or chain correct or chain empty -> continue to next fresh
      4. Wrong chain -> trigger ICS (budget = min(max_ics_iterations, N - len(buffer))):
         a. Localize error: standard CoT call with L1/L2 prompt
            -> parse \boxed{step_number} from response
         b. Backtrack: slice response_ids at thought_boundaries[error_step - 2]
         c. Regenerate from prefix (thought-by-thought, new request_id)
         d. Append correction to buffer; check correctness; if correct -> break
    Signal done to other slots

  Slots 1..N-1 (workers):
    await coordinator done, then take buffer[my_slot]

  All N slots return AgentLoopOutput -- trainer sees N trajectories as usual
```

### Phase 3: Process GRPO (Best-of-K with Rubric Scoring)

Instead of N independent chains with outcome reward, process GRPO generates K
thought variants per step, scores each with ThoughtEvaluator's rubric, selects the
best to extend the prefix, and uses rubric scores as per-thought rewards for GRPO.

```
GreedyProcessAgentLoop.run() -- N concurrent calls per prompt:

  Slot 0 (coordinator) -- _run_process_loop():
    prefix = []
    buffer = []
    for step in 0..max_thoughts:
      1. Generate K thought variants from current prefix (K parallel vLLM calls)
      2. Score all K with ThoughtEvaluator rubric (K parallel eval calls)
      3. All K (prefix, variant, rubric_score, group_id=step) -> buffer
      4. Best-scoring variant extends prefix
      5. If \boxed{} found in best -> break
      6. If prompt_ids would exceed prompt_length -> break
    Signal done to other slots

  Slots 1..N-1 (workers):
    await coordinator done, then take buffer[my_slot]
    Overflow slots wrap around to real entries (my_slot % len(buffer))

  All N slots return AgentLoopOutput
```

**Process GRPO output structure:**

```
prompt_ids   = base template + prefix_best (variable, up to ~5420)
response_ids = single_thought only (<=256 tokens)
response_mask = [1]*len(thought) (all trained)
response_logprobs = thought_logprobs only
```

The prefix IS the prompt -- it belongs in `prompt_ids`. No `response_mask` gymnastics
or prefix logprobs tracking needed. `response_length=4096` is kept large for val
compatibility (full chain generation).

**How it maps to standard GRPO:**

A problem with S steps x K variants produces S GRPO groups of K samples. Each group
shares a prefix (template + best thoughts from prior steps) and contains K single-thought
completions with K rubric scores. All entries from the same prompt share the same `uid`,
so GRPO normalizes advantages across all entries for that prompt.

**Rubric dimensions** (from `evaluation/thought_evaluator.py`):
- Forward Progress (0.4): Does this thought advance problem-solving?
- Substantiveness (0.4): Real reasoning vs filler/meta-commentary?
- Coherence (0.2): Logical connection without contradictions?

Each dimension scored 1-5 (numbers spelled as words in the prompt to avoid regex confusion),
normalized to 0-1, weighted sum gives composite score (0-1).

**Key features:**
- **Empty thought short-circuit**: thoughts with <5 chars of content auto-score 0.1
- **Evaluator parser**: uses last regex match (not first) to avoid picking up echoed prompt content
- **freeze_evaluator**: when `freeze_evaluator=true`, rubric evaluation uses the base model
  (LoRA disabled) via `use_lora=False` in vLLM, preventing the evaluator from drifting with training
- **Prompt template**: frames generation as "produce the single next step" with rubric quality criteria (no in-context examples)

## How This Differs from Vanilla GRPO + CoT

### Generation

| | Vanilla GRPO + CoT | Thought-Level GRPO |
|---|---|---|
| Generation calls | 1 vLLM call per sample | Up to 20 calls per sample (1 per thought) |
| Stop condition | EOS token or max length | `</thought>` delimiter or `\boxed{}` terminal |
| Reasoning structure | Free-form stream | Discrete bounded steps (max 256 tokens each) |
| Prompt format | `apply_chat_template()` | Raw `tokenizer.encode()` with custom template |
| Prefix caching | N/A (single call) | Same `request_id` across thoughts -> KV cache reuse |

### Training

| | Vanilla GRPO + CoT | Thought-Level GRPO |
|---|---|---|
| Reward | Terminal (correct/incorrect) | Same (Phase 1/2) or rubric (Phase 3) |
| Advantage | GRPO group-normalize | Same |
| Loss | PPO clipped objective | PPO clipped (vanilla mode) |
| Response mask | All 1s | All 1s (no injected tokens) |

The model learns to produce explicit `</thought>` delimiters (they're in the output with
real log-probs, so they get policy gradient).

### Why `include_stop_str_in_output=True` matters

vLLM's `stop` parameter can either strip or include the stop string. We include it because:
1. The `</thought>` tokens are model-generated with natural log-probs
2. `response_mask` is all 1s -- every token gets policy gradient
3. The model learns to produce the delimiter itself

### DynPad (Dynamic Padding Trim)

VERL's `_postprocess` trims pure-padding columns from batched tensors based on attention
mask. For thought chains, most sequences are much shorter than `response_length`, so
DynPad typically achieves 7-10x sequence length reduction (e.g. ~10240 down to ~1000-1500
for typical batches). This dramatically reduces memory and compute in the actor update.

## File Structure

```
training/
  __init__.py                          # Imports agent loops + thought_grpo_loss (triggers @register)
  thought_agent_loop.py                # Phase 1: ThoughtAgentLoop (N independent chains)
  thought_ics_agent_loop.py            # Phase 2: ThoughtICSAgentLoop (ICS buffer filling)
  greedy_process_agent_loop.py         # Phase 3: GreedyProcessAgentLoop (best-of-K + rubric)
  thought_grpo_loss.py                 # Pooled thought normalization loss (registered as "thought_grpo")
  prompt_templates.py                  # Thought-by-thought prompt templates (with/without examples)
  reward_fn.py                         # Math correctness reward (wraps verl math_reward)
  process_reward_fn.py                 # Process reward: rubric for training, math correctness for val
  config/
    thought_agent_config.yaml          # Agent loop configs (thought_agent, ICS, process)
    thought_agent_config_freeze_eval.yaml  # Same but freeze_evaluator=true for process GRPO
    thought_grpo_math500.yaml          # Thought GRPO / SC-GRPO training recipe
    process_grpo_math500.yaml          # Process GRPO recipe (n=160, greedy_process_agent)
    process_grpo_smoke.yaml            # Process GRPO smoke test (bs=8, 2 GPUs)
  scripts/
    launch_training.sh                 # Launch script with env setup
    prepare_datasets.py                # Prepare MATH-500, GPQA, CSQA, MathQA (400/100 splits)
    prepare_math500.py                 # Convert MATH-500 JSON to VERL parquet format
    prepare_math_level5.py             # Download Hendrycks MATH, filter Level 5, convert to parquet
    prepare_aime.py                    # Prepare AIME dataset
    save_experiment.py                 # Post-training: collects config, metrics, checkpoint
  tests/
    test_thought_agent_loop.py         # Unit tests for thought agent loop
    test_greedy_process_agent_loop.py  # Unit tests for process agent loop
    test_process_grpo_integration.py   # Integration tests for process GRPO
batch_scripts/
  submit_thought_grpo.sh               # Thought-level GRPO (independent rollouts)
  submit_thought_scgrpo.sh             # Self-correction GRPO (ICS + 2 epochs)
  submit_process_grpo_v2.sh            # Process GRPO with freeze_evaluator
evaluation/
  thought_evaluator.py                 # ThoughtEvaluator: rubric-based scoring
```

## VERL Integration Details

### Agent Loop Registration

VERL uses a registry pattern. Two mechanisms:

1. **`@register("thought_agent")` decorator** (in `thought_agent_loop.py`): writes to
   `_agent_loop_registry` at import time. Runs in the driver process.

2. **YAML config loading** (in `AgentLoopWorkerBase.__init__`): reads
   `thought_agent_config.yaml`, populates registry with full config including `_target_`,
   `max_thoughts`, etc. Ray workers use this path.

**Requirement**: `training` must be importable in worker processes.
The launch script sets `PYTHONPATH="${SCPO_DIR}:$PYTHONPATH"`.

### Prompt Construction

We bypass `apply_chat_template()` entirely. The template is in `prompt_templates.py`:
- `prompt_template_no_examples()` (default, `use_examples=false`): concise instructions
  framing generation as "produce the single next step" with quality criteria
- `prompt_template_with_examples()`: includes 2 worked examples

```python
prompt_text = self.template.format(question=question)
prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
```

### Prefix Caching

Each `ThoughtAgentLoop.run()` call uses the same `request_id` (UUID) across all
thought steps. VERL's `AsyncLLMServerManager` routes same-ID requests to the same
vLLM server. With `enable_prefix_caching: True`, vLLM reuses the KV cache.

### Data Flow Through the Trainer

```
Dataset (parquet):
  prompt: [{"role": "user", "content": "..."}]
  data_source: "math500"
  reward_model: {"ground_truth": "42"}
  extra_info: {"index": 0, "subject": "Algebra"}

DataLoader -> DataProto batch (32 prompts)
  batch.repeat(n=8) -> 256 samples
  -> generate_sequences() -> agent loop
  -> reward_fn() -> GRPO advantage -> actor update
```

### Relative Path Resolution

Two config paths are relative to CWD:
- `agent_loop_config_path`: resolved by `resolve_config_path()`
- `custom_reward_function.path`: resolved by `load_module()` against CWD

The launch script does `cd "$SCPO_DIR"` to ensure both resolve correctly.

## Config Reference

### `thought_grpo_math500.yaml` (Thought GRPO / SC-GRPO)

| Key | Value | Notes |
|-----|-------|-------|
| `data.train_batch_size` | 32 | Unique prompts per step |
| `data.return_raw_chat` | True | Pass chat messages to agent loop |
| `model.path` | `meta-llama/Llama-3.2-3B-Instruct` | Base config (overridden by batch scripts for 7B/8B) |
| `model.lora_rank` | 64 | LoRA rank and alpha |
| `rollout.n` | 8 | Independent chains per prompt |
| `rollout.prompt_length` | 2048 | Left-pad prompts to this length |
| `rollout.response_length` | 5120 | Right-pad responses to this length |
| `rollout.max_model_len` | 8192 | vLLM KV cache budget |
| `rollout.max_num_seqs` | 256 | vLLM concurrent sequences |
| `rollout.log_prob_micro_batch_size_per_gpu` | 16 | Log-prob computation batch size |
| `actor.policy_loss.loss_mode` | `vanilla` | Standard PPO clipped loss |
| `actor.ppo_micro_batch_size_per_gpu` | 4 | PPO micro batch (Qwen 7B can use 8) |
| `algorithm.adv_estimator` | grpo | GRPO advantage estimation |
| `trainer.total_epochs` | 2 | Full passes over training data |

### `process_grpo_math500.yaml` (Process GRPO)

Same base as thought GRPO except:

| Key | Value | Notes |
|-----|-------|-------|
| `rollout.n` | 160 | max_thoughts (20) x K (8) |
| `rollout.prompt_length` | 6144 | Larger: prefix grows as thoughts accumulate |
| `rollout.response_length` | 4096 | Single thought in training; full chain in val |
| `rollout.max_model_len` | 12288 | prompt_length + response_length headroom |
| `rollout.max_num_seqs` | 64 | Lower due to larger sequences |
| `rollout.log_prob_micro_batch_size_per_gpu` | 2 | Lower due to memory |
| `rollout.agent.default_agent_loop` | `greedy_process_agent` | Best-of-K agent |
| `rollout.logprobs_mode` | `processed_logprobs` | Logprobs computed by agent loop |
| `custom_reward_function.path` | `training/process_reward_fn.py` | Rubric for train, math for val |

### `process_grpo_smoke.yaml` (Process GRPO smoke test)

Smaller config for quick testing: `train_batch_size=8`, `n_gpus_per_node=2`,
`tensor_model_parallel_size=2`, `ppo_micro_batch_size_per_gpu=8`,
`log_prob_micro_batch_size_per_gpu=16`, `max_num_seqs=256`, no wandb, no val.

### `thought_agent_config.yaml` (Agent Loop Parameters)

**`thought_agent`** (Phase 1):

| Key | Value | Notes |
|-----|-------|-------|
| `max_thoughts` | 20 | Maximum reasoning steps per chain |
| `max_tokens_per_thought` | 256 | Token budget per thought |
| `thought_delimiter` | `</thought>` | Stop string for vLLM |
| `use_examples` | false | No in-context examples in prompt |

**`thought_ics_agent`** (Phase 2) -- inherits all above, plus:

| Key | Value | Notes |
|-----|-------|-------|
| `autonomy_level` | 2 | 1=oracle (sees ground truth), 2=binary (knows it's wrong) |
| `localization_temp` | 0.3 | Temperature for error localization call |
| `localization_max_tokens` | 2048 | Max tokens for localization (discarded, not training data) |

**`greedy_process_agent`** (Phase 3) -- inherits thought_agent fields, plus:

| Key | Value | Notes |
|-----|-------|-------|
| `K` | 8 | Number of thought variants per step |
| `eval_max_tokens` | 512 | Max tokens for rubric evaluation response |
| `eval_temperature` | 0.3 | Temperature for rubric evaluation |
| `freeze_evaluator` | false/true | false in default config, true in `_freeze_eval` variant |

### `thought_agent_config_freeze_eval.yaml`

Same as `thought_agent_config.yaml` but `greedy_process_agent.freeze_evaluator=true`.
Used by `submit_process_grpo_v2.sh` to keep rubric evaluator on the base model
(LoRA disabled during evaluation calls).

## Running Thought GRPO (Independent Rollouts)

### Local (no SLURM)

```bash
CUDA_VISIBLE_DEVICES=2,3 RAY_TMPDIR=/tmp/ray_qwen7b \
    bash batch_scripts/submit_thought_grpo.sh qwen7b_math500 --local
```

### SLURM

```bash
bash batch_scripts/submit_thought_grpo.sh qwen7b_math500
bash batch_scripts/submit_thought_grpo.sh all
```

## Running Self-Correction GRPO (ICS)

Uses `submit_thought_scgrpo.sh`. Defaults to 2 epochs and `VERL_LOGGING_LEVEL=INFO`.

### Local

```bash
CUDA_VISIBLE_DEVICES=2,3 RAY_TMPDIR=/tmp/ray_qwen7b_ics \
    bash batch_scripts/submit_thought_scgrpo.sh qwen7b_math500 --local
```

### SLURM

```bash
bash batch_scripts/submit_thought_scgrpo.sh qwen7b_math500
bash batch_scripts/submit_thought_scgrpo.sh all
```

### Available SC-GRPO jobs

| Job name | Model | Dataset |
|----------|-------|---------|
| `qwen7b_math500` | Qwen 2.5 7B | MATH-500 |
| `llama8b_math500` | Llama 3.1 8B | MATH-500 |
| `qwen7b_mathlvl5` | Qwen 2.5 7B | MATH Level 5 |
| `llama8b_mathlvl5` | Llama 3.1 8B | MATH Level 5 |
| `qwen7b_aime` | Qwen 2.5 7B | AIME |
| `llama8b_aime` | Llama 3.1 8B | AIME |
| `qwen7b_numina_oly` | Qwen 2.5 7B | NuminaMath Olympiads |
| `llama8b_numina_oly` | Llama 3.1 8B | NuminaMath Olympiads |
| `qwen7b_numina_aops` | Qwen 2.5 7B | NuminaMath AoPS |
| `llama8b_numina_aops` | Llama 3.1 8B | NuminaMath AoPS |
| `qwen7b_numina_amc` | Qwen 2.5 7B | NuminaMath AMC |
| `llama8b_numina_amc` | Llama 3.1 8B | NuminaMath AMC |
| `qwen7b_openmath2` | Qwen 2.5 7B | OpenMath2 |
| `llama8b_openmath2` | Llama 3.1 8B | OpenMath2 |

Run `bash batch_scripts/submit_thought_scgrpo.sh list` for the full list including
ablation variants.

### Enabling ICS mode manually

```bash
bash training/scripts/launch_training.sh \
    actor_rollout_ref.rollout.agent.default_agent_loop=thought_ics_agent \
    trainer.total_epochs=2
```

## Running Process GRPO (Best-of-K with Rubric Scoring)

### With freeze_evaluator (recommended)

Uses `submit_process_grpo_v2.sh` with `process_grpo_smoke` config and
`thought_agent_config_freeze_eval.yaml` (freeze_evaluator=true):

```bash
# Local
CUDA_VISIBLE_DEVICES=2,3 RAY_TMPDIR=/tmp/ray_qwen7b_pgrpo \
    bash batch_scripts/submit_process_grpo_v2.sh pgrpo_v2_qwen7b --local

# SLURM (both models)
bash batch_scripts/submit_process_grpo_v2.sh all
```

Jobs: `pgrpo_v2_qwen7b`, `pgrpo_v2_llama8b`.

### Enabling process GRPO manually

```bash
bash training/scripts/launch_training.sh \
    --config-name=process_grpo_math500 \
    actor_rollout_ref.rollout.agent.default_agent_loop=greedy_process_agent
```

### What to watch in logs

- `[ProcessGRPO] Coordinator starting for: ...` -- coordinator begins best-of-K loop
- `[ProcessGRPO] Step 0: 8 variants, scores=[0.72, 0.45, ...], best=0 (0.72)` -- per-step
- `[ProcessGRPO] {question}... | steps=3 entries=24 found_answer=True` -- per-problem
- `[ProcessGRPO] Slot 15: wrapped to entry 3 (buffer has 12 entries)` -- overflow slot

## 7-8B GPU Overrides

All batch scripts include these overrides for 7-8B models on 2x A100 40GB:

| Override | Why |
|----------|-----|
| `external_lib=training` | Register agent loops inside Ray workers |
| `tensor_model_parallel_size=2` | Shard vLLM across both GPUs |
| `load_format=safetensors` | vLLM preloads base weights from disk |
| `layered_summon=true` | Sync LoRA weights one layer at a time |
| `param_offload=true` | Keep FSDP params on CPU when not in forward/backward |
| `n_gpus_per_node=2` | Tell VERL to use 2 GPUs |

**Model-specific notes:**
- **Qwen 7B**: `ppo_micro_batch_size_per_gpu=8`, ~5.7 min/step at bs=8
- **Llama 8B**: `ppo_micro_batch_size_per_gpu=4` (OOMs at 8 on deep batches), ~8-10 min/step at bs=8
- Rollout is ~50s for bs=8 with K=8; training step dominates

## Dataset Preparation

VERL expects parquet with columns `{prompt, data_source, reward_model, extra_info}`.

### MATH-500

```bash
python training/scripts/prepare_math500.py
```

Creates `~/data/rlhf/math500/{train,test}.parquet` (400/100 split, seed 42).

### MATH Level 5

```bash
python training/scripts/prepare_math_level5.py [--n 500] [--train-ratio 0.8] [--seed 42]
```

Creates `~/data/rlhf/math_level5/{train,test}.parquet`.

### AIME

```bash
python training/scripts/prepare_aime.py
```

Creates `~/data/rlhf/aime/{train,test}.parquet`.

### GPQA, CSQA, MathQA (+ MATH-500 refresh)

```bash
python training/scripts/prepare_datasets.py [--datasets math500 gpqa csqa mathqa] [--n-train 400] [--n-test 100] [--seed 42]
```

Creates `~/data/rlhf/{dataset}/{train,test}.parquet` for each dataset.

- **math500**: 400/100 (from TREE local JSON)
- **gpqa**: 348/100 (only 448 total, from Wanfq/gpqa on HF)
- **csqa**: 400/100 (from tau/commonsense_qa on HF)
- **mathqa**: 400/100 (from swiss-ai/math_qa on HF)

MC datasets (gpqa, csqa, mathqa) have answers uppercased to match verl's case-sensitive
`is_equiv()`. GPQA requires `max_prompt_length=4096` due to long science prompts.

## Environment

- **Cluster**: A100 40GB GPUs
- **Conda env**: `scpo` at `/home/${USER}/miniconda3/envs/scpo`
- **Key deps**: Python 3.12, torch, vllm, verl (pip-installed in scpo env)
- **Critical**: `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` required for
  tensordict C extensions (set in launch script)
- **vLLM v1**: `VLLM_USE_V1=1` exported by launch script

## Experiment Tracking

After training, `save_experiment.py` collects artifacts into
`experiments/{experiment_name}_{YYYYMMDD_HHMMSS}/`:

```
experiments/
  math500_qwen7b_20260222_065839/
    config.yaml          # Full resolved Hydra config
    overrides.yaml       # CLI overrides
    metrics.json         # wandb-summary.json
    training_log.jsonl   # Per-step metrics
    checkpoint/          # Final LoRA adapter weights (moved, not copied)
```

Called automatically by `launch_training.sh` after training exits.
