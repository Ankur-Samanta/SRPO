# ICS Rollout Buffer for SCGRPO

How Iterative Self-Correction (ICS) fills the rollout buffer that SCGRPO trains on.

## Files

| File | Role |
|------|------|
| `thought_agent_loop.py` | Base class. Thought-by-thought generation via vLLM with `</thought>` as stop token. Produces `_ThoughtChainResult` (token IDs, logprobs, thought boundaries). |
| `thought_ics_agent_loop.py` | Subclass. Adds ICS: error localization, backtracking, and prefix-based regeneration. Coordinates N rollout slots per prompt. |

## Data structures

**`_ThoughtChainResult`** — one trajectory:
- `prompt_ids` / `response_ids` — tokenized prompt and response
- `response_logprobs` — per-token log-probs (needed for GRPO importance weighting)
- `thought_boundaries` — list of `(start_idx, end_idx)` in `response_ids`, one per thought
- `decoded_thoughts` — text of each thought step
- `found_answer` — whether `\boxed{}` was produced

**`_ICSBuffer`** — per-prompt coordination state:
- `trajectories` — list of `_ThoughtChainResult`, filled by the coordinator
- `done` — `asyncio.Event`, set when the coordinator finishes
- `next_slot` — atomic counter for slot claiming

## Slot coordination

VERL spawns N rollout workers per prompt (N = `rollout_n`, typically 8). All N workers enter `ThoughtICSAgentLoop.run()` and claim a slot from a shared `_ICSBuffer` keyed by question text:

```
Slot 0 (coordinator):
    calls _fill_rollout_buffer()       # generates all N trajectories
    stores them in buffer.trajectories
    sets buffer.done

Slots 1..N-1 (followers):
    await buffer.done.wait()           # block until coordinator finishes
    take buffer.trajectories[my_slot]  # claim pre-generated trajectory
```

This avoids N independent generations per prompt. One worker does all the work; the rest just consume. If the coordinator fails to produce enough trajectories, a follower falls back to vanilla `_generate_thought_chain()`.

## `_fill_rollout_buffer()` — the core loop

Runs until `len(buffer) >= N`:

```
while len(buffer) < N:

    1. GENERATE fresh chain
       chain = _generate_thought_chain(sampling_params, question)
       buffer.append(chain)
       if len(buffer) >= N: break

    2. CHECK CORRECTNESS
       oracle_correct = _check_correctness(chain, ground_truth)
       (optionally also run self-verifier in shadow mode)
       if correct: continue to next fresh chain

    3. ICS CORRECTION LOOP (triggered only for wrong chains)
       budget = min(max_ics_iterations, N - len(buffer))

       for ics_iter in range(budget):

           a. LOCALIZE ERROR
              error_step = _localize_error(question, ground_truth, decoded_thoughts)
              - L1 autonomy: model sees ground truth answer
              - L2 autonomy: model only knows "your answer is wrong"
              - Returns 1-indexed step number (0 = no error found -> break)

           b. BACKTRACK
              Cut chain at thought_boundaries[error_step - 2]
              Keep thoughts[0 : error_step-1] as prefix

           c. REGENERATE from prefix
              correction = _generate_thought_chain_from_prefix(
                  question, prefix_response_ids, prefix_logprobs, ...)
              buffer.append(correction)

           d. EVALUATE correction
              if correct: break
              else: current_chain = correction, continue loop
```

Every trajectory — fresh or corrected — occupies a buffer slot. No rollout budget is wasted.

### What the buffer looks like after filling

For a prompt with N=8 and one wrong chain that takes 3 ICS iterations to correct:

```
slot 0: fresh chain (correct)       <- no ICS triggered
slot 1: fresh chain (wrong)         <- ICS triggered
slot 2: correction iter 1 (wrong)   <- backtrack + regenerate
slot 3: correction iter 2 (wrong)   <- backtrack + regenerate
slot 4: correction iter 3 (correct) <- ICS succeeded, break
slot 5: fresh chain (correct)       <- back to fresh generation
slot 6: fresh chain (correct)
slot 7: fresh chain (wrong)         <- ICS triggered, but buffer full
```

## Error localization

`_localize_error()` makes a single CoT call (no stop tokens, temperature 0.3) asking the model to identify the first erroneous step.

**L2 prompt** (default — binary feedback, no ground truth):
```
Problem: {question}

Current reasoning chain (WRONG - got incorrect answer):
Step 1: ...
Step 2: ...
...

Your answer is incorrect. Analyze the reasoning chain step by step
to identify where the error occurred. Which step number (1 to N)
contains the first critical error?

Put ONLY the step number in: \boxed{step_number}
```

The model's localization response is discarded after parsing `\boxed{}` — it is not training data for the policy (unless auxiliary localization training is enabled).

**Random localization baseline**: `random_localization=True` skips the LLM call and picks a uniform random step. Used as an ablation to measure localization quality.

## Backtracking and prefix-based regeneration

`_generate_thought_chain_from_prefix()` continues generation from a truncated prefix. Two modes:

**Without context** (`use_context=False`, default):
- Uses the original prompt + prefix tokens
- Relies on vLLM prefix caching for KV reuse
- Model has no explicit knowledge of the previous attempt

**With context** (`use_context=True`):
- Injects the failed chain and error analysis into the prompt:
  ```
  {original_prompt}

  ### Previous Failed Attempt
  The following reasoning chain led to an incorrect answer:
  Step 1: ...

  ### Error Analysis
  {error_reasoning}

  Now let's try again with the correct approach:
  ```
- After generation, swaps `prompt_ids` back to the original prompt for GRPO training consistency
- Runs a **scoring pass** to recompute on-policy log-probs under the original prompt (since tokens were generated conditioned on the context-enriched prompt)

## How SCGRPO consumes the buffer

Each of the N trajectories per prompt gets an outcome reward (1.0 if correct, 0.0 if wrong). GRPO computes group-normalized advantages:

```
advantage_i = (r_i - mean(r_1..N)) / (std(r_1..N) + eps)
```

ICS enriches these groups with **contrastive signal**: a wrong chain and its corrected version share the same prefix up to the error step, so the policy learns which continuations from that prefix lead to correct vs. incorrect answers. This is stronger than N independent samples because the contrast is structurally aligned.

The `thought_segment_ids` field (set in `_chain_result_to_output()`) maps each response token to its thought index. The GRPO loss function can use this to apply per-thought weighting or masking via `thought_grpo_loss.py`.

## Eval behavior

During validation (temperature=0), ICS is skipped entirely. The loop falls back to `super().run()` (vanilla `ThoughtAgentLoop`), generating a single chain without any correction. This means eval accuracy reflects the model's standalone performance, not ICS-assisted performance.

Exception: `force_ics_at_eval=True` overrides this for dedicated ICS evaluation runs.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_n` | from config | Number of trajectories per prompt (N) |
| `max_ics_iterations` | `rollout_n - 1` | Max correction attempts per wrong chain |
| `autonomy_level` | 2 | 1 = model sees ground truth; 2 = binary "you're wrong" |
| `use_context` | False | Inject failed chain + error analysis into regeneration prompt |
| `localization_temp` | 0.3 | Temperature for error localization call |
| `localization_max_tokens` | 2048 | Max tokens for localization response |
| `random_localization` | False | Skip LLM, pick random error step (ablation) |
| `force_ics_at_eval` | False | Run ICS during validation |

## Auxiliary training modes

These are orthogonal to the main SCGRPO training and use data collected during ICS:

| Mode | Data source | Signal |
|------|-------------|--------|
| `train_localization` | Successful localization prompts | GRPO on K rollouts, reward = (predicted step == ground truth step) |
| `train_loc_ppo` | All localization prompts | PPO with reward = correction outcome (1.0 if fix worked) |
| `train_loc_sft` | Successful localizations only | NLL on localization response |
| `train_loc_kto` | All localizations | KTO sigmoid loss (success = desirable, failure = undesirable) |
| `train_verifier_*` | Self-verification prompts | Shadow-mode: reward = (verifier prediction == oracle), does not gate ICS |

Only one localization mode and one verifier mode can be active at a time.
