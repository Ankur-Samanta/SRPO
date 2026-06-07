#!/bin/bash
# Smoke test all baselines: SCoRe, Critique-GRPO, SPO
#
# Runs each baseline for 1 epoch on MATH-500 with minimal settings:
#   - Qwen 2.5 7B, 2 GPUs, TP=2
#   - train_batch_size=4, rollout n=4
#   - No checkpointing, no validation, no wandb
#
# Usage:
#   bash batch_scripts/smoke_baselines.sh <baseline> [--local]
#   bash batch_scripts/smoke_baselines.sh all [--local]
#
# Baselines: score, critique_grpo, spo, all

set -e

SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"

mkdir -p "${SCPO_DIR}/batch_scripts/logs"

export VERL_LOGGING_LEVEL=INFO
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ─── Shared smoke overrides ────────────────────────────────────────────────
SMOKE_OVERRIDES=(
    actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    trainer.n_gpus_per_node=2
    # Minimal settings for smoke test
    trainer.total_epochs=1
    trainer.save_freq=999
    trainer.test_freq=999
    trainer.val_before_train=False
    'trainer.logger=["console"]'
    data.train_batch_size=4
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
)

# ─── Per-baseline smoke commands ───────────────────────────────────────────

smoke_score() {
    echo "=== Smoke: SCoRe ==="
    bash "${SCPO_DIR}/baselines/score/scripts/launch_training.sh" \
        "${SMOKE_OVERRIDES[@]}" \
        actor_rollout_ref.model.external_lib=baselines.score \
        actor_rollout_ref.rollout.agent.default_agent_loop=score_agent \
        actor_rollout_ref.rollout.agent.agent_loop_config_path=baselines/score/config/score_agent_config.yaml \
        trainer.project_name=smoke_baselines \
        trainer.experiment_name=smoke_score
}

smoke_critique_grpo() {
    echo "=== Smoke: Critique-GRPO ==="
    bash "${SCPO_DIR}/baselines/critique_grpo/scripts/launch_training.sh" \
        "${SMOKE_OVERRIDES[@]}" \
        actor_rollout_ref.model.external_lib=baselines.critique_grpo \
        actor_rollout_ref.rollout.agent.default_agent_loop=critique_grpo_agent \
        actor_rollout_ref.rollout.agent.agent_loop_config_path=baselines/critique_grpo/config/critique_grpo_agent_config.yaml \
        trainer.project_name=smoke_baselines \
        trainer.experiment_name=smoke_critique_grpo
}

smoke_spo() {
    echo "=== Smoke: SPO-Tree ==="
    bash "${SCPO_DIR}/baselines/spo/scripts/launch_tree_training.sh" \
        "${SMOKE_OVERRIDES[@]}" \
        actor_rollout_ref.model.external_lib=baselines.spo \
        actor_rollout_ref.rollout.agent.default_agent_loop=spo_tree_agent \
        actor_rollout_ref.rollout.agent.agent_loop_config_path=baselines/spo/config/spo_tree_agent_config.yaml \
        trainer.project_name=smoke_baselines \
        trainer.experiment_name=smoke_spo_tree
}

# ─── SLURM submission ──────────────────────────────────────────────────────

submit() {
    local job_name=$1
    sbatch --partition=q1 \
           --nodes=1 \
           --gpus-per-node=2 \
           --cpus-per-gpu=10 \
           --time=01:00:00 \
           --job-name="${job_name}" \
           --output="batch_scripts/logs/${job_name}.out" \
           --error="batch_scripts/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SCPO_DIR} && bash batch_scripts/smoke_baselines.sh ${job_name#smoke_} --local'"
    echo "Submitted: ${job_name}"
}

# ─── Dispatch ──────────────────────────────────────────────────────────────

BASELINE="${1:?Usage: $0 <score|critique_grpo|spo|all> [--local]}"
MODE="${2:-slurm}"

case $BASELINE in
    score)
        [ "$MODE" = "--local" ] && smoke_score || submit smoke_score ;;
    critique_grpo)
        [ "$MODE" = "--local" ] && smoke_critique_grpo || submit smoke_critique_grpo ;;
    spo)
        [ "$MODE" = "--local" ] && smoke_spo || submit smoke_spo ;;
    all)
        if [ "$MODE" = "--local" ]; then
            smoke_score
            smoke_critique_grpo
            smoke_spo
        else
            submit smoke_score
            submit smoke_critique_grpo
            submit smoke_spo
        fi
        ;;
    *)
        echo "Unknown baseline: $BASELINE"
        echo "Usage: $0 <score|critique_grpo|spo|all> [--local]"
        exit 1
        ;;
esac
