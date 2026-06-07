"""Monkey-patch verl's RayPPOTrainer to keep best + last LoRA checkpoints only.

Works with verl's existing checkpoint logic (save_freq, test_freq, etc).
After each checkpoint save, compares the latest val-core reward to the
best seen so far. If it's not the best, deletes the checkpoint. The best
checkpoint is kept in a "best/" directory alongside the normal global_step_N/.

Setup:
  1. Set trainer.save_freq=1 and trainer.test_freq=1 in your config
     (so we get a checkpoint + val metric every step)
  2. Set trainer.max_actor_ckpt_to_keep=1 (verl keeps only latest step dir)
  3. Import this module before training:
       import training.best_checkpoint_patch

Result: at the end of training you get:
  {default_local_dir}/best/       — best checkpoint (copied when new best found)
  {default_local_dir}/global_step_N/  — last checkpoint (verl's default behavior)
"""

import os
import shutil

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

_original_validate = RayPPOTrainer._validate

# State tracked across calls
_best_reward = float("-inf")
_best_step = -1


def _get_val_core_reward(val_metrics: dict) -> float:
    """Extract val-core reward/mean@1 from validation metrics dict."""
    for key, value in val_metrics.items():
        if "val-core" in key and "reward/mean@1" in key:
            return float(value)
    return float("-inf")


def _patched_validate(self, merged=False):
    """After validation, check if current checkpoint is the best and copy it."""
    global _best_reward, _best_step

    val_metrics = _original_validate(self, merged=merged)

    reward = _get_val_core_reward(val_metrics)
    step = self.global_steps

    if reward > _best_reward:
        _best_reward = reward
        _best_step = step

        # Copy only the LoRA adapter to best/
        ckpt_dir = self.config.trainer.default_local_dir
        src_lora = os.path.join(ckpt_dir, f"global_step_{step}", "actor", "lora_adapter")
        dst = os.path.join(ckpt_dir, "best")
        dst_lora = os.path.join(dst, "actor", "lora_adapter")

        if os.path.exists(src_lora):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(os.path.join(dst, "actor"), exist_ok=True)
            shutil.copytree(src_lora, dst_lora)
            with open(os.path.join(dst, "best_metadata.txt"), "w") as f:
                f.write(f"global_step={step}\n")
                f.write(f"val_reward={reward}\n")
            print(f"[BestCkpt] New best: reward={reward:.4f} at step {step} → {dst_lora}")
        else:
            print(f"[BestCkpt] New best: reward={reward:.4f} at step {step} (no lora_adapter found)")
    else:
        print(f"[BestCkpt] Step {step}: reward={reward:.4f} (best={_best_reward:.4f} @ step {_best_step})")

    return val_metrics


RayPPOTrainer._validate = _patched_validate
print("[BestCkpt] Patched RayPPOTrainer._validate with best-checkpoint tracking")
