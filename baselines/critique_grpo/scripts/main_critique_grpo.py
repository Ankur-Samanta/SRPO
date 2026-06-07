"""Critique-GRPO training entrypoint.

Patches verl's TaskRunner.run() to import baselines.critique_grpo inside
the Ray actor process where compute_advantage actually executes.
"""

import baselines.critique_grpo  # noqa: F401

from verl.trainer.main_ppo import TaskRunner, main

_original_run = TaskRunner.run


def _patched_run(self, config):
    import baselines.critique_grpo  # noqa: F401
    return _original_run(self, config)


TaskRunner.run = _patched_run

if __name__ == "__main__":
    main()
