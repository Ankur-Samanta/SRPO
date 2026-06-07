"""SCoRe training entrypoint.

Patches verl's TaskRunner.run() to import baselines.score inside the Ray
actor process where compute_advantage actually executes. Without this,
the monkey-patch only applies in the launcher process (which is not the
same process as the Ray TaskRunner actor).
"""

import baselines.score  # noqa: F401 — registers in launcher (for pre-flight validation)

from verl.trainer.main_ppo import TaskRunner, main

_original_run = TaskRunner.run


def _patched_run(self, config):
    import baselines.score  # noqa: F401 — re-import inside Ray actor to apply patch
    return _original_run(self, config)


TaskRunner.run = _patched_run

if __name__ == "__main__":
    main()
