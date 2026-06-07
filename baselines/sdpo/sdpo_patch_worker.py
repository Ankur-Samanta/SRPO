"""SDPO-full: patch the FSDP worker to wire up the EMA teacher module.

Upstream (lasgroup/SDPO/verl/workers/fsdp_workers.py) assigns
``self.actor.teacher_module = self.ref_module_fsdp`` at the end of ``init_model``
when SDPO is configured with EMA. That is: the teacher REUSES the ref model that
verl already builds. This file replicates that with two adaptations:

    1. LoRA sync: verl's ref and actor are both LoRA-wrapped (if configured) but
       ``peft`` initializes ``lora_A`` randomly, so the two start with
       DIFFERENT adapter weights. Paper-faithful SDPO needs teacher == actor at
       t=0, so we copy the actor's state_dict into the ref after both are built.

    2. LoRA-only EMA: ``_update_teacher`` iterates named_parameters and only
       EMA-averages parameters where the student has ``requires_grad=True``
       (the LoRA adapters). Base-model params remain frozen.

Reference:
    https://github.com/lasgroup/SDPO/blob/main/verl/workers/fsdp_workers.py#L893
"""

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def _patch_actor_config_for_self_distillation():
    """Make FSDPActorConfig.__init__ tolerate an extra ``self_distillation`` kwarg.

    Hydra instantiates FSDPActorConfig via ``_target_`` at startup with every
    key under ``actor_rollout_ref.actor`` as a kwarg. The base dataclass doesn't
    declare ``self_distillation``, so we wrap __init__ to peel it off and stash
    it on the instance. The worker then reads it via ``self.config.actor.self_distillation``.
    """
    from verl.workers.config.actor import FSDPActorConfig

    if getattr(FSDPActorConfig, "_sdpo_full_init_patched", False):
        return

    original_init = FSDPActorConfig.__init__

    def patched_init(self, *args, **kwargs):
        sd_cfg = kwargs.pop("self_distillation", None)
        original_init(self, *args, **kwargs)
        # stash on instance; BaseConfig.get() will surface it via self.config.actor.get("self_distillation")
        object.__setattr__(self, "self_distillation", sd_cfg)

    FSDPActorConfig.__init__ = patched_init
    FSDPActorConfig._sdpo_full_init_patched = True
    logger.info("SDPO-full: patched FSDPActorConfig.__init__ to accept self_distillation kwarg")


_patch_actor_config_for_self_distillation()


def _get_sdpo_cfg(worker) -> Optional[object]:
    """Return the SelfDistillationConfig if SDPO-full is active, else None.

    Only fires for loss_mode='sdpo_full'. Workers only see actor_rollout_ref,
    not the top-level algorithm config, so self_distillation must be under
    ``actor_rollout_ref.actor`` (added via Hydra ``+`` override in the launch
    script) -- not under ``algorithm`` where sdpo-light keeps it.
    """
    actor_cfg = getattr(worker.config, "actor", None)
    if actor_cfg is None:
        return None
    policy_loss = actor_cfg.get("policy_loss", None)
    if policy_loss is None or policy_loss.get("loss_mode", "vanilla") != "sdpo_full":
        return None
    sd_cfg = actor_cfg.get("self_distillation", None)
    if sd_cfg is None:
        logger.warning(
            "SDPO-full: loss_mode=sdpo_full but no self_distillation under "
            "actor_rollout_ref.actor. Add the Hydra override "
            "'+actor_rollout_ref.actor.self_distillation=${algorithm.self_distillation}' "
            "so the worker can see the config."
        )
    return sd_cfg


def _sync_actor_to_teacher(actor_fsdp_module, teacher_fsdp_module) -> None:
    """Copy the actor's full state_dict into the teacher (FSDP-aware).

    Uses FSDP2's fsdp2_load_full_state_dict when available; falls back to
    FSDP1 summon_full_params semantics otherwise.
    """
    from verl.utils.fsdp_utils import FSDPModule, fsdp_version

    if fsdp_version(actor_fsdp_module) == 2 or isinstance(teacher_fsdp_module, FSDPModule):
        from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

        full_state = actor_fsdp_module.state_dict()
        fsdp2_load_full_state_dict(teacher_fsdp_module, full_state, device_mesh=None, cpu_offload=None)
    else:
        # FSDP1 path: load_state_dict works if both use the same wrap policy.
        full_state = actor_fsdp_module.state_dict()
        teacher_fsdp_module.load_state_dict(full_state)


def _update_teacher(self) -> None:
    """EMA-update self.teacher_module toward self.actor_module.

    Iterates parameters by name and filters to those where the student has
    ``requires_grad=True`` (i.e. the LoRA adapters when LoRA is active, or all
    params when full finetuning). Teacher params are frozen; this just updates
    their ``.data`` in-place.

    Reference: lasgroup/SDPO dp_actor.py::_update_teacher
    """
    sd_cfg = getattr(self.config, "self_distillation", None)
    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
    if not sd_cfg or loss_mode != "sdpo":
        return
    teacher_regularization = getattr(sd_cfg, "teacher_regularization", "ema")
    if teacher_regularization != "ema":
        return
    update_rate = getattr(sd_cfg, "teacher_update_rate", 0.0)
    if update_rate == 0.0:
        return
    if self.teacher_module is None or self.teacher_module is self.actor_module:
        raise ValueError("SDPO EMA teacher requires a separate teacher_module on the actor.")

    student_params = dict(self.actor_module.named_parameters())
    with torch.no_grad():
        for t_name, t_param in self.teacher_module.named_parameters():
            s_param = student_params.get(t_name)
            if s_param is None:
                continue
            if not s_param.requires_grad:
                # LoRA mode: skip base-model params. Full FT: all params trainable.
                continue
            s_data = s_param.data.to(device=t_param.device)
            t_param.data.mul_(1.0 - update_rate).add_(s_data, alpha=update_rate)


def patch_worker_init_model():
    """Monkey-patch ActorRolloutRefWorker.init_model to set up the SDPO EMA teacher.

    After the original init_model finishes (actor + ref both built), if SDPO-full
    with EMA is configured:

        1. Sync actor's state_dict into ref (so teacher == actor at t=0).
        2. Freeze all teacher params (requires_grad=False).
        3. Assign self.actor.teacher_module = self.ref_module_fsdp.

    For teacher_regularization="ema" with update_rate=0.0 (no-update mode),
    teacher still needs to be a separate module (the frozen ref).
    """
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    # Add teacher_module slot + _update_teacher method to the actor class
    if not hasattr(DataParallelPPOActor, "_sdpo_full_teacher_patched"):
        # Ensure every actor instance has a teacher_module attribute
        original_init = DataParallelPPOActor.__init__

        def patched_init(self, config, actor_module, actor_optimizer=None):
            original_init(self, config, actor_module, actor_optimizer)
            self.teacher_module = None

        DataParallelPPOActor.__init__ = patched_init
        DataParallelPPOActor._update_teacher = _update_teacher
        DataParallelPPOActor._sdpo_full_teacher_patched = True
        logger.info("SDPO-full: added teacher_module slot + _update_teacher to DataParallelPPOActor")

    if hasattr(ActorRolloutRefWorker, "_sdpo_full_init_patched"):
        return

    # verl's @register decorator sets MAGIC_ATTR on the method so that
    # RayWorkerGroup.__getattr__ can dispatch method calls across workers.
    # Plain monkey-patching strips this, so copy it to our wrapper.
    from verl.single_controller.base.decorator import MAGIC_ATTR
    import functools

    original_init_model = ActorRolloutRefWorker.init_model
    original_magic = getattr(original_init_model, MAGIC_ATTR, None)

    @functools.wraps(original_init_model)
    def patched_init_model(self, *args, **kwargs):
        result = original_init_model(self, *args, **kwargs)

        sd_cfg = _get_sdpo_cfg(self)
        if sd_cfg is None:
            return result
        if not getattr(self, "_is_actor", False):
            return result
        teacher_regularization = sd_cfg.get("teacher_regularization", "ema") if hasattr(sd_cfg, "get") else getattr(sd_cfg, "teacher_regularization", "ema")
        if teacher_regularization == "trust-region":
            logger.warning("SDPO-full: trust-region teacher not implemented; falling back to EMA path.")
            teacher_regularization = "ema"

        # EMA mode: need a separate teacher module.
        ref_module = getattr(self, "ref_module_fsdp", None)
        if ref_module is None:
            logger.warning(
                "SDPO-full: no ref_module_fsdp available; teacher will be self.actor_module "
                "(equivalent to teacher_update_rate=0 on current-model). Configure "
                "actor_rollout_ref.ref to enable EMA teacher."
            )
            self.actor.teacher_module = self.actor_module_fsdp
            return result

        # Sync actor's state_dict into ref so teacher == actor at t=0 (LoRA adapters too).
        try:
            _sync_actor_to_teacher(self.actor_module_fsdp, ref_module)
            logger.info("SDPO-full: synced actor state_dict → teacher (ref_module_fsdp)")
        except Exception as exc:
            logger.warning(
                "SDPO-full: failed to sync actor→teacher state_dict (%s). "
                "Teacher will start from ref's independent init.", exc
            )

        # Freeze teacher params.
        for p in ref_module.parameters():
            p.requires_grad = False

        self.actor.teacher_module = ref_module
        logger.info("SDPO-full: wired ref_module_fsdp as EMA teacher (update_rate=%s)",
                    getattr(sd_cfg, "teacher_update_rate", "?"))
        return result

    # Restore verl's dispatch metadata so RayWorkerGroup can still call init_model.
    if original_magic is not None:
        setattr(patched_init_model, MAGIC_ATTR, original_magic)

    ActorRolloutRefWorker.init_model = patched_init_model
    ActorRolloutRefWorker._sdpo_full_init_patched = True
    logger.info("SDPO-full: patched ActorRolloutRefWorker.init_model for EMA teacher setup")
