"""Unit tests for SDPO core loss computation."""

import torch
import pytest


def test_self_distillation_loss_reverse_kl_logprob_only():
    """Non-full-logit mode: reverse KL on log-probs."""
    from baselines.sdpo.sdpo_core import (
        SelfDistillationConfig,
        compute_self_distillation_loss,
    )

    cfg = SelfDistillationConfig(
        full_logit_distillation=False,
        alpha=1.0,
        is_clip=None,
    )

    bs, seq_len = 4, 10
    student_log_probs = torch.randn(bs, seq_len) - 1.0
    teacher_log_probs = torch.randn(bs, seq_len) - 1.0
    response_mask = torch.ones(bs, seq_len)

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
    )

    assert loss.shape == ()
    assert loss.requires_grad


def test_self_distillation_mask_zeros_out_non_distilled():
    """Samples with mask=0 should not contribute to loss."""
    from baselines.sdpo.sdpo_core import (
        SelfDistillationConfig,
        compute_self_distillation_loss,
    )

    cfg = SelfDistillationConfig(
        full_logit_distillation=False,
        alpha=1.0,
        is_clip=None,
    )

    bs, seq_len = 4, 10
    student_log_probs = torch.randn(bs, seq_len) - 1.0
    teacher_log_probs = torch.randn(bs, seq_len) - 1.0
    response_mask = torch.ones(bs, seq_len)

    # Only first 2 samples have distillation targets
    self_distillation_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

    loss_masked, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        self_distillation_mask=self_distillation_mask,
    )

    # With all mask=0, loss should be 0
    all_zero_mask = torch.zeros(bs)
    loss_zero, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        self_distillation_mask=all_zero_mask,
    )

    assert loss_zero.item() == pytest.approx(0.0, abs=1e-6)


def test_is_clip_requires_old_log_probs():
    """IS clipping should raise if old_log_probs not provided."""
    from baselines.sdpo.sdpo_core import (
        SelfDistillationConfig,
        compute_self_distillation_loss,
    )

    cfg = SelfDistillationConfig(
        full_logit_distillation=False,
        alpha=1.0,
        is_clip=2.0,
    )

    bs, seq_len = 2, 5
    student_log_probs = torch.randn(bs, seq_len)
    teacher_log_probs = torch.randn(bs, seq_len)
    response_mask = torch.ones(bs, seq_len)

    with pytest.raises(ValueError, match="old_log_probs is required"):
        compute_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            self_distillation_config=cfg,
        )


def test_is_clip_with_old_log_probs():
    """IS clipping should work when old_log_probs provided."""
    from baselines.sdpo.sdpo_core import (
        SelfDistillationConfig,
        compute_self_distillation_loss,
    )

    cfg = SelfDistillationConfig(
        full_logit_distillation=False,
        alpha=1.0,
        is_clip=2.0,
    )

    bs, seq_len = 2, 5
    student_log_probs = torch.randn(bs, seq_len) - 1.0
    teacher_log_probs = torch.randn(bs, seq_len) - 1.0
    old_log_probs = torch.randn(bs, seq_len) - 1.0
    response_mask = torch.ones(bs, seq_len)

    loss, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        old_log_probs=old_log_probs,
    )

    assert loss.shape == ()
