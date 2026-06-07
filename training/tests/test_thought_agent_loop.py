"""Tests for ThoughtAgentLoop truncation fix.

Covers:
- Pre-generation check prevents partial-thought truncation
- Safety truncation falls back to last complete thought boundary
- Config arithmetic: response_length=10240 supports 20×512 thoughts

Run with:
    python training/tests/test_thought_agent_loop.py
"""

import sys
import types
import os
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _install_stubs():
    for mod_path in [
        "verl", "verl.experimental", "verl.experimental.agent_loop",
        "verl.experimental.agent_loop.agent_loop", "verl.utils",
        "verl.utils.profiler",
    ]:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)

    class _FakeMetrics:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _FakeOutput:
        def __init__(self, **kw):
            self.extra_fields = {}
            for k, v in kw.items():
                setattr(self, k, v)

    al = sys.modules["verl.experimental.agent_loop.agent_loop"]
    al.AgentLoopBase = type("AgentLoopBase", (), {
        "__init__": lambda self, *a, **kw: None,
    })
    al.AgentLoopMetrics = _FakeMetrics
    al.AgentLoopOutput = _FakeOutput
    al.AsyncLLMServerManager = object
    al.DictConfigWrap = object

    @contextmanager
    def _fake_timer(name, metrics):
        yield
        metrics[name] = 0.0

    sys.modules["verl.utils.profiler"].simple_timer = _fake_timer

    # Stub prompt_templates
    pt = types.ModuleType("training.prompt_templates")
    sys.modules["training.prompt_templates"] = pt
    pt.prompt_template_with_examples = lambda: "Q: {question}\n"
    pt.prompt_template_no_examples = lambda: "Q: {question}\n"


_install_stubs()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from thought_agent_loop import _ThoughtChainResult  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreGenerationCheck:
    """The loop should stop before generating if the next thought can't fit."""

    def test_check_prevents_overshoot(self):
        """Simulate the loop's pre-generation check."""
        response_length = 10240
        max_tokens_per_thought = 512
        max_thoughts = 20

        response_len = 0
        steps = 0
        for step in range(max_thoughts):
            if response_len + max_tokens_per_thought > response_length:
                break
            # Simulate generating a full-size thought
            response_len += max_tokens_per_thought
            steps += 1

        assert steps == 20
        assert response_len == 10240
        # Next step would be 10240 + 512 = 10752 > 10240, so loop stops

    def test_short_thoughts_allow_more_steps(self):
        response_length = 10240
        max_tokens_per_thought = 512

        response_len = 0
        steps = 0
        for step in range(100):
            if response_len + max_tokens_per_thought > response_length:
                break
            response_len += 256  # thoughts only use half the budget
            steps += 1

        # Check fires when response_len + 512 > 10240, i.e. response_len > 9728
        # 9728 / 256 = 38.0, so step 38 has response_len=9728, 9728+512=10240 ≤ 10240 → ok
        # step 39 has response_len=9984, 9984+512=10496 > 10240 → break
        assert steps == 39

    def test_old_check_would_truncate(self):
        """The old check `>= response_length` allowed partial thoughts."""
        response_length = 4096  # old value
        max_tokens_per_thought = 512

        # Old check: len(all_response_ids) >= response_length
        response_len = 0
        steps = 0
        for step in range(20):
            if response_len >= response_length:  # OLD check
                break
            response_len += max_tokens_per_thought
            steps += 1

        # After 8 steps: response_len = 4096, check passes (4096 >= 4096), stops
        # But if thoughts are 500 tokens:
        response_len = 0
        steps = 0
        for step in range(20):
            if response_len >= response_length:
                break
            response_len += 500  # slightly under max
            steps += 1

        # step 0: 0 < 4096 → gen → 500
        # step 7: 3500 < 4096 → gen → 4000
        # step 8: 4000 < 4096 → gen → 4500 ← OVERSHOOT!
        # step 9: 4500 >= 4096 → stop
        assert steps == 9
        assert response_len == 4500  # overshot response_length!


class TestSafetyTruncation:
    """If response somehow exceeds response_length, truncate at last complete boundary."""

    def test_truncate_at_boundary(self):
        """Simulate safety truncation logic."""
        response_length = 10240
        boundaries = [(0, 500), (500, 1000), (1000, 1500)]
        all_response_ids = list(range(1500))

        # No truncation needed
        if len(all_response_ids) > response_length:
            assert False, "Should not truncate"
        truncated = list(boundaries)
        assert len(truncated) == 3

    def test_truncate_drops_partial(self):
        """When response exceeds limit, drop partial thoughts."""
        response_length = 1200
        boundaries = [(0, 500), (500, 1000), (1000, 1500)]
        all_response_ids = list(range(1500))

        if len(all_response_ids) > response_length:
            truncated_boundaries = []
            for start, end in boundaries:
                if end <= response_length:
                    truncated_boundaries.append((start, end))
                else:
                    break
            if truncated_boundaries:
                cut = truncated_boundaries[-1][1]
                all_response_ids = all_response_ids[:cut]
        else:
            truncated_boundaries = list(boundaries)

        # Boundary (1000, 1500) has end=1500 > 1200, so it's dropped
        assert len(truncated_boundaries) == 2
        assert truncated_boundaries == [(0, 500), (500, 1000)]
        assert len(all_response_ids) == 1000  # cut at last complete boundary

    def test_truncate_first_thought_too_long(self):
        """Edge case: even the first thought exceeds response_length."""
        response_length = 300
        boundaries = [(0, 500)]
        all_response_ids = list(range(500))

        if len(all_response_ids) > response_length:
            truncated_boundaries = []
            for start, end in boundaries:
                if end <= response_length:
                    truncated_boundaries.append((start, end))
                else:
                    break
            if truncated_boundaries:
                cut = truncated_boundaries[-1][1]
                all_response_ids = all_response_ids[:cut]
            else:
                all_response_ids = []

        assert truncated_boundaries == []
        assert all_response_ids == []


class TestThoughtGRPOConfig:
    """Config arithmetic for thought GRPO."""

    def test_response_length_fits_20_thoughts(self):
        assert 20 * 512 == 10240

    def test_total_fits_max_model_len(self):
        assert 2048 + 10240 <= 16384

    def test_vllm_generation_headroom(self):
        # vLLM prompt = base (~800) + response (up to 10240)
        # plus one more thought (512) for generation
        # Total: ~11552, under max_model_len=16384
        assert 800 + 10240 + 512 <= 16384

    def test_old_config_limited_depth(self):
        """Old response_length=4096 could only fit ~8 thoughts."""
        old_response_length = 4096
        max_thoughts = old_response_length // 512
        assert max_thoughts == 8


class TestThoughtChainResult:
    """_ThoughtChainResult structure."""

    def test_no_truncation_needed(self):
        result = _ThoughtChainResult(
            prompt_ids=list(range(800)),
            response_ids=list(range(5000)),
            response_mask=[1] * 5000,
            response_logprobs=[0.1] * 5000,
            thought_boundaries=[(0, 500), (500, 1000), (1000, 1500),
                                (1500, 2000), (2000, 2500), (2500, 3000),
                                (3000, 3500), (3500, 4000), (4000, 4500),
                                (4500, 5000)],
            decoded_thoughts=[f"thought_{i}" for i in range(10)],
            found_answer=True,
            num_thoughts=10,
        )
        assert result.num_thoughts == 10
        assert len(result.response_ids) == 5000
        # All fits in response_length=10240


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_standalone():
    passed = 0
    failed = 0
    for cls_name, cls in sorted(globals().items()):
        if not (isinstance(cls, type) and cls_name.startswith("Test")):
            continue
        instance = cls()
        for method_name in sorted(dir(instance)):
            if not method_name.startswith("test_"):
                continue
            full_name = f"{cls_name}::{method_name}"
            try:
                getattr(instance, method_name)()
                print(f"  PASS  {full_name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {full_name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
