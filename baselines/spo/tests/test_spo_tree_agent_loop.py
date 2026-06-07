"""Tests for SPO-tree agent loop: path decoding, tree-state coordination,
segment budget, and output layout.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tok.decode = MagicMock(
        side_effect=lambda ids, skip_special_tokens=False: f"decoded_{len(ids)}_tokens"
    )
    return tok


def _make_mock_config(response_length=5000, prompt_length=100):
    config = MagicMock()
    config.actor_rollout_ref.rollout.prompt_length = prompt_length
    config.actor_rollout_ref.rollout.response_length = response_length
    return config


def _make_tree_loop(
    branching_factors=(2, 2, 2),
    segment_tokens=10,
    response_length=200,
    mc_temperature=0.7,
):
    from baselines.spo.spo_tree_agent_loop import SPOTreeAgentLoop

    config = _make_mock_config(response_length=response_length)
    with patch.object(SPOTreeAgentLoop, "__init__", lambda self, *a, **kw: None):
        loop = SPOTreeAgentLoop.__new__(SPOTreeAgentLoop)
    loop.B = tuple(branching_factors)
    loop.L = len(loop.B)
    loop.N = 1
    for b in loop.B:
        loop.N *= b
    loop.M = segment_tokens
    loop.mc_temperature = mc_temperature
    loop.prompt_length = config.actor_rollout_ref.rollout.prompt_length
    loop.response_length = response_length
    loop.template = "Q: {question}\n"
    loop.tokenizer = _make_mock_tokenizer()
    loop.server_manager = AsyncMock()
    loop.config = config
    loop.loop = asyncio.new_event_loop()
    # Reset the class-level registry between tests.
    from baselines.spo.spo_tree_agent_loop import _TreeRegistry
    SPOTreeAgentLoop._registry = _TreeRegistry()
    return loop


# ---------------------------------------------------------------------------
# Path decoding
# ---------------------------------------------------------------------------


class TestDecodePath:
    def test_depth3_round_trip(self):
        """Every leaf index decodes to a unique path."""
        loop = _make_tree_loop(branching_factors=(2, 3, 4))  # N = 24
        seen = set()
        for idx in range(loop.N):
            path = loop._decode_path(idx)
            assert len(path) == 3
            for d, bd in enumerate(loop.B):
                assert 0 <= path[d] < bd
            assert path not in seen
            seen.add(path)
        assert len(seen) == loop.N

    def test_depth1(self):
        loop = _make_tree_loop(branching_factors=(5,))
        for i in range(5):
            assert loop._decode_path(i) == (i,)

    def test_out_of_range(self):
        loop = _make_tree_loop(branching_factors=(2, 2))  # N=4
        with pytest.raises(ValueError):
            loop._decode_path(4)
        with pytest.raises(ValueError):
            loop._decode_path(-1)

    def test_stride_ordering(self):
        """Leaf 0 → all zeros; leaf N-1 → (B_0-1, B_1-1, ...)."""
        loop = _make_tree_loop(branching_factors=(3, 4, 5))  # N=60
        assert loop._decode_path(0) == (0, 0, 0)
        assert loop._decode_path(59) == (2, 3, 4)


# ---------------------------------------------------------------------------
# Segment budget
# ---------------------------------------------------------------------------


class TestSegmentBudget:
    def test_interior_capped_at_M(self):
        loop = _make_tree_loop(segment_tokens=10, response_length=200)
        # Prefix has 5 prompt tokens + 0 response tokens → response_used=-5?
        # Our internal accounting uses prefix_len (including prompt) and
        # subtracts prompt_len. We pass prompt_len=5 via mock.
        budget = loop._segment_budget(prefix_len=5, is_leaf_depth=False, prompt_len=5)
        assert budget == 10

    def test_leaf_uses_remaining(self):
        loop = _make_tree_loop(segment_tokens=10, response_length=100)
        # After 2 interior segments (20 tokens used), leaf should get 80.
        budget = loop._segment_budget(
            prefix_len=5 + 20, is_leaf_depth=True, prompt_len=5
        )
        assert budget == 80

    def test_exhausted_budget_returns_zero(self):
        loop = _make_tree_loop(segment_tokens=10, response_length=20)
        # response_used = 25 − 5 = 20, remaining = 0 → budget 0.
        budget = loop._segment_budget(
            prefix_len=25, is_leaf_depth=True, prompt_len=5
        )
        assert budget == 0

    def test_interior_clamped_when_near_end(self):
        loop = _make_tree_loop(segment_tokens=10, response_length=100)
        # response_used=95, remaining=5 → interior returns min(10, 5) = 5.
        budget = loop._segment_budget(
            prefix_len=5 + 95, is_leaf_depth=False, prompt_len=5
        )
        assert budget == 5


# ---------------------------------------------------------------------------
# _build_output
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def test_extra_fields_present(self):
        loop = _make_tree_loop(branching_factors=(2, 2, 2), segment_tokens=10,
                               response_length=40)
        # 25 response tokens: 10 in level 1, 10 in level 2, 5 in leaf.
        resp = list(range(25))
        logprobs = [-0.1] * 25

        output = loop._build_output(
            prompt_ids=[1, 2, 3],
            response_ids=resp,
            response_logprobs=logprobs,
            path=(0, 1, 1),
            found_answer=True,
            uid="prompt_abc",
            generate_duration=0.2,
        )

        ef = output.extra_fields
        assert ef["tree_path"] == [0, 1, 1]
        assert ef["tree_branching"] == [2, 2, 2]
        # Segment positions: [10, 20, 25] (leaf extends to response length 25).
        assert ef["segment_positions"] == [10, 20, 25]
        # Per-token segment ids: 10 of seg 1, 10 of seg 2, 5 of seg 3, padding 0.
        seg_ids = ef["segment_token_ids"]
        assert len(seg_ids) == loop.response_length
        assert seg_ids[:10] == [1] * 10
        assert seg_ids[10:20] == [2] * 10
        assert seg_ids[20:25] == [3] * 5
        assert all(s == 0 for s in seg_ids[25:])
        assert ef["uid"] == "prompt_abc"
        assert ef["found_answer"] is True
        assert ef["num_segments"] == 3

    def test_leaf_seg_extends_to_response_end(self):
        """Leaf segment covers from (L-1)·M to end of response."""
        loop = _make_tree_loop(branching_factors=(2, 2), segment_tokens=5,
                               response_length=50)
        resp = list(range(40))  # 5 for segment 1 + 35 leaf
        out = loop._build_output(
            prompt_ids=[1], response_ids=resp, response_logprobs=[0.0] * 40,
            path=(1, 0), found_answer=False, uid=None, generate_duration=0.0,
        )
        seg = out.extra_fields["segment_token_ids"]
        assert seg[:5] == [1] * 5
        assert seg[5:40] == [2] * 35


# ---------------------------------------------------------------------------
# Tree coordination
# ---------------------------------------------------------------------------


class TestTreeCoordination:
    def test_segment_cached_across_callers(self):
        """The same node_key generates its segment exactly once."""
        from baselines.spo.spo_tree_agent_loop import _TreeState, _NodeResult

        state = _TreeState(
            prompt_ids=[1, 2],
            branching=(2, 2),
            total_leaves=4,
        )
        call_count = {"n": 0}

        async def slow_gen(prefix):
            call_count["n"] += 1
            await asyncio.sleep(0)
            return _NodeResult(token_ids=[99], logprobs=[-1.0])

        async def driver():
            # Two concurrent callers request the same node.
            r1, r2 = await asyncio.gather(
                state.get_or_generate_segment((0,), [1, 2], slow_gen),
                state.get_or_generate_segment((0,), [1, 2], slow_gen),
            )
            return r1, r2

        loop = asyncio.new_event_loop()
        try:
            r1, r2 = loop.run_until_complete(driver())
        finally:
            loop.close()

        assert r1.token_ids == [99]
        assert r2.token_ids == [99]
        assert call_count["n"] == 1

    def test_distinct_nodes_both_generate(self):
        from baselines.spo.spo_tree_agent_loop import _TreeState, _NodeResult

        state = _TreeState(
            prompt_ids=[1],
            branching=(2, 2),
            total_leaves=4,
        )
        calls = []

        async def gen(prefix):
            calls.append(tuple(prefix))
            return _NodeResult(token_ids=[len(calls)], logprobs=[0.0])

        async def driver():
            r_a = await state.get_or_generate_segment((0,), [1], gen)
            r_b = await state.get_or_generate_segment((1,), [1], gen)
            return r_a, r_b

        loop = asyncio.new_event_loop()
        try:
            r_a, r_b = loop.run_until_complete(driver())
        finally:
            loop.close()

        assert r_a.token_ids != r_b.token_ids
        assert len(calls) == 2

    def test_claim_leaf_index_unique(self):
        from baselines.spo.spo_tree_agent_loop import _TreeState

        state = _TreeState(
            prompt_ids=[1], branching=(4,), total_leaves=4,
        )

        async def driver():
            indices = await asyncio.gather(*[state.claim_leaf_index() for _ in range(4)])
            return indices

        loop = asyncio.new_event_loop()
        try:
            indices = loop.run_until_complete(driver())
        finally:
            loop.close()

        assert sorted(indices) == [0, 1, 2, 3]

    def test_claim_resets_on_overflow(self):
        """Claims past total_leaves reset the counter (and clear the cache)
        rather than raising — protects against cross-step key collisions.
        """
        from baselines.spo.spo_tree_agent_loop import _TreeState, _NodeResult

        state = _TreeState(
            prompt_ids=[1], branching=(2, 2), total_leaves=4,
        )
        # Pre-populate the cache to confirm it gets cleared on reset.
        state._node_cache[(0,)] = _NodeResult(token_ids=[99], logprobs=[0.0])

        async def driver():
            first_batch = [await state.claim_leaf_index() for _ in range(4)]
            # Next claim would overflow (idx=4 into N=4); must reset to 0.
            overflow = await state.claim_leaf_index()
            return first_batch, overflow

        loop = asyncio.new_event_loop()
        try:
            first_batch, overflow = loop.run_until_complete(driver())
        finally:
            loop.close()

        assert sorted(first_batch) == [0, 1, 2, 3]
        assert overflow == 0  # reset back to 0
        assert (0,) not in state._node_cache  # cache cleared
