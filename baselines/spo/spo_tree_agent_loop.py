"""SPO-tree agent loop (paper arXiv:2505.23564 §5).

Generates the tree-structured rollout described in Section 5. Each prompt
produces N = ∏ B_l leaf trajectories that share ancestor segments in a
balanced tree of depth L = |B| and branching factors B_0, B_1, ..., B_{L-1}.

Partition: fixed-length (paper §5.1) — each tree level adds exactly M tokens
(``segment_tokens``). The cutpoint-based partition is exclusive to SPO-chain.

Tree coordination: one ``_TreeState`` instance per prompt (keyed on the
prompt ids) holds the generated segment cache, per-node asyncio locks, and
an atomic leaf-index counter. VERL fans out N agent-loop coroutines per
prompt (``rollout.n = ∏B``); each claims a unique leaf index and walks its
root-to-leaf path, generating a segment only when it is the first to reach
that node. All others await the cached result, so each interior node is
sampled exactly once regardless of how many descendants traverse it.

After depth L the leaf continues to termination (its segment plus the rest
of the trajectory until EOS or the response budget). The trajectory's
outcome reward becomes V̂(leaf) downstream.

Emits:
    response_ids         : prompt-free leaf trajectory (all segments + tail)
    response_logprobs    : token logprobs from vLLM
    extra_fields:
        tree_path          : tuple[int] of length L, child index at each depth
        tree_branching     : tuple[int] of length L, the B vector
        segment_positions  : [M, 2M, ..., L·M] — absolute token positions
                             where each level's segment ends (segment k
                             spans [(k-1)·M, k·M))
        segment_token_ids  : (response_length,) 1-indexed segment id per token;
                             tokens past L·M get id L (leaf segment extends)
        uid                : prompt uid (for sibling grouping downstream)

Reference: SPO (arXiv:2505.23564), §5.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
)
from verl.utils.profiler import simple_timer

from training.prompt_templates import (
    prompt_template_no_examples,
    prompt_template_with_examples,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Per-prompt tree state
# ---------------------------------------------------------------------------


@dataclass
class _NodeResult:
    """Cached output for a generated tree segment."""

    token_ids: list[int]
    logprobs: list[float]


@dataclass
class _TreeState:
    """Shared tree state for a single prompt (across all N agent loops)."""

    prompt_ids: list[int]
    branching: tuple[int, ...]
    total_leaves: int

    _leaf_counter: int = 0
    _counter_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _node_cache: dict[tuple[int, ...], _NodeResult] = field(default_factory=dict)
    _node_locks: dict[tuple[int, ...], asyncio.Lock] = field(default_factory=dict)
    _locks_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def claim_leaf_index(self) -> int:
        async with self._counter_lock:
            if self._leaf_counter >= self.total_leaves:
                # Tree state was reused past its leaf budget — happens when
                # two different (step, prompt) invocations collide on the
                # same registry key (e.g., prompt-hash key matches across
                # steps). Reset counter and cache so this call sees a fresh
                # tree. Sibling sharing within the original step is lost
                # for the overflow caller, but the algorithm remains valid.
                logger.warning(
                    "[SPO-tree] tree state reused past N=%d; resetting counter + cache.",
                    self.total_leaves,
                )
                self._leaf_counter = 0
                self._node_cache.clear()
                self._node_locks.clear()
            idx = self._leaf_counter
            self._leaf_counter += 1
            return idx

    async def _get_node_lock(self, node_key: tuple[int, ...]) -> asyncio.Lock:
        async with self._locks_lock:
            lock = self._node_locks.get(node_key)
            if lock is None:
                lock = asyncio.Lock()
                self._node_locks[node_key] = lock
            return lock

    async def get_or_generate_segment(
        self,
        node_key: tuple[int, ...],
        prefix_ids: list[int],
        generate_fn,
    ) -> _NodeResult:
        """Return cached segment for node_key; generate via generate_fn if new.

        Only the first caller generates; concurrent callers block on the per-
        node lock and then read the populated cache entry.
        """
        if node_key in self._node_cache:
            return self._node_cache[node_key]

        node_lock = await self._get_node_lock(node_key)
        async with node_lock:
            if node_key in self._node_cache:
                return self._node_cache[node_key]
            result = await generate_fn(prefix_ids)
            self._node_cache[node_key] = result
            return result


class _TreeRegistry:
    """Class-level registry of per-prompt tree states."""

    def __init__(self):
        self._states: dict[int, _TreeState] = {}
        self._reg_lock = asyncio.Lock()

    async def get_or_create(
        self,
        key: int,
        prompt_ids: list[int],
        branching: tuple[int, ...],
        total_leaves: int,
    ) -> _TreeState:
        async with self._reg_lock:
            state = self._states.get(key)
            if state is None:
                state = _TreeState(
                    prompt_ids=prompt_ids,
                    branching=branching,
                    total_leaves=total_leaves,
                )
                self._states[key] = state
            return state

    async def evict(self, key: int) -> None:
        async with self._reg_lock:
            self._states.pop(key, None)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class SPOTreeAgentLoop(AgentLoopBase):
    """SPO-tree agent loop (paper §5).

    One invocation = one leaf trajectory. Per-prompt tree coordination is
    done through a class-level registry so sibling leaves share ancestor
    segments.
    """

    # Class-level shared registry — outlives individual agent loop instances.
    _registry: _TreeRegistry = _TreeRegistry()

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        branching_factors: list[int] = (6, 6, 6),
        segment_tokens: int = 600,
        mc_temperature: float = 0.7,
        use_examples: bool = True,
        **kwargs,
    ):
        """
        Args:
            branching_factors: B = [B_0, ..., B_{L-1}]. Number of children at
                each depth. Total leaves = ∏B = rollout.n.
            segment_tokens: M. Every level generates M tokens before branching.
                Leaves continue past M to termination.
            mc_temperature: Sampling temperature for tree segments. The paper
                uses π_θ_old directly; we expose it here for compatibility
                with the existing training-rollout temperature.
            use_examples: Whether to use the few-shot thought template.
        """
        super().__init__(trainer_config, server_manager, **kwargs)
        self.B = tuple(int(b) for b in branching_factors)
        assert all(b >= 1 for b in self.B), "all branching factors must be ≥ 1"
        self.L = len(self.B)
        self.N = int(math.prod(self.B))
        self.M = int(segment_tokens)
        self.mc_temperature = float(mc_temperature)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        if use_examples:
            self.template = prompt_template_with_examples()
        else:
            self.template = prompt_template_no_examples()

    # ------------------------------------------------------------------
    # Path decoding
    # ------------------------------------------------------------------

    def _decode_path(self, leaf_idx: int) -> tuple[int, ...]:
        """Convert leaf index in [0, N) to (i_0, i_1, ..., i_{L-1})."""
        if leaf_idx < 0 or leaf_idx >= self.N:
            raise ValueError(
                f"leaf_idx={leaf_idx} out of range [0, {self.N})"
            )
        path: list[int] = []
        remaining = leaf_idx
        # Iterate from deepest to shallowest using place values ∏_{l>k} B_l.
        strides: list[int] = [1] * self.L
        for k in range(self.L - 2, -1, -1):
            strides[k] = strides[k + 1] * self.B[k + 1]
        for k in range(self.L):
            path.append(remaining // strides[k])
            remaining = remaining % strides[k]
        return tuple(path)

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        messages = kwargs["raw_prompt"]
        uid = kwargs.get("uid", None)
        question = self._extract_question(messages)

        metrics: dict[str, Any] = {}

        prompt_text = self.template.format(question=question)
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(prompt_text, add_special_tokens=True),
        )
        prompt_ids = list(prompt_ids)

        # Prefer the prompt uid as the tree-state key: siblings (same prompt,
        # same step) share uid → share tree; different steps get distinct
        # uids → distinct trees. Fall back to prompt-hash if uid is absent
        # (overflow-reset in _TreeState handles cross-step collisions).
        if uid is not None:
            tree_key = hash(("uid", str(uid)))
        else:
            tree_key = hash(("ph", tuple(prompt_ids)))
        state = await self._registry.get_or_create(
            tree_key, prompt_ids, self.B, self.N
        )

        leaf_idx = await state.claim_leaf_index()
        path = self._decode_path(leaf_idx)

        is_validation = sampling_params.get("temperature", 1.0) == 0

        # ----- Walk the tree, generating/reading interior segments -----
        segments: list[_NodeResult] = []
        accumulated = list(prompt_ids)
        with simple_timer("generate_sequences", metrics):
            for depth in range(self.L):
                node_key = path[: depth + 1]
                is_leaf_depth = depth == self.L - 1

                async def _gen(prefix, _leaf=is_leaf_depth):
                    return await self._generate_segment(
                        prefix,
                        sampling_params,
                        max_new_tokens=self._segment_budget(
                            prefix_len=len(prefix),
                            is_leaf_depth=_leaf,
                            prompt_len=len(prompt_ids),
                        ),
                        is_leaf_depth=_leaf,
                        validation=is_validation,
                    )

                seg = await state.get_or_generate_segment(node_key, accumulated, _gen)
                segments.append(seg)
                accumulated = accumulated + seg.token_ids
                if is_leaf_depth:
                    break

        # Compose response (everything after the prompt).
        response_ids: list[int] = []
        response_logprobs: list[float] = []
        for seg in segments:
            response_ids.extend(seg.token_ids)
            response_logprobs.extend(seg.logprobs)

        # Clamp to response_length (should already fit given budget math).
        if len(response_ids) > self.response_length:
            response_ids = response_ids[: self.response_length]
            response_logprobs = response_logprobs[: self.response_length]

        # Truthful "found_answer" diagnostic.
        full_decoded = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                response_ids, skip_special_tokens=False
            ),
        )
        found_answer = "\\boxed{" in full_decoded

        return self._build_output(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            path=path,
            found_answer=found_answer,
            uid=uid,
            generate_duration=metrics.get("generate_sequences", 0.0),
        )

    # ------------------------------------------------------------------
    # Segment generation
    # ------------------------------------------------------------------

    def _segment_budget(
        self,
        prefix_len: int,
        is_leaf_depth: bool,
        prompt_len: int,
    ) -> int:
        """Token budget for a single segment.

        Interior segments are capped at M. Leaf segments are allowed to use
        the remainder of the response budget so they can run to termination.
        """
        response_len_used = prefix_len - prompt_len
        remaining = self.response_length - response_len_used
        if remaining <= 0:
            return 0
        if is_leaf_depth:
            # Leaf: let the model finish the trajectory (M-token minimum).
            return remaining
        return min(self.M, remaining)

    async def _generate_segment(
        self,
        prefix_ids: list[int],
        sampling_params: dict,
        max_new_tokens: int,
        is_leaf_depth: bool,
        validation: bool,
    ) -> _NodeResult:
        if max_new_tokens <= 0:
            return _NodeResult(token_ids=[], logprobs=[])

        params = dict(sampling_params)
        if not validation:
            params["temperature"] = self.mc_temperature
        params["max_new_tokens"] = max_new_tokens
        params.pop("max_tokens", None)
        params.pop("stop", None)
        params["include_stop_str_in_output"] = True

        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=list(prefix_ids),
            sampling_params=params,
        )

        token_ids = list(output.token_ids) if output.token_ids else []
        logprobs = list(output.log_probs) if output.log_probs else [0.0] * len(token_ids)

        # For interior segments, truncate to exactly M tokens so the tree has
        # the expected shape. Leaves keep whatever vLLM returned.
        if not is_leaf_depth and len(token_ids) > self.M:
            token_ids = token_ids[: self.M]
            logprobs = logprobs[: self.M]

        return _NodeResult(token_ids=token_ids, logprobs=logprobs)

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------

    def _build_output(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: list[float],
        path: tuple[int, ...],
        found_answer: bool,
        uid: Optional[str],
        generate_duration: float,
    ) -> AgentLoopOutput:
        response_len = len(response_ids)

        # Segment boundaries: each interior segment is exactly M tokens.
        # Leaf segment covers [(L-1)·M, end_of_response).
        segment_positions: list[int] = []
        cum = 0
        for depth in range(self.L):
            if depth == self.L - 1:
                segment_positions.append(response_len)
            else:
                cum += self.M
                segment_positions.append(min(cum, response_len))

        # 1-indexed segment id per token; tokens past the last interior
        # boundary belong to the leaf segment (id = L).
        seg_token_ids = [0] * self.response_length
        prev = 0
        for depth, end in enumerate(segment_positions):
            for t in range(prev, min(end, response_len, self.response_length)):
                seg_token_ids[t] = depth + 1
            prev = end

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * response_len,
            response_logprobs=response_logprobs if response_logprobs else None,
            multi_modal_data={},
            num_turns=self.L,
            metrics=AgentLoopMetrics(
                generate_sequences=generate_duration,
                tool_calls=0.0,
            ),
        )

        output.extra_fields.update(
            {
                # SPO-tree
                "tree_path": list(path),
                "tree_branching": list(self.B),
                "segment_positions": segment_positions,
                "segment_token_ids": seg_token_ids,
                "segment_tokens_per_level": self.M,
                "found_answer": found_answer,
                "num_segments": self.L,
                # TGRPO-compat aliases
                "thought_segment_ids": seg_token_ids,
                "num_thoughts": self.L,
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        if uid is not None:
            output.extra_fields["uid"] = uid
        return output

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_question(self, messages) -> str:
        if isinstance(messages, str):
            return messages
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part["text"]
        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                return content
        return ""
