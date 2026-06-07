#!/usr/bin/env python3
"""
Thought by Thought reasoning using an MDP formulation.

Components:
- Agent: LLM policy that generates thoughts by thoughts
- Environment: State transitions and termination rules
- Search: Tree exploration (BFS, DFS)
- Utilities: Tree manipulation methods

Copied from TREE/tree_of_thought.py with additional prompt template functions
for SFT training with and without in-context examples.
"""

import os
import re
import json
import logging
import time
from typing import (
    List,
    Optional,
    Dict,
    Tuple,
    Callable,
    Any,
    Union
)
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT TEMPLATE FUNCTIONS
# ============================================================================

def prompt_template_with_examples() -> str:
    """
    Prompt with 2 in-context examples (book arrangement, rectangle).

    This is the original template from the ToT environment, providing
    concrete worked examples to guide the model's reasoning format.

    Returns:
        Template string with {question} placeholder
    """
    return """You are solving a problem step-by-step.

Instructions:
1. State your next reasoning step (one observation, calculation, or deduction)
2. End each thought with </thought>
3. Continue until you reach the final answer, then write it in \\boxed{{answer}} format

Examples:

Q: In how many ways can 5 distinct books be arranged on a shelf if 2 specific books must not be adjacent?
Total arrangements without restrictions is 5! = 120</thought>
I need to subtract arrangements where the 2 specific books ARE adjacent</thought>
If I treat the 2 books as a single unit, I have 4 units to arrange: 4! = 24 ways</thought>
The 2 books within their unit can be arranged in 2! = 2 ways</thought>
So arrangements with the books adjacent = 24 × 2 = 48</thought>
Therefore, arrangements where they are NOT adjacent = 120 - 48 = \\boxed{{72}}</thought>

Q: A rectangle has area 48 and perimeter 28. What is the length of its diagonal?
Let length = l and width = w. From the area: lw = 48</thought>
From the perimeter: 2l + 2w = 28, so l + w = 14</thought>
From l + w = 14, we get w = 14 - l. Substituting into lw = 48: l(14 - l) = 48</thought>
Expanding: 14l - l² = 48, so l² - 14l + 48 = 0. Factoring: (l - 6)(l - 8) = 0</thought>
So l = 8 and w = 6 (or vice versa). Using the Pythagorean theorem: d² = 8² + 6² = 64 + 36 = 100</thought>
Therefore d = 10, so the answer is \\boxed{{10}}</thought>

Q: {question}
"""


def prompt_template_no_examples() -> str:
    """
    Prompt with generic template (no specific examples).

    This template provides format guidance without concrete examples,
    reducing token overhead and suitable for fine-tuned models that
    have already learned the format.

    Returns:
        Template string with {question} placeholder
    """
    return """You are solving a problem step-by-step.

Instructions:
1. State your next reasoning step (one observation, calculation, or deduction)
2. End each thought with </thought>
3. Continue until you reach the final answer, then write it in \\boxed{{answer}} format

Format:
- Each thought should be a single logical step
- Thoughts are separated by </thought>
- The final thought should contain \\boxed{{your_answer}}

Example format:
[First observation or setup]</thought>
[Next logical step]</thought>
[Continue reasoning...]</thought>
[Final calculation with answer in] \\boxed{{answer}}</thought>

Q: {question}
"""


# ============================================================================
# TREE DATA STRUCTURE (Shared by both architectures)
# ============================================================================

@dataclass
class ThoughtNode:
    """Node in the reasoning tree. Contains a thought, completion status, and tree links."""
    thought: str
    is_done: bool
    confidence: float
    depth: int
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    node_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_path(self) -> List[str]:
        """
        Get the path of thoughts from root to this node.

        Returns:
            List of thoughts representing the reasoning trajectory
        """
        path = []
        node = self
        while node is not None:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))

    def __repr__(self) -> str:
        thought_preview = self.thought[:50] + "..." if len(self.thought) > 50 else self.thought
        return f"ThoughtNode(id={self.node_id}, depth={self.depth}, done={self.is_done}, thought='{thought_preview}')"

    def __hash__(self) -> int:
        return hash(self.node_id)


# ============================================================================
# LAYER 1: CORE MDP COMPONENTS
# ============================================================================

@dataclass
class ToTState:
    """
    State in the reasoning MDP. Includes question, thought history, depth, and tree node.
    Immutable - each action creates a new state.
    """
    question: str
    thought_history: Tuple[str, ...]  # Immutable sequence
    depth: int
    node: Optional[ThoughtNode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.question, self.thought_history, self.depth))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToTState):
            return NotImplemented
        return (
            self.question == other.question and
            self.thought_history == other.thought_history and
            self.depth == other.depth
        )


@dataclass
class ToTAction:
    """Action in the reasoning MDP. A single thought with terminal flag and confidence."""
    thought: str
    is_terminal: bool
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        thought_preview = self.thought[:50] + "..." if len(self.thought) > 50 else self.thought
        return f"ToTAction(thought='{thought_preview}', terminal={self.is_terminal}, conf={self.confidence:.2f})"


class ToTAgent:
    """
    Agent that generates thoughts. Implements policy π(a|s) using an LLM.
    Supports single and batched action generation.
    """

    def __init__(
        self,
        model_manager: Any,
        temperature: float = 0.7,
        max_tokens: int = 150,
        top_p: float = 0.9,
        top_k: int = 50,
        min_tokens: int = 5,
        stop_sequences: Optional[List[str]] = None
    ):
        """
        Initialize the ToT agent.

        Args:
            model_manager: SIERA model manager with vLLM backend
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens per thought
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            min_tokens: Minimum tokens to generate
            stop_sequences: Stop tokens for generation
        """
        self.model_manager = model_manager
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.min_tokens = min_tokens
        self.stop_sequences = stop_sequences or ["</thought>", "\n\n"]

        # Statistics
        self.total_generations = 0
        self.total_tokens = 0

    def act(
        self,
        state: ToTState,
        prompt_fn: Callable[[ToTState], str],
        **gen_kwargs
    ) -> ToTAction:
        """
        Generate a single action for the given state.

        Implements π(a|s) - samples one action from the policy distribution.

        Args:
            state: Current state
            prompt_fn: Function that converts state to prompt
            **gen_kwargs: Additional generation parameters

        Returns:
            Sampled action
        """
        actions = self.act_batch(
            states=[state],
            n_samples_per_state=1,
            prompt_fn=prompt_fn,
            **gen_kwargs
        )
        return actions[0][0]

    def act_batch(
        self,
        states: List[ToTState],
        n_samples_per_state: int,
        prompt_fn: Callable[[ToTState], str],
        **gen_kwargs
    ) -> List[List[ToTAction]]:
        """
        Efficient batched action generation.

        For each state, samples n actions from π(a|s).
        This is the workhorse method that enables efficient tree search.

        Args:
            states: List of states to generate actions for
            n_samples_per_state: Number of actions to sample per state
            prompt_fn: Function that converts state to prompt
            **gen_kwargs: Additional generation parameters

        Returns:
            actions[i][j] = j-th action for i-th state
        """
        if not states:
            return []

        # Build prompts: n_samples for each state
        prompts = []
        for state in states:
            prompt = prompt_fn(state)
            prompts.extend([prompt] * n_samples_per_state)

        # Batch generate
        outputs = self._generate(prompts, **gen_kwargs)

        # Group by state
        actions = []
        for i in range(len(states)):
            state_actions = []
            for j in range(n_samples_per_state):
                idx = i * n_samples_per_state + j
                action = self._parse_action(outputs[idx])
                state_actions.append(action)
            actions.append(state_actions)

        return actions

    def _generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Call LLM to generate text.

        Handles batching and applies generation parameters.

        Args:
            prompts: List of prompts
            **kwargs: Override generation parameters

        Returns:
            List of generated texts
        """
        gen_params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_tokens": self.min_tokens,
            "stop": self.stop_sequences,
            **kwargs
        }

        try:
            outputs = self.model_manager.generate(prompts=prompts, **gen_params)
            self.total_generations += len(prompts)
            return outputs
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ["ERROR: Generation failed"] * len(prompts)

    def _parse_action(
        self,
        output: str
    ) -> ToTAction:
        """
        Parse LLM output into an action.

        Args:
            output: Raw LLM output

        Returns:
            Parsed action with thought and terminal flag
        """
        thought = output.strip()

        # Check if this is a terminal thought (contains \boxed{})
        is_terminal = '\\boxed{' in thought

        # Default confidence (could be computed from logprobs in future)
        confidence = 1.0

        return ToTAction(
            thought=thought,
            is_terminal=is_terminal,
            confidence=confidence
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_generations": self.total_generations,
            "total_tokens": self.total_tokens
        }


class ToTEnvironment:
    """
    Environment for Tree of Thought reasoning MDP.

    The environment manages:
    - State transitions: step(state, action) -> (next_state, reward, done, info)
    - Episode initialization: reset(question) -> initial_state
    - Termination conditions: is_terminal(state) -> bool
    - Prompt construction: state_to_prompt(state) -> str
    - Tree structure maintenance (via ThoughtNode)
    - Reward calculation

    The environment owns the "rules" of the reasoning task.
    """

    def __init__(
        self,
        max_depth: int = 15,
        prompt_template: Optional[str] = None,
        reward_fn: Optional[Callable[[ToTState, ToTAction, ToTState], float]] = None,
        use_examples: bool = True
    ):
        """
        Initialize the ToT environment.

        Args:
            max_depth: Maximum depth of reasoning tree
            prompt_template: Template for constructing prompts (None = use default)
            reward_fn: Custom reward function (None = use default)
            use_examples: Whether to use in-context examples in default template
        """
        self.max_depth = max_depth
        self.use_examples = use_examples
        self.prompt_template = prompt_template or self._default_prompt_template()
        self.reward_fn = reward_fn or self._default_reward

        # Tree management
        self.root: Optional[ThoughtNode] = None
        self.node_counter = 0

        # Statistics
        self.total_steps = 0
        self.total_episodes = 0

    def reset(
        self,
        question: str
    ) -> ToTState:
        """
        Initialize a new reasoning episode.

        Args:
            question: The question to solve

        Returns:
            Initial state
        """
        # Create root node
        self.root = ThoughtNode(
            thought=question,
            is_done=False,
            confidence=1.0,
            depth=0,
            node_id="root"
        )
        self.node_counter = 0
        self.total_episodes += 1

        # Create initial state
        return ToTState(
            question=question,
            thought_history=tuple(),
            depth=0,
            node=self.root
        )

    def step(
        self,
        state: ToTState,
        action: ToTAction
    ) -> Tuple[ToTState, float, bool, Dict[str, Any]]:
        """
        Execute state transition.

        Standard RL interface: (s, a) -> (s', r, done, info)

        Args:
            state: Current state
            action: Action to take

        Returns:
            next_state: Resulting state
            reward: Reward for this transition
            done: Whether episode is complete
            info: Additional information
        """
        # Create new state (immutable)
        new_history = state.thought_history + (action.thought,)
        new_depth = state.depth + 1

        # Create tree node
        new_node = ThoughtNode(
            thought=action.thought,
            is_done=action.is_terminal,
            confidence=action.confidence,
            depth=new_depth,
            parent=state.node,
            node_id=self._create_node_id()
        )

        # Link to parent
        if state.node is not None:
            state.node.children.append(new_node)

        # Create next state
        next_state = ToTState(
            question=state.question,
            thought_history=new_history,
            depth=new_depth,
            node=new_node
        )

        # Compute reward
        reward = self.reward_fn(state, action, next_state)

        # Check termination
        done = self.is_terminal(next_state)

        # Info
        info = {
            'node_id': new_node.node_id,
            'is_terminal': action.is_terminal,
            'depth': new_depth,
            'thought': action.thought
        }

        self.total_steps += 1

        return next_state, reward, done, info

    def is_terminal(
        self,
        state: ToTState
    ) -> bool:
        """
        Check if state is terminal.

        Args:
            state: State to check

        Returns:
            True if state is terminal
        """
        # Terminal if we found an answer
        has_answer = (
            len(state.thought_history) > 0 and
            '\\boxed{' in state.thought_history[-1]
        )

        # Terminal if we reached max depth
        at_max_depth = state.depth >= self.max_depth

        return has_answer or at_max_depth

    def state_to_prompt(
        self,
        state: ToTState
    ) -> str:
        """
        Convert state to LLM prompt.

        Args:
            state: State to convert

        Returns:
            Formatted prompt string
        """
        prompt = self.prompt_template.format(question=state.question)

        # Append thought history
        for thought in state.thought_history:
            prompt += f"{thought}</thought>\n"

        return prompt

    def _default_reward(
        self,
        state: ToTState,
        action: ToTAction,
        next_state: ToTState
    ) -> float:
        """
        Default reward function.

        Sparse rewards:
        - +1 if found answer (terminal action)
        - -1 if reached max depth without answer
        - 0 otherwise

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward value
        """
        if action.is_terminal:
            return 1.0  # Found an answer
        elif next_state.depth >= self.max_depth:
            return -1.0  # Timeout without answer
        else:
            return 0.0  # Intermediate step

    def _create_node_id(self) -> str:
        """Generate unique node ID."""
        self.node_counter += 1
        return f"node_{self.node_counter}"

    def _default_prompt_template(self) -> str:
        """Default prompt template for reasoning."""
        if self.use_examples:
            return prompt_template_with_examples()
        else:
            return prompt_template_no_examples()

    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "total_nodes": self.node_counter
        }


# ============================================================================
# LAYER 2: TREE SEARCH STRATEGIES
# ============================================================================

class TreeSearch:
    """
    Tree search using RL primitives.

    Implements various search strategies (BFS, DFS, etc.) using the agent-environment
    interface. The search orchestrates the interaction:

    1. Agent generates actions: π(a|s)
    2. Environment executes transitions: (s, a) -> (s', r, done, info)
    3. Search strategy decides which states to expand next

    All strategies use batched generation for efficiency.

    Advanced features:
    - Backtracking: Navigate back up the tree
    - Pruning: Remove unpromising subtrees
    - Path selection: Find best trajectories
    - Tree editing: Branch, rollback operations
    """

    def __init__(
        self,
        agent: ToTAgent,
        env: ToTEnvironment,
        strategy: str = "dfs",
        n_rollouts: int = 3,
        max_batch_size: int = 32,
        auto_save_path: Optional[str] = None,
        enable_wandb: bool = False
    ):
        """
        Initialize tree search.

        Args:
            agent: ToT agent (policy)
            env: ToT environment (dynamics)
            strategy: Search strategy ("bfs" or "dfs")
            n_rollouts: Number of actions to sample per state
            max_batch_size: Maximum batch size for generation
            auto_save_path: Optional path to save tree progress
            enable_wandb: Whether to enable wandb logging
        """
        self.agent = agent
        self.env = env
        self.strategy = strategy
        self.n_rollouts = n_rollouts
        self.max_batch_size = max_batch_size
        self.auto_save_path = auto_save_path
        self.enable_wandb = enable_wandb

        if strategy not in ["dfs", "bfs"]:
            raise ValueError(f"Strategy must be 'dfs' or 'bfs', got '{strategy}'")

        # Statistics
        self.expansion_counter = 0

        # State tracking for advanced operations
        self.state_to_node_map: Dict[ToTState, ThoughtNode] = {}
        self.node_values: Dict[str, float] = {}  # For value-based search

    def search(
        self,
        question: str,
        verbose: bool = True
    ) -> ThoughtNode:
        """
        Execute tree search.

        Args:
            question: Question to solve
            verbose: Whether to print progress

        Returns:
            Root node of the generated tree
        """
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"TREE SEARCH - {self.strategy.upper()}")
            logger.info(f"Question: {question}")
            logger.info(f"N rollouts per state: {self.n_rollouts}")
            logger.info(f"Max depth: {self.env.max_depth}")
            logger.info(f"{'='*80}\n")

        # Initialize episode
        initial_state = self.env.reset(question)

        # Dispatch to search strategy
        if self.strategy == "bfs":
            root = self._search_bfs(initial_state, verbose)
        else:  # dfs
            root = self._search_dfs(initial_state, verbose)

        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info("SEARCH COMPLETE")
            self._print_search_stats()
            logger.info(f"{'='*80}\n")

        return root

    def _search_bfs(
        self,
        initial_state: ToTState,
        verbose: bool
    ) -> ThoughtNode:
        """
        Breadth-first search.

        Expands all states at depth d before moving to depth d+1.
        Uses batched generation to expand all states at each level in parallel.

        Args:
            initial_state: Starting state
            verbose: Print progress

        Returns:
            Root node of tree
        """
        current_level = [initial_state]
        depth = 0

        while current_level and depth < self.env.max_depth:
            depth += 1

            if verbose:
                logger.info(f"\n{'='*80}")
                logger.info(f"BFS DEPTH {depth}")
                logger.info(f"{'='*80}")

            # Filter non-terminal states
            to_expand = [s for s in current_level if not self.env.is_terminal(s)]

            if not to_expand:
                if verbose:
                    logger.info("All branches completed!")
                break

            if verbose:
                logger.info(f"Expanding {len(to_expand)} states")
                logger.info(f"Total actions to generate: {len(to_expand) * self.n_rollouts}")

            # Generate actions for all states (batched)
            action_lists = self.agent.act_batch(
                states=to_expand,
                n_samples_per_state=self.n_rollouts,
                prompt_fn=self.env.state_to_prompt
            )

            # Execute transitions
            next_level = []
            for state, actions in zip(to_expand, action_lists):
                for action in actions:
                    next_state, reward, done, info = self.env.step(state, action)
                    next_level.append(next_state)

            self.expansion_counter += 1

            if verbose:
                logger.info(f"Generated {len(next_level)} new states at depth {depth}")
                done_count = sum(1 for s in next_level if self.env.is_terminal(s))
                logger.info(f"  - Terminal: {done_count}")
                logger.info(f"  - Active: {len(next_level) - done_count}")

            # Auto-save
            self._auto_save()

            current_level = next_level

        return self.env.root

    def _search_dfs(
        self,
        initial_state: ToTState,
        verbose: bool
    ) -> ThoughtNode:
        """
        Depth-first search with sibling batching.

        For each state:
        1. Generate all children together (batched)
        2. Recursively explore each child's subtree before moving to next sibling

        Args:
            initial_state: Starting state
            verbose: Print progress

        Returns:
            Root node of tree
        """

        def dfs_expand(state: ToTState) -> None:
            """Recursively expand state and all descendants."""
            # Base cases
            if self.env.is_terminal(state):
                return

            if verbose:
                logger.info(f"\n{'='*80}")
                logger.info(f"DFS: Expanding depth {state.depth}")
                if state.node:
                    logger.info(f"Node: {state.node.node_id}")
                logger.info(f"{'='*80}")

            # Generate all children (siblings batched together)
            action_lists = self.agent.act_batch(
                states=[state],
                n_samples_per_state=self.n_rollouts,
                prompt_fn=self.env.state_to_prompt
            )
            actions = action_lists[0]

            # Execute transitions to create children
            child_states = []
            for action in actions:
                next_state, reward, done, info = self.env.step(state, action)
                child_states.append(next_state)

            self.expansion_counter += 1

            if verbose:
                logger.info(f"Generated {len(child_states)} children")
                done_count = sum(1 for s in child_states if self.env.is_terminal(s))
                logger.info(f"  - Terminal: {done_count}")
                logger.info(f"  - Active: {len(child_states) - done_count}")

            # Auto-save
            self._auto_save()

            # Recursively explore each child
            for i, child in enumerate(child_states):
                if verbose:
                    logger.info(f"\n{'='*40}")
                    logger.info(f"DFS: Exploring child {i+1}/{len(child_states)}")
                    logger.info(f"{'='*40}")

                dfs_expand(child)

        # Start DFS from initial state
        dfs_expand(initial_state)

        return self.env.root

    def _auto_save(self) -> None:
        """Auto-save tree if enabled."""
        if self.auto_save_path and self.env.root:
            try:
                save_tree_json(self.env.root, self.auto_save_path, self._get_stats())
            except Exception as e:
                logger.warning(f"Auto-save failed: {e}")

    def _print_search_stats(self) -> None:
        """Print search statistics."""
        logger.info(f"Search Statistics:")
        logger.info(f"  - Expansions: {self.expansion_counter}")
        logger.info(f"  - Agent generations: {self.agent.total_generations}")
        logger.info(f"  - Environment steps: {self.env.total_steps}")
        logger.info(f"  - Total nodes: {self.env.node_counter}")

    def _get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "strategy": self.strategy,
            "n_rollouts": self.n_rollouts,
            "max_depth": self.env.max_depth,
            "expansions": self.expansion_counter,
            **self.agent.get_stats(),
            **self.env.get_stats()
        }

    # ========================================================================
    # CORE TREE MANIPULATION METHODS
    # ========================================================================

    def backtrack(
        self,
        state: ToTState,
        steps: int = 1
    ) -> Optional[ToTState]:
        """
        Navigate back up the tree from current state.

        Essential for: Interactive exploration, MCTS selection phase

        Args:
            state: Current state
            steps: Number of steps to go back (default: 1)

        Returns:
            State after backtracking, or None if at root
        """
        if not state.node:
            return None

        current_node = state.node
        for _ in range(steps):
            if current_node.parent is None:
                return None
            current_node = current_node.parent

        # Reconstruct state from node
        path = current_node.get_path()
        return ToTState(
            question=state.question,
            thought_history=tuple(path[1:]) if len(path) > 1 else tuple(),
            depth=current_node.depth,
            node=current_node
        )

    def get_leaf_states(
        self,
        root: Optional[ThoughtNode] = None
    ) -> List[ToTState]:
        """
        Get all leaf states (unexpanded nodes).

        Essential for: Finding frontier to expand next

        Args:
            root: Root node (default: use environment root)

        Returns:
            List of leaf states
        """
        if root is None:
            root = self.env.root

        if root is None:
            return []

        leaves = []

        def find_leaves(node: ThoughtNode) -> None:
            if not node.children:
                path = node.get_path()
                state = ToTState(
                    question=path[0] if path else "",
                    thought_history=tuple(path[1:]) if len(path) > 1 else tuple(),
                    depth=node.depth,
                    node=node
                )
                leaves.append(state)
            else:
                for child in node.children:
                    find_leaves(child)

        find_leaves(root)
        return leaves

    def expand_leaf(
        self,
        leaf_state: ToTState,
        n_children: Optional[int] = None
    ) -> List[ToTState]:
        """
        Expand a leaf by generating children.

        Essential for: All tree search algorithms

        Args:
            leaf_state: Leaf state to expand
            n_children: Number of children (default: use n_rollouts)

        Returns:
            List of newly created child states
        """
        if n_children is None:
            n_children = self.n_rollouts

        # Generate actions
        action_lists = self.agent.act_batch(
            states=[leaf_state],
            n_samples_per_state=n_children,
            prompt_fn=self.env.state_to_prompt
        )
        actions = action_lists[0]

        # Execute transitions
        child_states = []
        for action in actions:
            next_state, reward, done, info = self.env.step(leaf_state, action)
            child_states.append(next_state)

        self.expansion_counter += 1
        self._auto_save()

        return child_states

    def prune_subtree(
        self,
        state: ToTState
    ) -> bool:
        """
        Remove subtree rooted at this state.

        Essential for: Beam search, pruning bad branches

        Args:
            state: Root of subtree to prune

        Returns:
            True if successfully pruned, False if can't prune (e.g., root)
        """
        if not state.node or not state.node.parent:
            return False

        parent = state.node.parent
        if state.node in parent.children:
            parent.children.remove(state.node)
            return True

        return False

    def set_node_value(
        self,
        state: ToTState,
        value: float
    ) -> None:
        """
        Set value for a node.

        Essential for: MCTS, value-based search, AlphaZero-style algorithms

        Args:
            state: State to set value for
            value: Value to assign
        """
        if state.node:
            self.node_values[state.node.node_id] = value

    def get_node_value(
        self,
        state: ToTState,
        default: float = 0.0
    ) -> float:
        """
        Get value for a node.

        Essential for: MCTS, value-based search

        Args:
            state: State to get value for
            default: Default value if not set

        Returns:
            Node value
        """
        if state.node:
            return self.node_values.get(state.node.node_id, default)
        return default

    def backpropagate_value(
        self,
        leaf_state: ToTState,
        value: float,
        discount: float = 0.95
    ) -> None:
        """
        Backpropagate value from leaf to root.

        Essential for: MCTS backpropagation phase

        Args:
            leaf_state: Leaf state where value originates
            value: Value to propagate
            discount: Discount factor per step (default: 0.95)
        """
        current_state = leaf_state
        current_value = value

        while current_state.node is not None:
            # Take max of existing and new value (like Bellman backup)
            existing_value = self.get_node_value(current_state, default=0.0)
            new_value = max(existing_value, current_value)
            self.set_node_value(current_state, new_value)

            # Move to parent
            parent = self.backtrack(current_state, steps=1)
            if parent is None:
                break

            current_state = parent
            current_value *= discount


# ============================================================================
# LAYER 3: UTILITIES
# ============================================================================

def get_completed_paths(
    root: ThoughtNode
) -> List[List[str]]:
    """
    Get all paths that reached completion.

    Args:
        root: Root node of tree

    Returns:
        List of completed thought paths
    """
    completed = []

    def traverse(node: ThoughtNode) -> None:
        if node.is_done:
            completed.append(node.get_path())
        for child in node.children:
            traverse(child)

    traverse(root)
    return completed


def extract_boxed_answers(
    root: ThoughtNode
) -> List[Tuple[str, List[str]]]:
    """
    Extract all boxed answers from completed paths.

    Args:
        root: Root node of tree

    Returns:
        List of (answer, path) tuples
    """
    boxed_pattern = re.compile(r'\\boxed\{([^}]+)\}')
    results = []

    for path in get_completed_paths(root):
        # Check last thought for boxed answer
        if len(path) > 1:  # Skip root
            last_thought = path[-1]
            match = boxed_pattern.search(last_thought)

            if match:
                answer = match.group(1)
                results.append((answer, path))

    return results


def save_tree_json(
    root: ThoughtNode,
    filepath: str,
    stats: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save tree structure to JSON file.

    Args:
        root: Root node
        filepath: Output file path
        stats: Optional statistics to include
    """

    def node_to_dict(node: ThoughtNode) -> Dict[str, Any]:
        """Convert node and subtree to dictionary."""
        return {
            "node_id": node.node_id,
            "thought": node.thought,
            "is_done": node.is_done,
            "confidence": node.confidence,
            "depth": node.depth,
            "children": [node_to_dict(child) for child in node.children]
        }

    tree_dict = {
        "root": node_to_dict(root),
        "stats": stats or {}
    }

    with open(filepath, 'w') as f:
        json.dump(tree_dict, f, indent=2)

    logger.info(f"Tree saved to {filepath}")


def print_tree(
    root: ThoughtNode,
    max_depth: Optional[int] = None
) -> None:
    """
    Print tree structure.

    Args:
        root: Root node
        max_depth: Maximum depth to print (None = all)
    """

    def print_node(
        node: ThoughtNode,
        prefix: str = "",
        is_last: bool = True
    ) -> None:
        if max_depth is not None and node.depth > max_depth:
            return

        # Format node info
        marker = "└── " if is_last else "├── "
        done_marker = "[✓]" if node.is_done else "[ ]"
        conf_str = f"({node.confidence:.2f})" if node.depth > 0 else ""

        thought_preview = node.thought[:60] + "..." if len(node.thought) > 60 else node.thought

        if node.depth == 0:
            logger.info(f"{done_marker} ROOT: {thought_preview}")
        else:
            logger.info(f"{prefix}{marker}{done_marker} {conf_str} {thought_preview}")

        # Print children
        if node.children:
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                print_node(child, new_prefix, i == len(node.children) - 1)

    logger.info("\n" + "="*80)
    logger.info("TREE STRUCTURE")
    logger.info("="*80)
    print_node(root)
    logger.info("="*80 + "\n")


def print_tree_stats(
    root: ThoughtNode
) -> None:
    """
    Print tree statistics.

    Args:
        root: Root node
    """
    total_nodes = 0
    done_nodes = 0
    max_depth = 0

    def traverse(node: ThoughtNode) -> None:
        nonlocal total_nodes, done_nodes, max_depth
        total_nodes += 1
        if node.is_done:
            done_nodes += 1
        max_depth = max(max_depth, node.depth)
        for child in node.children:
            traverse(child)

    traverse(root)

    completed_paths = get_completed_paths(root)
    boxed_answers = extract_boxed_answers(root)

    logger.info("\n" + "="*80)
    logger.info("TREE STATISTICS")
    logger.info("="*80)
    logger.info(f"Total nodes: {total_nodes}")
    logger.info(f"Completed nodes: {done_nodes}")
    logger.info(f"Active nodes: {total_nodes - done_nodes}")
    logger.info(f"Max depth reached: {max_depth}")
    logger.info(f"Completed paths: {len(completed_paths)}")
    logger.info(f"Paths with boxed answers: {len(boxed_answers)}")
    logger.info("="*80 + "\n")


def print_completed_paths(
    root: ThoughtNode
) -> None:
    """
    Print all completed reasoning paths.

    Args:
        root: Root node
    """
    logger.info("\n" + "="*80)
    logger.info("COMPLETED REASONING PATHS")
    logger.info("="*80 + "\n")

    completed = get_completed_paths(root)

    for i, path in enumerate(completed, 1):
        logger.info(f"Path {i}:")
        logger.info(f"  Question: {path[0]}")
        for j, thought in enumerate(path[1:], 1):
            logger.info(f"  Step {j}: {thought}")
        logger.info("")

    logger.info(f"Total completed paths: {len(completed)}")
    logger.info("="*80 + "\n")


def print_boxed_answers(
    root: ThoughtNode
) -> None:
    """
    Print all extracted boxed answers.

    Args:
        root: Root node
    """
    logger.info("\n" + "="*80)
    logger.info("EXTRACTED ANSWERS")
    logger.info("="*80 + "\n")

    answers = extract_boxed_answers(root)

    if not answers:
        logger.info("No boxed answers found!")
    else:
        # Group by answer
        answer_groups: Dict[str, List[List[str]]] = {}
        for answer, path in answers:
            if answer not in answer_groups:
                answer_groups[answer] = []
            answer_groups[answer].append(path)

        logger.info(f"Found {len(answer_groups)} unique answer(s):\n")

        for answer, paths in answer_groups.items():
            logger.info(f"Answer: \\boxed{{{answer}}}")
            logger.info(f"  - Found in {len(paths)} path(s)")
            total_completed = len(get_completed_paths(root))
            if total_completed > 0:
                logger.info(f"  - Confidence: {len(paths) / total_completed:.1%} of completed paths")
            logger.info("")

    logger.info("="*80 + "\n")


# ============================================================================
# BATCHED THOUGHT CHAIN GENERATION
# ============================================================================

@dataclass
class ChainState:
    """State of a single thought chain during batched generation."""
    problem_idx: int
    rollout_idx: int
    prefix: str             # accumulated prompt + thoughts
    thoughts: List[str]
    is_done: bool
    has_answer: bool


def generate_thought_chains_batched(
    manager,
    questions: List[str],
    n_rollouts: int = 8,
    use_examples: bool = True,
    max_thoughts: int = 10,
    max_tokens_per_thought: int = 300,
    temperature: float = 0.7,
) -> List[List[List[str]]]:
    """
    Batched depth-synchronized thought chain generation.

    At each depth step, collects ALL active chains across all problems and
    rollouts into a single batch call. This replaces sequential per-chain
    generation with O(max_thoughts) batch calls instead of
    O(n_problems * n_rollouts * max_thoughts) individual calls.

    Args:
        manager: BaseModelManager with generate() method
        questions: List of problem questions
        n_rollouts: Number of independent rollouts per problem
        use_examples: Whether to use in-context examples in prompt template
        max_thoughts: Maximum thoughts per chain
        max_tokens_per_thought: Max tokens per thought generation
        temperature: Sampling temperature

    Returns:
        results[prob_idx][rollout_idx] = list of thought strings
    """
    if use_examples:
        template = prompt_template_with_examples()
    else:
        template = prompt_template_no_examples()

    # Initialize all chain states
    chains: List[ChainState] = []
    for prob_idx, question in enumerate(questions):
        base_prompt = template.format(question=question)
        for rollout_idx in range(n_rollouts):
            chains.append(ChainState(
                problem_idx=prob_idx,
                rollout_idx=rollout_idx,
                prefix=base_prompt,
                thoughts=[],
                is_done=False,
                has_answer=False,
            ))

    logger.info(f"Batched generation: {len(questions)} problems x {n_rollouts} rollouts = {len(chains)} chains")

    # Depth-synchronized loop
    for depth in range(max_thoughts):
        # Collect active chains
        active = [c for c in chains if not c.is_done]
        if not active:
            logger.info(f"  Depth {depth}: all chains done, stopping early")
            break

        logger.info(f"  Depth {depth}: generating for {len(active)} active chains")

        # Build prompts for all active chains
        prompts = [c.prefix for c in active]

        # Single batched call
        outputs = manager.generate(
            prompts=prompts,
            max_tokens=max_tokens_per_thought,
            temperature=temperature,
            stop=["</thought>"],
            min_tokens=5,
            n=1,
        )

        # Update each active chain
        for chain, output in zip(active, outputs):
            thought = output.strip()
            if not thought:
                chain.is_done = True
                continue

            chain.thoughts.append(thought)
            chain.prefix = chain.prefix + thought + "</thought>\n"

            if "\\boxed{" in thought:
                chain.is_done = True
                chain.has_answer = True

    # Organize results: results[prob_idx][rollout_idx] = list of thoughts
    results: List[List[List[str]]] = [
        [[] for _ in range(n_rollouts)]
        for _ in range(len(questions))
    ]
    for chain in chains:
        results[chain.problem_idx][chain.rollout_idx] = chain.thoughts

    # Log summary
    total_thoughts = sum(len(c.thoughts) for c in chains)
    answered = sum(1 for c in chains if c.has_answer)
    logger.info(f"  Done: {total_thoughts} total thoughts, {answered}/{len(chains)} chains found answers")

    return results
