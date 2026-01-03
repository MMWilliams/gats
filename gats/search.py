"""GATS 2.0 MCTS Search Controller with PUCT.

Ultra-minimal logging - summary only. Set GATS_LOG=1 to enable.
"""
from __future__ import annotations
import math
import random
import logging
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Any
from pathlib import Path
from .core import State, Candidate, ActionSpec


# =============================================================================
# MINIMAL LOGGING - disabled by default
# =============================================================================

_LOG_ENABLED = os.environ.get("GATS_LOG", "1") == "1"
_logger: logging.Logger | None = None

def _init_logger() -> logging.Logger:
    """Lazy init logger only if enabled."""
    global _logger
    if _logger is not None:
        return _logger
    
    _logger = logging.getLogger("gats.search")
    _logger.setLevel(logging.INFO)
    
    if _logger.handlers:
        return _logger
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "search_logs.jsonl"
    
    fh = logging.FileHandler(log_file, mode='w', encoding="utf-8")  # mode='w' overwrites
    fh.setFormatter(logging.Formatter())
    _logger.addHandler(fh)
    
    return _logger

def _log(data: dict[str, Any]) -> None:
    """Emit single compact log line. No-op if GATS_LOG!=1."""
    if not _LOG_ENABLED:
        return
    logger = _init_logger()
    logger.info(json.dumps(data, separators=(',', ':')))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Node:
    """MCTS tree node."""
    state: State
    parent: Node | None = None
    action: Candidate | None = None
    children: dict[str, Node] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    prior: float = 0.5
    depth: int = 0
    
    @property
    def q_value(self) -> float:
        return self.value / max(1, self.visits)
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# =============================================================================
# SEARCH STATISTICS
# =============================================================================

@dataclass
class SearchStats:
    """Minimal search statistics."""
    iters: int = 0
    expanded: int = 0
    goals: int = 0
    
    def to_dict(self) -> dict[str, int]:
        return {"i": self.iters, "n": self.expanded, "g": self.goals}


# =============================================================================
# MCTS SEARCH
# =============================================================================

class MCTS:
    """Monte Carlo Tree Search with PUCT selection and rollout."""
    
    def __init__(
        self,
        world_model: Callable[[State, Candidate], tuple[State, float]],
        value_fn: Callable[[State], float],
        c_puct: float = 1.5,
        max_depth: int = 20,
        rollout_depth: int = 10
    ) -> None:
        self.world_model = world_model
        self.value_fn = value_fn
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth
        self.transposition: dict[str, Node] = {}
        self.nodes_expanded = 0
        self._action_model = None
        self._stats: SearchStats = SearchStats()
    
    def set_action_model(self, action_model) -> None:
        """Set action model for legal action queries during simulation."""
        self._action_model = action_model
    
    def search(
        self,
        root_state: State,
        legal_candidates: list[Candidate],
        budget: int = 100
    ) -> Candidate | None:
        """Execute MCTS search from root state.
        
        Args:
            root_state: Starting state for search
            legal_candidates: Pre-verified legal actions at root
            budget: Number of MCTS iterations
            
        Returns:
            Best candidate action or None if no valid actions
        """
        self._stats = SearchStats()
        
        if not legal_candidates:
            return None
        
        root = self._get_or_create_node(root_state, depth=0)
        
        # Pre-expand root with all candidates
        for cand in legal_candidates:
            key = f"{cand.action_id}"
            if key not in root.children:
                next_state, prob = self.world_model(root_state, cand)
                if prob > 0:
                    child = Node(
                        next_state,
                        parent=root,
                        action=cand,
                        prior=cand.prior,
                        depth=1
                    )
                    child.value = self._simulate(next_state, depth=self.rollout_depth)
                    child.visits = 1
                    root.children[key] = child
                    self.nodes_expanded += 1
                    self._stats.expanded += 1
                    
                    if next_state.is_goal():
                        self._stats.goals += 1
        
        # Run MCTS iterations
        for iteration in range(budget):
            self._stats.iters = iteration + 1
            
            node = self._select(root)
            
            if node.state.is_goal():
                self._stats.goals += 1
                self._backprop(node, 1.0)
                continue
            
            value = self._expand_and_simulate(node)
            self._backprop(node, value)
        
        if not root.children:
            return legal_candidates[0] if legal_candidates else None
        
        # Select by Q-value
        sorted_children = sorted(
            root.children.items(),
            key=lambda x: x[1].q_value,
            reverse=True
        )
        
        best_child = sorted_children[0][1]
        best_action = best_child.action
        
        # Single summary log
        _log({"a": best_action.action_id, "q": round(best_child.q_value, 2), **self._stats.to_dict()})
        
        return best_action
    
    def _get_or_create_node(self, state: State, depth: int = 0) -> Node:
        """Get existing node from transposition table or create new one."""
        h = state.hash()
        if h not in self.transposition:
            self.transposition[h] = Node(state, depth=depth)
        return self.transposition[h]
    
    def _select(self, node: Node) -> Node:
        """Select leaf node via PUCT traversal."""
        depth = 0
        while node.children and depth < self.max_depth:
            node = self._puct_select(node)
            depth += 1
        return node
    
    def _puct_select(self, node: Node) -> Node:
        """Select child using PUCT formula."""
        total_visits = sum(c.visits for c in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)
        
        best_score = float('-inf')
        best_child = None
        
        for child in node.children.values():
            q = child.q_value
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_and_simulate(self, node: Node) -> float:
        """Expand node and run simulation from new child."""
        if node.state.is_goal():
            return 1.0
        
        if self._action_model is None:
            return self.value_fn(node.state)
        
        legal = self._action_model.get_legal_actions(node.state)
        if not legal:
            return 0.0
        
        # Expand one new child
        for spec in legal:
            key = spec.action_id
            if key not in node.children:
                cand = Candidate(spec.action_id, {}, prior=0.5)
                next_state, prob = self.world_model(node.state, cand)
                if prob > 0:
                    child = Node(
                        next_state,
                        parent=node,
                        action=cand,
                        prior=0.5,
                        depth=node.depth + 1
                    )
                    node.children[key] = child
                    self.nodes_expanded += 1
                    self._stats.expanded += 1
                    return self._simulate(next_state, depth=self.rollout_depth)
        
        # All children expanded - return average
        if node.children:
            return sum(c.q_value for c in node.children.values()) / len(node.children)
        
        return 0.0
    
    def _simulate(self, state: State, depth: int | None = None) -> float:
        """Single greedy rollout to estimate state value.
        
        Args:
            state: State to simulate from
            depth: Maximum simulation depth
            
        Returns:
            Estimated value in [0, 1]
        """
        if depth is None:
            depth = self.rollout_depth
        
        current_state = state
        
        for _ in range(depth):
            if current_state.is_goal():
                return 1.0
            
            if self._action_model is None:
                return self.value_fn(current_state)
            
            legal = self._action_model.get_legal_actions(current_state)
            if not legal:
                return 0.0
            
            # Greedy selection: pick action that produces most goal items or new items
            def score(spec: ActionSpec) -> float:
                goal_items = spec.effects_add & current_state.goal
                new_items = spec.effects_add - current_state.inventory
                return len(goal_items) * 10 + len(new_items) + random.random() * 0.1
            
            best_spec = max(legal, key=score)
            cand = Candidate(best_spec.action_id)
            current_state, prob = self.world_model(current_state, cand)
            
            if prob <= 0:
                return self.value_fn(current_state)
        
        return self.value_fn(current_state)
    
    def _backprop(self, node: Node, value: float) -> None:
        """Backpropagate value through tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent


# =============================================================================
# WORLD MODEL
# =============================================================================

class WorldModel:
    """Layered world model for state prediction."""
    
    def __init__(self, action_model) -> None:
        from .verifier import ActionModel
        self.action_model: ActionModel = action_model
    
    def predict(self, state: State, candidate: Candidate) -> tuple[State, float]:
        """Predict next state and success probability.
        
        Args:
            state: Current state
            candidate: Action to apply
            
        Returns:
            Tuple of (next_state, success_probability)
        """
        spec = self.action_model.resolve(candidate.action_id)
        
        if spec is None:
            return state, 0.0
        
        if not spec.preconditions <= state.inventory:
            return state, 0.0
        
        new_inv = (state.inventory | spec.effects_add) - spec.effects_remove
        new_state = state.with_inventory(new_inv)
        
        return new_state, 1.0