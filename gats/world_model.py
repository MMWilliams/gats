"""
GATS 2.0 Multi-Layer World Model

Implements a hierarchical world model with three layers:
- Layer 1: Exact effects from ActionSpec (deterministic)
- Layer 2: Learned transitions from execution logs (statistical)
- Layer 3: Generative fallback for unknown actions (heuristic/LLM)

Reference: Follows the "World Models" paradigm from Ha & Schmidhuber (2018)
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gats.core import State, ActionSpec, Candidate, ActionModel, VerificationResult


@dataclass
class TransitionRecord:
    """Record of a state transition for learning."""
    state_before: frozenset[str]
    action_id: str
    args: dict[str, Any]
    state_after: frozenset[str]
    success: bool
    cost: float
    timestamp: float = 0.0


@dataclass
class TransitionStats:
    """Statistics for a state-action pair."""
    total_count: int = 0
    success_count: int = 0
    effects_observed: dict[frozenset[str], int] = field(default_factory=dict)
    avg_cost: float = 1.0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.total_count)
    
    def most_likely_effects(self) -> tuple[frozenset[str], float]:
        """Return most frequently observed effects and their probability."""
        if not self.effects_observed:
            return frozenset(), 0.0
        
        best_effects = max(self.effects_observed, key=self.effects_observed.get)
        prob = self.effects_observed[best_effects] / self.total_count
        return best_effects, prob


# =============================================================================
# Layer 1: Exact Effects (Deterministic)
# =============================================================================

class Layer1ExactModel:
    """
    Layer 1: Deterministic world model using ActionSpec effects.
    
    This is the baseline - if an action has known preconditions and effects,
    we can predict the next state exactly.
    """
    
    def __init__(self, action_model: ActionModel):
        self.action_model = action_model
    
    def can_predict(self, state: State, candidate: Candidate) -> bool:
        """Check if this layer can make a prediction."""
        spec = self.action_model.resolve(candidate.action_id)
        return spec is not None
    
    def predict(self, state: State, candidate: Candidate) -> tuple[State, float]:
        """
        Predict next state using exact ActionSpec effects.
        
        Returns:
            (next_state, probability) where probability is 1.0 if valid, 0.0 otherwise
        """
        result = self.action_model.verify(candidate, state)
        if not result.is_valid:
            return state, 0.0
        
        new_inventory = (
            state.inventory | result.compiled_effects_add
        ) - result.compiled_effects_remove
        
        return state.with_inventory(new_inventory), 1.0


# =============================================================================
# Layer 2: Learned Transitions (Statistical)
# =============================================================================

class Layer2LearnedModel:
    """
    Layer 2: Statistical world model learned from execution logs.
    
    Learns:
    - Transition probabilities P(s' | s, a)
    - Effect distributions for each action
    - Context-dependent success rates
    
    Uses a combination of:
    - Exact state matching (when available)
    - Feature-based generalization (for unseen states)
    """
    
    def __init__(
        self,
        min_observations: int = 3,
        confidence_threshold: float = 0.7
    ):
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold
        
        # Exact state-action statistics
        self._exact_stats: dict[tuple[str, str], TransitionStats] = defaultdict(TransitionStats)
        
        # Action-only statistics (aggregated across states)
        self._action_stats: dict[str, TransitionStats] = defaultdict(TransitionStats)
        
        # Feature-based statistics (inventory size buckets, goal overlap)
        self._feature_stats: dict[tuple[str, int, int], TransitionStats] = defaultdict(TransitionStats)
    
    def _state_key(self, state: State) -> str:
        """Create hashable key for state."""
        return str(sorted(state.inventory))
    
    def _feature_key(self, state: State, action_id: str) -> tuple[str, int, int]:
        """Create feature-based key for generalization."""
        inv_size_bucket = len(state.inventory) // 3  # Bucket by inventory size
        goal_overlap = len(state.inventory & state.goal)
        return (action_id, inv_size_bucket, goal_overlap)
    
    def record_transition(self, record: TransitionRecord) -> None:
        """Record an observed transition for learning."""
        state_key = str(sorted(record.state_before))
        action_key = record.action_id
        
        # Update exact state-action stats
        exact_key = (state_key, action_key)
        stats = self._exact_stats[exact_key]
        stats.total_count += 1
        if record.success:
            stats.success_count += 1
            effects = record.state_after - record.state_before
            stats.effects_observed[effects] = stats.effects_observed.get(effects, 0) + 1
        stats.avg_cost = (stats.avg_cost * (stats.total_count - 1) + record.cost) / stats.total_count
        
        # Update action-only stats
        action_stats = self._action_stats[action_key]
        action_stats.total_count += 1
        if record.success:
            action_stats.success_count += 1
            effects = record.state_after - record.state_before
            action_stats.effects_observed[effects] = action_stats.effects_observed.get(effects, 0) + 1
        action_stats.avg_cost = (action_stats.avg_cost * (action_stats.total_count - 1) + record.cost) / action_stats.total_count
        
        # Update feature-based stats
        inv_size_bucket = len(record.state_before) // 3
        goal_overlap = 0  # We don't have goal info in record, use 0
        feature_key = (action_key, inv_size_bucket, goal_overlap)
        feat_stats = self._feature_stats[feature_key]
        feat_stats.total_count += 1
        if record.success:
            feat_stats.success_count += 1
            effects = record.state_after - record.state_before
            feat_stats.effects_observed[effects] = feat_stats.effects_observed.get(effects, 0) + 1
    
    def load_from_logs(self, log_path: Path) -> int:
        """
        Load transitions from JSONL execution logs.
        
        Expected format:
        {"state_before": [...], "action_id": "...", "args": {...}, 
         "state_after": [...], "success": true, "cost": 1.0}
        
        Returns:
            Number of transitions loaded
        """
        count = 0
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    record = TransitionRecord(
                        state_before=frozenset(data.get("state_before", [])),
                        action_id=data["action_id"],
                        args=data.get("args", {}),
                        state_after=frozenset(data.get("state_after", [])),
                        success=data.get("success", True),
                        cost=data.get("cost", 1.0),
                        timestamp=data.get("timestamp", 0.0)
                    )
                    self.record_transition(record)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        return count
    
    def can_predict(self, state: State, candidate: Candidate) -> bool:
        """Check if we have enough observations to make a confident prediction."""
        state_key = self._state_key(state)
        exact_key = (state_key, candidate.action_id)
        
        # Check exact match first
        if exact_key in self._exact_stats:
            stats = self._exact_stats[exact_key]
            if stats.total_count >= self.min_observations:
                return True
        
        # Check action-level stats
        if candidate.action_id in self._action_stats:
            stats = self._action_stats[candidate.action_id]
            if stats.total_count >= self.min_observations * 2:  # Need more for generalization
                return True
        
        return False
    
    def predict(self, state: State, candidate: Candidate) -> tuple[State, float]:
        """
        Predict next state using learned statistics.
        
        Returns:
            (next_state, probability) based on observed transitions
        """
        state_key = self._state_key(state)
        exact_key = (state_key, candidate.action_id)
        
        # Try exact match
        if exact_key in self._exact_stats:
            stats = self._exact_stats[exact_key]
            if stats.total_count >= self.min_observations:
                effects, prob = stats.most_likely_effects()
                if prob >= self.confidence_threshold:
                    new_inventory = state.inventory | effects
                    return state.with_inventory(new_inventory), prob * stats.success_rate
        
        # Fall back to action-level stats
        if candidate.action_id in self._action_stats:
            stats = self._action_stats[candidate.action_id]
            if stats.total_count >= self.min_observations:
                effects, prob = stats.most_likely_effects()
                # Reduce confidence for generalized prediction
                adjusted_prob = prob * stats.success_rate * 0.8
                if adjusted_prob >= self.confidence_threshold * 0.5:
                    new_inventory = state.inventory | effects
                    return state.with_inventory(new_inventory), adjusted_prob
        
        # Can't make confident prediction
        return state, 0.0
    
    def get_stats(self) -> dict[str, Any]:
        """Return learning statistics."""
        return {
            "exact_state_action_pairs": len(self._exact_stats),
            "action_patterns": len(self._action_stats),
            "feature_patterns": len(self._feature_stats),
            "total_observations": sum(s.total_count for s in self._action_stats.values()),
        }


# =============================================================================
# Layer 3: Generative Fallback (Heuristic/LLM)
# =============================================================================

class Layer3GenerativeModel:
    """
    Layer 3: Generative world model for unknown actions.
    
    Uses heuristics and optional LLM to predict effects when:
    - Action is not in ActionSpec (Layer 1 fails)
    - Not enough observations (Layer 2 fails)
    
    Strategies:
    1. Name-based heuristics (e.g., "get_X" likely adds "X_result")
    2. Similarity to known actions
    3. LLM-based prediction (if configured)
    """
    
    def __init__(
        self,
        action_model: ActionModel | None = None,
        llm_predictor: Callable[[State, Candidate], tuple[frozenset[str], float]] | None = None,
        default_confidence: float = 0.3
    ):
        self.action_model = action_model
        self.llm_predictor = llm_predictor
        self.default_confidence = default_confidence
        
        # Common action patterns for heuristic prediction
        self._action_patterns = {
            "get_": ("_data", "result"),
            "fetch_": ("_data", "result"),
            "search_": ("_results", "list"),
            "create_": ("_id", "created"),
            "update_": ("_updated", "success"),
            "delete_": ("_deleted", "success"),
            "send_": ("_sent", "status"),
            "book_": ("_confirmation", "booked"),
            "check_": ("_status", "checked"),
        }
    
    def _heuristic_predict(self, state: State, candidate: Candidate) -> tuple[frozenset[str], float]:
        """
        Use naming heuristics to predict effects.
        
        Examples:
        - "get_weather" → {"weather_data"} or {"weather_result"}
        - "search_flights" → {"flights_results"} or {"flights_list"}
        - "book_hotel" → {"hotel_confirmation"} or {"hotel_booked"}
        """
        action_id = candidate.action_id
        
        # Try to match action patterns
        for prefix, (suffix1, suffix2) in self._action_patterns.items():
            if action_id.startswith(prefix):
                base = action_id[len(prefix):]
                # Generate possible effect names
                effects = frozenset([f"{base}{suffix1}", f"{base}_{suffix2}"])
                return effects, 0.4  # Low confidence for heuristic
        
        # Handle "tool.endpoint" format
        if "." in action_id:
            tool, endpoint = action_id.rsplit(".", 1)
            for prefix, (suffix1, suffix2) in self._action_patterns.items():
                if endpoint.startswith(prefix):
                    base = endpoint[len(prefix):]
                    effects = frozenset([f"{base}{suffix1}", f"{endpoint}_result"])
                    return effects, 0.4
            
            # Default: endpoint_result
            return frozenset([f"{endpoint}_result"]), 0.3
        
        # Default fallback: action_result
        return frozenset([f"{action_id}_result"]), 0.2
    
    def _similarity_predict(self, state: State, candidate: Candidate) -> tuple[frozenset[str], float]:
        """
        Find similar known actions and use their effects.
        """
        if self.action_model is None:
            return frozenset(), 0.0
        
        action_id = candidate.action_id
        best_match = None
        best_score = 0.0
        
        # Get all known actions
        legal = self.action_model.get_legal_actions(state)
        
        for spec in legal:
            # Simple string similarity
            score = self._string_similarity(action_id, spec.action_id)
            if score > best_score:
                best_score = score
                best_match = spec
        
        if best_match and best_score > 0.5:
            # Use similar action's effects
            return best_match.effects_add, best_score * 0.5
        
        return frozenset(), 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple Jaccard similarity on character trigrams."""
        def trigrams(s: str) -> set[str]:
            s = s.lower()
            return {s[i:i+3] for i in range(len(s) - 2)}
        
        t1, t2 = trigrams(s1), trigrams(s2)
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)
    
    def can_predict(self, state: State, candidate: Candidate) -> bool:
        """Layer 3 can always attempt a prediction."""
        return True
    
    def predict(self, state: State, candidate: Candidate) -> tuple[State, float]:
        """
        Generate prediction using heuristics or LLM.
        
        Returns:
            (next_state, probability) - probability reflects confidence
        """
        # Try LLM predictor first if available
        if self.llm_predictor:
            try:
                effects, confidence = self.llm_predictor(state, candidate)
                if confidence > 0:
                    new_inventory = state.inventory | effects
                    return state.with_inventory(new_inventory), confidence
            except Exception:
                pass  # Fall back to heuristics
        
        # Try similarity-based prediction
        effects, confidence = self._similarity_predict(state, candidate)
        if confidence > 0.3:
            new_inventory = state.inventory | effects
            return state.with_inventory(new_inventory), confidence
        
        # Fall back to heuristics
        effects, confidence = self._heuristic_predict(state, candidate)
        new_inventory = state.inventory | effects
        return state.with_inventory(new_inventory), confidence


# =============================================================================
# Multi-Layer World Model (Main Interface)
# =============================================================================

class LayeredWorldModel:
    """
    Multi-layer world model that cascades through layers:
    
    1. Layer 1 (Exact): Use ActionSpec if available
    2. Layer 2 (Learned): Use statistical model if confident
    3. Layer 3 (Generative): Fall back to heuristics/LLM
    
    Each layer returns (next_state, probability). The model uses
    the first layer that returns probability > threshold.
    """
    
    def __init__(
        self,
        action_model: ActionModel,
        layer2_min_obs: int = 3,
        layer3_llm: Callable | None = None,
        confidence_threshold: float = 0.5
    ):
        self.action_model = action_model
        self.confidence_threshold = confidence_threshold
        
        # Initialize layers
        self.layer1 = Layer1ExactModel(action_model)
        self.layer2 = Layer2LearnedModel(min_observations=layer2_min_obs)
        self.layer3 = Layer3GenerativeModel(
            action_model=action_model,
            llm_predictor=layer3_llm
        )
        
        # Track which layer was used for debugging
        self._last_layer_used: int = 0
    @property
    def last_layer(self) -> int:
        return self._last_layer_used
    def predict(self, state: State, candidate: Candidate) -> tuple[State, float]:
        """
        Predict next state using layered approach.
        
        Returns:
            (next_state, probability) from the first confident layer
        """
        # Layer 1: Exact effects
        if self.layer1.can_predict(state, candidate):
            next_state, prob = self.layer1.predict(state, candidate)
            if prob > 0:
                self._last_layer_used = 1
                return next_state, prob
        
        # Layer 2: Learned model
        if self.layer2.can_predict(state, candidate):
            next_state, prob = self.layer2.predict(state, candidate)
            if prob >= self.confidence_threshold:
                self._last_layer_used = 2
                return next_state, prob
        
        # Layer 3: Generative fallback
        next_state, prob = self.layer3.predict(state, candidate)
        self._last_layer_used = 3
        return next_state, prob
    
    def __call__(self, state: State, candidate: Candidate) -> tuple[State, float]:
        return self.predict(state, candidate)
    
    def record_transition(self, record: TransitionRecord) -> None:
        """Record a transition for Layer 2 learning."""
        self.layer2.record_transition(record)
    
    def load_logs(self, log_path: Path) -> int:
        """Load execution logs for Layer 2 learning."""
        return self.layer2.load_from_logs(log_path)
    
    def get_layer_stats(self) -> dict[str, Any]:
        """Return statistics about each layer."""
        return {
            "layer1_actions": len(self.action_model._specs) if hasattr(self.action_model, '_specs') else 0,
            "layer2": self.layer2.get_stats(),
            "layer3_patterns": len(self.layer3._action_patterns),
            "last_layer_used": self._last_layer_used,
        }


# =============================================================================
# LLM-based Effect Predictor (Optional)
# =============================================================================

def create_llm_predictor(
    api_call: Callable[[str], str],
    max_effects: int = 5
) -> Callable[[State, Candidate], tuple[frozenset[str], float]]:
    """
    Create an LLM-based effect predictor.
    
    Args:
        api_call: Function that takes a prompt and returns LLM response
        max_effects: Maximum number of effects to predict
    
    Returns:
        Predictor function for Layer 3
    """
    def predictor(state: State, candidate: Candidate) -> tuple[frozenset[str], float]:
        prompt = f"""Given an agent state and action, predict the effects.

Current inventory: {sorted(state.inventory)}
Goal items: {sorted(state.goal)}
Action: {candidate.action_id}
Arguments: {candidate.args}

What items will be added to inventory after this action?
Reply with a JSON list of item names only, e.g., ["weather_data", "forecast_result"]
"""
        try:
            response = api_call(prompt)
            # Parse JSON response
            effects = json.loads(response)
            if isinstance(effects, list):
                return frozenset(effects[:max_effects]), 0.6
        except (json.JSONDecodeError, Exception):
            pass
        
        return frozenset(), 0.0
    
    return predictor


# =============================================================================
# Backward-compatible WorldModel wrapper
# =============================================================================

class WorldModel(LayeredWorldModel):
    """
    Backward-compatible WorldModel that extends LayeredWorldModel.
    
    Drop-in replacement for the basic WorldModel in gats/core.py
    """
    
    def __init__(
        self,
        action_model: ActionModel,
        enable_learning: bool = True,
        enable_generative: bool = True
    ):
        super().__init__(action_model)
        self._enable_learning = enable_learning
        self._enable_generative = enable_generative
    
    def predict(self, state: State, candidate: Candidate) -> tuple[State, float]:
        """
        Predict with optional layer disabling for testing.
        """
        # Layer 1 always enabled
        if self.layer1.can_predict(state, candidate):
            next_state, prob = self.layer1.predict(state, candidate)
            if prob > 0:
                self._last_layer_used = 1
                return next_state, prob
        
        # Layer 2 if learning enabled
        if self._enable_learning and self.layer2.can_predict(state, candidate):
            next_state, prob = self.layer2.predict(state, candidate)
            if prob >= self.confidence_threshold:
                self._last_layer_used = 2
                return next_state, prob
        
        # Layer 3 if generative enabled
        if self._enable_generative:
            next_state, prob = self.layer3.predict(state, candidate)
            self._last_layer_used = 3
            return next_state, prob
        
        # No prediction possible
        return state, 0.0


if __name__ == "__main__":
    # Demo usage with inline types (for standalone testing)
    from dataclasses import dataclass, field
    from typing import Any
    
    @dataclass(frozen=True)
    class DemoActionSpec:
        action_id: str
        preconditions: frozenset[str] = field(default_factory=frozenset)
        effects_add: frozenset[str] = field(default_factory=frozenset)
        effects_remove: frozenset[str] = frozenset()
        cost: float = 1.0
    
    @dataclass
    class DemoState:
        goal: frozenset[str]
        inventory: frozenset[str] = frozenset()
        
        def is_goal(self) -> bool:
            return self.goal <= self.inventory
        
        def with_inventory(self, inv: frozenset[str]) -> "DemoState":
            return DemoState(self.goal, inv)
    
    @dataclass
    class DemoCandidate:
        action_id: str
        args: dict = field(default_factory=dict)
    
    class DemoActionModel:
        def __init__(self, specs):
            self._specs = {s.action_id: s for s in specs}
        
        def resolve(self, action_id):
            return self._specs.get(action_id)
        
        def get_legal_actions(self, state):
            return [s for s in self._specs.values() if s.preconditions <= state.inventory]
        
        def verify(self, candidate, state):
            spec = self.resolve(candidate.action_id)
            if not spec:
                return type('VR', (), {'is_valid': False})()
            if not spec.preconditions <= state.inventory:
                return type('VR', (), {'is_valid': False})()
            return type('VR', (), {
                'is_valid': True,
                'compiled_effects_add': spec.effects_add,
                'compiled_effects_remove': spec.effects_remove
            })()
    
    # Create sample actions
    specs = [
        DemoActionSpec(
            action_id="get_weather",
            preconditions=frozenset(["has_location"]),
            effects_add=frozenset(["weather_data"]),
        ),
        DemoActionSpec(
            action_id="book_flight",
            preconditions=frozenset(["flight_list"]),
            effects_add=frozenset(["booking_confirmation"]),
        ),
    ]
    
    action_model = DemoActionModel(specs)
    
    # Create layered world model with demo types
    layer1 = Layer1ExactModel(action_model)
    layer2 = Layer2LearnedModel()
    layer3 = Layer3GenerativeModel(action_model=action_model)
    
    # Test Layer 1 (exact)
    state = DemoState(
        goal=frozenset(["weather_data"]),
        inventory=frozenset(["has_location"])
    )
    candidate = DemoCandidate("get_weather", {"location": "NYC"})
    
    next_state, prob = layer1.predict(state, candidate)
    print(f"Layer 1 prediction (exact effects):")
    print(f"  Action: {candidate.action_id}")
    print(f"  Before: {state.inventory}")
    print(f"  After: {next_state.inventory}")
    print(f"  Probability: {prob}")
    
    # Test Layer 2 (learned) - add some training data
    print(f"\nLayer 2 (learned from logs):")
    for i in range(5):
        layer2.record_transition(TransitionRecord(
            state_before=frozenset(["task_context"]),
            action_id="search_hotels",
            args={"city": "Paris"},
            state_after=frozenset(["task_context", "hotel_list"]),
            success=True,
            cost=2.0
        ))
    
    candidate2 = DemoCandidate("search_hotels", {"city": "Paris"})
    state2 = DemoState(goal=frozenset(["hotel_list"]), inventory=frozenset(["task_context"]))
    
    if layer2.can_predict(state2, candidate2):
        next_state2, prob2 = layer2.predict(state2, candidate2)
        print(f"  Action: {candidate2.action_id}")
        print(f"  Before: {state2.inventory}")
        print(f"  After: {next_state2.inventory}")
        print(f"  Probability: {prob2:.2f}")
    print(f"  Stats: {layer2.get_stats()}")
    
    # Test Layer 3 (generative) - unknown action
    print(f"\nLayer 3 prediction (generative/heuristic):")
    candidate3 = DemoCandidate("fetch_restaurants", {"query": "pizza"})
    state3 = DemoState(goal=frozenset(["restaurants_data"]), inventory=frozenset())
    
    next_state3, prob3 = layer3.predict(state3, candidate3)
    print(f"  Action: {candidate3.action_id}")
    print(f"  Before: {state3.inventory}")
    print(f"  After: {next_state3.inventory}")
    print(f"  Probability: {prob3:.2f}")
    print(f"  (Heuristic: 'fetch_X' → 'X_data')")
    
    # Test another pattern
    candidate4 = DemoCandidate("create_booking", {})
    next_state4, prob4 = layer3.predict(state3, candidate4)
    print(f"\n  Action: {candidate4.action_id}")
    print(f"  After: {next_state4.inventory}")
    print(f"  Probability: {prob4:.2f}")
    print(f"  (Heuristic: 'create_X' → 'X_id')")