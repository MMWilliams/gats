"""GATS 2.0 Core Data Types."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
import hashlib
import json

class FailCode(Enum):
    """Verification failure codes."""
    NOT_FOUND = auto()
    ARG_SCHEMA_MISMATCH = auto()
    PRECONDITION_MISSING = auto()
    PERMISSION_DENIED = auto()
    TYPE_MISMATCH = auto()

@dataclass(frozen=True)
class ActionSpec:
    """Action specification with preconditions and effects."""
    action_id: str
    description: str
    args_schema: dict[str, type]
    preconditions: frozenset[str]  # Required inventory items
    effects_add: frozenset[str]    # Items added to inventory
    effects_remove: frozenset[str] = frozenset()
    cost: float = 1.0
    
    def validate_args(self, args: dict[str, Any]) -> bool:
        """Check if args match schema."""
        if set(args.keys()) != set(self.args_schema.keys()):
            return False
        return all(isinstance(args[k], t) for k, t in self.args_schema.items())

@dataclass
class State:
    """Agent state - immutable view for search."""
    goal: frozenset[str]  # Target inventory items
    inventory: frozenset[str] = frozenset()
    constraints: dict[str, Any] = field(default_factory=dict)
    
    def hash(self) -> str:
        """Stable hash for transposition table."""
        data = (tuple(sorted(self.goal)), tuple(sorted(self.inventory)))
        return hashlib.md5(json.dumps(data).encode()).hexdigest()[:16]
    
    def is_goal(self) -> bool:
        """Check if goal is satisfied."""
        return self.goal <= self.inventory
    
    def with_inventory(self, inv: frozenset[str]) -> State:
        """Return new state with updated inventory."""
        return State(self.goal, inv, self.constraints)

@dataclass
class Candidate:
    """Proposed action candidate."""
    action_id: str
    args: dict[str, Any] = field(default_factory=dict)
    prior: float = 0.5  # Policy prior for MCTS
    
@dataclass
class VerificationResult:
    """Deterministic verification output."""
    is_valid: bool
    fail_code: FailCode | None = None
    missing_preconditions: frozenset[str] = frozenset()
    repair_suggestions: list[str] = field(default_factory=list)
    compiled_effects_add: frozenset[str] = frozenset()
    compiled_effects_remove: frozenset[str] = frozenset()

@dataclass
class Event:
    """Single event in execution log."""
    step: int
    action_id: str
    args: dict[str, Any]
    success: bool
    state_before: str  # State hash
    state_after: str
    
@dataclass
class Episode:
    """Complete execution episode."""
    task_id: str
    events: list[Event] = field(default_factory=list)
    success: bool = False
    total_cost: float = 0.0
    nodes_expanded: int = 0
class ActionModel:
    """Action model providing legal actions and verification."""
    
    def __init__(self, action_specs: list[ActionSpec]):
        self._specs = {spec.action_id: spec for spec in action_specs}
    
    def resolve(self, action_id: str) -> ActionSpec | None:
        """Resolve action ID to specification."""
        return self._specs.get(action_id)
    
    def get_legal_actions(self, state: State) -> list[ActionSpec]:
        """Get all actions whose preconditions are met."""
        legal = []
        for spec in self._specs.values():
            if spec.preconditions <= state.inventory:
                legal.append(spec)
        return legal
    
    def verify(self, candidate: Candidate, state: State) -> VerificationResult:
        """Verify that action is valid in current state."""
        spec = self.resolve(candidate.action_id)
        if spec is None:
            return VerificationResult(
                is_valid=False,
                fail_code=FailCode.NOT_FOUND
            )
        
        if not spec.preconditions <= state.inventory:
            missing = spec.preconditions - state.inventory
            return VerificationResult(
                is_valid=False,
                fail_code=FailCode.PRECONDITION_MISSING,
                missing_preconditions=missing
            )
        
        return VerificationResult(
            is_valid=True,
            compiled_effects_add=spec.effects_add,
            compiled_effects_remove=spec.effects_remove
        )


class WorldModel:
    """World model for state transitions."""
    
    def __init__(self, action_model: ActionModel):
        self.action_model = action_model
    
    def predict(self, state: State, candidate: Candidate) -> tuple[State, float]:
        """Predict next state given action."""
        result = self.action_model.verify(candidate, state)
        if not result.is_valid:
            return state, 0.0
        
        new_inventory = (
            state.inventory | result.compiled_effects_add
        ) - result.compiled_effects_remove
        
        return state.with_inventory(new_inventory), 1.0
    
    def __call__(self, state: State, candidate: Candidate) -> tuple[State, float]:
        return self.predict(state, candidate)