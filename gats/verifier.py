"""GATS 2.0 Action Model Compiler (Verifier).

Deterministic verification - treats actions like a type system.
"""
from __future__ import annotations
from .core import ActionSpec, State, Candidate, VerificationResult, FailCode

class ActionModel:
    """Ground-truth action model with verification."""
    
    def __init__(self) -> None:
        self._actions: dict[str, ActionSpec] = {}
        self._aliases: dict[str, str] = {}  # alias -> canonical
        self._produces: dict[str, list[str]] = {}  # item -> actions that produce it
    
    def register(self, spec: ActionSpec, aliases: list[str] | None = None) -> None:
        """Register an action specification."""
        self._actions[spec.action_id] = spec
        for item in spec.effects_add:
            self._produces.setdefault(item, []).append(spec.action_id)
        for alias in (aliases or []):
            self._aliases[alias] = spec.action_id
    
    def resolve(self, action_id: str) -> ActionSpec | None:
        """Resolve action ID (including aliases)."""
        canonical = self._aliases.get(action_id, action_id)
        return self._actions.get(canonical)
    
    def verify(self, candidate: Candidate, state: State) -> VerificationResult:
        """Compile candidate against action model. Deterministic."""
        spec = self.resolve(candidate.action_id)
        
        # Check existence
        if spec is None:
            suggestions = self._find_similar(candidate.action_id)
            return VerificationResult(False, FailCode.NOT_FOUND, repair_suggestions=suggestions)
        
        # Check args schema
        if not spec.validate_args(candidate.args):
            return VerificationResult(False, FailCode.ARG_SCHEMA_MISMATCH)
        
        # Check preconditions
        missing = spec.preconditions - state.inventory
        if missing:
            suggestions = self._suggest_producers(missing)
            return VerificationResult(
                False, FailCode.PRECONDITION_MISSING,
                missing_preconditions=missing, repair_suggestions=suggestions
            )
        
        # Valid - return compiled effects
        return VerificationResult(
            True,
            compiled_effects_add=spec.effects_add,
            compiled_effects_remove=spec.effects_remove
        )
    
    def verify_batch(self, candidates: list[Candidate], state: State) -> list[VerificationResult]:
        """Verify multiple candidates."""
        return [self.verify(c, state) for c in candidates]
    
    def _find_similar(self, action_id: str, max_results: int = 3) -> list[str]:
        """Find similar action IDs for repair suggestions."""
        # Simple substring matching
        matches = [aid for aid in self._actions if action_id[:3] in aid or aid[:3] in action_id]
        return matches[:max_results]
    
    def _suggest_producers(self, missing: frozenset[str]) -> list[str]:
        """Suggest actions that produce missing items."""
        suggestions = []
        for item in missing:
            if item in self._produces:
                suggestions.extend(self._produces[item])
        return list(set(suggestions))
    
    def get_legal_actions(self, state: State) -> list[ActionSpec]:
        """Return all actions whose preconditions are met."""
        return [s for s in self._actions.values() if s.preconditions <= state.inventory]
    
    @property
    def all_actions(self) -> list[ActionSpec]:
        return list(self._actions.values())
