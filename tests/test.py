#!/usr/bin/env python3
"""GATS 2.0 Tests - Property-based and unit tests."""
import random
from gats import (
    ActionSpec, ActionModel, State, Candidate, VerificationResult, FailCode,
    Agent, GreedyAgent, Episode
)
from bench import BenchmarkConfig, generate_chain_task

def test_verifier_rejects_invalid():
    """Verifier correctly rejects invalid actions."""
    model = ActionModel()
    model.register(ActionSpec(
        action_id="test_action",
        description="Test",
        args_schema={"x": int},
        preconditions=frozenset(["prereq"]),
        effects_add=frozenset(["output"])
    ))
    
    state = State(goal=frozenset(["output"]), inventory=frozenset())
    
    # Test NOT_FOUND
    result = model.verify(Candidate("nonexistent"), state)
    assert not result.is_valid
    assert result.fail_code == FailCode.NOT_FOUND
    
    # Test ARG_SCHEMA_MISMATCH
    result = model.verify(Candidate("test_action", {"x": "string"}), state)
    assert not result.is_valid
    assert result.fail_code == FailCode.ARG_SCHEMA_MISMATCH
    
    # Test PRECONDITION_MISSING
    result = model.verify(Candidate("test_action", {"x": 1}), state)
    assert not result.is_valid
    assert result.fail_code == FailCode.PRECONDITION_MISSING
    assert "prereq" in result.missing_preconditions
    
    # Test valid
    state_with_prereq = State(goal=frozenset(["output"]), inventory=frozenset(["prereq"]))
    result = model.verify(Candidate("test_action", {"x": 1}), state_with_prereq)
    assert result.is_valid
    assert "output" in result.compiled_effects_add
    
    print("✓ test_verifier_rejects_invalid")

def test_state_hash_stable():
    """State hash is deterministic."""
    s1 = State(goal=frozenset(["a", "b"]), inventory=frozenset(["x", "y"]))
    s2 = State(goal=frozenset(["b", "a"]), inventory=frozenset(["y", "x"]))
    assert s1.hash() == s2.hash()
    
    s3 = State(goal=frozenset(["a"]), inventory=frozenset(["x", "y"]))
    assert s1.hash() != s3.hash()
    
    print("✓ test_state_hash_stable")

def test_no_invalid_execution():
    """Invariant: agent never executes invalid action."""
    for seed in range(20):
        config = BenchmarkConfig(depth=5, n_tools=30, seed=seed)
        model, state = generate_chain_task(config)
        agent = Agent(model, search_budget=50)
        episode = agent.solve(state, f"test_{seed}")
        
        # Replay and verify each action was legal at time of execution
        replay_state = state
        for event in episode.events:
            candidate = Candidate(event.action_id, event.args)
            result = model.verify(candidate, replay_state)
            assert result.is_valid, f"Invalid execution: {event.action_id} at step {event.step}"
            
            # Apply effects
            new_inv = (replay_state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            replay_state = replay_state.with_inventory(new_inv)
    
    print("✓ test_no_invalid_execution (20 episodes)")

def test_replay_determinism():
    """Replaying events reproduces same end state."""
    config = BenchmarkConfig(depth=4, seed=123)
    model, state = generate_chain_task(config)
    
    agent = Agent(model, search_budget=30)
    episode = agent.solve(state, "determinism_test")
    
    # Replay
    replay_state = state
    for event in episode.events:
        result = model.verify(Candidate(event.action_id, event.args), replay_state)
        new_inv = (replay_state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
        replay_state = replay_state.with_inventory(new_inv)
    
    # Verify final state matches
    assert event.state_after == replay_state.hash()
    
    print("✓ test_replay_determinism")

def test_goal_detection():
    """Goal detection works correctly."""
    s = State(goal=frozenset(["a", "b"]), inventory=frozenset(["a"]))
    assert not s.is_goal()
    
    s2 = s.with_inventory(frozenset(["a", "b", "c"]))
    assert s2.is_goal()
    
    print("✓ test_goal_detection")

def test_legal_actions():
    """get_legal_actions returns only executable actions."""
    model = ActionModel()
    model.register(ActionSpec("a1", "desc", {}, frozenset(["x"]), frozenset(["y"])))
    model.register(ActionSpec("a2", "desc", {}, frozenset(["y"]), frozenset(["z"])))
    model.register(ActionSpec("a3", "desc", {}, frozenset(), frozenset(["w"])))  # No prereqs
    
    state = State(goal=frozenset(["z"]), inventory=frozenset(["x"]))
    legal = model.get_legal_actions(state)
    
    assert len(legal) == 2  # a1 (has x) and a3 (no prereqs)
    legal_ids = {a.action_id for a in legal}
    assert "a1" in legal_ids
    assert "a3" in legal_ids
    assert "a2" not in legal_ids
    
    print("✓ test_legal_actions")

def run_all_tests():
    """Run all tests."""
    print("=== GATS 2.0 Tests ===\n")
    
    test_verifier_rejects_invalid()
    test_state_hash_stable()
    test_goal_detection()
    test_legal_actions()
    test_no_invalid_execution()
    test_replay_determinism()
    
    print("\n=== All tests passed ===")

if __name__ == "__main__":
    run_all_tests()
