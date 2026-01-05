#!/usr/bin/env python3
"""GATS 2.0 Module Tests - Event Log, Calibration, LLM."""
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from gats.core import ActionSpec, State, Candidate, ActionModel
from gats.world_model import LayeredWorldModel, TransitionRecord
from gats.event_log import EventLog, LogEvent, LogReader, validate_replay, create_logger
from gats.calibration import (
    Prediction, CalibrationTracker, compute_ece, compute_brier,
    reliability_diagram, calibration_by_layer, calibration_report
)
from gats.llm import (
    MockPredictor, CachedPredictor, OllamaPredictor, VLLMPredictor,
    create_predictor, parse_json_list
)


def test_event_log():
    """Test event logging and replay."""
    print("\n[Event Log Tests]")
    
    # Create temp log
    log_path = Path("/tmp/gats_test.jsonl")
    
    with EventLog(log_path) as log:
        log.start_episode("test_task")
        
        # Log some events
        for i in range(3):
            event = LogEvent.from_execution(
                task_id="test_task",
                step=i,
                action=Candidate(f"action_{i}"),
                state_before=State(frozenset(["goal"]), frozenset([f"item_{j}" for j in range(i)])),
                state_after=State(frozenset(["goal"]), frozenset([f"item_{j}" for j in range(i+1)])),
                success=True,
                cost=1.0,
                layer_used=1,
                confidence=0.9
            )
            log.log(event)
        
        summary = log.end_episode("test_task", success=True)
    
    assert summary.n_steps == 3, f"Expected 3 steps, got {summary.n_steps}"
    assert summary.success, "Expected success"
    print(f"  ✓ EventLog: logged {summary.n_steps} events, cost={summary.total_cost:.1f}")
    
    # Read back
    reader = LogReader(log_path)
    events = list(reader.read_events())
    assert len(events) == 3, f"Expected 3 events, got {len(events)}"
    print(f"  ✓ LogReader: read {len(events)} events")
    
    # Verify event hash
    h = events[0].hash()
    assert len(h) == 16, "Hash should be 16 chars"
    print(f"  ✓ Event hash: {h}")
    
    # Cleanup
    log_path.unlink(missing_ok=True)


def test_calibration():
    """Test calibration metrics."""
    print("\n[Calibration Tests]")
    
    # Create test predictions
    predictions = [
        Prediction(0.9, True, layer=1),   # High conf, correct
        Prediction(0.9, False, layer=1),  # High conf, wrong (overconfident)
        Prediction(0.5, True, layer=2),   # Mid conf, correct
        Prediction(0.5, False, layer=2),  # Mid conf, wrong
        Prediction(0.1, False, layer=3),  # Low conf, correct (wrong prediction, correct behavior)
        Prediction(0.1, True, layer=3),   # Low conf, wrong (underconfident)
    ]
    
    # Test ECE
    ece = compute_ece(predictions)
    assert 0 <= ece <= 1, f"ECE should be in [0,1], got {ece}"
    print(f"  ✓ ECE: {ece:.4f}")
    
    # Test Brier
    brier = compute_brier(predictions)
    assert 0 <= brier <= 1, f"Brier should be in [0,1], got {brier}"
    print(f"  ✓ Brier: {brier:.4f}")
    
    # Test reliability diagram
    diagram = reliability_diagram(predictions, n_bins=5)
    assert len(diagram.accuracies) == 5
    print(f"  ✓ Reliability diagram: {len(diagram.bin_edges)} bin edges")
    
    # Test per-layer calibration
    layer_cal = calibration_by_layer(predictions)
    assert len(layer_cal) == 3, f"Expected 3 layers, got {len(layer_cal)}"
    print(f"  ✓ Per-layer: L1={layer_cal[1].accuracy:.1%}, L2={layer_cal[2].accuracy:.1%}, L3={layer_cal[3].accuracy:.1%}")
    
    # Test tracker
    tracker = CalibrationTracker()
    for p in predictions:
        tracker.record(p.confidence, p.correct, p.layer)
    
    assert tracker.n == 6
    assert abs(tracker.ece - ece) < 0.001
    print(f"  ✓ CalibrationTracker: n={tracker.n}, ece={tracker.ece:.4f}")


def test_llm_parsing():
    """Test LLM response parsing."""
    print("\n[LLM Parsing Tests]")
    
    # Test JSON list parsing
    test_cases = [
        ('["item1", "item2"]', ["item1", "item2"]),
        ('["weather_data"]', ["weather_data"]),
        ('Result: ["result_data", "extra"]', ["result_data", "extra"]),
        ('["item1",\n"item2"]', ["item1", "item2"]),
        ('Some text ["a"] more text', ["a"]),
    ]
    
    for text, expected in test_cases:
        result = parse_json_list(text)
        assert result == expected, f"parse_json_list({text!r}) = {result}, expected {expected}"
    
    print(f"  ✓ parse_json_list: {len(test_cases)} cases passed")


def test_llm_predictor():
    """Test LLM predictor interface."""
    print("\n[LLM Predictor Tests]")
    
    # Test mock predictor
    mock = MockPredictor()
    assert mock.is_available()
    
    response = mock.predict_effects("get_weather", frozenset(["location"]))
    assert response.success
    assert "weather_data" in response.effects
    print(f"  ✓ MockPredictor: get_weather → {response.effects}")
    
    response = mock.predict_effects("search_hotels", frozenset())
    assert "hotels_results" in response.effects
    print(f"  ✓ MockPredictor: search_hotels → {response.effects}")
    
    # Test cached predictor
    cached = CachedPredictor(mock, maxsize=10)
    
    # First call - cache miss
    r1 = cached.predict_effects("test_action", frozenset(["a"]))
    assert cached._misses == 1
    
    # Second call - cache hit
    r2 = cached.predict_effects("test_action", frozenset(["a"]))
    assert cached._hits == 1
    assert r1.effects == r2.effects
    
    print(f"  ✓ CachedPredictor: hit_rate={cached.hit_rate:.1%}")


def test_world_model_integration():
    """Test world model with all three layers."""
    print("\n[World Model Integration Tests]")
    
    # Create action model
    specs = [
        ActionSpec("get_weather", "", {}, frozenset(["location"]), frozenset(["weather_data"])),
        ActionSpec("book_flight", "", {}, frozenset(["flight_list"]), frozenset(["booking"])),
    ]
    action_model = ActionModel(specs)
    
    # Create layered world model
    wm = LayeredWorldModel(action_model)
    
    # Test L1 prediction
    state = State(frozenset(["weather_data"]), frozenset(["location"]))
    cand = Candidate("get_weather")
    
    next_state, prob = wm.predict(state, cand)
    assert wm.last_layer == 1, f"Expected L1, got L{wm.last_layer}"
    assert prob == 1.0
    assert "weather_data" in next_state.inventory
    print(f"  ✓ L1 prediction: get_weather → {next_state.inventory}")
    
    # Train L2
    for _ in range(10):
        wm.record_transition(TransitionRecord(
            state_before=frozenset(["ctx"]),
            action_id="search_hotels",
            args={},
            state_after=frozenset(["ctx", "hotel_list"]),
            success=True,
            cost=2.0
        ))
    
    # Test L2 prediction
    state2 = State(frozenset(), frozenset(["ctx"]))
    cand2 = Candidate("search_hotels")
    
    next_state2, prob2 = wm.predict(state2, cand2)
    assert wm.last_layer == 2, f"Expected L2, got L{wm.last_layer}"
    assert "hotel_list" in next_state2.inventory
    print(f"  ✓ L2 prediction: search_hotels → {next_state2.inventory} (p={prob2:.2f})")
    
    # Test L3 prediction (unknown action)
    state3 = State(frozenset(), frozenset())
    cand3 = Candidate("fetch_news")
    
    next_state3, prob3 = wm.predict(state3, cand3)
    assert wm.last_layer == 3, f"Expected L3, got L{wm.last_layer}"
    print(f"  ✓ L3 prediction: fetch_news → {next_state3.inventory} (p={prob3:.2f})")
    
    # Stats
    if hasattr(wm, 'get_stats'):
        stats = wm.get_stats()
        print(f"  ✓ Stats: L1={stats['layer1_actions']} actions, L2={stats['layer2']['total_observations']} obs")
    else:
        print(f"  ✓ Stats: get_stats() not available (using existing world_model.py)")    


def test_replay_validation():
    """Test event replay validation."""
    print("\n[Replay Validation Tests]")
    
    # Create action model
    specs = [
        ActionSpec("step_1", "", {}, frozenset(["start"]), frozenset(["mid"])),
        ActionSpec("step_2", "", {}, frozenset(["mid"]), frozenset(["goal"])),
    ]
    action_model = ActionModel(specs)
    
    # Create valid event sequence
    events = [
        LogEvent(
            ts=time.time(),
            task_id="test",
            step=0,
            action_id="step_1",
            args=(),
            state_before=("start",),
            state_after=("mid", "start"),
            goal=("goal",),
            success=True,
            cost=1.0
        ),
        LogEvent(
            ts=time.time(),
            task_id="test",
            step=1,
            action_id="step_2",
            args=(),
            state_before=("mid", "start"),
            state_after=("goal", "mid", "start"),
            goal=("goal",),
            success=True,
            cost=1.0
        ),
    ]
    
    is_valid, errors = validate_replay(events, action_model)
    assert is_valid, f"Should be valid, got errors: {errors}"
    print(f"  ✓ Valid sequence: {len(events)} events validated")
    
    # Test invalid sequence (missing precondition)
    invalid_events = [
        LogEvent(
            ts=time.time(),
            task_id="test",
            step=0,
            action_id="step_2",  # Requires "mid" which we don't have
            args=(),
            state_before=("start",),
            state_after=("goal", "start"),
            goal=("goal",),
            success=True,
            cost=1.0
        ),
    ]
    
    is_valid2, errors2 = validate_replay(invalid_events, action_model)
    assert not is_valid2, "Should detect invalid execution"
    print(f"  ✓ Invalid sequence detected: {len(errors2)} error(s)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("GATS 2.0 Module Tests")
    print("=" * 60)
    
    test_event_log()
    test_calibration()
    test_llm_parsing()
    test_llm_predictor()
    test_world_model_integration()
    test_replay_validation()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()