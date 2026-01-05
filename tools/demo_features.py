#!/usr/bin/env python3
"""GATS 2.0 Feature Demo - Event Log, Calibration, LLM Integration.

Demonstrates:
1. Event logging with source-of-truth replay
2. Calibration metrics and reliability diagrams
3. Open model integration layer
4. End-to-end workflow with metrics
"""
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gats.core import ActionSpec, State, Candidate, ActionModel, Episode
from gats.world_model import LayeredWorldModel, TransitionRecord
from gats.event_log import EventLog, LogEvent, validate_replay, create_logger
from gats.calibration import (
    CalibrationTracker, Prediction, 
    calibration_report, reliability_diagram,
    find_temperature, apply_temperature
)
from gats.llm import MockPredictor, CachedPredictor, create_predictor


# =============================================================================
# Demo Action Model
# =============================================================================

def create_demo_action_model() -> ActionModel:
    """Create demo action model with travel planning theme."""
    specs = [
        ActionSpec("parse_request", "", {}, frozenset(["user_input"]), frozenset(["parsed_intent"])),
        ActionSpec("search_flights", "", {}, frozenset(["parsed_intent"]), frozenset(["flight_options"])),
        ActionSpec("search_hotels", "", {}, frozenset(["parsed_intent"]), frozenset(["hotel_options"])),
        ActionSpec("check_weather", "", {}, frozenset(["parsed_intent"]), frozenset(["weather_data"])),
        ActionSpec("book_flight", "", {}, frozenset(["flight_options"]), frozenset(["flight_booked"]), cost=5.0),
        ActionSpec("book_hotel", "", {}, frozenset(["hotel_options"]), frozenset(["hotel_booked"]), cost=4.0),
        ActionSpec("send_confirmation", "", {}, frozenset(["flight_booked", "hotel_booked"]), frozenset(["trip_confirmed"])),
        # Trap: cheap but leads nowhere
        ActionSpec("quick_search", "", {}, frozenset(["user_input"]), frozenset(["quick_results"]), cost=0.5),
    ]
    return ActionModel(specs)


# =============================================================================
# Demo 1: Event Logging with Replay Validation
# =============================================================================

def demo_event_logging():
    """Demonstrate event logging and replay validation."""
    print("\n" + "=" * 60)
    print("DEMO 1: Event Logging as Source of Truth")
    print("=" * 60)
    
    action_model = create_demo_action_model()
    world_model = LayeredWorldModel(action_model)
    
    # Create logger
    log_path = Path("/tmp/gats_demo.jsonl")
    log = EventLog(log_path, buffer_size=10)
    
    # Simulate episode
    task_id = "demo_trip_planning"
    log.start_episode(task_id)
    
    state = State(
        goal=frozenset(["trip_confirmed"]),
        inventory=frozenset(["user_input"])
    )
    
    actions = ["parse_request", "search_flights", "search_hotels", 
               "book_flight", "book_hotel", "send_confirmation"]
    
    print(f"\nExecuting task: {task_id}")
    print(f"Goal: {state.goal}")
    print("-" * 40)
    
    events_logged = []
    for step, action_id in enumerate(actions):
        cand = Candidate(action_id)
        next_state, prob = world_model.predict(state, cand)
        
        if prob > 0:
            event = LogEvent.from_execution(
                task_id=task_id,
                step=step,
                action=cand,
                state_before=state,
                state_after=next_state,
                success=True,
                cost=action_model.resolve(action_id).cost,
                layer_used=world_model.last_layer,
                confidence=prob
            )
            log.log(event)
            events_logged.append(event)
            
            print(f"  Step {step}: {action_id} → +{next_state.inventory - state.inventory}")
            state = next_state
    
    summary = log.end_episode(task_id, success=state.is_goal())
    log.flush()
    
    print("-" * 40)
    print(f"Episode complete: success={summary.success}, cost={summary.total_cost:.1f}")
    
    # Replay validation
    print("\nReplay Validation:")
    is_valid, errors = validate_replay(events_logged, action_model)
    print(f"  Valid: {is_valid}")
    if errors:
        for e in errors[:3]:
            print(f"  Error: {e}")
    
    # Cleanup
    log_path.unlink(missing_ok=True)
    return events_logged


# =============================================================================
# Demo 2: Calibration Metrics
# =============================================================================

def demo_calibration():
    """Demonstrate calibration metrics and temperature scaling."""
    print("\n" + "=" * 60)
    print("DEMO 2: Calibration Metrics")
    print("=" * 60)
    
    # Simulate predictions from a slightly overconfident model
    random.seed(42)
    predictions = []
    
    # Layer 1 (exact) - perfectly calibrated
    for _ in range(50):
        predictions.append(Prediction(1.0, True, layer=1))
    
    # Layer 2 (learned) - slightly overconfident
    for _ in range(30):
        conf = random.uniform(0.7, 0.9)
        correct = random.random() < conf - 0.1  # Actual accuracy lower
        predictions.append(Prediction(conf, correct, layer=2))
    
    # Layer 3 (generative) - underconfident
    for _ in range(20):
        conf = random.uniform(0.3, 0.5)
        correct = random.random() < conf + 0.15  # Actual accuracy higher
        predictions.append(Prediction(conf, correct, layer=3))
    
    # Print calibration report
    print(calibration_report(predictions, n_bins=5))
    
    # Temperature scaling
    print("\n" + "-" * 60)
    print("Temperature Scaling (Post-hoc Calibration)")
    print("-" * 60)
    
    T = find_temperature(predictions)
    print(f"Optimal temperature: T = {T:.3f}")
    
    if T > 1.0:
        print("  → Model is overconfident (T > 1 softens predictions)")
    else:
        print("  → Model is underconfident (T < 1 sharpens predictions)")
    
    # Apply scaling and compare
    from gats.calibration import compute_ece
    original_ece = compute_ece(predictions)
    scaled = apply_temperature(predictions, T)
    scaled_ece = compute_ece(scaled)
    
    print(f"\nECE before: {original_ece:.4f}")
    print(f"ECE after:  {scaled_ece:.4f}")
    print(f"Improvement: {(original_ece - scaled_ece) / original_ece * 100:.1f}%")


# =============================================================================
# Demo 3: LLM Integration
# =============================================================================

def demo_llm_integration():
    """Demonstrate LLM predictor integration."""
    print("\n" + "=" * 60)
    print("DEMO 3: Open Model Integration")
    print("=" * 60)
    
    # Create cached mock predictor (simulates LLM)
    base_predictor = MockPredictor()
    predictor = CachedPredictor(base_predictor, maxsize=100)
    
    # Test predictions
    test_actions = [
        ("get_weather", frozenset(["location"])),
        ("search_flights", frozenset(["origin", "dest"])),
        ("create_booking", frozenset(["flight_id"])),
        ("fetch_recommendations", frozenset(["user_prefs"])),
        ("unknown_action_xyz", frozenset()),
    ]
    
    print("\nLLM Effect Predictions:")
    print("-" * 50)
    
    for action, inventory in test_actions:
        response = predictor.predict_effects(action, inventory)
        print(f"  {action:25} → {response.effects}")
    
    # Test caching
    print("\nCache Performance:")
    for action, inventory in test_actions:
        predictor.predict_effects(action, inventory)  # Cache hit
    
    print(f"  Hits: {predictor._hits}, Misses: {predictor._misses}")
    print(f"  Hit rate: {predictor.hit_rate:.1%}")
    
    # Check local model availability
    print("\nLocal Model Availability:")
    ollama = create_predictor("ollama")
    vllm = create_predictor("vllm")
    
    print(f"  Ollama: {'available' if ollama.is_available() else 'not available'}")
    print(f"  vLLM:   {'available' if vllm.is_available() else 'not available'}")


# =============================================================================
# Demo 4: End-to-End with Metrics
# =============================================================================

def demo_end_to_end():
    """Demonstrate full workflow with all features."""
    print("\n" + "=" * 60)
    print("DEMO 4: End-to-End Workflow with Metrics")
    print("=" * 60)
    
    # Setup
    action_model = create_demo_action_model()
    
    # Create world model with LLM fallback (using mock)
    llm_predictor = MockPredictor()
    
    def llm_fn(action: str, inventory: frozenset) -> tuple[frozenset[str], float]:
        response = llm_predictor.predict_effects(action, inventory)
        return response.effects, response.confidence
    
    world_model = LayeredWorldModel(action_model, layer3_llm=llm_fn)
    
    # Train Layer 2 with some history
    print("\nTraining Layer 2 from historical data...")
    for _ in range(15):
        world_model.record_transition(TransitionRecord(
            state_before=frozenset(["parsed_intent"]),
            action_id="check_weather",
            args={},
            state_after=frozenset(["parsed_intent", "weather_data"]),
            success=True,
            cost=1.0
        ))
    try:
        stats = world_model.get_layer_stats()
        print(f"  Recorded {stats.get('layer2', {}).get('total_observations', '?')} transitions")
    except:
        print(f"  Recorded transitions")    
    # Run multiple episodes with calibration tracking
    calibration = CalibrationTracker()
    n_episodes = 20
    
    print(f"\nRunning {n_episodes} episodes...")
    
    successes = 0
    total_cost = 0
    layer_usage = {1: 0, 2: 0, 3: 0}
    
    goals = [
        frozenset(["flight_booked"]),
        frozenset(["hotel_booked"]),
        frozenset(["weather_data"]),
        frozenset(["trip_confirmed"]),
    ]
    
    for ep in range(n_episodes):
        goal = random.choice(goals)
        state = State(goal=goal, inventory=frozenset(["user_input"]))
        episode_cost = 0
        
        for step in range(10):
            if state.is_goal():
                successes += 1
                break
            
            legal = action_model.get_legal_actions(state)
            if not legal:
                break
            
            # Pick action
            spec = random.choice(legal)
            cand = Candidate(spec.action_id)
            
            # Predict
            next_state, prob = world_model.predict(state, cand)
            layer = world_model.last_layer
            layer_usage[layer] += 1
            
            # Record calibration (correct if we get expected effects)
            expected = spec.effects_add
            actual = next_state.inventory - state.inventory
            correct = expected <= actual
            calibration.record(prob, correct, layer)
            
            state = next_state
            episode_cost += spec.cost
        
        total_cost += episode_cost
    
    # Print results
    print(f"\nResults:")
    print(f"  Success rate: {successes/n_episodes:.1%}")
    print(f"  Avg cost: {total_cost/n_episodes:.1f}")
    print(f"  Layer usage: L1={layer_usage[1]}, L2={layer_usage[2]}, L3={layer_usage[3]}")
    
    print("\nCalibration Summary:")
    print(f"  Predictions: {calibration.n}")
    print(f"  Accuracy: {calibration.accuracy:.1%}")
    print(f"  ECE: {calibration.ece:.4f}")
    print(f"  Brier: {calibration.brier:.4f}")
    
    # Per-layer breakdown
    layer_cal = calibration.by_layer()
    print("\nPer-Layer Calibration:")
    for layer, cal in sorted(layer_cal.items()):
        name = {1: "Exact", 2: "Learned", 3: "Generative"}.get(layer, f"L{layer}")
        status = "✓" if not cal.overconfident and not cal.underconfident else "⚠️"
        print(f"  {name}: acc={cal.accuracy:.1%}, conf={cal.avg_confidence:.1%}, ece={cal.ece:.3f} {status}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("GATS 2.0 Feature Demonstration")
    print("=" * 60)
    print("\nThis demo shows the three new features:")
    print("  4. Event Log as Source of Truth")
    print("  5. Calibration Metrics")
    print("  6. Open Model Replication")
    
    demo_event_logging()
    demo_calibration()
    demo_llm_integration()
    demo_end_to_end()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()