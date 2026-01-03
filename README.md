# GATS 2.0: Graph-Augmented Tree Search for Agents

Minimal, research-grade implementation of uncertainty-aware search for tool-use agents with calibrated world models and external benchmark validation.

## Core Idea

GATS 2.0 enforces **soundness** (no illegal actions) via a deterministic action compiler, while using **MCTS** to optimize action selection under budget constraints. A **3-layer world model** provides uncertainty-calibrated predictions for planning.

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Proposer   │────▶│   Verifier   │────▶│    MCTS    │────▶ Execute
│ (candidates)│     │ (compile)    │     │  (search)  │
└─────────────┘     └──────────────┘     └────────────┘
        ▲                   │                    │
        └───────────────────┴────────────────────┘
                    (feedback loop)

World Model Layers:
  L1 (Exact)      p=1.0   ActionSpec definitions
  L2 (Learned)    p=var   Log-based transition stats
  L3 (Genertic)   p=0.4   LLM/heuristic fallback
```

## Quick Start

```bash
# Run all tests
python test.py           # Core unit tests
python test_modules.py   # New feature tests

# Internal benchmarks
python run.py all

# External benchmarks (API-Bank + ToolBench)
python run_external.py all

# Feature demo (event log, calibration, LLM)
python demo_features.py
```

## Features

### 1. Uncertainty-Aware Search (MCTS)
PUCT-based tree search with progressive widening and transposition tables.

### 2. Multi-Layer World Model
```python
from gats.world_model import LayeredWorldModel

wm = LayeredWorldModel(action_model, layer3_llm=llm_fn)
next_state, prob = wm.predict(state, candidate)
print(f"Layer used: L{wm._last_layer_used}, confidence: {prob}")
```

### 3. Formal Verification
Deterministic action-model compiler ensures executed actions are always valid.

```python
result = action_model.verify(candidate, state)
if result.is_valid:
    new_state = apply(result.compiled_effects)
else:
    print(result.fail_code, result.repair_suggestions)
```

### 4. Event Log as Source of Truth
Append-only logging with replay validation for reproducibility.

```python
from gats.event_log import EventLog, LogEvent, validate_replay

log = EventLog("logs/run.jsonl")
log.start_episode("task_1")
log.log(LogEvent.from_execution(task_id, step, candidate, state_before, state_after, success, cost, layer, conf))
log.end_episode("task_1", success=True)

# Validate replay
errors = validate_replay("logs/run.jsonl", action_model)
assert len(errors) == 0
```

### 5. Calibration Metrics
Standard metrics (ECE, Brier, MCE) with per-layer analysis and temperature scaling.

```python
from gats.calibration import CalibrationTracker, calibration_report

tracker = CalibrationTracker()
tracker.record(confidence=0.9, correct=True, layer=1)
tracker.record(confidence=0.7, correct=False, layer=2)

print(f"ECE: {tracker.ece:.4f}")
print(f"Brier: {tracker.brier:.4f}")
print(tracker.report())  # Full report with reliability diagram
```

### 6. Open Model Integration
Pluggable LLM backends for Layer 3 predictions.

```python
from gats.llm import create_predictor, CachedPredictor

# Auto-detect: Ollama → vLLM → HuggingFace
predictor = create_predictor("auto")

# Or specify backend
predictor = create_predictor("ollama", model="llama3.2")

# With caching
cached = CachedPredictor(predictor, maxsize=1000)
response = cached.predict_effects("get_weather", frozenset(["location"]))
print(f"Effects: {response.effects}, Cache hit rate: {cached.hit_rate:.1%}")
```

## Architecture

### Data Types (`gats/core.py`)
- `ActionSpec`: Action with preconditions, effects, costs
- `State`: Immutable agent state with goal and inventory
- `Candidate`: Proposed action from proposer
- `VerificationResult`: Deterministic compilation result

### Verifier (`gats/verifier.py`)
Deterministic action-model compiler. **Key property**: executed actions are always verified.

### World Model (`gats/world_model.py`)
3-layer cascade: L1 (exact) → L2 (learned) → L3 (generative)

### MCTS (`gats/search.py`)
PUCT-based tree search with progressive widening and transposition table.

### Agent (`gats/agent.py`)
Main loop enforcing the invariant: `Executed(a) ⟹ Verified(a) = true`

### Event Log (`gats/event_log.py`)
Append-only logging with deterministic hashing and replay validation.

### Calibration (`gats/calibration.py`)
ECE, Brier, MCE metrics with reliability diagrams and temperature scaling.

### LLM (`gats/llm.py`)
Pluggable backends: Ollama, vLLM, HuggingFace, OpenAI-compatible.

## Benchmarks

### Internal Benchmarks

| Task Type | TreeSearch | Greedy |
|-----------|------------|--------|
| Deceptive | 100% | 100% |
| Cost-optimal | 100% | 100% |
| Resource | 100% | 0% |
| Dead-end | 100% | 100% |
| **Overall** | **100%** | **75%** |

### External Benchmarks (API-Bank + ToolBench)

```
================================================================================
API-Bank External Validity Benchmark (100 tasks)
================================================================================
Agent        Overall SR    API Acc   Avg Cost
GATS             100.0%      84.1%       3.7
Greedy           100.0%       3.0%       6.0
Random            99.0%      18.4%      10.3

================================================================================
ToolBench External Validity Benchmark (100 tasks)
================================================================================
Agent        Overall SR   Avg Cost
GATS             100.0%       3.3
Greedy           100.0%       4.1
Random           100.0%       5.7
```

**Key findings:**
- API accuracy: GATS 84.1% vs Greedy 3.0% (+81pp)
- Cost efficiency: GATS 3.5 vs Greedy 5.0 (-30%)

## File Structure

```
gats/
├── __init__.py
├── core.py          # Data types
├── verifier.py      # Action compiler
├── search.py        # MCTS
├── agent.py         # Main agent
├── world_model.py   # 3-layer world model
├── event_log.py     # Logging + replay validation
├── calibration.py   # ECE, Brier, reliability diagrams
└── llm.py           # LLM backends (Ollama, vLLM, HF)

bench/
├── __init__.py
├── tasks.py         # Internal task generators
├── api_bank.py      # API-Bank benchmark adapter
└── toolbench.py     # ToolBench benchmark adapter

test.py              # Core unit tests
test_modules.py      # New feature tests
run.py               # Internal benchmarks
run_external.py      # External benchmarks
demo_features.py     # Feature demonstration
```

## Extending

### Custom Actions
```python
model.register(ActionSpec(
    action_id="fetch_data",
    description="Fetch data from API",
    args_schema={"url": str},
    preconditions=frozenset(["auth_token"]),
    effects_add=frozenset(["api_response"]),
    cost=1.0
))
```

### Custom LLM Backend
```python
from gats.llm import LLMPredictor, LLMResponse

class MyPredictor(LLMPredictor):
    def predict_effects(self, action, inventory, goal=None) -> LLMResponse:
        # Your implementation
        return LLMResponse(text="...", effects=frozenset(["result"]), confidence=0.8)
    
    def is_available(self) -> bool:
        return True
```

### Training Layer 2 from Logs
```python
from gats.event_log import train_from_logs
from gats.world_model import Layer2Learned

transitions = train_from_logs("logs/historical.jsonl")
layer2 = Layer2Learned()
for t in transitions:
    layer2.record_transition(t.state_before, t.action_id, t.args, t.state_after, t.success, t.cost)
```

## Key Invariants

1. **Soundness**: Invalid execution rate = 0 (enforced by architecture)
2. **Replay determinism**: Event logs reproduce identical end states
3. **Inventory typing**: State only changes via executor results
4. **Calibration**: Per-layer ECE tracking for uncertainty quantification

## Publication Checklist

- [x] Soundness at scale (0 invalid executions)
- [x] Controlled difficulty sweep
- [x] Ablation studies
- [x] Anytime improvement curves
- [x] Reproducible (fixed seeds, event logs)
- [x] External benchmark (API-Bank, ToolBench)
- [x] Open model replication (Ollama, vLLM, HF)
- [x] Calibration metrics (ECE, Brier, reliability diagrams)
- [x] Event logging with replay validation

## Citation

```bibtex
@article{gats2026,
  title={GATS: Graph-Augmented Tree Search for Uncertainty-Aware Agent Evaluation},
  author={...},
  year={2026}
}
```

## License

MIT