# GATS 2.0: Graph-Augmented Tree Search for Agents

Minimal, research-grade implementation of uncertainty-aware search for tool-use agents.

## Core Idea

GATS 2.0 enforces **soundness** (no illegal actions) via a deterministic action compiler, while using **MCTS** to optimize action selection under budget constraints.

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Proposer   │────▶│   Verifier   │────▶│    MCTS    │────▶ Execute
│ (candidates)│     │ (compile)    │     │  (search)  │
└─────────────┘     └──────────────┘     └────────────┘
        ▲                   │                    │
        └───────────────────┴────────────────────┘
                    (feedback loop)
```

## Quick Start

```bash
# Demo
python run.py demo

# Benchmarks  
python run.py bench

# Ablation studies
python run.py ablate

# Anytime improvement curves
python run.py anytime

# Run tests
python test.py
```

## Architecture

### Data Types (`gats/core.py`)
- `ActionSpec`: Action with preconditions, effects, costs
- `State`: Immutable agent state with goal and inventory
- `Candidate`: Proposed action from proposer
- `VerificationResult`: Deterministic compilation result
- `Episode`/`Event`: Execution logging

### Verifier (`gats/verifier.py`)
Deterministic action-model compiler. **Key property**: executed actions are always verified.

```python
result = action_model.verify(candidate, state)
if result.is_valid:
    # Safe to execute - effects are compiled from ground truth
    new_state = apply(result.compiled_effects)
else:
    # Repair suggestions provided
    print(result.fail_code, result.repair_suggestions)
```

### MCTS (`gats/search.py`)
PUCT-based tree search with:
- Progressive widening
- Transposition table
- Configurable search budget

### Agent (`gats/agent.py`)
Main loop enforcing the invariant: `Executed(a) ⟹ Verified(a) = true`

## Key Invariants

1. **Soundness**: Invalid execution rate = 0 (enforced by architecture)
2. **Replay determinism**: Event logs reproduce identical end states
3. **Inventory typing**: State only changes via executor results

## Benchmarks

| Config | D=3 | D=4 | D=5 | D=6 | D=7 |
|--------|-----|-----|-----|-----|-----|
| GATS   | 100%| 100%| 100%| 100%| 100%|
| Random | 97% | 93% | 70% | 43% | 27% |

## File Structure

```
gats2/
├── gats/
│   ├── core.py      # Data types
│   ├── verifier.py  # Action compiler
│   ├── search.py    # MCTS
│   └── agent.py     # Main agent
├── bench/
│   └── tasks.py     # Task generators
├── run.py           # Entry point
└── test.py          # Tests
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

### LLM Proposer
Implement the `Proposer` protocol:
```python
class LLMProposer:
    def propose(self, state, legal_actions, k=5) -> list[Candidate]: ...
    def repair(self, candidate, feedback) -> Candidate: ...
```

## Publication Checklist

- [x] Soundness at scale (0 invalid executions)
- [x] Controlled difficulty sweep
- [x] Ablation studies
- [x] Anytime improvement curves
- [x] Reproducible (fixed seeds, event logs)
- [ ] External benchmark (API-Bank, WebArena)
- [ ] Open model replication
