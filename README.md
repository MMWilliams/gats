# GATS: Graph-Augmented Tree Search

**A planning framework that achieves 100% success rate with zero LLM calls during inference.**

## Overview

GATS (Graph-Augmented Tree Search) is a planning framework for LLM agents that eliminates inference-time LLM calls while outperforming existing methods like LATS and ReAct on complex planning tasks.

### How It Works

GATS uses a **layered world model** with UCB1 tree search:

```
L1 (Symbolic)  → Exact precondition-effect matching
L2 (Learned)   → Statistics from execution logs  
L3 (LLM)       → Fallback for unknown actions (cached)
```

During planning, GATS queries L1 → L2 → L3 in order. The LLM is only called for genuinely unknown actions, then cached for reuse.

---

## Installation

```bash
git clone https://github.com/yourusername/gats.git
cd gats
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- numpy
- requests

---

## Results Summary

### Main Evaluation (100 Synthetic Tasks)

| Method | Success Rate | Optimality | Avg Cost | Nodes |
|--------|-------------|------------|----------|-------|
| **GATS b=10** | **100.0%** | 1.00 | 4.2 | 167 |
| **GATS b=20** | **100.0%** | 1.00 | 4.2 | 334 |
| LATS b=10 | 92.0% | 0.99 | 4.0 | 37 |
| LATS b=5 | 70.7% | 0.99 | 3.8 | 17 |
| ReAct | 64.0% | 0.54 | 12.8 | 13 |
| Greedy (Oracle) | 100.0% | 1.00 | 4.2 | 17 |

### World Model Ablations

| Method | Success Rate | Description |
|--------|-------------|-------------|
| GATS b=10 | 100.0% | Full model (L1 + L2 + L3) |
| GATS no_l1 | 100.0% | Without symbolic layer |
| GATS no_l3 | 100.0% | Without LLM fallback |
| GATS b=5 | 84.0% | Reduced search budget |
| GATS b=1 | 0.0% | Minimal budget (greedy) |

### Budget Scaling

| Budget | GATS SR | LATS SR | Δ |
|--------|---------|---------|---|
| b=1 | 0.0% | — | — |
| b=5 | 84.0% | 70.7% | +13.3% |
| b=10 | 100.0% | 92.0% | +8.0% |
| b=20 | 100.0% | — | — |

### API-Bank Benchmark

| Level | GATS | LATS | ReAct | Description |
|-------|------|------|-------|-------------|
| L1 | 100.0% | 100.0% | 100.0% | Single API selection |
| L2 | 100.0% | 100.0% | 100.0% | Multi-API selection |
| L3 | 100.0% | 100.0% | 100.0% | Multi-step (constructed) |

*Note: API-Bank tests single-step API selection, so all methods achieve 100%.*

### Stress Test (12 Categories, 120 Tasks)

| Category | GATS b=20 | LATS b=20 | ReAct | Δ (GATS-LATS) |
|----------|-----------|-----------|-------|---------------|
| coding_task | 100.0% | 63.3% | 0.0% | **+36.7%** |
| deep_horizon | 100.0% | 63.3% | 0.0% | **+36.7%** |
| web_navigation | 100.0% | 63.3% | 0.0% | **+36.7%** |
| resource_puzzle | 100.0% | 86.7% | 16.7% | **+13.3%** |
| trap_heavy | 100.0% | 96.7% | 16.7% | +3.3% |
| commitment_cascade | 100.0% | 96.7% | 66.7% | +3.3% |
| memory_limit | 100.0% | 96.7% | 20.0% | +3.3% |
| critical_choice | 100.0% | 100.0% | 63.3% | 0.0% |
| deceptive | 100.0% | 100.0% | 63.3% | 0.0% |
| high_branching | 100.0% | 100.0% | 36.7% | 0.0% |
| no_backtrack | 100.0% | 100.0% | 0.0% | 0.0% |
| very_long_horizon | 100.0% | 100.0% | 3.3% | 0.0% |
| **Overall** | **100.0%** | **88.9%** | **23.9%** | **+11.1%** |

---

## Reproduce Results

### 1. Main Evaluation + Ablations

```bash
python run_ablation_study.py
```

**Output:**
```
Method                  Success Optimality   Avg Cost
──────────────────────────────────────────────────────
greedy                  100.0%       1.00        4.2
gats_b10                100.0%       1.00        4.2
gats_b20                100.0%       1.00        4.2
gats_no_l1              100.0%       1.00        4.2
gats_no_l3              100.0%       1.00        4.2
lats_b10                 92.0%       0.99        4.0
react                    64.0%       0.54       12.8
```

### 2. API-Bank Evaluation

```bash
python download_api_bank.py
python run_gats_eval_full.py --n-tasks 150 --quick
```

### 3. Stress Test

```bash
python run_stress_test.py --n-per-category 10 --seeds 42 123 456
```

**Output:**
```
GATS b=20:  100.0% overall
LATS b=20:   88.9% overall
ReAct:       23.9% overall
```

### 4. Quick Validation (~2 min)

```bash
python run_stress_test.py --quick
```

### 5. Full Reproduction (Windows)

```bash
python scripts/reproduce.py
```

Or run each step manually:
```bash
python download_api_bank.py
python download_level3.py
python run_ablation_study.py
python run_stress_test.py --n-per-category 10 --seeds 42 123 456
```

---

## Project Structure

```
gats2/
│
├── gats/                        # Core GATS implementation
│   ├── __init__.py
│   ├── agent.py                 # GATS planning agent
│   ├── core.py                  # State, Action, Plan primitives
│   ├── search.py                # UCB1 tree search algorithm
│   ├── world_model.py           # L1/L2/L3 layered world model
│   ├── verifier.py              # Formal plan verification
│   ├── calibration.py           # Confidence calibration
│   ├── llm.py                   # LLM interface (L3 layer)
│   └── event_log.py             # Execution logging
│
├── agents/                      # Baseline agents
│   └── tot.py                   # Tree of Thoughts baseline
│
├── bench/                       # Benchmarks
│   ├── __init__.py
│   ├── tasks.py                 # Synthetic task generation
│   ├── api_bank.py              # API-Bank loader
│   ├── api_bank_real.py         # Real API-Bank evaluation
│   ├── toolbench.py             # ToolBench loader
│   └── toolbench_real.py        # Real ToolBench evaluation
│
├── analysis/                    # Analysis tools
│   └── statistics.py            # Statistical analysis
│
├── scripts/                     # Automation scripts
│   ├── reproduce.py             # Full reproduction (cross-platform)
│   ├── download_datasets.py     # Download benchmarks
│   ├── run_ablations.py         # Ablation studies
│   ├── run_llm_experiments.py   # LLM-based experiments
│   ├── test_pipeline.py         # CI/CD tests
│   ├── api_bank_real.py         # API-Bank real evaluation
│   └── toolbench_real.py        # ToolBench real evaluation
│
├── run_gats_eval.py             # Basic GATS evaluation
├── run_gats_eval_full.py        # Full evaluation (synthetic + API-Bank)
├── run_stress_test.py           # Stress test (12 categories)
├── run_ablation_study.py        # Ablation studies
├── run_experiment.py            # General experiment runner
├── run_external.py              # External benchmark runner
│
├── download_api_bank.py         # Download API-Bank data
├── download_level3.py           # Generate Level 3 multi-step tasks
│
├── test.py                      # Unit tests
├── test_modules.py              # Module tests
│
├── results/                     # Output directory
│   ├── gats_eval.json           # Main evaluation results
│   ├── stress_test.json         # Stress test results
│   └── tables.tex               # LaTeX tables
│
├── requirements.txt
├── README.md
└── gats_paper.pdf               # Full paper (12 pages)
```

---

## Stress Test Categories

| Category | Steps | Description |
|----------|-------|-------------|
| coding_task | 11 | Script/API/pipeline development |
| web_navigation | 10-13 | Email, flight, hotel booking |
| deep_horizon | 8-12 | Long paths with shortcut traps |
| critical_choice | 8 | Memory allocation (wrong = stuck) |
| no_backtrack | 8-12 | Maze with locking doors |
| high_branching | 4 | 4-6 choices per step |
| resource_puzzle | 7 | Limited resources, correct order |
| trap_heavy | 5 | 3-7 attractive dead-ends |
| deceptive | 5 | Quick-gains path is a trap |
| memory_limit | 7 | Tool sequencing |
| very_long_horizon | 12-15 | Extended task with periodic traps |
| commitment_cascade | 4 | Early choices lock future options |

---

## Key Findings

### GATS vs LATS
- **+8% on synthetic tasks** (100% vs 92%)
- **+11% on stress test** (100% vs 88.9%)
- **Zero LLM calls** vs ~60 for LATS
- **Deterministic** (0% variance) vs 2% for LATS

### GATS vs ReAct
- **+36% on synthetic tasks** (100% vs 64%)
- **+76% on stress test** (100% vs 23.9%)

### When Methods Are Equal
- API-Bank L1/L2/L3: Single-step tasks (all 100%)
- Simple tasks with no dead-ends

### When GATS Excels
- Long horizon (10+ steps)
- High branching (many choices)
- Dead-ends (irreversible mistakes)
- Resource constraints

---

## Citation

```bibtex
@article{williams2026gats,
  title={GATS: Graph-Augmented Tree Search with Layered World Models 
         for Efficient Agent Planning},
  author={Williams, Maureese},
  year={2026}
}
```

## License

MIT License

## Author

**Maureese Williams**  
maureesewilliams@gmail.com