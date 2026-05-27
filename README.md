# GATS — Graph-Augmented Tree Search with Layered World Models

Reproducibility repository for the preprint
**"GATS: Graph-Augmented Tree Search with Layered World Models for Efficient
Agent Planning"** ([paper/gats_preprint.pdf](paper/gats_preprint.pdf)).

GATS combines UCB1 tree search with a three-layer world model
(symbolic → learned → LLM fallback) to plan multi-step agent tasks with
**zero LLM inference calls** during planning, while matching or exceeding
LATS and ReAct.

## Headline results (from `python reproduce.py`)

| Benchmark                          | GATS *b*=20 | LATS *b*=20 | ReAct  |
| ---------------------------------- | ----------: | ----------: | -----: |
| 100 synthetic multi-step tasks     |    **100%** |        92 % |   64 % |
| 12-category stress test (120 tasks)|    **100%** |      88.9 % | 23.9 % |
| LLM calls per task (planning)      |       **0** |          37 |     13 |

---

## Reproduce everything with one command

```bash
git clone https://github.com/MMWilliams/gats
cd gats
pip install -r requirements.txt
python reproduce.py
```

Total wall time: **~2 minutes** on a laptop. No GPU. No API keys.
The pipeline is fully deterministic — every random number is seeded.

### What you get

After `reproduce.py` finishes, `results/` contains:

```
results/
├── main_eval.json         raw output of the main evaluation (Tables 1, 2, 3, 5)
├── stress_test.json       raw output of the 12-category stress test (Table 6)
├── sensitivity.json       hyperparameter sensitivity (Table 7)
├── summary.csv            tidy-format aggregate for downstream tooling
├── tables.tex             booktabs LaTeX of the main tables
└── figures/
    ├── fig1_main_results.{pdf,png}      Success rate per method (Table 1)
    ├── fig2_budget_ablation.{pdf,png}   GATS scaling vs search budget (Table 2)
    ├── fig3_llm_calls.{pdf,png}         Cost-vs-performance Pareto plot
    ├── fig4_stress_test.{pdf,png}       Per-category heatmap (Table 6)
    └── fig5_world_model.{pdf,png}       Layered world-model ablation (Table 3)
```

### Useful flags

```bash
python reproduce.py --quick      # ~10s smoke test (smaller seeds/tasks)
python reproduce.py --only stress   # run just the stress test
python reproduce.py --no-figs    # data only, no plots
python figures.py                # re-render figures from existing JSON
```

To replicate the LLM-backed runs from the paper (Llama 3.2 via Ollama):

```bash
pip install requests
ollama serve            # in a separate shell
python experiments/run_gats_eval.py --backend ollama --n-tasks 100 \
    --seeds 42 123 456 789 1000
```

---

## Repository structure

```
gats/
├── reproduce.py            single entry point — runs every experiment + figures
├── figures.py              renders publication-quality PDF/PNG plots
├── experiments/
│   ├── run_gats_eval.py    main evaluation + ablations (Tables 1, 2, 3, 5)
│   └── run_stress_test.py  12-category stress test (Table 6)
├── tests/
│   └── test_reproducibility.py   pytest smoke tests for the pipeline
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── gats_preprint.pdf
├── requirements.txt
└── pyproject.toml
```

That's the whole thing. Both `run_gats_eval.py` and `run_stress_test.py` are
self-contained — they don't depend on a separate `gats/` library — so there
is no install step beyond `pip install -r requirements.txt` and no hidden
state.

---

## How GATS works (one paragraph)

At each planning step, GATS runs a budgeted UCB1 tree search over the
applicable actions. For every candidate next state, the **layered world
model** predicts the effect:

- **L1 (Symbolic)** — exact precondition/effect matching when the action is
  known. Returns the next state deterministically with confidence 1.0.
- **L2 (Learned)** — statistical effect prediction from execution-log
  counts, returning the most-frequent observed effect.
- **L3 (LLM)** — fallback for novel actions, called once and cached.

States are scored by a BFS reachability heuristic (admissible — no LLM
calls). On benchmarks where L1 has full coverage, L3 is never invoked at
planning time. See §3 of the paper for the full algorithm.

---

## Citation

```bibtex
@article{williams2026gats,
  title  = {GATS: Graph-Augmented Tree Search with Layered World Models
            for Efficient Agent Planning},
  author = {Williams, Maureese},
  year   = {2026},
  eprint = {arXiv:2601.XXXXX}
}
```

---

## License

MIT — see [pyproject.toml](pyproject.toml).
