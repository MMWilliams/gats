# GATS development: MCTS, layered models, and transition evidence

This is the active development implementation requested on November 1, 2026.
It replaces the business integration runtime's root-action UCB1 bandit with
full Monte Carlo tree search. The frozen v2.1/v2.2 research sources and their
tables are preserved. Current paper:
`C:/Users/reese/gats-submissions/jair/gats_v2_2_2026_09_17.pdf`.

## Run and inspect

From `C:/Users/reese/gats_clean/gats2`, using Python 3.10 or later:

```powershell
python -m gats_mcts benchmark --output results/mcts-new --seeds 11 29 47
python -m gats_mcts inspect --database results/mcts-new/graph.sqlite
python -m gats_mcts audit --database results/mcts-new/graph.sqlite --output results/mcts-new/model-audit.json --max-states 256
python -m pytest tests/test_mcts_model.py tests/test_mcts_search.py tests/test_mcts_graph.py tests/test_mcts_scenarios.py -q
python -m integrations.run_deep
```

The benchmark needs a fresh output directory and writes:

- `report.json`: every case, condition, seed, executed plan, actual outcome,
  model error, search work, coverage, and reproducible outcome digest.
- `graph.sqlite`: the populated reusable world-model graph.
- `graph.json`: readable states, predicted distributions, observed transitions,
  source identifiers, action catalogs, and a content checksum.
- `traces/*.jsonl`: actual sandbox executions used for training, with stable
  event IDs; repeated ingestion does not inflate sample counts.
- `catalogs/*.json`: the public actions and supplied L1 references for each condition.

All data is synthetic or from the existing in-memory company fixtures. Business
approvals in the benchmark are simulated. The actual integration runtime keeps
its live precondition checks and explicit human-approval callback.

## Model contract

Public `Action` objects contain names, positive/negative preconditions, positive
costs, and optional dependency/write footprints. They contain neither execution
callbacks nor hidden effects. The evaluator owns execution. L1 accepts only
explicit supplied `Contract` objects, uses add-wins deletions, and evaluates all
conditional guards against the same pre-state.

L2 keeps outcome counts from observations, including failed/no-op outcomes and
deletions. Exact-state evidence overrides a reference immediately. A reference
contradicted by an observation is quarantined across states; it cannot silently
return as an exact prediction on another cache miss.

By default, observations apply only to an exact state/action pair. Cross-state
transfer requires an explicit **complete** `context` and `writes` declaration.
The L2 key includes that context, preconditions, negative guards, and all written
facts. Outcomes are learned assignments to the write footprint; unrelated facts
in the query state remain unchanged. At least two samples are required for this
transfer. This supports irrelevant-fact changes without assuming an arbitrary
delta learned in one state is a universal action schema. Incorrect or incomplete
declared dependencies can still produce incorrect generalization.

Predictions carry empirical outcome frequencies, support, and a Wilson lower
bound for the modal outcome. These are descriptive statistics, not guaranteed
calibration or a probability that the model itself is correct. L1 probability
one means deterministic under its supplied contract; it does not certify the
contract. Unmodeled actions abstain. There is no invented L3 output.

## Search and graph semantics

MCTS performs PUCT selection at each decision node, expands one new successor,
rolls out through model predictions, samples empirical chance outcomes, and
backs discounted return through the entire selected path. Costs affect both
guidance and return. The rollout uses optimistic h-add guidance from public
contracts and observed additions; it never asks an execution oracle for value.
That heuristic is not a proof of reachability. Terminal reward is one; bounded
unfinished rollouts receive at most one quarter as heuristic value.

Search has explicit simulation, depth and prediction-request limits. Cycles are
cut within simulations. Deterministic no-ops are omitted; stochastic transitions
with a no-op outcome retain that outcome. Root choice uses visit count, then
mean value and prior. Ties and sampling have a reproducible local seed.
Value statistics reset for each decision/goal; the transition graph persists.
This prevents one task's backed-up rewards contaminating a later task.

The graph uses full SHA256 state IDs over canonical JSON and a namespace-bound
catalog. Catalog or model-policy changes require a new namespace. Observation
event IDs are idempotent; conflicting duplicates are rejected. Observations
invalidate all cached predictions for that action. Stored predictions carry
model revisions, so a restarted model rebuilds evidence and reuses only current
predictions. SQLite foreign keys and the inspector check structural integrity;
the export checksum detects accidental modification, not a malicious writer who
can recompute the checksum. Observation source strings identify provenance but
do not authenticate an external sensor. Use one model writer per namespace.

The graph has two separate kinds of knowledge: model predictions and observed
executions. Search never turns a simulation into an observation. In the business
runtime, denied or refused actions never create L2 evidence. Its returned plan is
a modeled sequence; every execution step still checks current system facts.

## Ingest your own sourced traces

Start with a generated catalog or create the same JSON schema. Each JSONL row
needs `event`, `source`, `before` (facts), `action` (name), and `after` (facts).
Provide complete observed state snapshots. Use stable unique execution event IDs.

```powershell
python -m gats_mcts ingest --database my-graph.sqlite --namespace my-workflow-v1 --catalog catalog.json --traces executions.jsonl
python -m gats_mcts inspect --database my-graph.sqlite --export my-graph.json
python -m gats_mcts plan --database my-graph.sqlite --namespace my-workflow-v1 --state '[\"ready\"]' --goal '[\"done\"]'
```

The escaped quotes above work with Windows PowerShell's native argument passing.
`plan` returns one recommended action and search statistics; it executes nothing.
The Python API accepts frozensets directly, avoiding command-shell JSON quoting.

## Evaluation boundaries

The panel covers 13 research fixtures and ten established business workflows.
It includes long chains, parallel prerequisites, consumed/regenerated resources,
irreversible traps, a 40-way distraction set, nonuniform costs, cycles, no-ops,
conditional effects, negative guards, impossible goals and already-complete goals.

The four conditions are exact L1; cold L2 with no references or demonstrations;
L2 populated from sandbox execution traces; and corrupted L1 references plus
the same traces. Training demonstrations come from a bounded best-first teacher
using public exact contracts, then are checked by actual sandbox execution.
Bounded exploration of up to 128 reachable states also samples recovery and
off-path contexts. Each of three resets contributes one execution per distinct
declared action context; repeated contexts along a path do not inflate support.
Evaluation uses new
irrelevant episode facts, so L2 must transfer through its declared footprints.

The separate model audit uses a larger reachable-state cap when requested,
compares all applicable transitions in those states against sandbox execution,
excludes evaluation observations, and reports abstentions and cap truncation.
This is supervised **known-workflow transfer**, not a blind test of new schemas,
unseen domains, partial observation, real API behavior, or stochastic production
systems. Cold-model failures remain in the report. Stochastic outcome support
has separate regression tests, not a broad stochastic benchmark. Finite MCTS
budgets cannot guarantee an optimal or successful plan. For actual outcomes and
the seven pre-existing legacy failures, see [VERIFICATION.md](VERIFICATION.md).

Algorithm background: [Browne et al., A Survey of Monte Carlo Tree Search Methods](https://repository.essex.ac.uk/4117/).
