# MCTS and world-model development verification

Working directory: `C:/Users/reese/gats_clean/gats2`. Python 3.13.5.
This is a new development implementation. Frozen v2.1/v2.2 sources and result
tables are historical evidence and must retain their bytes.

| Requirement or material risk | Source | Runnable check | Explicit pass criterion | Result/evidence |
| --- | --- | --- | --- | --- |
| Full MCTS, bounded search, multi-step return and cost handling | User request; Browne et al. 2012 MCTS phases | `python -m pytest tests/test_mcts_search.py -q` | Delayed reward, trap, cost, cycle, stochastic-outcome, depth and query-budget cases pass; no environment import or execution in planner | Passed in final 36-test development suite; AST boundary check included |
| L1 applies preconditions, negative guards, simultaneous conditional effects and deletions | v2.2 add-wins STRIPS contract; extension to explicit conditional rules | `python -m pytest tests/test_mcts_model.py -q` | Independent finite-state truth tables agree; invalid actions rejected; applicable unknown effects abstain | Passed, including exhaustive 64-state conditional truth table |
| L2 learns only observations; bounded generalization and contradicted references | v2.2 review: missing schemas and exact-state limitation; inferred risk of context aliasing | `python -m pytest tests/test_mcts_model.py -q` | Different contexts remain distinct; transfer only with explicit complete read/write footprint; empirical outcomes preserve frequencies; contradicted references quarantined | Passed; no new-test failures or skips |
| Persistent populated graph distinguishes predictions and observations | User request; v2.2 cache transparency; integration provenance precedent | `python -m pytest tests/test_mcts_graph.py -q` | Restart retains evidence; changed observations invalidate predictions; namespaces isolated; duplicate/conflicting events and corrupt exports detected | Passed; transaction batching, rollback and simulation/evidence boundary checked |
| Broad scenario execution with L1 and populated L2 | User request; existing integration lab cases | `python -m gats_mcts benchmark --output results/mcts-populated --seeds 11 29 47` | Every claimed success checked by sandbox environment; full case/condition denominator, failures, query counts and finite-training coverage reported; zero hidden-effect access by planner | Exit 0; 276/276 expected outcomes, 23 scenarios x 4 conditions x 3 seeds; source hashes unchanged |
| Repeatability, graph integrity, no evidence inflation | Existing deterministic research contract | `python -m pytest tests/test_mcts_scenarios.py -q` | Fixed-seed repeat has identical normalized outputs; reingesting same trace adds zero evidence; graph validation passes | Passed; repeated four-scenario panel matches rows, graph and outcome digests; CLI ingestion/inspection/planning exercised |
| Existing repository gates | pyproject.toml; tests/test_paper_claims.py; tests/test_reproducibility.py | `python -m pytest tests -q -ra --junitxml=results/mcts-tests.xml` | Record pre-change failures; no new failure in unchanged legacy checks | Failed verification: 44 passed, 7 failed, 0 skipped. The same seven legacy failures occurred before edits (8 passed, 7 failed baseline). All 36 new tests pass |
| Business runtime uses MCTS and learns only after guarded execution | User request to replace UCB1; integrations/core.py execution and audit contract | `python -m integrations.run_deep` and `python -m pytest tests/test_mcts_scenarios.py -q` | All integration, injection, audit, graph, generated-system and crash-resume checks pass; denied actions add no L2 evidence | Exit 0, ALL LAYERS PASSED in 152.7s: 12 integration cases, 120/120 generated systems, 240-tool scale, 3,894/3,894 tamper mutations, 604/604 invalid-action refusals, 114/114 crash-resume prefixes |
| Accuracy audit does not inspect only chosen trajectories | v2.2 final research review: policy-selected model error; user request for broad model correctness | `python -m gats_mcts audit --database results/mcts-populated/graph.sqlite --output results/mcts-populated/model-audit.json --max-states 256` | No incorrect modeled successor in audited deterministic cases; L1 covers every audited applicable transition; publish L2 abstentions and state caps | Exit 0; L1, trained L2 and mixed/corrupt each 6,609/6,609 correct, 0 abstentions; cold L2 abstains on all 6,609. 884 reachable states examined; 21 scenarios exhausted, 2 hit the cap |
| Provenance explanations retain linear DAG traversal rather than scanning every edge at every ancestor | Existing integrations/graph_db.py why() complexity contract; slow deep validation observed | `python -m pytest tests/test_mcts_graph.py -q` and `python -m integrations.run_deep` | One edge scan per explanation query; full explanation agrees with independent raw-record oracle | Regression demonstrated 8 scans before the fix and 1 afterward; all 7 graph tests pass. Deep corpus: 1,442 nodes / 2,669 edges, 60/60 corruptions detected, 549/549 why queries and 11/11 blast-radius queries match independent oracles |

Manual review: inspect public descriptors and planner imports for hidden effects;
inspect generated scenario/coverage summaries and provenance. These checks cannot
establish perfection outside the finite cases, accuracy of supplied real-world
contracts, or calibrated probabilities from sparse observations. Scenario data
will be explicitly labeled sandbox-generated, not real production telemetry.

Algorithm reference: [Browne et al., A Survey of Monte Carlo Tree Search Methods](https://repository.essex.ac.uk/4117/).

## Final development evidence, November 1, 2026

The scenario panel's L1, trained-L2, and mixed/corrupted-reference conditions
each solve 66/69 cells: all 22 achievable fixtures over three seeds. The
deliberately impossible fixture accounts for the three failures in each
condition. Cold L2 solves only the three already-complete cells and fails the
remaining 66, as expected. No executed transition is mispredicted. This is a
populated known-workflow development panel, not a publication holdout.

The final corpus has 1,092 independently reset sandbox training events. The
SQLite graph contains 92 isolated catalogs, 16,833 states, 4,416 observation
records (training stored in two model conditions plus evaluation executions),
and 19,460 cached predictions. Do not count duplicated training events across
model conditions as additional independent evidence. The inspector exits 0
with no integrity errors. Native PowerShell CLI planning was also checked on
a copy of the database: a new-state chain query returned `step_00`, with 64
simulations and tree depth four, without executing an action.

Artifacts:

- `results/mcts-populated/report.json` — source hashes, environment and full rows.
- `results/mcts-populated/model-audit.json` — transition coverage and cap flags.
- `results/mcts-populated/graph.sqlite` and `graph.json` — populated graph and export.
- `results/mcts-tests.xml` — final full-suite JUnit result.
- `integrations/results/deep_results.json` — final integration, scale, audit,
  crash-resume and independent provenance-graph checks.

Outcome SHA256: `8c5de80dc621a4f107c0084d44262654c933a8a6f28dffc5866af68120351f25`.
Graph export SHA256: `d63658915e4bb1705d3288298ce791f80648ea9f4feba898bcbfeaca9759405c`.
Final runtime source files match the report's source hashes. `git diff --check`
and `python -m compileall -q gats_mcts integrations/core.py` exit 0.

## Pre-existing failures and development findings

The unchanged historical `test_paper_claims.py` fails three tests because its
task loader lacks `task_seed`, one because the old planner never queries L1,
two because the old model lacks the expected L2 observation API, and one
because its BFS horizon is shorter than the asserted 20-step contract. These
failures were demonstrated before implementation and remain visible; no tests,
assertions, thresholds or discovery rules were weakened. The full repository
gate therefore has **failed verification**, despite passing development checks.

The first population audit found an unobserved recovery action in the
impossible-goal fixture. Demonstrations alone did not cover it. Bounded
reachable-state exploration now supplies actual executions for such contexts;
deduplication prevents repeated contexts along a demonstration inflating support.

Initial long runs exposed per-edge SQLite commits and repeated fixed-point
heuristic construction. Predictions now commit in a single transaction per
decision, and compiled worklist h-add values are reused while goal and model
revision are unchanged. Rollback and evidence separation have regression tests.
The final scale test plans 240 actions over 240 tools in 8.03 seconds on this host.
Earlier interrupted profiling/population runs in `results/mcts`,
`results/mcts-2026-11-01`, and `results/mcts-validated` are incomplete development
artifacts, not completed evaluation evidence. Use `results/mcts-populated`.

Remaining scientific work is tracked in `.claude/TODO.md`. No universal
perfection, unseen-schema generalization, real API reliability, or comparative
planning superiority is established by this finite sandbox corpus.

The existing provenance graph's `why()` query now indexes prerequisite edges
once per public query, sharing the index through recursive supports. The index
is local, so adding records or deserializing a graph cannot leave stale adjacency.
The independent raw-record oracle and all corruption assertions remain unchanged.
