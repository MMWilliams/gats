# GATS research backlog

- Reconcile the seven pre-existing `tests/test_paper_claims.py` failures with
  the historical `experiments/run_gats_eval.py` interface. Preserve frozen
  research bytes and claim provenance; do not weaken assertions to turn the
  archived experiment green. See `gats_mcts/VERIFICATION.md` for the baseline.
- Evaluate MCTS on a newly frozen, structurally held-out domain panel with
  matched search/model budgets and informed baselines. Current results are
  populated known-workflow transfer, not new-schema generalization.
- Add real, authorized execution telemetry with authenticated provenance and
  independently checked complete dependency/write declarations. Current corpus
  consists entirely of sandbox executions and simulated approvals.
- Extend the scenario panel to stochastic and partially observed environments;
  current stochastic coverage is regression tests, not a broad experiment.
- Extend bounded coverage for `business_cascade` and `business_voltway_pro`:
  the 256-state audit cap was reached. Other 21 finite scenarios were exhausted.
- Seek coauthor/outside review of the existing v2.2 manuscript before any new
  publication claims; the MCTS development results do not modify that paper.
