# GATS research backlog

- Long-horizon program run (reese 11-02): a multi-phase enterprise program over
  the fifty corporate workstreams, planned up front and executed with repair
  for about twelve hours; see `.claude/HANDOFF.md` while it is in flight.
- Simultaneous writers: two processes appending to one shared graph at the
  same instant are not exercised (WAL + busy timeout only tested sequentially
  and interleaved in one process).
- Stochastic drift in the corporate panel (today's drift is deterministic).
- t17 and t22: agent A's tie-breaks never executed the silently failing tool,
  so its drift was learned only by B; consider a tie-break that prefers tools
  with evidence.

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
