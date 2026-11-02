# GATS research backlog

- Long-horizon program run (reese 11-02): 95 months x 2 cases launched as
  scheduled task `GATS-Program` into `C:\Users\reese\gats-program`; when it
  finishes, read `report.json`, record the learning curve and the month timing
  in `integrations/PROGRAM_RUN.md`, and delete the task
  (`schtasks /delete /tn GATS-Program /f`). See `.claude/HANDOFF.md`.
- Program design follow-ups: months are structurally identical, so the monthly
  plan is the same ordering problem; add month-to-month variation (volume,
  capacity tokens such as limited legal or CFO slots) and real top-level drift.
- Evidence replay per model instance is O(all observations of the namespace);
  fine at this scale, but a much longer run needs aggregated evidence or
  long-lived model instances.
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
