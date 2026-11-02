# HANDOFF: GATS MCTS hardening, shared graph, corporate pass, 12-hour program (2026-11-02)

## Goal
Done means: (1) hardening + shared graph + fifty-task two-agent pass merged to master and pushed
(DONE, 578dda2); (2) the long-horizon program run (`GATS-Program`, 95 months x 2 cases) completes
with `status: complete`, all months `chain_valid`, no integrity errors, every cell success, and its
learning curve is recorded in `integrations/PROGRAM_RUN.md`.

## Files changed (this session, branch mcts-corporate -> master)
- gats_mcts/model.py: truth-first footprints, L1r repaired contracts (Repair), from_graph, sync, MODEL_VERSION 2, cache bound.
- gats_mcts/graph.py: WAL + busy timeout, catalog/namespaces/observations_since, store_predictions flag, L1r layer.
- gats_mcts/search.py: SearchResult.goal_hits; rollout returns (value, reached).
- integrations/core.py: Tool.hidden_del; AuditedRuntime(graph, agent, task), retrieved, step(), run() GATS loop,
  _toward_unmodeled exploration, divergence tracking, event ids agent/task/seq.
- integrations/corporate_tasks.py (50 tasks), run_corporate.py (A/B/compare), run_program.py (months program).
- tests: test_mcts_hardening.py, test_corporate.py, test_program.py; test_mcts_model.py footprint assertion updated.
- Docs: gats_mcts/README.md, VERIFICATION.md, integrations/CORPORATE_RESULTS.md, PROGRAM_RUN.md.

## Decisions and why
- Exploration lives in the runtime (labeled basis `exploration`), never in the planner: search.py stays pure.
- When no rollout reaches the goal and nothing unmodeled is reachable, the runtime stops (`no_modeled_path`)
  instead of executing the most-visited hopeless action; this is what gives pass 2 zero wasted executions.
- L1r needs two distinct observed states and every common pre-state fact in the query state (guard that
  keeps the older projection tests honest).
- Program run: predictions not stored (`store_predictions=False`), one plan per month from live truth,
  95 months chosen from month-1 calibration (452.7 s planning + 5 s execution at 2 cases).

## Approaches that failed
- Unguarded L1r (any consistent delta) broke two existing projection tests: aliasing of context facts.
- Checking "no goal-reaching rollout" via `max(mean) <= 0.25` is unsound under averaging; `goal_hits` instead.
- Top-level event ids without month/segment collided across months and after resume.
- A plan-once program with one case per workstream closed a month in 1 s; useless as a 12-hour task.

## Next step
Wait for `GATS-Program` to finish (started 2026-11-02, about 12 h). Then:
`python -m integrations.run_program status --output C:/Users/reese/gats-program`, read `report.json`,
write the learning curve (explorations/divergences/served layers per month, planning seconds) into
`integrations/PROGRAM_RUN.md`, run `python -m gats_mcts inspect --database C:/Users/reese/gats-program/shared.sqlite`,
commit, push, `schtasks /delete /tn GATS-Program /f`. If it stalled, `run.log` and `status.json` say where.

## Test command
`py -3.13 -m pytest tests -q -rN --tb=no` -> 50 passed, 7 failed (the seven legacy test_paper_claims failures,
pre-existing). `py -3.13 -m integrations.run_deep` -> ALL LAYERS PASSED (106 s).
