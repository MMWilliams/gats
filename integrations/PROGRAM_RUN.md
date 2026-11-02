# The long-horizon program run

Started November 2, 2026 at 13:52 UTC (8:52 AM local) as a detached background
process (`C:\Users\reese\gats-program\run.cmd`; a Windows scheduled task was
not permitted in that session), from `C:\Users\reese\gats_clean\gats2` at
master 70f0aa0, writing to `C:\Users\reese\gats-program`. Expected to finish
about 11.5 to 12 hours later.

```
python -m integrations.run_program run --months 95 --cases 2 --output C:/Users/reese/gats-program --database C:/Users/reese/gats-program/shared.sqlite --resume
```

## The task

Run the company for 95 months. Every month, two cases of each of the 48
reachable corporate workstreams must be completed and the month closed; the
close opens the next month. Workstreams depend on each other inside a case:
vendor security review before vendor onboarding before the PO before the
supplier invoice; invoice, expenses, payroll and bank reconciliation before the
month-end close; the close before the regulatory filing and the board book;
NDA and MSA before the enterprise deal; the deal before customer onboarding,
the renewal, the escalation and the return; the hire before the access
request; CAB before the patch; CAB and the continuity test before the database
migration; access, CAB and the patch before the SOC 2 evidence.

That is 95 monthly plans of 97 abstract steps (9,215 planned actions), each
`run:` step a full audited GATS episode on a fresh company system (about 9,120
episodes, roughly 65,000 guarded tool executions), all on one shared graph that
started from the corporate two-agent pass.

## How it works

- Level 1 creates the month's plan with MCTS over the abstract actions (97
  actions, horizon 97, 48 simulations per decision, one search per step).
  Calibration on two months: planning 452.7 s then 403.1 s, execution of the
  96 cells 5.0 s then 5.4 s, 192/192 cells successful, month 2 with no
  exploration and no surprise. The plan is created from live truth, saved to
  `plans/mNNN.json`, then executes it step by step. A refusal triggers a replan
  from live truth; ten refusals in a month stall the program.
- Level 2 runs each workstream with the GATS loop of the corporate pass. The
  shared graph carries drift evidence from the corporate pass and every earlier
  month, so from the second case of month 1 on, repaired contracts replace
  exploration.
- Every abstract step appends to `records.jsonl` and rewrites `status.json`.
  Each workstream episode's full audit chain goes to `cells/mNNN.jsonl`.
  `report.json` gains one month report per closed month.

## Check on it

```
python -m integrations.run_program status --output C:/Users/reese/gats-program
Get-Content C:\Users\reese\gats-program\run.log -Tail 5
```

`status.json` shows the month, the phase (planning or executing), the step
index, cells completed and the chain head. Expected finish: about 11.5 to 12
hours after the start in `started_at` (95 months x about 7.2 minutes).

## If it stops

A detached process does not survive a reboot, and may not survive the session
that started it. Run `C:\Users\reese\gats-program\run.cmd` again (or register
it: `schtasks /create /tn GATS-Program /tr C:\Users\reese\gats-program\run.cmd /sc once /st 23:59`
then `schtasks /run /tn GATS-Program`). The launcher already passes
`--resume`: the program state is rebuilt from `records.jsonl`,
finished workstreams are not rerun, and the current month's plan is reused.
Observation event ids carry a run-segment number, so a resumed month never
collides with the records written before the interruption. Per-month audit
chains are verified per segment.

## When it is done

`report.json` has `status: complete`, 95 month reports with `chain_valid`
true, `integrity_errors` empty and every cell `success`. The month reports
give the learning curve: explorations and surprises per month, predictions
served by L1, L1r and L2, planning seconds, and the growing observation count.
`python -m gats_mcts inspect --database C:/Users/reese/gats-program/shared.sqlite`
validates the graph.

## Limits

Synthetic systems, simulated approvals, deterministic drift. The 95 months are
structurally identical, so the monthly plan is the same ordering problem each
time; the compute is real MCTS at width and depth 97, not a new problem each
month. The machine must stay awake for the run.
