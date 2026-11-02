# Fifty corporate tasks, two agents, one shared world model

Run on November 2, 2026 from branch `mcts-corporate` with Python 3.13 on
Windows 11. Everything is synthetic, deterministic and stdlib only. Human
approvals are simulated. These are known-workflow results, not production
telemetry and not a publication claim.

## Setup

`integrations/corporate_tasks.py` defines fifty workflows across finance, HR,
legal, compliance, IT, security, sales, marketing, procurement, facilities and
operations. Each is its own company system: a published tool contract
(preconditions and declared effects), private ground truth, one or two human
gates on the high-risk step, and usually a valid but unnecessary distractor.

Twenty-four systems drift from their published contract the way real stacks
do: fourteen tools add an undocumented fact, eight report success and
silently deliver nothing (a slower fallback tool exists), two reset an earlier
fact without saying so. Two tasks (t14 visa sponsorship, t30 laptop refresh)
cannot be finished at all: the only tool that can deliver a required fact
silently fails. Seven systems contain a trap tool that destroys progress.

Each task is run as a case: the same system plus one irrelevant fact
(`case:c1`, `case:c2`, ...) the model has never seen, so evidence has to
transfer through repaired contracts rather than replay an exact state.

- **Agent A** runs cases c1 and c2 cold. Its only knowledge is the published
  contract (L1). Every guarded execution is written to `shared.sqlite` (L2).
- **Agent B** runs in a separate process with no memory of A. For each task it
  retrieves A's catalog and evidence from the graph, runs case c1 (seen by A)
  and case c3 (seen by nobody), and writes its own executions back.

Each decision is one MCTS search with 48 simulations from live truth. When no
simulation reaches the goal the runtime heads for the nearest tool the model
cannot predict and records those steps as exploration; when no such tool is
reachable it stops.

## Results

| | Agent A (c1, c2) | Agent B (c1, c3) |
| --- | ---: | ---: |
| Cells (task x case) | 100 | 100 |
| Goals reached | 96 | 96 |
| Expected outcomes (the 4 misses are the 2 unreachable tasks) | 100 | 100 |
| Executions | 649 | 631 |
| Exploration steps | 77 | 0 |
| Surprises (observed state not among predicted outcomes) | 25 | 1 |
| Drift events flagged by the audit layer | 45 | 33 |
| Predictions served from L1 / L1r / L2 | 33,166 / 214 / 27 | 29,503 / 1,276 / 1,548 |
| Catalogs retrieved from the graph | 50 of 100 (A's own, on c2) | 100 of 100 |
| Executions spent on the two unreachable tasks | 16 | 0 |

Comparison checks (`results/corporate/comparison.json`), all true:
B retrieved every catalog from the graph; B's success rate is not below A's;
B was never surprised by a tool A had already run; B's surprises are not above
A's; B explored nothing; B used preserved evidence (L2 and L1r served); B
executed no more than A on the shared case; B never executed an unreachable
task; the graph passes integrity validation after both agents wrote to it.

B's one surprise is t22 litigation hold, case c3: the MCTS tie-break chose
`suspend_retention`, whose silent failure A never observed because A's
tie-breaks picked the manual override both times. B learned it and finished
with the fallback. Preserved evidence was never wrong; it was simply absent.

Graph after both passes: 50 catalogs, 2,563 states, 1,280 observations, 3,859
cached predictions, no integrity errors, export SHA256
`0503c84334c8261f20c909478e64a271a6f04ffa9374d1e292ab102249b9d333`.
Agent A's outcome digest is identical across two fresh runs:
`6fa45220e348111837d72b9c5b3dc387317501c9cd03e44ddba7952acfe20644`.

## What the second pass bought

- On the unseen case, every drifted tool A had run twice was predicted by a
  repaired contract (`L1r`): no surprise, no exploration, no wasted step.
- On the two unreachable tasks, A spent 16 executions finding out; B spent 0
  because the graph already held the no-op evidence for the failing tool.
- On the seen case, exact evidence (L2) replaced L1 for every executed step.

## Caveats

Exact-state L2 keys; L1r needs two distinct observed states and applies only
where every common pre-state fact is present. Drift here is deterministic;
stochastic drift is covered by regression tests, not by this experiment. Two
agents wrote sequentially (and, in the unit test, interleaved from one
process); simultaneous writers from two processes are not exercised.

Reproduce: the three `run_corporate` commands in `gats_mcts/README.md`, then
`python -m gats_mcts inspect --database results/corporate/shared.sqlite`.
