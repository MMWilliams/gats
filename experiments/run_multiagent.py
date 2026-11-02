#!/usr/bin/env python3
"""
Multi-agent shared-world-model experiment.

Why this design
---------------
In the main benchmark every action carries a complete symbolic spec, so L1
always answers and the learned layer L2 is never reached. Sharing L2 in that
setting would measure exactly nothing.

The interesting — and realistic — regime is PARTIAL knowledge: a fleet of
agents that does NOT know every action up front and has to learn the missing
dynamics from execution. That is precisely the "learn the specs instead of
hand-writing them" problem, and it is where sharing can pay off.

Setup
-----
A population of N agents runs the same task distribution in sequence. A fixed
fraction of action specs is withheld from L1, so those actions must be resolved
by L2 (learned from observed executions) or, failing that, L3 (a guess).

    INDEPENDENT : every agent keeps its own world model. Agent k starts as
                  ignorant as agent 0.
    SHARED      : all agents read and write ONE world model, so every executed
                  transition becomes a prior for every later agent.

Reported per agent index: success rate, and how often the agent had to fall
back to guessing (L3). If sharing works, later agents in the SHARED condition
should guess less and succeed more, while INDEPENDENT stays flat.
"""
from __future__ import annotations
import sys, json, argparse, random, statistics
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from run_gats_eval_wm import (          # the corrected harness: WM is in the loop
    State, Action, LayeredWorldModel, GATSPlanner,
    load_planning_tasks,
)


def strip_specs(wm: LayeredWorldModel, hidden_frac: float, rng: random.Random) -> None:
    """Withhold a fraction of action specs from the symbolic layer.

    L1 answers only for actions it knows; anything hidden must come from L2
    (if the fleet has observed it) or L3 (a guess).
    """
    names = sorted(wm.actions.keys())
    n_hide = int(len(names) * hidden_frac)
    for name in rng.sample(names, n_hide):
        del wm.actions[name]


def build_wm(tasks, hidden_frac: float, seed: int) -> LayeredWorldModel:
    """One world model over the UNION of every action in the task distribution.

    Building it from a single task (as an earlier version did) silently left
    every other task's actions unknown, so L3 fired even at hidden=0 and the
    'hidden fraction' was confounded with 'actions from other tasks'.
    """
    union = {}
    for task in tasks:
        for a in task.actions:
            union[a.name] = a
    wm = LayeredWorldModel(list(union.values()))
    strip_specs(wm, hidden_frac, random.Random(seed))
    return wm


def run_condition(tasks, n_agents: int, hidden_frac: float, shared: bool,
                  budget: int, seed: int, explore: float = 0.0) -> List[Dict]:
    rng = random.Random(seed)
    per_agent = []
    shared_wm = build_wm(tasks, hidden_frac, seed) if shared else None

    for agent_i in range(n_agents):
        order = list(tasks)
        rng.shuffle(order)

        # an independent agent still accumulates its own experience across its
        # own tasks — it simply never sees anybody else's
        wm = shared_wm if shared else build_wm(tasks, hidden_frac, seed)

        successes = 0
        layer_tot = {1: 0, 2: 0, 3: 0}

        for task in order:
            before = dict(wm.layer_usage)
            planner = GATSPlanner(wm, budget=budget, explore_bonus=explore)
            initial = State(frozenset(task.goal_facts), frozenset(task.initial_facts))
            result = planner.plan(initial, task.actions)

            if result.success:
                successes += 1
            for k in (1, 2, 3):
                layer_tot[k] += wm.layer_usage[k] - before.get(k, 0)

        total = sum(layer_tot.values()) or 1
        per_agent.append({
            "agent": agent_i,
            "success_rate": successes / len(order),
            "l1": layer_tot[1], "l2": layer_tot[2], "l3": layer_tot[3],
            "guess_frac": layer_tot[3] / total,
            "learned_frac": layer_tot[2] / total,
        })
    return per_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--hidden", type=float, default=0.5,
                    help="fraction of action specs withheld from L1")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--explore", type=float, default=0.0,
                    help="v2 optimism bonus for unpredictable reversible actions")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--output", default="results/multiagent.json")
    args = ap.parse_args()

    tasks = load_planning_tasks("data/api_bank", args.n_tasks, use_hard=True)
    print(f"{len(tasks)} tasks · {args.agents} agents · "
          f"{args.hidden:.0%} of specs withheld · {len(args.seeds)} seeds\n")

    out = {}
    for cond, shared in (("independent", False), ("shared", True)):
        runs = [run_condition(tasks, args.agents, args.hidden, shared,
                              args.budget, s, args.explore) for s in args.seeds]
        # average across seeds, per agent index
        merged = []
        for i in range(args.agents):
            merged.append({
                "agent": i,
                "success_rate": statistics.mean(r[i]["success_rate"] for r in runs),
                "guess_frac":   statistics.mean(r[i]["guess_frac"] for r in runs),
                "learned_frac": statistics.mean(r[i]["learned_frac"] for r in runs),
            })
        out[cond] = merged

        print(f"── {cond.upper()}")
        print(f"   {'agent':>5} {'success':>9} {'guessed(L3)':>13} {'learned(L2)':>13}")
        for m in merged:
            print(f"   {m['agent']:>5} {m['success_rate']:>8.1%} "
                  f"{m['guess_frac']:>12.1%} {m['learned_frac']:>12.1%}")
        print()

    first = out["shared"][0]["success_rate"]
    last  = out["shared"][-1]["success_rate"]
    ind_f = out["independent"][0]["success_rate"]
    ind_l = out["independent"][-1]["success_rate"]
    print(f"SHARED      agent 0 -> agent {args.agents-1}: {first:.1%} -> {last:.1%}")
    print(f"INDEPENDENT agent 0 -> agent {args.agents-1}: {ind_f:.1%} -> {ind_l:.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
