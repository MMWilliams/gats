"""
Two agents, one shared world model, fifty corporate tasks.

  python -m integrations.run_corporate run --database results/corporate/shared.sqlite --agent A --cases c1 c2 --output results/corporate/A.json
  python -m integrations.run_corporate run --database results/corporate/shared.sqlite --agent B --cases c1 c3 --output results/corporate/B.json
  python -m integrations.run_corporate compare results/corporate/A.json results/corporate/B.json --output results/corporate/comparison.json

Agent A meets every system cold: the published tool contract (L1) is all it
knows. Every guarded execution is written to the shared SQLite graph (L2).
Agent B is a separate process with no memory of A. It retrieves A's catalog
and evidence from the graph, runs the same fifty tasks (one case A already
saw, one case nobody has seen) and writes its own executions back. The
comparison shows what the preserved evidence bought the second pass.

Stdlib only, deterministic, no LLM calls. All systems are synthetic and all
human approvals are simulated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from gats_mcts import TransitionGraph
from gats_mcts.graph import canonical

from .core import AuditedRuntime, verify_chain
from .corporate_tasks import BY_ID, TASKS, instance

APPROVE = lambda tool: True
LAYERS = ("L1", "L1r", "L2")


def run_agent(database, agent, cases, tasks=None, budget=48, progress=None):
    graph = TransitionGraph(database)
    tasks = [BY_ID[t] for t in tasks] if tasks else TASKS
    rows = []
    try:
        for task in tasks:
            for case in cases:
                rt = AuditedRuntime(instance(task, case), task.goal, budget=budget, max_steps=40,
                                    horizon=24, graph=graph, agent=agent, task=f"{task.id}/{case}")
                success = rt.run(human_approver=APPROVE)
                executions = [r for r in rt.records if r["type"] == "execution"]
                usage = rt.world_model.usage
                rows.append({
                    "task": task.id, "department": task.department, "title": task.title,
                    "case": case, "agent": agent, "retrieved_from_graph": rt.retrieved,
                    "expected_success": task.reachable, "success": success, "status": rt.status,
                    "executions": len(executions), "plan": [r["action"] for r in executions],
                    "explorations": rt.explorations, "divergences": rt.divergences,
                    "divergent_actions": rt.divergent_actions, "drift_events": len(rt.drift_events),
                    "searches": len(rt.search_results),
                    "simulations": sum(r.simulations for r in rt.search_results),
                    "model_requests": sum(r.model_requests for r in rt.search_results),
                    "served": {layer: usage.get("served_" + layer, 0) for layer in LAYERS},
                    "chain_valid": verify_chain(rt.records), "chain_head": rt.chain_head(),
                    "coverage": rt.world_model.coverage(),
                })
                if progress:
                    r = rows[-1]
                    progress(f"{agent} {task.id}/{case} {'ok ' if success else 'NO '} steps={r['executions']:<2} "
                             f"explore={r['explorations']} diverge={r['divergences']} served={r['served']}")
        integrity = graph.validate()
        export = graph.export()
    finally:
        graph.close()
    report = {"schema": "gats-corporate-shared-graph-v1", "agent": agent, "database": str(database),
              "cases": list(cases), "tasks": len(tasks), "budget": budget,
              "scope": "Fifty synthetic corporate workflows with simulated human approvals. About half "
                       "drift from their published contracts. Exact-state L2 keys; L1r repairs need two "
                       "distinct observed states. Known-workflow transfer, not production telemetry.",
              "summary": summarize(rows), "rows": rows,
              "graph": {"states": len(export["states"]), "observations": len(export["observations"]),
                        "predictions": len(export["predictions"]), "catalogs": len(export["catalogs"]),
                        "integrity_errors": integrity, "sha256": export["sha256"]},
              "passed": not integrity and all(r["chain_valid"] and r["success"] == r["expected_success"]
                                              for r in rows)}
    report["outcome_sha256"] = hashlib.sha256(canonical(rows).encode("utf-8")).hexdigest()
    return report


def summarize(rows):
    def total(key, subset=rows):
        return sum(r[key] for r in subset)
    unreachable = [r for r in rows if not r["expected_success"]]
    return {"cells": len(rows), "successes": sum(r["success"] for r in rows),
            "expected_outcomes": sum(r["success"] == r["expected_success"] for r in rows),
            "executions": total("executions"), "explorations": total("explorations"),
            "divergences": total("divergences"), "drift_events": total("drift_events"),
            "model_requests": total("model_requests"),
            "served": {layer: sum(r["served"][layer] for r in rows) for layer in LAYERS},
            "retrieved_from_graph": sum(r["retrieved_from_graph"] for r in rows),
            "executions_on_unreachable_tasks": total("executions", unreachable)}


def compare(first, second):
    a, b = first["summary"], second["summary"]
    shared = sorted(set(first["cases"]) & set(second["cases"]))
    by_cell = {(r["task"], r["case"]): r for r in first["rows"]}
    cells = []
    for r in second["rows"]:
        prior = by_cell.get((r["task"], r["case"]))
        cells.append({"task": r["task"], "case": r["case"], "title": r["title"],
                      "seen_by_first": prior is not None,
                      "first": prior and {k: prior[k] for k in ("success", "executions", "explorations", "divergences")},
                      "second": {k: r[k] for k in ("success", "executions", "explorations", "divergences", "served")}})
    second_on_shared = [r for r in second["rows"] if r["case"] in shared]
    first_on_shared = [r for r in first["rows"] if r["case"] in shared]
    executed_by_first = {}
    for r in first["rows"]:
        executed_by_first.setdefault(r["task"], set()).update(r["plan"])
    checks = {
        "second_retrieved_every_catalog_from_graph": b["retrieved_from_graph"] == b["cells"],
        "second_successes_not_below_first": b["successes"] / b["cells"] >= a["successes"] / a["cells"],
        # Preserved evidence is never wrong: a surprise needs a tool the first agent never ran.
        "second_never_surprised_where_first_has_evidence": all(
            not set(r["divergent_actions"]) & executed_by_first.get(r["task"], set()) for r in second["rows"]),
        "second_divergences_not_above_first": b["divergences"] <= a["divergences"],
        "second_explorations_zero": b["explorations"] == 0,
        "second_used_preserved_evidence": b["served"]["L2"] + b["served"]["L1r"] > 0,
        "second_executions_on_shared_cases_not_above_first":
            sum(r["executions"] for r in second_on_shared) <= sum(r["executions"] for r in first_on_shared),
        "second_never_executes_unreachable_tasks": b["executions_on_unreachable_tasks"] == 0,
        "both_graph_integrity_clean": not first["graph"]["integrity_errors"] and not second["graph"]["integrity_errors"],
    }
    return {"schema": "gats-corporate-two-agent-comparison-v1",
            "first": {"agent": first["agent"], "cases": first["cases"], "summary": a},
            "second": {"agent": second["agent"], "cases": second["cases"], "summary": b},
            "shared_cases": shared, "cells": cells, "checks": checks, "passed": all(checks.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="one agent runs every task; evidence goes to the shared graph")
    run.add_argument("--database", required=True)
    run.add_argument("--agent", required=True)
    run.add_argument("--cases", nargs="+", default=["c1"])
    run.add_argument("--tasks", nargs="*", help="task ids; default all fifty")
    run.add_argument("--budget", type=int, default=48, help="MCTS simulations per decision")
    run.add_argument("--output", required=True)
    cmp = commands.add_parser("compare", help="what the second agent gained from the first's evidence")
    cmp.add_argument("first")
    cmp.add_argument("second")
    cmp.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run_agent(args.database, args.agent, args.cases, args.tasks, args.budget, progress=print)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"summary": report["summary"], "graph": report["graph"], "passed": report["passed"],
                          "outcome_sha256": report["outcome_sha256"]}, indent=2))
        return 0 if report["passed"] else 1
    first = json.loads(Path(args.first).read_text(encoding="utf-8"))
    second = json.loads(Path(args.second).read_text(encoding="utf-8"))
    result = compare(first, second)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("first", "second", "checks", "passed")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
