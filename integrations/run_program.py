"""
Run the company for N months: one long-horizon program planned and executed by
GATS at two levels, on the shared graph.

  python -m integrations.run_program run --months 60 --cases 2 --output C:/Users/reese/gats-program --database C:/Users/reese/gats-program/shared.sqlite
  python -m integrations.run_program run --months 60 --cases 2 --output ... --database ... --resume
  python -m integrations.run_program status --output C:/Users/reese/gats-program

The program is itself a company system. Its tools are whole workstreams:
`run:t02:a` executes this month's supplier-invoice workflow for case a against
a fresh instance of that system with the low-level GATS runtime (MCTS, live
precondition checks, human gates, the shared graph). Its published contract
says the workstream completes; its ground truth is whether the low-level
episode actually reached the goal. A month closes only when every case of
every workstream is done; closing opens the next month.

Level 1, the program: at the start of each month one MCTS plan is created
from live truth over the abstract actions and their dependencies (vendor
onboarding before the PO, the PO before the invoice; invoice, payroll and
bank reconciliation before the month-end close; the close before the
regulatory filing and the board book; NDA and MSA before the enterprise deal;
the deal before customer onboarding, and so on). The plan is saved to
plans/mNNN.json before anything executes, then followed step by step; a
refusal (a dependency that never materialized) triggers a replan from live
truth.

Level 2, the workstreams: the GATS loop of the corporate pass, on the same
shared graph, so every month's drift evidence serves the next month's plans
at both levels.

Checkpoints: every abstract step appends to records.jsonl and rewrites
status.json; `--resume` rebuilds the program state from the records without
re-running finished workstreams. Synthetic systems, simulated approvals.
"""
from __future__ import annotations

import argparse
import json
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from gats_mcts import TransitionGraph

from .core import AuditedRuntime, CompanySystem, Tool, ToolError, verify_chain
from .corporate_tasks import BY_ID, TASKS, instance

APPROVE = lambda tool: True
LAYERS = ("L1", "L1r", "L2")
UNREACHABLE = {"t14", "t30"}  # covered by the corporate pass; a month could never close with them
WORKSTREAMS = [t for t in TASKS if t.id not in UNREACHABLE]
DEPENDENCIES = {
    "t01": ["t41"], "t02": ["t01"], "t04": ["t02", "t03", "t08", "t11"], "t07": ["t33"],
    "t20": ["t25", "t26", "t27"], "t21": ["t04"], "t25": ["t09"], "t27": ["t26"],
    "t31": ["t26", "t47"], "t33": ["t17", "t18"], "t34": ["t33"], "t35": ["t34"],
    "t38": ["t34"], "t41": ["t29"], "t43": ["t33"], "t48": ["t01"], "t50": ["t04", "t21"],
}
MAX_REPAIRS_PER_MONTH = 10


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProgramSystem(CompanySystem):
    """The program as a company: its tools run whole workstreams for the current month."""

    def __init__(self, months, cases, graph, output, workstreams=WORKSTREAMS, budget=48):
        self.workstreams = list(workstreams)
        self.cases = string.ascii_lowercase[:cases]
        ids = {t.id for t in self.workstreams}
        done = {f"done:{t.id}:{c}" for t in self.workstreams for c in self.cases}
        opened = {f"open:{t.id}:{c}" for t in self.workstreams for c in self.cases}
        tools = [Tool(f"run:{t.id}:{c}",
                      frozenset({f"open:{t.id}:{c}"} | {f"done:{d}:{c}" for d in DEPENDENCIES.get(t.id, ()) if d in ids}),
                      frozenset({f"done:{t.id}:{c}"}), frozenset({f"open:{t.id}:{c}"}))
                 for t in self.workstreams for c in self.cases]
        for month in range(1, months + 1):
            opens = {"program:complete"} if month == months else {f"month:{month + 1}"} | opened
            tools.append(Tool(f"close_month_{month:03d}", frozenset({f"month:{month}"} | done),
                              frozenset(opens), frozenset({f"month:{month}"} | done), risk="high"))
        super().__init__("program", {"month:1"} | opened, tools)
        self.months, self.graph, self.output, self.budget = months, graph, Path(output), budget
        self.cells = []

    def month(self):
        return next(int(f.split(":")[1]) for f in self._facts if f.startswith("month:"))

    def call(self, tool_name):
        tool = self.tools[tool_name]
        missing = tool.preconditions - self._facts
        if missing:
            raise ToolError(tool_name, frozenset(missing))
        before = frozenset(self._facts)
        if tool_name.startswith("run:"):
            _, task_id, case = tool_name.split(":")
            self._facts -= tool.effects_del
            if self._run_workstream(BY_ID[task_id], case):
                self._facts |= tool.effects_add
        else:
            self._facts |= tool.effects_add
            self._facts -= tool.effects_del
        after = frozenset(self._facts)
        self.call_log.append(tool_name)
        return after - before, before - after

    def _run_workstream(self, task, case):
        month = self.month()
        label = f"m{month:03d}{case}"
        runtime = AuditedRuntime(instance(task, label), task.goal, budget=self.budget, max_steps=40,
                                 horizon=24, graph=self.graph, agent="program", task=f"{task.id}/{label}")
        started = time.perf_counter()
        success = runtime.run(human_approver=APPROVE)
        usage = runtime.world_model.usage
        self.cells.append({
            "month": month, "task": task.id, "case": case, "success": success, "status": runtime.status,
            "executions": sum(r["type"] == "execution" for r in runtime.records),
            "explorations": runtime.explorations, "divergences": runtime.divergences,
            "drift_events": len(runtime.drift_events),
            "served": {layer: usage.get("served_" + layer, 0) for layer in LAYERS},
            "retrieved_from_graph": runtime.retrieved, "chain_valid": verify_chain(runtime.records),
            "seconds": round(time.perf_counter() - started, 3)})
        folder = self.output / "cells"
        folder.mkdir(exist_ok=True)
        with (folder / f"m{month:03d}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task": task.id, "case": case, "records": runtime.records}) + "\n")
        return success

    def restore(self, records):
        """Rebuild program facts from checkpointed abstract executions without re-running them."""
        for rec in records:
            self._facts |= set(rec["added"])
            self._facts -= set(rec["removed"])


def month_summary(cells, month, seconds, planning_seconds, repairs, chain_valid):
    rows = [c for c in cells if c["month"] == month]
    return {"month": month, "seconds": round(seconds, 1), "planning_seconds": planning_seconds,
            "repairs": repairs, "chain_valid": chain_valid,
            "cells": len(rows), "successes": sum(c["success"] for c in rows),
            "executions": sum(c["executions"] for c in rows),
            "explorations": sum(c["explorations"] for c in rows),
            "divergences": sum(c["divergences"] for c in rows),
            "drift_events": sum(c["drift_events"] for c in rows),
            "served": {layer: sum(c["served"][layer] for c in rows) for layer in LAYERS}}


def plan_month(runtime, path, progress):
    """One modeled plan for the month from live truth, saved before anything executes."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    sequence = runtime.plan()
    if sequence is None:
        raise RuntimeError("no modeled plan for the month")
    plan = {"sequence": sequence, "planning_seconds": round(time.perf_counter() - started, 1),
            "searches": len(runtime.search_results),
            "simulations": sum(r.simulations for r in runtime.search_results),
            "model_requests": sum(r.model_requests for r in runtime.search_results), "created_at": now()}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if progress:
        progress(f"plan {path.stem}: {len(sequence)} steps in {plan['planning_seconds']}s")
    return plan


def run_program(months, cases, output, database, *, resume=False, limit=None, budget=48,
                workstreams=WORKSTREAMS, progress=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records_path, status_path, report_path = output / "records.jsonl", output / "status.json", output / "report.json"
    envelopes = []
    if records_path.exists():
        if not resume:
            raise ValueError("output already holds a program run; pass --resume or choose a new directory")
        envelopes = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    graph = TransitionGraph(database, store_predictions=False)
    system = ProgramSystem(months, cases, graph, output, workstreams, budget)
    system.restore(e["record"] for e in envelopes if e["record"]["type"] == "execution")
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {
        "started_at": now(), "months": months, "cases": cases, "status": "running"}
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "schema": "gats-program-v2", "months": months, "cases": cases,
        "workstreams": [t.id for t in system.workstreams],
        "dependencies": {k: v for k, v in DEPENDENCIES.items() if k in {t.id for t in system.workstreams}},
        "month_reports": [], "status": "running"}
    size = len(system.workstreams) * cases + 8
    # Observation event ids carry the month and the run segment, so a resumed
    # month never reuses an id from before the interruption.
    segment = status.get("segments", 0) + 1
    status["segments"] = segment
    executed_total = sum(e["record"]["type"] == "execution" for e in envelopes)
    done_this_call = 0
    try:
        month = system.month() if "program:complete" not in system.observe() else months + 1
        month_index = sum(1 for e in envelopes if e["month"] == month and e["record"]["type"] == "execution")
        while month <= months and (limit is None or done_this_call < limit):
            goal = {"program:complete"} if month == months else {f"month:{month + 1}"}
            runtime = AuditedRuntime(system, goal, budget=budget, max_steps=size, horizon=size,
                                     graph=graph, agent="program", task=f"m{month:03d}/s{segment}")
            status.update(month=month, phase="planning", updated_at=now())
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            plan = plan_month(runtime, output / "plans" / f"m{month:03d}.json", progress)
            steps = list(plan["sequence"])
            index, repairs, month_started = month_index, 0, time.perf_counter()
            while index < len(steps) and (limit is None or done_this_call < limit):
                name = steps[index]
                ok = runtime.step(name, APPROVE, "plan")
                record = runtime.records[-1]
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"month": month, "index": index, "action": name,
                                             "at": now(), "record": record}) + "\n")
                if ok:
                    index += 1
                    executed_total += 1
                    done_this_call += 1
                else:
                    # A dependency never materialized: replan the rest of the month from live truth.
                    repairs += 1
                    repaired = runtime.plan() if repairs <= MAX_REPAIRS_PER_MONTH else None
                    if not repaired:
                        status.update(status="stalled", stalled_at=now(), action=name)
                        report["status"] = "stalled"
                        break
                    steps = steps[:index] + repaired
                status.update(phase="executing", index=index, steps_in_month=len(steps), updated_at=now(),
                              executed_total=executed_total, chain_head=runtime.chain_head(),
                              cells=len(system.cells))
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                if progress and record["type"] == "execution":
                    progress(f"m{month:03d} {index:>3}/{len(steps)} {name:<18} "
                             f"{'ok' if ok else 'REFUSED'} cells={len(system.cells)}")
            if status["status"] == "stalled" or index < len(steps):
                break  # stalled, or the step limit was reached mid-month; resume continues here
            summary = month_summary(system.cells, month, time.perf_counter() - month_started,
                                    plan["planning_seconds"], repairs, verify_chain(runtime.records))
            summary["graph_observations"] = graph.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            report["month_reports"].append(summary)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if progress:
                progress(f"month {month} closed in {summary['seconds']}s (plan {summary['planning_seconds']}s): "
                         f"{summary['successes']}/{summary['cells']} cells, explore={summary['explorations']} "
                         f"diverge={summary['divergences']} repairs={repairs}")
            month += 1
            month_index = 0
        if "program:complete" in system.observe():
            status.update(status="complete", finished_at=now())
            report["status"] = "complete"
        status.update(updated_at=now(), executed_total=executed_total, cells=len(system.cells))
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        report.update(status=report["status"], integrity_errors=graph.validate(), cells=system.cells,
                      updated_at=now())
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        graph.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--months", type=int, required=True)
    run.add_argument("--cases", type=int, default=1, help="case instances of every workstream per month")
    run.add_argument("--output", required=True)
    run.add_argument("--database", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limit", type=int, help="abstract steps to execute in this invocation")
    run.add_argument("--budget", type=int, default=48)
    status = commands.add_parser("status")
    status.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        print(Path(args.output, "status.json").read_text(encoding="utf-8"))
        return 0
    report = run_program(args.months, args.cases, args.output, args.database, resume=args.resume,
                         limit=args.limit, budget=args.budget, progress=print)
    print(json.dumps({"status": report["status"], "months_closed": len(report["month_reports"]),
                      "cells": len(report["cells"]), "integrity_errors": report["integrity_errors"]}, indent=2))
    return 0 if report["status"] in {"complete", "running"} else 1


if __name__ == "__main__":
    sys.exit(main())
