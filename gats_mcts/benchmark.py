"""Reproducible development panel and populated graph; no publication claims."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from .domain import Contract, Effect, catalog_data
from .graph import TransitionGraph, canonical
from .model import LayeredWorldModel
from .scenarios import research_scenarios, business_scenarios, training_traces
from .search import MCTSPlanner


CONDITIONS = ("l1", "l2_cold", "l2_trained", "mixed_corrupt")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "gats_mcts").glob("*.py")) + [root / "integrations" / name for name in
            ("core.py", "companies.py", "complex_cases.py")]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def episode(case, model, seed, *, simulations=128, max_steps=64):
    state = case.initial | {f"episode:evaluation:{seed}"}
    planner = MCTSPlanner(model, simulations=simulations, max_depth=64, seed=seed)
    path, errors, requests, simulations_done, deepest = [], 0, 0, 0, 0
    before_usage = model.usage.copy()
    termination = "step_limit"
    for step in range(max_steps):
        if case.goal <= state:
            termination = "goal"
            break
        result = planner.search(state, case.goal)
        requests += result.model_requests
        simulations_done += result.simulations
        deepest = max(deepest, result.max_tree_depth)
        if result.action is None:
            termination = result.status
            break
        action = model.by_name[result.action]
        if not action.applicable(state):
            raise AssertionError("planner returned inapplicable action")
        prediction = model.predict(state, action.name)
        after = case.execute(state, action.name)
        if prediction is None or after not in {o.state for o in prediction.outcomes}:
            errors += 1
        model.observe(state, action.name, after,
                      event=f"eval/{seed}/{step}/{action.name}", source="sandbox-evaluation:" + case.name)
        path.append(action.name)
        state = after
    if case.goal <= state:
        termination = "goal"
    return {"success": case.goal <= state, "plan": path, "final_state": sorted(state),
            "cost": sum(model.by_name[n].cost for n in path), "prediction_errors": errors,
            "model_requests": requests, "simulations": simulations_done, "max_tree_depth": deepest,
            "termination": termination, "usage": dict(model.usage - before_usage)}


def run_panel(output, seeds=(11, 29, 47), *, simulations=128, cases=None, progress=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "graph.sqlite"
    if database.exists():
        raise ValueError("output graph already exists; choose a fresh output directory to preserve its evidence")
    graph = TransitionGraph(database)
    sources = source_hashes()
    cases = list(cases if cases is not None else research_scenarios() + business_scenarios())
    rows, coverage, datasets = [], [], []
    try:
        for case in cases:
            traces = training_traces(case)
            trace_path = output / "traces" / (case.name + ".jsonl")
            trace_path.parent.mkdir(exist_ok=True)
            trace_bytes = "".join(canonical(r) + "\n" for r in traces).encode("utf-8")
            trace_path.write_bytes(trace_bytes)
            datasets.append({"scenario": case.name, "events": len(traces),
                             "sha256": hashlib.sha256(trace_bytes).hexdigest(),
                             "collection": "demonstrations plus up to 128 reachable states; "
                                           "one sample per declared context per reset, three resets",
                             "provenance": case.provenance})
            for condition in CONDITIONS:
                references = case.contracts if condition == "l1" else {}
                if condition == "mixed_corrupt":
                    references = {name: (Contract(Effect(c.effect.add)) if i % 2 else Contract())
                                  for i, (name, c) in enumerate(sorted(case.contracts.items()))}
                namespace = case.name + "/" + condition
                model = LayeredWorldModel(case.actions, references, graph=graph, namespace=namespace)
                if condition in {"l2_trained", "mixed_corrupt"}:
                    # Warm some supplied predictions before learning to exercise invalidation.
                    for a in model.applicable(case.initial):
                        model.predict(case.initial, a.name)
                    model.ingest(traces)
                    if model.ingest(traces) != 0:
                        raise AssertionError("duplicate traces inflated support")
                catalog_path = output / "catalogs" / (case.name + "-" + condition + ".json")
                catalog_path.parent.mkdir(exist_ok=True)
                catalog_path.write_text(json.dumps(catalog_data(case.actions, references), indent=2), encoding="utf-8")
                before = model.coverage()
                for seed in seeds:
                    row = episode(case, model, seed, simulations=simulations)
                    expected = (case.goal <= case.initial if condition == "l2_cold" else case.expected_success)
                    row.update(scenario=case.name, family=case.family, condition=condition, seed=seed,
                               expected_success=expected, passed=(row["success"] == expected))
                    rows.append(row)
                coverage.append({"namespace": namespace, "before_evaluation": before,
                                 "after_evaluation": model.coverage()})
            if progress:
                subset = [r for r in rows if r["scenario"] == case.name]
                progress(f"{case.name}: {sum(r['passed'] for r in subset)}/{len(subset)} expected outcomes")
        integrity = graph.validate()
        export = graph.export()
        (output / "graph.json").write_text(json.dumps(export, indent=2), encoding="utf-8")
        summary = {}
        for condition in CONDITIONS:
            subset = [r for r in rows if r["condition"] == condition]
            summary[condition] = {"cells": len(subset), "successes": sum(r["success"] for r in subset),
                                  "expected_outcomes": sum(r["passed"] for r in subset),
                                  "prediction_errors": sum(r["prediction_errors"] for r in subset),
                                  "model_requests": sum(r["model_requests"] for r in subset)}
        result = {"schema": "gats-mcts-development-v1", "seeds": list(seeds), "simulations_per_decision": simulations,
                  "environment": {"python": sys.version, "platform": platform.platform()},
                  "source_sha256": sources, "source_unchanged_during_run": sources == source_hashes(),
                  "scope": "Fully observed deterministic sandbox workflows. L2 uses supervised scenario traces, "
                           "complete declared context/write footprints, and unseen irrelevant episode facts. "
                           "This measures known-workflow transfer, not novel-domain or production generalization.",
                  "summary": summary, "rows": rows, "coverage": coverage, "datasets": datasets,
                  "graph": {"states": len(export["states"]), "predictions": len(export["predictions"]),
                            "observations": len(export["observations"]), "sha256": export["sha256"],
                            "integrity_errors": integrity},
                  "passed": not integrity and all(r["passed"] for r in rows) and sources == source_hashes()}
        result["outcome_sha256"] = hashlib.sha256(canonical(rows).encode("utf-8")).hexdigest()
        (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        graph.close()
