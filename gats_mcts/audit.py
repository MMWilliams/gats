"""Audit model accuracy beyond chosen plans on bounded reachable sandbox states."""
from collections import deque
import json
from pathlib import Path

from .benchmark import CONDITIONS
from .domain import read_catalog
from .graph import TransitionGraph
from .model import LayeredWorldModel
from .scenarios import business_scenarios, research_scenarios


def reachable_samples(case, max_states):
    queue, seen, rows = deque([case.initial]), {case.initial}, []
    truncated = False
    while queue:
        state = queue.popleft()
        for action in sorted(case.actions, key=lambda a: a.name):
            if not action.applicable(state):
                continue
            # A new irrelevant marker checks footprint transfer, not exact-key replay.
            before = state | {"episode:model-audit:unseen"}
            after = case.execute(before, action.name)
            rows.append((before, action.name, after))
            successor = after - {"episode:model-audit:unseen"}
            if successor not in seen:
                if len(seen) < max_states:
                    seen.add(successor)
                    queue.append(successor)
                else:
                    truncated = True
    return rows, len(seen), truncated


def audit_population(database, output, *, max_states=128, cases=None):
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if not Path(database).is_file():
        raise ValueError("database does not exist")
    source = TransitionGraph(database)
    reports = []
    try:
        for case in cases if cases is not None else research_scenarios() + business_scenarios():
            samples, states, truncated = reachable_samples(case, max_states)
            for condition in CONDITIONS:
                namespace = case.name + "/" + condition
                row = source.db.execute("SELECT body FROM catalogs WHERE namespace=?", (namespace,)).fetchone()
                if row is None:
                    raise ValueError("missing benchmark namespace: " + namespace)
                catalog = json.loads(row[0])
                actions, contracts = read_catalog(catalog)
                memory = TransitionGraph()
                try:
                    model = LayeredWorldModel(actions, contracts, graph=memory,
                                              min_support=catalog["model"]["min_support"])
                    # Evaluation observations are deliberately excluded from this audit.
                    model.ingest(r for r in source.observations(namespace)
                                 if r["source"].startswith("sandbox-reset-execution:"))
                    correct = missing = wrong = 0
                    examples = []
                    with memory.prediction_batch():
                        for before, name, after in samples:
                            prediction = model.predict(before, name)
                            if prediction is None:
                                missing += 1
                                kind = "unmodeled"
                            elif len(prediction.outcomes) == 1 and prediction.outcomes[0].state == after:
                                correct += 1
                                continue
                            else:
                                wrong += 1
                                kind = "incorrect"
                            if len(examples) < 5:
                                examples.append({"kind": kind, "action": name,
                                                 "before": sorted(before), "truth": sorted(after)})
                    reports.append({"scenario": case.name, "condition": condition,
                                    "reachable_states_examined": states, "state_cap_reached": truncated,
                                    "transition_checks": len(samples), "correct": correct,
                                    "abstentions": missing, "incorrect": wrong, "examples": examples,
                                    "passed": wrong == 0 and (condition != "l1" or missing == 0)})
                finally:
                    memory.close()
        result = {"scope": "Bounded reachable-state audit against separate sandbox execution. "
                           "Uses training observations only and an unseen irrelevant episode fact. "
                           "State-cap flags and abstentions are retained; not proof outside sampled states.",
                  "max_states_per_scenario": max_states, "rows": reports,
                  "summary": {condition: {key: sum(r[key] for r in reports if r["condition"] == condition)
                                          for key in ("transition_checks", "correct", "abstentions", "incorrect")}
                              for condition in CONDITIONS},
                  "passed": all(r["passed"] for r in reports)}
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        source.close()
