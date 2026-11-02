import json

from gats_mcts.benchmark import run_panel
from gats_mcts.scenarios import research_scenarios
from gats_mcts.__main__ import main


def test_seeded_panel_repeat_and_trained_transfer(tmp_path):
    cases = [c for c in research_scenarios() if c.name in {"consume_regenerate", "conditional_priority", "already_done", "unreachable"}]
    first = run_panel(tmp_path / "first", seeds=(11,), simulations=32, cases=cases)
    second = run_panel(tmp_path / "second", seeds=(11,), simulations=32, cases=cases)
    assert first["passed"] and second["passed"]
    assert first["rows"] == second["rows"]
    assert first["outcome_sha256"] == second["outcome_sha256"]
    assert first["graph"] == second["graph"]
    learned = [r for r in first["rows"] if r["condition"] == "l2_trained" and r["plan"]]
    assert learned and all(r["usage"]["served_L2"] > 0 for r in learned)
    assert all(r["usage"].get("served_L1", 0) == 0 for r in learned)
    assert all(r["prediction_errors"] == 0 for r in first["rows"])
    assert first["graph"]["observations"] > 0 and first["graph"]["predictions"] > 0
    from gats_mcts.audit import audit_population
    audit = audit_population(tmp_path / "first" / "graph.sqlite", tmp_path / "audit.json", cases=cases)
    assert audit["passed"]
    assert audit["summary"]["l1"]["transition_checks"] > 0
    assert audit["summary"]["l2_trained"]["incorrect"] == 0
    assert audit["summary"]["l2_trained"]["abstentions"] == 0
    assert audit["summary"]["mixed_corrupt"]["incorrect"] == 0
    assert audit["summary"]["l2_cold"]["abstentions"] > 0


def test_real_cli_trace_ingest_inspect_and_plan(tmp_path, capsys):
    catalog = tmp_path / "catalog.json"
    traces = tmp_path / "traces.jsonl"
    db = str(tmp_path / "graph.sqlite")
    catalog.write_text(json.dumps({"actions": [{"name": "complete", "preconditions": ["ready"],
                                                "context": [], "writes": ["ready", "done"]}], "contracts": {}}))
    traces.write_text("\n".join(json.dumps({"event": str(i), "before": ["ready", f"train{i}"],
                                           "action": "complete", "after": ["done", f"train{i}"], "source": "sandbox"})
                                for i in range(2)))
    command = ["ingest", "--database", db, "--namespace", "demo", "--catalog", str(catalog), "--traces", str(traces)]
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["new_observations"] == 2
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["new_observations"] == 0
    assert main(["plan", "--database", db, "--namespace", "demo", "--state", '["ready","unseen"]', "--goal", '["done"]']) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "complete"
    assert main(["inspect", "--database", db]) == 0
    assert json.loads(capsys.readouterr().out)["integrity_errors"] == []


def test_audited_runtime_uses_mcts_and_learns_only_guarded_executions():
    from integrations.companies import crestline, CRESTLINE
    from integrations.core import AuditedRuntime, verify_chain
    runtime = AuditedRuntime(crestline(), CRESTLINE["goal"])
    plan = runtime.plan()
    assert plan and runtime.search_results
    assert all(r.simulations > 0 for r in runtime.search_results)
    assert any(r.max_tree_depth > 1 for r in runtime.search_results)
    assert runtime.world_model.coverage()["observations"] == 0
    runtime.attempt_injection(CRESTLINE["lie_actions"], CRESTLINE["claimed_true"])
    assert runtime.world_model.coverage()["observations"] == 0
    assert not runtime.execute(plan)  # The human checkpoint blocks execution.
    executed = [r for r in runtime.records if r["type"] == "execution"]
    assert runtime.world_model.coverage()["observations"] == len(executed)
    assert runtime.status == "blocked_awaiting_human"
    resumed = runtime.plan()
    assert runtime.execute(resumed, human_approver=lambda _: True)
    assert verify_chain(runtime.records)
    assert runtime.world_model.graph.validate() == []
