import copy

import pytest

from gats_mcts import Action, Contract, Effect, LayeredWorldModel, TransitionGraph


def test_restart_namespace_separation_and_cache_invalidation(tmp_path):
    path = tmp_path / "graph.sqlite"
    graph = TransitionGraph(path)
    actions = [Action("send", {"ready"})]
    contracts = {"send": Contract(Effect({"done"}))}
    model = LayeredWorldModel(actions, contracts, graph=graph, namespace="a")
    assert model.predict({"ready"}, "send").layer == "L1"
    model.observe({"ready"}, "send", {"failed"}, event="event-1", source="sensor")
    assert model.predict({"ready"}, "send").layer == "L2"
    other = LayeredWorldModel(actions, contracts, graph=graph, namespace="b")
    assert other.predict({"ready"}, "send").layer == "L1"
    export = graph.export()
    assert graph.validate() == [] and graph.verify_export(export)
    graph.close()
    graph = TransitionGraph(path)
    model = LayeredWorldModel(actions, contracts, graph=graph, namespace="a")
    assert model.predict({"ready"}, "send").outcomes[0].state == {"failed"}
    assert model.coverage()["quarantined"] == ["send"]
    assert model.usage["cache"] == 1
    assert not model.observe({"ready"}, "send", {"failed"}, event="event-1", source="sensor")
    assert model.coverage()["observations"] == 1
    assert graph.export() == export
    graph.close()


def test_same_namespace_rejects_changed_contract_and_model_policy():
    graph = TransitionGraph()
    actions = [Action("a")]
    LayeredWorldModel(actions, graph=graph)
    with pytest.raises(ValueError, match="different catalog"):
        LayeredWorldModel(actions, {"a": Contract(Effect({"x"}))}, graph=graph)
    with pytest.raises(ValueError, match="different catalog"):
        LayeredWorldModel(actions, graph=graph, min_support=3)


def test_predicted_transitions_never_become_observations():
    model = LayeredWorldModel([Action("a")], {"a": Contract(Effect({"x"}))})
    model.predict(set(), "a")
    export = model.graph.export()
    assert len(export["predictions"]) == 1 and export["observations"] == []
    assert model.coverage()["observations"] == 0
    altered = copy.deepcopy(export)
    altered["states"][0]["facts"] = ["forged"]
    assert not model.graph.verify_export(altered)


def test_hash_uses_unambiguous_fact_encoding_and_integrity_detects_corruption():
    from gats_mcts.graph import state_id
    assert state_id({"a\nb"}) != state_id({"a", "b"})
    model = LayeredWorldModel([Action("a")], {"a": Contract(Effect({"x"}))})
    model.predict(set(), "a")
    model.graph.db.execute("UPDATE states SET facts='[\"forged\"]'")
    assert "state hash mismatch" in model.graph.validate()


def test_simulation_batch_commits_once_and_cannot_create_execution_evidence(tmp_path):
    graph = TransitionGraph(tmp_path / "graph.sqlite")
    model = LayeredWorldModel([Action("a")], {"a": Contract(Effect({"x"}))}, graph=graph)
    sql = []
    graph.db.set_trace_callback(sql.append)
    with graph.prediction_batch():
        for i in range(20):
            model.predict({str(i)}, "a")
        with pytest.raises(RuntimeError, match="simulation batch"):
            model.observe(set(), "a", {"x"}, event="not-real", source="bad")
    assert sql.count("COMMIT") == 1
    assert len(graph.export()["predictions"]) == 20
    assert graph.export()["observations"] == []
    graph.close()


def test_failed_prediction_batch_rolls_back(tmp_path):
    graph = TransitionGraph(tmp_path / "graph.sqlite")
    model = LayeredWorldModel([Action("a")], {"a": Contract(Effect({"x"}))}, graph=graph)
    with pytest.raises(RuntimeError):
        with graph.prediction_batch():
            model.predict(set(), "a")
            raise RuntimeError("simulated interruption")
    assert graph.export()["predictions"] == []
    assert graph.export()["states"] == []
    graph.close()


def test_provenance_explanation_scans_edges_once_and_matches_raw_records():
    from integrations.core import AuditedRuntime, CompanySystem, Tool
    from integrations.graph_db import ProvenanceGraph
    from integrations.graph_validation import oracle_why_matches
    system = CompanySystem("chain", {"p0"}, [
        Tool(f"step{i}", frozenset({f"p{i}"}), frozenset({f"p{i+1}"})) for i in range(8)])
    runtime = AuditedRuntime(system, {"p8"})
    assert runtime.execute(runtime.plan())
    graph = ProvenanceGraph()
    graph.add_run("chain", system.initial_facts, runtime.records)

    class CountedEdges(list):
        scans = 0

        def __iter__(self):
            self.scans += 1
            return super().__iter__()

    graph.edges = CountedEdges(graph.edges)
    explanation = graph.why("chain", "p8")
    assert explanation["action"] == "step7"
    # why() promises linear DAG traversal; one edge index per public query
    # avoids a full scan at every ancestor. No timing-dependent threshold.
    assert graph.edges.scans == 1
    assert oracle_why_matches(graph, "chain", system.initial_facts, runtime.records, "p8")
