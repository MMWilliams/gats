"""Hardening: truth-first footprints, repaired contracts, and a shared graph."""
import pytest

from gats_mcts import Action, Contract, Effect, LayeredWorldModel, TransitionGraph


def test_footprint_violation_keeps_truth_and_disables_projection_only():
    a = Action("credit", {"confirmed"}, context={"confirmed"}, writes={"credited"})
    model = LayeredWorldModel([a], {"credit": Contract(Effect({"credited"}))}, min_support=3)
    # The real system also books an undocumented ledger entry: truth is kept.
    for i in range(2):
        after = {"confirmed", "credited", "ledger_pending", f"case:{i}"}
        assert model.observe({"confirmed", f"case:{i}"}, "credit", after, event=str(i), source="audit")
    assert model.coverage()["footprint_violations"] == ["credit"]
    assert model.coverage()["quarantined"] == ["credit"]
    exact = model.predict({"confirmed", "case:0"}, "credit")
    assert exact.layer == "L2"
    assert exact.outcomes[0].state == {"confirmed", "credited", "ledger_pending", "case:0"}
    # Below min_support nothing transfers: not the declared footprint, not a repair.
    assert model.predict({"confirmed", "case:9"}, "credit") is None
    model.observe({"confirmed", "case:2"}, "credit",
                  {"confirmed", "credited", "ledger_pending", "case:2"}, event="2", source="audit")
    p = model.predict({"confirmed", "case:9"}, "credit")
    assert p.layer == "L1r" and p.support == 3
    assert p.outcomes[0].state == {"confirmed", "credited", "ledger_pending", "case:9"}


def test_repaired_contract_transfers_consistent_delta_and_retracts_on_conflict():
    a = Action("notify", {"approved"})
    model = LayeredWorldModel([a], {"notify": Contract(Effect({"notified"}))}, min_support=2)
    # Silent failure in two distinct states: the declared effect never lands.
    for i in range(2):
        model.observe({"approved", f"case:{i}"}, "notify", {"approved", f"case:{i}"}, event=str(i), source="audit")
    p = model.predict({"approved", "case:new"}, "notify")
    assert p.layer == "L1r" and p.support == 2 and 0 < p.confidence_lower < 1
    assert p.outcomes[0].state == {"approved", "case:new"}
    # It worked once: the evidence is no longer one delta, so the repair is withdrawn.
    model.observe({"approved", "case:2"}, "notify", {"approved", "notified", "case:2"}, event="2", source="audit")
    assert model.predict({"approved", "case:other"}, "notify") is None
    assert model.predict({"approved", "case:2"}, "notify").layer == "L2"
    assert model.coverage()["repaired"] == []


def test_repair_needs_two_distinct_states_and_learns_deletions():
    a = Action("ship", {"paid"})
    model = LayeredWorldModel([a], {"ship": Contract(Effect({"sent"}))}, min_support=2)
    for i in range(3):
        model.observe({"paid", "stock"}, "ship", {"paid", "sent", "manifest"}, event=str(i), source="audit")
    assert model.predict({"paid", "stock", "other"}, "ship") is None
    model.observe({"paid", "stock", "x"}, "ship", {"paid", "sent", "manifest", "x"}, event="x", source="audit")
    assert model.coverage()["repaired"] == ["ship"]
    p = model.predict({"paid", "stock", "other"}, "ship")
    assert p.layer == "L1r" and p.outcomes[0].state == {"paid", "sent", "manifest", "other"}
    # "stock" was present in every observed pre-state: without it the delta is unproven.
    assert model.predict({"paid", "other"}, "ship") is None
    assert model.graph.validate() == []


def test_shared_graph_second_connection_retrieves_catalog_and_syncs_new_evidence(tmp_path):
    path = tmp_path / "shared.sqlite"
    writer = TransitionGraph(path)
    actions = [Action("send", {"ready"})]
    contracts = {"send": Contract(Effect({"done"}))}
    a = LayeredWorldModel(actions, contracts, graph=writer, namespace="acme")
    reader = TransitionGraph(path)
    assert reader.namespaces() == ["acme"]
    b = LayeredWorldModel.from_graph(reader, "acme")
    assert b.actions == a.actions and dict(b.contracts) == dict(a.contracts)
    assert b.min_support == a.min_support
    assert b.predict({"ready"}, "send").layer == "L1"
    a.observe({"ready"}, "send", {"failed"}, event="A/1", source="agent:A")
    # B sees A's execution evidence without restarting, and its cache is invalidated.
    assert b.sync() == 1
    assert b.predict({"ready"}, "send").layer == "L2"
    assert b.coverage()["quarantined"] == ["send"]
    b.observe({"ready", "x"}, "send", {"failed", "x"}, event="B/1", source="agent:B")
    assert a.sync() == 1 and a.coverage()["observations"] == 2 and b.sync() == 0
    assert a.revisions == b.revisions
    # Interleaved writers in separate namespaces on one file never collide.
    c = LayeredWorldModel(actions, contracts, graph=reader, namespace="other")
    for i in range(20):
        a.observe({"ready", str(i)}, "send", {"done", str(i)}, event=f"A/{i+2}", source="agent:A")
        c.observe({"ready", str(i)}, "send", {"done", str(i)}, event=f"C/{i}", source="agent:C")
    assert writer.validate() == [] and reader.export() == writer.export()
    with pytest.raises(ValueError, match="namespace"):
        LayeredWorldModel.from_graph(reader, "missing")
    writer.close()
    reader.close()
