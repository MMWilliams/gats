import itertools

import pytest

from gats_mcts import Action, Contract, Effect, LayeredWorldModel, Rule


def test_l1_independent_truth_table_and_simultaneous_conditions():
    action = Action("dispatch", {"paid"}, {"blocked"})
    contract = Contract(Effect({"sent"}, {"stock"}), (
        Rule({"priority"}, Effect({"express"})),
        Rule({"sent"}, Effect({"receipt"})),
        Rule({"stock"}, Effect({"packed"}), {"damaged"}),
    ))
    model = LayeredWorldModel([action], {action.name: contract})
    vocabulary = ["paid", "blocked", "stock", "priority", "sent", "damaged"]
    for bits in itertools.product((False, True), repeat=len(vocabulary)):
        before = frozenset(f for f, bit in zip(vocabulary, bits) if bit)
        prediction = model.predict(before, "dispatch")
        if "paid" not in before or "blocked" in before:
            assert prediction is None
            continue
        expected = set(before)
        expected.discard("stock")
        expected.add("sent")
        if "priority" in before:
            expected.add("express")
        if "sent" in before:
            expected.add("receipt")
        if "stock" in before and "damaged" not in before:
            expected.add("packed")
        assert prediction.outcomes[0].state == expected
        assert prediction.layer == "L1"


def test_add_wins_and_all_rules_read_prestate():
    contract = Contract(Effect({"b"}, {"a"}), (
        Rule({"a"}, Effect({"a", "c"})), Rule({"b"}, Effect({"d"}))))
    assert contract.apply(frozenset({"a"})) == {"a", "b", "c"}


def test_unknown_effects_abstain_and_public_actions_have_no_effects():
    a = Action("unknown", {"ready"})
    model = LayeredWorldModel([a])
    assert not hasattr(a, "effect") and not hasattr(a, "apply")
    assert model.predict({"ready"}, a.name) is None
    assert model.coverage()["uncovered_actions"] == ["unknown"]


def test_exact_evidence_does_not_leak_to_another_context():
    a = Action("route", {"ticket"})
    model = LayeredWorldModel([a])
    model.observe({"ticket", "vip"}, "route", {"ticket", "vip", "priority"}, event="1", source="test")
    assert model.predict({"ticket"}, "route") is None
    assert model.predict({"ticket", "vip"}, "route").outcomes[0].state == {"ticket", "vip", "priority"}


def test_projected_l2_transfers_preserves_unrelated_facts_and_deletes():
    a = Action("route", {"ticket"}, context={"vip"}, writes={"ticket", "priority", "normal"})
    model = LayeredWorldModel([a], min_support=2)
    for i in range(2):
        model.observe({"ticket", "vip", f"train:{i}"}, "route", {"vip", "priority", f"train:{i}"},
                      event=str(i), source="sandbox")
    p = model.predict({"ticket", "vip", "unseen-customer"}, "route")
    assert p.layer == "L2" and p.support == 2 and 0 < p.confidence_lower < 1
    assert p.outcomes[0].state == {"vip", "priority", "unseen-customer"}
    assert model.predict({"ticket", "unseen-customer"}, "route") is None


def test_write_facts_are_part_of_context_not_implicitly_copied_from_training():
    a = Action("maybe", context=set(), writes={"flag"})
    model = LayeredWorldModel([a])
    for i in range(2):
        model.observe({"flag", str(i)}, "maybe", {"flag", str(i)}, event=str(i), source="sandbox")
    assert model.predict({"new"}, "maybe") is None


def test_outcome_distribution_and_no_op_are_retained():
    model = LayeredWorldModel([Action("try", {"ready"})])
    for i in range(4):
        model.observe({"ready"}, "try", {"done"} if i < 3 else {"ready"}, event=str(i), source="sensor")
    p = model.predict({"ready"}, "try")
    assert {o.state: o.probability for o in p.outcomes} == {frozenset({"done"}): .75, frozenset({"ready"}): .25}
    assert p.support == 4 and p.confidence_lower < .75


def test_reference_contradiction_quarantines_unseen_states_and_survives_cache():
    a = Action("send", {"ready"})
    model = LayeredWorldModel([a], {"send": Contract(Effect({"done"}))})
    assert model.predict({"ready"}, "send").layer == "L1"
    model.observe({"ready"}, "send", {"failed"}, event="fail", source="sensor")
    p = model.predict({"ready"}, "send")
    assert p.layer == "L2" and p.outcomes[0].state == {"failed"}
    assert model.predict({"ready", "other"}, "send") is None
    assert model.coverage()["quarantined"] == ["send"]


def test_invalid_logs_and_batch_conflicts_do_not_partially_populate():
    model = LayeredWorldModel([Action("send", {"ready"}, context=set(), writes={"done"})])
    good = dict(event="same", before=["ready"], action="send", after=["ready", "done"], source="test")
    wrong = dict(good, after=["ready"])
    with pytest.raises(ValueError, match="conflicting"):
        model.ingest([good, wrong])
    assert model.coverage()["observations"] == 0
    with pytest.raises(ValueError, match="preconditions"):
        model.observe(set(), "send", {"done"}, event="bad", source="test")
    assert model.ingest([good]) == 1
    assert model.ingest([good]) == 0
    # Truth outside the declared footprint is kept; only the footprint loses trust.
    assert model.observe({"ready"}, "send", {"done"}, event="drift", source="test")
    assert model.coverage()["footprint_violations"] == ["send"]
    assert model.coverage()["observations"] == 2


@pytest.mark.parametrize("kwargs", [{"cost": 0}, {"cost": float("nan")}, {"context": set()},
                                     {"preconditions": {"x"}, "forbidden": {"x"}}])
def test_invalid_actions_rejected(kwargs):
    with pytest.raises(ValueError):
        Action("bad", **kwargs)


def test_effect_bearing_action_rejected():
    class Leaky:
        effect = {"hidden"}
    with pytest.raises(TypeError):
        LayeredWorldModel([Leaky()])
