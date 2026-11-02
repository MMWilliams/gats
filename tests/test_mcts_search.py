import ast
from pathlib import Path

import pytest

from gats_mcts import Action, Contract, Effect, LayeredWorldModel, MCTSPlanner


def model_for(rows):
    return LayeredWorldModel([Action(n, p, cost=c) for n, p, _, _, c in rows],
                            {n: Contract(Effect(a, d)) for n, _, a, d, _ in rows})


def test_delayed_goal_beats_irreversible_trap_and_has_deep_backups():
    model = model_for([
        ("trap", {"start"}, {"shiny"}, {"start"}, 1),
        ("prepare", {"start"}, {"ready"}, {"start"}, 1),
        ("work", {"ready"}, {"almost"}, {"ready"}, 1),
        ("finish", {"almost"}, {"goal"}, set(), 1),
    ])
    result = MCTSPlanner(model, simulations=128, seed=3).search({"start"}, {"goal"})
    assert result.action == "prepare"
    assert result.max_tree_depth >= 3 and result.expansions > 0
    assert result.root["prepare"]["mean"] > result.root["trap"]["mean"]
    assert result.simulations == 128
    assert sum(r["visits"] for r in result.root.values()) == 128
    assert model.coverage()["observations"] == 0


def test_nonuniform_cost_prefers_two_cheap_steps_over_expensive_goal():
    model = model_for([
        ("costly", {"start"}, {"goal"}, set(), 20),
        ("cheap", {"start"}, {"mid"}, {"start"}, 1),
        ("finish", {"mid"}, {"goal"}, set(), 1),
    ])
    assert MCTSPlanner(model, simulations=256).search({"start"}, {"goal"}).action == "cheap"


def test_cycle_noop_terminal_and_budget_bounds():
    model = model_for([
        ("noop", {"a"}, {"a"}, set(), 1),
        ("forward", {"a"}, {"b"}, {"a"}, 1),
        ("back", {"b"}, {"a"}, {"b"}, 1),
    ])
    result = MCTSPlanner(model, simulations=7, max_depth=3).search({"a"}, {"missing"})
    assert result.simulations == 7 and result.max_tree_depth <= 3
    assert "noop" not in result.root
    assert MCTSPlanner(model).search({"a"}, {"a"}).status == "goal"
    assert MCTSPlanner(model).search(set(), {"a"}).action is None
    limited = MCTSPlanner(model, max_model_queries=1).search({"a"}, {"missing"})
    assert limited.status == "query_limit" and limited.model_requests == 1


def test_same_seed_warm_graph_and_changed_goal_do_not_share_values():
    model = model_for([("left", {"s"}, {"l"}, {"s"}, 1), ("right", {"s"}, {"r"}, {"s"}, 1)])
    planner = MCTSPlanner(model, seed=91)
    first = planner.search({"s"}, {"l"})
    assert planner.search({"s"}, {"l"}) == first
    assert first.action == "left"
    assert planner.search({"s"}, {"r"}).action == "right"


def test_empirical_stochastic_outcomes_are_sampled_in_search():
    actions = [Action("risky", {"start"}), Action("safe", {"start"})]
    model = LayeredWorldModel(actions)
    for i in range(20):
        model.observe({"start"}, "risky", {"goal"} if i == 0 else {"dead"}, event=f"r{i}", source="sandbox")
        model.observe({"start"}, "safe", {"goal"}, event=f"s{i}", source="sandbox")
    result = MCTSPlanner(model, simulations=500).search({"start"}, {"goal"})
    assert result.action == "safe"
    assert result.root["safe"]["mean"] > result.root["risky"]["mean"]


def test_planner_has_no_environment_import_or_execution_callback():
    path = Path(__file__).parents[1] / "gats_mcts" / "search.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module in {"__future__", "dataclasses", "domain", "graph", "heuristic"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"execute", "apply", "call", "step"}


@pytest.mark.parametrize("kwargs", [{"simulations": 0}, {"max_depth": 0}, {"max_model_queries": 0},
                                     {"discount": 1}, {"exploration": float("nan")}])
def test_invalid_search_budget(kwargs):
    with pytest.raises(ValueError):
        MCTSPlanner(LayeredWorldModel([]), **kwargs)


def test_additive_guidance_known_costs_and_model_update_invalidation():
    from gats_mcts.heuristic import AdditiveHeuristic
    operators = [(frozenset({"s"}), frozenset({"a"}), 2),
                 (frozenset({"s"}), frozenset({"b"}), 3),
                 (frozenset({"a", "b"}), frozenset({"goal"}), 5)]
    heuristic = AdditiveHeuristic(operators, frozenset({"goal"}))
    assert heuristic(frozenset({"s"})) == 10
    assert heuristic(frozenset({"s", "a"})) == 8
    assert heuristic(frozenset({"goal"})) == 0
    assert heuristic(frozenset()) == float("inf")
    model = LayeredWorldModel([Action("a", {"s"})])
    planner = MCTSPlanner(model)
    assert planner.search({"s"}, {"goal"}).action is None
    model.observe({"s"}, "a", {"goal"}, event="learn", source="test")
    assert planner.search({"s"}, {"goal"}).action == "a"
