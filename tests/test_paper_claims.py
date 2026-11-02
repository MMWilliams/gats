"""Regression tests encoding the paper's verifiable claims.

Added by the continuous-audit loop (2026-07-19). Each test guards a claim the
paper makes (determinism, layered-WM usage, ablation meaningfulness, edge
behavior) so future refactors can't silently regress them. All tests run in
seconds — no full benchmark required.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from run_gats_eval import (
    Action,
    GATSPlanner,
    LayeredWorldModel,
    State,
    bfs_to_goal,
    load_planning_tasks,
)
from run_stress_test import generate_all_stress_tasks


def _mk(name, pre, add, dele=()):
    return Action(name, {}, frozenset(pre), frozenset(add), frozenset(dele))


def _task_fingerprint(tasks):
    return [(t.task_id, tuple(sorted(a.name for a in t.actions))) for t in tasks]


# --- Determinism (README: "fully deterministic", abstract: "zero variance") ---

def test_task_generation_deterministic():
    a = load_planning_tasks("data/api_bank", n_tasks=30, task_seed=7)
    b = load_planning_tasks("data/api_bank", n_tasks=30, task_seed=7)
    assert _task_fingerprint(a) == _task_fingerprint(b)


def test_task_seed_actually_varies_tasks():
    a = load_planning_tasks("data/api_bank", n_tasks=30, task_seed=7)
    b = load_planning_tasks("data/api_bank", n_tasks=30, task_seed=8)
    assert _task_fingerprint(a) != _task_fingerprint(b)


def test_stress_generators_deterministic():
    a = generate_all_stress_tasks(n_per_category=3)
    b = generate_all_stress_tasks(n_per_category=3)
    assert _task_fingerprint(a) == _task_fingerprint(b)


def test_gats_plans_identical_across_runs():
    tasks = load_planning_tasks("data/api_bank", n_tasks=10, task_seed=7)
    plans = []
    for _ in range(2):
        random.seed(42)
        run = []
        for t in tasks:
            planner = GATSPlanner(LayeredWorldModel(t.actions), budget=10)
            initial = State(frozenset(t.goal_facts), frozenset(t.initial_facts))
            r = planner.plan(initial, t.actions)
            run.append((r.success, tuple(a.name for a in r.plan)))
        plans.append(run)
    assert plans[0] == plans[1]


# --- Layered world model is actually in the planning loop (Tables 3/4) ---

def test_l1_used_during_planning():
    acts = [_mk("A", ["start"], ["m"]), _mk("B", ["m"], ["goal"]),
            _mk("Trap", ["start"], ["dead"])]
    wm = LayeredWorldModel(acts)
    p = GATSPlanner(wm, budget=10)
    r = p.plan(State(frozenset(["goal"]), frozenset(["start"])), acts)
    assert r.success
    assert wm.layer_usage[1] > 0, "L1 must be queried during planning"


def test_no_l1_ablation_is_distinct():
    acts = [_mk("A", ["start"], ["m"]), _mk("B", ["m"], ["goal"])]
    wm = LayeredWorldModel(acts, use_l1=False)
    for a in acts:
        for _ in range(wm.L2_SATURATION):
            wm.observe(a.name, a.effects_add, a.effects_del)
    p = GATSPlanner(wm, budget=10)
    r = p.plan(State(frozenset(["goal"]), frozenset(["start"])), acts)
    assert r.success
    assert wm.layer_usage[1] == 0
    assert wm.layer_usage[2] > 0, "no_l1 must route predictions through L2"


def test_l2_confidence_matches_eq6_and_tau2_gate():
    wm = LayeredWorldModel([])
    st = State(frozenset(), frozenset())
    for i in range(1, 15):
        wm.observe("X", frozenset(["y"]), frozenset())
        _, conf = wm.predict(st, "X")
        eq6 = min(i / wm.L2_SATURATION, 1.0)
        if eq6 >= wm.L2_TAU:
            assert wm.last_layer == 2, f"count={i}: expected L2 above tau_2"
            assert conf == eq6, f"count={i}: Eq. 6 confidence"
        else:
            assert wm.last_layer == 3, f"count={i}: below tau_2 must fall through to L3 (Eq. 5)"
            assert conf == 0.5, "L3 confidence per Eq. 3"


# --- Edge behavior (documented boundaries) ---

def test_goal_already_satisfied():
    acts = [_mk("A", ["start"], ["x"])]
    st = State(frozenset(["start"]), frozenset(["start"]))
    r = GATSPlanner(LayeredWorldModel(acts), budget=10).plan(st, acts)
    assert r.success and len(r.plan) == 0


def test_no_applicable_actions_fails_cleanly():
    acts = [_mk("A", ["missing"], ["goal"])]
    st = State(frozenset(["goal"]), frozenset(["start"]))
    r = GATSPlanner(LayeredWorldModel(acts), budget=10).plan(st, acts)
    assert not r.success and r.plan == []


def test_bfs_horizon_boundary():
    def chain(n):
        return [_mk(f"S{i}", [f"p{i}"], [f"p{i+1}"]) for i in range(n)]
    acts19 = chain(19)
    st19 = State(frozenset(["p19"]), frozenset(["p0"]))
    assert bfs_to_goal(st19, acts19)[0], "19-step chain inside d=20 horizon"
    acts25 = chain(25)
    st25 = State(frozenset(["p25"]), frozenset(["p0"]))
    assert not bfs_to_goal(st25, acts25)[0], "25-step chain beyond horizon (paper Section 6 cap)"


def test_delete_effect_consume_regen():
    acts = [
        _mk("Use", ["res", "start"], ["used"], ["res"]),
        _mk("Regen", ["used"], ["res", "stage2"]),
        _mk("Finish", ["res", "stage2"], ["goal"]),
    ]
    st = State(frozenset(["goal"]), frozenset(["start", "res"]))
    r = GATSPlanner(LayeredWorldModel(acts), budget=20).plan(st, acts)
    assert r.success
    assert [a.name for a in r.plan] == ["Use", "Regen", "Finish"]


def test_dead_ends_do_not_leak_across_episodes():
    acts = [_mk(f"T{i}", ["start"], [f"t{i}"]) for i in range(9)]
    acts.append(_mk("Goal", [f"t{i}" for i in range(9)], ["goal"]))
    st = State(frozenset(["goal"]), frozenset(["start"]))
    p = GATSPlanner(LayeredWorldModel(acts), budget=20, max_steps=40)
    assert p.plan(st, acts).success == p.plan(st, acts).success
