"""Trusted sandbox fixtures. Only this evaluation side can see execution truth."""
from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .domain import Action, Contract, Effect, Rule


@dataclass
class Scenario:
    name: str
    family: str
    initial: frozenset
    goal: frozenset
    actions: tuple
    contracts: dict
    execute: Callable
    expected_success: bool = True
    provenance: str = "synthetic fixture; fully observed deterministic sandbox"


def strips(name, initial, goal, rows, *, expected_success=True):
    """Rows contain evaluator truth; public actions omit all effects."""
    actions, contracts, truth = [], {}, {}
    for row in rows:
        action, pre, add, delete, *costs = row
        pre, add, delete = map(frozenset, (pre, add, delete))
        actions.append(Action(action, pre, cost=costs[0] if costs else 1,
                              context=pre, writes=add | delete))
        contracts[action] = Contract(Effect(add, delete))
        truth[action] = (pre, add, delete)

    def execute(state, name):
        pre, added, removed = truth[name]
        if not pre <= state:
            raise ValueError("sandbox precondition failure")
        # Environment implementation is separate from Contract.apply.
        next_state = set(state)
        next_state.difference_update(removed)
        next_state.update(added)
        return frozenset(next_state)

    return Scenario(name, "research", frozenset(initial), frozenset(goal), tuple(actions),
                    contracts, execute, expected_success)


def research_scenarios():
    result = []
    for length in (4, 12, 32):
        result.append(strips(f"chain_{length}", {"p0"}, {f"p{length}"},
                             [(f"step_{i:02d}", {f"p{i}"}, {f"p{i+1}"}, {f"p{i}"}) for i in range(length)]))
    result.append(strips("parallel_join", {"ticket"}, {"done"}, [
        (f"prepare_{i}", {"ticket"}, {f"ready_{i}"}, set()) for i in range(6)] +
        [("join", {f"ready_{i}" for i in range(6)}, {"done"}, set())]))
    result.append(strips("consume_regenerate", {"resource", "start"}, {"done"}, [
        ("consume", {"resource", "start"}, {"used"}, {"resource", "start"}),
        ("regenerate", {"used"}, {"resource", "stage2"}, {"used"}),
        ("finish", {"resource", "stage2"}, {"done"}, {"resource"})]))
    result.append(strips("irreversible_trap", {"permit"}, {"done"}, [
        ("tempting", {"permit"}, {"shiny"}, {"permit"}),
        ("prepare", {"permit"}, {"ready"}, {"permit"}),
        ("finish", {"ready"}, {"done"}, set())]))
    result.append(strips("wide_40", {"ticket"}, {"done"}, [
        (f"distractor_{i:02d}", {"ticket"}, {f"unused_{i}"}, {"ticket"}) for i in range(40)] +
        [("useful", {"ticket"}, {"ready"}, {"ticket"}), ("finish", {"ready"}, {"done"}, set())]))
    result.append(strips("cost_tradeoff", {"start"}, {"done"}, [
        ("expensive", {"start"}, {"done"}, set(), 20),
        ("cheap", {"start"}, {"ready"}, {"start"}, 1),
        ("finish", {"ready"}, {"done"}, set(), 1)]))
    result.append(strips("cycle_and_noop", {"a"}, {"done"}, [
        ("noop", {"a"}, {"a"}, set()), ("forward", {"a"}, {"b"}, {"a"}),
        ("back", {"b"}, {"a"}, {"b"}), ("finish", {"b"}, {"done"}, set())]))
    result.append(strips("unreachable", {"a"}, {"done"}, [
        ("forward", {"a"}, {"b"}, {"a"}), ("back", {"b"}, {"a"}, {"b"})], expected_success=False))
    result.append(strips("already_done", {"done"}, {"done"}, []))

    actions = (Action("unblock", {"blocked"}, context={"blocked"}, writes={"blocked"}),
               Action("dispatch", {"paid", "stock"}, {"blocked"},
                      context={"paid", "stock", "priority", "blocked"},
                      writes={"stock", "sent", "express"}))
    contracts = {"unblock": Contract(Effect(delete={"blocked"})),
                 "dispatch": Contract(Effect({"sent"}, {"stock"}),
                                      (Rule({"priority"}, Effect({"express"})),))}

    def conditional_truth(state, action):
        state = set(state)
        if action == "unblock":
            if "blocked" not in state:
                raise ValueError("not blocked")
            state.remove("blocked")
        else:
            if not {"paid", "stock"} <= state or "blocked" in state:
                raise ValueError("cannot dispatch")
            state.remove("stock")
            state.add("sent")
            if "priority" in state:
                state.add("express")
        return frozenset(state)

    for priority in (False, True):
        result.append(Scenario("conditional_priority" if priority else "negative_guard", "research",
                               frozenset({"paid", "stock", "blocked"} | ({"priority"} if priority else set())),
                               frozenset({"sent", "express"} if priority else {"sent"}),
                               actions, contracts, conditional_truth))
    return result


def business_scenarios():
    from integrations.companies import ALL_COMPANIES
    from integrations.complex_cases import ALL_COMPLEX
    cases = []
    for case in ALL_COMPANIES + ALL_COMPLEX:
        builder = case["builder"]
        system = builder()
        actions, contracts = [], {}
        for a in system.published_contract():
            actions.append(Action(a.name, a.preconditions, context=a.preconditions,
                                  writes=a.effects_add | a.effects_del))
            contracts[a.name] = Contract(Effect(a.effects_add, a.effects_del))

        def execute(state, action, builder=builder):
            # Each sample resets this in-memory fixture, never a customer API.
            sandbox = builder()
            sandbox._facts = set(state)
            # All required human gates are simulated approvals for this fixture.
            sandbox.call(action)
            return sandbox.observe()

        cases.append(Scenario("business_" + case["name"], "business", system.observe(),
                              frozenset(case["goal"]), tuple(actions), contracts, execute,
                              provenance="existing integrations company fixture; simulated human approvals"))
    return cases


def demonstration(case):
    """Supervised training path from public exact contracts, validated by execution.

    This teacher uses greedy best-first h-add; it is training-data construction,
    not a benchmark competitor or part of MCTS. Unreachable fixtures have no demo.
    """
    if not case.expected_success:
        return []
    actions = sorted(case.actions, key=lambda a: a.name)
    operators = [(a.preconditions, case.contracts[a.name].effect.add, a.cost) for a in actions]
    for a in actions:
        operators.extend((a.preconditions | r.when, r.effect.add, a.cost) for r in case.contracts[a.name].rules)

    def heuristic(state):
        costs = {f: 0.0 for f in state}
        changed = True
        while changed:
            changed = False
            for pre, add, cost in operators:
                if pre <= costs.keys():
                    value = cost + sum(costs[p] for p in sorted(pre))
                    for f in sorted(add):
                        if value < costs.get(f, math.inf):
                            costs[f] = value
                            changed = True
        return sum(costs.get(f, math.inf) for f in sorted(case.goal))

    queue = [(heuristic(case.initial), 0, case.initial, ())]
    ids, seen = itertools.count(1), {case.initial}
    for _ in range(20000):
        if not queue:
            break
        _, _, state, path = heapq.heappop(queue)
        if case.goal <= state:
            return list(path)
        for a in actions:
            if a.applicable(state):
                after = case.contracts[a.name].apply(state)
                if after not in seen:
                    seen.add(after)
                    heapq.heappush(queue, (heuristic(after), next(ids), after, (*path, a.name)))
    raise RuntimeError(f"no demonstration within teacher cap: {case.name}")


def training_traces(case, repeats=3, exploration_cap=128):
    witness = demonstration(case)
    states, state = [case.initial], case.initial
    for name in witness:
        state = case.execute(state, name)
        states.append(state)
    if case.expected_success and not case.goal <= state:
        raise AssertionError("teacher path fails independent sandbox execution")
    # Populate branches that a successful demonstration never needs, including
    # recovery transitions in impossible-goal fixtures. Execution, not supplied
    # effects, determines these successors. The finite cap is part of the corpus.
    queue, seen = deque([case.initial]), {case.initial}
    while queue:
        before = queue.popleft()
        states.append(before)
        for action in sorted(case.actions, key=lambda a: a.name):
            if action.applicable(before):
                after = case.execute(before, action.name)
                if after not in seen and len(seen) < exploration_cap:
                    seen.add(after)
                    queue.append(after)
    # One actual sample per complete context per reset is enough. Repeated
    # identical contexts along a demonstration must not inflate support.
    contexts = {}
    for state in states:
        for action in sorted(case.actions, key=lambda a: a.name):
            if action.applicable(state):
                contexts.setdefault((action.name, action.evidence_key(state)), state)
    rows = []
    for repeat in range(repeats):
        for step, ((name, _), state) in enumerate(contexts.items()):
            before = state | {f"episode:training:{repeat}"}
            after = case.execute(before, name)
            rows.append({"event": f"train/{case.name}/{repeat}/{step}/{name}",
                         "before": sorted(before), "action": name, "after": sorted(after),
                         "source": "sandbox-reset-execution:" + case.name})
    return rows
