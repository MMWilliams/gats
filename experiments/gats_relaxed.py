"""Development prototype: persistent graph UCB with a counted h-max value.

The value is delete-relaxed planning distance, not a hidden exact BFS. This
prototype is separate from the submitted GATS evaluator and its results.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from gats_rebuilt import Edge, LayeredTransitionModel
from run_gats_eval import Action, State


class RelaxedDistance:
    """h-max distance with explicit action-scan accounting and state cache."""

    def __init__(self, actions: list[Action]):
        self.actions = actions
        self.cache: dict[frozenset, float] = {}
        self.calls = 0
        self.cache_hits = 0
        self.action_scans = 0

    def __call__(self, state: State) -> float:
        self.calls += 1
        if state.inventory in self.cache:
            self.cache_hits += 1
            return self.cache[state.inventory]
        if state.is_goal():
            self.cache[state.inventory] = 0.0
            return 0.0
        cost = {fact: 0 for fact in state.inventory}
        # Each successful pass lowers at least one integer fact cost. In
        # positive-cost delete relaxation the fixed point is finite.
        while True:
            changed = False
            for action in self.actions:
                self.action_scans += 1
                if not action.preconditions.issubset(cost):
                    continue
                next_cost = 1 + max((cost[p] for p in action.preconditions),
                                    default=0)
                for fact in action.effects_add:
                    if next_cost < cost.get(fact, math.inf):
                        cost[fact] = next_cost
                        changed = True
            if not changed:
                break
        distance = max((cost.get(fact, math.inf) for fact in state.goal),
                       default=0)
        self.cache[state.inventory] = float(distance)
        return float(distance)


@dataclass(frozen=True)
class Result:
    success: bool
    plan: tuple[str, ...]
    model_queries: int
    model_cache_hits: int
    graph_cache_hits: int
    graph_edges: int
    heuristic_calls: int
    heuristic_cache_hits: int
    heuristic_action_scans: int
    layer_usage: dict[str, int]


class RelaxedGraphPlanner:
    """One-new-edge-per-simulation UCB graph search with h-max leaf value."""

    def __init__(self, model: LayeredTransitionModel, actions: list[Action], *,
                 budget: int = 20, max_depth: int = 8, max_steps: int = 64,
                 graph_memory: bool = True, seed: int = 0):
        if min(budget, max_depth, max_steps) < 1:
            raise ValueError("positive limits required")
        self.model = model
        self.actions = sorted(actions, key=lambda action: action.name)
        self.budget = budget
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.graph_memory = graph_memory
        self.rng = random.Random(seed)
        self.heuristic = RelaxedDistance(self.actions)
        self.edges: dict[tuple[frozenset, str], Edge] = {}
        self.graph_cache_hits = 0

    def _select(self, state: State) -> Action | None:
        applicable = [action for action in self.actions
                      if action.is_applicable(state)]
        if not applicable:
            return None
        unvisited = [action for action in applicable
                     if (state.inventory, action.name) not in self.edges or
                     self.edges[(state.inventory, action.name)].visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)
        total = sum(self.edges[(state.inventory, a.name)].visits
                    for a in applicable)
        def score(action):
            edge = self.edges[(state.inventory, action.name)]
            return edge.mean_value + math.sqrt(2 * math.log(max(2, total)) /
                                               edge.visits)
        best = max(score(action) for action in applicable)
        return self.rng.choice([action for action in applicable
                                if score(action) == best])

    def _simulate(self, root: State) -> None:
        state = root
        path: list[Edge] = []
        seen = set()
        for _ in range(self.max_depth):
            if state.is_goal() or state.inventory in seen:
                break
            seen.add(state.inventory)
            action = self._select(state)
            if action is None:
                break
            key = (state.inventory, action.name)
            new_edge = key not in self.edges
            if new_edge:
                prediction = self.model.predict(state, action)
                if prediction is None:
                    break
                self.edges[key] = Edge(prediction.state, prediction.confidence,
                                       prediction.layer)
            else:
                self.graph_cache_hits += 1
            edge = self.edges[key]
            path.append(edge)
            state = edge.next_state
            if new_edge:
                break
        distance = self.heuristic(state)
        value = 0.0 if math.isinf(distance) else 1.0 / (1.0 + distance)
        for edge in path:
            edge.visits += 1
            edge.value_sum += value * edge.confidence

    def plan(self, initial: State) -> Result:
        state = initial
        plan: list[str] = []
        seen_real = set()
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in seen_real:
                break
            seen_real.add(state.inventory)
            if not self.graph_memory:
                self.edges.clear()
            for _ in range(self.budget):
                self._simulate(state)
            candidates = [(action, self.edges[(state.inventory, action.name)])
                          for action in self.actions if action.is_applicable(state)
                          and (state.inventory, action.name) in self.edges]
            if not candidates:
                break
            high = max(edge.mean_value for _, edge in candidates)
            action = self.rng.choice([action for action, edge in candidates
                                      if edge.mean_value == high])
            before = state
            state = action.apply(state)
            self.model.observe(before, action, state)
            plan.append(action.name)
        return Result(state.is_goal(), tuple(plan),
                      sum(self.model.usage[layer] for layer in ("L1", "L2", "L3")),
                      self.model.usage["cache"], self.graph_cache_hits,
                      len(self.edges), self.heuristic.calls,
                      self.heuristic.cache_hits, self.heuristic.action_scans,
                      dict(self.model.usage))


class RelaxedGreedyPlanner:
    """All-successor one-step h-max control with the same real-step cap."""

    def __init__(self, actions: list[Action], *, max_steps: int = 64):
        self.actions = sorted(actions, key=lambda action: action.name)
        self.max_steps = max_steps
        self.heuristic = RelaxedDistance(self.actions)

    def plan(self, initial: State) -> Result:
        state = initial
        plan: list[str] = []
        seen = set()
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in seen:
                break
            seen.add(state.inventory)
            candidates = [(self.heuristic(action.apply(state)), action)
                          for action in self.actions if action.is_applicable(state)
                          and action.apply(state).inventory not in seen]
            if not candidates:
                break
            _, action = min(candidates, key=lambda item: (item[0], item[1].name))
            state = action.apply(state)
            plan.append(action.name)
        return Result(state.is_goal(), tuple(plan), 0, 0, 0, 0,
                      self.heuristic.calls, self.heuristic.cache_hits,
                      self.heuristic.action_scans, {})
