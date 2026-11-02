"""Research implementation of graph search over a layered transition model.

This is deliberately separate from run_gats_eval.py. Its results must not be
combined with the published tables until a new evaluation is completed.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from run_gats_eval import Action, State


@dataclass(frozen=True)
class Prediction:
    state: State
    confidence: float
    layer: str


class LayeredTransitionModel:
    """Exact specifications, observed effects, then a supplied predictor.

    No network/model call is hidden here. An L3 experiment must explicitly
    supply a predictor and measure its own token and runtime costs.
    """

    def __init__(
        self,
        specifications: Iterable[Action] = (),
        *,
        min_observations: int = 3,
        predictor: Callable[[State, Action], State] | None = None,
    ) -> None:
        self.specifications = {action.name: action for action in specifications}
        self.min_observations = min_observations
        self.predictor = predictor
        self.observations: dict[str, Counter[tuple[frozenset, frozenset]]] = {}
        self.cache: dict[tuple[frozenset, str], Prediction] = {}
        self.usage = Counter()

    def observe(self, before: State, action: Action, after: State) -> None:
        added = after.inventory - before.inventory
        removed = before.inventory - after.inventory
        counts = self.observations.setdefault(action.name, Counter())
        counts[(added, removed)] += 1
        for key in [key for key in self.cache if key[1] == action.name]:
            del self.cache[key]

    def predict(self, state: State, action: Action) -> Prediction | None:
        key = (state.inventory, action.name)
        if key in self.cache:
            self.usage["cache"] += 1
            return self.cache[key]

        specification = self.specifications.get(action.name)
        if specification is not None:
            if not specification.is_applicable(state):
                return None
            result = Prediction(specification.apply(state), 1.0, "L1")
        else:
            counts = self.observations.get(action.name, Counter())
            n = sum(counts.values())
            if n >= self.min_observations:
                (added, removed), mode_n = counts.most_common(1)[0]
                next_state = state.with_inventory((state.inventory | added) - removed)
                result = Prediction(next_state, mode_n / n, "L2")
            elif self.predictor is not None:
                result = Prediction(self.predictor(state, action), 0.5, "L3")
            else:
                self.usage["unmodeled"] += 1
                return None

        self.cache[key] = result
        self.usage[result.layer] += 1
        return result


@dataclass
class Edge:
    next_state: State
    confidence: float
    layer: str
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class PlanningResult:
    success: bool
    plan: tuple[str, ...]
    model_queries: int
    graph_edges: int
    cache_hits: int


class GraphSearchPlanner:
    """Persistent graph MCTS with a fixed simulation budget per real step.

    Reward is goal reachability along sampled paths. This planner has no BFS
    oracle; its weakness on wide or long tasks is part of the experiment.
    """

    def __init__(self, model: LayeredTransitionModel, *, budget: int = 20,
                 max_depth: int = 20, max_steps: int = 30,
                 exploration: float = 1.0, seed: int = 0) -> None:
        if budget < 1 or max_depth < 1 or max_steps < 1:
            raise ValueError("budget and limits must be positive")
        self.model = model
        self.budget = budget
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.edges: dict[tuple[frozenset, str], Edge] = {}
        self.action_map: dict[str, Action] = {}
        self.cache_hits = 0

    def _edge(self, state: State, action: Action) -> Edge | None:
        key = (state.inventory, action.name)
        if key in self.edges:
            self.cache_hits += 1
            return self.edges[key]
        prediction = self.model.predict(state, action)
        if prediction is None:
            return None
        edge = Edge(prediction.state, prediction.confidence, prediction.layer)
        self.edges[key] = edge
        return edge

    def _select(self, state: State, actions: list[Action]) -> Action | None:
        applicable = [a for a in actions if a.is_applicable(state)]
        if not applicable:
            return None
        applicable.sort(key=lambda a: a.name)
        unvisited = [a for a in applicable if (state.inventory, a.name) not in self.edges
                     or self.edges[(state.inventory, a.name)].visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)
        total = sum(self.edges[(state.inventory, a.name)].visits for a in applicable)
        scores = {a.name: (
            self.edges[(state.inventory, a.name)].mean_value
            + self.exploration * math.sqrt(2 * math.log(max(2, total)) /
                                           self.edges[(state.inventory, a.name)].visits)
        ) for a in applicable}
        high = max(scores.values())
        return self.rng.choice([a for a in applicable if scores[a.name] == high])

    def _simulate(self, root: State, actions: list[Action]) -> None:
        state = root
        path: list[Edge] = []
        seen = set()
        for _ in range(self.max_depth):
            if state.is_goal() or state.inventory in seen:
                break
            seen.add(state.inventory)
            action = self._select(state, actions)
            if action is None:
                break
            edge = self._edge(state, action)
            if edge is None:
                break
            path.append(edge)
            state = edge.next_state
        reward = 1.0 if state.is_goal() else 0.0
        for edge in path:
            edge.visits += 1
            edge.value_sum += reward * edge.confidence

    def plan(self, initial: State, actions: list[Action]) -> PlanningResult:
        if len({a.name for a in actions}) != len(actions):
            raise ValueError("action names must be unique")
        self.action_map = {a.name: a for a in actions}
        state = initial
        plan: list[str] = []
        seen_real = set()
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in seen_real:
                break
            seen_real.add(state.inventory)
            for _ in range(self.budget):
                self._simulate(state, actions)
            candidates = [(a, self.edges[(state.inventory, a.name)])
                          for a in actions if a.is_applicable(state)
                          and (state.inventory, a.name) in self.edges]
            if not candidates:
                break
            # Tie choices are seeded; input list order does not influence them.
            candidates.sort(key=lambda item: item[0].name)
            score = lambda item: (item[1].mean_value, item[1].visits)
            best_score = max(map(score, candidates))
            action, _ = self.rng.choice([item for item in candidates
                                         if score(item) == best_score])
            before = state
            state = action.apply(state)  # real execution is separate from prediction
            self.model.observe(before, action, state)
            if action.name not in self.model.specifications:
                # An updated empirical model can change predicted successors.
                # Clear search statistics rather than retain stale backups.
                self.edges.clear()
            plan.append(action.name)
        return PlanningResult(state.is_goal(), tuple(plan),
                              sum(self.model.usage[layer] for layer in ("L1", "L2", "L3")),
                              len(self.edges), self.cache_hits)
