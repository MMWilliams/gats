"""Claim-aligned, oracle-free graph planner for the next GATS experiment.

This module is intentionally separate from both the submitted evaluator and
the first relaxed-UCB prototype.  It uses bounded best-first search over a
transition graph and an h-max delete-relaxed heuristic.  It never calls the
exact BFS helper from ``run_gats_eval``.

The graph is the experimental mechanism: when ``graph_memory`` is true,
predicted transitions and their search metadata survive real decisions.  The
reset control clears both at each decision.  All transition predictions and
heuristic work are counted so that a positive result cannot be obtained by
silently hiding computation in the value function.
"""
from __future__ import annotations

import heapq
import itertools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from gats_rebuilt import Edge, LayeredTransitionModel
from gats_relaxed import RelaxedDistance
from run_gats_eval import Action, State


@dataclass(frozen=True)
class NextResult:
    success: bool
    plan: tuple[str, ...]
    model_queries: int
    model_cache_hits: int
    graph_cache_hits: int
    graph_edges: int
    search_expansions: int
    heuristic_calls: int
    heuristic_cache_hits: int
    heuristic_action_scans: int
    layer_usage: dict[str, int]
    frontier_peak: int
    fallback_steps: int
    exploration_steps: int
    heuristic_visible_actions: int


class RelaxedAdditiveDistance:
    """Delete-relaxed additive cost with explicit work accounting."""

    def __init__(self, actions: list[Action]):
        self.actions = actions
        self.watchers: dict[str, list[int]] = {}
        self.preconditions: list[frozenset] = []
        self.effects: list[frozenset] = []
        for index, action in enumerate(actions):
            self.preconditions.append(action.preconditions)
            self.effects.append(action.effects_add)
            for fact in action.preconditions:
                self.watchers.setdefault(fact, []).append(index)
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
        cost = {fact: 0.0 for fact in state.inventory}
        heap: list[tuple[float, str]] = [(0.0, fact) for fact in state.inventory]
        heapq.heapify(heap)
        remaining = [len(preconditions) for preconditions in self.preconditions]
        sums = [0.0 for _ in self.preconditions]
        fired = [False for _ in self.preconditions]
        for index, count in enumerate(remaining):
            if count == 0:
                fired[index] = True
                value = 1.0
                for fact in self.effects[index]:
                    if value < cost.get(fact, math.inf):
                        cost[fact] = value
                        heapq.heappush(heap, (value, fact))
        while heap:
            value, fact = heapq.heappop(heap)
            if value != cost.get(fact):
                continue
            for index in self.watchers.get(fact, ()):
                self.action_scans += 1
                if fired[index]:
                    continue
                remaining[index] -= 1
                sums[index] += value
                if remaining[index] == 0:
                    fired[index] = True
                    next_cost = 1.0 + sums[index]
                    for effect in self.effects[index]:
                        if next_cost < cost.get(effect, math.inf):
                            cost[effect] = next_cost
                            heapq.heappush(heap, (next_cost, effect))
        distance = sum(cost.get(fact, math.inf) for fact in state.goal)
        self.cache[state.inventory] = float(distance)
        return float(distance)


def backward_relevant_actions(actions: list[Action], goal: frozenset) -> list[Action]:
    """Keep actions in the backward relaxed dependency closure of ``goal``."""
    needed = set(goal)
    relevant: set[str] = set()
    changed = True
    while changed:
        changed = False
        for action in actions:
            if action.name in relevant or not (action.effects_add & needed):
                continue
            relevant.add(action.name)
            before = len(needed)
            needed.update(action.preconditions)
            changed = changed or len(needed) != before
    return sorted((action for action in actions if action.name in relevant),
                  key=lambda action: action.name)


class LiftedEffectPredictor:
    """Infer grounded effects from visible examples of an operator schema.

    The predictor sees only the supplied visible ``Action`` objects. Grounded
    constants occurring in an action name are replaced by positional
    variables, then instantiated for a hidden action with the same operator
    name. No hidden action effects or goal facts are consulted.
    """

    _space = re.compile(r"[()]\s*")

    def __init__(self, visible_actions: list[Action]):
        self.schemas: dict[tuple[str, tuple[int, ...]],
                          tuple[frozenset, frozenset]] = {}
        self.operator_schemas: dict[str, tuple[frozenset, frozenset]] = {}
        samples: defaultdict[tuple[str, tuple[int, ...]], Counter] = defaultdict(Counter)
        for action in visible_actions:
            operator, args = self._action_tokens(action.name)
            pattern = self._equality_pattern(args)
            add = tuple(sorted(self._normalize_fact(fact, args)
                               for fact in action.effects_add))
            delete = tuple(sorted(self._normalize_fact(fact, args)
                                  for fact in action.effects_del))
            samples[(operator, pattern)][(add, delete)] += 1
        operator_samples: defaultdict[str, Counter] = defaultdict(Counter)
        for (operator, pattern), counts in samples.items():
            (add, delete), _ = counts.most_common(1)[0]
            schema = (frozenset(add), frozenset(delete))
            self.schemas[(operator, pattern)] = schema
            operator_samples[operator][schema] += sum(counts.values())
        for operator, counts in operator_samples.items():
            self.operator_schemas[operator] = counts.most_common(1)[0][0]

    @staticmethod
    def _action_tokens(name: str) -> tuple[str, tuple[str, ...]]:
        tokens = name.strip().strip("()").split()
        if not tokens:
            raise ValueError("empty grounded action name")
        return tokens[0], tuple(tokens[1:])

    @staticmethod
    def _equality_pattern(args: tuple[str, ...]) -> tuple[int, ...]:
        positions: dict[str, int] = {}
        pattern: list[int] = []
        for arg in args:
            if arg not in positions:
                positions[arg] = len(positions)
            pattern.append(positions[arg])
        return tuple(pattern)

    @staticmethod
    def _fact_tokens(fact: str) -> tuple[str, ...]:
        return tuple(fact.strip().strip("()").split())

    @classmethod
    def _normalize_fact(cls, fact: str, args: tuple[str, ...]) -> tuple[str, ...]:
        arg_positions = {value: index for index, value in enumerate(args)}
        return tuple("?" + str(arg_positions[token])
                     if token in arg_positions else token
                     for token in cls._fact_tokens(fact))

    @staticmethod
    def _instantiate(fact: tuple[str, ...], args: tuple[str, ...]) -> str:
        values = [args[int(token[1:])] if token.startswith("?") else token
                  for token in fact]
        return "(" + " ".join(values) + ")"

    def __call__(self, state: State, action: Action) -> State:
        operator, args = self._action_tokens(action.name)
        pattern = self._equality_pattern(args)
        schema = self.schemas.get((operator, pattern))
        if schema is None:
            schema = self.operator_schemas.get(operator)
        if schema is None:
            return state
        add, delete = schema
        added = frozenset(self._instantiate(fact, args) for fact in add)
        removed = frozenset(self._instantiate(fact, args) for fact in delete)
        return state.with_inventory((state.inventory | added) - removed)


class BestFirstGraphPlanner:
    """Bounded best-first search with persistent transition-graph memory.

    The search priority is ``path_cost + h_max``.  h-max ignores delete
    effects, so it is a relaxation rather than an exact shortest-path oracle.
    The planner only expands transitions through ``LayeredTransitionModel``;
    real execution uses the supplied action, and observations update the
    model just as they would in an incomplete-model experiment.
    """

    def __init__(
        self,
        model: LayeredTransitionModel,
        actions: list[Action],
        *,
        budget: int = 500,
        max_depth: int = 64,
        max_steps: int = 64,
        graph_memory: bool = True,
        heuristic: str = "hmax",
        total_budget: bool = False,
        allow_fallback: bool = False,
        commit_plan: bool = False,
        heuristic_actions: list[Action] | None = None,
        explore_unknown: bool = False,
        exploration_cap: int = 32,
        heuristic_weight: float = 1.0,
    ) -> None:
        if min(budget, max_depth, max_steps) < 1:
            raise ValueError("budget, max_depth, and max_steps must be positive")
        if len({action.name for action in actions}) != len(actions):
            raise ValueError("action names must be unique")
        self.model = model
        self.actions = sorted(actions, key=lambda action: action.name)
        self.action_by_name = {action.name: action for action in self.actions}
        self.search_actions = self.actions
        self.budget = budget
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.graph_memory = graph_memory
        self.total_budget = total_budget
        self.allow_fallback = allow_fallback
        self.commit_plan = commit_plan
        self.heuristic_source = sorted(
            heuristic_actions if heuristic_actions is not None else self.actions,
            key=lambda action: action.name)
        self.explore_unknown = explore_unknown
        self.exploration_cap = exploration_cap
        self.remaining_budget: int | None = None
        if heuristic not in {"hmax", "hadd"}:
            raise ValueError("heuristic must be hmax or hadd")
        if heuristic_weight <= 0:
            raise ValueError("heuristic weight must be positive")
        self.heuristic_name = heuristic
        self.heuristic_weight = heuristic_weight
        self.heuristic = (RelaxedDistance(self.search_actions) if heuristic == "hmax"
                          else RelaxedAdditiveDistance(self.search_actions))
        self.edges: dict[tuple[frozenset, str], Edge] = {}
        self.expanded_states: set[frozenset] = set()
        self.graph_cache_hits = 0
        self.search_expansions = 0
        self.frontier_peak = 0
        self.fallback_steps = 0
        self.exploration_steps = 0
        self.probed: set[tuple[frozenset, str]] = set()

    def _clear_memory(self) -> None:
        self.edges.clear()
        self.expanded_states.clear()
        # A model prediction cache is part of the retained transition graph
        # for this comparison. Clearing it makes the reset control honest.
        self.model.cache.clear()

    def _edge(self, state: State, action: Action) -> Edge | None:
        key = (state.inventory, action.name)
        if key in self.edges:
            self.graph_cache_hits += 1
            return self.edges[key]
        prediction = self.model.predict(state, action)
        if prediction is None:
            return None
        edge = Edge(prediction.state, prediction.confidence, prediction.layer)
        self.edges[key] = edge
        return edge

    def _probe_action(self, state: State) -> str | None:
        if (not self.explore_unknown or
                self.exploration_steps >= self.exploration_cap):
            return None
        # Exploration is allowed to inspect any applicable unknown action,
        # including one outside the current relaxed relevance closure. Without
        # this escape hatch an initially unknown prerequisite can never be
        # observed, which would make learning impossible by construction.
        for action in self.actions:
            key = (state.inventory, action.name)
            if (action.is_applicable(state) and
                    action.name not in self.model.specifications and
                    key not in self.probed):
                self.probed.add(key)
                return action.name
        return None

    def _priority(self, g: int, h: float, depth: int, serial: int):
        # Keep infinity values deterministic while preserving their meaning.
        return (math.isinf(h), g + self.heuristic_weight * h
                if not math.isinf(h) else math.inf,
                depth, serial)

    def _best_path(self, root: State) -> tuple[str, ...] | None:
        """Return the best bounded path, or a measured one-step fallback."""
        if root.is_goal():
            return None
        counter = itertools.count()
        frontier: list[tuple[tuple, int, State, tuple[str, ...]]] = []
        start_h = self.heuristic(root)
        heapq.heappush(frontier, (self._priority(0, start_h, 0, next(counter)),
                                  0, root, ()))
        best_depth: dict[frozenset, int] = {root.inventory: 0}
        expansions = 0
        best_partial: tuple[tuple, tuple[str, ...]] | None = None

        local_budget = (self.remaining_budget if self.total_budget
                        and self.remaining_budget is not None else self.budget)
        while frontier and expansions < local_budget:
            _, depth, state, path = heapq.heappop(frontier)
            if state.is_goal():
                if self.total_budget and self.remaining_budget is not None:
                    self.remaining_budget -= expansions
                return path if path else None
            if depth >= self.max_depth:
                continue
            if depth > best_depth.get(state.inventory, depth):
                continue
            if path:
                h = self.heuristic(state)
                partial_key = (math.isinf(h), h, state.goal_distance(),
                               depth, path)
                if best_partial is None or partial_key < best_partial[0]:
                    best_partial = (partial_key, path)
            expansions += 1
            self.search_expansions += 1
            self.expanded_states.add(state.inventory)
            for action in self.search_actions:
                if not action.is_applicable(state):
                    continue
                successor_depth = depth + 1
                edge = self._edge(state, action)
                if edge is None:
                    if action.name not in self.model.specifications:
                        # Preserve a path to an unknown transition as an
                        # exploration candidate. The caller will execute only
                        # the first action unless commit mode is explicitly
                        # requested, so no unverified suffix is hidden.
                        probe_path = path + (action.name,)
                        partial_key = (True, math.inf,
                                       successor_depth, successor_depth,
                                       probe_path)
                        if (best_partial is None or
                                partial_key < best_partial[0]):
                            best_partial = (partial_key, probe_path)
                    continue
                successor = edge.next_state
                prior = best_depth.get(successor.inventory)
                if prior is not None and successor_depth >= prior:
                    continue
                best_depth[successor.inventory] = successor_depth
                new_path = path + (action.name,)
                h = self.heuristic(successor)
                partial_key = (math.isinf(h), h, successor.goal_distance(),
                               successor_depth, new_path)
                if best_partial is None or partial_key < best_partial[0]:
                    best_partial = (partial_key, new_path)
                heapq.heappush(
                    frontier,
                    (self._priority(successor_depth, h, successor_depth,
                                    next(counter)), successor_depth,
                     successor, new_path),
                )
            self.frontier_peak = max(self.frontier_peak, len(frontier))
        # Bounded planning should still make a measured best-effort decision.
        # Returning the best partial path avoids converting search exhaustion
        # into an artificial zero-step failure. The fallback is logged and is
        # never presented as a proven complete plan.
        if self.total_budget and self.remaining_budget is not None:
            self.remaining_budget -= expansions
        if self.allow_fallback and best_partial is not None:
            self.fallback_steps += 1
            return (best_partial[1][0],)
        return None

    def plan(self, initial: State) -> NextResult:
        self.search_actions = backward_relevant_actions(self.actions, initial.goal)
        heuristic_actions = backward_relevant_actions(
            self.heuristic_source, initial.goal)
        self.heuristic = (RelaxedDistance(heuristic_actions)
                          if self.heuristic_name == "hmax"
                          else RelaxedAdditiveDistance(heuristic_actions))
        state = initial
        self.remaining_budget = self.budget if self.total_budget else None
        plan: list[str] = []
        seen_real: set[frozenset] = set()
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in seen_real:
                break
            seen_real.add(state.inventory)
            if not self.graph_memory:
                self._clear_memory()
            path = self._best_path(state)
            if not path:
                probe = self._probe_action(state)
                path = (probe,) if probe else None
            if not path:
                break
            names = path if self.commit_plan else path[:1]
            for action_name in names:
                action = self.action_by_name[action_name]
                if not action.is_applicable(state):
                    break
                if action.name not in self.model.specifications:
                    if self.exploration_steps >= self.exploration_cap:
                        break
                    self.exploration_steps += 1
                before = state
                state = action.apply(state)
                self.model.observe(before, action, state)
                plan.append(action.name)
                if state.is_goal():
                    break

        usage = dict(self.model.usage)
        return NextResult(
            success=state.is_goal(),
            plan=tuple(plan),
            model_queries=sum(usage.get(layer, 0) for layer in ("L1", "L2", "L3")),
            model_cache_hits=usage.get("cache", 0),
            graph_cache_hits=self.graph_cache_hits,
            graph_edges=len(self.edges),
            search_expansions=self.search_expansions,
            heuristic_calls=self.heuristic.calls,
            heuristic_cache_hits=self.heuristic.cache_hits,
            heuristic_action_scans=self.heuristic.action_scans,
            layer_usage=usage,
            frontier_peak=self.frontier_peak,
            fallback_steps=self.fallback_steps,
            exploration_steps=self.exploration_steps,
            heuristic_visible_actions=len(heuristic_actions),
        )
