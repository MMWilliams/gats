"""Bounded MCTS with PUCT selection, one expansion, rollout, and path backup.

Only the layered model is accessible here. This module never executes actions
and never imports the benchmark environment. Transition predictions persist;
goal-dependent tree statistics are local to each decision.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from .domain import facts
from .graph import canonical
from .heuristic import AdditiveHeuristic


@dataclass
class Branch:
    prediction: object
    prior: float
    visits: int = 0
    value: float = 0.0
    children: dict = field(default_factory=dict)

    @property
    def mean(self):
        return self.value / self.visits if self.visits else 0.0


@dataclass
class Node:
    state: frozenset
    branches: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    action: str | None
    simulations: int
    expansions: int
    model_requests: int
    max_tree_depth: int
    status: str
    root: dict


class QueryLimit(Exception):
    pass


class MCTSPlanner:
    def __init__(self, model, *, simulations=128, max_depth=64,
                 exploration=1.5, discount=0.97, rollout_epsilon=0.1,
                 max_model_queries=50000, seed=0):
        if any(type(v) is not int or v < 1 for v in (simulations, max_depth, max_model_queries)):
            raise ValueError("search budgets must be positive integers")
        if not math.isfinite(exploration) or exploration <= 0:
            raise ValueError("exploration must be finite and positive")
        if not 0 < discount < 1 or not 0 <= rollout_epsilon <= 1:
            raise ValueError("invalid discount or rollout epsilon")
        self.model, self.simulations, self.max_depth = model, simulations, max_depth
        self.exploration, self.discount = exploration, discount
        self.rollout_epsilon, self.max_model_queries, self.seed = rollout_epsilon, max_model_queries, seed
        self._heuristic_key = None
        self._heuristic = None

    def search(self, initial, goal):
        try:
            with self.model.graph.prediction_batch():
                return self._search(initial, goal)
        except BaseException:
            self.model.cache.clear()
            raise

    def _search(self, initial, goal):
        initial, goal = facts(initial), facts(goal)
        if goal <= initial:
            return SearchResult(None, 0, 0, 0, 0, "goal", {})
        rng = random.Random(hashlib.sha256(canonical([self.seed, sorted(initial), sorted(goal)]).encode()).digest())
        # Reuse pure heuristic values across modeled steps while the model and
        # goal stay unchanged. Observed outcomes invalidate this entire cache.
        heuristic_key = (goal, tuple(sorted(self.model.revisions.items())))
        if heuristic_key != self._heuristic_key:
            self._heuristic = AdditiveHeuristic(self.model.relaxed_operators(), goal)
            self._heuristic_key = heuristic_key
        distance = self._heuristic
        candidate_cache = {}
        requests = expansions = completed = deepest = 0
        status = "budget"

        def candidates(state):
            nonlocal requests
            if state in candidate_cache:
                return candidate_cache[state]
            choices = []
            for a in self.model.applicable(state):
                if requests >= self.max_model_queries:
                    raise QueryLimit
                requests += 1
                prediction = self.model.predict(state, a.name)
                if prediction is None or all(o.state == state for o in prediction.outcomes):
                    continue
                score = sum(o.probability * (1.0 / (1.0 + a.cost + distance(o.state)))
                            for o in prediction.outcomes)
                choices.append((a.name, prediction, score))
            total = sum(c[2] for c in choices)
            result = {name: (p, (score / total if total else 1/len(choices)))
                      for name, p, score in choices}
            candidate_cache[state] = result
            return result

        def sample(prediction):
            value = rng.random()
            for outcome in prediction.outcomes:
                value -= outcome.probability
                if value <= 0:
                    return outcome.state
            return prediction.outcomes[-1].state

        def pick(scores):
            best = max(scores.values())
            return rng.choice([k for k, v in scores.items() if v == best])

        def rollout(state, remaining, seen):
            multiplier = 1.0
            for _ in range(remaining):
                if goal <= state:
                    return multiplier
                choices = candidates(state)
                if not choices:
                    return 0.0
                name = (rng.choice(list(choices)) if rng.random() < self.rollout_epsilon else
                        pick({name: p for name, (_, p) in choices.items()}))
                state = sample(choices[name][0])
                multiplier *= self.discount ** self.model.by_name[name].cost
                if state in seen:
                    return 0.0
                seen.add(state)
            return multiplier * (1.0 if goal <= state else 0.25 / (1.0 + distance(state)))

        root = Node(initial)
        for _ in range(self.simulations):
            node, path, seen = root, [], {initial}
            reward = 0.0
            try:
                for depth in range(self.max_depth):
                    if goal <= node.state:
                        reward = 1.0
                        break
                    if not node.branches:
                        node.branches = {name: Branch(p, prior) for name, (p, prior) in candidates(node.state).items()}
                    if not node.branches:
                        break
                    unvisited = {name: b.prior for name, b in node.branches.items() if not b.visits}
                    if unvisited:
                        name = pick(unvisited)
                    else:
                        total = sum(b.visits for b in node.branches.values())
                        name = pick({name: b.mean + self.exploration * b.prior * math.sqrt(total) / (1 + b.visits)
                                     for name, b in node.branches.items()})
                    branch = node.branches[name]
                    path.append((branch, self.model.by_name[name].cost))
                    state = sample(branch.prediction)
                    deepest = max(deepest, depth + 1)
                    if goal <= state:
                        reward = 1.0
                        break
                    if state in seen:
                        break
                    seen.add(state)
                    if state not in branch.children:
                        branch.children[state] = Node(state)
                        expansions += 1
                        reward = rollout(state, self.max_depth - depth - 1, seen)
                        break
                    node = branch.children[state]
                else:
                    reward = 0.25 / (1 + distance(node.state))
            except QueryLimit:
                status = "query_limit"
                break  # Incomplete simulations do not fabricate value/visit evidence.
            for branch, cost in reversed(path):
                reward *= self.discount ** cost
                branch.visits += 1
                branch.value += reward
            completed += 1
        stats = {name: {"visits": b.visits, "mean": b.mean, "prior": b.prior}
                 for name, b in root.branches.items()}
        visited = {name: b for name, b in root.branches.items() if b.visits}
        action = (max(visited, key=lambda name: (visited[name].visits, visited[name].mean,
                                                visited[name].prior)) if visited else None)
        if action is None and status != "query_limit":
            status = "no_modeled_action"
        return SearchResult(action, completed, expansions, requests, deepest, status, stats)
