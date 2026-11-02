"""Worklist h-add guidance; optimistic, not a reachability or optimality proof.

Uses the fact-cost propagation technique already present in the corrected v2.1
planner, without importing or changing any frozen research source.
"""
from collections import defaultdict
import heapq
import math


class AdditiveHeuristic:
    def __init__(self, operators, goal, *, max_cache_entries=50000):
        self.goal = goal
        self.operators = list(operators)
        self.watchers = defaultdict(list)
        for index, (pre, _, _) in enumerate(self.operators):
            for fact in sorted(pre):
                self.watchers[fact].append(index)
        self.cache = {}
        self.max_cache_entries = max_cache_entries

    def __call__(self, state):
        if state in self.cache:
            return self.cache[state]
        costs = {f: 0.0 for f in state}
        queue = [(0.0, f) for f in state]
        heapq.heapify(queue)
        remaining = [len(pre) for pre, _, _ in self.operators]
        totals = [cost for _, _, cost in self.operators]
        fired = [False] * len(self.operators)

        def fire(index):
            fired[index] = True
            for fact in sorted(self.operators[index][1]):
                if totals[index] < costs.get(fact, math.inf):
                    costs[fact] = totals[index]
                    heapq.heappush(queue, (totals[index], fact))

        for index, count in enumerate(remaining):
            if not count:
                fire(index)
        while queue:
            cost, fact = heapq.heappop(queue)
            if cost != costs[fact]:
                continue
            for index in self.watchers.get(fact, ()):
                if fired[index]:
                    continue
                remaining[index] -= 1
                totals[index] += cost
                if not remaining[index]:
                    fire(index)
        value = sum(costs.get(f, math.inf) for f in sorted(self.goal))
        if len(self.cache) >= self.max_cache_entries:
            self.cache.clear()
        self.cache[state] = value
        return value
