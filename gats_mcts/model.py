"""L1 contracts and L2 observed outcome distributions, without an execution oracle."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from types import MappingProxyType

from .domain import Action, Contract, Outcome, Prediction, catalog_data, facts
from .graph import TransitionGraph


def wilson_lower(successes, total):
    """95% Wilson lower bound for the observed modal outcome, not a guarantee."""
    z = 1.959963984540054
    p = successes / total
    return (p + z*z/(2*total) - z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total)


class LayeredWorldModel:
    def __init__(self, actions, contracts=None, *, graph=None, namespace="default",
                 min_support=2):
        actions = tuple(actions)
        if type(min_support) is not int or min_support < 1:
            raise ValueError("min_support must be a positive integer")
        if any(type(a) is not Action for a in actions):
            raise TypeError("planner accepts only effect-free public Action descriptors")
        self.actions = tuple(sorted(actions, key=lambda a: a.name))
        self.by_name = MappingProxyType({a.name: a for a in self.actions})
        if len(self.by_name) != len(self.actions):
            raise ValueError("duplicate action names")
        contracts = dict(contracts or {})
        if not contracts.keys() <= self.by_name.keys() or any(type(c) is not Contract for c in contracts.values()):
            raise ValueError("invalid symbolic contracts")
        for name, c in contracts.items():
            a = self.by_name[name]
            if a.writes is not None:
                effects = [c.effect] + [r.effect for r in c.rules]
                if any(not (e.add | e.delete) <= a.writes for e in effects):
                    raise ValueError("contract effects exceed declared write footprint")
                reads = a.context | a.preconditions | a.forbidden | a.writes
                if any(not (r.when | r.unless) <= reads for r in c.rules):
                    raise ValueError("contract guards exceed declared context")
        self.contracts = MappingProxyType(contracts)
        self.min_support = min_support
        self.graph = graph if graph is not None else TransitionGraph()
        self.namespace = namespace
        catalog = catalog_data(self.actions, contracts)
        catalog["model"] = {"version": 1, "min_support": min_support}
        self.graph.register(namespace, catalog)
        self.exact = defaultdict(Counter)
        self.projected = defaultdict(Counter)
        self.additions = defaultdict(set)
        self.quarantined = set()
        self.revisions = Counter()
        self.usage = Counter()
        self.cache = {}
        for row in self.graph.observations(namespace):
            before, after = facts(row["before"]), facts(row["after"])
            self._validate_observation(before, row["action"], after)
            self._learn(before, row["action"], after)

    def _validate_observation(self, before, name, after):
        action = self.by_name[name]
        if not action.applicable(before):
            raise ValueError("observation violates action preconditions")
        if action.writes is not None and not (before ^ after) <= action.writes:
            raise ValueError("observation changes facts outside the write footprint")

    def _learn(self, before, name, after):
        action = self.by_name[name]
        self.exact[(before, name)][after] += 1
        if action.context is not None:
            self.projected[(action.evidence_key(before), name)][after & action.writes] += 1
        self.additions[name].update(after - before)
        contract = self.contracts.get(name)
        if contract is not None and contract.apply(before) != after:
            self.quarantined.add(name)
        self.revisions[name] += 1
        self.cache = {k: v for k, v in self.cache.items() if k[1] != name}

    def observe(self, before, name, after, *, event, source):
        before, after = facts(before), facts(after)
        self._validate_observation(before, name, after)
        added = self.graph.observe(self.namespace, event, before, name, after, source)
        if added:
            self._learn(before, name, after)
        return added

    def ingest(self, rows):
        """Validate a complete batch before writing; repeated event IDs are idempotent."""
        rows = list(rows)
        events = {r["event"]: r for r in self.graph.observations(self.namespace)}
        normalized = []
        for row in rows:
            before, after = facts(row["before"]), facts(row["after"])
            self._validate_observation(before, row["action"], after)
            if not all(isinstance(row[k], str) and row[k].strip() for k in ("event", "source")):
                raise ValueError("source and event id required")
            item = {"event": row["event"], "source": row["source"], "action": row["action"],
                    "before": sorted(before), "after": sorted(after)}
            if item["event"] in events and events[item["event"]] != item:
                raise ValueError("conflicting observation event id")
            events[item["event"]] = item
            normalized.append(item)
        return sum(self.observe(r["before"], r["action"], r["after"],
                                event=r["event"], source=r["source"]) for r in normalized)

    def applicable(self, state):
        return [a for a in self.actions if a.applicable(state)]

    def predict(self, state, name):
        state = facts(state)
        action = self.by_name[name]
        self.usage["requests"] += 1
        if not action.applicable(state):
            self.usage["inapplicable"] += 1
            return None
        key = (state, name)
        prediction = self.cache.get(key)
        if prediction is None:
            prediction = self.graph.get_prediction(self.namespace, state, name, self.revisions[name])
        if prediction is not None:
            self.usage["cache"] += 1
            self.usage["served_" + prediction.layer] += 1
            self.cache[key] = prediction
            return prediction

        # Exact observations immediately override defeasible references. Projected
        # transfer requires min_support; an exact observation is never disguised as
        # statistically well-established knowledge (support and Wilson bound travel).
        counts = self.exact.get(key)
        if action.context is not None:
            projected = self.projected.get((action.evidence_key(state), name))
            if projected and sum(projected.values()) >= self.min_support:
                counts = Counter({(state - action.writes) | post: n for post, n in projected.items()})
        if counts:
            total = sum(counts.values())
            outcomes = tuple(Outcome(s, n / total) for s, n in sorted(counts.items(), key=lambda v: tuple(sorted(v[0]))))
            prediction = Prediction(outcomes, "L2", total, wilson_lower(max(counts.values()), total))
        elif name in self.contracts and name not in self.quarantined:
            prediction = Prediction((Outcome(self.contracts[name].apply(state), 1.0),), "L1")
        else:
            self.usage["unmodeled"] += 1
            return None
        self.usage[prediction.layer] += 1
        self.usage["served_" + prediction.layer] += 1
        self.graph.put_prediction(self.namespace, state, name, self.revisions[name], prediction)
        self.cache[key] = prediction
        return prediction

    def relaxed_operators(self):
        """Optimistic rollout guidance from public contracts and actual evidence only."""
        result = []
        for a in self.actions:
            c = self.contracts.get(a.name)
            if c is not None and a.name not in self.quarantined:
                result.append((a.preconditions, c.effect.add, a.cost))
                result.extend((a.preconditions | r.when, r.effect.add, a.cost) for r in c.rules)
            if self.additions[a.name]:
                result.append((a.preconditions, frozenset(self.additions[a.name]), a.cost))
        return result

    def coverage(self):
        observed = {name for _, name in self.exact}
        return {"actions": len(self.actions), "l1_contracts": len(self.contracts),
                "quarantined": sorted(self.quarantined), "observed_actions": len(observed),
                "uncovered_actions": sorted(set(self.by_name) - (set(self.contracts) - self.quarantined) - observed),
                "exact_state_action_pairs": len(self.exact),
                "projected_contexts": len(self.projected),
                "observations": sum(self.revisions.values())}
