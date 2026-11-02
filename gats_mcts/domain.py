"""Effect-free public actions and explicitly supplied symbolic contracts."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def facts(value) -> frozenset[str]:
    if isinstance(value, str):
        raise TypeError("facts must be a collection, not a string")
    result = frozenset(value)
    if any(not isinstance(f, str) or not f for f in result):
        raise ValueError("facts must be nonempty strings")
    return result


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    preconditions: frozenset[str] = field(default_factory=frozenset)
    forbidden: frozenset[str] = field(default_factory=frozenset)
    cost: float = 1.0
    # BOTH fields opt into a complete dependency/write-footprint contract.
    # None means exact-state evidence only; it never implies an empty context.
    context: frozenset[str] | None = None
    writes: frozenset[str] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("action name required")
        for key in ("preconditions", "forbidden", "context", "writes"):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, facts(getattr(self, key)))
        if self.preconditions & self.forbidden:
            raise ValueError("contradictory action guards")
        if not math.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("cost must be finite and positive")
        if (self.context is None) != (self.writes is None):
            raise ValueError("context and writes must be provided together")

    def applicable(self, state):
        return self.preconditions <= state and not self.forbidden & state

    def evidence_key(self, state):
        if self.context is None:
            return state
        return state & (self.context | self.writes | self.preconditions | self.forbidden)


@dataclass(frozen=True, slots=True)
class Effect:
    add: frozenset[str] = field(default_factory=frozenset)
    delete: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        object.__setattr__(self, "add", facts(self.add))
        object.__setattr__(self, "delete", facts(self.delete))

    def apply(self, state):
        return (state - self.delete) | self.add  # add wins, v2.2 STRIPS semantics


@dataclass(frozen=True, slots=True)
class Rule:
    when: frozenset[str]
    effect: Effect
    unless: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        object.__setattr__(self, "when", facts(self.when))
        object.__setattr__(self, "unless", facts(self.unless))
        if self.when & self.unless:
            raise ValueError("contradictory conditional guards")


@dataclass(frozen=True, slots=True)
class Contract:
    effect: Effect = field(default_factory=Effect)
    rules: tuple[Rule, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "rules", tuple(self.rules))

    def apply(self, state):
        add, delete = self.effect.add, self.effect.delete
        for rule in self.rules:
            # All guards evaluate against the pre-state, never earlier effects.
            if rule.when <= state and not rule.unless & state:
                add |= rule.effect.add
                delete |= rule.effect.delete
        return (state - delete) | add


@dataclass(frozen=True, slots=True)
class Outcome:
    state: frozenset[str]
    probability: float


@dataclass(frozen=True, slots=True)
class Prediction:
    outcomes: tuple[Outcome, ...]
    layer: str
    support: int = 0
    confidence_lower: float = 0.0


def catalog_data(actions, contracts):
    def effect(e):
        return {"add": sorted(e.add), "delete": sorted(e.delete)}
    return {
        "actions": [{"name": a.name, "preconditions": sorted(a.preconditions),
                     "forbidden": sorted(a.forbidden), "cost": a.cost,
                     "context": sorted(a.context) if a.context is not None else None,
                     "writes": sorted(a.writes) if a.writes is not None else None}
                    for a in sorted(actions, key=lambda a: a.name)],
        "contracts": {name: {"effect": effect(c.effect),
                             "rules": [{"when": sorted(r.when), "unless": sorted(r.unless),
                                        "effect": effect(r.effect)} for r in c.rules]}
                      for name, c in sorted(contracts.items())},
    }


def read_catalog(data):
    actions = [Action(**a) for a in data["actions"]]
    contracts = {}
    for name, c in data.get("contracts", {}).items():
        rules = tuple(Rule(r["when"], Effect(**r["effect"]), r.get("unless", ()))
                      for r in c.get("rules", ()))
        contracts[name] = Contract(Effect(**c["effect"]), rules)
    return actions, contracts
