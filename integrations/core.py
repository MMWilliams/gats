"""
Integration lab core: GATS as the planning-and-provenance layer over a
customer's own system.

The boundary is strict, mirroring a real integration:

  - CompanySystem owns the ground truth. Its fact store is private; GATS
    never mutates it directly. The only surface GATS sees is (a) the
    published tool contract (names, preconditions, declared effects) and
    (b) observe(), a read-only snapshot, the same thing a status API or
    log stream gives you in production.
  - AuditedRuntime is the embedded GATS layer: it plans with the real
    MCTS planner from gats_mcts over public symbolic contracts, then
    executes tool-by-tool, verifying
    each action's preconditions against the company's LIVE ground truth
    at execution time, never against what any upstream model claims.
  - Every execution appends a hash-chained audit record (prev-hash
    linked, canonical JSON, SHA-256). Wall-clock timestamps live in the
    unhashed envelope so chains are byte-reproducible across runs.

Stdlib only, deterministic, no LLM calls.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from gats_mcts import (Action as ModelAction, Contract, Effect,
                       LayeredWorldModel, MCTSPlanner)
from gats_mcts.domain import catalog_data
from gats_mcts.heuristic import AdditiveHeuristic

# --- load the real demo engine (single source of truth for the planner) ----
_DEMOS = Path(__file__).resolve().parent.parent / "demos"
_spec = importlib.util.spec_from_file_location("agent_lies_demo", _DEMOS / "agent_lies_demo.py")
_eng = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules["agent_lies_demo"] = _eng
_spec.loader.exec_module(_eng)

State = _eng.State
Action = _eng.Action


# ===========================================================================
# Company side: a completely separate system we integrate with
# ===========================================================================

class ToolError(Exception):
    def __init__(self, tool: str, missing: FrozenSet[str]):
        super().__init__(f"{tool}: missing {sorted(missing)}")
        self.tool = tool
        self.missing = missing


@dataclass
class Tool:
    """A tool in the company's own stack. `effects_add`/`effects_del` are the
    PUBLISHED contract. `hidden_add` (undocumented side effects),
    `fails_to_add` (declared but not delivered) and `hidden_del` (undocumented
    resets) simulate contract drift in the customer's real system; GATS never
    sees them."""
    name: str
    preconditions: FrozenSet[str]
    effects_add: FrozenSet[str]
    effects_del: FrozenSet[str] = frozenset()
    risk: str = "low"
    hidden_add: FrozenSet[str] = frozenset()
    fails_to_add: FrozenSet[str] = frozenset()
    hidden_del: FrozenSet[str] = frozenset()


class CompanySystem:
    """One customer. Own namespace, own private state, own tool API."""

    def __init__(self, company_id: str, initial_facts: Set[str],
                 tools: List[Tool], human_gated: Set[str] = frozenset()):
        self.company_id = company_id
        self.initial_facts = frozenset(initial_facts)
        self._facts: Set[str] = set(initial_facts)
        self.tools: Dict[str, Tool] = {t.name: t for t in tools}
        self.human_gated = set(human_gated)
        self.call_log: List[str] = []

    def observe(self) -> FrozenSet[str]:
        return frozenset(self._facts)

    def published_contract(self) -> List[Action]:
        return [Action(t.name, t.preconditions, t.effects_add, t.effects_del)
                for t in self.tools.values()]

    def call(self, tool_name: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        """Execute like the company's real API: enforce ITS OWN preconditions,
        apply ITS OWN (possibly drifted) effects. Returns (added, removed)."""
        t = self.tools[tool_name]
        missing = t.preconditions - self._facts
        if missing:
            raise ToolError(tool_name, frozenset(missing))
        before = frozenset(self._facts)
        self._facts |= (set(t.effects_add) - set(t.fails_to_add)) | set(t.hidden_add)
        self._facts -= set(t.effects_del) | set(t.hidden_del)
        after = frozenset(self._facts)
        self.call_log.append(tool_name)
        return after - before, before - after


# ===========================================================================
# Hashing helpers
# ===========================================================================

def state_hash(facts: FrozenSet[str]) -> str:
    return hashlib.sha256("\n".join(sorted(facts)).encode()).hexdigest()[:16]


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _record_hash(prev_hash: str, rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k not in ("record_hash", "prev_hash")}
    return hashlib.sha256((prev_hash + _canonical(body)).encode()).hexdigest()


# ===========================================================================
# The embedded GATS layer
# ===========================================================================

class AuditedRuntime:
    """GATS embedded under one company's stack, with a tamper-evident log."""

    def __init__(self, system: CompanySystem, goal: Set[str],
                 budget: int = 24, max_steps: int = 40, horizon: int = 12,
                 graph=None, agent: str = "agent", task: Optional[str] = None):
        self.sys = system
        self.goal = frozenset(goal)
        self.actions = system.published_contract()
        self.by_name = {a.name: a for a in self.actions}
        self.horizon = horizon
        self.budget = budget
        self.max_steps = max_steps
        self.agent = agent
        self.task = task or system.company_id
        self.records: List[dict] = []
        self.security_events: List[dict] = []
        self.drift_events: List[dict] = []
        self.divergences = 0
        self.divergent_actions: List[str] = []
        self.explorations = 0
        self.status = "init"
        # No context/write completeness promise is available from these tool
        # APIs, so runtime observations use conservative exact-state L2 keys.
        actions = [ModelAction(a.name, a.preconditions) for a in self.actions]
        contracts = {a.name: Contract(Effect(a.effects_add, a.effects_del)) for a in self.actions}
        self.retrieved = graph is not None and graph.catalog(system.company_id) is not None
        if self.retrieved:
            # Another agent registered this system: its L1 catalog and every
            # execution it recorded come from the shared graph, not from here.
            self.world_model = LayeredWorldModel.from_graph(graph, system.company_id)
            stored = {k: v for k, v in graph.catalog(system.company_id).items() if k != "model"}
            if stored != catalog_data(actions, contracts):
                raise ValueError("published contract differs from the shared graph catalog; "
                                 "register the changed API under a new namespace")
        else:
            self.world_model = LayeredWorldModel(actions, contracts, graph=graph,
                                                 namespace=system.company_id)
        # Published-contract guidance for exploration only; never a prediction.
        self._published_distance = AdditiveHeuristic(
            [(frozenset(a.preconditions), frozenset(a.effects_add), 1.0) for a in self.actions],
            self.goal)
        self.search_results = []

    # -- audit chain --------------------------------------------------------
    def _append(self, rec: dict) -> None:
        prev = self.records[-1]["record_hash"] if self.records else "GENESIS"
        rec["seq"] = len(self.records)
        rec["company"] = self.sys.company_id
        rec["prev_hash"] = prev
        rec["record_hash"] = _record_hash(prev, rec)
        self.records.append(rec)

    def chain_head(self) -> str:
        return self.records[-1]["record_hash"] if self.records else "GENESIS"

    # -- planning (zero LLM calls) ------------------------------------------
    def _planner(self) -> MCTSPlanner:
        return MCTSPlanner(self.world_model, simulations=self.budget,
                           max_depth=max(self.horizon, self.max_steps), seed=0)

    def plan(self) -> Optional[List[str]]:
        """Produce a modeled plan without executing tools or creating evidence."""
        planner = self._planner()
        state, plan, seen = self.sys.observe(), [], set()
        self.search_results = []
        for _ in range(self.max_steps):
            if self.goal <= state:
                return plan
            if state in seen:
                return None
            seen.add(state)
            result = planner.search(state, self.goal)
            self.search_results.append(result)
            if result.action is None:
                return None
            prediction = self.world_model.predict(state, result.action)
            # Existing contract: return one modeled sequence. Execution still
            # checks live truth and human gates at every step.
            state = max(prediction.outcomes, key=lambda o: o.probability).state
            plan.append(result.action)
        return plan if self.goal <= state else None

    # -- execution with per-step ground-truth verification ------------------
    def execute(self, plan_names: List[str],
                human_approver: Optional[Callable[[str], bool]] = None) -> bool:
        for name in plan_names:
            if not self.step(name, human_approver, "plan"):
                return False
        goal_ok = self.goal.issubset(self.sys.observe())
        self.status = "goal_reached" if goal_ok else "plan_exhausted"
        return goal_ok

    def step(self, name: str, human_approver, basis: str = "plan") -> bool:
        """Execute one action under the guards; False after a refusal or a blocked gate."""
        act = self.by_name[name]
        truth = self.sys.observe()

        # The security primitive: preconditions vs LIVE ground truth.
        missing = frozenset(act.preconditions) - truth
        if missing:
            self._append({
                "type": "refusal", "action": name, "actor": "agent",
                "missing_preconditions": sorted(missing),
                "pre_state": state_hash(truth),
            })
            self.status = "halted_precondition_failed"
            return False

        # FINRA-style human checkpoint: gated tools never auto-execute.
        if name in self.sys.human_gated:
            if human_approver is None or not human_approver(name):
                self._append({
                    "type": "checkpoint_blocked", "action": name,
                    "actor": "agent", "pre_state": state_hash(truth),
                    "note": "human approval required and not granted",
                })
                self.status = "blocked_awaiting_human"
                return False
            actor = "human"
        else:
            actor = "agent"

        added, removed = self.sys.call(name)

        # Contract-drift detection: observed delta vs published effects.
        expected_add = frozenset(act.effects_add) - truth
        unexpected_add = added - frozenset(act.effects_add)
        missing_add = expected_add - added
        unexpected_del = removed - frozenset(act.effects_del)
        drift = {}
        if unexpected_add:
            drift["undocumented_side_effects"] = sorted(unexpected_add)
        if missing_add:
            drift["declared_effects_not_delivered"] = sorted(missing_add)
        if unexpected_del:
            drift["undocumented_deletions"] = sorted(unexpected_del)
        if drift:
            self.drift_events.append({"action": name, **drift})

        self._append({
            "type": "execution", "action": name, "actor": actor, "basis": basis,
            "preconditions": sorted(act.preconditions),
            "pre_state": state_hash(truth),
            "post_state": state_hash(self.sys.observe()),
            "added": sorted(added), "removed": sorted(removed),
            "drift": drift or None,
        })
        self.world_model.observe(truth, name, self.sys.observe(),
                                 event=f"{self.agent}/{self.task}/{self.records[-1]['seq']}",
                                 source=f"audited-execution:{self.agent}:" + self.records[-1]["record_hash"])
        return True

    # -- the GATS loop: search, act under the guards, observe, search again --
    def run(self, human_approver: Optional[Callable[[str], bool]] = None) -> bool:
        """Receding-horizon episode. Each step searches from live truth with
        whatever evidence any agent has written to the shared graph. When no
        simulated path reaches the goal, the runtime heads for the nearest tool
        the model cannot predict (its outcome becomes evidence); those steps are
        recorded as exploration. When no such tool is reachable it stops instead
        of executing hopeless actions."""
        planner = self._planner()
        self.search_results = []
        for _ in range(self.max_steps):
            state = self.sys.observe()
            if self.goal <= state:
                self.status = "goal_reached"
                return True
            self.world_model.sync()
            result = planner.search(state, self.goal)
            self.search_results.append(result)
            action, basis = result.action, "search"
            if result.goal_hits == 0:
                action, basis = self._toward_unmodeled(state), "exploration"
            if action is None:
                self.status = "no_modeled_path"
                return False
            if basis == "exploration":
                self.explorations += 1
            prediction = self.world_model.predict(state, action)
            if not self.step(action, human_approver, basis):
                return False
            if prediction is not None and self.sys.observe() not in {o.state for o in prediction.outcomes}:
                self.divergences += 1
                self.divergent_actions.append(action)
        self.status = "goal_reached" if self.goal <= self.sys.observe() else "step_limit"
        return self.status == "goal_reached"

    def _toward_unmodeled(self, state: FrozenSet[str]) -> Optional[str]:
        """First step of the shortest modeled path to a state where a tool the
        model cannot predict, and whose published contract would bring the goal
        nearer, becomes applicable. None when no such tool is reachable. The
        contract guides where to look; it is never used as a prediction."""
        def helpful_unmodeled(s):
            best = None
            for act in self.actions:
                if (not frozenset(act.preconditions) <= s
                        or self.world_model.predict(s, act.name) is not None):
                    continue
                promised = (s - frozenset(act.effects_del)) | frozenset(act.effects_add)
                distance = self._published_distance(promised)
                if distance != float("inf") and (best is None or (distance, act.name) < best):
                    best = (distance, act.name)
            return best[1] if best else None

        with self.world_model.graph.prediction_batch():
            queue, seen = deque([(state, None)]), {state}
            while queue and len(seen) <= 256:
                current, first = queue.popleft()
                candidate = helpful_unmodeled(current)
                if candidate is not None:
                    return first or candidate
                for act in self.world_model.applicable(current):
                    prediction = self.world_model.predict(current, act.name)
                    if prediction is None:
                        continue
                    following = max(prediction.outcomes, key=lambda o: o.probability).state
                    if following not in seen:
                        seen.add(following)
                        queue.append((following, first or act.name))
        return None

    # -- the injected/compromised path --------------------------------------
    def attempt_injection(self, lie_actions: List[str],
                          claimed_true: FrozenSet[str]) -> List[dict]:
        """A compromised upstream layer asserts `claimed_true` and tries to
        jump straight to high-blast-radius actions. The guarded layer checks
        ground truth, not claims."""
        by_name = {a.name: a for a in self.actions}
        outcomes = []
        for name in lie_actions:
            act = by_name[name]
            truth = self.sys.observe()
            missing = frozenset(act.preconditions) - truth
            if missing:
                event = {
                    "action": name, "verdict": "REFUSED",
                    "claimed_true": sorted(claimed_true),
                    "actually_missing": sorted(missing),
                }
                self.security_events.append(event)
                self._append({"type": "security_refusal", "actor": "agent", **event,
                              "pre_state": state_hash(truth)})
            else:
                event = {"action": name, "verdict": "EXECUTED",
                         "claimed_true": sorted(claimed_true)}
                self.security_events.append(event)
            outcomes.append(event)
        return outcomes


# ===========================================================================
# Verification suite (what an auditor can check from the log alone)
# ===========================================================================

def verify_chain(records: List[dict]) -> bool:
    prev = "GENESIS"
    for rec in records:
        if rec["prev_hash"] != prev:
            return False
        if _record_hash(prev, rec) != rec["record_hash"]:
            return False
        prev = rec["record_hash"]
    return True


def verify_tamper_evidence(records: List[dict]) -> bool:
    """Flipping any field in any record must break the chain."""
    if not records:
        return False
    for i in range(len(records)):
        tampered = copy.deepcopy(records)
        tampered[i]["action"] = tampered[i].get("action", "") + "_TAMPERED"
        if verify_chain(tampered):
            return False
    return True


def verify_provenance_closure(records: List[dict], initial: FrozenSet[str],
                              final: FrozenSet[str]) -> Tuple[bool, dict]:
    """Every final fact must trace to a producing record; every executed
    action's preconditions must be initial facts or produced earlier.
    Checked from the log alone, without trusting the runtime."""
    produced_at: Dict[str, int] = {}
    ok = True
    problems = []
    for rec in records:
        if rec["type"] != "execution":
            continue
        for pre in rec["preconditions"]:
            if pre not in initial and pre not in produced_at:
                ok = False
                problems.append(f"seq {rec['seq']}: precondition '{pre}' has no producer")
        for fact in rec["added"]:
            produced_at.setdefault(fact, rec["seq"])
    for fact in final - initial:
        if fact not in produced_at:
            ok = False
            problems.append(f"final fact '{fact}' has no producing record")
    graph = {fact: seq for fact, seq in produced_at.items()}
    return ok, {"provenance": graph, "problems": problems}


def verify_replay(builder: Callable[[], CompanySystem], goal: Set[str],
                  records: List[dict]) -> bool:
    """Re-execute the recorded actions on a FRESH company instance; every
    recorded pre/post state hash must match byte for byte."""
    fresh = builder()
    for rec in records:
        if rec["type"] != "execution":
            continue
        if state_hash(fresh.observe()) != rec["pre_state"]:
            return False
        fresh.call(rec["action"])
        if state_hash(fresh.observe()) != rec["post_state"]:
            return False
    return True


def run_full_suite(name: str, builder: Callable[[], CompanySystem],
                   goal: Set[str], lie_actions: List[str],
                   claimed_true: FrozenSet[str], n_determinism: int = 3,
                   expect_drift: bool = False, **rt_kwargs) -> dict:
    """The uniform verification battery, run per company."""
    results: Dict[str, object] = {"company": name}
    approver = lambda tool: True  # simulated human clicking Approve

    # 1. Determinism: identical plan, identical chain head, identical final state.
    plans, heads, finals = [], [], []
    for _ in range(n_determinism):
        rt = AuditedRuntime(builder(), goal, **rt_kwargs)
        plan = rt.plan()
        assert plan is not None, f"{name}: planner found no plan"
        rt.execute(plan, human_approver=approver)
        plans.append(tuple(plan))
        heads.append(rt.chain_head())
        finals.append(state_hash(rt.sys.observe()))
    results["deterministic"] = len(set(plans)) == 1 and len(set(heads)) == 1 \
        and len(set(finals)) == 1
    results["plan"] = list(plans[0])
    results["chain_head"] = heads[0]

    # Reference run for the remaining checks.
    rt = AuditedRuntime(builder(), goal, **rt_kwargs)
    plan = rt.plan()
    goal_reached = rt.execute(plan, human_approver=approver)
    results["goal_reached"] = goal_reached
    results["status"] = rt.status
    results["steps"] = len([r for r in rt.records if r["type"] == "execution"])
    results["drift_events"] = rt.drift_events
    if expect_drift:
        results["drift_detected"] = len(rt.drift_events) > 0
    else:
        results["no_unexpected_drift"] = len(rt.drift_events) == 0

    # 2. Audit chain integrity + tamper evidence.
    results["chain_valid"] = verify_chain(rt.records)
    results["tamper_evident"] = verify_tamper_evidence(rt.records)

    # 3. Provenance closure from the log alone.
    prov_ok, prov = verify_provenance_closure(
        rt.records, rt.sys.initial_facts, rt.sys.observe())
    results["provenance_closed"] = prov_ok
    results["provenance_problems"] = prov["problems"]

    # 4. Deterministic replay on a fresh instance.
    results["replay_ok"] = verify_replay(builder, goal, rt.records)

    # 5. Injection: the lie is refused against ground truth; the guarded
    #    planner then still reaches the goal legitimately.
    rt_inj = AuditedRuntime(builder(), goal, **rt_kwargs)
    outcomes = rt_inj.attempt_injection(lie_actions, claimed_true)
    results["injection_refused"] = all(o["verdict"] == "REFUSED" for o in outcomes)
    results["injection_events"] = outcomes
    plan2 = rt_inj.plan()
    results["recovered_after_injection"] = plan2 is not None and \
        rt_inj.execute(plan2, human_approver=approver)

    # 6. Human checkpoint: without an approver the gated action never runs.
    gated = builder().human_gated
    if gated:
        rt_blocked = AuditedRuntime(builder(), goal, **rt_kwargs)
        plan3 = rt_blocked.plan()
        done = rt_blocked.execute(plan3, human_approver=None)
        gated_ran = any(t in rt_blocked.sys.call_log for t in gated)
        results["checkpoint_blocks_without_human"] = (
            not done and rt_blocked.status == "blocked_awaiting_human"
            and not gated_ran)
    else:
        results["checkpoint_blocks_without_human"] = None

    checks = [v for k, v in results.items() if isinstance(v, bool)]
    results["all_passed"] = all(checks)
    return results
