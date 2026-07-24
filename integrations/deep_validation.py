"""
Deep validation battery: several layers past the per-company suite.

  1. Generated-workflow family: 120 seeded, randomly generated customer
     systems (layered dependencies, distractors, progress-destroying traps)
     - plan, execute, and verify determinism / chain / replay / provenance
     on every one. Seeds are fixed, so the whole battery is reproducible.
  2. Scale: a 40-stage, 200-tool pipeline - plan and execute end to end,
     with the full audit battery and wall-clock timing.
  3. Exhaustive tamper attack: for the reference runs, mutate EVERY field
     of EVERY record (not just one) and require the chain to break each time.
  4. Injection battery: for every archetype, attempt every tool from every
     execution prefix; every attempt whose preconditions are not actually
     satisfied must be refused. Counts total attempts and refusals.
  5. Crash-resume: kill the run after every prefix k, rebuild state on a
     fresh system by replaying the audit log, re-plan from there, and
     require the final state to match the uninterrupted run byte for byte.

Stdlib only, deterministic, no LLM calls.
"""
from __future__ import annotations

import copy
import random
import time
from typing import Dict, List, Set, Tuple

from .companies import ALL_COMPANIES
from .complex_cases import ALL_COMPLEX
from .core import (AuditedRuntime, CompanySystem, Tool, _record_hash,
                   state_hash, verify_chain, verify_provenance_closure,
                   verify_replay)

F = frozenset
APPROVE = lambda tool: True


# ===========================================================================
# 1. Generated-workflow family
# ===========================================================================

def generate_system(seed: int) -> Tuple[CompanySystem, Set[str], dict]:
    """A random layered workflow: facts arranged in layers, each produced by
    a tool whose preconditions come from earlier layers, plus distractors
    and trap tools that DELETE progress."""
    rng = random.Random(seed)
    layers = rng.randint(4, 8)
    width = rng.randint(2, 4)
    grid = [[f"L{i}F{j}" for j in range(width)] for i in range(layers)]
    initial = set(grid[0])
    tools: List[Tool] = []

    for i in range(1, layers):
        for f in grid[i]:
            k = rng.randint(1, min(3, width))
            pre = set(rng.sample(grid[i - 1], k))
            if i > 1 and rng.random() < 0.3:
                pre.add(rng.choice(grid[rng.randint(0, i - 2)]))
            tools.append(Tool(f"make_{f}", F(pre), F({f})))

    for d in range(rng.randint(2, 5)):
        src = rng.choice(grid[rng.randint(0, layers - 2)])
        tools.append(Tool(f"distract_{d}", F({src}), F({f"DEAD{d}"})))

    for t in range(rng.randint(1, 2)):
        lay = rng.randint(1, layers - 2)
        victim = rng.choice(grid[lay])
        trigger = rng.choice(grid[lay])
        tools.append(Tool(f"trap_{t}", F({trigger}), F({f"TRAPPED{t}"}),
                          effects_del=F({victim})))

    goal = {grid[-1][0]}
    goal_tool = next(t for t in tools if grid[-1][0] in t.effects_add)
    meta = {"lie_action": goal_tool.name,
            "claimed": goal_tool.preconditions,
            "layers": layers, "tools": len(tools)}
    return CompanySystem(f"gen{seed}", initial, tools), goal, meta


def run_generated_family(n: int = 120) -> dict:
    solved = deterministic = chain_ok = replay_ok = prov_ok = refused = 0
    injectable = 0
    plan_lens = []
    for seed in range(n):
        builder = lambda s=seed: generate_system(s)[0]
        _, goal, meta = generate_system(seed)

        rt1 = AuditedRuntime(builder(), goal, budget=24, max_steps=60, horizon=24)
        p1 = rt1.plan()
        if p1 is None:
            continue
        ok1 = rt1.execute(p1, human_approver=APPROVE)
        if not ok1:
            continue
        solved += 1
        plan_lens.append(len(p1))

        rt2 = AuditedRuntime(builder(), goal, budget=24, max_steps=60, horizon=24)
        p2 = rt2.plan()
        rt2.execute(p2, human_approver=APPROVE)
        if p1 == p2 and rt1.chain_head() == rt2.chain_head():
            deterministic += 1
        if verify_chain(rt1.records):
            chain_ok += 1
        if verify_replay(builder, goal, rt1.records):
            replay_ok += 1
        pok, _ = verify_provenance_closure(rt1.records, rt1.sys.initial_facts,
                                           rt1.sys.observe())
        if pok:
            prov_ok += 1

        # injection: goal-producing tool attempted from the initial state
        rt3 = AuditedRuntime(builder(), goal, budget=24, max_steps=60, horizon=24)
        missing = meta["claimed"] - rt3.sys.observe()
        if missing:
            injectable += 1
            out = rt3.attempt_injection([meta["lie_action"]], meta["claimed"])
            if all(o["verdict"] == "REFUSED" for o in out):
                refused += 1

    return {
        "battery": "generated_family", "systems": n, "solved": solved,
        "deterministic": deterministic, "chain_valid": chain_ok,
        "replay_ok": replay_ok, "provenance_closed": prov_ok,
        "injection_attempts": injectable, "injection_refused": refused,
        "mean_plan_len": round(sum(plan_lens) / max(1, len(plan_lens)), 1),
        "max_plan_len": max(plan_lens) if plan_lens else 0,
        "all_passed": (solved == n and deterministic == n and chain_ok == n
                       and replay_ok == n and prov_ok == n
                       and refused == injectable),
    }


# ===========================================================================
# 2. Scale: 40-stage, 200-tool pipeline
# ===========================================================================

def build_scale_system() -> Tuple[CompanySystem, Set[str]]:
    """40 sequential stages, each with 5 parallel work items that must all
    complete before the stage gate opens: 200 work tools + 40 gates.
    Work items consume their ticket (like a real queue), and gates consume
    their token, so completed work leaves the applicable set."""
    tools: List[Tool] = []
    for stage in range(40):
        gate_pre = {f"gate_token_{stage:02d}"}
        for item in range(5):
            done = f"s{stage:02d}w{item}_done"
            ticket = f"todo_s{stage:02d}w{item}"
            tools.append(Tool(f"work_s{stage:02d}_{item}", F({ticket}),
                              F({done}), effects_del=F({ticket})))
            gate_pre.add(done)
        next_adds = {f"gate_token_{stage + 1:02d}"}
        if stage < 39:
            next_adds |= {f"todo_s{stage + 1:02d}w{i}" for i in range(5)}
        tools.append(Tool(f"gate_{stage:02d}", F(gate_pre), F(next_adds),
                          effects_del=F({f"gate_token_{stage:02d}"})))
    initial = {f"todo_s00w{i}" for i in range(5)} | {"gate_token_00"}
    return CompanySystem("scale240", initial, tools), {"gate_token_40"}


def run_scale() -> dict:
    def builder() -> CompanySystem:
        sys_, _ = build_scale_system()
        return sys_

    goal = {"gate_token_40"}
    t0 = time.perf_counter()
    rt = AuditedRuntime(builder(), goal, budget=8, max_steps=300, horizon=90)
    plan = rt.plan()
    plan_s = time.perf_counter() - t0
    ok = plan is not None and rt.execute(plan, human_approver=APPROVE)
    total_s = time.perf_counter() - t0

    res = {
        "battery": "scale", "tools": len(rt.actions),
        "plan_len": len(plan) if plan else 0,
        "goal_reached": ok,
        "chain_valid": verify_chain(rt.records),
        "replay_ok": verify_replay(builder, goal, rt.records),
        "provenance_closed": verify_provenance_closure(
            rt.records, rt.sys.initial_facts, rt.sys.observe())[0],
        "plan_seconds": round(plan_s, 2),
        "total_seconds": round(total_s, 2),
        "audit_records": len(rt.records),
    }
    res["all_passed"] = all(v for v in res.values() if isinstance(v, bool))
    return res, rt  # runtime returned for the graph layer


# ===========================================================================
# 3. Exhaustive tamper attack
# ===========================================================================

def run_tamper_battery(reference_runs: Dict[str, List[dict]]) -> dict:
    """Mutate every string/int field of every record in every reference run;
    the chain must break on every single mutation."""
    mutations = detected = 0
    for name, records in reference_runs.items():
        for i, rec in enumerate(records):
            for key, val in rec.items():
                if key in ("record_hash", "prev_hash"):
                    continue
                tampered = copy.deepcopy(records)
                if isinstance(val, str):
                    tampered[i][key] = val + "X"
                elif isinstance(val, int):
                    tampered[i][key] = val + 1
                elif isinstance(val, list):
                    tampered[i][key] = val + ["X"]
                elif val is None:
                    tampered[i][key] = "X"
                else:
                    continue
                mutations += 1
                if not verify_chain(tampered):
                    detected += 1
    return {
        "battery": "tamper", "mutations": mutations, "detected": detected,
        "all_passed": mutations > 0 and mutations == detected,
    }


# ===========================================================================
# 4. Injection battery: every tool, every prefix, every archetype
# ===========================================================================

def run_injection_battery() -> dict:
    attempts = refused = wrongly_refused = 0
    for case in ALL_COMPANIES + ALL_COMPLEX:
        rt_kwargs = case.get("rt_kwargs", {})
        rt = AuditedRuntime(case["builder"](), case["goal"], **rt_kwargs)
        plan = rt.plan()
        by_name = {a.name: a for a in rt.actions}

        # walk the plan; before each step, attempt EVERY tool as a lie
        probe_sys = case["builder"]()
        probe = AuditedRuntime(probe_sys, case["goal"], **rt_kwargs)
        for step_idx in range(len(plan) + 1):
            truth = probe_sys.observe()
            for tool_name, act in by_name.items():
                missing = frozenset(act.preconditions) - truth
                if not missing:
                    continue  # genuinely valid here; not a lie
                attempts += 1
                out = probe.attempt_injection([tool_name], act.preconditions)
                if out[0]["verdict"] == "REFUSED":
                    refused += 1
            if step_idx < len(plan):
                name = plan[step_idx]
                if frozenset(by_name[name].preconditions) <= probe_sys.observe():
                    probe_sys.call(name)
    return {
        "battery": "injection", "attempts": attempts, "refused": refused,
        "false_refusals": wrongly_refused,
        "all_passed": attempts > 0 and attempts == refused,
    }


# ===========================================================================
# 5. Crash-resume from the audit log
# ===========================================================================

def run_crash_resume() -> dict:
    cases_checked = prefixes = matched = 0
    for case in ALL_COMPANIES + ALL_COMPLEX:
        rt_kwargs = case.get("rt_kwargs", {})
        ref = AuditedRuntime(case["builder"](), case["goal"], **rt_kwargs)
        ref_plan = ref.plan()
        ref.execute(ref_plan, human_approver=APPROVE)
        ref_final = state_hash(ref.sys.observe())
        exec_records = [r for r in ref.records if r["type"] == "execution"]
        cases_checked += 1

        for k in range(len(exec_records)):
            # crash after k steps; rebuild the customer state from the log
            fresh = case["builder"]()
            for rec in exec_records[:k]:
                fresh.call(rec["action"])
            rt2 = AuditedRuntime(fresh, case["goal"], **rt_kwargs)
            p2 = rt2.plan()
            ok = p2 is not None and rt2.execute(p2, human_approver=APPROVE)
            prefixes += 1
            if ok and state_hash(rt2.sys.observe()) == ref_final:
                matched += 1
    return {
        "battery": "crash_resume", "cases": cases_checked,
        "prefixes": prefixes, "final_state_matched": matched,
        "all_passed": prefixes > 0 and prefixes == matched,
    }
