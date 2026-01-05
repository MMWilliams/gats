#!/usr/bin/env python3
"""
GATS 2.0 Lean World Model + Benchmark

Single-file implementation with:
  - L1: Exact ActionSpec effects
  - L2: Log-based learning (primary)
  - L3: Open LLM fallback
  - Agent comparison benchmarks

Usage:
    python run.py [test|bench|agents|all]
"""
from __future__ import annotations
import sys, time, json, re, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from collections import defaultdict

# =============================================================================
# Core Types
# =============================================================================

@dataclass(frozen=True)
class Transition:
    """Single state transition record."""
    state: frozenset[str]
    action: str
    args: dict
    next_state: frozenset[str]
    success: bool = True
    cost: float = 1.0

@dataclass
class Stats:
    """Aggregated statistics for predictions."""
    n: int = 0
    ok: int = 0
    effects: dict[frozenset[str], int] = field(default_factory=dict)
    cost: float = 1.0
    
    @property
    def p_success(self) -> float: return self.ok / max(1, self.n)
    
    def best_effects(self) -> tuple[frozenset[str], float]:
        if not self.effects: return frozenset(), 0.0
        best = max(self.effects, key=self.effects.get)
        return best, self.effects[best] / self.n

@dataclass
class State:
    """Agent state with goal and inventory."""
    goal: frozenset[str] = field(default_factory=frozenset)
    inventory: frozenset[str] = field(default_factory=frozenset)
    def is_goal(self) -> bool: return self.goal <= self.inventory
    def with_inventory(self, inv): return State(self.goal, inv)

@dataclass
class Spec:
    """Action specification."""
    action_id: str
    preconditions: frozenset[str] = field(default_factory=frozenset)
    effects_add: frozenset[str] = field(default_factory=frozenset)
    effects_remove: frozenset[str] = field(default_factory=frozenset)
    cost: float = 1.0

class ActionModel:
    """Action model with precondition checking."""
    def __init__(self, specs: list[Spec]):
        self._specs = {s.action_id: s for s in specs}
    
    def resolve(self, aid: str) -> Spec | None: return self._specs.get(aid)
    
    def verify(self, cand, state):
        spec = self.resolve(cand.action_id)
        @dataclass
        class R:
            is_valid: bool = False
            compiled_effects_add: frozenset = field(default_factory=frozenset)
            compiled_effects_remove: frozenset = field(default_factory=frozenset)
        if not spec: return R()
        if not spec.preconditions <= state.inventory: return R()
        return R(True, spec.effects_add, spec.effects_remove)
    
    def get_legal(self, state: State) -> list[Spec]:
        return [s for s in self._specs.values() if s.preconditions <= state.inventory]

# =============================================================================
# Layer 1: Exact Effects
# =============================================================================

class L1Exact:
    """Deterministic predictions from ActionSpec."""
    
    def __init__(self, am=None):
        self.am = am
    
    def predict(self, state, action: str, args: dict = None) -> tuple[frozenset[str], float]:
        if not self.am: return state.inventory, 0.0
        spec = self.am.resolve(action)
        if not spec: return state.inventory, 0.0
        
        @dataclass
        class C: action_id: str; args: dict = field(default_factory=dict)
        result = self.am.verify(C(action, args or {}), state)
        if not getattr(result, 'is_valid', False): return state.inventory, 0.0
        
        adds = getattr(result, 'compiled_effects_add', frozenset())
        rems = getattr(result, 'compiled_effects_remove', frozenset())
        return (state.inventory | adds) - rems, 1.0

# =============================================================================
# Layer 2: Log-based Learning
# =============================================================================

class L2Learned:
    """Statistical model learned from execution logs."""
    
    def __init__(self, min_obs: int = 3, conf: float = 0.6):
        self.min_obs, self.conf = min_obs, conf
        self._by_action: dict[str, Stats] = defaultdict(Stats)
        self._by_state_action: dict[tuple, Stats] = defaultdict(Stats)
    
    def record(self, t: Transition) -> None:
        key = (tuple(sorted(t.state)), t.action)
        for stats in (self._by_action[t.action], self._by_state_action[key]):
            stats.n += 1
            if t.success:
                stats.ok += 1
                eff = t.next_state - t.state
                stats.effects[eff] = stats.effects.get(eff, 0) + 1
            stats.cost = (stats.cost * (stats.n-1) + t.cost) / stats.n
    
    def load_jsonl(self, path: Path) -> int:
        n = 0
        for line in Path(path).read_text().splitlines():
            try:
                d = json.loads(line)
                self.record(Transition(
                    state=frozenset(d.get('state_before', d.get('state', []))),
                    action=d.get('action_id', d.get('action', '')),
                    args=d.get('args', {}),
                    next_state=frozenset(d.get('state_after', d.get('next_state', []))),
                    success=d.get('success', True),
                    cost=d.get('cost', 1.0)
                ))
                n += 1
            except: pass
        return n
    
    def load_dir(self, dir_path: Path, pattern: str = "*.jsonl") -> int:
        return sum(self.load_jsonl(f) for f in Path(dir_path).glob(pattern))
    
    def predict(self, state, action: str, args: dict = None) -> tuple[frozenset[str], float]:
        key = (tuple(sorted(state.inventory)), action)
        
        if key in self._by_state_action:
            s = self._by_state_action[key]
            if s.n >= self.min_obs:
                eff, p = s.best_effects()
                if p >= self.conf:
                    return state.inventory | eff, p * s.p_success
        
        if action in self._by_action:
            s = self._by_action[action]
            if s.n >= self.min_obs:
                eff, p = s.best_effects()
                adj_p = p * s.p_success * 0.8
                if adj_p >= self.conf * 0.5:
                    return state.inventory | eff, adj_p
        
        return state.inventory, 0.0
    
    @property
    def stats(self) -> dict:
        return {'actions': len(self._by_action), 
                'state_actions': len(self._by_state_action),
                'total_obs': sum(s.n for s in self._by_action.values())}

# =============================================================================
# Layer 3: Generative Fallback
# =============================================================================

PATTERNS = {
    'get_': '_data', 'fetch_': '_data', 'search_': '_results',
    'create_': '_id', 'update_': '_updated', 'delete_': '_deleted',
    'send_': '_sent', 'book_': '_confirmation', 'check_': '_status',
}

class L3Generative:
    """Heuristic + optional LLM fallback."""
    
    def __init__(self, llm: Callable[[str, frozenset], frozenset] | None = None):
        self.llm = llm
    
    def predict(self, state, action: str, args: dict = None) -> tuple[frozenset[str], float]:
        if self.llm:
            try:
                eff = self.llm(action, state.inventory)
                if eff: return state.inventory | eff, 0.5
            except: pass
        
        for prefix, suffix in PATTERNS.items():
            if action.startswith(prefix):
                base = action[len(prefix):]
                return state.inventory | frozenset([f"{base}{suffix}"]), 0.4
        
        if '.' in action:
            _, endpoint = action.rsplit('.', 1)
            for prefix, suffix in PATTERNS.items():
                if endpoint.startswith(prefix):
                    base = endpoint[len(prefix):]
                    return state.inventory | frozenset([f"{base}{suffix}"]), 0.35
            return state.inventory | frozenset([f"{endpoint}_result"]), 0.3
        
        return state.inventory | frozenset([f"{action}_result"]), 0.2

# =============================================================================
# World Model
# =============================================================================

class WorldModel:
    """Cascading world model: L1 → L2 → L3"""
    
    def __init__(self, action_model=None, llm=None, min_obs: int = 3):
        self.l1 = L1Exact(action_model)
        self.l2 = L2Learned(min_obs=min_obs)
        self.l3 = L3Generative(llm)
        self._last_layer = 0
    
    def load_logs(self, path: Path) -> int:
        p = Path(path)
        return self.l2.load_dir(p) if p.is_dir() else self.l2.load_jsonl(p)
    
    def record(self, t: Transition) -> None:
        self.l2.record(t)
    
    def predict(self, state, action: str, args: dict = None) -> tuple[frozenset[str], float]:
        inv, p = self.l1.predict(state, action, args)
        if p > 0: self._last_layer = 1; return inv, p
        
        inv, p = self.l2.predict(state, action, args)
        if p > 0: self._last_layer = 2; return inv, p
        
        self._last_layer = 3
        return self.l3.predict(state, action, args)
    
    @property
    def last_layer(self) -> int: return self._last_layer
    
    @property
    def stats(self) -> dict: return {'l2': self.l2.stats, 'last_layer': self._last_layer}

# =============================================================================
# Open Model Predictors
# =============================================================================

def create_ollama_predictor(model: str = "llama3.2", host: str = "http://localhost:11434"):
    """Ollama local model predictor."""
    import requests
    def predict(action: str, inventory: frozenset) -> frozenset:
        prompt = f'Action "{action}" with state {sorted(inventory)}. Return JSON list of new items: '
        resp = requests.post(f"{host}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 50}
        }, timeout=10)
        if resp.ok:
            match = re.search(r'\[.*?\]', resp.json().get('response', ''))
            if match: return frozenset(json.loads(match.group()))
        return frozenset()
    return predict

def create_vllm_predictor(model: str = "meta-llama/Llama-3.2-3B-Instruct",
                          base_url: str = "http://localhost:8000/v1"):
    """vLLM server predictor (OpenAI-compatible)."""
    import requests
    def predict(action: str, inventory: frozenset) -> frozenset:
        prompt = f'Action: {action}\nState: {sorted(inventory)}\nNew items JSON: ['
        resp = requests.post(f"{base_url}/completions", json={
            "model": model, "prompt": prompt, "max_tokens": 50,
            "temperature": 0.1, "stop": ["\n", "]"]
        }, timeout=10)
        if resp.ok:
            text = "[" + resp.json()['choices'][0]['text'] + "]"
            match = re.search(r'\[.*?\]', text)
            if match: return frozenset(json.loads(match.group()))
        return frozenset()
    return predict

# =============================================================================
# Agents
# =============================================================================

@dataclass
class Episode:
    task_id: str = ""
    success: bool = False
    steps: int = 0
    cost: float = 0.0

class TreeSearchAgent:
    """GATS-style tree search with lookahead."""
    
    def __init__(self, am: ActionModel, wm: WorldModel, depth: int = 6):
        self.am, self.wm, self.depth = am, wm, depth
    
    def _search(self, state: State, goal: frozenset, cost: float, d: int) -> tuple[float, list]:
        if goal <= state.inventory: return cost, []
        if d >= self.depth: return cost + len(goal - state.inventory) * 2, None
        
        best_cost, best_path = float('inf'), None
        for spec in sorted(self.am.get_legal(state), key=lambda s: s.cost):
            inv, p = self.wm.predict(state, spec.action_id)
            if p > 0 and inv != state.inventory:
                fc, fp = self._search(state.with_inventory(inv), goal, cost + spec.cost, d + 1)
                if fc < best_cost:
                    best_cost, best_path = fc, [spec] + (fp or [])
        return best_cost, best_path
    
    def solve(self, state: State, task_id: str = "") -> Episode:
        ep = Episode(task_id)
        visited = set()
        
        for _ in range(50):
            if state.is_goal(): break
            h = tuple(sorted(state.inventory))
            if h in visited: break
            visited.add(h)
            
            _, path = self._search(state, state.goal, 0, 0)
            if not path:
                best = None
                for spec in self.am.get_legal(state):
                    inv, p = self.wm.predict(state, spec.action_id)
                    if p > 0 and tuple(sorted(inv)) not in visited:
                        if not best or spec.cost < best[0].cost:
                            best = (spec, inv)
                if not best: break
                spec, inv = best
            else:
                spec = path[0]
                inv, _ = self.wm.predict(state, spec.action_id)
            
            state = state.with_inventory(inv)
            ep.steps += 1
            ep.cost += spec.cost
        
        ep.success = state.is_goal()
        return ep

class GreedyAgent:
    """ReAct-style greedy agent - picks cheapest action."""
    
    def __init__(self, am: ActionModel, wm: WorldModel):
        self.am, self.wm = am, wm
    
    def solve(self, state: State, task_id: str = "") -> Episode:
        ep = Episode(task_id)
        visited = set()
        
        for _ in range(50):
            if state.is_goal(): break
            h = tuple(sorted(state.inventory))
            if h in visited: break
            visited.add(h)
            
            best = None
            for spec in sorted(self.am.get_legal(state), key=lambda s: s.cost):
                inv, p = self.wm.predict(state, spec.action_id)
                if p > 0 and tuple(sorted(inv)) not in visited:
                    best = (spec, inv)
                    break
            if not best: break
            
            spec, inv = best
            state = state.with_inventory(inv)
            ep.steps += 1
            ep.cost += spec.cost
        
        ep.success = state.is_goal()
        return ep

# =============================================================================
# Hard Tasks
# =============================================================================

def make_hard_tasks():
    """Tasks with traps that differentiate tree search from greedy."""
    specs = [
        # Deceptive: correct=8, trap=0.5 dead-end
        Spec("gather_data", frozenset(["start"]), frozenset(["raw_data"]), cost=2.0),
        Spec("analyze_data", frozenset(["raw_data"]), frozenset(["analysis"]), cost=2.0),
        Spec("draft_report", frozenset(["analysis"]), frozenset(["draft"]), cost=2.0),
        Spec("finalize", frozenset(["draft"]), frozenset(["final_report"]), cost=2.0),
        Spec("quick_summary", frozenset(["start"]), frozenset(["summary"]), cost=0.5),
        
        # Cost-optimal: booking+flight=5 vs scenic=11
        Spec("make_booking", frozenset(["origin"]), frozenset(["booked"]), cost=1.0),
        Spec("direct_flight", frozenset(["origin", "booked"]), frozenset(["dest"]), cost=4.0),
        Spec("via_waypoint", frozenset(["origin"]), frozenset(["dest"]), cost=6.0),
        Spec("scenic_start", frozenset(["origin"]), frozenset(["scenic"]), cost=0.5),
        Spec("scenic_end", frozenset(["scenic"]), frozenset(["dest"]), cost=10.0),
        
        # Resource: must conserve fuel
        Spec("scout", frozenset(["base"]), frozenset(["scouted"]), cost=2.0),
        Spec("plan", frozenset(["scouted"]), frozenset(["planned"]), cost=2.0),
        Spec("execute", frozenset(["planned", "fuel"]), frozenset(["mission_done"]), 
             frozenset(["fuel"]), cost=3.0),
        Spec("joy_ride", frozenset(["base", "fuel"]), frozenset(["fun"]),
             frozenset(["fuel"]), cost=0.5),
        
        # Dead-end trap
        Spec("explore", frozenset(["entrance"]), frozenset(["hall"]), cost=2.0),
        Spec("find_key", frozenset(["hall"]), frozenset(["key"]), cost=2.0),
        Spec("open_vault", frozenset(["hall", "key"]), frozenset(["vault"]), cost=2.0),
        Spec("get_treasure", frozenset(["vault"]), frozenset(["treasure"]), cost=2.0),
        Spec("rush_ahead", frozenset(["entrance"]), frozenset(["pit"]), cost=0.5),
        Spec("struggle", frozenset(["pit"]), frozenset(["stuck"]), cost=1.0),
    ]
    
    tasks = [
        *[{"id": f"decept_{i}", "type": "deceptive", 
           "goal": frozenset(["final_report"]), "inv": frozenset(["start"])} for i in range(5)],
        *[{"id": f"cost_{i}", "type": "cost_optimal",
           "goal": frozenset(["dest"]), "inv": frozenset(["origin"])} for i in range(5)],
        *[{"id": f"resource_{i}", "type": "resource",
           "goal": frozenset(["mission_done"]), "inv": frozenset(["base", "fuel"])} for i in range(5)],
        *[{"id": f"deadend_{i}", "type": "dead_end",
           "goal": frozenset(["treasure"]), "inv": frozenset(["entrance"])} for i in range(5)],
    ]
    return ActionModel(specs), tasks

# =============================================================================
# Tests & Benchmarks
# =============================================================================

def test_layers():
    """Test world model layers."""
    print("=" * 60)
    print("Layer Tests")
    print("=" * 60)
    
    wm = WorldModel()
    state = State(inventory=frozenset(['ctx']))
    
    inv, p = wm.predict(state, 'get_weather')
    assert wm.last_layer == 3
    print(f"✓ L3 heuristic: get_weather → {inv - state.inventory}, p={p:.2f}")
    
    for _ in range(10):
        wm.record(Transition(frozenset(['ctx']), 'search_hotels', {},
                            frozenset(['ctx', 'hotels']), True))
    
    inv, p = wm.predict(state, 'search_hotels')
    assert wm.last_layer == 2
    print(f"✓ L2 learned: search_hotels → {inv - state.inventory}, p={p:.2f}")
    
    specs = [Spec("get_data", frozenset(["ctx"]), frozenset(["data"]))]
    wm2 = WorldModel(ActionModel(specs))
    inv, p = wm2.predict(state, 'get_data')
    assert wm2.last_layer == 1 and p == 1.0
    print(f"✓ L1 exact: get_data → {inv - state.inventory}, p={p:.2f}")
    
    print("\n✓ All layer tests passed!")

def test_agents():
    """Test agent behavior."""
    print("\n" + "=" * 60)
    print("Agent Tests")
    print("=" * 60)
    
    am, _ = make_hard_tasks()
    wm = WorldModel(am)
    
    state = State(frozenset(["mission_done"]), frozenset(["base", "fuel"]))
    
    ep = TreeSearchAgent(am, wm).solve(state, "resource")
    assert ep.success
    print(f"✓ TreeSearch: success={ep.success}, cost={ep.cost:.1f}")
    
    ep = GreedyAgent(am, wm).solve(state, "resource")
    assert not ep.success
    print(f"✓ Greedy: success={ep.success} (correctly fails)")
    
    print("\n✓ All agent tests passed!")

def benchmark_perf():
    """Performance benchmark."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    wm = WorldModel()
    state = State(inventory=frozenset(['ctx']))
    
    for _ in range(100):
        wm.record(Transition(frozenset(['ctx']), 'search', {},
                            frozenset(['ctx', 'results']), True))
    
    for name, action in [("L2", "search"), ("L3", "unknown")]:
        start = time.perf_counter()
        for _ in range(10000):
            wm.predict(state, action)
        us = (time.perf_counter() - start) * 1e6 / 10000
        print(f"{name}: {us:.1f} µs/call")

def benchmark_agents():
    """Agent comparison benchmark."""
    print("\n" + "=" * 60)
    print("Agent Comparison Benchmark")
    print("=" * 60)
    
    am, tasks = make_hard_tasks()
    wm = WorldModel(am)
    
    results = {"TreeSearch": defaultdict(lambda: {"ok": 0, "n": 0, "cost": 0}),
               "Greedy": defaultdict(lambda: {"ok": 0, "n": 0, "cost": 0})}
    
    for task in tasks:
        state = State(task["goal"], task["inv"])
        for name, Agent in [("TreeSearch", TreeSearchAgent), ("Greedy", GreedyAgent)]:
            agent = Agent(am, wm) if name == "Greedy" else Agent(am, wm, depth=6)
            ep = agent.solve(state, task["id"])
            r = results[name][task["type"]]
            r["n"] += 1
            r["cost"] += ep.cost
            if ep.success: r["ok"] += 1
    
    print(f"\n{'Type':<15} {'Agent':<12} {'SR':>8} {'AvgCost':>10}")
    print("-" * 50)
    
    for ttype in ["deceptive", "cost_optimal", "resource", "dead_end"]:
        for agent in ["TreeSearch", "Greedy"]:
            r = results[agent][ttype]
            sr = r["ok"] / max(1, r["n"])
            avg = r["cost"] / max(1, r["n"])
            print(f"{ttype:<15} {agent:<12} {sr:>7.0%} {avg:>9.1f}")
        print()
    
    print("=" * 50)
    print("SUMMARY")
    print("-" * 50)
    for agent in ["TreeSearch", "Greedy"]:
        total_ok = sum(r["ok"] for r in results[agent].values())
        total_n = sum(r["n"] for r in results[agent].values())
        total_cost = sum(r["cost"] for r in results[agent].values())
        print(f"{agent:<12}: SR={total_ok/total_n:.0%}, AvgCost={total_cost/total_n:.1f}")

# =============================================================================
# Main
# =============================================================================

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if cmd in ("test", "all"):
        test_layers()
        test_agents()
    
    if cmd in ("bench", "all"):
        benchmark_perf()
    
    if cmd in ("agents", "all"):
        benchmark_agents()
    
    if cmd not in ("test", "bench", "agents", "all"):
        print(f"Usage: python {sys.argv[0]} [test|bench|agents|all]")
        print("\nCommands:")
        print("  test   - Run layer and agent tests")
        print("  bench  - Run performance benchmark")
        print("  agents - Run agent comparison")
        print("  all    - Run everything")

if __name__ == "__main__":
    main()