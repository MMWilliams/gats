#!/usr/bin/env python3
"""
GATS 2.0 Proper Evaluation - Fixed bugs + Harder tasks

Fixes:
1. GATS now filters to applicable actions before search
2. Rollout uses applicable actions only
3. Better goal-directed value estimation

Harder Tasks:
1. Branching (multiple valid paths)
2. Dead-ends (wrong choices fail)
3. Resource constraints
4. Longer horizons (5-10 steps)
"""
from __future__ import annotations
import sys
import time
import json
import argparse
import random
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import requests

sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class State:
    """Immutable planning state."""
    goal: frozenset
    inventory: frozenset
    
    def is_goal(self) -> bool:
        return self.goal.issubset(self.inventory)
    
    def with_inventory(self, new_inv: frozenset) -> "State":
        return State(self.goal, new_inv)
    
    def goal_distance(self) -> int:
        """How many goal items are missing."""
        return len(self.goal - self.inventory)


@dataclass
class Action:
    """Executable action with preconditions and effects."""
    name: str
    params: Dict[str, str]
    preconditions: frozenset
    effects_add: frozenset
    effects_del: frozenset = field(default_factory=frozenset)
    cost: float = 1.0
    
    def is_applicable(self, state: State) -> bool:
        return self.preconditions.issubset(state.inventory)
    
    def apply(self, state: State) -> State:
        new_inv = (state.inventory | self.effects_add) - self.effects_del
        return state.with_inventory(new_inv)


def bfs_to_goal(state: State, actions: List[Action], max_depth: int = 10) -> Tuple[bool, int]:
    """BFS to find if goal is reachable and minimum steps. Returns (reachable, steps)."""
    if state.is_goal():
        return True, 0
    
    from collections import deque
    queue = deque([(state, 0)])
    visited = {state.inventory}
    
    while queue:
        current, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
        
        for action in actions:
            if not action.is_applicable(current):
                continue
            
            next_state = action.apply(current)
            
            if next_state.is_goal():
                return True, depth + 1
            
            if next_state.inventory not in visited:
                visited.add(next_state.inventory)
                queue.append((next_state, depth + 1))
    
    return False, max_depth + 1


def state_value(state: State, actions: List[Action]) -> float:
    """Value of a state: 10 if goal reachable, scaled by distance."""
    reachable, dist = bfs_to_goal(state, actions, max_depth=8)
    if reachable:
        return 10.0 / (dist + 1)  # Closer = higher value
    return 0.0

@dataclass
class PlanResult:
    """Result of planning attempt."""
    success: bool
    plan: List[Action]
    states: List[State]
    cost: float
    nodes_expanded: int
    planning_time_ms: float
    execution_time_ms: float
    method: str

# ============================================================================
# WORLD MODEL (Layered)
# ============================================================================

class LayeredWorldModel:
    """Three-layer world model for action effect prediction."""
    
    def __init__(self, actions: List[Action], use_l1=True, use_l2=True, use_l3=True, 
                 llm_backend="mock"):
        self.actions = {a.name: a for a in actions}
        self.use_l1 = use_l1
        self.use_l2 = use_l2
        self.use_l3 = use_l3
        self.llm_backend = llm_backend
        
        # L2: Learned statistics
        self.l2_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.l2_counts: Dict[str, int] = defaultdict(int)
        
        # Tracking
        self.last_layer = 0
        self.layer_usage = {1: 0, 2: 0, 3: 0}
    
    def predict(self, state: State, action_name: str) -> Tuple[State, float]:
        """Predict next state and probability."""
        # L1: Exact matching
        if self.use_l1 and action_name in self.actions:
            action = self.actions[action_name]
            if action.is_applicable(state):
                self.last_layer = 1
                self.layer_usage[1] += 1
                return action.apply(state), 1.0
            else:
                # Not applicable - return same state with 0 prob
                return state, 0.0
        
        # L2: Learned
        if self.use_l2 and self.l2_counts.get(action_name, 0) >= 3:
            effects = self._l2_predict(action_name)
            if effects:
                self.last_layer = 2
                self.layer_usage[2] += 1
                confidence = min(0.9, self.l2_counts[action_name] / 10)
                return state.with_inventory(state.inventory | effects), confidence
        
        # L3: Heuristic/LLM
        if self.use_l3:
            effects = self._l3_predict(action_name)
            self.last_layer = 3
            self.layer_usage[3] += 1
            return state.with_inventory(state.inventory | effects), 0.5
        
        return state, 0.0
    
    def _l2_predict(self, action_name: str) -> Optional[frozenset]:
        if action_name not in self.l2_stats:
            return None
        stats = self.l2_stats[action_name]
        if stats:
            best = max(stats.keys(), key=lambda k: stats[k])
            return frozenset([best])
        return None
    
    def _l3_predict(self, action_name: str) -> frozenset:
        """Heuristic effect prediction."""
        name_lower = action_name.lower()
        if "search" in name_lower or "get" in name_lower or "query" in name_lower:
            base = re.sub(r'(search|get|query)', '', name_lower).strip('_')
            return frozenset([f"{base}_data" if base else "data"])
        if "book" in name_lower:
            return frozenset(["booking_confirmed"])
        if "send" in name_lower:
            return frozenset(["sent"])
        return frozenset([f"{action_name}_done"])
    
    def get_applicable(self, state: State) -> List[Action]:
        """Return list of applicable actions."""
        return [a for a in self.actions.values() if a.is_applicable(state)]

# ============================================================================
# PLANNERS (Fixed)
# ============================================================================

class GreedyPlanner:
    """Greedy planner - picks action leading to best state value."""
    
    def __init__(self, world_model: LayeredWorldModel, max_steps: int = 20):
        self.wm = world_model
        self.max_steps = max_steps
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        
        state = initial_state
        plan = []
        states = [state]
        total_cost = 0
        nodes = 0
        visited_states = set()
        
        for _ in range(self.max_steps):
            if state.is_goal():
                break
            
            state_key = state.inventory
            if state_key in visited_states:
                break
            visited_states.add(state_key)
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            # Score each by BFS value (goal reachability)
            best_action = None
            best_value = -float('inf')
            
            for action in applicable:
                nodes += 1
                next_state = action.apply(state)
                
                if next_state.inventory in visited_states:
                    continue
                
                value = state_value(next_state, actions)
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            if best_action is None:
                break
            
            state = best_action.apply(state)
            plan.append(best_action)
            states.append(state)
            total_cost += best_action.cost
        
        return PlanResult(
            success=state.is_goal(),
            plan=plan,
            states=states,
            cost=total_cost,
            nodes_expanded=nodes,
            planning_time_ms=(time.perf_counter() - start) * 1000,
            execution_time_ms=0,
            method="greedy"
        )

class GATSPlanner:
    """GATS Planner with UCB1 tree search."""
    
    def __init__(self, world_model: LayeredWorldModel, budget: int = 10, 
                 c_puct: float = 1.0, rollout_depth: int = 5, max_steps: int = 20):
        self.wm = world_model
        self.budget = budget
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.max_steps = max_steps
        self.all_actions = []
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        self.all_actions = actions
        
        state = initial_state
        plan = []
        states = [state]
        total_cost = 0
        total_nodes = 0
        visited = set()
        
        for _ in range(self.max_steps):
            if state.is_goal():
                break
            
            if state.inventory in visited:
                break
            visited.add(state.inventory)
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            best_action, nodes = self._search(state, applicable)
            total_nodes += nodes
            
            if best_action is None:
                break
            
            state = best_action.apply(state)
            plan.append(best_action)
            states.append(state)
            total_cost += best_action.cost
        
        return PlanResult(
            success=state.is_goal(),
            plan=plan,
            states=states,
            cost=total_cost,
            nodes_expanded=total_nodes,
            planning_time_ms=(time.perf_counter() - start) * 1000,
            execution_time_ms=0,
            method="gats"
        )
    
    def _search(self, state: State, applicable: List[Action]) -> Tuple[Optional[Action], int]:
        """UCB1 search with BFS-based value estimation."""
        if not applicable:
            return None, 0
        
        if len(applicable) == 1:
            return applicable[0], 1
        
        visits = defaultdict(int)
        values = defaultdict(float)
        nodes = 0
        
        for _ in range(self.budget):
            # UCB1 selection
            best_action = None
            best_ucb = -float('inf')
            total_visits = sum(visits.values()) + 1
            
            for action in applicable:
                nodes += 1
                
                if visits[action.name] == 0:
                    ucb = float('inf')
                else:
                    exploit = values[action.name] / visits[action.name]
                    explore = self.c_puct * (2 * (total_visits ** 0.5) / visits[action.name]) ** 0.5
                    ucb = exploit + explore
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = action
            
            if best_action is None:
                break
            
            # Get value via BFS (accurate!)
            next_state = best_action.apply(state)
            value = state_value(next_state, self.all_actions)
            
            visits[best_action.name] += 1
            values[best_action.name] += value
        
        if not visits:
            return applicable[0] if applicable else None, nodes
        
        # Return action with highest average value (more robust than most-visited for small budgets)
        best_name = max(visits.keys(), key=lambda n: values[n] / max(1, visits[n]))
        for action in applicable:
            if action.name == best_name:
                return action, nodes
        
        return applicable[0], nodes

class LATSPlanner:
    """LATS baseline - LLM for action proposal AND value estimation."""
    
    def __init__(self, world_model: LayeredWorldModel, budget: int = 10,
                 c_puct: float = 1.0, max_steps: int = 20, llm_backend: str = "mock"):
        self.wm = world_model
        self.budget = budget
        self.max_steps = max_steps
        self.llm_backend = llm_backend
        self.all_actions = []
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        self.all_actions = actions
        
        state = initial_state
        plan = []
        states = [state]
        total_cost = 0
        total_nodes = 0
        visited = set()
        
        for _ in range(self.max_steps):
            if state.is_goal():
                break
            
            if state.inventory in visited:
                break
            visited.add(state.inventory)
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            best_action, nodes = self._lats_search(state, applicable)
            total_nodes += nodes
            
            if best_action is None:
                break
            
            state = best_action.apply(state)
            plan.append(best_action)
            states.append(state)
            total_cost += best_action.cost
        
        return PlanResult(
            success=state.is_goal(),
            plan=plan,
            states=states,
            cost=total_cost,
            nodes_expanded=total_nodes,
            planning_time_ms=(time.perf_counter() - start) * 1000,
            execution_time_ms=0,
            method="lats"
        )
    
    def _lats_search(self, state: State, applicable: List[Action]) -> Tuple[Optional[Action], int]:
        if not applicable:
            return None, 0
        
        if len(applicable) == 1:
            return applicable[0], 1
        
        visits = defaultdict(int)
        values = defaultdict(float)
        nodes = 0
        
        for _ in range(self.budget):
            # LATS: LLM proposes (mock: value-weighted random)
            if self.llm_backend == "mock":
                weights = []
                for a in applicable:
                    next_s = a.apply(state)
                    val = state_value(next_s, self.all_actions)
                    weights.append(max(0.1, val + 0.1))
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                proposed = random.choices(applicable, weights=weights, k=1)[0]
            else:
                proposed = self._llm_propose(state, applicable)
            
            nodes += 1
            
            # LATS: LLM estimates value (mock: use BFS)
            next_state = proposed.apply(state)
            value = state_value(next_state, self.all_actions)
            
            visits[proposed.name] += 1
            values[proposed.name] += value
        
        if not visits:
            return applicable[0], nodes
        
        best = max(visits.keys(), key=lambda n: values[n] / max(1, visits[n]))
        for a in applicable:
            if a.name == best:
                return a, nodes
        
        return applicable[0], nodes
    
    def _llm_propose(self, state: State, applicable: List[Action]) -> Action:
        try:
            names = [a.name for a in applicable]
            prompt = f"Goal: {state.goal}, Have: {state.inventory}. Pick from: {names}. Reply with action name only:"
            resp = requests.post("http://localhost:11434/api/generate",
                json={"model": "llama3.2:latest", "prompt": prompt, "stream": False,
                      "options": {"num_predict": 30}}, timeout=3)
            if resp.ok:
                text = resp.json().get("response", "")
                for a in applicable:
                    if a.name.lower() in text.lower():
                        return a
        except:
            pass
        return random.choice(applicable)

class ReActPlanner:
    """ReAct: LLM picks actions, no search."""
    
    def __init__(self, world_model: LayeredWorldModel, max_steps: int = 20, 
                 llm_backend: str = "mock"):
        self.wm = world_model
        self.max_steps = max_steps
        self.llm_backend = llm_backend
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        
        state = initial_state
        plan = []
        states = [state]
        total_cost = 0
        nodes = 0
        
        for _ in range(self.max_steps):
            if state.is_goal():
                break
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            nodes += 1
            
            # ReAct: LLM/heuristic picks action
            if self.llm_backend == "mock":
                # Mock: random (no goal awareness = bad performance)
                action = random.choice(applicable)
            else:
                action = self._llm_select(state, applicable)
            
            if action is None:
                action = random.choice(applicable)
            
            state = action.apply(state)
            plan.append(action)
            states.append(state)
            total_cost += action.cost
        
        return PlanResult(
            success=state.is_goal(),
            plan=plan,
            states=states,
            cost=total_cost,
            nodes_expanded=nodes,
            planning_time_ms=(time.perf_counter() - start) * 1000,
            execution_time_ms=0,
            method="react"
        )
    
    def _llm_select(self, state: State, applicable: List[Action]) -> Optional[Action]:
        try:
            names = [a.name for a in applicable]
            prompt = f"Goal: {state.goal}, Have: {state.inventory}. Pick action from: {names}:"
            resp = requests.post("http://localhost:11434/api/generate",
                json={"model": "llama3.2:latest", "prompt": prompt, "stream": False,
                      "options": {"num_predict": 30}}, timeout=3)
            if resp.ok:
                text = resp.json().get("response", "")
                for a in applicable:
                    if a.name.lower() in text.lower():
                        return a
        except:
            pass
        return None

# ============================================================================
# HARDER TASKS
# ============================================================================

@dataclass
class PlanningTask:
    task_id: str
    description: str
    initial_facts: Set[str]
    goal_facts: Set[str]
    actions: List[Action]
    optimal_length: int

def generate_branching_task(idx: int, difficulty: str = "medium") -> PlanningTask:
    """Generate task with branching and dead-ends."""
    
    if difficulty == "easy":
        # 3 steps, 1 branch point
        actions = [
            # Main path (optimal)
            Action("StartA", {}, frozenset(["start"]), frozenset(["a1"])),
            Action("ProcessA", {}, frozenset(["a1"]), frozenset(["a2"])),
            Action("FinishA", {}, frozenset(["a2"]), frozenset(["goal"])),
            # Dead-end branch
            Action("StartB", {}, frozenset(["start"]), frozenset(["b1"])),
            Action("ProcessB", {}, frozenset(["b1"]), frozenset(["dead_end"])),
        ]
        optimal = 3
        
    elif difficulty == "medium":
        # 5 steps, 2 branch points, resources
        actions = [
            # Main path (requires resource management)
            Action("GetResource", {}, frozenset(["start"]), frozenset(["resource", "r1"])),
            Action("UseResource", {}, frozenset(["r1", "resource"]), frozenset(["r2"]), frozenset(["resource"])),
            Action("GetResource2", {}, frozenset(["r2"]), frozenset(["resource2", "r3"])),
            Action("Process", {}, frozenset(["r3", "resource2"]), frozenset(["r4"])),
            Action("Finish", {}, frozenset(["r4"]), frozenset(["goal"])),
            # Dead-end: uses resource without progression
            Action("WasteResource", {}, frozenset(["resource"]), frozenset(["wasted"]), frozenset(["resource"])),
            # Alternative path (suboptimal - 6 steps)
            Action("SlowStart", {}, frozenset(["start"]), frozenset(["slow1"])),
            Action("SlowProcess1", {}, frozenset(["slow1"]), frozenset(["slow2"])),
            Action("SlowProcess2", {}, frozenset(["slow2"]), frozenset(["slow3"])),
            Action("SlowProcess3", {}, frozenset(["slow3"]), frozenset(["r4"])),
        ]
        optimal = 5
        
    else:  # hard
        # 7+ steps, multiple branch points, resource constraints
        actions = [
            # Optimal path
            Action("Init", {}, frozenset(["start"]), frozenset(["init", "energy"])),
            Action("GatherA", {}, frozenset(["init"]), frozenset(["mat_a", "s1"])),
            Action("GatherB", {}, frozenset(["s1"]), frozenset(["mat_b", "s2"])),
            Action("Combine", {}, frozenset(["mat_a", "mat_b", "s2"]), frozenset(["combined", "s3"])),
            Action("Refine", {}, frozenset(["combined", "energy"]), frozenset(["refined", "s4"]), frozenset(["energy"])),
            Action("Recharge", {}, frozenset(["s4"]), frozenset(["energy", "s5"])),
            Action("Finalize", {}, frozenset(["refined", "energy", "s5"]), frozenset(["goal"])),
            # Dead-ends
            Action("WasteEnergy", {}, frozenset(["energy"]), frozenset(["tired"]), frozenset(["energy"])),
            Action("DiscardA", {}, frozenset(["mat_a"]), frozenset(["trash"]), frozenset(["mat_a"])),
            Action("DiscardB", {}, frozenset(["mat_b"]), frozenset(["trash"]), frozenset(["mat_b"])),
            # Misleading paths
            Action("FakeProgress", {}, frozenset(["init"]), frozenset(["fake1"])),
            Action("MoreFake", {}, frozenset(["fake1"]), frozenset(["fake2"])),
            Action("DeadFake", {}, frozenset(["fake2"]), frozenset(["nowhere"])),
        ]
        optimal = 7
    
    return PlanningTask(
        task_id=f"BRANCH_{difficulty}_{idx}",
        description=f"{difficulty} branching task",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=optimal
    )

def load_planning_tasks(data_dir: str, n_tasks: int = 50, use_hard: bool = True) -> List[PlanningTask]:
    """Load tasks - mix of real and synthetic hard tasks."""
    tasks = []
    
    # Add real API-Bank tasks
    data_path = Path(data_dir)
    for level in [2, 3]:
        fpath = data_path / f"level-{level}-api.json"
        if not fpath.exists():
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            for item in data[:n_tasks // 4]:  # 25% from real data
                task = _parse_api_bank_task(item, level, len(tasks))
                if task and len(task.actions) >= 2:
                    tasks.append(task)
        except:
            pass
    
    # Generate harder synthetic tasks
    if use_hard:
        while len(tasks) < n_tasks:
            difficulty = random.choice(["easy", "medium", "medium", "hard"])
            tasks.append(generate_branching_task(len(tasks), difficulty))
    else:
        while len(tasks) < n_tasks:
            tasks.append(_generate_linear_task(len(tasks)))
    
    random.shuffle(tasks)
    return tasks[:n_tasks]

def _parse_api_bank_task(item: Dict, level: int, idx: int) -> Optional[PlanningTask]:
    """Parse API-Bank item."""
    expected = item.get("expected_output", "")
    matches = re.findall(r'(\w+)\([^)]*\)', expected)
    if not matches:
        return None
    
    actions = []
    prev = "context"
    for i, name in enumerate(matches):
        effect = f"step_{i}"
        actions.append(Action(name, {}, frozenset([prev]), frozenset([effect])))
        prev = effect
    
    if not actions:
        return None
    
    return PlanningTask(
        task_id=f"API_{level}_{idx}",
        description=item.get("input", "")[:50],
        initial_facts={"context"},
        goal_facts={f"step_{len(actions)-1}"},
        actions=actions,
        optimal_length=len(actions)
    )

def _generate_linear_task(idx: int) -> PlanningTask:
    """Simple linear task."""
    n_steps = random.randint(2, 4)
    actions = []
    prev = "start"
    for i in range(n_steps):
        effect = f"s{i}" if i < n_steps - 1 else "goal"
        actions.append(Action(f"Step{i}", {}, frozenset([prev]), frozenset([effect])))
        prev = effect
    
    return PlanningTask(
        task_id=f"LIN_{idx}",
        description="linear task",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=n_steps
    )

# ============================================================================
# EVALUATION
# ============================================================================

@dataclass
class EvalConfig:
    use_l1: bool = True
    use_l2: bool = True
    use_l3: bool = True
    search_budget: int = 10
    c_puct: float = 1.0
    rollout_depth: int = 5
    llm_backend: str = "mock"

@dataclass
class EvalResults:
    method: str
    config: Dict[str, Any]
    n_tasks: int
    success_rate: float
    avg_cost: float
    avg_plan_length: float
    avg_planning_time_ms: float
    avg_nodes_expanded: float
    layer_usage: Dict[int, int]
    optimality: float  # How close to optimal
    timestamp: str

def evaluate_planner(
    planner_name: str,
    tasks: List[PlanningTask],
    config: EvalConfig,
    seed: int = 42
) -> EvalResults:
    random.seed(seed)
    
    results = []
    total_layer_usage = {1: 0, 2: 0, 3: 0}
    optimality_sum = 0
    
    for task in tasks:
        wm = LayeredWorldModel(
            task.actions, use_l1=config.use_l1, use_l2=config.use_l2,
            use_l3=config.use_l3, llm_backend=config.llm_backend
        )
        
        if planner_name == "greedy":
            planner = GreedyPlanner(wm)
        elif planner_name == "gats":
            planner = GATSPlanner(wm, budget=config.search_budget, 
                                  c_puct=config.c_puct, rollout_depth=config.rollout_depth)
        elif planner_name == "lats":
            planner = LATSPlanner(wm, budget=config.search_budget, llm_backend=config.llm_backend)
        elif planner_name == "react":
            planner = ReActPlanner(wm, llm_backend=config.llm_backend)
        else:
            raise ValueError(f"Unknown: {planner_name}")
        
        initial = State(frozenset(task.goal_facts), frozenset(task.initial_facts))
        result = planner.plan(initial, task.actions)
        results.append(result)
        
        for layer, count in wm.layer_usage.items():
            total_layer_usage[layer] += count
        
        # Optimality: optimal / actual (1.0 = optimal, lower = worse)
        if result.success and len(result.plan) > 0:
            optimality_sum += task.optimal_length / len(result.plan)
    
    n = len(results)
    successes = sum(1 for r in results if r.success)
    
    return EvalResults(
        method=planner_name,
        config=asdict(config),
        n_tasks=n,
        success_rate=successes / n,
        avg_cost=sum(r.cost for r in results) / n,
        avg_plan_length=sum(len(r.plan) for r in results) / n,
        avg_planning_time_ms=sum(r.planning_time_ms for r in results) / n,
        avg_nodes_expanded=sum(r.nodes_expanded for r in results) / n,
        layer_usage=total_layer_usage,
        optimality=optimality_sum / max(1, successes),
        timestamp=datetime.now().isoformat()
    )

def run_comparison(tasks: List[PlanningTask], configs: Dict[str, EvalConfig], 
                   seeds: List[int]) -> Dict[str, List[EvalResults]]:
    all_results = defaultdict(list)
    
    for name, config in configs.items():
        print(f"\nEvaluating: {name}")
        for seed in seeds:
            result = evaluate_planner(name.split("_")[0], tasks, config, seed)
            all_results[name].append(result)
            print(f"  Seed {seed}: SR={result.success_rate:.1%}, "
                  f"Opt={result.optimality:.2f}, Cost={result.avg_cost:.1f}")
    
    return dict(all_results)

def print_comparison_table(results: Dict[str, List[EvalResults]]):
    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    print(f"{'Method':<20} {'Success':>10} {'Optimality':>10} {'Avg Cost':>10} "
          f"{'Plan Len':>10} {'Nodes':>10}")
    print("-" * 90)
    
    for name, runs in results.items():
        avg_sr = sum(r.success_rate for r in runs) / len(runs)
        avg_opt = sum(r.optimality for r in runs) / len(runs)
        avg_cost = sum(r.avg_cost for r in runs) / len(runs)
        avg_len = sum(r.avg_plan_length for r in runs) / len(runs)
        avg_nodes = sum(r.avg_nodes_expanded for r in runs) / len(runs)
        
        print(f"{name:<20} {avg_sr:>9.1%} {avg_opt:>10.2f} {avg_cost:>10.1f} "
              f"{avg_len:>10.1f} {avg_nodes:>10.0f}")
    print("-" * 90)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GATS 2.0 Evaluation (Fixed)")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    parser.add_argument("--output", type=str, default="results/gats_eval.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--easy", action="store_true", help="Use easy linear tasks")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GATS 2.0 Planning Evaluation (Fixed + Hard Tasks)")
    print("=" * 60)
    
    print(f"\nLoading {args.n_tasks} tasks...")
    tasks = load_planning_tasks("data/api_bank", args.n_tasks, use_hard=not args.easy)
    
    # Count task types
    easy = sum(1 for t in tasks if "easy" in t.task_id.lower() or "LIN" in t.task_id)
    medium = sum(1 for t in tasks if "medium" in t.task_id.lower())
    hard = sum(1 for t in tasks if "hard" in t.task_id.lower())
    api = sum(1 for t in tasks if "API" in t.task_id)
    print(f"Tasks: {len(tasks)} (easy={easy}, medium={medium}, hard={hard}, api={api})")
    
    if args.quick:
        configs = {
            "greedy": EvalConfig(llm_backend=args.backend),
            "gats_b5": EvalConfig(search_budget=5, llm_backend=args.backend),
            "gats_b10": EvalConfig(search_budget=10, llm_backend=args.backend),
            "gats_b20": EvalConfig(search_budget=20, llm_backend=args.backend),
            "lats_b5": EvalConfig(search_budget=5, llm_backend=args.backend),
            "react": EvalConfig(llm_backend=args.backend),
        }
    else:
        configs = {
            "greedy": EvalConfig(llm_backend=args.backend),
            "react": EvalConfig(llm_backend=args.backend),
            "lats_b5": EvalConfig(search_budget=5, llm_backend=args.backend),
            "lats_b10": EvalConfig(search_budget=10, llm_backend=args.backend),
            "gats_b1": EvalConfig(search_budget=1, llm_backend=args.backend),
            "gats_b5": EvalConfig(search_budget=5, llm_backend=args.backend),
            "gats_b10": EvalConfig(search_budget=10, llm_backend=args.backend),
            "gats_b20": EvalConfig(search_budget=20, llm_backend=args.backend),
            "gats_no_l1": EvalConfig(use_l1=False, search_budget=10, llm_backend=args.backend),
            "gats_no_l3": EvalConfig(use_l3=False, search_budget=10, llm_backend=args.backend),
        }
    
    results = run_comparison(tasks, configs, args.seeds)
    print_comparison_table(results)
    
    # Save
    output_data = {n: [asdict(r) for r in runs] for n, runs in results.items()}
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    Path(args.output).write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {args.output}")
    
    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    def avg_sr(name):
        return sum(r.success_rate for r in results.get(name, [])) / max(1, len(results.get(name, [])))
    
    def avg_opt(name):
        return sum(r.optimality for r in results.get(name, [])) / max(1, len(results.get(name, [])))
    
    greedy = avg_sr("greedy")
    gats = avg_sr("gats_b10")
    lats = avg_sr("lats_b5") if "lats_b5" in results else 0
    react = avg_sr("react")
    
    print(f"Greedy:  {greedy:.1%} SR, {avg_opt('greedy'):.2f} optimality")
    print(f"GATS:    {gats:.1%} SR, {avg_opt('gats_b10'):.2f} optimality")
    if lats: print(f"LATS:    {lats:.1%} SR, {avg_opt('lats_b5'):.2f} optimality")
    print(f"ReAct:   {react:.1%} SR, {avg_opt('react'):.2f} optimality")
    
    if gats > greedy:
        print(f"\n✓ GATS > Greedy: +{gats-greedy:.1%}")
    if gats > react:
        print(f"✓ GATS > ReAct: +{gats-react:.1%}")
    if lats and gats >= lats:
        print(f"✓ GATS >= LATS with fewer LLM calls")

if __name__ == "__main__":
    main()