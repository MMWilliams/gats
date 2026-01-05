#!/usr/bin/env python3
"""
GATS 2.0 Evaluation with Real API-Bank and ToolBench Tasks

This version properly loads and evaluates on:
1. API-Bank Level 1/2/3 (multi-step API calling)
2. ToolBench (if available)
3. Synthetic tasks (as fallback/additional)
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
from collections import defaultdict, deque
import requests

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
    """BFS to find if goal is reachable and minimum steps."""
    if state.is_goal():
        return True, 0
    
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
    """Value of a state based on goal reachability."""
    reachable, dist = bfs_to_goal(state, actions, max_depth=8)
    if reachable:
        return 10.0 / (dist + 1)
    return 0.0


@dataclass
class PlanResult:
    success: bool
    plan: List[Action]
    states: List[State]
    cost: float
    nodes_expanded: int
    planning_time_ms: float
    execution_time_ms: float
    method: str


@dataclass
class PlanningTask:
    task_id: str
    description: str
    initial_facts: Set[str]
    goal_facts: Set[str]
    actions: List[Action]
    optimal_length: int
    source: str = "synthetic"  # "api_bank", "toolbench", or "synthetic"

# ============================================================================
# API-BANK LOADER
# ============================================================================

def load_api_bank_tasks(data_dir: str, max_tasks: int = 100) -> List[PlanningTask]:
    """Load real API-Bank tasks from Level 1, 2, and 3."""
    tasks = []
    data_path = Path(data_dir)
    
    # Load from each level proportionally
    tasks_per_level = max(10, max_tasks // 3)
    
    for level in [1, 2, 3]:
        fpath = data_path / f"level-{level}-api.json"
        if not fpath.exists():
            print(f"  Level {level}: not found at {fpath}")
            continue
        
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            level_tasks = 0
            
            for idx, item in enumerate(data):
                if level_tasks >= tasks_per_level:
                    break
                
                task = parse_api_bank_item(item, level, idx)
                if task:
                    tasks.append(task)
                    level_tasks += 1
            
            print(f"  Level {level}: loaded {level_tasks} tasks from {len(data)} items")
            
        except Exception as e:
            print(f"  Level {level}: error - {e}")
    
    return tasks


def parse_api_bank_item(item: Dict, level: int, idx: int) -> Optional[PlanningTask]:
    """Parse API-Bank item into planning task."""
    
    # PRIORITY: Use api_sequence if present (constructed L3 tasks)
    if "api_sequence" in item and isinstance(item["api_sequence"], list):
        api_names = item["api_sequence"]
        if not api_names:
            return None
        
        actions = []
        for i, api_name in enumerate(api_names):
            prec = frozenset(["start"]) if i == 0 else frozenset([f"step_{i-1}_done"])
            effect = frozenset([f"step_{i}_done"])
            actions.append(Action(
                name=api_name,
                params={},
                preconditions=prec,
                effects_add=effect,
                cost=1.0
            ))
        
        # Add distractors
        distractors = generate_distractors(set(api_names), len(api_names))
        actions.extend(distractors)
        
        n_steps = len(api_names)
        return PlanningTask(
            task_id=f"API_L{level}_{idx}",
            description=item.get("instruction", item.get("input", ""))[:100],
            initial_facts={"start"},
            goal_facts={f"step_{n_steps-1}_done"},
            actions=actions,
            optimal_length=n_steps,
            source=f"api_bank_L{level}_multi" if n_steps > 1 else f"api_bank_L{level}"
        )
    
    # FALLBACK: Parse from expected_output
    expected = ""
    for key in ["expected_output", "output", "answer", "api_call"]:
        if key in item and item[key]:
            expected = str(item[key])
            break
    
    if not expected:
        return None
    
    # Extract API calls
    api_pattern = r'\[?(\w+)\(([^)]*)\)\]?'
    matches = re.findall(api_pattern, expected)
    
    if not matches:
        return None
    
    # Build action sequence
    actions = []
    seen_apis = set()
    
    for i, (api_name, params_str) in enumerate(matches):
        params = {}
        for pm in re.finditer(r"(\w+)\s*=\s*['\"]?([^,'\")]+)['\"]?", params_str):
            params[pm.group(1)] = pm.group(2)
        
        prec = frozenset(["start"]) if i == 0 else frozenset([f"step_{i-1}_done"])
        effect = frozenset([f"step_{i}_done"])
        
        actions.append(Action(
            name=api_name,
            params=params,
            preconditions=prec,
            effects_add=effect,
            cost=1.0
        ))
        seen_apis.add(api_name)
    
    if not actions:
        return None
    
    # Add distractors
    distractors = generate_distractors(seen_apis, len(actions))
    actions.extend(distractors)
    
    n_steps = len(matches)
    return PlanningTask(
        task_id=f"API_L{level}_{idx}",
        description=item.get("input", item.get("instruction", ""))[:100],
        initial_facts={"start"},
        goal_facts={f"step_{n_steps-1}_done"},
        actions=actions,
        optimal_length=n_steps,
        source=f"api_bank_L{level}_multi" if n_steps > 1 else f"api_bank_L{level}"
    )


def generate_distractors(existing_apis: Set[str], n_steps: int) -> List[Action]:
    """Generate distractor actions that don't lead to goal."""
    distractors = []
    
    # Common API patterns that might seem relevant but aren't
    distractor_apis = [
        "GetUserInfo", "CheckStatus", "ValidateInput", "LogEvent",
        "CacheResult", "RetryOperation", "SendNotification", "UpdateRecord"
    ]
    
    for i, api in enumerate(distractor_apis[:min(3, n_steps)]):
        if api not in existing_apis:
            # Distractor can be applied from start but leads nowhere
            distractors.append(Action(
                name=f"Distractor_{api}",
                params={},
                preconditions=frozenset(["start"]),
                effects_add=frozenset([f"distractor_{i}_done"]),
                cost=1.0
            ))
    
    return distractors


# ============================================================================
# TOOLBENCH LOADER
# ============================================================================

def load_toolbench_tasks(data_dir: str, max_tasks: int = 50) -> List[PlanningTask]:
    """Load ToolBench tasks if available."""
    tasks = []
    data_path = Path(data_dir)
    
    # Check for ToolBench data
    for fname in ["toolbench.json", "toolbench_test.json", "data.json"]:
        fpath = data_path / fname
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                for idx, item in enumerate(data[:max_tasks]):
                    task = parse_toolbench_item(item, idx)
                    if task:
                        tasks.append(task)
                print(f"  ToolBench: loaded {len(tasks)} tasks from {fname}")
                break
            except Exception as e:
                print(f"  ToolBench: error loading {fname} - {e}")
    
    if not tasks:
        print("  ToolBench: not found")
    
    return tasks


def parse_toolbench_item(item: Dict, idx: int) -> Optional[PlanningTask]:
    """Parse ToolBench item into planning task."""
    # ToolBench format varies, try common patterns
    tools = item.get("tools", item.get("api_list", []))
    if not tools:
        return None
    
    actions = []
    for i, tool in enumerate(tools):
        if isinstance(tool, str):
            name = tool
            params = {}
        elif isinstance(tool, dict):
            name = tool.get("name", tool.get("api_name", f"Tool_{i}"))
            params = tool.get("parameters", {})
        else:
            continue
        
        prec = frozenset(["start"]) if i == 0 else frozenset([f"step_{i-1}_done"])
        effect = frozenset([f"step_{i}_done"])
        
        actions.append(Action(
            name=name,
            params=params if isinstance(params, dict) else {},
            preconditions=prec,
            effects_add=effect,
            cost=1.0
        ))
    
    if not actions:
        return None
    
    # Add distractors
    actions.extend(generate_distractors(set(a.name for a in actions), len(actions)))
    
    return PlanningTask(
        task_id=f"TOOL_{idx}",
        description=item.get("query", item.get("instruction", ""))[:100],
        initial_facts={"start"},
        goal_facts={f"step_{len(tools)-1}_done"},
        actions=actions,
        optimal_length=len(tools),
        source="toolbench"
    )


# ============================================================================
# SYNTHETIC TASK GENERATOR
# ============================================================================

def generate_synthetic_tasks(n_tasks: int, seed: int = 42) -> List[PlanningTask]:
    """Generate synthetic multi-step planning tasks."""
    random.seed(seed)
    tasks = []
    
    for i in range(n_tasks):
        difficulty = random.choice(["easy", "easy", "medium", "medium", "medium", "hard"])
        tasks.append(generate_branching_task(i, difficulty))
    
    return tasks


def generate_branching_task(idx: int, difficulty: str = "medium") -> PlanningTask:
    """Generate task with branching and dead-ends."""
    
    if difficulty == "easy":
        actions = [
            Action("StartA", {}, frozenset(["start"]), frozenset(["a1"])),
            Action("ProcessA", {}, frozenset(["a1"]), frozenset(["a2"])),
            Action("FinishA", {}, frozenset(["a2"]), frozenset(["goal"])),
            Action("StartB", {}, frozenset(["start"]), frozenset(["b1"])),
            Action("ProcessB", {}, frozenset(["b1"]), frozenset(["dead_end"])),
        ]
        optimal = 3
        
    elif difficulty == "medium":
        actions = [
            Action("GetResource", {}, frozenset(["start"]), frozenset(["resource", "r1"])),
            Action("UseResource", {}, frozenset(["r1", "resource"]), frozenset(["r2"]), frozenset(["resource"])),
            Action("GetResource2", {}, frozenset(["r2"]), frozenset(["resource2", "r3"])),
            Action("Process", {}, frozenset(["r3", "resource2"]), frozenset(["r4"])),
            Action("Finish", {}, frozenset(["r4"]), frozenset(["goal"])),
            Action("WasteResource", {}, frozenset(["resource"]), frozenset(["wasted"]), frozenset(["resource"])),
            Action("SlowStart", {}, frozenset(["start"]), frozenset(["slow1"])),
            Action("SlowProcess1", {}, frozenset(["slow1"]), frozenset(["slow2"])),
            Action("SlowProcess2", {}, frozenset(["slow2"]), frozenset(["slow3"])),
            Action("SlowProcess3", {}, frozenset(["slow3"]), frozenset(["r4"])),
        ]
        optimal = 5
        
    else:  # hard
        actions = [
            Action("Init", {}, frozenset(["start"]), frozenset(["init", "energy"])),
            Action("GatherA", {}, frozenset(["init"]), frozenset(["mat_a", "s1"])),
            Action("GatherB", {}, frozenset(["s1"]), frozenset(["mat_b", "s2"])),
            Action("Combine", {}, frozenset(["mat_a", "mat_b", "s2"]), frozenset(["combined", "s3"])),
            Action("Refine", {}, frozenset(["combined", "energy"]), frozenset(["refined", "s4"]), frozenset(["energy"])),
            Action("Recharge", {}, frozenset(["s4"]), frozenset(["energy", "s5"])),
            Action("Finalize", {}, frozenset(["refined", "energy", "s5"]), frozenset(["goal"])),
            Action("WasteEnergy", {}, frozenset(["energy"]), frozenset(["tired"]), frozenset(["energy"])),
            Action("DiscardA", {}, frozenset(["mat_a"]), frozenset(["trash"]), frozenset(["mat_a"])),
            Action("DiscardB", {}, frozenset(["mat_b"]), frozenset(["trash"]), frozenset(["mat_b"])),
            Action("FakeProgress", {}, frozenset(["init"]), frozenset(["fake1"])),
            Action("MoreFake", {}, frozenset(["fake1"]), frozenset(["fake2"])),
            Action("DeadFake", {}, frozenset(["fake2"]), frozenset(["nowhere"])),
        ]
        optimal = 7
    
    return PlanningTask(
        task_id=f"SYN_{difficulty}_{idx}",
        description=f"{difficulty} branching task",
        initial_facts={"start"},
        goal_facts={"goal"},
        actions=actions,
        optimal_length=optimal,
        source="synthetic"
    )


# ============================================================================
# COMBINED LOADER
# ============================================================================

def load_all_tasks(data_dir: str, n_tasks: int = 100, include_synthetic: bool = True) -> List[PlanningTask]:
    """Load tasks from all sources."""
    print(f"\nLoading tasks from {data_dir}...")
    
    all_tasks = []
    
    # Load API-Bank
    api_bank_tasks = load_api_bank_tasks(Path(data_dir) / "api_bank", max_tasks=n_tasks // 2)
    all_tasks.extend(api_bank_tasks)
    
    # Load ToolBench
    toolbench_tasks = load_toolbench_tasks(Path(data_dir) / "toolbench", max_tasks=n_tasks // 4)
    all_tasks.extend(toolbench_tasks)
    
    # Fill with synthetic if needed
    if include_synthetic and len(all_tasks) < n_tasks:
        n_synthetic = n_tasks - len(all_tasks)
        print(f"  Synthetic: generating {n_synthetic} tasks")
        synthetic_tasks = generate_synthetic_tasks(n_synthetic)
        all_tasks.extend(synthetic_tasks)
    
    # Shuffle
    random.shuffle(all_tasks)
    
    # Count by source
    sources = defaultdict(int)
    for t in all_tasks:
        sources[t.source] += 1
    
    print(f"\nTotal: {len(all_tasks)} tasks")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    
    return all_tasks[:n_tasks]


# ============================================================================
# WORLD MODEL
# ============================================================================

class LayeredWorldModel:
    """Three-layer world model."""
    
    def __init__(self, actions: List[Action], use_l1=True, use_l2=True, use_l3=True, 
                 llm_backend="mock"):
        self.actions = {a.name: a for a in actions}
        self.use_l1 = use_l1
        self.use_l2 = use_l2
        self.use_l3 = use_l3
        self.llm_backend = llm_backend
        self.l2_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.l2_counts: Dict[str, int] = defaultdict(int)
        self.layer_usage = {1: 0, 2: 0, 3: 0}
    
    def predict(self, state: State, action_name: str) -> Tuple[State, float]:
        if self.use_l1 and action_name in self.actions:
            action = self.actions[action_name]
            if action.is_applicable(state):
                self.layer_usage[1] += 1
                return action.apply(state), 1.0
            return state, 0.0
        
        if self.use_l2 and self.l2_counts.get(action_name, 0) >= 3:
            stats = self.l2_stats.get(action_name, {})
            if stats:
                best = max(stats.keys(), key=lambda k: stats[k])
                self.layer_usage[2] += 1
                return state.with_inventory(state.inventory | frozenset([best])), 0.8
        
        if self.use_l3:
            self.layer_usage[3] += 1
            return state.with_inventory(state.inventory | frozenset([f"{action_name}_done"])), 0.5
        
        return state, 0.0
    
    def get_applicable(self, state: State) -> List[Action]:
        return [a for a in self.actions.values() if a.is_applicable(state)]


# ============================================================================
# PLANNERS
# ============================================================================

class GreedyPlanner:
    def __init__(self, world_model: LayeredWorldModel, max_steps: int = 20):
        self.wm = world_model
        self.max_steps = max_steps
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        state, plan, states, cost, nodes, visited = initial_state, [], [initial_state], 0, 0, set()
        
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in visited:
                break
            visited.add(state.inventory)
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            best_action, best_value = None, -float('inf')
            for action in applicable:
                nodes += 1
                next_state = action.apply(state)
                if next_state.inventory not in visited:
                    value = state_value(next_state, actions)
                    if value > best_value:
                        best_value, best_action = value, action
            
            if best_action is None:
                break
            
            state = best_action.apply(state)
            plan.append(best_action)
            states.append(state)
            cost += best_action.cost
        
        return PlanResult(state.is_goal(), plan, states, cost, nodes,
                         (time.perf_counter() - start) * 1000, 0, "greedy")


class GATSPlanner:
    def __init__(self, world_model: LayeredWorldModel, budget: int = 10, 
                 c_puct: float = 1.0, max_steps: int = 20):
        self.wm = world_model
        self.budget = budget
        self.c_puct = c_puct
        self.max_steps = max_steps
        self.all_actions = []
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        self.all_actions = actions
        state, plan, states, cost, total_nodes, visited = initial_state, [], [initial_state], 0, 0, set()
        
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in visited:
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
            cost += best_action.cost
        
        return PlanResult(state.is_goal(), plan, states, cost, total_nodes,
                         (time.perf_counter() - start) * 1000, 0, "gats")
    
    def _search(self, state: State, applicable: List[Action]) -> Tuple[Optional[Action], int]:
        if len(applicable) <= 1:
            return (applicable[0] if applicable else None), 1
        
        visits, values, nodes = defaultdict(int), defaultdict(float), 0
        
        for _ in range(self.budget):
            best_action, best_ucb = None, -float('inf')
            total_visits = sum(visits.values()) + 1
            
            for action in applicable:
                nodes += 1
                if visits[action.name] == 0:
                    ucb = float('inf')
                else:
                    ucb = values[action.name] / visits[action.name] + \
                          self.c_puct * (2 * (total_visits ** 0.5) / visits[action.name]) ** 0.5
                if ucb > best_ucb:
                    best_ucb, best_action = ucb, action
            
            if best_action is None:
                break
            
            next_state = best_action.apply(state)
            value = state_value(next_state, self.all_actions)
            visits[best_action.name] += 1
            values[best_action.name] += value
        
        if not visits:
            return (applicable[0] if applicable else None), nodes
        
        best_name = max(visits.keys(), key=lambda n: values[n] / max(1, visits[n]))
        return next((a for a in applicable if a.name == best_name), applicable[0]), nodes


class LATSPlanner:
    def __init__(self, world_model: LayeredWorldModel, budget: int = 10,
                 max_steps: int = 20, llm_backend: str = "mock"):
        self.wm = world_model
        self.budget = budget
        self.max_steps = max_steps
        self.llm_backend = llm_backend
        self.all_actions = []
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        self.all_actions = actions
        state, plan, states, cost, total_nodes, visited = initial_state, [], [initial_state], 0, 0, set()
        
        for _ in range(self.max_steps):
            if state.is_goal() or state.inventory in visited:
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
            cost += best_action.cost
        
        return PlanResult(state.is_goal(), plan, states, cost, total_nodes,
                         (time.perf_counter() - start) * 1000, 0, "lats")
    
    def _search(self, state: State, applicable: List[Action]) -> Tuple[Optional[Action], int]:
        if len(applicable) <= 1:
            return (applicable[0] if applicable else None), 1
        
        visits, values, nodes = defaultdict(int), defaultdict(float), 0
        
        for _ in range(self.budget):
            # LATS: weighted random proposal
            weights = [max(0.1, state_value(a.apply(state), self.all_actions) + 0.1) for a in applicable]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            proposed = random.choices(applicable, weights=weights, k=1)[0]
            
            nodes += 1
            value = state_value(proposed.apply(state), self.all_actions)
            visits[proposed.name] += 1
            values[proposed.name] += value
        
        if not visits:
            return (applicable[0] if applicable else None), nodes
        
        best = max(visits.keys(), key=lambda n: values[n] / max(1, visits[n]))
        return next((a for a in applicable if a.name == best), applicable[0]), nodes


class ReActPlanner:
    def __init__(self, world_model: LayeredWorldModel, max_steps: int = 20, 
                 llm_backend: str = "mock"):
        self.wm = world_model
        self.max_steps = max_steps
        self.llm_backend = llm_backend
    
    def plan(self, initial_state: State, actions: List[Action]) -> PlanResult:
        start = time.perf_counter()
        state, plan, states, cost, nodes = initial_state, [], [initial_state], 0, 0
        
        for _ in range(self.max_steps):
            if state.is_goal():
                break
            
            applicable = [a for a in actions if a.is_applicable(state)]
            if not applicable:
                break
            
            nodes += 1
            action = random.choice(applicable)  # Mock: random selection
            
            state = action.apply(state)
            plan.append(action)
            states.append(state)
            cost += action.cost
        
        return PlanResult(state.is_goal(), plan, states, cost, nodes,
                         (time.perf_counter() - start) * 1000, 0, "react")


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
    optimality: float
    by_source: Dict[str, Dict[str, float]]
    timestamp: str


def evaluate_planner(planner_name: str, tasks: List[PlanningTask], config: EvalConfig, seed: int = 42) -> EvalResults:
    random.seed(seed)
    
    results = []
    total_layer_usage = {1: 0, 2: 0, 3: 0}
    optimality_sum = 0
    by_source = defaultdict(lambda: {"success": 0, "total": 0})
    
    for task in tasks:
        wm = LayeredWorldModel(task.actions, use_l1=config.use_l1, use_l2=config.use_l2,
                               use_l3=config.use_l3, llm_backend=config.llm_backend)
        
        if planner_name == "greedy":
            planner = GreedyPlanner(wm)
        elif planner_name == "gats":
            planner = GATSPlanner(wm, budget=config.search_budget, c_puct=config.c_puct)
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
        
        if result.success and len(result.plan) > 0:
            optimality_sum += task.optimal_length / len(result.plan)
        
        by_source[task.source]["total"] += 1
        if result.success:
            by_source[task.source]["success"] += 1
    
    n = len(results)
    successes = sum(1 for r in results if r.success)
    
    # Convert by_source to success rates
    by_source_rates = {}
    for src, data in by_source.items():
        by_source_rates[src] = {
            "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
            "count": data["total"]
        }
    
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
        by_source=by_source_rates,
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
            
            # Show per-source results
            sources_str = ", ".join(f"{s}={d['success_rate']:.0%}" for s, d in result.by_source.items())
            print(f"  Seed {seed}: SR={result.success_rate:.1%}, Opt={result.optimality:.2f} [{sources_str}]")
    
    return dict(all_results)


def print_comparison_table(results: Dict[str, List[EvalResults]]):
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)
    print(f"{'Method':<20} {'Success':>10} {'Optimality':>10} {'Avg Cost':>10} {'Plan Len':>10} {'Nodes':>10}")
    print("-" * 100)
    
    for name, runs in results.items():
        avg_sr = sum(r.success_rate for r in runs) / len(runs)
        avg_opt = sum(r.optimality for r in runs) / len(runs)
        avg_cost = sum(r.avg_cost for r in runs) / len(runs)
        avg_len = sum(r.avg_plan_length for r in runs) / len(runs)
        avg_nodes = sum(r.avg_nodes_expanded for r in runs) / len(runs)
        
        print(f"{name:<20} {avg_sr:>9.1%} {avg_opt:>10.2f} {avg_cost:>10.1f} {avg_len:>10.1f} {avg_nodes:>10.0f}")
    print("-" * 100)


def print_by_source_table(results: Dict[str, List[EvalResults]]):
    """Print results broken down by task source."""
    print("\n" + "=" * 80)
    print("RESULTS BY TASK SOURCE")
    print("=" * 80)
    
    # Collect all sources
    all_sources = set()
    for runs in results.values():
        for r in runs:
            all_sources.update(r.by_source.keys())
    
    sources = sorted(all_sources)
    
    header = f"{'Method':<20}" + "".join(f"{s:>15}" for s in sources)
    print(header)
    print("-" * 80)
    
    for name, runs in results.items():
        # Average across seeds
        source_rates = defaultdict(list)
        for r in runs:
            for src, data in r.by_source.items():
                source_rates[src].append(data["success_rate"])
        
        row = f"{name:<20}"
        for src in sources:
            if src in source_rates:
                avg = sum(source_rates[src]) / len(source_rates[src])
                row += f"{avg:>14.1%} "
            else:
                row += f"{'---':>15}"
        print(row)
    print("-" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GATS Evaluation with Real Benchmarks")
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    parser.add_argument("--output", type=str, default="results/gats_eval_full.json")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true", help="Only use synthetic tasks")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GATS 2.0 Evaluation with Real Benchmarks")
    print("=" * 60)
    
    # Load tasks
    if args.synthetic_only:
        print(f"\nGenerating {args.n_tasks} synthetic tasks...")
        tasks = generate_synthetic_tasks(args.n_tasks)
        print(f"Generated {len(tasks)} synthetic tasks")
    else:
        tasks = load_all_tasks(args.data_dir, args.n_tasks)
    
    # Define configs
    if args.quick:
        configs = {
            "greedy": EvalConfig(llm_backend=args.backend),
            "gats_b10": EvalConfig(search_budget=10, llm_backend=args.backend),
            "lats_b10": EvalConfig(search_budget=10, llm_backend=args.backend),
            "react": EvalConfig(llm_backend=args.backend),
        }
    else:
        configs = {
            "greedy": EvalConfig(llm_backend=args.backend),
            "react": EvalConfig(llm_backend=args.backend),
            "lats_b5": EvalConfig(search_budget=5, llm_backend=args.backend),
            "lats_b10": EvalConfig(search_budget=10, llm_backend=args.backend),
            "gats_b5": EvalConfig(search_budget=5, llm_backend=args.backend),
            "gats_b10": EvalConfig(search_budget=10, llm_backend=args.backend),
            "gats_b20": EvalConfig(search_budget=20, llm_backend=args.backend),
        }
    
    # Run
    results = run_comparison(tasks, configs, args.seeds)
    
    # Print tables
    print_comparison_table(results)
    print_by_source_table(results)
    
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
    
    gats = avg_sr("gats_b10")
    lats = avg_sr("lats_b10") if "lats_b10" in results else avg_sr("lats_b5")
    react = avg_sr("react")
    greedy = avg_sr("greedy")
    
    print(f"GATS b=10:  {gats:.1%}")
    print(f"LATS:       {lats:.1%}")
    print(f"ReAct:      {react:.1%}")
    print(f"Greedy:     {greedy:.1%}")
    
    if gats > lats:
        print(f"\n✓ GATS > LATS: +{(gats-lats)*100:.1f}%")
    if gats > react:
        print(f"✓ GATS > ReAct: +{(gats-react)*100:.1f}%")


if __name__ == "__main__":
    main()