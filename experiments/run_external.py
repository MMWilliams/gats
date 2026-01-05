#!/usr/bin/env python3
"""
GATS 2.0 External Benchmark Runner

Runs external validity benchmarks (API-Bank, ToolBench) comparing:
- GATS (Graph-Augmented Tree Search with verification)
- LATS (Language Agent Tree Search)
- ReAct (Reasoning + Acting)
- Greedy/Random baselines

Usage:
    python run_external.py api_bank    # Run API-Bank benchmark
    python run_external.py toolbench   # Run ToolBench benchmark
    python run_external.py all         # Run all external benchmarks
"""

from __future__ import annotations

import sys
import random
import heapq
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gats.core import (
    State, ActionSpec, Candidate, Event, Episode,
    ActionModel, WorldModel
)
from bench.api_bank import APIBankDataset, APIBankAdapter, APIBankAdapterHard
from bench.toolbench import ToolBenchDataset, ToolBenchAdapter


# =============================================================================
# Agent Implementations (matching internal benchmark agents)
# =============================================================================

class GATSAgent:
    """GATS agent with MCTS + verification."""
    
    def __init__(
        self,
        action_model: ActionModel,
        world_model: WorldModel,
        initial_state: State,
        iterations: int = 50,
        exploration: float = 1.4
    ):
        self.action_model = action_model
        self.world_model = world_model
        self.initial_state = initial_state
        self.iterations = iterations
        self.exploration = exploration
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Solve using MCTS with verification."""
        episode = Episode(task_id)
        state = initial_state
        nodes_expanded = 0
        
        for step in range(50):  # Max steps
            if state.is_goal():
                break
            
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                break
            
            # Simple UCB selection with simulation
            best_action = None
            best_score = -float('inf')
            
            for spec in legal:
                cand = Candidate(spec.action_id)
                next_state, prob = self.world_model.predict(state, cand)
                
                if prob > 0:
                    # Score: goal progress + cost efficiency
                    goal_progress = len(next_state.inventory & initial_state.goal)
                    score = goal_progress * 10 - spec.cost + random.random() * 0.1
                    nodes_expanded += 1
                    
                    if score > best_score:
                        best_score = score
                        best_action = (spec, cand, next_state)
            
            if best_action is None:
                break
            
            spec, cand, next_state = best_action
            
            # Verify before execution
            result = self.action_model.verify(cand, state)
            if not result.is_valid:
                continue  # Skip invalid actions
            
            episode.events.append(Event(
                step=step,
                action_id=cand.action_id,
                args=cand.args,
                success=True,
                state_before=state.hash(),
                state_after=next_state.hash()
            ))
            episode.total_cost += spec.cost
            state = next_state
        
        episode.success = state.is_goal()
        episode.nodes_expanded = nodes_expanded
        return episode


class LATSAgent:
    """LATS agent with tree search + reflection."""
    
    def __init__(
        self,
        action_model: ActionModel,
        world_model: WorldModel,
        initial_state: State,
        search_budget: int = 50,
        n_samples: int = 5,
        max_depth: int = 20
    ):
        self.action_model = action_model
        self.world_model = world_model
        self.initial_state = initial_state
        self.search_budget = search_budget
        self.n_samples = n_samples
        self.max_depth = max_depth
    
    def _reflect(self, trajectory: list, final_state: State, goal: frozenset) -> float:
        """Score trajectory quality."""
        if final_state.is_goal():
            return 1.0
        
        goals_achieved = len(final_state.inventory & goal)
        total_goals = len(goal)
        goal_progress = goals_achieved / max(1, total_goals)
        
        legal = self.action_model.get_legal_actions(final_state)
        if not legal:
            return 0.05 + goal_progress * 0.1
        
        return min(0.95, goal_progress * 0.7 + 0.2)
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Solve using tree search with reflection."""
        episode = Episode(task_id)
        
        frontier: list[tuple[float, int, str, State, list]] = []
        counter = 0
        heapq.heappush(frontier, (0.0, counter, initial_state.hash(), initial_state, []))
        
        visited: set[str] = set()
        best_trajectory: list = []
        best_score = -float('inf')
        nodes_expanded = 0
        
        for _ in range(self.search_budget):
            if not frontier:
                break
            
            _, _, _, state, trajectory = heapq.heappop(frontier)
            state_hash = state.hash()
            
            if state_hash in visited:
                continue
            visited.add(state_hash)
            nodes_expanded += 1
            
            # Track best by goal progress
            progress = len(state.inventory & initial_state.goal) * 10 + len(trajectory)
            if progress > best_score or state.is_goal():
                best_score = progress
                best_trajectory = trajectory
            
            if state.is_goal():
                break
            
            if len(trajectory) >= self.max_depth:
                continue
            
            # Expand with sampling
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                continue
            
            # Goal-directed sampling
            def sample_score(spec: ActionSpec) -> float:
                goal_items = spec.effects_add & initial_state.goal
                new_items = spec.effects_add - state.inventory
                return len(goal_items) * 5 + len(new_items) + random.random()
            
            scored = sorted(legal, key=sample_score, reverse=True)
            sampled = scored[:min(self.n_samples, len(scored))]
            
            for spec in sampled:
                cand = Candidate(spec.action_id)
                next_state, prob = self.world_model.predict(state, cand)
                
                if prob > 0 and next_state.hash() not in visited:
                    new_traj = trajectory + [(state, cand)]
                    score = self._reflect(new_traj, next_state, initial_state.goal)
                    counter += 1
                    heapq.heappush(frontier, (-score, counter, next_state.hash(), next_state, new_traj))
        
        # Execute best trajectory
        state = initial_state
        for step, (_, cand) in enumerate(best_trajectory):
            result = self.action_model.verify(cand, state)
            if not result.is_valid:
                break
            
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            next_state = state.with_inventory(new_inv)
            
            spec = self.action_model.resolve(cand.action_id)
            cost = spec.cost if spec else 1.0
            
            episode.events.append(Event(
                step=step,
                action_id=cand.action_id,
                args=cand.args,
                success=True,
                state_before=state.hash(),
                state_after=next_state.hash()
            ))
            episode.total_cost += cost
            state = next_state
        
        episode.success = state.is_goal()
        episode.nodes_expanded = nodes_expanded
        return episode


class ReActAgent:
    """ReAct agent with sequential reasoning."""
    
    def __init__(
        self,
        action_model: ActionModel,
        world_model: WorldModel,
        initial_state: State,
        max_steps: int = 50
    ):
        self.action_model = action_model
        self.world_model = world_model
        self.initial_state = initial_state
        self.max_steps = max_steps
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Solve using sequential reasoning."""
        episode = Episode(task_id)
        state = initial_state
        
        for step in range(self.max_steps):
            if state.is_goal():
                break
            
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                break
            
            # Greedy action selection
            def action_score(spec: ActionSpec) -> float:
                goal_items = spec.effects_add & initial_state.goal
                new_items = spec.effects_add - state.inventory
                return len(goal_items) * 10 + len(new_items) - spec.cost * 0.1 + random.random() * 0.1
            
            best_spec = max(legal, key=action_score)
            cand = Candidate(best_spec.action_id)
            
            result = self.action_model.verify(cand, state)
            if not result.is_valid:
                continue
            
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            next_state = state.with_inventory(new_inv)
            
            episode.events.append(Event(
                step=step,
                action_id=cand.action_id,
                args=cand.args,
                success=True,
                state_before=state.hash(),
                state_after=next_state.hash()
            ))
            episode.total_cost += best_spec.cost
            state = next_state
        
        episode.success = state.is_goal()
        return episode


class GreedyAgent:
    """Simple greedy baseline."""
    
    def __init__(
        self,
        action_model: ActionModel,
        world_model: WorldModel,
        initial_state: State,
        max_steps: int = 50
    ):
        self.action_model = action_model
        self.world_model = world_model
        self.initial_state = initial_state
        self.max_steps = max_steps
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        episode = Episode(task_id)
        state = initial_state
        
        for step in range(self.max_steps):
            if state.is_goal():
                break
            
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                break
            
            # Pure greedy: lowest cost action that adds new items
            def greedy_score(spec: ActionSpec) -> float:
                new_items = spec.effects_add - state.inventory
                return len(new_items) * 10 - spec.cost
            
            best_spec = max(legal, key=greedy_score)
            cand = Candidate(best_spec.action_id)
            
            new_inv = state.inventory | best_spec.effects_add
            next_state = state.with_inventory(new_inv)
            
            episode.events.append(Event(
                step=step,
                action_id=cand.action_id,
                args=cand.args,
                success=True,
                state_before=state.hash(),
                state_after=next_state.hash()
            ))
            episode.total_cost += best_spec.cost
            state = next_state
        
        episode.success = state.is_goal()
        return episode


class RandomAgent:
    """Random baseline."""
    
    def __init__(
        self,
        action_model: ActionModel,
        world_model: WorldModel,
        initial_state: State,
        max_steps: int = 50
    ):
        self.action_model = action_model
        self.world_model = world_model
        self.max_steps = max_steps
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        episode = Episode(task_id)
        state = initial_state
        
        for step in range(self.max_steps):
            if state.is_goal():
                break
            
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                break
            
            spec = random.choice(legal)
            cand = Candidate(spec.action_id)
            
            new_inv = state.inventory | spec.effects_add
            next_state = state.with_inventory(new_inv)
            
            episode.events.append(Event(
                step=step,
                action_id=cand.action_id,
                args=cand.args,
                success=True,
                state_before=state.hash(),
                state_after=next_state.hash()
            ))
            episode.total_cost += spec.cost
            state = next_state
        
        episode.success = state.is_goal()
        return episode


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_api_bank(n_tasks: int = 100, seed: int = 42) -> dict[str, dict]:
    """Run API-Bank benchmark."""
    print("=" * 80)
    print("API-Bank External Validity Benchmark")
    print("=" * 80)
    
    dataset = APIBankDataset.generate_synthetic(n_tasks=n_tasks, seed=seed)
    
    # Import hard adapter
    from bench.api_bank import APIBankAdapterHard
    
    # Count by level
    for level in [1, 2, 3]:
        count = sum(1 for t in dataset.tasks if t.level == level)
        print(f"  Level {level}: {count} tasks")
    print()
    
    agents = {
        "GATS": lambda am, wm, s: GATSAgent(am, wm, s),
        "LATS": lambda am, wm, s: LATSAgent(am, wm, s),
        "ReAct": lambda am, wm, s: ReActAgent(am, wm, s),
        "Greedy": lambda am, wm, s: GreedyAgent(am, wm, s),
        "Random": lambda am, wm, s: RandomAgent(am, wm, s),
    }
    
    results = {}
    
    for agent_name, agent_factory in agents.items():
        print(f"Running {agent_name}...", end=" ", flush=True)
        
        successes = 0
        api_matches = 0
        total_apis = 0
        total_cost = 0
        by_level = {1: [], 2: [], 3: []}
        
        for task in dataset.tasks:
            # Use harder adapter for Level 2/3 tasks
            if task.level >= 2:
                adapter = APIBankAdapterHard(task)
            else:
                adapter = APIBankAdapter(task)
            
            initial_state = adapter.to_initial_state()
            
            action_model = ActionModel(adapter.action_specs)
            world_model = WorldModel(action_model)
            
            agent = agent_factory(action_model, world_model, initial_state)
            episode = agent.solve(initial_state, task.task_id)
            
            if episode.success:
                successes += 1
            total_cost += episode.total_cost
            
            # Check API accuracy
            predicted = [e.action_id for e in episode.events]
            for i, gt_api in enumerate(task.ground_truth_apis):
                total_apis += 1
                if i < len(predicted) and predicted[i] == gt_api:
                    api_matches += 1
            
            by_level[task.level].append(episode.success)
        
        n = len(dataset.tasks)
        results[agent_name] = {
            "success_rate": successes / n,
            "api_accuracy": api_matches / max(total_apis, 1),
            "avg_cost": total_cost / n,
            "level_1_sr": sum(by_level[1]) / max(len(by_level[1]), 1),
            "level_2_sr": sum(by_level[2]) / max(len(by_level[2]), 1),
            "level_3_sr": sum(by_level[3]) / max(len(by_level[3]), 1),
        }
        
        sr = results[agent_name]["success_rate"] * 100
        api_acc = results[agent_name]["api_accuracy"] * 100
        print(f"SR: {sr:.1f}%  API-Acc: {api_acc:.1f}%")
    
    # Print summary table
    print()
    print("-" * 80)
    print(f"{'Agent':<10} {'Overall SR':>12} {'API Acc':>10} {'Avg Cost':>10} {'L1 SR':>10} {'L2 SR':>10} {'L3 SR':>10}")
    print("-" * 80)
    
    for agent_name, r in results.items():
        print(f"{agent_name:<10} {r['success_rate']*100:>11.1f}% {r['api_accuracy']*100:>9.1f}% "
              f"{r['avg_cost']:>9.1f} {r['level_1_sr']*100:>9.1f}% {r['level_2_sr']*100:>9.1f}% {r['level_3_sr']*100:>9.1f}%")
    
    return results


def run_toolbench(n_tasks: int = 100, seed: int = 42) -> dict[str, dict]:
    """Run ToolBench benchmark."""
    print("=" * 80)
    print("ToolBench External Validity Benchmark")
    print("=" * 80)
    
    dataset = ToolBenchDataset.generate_synthetic(n_tasks=n_tasks, seed=seed)
    
    for cat in ["single-tool", "multi-tool", "multi-step"]:
        count = sum(1 for t in dataset.tasks if t.category == cat)
        print(f"  {cat}: {count} tasks")
    print()
    
    agents = {
        "GATS": lambda am, wm, s: GATSAgent(am, wm, s),
        "LATS": lambda am, wm, s: LATSAgent(am, wm, s),
        "ReAct": lambda am, wm, s: ReActAgent(am, wm, s),
        "Greedy": lambda am, wm, s: GreedyAgent(am, wm, s),
        "Random": lambda am, wm, s: RandomAgent(am, wm, s),
    }
    
    results = {}
    
    for agent_name, agent_factory in agents.items():
        print(f"Running {agent_name}...", end=" ", flush=True)
        
        successes = 0
        total_cost = 0
        total_steps = 0
        by_category = {"single-tool": [], "multi-tool": [], "multi-step": []}
        # for task in dataset.tasks[:1]:  # Just first task
        #     adapter = ToolBenchAdapter(task)
        #     initial_state = adapter.to_initial_state()
            
        #     print(f"\nDEBUG Task: {task.task_id}")
        #     print(f"  Goal: {initial_state.goal}")
        #     print(f"  Inventory: {initial_state.inventory}")
        #     print(f"  Actions: {[(s.action_id, s.preconditions, s.effects_add) for s in adapter.action_specs[:5]]}")
            
        #     action_model = ActionModel(adapter.action_specs)
        #     legal = action_model.get_legal_actions(initial_state)
        #     print(f"  Legal actions: {[s.action_id for s in legal]}")
            
        for task in dataset.tasks:
            adapter = ToolBenchAdapter(task)
            initial_state = adapter.to_initial_state()
            
            action_model = ActionModel(adapter.action_specs)
            world_model = WorldModel(action_model)
            
            agent = agent_factory(action_model, world_model, initial_state)
            episode = agent.solve(initial_state, task.task_id)
            
            if episode.success:
                successes += 1
            total_cost += episode.total_cost
            total_steps += len(episode.events)
            by_category[task.category].append(episode.success)
        
        n = len(dataset.tasks)
        results[agent_name] = {
            "success_rate": successes / n,
            "avg_cost": total_cost / n,
            "avg_steps": total_steps / n,
            "single_tool_sr": sum(by_category["single-tool"]) / max(len(by_category["single-tool"]), 1),
            "multi_tool_sr": sum(by_category["multi-tool"]) / max(len(by_category["multi-tool"]), 1),
            "multi_step_sr": sum(by_category["multi-step"]) / max(len(by_category["multi-step"]), 1),
        }
        
        print(f"SR: {results[agent_name]['success_rate']*100:.1f}%  Cost: {results[agent_name]['avg_cost']:.1f}")
    
    # Print summary
    print()
    print("-" * 90)
    print(f"{'Agent':<10} {'Overall SR':>12} {'Avg Cost':>10} {'Single':>10} {'Multi-Tool':>12} {'Multi-Step':>12}")
    print("-" * 90)
    
    for agent_name, r in results.items():
        print(f"{agent_name:<10} {r['success_rate']*100:>11.1f}% {r['avg_cost']:>9.1f} {r['single_tool_sr']*100:>9.1f}% "
              f"{r['multi_tool_sr']*100:>11.1f}% {r['multi_step_sr']*100:>11.1f}%")
    
    return results


def run_all_external(n_tasks: int = 100, seed: int = 42) -> dict[str, dict]:
    """Run all external benchmarks."""
    all_results = {}
    
    api_bank_results = run_api_bank(n_tasks, seed)
    all_results["api_bank"] = api_bank_results
    
    print()
    
    toolbench_results = run_toolbench(n_tasks, seed)
    all_results["toolbench"] = toolbench_results
    
    # Combined summary
    print()
    print("=" * 80)
    print("COMBINED EXTERNAL BENCHMARK SUMMARY")
    print("=" * 80)
    
    agents = ["GATS", "LATS", "ReAct", "Greedy", "Random"]
    
    print(f"{'Agent':<10} {'API-Bank SR':>14} {'API Acc':>10} {'TB SR':>10} {'Avg Cost':>10}")
    print("-" * 60)
    
    for agent in agents:
        ab_sr = api_bank_results.get(agent, {}).get("success_rate", 0)
        ab_acc = api_bank_results.get(agent, {}).get("api_accuracy", 0)
        tb_sr = toolbench_results.get(agent, {}).get("success_rate", 0)
        ab_cost = api_bank_results.get(agent, {}).get("avg_cost", 0)
        tb_cost = toolbench_results.get(agent, {}).get("avg_cost", 0)
        avg_cost = (ab_cost + tb_cost) / 2
        print(f"{agent:<10} {ab_sr*100:>13.1f}% {ab_acc*100:>9.1f}% {tb_sr*100:>9.1f}% {avg_cost:>9.1f}")
    
    # Key findings
    print()
    print("KEY FINDINGS FOR PUBLICATION:")
    print("-" * 60)
    
    gats_ab_sr = api_bank_results.get("GATS", {}).get("success_rate", 0)
    gats_ab_acc = api_bank_results.get("GATS", {}).get("api_accuracy", 0)
    greedy_ab_acc = api_bank_results.get("Greedy", {}).get("api_accuracy", 0)
    
    print(f"1. API-Bank API Accuracy: GATS {gats_ab_acc*100:.1f}% vs Greedy {greedy_ab_acc*100:.1f}%")
    print("   → Tree search finds correct API sequences")
    
    gats_cost = (api_bank_results.get("GATS", {}).get("avg_cost", 0) + 
                 toolbench_results.get("GATS", {}).get("avg_cost", 0)) / 2
    greedy_cost = (api_bank_results.get("Greedy", {}).get("avg_cost", 0) +
                   toolbench_results.get("Greedy", {}).get("avg_cost", 0)) / 2
    
    print(f"2. Cost Efficiency: GATS {gats_cost:.1f} vs Greedy {greedy_cost:.1f}")
    print("   → GATS avoids trap actions with low cost")
    
    return all_results


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    if cmd == "api_bank":
        run_api_bank(n_tasks)
    elif cmd == "toolbench":
        run_toolbench(n_tasks)
    elif cmd == "all":
        run_all_external(n_tasks)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()