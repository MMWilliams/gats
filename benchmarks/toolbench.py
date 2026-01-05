"""
ToolBench Benchmark Adapter for GATS 2.0

ToolBench evaluates tool-use capabilities with 16,000+ real-world APIs.
Reference: Qin et al. "ToolLLM: Facilitating Large Language Models to Master Tools" (2023)

This adapter provides:
1. Synthetic ToolBench-style tasks for local testing
2. Interface for real ToolBench evaluation (requires external setup)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gats.core import State, ActionSpec


@dataclass
class ToolBenchTask:
    """Represents a ToolBench evaluation task."""
    task_id: str
    category: str  # e.g., "single-tool", "multi-tool", "multi-step"
    instruction: str
    tools: list[dict[str, Any]]
    ground_truth_calls: list[dict[str, Any]]
    
    @property
    def num_tools(self) -> int:
        return len(set(c["tool"] for c in self.ground_truth_calls))
    
    @property
    def num_steps(self) -> int:
        return len(self.ground_truth_calls)


@dataclass
class ToolBenchDataset:
    """ToolBench dataset container."""
    tasks: list[ToolBenchTask] = field(default_factory=list)
    
    @classmethod
    def generate_synthetic(
        cls,
        n_tasks: int = 50,
        seed: int = 42
    ) -> "ToolBenchDataset":
        """
        Generate synthetic ToolBench-style tasks.
        
        Categories:
        - single-tool: One API, one call
        - multi-tool: Multiple APIs, parallel calls  
        - multi-step: Sequential API calls with dependencies
        """
        random.seed(seed)
        
        # Tool library (subset of realistic tools)
        tools = {
            "weather_api": {
                "name": "weather_api",
                "description": "Get current weather and forecasts",
                "endpoints": {
                    "get_current": ["location"],
                    "get_forecast": ["location", "days"],
                    "get_alerts": ["location"]
                }
            },
            "maps_api": {
                "name": "maps_api", 
                "description": "Navigation and location services",
                "endpoints": {
                    "get_directions": ["origin", "destination", "mode"],
                    "search_places": ["query", "location", "radius"],
                    "get_distance": ["origin", "destination"]
                }
            },
            "calendar_api": {
                "name": "calendar_api",
                "description": "Calendar and scheduling",
                "endpoints": {
                    "create_event": ["title", "start_time", "end_time"],
                    "list_events": ["date", "calendar_id"],
                    "update_event": ["event_id", "updates"]
                }
            },
            "email_api": {
                "name": "email_api",
                "description": "Email sending and management",
                "endpoints": {
                    "send_email": ["to", "subject", "body"],
                    "search_emails": ["query", "folder"],
                    "get_email": ["email_id"]
                }
            },
            "finance_api": {
                "name": "finance_api",
                "description": "Stock prices and financial data",
                "endpoints": {
                    "get_price": ["symbol"],
                    "get_history": ["symbol", "period"],
                    "get_news": ["symbol"]
                }
            },
            "translation_api": {
                "name": "translation_api",
                "description": "Text translation services",
                "endpoints": {
                    "translate": ["text", "source_lang", "target_lang"],
                    "detect_language": ["text"]
                }
            }
        }
        
        # Task templates by category
        templates = {
            "single-tool": [
                ("What's the weather in {location}?", 
                 [{"tool": "weather_api", "endpoint": "get_current", "params": {"location": "{location}"}}]),
                ("Get directions from {origin} to {destination}",
                 [{"tool": "maps_api", "endpoint": "get_directions", 
                   "params": {"origin": "{origin}", "destination": "{destination}", "mode": "driving"}}]),
                ("What's the stock price of {symbol}?",
                 [{"tool": "finance_api", "endpoint": "get_price", "params": {"symbol": "{symbol}"}}]),
            ],
            "multi-tool": [
                ("Check weather in {location} and find nearby restaurants",
                 [{"tool": "weather_api", "endpoint": "get_current", "params": {"location": "{location}"}},
                  {"tool": "maps_api", "endpoint": "search_places", 
                   "params": {"query": "restaurants", "location": "{location}", "radius": "5km"}}]),
                ("Get {symbol} price and latest news",
                 [{"tool": "finance_api", "endpoint": "get_price", "params": {"symbol": "{symbol}"}},
                  {"tool": "finance_api", "endpoint": "get_news", "params": {"symbol": "{symbol}"}}]),
            ],
            "multi-step": [
                ("Schedule meeting about {topic}, then email {recipient} the details",
                 [{"tool": "calendar_api", "endpoint": "create_event",
                   "params": {"title": "{topic}", "start_time": "tomorrow 2pm", "end_time": "tomorrow 3pm"}},
                  {"tool": "email_api", "endpoint": "send_email",
                   "params": {"to": "{recipient}", "subject": "Meeting: {topic}", "body": "Meeting scheduled..."}}]),
                ("Translate '{text}' to {target_lang} and email to {recipient}",
                 [{"tool": "translation_api", "endpoint": "translate",
                   "params": {"text": "{text}", "source_lang": "en", "target_lang": "{target_lang}"}},
                  {"tool": "email_api", "endpoint": "send_email",
                   "params": {"to": "{recipient}", "subject": "Translation", "body": "<translated>"}}]),
            ]
        }
        
        # Sample values
        locations = ["New York", "London", "Tokyo", "Paris"]
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        tasks = []
        for i in range(n_tasks):
            category = random.choice(list(templates.keys()))
            template, calls_template = random.choice(templates[category])
            
            # Fill template
            instruction = template.format(
                location=random.choice(locations),
                origin=random.choice(locations),
                destination=random.choice(locations),
                symbol=random.choice(symbols),
                topic="Q1 Review",
                recipient="team@example.com",
                text="Hello, how are you?",
                target_lang="Spanish"
            )
            
            # Build ground truth calls
            gt_calls = []
            for call in calls_template:
                gt_calls.append({
                    "tool": call["tool"],
                    "endpoint": call["endpoint"],
                    "params": {k: v.format(
                        location=random.choice(locations),
                        origin=random.choice(locations),
                        destination=random.choice(locations),
                        symbol=random.choice(symbols),
                        topic="Q1 Review",
                        recipient="team@example.com",
                        text="Hello, how are you?",
                        target_lang="es"
                    ) if isinstance(v, str) else v for k, v in call["params"].items()}
                })
            
            # Select available tools (ground truth + distractors)
            used_tools = list(set(c["tool"] for c in gt_calls))
            available_tools = [tools[t] for t in used_tools]
            distractor_tools = [t for t in tools.keys() if t not in used_tools]
            available_tools.extend([tools[t] for t in random.sample(distractor_tools, min(2, len(distractor_tools)))])
            
            tasks.append(ToolBenchTask(
                task_id=f"toolbench_{i:04d}",
                category=category,
                instruction=instruction,
                tools=available_tools,
                ground_truth_calls=gt_calls
            ))
        
        return cls(tasks=tasks)


class ToolBenchAdapter:
    """
    Converts ToolBench tasks to GATS planning format.
    
    Challenge features:
    - Sequential dependencies between tool calls
    - Trap endpoints that look helpful but aren't
    - Limited initial context
    """
    
    def __init__(self, task: ToolBenchTask, add_traps: bool = True):
        self.task = task
        self.add_traps = add_traps
        self._action_specs: list[ActionSpec] = []
        self._build_action_specs()
    
    def _build_action_specs(self) -> None:
        """Convert tools to ActionSpecs with dependencies."""
        gt_calls = self.task.ground_truth_calls
        gt_endpoints = [(c["tool"], c["endpoint"]) for c in gt_calls]
        
        for tool in self.task.tools:
            tool_name = tool["name"]
            for endpoint, params in tool.get("endpoints", {}).items():
                action_id = f"{tool_name}.{endpoint}"
                is_correct = (tool_name, endpoint) in gt_endpoints
                
                # Build preconditions with dependencies
                preconditions = set()
                
                if is_correct:
                    # Find position in ground truth sequence
                    gt_idx = None
                    for i, (t, e) in enumerate(gt_endpoints):
                        if t == tool_name and e == endpoint:
                            gt_idx = i
                            break
                    
                    if gt_idx == 0:
                        # First call: needs initial context only
                        preconditions.add("task_context")
                    elif gt_idx is not None and gt_idx > 0:
                        # Depends on previous call's result
                        prev_t, prev_e = gt_endpoints[gt_idx - 1]
                        preconditions.add(f"{prev_e}_result")
                        preconditions.add("task_context")  # Also need context
                    
                    cost = 2.0
                else:
                    # Wrong endpoint: harder to access (NOT a trap by default)
                    preconditions.update(f"has_{p}" for p in params)
                    cost = 3.0
                
                # Effects
                effects_add = frozenset([f"{endpoint}_result"])
                
                self._action_specs.append(ActionSpec(
                    action_id=action_id,
                    description=f"Tool: {action_id}",
                    args_schema={},
                    preconditions=frozenset(preconditions),
                    effects_add=effects_add,
                    effects_remove=frozenset(),
                    cost=cost,
                ))
        
        # Add traps AFTER correct actions are set up
        if self.add_traps:
            self._add_trap_endpoints()
    
    def _add_trap_endpoints(self) -> None:
        """Add deceptive trap endpoints."""
        gt_calls = self.task.ground_truth_calls
        
        for call in gt_calls[:2]:
            trap_id = f"{call['tool']}.{call['endpoint']}_cached"
            result = f"{call['endpoint']}_result"
            
            # Trap returns wrong variant of expected result
            self._action_specs.append(ActionSpec(
                action_id=trap_id,
                description=f"Trap: {trap_id}",
                args_schema={},
                preconditions=frozenset(["task_context"]),
                effects_add=frozenset([f"{result}_stale"]),  # Wrong!
                effects_remove=frozenset(),
                cost=0.5,  # Very tempting
            ))
    
    def to_initial_state(self) -> State:
        """Create initial state."""
        # Start with task context only
        initial_inventory = frozenset(["task_context"])
        
        # Goal: results from all ground truth calls
        goal_items = frozenset(
            f"{c['endpoint']}_result" for c in self.task.ground_truth_calls
        )
        
        return State(inventory=initial_inventory, goal=goal_items)
    
    @property
    def action_specs(self) -> list[ActionSpec]:
        return self._action_specs


def run_toolbench_benchmark(
    agents: dict[str, Any],
    n_tasks: int = 50,
    seed: int = 42
) -> dict[str, dict]:
    """Run ToolBench benchmark comparison."""
    from gats.core import ActionModel, WorldModel
    
    print("=" * 70)
    print("ToolBench Benchmark (Synthetic)")
    print("=" * 70)
    
    dataset = ToolBenchDataset.generate_synthetic(n_tasks=n_tasks, seed=seed)
    
    # Count by category
    for cat in ["single-tool", "multi-tool", "multi-step"]:
        count = sum(1 for t in dataset.tasks if t.category == cat)
        print(f"  {cat}: {count} tasks")
    print()
    
    all_results = {}
    
    for agent_name, agent_factory in agents.items():
        print(f"Running {agent_name}...", end=" ", flush=True)
        
        successes = 0
        total_steps = 0
        total_cost = 0
        by_category = {"single-tool": [], "multi-tool": [], "multi-step": []}
        
        for task in dataset.tasks:
            adapter = ToolBenchAdapter(task)
            initial_state = adapter.to_initial_state()
            
            action_model = ActionModel(adapter.action_specs)
            world_model = WorldModel(action_model)
            
            agent = agent_factory(action_model, world_model, initial_state)
            episode = agent.solve(initial_state, task.task_id)
            
            if episode.success:
                successes += 1
            total_steps += len(episode.events)
            total_cost += episode.total_cost
            
            by_category[task.category].append(episode.success)
        
        n = len(dataset.tasks)
        results = {
            "success_rate": successes / n,
            "avg_steps": total_steps / n,
            "avg_cost": total_cost / n,
        }
        
        for cat, outcomes in by_category.items():
            if outcomes:
                results[f"{cat}_sr"] = sum(outcomes) / len(outcomes)
        
        all_results[agent_name] = results
        print(f"SR: {results['success_rate']*100:.1f}%")
    
    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"{'Agent':<12} {'Overall SR':<12} {'Single':<10} {'Multi-Tool':<12} {'Multi-Step':<12}")
    print("-" * 70)
    
    for agent_name, results in all_results.items():
        overall = results["success_rate"] * 100
        single = results.get("single-tool_sr", 0) * 100
        multi_t = results.get("multi-tool_sr", 0) * 100
        multi_s = results.get("multi-step_sr", 0) * 100
        print(f"{agent_name:<12} {overall:>10.1f}% {single:>8.1f}% {multi_t:>10.1f}% {multi_s:>10.1f}%")
    
    return all_results


if __name__ == "__main__":
    dataset = ToolBenchDataset.generate_synthetic(n_tasks=10)
    for task in dataset.tasks[:3]:
        print(f"Task: {task.task_id} ({task.category})")
        print(f"  Instruction: {task.instruction}")
        print(f"  Tools: {[t['name'] for t in task.tools]}")
        print(f"  Ground truth: {[c['tool']+'.'+c['endpoint'] for c in task.ground_truth_calls]}")
        print()