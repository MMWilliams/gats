"""
API-Bank Benchmark Adapter for GATS 2.0

API-Bank is a benchmark for evaluating tool-augmented LLMs with 300+ real-world APIs.
Reference: Li et al. "API-Bank: A Benchmark for Tool-Augmented LLMs" (2023)

This adapter converts API-Bank tasks into the GATS planning format for evaluation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gats.core import State, ActionSpec


@dataclass
class APIBankTask:
    """Represents a single API-Bank task."""
    task_id: str
    level: int  # 1, 2, or 3
    query: str
    ground_truth_apis: list[str]
    ground_truth_params: list[dict[str, Any]]
    available_apis: list[dict[str, Any]]
    
    @property
    def num_steps(self) -> int:
        return len(self.ground_truth_apis)


@dataclass
class APIBankDataset:
    """API-Bank dataset container."""
    tasks: list[APIBankTask] = field(default_factory=list)
    
    @classmethod
    def load_from_file(cls, path: Path) -> "APIBankDataset":
        """Load API-Bank dataset from JSONL file."""
        tasks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                task = APIBankTask(
                    task_id=data.get("id", f"task_{len(tasks)}"),
                    level=data.get("level", 1),
                    query=data["query"],
                    ground_truth_apis=data["api_calls"],
                    ground_truth_params=data.get("parameters", [{}] * len(data["api_calls"])),
                    available_apis=data.get("available_apis", [])
                )
                tasks.append(task)
        return cls(tasks=tasks)
    
    @classmethod
    def generate_synthetic(cls, n_tasks: int = 50, seed: int = 42) -> "APIBankDataset":
        """
        Generate synthetic API-Bank-style tasks for testing.
        
        Creates tasks across 3 difficulty levels:
        - Level 1: Single API call (e.g., get_weather)
        - Level 2: Multi-step API calls (e.g., search_flights -> book_flight)
        - Level 3: API calls requiring retrieval/context
        """
        random.seed(seed)
        
        # Define synthetic API library
        apis = {
            # Weather APIs
            "get_weather": {"params": ["location"], "returns": "weather_data"},
            "get_forecast": {"params": ["location", "days"], "returns": "forecast_data"},
            
            # Travel APIs
            "search_flights": {"params": ["origin", "destination", "date"], "returns": "flight_list"},
            "book_flight": {"params": ["flight_id", "passenger_info"], "returns": "booking_confirmation"},
            "search_hotels": {"params": ["location", "checkin", "checkout"], "returns": "hotel_list"},
            "book_hotel": {"params": ["hotel_id", "guest_info"], "returns": "hotel_confirmation"},
            
            # Finance APIs
            "get_stock_price": {"params": ["symbol"], "returns": "price_data"},
            "get_exchange_rate": {"params": ["from_currency", "to_currency"], "returns": "rate"},
            "convert_currency": {"params": ["amount", "from_currency", "to_currency"], "returns": "converted_amount"},
            
            # Communication APIs
            "send_email": {"params": ["recipient", "subject", "body"], "returns": "send_status"},
            "send_sms": {"params": ["phone_number", "message"], "returns": "send_status"},
            "schedule_meeting": {"params": ["attendees", "time", "duration"], "returns": "meeting_id"},
            
            # Data APIs
            "search_database": {"params": ["query", "table"], "returns": "results"},
            "update_record": {"params": ["record_id", "fields"], "returns": "update_status"},
            "create_record": {"params": ["table", "data"], "returns": "record_id"},
        }
        
        # Level 1 templates (single API)
        level1_templates = [
            ("What's the weather in {location}?", ["get_weather"]),
            ("Get the stock price for {symbol}", ["get_stock_price"]),
            ("Send an email to {recipient} about {subject}", ["send_email"]),
            ("What's the exchange rate from {from_currency} to {to_currency}?", ["get_exchange_rate"]),
        ]
        
        # Level 2 templates (multi-step)
        level2_templates = [
            ("Book a flight from {origin} to {destination} on {date}", 
             ["search_flights", "book_flight"]),
            ("Convert {amount} {from_currency} to {to_currency}", 
             ["get_exchange_rate", "convert_currency"]),
            ("Find and book a hotel in {location} from {checkin} to {checkout}",
             ["search_hotels", "book_hotel"]),
            ("Search for {query} in {table} and update the first result",
             ["search_database", "update_record"]),
        ]
        
        # Level 3 templates (retrieval + planning)
        level3_templates = [
            ("Plan a trip to {destination}: find flights, book hotel, check weather",
             ["search_flights", "book_flight", "search_hotels", "book_hotel", "get_weather"]),
            ("Prepare meeting with {attendees}: check availability, schedule, send invites",
             ["search_database", "schedule_meeting", "send_email"]),
            ("Financial report: get {symbol} price, compare with {symbol2}, email summary",
             ["get_stock_price", "get_stock_price", "send_email"]),
        ]
        
        tasks = []
        all_templates = (
            [(t, 1) for t in level1_templates] +
            [(t, 2) for t in level2_templates] +
            [(t, 3) for t in level3_templates]
        )
        
        # Sample values for template filling
        locations = ["New York", "London", "Tokyo", "Paris", "Sydney"]
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        currencies = ["USD", "EUR", "GBP", "JPY", "AUD"]
        dates = ["2025-03-01", "2025-03-15", "2025-04-01"]
        
        for i in range(n_tasks):
            template_info, level = random.choice(all_templates)
            query_template, api_calls = template_info
            
            # Fill template with random values
            query = query_template.format(
                location=random.choice(locations),
                symbol=random.choice(symbols),
                symbol2=random.choice(symbols),
                recipient="user@example.com",
                subject="Meeting Update",
                from_currency=random.choice(currencies),
                to_currency=random.choice([c for c in currencies if c != currencies[0]]),
                origin=random.choice(locations),
                destination=random.choice(locations),
                date=random.choice(dates),
                amount=random.randint(100, 10000),
                checkin=dates[0],
                checkout=dates[1],
                query="status=active",
                table="users",
                attendees="team@example.com",
            )
            
            # Generate parameters for each API call
            params = []
            for api_name in api_calls:
                api_info = apis.get(api_name, {})
                param_dict = {p: f"<{p}>" for p in api_info.get("params", [])}
                params.append(param_dict)
            
            # Build available APIs list (ground truth + distractors)
            available = [{"name": api, **apis[api]} for api in api_calls]
            distractors = random.sample(
                [a for a in apis.keys() if a not in api_calls],
                min(5, len(apis) - len(api_calls))
            )
            for d in distractors:
                available.append({"name": d, **apis[d]})
            random.shuffle(available)
            
            tasks.append(APIBankTask(
                task_id=f"synthetic_{i:04d}",
                level=level,
                query=query,
                ground_truth_apis=api_calls,
                ground_truth_params=params,
                available_apis=available
            ))
        
        return cls(tasks=tasks)


class APIBankAdapter:
    """
    Converts API-Bank tasks to GATS planning format.
    
    Translation:
    - APIs become ActionSpecs
    - API parameters become action preconditions (required info items)
    - API returns become effects_add
    - Query intent becomes goal state
    
    Challenge features:
    - Multi-step dependencies (API B requires output of API A)
    - Trap actions (look promising but lead to dead ends)
    - Limited initial information
    """
    
    def __init__(self, task: APIBankTask, add_traps: bool = True):
        self.task = task
        self.add_traps = add_traps
        self._action_specs: list[ActionSpec] = []
        self._build_action_specs()
    
    def _build_action_specs(self) -> None:
        """Convert available APIs to ActionSpecs with dependencies."""
        gt_apis = self.task.ground_truth_apis
        
        for i, api in enumerate(self.task.available_apis):
            name = api["name"]
            params = api.get("params", [])
            returns = api.get("returns", "result")
            
            is_correct = name in gt_apis
            
            # Build preconditions with dependencies
            preconditions = set()
            
            if is_correct:
                # Correct API: add dependency on previous API's output (for multi-step)
                gt_idx = gt_apis.index(name)
                if gt_idx > 0:
                    # Depends on previous ground truth API's result
                    prev_api = gt_apis[gt_idx - 1]
                    for a in self.task.available_apis:
                        if a["name"] == prev_api:
                            preconditions.add(a.get("returns", "result"))
                            break
                else:
                    # First step: needs basic info
                    preconditions.add("query_parsed")
            else:
                # Wrong API: either easy access (trap) or blocked
                if self.add_traps and random.random() < 0.3:
                    # Trap: easy preconditions, but doesn't help goal
                    preconditions.add("query_parsed")
                else:
                    # Blocked: needs something unavailable
                    preconditions.update(f"has_{p}_info" for p in params)
            
            # Effects
            effects_add = frozenset([returns])
            
            # Cost: traps have low cost to be tempting
            if is_correct:
                cost = 2.0  # Correct path has moderate cost
            elif self.add_traps and "query_parsed" in preconditions:
                cost = 1.0  # Trap: tempting low cost
            else:
                cost = 3.0  # Wrong path: higher cost
            
            self._action_specs.append(ActionSpec(
                action_id=name,
                description=f"API call: {name}",
                args_schema={},
                preconditions=frozenset(preconditions),
                effects_add=effects_add,
                effects_remove=frozenset(),
                cost=cost,
            ))
        
        # Add trap actions that look like they help but don't
        if self.add_traps:
            self._add_trap_actions()
    
    def _add_trap_actions(self) -> None:
        """Add deceptive trap actions."""
        gt_apis = self.task.ground_truth_apis
        
        # For each correct API, add a trap variant
        for api_name in gt_apis[:2]:  # First 2 ground truth APIs get traps
            trap_name = f"{api_name}_fast"
            
            # Find the API's return value
            returns = None
            for api in self.task.available_apis:
                if api["name"] == api_name:
                    returns = api.get("returns", "result")
                    break
            
            if returns:
                # Trap produces fake result that looks like real one
                    self._action_specs.append(ActionSpec(
                    action_id=trap_name,
                    description=f"Trap: {trap_name}",
                    args_schema={},
                    preconditions=frozenset(["query_parsed"]),  # Easy access
                    effects_add=frozenset([f"{returns}_cached"]),  # Wrong item!
                    effects_remove=frozenset(),
                    cost=0.5,  # Very tempting low cost
                ))
    
    def to_initial_state(self) -> State:
        """
        Create initial state with limited information.
        """
        # Start with just the parsed query - other info must be obtained
        initial_inventory = frozenset(["query_parsed"])
        
        # Goal: obtain return values from ALL ground truth APIs
        goal_items = set()
        for api_name in self.task.ground_truth_apis:
            for api in self.task.available_apis:
                if api["name"] == api_name:
                    goal_items.add(api.get("returns", "result"))
                    break
        
        return State(
            inventory=initial_inventory,
            goal=frozenset(goal_items)
        )
    
    @property
    def action_specs(self) -> list[ActionSpec]:
        return self._action_specs


class APIBankAdapterHard(APIBankAdapter):
    """
    Harder variant with stricter dependency chains.
    
    For Level 2/3 tasks, goals can ONLY be achieved through correct sequence.
    """
    
    def __init__(self, task: APIBankTask):
        super().__init__(task, add_traps=True)
        self._enforce_strict_dependencies()
    
    def _enforce_strict_dependencies(self) -> None:
        """Make goal items only obtainable through correct path."""
        gt_apis = self.task.ground_truth_apis
        
        if len(gt_apis) < 2:
            return
        
        # For multi-step: final goal requires all intermediate results
        final_api = gt_apis[-1]
        required_intermediates = set()
        
        for api_name in gt_apis[:-1]:
            for api in self.task.available_apis:
                if api["name"] == api_name:
                    required_intermediates.add(api.get("returns", "result"))
                    break
        
        # Update final API to require ALL intermediate results
        for j, spec in enumerate(self._action_specs):
            if spec.action_id == final_api:
                new_preconds = spec.preconditions | frozenset(required_intermediates)
                self._action_specs[j] = ActionSpec(
                    action_id=spec.action_id,
                    description=spec.description,
                    args_schema=spec.args_schema,
                    preconditions=new_preconds,
                    effects_add=spec.effects_add,
                    effects_remove=spec.effects_remove,
                    cost=spec.cost,
                )
                break


def evaluate_api_bank(
    agent_factory,
    dataset: APIBankDataset,
    max_tasks: int | None = None
) -> dict[str, Any]:
    """
    Evaluate an agent on API-Bank tasks.
    
    Args:
        agent_factory: Callable that creates agent given (action_specs, initial_state)
        dataset: API-Bank dataset to evaluate on
        max_tasks: Maximum number of tasks to evaluate (None = all)
    
    Returns:
        Dictionary with evaluation metrics
    """
    from gats.core import ActionModel, WorldModel
    
    tasks = dataset.tasks[:max_tasks] if max_tasks else dataset.tasks
    
    results = {
        "total": len(tasks),
        "by_level": {1: [], 2: [], 3: []},
        "overall": {
            "success": 0,
            "api_accuracy": 0.0,
            "param_accuracy": 0.0,
            "avg_steps": 0.0,
            "avg_cost": 0.0,
        }
    }
    
    for task in tasks:
        adapter = APIBankAdapter(task)
        initial_state = adapter.to_initial_state()
        
        # Create models
        action_model = ActionModel(adapter.action_specs)
        world_model = WorldModel(action_model)
        
        # Create and run agent
        agent = agent_factory(action_model, world_model, initial_state)
        episode = agent.solve(initial_state, task.task_id)
        
        # Extract predicted API calls
        predicted_apis = [e.action_id for e in episode.events]
        
        # Calculate metrics
        api_correct = sum(
            1 for i, api in enumerate(predicted_apis)
            if i < len(task.ground_truth_apis) and api == task.ground_truth_apis[i]
        )
        api_accuracy = api_correct / max(len(task.ground_truth_apis), 1)
        
        # Check if goal was achieved
        success = episode.success
        
        task_result = {
            "task_id": task.task_id,
            "success": success,
            "api_accuracy": api_accuracy,
            "predicted_apis": predicted_apis,
            "ground_truth": task.ground_truth_apis,
            "steps": len(episode.events),
            "cost": episode.total_cost,
        }
        
        results["by_level"][task.level].append(task_result)
        
        if success:
            results["overall"]["success"] += 1
        results["overall"]["api_accuracy"] += api_accuracy
        results["overall"]["avg_steps"] += len(episode.events)
        results["overall"]["avg_cost"] += episode.total_cost
    
    # Compute averages
    n = len(tasks)
    if n > 0:
        results["overall"]["success_rate"] = results["overall"]["success"] / n
        results["overall"]["api_accuracy"] /= n
        results["overall"]["avg_steps"] /= n
        results["overall"]["avg_cost"] /= n
    
    # Per-level summary
    for level in [1, 2, 3]:
        level_tasks = results["by_level"][level]
        if level_tasks:
            results[f"level_{level}_success_rate"] = sum(
                1 for t in level_tasks if t["success"]
            ) / len(level_tasks)
            results[f"level_{level}_api_accuracy"] = sum(
                t["api_accuracy"] for t in level_tasks
            ) / len(level_tasks)
    
    return results


def run_api_bank_benchmark(
    agents: dict[str, Any],
    n_tasks: int = 50,
    seed: int = 42
) -> dict[str, dict]:
    """
    Run API-Bank benchmark comparison across agents.
    
    Args:
        agents: Dictionary mapping agent name to factory function
        n_tasks: Number of synthetic tasks to generate
        seed: Random seed for reproducibility
    
    Returns:
        Results dictionary keyed by agent name
    """
    print("=" * 70)
    print("API-Bank Benchmark (Synthetic)")
    print("=" * 70)
    
    # Generate synthetic dataset
    dataset = APIBankDataset.generate_synthetic(n_tasks=n_tasks, seed=seed)
    
    print(f"Generated {len(dataset.tasks)} tasks:")
    for level in [1, 2, 3]:
        count = sum(1 for t in dataset.tasks if t.level == level)
        print(f"  Level {level}: {count} tasks")
    print()
    
    all_results = {}
    
    for agent_name, agent_factory in agents.items():
        print(f"Running {agent_name}...", end=" ", flush=True)
        
        results = evaluate_api_bank(agent_factory, dataset)
        all_results[agent_name] = results
        
        sr = results["overall"].get("success_rate", 0) * 100
        api_acc = results["overall"]["api_accuracy"] * 100
        print(f"SR: {sr:.1f}%  API-Acc: {api_acc:.1f}%")
    
    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"{'Agent':<12} {'Overall SR':<12} {'API Acc':<12} {'L1 SR':<10} {'L2 SR':<10} {'L3 SR':<10}")
    print("-" * 70)
    
    for agent_name, results in all_results.items():
        overall_sr = results["overall"].get("success_rate", 0) * 100
        api_acc = results["overall"]["api_accuracy"] * 100
        l1_sr = results.get("level_1_success_rate", 0) * 100
        l2_sr = results.get("level_2_success_rate", 0) * 100
        l3_sr = results.get("level_3_success_rate", 0) * 100
        
        print(f"{agent_name:<12} {overall_sr:>10.1f}% {api_acc:>10.1f}% {l1_sr:>8.1f}% {l2_sr:>8.1f}% {l3_sr:>8.1f}%")
    
    return all_results


if __name__ == "__main__":
    # Test synthetic generation
    dataset = APIBankDataset.generate_synthetic(n_tasks=10)
    for task in dataset.tasks[:3]:
        print(f"Task: {task.task_id} (Level {task.level})")
        print(f"  Query: {task.query}")
        print(f"  Ground truth: {task.ground_truth_apis}")
        print()