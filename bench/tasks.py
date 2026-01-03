"""GATS 2.0 Benchmark Task Generators."""
from __future__ import annotations
import random
from dataclasses import dataclass
from gats import ActionSpec, ActionModel, State

@dataclass
class BenchmarkConfig:
    depth: int = 5
    n_tools: int = 50
    branching: int = 2
    distractors: int = 10
    dead_ends: int = 3
    seed: int = 42

def generate_chain_task(config: BenchmarkConfig) -> tuple[ActionModel, State]:
    """Generate controlled dependency-chain benchmark.
    
    Creates a task where the goal requires traversing a chain of dependencies,
    with distractors and alternative paths to test search quality.
    """
    random.seed(config.seed)
    model = ActionModel()
    
    # Build main dependency chain: item_0 -> item_1 -> ... -> item_N
    items = [f"item_{i}" for i in range(config.depth + 1)]
    
    for i in range(config.depth):
        for b in range(config.branching):
            # Multiple actions can produce the same next item
            spec = ActionSpec(
                action_id=f"step_{i}_v{b}",
                description=f"Step {i} variant {b}",
                args_schema={},
                preconditions=frozenset([items[i]]),
                effects_add=frozenset([items[i + 1]]),
                cost=1.0 + b * 0.1  # Slight cost difference
            )
            model.register(spec)
    
    # Add distractor actions (valid but don't help)
    for d in range(config.distractors):
        distractor_item = f"distractor_{d}"
        spec = ActionSpec(
            action_id=f"distract_{d}",
            description=f"Distractor {d}",
            args_schema={},
            preconditions=frozenset([items[0]]),  # Always available
            effects_add=frozenset([distractor_item]),
            cost=0.5
        )
        model.register(spec)
    
    # Fill remaining tool slots with near-miss actions
    n_real = config.depth * config.branching + config.distractors
    for t in range(config.n_tools - n_real):
        # Near-miss: requires items we don't have
        fake_prereq = f"fake_prereq_{t}"
        spec = ActionSpec(
            action_id=f"nearmiss_{t}",
            description=f"Near-miss tool {t}",
            args_schema={},
            preconditions=frozenset([fake_prereq]),
            effects_add=frozenset([f"nearmiss_out_{t}"]),
            cost=1.0
        )
        model.register(spec, aliases=[f"nm{t}"])
    
    # Initial state with item_0, goal is item_N
    initial_state = State(
        goal=frozenset([items[-1]]),
        inventory=frozenset([items[0]])
    )
    
    return model, initial_state
def generate_deceptive_task(config: BenchmarkConfig) -> tuple[ActionModel, State]:
    """Generate task where traps consume resources - no recovery.
    
    Taking a trap uses up your current item, blocking the correct path.
    Only lookahead can avoid the trap.
    """
    random.seed(config.seed)
    model = ActionModel()
    items = [f"item_{i}" for i in range(config.depth + 1)]
    goal_item = items[-1]
    
    actions_to_register = []
    
    # Main chain (correct path) - consumes current item
    for i in range(config.depth):
        actions_to_register.append(ActionSpec(
            action_id=f"path_a_{i}",
            description=f"Option A at level {i}",
            args_schema={},
            preconditions=frozenset([items[i]]),
            effects_add=frozenset([items[i + 1]]),
            effects_remove=frozenset([items[i]]),  # Consumes item!
            cost=2.0
        ))
    
    # Traps - also consume current item, but lead nowhere
    for i in range(config.depth - 1):
        for d in range(config.dead_ends):
            fake_item = f"fake_{i}_{d}"
            actions_to_register.append(ActionSpec(
                action_id=f"path_b_{i}_{d}",
                description=f"Option B at level {i}",
                args_schema={},
                preconditions=frozenset([items[i]]),
                effects_add=frozenset([fake_item]),
                effects_remove=frozenset([items[i]]),  # Consumes item!
                cost=0.5  # Looks better but dead end
            ))
            
            # Dead end continuation
            actions_to_register.append(ActionSpec(
                action_id=f"deadend_{i}_{d}",
                description=f"Continue B",
                args_schema={},
                preconditions=frozenset([fake_item]),
                effects_add=frozenset([f"stuck_{i}_{d}"]),
                effects_remove=frozenset([fake_item]),
                cost=0.5
            ))
    
    random.shuffle(actions_to_register)
    for spec in actions_to_register:
        model.register(spec)
    
    return model, State(goal=frozenset([goal_item]), inventory=frozenset([items[0]]))

def generate_multigoal_task(config: BenchmarkConfig) -> tuple[ActionModel, State]:
    """Generate task requiring multiple goals."""
    random.seed(config.seed)
    model = ActionModel()
    
    n_goals = min(3, config.depth)
    goals = [f"goal_{g}" for g in range(n_goals)]
    
    model.register(ActionSpec(
        action_id="get_key",
        description="Get shared key",
        args_schema={},
        preconditions=frozenset(["start"]),
        effects_add=frozenset(["key"]),
        cost=1.0
    ))
    
    for g, goal in enumerate(goals):
        chain_len = random.randint(2, config.depth // n_goals + 1)
        prev = "key"
        for step in range(chain_len):
            next_item = goal if step == chain_len - 1 else f"g{g}_step_{step}"
            model.register(ActionSpec(
                action_id=f"toward_{goal}_{step}",
                description=f"Step {step} toward {goal}",
                args_schema={},
                preconditions=frozenset([prev]),
                effects_add=frozenset([next_item]),
                cost=1.0
            ))
            prev = next_item
    
    for d in range(config.distractors):
        model.register(ActionSpec(
            action_id=f"waste_{d}",
            description=f"Waste action {d}",
            args_schema={},
            preconditions=frozenset(["start"]),
            effects_add=frozenset([f"junk_{d}"]),
            cost=0.5
        ))
    
    return model, State(goal=frozenset(goals), inventory=frozenset(["start"]))


def generate_costly_task(config: BenchmarkConfig) -> tuple[ActionModel, State]:
    """Generate task where cost optimization matters."""
    random.seed(config.seed)
    model = ActionModel()
    
    # Expensive short path (2 steps, cost 10 each)
    model.register(ActionSpec(
        action_id="expensive_1",
        description="Expensive step 1",
        args_schema={},
        preconditions=frozenset(["start"]),
        effects_add=frozenset(["exp_mid"]),
        cost=10.0
    ))
    model.register(ActionSpec(
        action_id="expensive_2",
        description="Expensive step 2",
        args_schema={},
        preconditions=frozenset(["exp_mid"]),
        effects_add=frozenset(["goal"]),
        cost=10.0
    ))
    
    # Cheap long path (depth steps, cost 1 each)
    items = ["start"] + [f"cheap_{i}" for i in range(config.depth - 1)] + ["goal"]
    for i in range(len(items) - 1):
        model.register(ActionSpec(
            action_id=f"cheap_{i}",
            description=f"Cheap step {i}",
            args_schema={},
            preconditions=frozenset([items[i]]),
            effects_add=frozenset([items[i + 1]]),
            cost=1.0
        ))
    
    return model, State(goal=frozenset(["goal"]), inventory=frozenset(["start"]))
def generate_partial_obs_task(config: BenchmarkConfig) -> tuple[ActionModel, State]:
    """Generate task with partial observability.
    
    Some actions reveal information needed to choose subsequent actions.
    """
    random.seed(config.seed)
    model = ActionModel()
    
    # Observation action reveals the correct path
    model.register(ActionSpec(
        action_id="observe",
        description="Reveal which path is correct",
        args_schema={},
        preconditions=frozenset(["start"]),
        effects_add=frozenset(["knowledge"]),
        cost=0.5
    ))
    
    # Two paths, only one leads to goal
    correct_path = random.randint(0, 1)
    for path in range(2):
        prereqs = frozenset(["start", "knowledge"]) if path == correct_path else frozenset(["start"])
        effects = frozenset(["goal_item"]) if path == correct_path else frozenset([f"dead_end_{path}"])
        model.register(ActionSpec(
            action_id=f"path_{path}",
            description=f"Take path {path}",
            args_schema={},
            preconditions=prereqs,
            effects_add=effects,
            cost=1.0
        ))
    
    initial_state = State(
        goal=frozenset(["goal_item"]),
        inventory=frozenset(["start"])
    )
    
    return model, initial_state

@dataclass 
class BenchmarkResult:
    """Results from running a benchmark suite."""
    success_rate: float
    avg_cost: float
    avg_steps: float
    avg_nodes: float
    invalid_executions: int  # Should always be 0
    
def run_benchmark(
    agent_factory,
    task_generator,
    n_tasks: int = 100,
    depth_range: tuple[int, int] = (2, 10)
) -> dict[int, BenchmarkResult]:
    """Run benchmark across difficulty levels."""
    results = {}
    
    for depth in range(depth_range[0], depth_range[1] + 1):
        successes, costs, steps, nodes = [], [], [], []
        invalid = 0
        
        for seed in range(n_tasks):
            config = BenchmarkConfig(depth=depth, seed=seed)
            model, state = task_generator(config)
            agent = agent_factory(model)
            episode = agent.solve(state, f"depth{depth}_seed{seed}")
            
            successes.append(episode.success)
            costs.append(episode.total_cost)
            steps.append(len(episode.events))
            nodes.append(episode.nodes_expanded)
            
            # Check invariant: no invalid executions
            # (In this framework, this is enforced by assertion)
        
        results[depth] = BenchmarkResult(
            success_rate=sum(successes) / n_tasks,
            avg_cost=sum(costs) / n_tasks,
            avg_steps=sum(steps) / n_tasks,
            avg_nodes=sum(nodes) / n_tasks,
            invalid_executions=invalid
        )
    
    return results
