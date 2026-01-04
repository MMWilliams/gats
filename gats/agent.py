"""GATS 2.0 Agent - Main search-based agent runtime.

Ultra-minimal logging - episode summaries only. Set GATS_LOG=1 to enable.
"""
from __future__ import annotations
import random
import logging
import json
import uuid
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Protocol, Any
from pathlib import Path

from .core import State, Candidate, Episode, Event, ActionSpec
from .verifier import ActionModel
from .search import MCTS, WorldModel


# =============================================================================
# MINIMAL LOGGING - disabled by default
# =============================================================================

_LOG_ENABLED = os.environ.get("GATS_LOG", "1") == "1"
_logger: logging.Logger | None = None

def _init_logger() -> logging.Logger:
    """Lazy init logger only if enabled."""
    global _logger
    if _logger is not None:
        return _logger
    
    _logger = logging.getLogger("gats.agent")
    _logger.setLevel(logging.INFO)
    
    if _logger.handlers:
        return _logger
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "agent_logs.jsonl"
    
    fh = logging.FileHandler(log_file, mode='w', encoding="utf-8")  # mode='w' overwrites
    fh.setFormatter(logging.Formatter())
    _logger.addHandler(fh)
    
    return _logger


def _log(data: dict[str, Any]) -> None:
    """Emit single compact log line. No-op if GATS_LOG!=1."""
    if not _LOG_ENABLED:
        return
    logger = _init_logger()
    logger.info(json.dumps(data, separators=(',', ':')))


# =============================================================================
# PROPOSER INTERFACES
# =============================================================================

class Proposer(Protocol):
    """Interface for action proposal (LLM or heuristic)."""
    
    def propose(
        self,
        state: State,
        legal_actions: list[ActionSpec],
        k: int = 5
    ) -> list[Candidate]:
        """Propose candidate actions for state."""
        ...
    
    def repair(self, candidate: Candidate, feedback: str) -> Candidate:
        """Attempt to repair invalid candidate."""
        ...


class HeuristicProposer:
    """Simple heuristic proposer (no LLM needed for benchmarks)."""
    
    def propose(
        self,
        state: State,
        legal_actions: list[ActionSpec],
        k: int = 5
    ) -> list[Candidate]:
        """Propose actions using goal-directed heuristic."""
        if not legal_actions:
            return []
        
        def score(spec: ActionSpec) -> float:
            new_items = spec.effects_add - state.inventory
            goal_items = spec.effects_add & state.goal
            already_have = spec.effects_add & state.inventory
            
            if not new_items and already_have:
                return -10.0
            
            goal_score = len(goal_items) * 20
            new_score = len(new_items)
            
            if not goal_items and spec.cost < 1.0:
                return new_score * 0.1 + random.random() * 0.01
            
            return goal_score + new_score - spec.cost * 0.1 + random.random() * 0.01
        
        sorted_actions = sorted(legal_actions, key=score, reverse=True)
        candidates = [
            Candidate(s.action_id, {}, prior=max(0.1, (score(s) + 10) / 30))
            for s in sorted_actions[:k]
        ]
        
        return candidates
    
    def repair(self, candidate: Candidate, feedback: str) -> Candidate:
        """No-op repair for heuristic proposer."""
        return candidate


class BlindProposer:
    """Proposer without oracle knowledge of action naming."""
    
    def propose(
        self,
        state: State,
        legal_actions: list[ActionSpec],
        k: int = 5
    ) -> list[Candidate]:
        """Propose actions using blind heuristic."""
        if not legal_actions:
            return []
        
        def score(spec: ActionSpec) -> float:
            new_items = spec.effects_add - state.inventory
            goal_items = spec.effects_add & state.goal
            already_have = spec.effects_add & state.inventory
            
            if not new_items and already_have:
                return -10.0
            
            goal_score = len(goal_items) * 20
            new_score = len(new_items)
            
            if not goal_items and spec.cost < 1.0:
                return new_score * 0.1 + random.random() * 0.01
            
            return goal_score + new_score - spec.cost * 0.1 + random.random() * 0.01
        
        sorted_actions = sorted(legal_actions, key=score, reverse=True)
        candidates = [
            Candidate(s.action_id, {}, prior=max(0.1, (score(s) + 10) / 30))
            for s in sorted_actions[:k]
        ]
        
        return candidates
    
    def repair(self, candidate: Candidate, feedback: str) -> Candidate:
        return candidate


class NaiveProposer:
    """Truly naive proposer - only sees: new items good, lower cost good.
    
    No trap detection. Used for fair ablation testing.
    """
    
    def propose(
        self,
        state: State,
        legal_actions: list[ActionSpec],
        k: int = 5
    ) -> list[Candidate]:
        """Propose actions with naive scoring (no trap detection)."""
        if not legal_actions:
            return []
        
        def score(spec: ActionSpec) -> float:
            new_items = spec.effects_add - state.inventory
            goal_items = spec.effects_add & state.goal
            
            if not new_items:
                return -10.0
            
            return len(goal_items) * 10 + len(new_items) - spec.cost + random.random() * 0.01
        
        sorted_actions = sorted(legal_actions, key=score, reverse=True)
        candidates = [
            Candidate(s.action_id, {}, prior=max(0.1, (score(s) + 5) / 15))
            for s in sorted_actions[:k]
        ]
        
        return candidates
    
    def repair(self, candidate: Candidate, feedback: str) -> Candidate:
        return candidate


class UnbiasedProposer:
    """Proposes all legal actions with equal prior. Lets MCTS decide."""
    
    def propose(
        self,
        state: State,
        legal_actions: list[ActionSpec],
        k: int = 10
    ) -> list[Candidate]:
        """Propose all legal actions with equal prior."""
        if not legal_actions:
            return []
        
        actions = list(legal_actions)
        random.shuffle(actions)
        return [Candidate(s.action_id, {}, prior=0.5) for s in actions[:k]]
    
    def repair(self, candidate: Candidate, feedback: str) -> Candidate:
        return candidate


# =============================================================================
# MAIN AGENT
# =============================================================================

class Agent:
    """Search-based agent with verification guarantee."""
    
    def __init__(
        self,
        action_model: ActionModel,
        proposer: Proposer | None = None,
        search_budget: int = 100,
        max_steps: int = 50
    ) -> None:
        """Initialize agent.
        
        Args:
            action_model: Ground-truth action model for verification
            proposer: Action proposal strategy (defaults to HeuristicProposer)
            search_budget: MCTS iterations per step
            max_steps: Maximum episode length
        """
        self.action_model = action_model
        self.proposer = proposer or HeuristicProposer()
        self.world_model = WorldModel(action_model)
        self.search_budget = search_budget
        self.max_steps = max_steps
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Run agent to solve task.
        
        Args:
            initial_state: Starting state with goal specification
            task_id: Identifier for logging correlation
            
        Returns:
            Episode containing execution trace and results
        """
        episode = Episode(task_id)
        state = initial_state
        goal_items = initial_state.goal
        
        def value_fn(s: State) -> float:
            """Compute value estimate for state."""
            if s.is_goal():
                return 1.0
            
            legal = self.action_model.get_legal_actions(s)
            if not legal:
                return 0.0
            
            can_progress = any(
                spec.effects_add - s.inventory
                for spec in legal
            )
            
            if not can_progress:
                return 0.0
            
            achieved = len(s.inventory & goal_items)
            return achieved / max(1, len(goal_items)) * 0.8 + 0.1
        
        mcts = MCTS(
            world_model=self.world_model.predict,
            value_fn=value_fn,
            max_depth=self.search_budget
        )
        mcts.set_action_model(self.action_model)
        
        for step in range(self.max_steps):
            if state.is_goal():
                episode.success = True
                break
            
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                break
            
            candidates = self.proposer.propose(state, legal, k=10)
            
            # Verify candidates
            verified = []
            for c in candidates:
                result = self.action_model.verify(c, state)
                if result.is_valid:
                    verified.append((c, result))
            
            if not verified:
                break
            
            # MCTS search
            action = mcts.search(state, [c for c, _ in verified], self.search_budget)
            
            if action is None:
                break
            
            # Final verification before execution
            result = self.action_model.verify(action, state)
            assert result.is_valid, "INVARIANT VIOLATED: executing unverified action"
            
            # Execute action
            state_before = state.hash()
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            state = state.with_inventory(new_inv)
            
            spec = self.action_model.resolve(action.action_id)
            cost = spec.cost if spec else 1.0
            
            event = Event(
                step=step,
                action_id=action.action_id,
                args=action.args,
                success=True,
                state_before=state_before,
                state_after=state.hash()
            )
            episode.events.append(event)
            episode.total_cost += cost
        
        episode.nodes_expanded = mcts.nodes_expanded
        
        # Single summary log for entire episode
        _log({"t": task_id, "ok": episode.success, "s": len(episode.events), 
              "c": round(episode.total_cost, 1), "n": episode.nodes_expanded})
        
        return episode


# =============================================================================
# BASELINE AGENTS
# =============================================================================

class GreedyAgent:
    """Baseline: greedy selection without search."""
    
    def __init__(self, action_model: ActionModel, max_steps: int = 50) -> None:
        self.action_model = action_model
        self.max_steps = max_steps
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Solve task using greedy action selection."""
        episode = Episode(task_id)
        state = initial_state
        
        for step in range(self.max_steps):
            if state.is_goal():
                episode.success = True
                break
            
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                break
            
            def score(spec: ActionSpec) -> float:
                new_items = spec.effects_add - state.inventory
                goal_items = spec.effects_add & state.goal
                is_distractor = "distract" in spec.action_id
                return len(goal_items) * 10 + (0 if is_distractor else len(new_items))
            
            best = max(legal, key=score)
            candidate = Candidate(best.action_id)
            result = self.action_model.verify(candidate, state)
            
            if not result.is_valid:
                break
            
            state_before = state.hash()
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            state = state.with_inventory(new_inv)
            
            episode.events.append(Event(
                step=step,
                action_id=best.action_id,
                args={},
                success=True,
                state_before=state_before,
                state_after=state.hash()
            ))
            episode.total_cost += best.cost
        
        _log({"t": task_id, "m": "greedy", "ok": episode.success, "s": len(episode.events), "c": round(episode.total_cost, 1)})
        
        return episode


class RandomAgent:
    """Baseline: random legal action selection."""
    
    def __init__(self, action_model: ActionModel, max_steps: int = 50) -> None:
        self.action_model = action_model
        self.max_steps = max_steps
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Solve task using random action selection."""
        episode = Episode(task_id)
        state = initial_state
        
        for step in range(self.max_steps):
            if state.is_goal():
                episode.success = True
                break
            
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                break
            
            spec = random.choice(legal)
            candidate = Candidate(spec.action_id)
            result = self.action_model.verify(candidate, state)
            
            if not result.is_valid:
                break
            
            state_before = state.hash()
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            state = state.with_inventory(new_inv)
            
            episode.events.append(Event(
                step=step,
                action_id=spec.action_id,
                args={},
                success=True,
                state_before=state_before,
                state_after=state.hash()
            ))
            episode.total_cost += spec.cost
        
        _log({"t": task_id, "m": "random", "ok": episode.success, "s": len(episode.events)})
        
        return episode


class CostAwareAgent:
    """Agent that optimizes for minimal cost."""
    
    def __init__(
        self,
        action_model: ActionModel,
        search_budget: int = 100,
        max_steps: int = 50,
        cost_weight: float = 0.5
    ) -> None:
        self.action_model = action_model
        self.world_model = WorldModel(action_model)
        self.search_budget = search_budget
        self.max_steps = max_steps
        self.cost_weight = cost_weight
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Solve task with cost-aware planning."""
        episode = Episode(task_id)
        state = initial_state
        goal_items = initial_state.goal
        
        def value_fn(s: State) -> float:
            if s.is_goal():
                return 1.0
            return len(s.inventory & goal_items) / max(1, len(goal_items)) * 0.9
        
        def score_action(spec: ActionSpec) -> float:
            goal_produced = spec.effects_add & goal_items
            new = spec.effects_add - state.inventory
            return len(goal_produced) * 10 + len(new) - spec.cost * self.cost_weight
        
        mcts = MCTS(world_model=self.world_model.predict, value_fn=value_fn)
        mcts.set_action_model(self.action_model)
        
        for step in range(self.max_steps):
            if state.is_goal():
                episode.success = True
                break
            
            legal = self.action_model.get_legal_actions(state)
            if not legal:
                break
            
            sorted_legal = sorted(legal, key=score_action, reverse=True)[:5]
            candidates = [
                Candidate(s.action_id, {}, prior=max(0.1, score_action(s) / 10))
                for s in sorted_legal
            ]
            
            verified = [(c, self.action_model.verify(c, state)) for c in candidates]
            verified = [(c, r) for c, r in verified if r.is_valid]
            
            if not verified:
                break
            
            action = mcts.search(state, [c for c, _ in verified], self.search_budget)
            if action is None:
                break
            
            result = self.action_model.verify(action, state)
            assert result.is_valid
            
            state_before = state.hash()
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            state = state.with_inventory(new_inv)
            
            spec = self.action_model.resolve(action.action_id)
            cost = spec.cost if spec else 1.0
            
            episode.events.append(Event(
                step=step,
                action_id=action.action_id,
                args=action.args,
                success=True,
                state_before=state_before,
                state_after=state.hash()
            ))
            episode.total_cost += cost
        
        episode.nodes_expanded = mcts.nodes_expanded
        
        _log({"t": task_id, "m": "cost", "ok": episode.success, "s": len(episode.events), 
              "c": round(episode.total_cost, 1), "n": episode.nodes_expanded})
        
        return episode
    

# =============================================================================
# LATS AGENT (Language Agent Tree Search)
# =============================================================================

class LATSAgent:
    """LATS: Tree search with reflection and backtracking.
    
    Key differences from GATS:
    - Uses reflection to generate verbal feedback on failed paths
    - Backtracking based on reflection quality
    - No formal verification - relies on simulation
    """
    
    def __init__(
        self,
        action_model: ActionModel,
        search_budget: int = 100,
        max_steps: int = 50,
        n_samples: int = 5,
        max_depth: int = 10
    ) -> None:
        self.action_model = action_model
        self.world_model = WorldModel(action_model)
        self.search_budget = search_budget
        self.max_steps = max_steps
        self.n_samples = n_samples
        self.max_depth = max_depth
    
    # def _reflect(self, trajectory: list[tuple[State, Candidate]], final_state: State, goal: frozenset[str]) -> float:
    #     """Simulate LLM reflection on trajectory quality."""
    #     if final_state.is_goal():
    #         return 1.0
        
    #     # Multi-goal aware: track progress toward ALL goals
    #     goals_achieved = len(final_state.inventory & goal)
    #     total_goals = len(goal)
    #     goal_progress = goals_achieved / max(1, total_goals)
        
    #     # Penalize dead-ends
    #     legal = self.action_model.get_legal_actions(final_state)
    #     if not legal:
    #         return 0.05 + goal_progress * 0.1
        
    #     # Reward new items that could lead to goals
    #     new_items = len(final_state.inventory)
        
    #     return min(0.95, goal_progress * 0.7 + 0.2 + new_items * 0.01)
    def _reflect(self, trajectory, final_state, goal) -> tuple[str, float]:
        """Use LLM to generate verbal reflection and score."""
        prompt = self._build_reflection_prompt(trajectory, final_state, goal)
        response = self.llm.generate(prompt)
        return response.text, self._parse_score(response)
    def _expand_node(
        self, 
        state: State, 
        trajectory: list[tuple[State, Candidate]]
    ) -> list[tuple[State, Candidate, float]]:
        """Expand node by sampling actions with goal-directed bias."""
        legal = self.action_model.get_legal_actions(state)
        if not legal:
            return []
        
        # Score actions for sampling (goal-directed but not perfect)
        def sample_score(spec: ActionSpec) -> float:
            goal_items = spec.effects_add & state.goal
            new_items = spec.effects_add - state.inventory
            return len(goal_items) * 5 + len(new_items) + random.random()
        
        # Sample with bias toward promising actions
        scored = [(spec, sample_score(spec)) for spec in legal]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Take top actions + some random exploration
        n_top = min(self.n_samples // 2 + 1, len(scored))
        n_random = min(self.n_samples - n_top, len(scored) - n_top)
        
        sampled = [s[0] for s in scored[:n_top]]
        if n_random > 0 and len(scored) > n_top:
            remaining = [s[0] for s in scored[n_top:]]
            sampled.extend(random.sample(remaining, n_random))
        
        expansions = []
        for spec in sampled:
            cand = Candidate(spec.action_id, {})
            next_state, prob = self.world_model.predict(state, cand)
            if prob > 0:
                new_traj = trajectory + [(state, cand)]
                score = self._reflect(new_traj, next_state, state.goal)
                expansions.append((next_state, cand, score))
        
        return sorted(expansions, key=lambda x: x[2], reverse=True)
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Solve using LATS tree search with reflection."""
        episode = Episode(task_id)
        
        import heapq
        # Priority queue: (negative_score, counter, state_hash, state, trajectory)
        frontier: list[tuple[float, int, str, State, list[tuple[State, Candidate]]]] = []
        counter = 0
        heapq.heappush(frontier, (0.0, counter, initial_state.hash(), initial_state, []))
        
        visited: set[str] = set()
        best_trajectory: list[tuple[State, Candidate]] = []
        best_score = -float('inf')
        best_state = initial_state
        goal_found = False
        nodes_expanded = 0
        
        for _ in range(self.search_budget):
            if not frontier or goal_found:
                break
            
            neg_score, _, _, state, trajectory = heapq.heappop(frontier)
            state_hash = state.hash()
            
            if state_hash in visited:
                continue
            visited.add(state_hash)
            nodes_expanded += 1
            
            # Track best trajectory by progress toward goal
            progress_score = len(state.inventory & initial_state.goal) * 10 + len(trajectory)
            if progress_score > best_score or state.is_goal():
                best_score = progress_score
                best_trajectory = trajectory
                best_state = state
            
            if state.is_goal():
                goal_found = True
                break
            
            if len(trajectory) >= self.max_depth:
                continue
            
            expansions = self._expand_node(state, trajectory)
            
            for next_state, cand, score in expansions:
                if next_state.hash() not in visited:
                    counter += 1
                    new_traj = trajectory + [(state, cand)]
                    heapq.heappush(frontier, (-score, counter, next_state.hash(), next_state, new_traj))
        
        # If no complete trajectory found, use iterative deepening from best state
        if not goal_found and best_state != initial_state:
            # Continue from best state found
            current_state = best_state
            current_traj = best_trajectory
            
            for _ in range(self.max_steps - len(current_traj)):
                if current_state.is_goal():
                    break
                
                legal = self.action_model.get_legal_actions(current_state)
                if not legal:
                    break
                
                # Greedy selection from current state
                def greedy_score(spec: ActionSpec) -> float:
                    # Prioritize actions that produce goal items we don't have yet
                    remaining_goals = initial_state.goal - current_state.inventory
                    goal_items = spec.effects_add & remaining_goals
                    new_items = spec.effects_add - current_state.inventory
                    return len(goal_items) * 20 + len(new_items) + random.random() * 0.1
                
                best_spec = max(legal, key=greedy_score)
                cand = Candidate(best_spec.action_id, {})
                next_state, prob = self.world_model.predict(current_state, cand)
                
                if prob > 0:
                    current_traj = current_traj + [(current_state, cand)]
                    current_state = next_state
                else:
                    break
            
            best_trajectory = current_traj
        
        # Execute best trajectory found
        state = initial_state
        for step, (_, cand) in enumerate(best_trajectory):
            result = self.action_model.verify(cand, state)
            if not result.is_valid:
                break
            
            state_before = state.hash()
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            state = state.with_inventory(new_inv)
            
            spec = self.action_model.resolve(cand.action_id)
            cost = spec.cost if spec else 1.0
            
            episode.events.append(Event(
                step=step,
                action_id=cand.action_id,
                args=cand.args,
                success=True,
                state_before=state_before,
                state_after=state.hash()
            ))
            episode.total_cost += cost
        
        episode.success = state.is_goal()
        episode.nodes_expanded = nodes_expanded
        
        _log({"t": task_id, "m": "lats", "ok": episode.success, 
              "s": len(episode.events), "c": round(episode.total_cost, 1), 
              "n": nodes_expanded})
        
        return episode

# =============================================================================
# REACT AGENT (Reasoning + Acting)
# =============================================================================

class ReActAgent:
    """ReAct: Interleaved reasoning and acting without tree search.
    
    Key differences from GATS:
    - No lookahead/tree search
    - Single-step reasoning before each action
    - No backtracking capability
    """
    
    def __init__(
        self,
        action_model: ActionModel,
        max_steps: int = 50,
        reasoning_quality: float = 0.7  # Simulates LLM reasoning accuracy
    ) -> None:
        self.action_model = action_model
        self.max_steps = max_steps
        self.reasoning_quality = reasoning_quality
    
    def _reason_and_act(self, state: State) -> Candidate | None:
        """Simulate ReAct's reason-then-act pattern."""
        legal = self.action_model.get_legal_actions(state)
        if not legal:
            return None
        
        # Simulate reasoning: score actions based on goal proximity
        def score(spec: ActionSpec) -> float:
            goal_items = spec.effects_add & state.goal
            new_items = spec.effects_add - state.inventory
            loses_items = spec.effects_remove & state.inventory
            
            base_score = len(goal_items) * 10 + len(new_items) - len(loses_items) * 5
            
            # ReAct doesn't do lookahead - can be fooled by low-cost traps
            if spec.cost < 1.0 and new_items:
                base_score += 2  # Biased toward cheap actions
            
            return base_score + random.random() * 0.1
        
        # With reasoning_quality probability, pick best; otherwise random
        if random.random() < self.reasoning_quality:
            best = max(legal, key=score)
        else:
            # Reasoning failure - pick suboptimally
            sorted_actions = sorted(legal, key=score, reverse=True)
            idx = min(len(sorted_actions) - 1, random.randint(0, 2))
            best = sorted_actions[idx]
        
        return Candidate(best.action_id, {})
    
    def solve(self, initial_state: State, task_id: str = "task") -> Episode:
        """Solve using ReAct pattern."""
        episode = Episode(task_id)
        state = initial_state
        
        for step in range(self.max_steps):
            if state.is_goal():
                episode.success = True
                break
            
            candidate = self._reason_and_act(state)
            if candidate is None:
                break
            
            result = self.action_model.verify(candidate, state)
            if not result.is_valid:
                # ReAct doesn't guarantee valid actions
                continue
            
            state_before = state.hash()
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            state = state.with_inventory(new_inv)
            
            spec = self.action_model.resolve(candidate.action_id)
            cost = spec.cost if spec else 1.0
            
            episode.events.append(Event(
                step=step,
                action_id=candidate.action_id,
                args=candidate.args,
                success=True,
                state_before=state_before,
                state_after=state.hash()
            ))
            episode.total_cost += cost
        
        _log({"t": task_id, "m": "react", "ok": episode.success, 
              "s": len(episode.events), "c": round(episode.total_cost, 1)})
        
        return episode
