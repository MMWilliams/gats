"""GATS 2.0 Event Log - Source of Truth for Reproducibility.

Structured event logging with:
- Immutable event schema
- Replay validation
- Log-based learning integration
"""
from __future__ import annotations
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, Callable

from .core import State, Candidate, ActionSpec, Episode, Event


# =============================================================================
# Event Schema
# =============================================================================

@dataclass(frozen=True)
class LogEvent:
    """Immutable event record - source of truth."""
    ts: float                          # Unix timestamp
    task_id: str                       # Task correlation ID
    step: int                          # Step number in episode
    action_id: str                     # Action taken
    args: tuple[tuple[str, Any], ...]  # Args as sorted tuple for hashing
    state_before: tuple[str, ...]      # Inventory before (sorted)
    state_after: tuple[str, ...]       # Inventory after (sorted)
    goal: tuple[str, ...]              # Goal items (sorted)
    success: bool                      # Action succeeded
    cost: float                        # Action cost
    layer_used: int = 0                # World model layer (1/2/3)
    confidence: float = 1.0            # Prediction confidence
    
    @classmethod
    def from_execution(
        cls,
        task_id: str,
        step: int,
        action: Candidate,
        state_before: State,
        state_after: State,
        success: bool,
        cost: float,
        layer_used: int = 0,
        confidence: float = 1.0
    ) -> "LogEvent":
        return cls(
            ts=time.time(),
            task_id=task_id,
            step=step,
            action_id=action.action_id,
            args=tuple(sorted(action.args.items())),
            state_before=tuple(sorted(state_before.inventory)),
            state_after=tuple(sorted(state_after.inventory)),
            goal=tuple(sorted(state_before.goal)),
            success=success,
            cost=cost,
            layer_used=layer_used,
            confidence=confidence
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "task_id": self.task_id,
            "step": self.step,
            "action_id": self.action_id,
            "args": dict(self.args),
            "state_before": list(self.state_before),
            "state_after": list(self.state_after),
            "goal": list(self.goal),
            "success": self.success,
            "cost": self.cost,
            "layer_used": self.layer_used,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogEvent":
        return cls(
            ts=d["ts"],
            task_id=d["task_id"],
            step=d["step"],
            action_id=d["action_id"],
            args=tuple(sorted(d.get("args", {}).items())),
            state_before=tuple(sorted(d.get("state_before", []))),
            state_after=tuple(sorted(d.get("state_after", []))),
            goal=tuple(sorted(d.get("goal", []))),
            success=d.get("success", True),
            cost=d.get("cost", 1.0),
            layer_used=d.get("layer_used", 0),
            confidence=d.get("confidence", 1.0)
        )
    
    def hash(self) -> str:
        """Deterministic hash for integrity verification."""
        data = (self.task_id, self.step, self.action_id, self.args,
                self.state_before, self.state_after, self.success)
        return hashlib.sha256(str(data).encode()).hexdigest()[:16]


@dataclass
class EpisodeSummary:
    """Summary of completed episode."""
    task_id: str
    success: bool
    n_steps: int
    total_cost: float
    duration_s: float
    events: list[LogEvent] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "n_steps": self.n_steps,
            "total_cost": self.total_cost,
            "duration_s": self.duration_s,
            "event_hashes": [e.hash() for e in self.events]
        }


# =============================================================================
# Event Logger
# =============================================================================

class EventLog:
    """Append-only event log with integrity checking."""
    
    def __init__(self, path: Path | str | None = None, buffer_size: int = 100):
        self.path = Path(path) if path else None
        self.buffer_size = buffer_size
        self._buffer: list[LogEvent] = []
        self._episode_events: dict[str, list[LogEvent]] = {}
        self._episode_starts: dict[str, float] = {}
        
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_episode(self, task_id: str) -> None:
        """Mark episode start for duration tracking."""
        self._episode_starts[task_id] = time.time()
        self._episode_events[task_id] = []
    
    def log(self, event: LogEvent) -> None:
        """Append event to log."""
        self._buffer.append(event)
        
        if event.task_id in self._episode_events:
            self._episode_events[event.task_id].append(event)
        
        if len(self._buffer) >= self.buffer_size:
            self.flush()
    
    def end_episode(self, task_id: str, success: bool) -> EpisodeSummary:
        """Finalize episode and return summary."""
        events = self._episode_events.pop(task_id, [])
        start = self._episode_starts.pop(task_id, time.time())
        
        summary = EpisodeSummary(
            task_id=task_id,
            success=success,
            n_steps=len(events),
            total_cost=sum(e.cost for e in events),
            duration_s=time.time() - start,
            events=events
        )
        
        # Log summary
        if self.path:
            summary_path = self.path.with_suffix('.summaries.jsonl')
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(summary.to_dict()) + '\n')
        
        return summary
    
    def flush(self) -> None:
        """Write buffer to disk."""
        if not self.path or not self._buffer:
            return
        
        with open(self.path, 'a', encoding='utf-8') as f:
            for event in self._buffer:
                f.write(json.dumps(event.to_dict()) + '\n')
        
        self._buffer.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.flush()


# =============================================================================
# Log Reader & Replay
# =============================================================================

class LogReader:
    """Read and replay event logs."""
    
    def __init__(self, path: Path | str):
        self.path = Path(path)
    
    def read_events(self) -> Iterator[LogEvent]:
        """Stream events from log file."""
        if not self.path.exists():
            return
        
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    yield LogEvent.from_dict(json.loads(line.strip()))
                except (json.JSONDecodeError, KeyError):
                    continue
    
    def read_episodes(self) -> Iterator[list[LogEvent]]:
        """Group events by episode."""
        episodes: dict[str, list[LogEvent]] = {}
        
        for event in self.read_events():
            episodes.setdefault(event.task_id, []).append(event)
        
        for events in episodes.values():
            yield sorted(events, key=lambda e: e.step)
    
    def count_events(self) -> int:
        """Count total events in log."""
        return sum(1 for _ in self.read_events())


def validate_replay(
    events: list[LogEvent],
    action_model,
    initial_inventory: frozenset[str] | None = None
) -> tuple[bool, list[str]]:
    """
    Validate that logged events can be replayed.
    
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    if not events:
        return True, []
    
    # Reconstruct initial state
    inv = frozenset(events[0].state_before) if initial_inventory is None else initial_inventory
    goal = frozenset(events[0].goal)
    state = State(goal=goal, inventory=inv)
    
    for i, event in enumerate(events):
        # Check state consistency
        expected_inv = tuple(sorted(state.inventory))
        if event.state_before != expected_inv:
            errors.append(f"Step {i}: state mismatch. Expected {expected_inv}, got {event.state_before}")
        
        # Verify action was legal
        cand = Candidate(event.action_id, dict(event.args))
        result = action_model.verify(cand, state)
        
        if not result.is_valid and event.success:
            errors.append(f"Step {i}: action {event.action_id} marked success but verification failed")
        
        # Apply effects
        if event.success:
            new_inv = (state.inventory | result.compiled_effects_add) - result.compiled_effects_remove
            state = state.with_inventory(new_inv)
            
            # Check post-state
            actual_after = tuple(sorted(state.inventory))
            if event.state_after != actual_after:
                errors.append(f"Step {i}: post-state mismatch. Expected {event.state_after}, got {actual_after}")
    
    return len(errors) == 0, errors


# =============================================================================
# Log-based Learning Integration
# =============================================================================

def extract_transitions(events: Iterator[LogEvent]):
    """Extract state transitions for Layer 2 learning."""
    from .world_model import TransitionRecord
    
    for event in events:
        if event.success:
            yield TransitionRecord(
                state_before=frozenset(event.state_before),
                action_id=event.action_id,
                args=dict(event.args),
                state_after=frozenset(event.state_after),
                success=True,
                cost=event.cost,
                timestamp=event.ts
            )


def train_from_logs(world_model, log_paths: list[Path]) -> dict[str, int]:
    """
    Train Layer 2 world model from historical logs.
    
    Returns:
        Statistics about training
    """
    stats = {"files": 0, "events": 0, "transitions": 0}
    
    for path in log_paths:
        if not path.exists():
            continue
        
        stats["files"] += 1
        reader = LogReader(path)
        
        for event in reader.read_events():
            stats["events"] += 1
            
            if event.success:
                from .world_model import TransitionRecord
                world_model.record_transition(TransitionRecord(
                    state_before=frozenset(event.state_before),
                    action_id=event.action_id,
                    args=dict(event.args),
                    state_after=frozenset(event.state_after),
                    success=True,
                    cost=event.cost,
                    timestamp=event.ts
                ))
                stats["transitions"] += 1
    
    return stats


# =============================================================================
# Convenience Functions
# =============================================================================

def create_logger(name: str = "gats", log_dir: str = "logs") -> EventLog:
    """Create event logger with timestamped filename."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{name}_{timestamp}.jsonl"
    return EventLog(path)


def merge_logs(log_paths: list[Path], output_path: Path) -> int:
    """Merge multiple log files into one."""
    n = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for path in log_paths:
            for event in LogReader(path).read_events():
                out.write(json.dumps(event.to_dict()) + '\n')
                n += 1
    return n