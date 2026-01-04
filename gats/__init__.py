"""GATS 2.0 - Graph-Augmented Tree Search for Agents."""
from .core import ActionSpec, State, Candidate, VerificationResult, Episode, Event, FailCode, ActionModel, WorldModel
from .verifier import ActionModel
from .search import MCTS, WorldModel
from .agent import Agent, GreedyAgent, RandomAgent, CostAwareAgent, LATSAgent, ReActAgent, HeuristicProposer, BlindProposer, NaiveProposer, UnbiasedProposer
# Add to existing __all__ and imports:
from .event_log import EventLog, LogEvent, LogReader, validate_replay, train_from_logs
from .calibration import (
    Prediction, CalibrationTracker, compute_ece, compute_brier,
    reliability_diagram, calibration_by_layer, calibration_report
)
from .llm import (
    LLMPredictor, OllamaPredictor, VLLMPredictor, HFPredictor,
    CachedPredictor, create_predictor, LLMConfig, LLMResponse
)
__all__ = [
    "ActionSpec", "State", "Candidate", "VerificationResult", "Episode", "Event", "FailCode",
    "ActionModel", "MCTS", "WorldModel",
    "Agent", "GreedyAgent", "RandomAgent", "CostAwareAgent", "LATSAgent", "ReActAgent",
    "HeuristicProposer", "BlindProposer", "NaiveProposer","UnbiasedProposer"
]