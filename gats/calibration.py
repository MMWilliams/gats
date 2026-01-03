"""GATS 2.0 Calibration Metrics.

Measures how well predicted confidence matches actual accuracy:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams
- Per-layer calibration analysis
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Sequence
from collections import defaultdict


# =============================================================================
# Core Metrics
# =============================================================================

@dataclass
class Prediction:
    """Single prediction with confidence and outcome."""
    confidence: float  # Predicted probability [0, 1]
    correct: bool      # Actual outcome
    layer: int = 0     # Which layer made prediction (1/2/3)
    action_id: str = ""


@dataclass
class CalibrationBin:
    """Statistics for one confidence bin."""
    lower: float
    upper: float
    n_samples: int = 0
    n_correct: int = 0
    sum_confidence: float = 0.0
    
    @property
    def accuracy(self) -> float:
        return self.n_correct / max(1, self.n_samples)
    
    @property
    def avg_confidence(self) -> float:
        return self.sum_confidence / max(1, self.n_samples)
    
    @property
    def gap(self) -> float:
        """Calibration gap: |accuracy - confidence|"""
        return abs(self.accuracy - self.avg_confidence)


def compute_ece(predictions: Sequence[Prediction], n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    
    ECE = Σ (n_b / N) * |acc(b) - conf(b)|
    
    Lower is better. 0 = perfectly calibrated.
    """
    if not predictions:
        return 0.0
    
    bins = [CalibrationBin(i/n_bins, (i+1)/n_bins) for i in range(n_bins)]
    
    for p in predictions:
        idx = min(int(p.confidence * n_bins), n_bins - 1)
        bins[idx].n_samples += 1
        bins[idx].sum_confidence += p.confidence
        if p.correct:
            bins[idx].n_correct += 1
    
    n = len(predictions)
    return sum((b.n_samples / n) * b.gap for b in bins if b.n_samples > 0)


def compute_mce(predictions: Sequence[Prediction], n_bins: int = 10) -> float:
    """
    Maximum Calibration Error.
    
    MCE = max_b |acc(b) - conf(b)|
    
    Worst-case calibration gap.
    """
    if not predictions:
        return 0.0
    
    bins = [CalibrationBin(i/n_bins, (i+1)/n_bins) for i in range(n_bins)]
    
    for p in predictions:
        idx = min(int(p.confidence * n_bins), n_bins - 1)
        bins[idx].n_samples += 1
        bins[idx].sum_confidence += p.confidence
        if p.correct:
            bins[idx].n_correct += 1
    
    return max((b.gap for b in bins if b.n_samples > 0), default=0.0)


def compute_brier(predictions: Sequence[Prediction]) -> float:
    """
    Brier Score.
    
    Brier = (1/N) Σ (confidence - outcome)²
    
    Lower is better. 0 = perfect predictions.
    """
    if not predictions:
        return 0.0
    
    return sum((p.confidence - float(p.correct)) ** 2 for p in predictions) / len(predictions)


def compute_log_loss(predictions: Sequence[Prediction], eps: float = 1e-15) -> float:
    """
    Log Loss (Binary Cross-Entropy).
    
    More sensitive to confident wrong predictions.
    """
    if not predictions:
        return 0.0
    
    total = 0.0
    for p in predictions:
        conf = max(eps, min(1 - eps, p.confidence))
        if p.correct:
            total -= math.log(conf)
        else:
            total -= math.log(1 - conf)
    
    return total / len(predictions)


# =============================================================================
# Reliability Diagram
# =============================================================================

@dataclass
class ReliabilityDiagram:
    """Data for plotting reliability diagram."""
    bin_edges: list[float]
    accuracies: list[float]
    confidences: list[float]
    counts: list[int]
    ece: float
    mce: float
    brier: float
    
    def to_dict(self) -> dict:
        return {
            "bin_edges": self.bin_edges,
            "accuracies": self.accuracies,
            "confidences": self.confidences,
            "counts": self.counts,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier
        }
    
    def __str__(self) -> str:
        lines = [
            f"ECE: {self.ece:.4f}  MCE: {self.mce:.4f}  Brier: {self.brier:.4f}",
            "",
            "Confidence |  Accuracy  |  Gap   |   N",
            "-" * 45
        ]
        
        for i in range(len(self.accuracies)):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            acc, conf, n = self.accuracies[i], self.confidences[i], self.counts[i]
            gap = abs(acc - conf)
            bar = "█" * int(acc * 10) if n > 0 else ""
            lines.append(f"[{lo:.1f}-{hi:.1f}] | {acc:>5.1%} {bar:<10} | {gap:>5.2f} | {n:>4}")
        
        return "\n".join(lines)


def reliability_diagram(predictions: Sequence[Prediction], n_bins: int = 10) -> ReliabilityDiagram:
    """Compute reliability diagram data."""
    bins = [CalibrationBin(i/n_bins, (i+1)/n_bins) for i in range(n_bins)]
    
    for p in predictions:
        idx = min(int(p.confidence * n_bins), n_bins - 1)
        bins[idx].n_samples += 1
        bins[idx].sum_confidence += p.confidence
        if p.correct:
            bins[idx].n_correct += 1
    
    return ReliabilityDiagram(
        bin_edges=[b.lower for b in bins] + [1.0],
        accuracies=[b.accuracy for b in bins],
        confidences=[b.avg_confidence for b in bins],
        counts=[b.n_samples for b in bins],
        ece=compute_ece(predictions, n_bins),
        mce=compute_mce(predictions, n_bins),
        brier=compute_brier(predictions)
    )


# =============================================================================
# Per-Layer Analysis
# =============================================================================

@dataclass
class LayerCalibration:
    """Calibration metrics for one world model layer."""
    layer: int
    n_predictions: int
    accuracy: float
    avg_confidence: float
    ece: float
    brier: float
    
    @property
    def overconfident(self) -> bool:
        return self.avg_confidence > self.accuracy + 0.05
    
    @property
    def underconfident(self) -> bool:
        return self.avg_confidence < self.accuracy - 0.05


def calibration_by_layer(predictions: Sequence[Prediction]) -> dict[int, LayerCalibration]:
    """Compute calibration metrics per world model layer."""
    by_layer: dict[int, list[Prediction]] = defaultdict(list)
    
    for p in predictions:
        by_layer[p.layer].append(p)
    
    results = {}
    for layer, preds in by_layer.items():
        n = len(preds)
        correct = sum(1 for p in preds if p.correct)
        
        results[layer] = LayerCalibration(
            layer=layer,
            n_predictions=n,
            accuracy=correct / max(1, n),
            avg_confidence=sum(p.confidence for p in preds) / max(1, n),
            ece=compute_ece(preds),
            brier=compute_brier(preds)
        )
    
    return results


def print_layer_calibration(layer_cal: dict[int, LayerCalibration]) -> None:
    """Print layer calibration summary."""
    print(f"{'Layer':<8} {'N':>8} {'Acc':>8} {'Conf':>8} {'ECE':>8} {'Status':<15}")
    print("-" * 60)
    
    for layer in sorted(layer_cal.keys()):
        c = layer_cal[layer]
        status = "overconfident" if c.overconfident else ("underconfident" if c.underconfident else "calibrated")
        name = {1: "Exact", 2: "Learned", 3: "Generative"}.get(layer, f"L{layer}")
        print(f"{name:<8} {c.n_predictions:>8} {c.accuracy:>7.1%} {c.avg_confidence:>7.1%} {c.ece:>7.3f}  {status}")


# =============================================================================
# Calibration from Event Logs
# =============================================================================

def predictions_from_events(events, ground_truth_fn=None) -> list[Prediction]:
    """
    Extract predictions from event log.
    
    Args:
        events: Iterable of LogEvent
        ground_truth_fn: Optional function to determine correctness
                        (default: use event.success)
    """
    predictions = []
    
    for event in events:
        correct = ground_truth_fn(event) if ground_truth_fn else event.success
        predictions.append(Prediction(
            confidence=event.confidence,
            correct=correct,
            layer=event.layer_used,
            action_id=event.action_id
        ))
    
    return predictions


def calibration_report(predictions: Sequence[Prediction], n_bins: int = 10) -> str:
    """Generate full calibration report."""
    lines = ["=" * 60, "CALIBRATION REPORT", "=" * 60, ""]
    
    # Overall metrics
    n = len(predictions)
    correct = sum(1 for p in predictions if p.correct)
    lines.append(f"Total predictions: {n}")
    lines.append(f"Overall accuracy:  {correct/max(1,n):.1%}")
    lines.append(f"Avg confidence:    {sum(p.confidence for p in predictions)/max(1,n):.1%}")
    lines.append("")
    
    # Reliability diagram
    diagram = reliability_diagram(predictions, n_bins)
    lines.append(str(diagram))
    lines.append("")
    
    # Per-layer breakdown
    layer_cal = calibration_by_layer(predictions)
    if len(layer_cal) > 1:
        lines.append("-" * 60)
        lines.append("PER-LAYER CALIBRATION")
        lines.append("-" * 60)
        
        for layer in sorted(layer_cal.keys()):
            c = layer_cal[layer]
            name = {1: "L1 (Exact)", 2: "L2 (Learned)", 3: "L3 (Generative)"}.get(layer, f"Layer {layer}")
            status = "⚠️ overconf" if c.overconfident else ("⚠️ underconf" if c.underconfident else "✓")
            lines.append(f"  {name}: n={c.n_predictions}, acc={c.accuracy:.1%}, conf={c.avg_confidence:.1%}, ECE={c.ece:.3f} {status}")
    
    return "\n".join(lines)


# =============================================================================
# Calibration Tracker (Online)
# =============================================================================

class CalibrationTracker:
    """Track calibration metrics online during execution."""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._predictions: list[Prediction] = []
    
    def record(self, confidence: float, correct: bool, layer: int = 0, action_id: str = "") -> None:
        """Record a single prediction."""
        self._predictions.append(Prediction(confidence, correct, layer, action_id))
    
    @property
    def n(self) -> int:
        return len(self._predictions)
    
    @property
    def ece(self) -> float:
        return compute_ece(self._predictions, self.n_bins)
    
    @property
    def brier(self) -> float:
        return compute_brier(self._predictions)
    
    @property
    def accuracy(self) -> float:
        if not self._predictions:
            return 0.0
        return sum(1 for p in self._predictions if p.correct) / len(self._predictions)
    
    def diagram(self) -> ReliabilityDiagram:
        return reliability_diagram(self._predictions, self.n_bins)
    
    def by_layer(self) -> dict[int, LayerCalibration]:
        return calibration_by_layer(self._predictions)
    
    def report(self) -> str:
        return calibration_report(self._predictions, self.n_bins)
    
    def reset(self) -> None:
        self._predictions.clear()


# =============================================================================
# Temperature Scaling (Post-hoc Calibration)
# =============================================================================

def find_temperature(predictions: Sequence[Prediction], lr: float = 0.01, n_iter: int = 100) -> float:
    """
    Find optimal temperature for calibration via temperature scaling.
    
    Softens/sharpens confidence: p' = sigmoid(logit(p) / T)
    
    Returns optimal temperature T (T > 1 = soften, T < 1 = sharpen)
    """
    if not predictions:
        return 1.0
    
    def logit(p: float) -> float:
        p = max(1e-7, min(1 - 1e-7, p))
        return math.log(p / (1 - p))
    
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))
    
    def scaled_nll(T: float) -> float:
        total = 0.0
        for p in predictions:
            scaled = sigmoid(logit(p.confidence) / T)
            scaled = max(1e-7, min(1 - 1e-7, scaled))
            if p.correct:
                total -= math.log(scaled)
            else:
                total -= math.log(1 - scaled)
        return total / len(predictions)
    
    # Grid search + refinement
    best_T, best_nll = 1.0, scaled_nll(1.0)
    
    for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        nll = scaled_nll(T)
        if nll < best_nll:
            best_T, best_nll = T, nll
    
    # Gradient descent refinement
    T = best_T
    for _ in range(n_iter):
        eps = 0.001
        grad = (scaled_nll(T + eps) - scaled_nll(T - eps)) / (2 * eps)
        T -= lr * grad
        T = max(0.1, min(10.0, T))
    
    return T


def apply_temperature(predictions: Sequence[Prediction], T: float) -> list[Prediction]:
    """Apply temperature scaling to predictions."""
    def logit(p: float) -> float:
        p = max(1e-7, min(1 - 1e-7, p))
        return math.log(p / (1 - p))
    
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))
    
    return [
        Prediction(
            confidence=sigmoid(logit(p.confidence) / T),
            correct=p.correct,
            layer=p.layer,
            action_id=p.action_id
        )
        for p in predictions
    ]