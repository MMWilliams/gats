# analysis/statistics.py

"""
Statistical Analysis for GATS Experiments

Provides:
- Confidence intervals (bootstrap and analytical)
- Significance tests (paired t-test, Wilcoxon)
- Effect size calculations (Cohen's d)
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Sequence


@dataclass
class StatisticalResult:
    """Result of a statistical comparison."""
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool  # at alpha=0.05
    
    def to_latex(self, metric_name: str = "") -> str:
        """Format for LaTeX table."""
        sig_marker = "*" if self.significant else ""
        return f"{self.mean_a:.1f}±{self.std_a:.1f} vs {self.mean_b:.1f}±{self.std_b:.1f}{sig_marker}"


def bootstrap_ci(
    data: Sequence[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    statistic: callable = np.mean,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns:
        (mean, ci_low, ci_high)
    """
    data = np.array(data)
    n = len(data)
    
    # Bootstrap resampling
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))
    
    boot_stats = np.array(boot_stats)
    
    # Percentile method
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return statistic(data), ci_low, ci_high


def paired_comparison(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alpha: float = 0.05,
) -> StatisticalResult:
    """
    Compare two methods using paired t-test.
    
    Args:
        scores_a: Scores from method A (per-task)
        scores_b: Scores from method B (per-task)
        alpha: Significance level
    
    Returns:
        StatisticalResult with all statistics
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    
    assert len(a) == len(b), "Must have paired observations"
    
    # Basic statistics
    mean_a, std_a = np.mean(a), np.std(a, ddof=1)
    mean_b, std_b = np.mean(b), np.std(b, ddof=1)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(a, b)
    
    # Effect size (Cohen's d for paired samples)
    diff = a - b
    effect_size = np.mean(diff) / np.std(diff, ddof=1)
    
    # Confidence interval on difference
    diff_mean, diff_ci_low, diff_ci_high = bootstrap_ci(diff)
    
    return StatisticalResult(
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        difference=diff_mean,
        ci_low=diff_ci_low,
        ci_high=diff_ci_high,
        p_value=p_value,
        effect_size=effect_size,
        significant=p_value < alpha,
    )


def multiple_comparison_correction(
    p_values: Sequence[float],
    method: str = "bonferroni",
) -> list[float]:
    """
    Correct p-values for multiple comparisons.
    
    Args:
        p_values: Raw p-values
        method: "bonferroni" or "holm"
    
    Returns:
        Corrected p-values
    """
    p = np.array(p_values)
    n = len(p)
    
    if method == "bonferroni":
        return list(np.minimum(p * n, 1.0))
    
    elif method == "holm":
        # Holm-Bonferroni step-down
        sorted_idx = np.argsort(p)
        corrected = np.zeros(n)
        
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min(p[idx] * (n - i), 1.0)
        
        # Enforce monotonicity
        for i in range(1, n):
            idx = sorted_idx[i]
            prev_idx = sorted_idx[i - 1]
            corrected[idx] = max(corrected[idx], corrected[prev_idx])
        
        return list(corrected)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate_across_seeds(
    results: list[dict],
    metric: str = "success_rate",
) -> tuple[float, float, list[float]]:
    """
    Aggregate results across random seeds.
    
    Returns:
        (mean, std, per_seed_values)
    """
    values = [r[metric] for r in results]
    return np.mean(values), np.std(values, ddof=1), values


def generate_comparison_table(
    all_results: dict[str, list[dict]],  # agent_name -> list of results
    metrics: list[str] = ["success_rate", "avg_cost"],
    baseline: str = "greedy",
) -> str:
    """
    Generate LaTeX comparison table with significance markers.
    
    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Agent comparison on API-Bank. * indicates $p < 0.05$ vs baseline.}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Agent & " + " & ".join(metrics) + r" \\",
        r"\midrule",
    ]
    
    # Get baseline scores
    baseline_results = all_results.get(baseline, [])
    
    for agent_name, results in all_results.items():
        row_parts = [agent_name]
        
        for metric in metrics:
            mean, std, values = aggregate_across_seeds(results, metric)
            
            # Compare to baseline if not baseline
            sig_marker = ""
            if agent_name != baseline and baseline_results:
                _, _, baseline_values = aggregate_across_seeds(baseline_results, metric)
                result = paired_comparison(values, baseline_values)
                if result.significant:
                    sig_marker = "*"
            
            row_parts.append(f"{mean:.1%}$\\pm${std:.1%}{sig_marker}")
        
        lines.append(" & ".join(row_parts) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)