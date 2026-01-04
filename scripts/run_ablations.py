#!/usr/bin/env python3
"""GATS 2.0 Ablation Studies - Systematic evaluation of component contributions."""
from __future__ import annotations
import sys
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class AblationResult:
    """Result from a single ablation configuration."""
    name: str
    config: Dict[str, Any]
    api_accuracy: float
    param_accuracy: float
    avg_latency_ms: float
    n_tasks: int
    seed: int
    benchmark: str

@dataclass
class AblationStudy:
    """Complete ablation study results."""
    study_name: str
    baseline: AblationResult
    ablations: List[AblationResult]
    timestamp: str

# ============================================================================
# ABLATION CONFIGURATIONS
# ============================================================================

ABLATION_CONFIGS = {
    # World Model Layer Ablations
    "no_l1": {
        "description": "Disable L1 (exact matching)",
        "world_model": {"use_l1": False, "use_l2": True, "use_l3": True}
    },
    "no_l2": {
        "description": "Disable L2 (learned from logs)", 
        "world_model": {"use_l1": True, "use_l2": False, "use_l3": True}
    },
    "no_l3": {
        "description": "Disable L3 (LLM prediction)",
        "world_model": {"use_l1": True, "use_l2": True, "use_l3": False}
    },
    "l1_only": {
        "description": "L1 only (pure symbolic)",
        "world_model": {"use_l1": True, "use_l2": False, "use_l3": False}
    },
    "l3_only": {
        "description": "L3 only (pure LLM)",
        "world_model": {"use_l1": False, "use_l2": False, "use_l3": True}
    },
    
    # Search Budget Ablations
    "budget_1": {
        "description": "Search budget = 1 (greedy)",
        "search": {"budget": 1, "c_puct": 1.0}
    },
    "budget_5": {
        "description": "Search budget = 5",
        "search": {"budget": 5, "c_puct": 1.0}
    },
    "budget_10": {
        "description": "Search budget = 10",
        "search": {"budget": 10, "c_puct": 1.0}
    },
    "budget_20": {
        "description": "Search budget = 20",
        "search": {"budget": 20, "c_puct": 1.0}
    },
    "budget_50": {
        "description": "Search budget = 50",
        "search": {"budget": 50, "c_puct": 1.0}
    },
    
    # Exploration Ablations (c_puct)
    "cpuct_0.5": {
        "description": "c_puct = 0.5 (less exploration)",
        "search": {"budget": 10, "c_puct": 0.5}
    },
    "cpuct_1.0": {
        "description": "c_puct = 1.0 (balanced)",
        "search": {"budget": 10, "c_puct": 1.0}
    },
    "cpuct_2.0": {
        "description": "c_puct = 2.0 (more exploration)",
        "search": {"budget": 10, "c_puct": 2.0}
    },
    "cpuct_5.0": {
        "description": "c_puct = 5.0 (high exploration)",
        "search": {"budget": 10, "c_puct": 5.0}
    },
    
    # Rollout Depth Ablations
    "depth_1": {
        "description": "Rollout depth = 1",
        "search": {"rollout_depth": 1}
    },
    "depth_3": {
        "description": "Rollout depth = 3",
        "search": {"rollout_depth": 3}
    },
    "depth_5": {
        "description": "Rollout depth = 5",
        "search": {"rollout_depth": 5}
    },
    "depth_10": {
        "description": "Rollout depth = 10",
        "search": {"rollout_depth": 10}
    },
    
    # Backend Ablations
    "backend_mock": {
        "description": "Mock/Heuristic backend",
        "backend": "mock"
    },
    "backend_ollama": {
        "description": "Ollama LLM backend",
        "backend": "ollama"
    },
}

# Predefined ablation groups
ABLATION_GROUPS = {
    "layers": ["no_l1", "no_l2", "no_l3", "l1_only", "l3_only"],
    "budget": ["budget_1", "budget_5", "budget_10", "budget_20", "budget_50"],
    "cpuct": ["cpuct_0.5", "cpuct_1.0", "cpuct_2.0", "cpuct_5.0"],
    "depth": ["depth_1", "depth_3", "depth_5", "depth_10"],
    "backend": ["backend_mock", "backend_ollama"],
    "quick": ["no_l1", "no_l3", "budget_1", "budget_10"],
    "full": list(ABLATION_CONFIGS.keys()),
}

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def run_single_ablation(
    name: str,
    config: Dict[str, Any],
    benchmark: str,
    n_tasks: int,
    seed: int,
    backend: str = "auto"
) -> AblationResult:
    """Run a single ablation configuration."""
    from run_experiment import run_experiment, APIPredictor, load_api_bank, load_toolbench, compare_api_calls
    import random
    
    random.seed(seed)
    
    # Override backend if specified in config
    if "backend" in config:
        backend = config["backend"]
    
    # Load tasks
    if benchmark == "api_bank":
        tasks = load_api_bank("data/api_bank", n_tasks)
    else:
        tasks = load_toolbench("data/toolbench", n_tasks)
    
    if not tasks:
        return AblationResult(
            name=name, config=config, api_accuracy=0, param_accuracy=0,
            avg_latency_ms=0, n_tasks=0, seed=seed, benchmark=benchmark
        )
    
    # Initialize predictor with config
    predictor = APIPredictor(backend)
    
    # Run evaluation
    api_correct = 0
    param_correct = 0
    total_latency = 0
    
    for task in tasks:
        start = time.perf_counter()
        
        predicted = predictor.predict(task["instruction"], task["input"])
        latency = (time.perf_counter() - start) * 1000
        total_latency += latency
        
        api_match, param_match = compare_api_calls(predicted, task["expected_output"])
        if api_match:
            api_correct += 1
        if param_match:
            param_correct += 1
    
    n = len(tasks)
    return AblationResult(
        name=name,
        config=config,
        api_accuracy=api_correct / n,
        param_accuracy=param_correct / n,
        avg_latency_ms=total_latency / n,
        n_tasks=n,
        seed=seed,
        benchmark=benchmark
    )

def run_ablation_study(
    study_name: str,
    ablation_names: List[str],
    benchmark: str = "api_bank",
    n_tasks: int = 50,
    seeds: List[int] = [42],
    backend: str = "auto",
    parallel: bool = False
) -> AblationStudy:
    """Run a complete ablation study."""
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {study_name}")
    print(f"{'='*60}")
    print(f"Benchmark: {benchmark}")
    print(f"Tasks: {n_tasks}")
    print(f"Seeds: {seeds}")
    print(f"Ablations: {ablation_names}")
    print()
    
    # Run baseline
    print("Running baseline...")
    baseline_results = []
    for seed in seeds:
        result = run_single_ablation(
            "baseline", {}, benchmark, n_tasks, seed, backend
        )
        baseline_results.append(result)
        print(f"  Seed {seed}: API={result.api_accuracy:.1%}")
    
    # Average baseline
    baseline = AblationResult(
        name="baseline",
        config={},
        api_accuracy=sum(r.api_accuracy for r in baseline_results) / len(baseline_results),
        param_accuracy=sum(r.param_accuracy for r in baseline_results) / len(baseline_results),
        avg_latency_ms=sum(r.avg_latency_ms for r in baseline_results) / len(baseline_results),
        n_tasks=baseline_results[0].n_tasks,
        seed=seeds[0],
        benchmark=benchmark
    )
    
    # Run ablations
    all_ablation_results = []
    
    for abl_name in ablation_names:
        if abl_name not in ABLATION_CONFIGS:
            print(f"Warning: Unknown ablation '{abl_name}', skipping")
            continue
        
        config = ABLATION_CONFIGS[abl_name]
        print(f"\nRunning ablation: {abl_name}")
        print(f"  Config: {config.get('description', 'N/A')}")
        
        abl_results = []
        for seed in seeds:
            result = run_single_ablation(
                abl_name, config, benchmark, n_tasks, seed, backend
            )
            abl_results.append(result)
            print(f"  Seed {seed}: API={result.api_accuracy:.1%}")
        
        # Average results
        avg_result = AblationResult(
            name=abl_name,
            config=config,
            api_accuracy=sum(r.api_accuracy for r in abl_results) / len(abl_results),
            param_accuracy=sum(r.param_accuracy for r in abl_results) / len(abl_results),
            avg_latency_ms=sum(r.avg_latency_ms for r in abl_results) / len(abl_results),
            n_tasks=abl_results[0].n_tasks,
            seed=seeds[0],
            benchmark=benchmark
        )
        all_ablation_results.append(avg_result)
    
    return AblationStudy(
        study_name=study_name,
        baseline=baseline,
        ablations=all_ablation_results,
        timestamp=datetime.now().isoformat()
    )

def print_ablation_table(study: AblationStudy):
    """Print formatted ablation results table."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {study.study_name}")
    print(f"{'='*70}")
    print(f"{'Configuration':<25} {'API Acc':>10} {'Param Acc':>10} {'Latency':>10} {'Δ API':>10}")
    print("-" * 70)
    
    baseline_api = study.baseline.api_accuracy
    print(f"{'baseline':<25} {baseline_api:>9.1%} {study.baseline.param_accuracy:>9.1%} "
          f"{study.baseline.avg_latency_ms:>8.1f}ms {'---':>10}")
    
    for abl in study.ablations:
        delta = abl.api_accuracy - baseline_api
        delta_str = f"{delta:+.1%}"
        print(f"{abl.name:<25} {abl.api_accuracy:>9.1%} {abl.param_accuracy:>9.1%} "
              f"{abl.avg_latency_ms:>8.1f}ms {delta_str:>10}")
    
    print("-" * 70)

def save_ablation_results(study: AblationStudy, output_path: str):
    """Save ablation results to JSON."""
    results = {
        "study_name": study.study_name,
        "timestamp": study.timestamp,
        "baseline": asdict(study.baseline),
        "ablations": [asdict(a) for a in study.ablations]
    }
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

def generate_latex_table(study: AblationStudy) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + study.study_name.replace("_", " ").title() + r"}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Configuration & API Acc. & Param Acc. & $\Delta$ \\",
        r"\midrule",
    ]
    
    baseline_api = study.baseline.api_accuracy
    lines.append(f"Baseline & {baseline_api:.1%} & {study.baseline.param_accuracy:.1%} & --- \\\\")
    
    for abl in study.ablations:
        delta = abl.api_accuracy - baseline_api
        delta_str = f"{delta:+.1%}"
        name = abl.name.replace("_", r"\_")
        lines.append(f"{name} & {abl.api_accuracy:.1%} & {abl.param_accuracy:.1%} & {delta_str} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GATS 2.0 Ablation Studies")
    parser.add_argument("--study", choices=list(ABLATION_GROUPS.keys()) + ["custom"],
                       default="quick", help="Predefined study group or 'custom'")
    parser.add_argument("--ablations", type=str, nargs="*", 
                       help="Custom ablation names (use with --study custom)")
    parser.add_argument("--benchmark", choices=["api_bank", "toolbench"], default="api_bank")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--backend", choices=["auto", "ollama", "mock"], default="auto")
    parser.add_argument("--output", type=str, default="results/ablations.json")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    parser.add_argument("--list", action="store_true", help="List available ablations")
    args = parser.parse_args()
    
    if args.list:
        print("Available ablation configurations:")
        print("-" * 50)
        for name, config in ABLATION_CONFIGS.items():
            print(f"  {name:<20} {config.get('description', '')}")
        print("\nPredefined groups:")
        for group, ablations in ABLATION_GROUPS.items():
            print(f"  {group:<15} {ablations}")
        return
    
    # Get ablation list
    if args.study == "custom":
        if not args.ablations:
            print("Error: --ablations required with --study custom")
            return
        ablation_names = args.ablations
        study_name = "custom_study"
    else:
        ablation_names = ABLATION_GROUPS[args.study]
        study_name = f"{args.study}_ablation"
    
    # Run study
    study = run_ablation_study(
        study_name=study_name,
        ablation_names=ablation_names,
        benchmark=args.benchmark,
        n_tasks=args.n_tasks,
        seeds=args.seeds,
        backend=args.backend
    )
    
    # Print results
    print_ablation_table(study)
    
    # Save results
    save_ablation_results(study, args.output)
    
    # Generate LaTeX if requested
    if args.latex:
        latex = generate_latex_table(study)
        latex_path = args.output.replace(".json", ".tex")
        Path(latex_path).write_text(latex)
        print(f"LaTeX table saved to {latex_path}")

if __name__ == "__main__":
    main()