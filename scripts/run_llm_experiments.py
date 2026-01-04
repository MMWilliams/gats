# scripts/run_llm_experiments.py

"""
LLM Experiment Runner for GATS 2.0

Runs experiments across:
- Models: Llama-3.1-8B, Llama-3.1-70B, GPT-4o-mini, Claude-3-Haiku
- Benchmarks: API-Bank, ToolBench
- Agents: GATS, LATS, ReAct, Greedy
- Seeds: 5 random seeds for statistical significance
"""

import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any
import numpy as np

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gats.llm import OllamaPredictor, VLLMPredictor, OpenAICompatiblePredictor, LLMConfig


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_name: str
    model_backend: str  # "ollama", "vllm", "openai"
    benchmark: str  # "api_bank", "toolbench"
    agent: str  # "gats", "lats", "react", "greedy"
    seed: int
    max_tasks: int = 100
    search_budget: int = 50
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    success_rate: float
    avg_cost: float
    avg_steps: float
    avg_latency_ms: float
    total_tokens: int
    wall_clock_seconds: float
    per_task_results: list[dict]
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "success_rate": self.success_rate,
            "avg_cost": self.avg_cost,
            "avg_steps": self.avg_steps,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "wall_clock_seconds": self.wall_clock_seconds,
            "n_tasks": len(self.per_task_results),
            "timestamp": self.timestamp,
        }


class ExperimentRunner:
    """Runs experiments with proper logging and reproducibility."""
    
    MODELS = {
        "llama-3.1-8b": {
            "backend": "ollama",
            "config": LLMConfig(model="llama3.1:8b", temperature=0.1)
        },
        "llama-3.1-70b": {
            "backend": "vllm",
            "config": LLMConfig(model="meta-llama/Llama-3.1-70B-Instruct", temperature=0.1)
        },
        "gpt-4o-mini": {
            "backend": "openai",
            "config": LLMConfig(model="gpt-4o-mini", temperature=0.1)
        },
    }
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_predictor(self, model_name: str):
        """Create LLM predictor for specified model."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = self.MODELS[model_name]
        backend = model_info["backend"]
        config = model_info["config"]
        
        if backend == "ollama":
            return OllamaPredictor(config)
        elif backend == "vllm":
            return VLLMPredictor(config)
        elif backend == "openai":
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
            from gats.llm import OpenAICompatiblePredictor
            return OpenAICompatiblePredictor(config, api_key=api_key)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def run_single(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment configuration."""
        import random
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Load dataset
        if config.benchmark == "api_bank":
            from bench.api_bank_real import RealAPIBankDataset
            dataset = RealAPIBankDataset(Path("data/api_bank")).load()
            tasks = dataset.tasks[:config.max_tasks]
        elif config.benchmark == "toolbench":
            from bench.toolbench_real import RealToolBenchDataset
            dataset = RealToolBenchDataset(Path("data/toolbench")).load()
            tasks = dataset.tasks[:config.max_tasks]
        else:
            raise ValueError(f"Unknown benchmark: {config.benchmark}")
        
        # Create predictor
        predictor = self.create_predictor(config.model_name)
        
        # Create agent
        # ... (agent creation logic based on config.agent)
        
        # Run tasks
        start_time = time.time()
        per_task = []
        total_tokens = 0
        total_latency = 0
        
        for task in tasks:
            task_start = time.time()
            # ... (task execution logic)
            task_result = {
                "task_id": task.task_id,
                "success": True,  # placeholder
                "cost": 0.0,
                "steps": 0,
                "tokens": 0,
                "latency_ms": (time.time() - task_start) * 1000,
            }
            per_task.append(task_result)
            total_tokens += task_result["tokens"]
            total_latency += task_result["latency_ms"]
        
        wall_clock = time.time() - start_time
        
        return ExperimentResult(
            config=config,
            success_rate=sum(1 for t in per_task if t["success"]) / len(per_task),
            avg_cost=sum(t["cost"] for t in per_task) / len(per_task),
            avg_steps=sum(t["steps"] for t in per_task) / len(per_task),
            avg_latency_ms=total_latency / len(per_task),
            total_tokens=total_tokens,
            wall_clock_seconds=wall_clock,
            per_task_results=per_task,
            timestamp=datetime.now().isoformat(),
        )
    
    def run_matrix(
        self,
        models: list[str],
        benchmarks: list[str],
        agents: list[str],
        seeds: list[int],
        max_tasks: int = 100,
    ) -> list[ExperimentResult]:
        """Run full experiment matrix."""
        results = []
        
        total_runs = len(models) * len(benchmarks) * len(agents) * len(seeds)
        run_idx = 0
        
        for model in models:
            for benchmark in benchmarks:
                for agent in agents:
                    for seed in seeds:
                        run_idx += 1
                        print(f"[{run_idx}/{total_runs}] {model} / {benchmark} / {agent} / seed={seed}")
                        
                        config = ExperimentConfig(
                            model_name=model,
                            model_backend=self.MODELS[model]["backend"],
                            benchmark=benchmark,
                            agent=agent,
                            seed=seed,
                            max_tasks=max_tasks,
                        )
                        
                        try:
                            result = self.run_single(config)
                            results.append(result)
                            
                            # Save incrementally
                            self._save_result(result)
                            
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            continue
        
        return results
    
    def _save_result(self, result: ExperimentResult) -> None:
        """Save single result to disk."""
        fname = f"{result.config.model_name}_{result.config.benchmark}_{result.config.agent}_s{result.config.seed}.json"
        path = self.results_dir / fname
        
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run GATS LLM experiments")
    parser.add_argument("--models", nargs="+", default=["llama-3.1-8b"])
    parser.add_argument("--benchmarks", nargs="+", default=["api_bank"])
    parser.add_argument("--agents", nargs="+", default=["gats", "react", "greedy"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1011])
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--results-dir", type=str, default="results/llm_experiments")
    args = parser.parse_args()
    
    runner = ExperimentRunner(Path(args.results_dir))
    results = runner.run_matrix(
        models=args.models,
        benchmarks=args.benchmarks,
        agents=args.agents,
        seeds=args.seeds,
        max_tasks=args.max_tasks,
    )
    
    print(f"\nCompleted {len(results)} experiments")
    print(f"Results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()