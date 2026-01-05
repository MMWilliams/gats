#!/usr/bin/env python3
"""
GATS Full Reproduction Script
Cross-platform (Windows/Linux/Mac)

Runs all evaluations:
1. Main evaluation + ablations
2. API-Bank benchmark
3. Stress test (12 categories)
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"[RUNNING] {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully")
        return True
    else:
        print(f"\n✗ {description} failed with code {result.returncode}")
        return False


def main():
    print("="*60)
    print("GATS FULL REPRODUCTION")
    print("="*60)
    
    # Track results
    results = {}
    
    # 1. Download datasets
    results["download_api_bank"] = run_command(
        f"{sys.executable} download_api_bank.py",
        "Download API-Bank data"
    )
    
    results["download_level3"] = run_command(
        f"{sys.executable} download_level3.py",
        "Generate Level 3 multi-step tasks"
    )
    
    # 2. Main evaluation + ablations
    results["ablation_study"] = run_command(
        f"{sys.executable} run_ablation_study.py",
        "Main evaluation + ablation study"
    )
    
    # 3. Stress test
    results["stress_test"] = run_command(
        f"{sys.executable} run_stress_test.py --n-per-category 10 --seeds 42 123 456",
        "Stress test (12 categories, 120 tasks)"
    )
    
    # Summary
    print("\n" + "="*60)
    print("REPRODUCTION SUMMARY")
    print("="*60)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:<25} {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    else:
        print("SOME EXPERIMENTS FAILED - Check output above")
    print("="*60)
    
    print("\nResults saved to:")
    print("  - results/gats_eval.json")
    print("  - results/stress_test.json")
    print("  - results/tables.tex")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())