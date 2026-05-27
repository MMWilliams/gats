#!/usr/bin/env python3
"""
GATS — single-command reproduction of every paper result.

Runs the three paper-producing evaluations and renders publication-quality
figures. All results land in ``results/`` as JSON, CSV summaries, and
PDF/PNG figures.

Usage
-----
    python reproduce.py             # full reproduction (~3 min, paper config)
    python reproduce.py --quick     # smoke test (~20 s, smaller seeds/tasks)
    python reproduce.py --no-figs   # skip figure generation
    python reproduce.py --only stress   # only run stress test stage
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXPERIMENTS = ROOT / "experiments"


# --- paper-matching config (Tables 1, 2, 3, 5 + Table 6 + Table 7) ----------

FULL_CONFIG = {
    "main": {
        "script": "run_gats_eval.py",
        "args": ["--n-tasks", "100", "--seeds", "42", "123", "456",
                 "--backend", "mock",
                 "--output", str(RESULTS / "main_eval.json")],
        "label": "Main eval + ablations (Tables 1, 2, 3, 5)",
    },
    "stress": {
        "script": "run_stress_test.py",
        "args": ["--n-per-category", "10", "--seeds", "42", "123", "456",
                 "--output", str(RESULTS / "stress_test.json")],
        "label": "12-category stress test (Table 6)",
    },
    "sensitivity": {
        "script": "run_gats_eval.py",
        "args": ["--n-tasks", "50", "--seeds", "42",
                 "--backend", "mock",
                 "--output", str(RESULTS / "sensitivity.json")],
        "label": "Hyperparameter sensitivity probe (Table 7)",
    },
}

QUICK_OVERRIDES = {
    "main":        ["--n-tasks", "30", "--seeds", "42",
                    "--backend", "mock",
                    "--output", str(RESULTS / "main_eval.json"),
                    "--quick"],
    "stress":      ["--n-per-category", "3", "--seeds", "42",
                    "--output", str(RESULTS / "stress_test.json")],
    "sensitivity": ["--n-tasks", "20", "--seeds", "42",
                    "--backend", "mock",
                    "--output", str(RESULTS / "sensitivity.json")],
}


def run_stage(name: str, cfg: dict, quick: bool) -> bool:
    script = EXPERIMENTS / cfg["script"]
    args = QUICK_OVERRIDES[name] if quick else cfg["args"]
    cmd = [sys.executable, str(script), *args]
    banner = f"  [{name}] {cfg['label']}"
    print("\n" + "=" * 78)
    print(banner)
    print("=" * 78)
    print("  $", " ".join(cmd))
    print()
    t0 = time.perf_counter()
    # Force UTF-8 stdout so child scripts can print unicode safely on Windows.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    rc = subprocess.run(cmd, env=env).returncode
    dt = time.perf_counter() - t0
    status = "OK" if rc == 0 else f"FAIL (exit {rc})"
    print(f"\n  [{name}] {status} in {dt:.1f}s")
    return rc == 0


def print_summary(quick: bool):
    """Print a tight summary table from the JSONs."""
    print("\n" + "=" * 78)
    print("  REPRODUCTION SUMMARY" + ("  (quick mode)" if quick else ""))
    print("=" * 78)

    main_path = RESULTS / "main_eval.json"
    if main_path.exists():
        data = json.loads(main_path.read_text())
        print("\n  Main eval (mean over seeds)")
        print(f"  {'method':<14} {'success':>8}  {'optimality':>10}  {'avg cost':>9}")
        print("  " + "-" * 48)
        order = ["greedy", "react", "lats_b5", "lats_b10",
                 "gats_b5", "gats_b10", "gats_b20"]
        for m in order:
            runs = data.get(m, [])
            if not runs:
                continue
            sr = sum(r["success_rate"] for r in runs) / len(runs)
            opt = sum(r["optimality"] for r in runs) / len(runs)
            cost = sum(r["avg_cost"] for r in runs) / len(runs)
            print(f"  {m:<14} {sr:>8.1%}  {opt:>10.2f}  {cost:>9.2f}")

    stress_path = RESULTS / "stress_test.json"
    if stress_path.exists():
        data = json.loads(stress_path.read_text())
        print("\n  Stress test (overall success rate)")
        print(f"  {'method':<14} {'overall':>10}")
        print("  " + "-" * 28)
        for m in ["react", "lats_b10", "lats_b20", "gats_b10", "gats_b20"]:
            cats = data.get(m, {})
            if not cats:
                continue
            overall = sum(cats.values()) / len(cats)
            print(f"  {m:<14} {overall:>10.1%}")

    print("\n  Artifacts:")
    for p in sorted(RESULTS.glob("*")):
        if p.is_file():
            print(f"    - {p.relative_to(ROOT)}")
    figs = RESULTS / "figures"
    if figs.exists():
        for p in sorted(figs.glob("*")):
            print(f"    - {p.relative_to(ROOT)}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test config (smaller seeds/tasks, ~20s).")
    p.add_argument("--no-figs", action="store_true",
                   help="Skip figure generation.")
    p.add_argument("--only", choices=list(FULL_CONFIG.keys()),
                   help="Run only one stage.")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)

    stages = [args.only] if args.only else list(FULL_CONFIG.keys())

    t0 = time.perf_counter()
    all_ok = True
    for name in stages:
        ok = run_stage(name, FULL_CONFIG[name], args.quick)
        all_ok = all_ok and ok

    if all_ok and not args.no_figs:
        print("\n" + "=" * 78)
        print("  [figures] Rendering publication figures")
        print("=" * 78)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        rc = subprocess.run(
            [sys.executable, str(ROOT / "figures.py"),
             "--results-dir", str(RESULTS)],
            env=env,
        ).returncode
        if rc != 0:
            print(f"  [figures] FAIL (exit {rc})")
            all_ok = False

    print_summary(args.quick)

    dt = time.perf_counter() - t0
    print(f"  Total wall time: {dt:.1f}s")
    print("  " + ("All stages succeeded." if all_ok else "Some stages FAILED."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
