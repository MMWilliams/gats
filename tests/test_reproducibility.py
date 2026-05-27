"""Smoke tests — verify the paper-reproduction pipeline runs end to end.

These tests do NOT validate the paper's exact numbers (that's what
``python reproduce.py`` does). They confirm:

  1. Each experiment script imports and runs to completion.
  2. The expected JSON artifacts are produced with the right shape.
  3. ``figures.py`` consumes those JSONs and writes both PDF + PNG.

Run with: pytest tests/ -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def _run(cmd: list[str], cwd: Path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run(cmd, cwd=cwd, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


def test_main_eval_smoke(tmp_path: Path):
    out = tmp_path / "main.json"
    _run([sys.executable, str(EXPERIMENTS / "run_gats_eval.py"),
          "--n-tasks", "10", "--seeds", "42", "--backend", "mock",
          "--quick", "--output", str(out)], cwd=ROOT)
    data = json.loads(out.read_text())
    assert "gats_b10" in data
    assert data["gats_b10"][0]["success_rate"] >= 0.0
    assert data["gats_b10"][0]["n_tasks"] == 10


def test_stress_test_smoke(tmp_path: Path):
    out = tmp_path / "stress.json"
    _run([sys.executable, str(EXPERIMENTS / "run_stress_test.py"),
          "--n-per-category", "2", "--seeds", "42",
          "--output", str(out)], cwd=ROOT)
    data = json.loads(out.read_text())
    assert "gats_b20" in data
    # 12 categories should all be present
    assert len(data["gats_b20"]) == 12


def test_figures_pipeline(tmp_path: Path):
    # Produce just enough data for figures.py to chew on.
    results = tmp_path / "results"
    results.mkdir()
    _run([sys.executable, str(EXPERIMENTS / "run_gats_eval.py"),
          "--n-tasks", "10", "--seeds", "42", "--backend", "mock",
          "--quick", "--output", str(results / "main_eval.json")], cwd=ROOT)
    _run([sys.executable, str(EXPERIMENTS / "run_stress_test.py"),
          "--n-per-category", "2", "--seeds", "42",
          "--output", str(results / "stress_test.json")], cwd=ROOT)
    _run([sys.executable, str(ROOT / "figures.py"),
          "--results-dir", str(results)], cwd=ROOT)

    figs = results / "figures"
    for name in ("fig1_main_results", "fig2_budget_ablation",
                 "fig3_llm_calls", "fig4_stress_test"):
        assert (figs / f"{name}.pdf").exists(), f"missing {name}.pdf"
        assert (figs / f"{name}.png").exists(), f"missing {name}.png"
    assert (results / "summary.csv").exists()
    assert (results / "tables.tex").exists()
