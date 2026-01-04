#!/bin/bash
set -euo pipefail

# =============================================================================
# GATS 2.0 Full Reproduction Script
# =============================================================================
# This script reproduces all experiments from the paper.
# Estimated runtime: 4-6 hours (with GPU), 12-24 hours (CPU only)
# Estimated cost: ~$100 (if using API models)
# =============================================================================

echo "========================================"
echo "GATS 2.0 Reproduction Script"
echo "========================================"
echo ""

# Check dependencies
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "pip required"; exit 1; }

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RESULTS_DIR="results/reproduction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Step 1: Install dependencies
echo "[1/6] Installing dependencies..."
pip install -e ".[all]" --quiet

# Step 2: Download datasets
echo "[2/6] Downloading datasets..."
bash scripts/download_api_bank.sh
bash scripts/download_toolbench.sh

# Step 3: Run unit tests
echo "[3/6] Running unit tests..."
python -m pytest tests/ -v --tb=short 2>&1 | tee "$RESULTS_DIR/test_results.txt"

# Step 4: Run internal benchmarks
echo "[4/6] Running internal benchmarks..."
python run.py all 2>&1 | tee "$RESULTS_DIR/internal_benchmarks.txt"

# Step 5: Run external benchmarks
echo "[5/6] Running external benchmarks..."
python run_external.py all 200 2>&1 | tee "$RESULTS_DIR/external_benchmarks.txt"

# Step 6: Run LLM experiments (optional - requires API key or local model)
echo "[6/6] Running LLM experiments..."
if command -v ollama >/dev/null 2>&1; then
    echo "  Using Ollama..."
    python scripts/run_llm_experiments.py \
        --models llama-3.1-8b \
        --benchmarks api_bank toolbench \
        --agents gats lats react greedy \
        --seeds 42 123 456 789 1011 \
        --max-tasks 100 \
        --results-dir "$RESULTS_DIR/llm_experiments" \
        2>&1 | tee "$RESULTS_DIR/llm_experiments.txt"
else
    echo "  Ollama not found, skipping LLM experiments"
    echo "  Install with: curl -fsSL https://ollama.com/install.sh | sh"
fi

# Generate tables and figures
echo ""
echo "[7/7] Generating tables and figures..."
python analysis/generate_tables.py "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/tables.txt"
python analysis/generate_figures.py "$RESULTS_DIR" 2>&1

echo ""
echo "========================================"
echo "Reproduction complete!"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Key files:"
echo "  - $RESULTS_DIR/test_results.txt"
echo "  - $RESULTS_DIR/internal_benchmarks.txt"
echo "  - $RESULTS_DIR/external_benchmarks.txt"
echo "  - $RESULTS_DIR/tables/"
echo "  - $RESULTS_DIR/figures/"