#!/usr/bin/env bash
# run_all.sh — Full pipeline: train → evaluate → ablate, all 5 datasets
# Saves all stdout/stderr to logs/run_all.log

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/run_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Graph Retrieval — Full Pipeline Run"
echo " Started: $(date)"
echo " Log: $LOG"
echo "============================================================"

DATASETS=(mutag aids imdb-binary proteins reddit-binary)

# ── Phase 1: Train (with GED-based triplet oracle) ───────────────────────────
echo ""
echo "==== PHASE 1: TRAINING ===="
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "---- Training: $ds  [$(date +%H:%M:%S)] ----"
    python3 "$ROOT/main.py" --dataset "$ds"
    echo "---- Done: $ds  [$(date +%H:%M:%S)] ----"
done

# ── Phase 2: Evaluate (with GED ground truth) ────────────────────────────────
echo ""
echo "==== PHASE 2: EVALUATION (GED ground truth) ===="
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "---- Evaluating: $ds  [$(date +%H:%M:%S)] ----"
    python3 "$ROOT/evaluate.py" --dataset "$ds" --mode both --gt ged
    echo "---- Done: $ds  [$(date +%H:%M:%S)] ----"
done

# ── Phase 3: LSH Ablation ────────────────────────────────────────────────────
echo ""
echo "==== PHASE 3: LSH ABLATION ===="
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "---- Ablating: $ds  [$(date +%H:%M:%S)] ----"
    python3 "$ROOT/ablate_lsh.py" --dataset "$ds" --gt ged
    echo "---- Done: $ds  [$(date +%H:%M:%S)] ----"
done

echo ""
echo "============================================================"
echo " ALL DONE — $(date)"
echo "============================================================"
