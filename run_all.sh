#!/usr/bin/env bash
# run_all.sh — Phase 1 only: retrain reddit-binary after model fix


set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo " Graph Retrieval — Phase 1: reddit-binary only"
echo " Started: $(date)"
echo "============================================================"

# ── Phase 1: Train reddit-binary (with GED-based triplet oracle) ─────────────
echo ""
echo "==== PHASE 1: TRAINING ===="

echo "---- Training: reddit-binary on GPU 0 ----"
python3 "$ROOT/main.py" --dataset reddit-binary --gpu 0
wait

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo "============================================================"

# ── Phases 2 & 3 commented out (run separately after Phase 1 succeeds) ───────
# echo "==== PHASE 2: EVALUATION (GED ground truth) ===="
# CUDA_VISIBLE_DEVICES=0 python3 "$ROOT/evaluate.py" --dataset reddit-binary --mode both --gt ged
#
# echo "==== PHASE 3: LSH ABLATION ===="
# CUDA_VISIBLE_DEVICES=0 python3 "$ROOT/ablate_lsh.py" --dataset reddit-binary --gt ged
