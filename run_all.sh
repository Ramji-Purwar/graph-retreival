#!/usr/bin/env bash
# run_all.sh — Full pipeline: train → evaluate → ablate, all 5 datasets


set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo " Graph Retrieval — Full Pipeline Run"
echo " Started: $(date)"

echo "============================================================"

DATASETS=(mutag aids imdb-binary proteins reddit-binary)

# ── Phase 1: Train (with GED-based triplet oracle) ───────────────────────────
echo ""
echo "==== PHASE 1: TRAINING ===="

echo "---- Training: mutag on GPU 0 & aids on GPU 1 ----"
python3 "$ROOT/main.py" --dataset mutag --gpu 0 &
python3 "$ROOT/main.py" --dataset aids --gpu 1 &
wait

echo "---- Training: imdb-binary on GPU 0 & proteins on GPU 1 ----"
python3 "$ROOT/main.py" --dataset imdb-binary --gpu 0 &
python3 "$ROOT/main.py" --dataset proteins --gpu 1 &
wait

echo "---- Training: reddit-binary on GPU 0 ----"
python3 "$ROOT/main.py" --dataset reddit-binary --gpu 0 &
wait

# ── Phase 2: Evaluate (with GED ground truth) ────────────────────────────────
echo ""
echo "==== PHASE 2: EVALUATION (GED ground truth) ===="

echo "---- Evaluating: mutag on GPU 0 & aids on GPU 1 ----"
CUDA_VISIBLE_DEVICES=0 python3 "$ROOT/evaluate.py" --dataset mutag --mode both --gt ged &
CUDA_VISIBLE_DEVICES=1 python3 "$ROOT/evaluate.py" --dataset aids --mode both --gt ged &
wait

echo "---- Evaluating: imdb-binary on GPU 0 & proteins on GPU 1 ----"
CUDA_VISIBLE_DEVICES=0 python3 "$ROOT/evaluate.py" --dataset imdb-binary --mode both --gt ged &
CUDA_VISIBLE_DEVICES=1 python3 "$ROOT/evaluate.py" --dataset proteins --mode both --gt ged &
wait

echo "---- Evaluating: reddit-binary on GPU 0 ----"
CUDA_VISIBLE_DEVICES=0 python3 "$ROOT/evaluate.py" --dataset reddit-binary --mode both --gt ged &
wait

# ── Phase 3: LSH Ablation ────────────────────────────────────────────────────
echo ""
echo "==== PHASE 3: LSH ABLATION ===="

echo "---- Ablating: mutag on GPU 0 & aids on GPU 1 ----"
CUDA_VISIBLE_DEVICES=0 python3 "$ROOT/ablate_lsh.py" --dataset mutag --gt ged &
CUDA_VISIBLE_DEVICES=1 python3 "$ROOT/ablate_lsh.py" --dataset aids --gt ged &
wait

echo "---- Ablating: imdb-binary on GPU 0 & proteins on GPU 1 ----"
CUDA_VISIBLE_DEVICES=0 python3 "$ROOT/ablate_lsh.py" --dataset imdb-binary --gt ged &
CUDA_VISIBLE_DEVICES=1 python3 "$ROOT/ablate_lsh.py" --dataset proteins --gt ged &
wait

echo "---- Ablating: reddit-binary on GPU 0 ----"
CUDA_VISIBLE_DEVICES=0 python3 "$ROOT/ablate_lsh.py" --dataset reddit-binary --gt ged &
wait

echo ""
echo "============================================================"
echo " ALL DONE — $(date)"
echo "============================================================"
