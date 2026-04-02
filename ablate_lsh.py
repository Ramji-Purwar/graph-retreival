import argparse
import json
import os
import sys
import time
import pickle
import io

import numpy as np
import torch

from config import CONFIGS
from core.lsh import LSHIndex


W_GRID = [0.5, 1.0, 2.0, 4.0]
L_GRID = [5, 10, 20]
K_FUNCS = 4
HOLDOUT_FRAC = 0.10
EVAL_K = 10


class _CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                        weights_only=False)
        return super().find_class(module, name)


def _load_outputs(dataset_name):
    out_dir = f"outputs/{dataset_name}"
    embeddings = np.load(f"{out_dir}/embeddings.npy")
    with open(f"{out_dir}/dataset.pkl", "rb") as f:
        dataset = _CPU_Unpickler(f).load()
    return embeddings, dataset


def _load_ground_truth(dataset_name, dataset, cfg, use_ged: bool):
    if use_ged:
        from core.ged_cache import build_or_load_ged_matrix, get_ground_truth_ged
        ged_matrix, threshold = build_or_load_ged_matrix(dataset, dataset_name, cfg)
        return get_ground_truth_ged(ged_matrix, threshold)
    # label-proxy fallback
    gt = {}
    for i, g in enumerate(dataset):
        lbl = g.y.item()
        gt[i] = {j for j, d in enumerate(dataset) if d.y.item() == lbl and j != i}
    return gt


# Metrics

def _recall_at_k(retrieved, relevant):
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def _precision_at_k(retrieved, relevant):
    if not retrieved:
        return 0.0
    return len(set(retrieved) & relevant) / len(retrieved)


def _average_precision(retrieved, relevant):
    hits = score = 0.0
    for i, idx in enumerate(list(retrieved)):
        if idx in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0


def _approx_quality(lsh_top, bf_top):
    s1, s2 = set(lsh_top), set(bf_top)
    return len(s1 & s2) / len(s2) if s2 else 0.0


def _brute_force(z_q, embeddings, k):
    dists = np.linalg.norm(embeddings - z_q, axis=1)
    return np.argsort(dists)[1:k + 1]


# LSH helpers

def _build_lsh(embeddings, out_dim, n_tables, n_funcs, w):
    lsh = LSHIndex(dim=out_dim, n_tables=n_tables, n_funcs=n_funcs, w=w)
    lsh.index(embeddings)
    return lsh


def _table_memory_bytes(lsh):
    return sum(sys.getsizeof(table) for (_, _, table) in lsh.tables)


# Phase 1: w grid search

def phase1_w_search(embeddings, gt, cfg, seed=42):
    N = len(embeddings)
    out_dim = cfg["out_dim"]
    rng = np.random.default_rng(seed)

    hold_n = max(1, int(N * HOLDOUT_FRAC))
    all_idx = np.arange(N)
    rng.shuffle(all_idx)
    query_idx = all_idx[:hold_n]
    corpus_idx = all_idx[hold_n:]

    corpus_emb = embeddings[corpus_idx]

    print(f"\n{'─'*60}")
    print(f"  Phase 1 — w grid search  (hold-out n={hold_n}, k={EVAL_K})")
    print(f"{'─'*60}")
    print(f"  {'w':>6}  {'Recall@10':>10}  {'Build(ms)':>10}  {'Query(ms)':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")

    results = {}
    for w in W_GRID:
        t0  = time.time()
        lsh = _build_lsh(corpus_emb, out_dim, n_tables=10, n_funcs=K_FUNCS, w=w)
        build_ms = (time.time() - t0) * 1000

        recalls, qtimes = [], []
        for qi in query_idx:
            z_q     = embeddings[qi]
            rel_all = gt[qi]                               # ground-truth in full corpus
            rel_sub = {corpus_idx.tolist().index(j)        # remap to corpus subset indices
                       for j in rel_all if j in corpus_idx}

            t0   = time.time()
            top  = lsh.query(z_q, corpus_emb, k=EVAL_K)
            qtimes.append((time.time() - t0) * 1000)

            top = [x for x in top][:EVAL_K]
            recalls.append(_recall_at_k(top, rel_sub))

        mean_rec = float(np.mean(recalls))
        mean_qt  = float(np.mean(qtimes))

        results[w] = {"Recall@10": mean_rec, "build_ms": build_ms, "query_ms": mean_qt}
        print(f"  {w:>6.1f}  {mean_rec:>10.4f}  {build_ms:>10.2f}  {mean_qt:>10.4f}")

    best_w = max(results, key=lambda x: results[x]["Recall@10"])
    print(f"\n  ✓ Best w = {best_w}  (Recall@10 = {results[best_w]['Recall@10']:.4f})\n")
    return best_w, results


# Phase 2: L ablation

def phase2_l_ablation(embeddings, gt, cfg, best_w):
    N = len(embeddings)
    out_dim = cfg["out_dim"]

    print(f"{'─'*60}")
    print(f"  Phase 2 — L ablation  (w={best_w}, K={K_FUNCS})")
    print(f"{'─'*60}")

    results = {}
    for L in L_GRID:
        t_build = time.time()
        lsh = _build_lsh(embeddings, out_dim, n_tables=L, n_funcs=K_FUNCS, w=best_w)
        build_s = time.time() - t_build
        mem_kb = _table_memory_bytes(lsh) / 1024

        k_results = {}
        for k in [5, 10, 20]:
            precs, recs, aps, aqs, qtimes, bftimes = [], [], [], [], [], []

            for i in range(N):
                z_q     = embeddings[i]
                relevant = gt[i]

                t0      = time.time()
                lsh_top = lsh.query(z_q, embeddings, k=k)
                qtimes.append((time.time() - t0) * 1000)
                lsh_top = [x for x in lsh_top if x != i][:k]

                t0     = time.time()
                bf_top = _brute_force(z_q, embeddings, k).tolist()
                bftimes.append((time.time() - t0) * 1000)

                precs.append(_precision_at_k(lsh_top, relevant))
                recs.append(_recall_at_k(lsh_top, relevant))
                aps.append(_average_precision(lsh_top, relevant))
                aqs.append(_approx_quality(lsh_top, bf_top))

            k_results[k] = {
                "Precision@k":    float(np.mean(precs)),
                "Recall@k":       float(np.mean(recs)),
                "MAP":            float(np.mean(aps)),
                "Approx Quality": float(np.mean(aqs)),
                "Query Time(ms)": float(np.mean(qtimes)),
                "BF Time(ms)":    float(np.mean(bftimes)),
            }

        results[L] = {
            "build_time_s": round(build_s, 4),
            "memory_kb":    round(mem_kb, 2),
            "metrics":      k_results,
        }

    for k in [5, 10, 20]:
        col = 14
        print(f"\n  k = {k}")
        hdr = (f"  {'Metric':<22}"
               + "".join(f"  {'L='+str(L):>{col}}" for L in L_GRID))
        print(hdr)
        print("  " + "─" * (22 + (col + 2) * len(L_GRID)))

        for m in ["Precision@k", "Recall@k", "MAP", "Approx Quality", "Query Time(ms)"]:
            row = f"  {m:<22}"
            for L in L_GRID:
                v = results[L]["metrics"][k][m]
                row += f"  {v:>{col}.4f}"
            print(row)

    print(f"\n  {'L':>4}  {'Build(s)':>10}  {'Memory(KB)':>12}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*12}")
    for L in L_GRID:
        print(f"  {L:>4}  {results[L]['build_time_s']:>10.4f}  {results[L]['memory_kb']:>12.2f}")

    return results


def ablate(dataset_name: str, use_ged: bool):
    cfg = CONFIGS[dataset_name]

    print(f"\n{'═'*60}")
    print(f"  LSH Ablation — {dataset_name.upper()}")
    print(f"  GT oracle : {'GED' if use_ged else 'label-proxy'}")
    print(f"{'═'*60}")

    embeddings, dataset = _load_outputs(dataset_name)
    gt = _load_ground_truth(dataset_name, dataset, cfg, use_ged)

    best_w, w_results = phase1_w_search(embeddings, gt, cfg)
    l_results         = phase2_l_ablation(embeddings, gt, cfg, best_w)

    out_dir = f"outputs/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/lsh_ablation.json"

    save = {
        "dataset":        dataset_name,
        "gt_type":        "ged" if use_ged else "label",
        "w_grid_search":  {str(k): v for k, v in w_results.items()},
        "best_w":         best_w,
        "L_ablation":     {str(k): v for k, v in l_results.items()},
        "K_fixed":        K_FUNCS,
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=4)
    print(f"\n  Saved → {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSH parameter ablation (Proposal §2.3)")
    parser.add_argument(
        "--dataset", type=str, default="mutag",
        choices=["all", "aids", "imdb-binary", "mutag", "proteins", "reddit-binary"],
    )
    parser.add_argument(
        "--gt", type=str, default="label",
        choices=["label", "ged"],
        help="label: class-label proxy (fast)  |  ged: GED-based ground truth",
    )
    args = parser.parse_args()

    datasets = (
        ["aids", "imdb-binary", "mutag", "proteins", "reddit-binary"]
        if args.dataset == "all"
        else [args.dataset]
    )
    for ds in datasets:
        ablate(ds, use_ged=(args.gt == "ged"))
