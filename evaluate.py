import torch
import numpy as np
import pickle
import time
from core.lsh import LSHIndex
from core.model import GINEncoder
from config import CONFIGS


def load_outputs(dataset_name):
    out_dir = f"outputs/{dataset_name}"
    embeddings = np.load(f"{out_dir}/embeddings.npy")
    with open(f"{out_dir}/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    return embeddings, dataset


def get_ground_truth(dataset):
    """
    Ground truth: graphs with the same label as query are relevant.
    Returns a dict: query_idx -> set of relevant indices
    """
    gt = {}
    for i, g in enumerate(dataset):
        label = g.y.item()
        gt[i] = set(j for j, d in enumerate(dataset) if d.y.item() == label and j != i)
    return gt


def brute_force_query(z_q, embeddings, k=5):
    dists = np.linalg.norm(embeddings - z_q, axis=1)
    top_k = np.argsort(dists)[1:k+1]  # exclude self (index 0)
    return top_k


def precision_at_k(retrieved, relevant):
    retrieved = list(retrieved)
    return len(set(retrieved) & relevant) / len(retrieved) if retrieved else 0.0


def recall_at_k(retrieved, relevant):
    retrieved = list(retrieved)
    return len(set(retrieved) & relevant) / len(relevant) if relevant else 0.0


def average_precision(retrieved, relevant):
    retrieved = list(retrieved)
    hits, score = 0, 0.0
    for i, idx in enumerate(retrieved):
        if idx in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0


def approximation_quality(lsh_retrieved, bf_retrieved):
    """
    Ratio of overlap between LSH results and brute-force results.
    1.0 = perfect match with brute-force.
    """
    lsh_set = set(lsh_retrieved)
    bf_set  = set(bf_retrieved)
    return len(lsh_set & bf_set) / len(bf_set) if bf_set else 0.0


def evaluate(dataset_name, k=5):
    cfg = CONFIGS[dataset_name]

    print(f"\nEvaluating on {dataset_name.upper()} | k={k}\n")

    # Load saved outputs
    embeddings, dataset = load_outputs(dataset_name)
    gt = get_ground_truth(dataset)

    # Build LSH index
    lsh = LSHIndex(
        dim      = cfg["out_dim"],
        n_tables = cfg["n_tables"],
        n_funcs  = cfg["n_funcs"],
        w        = cfg["w"],
    )
    lsh.index(embeddings)

    # Per-query evaluation
    lsh_prec, lsh_rec, lsh_ap, lsh_aq = [], [], [], []
    bf_prec,  bf_rec,  bf_ap          = [], [], []
    lsh_times, bf_times               = [], []

    for i in range(len(dataset)):
        z_q     = embeddings[i]
        relevant = gt[i]

        # LSH query
        t0 = time.time()
        lsh_top = lsh.query(z_q, embeddings, k=k)
        lsh_times.append(time.time() - t0)

        # Brute-force query
        t0 = time.time()
        bf_top = brute_force_query(z_q, embeddings, k=k)
        bf_times.append(time.time() - t0)

        # exclude self from lsh results
        lsh_top = [x for x in lsh_top if x != i][:k]

        # LSH metrics
        lsh_prec.append(precision_at_k(lsh_top, relevant))
        lsh_rec.append(recall_at_k(lsh_top, relevant))
        lsh_ap.append(average_precision(lsh_top, relevant))
        lsh_aq.append(approximation_quality(lsh_top, bf_top))

        # Brute-force metrics
        bf_prec.append(precision_at_k(bf_top, relevant))
        bf_rec.append(recall_at_k(bf_top, relevant))
        bf_ap.append(average_precision(bf_top, relevant))

    # Print results
    print(f"{'Metric':<25} {'LSH-ANN':>10} {'Brute-force':>12}")
    print("-" * 50)
    print(f"{'Precision@k':<25} {np.mean(lsh_prec):>10.4f} {np.mean(bf_prec):>12.4f}")
    print(f"{'Recall@k':<25} {np.mean(lsh_rec):>10.4f} {np.mean(bf_rec):>12.4f}")
    print(f"{'MAP':<25} {np.mean(lsh_ap):>10.4f} {np.mean(bf_ap):>12.4f}")
    print(f"{'Avg query time (ms)':<25} {np.mean(lsh_times)*1000:>10.4f} {np.mean(bf_times)*1000:>12.4f}")
    print(f"{'Approximation quality':<25} {np.mean(lsh_aq):>10.4f} {'—':>12}")
    print("-" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mutag",
                        choices=["aids", "imdb-binary", "mutag", "proteins", "reddit-binary"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.dataset, k=args.k)