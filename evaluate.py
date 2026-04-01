import torch
import numpy as np
import pickle
import time
import json
import os
import argparse
from core.lsh import LSHIndex
from core.model import GINEncoder
from core.train import embed_all
from config import CONFIGS


def load_outputs(dataset_name):
    out_dir = f"outputs/{dataset_name}"
    embeddings = np.load(f"{out_dir}/embeddings.npy")
    with open(f"{out_dir}/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    return embeddings, dataset


def get_ground_truth_label(dataset):
    gt = {}
    for i, g in enumerate(dataset):
        label = g.y.item()
        gt[i] = set(j for j, d in enumerate(dataset) if d.y.item() == label and j != i)
    return gt


def load_ged_ground_truth(dataset_name, dataset, cfg):
    from core.ged_cache import build_or_load_ged_matrix, get_ground_truth_ged
    ged_matrix, threshold = build_or_load_ged_matrix(dataset, dataset_name, cfg)
    gt = get_ground_truth_ged(ged_matrix, threshold)
    return gt, threshold


def brute_force_query(z_q, embeddings, k=5):
    dists = np.linalg.norm(embeddings - z_q, axis=1)
    top_k = np.argsort(dists)[1:k+1]   # exclude self
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
    """Overlap between LSH results and brute-force results. 1.0 = perfect."""
    lsh_set = set(lsh_retrieved)
    bf_set  = set(bf_retrieved)
    return len(lsh_set & bf_set) / len(bf_set) if bf_set else 0.0


def eval_embeddings(embeddings, dataset, cfg, gt):
    import sys

    # Build LSH index and time it
    t_idx_start = time.time()
    lsh = LSHIndex(
        dim      = cfg["out_dim"],
        n_tables = cfg["n_tables"],
        n_funcs  = cfg["n_funcs"],
        w        = cfg["w"],
    )
    lsh.index(embeddings)
    index_time = time.time() - t_idx_start

    # Approximate memory of hash tables (sum of defaultdict sizes)
    table_mem_bytes = sum(sys.getsizeof(table) for (_, _, table) in lsh.tables)

    metrics_by_k = {}
    for k in [5, 10, 20]:
        lsh_prec, lsh_rec, lsh_ap, lsh_aq = [], [], [], []
        bf_prec,  bf_rec,  bf_ap          = [], [], []
        lsh_times, bf_times               = [], []

        for i in range(len(dataset)):
            z_q      = embeddings[i]
            relevant = gt[i]

            # LSH query
            t0 = time.time()
            lsh_top = lsh.query(z_q, embeddings, k=k)
            lsh_times.append(time.time() - t0)

            # Brute-force query
            t0 = time.time()
            bf_top = brute_force_query(z_q, embeddings, k=k)
            bf_times.append(time.time() - t0)

            # Exclude self from LSH results
            lsh_top = [x for x in lsh_top if x != i][:k]

            lsh_prec.append(precision_at_k(lsh_top, relevant))
            lsh_rec.append(recall_at_k(lsh_top, relevant))
            lsh_ap.append(average_precision(lsh_top, relevant))
            lsh_aq.append(approximation_quality(lsh_top, bf_top))

            bf_prec.append(precision_at_k(bf_top, relevant))
            bf_rec.append(recall_at_k(bf_top, relevant))
            bf_ap.append(average_precision(bf_top, relevant))

        metrics_by_k[k] = {
            "lsh": {
                "Precision@k":    float(np.mean(lsh_prec)),
                "Recall@k":       float(np.mean(lsh_rec)),
                "MAP":            float(np.mean(lsh_ap)),
                "Approx Quality": float(np.mean(lsh_aq)),
                "Query Time(ms)": float(np.mean(lsh_times) * 1000),
            },
            "bf": {
                "Precision@k":    float(np.mean(bf_prec)),
                "Recall@k":       float(np.mean(bf_rec)),
                "MAP":            float(np.mean(bf_ap)),
                "Query Time(ms)": float(np.mean(bf_times) * 1000),
            },
            "index": {
                "construction_time_s": round(index_time, 4),
                "table_mem_bytes":     table_mem_bytes,
                "table_mem_kb":        round(table_mem_bytes / 1024, 2),
            },
        }

    return metrics_by_k


def print_table1(results_trained, results_untrained, k=10):
    """
    Proposal Table 1: three systems side-by-side at a given k.
      Brute-force (Trained GIN + exhaustive)
      LSH-ANN     (Trained GIN + LSH)
      Untrained-ANN (Random GIN + LSH)
    """
    bf  = results_trained[k]["bf"]
    lsh = results_trained[k]["lsh"]
    un  = results_untrained[k]["lsh"]

    col_w   = 16
    metrics = ["Precision@k", "Recall@k", "MAP", "Query Time(ms)"]

    header = (
        f"{'Metric':<22}"
        f" {'Brute-force':>{col_w}}"
        f" {'LSH-ANN':>{col_w}}"
        f" {'Untrained-ANN':>{col_w}}"
    )
    sep = "─" * len(header)

    print(f"\n{'─'*4} Table 1 — Three-System Comparison  (k={k}) {'─'*4}")
    print(header)
    print(sep)
    for m in metrics:
        print(
            f"  {m:<20}"
            f" {bf.get(m, float('nan')):>{col_w}.4f}"
            f" {lsh.get(m, float('nan')):>{col_w}.4f}"
            f" {un.get(m, float('nan')):>{col_w}.4f}"
        )
    # Approx Quality: N/A for brute-force
    aq_lsh = lsh.get("Approx Quality", float("nan"))
    aq_un  = un.get("Approx Quality", float("nan"))
    print(
        f"  {'Approx Quality':<20}"
        f" {'—':>{col_w}}"
        f" {aq_lsh:>{col_w}.4f}"
        f" {aq_un:>{col_w}.4f}"
    )
    print(sep)


def print_multik_table(results, label):
    """Multi-k table (k=5,10,20) for one system."""
    print(f"\n{'─'*4} {label} {'─'*4}")
    print(f"{'Metric':<25} {'k=5':>10} {'k=10':>10} {'k=20':>10}")
    print("─" * 57)
    metrics = ["Precision@k", "Recall@k", "MAP", "Approx Quality", "Query Time(ms)"]
    for m in metrics:
        v5  = results[5]["lsh"].get(m, float("nan"))
        v10 = results[10]["lsh"].get(m, float("nan"))
        v20 = results[20]["lsh"].get(m, float("nan"))
        print(f"  {m:<23} {v5:>10.4f} {v10:>10.4f} {v20:>10.4f}")
    print()


def print_index_stats(results, dataset_name):
    """Index construction time and memory (from k=5 entry, same for all k)."""
    idx = results[5]["index"]
    print(f"{'─'*4} Index Stats — {dataset_name} {'─'*4}")
    print(f"  Construction time : {idx['construction_time_s']:.4f} s")
    print(f"  Hash table memory : {idx['table_mem_kb']:.2f} KB")
    print()


def evaluate(dataset_name, mode="both", use_ged=False):
    cfg    = CONFIGS[dataset_name]
    gt_tag = f"GED ({cfg['ged_method']})" if use_ged else "label-proxy"

    print(f"\n{'═'*60}")
    print(f"  Dataset  : {dataset_name.upper()}")
    print(f"  Mode     : {mode}")
    print(f"  GT       : {gt_tag}")
    print(f"{'═'*60}\n")

    trained_embeddings, dataset = load_outputs(dataset_name)

    # Ground truth
    if use_ged:
        gt, threshold = load_ged_ground_truth(dataset_name, dataset, cfg)
        gt_meta = {"type": "ged", "method": cfg["ged_method"], "threshold": threshold}
    else:
        gt = get_ground_truth_label(dataset)
        gt_meta = {"type": "label"}

    # Trained GIN
    results_trained = None
    if mode in ["trained", "both"]:
        print("▶ Evaluating Trained GIN...")
        results_trained = eval_embeddings(trained_embeddings, dataset, cfg, gt)
        print_multik_table(results_trained, label="Trained GIN + LSH")
        print_index_stats(results_trained, dataset_name)

    # Untrained GIN (random weights)
    results_untrained = None
    if mode in ["untrained", "both"]:
        print("▶ Evaluating Untrained GIN (random weights)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        untrained_model = GINEncoder(
            in_dim     = cfg["in_dim"],
            hidden_dim = cfg["hidden_dim"],
            out_dim    = cfg["out_dim"],
            num_layers = cfg["num_layers"],
        ).to(device)
        untrained_embeddings = embed_all(untrained_model, dataset, device).numpy()
        results_untrained = eval_embeddings(untrained_embeddings, dataset, cfg, gt)
        print_multik_table(results_untrained, label="Untrained GIN + LSH")

    # Table 1: all three systems side-by-side
    if mode == "both" and results_trained and results_untrained:
        for k in [5, 10, 20]:
            print_table1(results_trained, results_untrained, k=k)

    # Save JSON
    out_dir  = f"outputs/{dataset_name}"
    out_file = f"{out_dir}/evaluation_results.json"
    os.makedirs(out_dir, exist_ok=True)

    save_data = {
        "dataset": dataset_name,
        "mode":    mode,
        "gt_meta": gt_meta,
        "metrics": {},
    }
    if results_trained:
        save_data["metrics"]["trained"]   = results_trained
    if results_untrained:
        save_data["metrics"]["untrained"] = results_untrained

    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=4)
    print(f"Saved → {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate graph retrieval pipeline")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "aids", "imdb-binary", "mutag", "proteins", "reddit-binary"],
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["trained", "untrained", "both"],
    )
    parser.add_argument(
        "--gt", type=str, default="label",
        choices=["label", "ged"],
        help=(
            "label : same class label = relevant (fast proxy, default)\n"
            "ged   : exact GED for MUTAG/AIDS/PROTEINS, beam B=20 for IMDB-B/Reddit\n"
            "        Pairwise GED matrix computed once and cached to outputs/{dataset}/ged_matrix.npy"
        ),
    )
    args = parser.parse_args()

    datasets_to_run = (
        ["aids", "imdb-binary", "mutag", "proteins", "reddit-binary"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds in datasets_to_run:
        evaluate(ds, mode=args.mode, use_ged=(args.gt == "ged"))