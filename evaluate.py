import torch
import numpy as np
import pickle
import time
import json
import os
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


def eval_embeddings(embeddings, dataset, cfg):
    gt = get_ground_truth(dataset)

    # Build LSH index
    lsh = LSHIndex(
        dim      = cfg["out_dim"],
        n_tables = cfg["n_tables"],
        n_funcs  = cfg["n_funcs"],
        w        = cfg["w"],
    )
    lsh.index(embeddings)

    metrics_by_k = {}
    for k in [5, 10, 20]:
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

        metrics_by_k[k] = {
            "lsh": {
                "Precision@k": float(np.mean(lsh_prec)),
                "Recall@k": float(np.mean(lsh_rec)),
                "MAP": float(np.mean(lsh_ap)),
                "Approx Quality": float(np.mean(lsh_aq)),
                "Query Time(ms)": float(np.mean(lsh_times)*1000)
            },
            "bf": {
                "Precision@k": float(np.mean(bf_prec)),
                "Recall@k": float(np.mean(bf_rec)),
                "MAP": float(np.mean(bf_ap)),
                "Query Time(ms)": float(np.mean(bf_times)*1000)
            }
        }
    return metrics_by_k


def evaluate(dataset_name, mode="trained"):
    cfg = CONFIGS[dataset_name]

    print(f"\nEvaluating on {dataset_name.upper()} | mode={mode}\n")

    # Load saved outputs
    trained_embeddings, dataset = load_outputs(dataset_name)

    results_trained = None
    results_untrained = None

    if mode in ["trained", "both"]:
        results_trained = eval_embeddings(trained_embeddings, dataset, cfg)

    if mode in ["untrained", "both"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GINEncoder(
            in_dim     = cfg["in_dim"],
            hidden_dim = cfg["hidden_dim"],
            out_dim    = cfg["out_dim"],
            num_layers = cfg["num_layers"],
        ).to(device)
        untrained_embeddings = embed_all(model, dataset, device).numpy()
        results_untrained = eval_embeddings(untrained_embeddings, dataset, cfg)

    if mode == "both":
        print(f"{'Metric':<25} {'k=5':<10} {'k=10':<10} {'k=20':<10}")
        print("-" * 55)
        print("[TRAINED GIN]")
        for metric in ["Precision@k", "Recall@k", "MAP", "Approx Quality", "Query Time(ms)"]:
            m5 = results_trained[5]["lsh"][metric]
            m10 = results_trained[10]["lsh"][metric]
            m20 = results_trained[20]["lsh"][metric]
            print(f"  {metric:<23} {m5:<10.4f} {m10:<10.4f} {m20:<10.4f}")
            
        print("[UNTRAINED GIN]")
        for metric in ["Precision@k", "Recall@k", "MAP", "Approx Quality", "Query Time(ms)"]:
            m5 = results_untrained[5]["lsh"][metric]
            m10 = results_untrained[10]["lsh"][metric]
            m20 = results_untrained[20]["lsh"][metric]
            print(f"  {metric:<23} {m5:<10.4f} {m10:<10.4f} {m20:<10.4f}")

    else:
        results = results_trained if mode == "trained" else results_untrained
        for k in [5, 10, 20]:
            print(f"--- k={k} ---")
            print(f"{'Metric':<25} {'LSH-ANN':>10} {'Brute-force':>12}")
            print("-" * 50)
            res_lsh = results[k]["lsh"]
            res_bf = results[k]["bf"]
            print(f"{'Precision@k':<25} {res_lsh['Precision@k']:>10.4f} {res_bf['Precision@k']:>12.4f}")
            print(f"{'Recall@k':<25} {res_lsh['Recall@k']:>10.4f} {res_bf['Recall@k']:>12.4f}")
            print(f"{'MAP':<25} {res_lsh['MAP']:>10.4f} {res_bf['MAP']:>12.4f}")
            print(f"{'Avg query time (ms)':<25} {res_lsh['Query Time(ms)']:>10.4f} {res_bf['Query Time(ms)']:>12.4f}")
            print(f"{'Approximation quality':<25} {res_lsh['Approx Quality']:>10.4f} {'—':>12}")
            print("-" * 50)
            print()
            
    # Save to JSON
    out_dir = f"outputs/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/evaluation_results.json"
    
    save_data = {
        "dataset": dataset_name,
        "mode": mode,
        "metrics": {}
    }
    
    if mode in ["trained", "both"]:
        save_data["metrics"]["trained"] = results_trained
    if mode in ["untrained", "both"]:
        save_data["metrics"]["untrained"] = results_untrained
        
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=4)
        
    print(f"\nSaved evaluation results to {out_file}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "aids", "imdb-binary", "mutag", "proteins", "reddit-binary"])
    parser.add_argument("--mode", type=str, default="both",
                        choices=["trained", "untrained", "both"])
    args = parser.parse_args()
    
    datasets_to_run = ["aids", "imdb-binary", "mutag", "proteins", "reddit-binary"] if args.dataset == "all" else [args.dataset]
    
    for ds in datasets_to_run:
        evaluate(ds, mode=args.mode)