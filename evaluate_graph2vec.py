import os
import json
import argparse
import pickle
import numpy as np

from config import CONFIGS
from core.dataset import load_jsonl
from core.graph2vec import graph2vec_embed

# Reuse evaluation machinery from evaluate.py — no logic duplication.
from evaluate import (
    eval_embeddings,
    get_ground_truth_label,
    load_ged_ground_truth,
    print_multik_table,
    print_index_stats,
)


def evaluate_graph2vec(dataset_name: str, use_ged: bool = False):
    """Run the full Graph2Vec → LSH evaluation for one dataset."""
    cfg    = CONFIGS[dataset_name]
    gt_tag = f"GED ({cfg['ged_method']})" if use_ged else "label-proxy"

    print(f"\n{'═' * 60}")
    print(f"  Dataset  : {dataset_name.upper()}")
    print(f"  Encoder  : Graph2Vec")
    print(f"  GT       : {gt_tag}")
    print(f"{'═' * 60}\n")

    dataset = load_jsonl(cfg["path"], max_degree=cfg["max_degree"])
    print(f"Loaded {len(dataset)} graphs from {cfg['path']}")

    if use_ged:
        gt, threshold = load_ged_ground_truth(dataset_name, dataset, cfg)
        gt_meta = {"type": "ged", "method": cfg["ged_method"],
                   "threshold": threshold}
    else:
        gt = get_ground_truth_label(dataset)
        gt_meta = {"type": "label"}

    embeddings = graph2vec_embed(dataset, cfg, dataset_name)
    assert embeddings.shape == (len(dataset), cfg["out_dim"]), (
        f"Expected shape ({len(dataset)}, {cfg['out_dim']}), "
        f"got {embeddings.shape}"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"Embeddings are not L2-normalised (mean norm = {norms.mean():.4f})"
    )

    # --- DIAGNOSTICS: Check for embedding collapse ---
    print("\n[Diagnostics] Running embedding space checks...")
    
    # Check 1: Are embeddings actually different from each other?
    print(f"Embedding std per dim: {embeddings.std(axis=0).mean():.6f}")
    
    # Distances between all pairs
    pw_dists = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=2).flatten()
    sorted_dists = np.sort(pw_dists)
    # The first N distances are 0 (self-distances), so the first non-zero distance is at index N
    min_dist = sorted_dists[len(dataset)]
    print(f"Min pairwise distance: {min_dist:.6f}")

    # Check 2: Are same-class graphs close in embedding space?
    class_0_idx = [i for i, d in enumerate(dataset) if d.y.item() == 0]
    class_1_idx = [i for i, d in enumerate(dataset) if d.y.item() == 1]
    
    if len(class_0_idx) > 0 and len(class_1_idx) > 0:
        c0_sample = class_0_idx[:10]
        c1_sample = class_1_idx[:10]
        within_class = np.linalg.norm(
            embeddings[c0_sample, None] - embeddings[None, c0_sample], axis=2).mean()
        across_class = np.linalg.norm(
            embeddings[c0_sample, None] - embeddings[None, c1_sample], axis=2).mean()
        print(f"Within-class avg distance: {within_class:.4f}")
        print(f"Across-class avg distance: {across_class:.4f}")

    # Check 3: Print 3 sample node label conversions (only for featured datasets)
    from core.graph2vec import FEATURELESS_DATASETS
    if dataset_name.lower() not in FEATURELESS_DATASETS:
        for i, data in enumerate(dataset[:3]):
            labels = data.x.argmax(dim=1).tolist()
            print(f"Graph {i}: {len(labels)} nodes, argmax labels sample: {labels[:5]}")
    else:
        # Just show degrees instead of features
        for i, data in enumerate(dataset[:3]):
            import networkx as nx
            from core.graph2vec import pyg_to_networkx
            g = pyg_to_networkx(data, use_degree_label=True)
            labels = [g.nodes[n]["label"] for n in list(g.nodes())]
            print(f"Graph {i}: {len(labels)} nodes, degree labels sample: {labels[:5]}")
    print("-------------------------------------------------")

    print("Evaluating Graph2Vec embeddings ...")
    results_g2v = eval_embeddings(embeddings, dataset, cfg, gt)

    print_multik_table(results_g2v, label="Graph2Vec + LSH")
    print_index_stats(results_g2v, dataset_name)

    out_dir  = f"outputs/{dataset_name}"
    out_file = f"{out_dir}/evaluation_results_graph2vec.json"
    os.makedirs(out_dir, exist_ok=True)

    save_data = {
        "dataset": dataset_name,
        "mode":    "graph2vec",
        "gt_meta": gt_meta,
        "metrics": {
            "graph2vec": results_g2v,
        },
    }
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=4)
    print(f"Saved → {out_file}")

    # Also save raw embeddings for potential re-use / plotting
    emb_file = f"{out_dir}/embeddings_graph2vec.npy"
    np.save(emb_file, embeddings)
    print(f"Saved → {emb_file}")

    gin_results_file = f"{out_dir}/evaluation_results.json"
    if os.path.exists(gin_results_file):
        _print_comparison(gin_results_file, results_g2v)
    else:
        print(f"\n[info] No trained GIN results found at {gin_results_file} "
              f"— skipping comparison table.")

    if os.path.exists(gin_results_file):
        _validate_schema(gin_results_file, save_data)

    print()


def _print_comparison(gin_json_path: str, results_g2v: dict):
    with open(gin_json_path) as f:
        gin_data = json.load(f)

    gin_metrics = gin_data.get("metrics", {})
    has_trained   = "trained"   in gin_metrics
    has_untrained = "untrained" in gin_metrics

    if not has_trained:
        return

    for k in [5, 10, 20]:
        bf  = gin_metrics["trained"][str(k)]["bf"]
        lsh = gin_metrics["trained"][str(k)]["lsh"]
        g2v = results_g2v[k]["lsh"]

        systems = [
            ("Brute-force",  bf),
            ("LSH-ANN",      lsh),
        ]
        if has_untrained:
            un = gin_metrics["untrained"][str(k)]["lsh"]
            systems.append(("Untrained-ANN", un))
        systems.append(("Graph2Vec-ANN", g2v))

        col_w   = 16
        n_cols  = len(systems)
        metrics = ["Precision@k", "Recall@k", "MAP", "Query Time(ms)"]

        header = f"{'Metric':<22}"
        for name, _ in systems:
            header += f" {name:>{col_w}}"
        sep = "─" * len(header)

        print(f"\n{'─'*4} All Systems Comparison  (k={k}) {'─'*4}")
        print(header)
        print(sep)

        for m in metrics:
            row = f"  {m:<20}"
            for _, vals in systems:
                row += f" {vals.get(m, float('nan')):>{col_w}.4f}"
            print(row)

        # Approx Quality row (N/A for brute-force)
        row = f"  {'Approx Quality':<20}"
        for name, vals in systems:
            if name == "Brute-force":
                row += f" {'—':>{col_w}}"
            else:
                row += f" {vals.get('Approx Quality', float('nan')):>{col_w}.4f}"
        print(row)
        print(sep)


def _validate_schema(gin_json_path: str, new_data: dict):
    """Light schema check: verify top-level keys match existing results."""
    with open(gin_json_path) as f:
        existing = json.load(f)

    expected_top = {"dataset", "mode", "gt_meta", "metrics"}
    actual_top   = set(new_data.keys())
    if actual_top != expected_top:
        print(f"[warn] Schema mismatch — expected keys {expected_top}, "
              f"got {actual_top}")
        return

    # Check that per-k metric structure matches
    # (use the first available system from existing results)
    for system_key in ("trained", "untrained"):
        if system_key in existing["metrics"]:
            ref_ks = set(existing["metrics"][system_key].keys())
            break
    else:
        return

    g2v_ks = set(str(k) for k in new_data["metrics"]["graph2vec"].keys())
    if g2v_ks != ref_ks:
        print(f"[warn] k-value mismatch — existing has {ref_ks}, "
              f"Graph2Vec has {g2v_ks}")
        return

    # Check per-k sub-keys (lsh, bf, index)
    for system_key in ("trained", "untrained"):
        if system_key in existing["metrics"]:
            sample_k = list(existing["metrics"][system_key].keys())[0]
            expected_subkeys = set(existing["metrics"][system_key][sample_k].keys())
            break

    sample_g2v_k = list(new_data["metrics"]["graph2vec"].keys())[0]
    actual_subkeys = set(new_data["metrics"]["graph2vec"][sample_g2v_k].keys())
    if actual_subkeys != expected_subkeys:
        print(f"[warn] Per-k sub-key mismatch — expected {expected_subkeys}, "
              f"got {actual_subkeys}")
        return

    print("[✓] JSON schema matches existing evaluation results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Graph2Vec baseline on graph retrieval"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "aids", "imdb-binary", "mutag", "proteins",
                 "reddit-binary"],
    )
    parser.add_argument(
        "--gt", type=str, default="label",
        choices=["label", "ged"],
        help=(
            "label : same class label = relevant (fast proxy, default)\n"
            "ged   : exact GED for MUTAG, beam B=20 for others\n"
            "        Reuses cached GED matrix from outputs/{dataset}/"
        ),
    )
    args = parser.parse_args()

    datasets_to_run = (
        ["aids", "imdb-binary", "mutag", "proteins", "reddit-binary"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds in datasets_to_run:
        evaluate_graph2vec(ds, use_ged=(args.gt == "ged"))
