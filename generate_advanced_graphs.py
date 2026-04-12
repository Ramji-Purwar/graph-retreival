"""
Advanced analyses for the project report.
Generates: training curves, t-SNE plots, GED oracle validation,
distance distributions, candidate set sizes, scaling experiment,
NN purity, retrieval examples, memory-recall tradeoff, w×L grid.
"""
import json, os, sys, io, pickle, time, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# ── project imports ──
sys.path.insert(0, ".")
from config import CONFIGS
from core.model import GINEncoder
from core.dataset import load_jsonl
from core.lsh import LSHIndex
from core.train import sample_triplets, _batch_encode

OUT = "graph"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.dpi": 200,
})

DATASETS = ["mutag", "proteins", "aids", "imdb-binary", "reddit-binary"]
PRETTY = {"mutag": "MUTAG", "proteins": "PROTEINS", "aids": "AIDS",
           "imdb-binary": "IMDB-Binary", "reddit-binary": "Reddit-Binary"}
COLORS5 = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#8b5cf6"]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                        weights_only=False)
        return super().find_class(module, name)


def load_stuff(ds):
    d = f"outputs/{ds}"
    emb = np.load(f"{d}/embeddings.npy")
    with open(f"{d}/dataset.pkl", "rb") as f:
        dataset = CPU_Unpickler(f).load()
    return emb, dataset


def load_ged(ds, kind="ged"):
    """kind = 'ged' (eval, exact/B=20) or 'train' (B=5)"""
    d = f"outputs/{ds}"
    prefix = "train_" if kind == "train" else ""
    mat = np.load(f"{d}/{prefix}ged_matrix.npy")
    thr = float(np.load(f"{d}/{prefix}ged_threshold.npy"))
    return mat, thr


# ====================================================================
# 1. TRAINING CURVES  (retrain MUTAG quickly with logging)
# ====================================================================
def gen_training_curves():
    print("\n=== Training curves (MUTAG, 50 epochs) ===")
    ds_name = "mutag"
    cfg = CONFIGS[ds_name]
    dataset = load_jsonl(cfg["path"], max_degree=cfg["max_degree"])
    device = torch.device("cpu")

    ged_matrix, ged_threshold = load_ged(ds_name, "train")

    model = GINEncoder(in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"],
                       out_dim=cfg["out_dim"], num_layers=cfg["num_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    epoch_losses = []
    epoch_precisions = []

    # ground truth for quick precision eval
    eval_ged, eval_thr = load_ged(ds_name, "ged")
    gt = {}
    for i in range(len(dataset)):
        gt[i] = set(j for j in range(len(dataset)) if j != i and eval_ged[i, j] <= eval_thr)

    for epoch in range(50):
        model.train()
        triplets = sample_triplets(dataset, n=512,
                                   ged_matrix=ged_matrix, threshold=ged_threshold)
        total_loss = 0.0
        n_batches = 0
        for bs in range(0, len(triplets), 64):
            batch_t = triplets[bs:bs+64]
            a_idx = [t[0] for t in batch_t]
            p_idx = [t[1] for t in batch_t]
            n_idx = [t[2] for t in batch_t]
            optimizer.zero_grad()
            za = _batch_encode(model, a_idx, dataset, device)
            zp = _batch_encode(model, p_idx, dataset, device)
            zn = _batch_encode(model, n_idx, dataset, device)
            loss = loss_fn(za, zp, zn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)

        # quick P@10 eval every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            loader = DataLoader(dataset, batch_size=256, num_workers=0)
            embs = []
            with torch.no_grad():
                for batch in loader:
                    z = model(batch.x, batch.edge_index, batch.batch)
                    embs.append(z.cpu())
            embs = torch.cat(embs).numpy()
            precs = []
            for i in range(len(dataset)):
                dists = np.linalg.norm(embs - embs[i], axis=1)
                top10 = np.argsort(dists)[1:11]
                precs.append(len(set(top10) & gt[i]) / 10)
            epoch_precisions.append((epoch + 1, np.mean(precs)))
            print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | P@10: {np.mean(precs):.4f}")
        else:
            print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Training Diagnostics — MUTAG (50 Epochs, Triplet Loss)", fontweight="bold")

    ax1.plot(range(1, 51), epoch_losses, color="#2563eb", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Triplet Loss")
    ax1.set_title("Training Loss Curve")
    ax1.grid(alpha=0.3)

    ep, pr = zip(*epoch_precisions)
    ax2.plot(ep, pr, "o-", color="#16a34a", linewidth=2, markersize=7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Precision@10 (GED ground truth)")
    ax2.set_title("Validation Precision@10 vs Epoch")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved training_curves.png")
    return epoch_losses, epoch_precisions


# ====================================================================
# 2. t-SNE / UMAP of trained vs untrained embeddings
# ====================================================================
def gen_tsne_plots():
    print("\n=== t-SNE plots ===")
    from sklearn.manifold import TSNE

    for ds_name in ["mutag", "imdb-binary"]:
        cfg = CONFIGS[ds_name]
        emb_trained, dataset = load_stuff(ds_name)
        labels = np.array([d.y.item() for d in dataset])

        # untrained embeddings
        device = torch.device("cpu")
        untrained_model = GINEncoder(
            in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"],
            out_dim=cfg["out_dim"], num_layers=cfg["num_layers"],
        ).to(device)
        untrained_model.eval()
        loader = DataLoader(dataset, batch_size=256, num_workers=0)
        emb_untrained = []
        with torch.no_grad():
            for batch in loader:
                z = untrained_model(batch.x, batch.edge_index, batch.batch)
                emb_untrained.append(z.cpu())
        emb_untrained = torch.cat(emb_untrained).numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"t-SNE Visualization — {PRETTY[ds_name]}", fontweight="bold")

        for ax, emb, title in [(ax1, emb_untrained, "Untrained GIN"),
                                (ax2, emb_trained, "Trained GIN (Triplet Loss)")]:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
            z2d = tsne.fit_transform(emb)
            unique_labels = sorted(set(labels))
            cmap = plt.cm.Set1 if len(unique_labels) <= 9 else plt.cm.tab20
            for li, lbl in enumerate(unique_labels):
                mask = labels == lbl
                ax.scatter(z2d[mask, 0], z2d[mask, 1], c=[cmap(li / max(len(unique_labels)-1, 1))],
                           label=f"Class {lbl}", s=20, alpha=0.7)
            ax.set_title(title)
            ax.legend(fontsize=8, markerscale=2)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(f"{OUT}/tsne_{ds_name.replace('-','_')}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved tsne_{ds_name.replace('-','_')}.png")


# ====================================================================
# 3. GED Oracle Validation (beam B=5 vs exact on MUTAG)
# ====================================================================
def gen_ged_oracle_validation():
    print("\n=== GED Oracle Validation (MUTAG) ===")
    exact_mat, exact_thr = load_ged("mutag", "ged")
    beam_mat, beam_thr = load_ged("mutag", "train")

    N = exact_mat.shape[0]
    triu = np.triu_indices(N, k=1)
    exact_vals = exact_mat[triu]
    beam_vals = beam_mat[triu]

    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(exact_vals, beam_vals)
    sr, _ = spearmanr(exact_vals, beam_vals)

    # Ranking accuracy: what % of pos/neg pairs agree
    exact_pos = exact_vals <= exact_thr
    beam_pos = beam_vals <= beam_thr
    agree = (exact_pos == beam_pos).mean()

    print(f"  Pearson r  = {pr:.4f}")
    print(f"  Spearman ρ = {sr:.4f}")
    print(f"  Ranking agreement = {agree:.4f} ({agree*100:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GED Oracle Validation — MUTAG\n(Beam Search B=5 vs Exact GED)", fontweight="bold")

    # Scatter
    subsample = np.random.RandomState(42).choice(len(exact_vals), size=min(3000, len(exact_vals)), replace=False)
    ax1.scatter(exact_vals[subsample], beam_vals[subsample], s=8, alpha=0.4, c="#2563eb")
    lims = [0, max(exact_vals.max(), beam_vals.max()) + 1]
    ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="y = x")
    ax1.set_xlabel("Exact GED")
    ax1.set_ylabel("Beam Search GED (B=5)")
    ax1.set_title(f"Scatter (Pearson r = {pr:.3f}, Spearman ρ = {sr:.3f})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Confusion matrix of pos/neg
    tp = ((exact_pos) & (beam_pos)).sum()
    tn = ((~exact_pos) & (~beam_pos)).sum()
    fp = ((~exact_pos) & (beam_pos)).sum()
    fn = ((exact_pos) & (~beam_pos)).sum()
    conf = np.array([[tp, fn], [fp, tn]])
    im = ax2.imshow(conf, cmap="Blues")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Pos (exact)", "Neg (exact)"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Pos (beam)", "Neg (beam)"])
    ax2.set_title(f"Ranking Agreement: {agree*100:.1f}%")
    for ii in range(2):
        for jj in range(2):
            ax2.text(jj, ii, f"{conf[ii,jj]}", ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if conf[ii,jj] > conf.max()/2 else "black")
    fig.colorbar(im, ax=ax2, shrink=0.8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/ged_oracle_validation.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved ged_oracle_validation.png")
    return {"pearson": pr, "spearman": sr, "agreement": agree,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


# ====================================================================
# 4. Pairwise Distance Distribution (MUTAG vs IMDB-Binary)
# ====================================================================
def gen_distance_distribution():
    print("\n=== Distance distributions ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Pairwise L2 Distance Distribution in Embedding Space", fontweight="bold")

    for ax, ds_name, color in [(axes[0], "mutag", "#2563eb"), (axes[1], "imdb-binary", "#dc2626")]:
        emb, _ = load_stuff(ds_name)
        N = emb.shape[0]
        # sample pairs
        rng = np.random.RandomState(42)
        n_sample = min(5000, N*(N-1)//2)
        idx_i = rng.randint(0, N, size=n_sample)
        idx_j = rng.randint(0, N, size=n_sample)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        dists = np.linalg.norm(emb[idx_i] - emb[idx_j], axis=1)

        ax.hist(dists, bins=60, color=color, alpha=0.75, edgecolor="white", linewidth=0.5, density=True)
        ax.axvline(np.median(dists), color="black", linestyle="--", linewidth=1.5,
                   label=f"Median = {np.median(dists):.3f}")
        ax.set_xlabel("L2 Distance")
        ax.set_ylabel("Density")
        ax.set_title(f"{PRETTY[ds_name]}\n(mean={np.mean(dists):.3f}, std={np.std(dists):.3f})")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/distance_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved distance_distribution.png")


# ====================================================================
# 5. LSH Candidate Set Sizes
# ====================================================================
def gen_candidate_sizes():
    print("\n=== LSH candidate set sizes ===")
    results = {}
    for ds_name in DATASETS:
        cfg = CONFIGS[ds_name]
        emb, dataset = load_stuff(ds_name)

        lsh = LSHIndex(dim=cfg["out_dim"], n_tables=cfg["n_tables"],
                       n_funcs=cfg["n_funcs"], w=cfg["w"])
        lsh.index(emb)

        cand_sizes = []
        for i in range(len(dataset)):
            z_q = emb[i]
            candidates = set()
            for a, b, table in lsh.tables:
                bucket = tuple(((a @ z_q + b) / lsh.w).astype(int))
                candidates.update(table.get(bucket, []))
            cand_sizes.append(len(candidates))

        results[ds_name] = {
            "mean": np.mean(cand_sizes),
            "median": np.median(cand_sizes),
            "min": int(np.min(cand_sizes)),
            "max": int(np.max(cand_sizes)),
            "std": np.std(cand_sizes),
            "corpus_size": len(dataset),
        }
        print(f"  {ds_name}: mean={np.mean(cand_sizes):.1f}, median={np.median(cand_sizes):.0f}, "
              f"min={np.min(cand_sizes)}, max={np.max(cand_sizes)}, N={len(dataset)}")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Average LSH Candidate Set Size vs Corpus Size", fontweight="bold")

    ds_labels = [PRETTY[ds] for ds in DATASETS]
    means = [results[ds]["mean"] for ds in DATASETS]
    corpus = [results[ds]["corpus_size"] for ds in DATASETS]

    x = np.arange(len(DATASETS))
    bars = ax.bar(x, means, color=COLORS5, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2 = ax.twinx()
    ax2.plot(x, corpus, "ko--", linewidth=2, markersize=7, label="Corpus Size")
    ax2.set_ylabel("Corpus Size (N)")
    ax2.legend(loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("Avg. Candidate Set |C|")
    ax.grid(axis="y", alpha=0.3)

    for bar, val, cs in zip(bars, means, corpus):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.0f}\n({val/cs*100:.1f}%)", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/candidate_set_sizes.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved candidate_set_sizes.png")
    return results


# ====================================================================
# 6. Scaling Experiment (Reddit-Binary subsamples)
# ====================================================================
def gen_scaling_experiment():
    print("\n=== Scaling experiment (Reddit-Binary subsamples) ===")
    emb_full, dataset_full = load_stuff("reddit-binary")
    cfg = CONFIGS["reddit-binary"]
    sizes = [200, 500, 1000, 1500, 2000]
    n_queries = 50
    k = 10

    rng = np.random.RandomState(42)
    results = {"sizes": sizes, "lsh_times": [], "bf_times": [],
               "cand_sizes": [], "speedup": []}

    for sz in sizes:
        idx = rng.choice(len(dataset_full), size=sz, replace=False)
        emb = emb_full[idx]

        # build LSH
        lsh = LSHIndex(dim=cfg["out_dim"], n_tables=cfg["n_tables"],
                       n_funcs=cfg["n_funcs"], w=cfg["w"])
        lsh.index(emb)

        query_idx = rng.choice(sz, size=n_queries, replace=False)
        lsh_ts, bf_ts, csizes = [], [], []

        for qi in query_idx:
            z_q = emb[qi]

            # LSH
            t0 = time.time()
            top_lsh = lsh.query(z_q, emb, k=k)
            lsh_ts.append((time.time() - t0) * 1000)

            # Brute-force
            t0 = time.time()
            dists = np.linalg.norm(emb - z_q, axis=1)
            _ = np.argsort(dists)[1:k+1]
            bf_ts.append((time.time() - t0) * 1000)

            # Candidate size
            candidates = set()
            for a, b, table in lsh.tables:
                bucket = tuple(((a @ z_q + b) / lsh.w).astype(int))
                candidates.update(table.get(bucket, []))
            csizes.append(len(candidates))

        results["lsh_times"].append(np.mean(lsh_ts))
        results["bf_times"].append(np.mean(bf_ts))
        results["cand_sizes"].append(np.mean(csizes))
        su = np.mean(bf_ts) / np.mean(lsh_ts) if np.mean(lsh_ts) > 0 else 0
        results["speedup"].append(su)
        print(f"  N={sz}: LSH={np.mean(lsh_ts):.3f}ms, BF={np.mean(bf_ts):.3f}ms, "
              f"|C|={np.mean(csizes):.0f}, Speedup={su:.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scalability Analysis — Reddit-Binary Subsamples", fontweight="bold")

    ax1.plot(sizes, results["lsh_times"], "o-", color="#2563eb", linewidth=2, markersize=7, label="LSH-ANN")
    ax1.plot(sizes, results["bf_times"], "s--", color="#dc2626", linewidth=2, markersize=7, label="Brute-Force")
    ax1.set_xlabel("Corpus Size (N)")
    ax1.set_ylabel("Avg Query Time (ms)")
    ax1.set_title("Query Time vs Corpus Size")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(sizes, results["cand_sizes"], "o-", color="#16a34a", linewidth=2, markersize=7)
    ax2r = ax2.twinx()
    ax2r.plot(sizes, [c/s*100 for c, s in zip(results["cand_sizes"], sizes)],
              "s--", color="#f59e0b", linewidth=2, markersize=7)
    ax2.set_xlabel("Corpus Size (N)")
    ax2.set_ylabel("Avg |C| (candidates)", color="#16a34a")
    ax2r.set_ylabel("|C|/N × 100 (%)", color="#f59e0b")
    ax2.set_title("Candidate Set Size vs Corpus Size")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/scaling_experiment.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved scaling_experiment.png")
    return results


# ====================================================================
# 7. Nearest Neighbor Purity
# ====================================================================
def gen_nn_purity():
    print("\n=== Nearest Neighbor Purity ===")
    results = {}
    for ds_name in DATASETS:
        emb, dataset = load_stuff(ds_name)
        labels = np.array([d.y.item() for d in dataset])
        purities = {k: [] for k in [5, 10, 20]}
        for i in range(len(dataset)):
            dists = np.linalg.norm(emb - emb[i], axis=1)
            for k in [5, 10, 20]:
                topk = np.argsort(dists)[1:k+1]
                purity = (labels[topk] == labels[i]).mean()
                purities[k].append(purity)
        results[ds_name] = {k: np.mean(v) for k, v in purities.items()}
        print(f"  {ds_name}: P@5={results[ds_name][5]:.4f}, "
              f"P@10={results[ds_name][10]:.4f}, P@20={results[ds_name][20]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Nearest Neighbor Class Purity (Trained GIN Embeddings)", fontweight="bold")

    x = np.arange(len(DATASETS))
    w = 0.25
    for i, k in enumerate([5, 10, 20]):
        vals = [results[ds][k] for ds in DATASETS]
        ax.bar(x + (i-1)*w, vals, w, label=f"k={k}", color=COLORS5[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[ds] for ds in DATASETS])
    ax.set_ylabel("Class Purity (fraction same class)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    fig.tight_layout()
    fig.savefig(f"{OUT}/nn_purity.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved nn_purity.png")
    return results


# ====================================================================
# 8. Concrete Retrieval Examples (MUTAG)
# ====================================================================
def gen_retrieval_examples():
    print("\n=== Retrieval examples (MUTAG) ===")
    emb, dataset = load_stuff("mutag")
    labels = np.array([d.y.item() for d in dataset])

    # pick 3 diverse queries
    queries = [0, 50, 100]
    examples = {}
    for qi in queries:
        dists = np.linalg.norm(emb - emb[qi], axis=1)
        top5 = np.argsort(dists)[1:6]
        examples[qi] = {
            "query": {"index": qi, "nodes": dataset[qi].num_nodes,
                      "edges": dataset[qi].num_edges // 2, "class": labels[qi]},
            "results": [{"index": int(j), "nodes": dataset[j].num_nodes,
                         "edges": dataset[j].num_edges // 2, "class": int(labels[j]),
                         "distance": float(dists[j])} for j in top5]
        }
        print(f"  Query {qi} (class={labels[qi]}, {dataset[qi].num_nodes}n):")
        for r in examples[qi]["results"]:
            print(f"    → Graph {r['index']:3d} | class={r['class']} | "
                  f"nodes={r['nodes']:2d} | dist={r['distance']:.4f}")

    # Visualize as a table figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    fig.suptitle("Concrete Retrieval Examples — MUTAG (Top-5 by L2 Distance)", fontweight="bold", y=1.02)

    for ax, qi in zip(axes, queries):
        ex = examples[qi]
        q = ex["query"]
        ax.axis("off")

        col_labels = ["Rank", "Graph ID", "Class", "Nodes", "Edges", "L2 Dist", "Same Class?"]
        cell_text = []
        colors_row = []
        for rank, r in enumerate(ex["results"], 1):
            same = "✓" if r["class"] == q["class"] else "✗"
            cell_text.append([str(rank), str(r["index"]), str(r["class"]),
                              str(r["nodes"]), str(r["edges"]),
                              f"{r['distance']:.4f}", same])
            colors_row.append("#e8f5e9" if r["class"] == q["class"] else "#ffebee")

        table = ax.table(cellText=cell_text, colLabels=col_labels,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#1565c0")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i, row_color in enumerate(colors_row):
            for j in range(len(col_labels)):
                table[i+1, j].set_facecolor(row_color)

        ax.set_title(f"Query: Graph {q['index']} (Class {q['class']}, {q['nodes']} nodes, {q['edges']} edges)",
                     fontsize=10, fontweight="bold", pad=2)

    fig.tight_layout()
    fig.savefig(f"{OUT}/retrieval_examples.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved retrieval_examples.png")
    return examples


# ====================================================================
# 9. Memory vs Recall Tradeoff
# ====================================================================
def gen_memory_recall_tradeoff():
    print("\n=== Memory vs Recall tradeoff ===")
    L_vals = [1, 2, 5, 10, 15, 20, 30]
    k = 10

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Memory vs Recall@10 Tradeoff (Varying L, Hash Tables)", fontweight="bold")

    for di, ds_name in enumerate(["mutag", "aids", "imdb-binary"]):
        cfg = CONFIGS[ds_name]
        emb, dataset = load_stuff(ds_name)
        eval_ged, eval_thr = load_ged(ds_name, "ged")
        gt = {}
        for i in range(len(dataset)):
            gt[i] = set(j for j in range(len(dataset)) if j != i and eval_ged[i, j] <= eval_thr)

        recalls, mems = [], []
        for L in L_vals:
            lsh = LSHIndex(dim=cfg["out_dim"], n_tables=L, n_funcs=4, w=cfg["w"])
            lsh.index(emb)
            mem = sum(sys.getsizeof(t) for (_, _, t) in lsh.tables) / 1024
            mems.append(mem)

            recs = []
            for i in range(min(200, len(dataset))):  # subsample for speed
                z_q = emb[i]
                top = lsh.query(z_q, emb, k=k)
                top = [x for x in top if x != i][:k]
                recs.append(len(set(top) & gt[i]) / max(len(gt[i]), 1))
            recalls.append(np.mean(recs))

        ax.plot(mems, recalls, "o-", color=COLORS5[di], linewidth=2, markersize=7,
                label=PRETTY[ds_name])
        # annotate L values
        for mi, (m, r, l_val) in enumerate(zip(mems, recalls, L_vals)):
            if mi % 2 == 0:
                ax.annotate(f"L={l_val}", (m, r), textcoords="offset points",
                            xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Hash Table Memory (KB)")
    ax.set_ylabel("Recall@10")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/memory_recall_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved memory_recall_tradeoff.png")


# ====================================================================
# 10. w × L Grid Search (MUTAG)
# ====================================================================
def gen_wl_grid():
    print("\n=== w × L grid search (MUTAG) ===")
    ds_name = "mutag"
    cfg = CONFIGS[ds_name]
    emb, dataset = load_stuff(ds_name)
    eval_ged, eval_thr = load_ged(ds_name, "ged")
    gt = {}
    for i in range(len(dataset)):
        gt[i] = set(j for j in range(len(dataset)) if j != i and eval_ged[i, j] <= eval_thr)

    W_VALS = [0.5, 1.0, 2.0, 4.0]
    L_VALS = [5, 10, 20]
    k = 10

    grid_prec = np.zeros((len(W_VALS), len(L_VALS)))
    grid_rec = np.zeros((len(W_VALS), len(L_VALS)))
    grid_aq = np.zeros((len(W_VALS), len(L_VALS)))

    # brute force top-k for AQ
    bf_tops = {}
    for i in range(len(dataset)):
        dists = np.linalg.norm(emb - emb[i], axis=1)
        bf_tops[i] = set(np.argsort(dists)[1:k+1])

    for wi, w in enumerate(W_VALS):
        for li, L in enumerate(L_VALS):
            lsh = LSHIndex(dim=cfg["out_dim"], n_tables=L, n_funcs=4, w=w)
            lsh.index(emb)

            precs, recs, aqs = [], [], []
            for i in range(len(dataset)):
                z_q = emb[i]
                top = lsh.query(z_q, emb, k=k)
                top = [x for x in top if x != i][:k]
                precs.append(len(set(top) & gt[i]) / k if top else 0)
                recs.append(len(set(top) & gt[i]) / max(len(gt[i]), 1))
                aqs.append(len(set(top) & bf_tops[i]) / k if top else 0)

            grid_prec[wi, li] = np.mean(precs)
            grid_rec[wi, li] = np.mean(recs)
            grid_aq[wi, li] = np.mean(aqs)
            print(f"  w={w}, L={L}: P@10={np.mean(precs):.4f}, R@10={np.mean(recs):.4f}, AQ={np.mean(aqs):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("w × L Grid Search — MUTAG (k=10)", fontweight="bold")

    for ax, grid, title in [(axes[0], grid_prec, "Precision@10"),
                             (axes[1], grid_rec, "Recall@10"),
                             (axes[2], grid_aq, "Approx Quality")]:
        im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(L_VALS)))
        ax.set_xticklabels([f"L={l}" for l in L_VALS])
        ax.set_yticks(range(len(W_VALS)))
        ax.set_yticklabels([f"w={w}" for w in W_VALS])
        ax.set_title(title)
        for ii in range(len(W_VALS)):
            for jj in range(len(L_VALS)):
                ax.text(jj, ii, f"{grid[ii,jj]:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if grid[ii,jj] > (grid.max()+grid.min())/2 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/wl_grid_search.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved wl_grid_search.png")


# ====================================================================
# 11. IMDB-Binary L ablation — does increasing L help?
# ====================================================================
def gen_imdb_l_ablation_aq():
    print("\n=== IMDB-Binary: AQ vs L ===")
    ds_name = "imdb-binary"
    cfg = CONFIGS[ds_name]
    emb, dataset = load_stuff(ds_name)

    L_vals = [5, 10, 20, 30, 40]
    k = 10

    # brute-force baseline
    bf_tops = {}
    for i in range(min(200, len(dataset))):
        dists = np.linalg.norm(emb - emb[i], axis=1)
        bf_tops[i] = set(np.argsort(dists)[1:k+1])

    aqs, lsh_times = [], []
    for L in L_vals:
        lsh = LSHIndex(dim=cfg["out_dim"], n_tables=L, n_funcs=4, w=cfg["w"])
        lsh.index(emb)
        aq_list, qt_list = [], []
        for i in range(min(200, len(dataset))):
            z_q = emb[i]
            t0 = time.time()
            top = lsh.query(z_q, emb, k=k)
            qt_list.append((time.time() - t0) * 1000)
            top = [x for x in top if x != i][:k]
            aq_list.append(len(set(top) & bf_tops[i]) / k if top else 0)
        aqs.append(np.mean(aq_list))
        lsh_times.append(np.mean(qt_list))
        print(f"  L={L}: AQ={np.mean(aq_list):.4f}, QT={np.mean(qt_list):.3f}ms")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle("IMDB-Binary: Does Increasing L Improve Approximation Quality?", fontweight="bold")

    ax1.plot(L_vals, aqs, "o-", color="#2563eb", linewidth=2, markersize=7, label="Approx Quality")
    ax1.set_xlabel("L (Number of Hash Tables)")
    ax1.set_ylabel("Approximation Quality", color="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(L_vals, lsh_times, "s--", color="#dc2626", linewidth=2, markersize=7, label="Query Time")
    ax2.set_ylabel("Query Time (ms)", color="#dc2626")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/imdb_l_ablation_aq.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved imdb_l_ablation_aq.png")


# ====================================================================
# RUN ALL
# ====================================================================
if __name__ == "__main__":
    train_data = gen_training_curves()
    gen_tsne_plots()
    oracle_data = gen_ged_oracle_validation()
    gen_distance_distribution()
    cand_data = gen_candidate_sizes()
    scale_data = gen_scaling_experiment()
    purity_data = gen_nn_purity()
    example_data = gen_retrieval_examples()
    gen_memory_recall_tradeoff()
    gen_wl_grid()
    gen_imdb_l_ablation_aq()

    # Save numeric results for the report
    report_data = {
        "oracle": oracle_data,
        "candidate_sizes": cand_data,
        "scaling": scale_data,
        "nn_purity": purity_data,
        "retrieval_examples": example_data,
    }
    with open(f"{OUT}/advanced_results.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\nAll advanced figures saved to {OUT}/")
