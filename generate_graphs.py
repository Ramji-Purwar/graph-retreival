"""
Generate all figures for the project report.
Reads evaluation_results.json and lsh_ablation.json from outputs/
and saves plots to graph/
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
})

OUT = "graph"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["mutag", "proteins", "aids", "imdb-binary", "reddit-binary"]
PRETTY = {"mutag": "MUTAG", "proteins": "PROTEINS", "aids": "AIDS",
           "imdb-binary": "IMDB-Binary", "reddit-binary": "Reddit-Binary"}
COLORS = {"trained_lsh": "#2563eb", "trained_bf": "#16a34a", "untrained_lsh": "#dc2626"}
K_VALS = [5, 10, 20]


def load_eval(ds):
    with open(f"outputs/{ds}/evaluation_results.json") as f:
        return json.load(f)


def load_ablation(ds):
    with open(f"outputs/{ds}/lsh_ablation.json") as f:
        return json.load(f)


# ========================================================================
# Figure 1: Precision@k comparison across all datasets (Trained LSH vs BF vs Untrained)
# ========================================================================
def fig1_precision_comparison():
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    fig.suptitle("Precision@k — Trained LSH vs Brute-Force vs Untrained LSH", fontweight="bold", y=1.02)

    for ax, ds in zip(axes, DATASETS):
        data = load_eval(ds)
        trained = data["metrics"]["trained"]
        untrained = data["metrics"]["untrained"]

        for k in K_VALS:
            sk = str(k)
        
        # Extract values
        t_lsh = [trained[str(k)]["lsh"]["Precision@k"] for k in K_VALS]
        t_bf  = [trained[str(k)]["bf"]["Precision@k"] for k in K_VALS]
        u_lsh = [untrained[str(k)]["lsh"]["Precision@k"] for k in K_VALS]

        ax.plot(K_VALS, t_lsh, "o-", color=COLORS["trained_lsh"], label="Trained GIN + LSH", linewidth=2, markersize=6)
        ax.plot(K_VALS, t_bf, "s--", color=COLORS["trained_bf"], label="Trained GIN + BF", linewidth=2, markersize=6)
        ax.plot(K_VALS, u_lsh, "^:", color=COLORS["untrained_lsh"], label="Untrained GIN + LSH", linewidth=2, markersize=6)

        ax.set_title(PRETTY[ds])
        ax.set_xlabel("k")
        ax.set_xticks(K_VALS)

    axes[0].set_ylabel("Precision@k")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/precision_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved precision_comparison.png")


# ========================================================================
# Figure 2: MAP comparison across all datasets
# ========================================================================
def fig2_map_comparison():
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    fig.suptitle("Mean Average Precision (MAP) — Trained LSH vs Brute-Force vs Untrained LSH", fontweight="bold", y=1.02)

    for ax, ds in zip(axes, DATASETS):
        data = load_eval(ds)
        trained = data["metrics"]["trained"]
        untrained = data["metrics"]["untrained"]

        t_lsh = [trained[str(k)]["lsh"]["MAP"] for k in K_VALS]
        t_bf  = [trained[str(k)]["bf"]["MAP"] for k in K_VALS]
        u_lsh = [untrained[str(k)]["lsh"]["MAP"] for k in K_VALS]

        ax.plot(K_VALS, t_lsh, "o-", color=COLORS["trained_lsh"], label="Trained GIN + LSH", linewidth=2, markersize=6)
        ax.plot(K_VALS, t_bf, "s--", color=COLORS["trained_bf"], label="Trained GIN + BF", linewidth=2, markersize=6)
        ax.plot(K_VALS, u_lsh, "^:", color=COLORS["untrained_lsh"], label="Untrained GIN + LSH", linewidth=2, markersize=6)

        ax.set_title(PRETTY[ds])
        ax.set_xlabel("k")
        ax.set_xticks(K_VALS)

    axes[0].set_ylabel("MAP")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/map_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved map_comparison.png")


# ========================================================================
# Figure 3: Approximation Quality of LSH vs k
# ========================================================================
def fig3_approx_quality():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("LSH Approximation Quality vs k", fontweight="bold")

    markers = ["o", "s", "^", "D", "v"]
    colors_ds = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#8b5cf6"]

    for i, ds in enumerate(DATASETS):
        data = load_eval(ds)
        trained = data["metrics"]["trained"]
        aq = [trained[str(k)]["lsh"]["Approx Quality"] for k in K_VALS]
        ax.plot(K_VALS, aq, f"{markers[i]}-", color=colors_ds[i], label=PRETTY[ds], linewidth=2, markersize=7)

    ax.set_xlabel("k")
    ax.set_ylabel("Approximation Quality")
    ax.set_xticks(K_VALS)
    ax.set_ylim(0.3, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/approx_quality.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved approx_quality.png")


# ========================================================================
# Figure 4: Query Time — LSH vs Brute-Force (bar chart)
# ========================================================================
def fig4_query_time():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Average Query Time at k=10 — LSH vs Brute-Force", fontweight="bold")

    x = np.arange(len(DATASETS))
    w = 0.35

    lsh_times = []
    bf_times = []
    for ds in DATASETS:
        data = load_eval(ds)
        trained = data["metrics"]["trained"]
        lsh_times.append(trained["10"]["lsh"]["Query Time(ms)"])
        bf_times.append(trained["10"]["bf"]["Query Time(ms)"])

    bars1 = ax.bar(x - w/2, lsh_times, w, label="LSH-ANN", color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x + w/2, bf_times, w, label="Brute-Force", color="#16a34a", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[ds] for ds in DATASETS])
    ax.set_ylabel("Query Time (ms)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/query_time_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved query_time_comparison.png")


# ========================================================================
# Figure 5: LSH Ablation — L (number of tables) vs Precision/Recall/MAP for MUTAG
# ========================================================================
def fig5_lsh_ablation_L():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("LSH Ablation: Effect of Number of Hash Tables (L)\non MUTAG at k=10", fontweight="bold", y=1.05)

    # Do this for multiple datasets
    metrics_names = ["Precision@k", "Recall@k", "MAP"]
    L_GRID = [5, 10, 20]
    colors_ds = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#8b5cf6"]
    markers = ["o", "s", "^", "D", "v"]

    for mi, metric in enumerate(metrics_names):
        ax = axes[mi]
        for di, ds in enumerate(DATASETS):
            abl = load_ablation(ds)
            vals = []
            for L in L_GRID:
                vals.append(abl["L_ablation"][str(L)]["metrics"]["10"][metric])
            ax.plot(L_GRID, vals, f"{markers[di]}-", color=colors_ds[di], 
                    label=PRETTY[ds], linewidth=2, markersize=7)
        ax.set_title(metric)
        ax.set_xlabel("L (Number of Hash Tables)")
        ax.set_xticks(L_GRID)
        ax.grid(alpha=0.3)

    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/lsh_ablation_L.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved lsh_ablation_L.png")


# ========================================================================
# Figure 6: LSH w grid search results
# ========================================================================
def fig6_w_grid_search():
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    fig.suptitle("LSH Parameter Search: Recall@10 vs Bucket Width (w)", fontweight="bold", y=1.02)

    for ax, ds in zip(axes, DATASETS):
        abl = load_ablation(ds)
        ws = sorted([float(k) for k in abl["w_grid_search"].keys()])
        recalls = [abl["w_grid_search"][str(w) if '.' in str(w) else str(float(w))]["Recall@10"] for w in ws]

        # Try matching keys
        recall_vals = []
        for w in ws:
            key = str(w)
            if key not in abl["w_grid_search"]:
                key = str(int(w)) if w == int(w) else str(w)
            recall_vals.append(abl["w_grid_search"][key]["Recall@10"])

        ax.bar([str(w) for w in ws], recall_vals, color="#8b5cf6", alpha=0.85)
        best_w = abl["best_w"]
        ax.set_title(f"{PRETTY[ds]}\n(best w={best_w})")
        ax.set_xlabel("w")

    axes[0].set_ylabel("Recall@10")
    fig.tight_layout()
    fig.savefig(f"{OUT}/w_grid_search.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved w_grid_search.png")


# ========================================================================
# Figure 7: Index Construction Time & Memory
# ========================================================================
def fig7_index_stats():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LSH Index Construction Cost", fontweight="bold")

    build_times = []
    mem_kb = []
    for ds in DATASETS:
        data = load_eval(ds)
        idx = data["metrics"]["trained"]["5"]["index"]
        build_times.append(idx["construction_time_s"] * 1000)  # ms
        mem_kb.append(idx["table_mem_kb"])

    x = np.arange(len(DATASETS))
    labels = [PRETTY[ds] for ds in DATASETS]

    ax1.bar(x, build_times, color="#2563eb", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("Construction Time (ms)")
    ax1.set_title("Index Build Time")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, mem_kb, color="#16a34a", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Memory (KB)")
    ax2.set_title("Hash Table Memory")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/index_stats.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved index_stats.png")


# ========================================================================
# Figure 8: Trained vs Untrained Precision heatmap
# ========================================================================
def fig8_trained_vs_untrained_heatmap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Precision@k Heatmap — Trained vs Untrained GIN + LSH", fontweight="bold")

    # Build matrices
    trained_mat = np.zeros((len(DATASETS), len(K_VALS)))
    untrained_mat = np.zeros((len(DATASETS), len(K_VALS)))

    for i, ds in enumerate(DATASETS):
        data = load_eval(ds)
        for j, k in enumerate(K_VALS):
            trained_mat[i, j] = data["metrics"]["trained"][str(k)]["lsh"]["Precision@k"]
            untrained_mat[i, j] = data["metrics"]["untrained"][str(k)]["lsh"]["Precision@k"]

    for ax, mat, title in [(ax1, trained_mat, "Trained GIN + LSH"),
                            (ax2, untrained_mat, "Untrained GIN + LSH")]:
        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.6, vmax=1.0)
        ax.set_xticks(range(len(K_VALS)))
        ax.set_xticklabels([f"k={k}" for k in K_VALS])
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels([PRETTY[ds] for ds in DATASETS])
        ax.set_title(title)

        # Annotate cells
        for ii in range(len(DATASETS)):
            for jj in range(len(K_VALS)):
                ax.text(jj, ii, f"{mat[ii, jj]:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if mat[ii, jj] > 0.85 else "black")

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Precision@k")
    fig.tight_layout()
    fig.savefig(f"{OUT}/precision_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved precision_heatmap.png")


# ========================================================================
# Figure 9: Pipeline Architecture Diagram
# ========================================================================
def fig9_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.suptitle("End-to-End Graph Retrieval Pipeline", fontweight="bold", fontsize=14)

    boxes = [
        (1, 2, "Graph\nCorpus", "#e0f2fe"),
        (3.5, 2, "GIN\nEncoder", "#dbeafe"),
        (6, 2, "64-dim\nEmbeddings", "#bfdbfe"),
        (8.5, 2, "LSH\nIndex", "#93c5fd"),
        (11, 2, "Top-k\nRetrieval", "#60a5fa"),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x - 0.9, y - 0.7), 1.8, 1.4, 
                              facecolor=color, edgecolor="#1e40af",
                              linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=3)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.9
        x2 = boxes[i+1][0] - 0.9
        ax.annotate("", xy=(x2, 2), xytext=(x1, 2),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#1e40af"))

    # Training branch
    ax.annotate("Triplet Loss\n(GED Oracle)", xy=(3.5, 1.1), fontsize=9,
                ha="center", va="top", color="#b91c1c", fontweight="bold")
    ax.annotate("", xy=(3.5, 1.3), xytext=(3.5, 0.7),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#b91c1c", ls="--"))

    # Query branch
    ax.text(11, 0.8, "Query Graph →\nGIN Encode → LSH Hash\n→ Re-rank by L2", 
            ha="center", va="top", fontsize=8, color="#4338ca", style="italic")

    fig.tight_layout()
    fig.savefig(f"{OUT}/pipeline_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved pipeline_architecture.png")


# ========================================================================
# Run all
# ========================================================================
if __name__ == "__main__":
    fig1_precision_comparison()
    fig2_map_comparison()
    fig3_approx_quality()
    fig4_query_time()
    fig5_lsh_ablation_L()
    fig6_w_grid_search()
    fig7_index_stats()
    fig8_trained_vs_untrained_heatmap()
    fig9_pipeline()
    print(f"\nAll figures saved to {OUT}/")
