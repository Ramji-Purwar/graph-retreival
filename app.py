import os
import pickle
import time
import io
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template

from config import CONFIGS
from core.model import GINEncoder
from core.lsh import LSHIndex

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

# ── Helpers ──────────────────────────────────────────────────────────────────

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def load_dataset_artifacts(dataset_name):
    """Load model, embeddings and dataset for a given dataset name.
    Returns None if the outputs directory doesn't exist yet."""
    out_dir = f"outputs/{dataset_name}"
    if not os.path.isdir(out_dir):
        return None

    cfg = CONFIGS[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = np.load(f"{out_dir}/embeddings.npy")

    with open(f"{out_dir}/dataset.pkl", "rb") as f:
        dataset = CPU_Unpickler(f).load()

    model = GINEncoder(
        in_dim     = cfg["in_dim"],
        hidden_dim = cfg["hidden_dim"],
        out_dim    = cfg["out_dim"],
        num_layers = cfg["num_layers"],
    ).to(device)
    model.load_state_dict(torch.load(f"{out_dir}/model.pt", map_location=device))
    model.eval()

    lsh = LSHIndex(
        dim      = cfg["out_dim"],
        n_tables = cfg["n_tables"],
        n_funcs  = cfg["n_funcs"],
        w        = cfg["w"],
    )
    lsh.index(embeddings)

    print(f"[{dataset_name}] Loaded {len(dataset)} graphs | embeddings: {embeddings.shape}")
    return {"embeddings": embeddings, "dataset": dataset, "model": model, "lsh": lsh}


# ── Global state ──────────────────────────────────────────────────────────────

ALL_DATASETS = list(CONFIGS.keys())

# Discover which datasets have already been trained (outputs exist)
_cache = {}

def get_state(dataset_name):
    """Return cached state for a dataset, loading it on first access."""
    if dataset_name not in _cache:
        arts = load_dataset_artifacts(dataset_name)
        if arts is None:
            return None
        _cache[dataset_name] = arts
    return _cache[dataset_name]


# Determine the default dataset (first one that has outputs)
DEFAULT_DATASET = None
for ds in ALL_DATASETS:
    if os.path.isdir(f"outputs/{ds}"):
        DEFAULT_DATASET = ds
        break

if DEFAULT_DATASET is None:
    DEFAULT_DATASET = ALL_DATASETS[0]   # fallback (no outputs yet)

# Current active dataset (mutable via /switch)
current_dataset = DEFAULT_DATASET


# ── Utility functions ─────────────────────────────────────────────────────────

def graph_info(state, idx):
    g = state["dataset"][idx]
    edge_index = g.edge_index.cpu().numpy()
    seen = set()
    edges_list = []
    for col in range(edge_index.shape[1]):
        u, v = int(edge_index[0, col]), int(edge_index[1, col])
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            edges_list.append({"from": u, "to": v})
    return {
        "index":  int(idx),
        "nodes":  int(g.num_nodes),
        "edges":  int(g.num_edges),
        "label":  int(g.y.item()),
        "topology": {
            "nodes": list(range(g.num_nodes)),
            "edges": edges_list,
        },
    }


def brute_force_query(embeddings, z_q, k):
    dists = np.linalg.norm(embeddings - z_q, axis=1)
    top_k = np.argsort(dists)[1:k+1]
    return top_k.tolist()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    available = [ds for ds in ALL_DATASETS if os.path.isdir(f"outputs/{ds}")]
    state = get_state(current_dataset)
    total = len(state["dataset"]) if state else 0
    return render_template(
        "index.html",
        total=total,
        current_dataset=current_dataset,
        all_datasets=available,
    )


@app.route("/datasets")
def datasets():
    """Return list of available (trained) datasets."""
    available = [
        {
            "name": ds,
            "ready": os.path.isdir(f"outputs/{ds}"),
            "active": ds == current_dataset,
        }
        for ds in ALL_DATASETS
    ]
    return jsonify(available)


@app.route("/switch", methods=["POST"])
def switch():
    """Switch the active dataset."""
    global current_dataset
    data = request.json
    ds = data.get("dataset", "")
    if ds not in CONFIGS:
        return jsonify({"error": f"Unknown dataset: {ds}"}), 400
    if not os.path.isdir(f"outputs/{ds}"):
        return jsonify({"error": f"Dataset '{ds}' has not been trained yet. Run: python main.py --dataset {ds}"}), 400
    state = get_state(ds)
    if state is None:
        return jsonify({"error": f"Failed to load dataset '{ds}'"}), 500
    current_dataset = ds
    return jsonify({"dataset": ds, "total": len(state["dataset"])})


@app.route("/graph/<int:idx>")
def get_graph(idx):
    state = get_state(current_dataset)
    if state is None:
        return jsonify({"error": "No dataset loaded"}), 503
    if idx < 0 or idx >= len(state["dataset"]):
        return jsonify({"error": "Index out of range"}), 400
    return jsonify(graph_info(state, idx))


@app.route("/retrieve", methods=["POST"])
def retrieve():
    state = get_state(current_dataset)
    if state is None:
        return jsonify({"error": "No dataset loaded"}), 503

    data   = request.json
    idx    = int(data.get("index", 0))
    k      = int(data.get("k", 5))
    method = data.get("method", "lsh")

    embeddings = state["embeddings"]
    if idx < 0 or idx >= len(state["dataset"]):
        return jsonify({"error": "Index out of range"}), 400

    z_q = embeddings[idx]

    t0 = time.time()
    if method == "lsh":
        top_k = state["lsh"].query(z_q, embeddings, k=k)
        top_k = [x for x in top_k if x != idx][:k]
    else:
        top_k = brute_force_query(embeddings, z_q, k)
    elapsed = (time.time() - t0) * 1000  # ms

    results = [graph_info(state, i) for i in top_k]
    return jsonify({
        "query":   graph_info(state, idx),
        "results": results,
        "time_ms": round(elapsed, 4),
        "method":  method,
        "dataset": current_dataset,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)