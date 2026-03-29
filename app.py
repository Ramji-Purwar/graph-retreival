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

# Global state
DATASET_NAME = "mutag"
cfg          = CONFIGS[DATASET_NAME]
out_dir      = f"outputs/{DATASET_NAME}"

# Load saved outputs
embeddings = np.load(f"{out_dir}/embeddings.npy")
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

with open(f"{out_dir}/dataset.pkl", "rb") as f:
    dataset = CPU_Unpickler(f).load()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GINEncoder(
    in_dim     = cfg["in_dim"],
    hidden_dim = cfg["hidden_dim"],
    out_dim    = cfg["out_dim"],
    num_layers = cfg["num_layers"],
).to(device)
model.load_state_dict(torch.load(f"{out_dir}/model.pt", map_location=device))
model.eval()

# Build LSH index
lsh = LSHIndex(
    dim      = cfg["out_dim"],
    n_tables = cfg["n_tables"],
    n_funcs  = cfg["n_funcs"],
    w        = cfg["w"],
)
lsh.index(embeddings)

print(f"Loaded {len(dataset)} graphs | embeddings: {embeddings.shape}")


def graph_info(idx):
    g = dataset[idx]
    edge_index = g.edge_index.cpu().numpy()
    # Build unique undirected edges
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


def brute_force_query(z_q, k):
    dists = np.linalg.norm(embeddings - z_q, axis=1)
    top_k = np.argsort(dists)[1:k+1]
    return top_k.tolist()


# Routes
@app.route("/")
def index():
    return render_template("index.html", total=len(dataset))


@app.route("/graph/<int:idx>")
def get_graph(idx):
    if idx < 0 or idx >= len(dataset):
        return jsonify({"error": "Index out of range"}), 400
    return jsonify(graph_info(idx))


@app.route("/retrieve", methods=["POST"])
def retrieve():
    data   = request.json
    idx    = int(data.get("index", 0))
    k      = int(data.get("k", 5))
    method = data.get("method", "lsh")

    if idx < 0 or idx >= len(dataset):
        return jsonify({"error": "Index out of range"}), 400

    z_q = embeddings[idx]

    t0 = time.time()
    if method == "lsh":
        top_k = lsh.query(z_q, embeddings, k=k)
        top_k = [x for x in top_k if x != idx][:k]
    else:
        top_k = brute_force_query(z_q, k)
    elapsed = (time.time() - t0) * 1000  # ms

    results = [graph_info(i) for i in top_k]
    return jsonify({
        "query":   graph_info(idx),
        "results": results,
        "time_ms": round(elapsed, 4),
        "method":  method,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)