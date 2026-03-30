import os
import argparse
import torch
import numpy as np
import pickle

from config import CONFIGS
from core.dataset import load_jsonl
from core.model import GINEncoder
from core.train import train, embed_all
from core.lsh import LSHIndex


def main(dataset_name):
    # Load config
    cfg = CONFIGS[dataset_name]
    print(f"\nDataset : {dataset_name}")
    print(f"Config  : {cfg}\n")

    # Load dataset
    dataset = load_jsonl(cfg["path"], max_degree=cfg["max_degree"])
    print(f"Loaded {len(dataset)} graphs")

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}\n")

    model = GINEncoder(
        in_dim     = cfg["in_dim"],
        hidden_dim = cfg["hidden_dim"],
        out_dim    = cfg["out_dim"],
        num_layers = cfg["num_layers"],
    ).to(device)

    model = train(model, dataset, device, epochs=cfg["epochs"])

    # Embed all graphs
    embeddings = embed_all(model, dataset, device).numpy()
    print(f"\nEmbeddings shape: {embeddings.shape}")

    # Build LSH index
    lsh = LSHIndex(
        dim      = cfg["out_dim"],
        n_tables = cfg["n_tables"],
        n_funcs  = cfg["n_funcs"],
        w        = cfg["w"],
    )
    lsh.index(embeddings)
    print("LSH index built")

    # Save outputs
    out_dir = f"outputs/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    np.save(f"{out_dir}/embeddings.npy", embeddings)
    with open(f"{out_dir}/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved model, embeddings and dataset to {out_dir}/")

    # Example query
    k     = cfg["k"]
    z_q   = embeddings[0]
    top_k = lsh.query(z_q, embeddings, k=k)
    print(f"\nExample query (graph 0) → top-{k} similar graphs: {top_k}")
    for idx in top_k:
        g = dataset[idx]
        print(f"  Graph {idx:3d} | nodes: {g.num_nodes:3d} | label: {g.y.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mutag",
                        choices=["aids", "imdb-binary", "mutag", "proteins", "reddit-binary"],
                        help="Dataset to run the pipeline on")
    args = parser.parse_args()
    main(args.dataset)