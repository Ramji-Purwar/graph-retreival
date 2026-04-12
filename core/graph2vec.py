import numpy as np
import networkx as nx

from karateclub.graph_embedding.graph2vec import Graph2Vec

FEATURELESS_DATASETS = {"imdb-binary", "reddit-binary"}


def pyg_to_networkx(data, use_degree_label: bool):
    G = nx.Graph()

    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))

    # Add edges from the PyG edge_index (shape [2, E])
    edge_index = data.edge_index.numpy()
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src < dst:                       # avoid duplicate undirected edges
            G.add_edge(int(src), int(dst))

    # Assign node labels
    if use_degree_label:
        for n in G.nodes():
            G.nodes[n]["label"] = str(G.degree(n))
    else:
        x = data.x.numpy()                  # (num_nodes, feat_dim)
        for n in G.nodes():
            G.nodes[n]["label"] = str(int(x[n].argmax()))

    return G


def graph2vec_embed(dataset, cfg, dataset_name: str) -> np.ndarray:
    out_dim = cfg["out_dim"]
    use_degree = dataset_name.lower() in FEATURELESS_DATASETS

    label_strategy = "node degree" if use_degree else "argmax(node features)"
    print(f"[graph2vec] Converting {len(dataset)} PyG graphs → NetworkX  "
          f"(labels: {label_strategy})")

    nx_graphs = [pyg_to_networkx(data, use_degree_label=use_degree)
                 for data in dataset]

    # Note: Reddit-Binary has ~2k graphs averaging 429 nodes each.
    # WL vocabulary construction can take several minutes on large datasets.
    print(f"[graph2vec] Fitting Graph2Vec  "
          f"(dimensions={out_dim}, wl_iterations=3, seed=42) ...")

    model = Graph2Vec(
        dimensions=out_dim,
        wl_iterations=3,       # match GIN's 3-layer depth
        seed=42,               # reproducibility
        workers=4,
        epochs=100,            # Increased epochs for better convergence, especially on small datasets
        min_count=1,           # CRITICAL: Prevent discarding all WL subtrees on small datasets like MUTAG!
        down_sampling=0.0001,
        learning_rate=0.025,
    )
    model.fit(nx_graphs)
    embeddings = model.get_embedding()       # (N, out_dim), float32

    # L2-normalise to match GINEncoder.forward() which applies
    # F.normalize(x, p=2, dim=-1).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)         # guard against zero vectors
    embeddings = embeddings / norms

    print(f"[graph2vec] Embeddings shape: {embeddings.shape}  "
          f"(L2-normalised, mean norm = {np.linalg.norm(embeddings, axis=1).mean():.4f})")

    return embeddings.astype(np.float32)
