import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch_geometric.loader import DataLoader
from core.model import GINEncoder
from core.dataset import load_jsonl



def sample_triplets(dataset, n=512, ged_matrix=None, threshold=None):
    """
    Build (anchor, positive, negative) index triples.

    If a GED matrix + threshold are supplied (proposal §5.1, training oracle B=5):
      positive  = graph with GED(anchor, positive) <= threshold
      negative  = graph with GED(anchor, negative) >  threshold

    Falls back to label-based sampling if no matrix is provided
    (keeps backward-compatibility with pre-GED checkpoints).
    """
    N = len(dataset)

    if ged_matrix is not None and threshold is not None:
        triplets = []
        attempts = 0
        max_attempts = n * 20  # guard against datasets where positives are scarce

        while len(triplets) < n and attempts < max_attempts:
            a = random.randrange(N)
            dists = ged_matrix[a]  # shape (N,)

            pos_pool = [j for j in range(N) if j != a and dists[j] <= threshold]
            neg_pool = [j for j in range(N) if dists[j] >  threshold]

            if pos_pool and neg_pool:
                p   = random.choice(pos_pool)
                neg = random.choice(neg_pool)
                triplets.append((a, p, neg))

            attempts += 1

        if len(triplets) < n:
            print(f"[train] Warning: only {len(triplets)}/{n} GED triplets could be sampled.")

        return triplets

    # fallback: group by class label
    pos_idx = [i for i, d in enumerate(dataset) if d.y.item() == 1]
    neg_idx = [i for i, d in enumerate(dataset) if d.y.item() == 0]

    triplets = []
    for _ in range(n):
        a, p = random.sample(pos_idx, 2)
        neg  = random.choice(neg_idx)
        triplets.append((a, p, neg))

    return triplets


def build_or_load_train_ged_matrix(dataset, dataset_name: str):
    """
    Compute (or load from cache) the TRAINING GED matrix using beam search B=5.
    Per proposal §5.1: training oracle is ALWAYS beam search (B=5),
    regardless of dataset. Exact GED is reserved for evaluation only.
    Cache stored separately: train_ged_matrix.npy (never mixed with eval matrix).
    """
    from core.ged_cache import (
        TRAIN_BEAM_WIDTH,
        compute_ged_matrix, compute_threshold,
        load_ged_matrix, save_ged_matrix,
    )

    out_dir       = f"outputs/{dataset_name}"
    matrix_path   = f"{out_dir}/train_ged_matrix.npy"
    thresh_path   = f"{out_dir}/train_ged_threshold.npy"

    if os.path.exists(matrix_path) and os.path.exists(thresh_path):
        ged_matrix = load_ged_matrix(matrix_path)
        threshold  = float(np.load(thresh_path))
        print(f"[train] Loaded cached training GED matrix  threshold={threshold:.2f}")
        return ged_matrix, threshold

    # Training oracle: beam search B=5, always (proposal §5.1)
    method     = "beam"
    beam_width = TRAIN_BEAM_WIDTH   # B = 5

    print(f"[train] Computing training GED matrix  method={method}  beam_width={beam_width}")
    ged_matrix = compute_ged_matrix(dataset, method=method, beam_width=beam_width)
    threshold  = compute_threshold(ged_matrix)

    os.makedirs(out_dir, exist_ok=True)
    save_ged_matrix(ged_matrix, matrix_path)
    np.save(thresh_path, np.array(threshold))
    print(f"[train] Saved training GED matrix and threshold to {out_dir}/")

    return ged_matrix, threshold




def embed_all(model, dataset, device):
    loader = DataLoader(dataset, batch_size=32)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model(batch.x, batch.edge_index, batch.batch)
            embeddings.append(z.cpu())
    return torch.cat(embeddings)

def train(model, dataset, device, epochs=50, dataset_name: str = None, use_ged: bool = True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.TripletMarginLoss(margin=1.0)

    # Build / load training oracle
    ged_matrix, ged_threshold = None, None
    if use_ged and dataset_name is not None:
        try:
            ged_matrix, ged_threshold = build_or_load_train_ged_matrix(dataset, dataset_name)
        except Exception as e:
            print(f"[train] GED oracle unavailable ({e}); falling back to label-based triplets.")

    oracle = "GED (B=5)" if ged_matrix is not None else "class label"
    print(f"[train] Triplet oracle : {oracle}")

    for epoch in range(epochs):
        model.train()
        triplets   = sample_triplets(dataset, n=512,
                                     ged_matrix=ged_matrix,
                                     threshold=ged_threshold)
        total_loss = 0

        for a_i, p_i, n_i in triplets:
            a_data = dataset[a_i].to(device)
            p_data = dataset[p_i].to(device)
            n_data = dataset[n_i].to(device)

            ba = torch.zeros(a_data.num_nodes, dtype=torch.long, device=device)
            bp = torch.zeros(p_data.num_nodes, dtype=torch.long, device=device)
            bn = torch.zeros(n_data.num_nodes, dtype=torch.long, device=device)

            za = model(a_data.x, a_data.edge_index, ba)
            zp = model(p_data.x, p_data.edge_index, bp)
            zn = model(n_data.x, n_data.edge_index, bn)

            loss = loss_fn(za, zp, zn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(triplets):.4f}")

    return model
