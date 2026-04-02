import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
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
    use_pin = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=4,
        pin_memory=use_pin,
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=use_pin)
            z = model(batch.x, batch.edge_index, batch.batch)
            embeddings.append(z.cpu())
    return torch.cat(embeddings)

def _batch_encode(model, indices, dataset, device):
    """Encode a list of graph indices as a single batched GPU forward pass."""
    graphs = [dataset[i] for i in indices]
    batch  = Batch.from_data_list(graphs).to(device)
    return model(batch.x, batch.edge_index, batch.batch)   # (len(indices), out_dim)


def train(
    model,
    dataset,
    device,
    epochs: int = 50,
    dataset_name: str = None,
    use_ged: bool = True,
    triplet_batch_size: int = 64,   # how many triplets per GPU forward pass
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.TripletMarginLoss(margin=1.0)

    print(f"[train] Device : {device}")

    # Build / load training oracle
    ged_matrix, ged_threshold = None, None
    if use_ged and dataset_name is not None:
        try:
            ged_matrix, ged_threshold = build_or_load_train_ged_matrix(dataset, dataset_name)
        except Exception as e:
            print(f"[train] GED oracle unavailable ({e}); falling back to label-based triplets.")

    oracle = "GED (B=5)" if ged_matrix is not None else "class label"
    print(f"[train] Triplet oracle : {oracle}")
    print(f"[train] Triplet batch  : {triplet_batch_size}")

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        triplets = sample_triplets(
            dataset, n=512,
            ged_matrix=ged_matrix,
            threshold=ged_threshold,
        )
        total_loss = 0.0
        n_batches  = 0

        # Process triplets in mini-batches for efficient GPU utilisation
        for batch_start in range(0, len(triplets), triplet_batch_size):
            batch_triplets = triplets[batch_start : batch_start + triplet_batch_size]
            a_idx = [t[0] for t in batch_triplets]
            p_idx = [t[1] for t in batch_triplets]
            n_idx = [t[2] for t in batch_triplets]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type,
                                    enabled=(device.type == "cuda")):
                za = _batch_encode(model, a_idx, dataset, device)
                zp = _batch_encode(model, p_idx, dataset, device)
                zn = _batch_encode(model, n_idx, dataset, device)
                loss = loss_fn(za, zp, zn)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/n_batches:.4f}")

    return model
