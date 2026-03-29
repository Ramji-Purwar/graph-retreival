import torch
import torch.nn as nn
import random
from torch_geometric.loader import DataLoader
from core.model import GINEncoder
from core.dataset import load_jsonl


def sample_triplets(dataset, n=512):
    # Group indices by label
    pos_idx = [i for i, d in enumerate(dataset) if d.y.item() == 1]
    neg_idx = [i for i, d in enumerate(dataset) if d.y.item() == 0]

    triplets = []
    for _ in range(n):
        a, p = random.sample(pos_idx, 2)
        neg = random.choice(neg_idx)
        triplets.append((a, p, neg))

    return triplets


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

def train(model, dataset, device, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.TripletMarginLoss(margin=1.0)

    for epoch in range(epochs):
        model.train()
        triplets   = sample_triplets(dataset, n=512)
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
