from torch_geometric.nn import GINConv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F

class GINEncoder(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=64, out_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in_dim -> hidden_dim
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=-1)

