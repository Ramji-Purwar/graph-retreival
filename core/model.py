from torch_geometric.nn import GINConv, global_add_pool
import torch.nn as nn

class GINEncoder(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=64, out_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()

        # First layyer: in_dim -> hidden_dim
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )))

        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )))

        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.proj(x)

