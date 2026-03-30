import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

def load_jsonl(path, max_degree=30):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)

            edge_index = torch.tensor(rec['edge_index'], dtype=torch.long)
            y = torch.tensor(rec['y'], dtype=torch.long)
            num_nodes = rec['num_nodes']

            if 'node_feat' in rec:
                x = torch.tensor(rec['node_feat'], dtype=torch.float)
            else:
                # degree-based one hot encoding 
                deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
                deg = deg.clamp(max=max_degree)
                x = torch.zeros(num_nodes, max_degree+1)
                x[torch.arange(num_nodes), deg] = 1

            if 'edge_attr' in rec:
                edge_attr = torch.tensor(rec['edge_attr'], dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            else:
                data = Data(x=x, edge_index=edge_index, y=y)
            
            dataset.append(data)
    
    return dataset
    