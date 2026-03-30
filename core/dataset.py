import json
import torch
from torch_geometric.data import Data

def load_jsonl(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)

            x = torch.tensor(rec['node_feat'], dtype=torch.float)
            edge_index = torch.tensor(rec['edge_index'], dtype=torch.long)
            y = torch.tensor(rec['y'], dtype=torch.long)

            if 'edge_attr' in rec:
                edge_attr = torch.tensor(rec['edge_attr'], dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            else:
                data = Data(x=x, edge_index=edge_index, y=y)
            
            dataset.append(data)
    
    return dataset
    