import pandas as pd
import json
import os

df = pd.read_parquet('full.parquet')

def convert(o):
    if hasattr(o, 'tolist'):
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

with open('full.jsonl', 'w') as f:
    for _, row in df.iterrows():
        rec = {
            'edge_index': row['edge_index'],
            'node_feat': row['node_feat'],
            'edge_attr': row['edge_attr'],
            'y': row['y'],
            'num_nodes': int(row['num_nodes'])
        }
        f.write(json.dumps(rec, default=convert) + '\n')

print('Done')