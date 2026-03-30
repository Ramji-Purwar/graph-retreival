import json
from collections import defaultdict

# Load graph label
graph_label = {}
with open('REDDIT-BINARY.graph_labels') as f:
    for i, line in enumerate(f, start=1):
        label = int(line.strip())
        graph_label[i] = 0 if label == -1 else 1

# Load node -> graph mapping
node_to_graph = {}
with open('REDDIT-BINARY.graph_idx') as f:
    for node_id, line in enumerate(f, start=1):
        node_to_graph[node_id] = int(line.strip())

# Load edges
graph_edges = defaultdict(list)
with open('REDDIT-BINARY.edges') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) != 2:
            continue
        u, v = int(parts[0]), int(parts[1])
        g_id = node_to_graph[u]
        graph_edges[g_id].append((u, v))

# pre-graph node index mapping
graph_nodes = defaultdict(set)
for node_id, g_id in node_to_graph.items():
    graph_nodes[g_id].add(node_id)

count = 0
with open('full.jsonl', 'w') as out:
    for g_id in sorted(graph_edges.keys()):
        nodes = sorted(graph_nodes[g_id])
        node_map = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        edges = graph_edges[g_id]
        src = [node_map[u] for u, v in edges]
        dst = [node_map[v] for u, v in edges]

        rec = {
            'edge_index': [src, dst],
            'num_nodes': num_nodes,
            'y': [graph_label[g_id]],
        }
        out.write(json.dumps(rec) + '\n')
        count += 1

print("Done")
