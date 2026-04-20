import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
import os
import io
import pickle
from torch_geometric.utils import to_networkx

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_dataset(ds_name):
    base_dir = f"outputs/{ds_name}"
    dataset_path = os.path.join(base_dir, "dataset.pkl")
    with open(dataset_path, "rb") as f:
        dataset = CPU_Unpickler(f).load()
    return dataset

def plot_graph(dataset, g_idx, save_path):
    data = dataset[g_idx]
    G = to_networkx(data, to_undirected=True)
    
    # We will use distinct colors if node features exist
    if hasattr(data, 'x') and data.x is not None:
        node_labels = data.x.argmax(dim=1).numpy()
        palette = sns.color_palette("muted", 10)
        colors = [palette[int(l) % 10] for l in node_labels]
    else:
        colors = ["#4C72B0"] * G.number_of_nodes()
        
    plt.figure(figsize=(8, 8))
    
    # Kamada-Kawai layout spreads nodes better to prevent clustering
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42, k=0.15)
    
    # Draw nodes and edges (Reduced node size, preserved colors exactly)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=250, edgecolors="white", linewidths=1.2)
    nx.draw_networkx_edges(G, pos, edge_color="#888888", width=1.5, alpha=0.7)
    
    # Determine what this graph is for context dynamically
    class_label = data.y.item() if hasattr(data, 'y') and data.y is not None else "Unknown"
    ds_name = save_path.split("_")[0].split("/")[-1].upper()
    plt.title(f"{ds_name} Graph {g_idx} (Class: {class_label})", fontsize=22, fontweight='bold', pad=15)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    os.makedirs("presentation_graphs", exist_ok=True)
    dataset = load_dataset("reddit-binary")
    
    print(f"Total graphs in dataset: {len(dataset)}")
    plot_graph(dataset, 2, "presentation_graphs/reddit-binary_query.png")
    plot_graph(dataset, 7, "presentation_graphs/reddit-binary_result.png")
    print("Graphs successfully plotted.")
