# Scalable Graph Retrieval Pipeline

This repository contains the implementation of a scalable graph retrieval system that combines Graph Neural Network (GNN) based graph embeddings with Locality Sensitive Hashing (LSH). It enables approximate nearest neighbor search over large graph corpora without exhaustive pairwise comparison, addressing the NP-hard nature of exact graph similarity measures like Graph Edit Distance (GED).

## Motivation & Goal

Finding the most structurally and semantically similar graphs from a large corpus for a given query graph is a critical problem in domains like molecular chemistry, social networks, and biological analysis. Naive pairwise comparisons using polynomial-time graph similarity measures are computationally intractable for large datasets. 

This project solves this by encoding graphs into a shared vector space and performing fast structural retrieval, using a pipeline consisting of graph embedding generation, contrastive training, LSH-based indexing, and approximate nearest neighbor retrieval.

## Architecture & Pipeline

### 1. Graph Embedding Generation (GIN)
We use the **Graph Isomorphism Network (GIN)** to encode graphs into fixed-dimensional dense vectors. GIN is provably as expressive as the Weisfeiler-Leman graph isomorphism test. For datasets without node features, we use a degree-based one-hot encoding mechanism to preserve structural inductive bias.

### 2. Contrastive Training (Triplet Loss)
The embedding network is trained end-to-end using triplet loss. The oracle used to generate triplets (anchor, positive, negative) is approximate Graph Edit Distance computed via beam search, ensuring that structurally similar graphs cluster together in the embedding space.

### 3. Locality Sensitive Hashing (LSH)
To scale retrieval to large corpora, we construct an LSH index using random projection. Query graphs are hashed to retrieve candidate graphs efficiently, which are then re-ranked by exact Euclidean distance for the final top-k results.

### 4. Evaluation Metrics
The pipeline is evaluated on multiple datasets (MUTAG, PROTEINS, IMDB-B, Reddit-Binary, AIDS) using:
- **Precision@k** & **Recall@k**
- **Mean Average Precision (MAP)**
- **Query Time** (speedup vs brute-force exact search)
- **Approximation Quality**

*Ground-truth for evaluation is computed using exact GED for small graphs (≤30 nodes) and a separate high-quality beam search for larger datasets.*

## Interactive Visualization Web App

While the core of this project is the retrieval pipeline, we also include a lightweight web application for visualizing graph topology and querying the index interactively.

### Setup & Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the visualization app locally:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to `http://127.0.0.1:5000/`.

## Acknowledgments
This project was proposed and guided by Prof. Anirban Dasgupta, Department of Computer Science and Engineering, IIT Gandhinagar. 
