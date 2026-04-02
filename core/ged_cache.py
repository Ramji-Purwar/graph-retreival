import os
import time
import numpy as np
from tqdm import tqdm

from core.ged import exact_ged, beam_search_ged

EXACT_GED_DATASETS  = {"mutag", "aids", "proteins"}
BEAM_GED_DATASETS   = {"imdb-binary", "reddit-binary"}

EVAL_BEAM_WIDTH = 20
TRAIN_BEAM_WIDTH = 5
MAX_NODES_FOR_GED = 60   # pairs where EITHER graph has more nodes fall back to label proxy


def _compute_ged(g1, g2, method: str, beam_width: int) -> float:
    # Size guard: skip expensive GED for large graphs and use label proxy instead
    if g1.num_nodes > MAX_NODES_FOR_GED or g2.num_nodes > MAX_NODES_FOR_GED:
        # Same class  → treat as structurally similar  (GED = 0)
        # Diff class  → treat as dissimilar             (GED = large)
        lbl1 = g1.y.item() if g1.y is not None else -1
        lbl2 = g2.y.item() if g2.y is not None else -2
        return 0.0 if lbl1 == lbl2 else float(MAX_NODES_FOR_GED * 4)

    if method == "exact":
        return exact_ged(g1, g2)
    else:
        return beam_search_ged(g1, g2, beam_width=beam_width)


def compute_ged_matrix(
    dataset,
    method: str,
    beam_width: int = EVAL_BEAM_WIDTH,
    verbose: bool = True,
) -> np.ndarray:

    N = len(dataset)
    ged_matrix = np.zeros((N, N), dtype=np.float32)

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    if verbose:
        skipped = sum(
            1 for i, j in pairs
            if dataset[i].num_nodes > MAX_NODES_FOR_GED or dataset[j].num_nodes > MAX_NODES_FOR_GED
        )
        print(f"Computing {method} GED for {N} graphs ({len(pairs):,} pairs) "
              f"[{skipped:,} pairs exceed {MAX_NODES_FOR_GED}-node limit → label proxy]...")
        t0 = time.time()

    iterator = tqdm(pairs, desc=f"GED ({method})", disable=not verbose)

    for i, j in iterator:
        g = _compute_ged(dataset[i], dataset[j], method, beam_width)
        ged_matrix[i, j] = g
        ged_matrix[j, i] = g

    if verbose:
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s")

    return ged_matrix


def save_ged_matrix(ged_matrix: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, ged_matrix)
    print(f"Saved GED matrix to {path}  shape={ged_matrix.shape}")


def load_ged_matrix(path: str) -> np.ndarray:
    ged_matrix = np.load(path)
    print(f"Loaded GED matrix from {path}  shape={ged_matrix.shape}")
    return ged_matrix


def compute_threshold(ged_matrix: np.ndarray, sample_size: int = 2000) -> float:
    N = ged_matrix.shape[0]
    
    triu_idx = np.triu_indices(N, k=1)
    values = ged_matrix[triu_idx]

    if len(values) > sample_size:
        rng = np.random.default_rng(42)
        values = rng.choice(values, size=sample_size, replace=False)

    threshold = float(np.median(values))
    print(f"GED threshold (median of {len(values)} pairs): {threshold:.2f}")
    return threshold


def get_ground_truth_ged(ged_matrix: np.ndarray, threshold: float) -> dict:
    N = ged_matrix.shape[0]
    gt = {}
    for i in range(N):
        relevant = set(
            j for j in range(N)
            if j != i and ged_matrix[i, j] <= threshold
        )
        gt[i] = relevant
    return gt


def build_or_load_ged_matrix(
    dataset,
    dataset_name: str,
    cfg: dict,
    force_recompute: bool = False,
    verbose: bool = True,
):
    out_dir = f"outputs/{dataset_name}"
    matrix_path = f"{out_dir}/ged_matrix.npy"
    threshold_path = f"{out_dir}/ged_threshold.npy"

    # Determine method
    name_lower = dataset_name.lower()
    if name_lower in EXACT_GED_DATASETS:
        method = "exact"
        beam_width = None
    else:
        method = "beam"
        beam_width = EVAL_BEAM_WIDTH

    # Load from cache if available
    if not force_recompute and os.path.exists(matrix_path) and os.path.exists(threshold_path):
        ged_matrix = load_ged_matrix(matrix_path)
        threshold = float(np.load(threshold_path))
        print(f"Loaded cached GED threshold: {threshold:.2f}")
        return ged_matrix, threshold

    # Compute
    ged_matrix = compute_ged_matrix(
        dataset,
        method=method,
        beam_width=beam_width if beam_width else EVAL_BEAM_WIDTH,
        verbose=verbose,
    )

    threshold = compute_threshold(ged_matrix)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    save_ged_matrix(ged_matrix, matrix_path)
    np.save(threshold_path, np.array(threshold))
    print(f"Saved GED threshold to {threshold_path}")

    return ged_matrix, threshold