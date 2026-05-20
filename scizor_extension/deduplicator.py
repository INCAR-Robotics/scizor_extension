"""
SCIZOR Pass 2 — Semantic Deduplication.

Closely follows the Meta SemDedup approach used in the original SCIZOR:
  - Extract joint visual+action chunk embeddings across all demos
  - Reduce dimensionality with PCA (optional, stabilises k-means)
  - Cluster with MiniBatchKMeans
  - Within each cluster: compute pairwise cosine similarities
  - Per-chunk score = max cosine similarity to any other cluster member
  - Flag as duplicate if score > eps  (paper default: 0.99)
  - Map chunk scores back to per-frame scores

Chunk parameters (paper §3.2):
  - Duration: 2 seconds  (chunk_seconds)
  - Subsampled to 8 frames for visual encoding  (n_subsample)
  - Actions concatenated to visual features to form joint embedding
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Chunk extraction
# --------------------------------------------------------------------------- #

CHUNK_SECONDS = 2.0     # paper: 2-second chunks
N_SUBSAMPLE   = 8       # paper: 8 uniformly subsampled frames per chunk
DEDUP_EPS     = 0.99    # paper: ε_d — cosine similarity threshold


def extract_chunk_embeddings(
    encoder,                              # VisualEncoder instance
    frames_per_demo: List[List],          # raw video frames per demo
    actions_per_demo: Optional[List[np.ndarray]],   # [N, action_dim] or None
    dt: float,
    pca_components: int = 128,            # 0 to skip PCA
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Build a chunk embedding matrix and an index mapping rows → demo/frames.

    Returns:
        embeddings:  float32 [C, feat_dim] (PCA-reduced if pca_components > 0)
        chunk_index: list of (demo_idx, start_frame, end_frame) — one per row
    """
    from sklearn.decomposition import IncrementalPCA

    chunk_frames = int(CHUNK_SECONDS / dt)
    step         = max(1, chunk_frames)        # non-overlapping — paper §3.3

    raw_chunks:  List[np.ndarray] = []
    chunk_index: List[Tuple[int, int, int]] = []

    for demo_idx, frames in enumerate(frames_per_demo):
        N    = len(frames)
        acts = actions_per_demo[demo_idx] if actions_per_demo is not None else None

        for start in range(0, max(1, N - chunk_frames + 1), step):
            end = min(start + chunk_frames, N)
            window_frames = frames[start:end]

            # visual embedding for this chunk
            vis = encoder.encode_chunks(
                [window_frames], n_subsample=N_SUBSAMPLE
            )[0]  # [feat_dim]

            if acts is not None:
                act_window = acts[start:end]
                if len(act_window) < chunk_frames:
                    pad = np.zeros((chunk_frames - len(act_window), act_window.shape[1]), dtype=act_window.dtype)
                    act_window = np.concatenate([act_window, pad], axis=0)
                chunk_vec = np.concatenate([vis, act_window.flatten()])
            else:
                chunk_vec = vis

            raw_chunks.append(chunk_vec.astype(np.float32))
            chunk_index.append((demo_idx, start, end))

    if not raw_chunks:
        return np.zeros((0, 1), dtype=np.float32), []

    chunk_matrix = np.stack(raw_chunks)   # [C, D]

    # PCA: reduces high-dim joint embeddings before clustering
    n_safe = min(pca_components, chunk_matrix.shape[0] - 1, chunk_matrix.shape[1])
    if pca_components > 0 and n_safe > 0 and chunk_matrix.shape[1] > n_safe:
        pca = IncrementalPCA(n_components=n_safe, batch_size=512)
        chunk_matrix = pca.fit_transform(chunk_matrix).astype(np.float32)

    return chunk_matrix, chunk_index


# --------------------------------------------------------------------------- #
# Deduplication scoring (SemDedup-style)
# --------------------------------------------------------------------------- #

def compute_dedup_scores(
    embeddings: np.ndarray,
    n_clusters: int = 0,    # 0 → auto: max(2, sqrt(C))
) -> np.ndarray:
    """
    Per-chunk cosine similarity score = max similarity to any cluster-mate.

    Follows Meta SemDedup:
      1. L2-normalise embeddings (cosine similarity = dot product)
      2. k-means cluster
      3. Within each cluster: pairwise similarities, per-item max (excl. self)

    Returns float32 [C] scores in [0, 1].
    """
    from sklearn.cluster import MiniBatchKMeans

    C = len(embeddings)
    if C == 0:
        return np.zeros(0, dtype=np.float32)

    if n_clusters <= 0:
        n_clusters = max(2, int(np.sqrt(C)))
    n_clusters = min(n_clusters, C)

    # L2 normalise for cosine similarity
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)

    km     = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    labels = km.fit_predict(normed)

    scores = np.zeros(C, dtype=np.float32)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        cluster = normed[idx]               # [K, D]
        # pairwise cosine similarities (upper triangle, excl. diagonal)
        sims = cluster @ cluster.T          # [K, K]
        np.fill_diagonal(sims, 0.0)
        scores[idx] = sims.max(axis=1)      # per-chunk: max similarity to any other

    return scores


# --------------------------------------------------------------------------- #
# Map chunk scores → per-frame scores
# --------------------------------------------------------------------------- #

def chunk_to_frame_scores(
    chunk_scores: np.ndarray,
    chunk_index:  List[Tuple[int, int, int]],
    n_demos:      int,
    frames_per_demo: List[int],
) -> List[np.ndarray]:
    """
    Each frame inherits the maximum dedup score of all chunks it belongs to.

    Returns a list of float32 arrays, one per demo.
    """
    frame_scores = [np.zeros(n, dtype=np.float32) for n in frames_per_demo]

    for score, (demo_idx, start, end) in zip(chunk_scores, chunk_index):
        frame_scores[demo_idx][start:end] = np.maximum(
            frame_scores[demo_idx][start:end], score
        )

    return frame_scores
