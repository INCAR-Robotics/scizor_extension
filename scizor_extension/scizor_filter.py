"""
SCIZOR preprocessing step for INCAR.

Implements the full two-pass SCIZOR pipeline from:
  "SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning"
  Zhang et al., 2025 — https://arxiv.org/abs/2505.22626

Must be placed AFTER sample_dt and downsample_video so video frame indices
are aligned 1-to-1 with HDF5 signal indices.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import av
import numpy as np
import torch
from tqdm import tqdm

from incar.common import FeatureType, ProcessHook
from incar.extensions.processing_step import ProcessStep
from incar.extensions.utils import reduce_h5_indices, reduce_video_indices

if TYPE_CHECKING:
    from incar.config.dataset_config import DatasetConfig


# =========================================================================== #
# Helpers
# =========================================================================== #

def _load_video_frames(path: Path) -> List[np.ndarray]:
    container = av.open(str(path / "data.mp4"))
    frames = [fr.to_ndarray(format="rgb24") for fr in container.decode()]
    container.close()
    return frames


def _load_h5_array(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path / "data.h5", "r") as f:
        return f["data"][:]


def _apply_index_filter(
    root_path: str,
    demo: str,
    indices_to_keep: List[int],
    config: "DatasetConfig",
) -> None:
    for feat_name, feat_cfg in config.features.items():
        feat_path = Path(root_path) / demo / feat_name
        if not feat_path.exists():
            continue
        if feat_cfg.type == FeatureType.VISUAL:
            reduce_video_indices(feat_path, indices_to_keep, config.video_fps)
        else:
            reduce_h5_indices(feat_path, indices_to_keep)


def _print_summary(rows: list, total_before: int, total_after: int) -> None:
    pct = 100.0 * (1.0 - total_after / max(total_before, 1))
    print(f"\n[ScizorFull] {total_before} → {total_after} frames ({pct:.1f}% removed)")
    print(f"  {'Demo':<16} {'Before':>7} {'After':>7} {'Removed%':>9}")
    for demo, before, after in rows:
        r = 100.0 * (1.0 - after / max(before, 1))
        print(f"  {demo:<16} {before:>7} {after:>7} {r:>8.1f}%")


# =========================================================================== #
# ScizorFull — full two-pass SCIZOR
# =========================================================================== #

@ProcessStep.register_subclass("scizor_full")
@dataclass
class ScizorFull(ProcessStep):
    """
    Full SCIZOR curation step (DATASET hook only).

    Pass 1 — Suboptimal Transition Removal
        Trains a self-supervised task-progress predictor (frozen DINOv2 +
        self-attention transformer) on frame pairs from all demos.  Each frame
        is scored by how much its sub-trajectory lags behind expected progress.
        Frames in the top `suboptimality_percentile` % are removed.

    Pass 2 — Similarity-Based State-Action Deduplication
        Encodes 2-second non-overlapping chunks (video via Cosmos, actions
        concatenated) across all demos, clusters with k-means, and flags chunks
        whose maximum intra-cluster cosine similarity ≥ dedup_eps.

    A frame is removed if it is suboptimal OR belongs to a duplicate chunk.

    Paper: https://arxiv.org/abs/2505.22626
    """

    hooks: List[ProcessHook] = field(
        default_factory=lambda: [ProcessHook.DATASET]
    )

    # --- features ---
    video_feature:  str           = "wrist_cam"
    action_feature: Optional[str] = "right.commands.arm.ee.velocity"

    # --- encoders ---
    # Pass 1: DINOv2 per-frame CLS tokens for the progress predictor.
    #   "dinov2_vits14" (384-d, fast) | "dinov2_vitb14" (768-d, paper-exact)
    encoder: str = "dinov2_vits14"

    # Pass 2: Cosmos-Tokenize encodes the full 2-second clip as a temporal
    # sequence — the paper's method.  Defaults to None (reuses `encoder`).
    # Requires: pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git
    dedup_encoder: Optional[str] = None

    # --- Pass 1: progress predictor ---
    pairs_per_demo:           int   = 500
    num_steps:                int   = 10_000   # paper: 10 000
    batch_size:               int   = 128       # paper: 128
    lr:                       float = 1e-4      # paper: 1e-4
    lookahead_steps:          int   = 5
    gamma:                    float = 0.9       # temporal discount
    suboptimality_percentile: int   = 25        # remove top N% by suboptimality

    # --- Pass 2: deduplication ---
    dedup_eps:      float = 0.99   # paper: ε_d = 0.99
    n_clusters:     int   = 0      # 0 → auto: max(2, sqrt(n_chunks))
    pca_components: int   = 128    # PCA dim before k-means; 0 to disable

    # --- general ---
    min_demo_steps:  int = 20
    dino_batch_size: int = 64      # encoder inference batch size

    def __post_init__(self):
        self._enc       = None
        self._dedup_enc = None
        self._device    = None

    def _ensure_encoders(self):
        if self._enc is not None:
            return
        from .encoders import build_encoder
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ScizorFull] loading Pass 1 encoder '{self.encoder}' on {self._device}")
        self._enc = build_encoder(self.encoder, device=self._device)
        dedup_name = self.dedup_encoder or self.encoder
        if dedup_name != self.encoder:
            print(f"[ScizorFull] loading Pass 2 encoder '{dedup_name}' on {self._device}")
            self._dedup_enc = build_encoder(dedup_name, device=self._device)
        else:
            self._dedup_enc = self._enc

    # ------------------------------------------------------------------ #
    # Dataset hook
    # ------------------------------------------------------------------ #

    def process_dataset(self, root_path: str, config: "DatasetConfig"):
        self._ensure_encoders()
        self._validate_config(config)

        demos = sorted(
            [f.name for f in os.scandir(root_path) if f.is_dir()],
            key=lambda x: int(x.split("_")[-1]),
        )

        print(f"\n[ScizorFull] Loading video frames from {len(demos)} demos …")
        frames_per_demo: List[List[np.ndarray]] = []
        actions_per_demo: Optional[List[np.ndarray]] = [] if self.action_feature else None

        for demo in tqdm(demos, desc="  loading"):
            frames_per_demo.append(
                _load_video_frames(Path(root_path) / demo / self.video_feature)
            )
            if self.action_feature and actions_per_demo is not None:
                actions_per_demo.append(
                    _load_h5_array(Path(root_path) / demo / self.action_feature)
                )

        sub_scores   = self._pass1_suboptimality(frames_per_demo, config.data_timestep or 0.1)
        dedup_scores = self._pass2_deduplication(
            frames_per_demo, actions_per_demo,
            config.data_timestep or 0.1, len(demos), [len(f) for f in frames_per_demo],
        )

        print("\n[ScizorFull] Filtering …")
        all_sub = np.concatenate(sub_scores)
        sub_threshold = float(np.percentile(all_sub, 100 - self.suboptimality_percentile))
        print(f"  suboptimality threshold (p{100 - self.suboptimality_percentile}): "
              f"{sub_threshold:.4f}  →  removes top {self.suboptimality_percentile}%")
        print(f"  dedup eps: {self.dedup_eps}")

        total_before = total_after = 0
        rows = []

        for i, demo in enumerate(demos):
            N = len(frames_per_demo[i])
            total_before += N

            if N <= self.min_demo_steps:
                total_after += N
                rows.append((demo, N, N))
                continue

            remove = (sub_scores[i] > sub_threshold) | (dedup_scores[i] >= self.dedup_eps)
            keep   = ~remove
            keep[0] = keep[-1] = True

            if keep.sum() < self.min_demo_steps:
                order = np.argsort(sub_scores[i])
                keep  = np.zeros(N, dtype=bool)
                keep[order[: self.min_demo_steps]] = True
                keep[0] = keep[-1] = True

            indices = np.where(keep)[0].tolist()
            total_after += len(indices)
            rows.append((demo, N, len(indices)))

            if len(indices) < N:
                _apply_index_filter(root_path, demo, indices, config)

        _print_summary(rows, total_before, total_after)

    # ------------------------------------------------------------------ #
    # Pass 1 — suboptimal transition removal
    # ------------------------------------------------------------------ #

    def _pass1_suboptimality(
        self,
        frames_per_demo: List[List[np.ndarray]],
        dt: float,
    ) -> List[np.ndarray]:
        from .progress_predictor import train_progress_predictor, score_demo

        print(f"\n[ScizorFull] Pass 1 — extracting {self.encoder} features …")
        features_per_demo: List[np.ndarray] = []
        for frames in tqdm(frames_per_demo, desc="  encoding"):
            features_per_demo.append(
                self._enc.encode_frames(frames, batch_size=self.dino_batch_size)
            )

        print(f"[ScizorFull] Pass 1 — training progress predictor "
              f"({self.num_steps} steps, bs={self.batch_size}) …")
        model = train_progress_predictor(
            features_per_demo, dt=dt, device=self._device,
            pairs_per_demo=self.pairs_per_demo, num_steps=self.num_steps,
            batch_size=self.batch_size, lr=self.lr, feat_dim=self._enc.feat_dim,
        )

        print("[ScizorFull] Pass 1 — scoring frames …")
        return [
            score_demo(feats, model, self._device, dt,
                       lookahead_steps=self.lookahead_steps, gamma=self.gamma)
            for feats in tqdm(features_per_demo, desc="  scoring")
        ]

    # ------------------------------------------------------------------ #
    # Pass 2 — similarity-based deduplication
    # ------------------------------------------------------------------ #

    def _pass2_deduplication(
        self,
        frames_per_demo: List[List[np.ndarray]],
        actions_per_demo: Optional[List[np.ndarray]],
        dt: float,
        n_demos: int,
        n_frames: List[int],
    ) -> List[np.ndarray]:
        from .deduplicator import extract_chunk_embeddings, compute_dedup_scores, chunk_to_frame_scores

        print("\n[ScizorFull] Pass 2 — building chunk embeddings …")
        embeddings, chunk_index = extract_chunk_embeddings(
            self._dedup_enc, frames_per_demo, actions_per_demo,
            dt=dt, pca_components=self.pca_components,
        )

        if len(embeddings) == 0:
            return [np.zeros(n, dtype=np.float32) for n in n_frames]

        print(f"[ScizorFull] Pass 2 — clustering {len(embeddings)} chunks …")
        chunk_scores = compute_dedup_scores(embeddings, n_clusters=self.n_clusters)

        n_dup = int((chunk_scores >= self.dedup_eps).sum())
        print(f"  {n_dup}/{len(chunk_scores)} chunks flagged as duplicates (eps={self.dedup_eps})")

        return chunk_to_frame_scores(chunk_scores, chunk_index, n_demos, n_frames)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_config(self, config: "DatasetConfig"):
        if self.video_feature not in config.features:
            raise ValueError(f"[ScizorFull] '{self.video_feature}' not in dataset features")
        if config.features[self.video_feature].type != FeatureType.VISUAL:
            raise ValueError(f"[ScizorFull] '{self.video_feature}' must be a VISUAL feature")
        if self.action_feature and self.action_feature not in config.features:
            raise ValueError(
                f"[ScizorFull] action_feature '{self.action_feature}' not in dataset. "
                "Set action_feature to null to disable action conditioning."
            )
        if config.data_timestep is None:
            raise ValueError(
                "[ScizorFull] data_timestep not set — run sample_dt before scizor_full."
            )

    def process_single_frame(self, frame: dict): raise NotImplementedError
    def process_sequence(self, frames: dict):    raise NotImplementedError
