# scizor_extension

INCAR preprocessing extension implementing **SCIZOR** — a self-supervised two-pass dataset curation method for imitation learning that removes suboptimal and redundant frames before policy training.

> **Paper:** [SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning](https://arxiv.org/abs/2505.22626)  
> **Original code:** [UT-Austin-RPL/SCIZOR](https://github.com/UT-Austin-RPL/SCIZOR)

---

## Quick start

Add `scizor_full` to your preprocessing pipeline (after `sample_dt` and `downsample_video`):

```json
{
    "type": "scizor_full",
    "video_feature": "wrist_cam",
    "action_feature": "right.commands.arm.ee.velocity",
    "encoder": "dinov2_vits14",
    "dedup_encoder": "cosmos",
    "num_steps": 10000,
    "pairs_per_demo": 500,
    "batch_size": 128,
    "lr": 1e-4,
    "lookahead_steps": 5,
    "gamma": 0.9,
    "suboptimality_percentile": 25,
    "dedup_eps": 0.99,
    "pca_components": 128,
    "min_demo_steps": 20
}
```

---

## Installation

```bash
source ~/incar_env/bin/activate
pip install -e /path/to/scizor_extension
```

---

## Background

Teleoperated robot demonstrations contain two types of low-quality frames that hurt policy learning:

- **Suboptimal / idle frames** — operator hesitating, repositioning, or holding position without making task progress.
- **Redundant frames** — structurally near-identical segments that appear across many demos (e.g., every demo starts with the same reach-to-object motion).

SCIZOR detects both types without any manual annotation.

---

## How It Works

### Pass 1 — Task Progress Predictor (suboptimality scoring)

A lightweight self-attention transformer is trained on frame pairs `(frame_i, frame_j)` from all demos to classify how much time elapsed between them (5 fixed bins: 0–0.5 s, 0.5–1 s, 1–2 s, 2–5 s, 5+ s).

Training is fully **self-supervised**: the ground-truth label is the actual elapsed time.

At inference, each frame is scored:

```
V_{i,i+T} = T_actual − T_predicted      # sub-trajectory score
score(frame) = 0.5 × local_score        # temporally discounted sum of V
             + 0.5 × demo_mean_score    # blended with demo average
```

**High score = idle/suboptimal.  Low score = productive frame.**

Frames in the top `suboptimality_percentile` % are marked for removal.

### Pass 2 — Deduplication (SemDedup-style)

Non-overlapping 2-second chunks are extracted from all demos. Each chunk is encoded as a joint visual+action embedding (8 subsampled frames through DINOv2 or Cosmos, concatenated with the action sequence). Chunks are:

1. Reduced with PCA
2. Clustered with MiniBatchKMeans across all demos
3. Scored by max cosine similarity to any other chunk in the same cluster

Frames belonging to chunks with similarity ≥ `dedup_eps` are flagged as duplicates.

### Combined filter

A frame is **removed** if:
```
suboptimality_score  >  threshold(suboptimality_percentile)
         OR
dedup_score          ≥  dedup_eps
```

Both steps operate at the INCAR `DATASET` hook and modify the preprocessed dataset files in-place (video + HDF5 signals), keeping frame indices aligned across all features.

---

## Configuration reference

#### Encoder options

| Parameter | Pass | Paper value | Notes |
|---|---|---|---|
| `encoder` | Pass 1 — progress predictor | `dinov2_vitb14` | Per-frame CLS tokens fed to the transformer head. `dinov2_vits14` (384-d) is faster and works well in practice. |
| `dedup_encoder` | Pass 2 — deduplication | `"cosmos"` | Encodes the full 2-second clip as a temporal sequence. Defaults to `null` (reuses `encoder`). |

**`encoder` values:**

| Value | `feat_dim` | Notes |
|---|---|---|
| `"dinov2_vits14"` | 384 | Default. Fast, cached via `torch.hub`. |
| `"dinov2_vitb14"` | 768 | Matches paper's `model_dim=768`. More memory. |

**`dedup_encoder` values (Pass 2 only):**

| Value | Notes |
|---|---|
| `null` | Reuses `encoder` (DINOv2 mean-pooled). Simpler, no extra install. |
| `"cosmos"` | NVIDIA Cosmos-Tokenize `CV4x8x8`. Encodes the 2-second window as a full video clip — the paper's method. |

To match the paper exactly:
```json
"encoder":       "dinov2_vitb14",
"dedup_encoder": "cosmos"
```

#### Key hyperparameters

| Parameter | Default | Paper value | Notes |
|---|---|---|---|
| `num_steps` | 10 000 | 10 000 | Training steps for Pass 1. |
| `batch_size` | 128 | 128 | Training batch size. |
| `lr` | 1e-4 | 1e-4 | AdamW learning rate. |
| `suboptimality_percentile` | 25 | — | Remove top N% by suboptimality. |
| `dedup_eps` | 0.99 | 0.99 | Cosine similarity duplicate threshold. |
| `pairs_per_demo` | 500 | — | Frame pairs sampled per demo for training. |
| `lookahead_steps` | 5 | — | Max frame gap for scoring (in frames at dataset dt). |
| `gamma` | 0.9 | — | Temporal discount for score accumulation. |
| `pca_components` | 128 | — | PCA dim before k-means; 0 to disable. |
| `n_clusters` | 0 | — | k-means clusters; 0 = auto (`max(2, sqrt(n_chunks))`). |
| `min_demo_steps` | 20 | — | Demos shorter than this are left untouched. |

---

## Placement in the preprocessing pipeline

Place **after** `sample_dt` and `downsample_video` so video frame indices are aligned 1-to-1 with HDF5 signal indices:

```json
"steps": [
    { "type": "sample_dt", "dt": 0.1 },
    { "type": "filter_takeover", "..." },
    { "type": "downsample_video", "features": ["wrist_cam"], "new_size": [240, 320] },
    {
        "type": "scizor_full",
        "video_feature": "wrist_cam",
        "action_feature": "right.commands.arm.ee.velocity",
        "encoder": "dinov2_vits14",
        "dedup_encoder": "cosmos",
        "suboptimality_percentile": 25,
        "dedup_eps": 0.99
    },
    { "type": "filter_by_buttons", "..." },
    { "type": "image_transform", "..." }
]
```

---

## Repository structure

```
scizor_extension/
├── pyproject.toml
├── README.md
└── scizor_extension/
    ├── __init__.py
    ├── encoders.py              # Unified DINOv2 / Cosmos encoder interface
    ├── progress_predictor.py    # Pass 1: transformer head + training + scoring
    ├── deduplicator.py          # Pass 2: chunk features + k-means + similarity
    └── scizor_filter.py         # INCAR ProcessStep: ScizorFull
```

---

## Citation

```bibtex
@article{scizor2025,
  title   = {SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning},
  author  = {Zhang et al.},
  journal = {arXiv preprint arXiv:2505.22626},
  year    = {2025}
}
```
