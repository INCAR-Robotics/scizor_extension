"""
SCIZOR Pass 1 — Task Progress Predictor.

Architecture (from paper / UT-Austin-RPL/SCIZOR):
  - Frozen DINOv2 encoder (features extracted upstream)
  - Input: two tokens — learnable CLS token and delta = feat_j - feat_i
  - num_layers self-attention transformer blocks  (paper: 6)
  - CLS output token → linear head → 5-class temporal bin logits

Training:
  - Self-supervised: frame pairs (i, j) from same demo
  - Label: which of 5 time bins (j-i)*dt falls into
  - Pairs balanced across bins to avoid easy-bin collapse
  - AdamW lr=1e-4 (paper), ExponentialLR scheduler

Scoring (SCIZOR §3.1):
  - V_{i,i+T} = T_actual - T_predicted   (sub-trajectory suboptimality)
  - Distributed as (1/T)*V to every frame in [i, i+T]
  - Temporally discounted: γ^(distance from chunk start)
  - Blended 50/50 with the per-demo mean score
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List

# --------------------------------------------------------------------------- #
# Time bins (seconds) — fixed by the paper
# --------------------------------------------------------------------------- #
BIN_EDGES   = [0.0, 0.5, 1.0, 2.0, 5.0, float("inf")]
BIN_MIDS    = np.array([0.25, 0.75, 1.5, 3.5, 7.0], dtype=np.float32)
N_BINS      = 5


def _time_to_bin(t: float) -> int:
    for i in range(N_BINS):
        if BIN_EDGES[i] <= t < BIN_EDGES[i + 1]:
            return i
    return N_BINS - 1


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ProgressHead(nn.Module):
    """
    Two-token transformer: [feat_i, delta] → 5-class bin logits.

    Faithful to the paper's 'multi-layer self-attention transformer blocks'
    on the concatenated [CLS token, delta-feature] sequence.

    Args:
        feat_dim:   DINOv2 feature dimension (384 for ViT-S, 768 for ViT-B).
        num_layers: Transformer depth.  Paper uses 6; 4 works well for small datasets.
        num_heads:  Attention heads.  Must divide feat_dim.
    """

    def __init__(
        self,
        feat_dim:  int = 384,
        num_layers: int = 4,
        num_heads:  int = 8,
    ):
        super().__init__()
        # Learnable CLS token — paper appendix A.2: "concatenated with a CLS token"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(feat_dim, N_BINS)

    def forward(self, feat_i: torch.Tensor, feat_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_i, feat_j: [B, D] DINOv2 CLS tokens
        Returns:
            logits: [B, N_BINS]
        """
        B = feat_i.shape[0]
        delta = (feat_j - feat_i).unsqueeze(1)              # [B, 1, D]
        cls   = self.cls_token.expand(B, -1, -1)            # [B, 1, D]
        x   = torch.cat([cls, delta], dim=1)                # [B, 2, D]
        out = self.transformer(x)                           # [B, 2, D]
        return self.head(out[:, 0])                         # CLS output → [B, N_BINS]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class _PairDataset(Dataset):
    """
    Balanced frame-pair dataset sampled from DINOv2 features of all demos.

    Pairs are balanced across the 5 time bins so the classifier doesn't
    trivially predict the most common bin.
    """

    def __init__(
        self,
        features_list: List[np.ndarray],
        dt: float,
        pairs_per_demo: int = 500,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        max_gap_frames = int(BIN_EDGES[-2] / dt)   # 5 s → frame count
        per_bin = max(1, pairs_per_demo // N_BINS)

        # (feat_i, feat_j, bin_label) — stored as numpy to save RAM
        fi_list, fj_list, label_list = [], [], []

        for feats in features_list:
            N = len(feats)
            if N < 2:
                continue
            by_bin: List[list] = [[] for _ in range(N_BINS)]

            # over-sample then prune to keep bins balanced
            for _ in range(pairs_per_demo * 4):
                i = int(rng.integers(0, N - 1))
                max_j = min(N, i + max_gap_frames + 1)
                if max_j <= i + 1:
                    continue
                j = int(rng.integers(i + 1, max_j))
                t = (j - i) * dt
                b = _time_to_bin(t)
                by_bin[b].append((i, j))

            for b, bucket in enumerate(by_bin):
                rng.shuffle(bucket)
                for i, j in bucket[:per_bin]:
                    fi_list.append(feats[i])
                    fj_list.append(feats[j])
                    label_list.append(b)

        self._fi     = np.stack(fi_list)
        self._fj     = np.stack(fj_list)
        self._labels = np.array(label_list, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._fi[idx]),
            torch.from_numpy(self._fj[idx]),
            int(self._labels[idx]),
        )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_progress_predictor(
    features_list: List[np.ndarray],
    dt: float,
    device: str,
    pairs_per_demo: int = 500,
    num_steps: int = 10_000,      # paper: 10 000
    batch_size: int = 128,        # paper: 128
    lr: float = 1e-4,             # paper: 1e-4
    lr_gamma: float = 0.999,      # ExponentialLR decay per step
    feat_dim: int = 384,
    num_layers: int = 4,
) -> ProgressHead:
    """
    Train and return a ProgressHead on pairs drawn from all demos.

    Follows the SCIZOR training procedure:
      AdamW + ExponentialLR, cross-entropy loss, balanced bins.
    """
    dataset = _PairDataset(features_list, dt, pairs_per_demo)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    model     = ProgressHead(feat_dim=feat_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    criterion = nn.CrossEntropyLoss()

    step = 0
    model.train()
    while step < num_steps:
        for fi, fj, labels in loader:
            if step >= num_steps:
                break
            fi, fj = fi.to(device), fj.to(device)
            labels  = labels.clone().detach().to(dtype=torch.long, device=device)

            logits = model(fi, fj)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 1000 == 0 or step == num_steps:
                acc = (logits.argmax(1) == labels).float().mean().item()
                print(f"    step {step:>6}/{num_steps}  "
                      f"loss={loss.item():.4f}  acc={acc*100:.1f}%  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_demo(
    features: np.ndarray,
    model: ProgressHead,
    device: str,
    dt: float,
    lookahead_steps: int = 5,
    gamma: float = 0.9,
) -> np.ndarray:
    """
    Compute per-frame suboptimality scores for one demo.

    Formula (SCIZOR §3.1):
      1. For each lookahead k in [1 .. lookahead_steps]:
           V_{i,i+k} = k*dt  −  T_predicted(feat_i, feat_{i+k})
           Each frame t in [i, i+k] accumulates  γ^(t-i) * V/k
      2. Blend with demo mean: score = 0.5 * local + 0.5 * mean(local)

    High score → suboptimal/idle frame.
    Low score  → productive frame (predicted progress ≈ actual elapsed time).
    """
    N = len(features)
    raw = np.zeros(N, dtype=np.float64)

    fi_t = torch.from_numpy(features).to(device)   # [N, D]

    model.eval()
    with torch.no_grad():
        for k in range(1, lookahead_steps + 1):
            if k >= N:
                break
            i_idx = np.arange(0, N - k)
            j_idx = i_idx + k
            T_actual = k * dt

            logits = model(fi_t[i_idx], fi_t[j_idx])        # [M, 5]
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            T_pred = (probs * BIN_MIDS).sum(axis=1)          # expected seconds

            V = T_actual - T_pred                            # [M]

            for step in range(k + 1):
                discount = gamma ** step
                raw[i_idx + step] += discount * (V / k)

    # 50/50 blend with demo mean
    scores = 0.5 * raw + 0.5 * raw.mean()
    return scores.astype(np.float32)
