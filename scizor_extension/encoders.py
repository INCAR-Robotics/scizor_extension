"""
Unified visual encoder interface for SCIZOR.

Supported encoders (set via `encoder` field in the training config):

  "dinov2_vits14"  — DINOv2 ViT-S/14, feat_dim=384.  Fast, no extra deps.
                     Loaded via torch.hub (weights cached locally on first run).

  "dinov2_vitb14"  — DINOv2 ViT-B/14, feat_dim=768.  Closer to the paper's
                     reported architecture (model_dim=768 in their config).

  "cosmos"         — NVIDIA Cosmos-Tokenize video encoder, feat_dim=varies.
                     Requires `pip install cosmos-tokenizer` and a GPU.
                     Used in the original SCIZOR deduplication pass.

All encoders expose the same interface:
    encoder = build_encoder("dinov2_vits14", device="cuda")
    features = encoder.encode_frames(frames)   # list[np.ndarray HWC] → [N, D]
    features = encoder.encode_chunks(chunks)   # list of frame-lists → [C, D]
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class VisualEncoder(ABC):
    """Common interface for all SCIZOR visual encoders."""

    @property
    @abstractmethod
    def feat_dim(self) -> int: ...

    @abstractmethod
    def encode_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode a list of HWC uint8 RGB frames → float32 [N, feat_dim]."""
        ...

    def encode_chunks(
        self,
        chunks: List[List[np.ndarray]],
        n_subsample: int = 8,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode video chunks for deduplication.

        Each chunk is a list of frames; `n_subsample` frames are uniformly
        selected and their features are mean-pooled into a single vector.
        Returns float32 [C, feat_dim].
        """
        chunk_feats = []
        for frames in chunks:
            idx = np.linspace(0, len(frames) - 1, min(n_subsample, len(frames)), dtype=int)
            sub = [frames[i] for i in idx]
            feats = self.encode_frames(sub, batch_size=batch_size)
            chunk_feats.append(feats.mean(axis=0))
        return np.stack(chunk_feats).astype(np.float32)


# --------------------------------------------------------------------------- #
# DINOv2
# --------------------------------------------------------------------------- #
_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _dino_preprocess(frame_rgb: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(frame_rgb).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _DINO_MEAN) / _DINO_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


class DINOv2Encoder(VisualEncoder):
    """
    Frozen DINOv2 encoder loaded via torch.hub.

    Args:
        variant: one of "dinov2_vits14" (384-d) or "dinov2_vitb14" (768-d).
        device:  "cuda" or "cpu".
    """

    _DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768}

    def __init__(self, variant: str = "dinov2_vits14", device: str = "cuda"):
        if variant not in self._DIMS:
            raise ValueError(
                f"Unknown DINOv2 variant '{variant}'. "
                f"Choose from: {list(self._DIMS)}"
            )
        self._variant = variant
        self._device  = device
        self._dim     = self._DIMS[variant]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = torch.hub.load(
                "facebookresearch/dinov2", variant,
                verbose=False,
            ).to(device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @property
    def feat_dim(self) -> int:
        return self._dim

    def encode_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 64,
    ) -> np.ndarray:
        tensors = [_dino_preprocess(f) for f in frames]
        out: List[np.ndarray] = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(self._device)
            with torch.no_grad():
                feats = self._model(batch)
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cosmos
# --------------------------------------------------------------------------- #
class CosmosEncoder(VisualEncoder):
    """
    NVIDIA Cosmos-Tokenize causal video encoder.

    Used in the original SCIZOR deduplication pass.  Encodes the full 2-second
    clip as a temporal sequence, capturing motion patterns that per-frame
    DINOv2 features cannot.

    Requires:  pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git

    The encoder JIT checkpoint is downloaded automatically from HuggingFace Hub
    on first use and cached in ~/.cache/huggingface/.

    Args:
        model_name: HuggingFace repo ID for the Cosmos tokenizer model.
                    Default: "nvidia/Cosmos-0.1-Tokenizer-CV4x8x8"
                      CV  = continuous video (latent, not discrete)
                      4x8x8 = 4× temporal, 8× spatial compression
        device:     "cuda" (required — Cosmos uses bfloat16)
    """

    _SPATIAL = 224   # resize target; must be divisible by 16 (spatial alignment)

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-0.1-Tokenizer-CV4x8x8",
        device: str = "cuda",
    ):
        try:
            from cosmos_tokenizer.video_lib import CausalVideoTokenizer
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "cosmos-tokenizer is not installed. "
                "Run: pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git\n"
                f"Original error: {e}"
            )
        self._device = device
        try:
            enc_path = hf_hub_download(model_name, "encoder.jit", local_files_only=True)
        except Exception:
            print(f"[CosmosEncoder] downloading encoder.jit from {model_name} …")
            enc_path = hf_hub_download(model_name, "encoder.jit")
        self._tokenizer = CausalVideoTokenizer(checkpoint_enc=enc_path, device=device)
        self._tokenizer.eval()

        # probe output dim with a fixed 8-frame dummy clip
        dummy = self._make_video_tensor(
            [np.zeros((self._SPATIAL, self._SPATIAL, 3), dtype=np.uint8)] * 8
        )
        with torch.no_grad():
            out = self._tokenizer.encode(dummy)
        self._dim = int(np.prod(out[0].shape[1:]))  # exclude batch dim

    @property
    def feat_dim(self) -> int:
        return self._dim

    def _make_video_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Convert a list of HWC uint8 RGB frames to a [1, 3, T, H, W] bfloat16
        tensor in range [-1..1], as expected by Cosmos.
        """
        tensors = []
        for f in frames:
            img = Image.fromarray(f).resize((self._SPATIAL, self._SPATIAL), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [0,255] → [-1,1]
            tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))  # [3, H, W]
        # [T, 3, H, W] → [1, 3, T, H, W]
        return torch.stack(tensors).permute(1, 0, 2, 3).unsqueeze(0).to(
            dtype=torch.bfloat16, device=self._device
        )

    def encode_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
    ) -> np.ndarray:
        # For Pass 1 (per-frame features), encode each frame as a 1-frame clip.
        # Note: Cosmos is primarily a clip encoder; for per-frame use prefer DINOv2.
        out: List[np.ndarray] = []
        for f in frames:
            video = self._make_video_tensor([f])
            with torch.no_grad():
                latent = self._tokenizer.encode(video)[0]
            out.append(latent.view(-1).float().cpu().numpy())
        return np.stack(out).astype(np.float32)

    def encode_chunks(
        self,
        chunks: List[List[np.ndarray]],
        n_subsample: int = 8,
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        Encode each chunk as a short video clip — the natural Cosmos input.
        Subsamples to n_subsample frames and encodes the full temporal window.
        """
        out: List[np.ndarray] = []
        for frames in chunks:
            idx = np.linspace(0, len(frames) - 1, min(n_subsample, len(frames)), dtype=int)
            sub_frames = [frames[i] for i in idx]
            # Pad to exactly n_subsample frames so the temporal latent dim is constant
            while len(sub_frames) < n_subsample:
                sub_frames.append(sub_frames[-1])
            video = self._make_video_tensor(sub_frames)
            with torch.no_grad():
                latent = self._tokenizer.encode(video)[0]
            out.append(latent.view(-1).float().cpu().numpy())
        return np.stack(out).astype(np.float32)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_encoder(name: str, device: str = "cuda") -> VisualEncoder:
    """
    Build a visual encoder by name.

    Args:
        name:   One of:
                  "dinov2_vits14"   DINOv2 ViT-S/14  (default, feat_dim=384)
                  "dinov2_vitb14"   DINOv2 ViT-B/14  (paper variant, feat_dim=768)
                  "cosmos"          NVIDIA Cosmos tokenizer
        device: "cuda" or "cpu"
    """
    if name.startswith("dinov2"):
        return DINOv2Encoder(variant=name, device=device)
    if name == "cosmos" or name.startswith("nvidia/Cosmos"):
        model_name = name if name.startswith("nvidia/") else "nvidia/Cosmos-0.1-Tokenizer-CV4x8x8"
        return CosmosEncoder(model_name=model_name, device=device)
    raise ValueError(
        f"Unknown encoder '{name}'. "
        "Supported: 'dinov2_vits14', 'dinov2_vitb14', 'cosmos', "
        "'nvidia/Cosmos-0.1-Tokenizer-CV4x8x8', etc."
    )
