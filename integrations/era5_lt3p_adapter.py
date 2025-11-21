"""
Utilities to feed the local ERA5 typhoon dataset into the LT3P architecture.

This module builds PyTorch datasets/dataloaders directly from the
`typhoon_data_2018_2021_full` NPZ cases and normalizes the track coordinates
exactly like the original LT3P inference script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

Split = Literal["train", "val", "test"]


# LT3P’s plotting script maps normalized coordinates back to lat/lon via:
#   lon = norm_lon * 15.566 + 135.051
#   lat = norm_lat *  8.202 +  19.441
# Reuse the same constants so that our adaptation stays consistent.
LON_BASE = 135.051
LON_SCALE = 15.566
LAT_BASE = 19.441
LAT_SCALE = 8.202


def normalize_track(track: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) in degrees to LT3P-normalized coordinates."""
    lat = (track[..., 0] - LAT_BASE) / LAT_SCALE
    lon = (track[..., 1] - LON_BASE) / LON_SCALE
    return np.stack([lat, lon], axis=-1)


def denormalize_track(track_norm: np.ndarray) -> np.ndarray:
    """Convert normalized coordinates back to (lat, lon) degrees."""
    lat = track_norm[..., 0] * LAT_SCALE + LAT_BASE
    lon = track_norm[..., 1] * LON_SCALE + LON_BASE
    return np.stack([lat, lon], axis=-1)


@dataclass
class ERA5Case:
    frames: np.ndarray
    past_frames: np.ndarray
    future_frames: np.ndarray
    past_track: np.ndarray
    future_track: np.ndarray
    metadata: Dict[str, np.ndarray]

    @property
    def normalized_track(self) -> np.ndarray:
        full_track = np.concatenate([self.past_track, self.future_track], axis=0)
        return normalize_track(full_track)


class ERA5TyphoonDataset(Dataset):
    """
    Thin wrapper around the processed NPZ cases.

    Each item returns:
        {
            "frames": torch.float32 [T, C, H, W] where T=past+future,
            "past_frames": torch.float32 [T_past, C, H, W],
            "future_frames": torch.float32 [T_future, C, H, W],
            "track_norm": torch.float32 [T, 2],
            "meta": dict(...)
        }
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        past_len: int = 8,
        future_len: int = 12,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.past_len = past_len
        self.future_len = future_len
        self.dtype = dtype

        split_dir = self.root / split / "cases"
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.files = sorted(split_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No NPZ cases found under {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_case(self, idx: int) -> ERA5Case:
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as npz:
            past_frames = npz["past_frames"].astype(self.dtype)
            future_frames = npz["future_frames"].astype(self.dtype)
            frames = np.concatenate([past_frames, future_frames], axis=0)
            past_track = npz["track_past"].astype(np.float64)
            future_track = npz["track_future"].astype(np.float64)

            meta = {
                "case_id": npz["case_id"].item(),
                "storm_id": npz["storm_id"].item(),
                "storm_name": npz["storm_name"].item(),
                "year": int(npz["year"]),
                "window_index": int(npz["window_index"]),
                "start_idx": int(npz["start_idx"]),
            }

        return ERA5Case(
            frames=frames,
            past_frames=past_frames,
            future_frames=future_frames,
            past_track=past_track,
            future_track=future_track,
            metadata=meta,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self._load_case(idx)

        track_norm = case.normalized_track.astype(self.dtype)

        sample = {
            "frames": torch.from_numpy(case.frames),
            "past_frames": torch.from_numpy(case.past_frames),
            "future_frames": torch.from_numpy(case.future_frames),
            "track_norm": torch.from_numpy(track_norm),
            "meta": case.metadata,
        }
        return sample


def build_dataloader(
    root: str | Path,
    split: Split,
    *,
    past_len: int = 8,
    future_len: int = 12,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = ERA5TyphoonDataset(root=root, split=split, past_len=past_len, future_len=future_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )


def prepare_lt3p_batch(
    batch: Dict[str, torch.Tensor],
    *,
    input_seq_len: int = 8,
    output_seq_len: int = 12,
    use_future_frames_only: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dataset batch into the tensors required by TrajectoryTransformer.

    Returns:
        src_coords: (B, input_seq_len, 2)
        video_tensor: (B, video_T, C, H, W)
    """
    track = batch["track_norm"]
    src = track[:, :input_seq_len, :]

    if use_future_frames_only:
        video_tensor = batch["future_frames"]
    else:
        video_tensor = torch.cat([batch["past_frames"], batch["future_frames"]], dim=1)

    return src.float(), video_tensor.float()


def describe_dataset(root: str | Path) -> Dict[str, Tuple[int, Tuple[int, ...]]]:
    """
    Quick summary (count + tensor shapes) for sanity checks.
    """
    summary: Dict[str, Tuple[int, Tuple[int, ...]]] = {}
    for split in ("train", "val", "test"):
        try:
            dataset = ERA5TyphoonDataset(root, split=split)
        except Exception:
            continue
        first = dataset[0]
        summary[split] = (
            len(dataset),
            tuple(first["frames"].shape),
        )
    return summary

