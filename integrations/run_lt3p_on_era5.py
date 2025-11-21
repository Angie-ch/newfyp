"""
Minimal runner that feeds the local ERA5 dataset into the LT3P TrajectoryTransformer.

Usage:
    python integrations/run_lt3p_on_era5.py --split val --checkpoint ./LT3P/1900.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from integrations.era5_lt3p_adapter import (
    build_dataloader,
    prepare_lt3p_batch,
)
from LT3P.lt_tpc import TrajectoryTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/angiecheong/Desktop/fyp3/data/data/era5/typhoon_data_2018_2021_full"),
        help="Root directory that contains train/val/test/cases NPZ files.",
    )
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("/Users/angiecheong/Desktop/fyp3/LT3P/1900.pth"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--future-only-video", action="store_true", default=True)
    parser.add_argument("--max-batches", type=int, default=0, help="Optional cap for quick smoke tests.")
    return parser.parse_args()


def build_model(device: torch.device) -> TrajectoryTransformer:
    model = TrajectoryTransformer(
        input_seq_len=8,
        output_seq_len=12,
        video_dim=64,
        tensor_dim=64,
        embed_size=64,
        num_layers=1,
        heads=4,
        video_shape=(12, 24, 64, 64),
    )
    return model.to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataloader: DataLoader = build_dataloader(
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(device)
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()

    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{args.split} split")):
            src, video = prepare_lt3p_batch(
                batch,
                use_future_frames_only=args.future_only_video,
            )
            src = src.to(device)
            video = video.to(device)
            _ = model(src, video)
            processed += src.size(0)
            if args.max_batches and (batch_idx + 1) >= args.max_batches:
                break

    print(f"Processed {processed} samples from {args.split} split.")


if __name__ == "__main__":
    main()

