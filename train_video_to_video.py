"""
Train the video-to-video diffusion model using the local ERA5 typhoon dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from integrations.era5_lt3p_adapter import build_dataloader
from models.video_diffusion.video_to_video import (
    TyphoonVideoToVideoDiffusion,
    VideoDiffusionConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Users/angiecheong/Desktop/fyp3/data/data/era5/typhoon_data_2018_2021_full"))
    parser.add_argument("--split", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/video_diffusion"))
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    dataloader: DataLoader = build_dataloader(
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4,
    )

    cfg = VideoDiffusionConfig(
        channels=dataloader.dataset[0]["past_frames"].shape[1],
        cond_channels=dataloader.dataset[0]["past_frames"].shape[1],
        video_frames=dataloader.dataset[0]["future_frames"].shape[0],
        cond_frames=dataloader.dataset[0]["past_frames"].shape[0],
        lr=args.lr,
    )
    model = TyphoonVideoToVideoDiffusion(cfg, device=args.device)

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            past = batch["past_frames"].permute(0, 2, 1, 3, 4).contiguous()
            future = batch["future_frames"].permute(0, 2, 1, 3, 4).contiguous()
            loss = model.training_step(future_frames=future, past_frames=past)
            pbar.set_postfix({"loss": float(loss)})
            global_step += 1

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = args.save_dir / f"video_diffusion_epoch{epoch+1:03}.pt"
            model.save(str(ckpt_path))


if __name__ == "__main__":
    main()

