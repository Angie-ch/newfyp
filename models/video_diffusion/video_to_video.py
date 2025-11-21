"""
Video-to-video diffusion wrapper built on top of Imagen-PyTorch.

This mirrors the setup used in the “Improving Tropical Cyclone Forecasting With
Video Diffusion Models” repo referenced from the Awesome Diffusion Models list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from imagen_pytorch import Imagen, ImagenTrainer, Unet3D


@dataclass
class VideoDiffusionConfig:
    channels: int = 24
    cond_channels: int = 24
    video_frames: int = 12
    cond_frames: int = 8
    image_size: int = 64
    base_dim: int = 64
    dim_mults: tuple[int, ...] = (1, 2, 4)
    timesteps: int = 250
    lr: float = 3e-4
    cond_drop_prob: float = 0.1


class TyphoonVideoToVideoDiffusion:
    """
    Lightweight interface for training a conditional video diffusion model
    using past ERA5 frames to denoise future ERA5 frames.
    """

    def __init__(self, config: VideoDiffusionConfig, device: torch.device | str = "cuda"):
        self.cfg = config
        self.device = torch.device(device)
        self._build_model()

    def _build_model(self) -> None:
        unet = Unet3D(
            dim=self.cfg.base_dim,
            channels=self.cfg.channels,
            dim_mults=self.cfg.dim_mults,
            cond_on_text=False,
        )

        self.imagen = Imagen(
            unets=unet,
            image_sizes=(self.cfg.image_size,),
            timesteps=self.cfg.timesteps,
            condition_on_text=False,
            channels=self.cfg.channels,
            cond_drop_prob=self.cfg.cond_drop_prob,
        ).to(self.device)

        self.trainer = ImagenTrainer(
            self.imagen,
            use_ema=True,
            lr=self.cfg.lr,
            verbose=False,
        ).to(self.device)

    def training_step(
        self,
        future_frames: torch.Tensor,
        past_frames: torch.Tensor,
        *,
        unet_number: int = 1,
    ) -> torch.Tensor:
        """
        future_frames: (B, C, T_future, H, W)
        past_frames:   (B, C, T_past,  H, W)
        """
        future_frames = future_frames.to(self.device)
        past_frames = past_frames.to(self.device)

        loss = self.trainer(
            images=future_frames,
            cond_video_frames=past_frames,
            unet_number=unet_number,
            ignore_time=False,
        )
        self.trainer.update(unet_number=unet_number)
        if hasattr(loss, "detach"):
            return loss.detach()
        return torch.as_tensor(loss)

    @torch.no_grad()
    def sample(
        self,
        past_frames: torch.Tensor,
        *,
        video_frames: Optional[int] = None,
        cond_scale: float = 3.0,
    ) -> torch.Tensor:
        past_frames = past_frames.to(self.device)
        video_len = video_frames or self.cfg.video_frames
        sampled = self.imagen.sample(
            batch_size=past_frames.shape[0],
            cond_scale=cond_scale,
            use_tqdm=False,
            video_frames=video_len,
            cond_video_frames=past_frames,
        )
        return sampled.cpu()

    def save(self, path: str) -> None:
        self.trainer.save(path)

    def load(self, path: str) -> None:
        self.trainer.load(path)

