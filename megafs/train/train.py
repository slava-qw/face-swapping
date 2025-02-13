import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict
import lpips
from dataclasses import dataclass

from inference.megafs import HieRFE, Generator, resnet50
from inference.swapper import FaceTransferAttnModule
from loss_models import IdentityLoss, LandmarkLoss
from utils.img_utils import MyImagePathDataset


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 1e-2
    lpips_weight: float = 32.0
    id_weight: float = 24.0
    landmark_weight: float = 1e5
    norm_weight: float = 32.0
    embed_dim: int = 512
    num_heads: int = 4
    generator_channels: int = 1024
    latent_dim: int = 512
    num_latents: Tuple[int] = (4, 6, 8)
    backbone_depth: int = 50


class FTAMTrainer:
    """Trainer class for Face Transfer Attention Module (FTAM)"""

    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device

        # Initialize models
        self.hierfe = HieRFE(
            backbone=resnet50(pretrained=False),
            num_latents=config.num_latents,
            depth=config.backbone_depth
        ).to(device).eval()

        self.generator = Generator(
            size=config.generator_channels,
            style_dim=config.latent_dim,
            n_latent=8,
            channel_multiplier=2
        ).to(device).eval()

        self.ftam = FaceTransferAttnModule(
            num_heads=config.num_heads,
            embed_dim=config.embed_dim
        ).to(device)

        # Initialize loss functions
        self.id_loss = IdentityLoss(device)
        self.landmark_loss = LandmarkLoss(device)
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.mse_loss = nn.MSELoss()

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.ftam.parameters(),
            lr=config.learning_rate
        )

    def compute_loss(self, src: torch.Tensor, tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute multi-component loss for face transfer"""
        with torch.no_grad():
            src_enc, _ = self.hierfe(src)
            tgt_enc, tgt_struct = self.hierfe(tgt)

        # Perform face transfer
        swapped_enc = self.ftam(src_enc, tgt_enc)
        swapped_face, _ = self.generator(tgt_struct, [swapped_enc, None], randomize_noise=False)

        # Calculate loss components
        loss_dict = {
            'lpips': self.config.lpips_weight * self.lpips_loss(tgt, swapped_face, normalize=True),
            'identity': self.config.id_weight * self.id_loss(tgt, swapped_face),
            'landmark': self.config.landmark_weight * self.landmark_loss(tgt, swapped_face),
            'norm': self.config.norm_weight * self.mse_loss(src_enc, swapped_enc)
        }
        return loss_dict

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform single training step"""
        self.ftam.train()
        self.optimizer.zero_grad()

        src, tgt = batch
        loss_dict = self.compute_loss(src, tgt)
        total_loss = sum(loss_dict.values())

        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}

    def train(self, dataloader: DataLoader, num_epochs: int = 1):
        """Full training loop"""
        for epoch in range(num_epochs):
            for batch in dataloader:
                batch = [x.to(self.device) for x in batch]
                loss_metrics = self.train_step(batch)
                # TODO: Add logging/checkpoint saving


def main():
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MyImagePathDataset(img_root="path/to/images", pairs="pairs.txt")
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    trainer = FTAMTrainer(config, device)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
