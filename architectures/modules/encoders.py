import torch
import torch.nn as nn
from .base import BaseINR
from .decoders import LISA

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=32, in_dim=1):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_dim, 16, 7, padding=3),
            nn.Tanh(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.Tanh(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.Tanh(),
            nn.Conv1d(64, latent_dim, 1),
        )

    def forward(self, x):
        return self.conv_blocks(x)

class LISAEncoder(BaseINR):
    def __init__(self, latent_dim=32, hidden_features=256, num_layers=5):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = LISA(
            in_features=1,
            out_features=1,
            hidden_features=hidden_features,
            num_layers=num_layers,
            latent_dim=latent_dim
        )

    def forward(self, coord, lr_audio):
        latent = self.encoder(lr_audio.transpose(1, 2))
        latent = latent.transpose(1, 2)
        return self.decoder.query_features(coord, latent)