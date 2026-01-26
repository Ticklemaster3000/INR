"""
LISA-Enc: Exact replication of the original LISA encoder from the paper.
Reference: https://github.com/ml-postech/LISA/blob/master/models/audio_encoder.py

Paper: "Learning Continuous Representation of Audio for Arbitrary Scale Super Resolution" (ICASSP 2022)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """
    Original LISA ConvEncoder - Matches paper implementation exactly.
    
    Reference: https://github.com/ml-postech/LISA/blob/master/models/audio_encoder.py
    
    Architecture (matching original exactly):
        Conv1d(1 → 16, kernel=7, padding=3, stride=1) + Tanh
        Conv1d(16 → 32, kernel=3, padding=1, stride=1) + Tanh
        Conv1d(32 → 64, kernel=3, padding=1, stride=1) + Tanh
        Conv1d(64 → latent_dim, kernel=1, padding=0)
    
    Key features:
        - stride=1 throughout (preserves sequence length!)
        - Simple Tanh activation (no BatchNorm, no LeakyReLU)
        - Channel progression: 1→16→32→64→latent_dim
    """
    
    def __init__(self, latent_dim=32, in_dim=1, kernel_size=7, stride=1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        
        # Exact architecture from original paper
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_dim, 16, kernel_size, padding=3, stride=stride),
            nn.Tanh(),
            nn.Conv1d(16, 32, 3, padding=1, stride=1),
            nn.Tanh(),
            nn.Conv1d(32, 64, 3, padding=1, stride=1),
            nn.Tanh(),
            nn.Conv1d(64, latent_dim, 1, padding=0),
        )
    
    def forward(self, inp):
        """
        Args:
            inp: [B, in_dim, lr_length] - Low-resolution audio (channel-first)
            
        Returns:
            latent: [B, latent_dim, lr_length] - Encoded features (same length!)
        """
        assert inp.shape[1] == self.in_dim, f"Expected {self.in_dim} channels at dim 1, got {inp.shape[1]}"
        return self.conv_blocks(inp)


# Backward compatibility alias
GONEncoder = ConvEncoder


class LISAEncoder(nn.Module):
    """
    LISA-Enc: Complete LISA model with ConvEncoder - matching original paper exactly.
    
    Reference: https://github.com/ml-postech/LISA/blob/master/models/lisa.py#L240-L262
    
    This is the 'lisa-enc' model from the original paper:
    - ConvEncoder extracts latent features from low-res audio (stride=1, preserves length)
    - LISA's IMNET queries features at arbitrary coordinates
    - Local ensemble for smooth interpolation
    - Feature unfolding (prev/curr/next concatenation)
    """
    
    def __init__(
        self,
        latent_dim=32,
        hidden_features=256,
        num_layers=5,
        feat_unfold=True,
        local_ensemble=True,
    ):
        super().__init__()
        
        # Identifier for routing in training loop
        self.has_gon_encoder = True
        self.latent_dim = latent_dim
        
        # ConvEncoder (matching original exactly)
        self.encoder = ConvEncoder(latent_dim=latent_dim, in_dim=1)
        
        # LISA decoder
        from .models import LISA
        self.lisa = LISA(
            in_features=1,
            out_features=1,
            hidden_features=hidden_features,
            num_layers=num_layers,
            latent_dim=latent_dim,
            feat_unfold=feat_unfold,
            use_positional_encoding=True,
        )
    
    def forward(self, coord, lr_audio, train=False):
        """
        Forward pass matching original LISA-Enc paper.
        
        Args:
            coord: [B, hr_length, 1] - High-res query coordinates
            lr_audio: [B, lr_length, 1] - Low-res audio input
            train: bool - Whether in training mode
            
        Returns:
            pred: [B, hr_length, 1] - Predicted high-res audio
        """
        batch_size = coord.shape[0]
        
        # Encode: [B, lr_length, 1] -> [B, 1, lr_length] -> [B, latent_dim, lr_length]
        inp = lr_audio.transpose(1, 2)  # [B, 1, lr_length]
        latent = self.encoder(inp)  # [B, latent_dim, lr_length] (stride=1!)
        latent = latent.transpose(1, 2)  # [B, lr_length, latent_dim]
        
        # Query LISA at high-res coordinates
        pred = self.lisa.query_features(coord, latent)
        
        return pred
    
    def query_features(self, coord, lr_audio):
        """Wrapper for compatibility with training loop."""
        return self.forward(coord, lr_audio, train=self.training)


# Backward compatibility alias
LISA_WithGON = LISAEncoder


def build_lisa_with_gon(
    latent_dim=32,
    hidden_features=256,
    num_layers=5,
):
    """
    Helper function to build LISA-Enc (matching original paper).
    
    Note: latent_dim=32 is the default in original paper.
    Uses ConvEncoder with stride=1 (preserves sequence length).
    """
    return LISAEncoder(
        latent_dim=latent_dim,
        hidden_features=hidden_features,
        num_layers=num_layers,
        feat_unfold=True,
        local_ensemble=True,
    )
