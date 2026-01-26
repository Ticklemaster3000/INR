"""
GON (Gradient Origin Network) Encoder for LISA
Extracts rich latent representations from low-resolution audio
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GONEncoder(nn.Module):
    """
    Gradient Origin Network - Convolutional encoder for audio feature extraction.
    
    Takes low-resolution audio and produces rich latent representations
    that capture multi-scale temporal and spectral features.
    
    Architecture:
        Input: [B, 1, lr_length] - Low-res audio waveform
        ↓ Conv layers with increasing channels
        ↓ Feature extraction at multiple scales
        Output: [B, num_latents, latent_dim] - Rich learned features
    """
    
    def __init__(
        self,
        in_channels=1,
        latent_dim=64,
        hidden_channels=[64, 128, 256],
        kernel_sizes=[7, 5, 3],
        stride=2,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Build convolutional layers for feature extraction
        layers = []
        curr_channels = in_channels
        
        for i, (hidden_ch, kernel_size) in enumerate(zip(hidden_channels, kernel_sizes)):
            layers.extend([
                nn.Conv1d(
                    curr_channels,
                    hidden_ch,
                    kernel_size=kernel_size,
                    stride=stride if i < len(hidden_channels) - 1 else 1,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            curr_channels = hidden_ch
        
        self.conv_encoder = nn.Sequential(*layers)
        
        # Project to final latent dimension
        self.latent_projection = nn.Sequential(
            nn.Conv1d(hidden_channels[-1], latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh()  # Normalize latent features
        )
        
    def forward(self, lr_audio):
        """
        Args:
            lr_audio: [B, lr_length, 1] - Low-resolution audio
            
        Returns:
            latent: [B, num_latents, latent_dim] - Encoded features
        """
        # Reshape for conv1d: [B, lr_length, 1] -> [B, 1, lr_length]
        x = lr_audio.transpose(1, 2)
        
        # Extract features through conv layers
        features = self.conv_encoder(x)  # [B, hidden_channels[-1], reduced_length]
        
        # Project to latent space
        latent = self.latent_projection(features)  # [B, latent_dim, num_latents]
        
        # Transpose to [B, num_latents, latent_dim] for LISA
        latent = latent.transpose(1, 2)
        
        return latent


class LISA_WithGON(nn.Module):
    """
    LISA model integrated with GON encoder.
    Complete implementation as per the paper.
    """
    
    def __init__(
        self,
        encoder_config=None,
        lisa_config=None,
    ):
        super().__init__()
        
        # Identifier for routing in training loop
        self.has_gon_encoder = True
        
        # Default configs
        if encoder_config is None:
            encoder_config = {
                'in_channels': 1,
                'latent_dim': 64,
                'hidden_channels': [64, 128, 256],
                'kernel_sizes': [7, 5, 3],
                'stride': 2,
            }
        
        if lisa_config is None:
            lisa_config = {
                'in_features': 1,
                'out_features': 1,
                'hidden_features': 256,
                'num_layers': 5,
                'latent_dim': 64,
                'feat_unfold': True,
            }
        
        # GON Encoder for latent extraction
        self.encoder = GONEncoder(**encoder_config)
        
        # LISA's local implicit function (import from models.py)
        from .models import LISA
        self.lisa = LISA(**lisa_config)
        
    def forward(self, coord, lr_audio):
        """
        Forward pass with proper encoding.
        
        Args:
            coord: [B, hr_length, 1] - High-res coordinates
            lr_audio: [B, lr_length, 1] - Low-res audio input
            
        Returns:
            pred: [B, hr_length, 1] - Predicted high-res audio
        """
        # Encode low-res audio to rich latent features
        latent = self.encoder(lr_audio)  # [B, num_latents, latent_dim]
        
        # Use LISA to query high-res values
        pred = self.lisa.query_features(coord, latent)
        
        return pred
    
    def query_features(self, coord, lr_audio):
        """Wrapper for compatibility with training loop."""
        return self.forward(coord, lr_audio)


def build_lisa_with_gon(
    latent_dim=64,
    hidden_features=256,
    num_layers=5,
):
    """
    Helper function to build LISA with GON encoder.
    """
    encoder_config = {
        'in_channels': 1,
        'latent_dim': latent_dim,
        'hidden_channels': [64, 128, 256],
        'kernel_sizes': [7, 5, 3],
        'stride': 2,
    }
    
    lisa_config = {
        'in_features': 1,
        'out_features': 1,
        'hidden_features': hidden_features,
        'num_layers': num_layers,
        'latent_dim': latent_dim,
        'feat_unfold': True,
    }
    
    return LISA_WithGON(encoder_config, lisa_config)
