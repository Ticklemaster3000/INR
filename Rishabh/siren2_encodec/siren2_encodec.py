"""
SIREN²-EnCodec: Combined architecture for Audio Super-Resolution.

Combines:
1. EnCodec's SEANetEncoder - Multi-scale convolutional feature extraction
2. SIREN² - Coordinate-based decoder with spectral-adaptive noise

Architecture Flow:
    Low-Res Audio → SEANetEncoder → Latent
    HR Coordinates + Latent → SIREN² → High-Res Audio
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from seanet_encoder import SimplifiedSEANetEncoder
from siren_square import SIREN_square, compute_spectral_centroid


class BaseINR(nn.Module):
    """Base class for Implicit Neural Representations."""
    pass


class SIREN2_EnCodec(BaseINR):
    """
    SIREN²-EnCodec: Hybrid architecture for audio super-resolution.
    
    This model combines:
    - SEANetEncoder: Extracts rich latent features from low-resolution audio
    - SIREN²: Decodes coordinates + latent into high-resolution audio
    
    The key insight is that SEANet provides multi-scale audio features
    (via strided convs + residual blocks + LSTM), while SIREN² provides
    continuous coordinate-based decoding with spectral-adaptive initialization.
    
    Args:
        # SEANet Encoder params
        encoder_dim: Latent dimension from encoder (default: 128)
        encoder_ratios: Downsampling ratios (default: [4, 4, 2] = 32x)
        n_filters: Base filter count for encoder (default: 32)
        n_residual_layers: ResBlocks per stage (default: 1)
        lstm_layers: LSTM layers in encoder (default: 2)
        
        # SIREN² Decoder params
        siren_hidden: Hidden dimension for SIREN² (default: 256)
        siren_layers: Number of hidden layers (default: 4)
        omega_0: Frequency scaling (default: 30.0)
        first_omega: First layer frequency (default: 30.0)
        use_spectral_noise: Whether to use SIREN²'s noise injection
        
        # General params
        out_features: Output dimension (default: 1)
    """
    
    def __init__(
        self,
        # SEANet params
        encoder_dim: int = 128,
        encoder_ratios: list = [4, 4, 2],
        n_filters: int = 32,
        n_residual_layers: int = 1,
        lstm_layers: int = 2,
        
        # SIREN² params
        siren_hidden: int = 256,
        siren_layers: int = 4,
        omega_0: float = 30.0,
        first_omega: float = 30.0,
        use_spectral_noise: bool = True,
        spectral_centroid: float = 0.0,
        S0: float = 0.0,
        S1: float = 0.0,
        use_pretrained: bool = False,
        freeze_encoder: bool = True,
        
        # General
        out_features: int = 1,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.encoder_ratios = encoder_ratios
        self.hop_length = int(np.prod(encoder_ratios))
        self.use_spectral_noise = use_spectral_noise
        
        # SEANet Encoder
        self.encoder = SimplifiedSEANetEncoder(
            channels=1,
            dimension=encoder_dim,
            n_filters=n_filters,
            n_residual_layers=n_residual_layers,
            ratios=encoder_ratios,
            lstm=lstm_layers,
        )
        
        # Latent normalization - scale encoder output to match SIREN² expected input range
        # Encoder outputs small values (std ~0.02), SIREN² expects values in [-1, 1] range
        self.latent_scale = nn.Parameter(torch.ones(1, encoder_dim, 1) * 10.0)  # Learnable scale
        self.latent_bias = nn.Parameter(torch.zeros(1, encoder_dim, 1))  # Learnable bias
        
        # SIREN² Decoder
        # Input: 1 (coord) + encoder_dim (latent)
        # NOTE: Removed lr_interp to match reference SIREN² (coordinates only + encoder features)
        self.decoder = SIREN_square(
            omega_0=omega_0,
            in_dim=1 + encoder_dim,
            HL_dim=siren_hidden,
            out_dim=out_features,
            first_omega=first_omega,
            n_HLs=siren_layers,
            spectral_centroid=spectral_centroid,
            S0=S0,
            S1=S1,
        )
        
        # NOTE: Zero-init was removed because we're doing direct prediction (no residual)
        # The SIREN_square class handles its own weight initialization
        
        # Store current spectral centroid
        self._current_sc = 0.0
        
        # Load pretrained encoder weights if requested
        if use_pretrained:
            print("Loading pretrained EnCodec 24kHz encoder weights...")
            self.load_pretrained_encoder(freeze=freeze_encoder)
    
    def encode(self, lr_audio: torch.Tensor) -> torch.Tensor:
        """
        Encode low-resolution audio to latent representation.
        
        Args:
            lr_audio: [B, lr_len, 1] or [B, 1, lr_len] low-res audio
            
        Returns:
            latent: [B, encoder_dim, T_enc] encoded features (scaled)
        """
        # Ensure channel-first format: [B, 1, lr_len]
        if lr_audio.dim() == 3 and lr_audio.shape[-1] == 1:
            lr_audio = lr_audio.transpose(1, 2)  # [B, lr_len, 1] -> [B, 1, lr_len]
        elif lr_audio.dim() == 2:
            lr_audio = lr_audio.unsqueeze(1)  # [B, lr_len] -> [B, 1, lr_len]
        
        # Encode
        latent = self.encoder(lr_audio)
        
        # Apply learnable scaling (small encoder output -> SIREN² expected range)
        latent = latent * self.latent_scale + self.latent_bias
        
        return latent
    
    def load_pretrained_encoder(self, freeze: bool = True):
        """
        Load pretrained EnCodec 24kHz encoder weights.
        
        Args:
            freeze: If True, freeze encoder parameters (only train decoder)
        """
        try:
            from encodec import EncodecModel
        except ImportError:
            raise ImportError(
                "encodec library not found. Install with: pip install encodec"
            )
        
        # Load pretrained 24kHz model
        pretrained_model = EncodecModel.encodec_model_24khz()
        pretrained_model.set_target_bandwidth(6.0)  # 6 kbps bandwidth
        
        # Transfer encoder weights
        try:
            self.encoder.load_state_dict(
                pretrained_model.encoder.state_dict(),
                strict=False  # Allow minor architecture differences
            )
            print("✓ Pretrained encoder weights loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load all pretrained weights: {e}")
            print("  Continuing with partial weight transfer...")
        
        # Optionally freeze encoder parameters
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder weights frozen (decoder-only training)")
        else:
            print("✓ Encoder weights unfrozen (full fine-tuning enabled)")
    
    
    def decode(
        self,
        coord: torch.Tensor,
        latent: torch.Tensor,
        spectral_centroid: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Decode coordinates + latent to audio amplitudes.
        
        Args:
            coord: [B, hr_len, 1] query coordinates
            latent: [B, encoder_dim, T_enc] encoded features
            spectral_centroid: Optional override for noise injection
            
        Returns:
            pred: [B, hr_len, 1] predicted audio
        """
        batch_size = coord.shape[0]
        hr_len = coord.shape[1]
        
        # Update spectral noise if needed
        if self.use_spectral_noise and spectral_centroid is not None:
            if spectral_centroid != self._current_sc:
                self.decoder.update_spectral_centroid(spectral_centroid)
                self._current_sc = spectral_centroid
        
        # Interpolate latent to match HR coordinate positions
        # latent: [B, encoder_dim, T_enc] -> [B, encoder_dim, hr_len]
        latent_interp = F.interpolate(
            latent,
            size=hr_len,
            mode='linear',
            align_corners=False,
        )
        
        # Transpose to [B, hr_len, encoder_dim]
        latent_interp = latent_interp.transpose(1, 2)
        
        # Concatenate: [B, hr_len, 1 (coord) + encoder_dim (latent)]
        inp = torch.cat([coord, latent_interp], dim=-1)
        
        # SIREN² decode
        # Direct prediction (NO residual connection - matches reference SIREN²)
        pred = self.decoder(inp)
        
        return pred
    
    def forward(
        self,
        coord: torch.Tensor,
        lr_audio: torch.Tensor,
        spectral_centroid: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coord: [B, hr_len, 1] high-resolution query coordinates
            lr_audio: [B, lr_len, 1] or [B, 1, lr_len] low-resolution audio
            spectral_centroid: Optional spectral centroid for noise injection
            
        Returns:
            pred: [B, hr_len, 1] predicted high-resolution audio
        """
        # Compute spectral centroid from low-res audio if needed
        if self.use_spectral_noise and spectral_centroid is None:
            # Flatten and compute
            if lr_audio.dim() == 3:
                audio_flat = lr_audio[:, :, 0] if lr_audio.shape[-1] == 1 else lr_audio[:, 0, :]
            else:
                audio_flat = lr_audio
            spectral_centroid = compute_spectral_centroid(audio_flat[0])
        
        # Encode
        latent = self.encode(lr_audio)
        
        # Decode
        pred = self.decode(
            coord=coord,
            latent=latent,
            spectral_centroid=spectral_centroid,
        )
        
        return pred
    
    @property
    def has_gon_encoder(self) -> bool:
        """Flag for compatibility with training loop routing."""
        return True


def build_siren2_encodec(
    encoder_dim: int = 128,
    encoder_ratios: list = [4, 4, 2],
    siren_hidden: int = 256,
    siren_layers: int = 4,
    use_spectral_noise: bool = True,
) -> SIREN2_EnCodec:
    """
    Helper function to build SIREN²-EnCodec model.
    
    Default config is balanced for 16kHz audio with 32x downsampling.
    """
    return SIREN2_EnCodec(
        encoder_dim=encoder_dim,
        encoder_ratios=encoder_ratios,
        siren_hidden=siren_hidden,
        siren_layers=siren_layers,
        use_spectral_noise=use_spectral_noise,
    )


def test_siren2_encodec():
    """Test the combined model."""
    print("Testing SIREN2_EnCodec...")
    
    # Create model
    model = SIREN2_EnCodec(
        encoder_dim=128,
        encoder_ratios=[4, 4, 2],  # 32x downsample
        siren_hidden=256,
        siren_layers=4,
        use_spectral_noise=True,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
    
    # Test input
    batch_size = 2
    lr_len = 4000  # Low-res (e.g., 4kHz, 1 second)
    hr_len = 16000  # High-res (e.g., 16kHz, 1 second)
    
    lr_audio = torch.randn(batch_size, lr_len, 1)
    coord = torch.linspace(0, 1, hr_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
    
    print(f"\nInput shapes:")
    print(f"  coord: {coord.shape}")
    print(f"  lr_audio: {lr_audio.shape}")
    
    # Forward pass
    with torch.no_grad():
        pred = model(coord, lr_audio)
    
    print(f"\nOutput shape: {pred.shape}")
    assert pred.shape == (batch_size, hr_len, 1), f"Expected {(batch_size, hr_len, 1)}, got {pred.shape}"
    
    # Test encode/decode separately
    latent = model.encode(lr_audio)
    print(f"Latent shape: {latent.shape}")
    
    pred2 = model.decode(coord, latent, spectral_centroid=0.3)
    print(f"Decode output shape: {pred2.shape}")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_siren2_encodec()
