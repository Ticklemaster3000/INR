"""
SIREN²-EnCodec: Combined Architecture for Audio Super-Resolution

This package combines:
- EnCodec's SEANetEncoder (multi-scale convolutional feature extraction)
- SIREN² (coordinate-based decoder with spectral-adaptive noise)

Files:
- seanet_encoder.py: Simplified SEANet encoder from EnCodec
- siren_square.py: SIREN² implementation with spectral-adaptive noise
- siren2_encodec.py: Combined model (SEANet encoder + SIREN² decoder)

Usage:
    from siren2_encodec import SIREN2_EnCodec
    
    model = SIREN2_EnCodec(
        encoder_dim=128,
        encoder_ratios=[4, 4, 2],
        siren_hidden=256,
        siren_layers=4,
    )
    
    # Forward pass
    pred = model(coord, lr_audio)
"""

from .seanet_encoder import SimplifiedSEANetEncoder, SEANetResnetBlock
from .siren_square import SIREN_square, SineLayer, compute_spectral_centroid
from .siren2_encodec import SIREN2_EnCodec, build_siren2_encodec

__all__ = [
    'SIREN2_EnCodec',
    'build_siren2_encodec',
    'SimplifiedSEANetEncoder',
    'SEANetResnetBlock',
    'SIREN_square',
    'SineLayer',
    'compute_spectral_centroid',
]
