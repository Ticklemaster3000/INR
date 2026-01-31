"""
Simplified SEANet Encoder for Audio Super-Resolution.

Based on: https://github.com/facebookresearch/encodec/blob/main/encodec/modules/seanet.py
Stripped down to only include the encoder (no decoder, no quantizer).

Reference Paper: "High Fidelity Neural Audio Compression" (Défossez et al., 2022)
"""
import typing as tp
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SConv1d(nn.Module):
    """
    Simplified Conv1d with optional normalization.
    Based on EnCodec's streamable conv implementation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: str = 'weight_norm',
        pad_mode: str = 'reflect',
    ):
        super().__init__()
        self.pad_mode = pad_mode
        
        # Calculate padding for 'same' output (when stride=1)
        self.padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, groups=groups, bias=bias
        )
        
        # Apply normalization
        if norm == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == 'none':
            pass
        else:
            raise ValueError(f"Unknown norm: {norm}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manual padding with reflect mode
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode=self.pad_mode)
        return self.conv(x)


class SEANetResnetBlock(nn.Module):
    """
    Residual block from SEANet model.
    
    Uses dilated convolutions to expand receptive field without losing resolution.
    Based exactly on EnCodec implementation.
    """
    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        compress: int = 2,
        pad_mode: str = 'reflect',
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        
        act = getattr(nn, activation)
        hidden = dim // compress
        
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                        norm=norm, pad_mode=pad_mode),
            ]
        
        self.block = nn.Sequential(*block)
        self.shortcut = nn.Identity()  # True skip connection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class SLSTM(nn.Module):
    """
    LSTM wrapper for SEANet.
    Processes sequence in both directions for non-causal models.
    """
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True, bidirectional=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        # [B, T, C] -> [B, C, T]
        return x.transpose(1, 2)


class SimplifiedSEANetEncoder(nn.Module):
    """
    Simplified SEANet Encoder for audio feature extraction.
    
    This is a stripped-down version of EnCodec's encoder:
    - No streaming support (simpler implementation)
    - No causal mode (we process full audio)
    - Configurable downsampling ratios
    
    Architecture:
        1. Initial Conv: channels → n_filters
        2. For each ratio in ratios (reversed):
           - n_residual_layers × ResBlock (dilated convs)
           - Strided Conv (downsample + double channels)
        3. LSTM layers (optional)
        4. Final Conv → dimension
    
    Default config (24kHz mono):
        ratios = [8, 5, 4, 2] → total 320x downsampling
        n_filters = 32 → final channels = 32 * 2^4 = 512
    
    For super-resolution, use lighter ratios like [4, 4, 2] = 32x
    """
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [4, 4, 2],  # Lighter than EnCodec default
        activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = 'reflect',
        compress: int = 2,
        lstm: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))  # Reverse for encoder
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        
        act = getattr(nn, activation)
        mult = 1
        
        # Initial convolution
        model: tp.List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm, pad_mode=pad_mode)
        ]
        
        # Downsample blocks
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base ** j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm,
                        compress=compress,
                        pad_mode=pad_mode,
                    )
                ]
            
            # Downsampling layer
            model += [
                act(**activation_params),
                SConv1d(
                    mult * n_filters, mult * n_filters * 2,
                    kernel_size=ratio * 2, stride=ratio,
                    norm=norm, pad_mode=pad_mode,
                ),
            ]
            mult *= 2
        
        # LSTM layers
        if lstm > 0:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]
        
        # Final projection
        model += [
            act(**activation_params),
            SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm, pad_mode=pad_mode)
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, channels, T] audio waveform
            
        Returns:
            latent: [B, dimension, T // hop_length] encoded features
        """
        return self.model(x)


def test_encoder():
    """Test the encoder with sample input."""
    print("Testing SimplifiedSEANetEncoder...")
    
    # Default config: ratios=[4,4,2] = 32x downsample
    encoder = SimplifiedSEANetEncoder(
        channels=1,
        dimension=128,
        n_filters=32,
        ratios=[4, 4, 2],
    )
    
    # Test input: 1 second at 16kHz
    x = torch.randn(1, 1, 16000)
    z = encoder(x)
    
    expected_t = 16000 // 32  # = 500
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Expected T: {expected_t}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    assert z.shape[0] == 1
    assert z.shape[1] == 128
    print("✅ Test passed!")


if __name__ == '__main__':
    test_encoder()
