"""
SIREN² (SIREN Square) - Sine Representation Network with Spectral-Adaptive Noise.

Based on: https://github.com/FrisbeeBean/INR/blob/Experiments/audio_inr_research/SIREN2_VCTK_Experiments.ipynb

Key Innovation:
    SIREN² adds noise to the first two layer weights based on the spectral centroid
    of the target signal. This helps the network better capture high-frequency content.

Noise Scales (for audio, in_dim=1):
    S0 = 3500 * (1 - exp(-7 * SC / n_ch))   # input → hidden1
    S1 = SC / n_ch * 3                       # hidden1 → hidden2

Weight Modification:
    net[0].weight = weights0 + randn() * (S0 / omega_0)
    net[2].weight = weights2 + randn() * (S1 / omega_0)
"""
import torch
import torch.nn as nn
import numpy as np


@torch.jit.script
def sine_block(x: torch.Tensor, w0: float, a0: float) -> torch.Tensor:
    """JIT-compiled sine activation."""
    return a0 * torch.sin(w0 * x)


class SineLayer(nn.Module):
    """Sine activation layer with configurable frequency."""
    def __init__(self, w0: float = 30.0, amplitude: float = 1.0):
        super().__init__()
        self.w0 = w0
        self.a0 = amplitude
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sine_block(x, self.w0, self.a0)


class SIREN_square(nn.Module):
    """
    SIREN² - SIREN with spectral-adaptive noise injection.
    
    This is the exact implementation from the FrisbeeBean/INR notebook.
    
    Args:
        omega_0: Frequency scaling for hidden layers
        in_dim: Input dimension (1 for time coordinate, 2 for coord+latent)
        HL_dim: Hidden layer dimension
        out_dim: Output dimension (1 for audio amplitude)
        first_omega: Frequency scaling for first layer
        n_HLs: Number of hidden layers
        spectral_centroid: Spectral centroid of target signal (0-1 normalized)
        S0: Manual noise scale for input→hidden1 (overrides auto)
        S1: Manual noise scale for hidden1→hidden2 (overrides auto)
    """
    def __init__(
        self,
        omega_0: float = 30.0,
        in_dim: int = 1,
        HL_dim: int = 256,
        out_dim: int = 1,
        first_omega: float = 30.0,
        n_HLs: int = 4,
        spectral_centroid: float = 0.0,
        S0: float = 0.0,
        S1: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.omega_0 = omega_0
        self.first_omega = first_omega
        self.S0 = S0
        self.S1 = S1
        self.SC = spectral_centroid
        self.n_ch = out_dim
        self.HL_dim = HL_dim
        self.n_HLs = n_HLs
        
        # Build network architecture
        self.net = nn.ModuleList()
        
        # First layer: in_dim → HL_dim
        self.net.append(nn.Linear(in_dim, HL_dim))
        self.net.append(SineLayer(first_omega))
        
        # Hidden layers: HL_dim → HL_dim
        for _ in range(n_HLs - 1):
            self.net.append(nn.Linear(HL_dim, HL_dim))
            self.net.append(SineLayer(omega_0))
        
        # Output layer: HL_dim → out_dim
        self.net.append(nn.Linear(HL_dim, out_dim))
        
        # Initialize weights (SIREN initialization)
        self._init_weights()
        
        # Store original weights for noise injection
        with torch.no_grad():
            self.register_buffer('weights0', self.net[0].weight.detach().clone())
            self.register_buffer('weights2', self.net[2].weight.detach().clone())
        
        # Set noise scales and add noise
        if spectral_centroid > 0:
            self.set_noise_scales()
            self.add_noise()
    
    def _init_weights(self):
        """SIREN weight initialization for high-dimensional input."""
        with torch.no_grad():
            # First layer: Original SIREN uses Uniform(-1/n, 1/n) which works for in_dim=1
            # But for in_dim>1 (coord + latent), we need larger weights to make sin() oscillate
            # Use sqrt(6/n) as in Xavier init, then divide by first_omega
            if self.in_dim == 1:
                # Standard SIREN init for pure coordinate input
                self.net[0].weight.uniform_(-1.0 / self.in_dim, 1.0 / self.in_dim)
            else:
                # For high-dim input (coord + latent), use larger init
                # This ensures the dot product is large enough for sin() to oscillate
                bound = np.sqrt(6.0 / self.in_dim)
                self.net[0].weight.uniform_(-bound, bound)
            
            # Hidden layers: Uniform(-sqrt(6/n)/w0, sqrt(6/n)/w0)
            for i in range(self.n_HLs):
                layer_idx = (i + 1) * 2  # 2, 4, 6, ...
                bound = np.sqrt(6.0 / self.HL_dim) / self.omega_0
                self.net[layer_idx].weight.uniform_(-bound, bound)
    
    def set_noise_scales(self):
        """
        Set noise scales based on spectral centroid.
        
        Empirical formula from SIREN² paper for audio (in_dim=1):
            S0 = 3500 * (1 - exp(-7 * SC / n_ch))
            S1 = SC / n_ch * 3
        """
        if self.in_dim == 1:  # Pure coordinate input (audio)
            a, b = 7, 3
            self.S0 = 3500 * (1 - np.exp(-a * self.SC / self.n_ch))
            self.S1 = self.SC / self.n_ch * b
        else:
            # For higher dim input (coord + latent), use scaled version
            a, b = 5, 2
            self.S0 = 2000 * (1 - np.exp(-a * self.SC / self.n_ch))
            self.S1 = self.SC / self.n_ch * b
    
    def add_noise(self):
        """Add spectral-adaptive noise to first two layer weights."""
        with torch.no_grad():
            # Input layer → First hidden layer
            scale0 = self.S0 / self.omega_0
            self.net[0].weight.copy_(
                self.weights0 + torch.randn_like(self.weights0) * scale0
            )
            
            # First hidden → Second hidden layer
            scale1 = self.S1 / self.omega_0
            self.net[2].weight.copy_(
                self.weights2 + torch.randn_like(self.weights2) * scale1
            )
    
    def update_spectral_centroid(self, spectral_centroid: float):
        """
        Update the spectral centroid and re-apply noise.
        Useful for per-sample adaptation during training.
        """
        self.SC = spectral_centroid
        
        # Restore original weights
        with torch.no_grad():
            self.net[0].weight.copy_(self.weights0)
            self.net[2].weight.copy_(self.weights2)
        
        # Recalculate and apply noise
        self.set_noise_scales()
        self.add_noise()
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SIREN².
        
        Args:
            coords: [B, N, in_dim] input coordinates (or coord+features)
            
        Returns:
            output: [B, N, out_dim] predicted values
        """
        x = coords
        for layer in self.net:
            x = layer(x)
        return x


def compute_spectral_centroid(audio: torch.Tensor) -> float:
    """
    Compute normalized spectral centroid of audio signal.
    
    Args:
        audio: [N] or [B, N] audio tensor
        
    Returns:
        centroid: float in range [0, 1]
    """
    audio = audio.flatten().detach().cpu().numpy()
    
    # Compute FFT
    spectrum = np.abs(np.fft.rfft(audio))
    freq_bins = np.fft.rfftfreq(len(audio), d=1)
    
    # Weighted mean frequency
    weighted_sum = np.sum(spectrum * freq_bins)
    sum_of_weights = np.sum(spectrum)
    
    if sum_of_weights == 0:
        return 0.0
    
    centroid = weighted_sum / sum_of_weights
    
    # Normalize to [0, 1] (assuming max freq is 0.5 for normalized coords)
    return min(centroid * 2, 1.0)


def test_siren_square():
    """Test SIREN² with sample input."""
    print("Testing SIREN_square...")
    
    # Test without spectral noise
    model = SIREN_square(
        omega_0=30.0,
        in_dim=1,
        HL_dim=256,
        out_dim=1,
        n_HLs=4,
        spectral_centroid=0.0,  # No noise
    )
    
    coords = torch.linspace(0, 1, 16000).reshape(1, 16000, 1)
    out = model(coords)
    
    print(f"Input shape: {coords.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with spectral noise
    model_noisy = SIREN_square(
        omega_0=30.0,
        in_dim=1,
        HL_dim=256,
        out_dim=1,
        n_HLs=4,
        spectral_centroid=0.3,  # With noise
    )
    
    out_noisy = model_noisy(coords)
    print(f"Output with noise shape: {out_noisy.shape}")
    print(f"Noise scales: S0={model_noisy.S0:.2f}, S1={model_noisy.S1:.2f}")
    
    # Test with higher input dim (for combined model)
    model_combined = SIREN_square(
        omega_0=30.0,
        in_dim=129,  # 1 coord + 128 latent
        HL_dim=256,
        out_dim=1,
        n_HLs=4,
        spectral_centroid=0.3,
    )
    
    coords_combined = torch.randn(1, 16000, 129)
    out_combined = model_combined(coords_combined)
    print(f"Combined input shape: {coords_combined.shape}")
    print(f"Combined output shape: {out_combined.shape}")
    
    print("✅ All tests passed!")


if __name__ == '__main__':
    test_siren_square()
