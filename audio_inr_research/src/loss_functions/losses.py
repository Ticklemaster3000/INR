import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. TIME-DOMAIN LOSSES
# ==========================================

class MSELoss(nn.Module):
    """Standard Mean Squared Error."""
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)

class L1Loss(nn.Module):
    """L1 is often more robust to outliers than MSE."""
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)

# ==========================================
# 2. FREQUENCY-DOMAIN (SPECTRAL) LOSSES
# ==========================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    Computes the STFT loss over multiple resolutions (FFT sizes).
    Extremely effective for audio quality and high-frequency reconstruction.
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft_loss(self, x, y, n_fft, hop_length, win_length):
        # x, y: (Batch, Time)
        x_stft = torch.stft(x, n_fft, hop_length, win_length, return_complex=True, pad_mode='reflect')
        y_stft = torch.stft(y, n_fft, hop_length, win_length, return_complex=True, pad_mode='reflect')
        
        x_mag = torch.abs(x_stft) + 1e-8
        y_mag = torch.abs(y_stft) + 1e-8
        
        # Spectral Convergence Loss
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        # Log Magnitude Loss
        mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        
        return sc_loss + mag_loss

    def forward(self, pred, target):
        # Reshape to (Batch, Time) if necessary
        pred, target = pred.squeeze(-1), target.squeeze(-1)
        total_loss = 0
        for f, h, w in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            total_loss += self.stft_loss(pred, target, f, h, w)
        return total_loss / len(self.fft_sizes)

# ==========================================
# 3. THE PLUG-AND-PLAY ORCHESTRATOR
# ==========================================

class HybridLoss(nn.Module):
    """
    Combines Time-Domain and Spectral-Domain losses.
    Commonly: Total = alpha * MSE + beta * MultiSTFT
    """
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.time_loss = nn.MSELoss()
        self.spectral_loss = MultiResolutionSTFTLoss()

    def forward(self, pred, target):
        t_loss = self.time_loss(pred, target)
        s_loss = self.spectral_loss(pred, target)
        return self.alpha * t_loss + self.beta * s_loss

# ==========================================
# 4. REGISTRY
# ==========================================

LOSS_REGISTRY = {
    "mse": MSELoss,
    "l1": L1Loss,
    "stft": MultiResolutionSTFTLoss,
    "hybrid": HybridLoss
}

def get_loss(name, params={}):
    return LOSS_REGISTRY[name](**params)