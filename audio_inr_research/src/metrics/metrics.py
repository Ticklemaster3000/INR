import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Dict, List

# ==========================================
# 1. STANDARD SIGNAL METRICS
# ==========================================

def compute_psnr(preds: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Commonly used in SR to measure reconstruction fidelity.
    """
    mse = F.mse_loss(preds, target)
    if mse == 0:
        return float('inf')
    # Assuming audio is normalized between -1 and 1, max_val is 1.0 (or 2.0 range)
    # Most researchers use 1.0 as the reference peak.
    max_val = 1.0 
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def compute_snr(preds: torch.Tensor, target: torch.Tensor) -> float:
    noise = target - preds
    snr = 10 * torch.log10(torch.sum(target**2) / (torch.sum(noise**2) + 1e-8))
    return snr.item()

# ==========================================
# 2. PERCEPTUAL METRIC WRAPPERS
# ==========================================

def compute_visqol(preds: torch.Tensor, target: torch.Tensor, fs=16000) -> float:
    """
    Placeholder for ViSQOL. 
    Requires: pip install pystoi (as a common proxy) or the official Google ViSQOL binary.
    """
    # For now, we return a dummy or use a simpler perceptual proxy like STOI
    try:
        from pystoi import stoi
        return stoi(target.cpu().numpy(), preds.cpu().numpy(), fs, extended=False)
    except ImportError:
        print("Warning: pystoi not installed. Skipping perceptual metric.")
        return 0.0

# ==========================================
# 3. FREQUENCY-DOMAIN (LSD)
# ==========================================

def compute_lsd(preds: torch.Tensor, target: torch.Tensor) -> float:
    p_spec = torch.stft(preds.view(-1), n_fft=2048, hop_length=512, return_complex=True)
    t_spec = torch.stft(target.view(-1), n_fft=2048, hop_length=512, return_complex=True)
    
    p_log_power = torch.log10(torch.abs(p_spec)**2 + 1e-8)
    t_log_power = torch.log10(torch.abs(t_spec)**2 + 1e-8)
    
    dist = torch.sqrt(torch.mean((t_log_power - p_log_power)**2, dim=0))
    return torch.mean(dist).item()

# ==========================================
# 4. MODULAR TRACKER
# ==========================================

class MetricSuite:
    def __init__(self, metrics_to_use: List[str] = None):
        # The registry makes it 'Plug and Play'
        self.registry = {
            "psnr": compute_psnr,
            "snr": compute_snr,
            "lsd": compute_lsd,
            "visqol": compute_visqol,
            "rmse": lambda p, t: torch.sqrt(F.mse_loss(p, t)).item()
        }
        self.active_metrics = metrics_to_use or ["psnr", "lsd"]

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        # Ensure tensors are on CPU and flattened for metric calc
        p, t = preds.detach().cpu(), target.detach().cpu()
        
        results = {}
        for name in self.active_metrics:
            if name in self.registry:
                results[name] = self.registry[name](p, t)
        return results