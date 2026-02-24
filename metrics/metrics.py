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


def compute_lsd_hf(preds: torch.Tensor, target: torch.Tensor, sr: int = 16000) -> float:
    """
    Log-Spectral Distance focusing on High Frequencies (>8kHz).
    Important for evaluating super-resolution quality.
    """
    p_spec = torch.stft(preds.view(-1), n_fft=2048, hop_length=512, return_complex=True)
    t_spec = torch.stft(target.view(-1), n_fft=2048, hop_length=512, return_complex=True)
    
    # Get frequency bins corresponding to >8kHz
    freq_bins = torch.linspace(0, sr/2, p_spec.shape[0])
    hf_mask = freq_bins > 8000
    
    # Check if we have any high frequency bins
    if not hf_mask.any():
        return float('nan')  # Sample rate too low for HF analysis
    
    p_log_power = torch.log10(torch.abs(p_spec[hf_mask])**2 + 1e-8)
    t_log_power = torch.log10(torch.abs(t_spec[hf_mask])**2 + 1e-8)
    
    dist = torch.sqrt(torch.mean((t_log_power - p_log_power)**2))
    return dist.item()


def compute_spectral_convergence(preds: torch.Tensor, target: torch.Tensor) -> float:
    """
    Spectral Convergence: Measures frequency domain reconstruction accuracy.
    Lower is better.
    """
    p_spec = torch.stft(preds.view(-1), n_fft=2048, hop_length=512, return_complex=True)
    t_spec = torch.stft(target.view(-1), n_fft=2048, hop_length=512, return_complex=True)
    
    p_mag = torch.abs(p_spec)
    t_mag = torch.abs(t_spec)
    
    sc = torch.norm(t_mag - p_mag, p='fro') / (torch.norm(t_mag, p='fro') + 1e-8)
    return sc.item()


def compute_envelope_distance(preds: torch.Tensor, target: torch.Tensor) -> float:
    """
    Envelope Distance: Measures temporal envelope matching.
    Important for audio quality perception.
    """
    # Simple envelope extraction using Hilbert transform approximation
    # Using moving average as a simple envelope extractor
    window_size = 512
    kernel = torch.ones(1, 1, window_size, device=preds.device) / window_size
    
    # Flatten and reshape properly for 1D convolution
    pred_abs = torch.abs(preds.view(-1)).unsqueeze(0).unsqueeze(0)
    target_abs = torch.abs(target.view(-1)).unsqueeze(0).unsqueeze(0)
    
    # Pad for convolution - use replicate instead of reflect to avoid dimension issues
    pad = window_size // 2
    pred_abs = F.pad(pred_abs, (pad, pad), mode='replicate')
    target_abs = F.pad(target_abs, (pad, pad), mode='replicate')
    
    pred_env = F.conv1d(pred_abs, kernel).squeeze()
    target_env = F.conv1d(target_abs, kernel).squeeze()
    
    return F.l1_loss(pred_env, target_env).item()


def compute_pesq(preds: torch.Tensor, target: torch.Tensor, fs: int = 16000) -> float:
    """
    PESQ (Perceptual Evaluation of Speech Quality).
    Requires: pip install pesq
    Range: -0.5 to 4.5 (higher is better)
    """
    try:
        from pesq import pesq
        pred_np = preds.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        
        # PESQ requires specific sample rates
        if fs not in [8000, 16000]:
            print(f"Warning: PESQ requires fs=8000 or 16000, got {fs}. Using 16000.")
            fs = 16000
            
        return pesq(fs, target_np, pred_np, 'wb')  # 'wb' for wideband (16kHz)
    except ImportError:
        print("Warning: pesq not installed. Run: pip install pesq")
        return 0.0
    except Exception as e:
        print(f"Warning: PESQ calculation failed: {e}")
        return 0.0

# ==========================================
# 4. MODULAR TRACKER
# ==========================================

class MetricSuite:
    def __init__(self, metrics_to_use: List[str] = None, sample_rate: int = 16000):
        # The registry makes it 'Plug and Play'
        self.registry = {
            "psnr": compute_psnr,
            "snr": compute_snr,
            "lsd": compute_lsd,
            "lsd_hf": lambda p, t: compute_lsd_hf(p, t, self.sample_rate),
            "spectral_convergence": compute_spectral_convergence,
            "envelope_distance": compute_envelope_distance,
            "pesq": lambda p, t: compute_pesq(p, t, self.sample_rate),
            "visqol": compute_visqol,
            "rmse": lambda p, t: torch.sqrt(F.mse_loss(p, t)).item()
        }
        self.active_metrics = metrics_to_use or ["psnr", "lsd", "lsd_hf", "spectral_convergence", "envelope_distance", "pesq"]
        self.sample_rate = sample_rate

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        # Ensure tensors are on CPU and flattened for metric calc
        p, t = preds.detach().cpu(), target.detach().cpu()
        
        results = {}
        for name in self.active_metrics:
            if name in self.registry:
                try:
                    results[name] = self.registry[name](p, t)
                except Exception as e:
                    print(f"Warning: Failed to compute {name}: {e}")
                    results[name] = 0.0
        return results