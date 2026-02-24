"""
Phase 3: SIREN²+EnCodec with PRETRAINED Encoder
Architecture: Pretrained EnCodec Encoder → Latents → SIREN Decoder
Key Change: Load pretrained 24kHz encoder weights, resample audio 22.05→24kHz
"""
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import pandas as pd

# Import existing modules
import sys
sys.path.append(str(Path(__file__).parent))
from siren2_encodec import SIREN2_EnCodec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== CONFIG ==============
class Config:
    lr = 1e-4
    gamma = 0.99
    scheduler_step = 20
    nb_epochs = 500
    max_samples = 150000
    SR_FACTOR = 2  # Super-resolution factor
    TARGET_SR = 24000  # Pretrained EnCodec expects 24kHz

config = Config()

# ============== HELPERS ==============
def load_audio(file_path, max_samples):
    """Load and normalize audio"""
    waveform, sr = librosa.load(file_path, sr=None, mono=True)
    waveform = waveform / np.max(np.abs(waveform))
    n_samples = min(max_samples, len(waveform))
    waveform = waveform[:n_samples]
    return torch.tensor(waveform, dtype=torch.float32).view(-1, 1), n_samples, sr

def resample_audio(audio_tensor, orig_sr, target_sr):
    """
    Resample audio from original sample rate to target sample rate.
    
    Args:
        audio_tensor: [N, 1] audio tensor
        orig_sr: Original sample rate (e.g., 22050)
        target_sr: Target sample rate (e.g., 24000)
    
    Returns:
        resampled: [M, 1] resampled audio tensor
    """
    import torchaudio.functional as F_audio
    
    # Flatten for resampling
    audio_flat = audio_tensor.squeeze()  # [N]
    
    # Resample
    resampled = F_audio.resample(
        audio_flat.unsqueeze(0),  # [1, N]
        orig_freq=orig_sr,
        new_freq=target_sr
    ).squeeze(0)  # [M]
    
    return resampled.view(-1, 1)  # [M, 1]

def calculate_lsd(y_pred, y_true, sr=16000):
    """Calculate Log-Spectral Distance"""
    if not torch.is_tensor(y_pred): y_pred = torch.tensor(y_pred)
    if not torch.is_tensor(y_true): y_true = torch.tensor(y_true)
    
    device_lsd = y_true.device
    y_pred = y_pred.to(device_lsd)
    
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(device_lsd)
    
    stft_pred = torch.stft(y_pred.view(-1), n_fft=n_fft, hop_length=hop_length,
                          return_complex=True, window=window)
    stft_true = torch.stft(y_true.view(-1), n_fft=n_fft, hop_length=hop_length,
                          return_complex=True, window=window)
    
    spec_pred = torch.abs(stft_pred) ** 2
    spec_true = torch.abs(stft_true) ** 2
    
    eps = 1e-10
    frame_energy = torch.sum(spec_true, dim=0)
    frame_energy_db = 10 * torch.log10(frame_energy + eps)
    mask = frame_energy_db > (frame_energy_db.max() - 60)
    
    if mask.sum() == 0:
        return 0.0
    
    spec_pred = spec_pred[:, mask]
    spec_true = spec_true[:, mask]
    
    max_freq_bin = int((8000 / sr) * (n_fft // 2 + 1))
    spec_pred = spec_pred[:max_freq_bin]
    spec_true = spec_true[:max_freq_bin]
    
    spec_pred = torch.clamp(spec_pred, min=eps)
    spec_true = torch.clamp(spec_true, min=eps)
    
    log_ratio = torch.log10(spec_true / spec_pred)
    lsd = torch.mean(torch.sqrt(torch.mean(log_ratio ** 2, dim=0)))
    
    return lsd.item()

def calculate_metrics(y_pred, y_true, sr):
    """Calculate PSNR, LSD"""
    if torch.is_tensor(y_pred): y_pred_np = y_pred.detach().cpu().numpy().flatten()
    else: y_pred_np = y_pred.flatten()
    
    if torch.is_tensor(y_true): y_true_np = y_true.detach().cpu().numpy().flatten()
    else: y_true_np = y_true.flatten()
    
    t_pred = torch.tensor(y_pred_np).float()
    t_true = torch.tensor(y_true_np).float()
    
    # PSNR
    mse = np.mean((y_pred_np - y_true_np) ** 2)
    max_val = 2.0
    psnr = 100.0 if mse == 0 else 20 * np.log10(max_val / np.sqrt(mse))
    
    # LSD
    lsd = calculate_lsd(t_pred, t_true, sr=sr)
    
    return {"PSNR": psnr, "LSD": lsd}

def train(model, lr_audio_24k, target, config, device, nb_epochs):
    """
    Train SIREN²+EnCodec model with pretrained encoder.
    
    Args:
        model: SIREN2_EnCodec with pretrained encoder
        lr_audio_24k: [1, 1, N_24k] - LR audio resampled to 24kHz
        target: [N_orig, 1] - Target audio at ORIGINAL sample rate
        config: Training config
        device: torch device
        nb_epochs: Number of epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.gamma)
    criterion = nn.MSELoss()
    
    model.train()
    n_samples = target.shape[0]
    
    # Prepare coordinates [1, N, 1] (with batch dimension) at ORIGINAL resolution
    coords = torch.linspace(-1, 1, n_samples, device=device).view(1, -1, 1)
    
    # Add batch dimension to target [1, N, 1]
    target_batched = target.unsqueeze(0) if target.dim() == 2 else target
    
    for epoch in tqdm(range(nb_epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Forward pass (encoder processes 24kHz audio, decoder outputs at original resolution)
        output = model(coords, lr_audio_24k)
        
        loss = criterion(output, target_batched)
        loss.backward()
        optimizer.step()
        scheduler.step()

# ============== MAIN BENCHMARK ==============
def main():
    print("\n" + "="*80)
    print("SIREN²+EnCodec with PRETRAINED Encoder (24kHz)")
    print("="*80)
    print(f"Pretrained Encoder: EnCodec 24kHz (6 kbps)")
    print(f"Resampling: 22.05 kHz → {config.TARGET_SR} Hz")
    print(f"Training: Decoder only (encoder frozen)")
    print("="*80 + "\n")
    
    # All 8 test files
    data_dir = Path(r"E:\INR\CODE\INR\audio_inr_research\data\raw")
    test_files = [
        data_dir / "sample_0026.wav",
        data_dir / "sample_0027.wav",
        data_dir / "sample_0028.wav",
        data_dir / "sample_0029.wav",
        data_dir / "sample_0030.wav",
        data_dir / "sample_0031.wav",
        data_dir / "sample_0050.wav",
        data_dir / "sample_0075.wav",
    ]
    
    results = []
    
    for file_path in test_files:
        print(f"\\n{'='*80}")
        print(f"PROCESSING: {file_path.name}")
        print(f"{'='*80}")
        
        # Load Ground Truth (original sample rate: 22050 Hz)
        waveform_gt, n_samples, sample_rate = load_audio(file_path, config.max_samples)
        waveform_gt = waveform_gt.to(device)
        
        print(f"Original audio: {n_samples} samples @ {sample_rate} Hz")
        
        # ============== EXPERIMENT 1: FITTING ==============
        print(f"\\n[Task 1] Audio Representation (Fitting)")
        
        # RESAMPLE to 24kHz for encoder
        waveform_24k = resample_audio(waveform_gt, sample_rate, config.TARGET_SR)
        lr_audio_fit_24k = waveform_24k.transpose(0, 1).unsqueeze(0).to(device)  # [1, 1, N_24k]
        print(f"  Resampled for encoder: {waveform_24k.shape[0]} samples @ 24000 Hz")
        
        # Calculate spectral centroid
        SC_fit = np.sum(np.abs(np.fft.rfft(waveform_gt.cpu().numpy().flatten())) * np.fft.rfftfreq(n_samples, d=1)) / np.sum(np.abs(np.fft.rfft(waveform_gt.cpu().numpy().flatten())))
        SC_fit = SC_fit * 2 if SC_fit > 0 else 0
        
        model_fit = SIREN2_EnCodec(
            encoder_dim=128,
            siren_hidden=256,
            siren_layers=4,
            use_spectral_noise=True,
            spectral_centroid=SC_fit,
            S0=3000,
            S1=1.0,
            use_pretrained=True,  # ← Enable pretrained weights!
            freeze_encoder=True,  # ← Freeze encoder (decoder-only training)
        ).to(device)
        
        print(f"  > Training SIREN²+EnCodec (pretrained encoder) on Full data...")
        train(model_fit, lr_audio_fit_24k, waveform_gt, config, device, nb_epochs=config.nb_epochs)
        
        # Inference
        model_fit.eval()
        with torch.no_grad():
            coords_full = torch.linspace(-1, 1, n_samples, device=device).view(1, -1, 1)
            output_fit = model_fit(coords_full, lr_audio_fit_24k)
            output_fit = output_fit.squeeze(0)  # Remove batch dim for metrics
        
        scores_fit = calculate_metrics(output_fit, waveform_gt, sample_rate)
        print(f"    SIREN²+EnCodec (Pretrained) Fitting -> PSNR: {scores_fit['PSNR']:.2f} | LSD: {scores_fit['LSD']:.3f}")
        
        results.append({
            "File": file_path.name,
            "Task": "Fitting",
            "PSNR": scores_fit['PSNR'],
            "LSD": scores_fit['LSD']
        })
        
        # Save output
        output_dir = Path(r"E:\\INR\\CODE\\INR\\audio_inr_research\\siren2_encodec\\benchmark_outputs")
        sf.write(output_dir / f"{file_path.stem}_encodec_pretrained_fit.wav", output_fit.cpu().detach().squeeze().numpy(), sample_rate)
        
        del model_fit
        torch.cuda.empty_cache()
        
        # ============== EXPERIMENT 2: SUPER-RESOLUTION ==============
        print(f"\\n[Task 2] Super-Resolution (x{config.SR_FACTOR})")
        
        # LR Audio = Downsampled (for super-res task)
        waveform_lr = waveform_gt[::config.SR_FACTOR]
        
        # RESAMPLE LR to 24kHz for encoder
        waveform_lr_24k = resample_audio(waveform_lr, sample_rate // config.SR_FACTOR, config.TARGET_SR)
        lr_audio_sr_24k = waveform_lr_24k.transpose(0, 1).unsqueeze(0).to(device)  # [1, 1, M_24k]
        print(f"  Downsampled LR: {waveform_lr.shape[0]} samples")
        print(f"  Resampled for encoder: {waveform_lr_24k.shape[0]} samples @ 24000 Hz")
        
        # Calculate spectral centroid from LR audio
        SC_sr = np.sum(np.abs(np.fft.rfft(waveform_lr.cpu().numpy().flatten())) * np.fft.rfftfreq(len(waveform_lr), d=1)) / np.sum(np.abs(np.fft.rfft(waveform_lr.cpu().numpy().flatten())))
        SC_sr = SC_sr * 2 if SC_sr > 0 else 0
        
        model_sr = SIREN2_EnCodec(
            encoder_dim=128,
            siren_hidden=256,
            siren_layers=4,
            use_spectral_noise=True,
            spectral_centroid=SC_sr,
            S0=2000,
            S1=0.001,
            use_pretrained=True,  # ← Enable pretrained weights!
            freeze_encoder=True,  # ← Freeze encoder
        ).to(device)
        
        print(f"  > Training SIREN²+EnCodec (pretrained encoder) on Low-Res data...")
        train(model_sr, lr_audio_sr_24k, waveform_gt, config, device, nb_epochs=config.nb_epochs)
        
        # Inference on High-Res Grid
        model_sr.eval()
        with torch.no_grad():
            coords_full = torch.linspace(-1, 1, n_samples, device=device).view(1, -1, 1)
            output_sr = model_sr(coords_full, lr_audio_sr_24k)
            output_sr = output_sr.squeeze(0)  # Remove batch dim for metrics
        
        scores_sr = calculate_metrics(output_sr, waveform_gt, sample_rate)
        print(f"    SIREN²+EnCodec (Pretrained) Super-Res -> PSNR: {scores_sr['PSNR']:.2f} | LSD: {scores_sr['LSD']:.3f}")
        
        results.append({
            "File": file_path.name,
            "Task": "Super-Res",
            "PSNR": scores_sr['PSNR'],
            "LSD": scores_sr['LSD']
        })
        
        # Save output
        sf.write(output_dir / f"{file_path.stem}_encodec_pretrained_sr.wav", output_sr.cpu().detach().squeeze().numpy(), sample_rate)
        
        del model_sr
        torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(results)
    print(f"\\n{'='*80}")
    print("FINAL RESULTS (Phase 3: SIREN²+EnCodec PRETRAINED)")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    output_dir = Path(r"E:\\INR\\CODE\\INR\\audio_inr_research\\siren2_encodec\\benchmark_outputs")
    df.to_csv(output_dir / "benchmark_results_phase3_pretrained.csv", index=False)
    print(f"\\nResults saved to: {output_dir / 'benchmark_results_phase3_pretrained.csv'}")

if __name__ == "__main__":
    main()
