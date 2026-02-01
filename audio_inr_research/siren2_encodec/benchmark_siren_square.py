"""
Phase 1: Vanilla SIREN_square Benchmark
Matches reference notebook workflow: Instance-specific training (train→infer→reset per audio)
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
from siren_square import SIREN_square

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== CONFIG ==============
class Config:
    lr = 1e-4
    gamma = 0.99
    scheduler_step = 20
    n_HLs = 4
    nb_epochs = 500
    HL_dim = 222
    max_samples = 150000
    n_channels = 1
    SR_FACTOR = 2  # Super-resolution factor

config = Config()

# ============== HELPERS ==============
def load_audio(file_path, max_samples):
    """Load and normalize audio"""
    waveform, sr = librosa.load(file_path, sr=None, mono=True)
    waveform = waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]
    n_samples = min(max_samples, len(waveform))
    waveform = waveform[:n_samples]
    return torch.tensor(waveform, dtype=torch.float32).view(-1, 1), n_samples, sr

def spectral_centroid(audio_np):
    """Calculate spectral centroid"""
    audio_np = audio_np.flatten()  # Handle shape (N,1) -> (N,)
    spectrum = np.abs(np.fft.rfft(audio_np))
    freq_bins = np.fft.rfftfreq(len(audio_np), d=1)
    weighted_sum = np.sum(spectrum * freq_bins)
    sum_of_weights = np.sum(spectrum)
    centroid = weighted_sum / sum_of_weights if sum_of_weights != 0 else 0
    return (centroid * 2)

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

def train(model, coords, target, config, device, nb_epochs, batch_size):
    """Train model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.gamma)
    criterion = nn.MSELoss()
    
    model.train()
    n_samples = coords.shape[0]
    
    for epoch in tqdm(range(nb_epochs), desc="Training"):
        indices = torch.randperm(n_samples, device=device)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            coord_batch = coords[batch_idx]
            target_batch = target[batch_idx]
            
            optimizer.zero_grad()
            output = model(coord_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()

def batched_forward(model, coords, batch_size=65536):
    """Inference in batches"""
    model.eval()
    outputs = []
    n_samples = coords.shape[0]
    
    for i in range(0, n_samples, batch_size):
        coord_batch = coords[i:i+batch_size]
        with torch.no_grad():
            output = model(coord_batch)
        outputs.append(output)
    
    return torch.cat(outputs, dim=0)

# ============== MAIN BENCHMARK ==============
def main():
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
        print(f"\n{'='*80}")
        print(f"PROCESSING: {file_path.name}")
        print(f"{'='*80}")
        
        # Load Ground Truth
        waveform_gt, n_samples, sample_rate = load_audio(file_path, config.max_samples)
        waveform_gt = waveform_gt.to(device)
        
        # Create Full Coordinate Grid
        coords_full = torch.linspace(-1, 1, n_samples, device=device).view(-1, config.n_channels)
        
        # ============== EXPERIMENT 1: FITTING ==============
        print(f"\n[Task 1] Audio Representation (Fitting)")
        SC_fit = spectral_centroid(waveform_gt.detach().cpu().numpy())
        
        model_fit = SIREN_square(
            omega_0=30, in_dim=1, HL_dim=config.HL_dim, out_dim=1,
            first_omega=30, n_HLs=config.n_HLs,
            spectral_centroid=SC_fit, S0=3000, S1=1.0
        ).to(device)
        
        print(f"  > Training SIREN_square on Full High-Res data...")
        batch_size = min(n_samples, 512*512)
        train(model_fit, coords_full, waveform_gt, config, device, nb_epochs=config.nb_epochs, batch_size=batch_size)
        
        # Inference
        output_fit = batched_forward(model_fit, coords_full, batch_size=65536).to(device)
        scores_fit = calculate_metrics(output_fit, waveform_gt, sample_rate)
        print(f"    SIREN_square Fitting -> PSNR: {scores_fit['PSNR']:.2f} | LSD: {scores_fit['LSD']:.3f}")
        
        results.append({
            "File": file_path.name,
            "Task": "Fitting",
            "PSNR": scores_fit['PSNR'],
            "LSD": scores_fit['LSD']
        })
        
        # Save output
        output_dir = Path(r"E:\INR\CODE\INR\audio_inr_research\siren2_encodec\benchmark_outputs")
        output_dir.mkdir(exist_ok=True)
        sf.write(output_dir / f"{file_path.stem}_fit.wav", output_fit.cpu().detach().squeeze().numpy(), sample_rate)
        
        del model_fit
        torch.cuda.empty_cache()
        
        # ============== EXPERIMENT 2: SUPER-RESOLUTION ==============
        print(f"\n[Task 2] Super-Resolution (x{config.SR_FACTOR})")
        
        coords_train_sr = coords_full[::config.SR_FACTOR]
        waveform_train_sr = waveform_gt[::config.SR_FACTOR]
        
        SC_sr = spectral_centroid(waveform_train_sr.detach().cpu().numpy())
        
        model_sr = SIREN_square(
            omega_0=30, in_dim=1, HL_dim=config.HL_dim, out_dim=1,
            first_omega=30, n_HLs=config.n_HLs,
            spectral_centroid=SC_sr, S0=2000, S1=0.001
        ).to(device)
        
        print(f"  > Training SIREN_square on Low-Res ({sample_rate//config.SR_FACTOR} Hz) data...")
        batch_size_sr = min(coords_train_sr.shape[0], 512*512)
        train(model_sr, coords_train_sr, waveform_train_sr, config, device, nb_epochs=config.nb_epochs, batch_size=batch_size_sr)
        
        # Inference on High-Res Grid
        output_sr = batched_forward(model_sr, coords_full, batch_size=65536).to(device)
        scores_sr = calculate_metrics(output_sr, waveform_gt, sample_rate)
        print(f"    SIREN_square Super-Res -> PSNR: {scores_sr['PSNR']:.2f} | LSD: {scores_sr['LSD']:.3f}")
        
        results.append({
            "File": file_path.name,
            "Task": "Super-Res",
            "PSNR": scores_sr['PSNR'],
            "LSD": scores_sr['LSD']
        })
        
        # Save output
        sf.write(output_dir / f"{file_path.stem}_sr.wav", output_sr.cpu().detach().squeeze().numpy(), sample_rate)
        
        del model_sr
        torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    df.to_csv(output_dir / "benchmark_results_phase1.csv", index=False)
    print(f"\nResults saved to: {output_dir / 'benchmark_results_phase1.csv'}")

if __name__ == "__main__":
    main()
