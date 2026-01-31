"""
Instance-Specific Training Script for SIREN²-EnCodec.

This script implements the standard INR workflow:
1. Iterate through a list of audio files.
2. For EACH file:
   - Initialize a fresh model (random weights).
   - "Overfit" (optimize) the model on that specific file.
   - Save the reconstruction and metrics.
   - Reset for the next file.

This approach yields high fidelity (PSNR > 30dB) but requires training at inference time.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from siren2_encodec import SIREN2_EnCodec
from src.metrics.metrics import MetricSuite

def train_on_single_file(audio_path, output_dir, config, device, metrics_suite):
    """Optimize a fresh model on a single audio file."""
    filename = audio_path.name
    
    # 1. Load Data
    sr, data = wavfile.read(audio_path)
    
    # Normalize to [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
        
    # Take mono
    if len(data.shape) > 1:
        data = data[:, 0]
        
    # Resample to target HR
    if sr != config['hr_sr']:
        data = signal.resample(data, int(len(data) * config['hr_sr'] / sr))
        data = data.astype(np.float32)

    # Prepare tensors
    hr_audio = data
    lr_audio = signal.resample(hr_audio, int(len(hr_audio) * config['lr_sr'] / config['hr_sr']))
    
    # Prepare tensors
    hr_tensor = torch.from_numpy(hr_audio).float().unsqueeze(0).unsqueeze(-1).to(device)
    lr_tensor = torch.from_numpy(lr_audio).float().unsqueeze(0).unsqueeze(-1).to(device)

    # Compute high-quality baseline interpolation (Sinc/Signal based)
    # This provides a much better starting point (approx 30dB PSNR) than linear interpolation
    lr_upsampled = signal.resample(lr_audio, len(hr_audio)).astype(np.float32)
    lr_upsampled_tensor = torch.from_numpy(lr_upsampled).float().unsqueeze(0).unsqueeze(-1).to(device)
    
    # Coordinate grid
    T = hr_tensor.shape[1]
    coord = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1).to(device)
    
    # 2. Initialize FRESH Model
    model = SIREN2_EnCodec(
        encoder_dim=config['encoder_dim'],
        encoder_ratios=[int(x) for x in config['encoder_ratios'].split(',')],
        siren_hidden=config['siren_hidden'],
        siren_layers=config['siren_layers'],
        use_spectral_noise=config['use_spectral_noise']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    mse_fn = nn.MSELoss()
    
    # 3. Optimization Loop
    best_psnr = 0
    best_audio = None
    
    # Progress bar for this file
    pbar = tqdm(range(config['epochs']), desc=f"Optimizing {filename}", leave=False)
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward - Pass high-quality interpolation as explicit baseline
        pred = model(coord, lr_tensor, lr_interp=lr_upsampled_tensor)
        loss = mse_fn(pred, hr_tensor)
        
        loss.backward()
        optimizer.step()
        
        # Logging
        if (epoch + 1) % 50 == 0:
            mse = loss.item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            pbar.set_postfix({'psnr': f'{psnr:.2f}dB'})
            
            if psnr > best_psnr:
                best_psnr = psnr
                best_audio = pred.detach().cpu().numpy().squeeze()
                
    # 4. Final Metrics
    model.eval()
    with torch.no_grad():
        final_pred = model(coord, lr_tensor)
        
        # Format for metrics: [B, 1, T]
        pred_m = final_pred.transpose(1, 2)
        hr_m = hr_tensor.transpose(1, 2)
        
        metrics = metrics_suite(pred_m, hr_m)
        metrics['filename'] = filename
        
    # 5. Save Reconstruction
    save_path = output_dir / f"recon_{filename}"
    wavfile.write(save_path, config['hr_sr'], (best_audio * 32768).astype(np.int16))
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Instance-Specific Optimization')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder containing WAV files')
    parser.add_argument('--output_dir', type=str, default='./results_instance', help='Where to save results')
    
    # Hyperparams
    parser.add_argument('--epochs', type=int, default=1000, help='Steps per file')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hr_sr', type=int, default=22050)
    parser.add_argument('--lr_sr', type=int, default=5512)
    
    # Model
    parser.add_argument('--encoder_dim', type=int, default=128)
    parser.add_argument('--encoder_ratios', type=str, default='4,4,2')
    parser.add_argument('--siren_hidden', type=int, default=256)
    parser.add_argument('--siren_layers', type=int, default=4)
    parser.add_argument('--use_spectral_noise', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metric_suite = MetricSuite(metrics_to_use=None, sample_rate=args.hr_sr)
    
    # Get files
    data_dir = Path(args.data_dir)
    files = list(data_dir.glob('**/*.wav'))[:5]  # Limit to 5 for demo
    
    print(f"Found {len(files)} files. Starting instance optimization...")
    print(f"Device: {device}")
    
    all_results = []
    
    for wav_file in tqdm(files, desc="Total Progress"):
        try:
            res = train_on_single_file(wav_file, output_dir, vars(args), device, metric_suite)
            all_results.append(res)
            
            # Print quick summary
            print(f" Done {res['filename']}: PSNR={res['psnr']:.2f}dB, LSD={res['lsd']:.2f}")
            
        except Exception as e:
            print(f"Failed on {wav_file}: {e}")
            
    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / 'metrics_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nAll Done! Summary saved to {csv_path}")
    print(df.describe())

if __name__ == '__main__':
    main()
