"""
Evaluation script for trained Audio INR models.
"""
import os
import sys
import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np

# Add parent directory to path to access src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.architectures.models import build_model
from src.metrics.metrics import MetricSuite
from src.utils.coord_utils import make_audio_coord, downsample_audio


def load_audio(audio_path, sr=16000):
    """Load audio file."""
    audio, file_sr = sf.read(audio_path)
    
    # Resample if needed (simple method)
    if file_sr != sr:
        print(f"Warning: Resampling from {file_sr} to {sr} Hz")
        # You might want to use a proper resampling library here
    
    return torch.tensor(audio, dtype=torch.float32), sr


def evaluate_model(model, audio, downsample_factor, device):
    """
    Evaluate model on a single audio file.
    
    Args:
        model: trained model
        audio: torch.Tensor of audio signal
        downsample_factor: int
        device: torch.device
        
    Returns:
        pred: predicted high-res audio
        gt: ground truth high-res audio
    """
    model.eval()
    
    with torch.no_grad():
        # Ground truth
        gt = audio
        
        # Downsample
        lr_audio = downsample_audio(audio, downsample_factor)
        
        # Generate coordinates for super-resolution
        coord = make_audio_coord(len(gt)).unsqueeze(0).to(device)
        lr_audio_tensor = lr_audio.unsqueeze(0).unsqueeze(-1).to(device)
        
        # Predict with low-res conditioning
        if hasattr(model, 'query_features'):
            # LISA model
            batch_size = 1
            lr_len = lr_audio_tensor.shape[1]
            latent = lr_audio_tensor.view(batch_size, lr_len, 1)
            pred = model.query_features(coord, latent)
        else:
            # SIREN/MLP - interpolate and concatenate
            lr_upsampled = torch.nn.functional.interpolate(
                lr_audio_tensor.transpose(1, 2),
                size=coord.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            coord_with_lr = torch.cat([coord, lr_upsampled], dim=-1)
            pred = model(coord_with_lr)
        
        pred = pred.squeeze(0).squeeze(-1).cpu()
        
    return pred, gt


def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio INR models")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test audio files')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--model', type=str, default='siren', choices=['mlp', 'siren', 'lisa', 'moe'])
    parser.add_argument('--downsample_factor', type=int, default=4, help='Downsampling factor')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
    parser.add_argument('--save_audio', action='store_true', help='Save output audio files')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Try to load config from checkpoint, otherwise use defaults
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        model_config = {
            'in_features': 2 if args.model == 'siren' else 1,  # SIREN uses coord+lr, LISA just coord
            'out_features': 1,
            'hidden_features': saved_config.get('hidden_features', 256),
            'num_layers': saved_config.get('num_layers', 5),
        }
        print(f"Loaded config from checkpoint: hidden={model_config['hidden_features']}, layers={model_config['num_layers']}")
    else:
        # Fallback to command line or defaults
        model_config = {
            'in_features': 2 if args.model == 'siren' else 1,
            'out_features': 1,
            'hidden_features': 256,
            'num_layers': 5,
        }
        print(f"Using default config: hidden={model_config['hidden_features']}, layers={model_config['num_layers']}")
    
    if args.model == 'siren':
        model_config['first_omega_0'] = 30.0
        model_config['hidden_omega_0'] = 30.0
    elif args.model == 'lisa':
        model_config['latent_dim'] = 1  # Match training config
        model_config['feat_unfold'] = True
    
    # Create and load model
    model = build_model(args.model, model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Get test files
    test_dir = Path(args.test_dir)
    audio_files = list(test_dir.glob('*.wav')) + list(test_dir.glob('*.mp3')) + list(test_dir.glob('*.flac'))
    print(f"Found {len(audio_files)} test files")
    
    if len(audio_files) == 0:
        print("Error: No audio files found!")
        return
    
    # Setup metrics
    metrics = MetricSuite(
        metrics_to_use=["psnr", "snr", "lsd", "lsd_hf", "spectral_convergence", "envelope_distance", "pesq"],
        sample_rate=args.sr
    )
    
    # Evaluate
    all_results = []
    
    for audio_file in tqdm(audio_files, desc="Evaluating"):
        try:
            # Load audio
            audio, sr = load_audio(audio_file, args.sr)
            
            # Skip if too short
            if len(audio) < 1000:
                print(f"Skipping {audio_file.name}: too short")
                continue
            
            # Evaluate
            pred, gt = evaluate_model(model, audio, args.downsample_factor, device)
            
            # Compute metrics
            file_metrics = metrics(pred.unsqueeze(-1), gt.unsqueeze(-1))
            file_metrics['filename'] = audio_file.name
            all_results.append(file_metrics)
            
            # Save audio if requested
            if args.save_audio:
                output_path = output_dir / f"pred_{audio_file.stem}.wav"
                sf.write(output_path, pred.numpy(), args.sr)
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            continue
    
    # Aggregate results
    if len(all_results) == 0:
        print("No results generated!")
        return
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    # Average metrics
    metric_names = [k for k in all_results[0].keys() if k != 'filename']
    avg_metrics = {}
    
    for metric in metric_names:
        values = [r[metric] for r in all_results if metric in r]
        if len(values) > 0:
            avg_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
    
    # Print results
    for metric, stats in avg_metrics.items():
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Save results
    results = {
        'config': vars(args),
        'per_file_results': all_results,
        'aggregate_metrics': avg_metrics,
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")
    
    # Generate summary table
    print("\n" + "=" * 50)
    print("SUMMARY TABLE (for reporting)")
    print("=" * 50)
    print("| Metric | Value |")
    print("|--------|-------|")
    for metric, stats in avg_metrics.items():
        print(f"| {metric} | {stats['mean']:.4f} ± {stats['std']:.4f} |")


if __name__ == '__main__':
    main()
