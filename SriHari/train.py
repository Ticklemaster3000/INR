"""
Training script for Audio INR models (SIREN, LISA, etc.)
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from pathlib import Path

from src.architectures.models import build_model
from src.loss_functions.losses import get_loss
from src.metrics.metrics import MetricSuite
from src.utils.coord_utils import make_audio_coord, downsample_audio


class AudioSuperResDataset(Dataset):
    """Simple dataset for audio super-resolution."""
    
    def __init__(self, audio_files, downsample_factor=4, sr=16000, chunk_len=16000):
        self.audio_files = audio_files
        self.downsample_factor = downsample_factor
        self.sr = sr
        self.chunk_len = chunk_len
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio (implement based on your data format)
        import soundfile as sf
        audio, sr = sf.read(self.audio_files[idx])
        
        # Convert to tensor
        audio = torch.tensor(audio, dtype=torch.float32)
        
        # Trim or pad to chunk_len
        if len(audio) > self.chunk_len:
            start = torch.randint(0, len(audio) - self.chunk_len + 1, (1,)).item()
            audio = audio[start:start + self.chunk_len]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.chunk_len - len(audio)))
        
        # Downsample
        audio_lr = downsample_audio(audio, self.downsample_factor)
        
        # Generate coordinates
        coord_hr = make_audio_coord(len(audio))
        coord_lr = make_audio_coord(len(audio_lr))
        
        return {
            'coord': coord_hr,
            'gt': audio.unsqueeze(-1),
            'lr_audio': audio_lr.unsqueeze(-1),
            'lr_coord': coord_lr,
        }


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        coord = batch['coord'].to(device)
        gt = batch['gt'].to(device)
        lr_audio = batch['lr_audio'].to(device)
        
        # Forward pass - pass low-res audio as conditioning
        optimizer.zero_grad()
        
        # Debug: print model type
        if hasattr(model, 'has_gon_encoder'):
            print(f"DEBUG: Model has_gon_encoder = {model.has_gon_encoder}")
        else:
            print(f"DEBUG: Model does NOT have has_gon_encoder attribute")
        
        # Check if model has GON encoder (must check this FIRST)
        if hasattr(model, 'has_gon_encoder') and getattr(model, 'has_gon_encoder', False):
            # LISA with GON model - encoder extracts latent, then queries
            print("DEBUG: Using GON-LISA path")
            pred = model(coord, lr_audio)
        elif hasattr(model, 'query_features'):
            # Old LISA model - use lr_audio as simple latent
            print("DEBUG: Using OLD LISA path")
            batch_size = lr_audio.shape[0]
            lr_len = lr_audio.shape[1]
            latent = lr_audio.view(batch_size, lr_len, 1)
            pred = model.query_features(coord, latent)
        else:
            # SIREN/MLP - interpolate lr to hr and concatenate with coords
            print("DEBUG: Using SIREN/MLP path")
            lr_upsampled = torch.nn.functional.interpolate(
                lr_audio.transpose(1, 2), 
                size=coord.shape[1], 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            coord_with_lr = torch.cat([coord, lr_upsampled], dim=-1)
            pred = model(coord_with_lr)
            pred = model(coord_with_lr)
        
        # Compute loss
        loss = loss_fn(pred, gt)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, metrics, device):
    """Validate the model."""
    model.eval()
    all_metrics = {k: [] for k in metrics.active_metrics}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            coord = batch['coord'].to(device)
            gt = batch['gt'].to(device)
            lr_audio = batch['lr_audio'].to(device)
            
            # Forward pass with low-res conditioning
            if hasattr(model, 'has_gon_encoder') and model.has_gon_encoder:
                # LISA with GON encoder
                pred = model(coord, lr_audio)
            elif hasattr(model, 'query_features'):
                # Old LISA model
                batch_size = lr_audio.shape[0]
                lr_len = lr_audio.shape[1]
                latent = lr_audio.view(batch_size, lr_len, 1)
                pred = model.query_features(coord, latent)
            else:
                # SIREN/MLP
                lr_upsampled = torch.nn.functional.interpolate(
                    lr_audio.transpose(1, 2), 
                    size=coord.shape[1], 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
                coord_with_lr = torch.cat([coord, lr_upsampled], dim=-1)
                pred = model(coord_with_lr)
            
            # Compute metrics
            batch_metrics = metrics(pred, gt)
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)
    
    # Average metrics
    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if len(v) > 0}
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Audio INR models")
    parser.add_argument('--model', type=str, default='siren', choices=['mlp', 'siren', 'lisa', 'moe'])
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with audio files')
    parser.add_argument('--output_dir', type=str, default='./experiments', help='Output directory')
    parser.add_argument('--downsample_factor', type=int, default=4, help='Downsampling factor')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_features', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--loss', type=str, default='hybrid', choices=['mse', 'l1', 'stft', 'hybrid'])
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
    parser.add_argument('--chunk_len', type=int, default=16000, help='Audio chunk length')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    exp_name = f"{args.model}_ds{args.downsample_factor}_h{args.hidden_features}_l{args.num_layers}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load audio files
    data_dir = Path(args.data_dir)
    audio_files = list(data_dir.glob('*.wav')) + list(data_dir.glob('*.mp3')) + list(data_dir.glob('*.flac'))
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("Error: No audio files found!")
        return
    
    # Split train/val
    split = int(0.9 * len(audio_files))
    train_files = audio_files[:split]
    val_files = audio_files[split:]
    
    # Create datasets
    train_dataset = AudioSuperResDataset(train_files, args.downsample_factor, args.sr, args.chunk_len)
    val_dataset = AudioSuperResDataset(val_files, args.downsample_factor, args.sr, args.chunk_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model_config = {
        'in_features': 2,  # coordinate + lr_audio feature
        'out_features': 1,
        'hidden_features': args.hidden_features,
        'num_layers': args.num_layers,
    }
    
    if args.model == 'siren':
        model_config['first_omega_0'] = 30.0
        model_config['hidden_omega_0'] = 30.0
        model = build_model(args.model, model_config).to(device)
    elif args.model == 'lisa':
        # Use LISA with proper GON encoder
        from src.architectures.gon_encoder import build_lisa_with_gon
        model = build_lisa_with_gon(
            latent_dim=64,
            hidden_features=args.hidden_features,
            num_layers=args.num_layers
        ).to(device)
    else:
        model_config['latent_dim'] = 1
        model_config['feat_unfold'] = True
        model_config['in_features'] = 1
        model = build_model(args.model, model_config).to(device)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    loss_fn = get_loss(args.loss, {})
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Metrics
    metrics = MetricSuite(
        metrics_to_use=["psnr", "snr", "lsd", "lsd_hf", "spectral_convergence", "envelope_distance"],
        sample_rate=args.sr
    )
    
    # Training loop
    best_psnr = -float('inf')
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("=" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if epoch % 5 == 0:
            val_metrics = validate(model, val_loader, metrics, device)
            val_metrics_history.append({'epoch': epoch, **val_metrics})
            
            print("Validation Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Save best model
            if val_metrics.get('psnr', -float('inf')) > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                }, output_dir / 'best_model.pth')
                print(f"  → Saved best model (PSNR: {best_psnr:.4f})")
        
        # Save checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch{epoch}.pth')
        
        scheduler.step()
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_metrics': val_metrics_history,
        'best_psnr': best_psnr,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to {output_dir}")
    print(f"Best PSNR: {best_psnr:.4f}")


if __name__ == '__main__':
    main()
