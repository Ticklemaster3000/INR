import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from metrics import MetricSuite
from loss_functions.losses import get_loss

def train(
    model: torch.nn.Module,
    dataloader,
    save_path: str,
    epochs: int = 10,
    lr: float = 1e-4,
    optimizer_cls=torch.optim.Adam,
    loss_name: str = "mse",          # NEW
    loss_params: dict = None,        # NEW
    device: torch.device = None,
    scheduler=None,
    metrics_to_use=None,
    sample_rate: int = 16000,
):
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Loss Setup (Registry-based)
    if loss_params is None:
        loss_params = {}

    loss_fn = get_loss(loss_name, loss_params).to(device)

    optimizer = optimizer_cls(model.parameters(), lr=lr)

    # Metrics
    metric_suite = MetricSuite(
        metrics_to_use=metrics_to_use,
        sample_rate=sample_rate
    )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    history = {
        "loss": [],
        "metrics": []
    }

    print("=" * 60)
    print(f"Training started on device: {device}")
    print(f"Loss function: {loss_name}")
    print("=" * 60)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics_accumulator = {}

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for batch in loop:

            if "original" not in batch:
                raise ValueError("Batch must contain 'original' key")

            targets = batch["original"].to(device)

            # Optional: if using coordinate-based INR
            coords = batch.get("coords")
            if coords is not None:
                coords = coords.to(device)
                preds = model(coords)
            else:
                preds = model(targets)

            optimizer.zero_grad()

            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Metrics
            with torch.no_grad():
                batch_metrics = metric_suite(preds, targets)

                for k, v in batch_metrics.items():
                    if k not in epoch_metrics_accumulator:
                        epoch_metrics_accumulator[k] = 0.0
                    epoch_metrics_accumulator[k] += v

        # Epoch Averaging
        epoch_loss /= len(dataloader)
        history["loss"].append(epoch_loss)

        averaged_metrics = {
            k: v / len(dataloader)
            for k, v in epoch_metrics_accumulator.items()
        }

        history["metrics"].append(averaged_metrics)

        if scheduler:
            scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Loss: {epoch_loss:.6f}")
        for k, v in averaged_metrics.items():
            print(f"{k}: {v:.4f}")

    # Save Final Model
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "loss_name": loss_name
    }

    torch.save(final_checkpoint, save_path / "final_model.pth")

    print("=" * 60)
    print("✅ Training Complete")
    print("Final model + full metric history saved.")
    print("=" * 60)

    return history

'''
save_path : str
    Directory where final model checkpoint will be saved.
    Only one file is saved:
        "final_model.pth"

epochs : int
    Number of training epochs.

lr : float
    Learning rate for optimizer.

optimizer_cls : torch.optim.Optimizer
    Optimizer class (default: Adam).

--------------------------------------------------------------
LOSS CONFIGURATION
--------------------------------------------------------------

loss_name : str
    Name of loss function from LOSS_REGISTRY.

    Available options:

    "mse"
        Mean Squared Error (time-domain)

    "l1"
        Mean Absolute Error (time-domain)

    "stft"
        Multi-Resolution STFT Loss
        (Spectral Convergence + Log-Magnitude loss)

    "hybrid"
        alpha * MSE + beta * MultiResolutionSTFTLoss

loss_params : dict
    Parameters for selected loss.

    For "hybrid":
        {
            "alpha": float   # weight for MSE
            "beta": float    # weight for STFT loss
        }

    For "stft":
        {
            "fft_sizes": list[int]
            "hop_sizes": list[int]
            "win_lengths": list[int]
        }

    Default: {}

--------------------------------------------------------------
METRIC CONFIGURATION
--------------------------------------------------------------

metrics_to_use : list[str]
    List of evaluation metrics from MetricSuite.

    Available metrics:

    "psnr"
        Peak Signal-to-Noise Ratio (dB)
        Higher is better.

    "snr"
        Signal-to-Noise Ratio (dB)
        Higher is better.

    "lsd"
        Log Spectral Distance
        Lower is better.

    "lsd_hf"
        High-Frequency Log Spectral Distance (>8kHz)
        Lower is better.

    "spectral_convergence"
        || |S_target| - |S_pred| ||_F / || |S_target| ||_F
        Lower is better.

    "envelope_distance"
        L1 difference between signal envelopes
        Lower is better.

    "pesq"
        Perceptual Evaluation of Speech Quality
        Range: [-0.5, 4.5]
        Higher is better.

    "visqol"
        Perceptual speech similarity proxy
        Higher is better.

    "rmse"
        Root Mean Squared Error
        Lower is better.

    Default metrics:
        ["psnr", "lsd", "lsd_hf",
            "spectral_convergence",
            "envelope_distance",
            "pesq"]

sample_rate : int
    Required for frequency-based and perceptual metrics.
    Used by:
        - lsd_hf
        - pesq
        - visqol

    Default: 16000 Hz

--------------------------------------------------------------
DEVICE HANDLING
--------------------------------------------------------------

device : torch.device
    If None:
        Automatically selects CUDA if available.

scheduler : optional
    Learning rate scheduler.
    scheduler.step() is called at end of each epoch.

--------------------------------------------------------------
OUTPUT
--------------------------------------------------------------

Returns:
    history : dict

    {
        "loss": [float, float, ...],  # loss per epoch
        "metrics": [
            {metric_name: value, ...},  # epoch 1
            {metric_name: value, ...},  # epoch 2
            ...
        ]
    }

--------------------------------------------------------------
SAVED CHECKPOINT (final_model.pth)
--------------------------------------------------------------

    {
        "model_state_dict": model weights,
        "optimizer_state_dict": optimizer state,
        "history": full training history,
        "loss_name": selected loss
    }
'''