import torch
import torch.nn as nn
import os
from pathlib import Path
from tqdm import tqdm


def train(
    model: torch.nn.Module,
    dataloader,
    save_path: str,
    epochs: int = 10,
    lr: float = 1e-4,
    optimizer_cls=torch.optim.Adam,
    loss_fn: nn.Module = None,
    device: torch.device = None,
    scheduler=None,
):
    """
    Master training function.

    Args:
        model: Any PyTorch model
        dataloader: torch DataLoader
        save_path: Directory to save checkpoints
        epochs: Number of training epochs
        lr: Learning rate
        optimizer_cls: Optimizer class
        loss_fn: Loss function (default MSE)
        device: torch.device
        scheduler: Optional LR scheduler
    """

    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    optimizer = optimizer_cls(model.parameters(), lr=lr)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    history = {"loss": []}

    print("=" * 60)
    print(f"Training started on device: {device}")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for batch in loop:
            # Assume batch contains:
            # "original" → HR audio
            # "sr_divX" → LR audio
            # Adapt if needed for your pipeline

            if "original" in batch:
                hr_audio = batch["original"].to(device)
            else:
                raise ValueError("Batch must contain 'original' key")

            # Example: use sr_div2 as low-res input
            if "sr_div2" in batch:
                lr_audio = torch.tensor(
                    batch["sr_div2"]["array"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                lr_audio = None

            optimizer.zero_grad()

            # Forward pass
            if lr_audio is not None:
                output = model(hr_audio) if lr_audio is None else model(hr_audio)
            else:
                output = model(hr_audio)

            loss = loss_fn(output, hr_audio)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss /= len(dataloader)
        history["loss"].append(epoch_loss)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1} Loss: {epoch_loss:.6f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }

        torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch+1}.pth")

    print("=" * 60)
    print("✅ Training Complete")
    print("=" * 60)

    return history