import torch
from torch.utils.data import Dataset, DataLoader
from data_loader.datamodule import DownsampleAudioDataset
from architectures import builder
from train import train  # your master train function

# ----------------------------------
# Load original dataset
# ----------------------------------
dataset = DownsampleAudioDataset("data", factors=(2, 4))
sample = dataset[0]

waveform = sample["original"].float()
T = waveform.shape[0]

coords = torch.linspace(0, 1, T).unsqueeze(-1)  # (T, 1)
targets = waveform.unsqueeze(-1)               # (T, 1)


# ----------------------------------
# Create Coordinate Dataset
# ----------------------------------
class CoordinateDataset(Dataset):
    def __init__(self, coords, targets):
        self.coords = coords
        self.targets = targets

    def __len__(self):
        return 1  # single waveform experiment

    def __getitem__(self, idx):
        return {
            "original": self.targets,
            "coords": self.coords
        }


coord_dataset = CoordinateDataset(coords, targets)
dataloader = DataLoader(coord_dataset, batch_size=1)

model = builder.build_model({
    "type": "siren",
    "params": {
        "in_features": 1,
        "out_features": 1,
        "hidden_features": 256,
        "num_layers": 4,
    }
})

history = train(
    model=model,
    dataloader=dataloader,
    save_path="./checkpoints",
    epochs=200,
    lr=1e-4,
    metrics_to_use=["psnr", "lsd", "spectral_convergence"],
    sample_rate=16000
)