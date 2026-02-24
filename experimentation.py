import torch
from data_loader.datamodule import DownsampleAudioDataset
from architectures.modules.decoders import SIREN
import torch.nn as nn
import torch.optim as optim

# Load dataset
dataset = DownsampleAudioDataset("data", factors=(2, 4))

# Get one audio sample
sample = dataset[0]

waveform = sample["original"]  # shape: (T,)
waveform = waveform.float()

print("Waveform shape:", waveform.shape)

T = waveform.shape[0]

coords = torch.linspace(0, 1, T).unsqueeze(-1)  # (T, 1)
targets = waveform.unsqueeze(-1)               # (T, 1)

model = SIREN(
    in_features=1,
    out_features=1,
    hidden_features=256,
    num_layers=4,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)
coords = coords.to(device)
targets = targets.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

epochs = 200

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    preds = model(coords)

    loss = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    reconstructed = model(coords).cpu().squeeze().numpy()