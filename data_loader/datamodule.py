from torch.utils.data import Dataset, DataLoader
import torch
import soundfile as sf
from pathlib import Path
from .downsampler import AudioDownsampler


class DownsampleAudioDataset(Dataset):
    def __init__(self, raw_dir, factors=(2, 4)):
        self.raw_dir = Path(raw_dir)
        self.files = list(self.raw_dir.glob("*.wav"))
        self.downsampler = AudioDownsampler(factors)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        waveform, sr = sf.read(str(file))

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        downsampled = self.downsampler.process(waveform, sr)

        return {
            "filename": file.name,
            "original": waveform.squeeze(0),
            **downsampled
        }


class DownsampleAudioLoader:
    def __init__(self, raw_dir, factors=(2, 4)):
        self.dataset = DownsampleAudioDataset(raw_dir, factors)

    def build(self, batch_size=4, shuffle=True):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)