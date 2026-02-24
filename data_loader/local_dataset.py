from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import Features, Sequence, Value
from pathlib import Path
import soundfile as sf
import torch
from tqdm import tqdm
import gc

from .downsampler import AudioDownsampler


class LocalAudioDatasetBuilder:
    """
    Modular builder for local audio → HuggingFace dataset.
    """

    def __init__(self, raw_dir, downsample_factors=(2, 4)):
        self.raw_dir = Path(raw_dir)
        self.downsampler = AudioDownsampler(downsample_factors)

    def load_files(self):
        audio_files = (
            list(self.raw_dir.glob("*.wav"))
            + list(self.raw_dir.glob("*.mp3"))
            + list(self.raw_dir.glob("*.flac"))
        )

        examples = []

        for file in tqdm(audio_files, desc="Loading audio"):
            try:
                waveform, sr = sf.read(str(file))

                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=1)

                examples.append({
                    "waveform": waveform,
                    "sampling_rate": sr,
                    "filename": file.name,
                })
            except Exception:
                continue

        return examples

    def process_examples(self, examples):
        processed = []

        for i, example in enumerate(tqdm(examples, desc="Processing")):
            waveform = torch.tensor(
                example["waveform"],
                dtype=torch.float32
            ).unsqueeze(0)

            sr = example["sampling_rate"]

            downsampled = self.downsampler.process(waveform, sr)

            out = {
                "filename": example["filename"]
            }

            for k, v in downsampled.items():
                out[f"audio_{k}"] = v

            processed.append(out)

            if (i + 1) % 10 == 0:
                gc.collect()

        return processed

    def build_hf_dataset(self, chunk_size=500):
        examples = self.load_files()
        processed = self.process_examples(examples)

        # Dynamic feature creation
        features_dict = {
            "filename": Value("string")
        }

        for key in processed[0].keys():
            if key.startswith("audio_"):
                features_dict[key] = {
                    "array": Sequence(Value("float32")),
                    "sampling_rate": Value("int32"),
                }

        features = Features(features_dict)

        datasets = []
        for i in range(0, len(processed), chunk_size):
            chunk = processed[i:i + chunk_size]
            ds_chunk = Dataset.from_list(chunk, features=features)
            datasets.append(ds_chunk)

        return DatasetDict({
            "processed": concatenate_datasets(datasets)
        })