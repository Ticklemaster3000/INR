from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import Features, Sequence, Value
import io
import soundfile as sf
import torchaudio
import torch
import os
from tqdm import tqdm
import traceback
import gc
import sys
from pathlib import Path

RAW_DIR = "/home/victus/Desktop/INR/audio_inr_research/data/raw"
OUT_DIR = "/home/victus/Desktop/INR/audio_inr_research/data/processed"

def dataset_from_list_with_progress(data, features, chunk_size=500):
    datasets = []

    for i in tqdm(range(0, len(data), chunk_size), desc="Building Arrow chunks"):
        chunk = data[i : i + chunk_size]
        ds_chunk = Dataset.from_list(chunk, features=features)
        datasets.append(ds_chunk)

    return concatenate_datasets(datasets)



def load_local_audio_files():
    """Load audio files from raw directory and create examples."""
    raw_path = Path(RAW_DIR)
    
    if not raw_path.exists():
        print(f"❌ Raw directory not found: {RAW_DIR}")
        return None
    
    # Find all audio files
    audio_files = list(raw_path.glob("*.wav")) + list(raw_path.glob("*.mp3")) + list(raw_path.glob("*.flac"))
    
    if not audio_files:
        print(f"❌ No audio files found in {RAW_DIR}")
        return None
    
    print(f"✓ Found {len(audio_files)} audio files in {RAW_DIR}")
    
    examples = []
    for audio_file in audio_files:
        try:
            waveform, sr = sf.read(str(audio_file))
            
            # Convert to mono if stereo
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            
            examples.append({
                "audio": {
                    "array": waveform,
                    "sampling_rate": sr,
                },
                "filename": audio_file.name,
                "path": str(audio_file),
            })
        except Exception as e:
            print(f"  ⚠️  Error loading {audio_file.name}: {e}")
            continue
    
    if not examples:
        print("❌ No audio files were successfully loaded!")
        return None
    
    print(f"✓ Successfully loaded {len(examples)} audio files")
    return examples


def process_examples(examples):
    """Process a list of examples (similar to process_split)."""
    processed = []
    
    print(f"Processing {len(examples)} examples...")
    
    for i, example in enumerate(tqdm(examples, desc="Processing examples")):
        try:
            # Get audio data
            if isinstance(example["audio"], dict) and "array" in example["audio"]:
                waveform = example["audio"]["array"]
                sr = example["audio"]["sampling_rate"]
            else:
                continue
            
            # Convert to tensor
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

            sr2 = sr // 2
            sr4 = sr // 4

            resample2 = torchaudio.transforms.Resample(sr, sr2)
            resample4 = torchaudio.transforms.Resample(sr, sr4)

            audio_div2 = resample2(waveform).squeeze(0).numpy()
            audio_div4 = resample4(waveform).squeeze(0).numpy()

            # Copy non-audio metadata
            out = {k: v for k, v in example.items() if k != "audio"}

            # Store resampled audio
            out["audio_sr_div2"] = {
                "array": audio_div2,
                "sampling_rate": sr2,
            }
            out["audio_sr_div4"] = {
                "array": audio_div4,
                "sampling_rate": sr4,
            }

            processed.append(out)
            
            # Cleanup memory
            if (i + 1) % 10 == 0:
                gc.collect()
                print(f"  Processed {i + 1} examples, memory cleanup done")
                
        except Exception as e:
            print(f"  ⚠️  Error processing example {i}: {e}")
            continue

    if not processed:
        print("❌ No examples were successfully processed!")
        return None

    print(f"✓ Successfully processed {len(processed)} examples")

    # Infer features dynamically
    example_keys = processed[0].keys()
    features_dict = {}

    for k in example_keys:
        if k in ["audio_sr_div2", "audio_sr_div4"]:
            continue
        features_dict[k] = Value("string")

    features_dict["audio_sr_div2"] = {
        "array": Sequence(Value("float32")),
        "sampling_rate": Value("int32"),
    }

    features_dict["audio_sr_div4"] = {
        "array": Sequence(Value("float32")),
        "sampling_rate": Value("int32"),
    }

    features = Features(features_dict)

    return dataset_from_list_with_progress(processed, features=features)


def main():
    try:
        print("🚀 Starting local audio dataset processing...")
        print(f"Raw directory: {RAW_DIR}")
        print(f"Output directory: {OUT_DIR}")
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"✓ Created/verified output directory: {OUT_DIR}")

        print("📥 Loading audio files from local directory...")
        examples = load_local_audio_files()
        
        if not examples:
            print("❌ Failed to load audio files!")
            sys.exit(1)

        print(f"\n{'='*50}")
        print(f"Processing {len(examples)} audio files...")
        print(f"{'='*50}")
        
        result = process_examples(examples)
        if result is None:
            print("❌ Failed to process audio files!")
            sys.exit(1)

        print(f"\n{'='*50}")
        print(f"Creating final dataset...")
        print(f"{'='*50}")
        ds_final = DatasetDict({"processed": result})
        
        print(f"💾 Saving dataset to {OUT_DIR}...")
        ds_final.save_to_disk(OUT_DIR)

        print(f"✅ Dataset saved successfully to: {OUT_DIR}")
        print(f"✅ You can find the processed data at: {OUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
