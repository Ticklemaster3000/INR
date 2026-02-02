# 🚀 Quick Start Guide: SIREN vs LISA Comparison

**For friends using this repo** - Everything you need to run experiments!

---

## ⚡ Super Quick Start (5 minutes)

### 1. Setup

```bash
cd audio_inr_research
pip install -r requirements.txt
```

### 2. Get Data

```bash
python download_vits_samples.py
```

### 3. Train a Model

```bash
# SIREN (4x upsampling)
python siren_vs_lisa_comparison/scripts/train.py --model siren --downsample_factor 4

# LISA (4x upsampling)
python siren_vs_lisa_comparison/scripts/train.py --model lisa --downsample_factor 4
```

### 4. Evaluate

```bash
python siren_vs_lisa_comparison/scripts/evaluate.py \
    --checkpoint siren_vs_lisa_comparison/experiments/siren_ds4_h256_l5/best_model.pth \
    --model siren \
    --downsample_factor 4 \
    --output_dir my_results/siren
```

That's it! 🎉

---

## 🎯 What You Get

After running experiments, you'll have:

```
siren_vs_lisa_comparison/
├── experiments/          # Your trained models
│   ├── siren_ds4_*/     # SIREN checkpoints
│   └── lisa_ds4_*/      # LISA checkpoints
│
└── results/             # Evaluation outputs (gitignored)
    ├── siren/          # SIREN metrics + audio
    └── lisa/           # LISA metrics + audio

Note: Documentation is in the main audio_inr_research folder
```

---

## 🔧 Model Selection

### Use SIREN when:

- 2x upsampling tasks (28.05 dB PSNR)
- Good waveform reconstruction needed
- Simpler architecture preferred

### Use LISA-Enc when:

- 4x upsampling tasks (25.74 dB vs SIREN's 16.77 dB!) 🎉
- Perceptual quality matters (better LSD and PESQ)
- Speech enhancement
- Arbitrary scale support needed

---

## 📊 All Available Experiments

### Pre-trained Models Included

| Model    | Downsample | Config  | Test PSNR       | Location                         |
| -------- | ---------- | ------- | --------------- | -------------------------------- |
| SIREN    | 4x         | h256_l5 | 16.77 dB        | `experiments/siren_ds4_h256_l5/` |
| SIREN    | 2x         | h256_l5 | 28.05 dB        | `experiments/siren_ds2_h256_l5/` |
| LISA-Enc | 4x         | h256_l4 | **25.74 dB** 🎉 | `experiments/lisa_ds4_h256_l4/`  |
| LISA-Enc | 2x         | h256_l4 | 26.42 dB        | `experiments/lisa_ds2_h256_l4/`  |

_LISA-Enc uses exact architecture from the original paper (stride=1 ConvEncoder)_

---

## 🎮 Training Options

### Basic Training

```bash
python scripts/train.py --model siren --downsample_factor 4
```

### Custom Hyperparameters

```bash
python scripts/train.py \
    --model lisa \
    --downsample_factor 2 \
    --hidden_features 512 \     # Bigger network
    --num_layers 7 \             # Deeper network
    --lr 5e-5 \                  # Custom learning rate
    --epochs 150 \               # More training
    --experiment_name my_exp     # Custom name
```

### All Available Options

- `--model`: `siren` or `lisa` (default: `siren`)
- `--downsample_factor`: `2`, `4`, `8` (default: `4`)
- `--hidden_features`: Network width (default: `256`)
- `--num_layers`: Network depth (default: `5`)
- `--lr`: Learning rate (default: `1e-4`)
- `--epochs`: Training epochs (default: `100`)
- `--data_dir`: Audio data location (default: `data/raw`)
- `--experiment_name`: Custom experiment name

---

## 📈 Evaluation Options

### Basic Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint experiments/siren_ds4_h256_l5/best_model.pth \
    --model siren \
    --downsample_factor 4
```

### Save Audio Outputs

```bash
python scripts/evaluate.py \
    --checkpoint experiments/lisa_ds2_h256_l5/best_model.pth \
    --model lisa \
    --downsample_factor 2 \
    --save_audio true \          # Save reconstructed audio
    --output_dir my_outputs      # Custom output location
```

### Evaluate on Custom Data

```bash
python scripts/evaluate.py \
    --checkpoint my_model.pth \
    --model siren \
    --downsample_factor 4 \
    --test_dir path/to/my/audio/files
```

---

## 🔍 Understanding Results

### Metrics Explained

**Higher is Better:**

- **PSNR**: Signal quality (>25 dB = good, LISA-Enc @ 4x: 25.74 dB)
- **SNR**: Signal-to-noise (>7 dB = good for 4x)
- **PESQ**: Perceptual quality (1.0-4.5, >1.3 = acceptable)

**Lower is Better:**

- **LSD**: Frequency accuracy (<1.2 = good, <0.9 = excellent)
- **Spectral Convergence**: Spectral similarity (<0.35 = good)
- **Envelope Distance**: Amplitude envelope match (<0.005 = good)

### Quick Comparison

```bash
# Look at aggregate_metrics in the JSON files:
cat results/siren/evaluation_results.json | grep -A 20 "aggregate_metrics"
cat results/lisa/evaluation_results.json | grep -A 20 "aggregate_metrics"
```

---

## 🎵 Audio Examples

Reconstructed audio files are saved in:

```
results/
├── siren/
│   └── reconstructed/
│       ├── sample_0000.wav
│       ├── sample_0001.wav
│       └── ...
└── lisa/
    └── reconstructed/
        ├── sample_0000.wav
        └── ...
```

Listen to compare quality!

---

## 🧪 Common Experiments

### Experiment 1: Compare Architectures

```bash
# Train both models
python scripts/train.py --model siren --downsample_factor 4 --experiment_name siren_4x
python scripts/train.py --model lisa --downsample_factor 4 --experiment_name lisa_4x

# Evaluate both
python scripts/evaluate.py --checkpoint experiments/siren_4x/best_model.pth --model siren --downsample_factor 4 --output_dir results/siren_4x
python scripts/evaluate.py --checkpoint experiments/lisa_4x/best_model.pth --model lisa --downsample_factor 4 --output_dir results/lisa_4x
```

### Experiment 2: Test Different Scales

```bash
# 2x upsampling
python scripts/train.py --model siren --downsample_factor 2

# 4x upsampling
python scripts/train.py --model siren --downsample_factor 4

# 8x upsampling (very hard!)
python scripts/train.py --model siren --downsample_factor 8
```

### Experiment 3: Bigger Networks

```bash
# Standard
python scripts/train.py --model lisa --hidden_features 256 --num_layers 5

# Wide
python scripts/train.py --model lisa --hidden_features 512 --num_layers 5

# Deep
python scripts/train.py --model lisa --hidden_features 256 --num_layers 10

# Wide + Deep
python scripts/train.py --model lisa --hidden_features 512 --num_layers 10
```

---

## 🐛 Troubleshooting

### "No module named..."

```bash
pip install -r requirements.txt
```

### "CUDA out of memory"

```bash
# Use smaller batch processing or reduce model size
python scripts/train.py --model siren --hidden_features 128 --num_layers 3
```

### "Training loss not decreasing"

- Check learning rate (try 1e-5 or 5e-4)
- Verify data is correct format (16kHz, mono, .wav)
- Check model selection matches checkpoint

### "Poor audio quality"

- Train longer (--epochs 150 or 200)
- Use larger model (--hidden_features 512)
- Try different downsample_factor

---

## 💡 Pro Tips

### 1. Monitor Training

Training logs show:

- Loss every 10 epochs
- Best model is automatically saved
- Learning rate adjustments

### 2. Quick Quality Check

Listen to a few reconstructed samples first before analyzing all metrics.

### 3. Reproducibility

All configs are saved in `experiments/*/config.json` - you can rerun exact experiments!

### 4. Compare Side-by-Side

```bash
# Generate spectrograms (requires matplotlib)
import matplotlib.pyplot as plt
import librosa
import librosa.display

y_orig, sr = librosa.load('data/test/sample_0000.wav')
y_siren, sr = librosa.load('results/siren/reconstructed/sample_0000.wav')
y_lisa, sr = librosa.load('results/lisa/reconstructed/sample_0000.wav')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for ax, y, title in zip(axes, [y_orig, y_siren, y_lisa],
                        ['Original', 'SIREN', 'LISA']):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig('comparison.png')
```

---

## 📚 Learn More

- **SIREN Paper**: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- **Architecture Details**: See main repo documentation (`../SIREN_ARCHITECTURE_EXPLAINED.md`)
- **Full Code**: Check `../src/architectures/models.py`

---

## 🤝 Need Help?

1. Check the main folder documentation
2. Look at config.json in experiment folders
3. Open an issue or ask the maintainer

---

## 🎉 Have Fun!

This comparison is fully **plug-and-play**:

- No code changes needed
- Just pick a model and run
- All architectures work seamlessly

**Happy experimenting!** 🚀
