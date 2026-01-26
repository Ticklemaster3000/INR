# Audio INR Research: SIREN + LISA for Super-Resolution

## 📋 Project Overview

This repository implements **Implicit Neural Representations (INR)** for audio super-resolution, comparing:

1. **SIREN** (Sinusoidal Representation Networks) - Baseline
2. **LISA** (Local Implicit representation for Super resolution of Arbitrary scale) - Advanced

## 🎯 Assignment: Get Baseline Results by TODAY EVENING ⚡

### Tasks:

- ✅ Implement LISA architecture
- ✅ Add enhanced metrics (PSNR, SNR, LSD, LSD-HF, PESQ, Spectral Convergence, Envelope Distance)
- ⚡ **URGENT:** Train both SIREN and LISA models TODAY
- ⚡ **URGENT:** Generate baseline results with all metrics TODAY

**⏰ DEADLINE: TODAY EVENING - See [URGENT_TODAY.md](URGENT_TODAY.md) for fast-track plan!**

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your audio files in a directory (e.g., `data/raw/`):

```
data/
  raw/
    audio1.wav
    audio2.wav
    ...
```

Supported formats: WAV, MP3, FLAC

### 3. Train Models

**Train SIREN (Baseline):**

```bash
python train.py \
    --model siren \
    --data_dir data/raw \
    --output_dir experiments \
    --downsample_factor 4 \
    --epochs 100 \
    --batch_size 8 \
    --hidden_features 256 \
    --num_layers 5 \
    --loss hybrid \
    --sr 16000
```

**Train LISA (Advanced):**

```bash
python train.py \
    --model lisa \
    --data_dir data/raw \
    --output_dir experiments \
    --downsample_factor 4 \
    --epochs 100 \
    --batch_size 8 \
    --hidden_features 256 \
    --num_layers 5 \
    --loss hybrid \
    --sr 16000
```

### 4. Evaluate Models

```bash
python evaluate.py \
    --checkpoint experiments/siren_ds4_h256_l5/best_model.pth \
    --test_dir data/test \
    --output_dir eval_results/siren \
    --model siren \
    --downsample_factor 4 \
    --save_audio
```

## 📊 Metrics Implemented

1. **PSNR** (Peak Signal-to-Noise Ratio) - Overall quality
2. **SNR** (Signal-to-Noise Ratio) - Noise level
3. **LSD** (Log-Spectral Distance) - Frequency domain accuracy
4. **LSD-HF** (LSD High Frequency) - High frequency (>8kHz) reconstruction
5. **PESQ** (Perceptual Evaluation of Speech Quality) - Perceptual quality
6. **Spectral Convergence** - Spectral magnitude matching
7. **Envelope Distance** - Temporal envelope matching

## 🏗️ Architecture Details

### SIREN

- Sine activation functions: `sin(ω₀ * W * x + b)`
- Special weight initialization for stable training
- omega_0 = 30.0 for first layer, 30.0 for hidden layers
- Excellent for representing high-frequency details

### LISA

- **Local features** with neighborhood context (prev, current, next)
- **Positional encoding** for better coordinate representation
- **Feature interpolation** for arbitrary resolution
- **GON encoder** for latent code generation (to be added)

## 📁 Project Structure

```
audio_inr_research/
├── src/
│   ├── architectures/
│   │   └── models.py          # SIREN, LISA, MLP, MoE
│   ├── data_loaders/
│   │   └── Downsample.py      # Data preprocessing
│   ├── loss_functions/
│   │   └── losses.py          # MSE, L1, STFT, Hybrid
│   ├── metrics/
│   │   └── metrics.py         # All evaluation metrics
│   └── utils/
│       └── coord_utils.py     # Coordinate generation
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
└── requirements.txt
```

## 🔬 Experiments to Run

### Experiment 1: Baseline SIREN

```bash
python train.py --model siren --downsample_factor 4 --epochs 100
```

### Experiment 2: LISA

```bash
python train.py --model lisa --downsample_factor 4 --epochs 100
```

### Experiment 3: Different Downsampling Factors

```bash
# 2x downsampling
python train.py --model siren --downsample_factor 2 --epochs 100

# 4x downsampling
python train.py --model siren --downsample_factor 4 --epochs 100
```

## 📈 Expected Results Format

After evaluation, you'll get a summary table like:

| Metric         | SIREN       | LISA        |
| -------------- | ----------- | ----------- |
| PSNR           | 25.3 ± 2.1  | 27.8 ± 1.9  |
| SNR            | 18.5 ± 1.5  | 20.2 ± 1.3  |
| LSD            | 3.45 ± 0.5  | 2.89 ± 0.4  |
| LSD-HF         | 4.21 ± 0.6  | 3.56 ± 0.5  |
| PESQ           | 2.8 ± 0.3   | 3.2 ± 0.2   |
| Spectral Conv. | 0.45 ± 0.05 | 0.38 ± 0.04 |
| Envelope Dist. | 0.12 ± 0.02 | 0.09 ± 0.01 |

## 🐛 Troubleshooting

**CUDA Out of Memory:**

- Reduce `--batch_size` to 4 or 2
- Reduce `--chunk_len` to 8000
- Reduce `--hidden_features` to 128

**PESQ Installation Issues:**

```bash
pip install pesq --no-cache-dir
```

**No audio files found:**

- Check that files are in the correct directory
- Ensure files have .wav, .mp3, or .flac extensions

## 📝 Report Template for Tomorrow

```
## SIREN + LISA Baseline Results

**Team Member:** Sriharikrishnan TS

**Date:** [Tomorrow's date]

### Experimental Setup
- Dataset: [Your dataset name]
- Sample Rate: 16kHz
- Downsampling Factor: 4x
- Models: SIREN, LISA
- Training Epochs: 100
- Loss: Hybrid (L1 + Multi-Resolution STFT)

### Results

#### Quantitative Metrics
[Paste the summary table from evaluation]

#### Analysis
- LISA shows [X]% improvement over SIREN in PSNR
- High-frequency reconstruction (LSD-HF) improved by [Y]%
- Perceptual quality (PESQ) increased from [A] to [B]

### Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Next Steps
- [What to improve]
- [Additional experiments]
```

## 📚 References

1. **SIREN:** Implicit Neural Representations with Periodic Activation Functions (NeurIPS 2020)
2. **LISA:** Learning Continuous Representation of Audio for Arbitrary Scale Super Resolution (ICASSP 2022)
3. Original LISA: https://github.com/ml-postech/LISA

## 💡 Tips for Tomorrow's Meeting

1. **Have results ready** - Run both SIREN and LISA with same settings
2. **Visualize spectrograms** - Show before/after comparison
3. **Highlight improvements** - Focus on metrics where LISA beats SIREN
4. **Be ready to explain** - Understand why local features help
5. **Discuss limitations** - What didn't work well

---

**Good luck with your experiments! 🚀**

For questions or issues, check the code comments or reach out to the team.
