# SIREN vs LISA Audio Super-Resolution Comparison

This folder contains a comparison study between SIREN and a LISA-inspired architecture for audio super-resolution tasks.

## ⚠️ Important Note on LISA Implementation

Our LISA implementation is **inspired by** but **not identical to** the original paper:

**Original LISA Paper**: "Learning Continuous Representation of Audio for Arbitrary Scale Super Resolution" (ICASSP 2022)
**Reference**: https://github.com/ml-postech/LISA

| Aspect        | Original LISA (ml-postech)                                                      | Our Implementation                                    |
| ------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Encoder       | **GON** (Gradient Origin Networks) - computes latents via backprop at inference | **Feedforward ConvEncoder** - standard neural network |
| Inference     | Requires gradient computation (slower)                                          | Direct forward pass (faster)                          |
| Core Concepts | Local implicit representation, feature unfolding, positional encoding           | ✅ Same concepts preserved                            |

**Why the difference?** GON requires computing gradients through the decoder at inference time, which is computationally expensive and complex to implement correctly. Our simplified encoder-based approach achieves competitive results with simpler, faster inference.

## 📁 Folder Structure

```
siren_vs_lisa_comparison/
├── README.md              # This file
├── QUICKSTART.md          # Quick start guide
├── FILE_STRUCTURE.md      # Detailed structure info
├── scripts/               # Training and evaluation scripts
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── run_all.bat       # Batch script to run all experiments
└── experiments/          # Trained model checkpoints
    ├── siren_ds4_h256_l5/
    ├── siren_ds2_h256_l5/
    ├── lisa_ds4_h256_l5/
    └── lisa_ds2_h256_l5/

Note: results/ folder is gitignored (generated files, not needed for sharing)
```

## 🎯 Quick Start

### Prerequisites

```bash
pip install -r ../requirements.txt
```

### Training Both Models

```bash
# Train SIREN (4x downsampling)
python scripts/train.py --model siren --downsample_factor 4 --hidden_features 256 --num_layers 5 --lr 1e-4 --epochs 100

# Train LISA (4x downsampling)
python scripts/train.py --model lisa --downsample_factor 4 --hidden_features 256 --num_layers 5 --lr 1e-4 --epochs 100
```

### Evaluating Models

```bash
# Evaluate SIREN
python scripts/evaluate.py --checkpoint experiments/siren_ds4_h256_l5/best_model.pth --model siren --downsample_factor 4 --output_dir results/siren

# Evaluate LISA
python scripts/evaluate.py --checkpoint experiments/lisa_ds4_h256_l5/best_model.pth --model lisa --downsample_factor 4 --output_dir results/lisa
```

## 📊 Results Summary

### 4x Downsampling (Harder Task)

| Metric                     | SIREN     | LISA         | Winner           |
| -------------------------- | --------- | ------------ | ---------------- |
| **PSNR** ↑                 | 16.77 dB  | **17.14 dB** | ✅ LISA (+2.2%)  |
| **SNR** ↑                  | -1.23 dB  | **-0.86 dB** | ✅ LISA (+30%)   |
| **LSD** ↓                  | **2.13**  | 3.78         | ✅ SIREN (1.77x) |
| **Spectral Convergence** ↓ | 1.245     | **1.195**    | ✅ LISA (-4.0%)  |
| **PESQ** ↑                 | **1.197** | 1.066        | ✅ SIREN (+12%)  |

### 2x Downsampling (Easier Task)

| Metric     | SIREN        | LISA      | Winner         |
| ---------- | ------------ | --------- | -------------- |
| **PSNR** ↑ | **28.05 dB** | 27.41 dB  | ✅ SIREN       |
| **SNR** ↑  | **10.04 dB** | 9.41 dB   | ✅ SIREN       |
| **LSD** ↓  | **0.92**     | 1.06      | ✅ SIREN (15%) |
| **PESQ** ↑ | 1.635        | **1.933** | ✅ LISA (+18%) |

### Key Takeaways

- **SIREN**: Better frequency domain accuracy (LSD), stronger at 2x upsampling
- **LISA**: Better perceptual quality (PESQ), competitive spectral characteristics, arbitrary scale support
- **Use Case**: SIREN for accuracy-critical tasks, LISA for perceptual quality and flexible scaling

## 🔧 Architecture Details

Both models are implemented in `../src/architectures/models.py`:

- **SIREN**: Sinusoidal Representation Networks with periodic sin(ωx) activations (ω₀=30)
- **LISA-inspired**: Feedforward encoder + local implicit decoder with:
  - ConvEncoder for latent feature extraction
  - Feature unfolding (prev/curr/next concatenation)
  - Positional encoding (Fourier features)
  - Grid-sample based feature interpolation
  - Arbitrary scale support via coordinate queries

**Note**: Our LISA uses a simpler feedforward encoder instead of the original paper's GON (Gradient Origin Networks) approach.

## 📈 Experiment Configurations

All experiments used:

- Hidden features: 256
- Number of layers: 5
- Learning rate: 1e-4
- Epochs: 100
- Optimizer: Adam
- Loss: L1 + Multi-scale spectral loss
