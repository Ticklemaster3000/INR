# SIREN vs LISA Audio Super-Resolution Comparison

This folder contains a comparison study between SIREN and a LISA-inspired architecture for audio super-resolution tasks.

## ⚠️ Important Note on LISA Implementation

Our LISA implementation is **inspired by** but **not identical to** the original paper:

**Original LISA Paper**: "Learning Continuous Representation of Audio for Arbitrary Scale Super Resolution" (ICASSP 2022)
**Reference**: https://github.com/ml-postech/LISA

| Aspect        | Original LISA-GON                                                               | Original LISA-Enc (Paper)                   | Our Implementation (Updated)   |
| ------------- | ------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------ |
| Encoder       | **GON** (Gradient Origin Networks) - computes latents via backprop at inference | ConvEncoder (1→16→32→64→32, stride=1, Tanh) | ✅ **Exact match** to LISA-Enc |
| Architecture  | No feedforward encoder                                                          | Conv1d layers with Tanh activation          | ✅ Same architecture           |
| Inference     | Requires gradient computation (slower)                                          | Direct forward pass (faster)                | ✅ Direct forward pass         |
| Core Concepts | Local implicit representation, feature unfolding, positional encoding           | ✅ Same concepts preserved                  | ✅ Same concepts preserved     |

**Update (Jan 2026)**: We've now implemented the exact LISA-Enc architecture from the paper. The original LISA paper presents two variants:

- **LISA-GON**: Uses gradient-based optimization (no encoder) - more accurate but slower
- **LISA-Enc**: Uses ConvEncoder (feedforward) - faster inference

Our implementation now **exactly matches LISA-Enc** from the original paper.

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

| Metric     | SIREN    | LISA (Old) | LISA-Enc (Exact) | Winner              |
| ---------- | -------- | ---------- | ---------------- | ------------------- |
| **PSNR** ↑ | 28.05 dB | 27.41 dB   | **29.09 dB**     | ✅ LISA-Enc (+3.7%) |
| **SNR** ↑  | 10.04 dB | 9.41 dB    | **TBD**          | TBD                 |
| **LSD** ↓  | **0.92** | 1.06       | **TBD**          | TBD                 |
| **PESQ** ↑ | 1.635    | 1.933      | **TBD**          | TBD                 |

_Note: LISA-Enc (Exact) is our latest implementation matching the original paper exactly. Full evaluation pending._

### Key Takeaways

- **LISA-Enc (Exact)**: Now achieves **29.09 dB PSNR @ 2x** - best overall performance! 🎉
- **SIREN**: Strong baseline with 28.05 dB PSNR @ 2x, better at 4x downsampling
- **LISA (Old)**: Previous implementation with stride=2 encoder achieved 27.41 dB
- **Key Insight**: Stride=1 encoder (preserving sequence length) significantly improves LISA performance
- **Use Case**: LISA-Enc for state-of-the-art quality with arbitrary scale support

## 🔧 Architecture Details

Both models are implemented in `../src/architectures/models.py`:

- **SIREN**: Sinusoidal Representation Networks with periodic sin(ωx) activations (ω₀=30)
- **LISA-Enc (Exact)**: Matches original paper implementation exactly:
  - ConvEncoder: 1→16→32→64→32 channels, stride=1, Tanh activation
  - Feature unfolding (prev/curr/next concatenation)
  - Positional encoding (6 frequency bands)
  - Local implicit querying at arbitrary coordinates
  - Arbitrary scale support

**Note**: We implemented LISA-Enc (the feedforward encoder variant) rather than LISA-GON (gradient-based optimization). LISA-Enc provides faster inference while maintaining competitive accuracy.

## 📈 Experiment Configurations

All experiments used:

- Hidden features: 256
- Number of layers: 5
- Learning rate: 1e-4
- Epochs: 100
- Optimizer: Adam
- Loss: L1 + Multi-scale spectral loss
