# SIREN vs LISA Audio Super-Resolution Comparison

This folder contains a comparison study between SIREN and a LISA-inspired architecture for audio super-resolution tasks.

## ⚠️ Important Note on LISA Implementation

**Original LISA Paper**: "Learning Continuous Representation of Audio for Arbitrary Scale Super Resolution" (ICASSP 2022)  
**Reference**: https://github.com/ml-postech/LISA

We implement **LISA-Enc** - the feedforward encoder variant from the original paper:

| Component | LISA-Enc Architecture                           | Our Implementation     |
| --------- | ----------------------------------------------- | ---------------------- |
| Encoder   | ConvEncoder (1→16→32→64→32, stride=1, Tanh)     | ✅ **Exact match**     |
| Decoder   | Local implicit network with positional encoding | ✅ **Exact match**     |
| Features  | Feature unfolding, arbitrary scale support      | ✅ **Exact match**     |
| Inference | Direct forward pass (fast)                      | ✅ Direct forward pass |

**Note**: The original paper also presents LISA-GON (gradient-based optimization, no encoder), which is more accurate but significantly slower. We use LISA-Enc for practical fast inference.

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

#### Test Set Results

| Metric                     | SIREN    | LISA-Enc     | Winner           |
| -------------------------- | -------- | ------------ | ---------------- |
| **PSNR** ↑                 | 16.77 dB | **25.74 dB** | ✅ LISA (+53.5%) |
| **SNR** ↑                  | -1.23 dB | **7.74 dB**  | ✅ LISA (+729%)  |
| **LSD** ↓                  | **2.13** | 1.20         | ✅ LISA (-43.7%) |
| **Spectral Convergence** ↓ | 1.245    | **0.34**     | ✅ LISA (-72.7%) |
| **PESQ** ↑                 | 1.197    | **1.38**     | ✅ LISA (+15.3%) |

#### Training Best (Validation)

| Metric     | SIREN    | LISA-Enc     |
| ---------- | -------- | ------------ |
| **PSNR** ↑ | 16.77 dB | **30.94 dB** |

_🎉 LISA-Enc dominates 4x downsampling across all metrics!_

### 2x Downsampling (Easier Task)

#### Test Set Results

| Metric                     | SIREN        | LISA-Enc | Winner           |
| -------------------------- | ------------ | -------- | ---------------- |
| **PSNR** ↑                 | **28.05 dB** | 26.42 dB | ✅ SIREN (+6.2%) |
| **SNR** ↑                  | **10.04 dB** | 8.42 dB  | ✅ SIREN         |
| **LSD** ↓                  | 0.92         | **0.87** | ✅ LISA (-5.4%)  |
| **Spectral Convergence** ↓ | N/A          | **0.25** | ✅ LISA          |
| **PESQ** ↑                 | 1.635        | **1.77** | ✅ LISA (+8.3%)  |

#### Training Best (Validation)

| Metric     | SIREN    | LISA-Enc     |
| ---------- | -------- | ------------ |
| **PSNR** ↑ | 28.05 dB | **29.09 dB** |

_Note: Training validation PSNR differs from test set due to different data samples._

### Key Takeaways

- **SIREN**: Higher test PSNR/SNR (28.05 dB vs 26.42 dB) - better waveform reconstruction
- **LISA-Enc**: Better LSD (0.87 vs 0.92) and PESQ (1.77 vs 1.64) - better perceptual quality
- **Training vs Test**: LISA-Enc shows higher training PSNR (29.09 dB) but lower test PSNR - may benefit from more data/regularization
- **Use Case**: SIREN for accuracy, LISA-Enc for perceptual quality with arbitrary scale support

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
