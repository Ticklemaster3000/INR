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

| Metric                     | SIREN     | LISA         | Winner           |
| -------------------------- | --------- | ------------ | ---------------- |
| **PSNR** ↑                 | 16.77 dB  | **17.14 dB** | ✅ LISA (+2.2%)  |
| **SNR** ↑                  | -1.23 dB  | **-0.86 dB** | ✅ LISA (+30%)   |
| **LSD** ↓                  | **2.13**  | 3.78         | ✅ SIREN (1.77x) |
| **Spectral Convergence** ↓ | 1.245     | **1.195**    | ✅ LISA (-4.0%)  |
| **PESQ** ↑                 | **1.197** | 1.066        | ✅ SIREN (+12%)  |

### 2x Downsampling (Easier Task)

| Metric                     | SIREN        | LISA-Enc     | Winner          |
| -------------------------- | ------------ | ------------ | --------------- |
| **PSNR** ↑                 | 28.05 dB     | **29.09 dB** | ✅ LISA (+3.7%) |
| **SNR** ↑                  | **10.04 dB** | 7.52 dB      | ✅ SIREN        |
| **LSD** ↓                  | **0.92**     | **0.92**     | 🤝 Tie          |
| **Spectral Convergence** ↓ | N/A          | **0.30**     | ✅ LISA         |

### Key Takeaways

- **LISA-Enc**: Achieves **29.09 dB PSNR @ 2x** - best performance! 🎉
- **SIREN**: Strong baseline (28.05 dB @ 2x), competitive at higher downsampling factors
- **Use Case**: LISA-Enc for best quality with arbitrary scale support, SIREN for simplicity

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
