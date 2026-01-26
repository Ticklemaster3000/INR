# File Structure Overview

This folder contains a complete, self-contained comparison between SIREN and LISA architectures for audio super-resolution.

## 📂 What's Where

### `/scripts/` - Executable Scripts

- **train.py** - Train SIREN or LISA models
- **evaluate.py** - Evaluate trained models and generate metrics
- **run_all.bat** - Batch script to run all experiments (Windows)

### `/experiments/` - Trained Models

- **siren_ds4_h256_l5/** - SIREN 4x model checkpoints
  - best_model.pth (best performing checkpoint)
  - config.json (experiment configuration)
  - results.json (training metrics)
  - checkpoint_epoch\*.pth (periodic checkpoints)
- **siren_ds2_h256_l5/** - SIREN 2x model checkpoints
- **lisa_ds4_h256_l5/** - LISA 4x model checkpoints
- **lisa_ds2_h256_l5/** - LISA 2x model checkpoints

### Root Files

- **README.md** - Main documentation
- **QUICKSTART.md** - Quick start guide for friends
- **FILE_STRUCTURE.md** - This file
- **SETUP_COMPLETE.md** - Setup summary

### Gitignored (Not Shared)

- `/results/` - Generated evaluation outputs (audio files, metrics)
- `/docs/` - Internal documentation (use main folder docs instead)

## 🔌 Plug-and-Play Design

### Core Model Code (NOT in this folder)

The actual model implementations are in:

```
../src/architectures/models.py
```

This contains:

- `SirenNet` class
- `LisaNet` class
- `GONEncoder` class
- `build_model()` function

**Why separate?** So your friends can use the same models.py for any experiment without copying code!

### Shared Dependencies

These are also in the parent src/ folder and shared:

- `src/loss_functions/` - Loss functions
- `src/metrics/` - Evaluation metrics
- `src/utils/` - Utility functions
- `src/data_loaders/` - Data loading utilities

## 🎯 How It Works

### When you run train.py or evaluate.py:

1. Script adds parent path to sys.path
2. Imports from `../src/architectures/models.py`
3. Calls `build_model(model_name='siren')` or `build_model(model_name='lisa')`
4. Everything works seamlessly!

### Adding New Models

To add a new architecture:

1. Add the model class to `../src/architectures/models.py`
2. Update `build_model()` function
3. Run with `--model your_new_model`
4. No other changes needed!

## 📊 Results Files Explained

### evaluation_results.json Structure

```json
{
  "config": {
    "checkpoint": "path/to/model.pth",
    "model": "siren",
    "downsample_factor": 4,
    ...
  },
  "per_file_results": [
    {
      "filename": "sample_0000.wav",
      "psnr": 16.77,
      "snr": -1.23,
      "lsd": 2.13,
      "spectral_convergence": 1.24,
      "envelope_distance": 0.057,
      "pesq": 1.20
    },
    ...
  ],
  "aggregate_metrics": {
    "psnr": {
      "mean": 16.77,
      "std": 0.007,
      "min": 16.76,
      "max": 16.80
    },
    ...
  }
}
```

### config.json Structure (in experiments/)

```json
{
  "model": "siren",
  "downsample_factor": 4,
  "hidden_features": 256,
  "num_layers": 5,
  "learning_rate": 0.0001,
  "epochs": 100,
  "data_dir": "data/raw",
  ...
}
```

## 🚀 Usage Examples

### Example 1: Run Pre-trained Model

```bash
cd siren_vs_lisa_comparison
python scripts/evaluate.py \
    --checkpoint experiments/siren_ds4_h256_l5/best_model.pth \
    --model siren \
    --downsample_factor 4 \
    --output_dir my_results/test1
```

### Example 2: Train Your Own

```bash
cd siren_vs_lisa_comparison
python scripts/train.py \
    --model lisa \
    --downsample_factor 2 \
    --hidden_features 512 \
    --num_layers 7 \
    --epochs 150 \
    --experiment_name my_custom_lisa
```

### Example 3: Compare Results

```bash
# Evaluate both models
python scripts/evaluate.py --checkpoint experiments/siren_ds4_h256_l5/best_model.pth --model siren --downsample_factor 4 --output_dir compare/siren
python scripts/evaluate.py --checkpoint experiments/lisa_ds4_h256_l5/best_model.pth --model lisa --downsample_factor 4 --output_dir compare/lisa

# Check the aggregate_metrics in both JSON files
```

## 🔧 Customization

### Want to modify training?

Edit `scripts/train.py`:

- Change loss functions
- Adjust learning rate schedule
- Add new metrics
- Modify data augmentation

### Want to add new metrics?

Add to `../src/metrics/metrics.py`:

- Your metrics are automatically available to all experiments
- Plug-and-play design!

### Want to try new architectures?

Add to `../src/architectures/models.py`:

- Implement your model class
- Update `build_model()` function
- Use with `--model your_model_name`

## 📝 Important Paths

### When running from comparison folder:

- Data: `../../data/raw/` (relative to scripts/)
- Models source: `../../src/` (relative to scripts/)
- Output: `./experiments/` and `./results/`

### Absolute paths (from repo root):

```
audio_inr_research/
├── src/                          # Shared code
│   └── architectures/
│       └── models.py             # Model definitions
├── data/                         # Data
│   └── raw/
└── siren_vs_lisa_comparison/     # This folder
    ├── scripts/                  # Scripts (use ../.. to access src)
    ├── experiments/              # Model checkpoints
    └── results/                  # Evaluation outputs
```

## 🎓 For Your Friends

**This folder is completely self-contained for usage**:
✅ All scripts work out of the box  
✅ All results and checkpoints are here  
✅ Full documentation included  
✅ Just need to install requirements from parent folder

**What they need from parent folder**:

- `requirements.txt` (dependencies)
- `src/` folder (model implementations)
- `data/` folder (audio files)

**They DON'T need**:

- Any other experiment folders
- Training logs or temporary files
- Other documentation files

## 🎉 Summary

This comparison folder is designed to be:

- **Shareable**: Give this folder to friends
- **Reproducible**: All configs saved
- **Extensible**: Easy to add new models
- **Educational**: Full docs included

The plug-and-play design means models are centralized in `src/architectures/models.py`, so everyone uses the same tested code!
