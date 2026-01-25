import os
from pathlib import Path

def setup_inr_project(root_name="audio_inr_research"):
    # Define the directory structure
    structure = [
        "configs",                    # YAML files for experiment overrides
        "data/raw",                   # Original audio files
        "data/processed",             # Resampled/pre-processed audio
        "src/architectures",          # SIREN, WIRE, Mlp models
        "src/data_loaders",           # Audio sampling and coordinate logic
        "src/loss_functions",         # Spectral and time-domain losses
        "src/metrics",                # LSD, SNR, ViSQOL scripts
        "src/utils",                  # Logging and audio visualization
        "experiments",                # Folder for saved weights and logs
        "notebooks",                  # Prototyping
        "tests",                      # Unit tests for modules
    ]

    root = Path(root_name)
    root.mkdir(exist_ok=True)

    # Create directories and __init__.py files
    for folder in structure:
        folder_path = root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure all subfolders in src are Python packages
        if folder.startswith("src"):
            (folder_path / "__init__.py").touch()
            
    # Create top-level files
    (root / "src/__init__.py").touch()
    files_to_create = [
        "train.py", 
        "evaluate.py", 
        "requirements.txt", 
        ".gitignore", 
        "README.md"
    ]
    for file in files_to_create:
        (root / file).touch()

    print(f"✅ Research repository '{root_name}' created successfully!")

if __name__ == "__main__":
    setup_inr_project()