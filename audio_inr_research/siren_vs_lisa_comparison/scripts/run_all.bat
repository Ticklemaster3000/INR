@echo off
REM Quick start script for running SIREN and LISA experiments (Windows)

echo ================================================
echo Audio INR Research - Quick Start Script
echo ================================================

REM Check if data directory exists
if not exist "data\raw" (
    echo WARNING: data\raw directory not found!
    echo Please create it and add audio files:
    echo   mkdir data\raw
    echo   REM Copy your audio files to data\raw\
    exit /b 1
)

REM Create directories
if not exist "experiments" mkdir experiments
if not exist "eval_results" mkdir eval_results
echo Created experiment directories

REM Training parameters
set EPOCHS=100
set BATCH_SIZE=8
set HIDDEN=256
set LAYERS=5
set DS_FACTOR=4
set SR=16000

echo.
echo ================================================
echo Training Configuration:
echo ================================================
echo Epochs: %EPOCHS%
echo Batch Size: %BATCH_SIZE%
echo Hidden Features: %HIDDEN%
echo Num Layers: %LAYERS%
echo Downsample Factor: %DS_FACTOR%x
echo Sample Rate: %SR% Hz
echo.

REM Train SIREN
echo ================================================
echo 1. Training SIREN (Baseline)
echo ================================================
python train.py ^
    --model siren ^
    --data_dir data/raw ^
    --output_dir experiments ^
    --downsample_factor %DS_FACTOR% ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --hidden_features %HIDDEN% ^
    --num_layers %LAYERS% ^
    --loss hybrid ^
    --sr %SR%

if %ERRORLEVEL% NEQ 0 (
    echo SIREN training failed!
    exit /b 1
)

echo.
echo SIREN training complete!
echo.

REM Train LISA
echo ================================================
echo 2. Training LISA
echo ================================================
python train.py ^
    --model lisa ^
    --data_dir data/raw ^
    --output_dir experiments ^
    --downsample_factor %DS_FACTOR% ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --hidden_features %HIDDEN% ^
    --num_layers %LAYERS% ^
    --loss hybrid ^
    --sr %SR%

if %ERRORLEVEL% NEQ 0 (
    echo LISA training failed!
    exit /b 1
)

echo.
echo LISA training complete!
echo.

REM Evaluate models (if test data exists)
if exist "data\test" (
    echo ================================================
    echo 3. Evaluating Models
    echo ================================================
    
    echo Evaluating SIREN...
    python evaluate.py ^
        --checkpoint experiments\siren_ds%DS_FACTOR%_h%HIDDEN%_l%LAYERS%\best_model.pth ^
        --test_dir data\test ^
        --output_dir eval_results\siren ^
        --model siren ^
        --downsample_factor %DS_FACTOR% ^
        --sr %SR% ^
        --save_audio
    
    echo Evaluating LISA...
    python evaluate.py ^
        --checkpoint experiments\lisa_ds%DS_FACTOR%_h%HIDDEN%_l%LAYERS%\best_model.pth ^
        --test_dir data\test ^
        --output_dir eval_results\lisa ^
        --model lisa ^
        --downsample_factor %DS_FACTOR% ^
        --sr %SR% ^
        --save_audio
    
    echo.
    echo Evaluation complete!
    echo.
) else (
    echo WARNING: No test data found (data\test). Skipping evaluation.
    echo You can evaluate later using:
    echo   python evaluate.py --checkpoint ^<path^> --test_dir ^<path^> ...
)

echo ================================================
echo All Done!
echo ================================================
echo.
echo Results:
echo   - Training logs: experiments\
echo   - SIREN: experiments\siren_ds%DS_FACTOR%_h%HIDDEN%_l%LAYERS%\
echo   - LISA: experiments\lisa_ds%DS_FACTOR%_h%HIDDEN%_l%LAYERS%\
if exist "eval_results" (
    echo   - Evaluation: eval_results\
)
echo.
echo Next steps:
echo   1. Check results.json in each experiment directory
echo   2. Compare PSNR, LSD, and other metrics
echo   3. Prepare report for tomorrow's meeting
echo.
echo Good luck!
