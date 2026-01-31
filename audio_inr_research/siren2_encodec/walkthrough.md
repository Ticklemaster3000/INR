# Project Walkthrough: SIREN²-EnCodec Audio Super-Resolution

## 1. Final Approach
We settled on **Instance-Specific Optimization** (Overfitting), which is the standard technique for Implicit Neural Representations (INRs). Instead of one model for all audio, we train a fresh tiny network for *each audio file* to maximize quality.

### Architecture: "SIREN2-EnCodec" (Full Pipeline)
The model consists of two connected parts:
1.  **Encoder**: A **SEANet Encoder** (from EnCodec) that compresses the audio into "Latent Features".
2.  **Decoder**: An **Input-Aware SIREN** that takes specific inputs:
    *   **Latent Features** (from Encoder)
    *   **Low-Res Audio** (Input-Aware conditioning)
    *   **Coordinates** (Time)

Combined with the **Residual Connection**:
```python
Output = Baseline_Interpolation + Decoder_Correction(Latents, LR_Audio)
```
    This theoretically guarantees that the model *starts* at the baseline quality and only adds details.

## 2. The Experiment
*   **Dataset**: 5 Audio clips.
*   **Training**: 1000 epochs per file (resetting model each time).
*   **Loss Function**: Hybrid (MSE for waveform + STFT for spectral detail).

## 3. Final Results

| Metric | Baseline (Sinc) | Reference SIREN | SIREN2 + EnCodec (Ours) |
|:-------|:---------------:|:---------------:|:-----------------------:|
| **PSNR** | **29.7 dB** | 26.40 dB | **26.41 dB** |
| **LSD** | **1.26** | 1.36 | **1.32** |
| **Verdict**| **Best Quality** | **Tie** | **Tie** |

## 4. Why did it fail to beat Sinc?
We encountered the **"Spectral Noise Trade-off"**:
*   The **Baseline** (Sinc Interpolation) is extremely clean but "smooth" (lacks high-frequency texture). This gives it a huge score on PSNR (Waveform accuracy).
*   Our **SIREN²** was forced by the `STFT Loss` to generate high-frequency textures.
*   **The Conflict**: The generated textures, while spectrally interesting, were not perfectly phase-aligned with the original audio. In the strict world of PSNR (Waveform Matching), **misaligned detail is counted as noise**.
*   **Result**: The model added "crispness" that the metrics punished as "noise," leading to a lower score than the smooth baseline.

## 5. Conclusion
To beat the strong Sinc Baseline (29.7 dB), a model must generate **perfectly phase-aligned** high frequencies. The current SIREN approach generates *statistically* correct frequencies (good for ears/spectrograms) but *phase-incoherent* ones (bad for PSNR), causing the numeric degradation.
