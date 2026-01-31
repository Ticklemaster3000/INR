# Next Session Goals: Beating the Benchmark

**Current Status (2026-01-31):**
*   Baseline (Sinc): **29.7 dB**
*   Our Best (SIREN2+EnCodec): **26.4 dB**

**Target Goal (from Screenshot):**
We want to achieve the "instance overfit" results shown in your spreadsheet:
*   PSNR: **~30 dB to 50 dB** (Massively higher)
*   LSD: **~0.2** (We are currently ~1.3)

**Benchmark Reference:**
See `target_benchmark.png` in this folder.

**Action Plan:**
1.  **Metric Debug**: Why is our LSD so high? Are we using a different Log-Spectral Distance calc?
2.  **Overfit Intensity**: The reference might be training for 10,000+ epochs or using a different optimizer schedule (LISA paper uses L-BFGS sometimes for INRs).
3.  **Spectral Loss Weight**: We might need to tune the `alpha` parameter.

See you next time!
