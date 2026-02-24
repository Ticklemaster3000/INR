import torch
import torchaudio


class AudioDownsampler:
    """
    Modular audio downsampler.
    Supports arbitrary downsample factors.
    """

    def __init__(self, factors=(2, 4)):
        self.factors = factors

    def process(self, waveform: torch.Tensor, sr: int):
        """
        Args:
            waveform: (1, T)
            sr: original sampling rate

        Returns:
            dict of downsampled audio
        """
        outputs = {}

        for factor in self.factors:
            new_sr = sr // factor
            resampler = torchaudio.transforms.Resample(sr, new_sr)
            audio_resampled = resampler(waveform)

            outputs[f"sr_div{factor}"] = {
                "array": audio_resampled.squeeze(0).numpy(),
                "sampling_rate": new_sr,
            }

        return outputs