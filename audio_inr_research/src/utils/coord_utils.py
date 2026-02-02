"""
Coordinate generation utilities for Implicit Neural Representations.
"""
import torch
import numpy as np


def make_coord(shape, ranges=None, flatten=True):
    """
    Generate coordinate grid.
    
    Args:
        shape: tuple, shape of the coordinate grid
        ranges: list of tuples, coordinate ranges for each dimension
        flatten: bool, whether to flatten the output
        
    Returns:
        coord: torch.Tensor, coordinates
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_audio_coord(length, sr=16000, normalize=True):
    """
    Generate 1D coordinates for audio signal.
    
    Args:
        length: int, length of the audio signal in samples
        sr: int, sample rate
        normalize: bool, whether to normalize coordinates to [0, 1]
        
    Returns:
        coord: torch.Tensor of shape [length, 1]
    """
    if normalize:
        coord = torch.linspace(0, 1, length).unsqueeze(-1)
    else:
        coord = torch.arange(length, dtype=torch.float32).unsqueeze(-1) / sr
    return coord


def to_pixel_samples(audio):
    """
    Convert audio waveform to coordinate-value pairs.
    
    Args:
        audio: torch.Tensor of shape [batch, length] or [length]
        
    Returns:
        coord: torch.Tensor of shape [batch, length, 1]
        values: torch.Tensor of shape [batch, length, 1]
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    batch_size, length = audio.shape
    coord = make_audio_coord(length).unsqueeze(0).expand(batch_size, -1, -1)
    values = audio.unsqueeze(-1)
    
    return coord, values


def downsample_audio(audio, factor):
    """
    Downsample audio by a given factor.
    
    Args:
        audio: torch.Tensor
        factor: int, downsampling factor
        
    Returns:
        downsampled: torch.Tensor
    """
    if factor == 1:
        return audio
    
    # Simple decimation (you can use proper anti-aliasing filter)
    return audio[..., ::factor]


def chunked_coordinates(coord, chunk_len):
    """
    Split coordinates into chunks for efficient processing.
    
    Args:
        coord: torch.Tensor [batch, length, dim]
        chunk_len: int, length of each chunk
        
    Returns:
        chunks: list of torch.Tensor
    """
    batch_size, length, dim = coord.shape
    num_chunks = (length + chunk_len - 1) // chunk_len
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_len
        end = min((i + 1) * chunk_len, length)
        chunks.append(coord[:, start:end, :])
    
    return chunks
