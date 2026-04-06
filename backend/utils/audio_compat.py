"""
Monkey-patch torchaudio.load to use soundfile when torchcodec is broken.

torchcodec 0.11 requires libnvrtc.so.13 (not present with CUDA 12.8),
and torchaudio 2.11 only supports torchcodec as a backend.

This patch makes torchaudio.load/save work via soundfile+torch instead.

Usage: import this module BEFORE any code that calls torchaudio.load().
"""

import torch
import soundfile as sf
import numpy as np
import torchaudio
from typing import Optional, Tuple, Union, BinaryIO
import os


def _soundfile_load(
    uri: Union[BinaryIO, str, os.PathLike],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """Drop-in replacement for torchaudio.load using soundfile."""
    kwargs = dict(dtype="float32", always_2d=True)
    if frame_offset > 0:
        kwargs["start"] = frame_offset
    if num_frames is not None and num_frames > 0:
        kwargs["frames"] = num_frames
    data, sample_rate = sf.read(uri, **kwargs)
    # data shape: (frames, channels) -> torch tensor
    tensor = torch.from_numpy(data)
    if channels_first:
        tensor = tensor.T  # (channels, frames)
    return tensor, sample_rate


def _soundfile_save(
    uri: Union[BinaryIO, str, os.PathLike],
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
    compression: Optional[float] = None,
):
    """Drop-in replacement for torchaudio.save using soundfile."""
    if channels_first:
        src = src.T  # (channels, frames) -> (frames, channels)
    data = src.cpu().numpy()
    sf.write(uri, data, sample_rate)


# Apply patches
torchaudio.load = _soundfile_load
torchaudio.save = _soundfile_save
