"""SignalProcessing submodule: classical signal-processing building
blocks re-implemented as ``torch.nn.Module`` so they can be composed
with deep networks and trained end-to-end.

Currently hosts:

* :class:`STFT`           — short-time Fourier transform with an
  optional learnable analysis window; pair with :class:`ISTFT` for
  round-trip reconstruction.
* :class:`MelSpectrogram` — log-Mel spectrogram on top of STFT with
  a fixed or learnable triangular filter bank.
* :class:`WaveletFilterBank` — multi-scale wavelet decomposition
  using a (learnable) quadrature-mirror filter bank.
"""

from .STFT import STFT, ISTFT
from .MelSpectrogram import MelSpectrogram
from .WaveletFilterBank import WaveletFilterBank

__all__ = ["STFT", "ISTFT", "MelSpectrogram", "WaveletFilterBank"]
