"""Log-Mel spectrogram layer.

Sits on top of :class:`STFT`: a triangular mel filter bank reduces
``n_fft//2+1`` frequency bins to ``n_mels`` mel bands, then we
take ``log(.| + eps)``.
"""

import math
from typing import Optional

import torch

from .STFT import STFT


def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    f_max = f_max or sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    mel_min_f = float(_hz_to_mel(torch.tensor([f_min], dtype=torch.float32)).numpy()[0])
    mel_max_f = float(_hz_to_mel(torch.tensor([f_max], dtype=torch.float32)).numpy()[0])
    mel_points = torch.linspace(
        mel_min_f, mel_max_f, n_mels + 2
    )                                                          # [n_mels+2]
    hz_points = _mel_to_hz(mel_points)                          # [n_mels+2]
    bin_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs) # [n_freqs]
    fb = torch.zeros(n_mels, n_freqs, dtype=torch.float32)
    for m in range(1, n_mels + 1):
        f_left = float(hz_points[m - 1])
        f_center = float(hz_points[m])
        f_right = float(hz_points[m + 1])
        for k in range(n_freqs):
            f = float(bin_freqs[k])
            if f < f_left or f > f_right:
                continue
            if f <= f_center:
                fb[m - 1, k] = (f - f_left) / (f_center - f_left)
            else:
                fb[m - 1, k] = (f_right - f) / (f_right - f_center)
    return fb


class MelSpectrogram(torch.nn.Module):
    """
    Analogue:
        librosa.feature.melspectrogram
    Log-Mel spectrogram.

    Parameters
    ----------
    n_mels : int
        Number of mel bands.
    sample_rate : int
        Audio sample rate in Hz.
    win_length, hop_length, n_fft : int
        STFT parameters.
    learnable_window : bool
        Forwarded to :class:`STFT`.
    log_eps : float
        Small constant added before log for numerical stability.
    """

    def __init__(
        self,
        n_mels: int,
        sample_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        learnable_window: bool = False,
        log_eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if n_mels <= 0:
            raise ValueError(f"n_mels must be > 0, got {n_mels}")
        n_fft = n_fft or win_length
        self.n_mels = n_mels
        self.log_eps = log_eps

        self.stft = STFT(
            win_length=win_length, hop_length=hop_length,
            n_fft=n_fft, learnable_window=learnable_window,
            return_complex=True,
        )
        self.register_buffer(
            "mel_fb", _mel_filterbank(n_mels, n_fft, sample_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.stft(x)                                       # [B, F, T]
        mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-12)
        mel = torch.matmul(self.mel_fb, mag)                      # [B, n_mels, T]
        return torch.log(mel + self.log_eps)

    def extra_repr(self) -> str:
        return f"n_mels={self.n_mels}, sample_rate={self.stft.win_length}"