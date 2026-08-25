"""Multi-scale wavelet decomposition using a quadrature-mirror
filter bank (à la Mallat).

The layer holds a learnable low-pass filter of length ``filter_length``;
the high-pass filter is the mirror ``(-1)^n g[n]``. Each scale is
obtained by convolving with the filter pair and downsampling by 2.
The output is a list of detail coefficients, one per scale.
"""

import math

import torch


def _qmf(h: torch.Tensor) -> torch.Tensor:
    """Quadrature mirror: g[n] = (-1)^n h[N - 1 - n]."""
    n = h.shape[0]
    sign = torch.tensor(
        [(-1) ** i for i in range(n)], dtype=h.dtype
    )
    return sign * h.flip(0)


class WaveletFilterBank(torch.nn.Module):
    """
    Analogue:
        pywt / Mallat 1989 'A Theory for Multiresolution Signal Decomposition'
    Multi-scale wavelet decomposition.

    Parameters
    ----------
    n_scales : int
        Number of decomposition scales.
    filter_length : int, default 4
        Length of the (low-pass) analysis filter. Default 4 → Haar.
    learnable : bool, default True
        If True, the low-pass filter is a learnable parameter. The
        filter is renormalised to unit L1 norm at every forward to
        stay well-conditioned.
    """

    def __init__(
        self,
        n_scales: int,
        filter_length: int = 4,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if n_scales <= 0:
            raise ValueError(f"n_scales must be > 0, got {n_scales}")
        if filter_length not in (2, 4, 6, 8):
            raise ValueError("filter_length must be one of {2, 4, 6, 8}")

        self.n_scales = n_scales
        self.filter_length = filter_length
        self.learnable = learnable

        # Default to the Daubechies-D2 (Haar) low-pass for filter_length=2;
        # for longer filters start with a smoothed random filter.
        if filter_length == 2:
            init = torch.tensor([1.0 / math.sqrt(2)] * 2, dtype=torch.float32)
        else:
            init = torch.randn(filter_length, dtype=torch.float32)
            init = init / torch.sum(torch.abs(init))

        if learnable:
            self.low_pass = torch.nn.Parameter(init.clone().detach())
        else:
            self.register_buffer("low_pass", init)

    def _filters(self) -> tuple:
        h = self.low_pass / torch.sum(torch.abs(self.low_pass))
        g = _qmf(h)
        return h, g

    def _decompose_once(self, x: torch.Tensor) -> tuple:
        """Single-scale analysis: convolve with (h, g) and downsample by 2."""
        h, g = self._filters()
        # Treat 1D inputs as [B=1, C=1, T] and 2D as [B, T].
        squeeze = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze = True
        if x.ndim == 2:
            x = x.unsqueeze(1)                                   # [B, 1, T]
        h_ = h.reshape(1, 1, -1)
        g_ = g.reshape(1, 1, -1)
        approx = torch.nn.functional.conv1d(x, h_, stride=2, padding=0)
        detail = torch.nn.functional.conv1d(x, g_, stride=2, padding=0)
        if squeeze:
            approx = approx.squeeze(0).squeeze(0)
            detail = detail.squeeze(0).squeeze(0)
        else:
            approx = approx.squeeze(1)
            detail = detail.squeeze(1)
        return approx, detail

    def forward(self, x: torch.Tensor) -> list:
        """Return a list of ``n_scales`` detail-coefficient tensors,
        each shorter by a factor of 2 from the previous."""
        details = []
        for _ in range(self.n_scales):
            x, d = self._decompose_once(x)
            details.append(d)
        return details

    def extra_repr(self) -> str:
        return f"n_scales={self.n_scales}, filter_length={self.filter_length}, learnable={self.learnable}"