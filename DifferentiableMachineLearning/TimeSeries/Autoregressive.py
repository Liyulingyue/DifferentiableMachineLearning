"""Generic building block for AR / ARMA / FIR style models.

Implements the relation

    A(p) y(k) = B(q) u(k) + C(o) v(k)

where

    A(p) y(k) = y(k) + a_1 y(k-1) + a_2 y(k-2) + ... + a_p y(k-p)
    B(q) u(k) = b_0 u(k) + b_1 u(k-1) + ... + b_q u(k-q)
    C(o) v(k) = c_0 v(k) + c_1 v(k-1) + ... + c_o v(k-o)

The forward signature takes the *lagged* tensors directly:

* ``y``  – shape ``[batch, y_features]``   (most-recent ``p`` lags stacked)
* each entry of ``x_features`` corresponds to one exogenous stream; the
  matching tensor must have shape ``[batch, x_i_features]`` (the
  ``q_i + 1`` most-recent lags stacked)
* ``v``  – shape ``[batch, e_features]``   (the ``o + 1`` most-recent
  noise lags stacked, or zeros if not modelled)
"""

from typing import List, Optional, Sequence

import torch

from .utils import _check_shapes, _to_2d


class Autoregressive(torch.nn.Module):
    """
    Analogue:
        AR / ARMA / ARMAX in statsmodels.tsa.arima.model / Hamilton (1994) Time Series Analysis
    A single shared linear layer mixes all lagged inputs and a bias.

    Compared with the original ``PaddleAutoregressive.Autoregressive``,
    this implementation:

    * uses **one** ``Linear(in_features, 1)`` instead of N small ones,
      reducing parameters from ``sum(in_i) + N`` to ``sum(in_i) + 1``
      and producing a single coherent set of coefficients;
    * supports a leading batch dimension;
    * validates input shapes and rejects rank > 2.
    """

    def __init__(
        self,
        y_features: int = 0,
        x_features: Optional[Sequence[int]] = None,
        e_features: int = 0,
    ) -> None:
        super().__init__()
        if y_features < 0:
            raise ValueError(f"y_features must be >= 0, got {y_features}")
        if e_features < 0:
            raise ValueError(f"e_features must be >= 0, got {e_features}")
        x_features = list(x_features or [])

        self.y_features = y_features
        self.x_features = x_features
        self.e_features = e_features

        self.linear = torch.nn.Linear(
            y_features + sum(x_features) + e_features, 1, bias=True
        )
        self._widths: List[int] = []
        if y_features:
            self._widths.append(y_features)
        self._widths.extend(x_features)
        if e_features:
            self._widths.append(e_features)

        if not self._widths:
            raise ValueError(
                "Autoregressive needs at least one of "
                "y_features / x_features / e_features to be > 0."
            )

    def forward(self, *parts: torch.Tensor) -> torch.Tensor:
        expected = len(self._widths)
        if len(parts) != expected:
            raise ValueError(
                f"Expected {expected} tensor(s) matching the configured "
                f"feature widths (y={self.y_features}, x={self.x_features}, "
                f"v={self.e_features}), got {len(parts)}."
            )
        normed = [_to_2d(p) for p in parts]
        _check_shapes(
            normed,
            ["y", *[f"x[{i}]" for i in range(len(self.x_features))], "v"][
                : len(normed)
            ],
        )
        return self.linear(torch.cat(normed, dim=-1))