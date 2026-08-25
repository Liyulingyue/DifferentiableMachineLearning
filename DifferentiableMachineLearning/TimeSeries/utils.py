"""Utilities for the TimeSeries submodule."""

from typing import List

import torch


def _to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure ``tensor`` is 2D ``[batch, features]``."""
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 1D or 2D tensor [batch, features], got shape {tensor.shape}"
        )
    return tensor


def _check_shapes(tensors: List[torch.Tensor], names: List[str]) -> None:
    """All tensors must share the same batch size on dim 0."""
    if not tensors:
        return
    batch = tensors[0].shape[0]
    for t, n in zip(tensors, names):
        if t.shape[0] != batch:
            raise ValueError(
                f"Batch size mismatch: expected {batch} for {n!r}, got {t.shape[0]}"
            )