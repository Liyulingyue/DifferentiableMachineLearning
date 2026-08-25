"""K-Nearest-Neighbours re-implemented as a ``torch.nn.Module``.

The layer holds an *external* memory bank of stored points; queries are
answered by computing squared Euclidean distance to the bank and
returning either the ``k``-nearest neighbour indices, the average of
their values (a simple memory-augmented reconstruction), or — when a
separate ``value_bank`` is registered — the average of those values.

The memory banks are buffers (no gradient) by design: KNN is
non-parametric. The layer's only *learnable* asset is an optional
distance-temperature that scales similarities, useful when KNN is used
inside a deep network.
"""

from typing import Optional, Tuple

import torch

from .utils import _to_2d


class KNN(torch.nn.Module):
    """
    Analogue:
        sklearn.neighbors.KNeighborsClassifier / Cover & Hart 1967
    Top-k nearest-neighbour memory layer.

    Parameters
    ----------
    k : int
        Number of neighbours to retrieve.
    dim : int
        Feature dimension of the memory bank.
    temperature : float, default 1.0
        Multiplier on negative squared distance in the
        :meth:`soft_retrieval` softmax; has no effect on
        :meth:`hard_retrieval` or :meth:`forward`.
    """

    def __init__(self, k: int, dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.k = k
        self.dim = dim
        self.temperature = temperature

        # Empty buffers; populate via update_memory().
        self.register_buffer(
            "memory_bank", torch.zeros(0, dim, dtype=torch.float32)
        )
        self.register_buffer(
            "value_bank", torch.zeros(0, dim, dtype=torch.float32)
        )
        self._has_value_bank = False

    # -------------------------------------------------------------- memory
    @torch.no_grad()
    def update_memory(
        self,
        x: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> "KNN":
        """Append ``x`` (and optionally ``values``) to the memory bank."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        if values is not None:
            values = _to_2d(values)
            if values.shape[0] != x.shape[0]:
                raise ValueError(
                    f"values has {values.shape[0]} rows, expected {x.shape[0]}"
                )
            if not self._has_value_bank or self.value_bank.shape[0] == 0:
                self.value_bank = values.clone()
                self._has_value_bank = True
            else:
                if values.shape[1] != self.value_bank.shape[1]:
                    raise ValueError(
                        "values dim changed; value_bank holds "
                        f"{self.value_bank.shape[1]}, got {values.shape[1]}"
                    )
                self.value_bank = torch.cat([self.value_bank, values], dim=0)
        if self.memory_bank.shape[0] == 0:
            self.memory_bank = x.clone()
        else:
            self.memory_bank = torch.cat([self.memory_bank, x], dim=0)
        return self

    @torch.no_grad()
    def reset_memory(self) -> "KNN":
        self.memory_bank = torch.zeros(0, self.dim, dtype=torch.float32)
        self.value_bank = torch.zeros(0, self.dim, dtype=torch.float32)
        self._has_value_bank = False
        return self

    # ------------------------------------------------------------- queries
    def _distances(self, x: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean distance, shape ``[batch, bank]``."""
        x = _to_2d(x)
        if self.memory_bank.shape[0] == 0:
            raise RuntimeError(
                "KNN memory bank is empty; call update_memory() first."
            )
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        sq_x = torch.sum(x * x, dim=1, keepdim=True)
        sq_m = torch.sum(self.memory_bank * self.memory_bank, dim=1)
        cross = x @ self.memory_bank.T
        dist = torch.clamp(sq_x + sq_m - 2.0 * cross, min=0.0)
        return dist

    def hard_retrieval(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(indices [batch, k], distances [batch, k])``."""
        dist = self._distances(x)
        k = min(self.k, dist.shape[1])
        dists, idx = torch.topk(dist, k=k, dim=-1, largest=False)
        return idx, dists

    def soft_retrieval(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable soft assignment over neighbours, shape ``[batch, k]``."""
        dist = self._distances(x)
        k = min(self.k, dist.shape[1])
        dists, idx = torch.topk(dist, k=k, dim=-1, largest=False)
        weights = torch.nn.functional.softmax(-dists / self.temperature, dim=-1)
        return weights, idx

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "indices",
    ) -> torch.Tensor:
        """Dispatch helper.

        ``mode="indices"``  -> ``[batch, k]`` int64 neighbour indices.
        ``mode="average"``  -> ``[batch, dim]`` average of neighbour
        memory vectors (a self-reconstruction).
        ``mode="values"``   -> ``[batch, dim]`` average of neighbour
        ``value_bank`` rows; requires ``update_memory`` with ``values``.
        """
        idx, _ = self.hard_retrieval(x)
        if mode == "indices":
            return idx
        gathered_mem = self.memory_bank[idx]               # [batch, k, dim]
        if mode == "average":
            return torch.mean(gathered_mem, dim=1)
        if mode == "values":
            if not self._has_value_bank:
                raise RuntimeError(
                    "No value_bank registered; call update_memory(x, values)."
                )
            gathered_val = self.value_bank[idx]
            return torch.mean(gathered_val, dim=1)
        raise ValueError(
            f"Unknown mode {mode!r}; use 'indices', 'average', or 'values'."
        )

    def extra_repr(self) -> str:
        bank = self.memory_bank.shape[0]
        return f"k={self.k}, dim={self.dim}, bank={bank}, temperature={self.temperature}"