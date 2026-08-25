"""Sparse coding / dictionary learning re-implemented as a
``torch.nn.Module``.

The dictionary ``D: [n_atoms, n_features]`` is a learnable
:class:`torch.nn.Parameter` whose columns are unit-norm to break
the scaling degeneracy with the codes. The forward path runs
ISTA (or FISTA) for a fixed number of iterations to obtain sparse
codes ``z`` for the input batch, then the layer returns
``D.T @ z`` as the reconstruction.

The standard training objective is

    loss = (1/2) ||x - D^T z||_2^2 + lambda ||z||_1

which ``sparse_loss`` returns directly. ``fit`` wraps the loss in
an Adam update so the dictionary learns end-to-end with
differentiable ISTA.
"""

import torch


def _soft_threshold(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - t, min=0.0)


class SparseCoding(torch.nn.Module):
    """
    Analogue:
        sklearn.decomposition.DictionaryLearning (Olshausen & Field 1996; Lee & Seung 2001 ISTA/FISTA)
    Sparse coding with ISTA / FISTA encoder.

    Parameters
    ----------
    n_atoms : int
        Number of dictionary atoms (code dimension).
    n_features : int
        Data dimension.
    lmbda : float
        Sparsity penalty coefficient.
    encoder : {"ista", "fista"}
    n_iter : int
        Number of encoder iterations inside ``forward``.
    lr : float
        Step size for ISTA. Should satisfy ``lr < 2 / ||D D^T||_2``;
        the constructor uses the safe default ``1 / n_atoms``.
    """

    def __init__(
        self,
        n_atoms: int,
        n_features: int,
        lmbda: float = 0.1,
        encoder: str = "ista",
        n_iter: int = 50,
        lr: float = 0.0,
    ) -> None:
        super().__init__()
        if encoder not in {"ista", "fista"}:
            raise ValueError(f"Unknown encoder {encoder!r}")
        if n_atoms <= 0 or n_features <= 0:
            raise ValueError("n_atoms and n_features must be > 0")
        if n_iter <= 0:
            raise ValueError("n_iter must be > 0")

        self.n_atoms = n_atoms
        self.n_features = n_features
        self.lmbda = lmbda
        self.encoder = encoder
        self.n_iter = n_iter
        self.lr = lr if lr > 0 else 1.0 / n_atoms

        D = torch.randn(n_atoms, n_features, dtype=torch.float32)
        D = D / torch.clamp(torch.norm(D, dim=1, keepdim=True), min=1e-12)
        self.D = torch.nn.Parameter(D)

    def _renormalise(self) -> None:
        """Re-normalise dictionary columns after each step to break
        the scale degeneracy with the codes."""
        with torch.no_grad():
            D = self.D / torch.clamp(torch.norm(self.D, dim=1, keepdim=True), min=1e-12)
            self.D.data.copy_(D)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run ISTA / FISTA on a batch ``x: [batch, n_features]`` and
        return sparse codes ``[batch, n_atoms]``."""
        D = self.D                                          # [k, d]
        threshold = self.lmbda * self.lr
        if self.encoder == "ista":
            z = torch.zeros(x.shape[0], self.n_atoms, dtype=x.dtype)
            for _ in range(self.n_iter):
                # grad of (1/2)||x - zD||^2 wrt z is (zD - x) D^T
                residual = z @ D - x                         # [batch, d]
                grad = residual @ D.T                       # [batch, k]
                z = _soft_threshold(z - self.lr * grad, threshold)
        else:  # fista
            z = torch.zeros(x.shape[0], self.n_atoms, dtype=x.dtype)
            z_prev = z.clone()
            t = torch.tensor(1.0, dtype=x.dtype)
            for _ in range(self.n_iter):
                y = z + ((t - 1.0) / (t + 1.0)) * (z - z_prev)
                residual = y @ D - x
                grad = residual @ D.T
                z_new = _soft_threshold(y - self.lr * grad, threshold)
                z_prev = z
                z = z_new
                t = (1.0 + torch.sqrt(5.0 * t * t + 1.0)) / 2.0
        return z

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.D

    def forward(self, x: torch.Tensor) -> tuple:
        """Return (z, x_hat) where x_hat is the reconstruction."""
        z = self.encode(x)
        return z, self.reconstruct(z)

    def sparse_loss(self, x: torch.Tensor) -> torch.Tensor:
        z, x_hat = self.forward(x)
        return 0.5 * torch.mean(torch.sum((x - x_hat) ** 2, dim=1)) + \
            self.lmbda * torch.mean(torch.sum(torch.abs(z), dim=1))

    def fit(self, x: torch.Tensor, n_outer: int = 50, lr: float = 5e-2) -> "SparseCoding":
        """Train the dictionary with Adam on the sparse coding loss."""
        opt = torch.optim.Adam([self.D], lr=lr)
        for _ in range(n_outer):
            loss = self.sparse_loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            self._renormalise()
        return self

    def extra_repr(self) -> str:
        return (
            f"n_atoms={self.n_atoms}, n_features={self.n_features}, "
            f"lmbda={self.lmbda}, encoder={self.encoder!r}, n_iter={self.n_iter}"
        )