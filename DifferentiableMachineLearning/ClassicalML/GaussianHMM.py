"""Discrete-observation Gaussian-HMM-style layer.

Concretely this is a categorical-emission HMM (the title is the
shorthand used in speech-toolkit literature). The state-space is
fully observable through discrete emissions and the model is
trained with closed-form EM (Baum-Welch).

Parameters
----------
n_states : int
    Cardinality of the hidden Markov chain.
n_emissions : int
    Cardinality of the observed alphabet.

All three parameter tensors are stored as
:class:`torch.nn.Parameter` in *log* space so softmax gives the
required stochastic matrices:

* ``log_start``  : ``[n_states]``
* ``log_trans``  : ``[n_states, n_states]``
* ``log_emit``   : ``[n_states, n_emissions]``

The forward pass returns the per-timestep responsibilities
``gamma_t[k] = P(z_t = k | x_1:T)`` and is differentiable, so the
layer can act as a probabilistic sequence model in a deep pipeline.
"""

import torch

from typing import Tuple

from .utils import _to_2d


def _logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    m = torch.max(x, dim=dim, keepdim=True).values
    out = m + torch.log(
        torch.sum(torch.exp(x - m), dim=dim, keepdim=True)
    )
    if not keepdim:
        out = out.squeeze(dim)
    return out


class GaussianHMM(torch.nn.Module):
    """
    Analogue:
        hmmlearn.hmm.GaussianHMM (Baum & Welch 1970 EM)
    Categorical-emission HMM with closed-form EM.

    Parameters
    ----------
    n_states : int
    n_emissions : int
    """

    def __init__(self, n_states: int, n_emissions: int) -> None:
        super().__init__()
        if n_states <= 1 or n_emissions <= 1:
            raise ValueError("n_states and n_emissions must be >= 2")

        self.n_states = n_states
        self.n_emissions = n_emissions

        self.log_start = torch.nn.Parameter(
            torch.empty(n_states).uniform_(-0.5, 0.5)
        )
        self.log_trans = torch.nn.Parameter(
            torch.empty(n_states, n_states).uniform_(-0.5, 0.5)
        )
        self.log_emit = torch.nn.Parameter(
            torch.empty(n_states, n_emissions).uniform_(-0.5, 0.5)
        )

    # ----------------------------------------------------------------- ops
    def _start(self) -> torch.Tensor:
        return torch.nn.functional.softmax(self.log_start, dim=-1)

    def _trans(self) -> torch.Tensor:
        return torch.nn.functional.softmax(self.log_trans, dim=-1)

    def _emit(self) -> torch.Tensor:
        return torch.nn.functional.softmax(self.log_emit, dim=-1)

    def _log_emit(self) -> torch.Tensor:
        return torch.nn.functional.log_softmax(self.log_emit, dim=-1)

    # ----------------------------------------------------- forward-backward
    def _forward_backward(
        self, log_emit_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """log_emit_x: ``[T, n_states]`` — log emission probabilities
        at every timestep. Returns (gamma, xi, log_likelihood).
        """
        T = log_emit_x.shape[0]
        log_start = torch.log(self._start() + 1e-30)
        log_trans = torch.log(self._trans() + 1e-30)

        # alpha (forward)
        log_alpha = [log_start + log_emit_x[0]]               # [n_states]
        for t in range(1, T):
            prev = log_alpha[-1].unsqueeze(-1)               # [n_states, 1]
            log_alpha.append(
                _logsumexp(prev + log_trans, dim=0) + log_emit_x[t]
            )
        log_alpha = torch.stack(log_alpha, dim=0)            # [T, n_states]

        # beta (backward)
        log_beta = [torch.zeros(self.n_states, dtype=log_alpha.dtype)]
        for t in range(T - 2, -1, -1):
            post = log_beta[0]
            log_beta.insert(
                0,
                _logsumexp(
                    log_trans + post.unsqueeze(0) + log_emit_x[t + 1].unsqueeze(0),
                    dim=1,
                ),
            )
        log_beta = torch.stack(log_beta, dim=0)              # [T, n_states]

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - _logsumexp(log_gamma, dim=-1, keepdim=True)
        gamma = torch.exp(log_gamma)                          # [T, n_states]
        log_likelihood = float(
            _logsumexp(log_alpha[T - 1], dim=-1).item()
        )

        # xi: P(z_t = k, z_{t+1} = l | x). Use the same per-step log
        # form:  log xi_t[k, l] = log_alpha[t, k] + log_trans[k, l] +
        # log_emit_x[t+1, l] + log_beta[t+1, l] - log p(x).
        xi_list = []
        for t in range(T - 1):
            log_xi = (
                log_alpha[t].unsqueeze(-1)
                + log_trans
                + log_emit_x[t + 1].unsqueeze(0)
                + log_beta[t + 1].unsqueeze(0)
            )
            log_xi = log_xi - _logsumexp(log_xi, dim=-1, keepdim=True)
            xi_list.append(torch.exp(log_xi))
        xi = torch.stack(xi_list, dim=0) if xi_list else torch.zeros(
            0, self.n_states, self.n_states, dtype=log_alpha.dtype
        )
        return gamma, xi, log_likelihood

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return responsibilities ``[T, n_states]`` (differentiable)."""
        x = x.long()
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D sequence, got shape {x.shape}")
        if int(x.max()) >= self.n_emissions or int(x.min()) < 0:
            raise ValueError(
                f"Emission id out of range: min={int(x.min())}, "
                f"max={int(x.max())}, expected [0, {self.n_emissions})"
            )
        log_emit = self._log_emit()                           # [n_states, n_em]
        log_emit_x = log_emit[:, x].T                         # [T, n_states]
        gamma, _, _ = self._forward_backward(log_emit_x)
        return gamma

    def viterbi(self, x: torch.Tensor) -> torch.Tensor:
        """Most-likely state sequence (int64, shape ``[T]``)."""
        x = x.long()
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D sequence, got shape {x.shape}")
        log_emit = self._log_emit()
        log_emit_x = log_emit[:, x].T                          # [T, n_states]
        log_start = torch.log(self._start() + 1e-30)
        log_trans = torch.log(self._trans() + 1e-30)

        T = x.shape[0]
        V = [log_start + log_emit_x[0]]
        ptr = []
        for t in range(1, T):
            scores = V[-1].unsqueeze(-1) + log_trans          # [k, l]
            best = torch.max(scores, dim=0).values             # [l]
            idx = torch.argmax(scores, dim=0)                  # [l]
            V.append(best + log_emit_x[t])
            ptr.append(idx)
        ptr = torch.stack(ptr, dim=0) if ptr else torch.zeros(
            0, self.n_states, dtype=torch.int64
        )
        # Backtrack. ptr[t, k] holds the best previous state when at
        # time t we are in state k, so to recover z_{t-1} from z_t
        # we look up ptr[t, z_t].
        best_last = int(torch.argmax(V[-1]).item())
        path = [best_last]
        for t in range(T - 1, 0, -1):
            prev = int(ptr[t - 1, path[-1]].item())
            path.append(prev)
        path.reverse()
        return torch.tensor(path, dtype=torch.int64)

    # ------------------------------------------------------------------- EM
    @torch.no_grad()
    def fit_em(self, x: torch.Tensor, n_iter: int = 50) -> "GaussianHMM":
        x = x.long()
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D sequence, got shape {x.shape}")
        for _ in range(n_iter):
            log_emit = self._log_emit()
            log_emit_x = log_emit[:, x].T                     # [T, n_states]
            gamma, xi, _ = self._forward_backward(log_emit_x)
            # M-step: re-estimate start, trans, emit.
            new_start = gamma[0]
            # trans: Σ_t P(z_t=k, z_{t+1}=l | x) / Σ_t P(z_t=k | x)
            xi_sum = xi.sum(dim=0)                            # [K, L]
            gamma_sum = torch.clamp(gamma[:-1].sum(dim=0), min=1e-12)  # [K]
            new_trans = xi_sum / gamma_sum.unsqueeze(-1)
            new_trans = new_trans / torch.clamp(
                new_trans.sum(dim=1, keepdim=True), min=1e-12
            )
            # emit
            emit_count = torch.zeros(
                self.n_states, self.n_emissions, dtype=gamma.dtype
            )
            for e in range(self.n_emissions):
                mask = (x == e).to(gamma.dtype).unsqueeze(-1)
                emit_count[:, e] = (gamma * mask).sum(dim=0)
            emit_count = emit_count / torch.clamp(
                emit_count.sum(dim=1, keepdim=True), min=1e-12
            )
            self.log_start.data.copy_(torch.log(new_start + 1e-30))
            self.log_trans.data.copy_(torch.log(new_trans + 1e-30))
            self.log_emit.data.copy_(torch.log(emit_count + 1e-30))
        return self

    def extra_repr(self) -> str:
        return f"n_states={self.n_states}, n_emissions={self.n_emissions}"