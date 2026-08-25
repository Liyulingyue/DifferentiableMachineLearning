"""Linear-chain Conditional Random Field (Lafferty 2001) re-implemented
as a ``torch.nn.Module``.

The CRF sits on top of an emission-score module (here a single
``nn.Linear`` for convenience, but any ``[T, n_tags]`` tensor can
be passed in). The transition matrix is a learnable parameter, and
the forward pass returns per-position marginal posteriors plus the
log partition function — both differentiable, so the whole thing
trains with standard backprop.
"""

import torch

from typing import Tuple


def _logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = torch.max(x, dim=dim, keepdim=True).values
    return (m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)


class LinearChainCRF(torch.nn.Module):
    """
    Analogue:
        sklearn-crfsuite CRF / Lafferty, McCallum, Pereira 2001
    Linear-chain CRF.

    Parameters
    ----------
    n_features : int
        Size of the per-token feature vector.
    n_tags : int
        Number of output tags.
    """

    def __init__(self, n_features: int, n_tags: int) -> None:
        super().__init__()
        if n_features <= 0 or n_tags <= 1:
            raise ValueError("n_features must be > 0 and n_tags >= 2")
        self.n_features = n_features
        self.n_tags = n_tags
        self.emitter = torch.nn.Linear(n_features, n_tags)
        self.transitions = torch.nn.Parameter(
            torch.empty(n_tags, n_tags).uniform_(-0.1, 0.1)
        )
        self.start_transitions = torch.nn.Parameter(
            torch.empty(n_tags).uniform_(-0.1, 0.1)
        )
        self.end_transitions = torch.nn.Parameter(
            torch.empty(n_tags).uniform_(-0.1, 0.1)
        )

    def _emit_scores(self, features: torch.Tensor) -> torch.Tensor:
        """``[T, n_tags]``."""
        return self.emitter(features)

    def _forward_algorithm(
        self, emit: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(log_Z, alpha, beta)`` for the marginal pass."""
        T = emit.shape[0]
        # alpha[t, k] = log-sum-exp over all paths that end in tag k at t.
        alpha = [self.start_transitions + emit[0]]              # [n_tags]
        for t in range(1, T):
            score = alpha[-1].unsqueeze(-1) + self.transitions  # [n_tags, n_tags]
            alpha.append(
                _logsumexp(score, dim=0) + emit[t]
            )
        alpha = torch.stack(alpha, dim=0)                      # [T, n_tags]
        # beta[t, k] = log-sum-exp over all paths that start at t in tag k.
        beta = [self.end_transitions]                            # [n_tags]
        for t in range(T - 2, -1, -1):
            score = (
                self.transitions
                + emit[t + 1].unsqueeze(0)
                + beta[0].unsqueeze(0)
            )                                                   # [n_tags, n_tags]
            beta.insert(0, _logsumexp(score, dim=1))
        beta = torch.stack(beta, dim=0)                        # [T, n_tags]
        log_Z = _logsumexp(alpha[T - 1] + self.end_transitions, dim=-1)
        return log_Z, alpha, beta

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-tag marginal posteriors, shape ``[T, n_tags]``."""
        emit = self._emit_scores(features)
        _, alpha, beta = self._forward_algorithm(emit)
        log_gamma = alpha + beta - _logsumexp(alpha[-1] + self.end_transitions, dim=-1)
        return torch.exp(log_gamma)

    def nll(self, features: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood for training. ``tags`` is ``[T]`` int64."""
        emit = self._emit_scores(features)
        T = emit.shape[0]
        log_Z, _, _ = self._forward_algorithm(emit)
        # Score of the gold path.
        score = self.start_transitions[tags[0]] + emit[0, tags[0]]
        for t in range(1, T):
            score = score + self.transitions[tags[t - 1], tags[t]] + emit[t, tags[t]]
        score = score + self.end_transitions[tags[T - 1]]
        return log_Z - score

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Viterbi MAP decode, returns int64 ``[T]``."""
        emit = self._emit_scores(features)
        T = emit.shape[0]
        V = [self.start_transitions + emit[0]]
        ptr = []
        for t in range(1, T):
            scores = V[-1].unsqueeze(-1) + self.transitions       # [k, l]
            best = torch.max(scores, dim=0).values                 # [l]
            idx = torch.argmax(scores, dim=0)                      # [l]
            V.append(best + emit[t])
            ptr.append(idx)
        ptr = torch.stack(ptr, dim=0) if ptr else torch.zeros(
            0, self.n_tags, dtype=torch.int64
        )
        best_last = int(torch.argmax(V[-1] + self.end_transitions).item())
        path = [best_last]
        for t in range(T - 1, 0, -1):
            prev = int(ptr[t - 1, path[-1]].item())
            path.append(prev)
        path.reverse()
        return torch.tensor(path, dtype=torch.int64)

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, n_tags={self.n_tags}"