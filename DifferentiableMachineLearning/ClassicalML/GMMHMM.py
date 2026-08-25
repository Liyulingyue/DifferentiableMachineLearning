"""GMM-HMM (Rabiner 1989) re-implemented as a ``torch.nn.Module``.

Each hidden state ``k`` has its own Gaussian Mixture Model over the
observation space, parameterised by

* ``means``         [n_states, n_components, n_features]
* ``log_vars``      [n_states, n_components, n_features]   (diagonal)
* ``log_weights``   [n_states, n_components]
* ``log_start``     [n_states]
* ``log_trans``     [n_states, n_states]

The forward pass returns per-timestep responsibilities of the
hidden state, and ``viterbi`` returns the most-likely state
sequence. ``fit_em`` performs Baum-Welch EM with closed-form
M-step for the GMM emission parameters.
"""

import torch


def _logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    m = torch.max(x, dim=dim, keepdim=True).values
    out = m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))
    if not keepdim:
        out = out.squeeze(dim)
    return out


class GMMHMM(torch.nn.Module):
    """
    Analogue:
        hmmlearn.hmm.GMMHMM (Rabiner 1989)
    GMM-emission HMM.

    Parameters
    ----------
    n_states : int
    n_components : int
        Number of Gaussian components per state.
    n_features : int
        Observation dimension.
    """

    def __init__(self, n_states: int, n_components: int, n_features: int) -> None:
        super().__init__()
        if n_states < 2 or n_components < 1 or n_features < 1:
            raise ValueError("n_states>=2, n_components>=1, n_features>=1")
        self.n_states = n_states
        self.n_components = n_components
        self.n_features = n_features

        self.log_start = torch.nn.Parameter(
            torch.empty(n_states).uniform_(-0.5, 0.5)
        )
        self.log_trans = torch.nn.Parameter(
            torch.empty(n_states, n_states).uniform_(-0.5, 0.5)
        )
        self.means = torch.nn.Parameter(
            torch.empty(n_states, n_components, n_features).uniform_(-0.5, 0.5)
        )
        self.log_vars = torch.nn.Parameter(
            torch.zeros(n_states, n_components, n_features, dtype=torch.float32)
        )
        self.log_comp_weights = torch.nn.Parameter(
            torch.zeros(n_states, n_components, dtype=torch.float32)
        )

    def _start(self) -> torch.Tensor:
        return torch.nn.functional.softmax(self.log_start, dim=-1)

    def _trans(self) -> torch.Tensor:
        return torch.nn.functional.softmax(self.log_trans, dim=-1)

    # ---------------------------------------------- emission log-probs
    def _log_emission(self) -> torch.Tensor:
        """Return ``log p(y | state=k, mixture=c)`` evaluated at an
        empty placeholder. We instead compute log p per datum on
        the fly inside ``_log_emit_x``."""
        return None  # computed on the fly

    def _log_emit_x(self, x: torch.Tensor) -> torch.Tensor:
        """Log emission probabilities for sequence ``x: [T, n_features]``.

        Returns ``[T, n_states]`` log p(x_t | state=k).
        """
        # log p(x | state=k, comp=c) is sum_d -0.5 ((x_d - mu_cd)^2 / var_cd + log var + log 2π)
        # Vectorise: for each state k,
        #   log p(x | k) = logsumexp_c [ log_w[k, c] + sum_d log_gauss_cd(x) ].
        T = x.shape[0]
        # x: [T, F] -> [T, 1, 1, F]
        x4 = x.unsqueeze(1).unsqueeze(1)
        # means, log_vars, log_comp_weights: [K, C, F]
        diff = x4 - self.means.unsqueeze(0)
        var = torch.exp(self.log_vars.unsqueeze(0))
        # Per-component Gaussian log-density
        log_p_cd = -0.5 * (diff ** 2 / var) - 0.5 * self.log_vars.unsqueeze(0) \
            - 0.5 * torch.log(torch.tensor(2 * 3.141592653589793, dtype=x.dtype))
        # Sum over features
        log_p_c = log_p_cd.sum(dim=-1)                          # [T, K, C]
        # Add log component weights
        log_p_c = log_p_c + torch.nn.functional.log_softmax(
            self.log_comp_weights, dim=-1
        ).unsqueeze(0)
        # Logsumexp over components
        return _logsumexp(log_p_c, dim=-1)                      # [T, K]

    # ---------------------------------------------- forward-backward
    def _forward_backward(
        self, log_emit_x: torch.Tensor
    ) -> tuple:
        T = log_emit_x.shape[0]
        log_start = torch.log(self._start() + 1e-30)
        log_trans = torch.log(self._trans() + 1e-30)

        log_alpha = [log_start + log_emit_x[0]]
        for t in range(1, T):
            prev = log_alpha[-1].unsqueeze(-1)
            log_alpha.append(_logsumexp(prev + log_trans, dim=0) + log_emit_x[t])
        log_alpha = torch.stack(log_alpha, dim=0)

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
        log_beta = torch.stack(log_beta, dim=0)

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - _logsumexp(log_gamma, dim=-1, keepdim=True)
        gamma = torch.exp(log_gamma)
        log_lik = float(_logsumexp(log_alpha[T - 1], dim=-1).item())

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
        return gamma, xi, log_lik

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got {x.shape}")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected n_features={self.n_features}, got {x.shape[1]}"
            )
        log_emit_x = self._log_emit_x(x)
        gamma, _, _ = self._forward_backward(log_emit_x)
        return gamma

    def viterbi(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        log_emit_x = self._log_emit_x(x)
        T = x.shape[0]
        log_start = torch.log(self._start() + 1e-30)
        log_trans = torch.log(self._trans() + 1e-30)
        V = [log_start + log_emit_x[0]]
        ptr = []
        for t in range(1, T):
            scores = V[-1].unsqueeze(-1) + log_trans
            best = torch.max(scores, dim=0).values
            idx = torch.argmax(scores, dim=0)
            V.append(best + log_emit_x[t])
            ptr.append(idx)
        ptr = torch.stack(ptr, dim=0) if ptr else torch.zeros(
            0, self.n_states, dtype=torch.int64
        )
        best_last = int(torch.argmax(V[-1]).item())
        path = [best_last]
        for t in range(T - 1, 0, -1):
            prev = int(ptr[t - 1, path[-1]].item())
            path.append(prev)
        path.reverse()
        return torch.tensor(path, dtype=torch.int64)

    # ------------------------------------------------------------------- EM
    @torch.no_grad()
    def fit_em(self, x: torch.Tensor, n_iter: int = 30) -> "GMMHMM":
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        for _ in range(n_iter):
            log_emit_x = self._log_emit_x(x)
            gamma, xi, _ = self._forward_backward(log_emit_x)
            # M-step for start, trans (same as GaussianHMM)
            new_start = gamma[0]
            xi_sum = xi.sum(dim=0)
            gamma_sum = torch.clamp(gamma[:-1].sum(dim=0), min=1e-12)
            new_trans = xi_sum / gamma_sum.unsqueeze(-1)
            new_trans = new_trans / torch.clamp(
                new_trans.sum(dim=1, keepdim=True), min=1e-12
            )

            # M-step for the GMM emission parameters using the
            # per-state responsibilities.
            # gamma_t: [T, K], x: [T, F]
            T, K, C, F = self.n_states, self.n_states, self.n_components, self.n_features
            new_means = torch.zeros(K, C, F, dtype=x.dtype)
            new_vars = torch.zeros(K, C, F, dtype=x.dtype)
            new_comp = torch.zeros(K, C, dtype=x.dtype)
            for k in range(K):
                gk = gamma[:, k]                                  # [T]
                # gk_total / (sum_c gk_total * log_w[c]) gives a soft
                # assignment of each (t, c) to state k.
                # First compute log_emit_per_comp: [T, C]
                diff = x.unsqueeze(1) - self.means[k]             # [T, C, F]
                var = torch.exp(self.log_vars[k])
                log_p_cd = -0.5 * (diff ** 2 / var) - 0.5 * self.log_vars[k] \
                    - 0.5 * torch.log(torch.tensor(2 * 3.141592653589793, dtype=x.dtype))
                log_p_c = log_p_cd.sum(dim=-1) + torch.nn.functional.log_softmax(
                    self.log_comp_weights[k], dim=-1
                )                                                # [T, C]
                log_p_c = log_p_c - _logsumexp(log_p_c, dim=-1, keepdim=True)
                p_c = torch.exp(log_p_c)                         # [T, C]
                # weight of (t, c) into state k
                w = gk.unsqueeze(-1) * p_c                        # [T, C]
                sum_w = w.sum(dim=0) + 1e-12                     # [C]
                new_means[k] = (w.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=0) / sum_w.unsqueeze(-1)
                diff2 = (x.unsqueeze(1) - new_means[k]) ** 2
                new_vars[k] = (w.unsqueeze(-1) * diff2).sum(dim=0) / sum_w.unsqueeze(-1) + 1e-4
                new_comp[k] = sum_w / (sum_w.sum() + 1e-12)

            self.log_start.data.copy_(torch.log(new_start + 1e-30))
            self.log_trans.data.copy_(torch.log(new_trans + 1e-30))
            self.means.data.copy_(new_means)
            self.log_vars.data.copy_(torch.log(new_vars + 1e-12))
            self.log_comp_weights.data.copy_(torch.log(new_comp + 1e-30))
        return self

    def extra_repr(self) -> str:
        return (
            f"n_states={self.n_states}, n_components={self.n_components}, "
            f"n_features={self.n_features}"
        )