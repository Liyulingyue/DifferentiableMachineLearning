"""Bayesian linear regression with mean-field variational inference,
re-implemented as a ``torch.nn.Module``.

The variational family is a diagonal Gaussian
``q(w) = prod_i N(w_i; mu_i, sigma_i^2)`` over the regression
weights, parameterised by ``mu`` and ``log_sigma`` as
``nn.Parameter``. The prior is also diagonal Gaussian
``N(0, prior_std^2 I)``. The layer minimises the ELBO loss

    loss = E_q [ log p(y | x, w) ] + KL(q(w) || p(w))

estimated by a Monte-Carlo reparameterised sample, plus the
analytical KL term. ``forward`` returns the predictive mean
under the variational posterior; ``predict`` returns the predictive
mean and (epistemic) standard deviation by averaging multiple
samples.
"""

import torch


def _gauss_kl(
    mu_q: torch.Tensor, log_sigma_q: torch.Tensor,
    mu_p: torch.Tensor, log_sigma_p: torch.Tensor
) -> torch.Tensor:
    """KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ), summed over dims."""
    var_q = torch.exp(2.0 * log_sigma_q)
    var_p = torch.exp(2.0 * log_sigma_p)
    return 0.5 * torch.sum(
        (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
        + 2.0 * log_sigma_p
        - 2.0 * log_sigma_q
    )


class BayesianLinearVI(torch.nn.Module):
    """
    Analogue:
        Blei et al. 2017 'Variational Inference: A Review for Statisticians'; sklearn-style BayesianRidge but with explicit mean-field VI
    Bayesian linear regression with mean-field VI.

    Parameters
    ----------
    n_features : int
    n_outputs : int, default 1
    prior_std : float
        Standard deviation of the diagonal Gaussian prior on ``w``.
    noise_std : float
        Standard deviation of the observation noise ``p(y | x, w)``.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int = 1,
        prior_std: float = 1.0,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        if n_features <= 0 or n_outputs <= 0:
            raise ValueError("n_features and n_outputs must be > 0")
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.prior_std = prior_std
        self.noise_std = noise_std

        self.mu = torch.nn.Parameter(
            torch.zeros(n_features, n_outputs, dtype=torch.float32)
        )
        self.log_sigma = torch.nn.Parameter(
            torch.full((n_features, n_outputs), -3.0, dtype=torch.float32)
        )

    def _sample_weights(self, n_samples: int) -> torch.Tensor:
        eps = torch.randn(n_samples, self.n_features, self.n_outputs)
        return self.mu + torch.exp(self.log_sigma) * eps

    def forward(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Predictive mean under the variational posterior."""
        w = self._sample_weights(n_samples)                 # [S, F, O]
        # x: [B, F];  x @ w -> [B, S, O];  mean over S.
        pred = torch.einsum("bf, sfo -> bso", x, w).mean(dim=0)
        return pred

    def neg_elbo(self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Negative ELBO = -E_q[log p(y | x, w)] + KL(q || p)."""
        w = self._sample_weights(n_samples)
        # Predictive mean for each sample
        pred = torch.einsum("bf, sfo -> bso", x, w)        # [B, S, O]
        # Gaussian log-likelihood, summed over data and averaged over samples
        var = torch.tensor(self.noise_std ** 2, dtype=x.dtype)
        log_lik = -0.5 * ((y.unsqueeze(1) - pred) ** 2 / var).sum(dim=[0, 2]).mean()
        log_lik = log_lik - 0.5 * (x.shape[0] * self.n_outputs) * torch.log(
            2 * 3.141592653589793 * var
        )
        kl = _gauss_kl(
            self.mu, self.log_sigma,
            torch.zeros_like(self.mu),
            torch.log(torch.full_like(self.mu, self.prior_std)),
        )
        return -log_lik / x.shape[0] + kl / x.shape[0]

    def predict(self, x: torch.Tensor, n_samples: int = 50, return_std: bool = True):
        """Posterior-predictive mean and (epistemic) std."""
        w = self._sample_weights(n_samples)                 # [S, F, O]
        pred = torch.einsum("bf, sfo -> bso", x, w)        # [B, S, O]
        mean = pred.mean(dim=1)                             # [B, O]
        if not return_std:
            return mean
        # Epistemic std: std across samples.
        std = pred.std(dim=1)                               # [B, O]
        # Add aleatoric noise.
        std = torch.sqrt(std ** 2 + self.noise_std ** 2)
        return mean, std

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, n_outputs={self.n_outputs}, "
            f"prior_std={self.prior_std}, noise_std={self.noise_std}"
        )