"""Linear Dynamical System (Kalman filter / smoother) re-implemented
as a ``torch.nn.Module``.

Model:

    x_t = A x_{t-1} + b + w_t,   w_t ~ N(0, Q)
    y_t = C x_t + d + v_t,       v_t ~ N(0, R)

with learnable parameters ``A, b, C, d, log_chol_Q, log_chol_R``
(stored in *log-Cholesky* parameterisation so ``Q`` and ``R`` stay
positive definite by construction). The forward pass implements
the closed-form Kalman filter + Rauch-Tung-Striebel smoother; the
EM update ``fit_em`` re-estimates the parameters in closed form
from the smoothed sufficient statistics.
"""

from typing import List, Optional, Tuple

import torch

from .utils import _to_2d


def _sym(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim == 3:
        return 0.5 * (matrix + matrix.transpose(1, 2))
    return 0.5 * (matrix + matrix.transpose(-2, -1))


class KalmanFilter(torch.nn.Module):
    """
    Analogue:
        filterpy.kalman.KalmanFilter / Kalman 1960; Rauch-Tung-Striebel 1965 smoother
    Linear Gaussian state-space model with closed-form filtering
    and smoothing.

    Parameters
    ----------
    state_dim : int
    obs_dim : int
    log_chol_init : float
        Initial log of the diagonal of the Cholesky factors of ``Q``
        and ``R`` (broadcast across dims for an isotropic init).
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        log_chol_init: float = -1.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or obs_dim <= 0:
            raise ValueError("state_dim and obs_dim must be > 0")
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.A = torch.nn.Parameter(
            torch.eye(state_dim, dtype=torch.float32)
        )
        self.b = torch.nn.Parameter(
            torch.zeros(state_dim, dtype=torch.float32)
        )
        self.C = torch.nn.Parameter(
            torch.empty(obs_dim, state_dim).uniform_(-0.5, 0.5)
        )
        self.d = torch.nn.Parameter(
            torch.zeros(obs_dim, dtype=torch.float32)
        )
        self.log_chol_Q = torch.nn.Parameter(
            torch.full((state_dim, state_dim), log_chol_init, dtype=torch.float32)
        )
        self.log_chol_R = torch.nn.Parameter(
            torch.full((obs_dim, obs_dim), log_chol_init, dtype=torch.float32)
        )
        # Initial-state mean and covariance.
        self.register_buffer(
            "x0_mean", torch.zeros(state_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "P0", torch.eye(state_dim, dtype=torch.float32)
        )

    def _Q(self) -> torch.Tensor:
        L = torch.tril(self.log_chol_Q)
        return L @ L.T + 1e-4 * torch.eye(self.state_dim, dtype=torch.float32)

    def _R(self) -> torch.Tensor:
        L = torch.tril(self.log_chol_R)
        return L @ L.T + 1e-4 * torch.eye(self.obs_dim, dtype=torch.float32)

    def _filter_smooth(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Kalman forward + RTS backward.

        Returns
        -------
        x_filt : [T, state_dim]   filtered means
        P_filt : [T, state_dim, state_dim]
        x_smooth : [T, state_dim]   smoothed means
        P_smooth : [T, state_dim, state_dim]
        """
        T = y.shape[0]
        A, b, C, d, Q, R = self.A, self.b, self.C, self.d, self._Q(), self._R()
        I = torch.eye(self.state_dim, dtype=torch.float32)

        # Forward pass
        x_pred = []
        P_pred = []
        x_f = [self.x0_mean]
        P_f = [self.P0]
        for t in range(T):
            # Predict
            if t == 0:
                x_p = self.x0_mean
                P_p = self.P0
            else:
                x_p = A @ x_f[-1] + b
                P_p = A @ P_f[-1] @ A.T + Q
            x_pred.append(x_p)
            P_pred.append(P_p)
            # Update with y[t]
            y_pred = C @ x_p + d
            S = C @ P_p @ C.T + R + 1e-3 * torch.eye(self.obs_dim, dtype=torch.float32)
            # K = P_p C^T S^{-1}
            K = torch.linalg.solve(S, (P_p @ C.T).T).T
            x_f.append(x_p + K @ (y[t] - y_pred))
            P_f.append((I - K @ C) @ P_p)
        x_f = torch.stack(x_f[1:], dim=0)                       # [T, state]
        P_f = torch.stack(P_f[1:], dim=0)                       # [T, state, state]

        # Backward (RTS) smoothing pass
        x_smooth = [None] * T
        P_smooth = [None] * T
        x_smooth[T - 1] = x_f[T - 1]
        P_smooth[T - 1] = P_f[T - 1]
        for t in range(T - 2, -1, -1):
            C_back = torch.linalg.solve(
                P_pred[t + 1] + 1e-6 * I, (P_f[t] @ A.T).T
            ).T
            x_smooth[t] = x_f[t] + C_back @ (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t] = P_f[t] + C_back @ (P_smooth[t + 1] - P_pred[t + 1]) @ C_back.T
        x_smooth = torch.stack(x_smooth, dim=0)
        P_smooth = torch.stack(P_smooth, dim=0)
        return x_f, P_f, x_smooth, P_smooth

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Return smoothed state means, shape ``[T, state_dim]``."""
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(-1)
        if y.ndim != 2:
            raise ValueError(f"Expected [T, obs_dim] input, got {y.shape}")
        if y.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected obs_dim={self.obs_dim}, got {y.shape[1]}"
            )
        _, _, x_smooth, _ = self._filter_smooth(y)
        return x_smooth

    @torch.no_grad()
    def fit_em(self, y: torch.Tensor, n_iter: int = 10) -> "KalmanFilter":
        """EM update of the LDS parameters (Ghahramani 1996)."""
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(-1)
        T = y.shape[0]
        I_s = torch.eye(self.state_dim, dtype=torch.float32)
        I_o = torch.eye(self.obs_dim, dtype=torch.float32)
        for _ in range(n_iter):
            _, P_f, x_s, P_s = self._filter_smooth(y)

            # RTS smoother gains: G_t = P_s[t] A^T (P_{t+1|t})^{-1}
            A_b = self.A
            P_pred_next = torch.zeros(
                T - 1, self.state_dim, self.state_dim, dtype=torch.float32
            )
            for t in range(T - 1):
                P_pred_next[t] = A_b @ P_f[t] @ A_b.T + self._Q()
            P_pred_inv = torch.linalg.inv(P_pred_next + 1e-6 * I_s)
            A_exp = A_b.unsqueeze(0).expand(T - 1, -1, -1)
            Gs = torch.matmul(torch.matmul(P_s[:-1], A_exp.transpose(1, 2)), P_pred_inv)

            # Sufficient statistics over t=1..T-1.
            sum_x_prev = x_s[:-1].sum(dim=0)
            sum_x_curr = x_s[1:].sum(dim=0)
            sum_xx_prev = sum(
                P_s[t - 1] + torch.outer(x_s[t - 1], x_s[t - 1]) for t in range(1, T)
            )
            sum_xx_curr = sum(
                P_s[t] + torch.outer(x_s[t], x_s[t]) for t in range(1, T)
            )
            sum_x_xprev = sum(
                Gs[t - 1] @ P_s[t] + torch.outer(x_s[t], x_s[t - 1])
                for t in range(1, T)
            )

            cov = sum_xx_prev - torch.outer(sum_x_prev, sum_x_prev) / (T - 1) + 1e-3 * I_s
            A_new = sum_x_xprev @ torch.linalg.inv(cov)
            b_new = (sum_x_curr - A_new @ sum_x_prev) / (T - 1)
            self.A.data.copy_(A_new)
            self.b.data.copy_(b_new)
            Q_new = _sym((sum_xx_curr - A_new @ sum_x_xprev.T) / (T - 1)) + 1e-2 * I_s
            eigvals_Q = torch.linalg.eigvalsh(Q_new)
            eig_min = float(eigvals_Q[0].item())
            jitter = max(1e-1, -eig_min + 1e-2) if eig_min < 0 else 1e-4
            Q_chol = torch.linalg.cholesky(Q_new + jitter * I_s)
            self.log_chol_Q.data.copy_(Q_chol)

            y_curr = y[1:]
            sum_y = y_curr.sum(dim=0)
            sum_yx = sum(torch.outer(y_curr[t], x_s[t + 1]) for t in range(T - 1))
            sum_yy = sum(torch.outer(y_curr[t], y_curr[t]) for t in range(T - 1))
            cov_C = sum_xx_curr - torch.outer(sum_x_curr, sum_x_curr) / (T - 1) + 1e-3 * I_s
            C_new = sum_yx @ torch.linalg.inv(cov_C)
            d_new = (sum_y - C_new @ sum_x_curr) / (T - 1)
            self.C.data.copy_(C_new)
            self.d.data.copy_(d_new)
            R_new = _sym((sum_yy - C_new @ sum_yx.T) / (T - 1)) + 1e-2 * I_o
            eigvals_R = torch.linalg.eigvalsh(R_new)
            eig_min_R = float(eigvals_R[0].item())
            jitter_R = max(1e-1, -eig_min_R + 1e-2) if eig_min_R < 0 else 1e-4
            R_chol = torch.linalg.cholesky(R_new + jitter_R * I_o)
            self.log_chol_R.data.copy_(R_chol)
        return self

    def extra_repr(self) -> str:
        return f"state_dim={self.state_dim}, obs_dim={self.obs_dim}"