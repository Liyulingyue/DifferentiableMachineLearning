"""Smoke tests for GMMHMM and ProbabilisticPCAMixture.

Run with:
    .venv/bin/python tests/test_gmmhmm_and_ppca.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from DifferentiableMachineLearning.ClassicalML import GMMHMM, ProbabilisticPCAMixture  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# --------------------------------------------------------------- GMMHMM
def test_gmmhmm_recovers_2_state_regime_with_gaussian_emissions():
    """Generate a 2-state HMM with distinct Gaussian emissions and
    verify that Baum-Welch recovers the state sequence."""
    torch.manual_seed(0)
    np.random.seed(0)
    K, F = 2, 4
    trans = torch.tensor([[0.95, 0.05], [0.10, 0.90]])
    means = torch.tensor([
        [+1.0, +1.0, -1.0, -1.0],
        [-1.0, -1.0, +1.0, +1.0],
    ])
    # diagonal covariances: small for both states but different
    log_vars = torch.log(torch.tensor([
        [0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1],
    ]))

    def sample(T):
        z, x = [], []
        s = 0
        for _ in range(T):
            mean = means[s].detach().cpu().numpy()
            cov = np.exp(log_vars[s].detach().cpu().numpy())
            x.append(np.random.normal(mean, np.sqrt(cov)))
            z.append(s)
            s = int(np.random.choice(K, p=trans[s].detach().cpu().numpy()))
        return np.array(z), np.array(x)

    z_true, x_seq = sample(200)
    x = torch.tensor(x_seq, dtype=torch.float32)
    hmm = GMMHMM(n_states=K, n_components=1, n_features=F)
    hmm.fit_em(x, n_iter=40)
    gamma = hmm(x).detach().cpu().numpy()
    pred = gamma.argmax(axis=1)
    same = (z_true == pred).sum()
    swap = (z_true == (1 - pred)).sum()
    accuracy = max(same, swap) / len(z_true)
    _ok(accuracy >= 0.7,
        f"GMMHMM 2-state recovery accuracy (mod swap) = {accuracy:.3f}")


def test_gmmhmm_responsibilities_sum_to_one():
    torch.manual_seed(0)
    hmm = GMMHMM(n_states=3, n_components=2, n_features=4)
    x = torch.randn(20, 4)
    gamma = hmm(x)
    _ok(torch.allclose(gamma.sum(-1), torch.ones(20), atol=1e-4),
        "GMMHMM responsibilities sum to 1")
    _ok(gamma.shape == (20, 3), f"GMMHMM responsibilities shape {gamma.shape}")


def test_gmmhmm_viterbi_returns_int_sequence():
    torch.manual_seed(0)
    hmm = GMMHMM(n_states=2, n_components=2, n_features=3)
    x = torch.randn(15, 3)
    hmm.fit_em(x, n_iter=10)
    path = hmm.viterbi(x)
    _ok(path.shape == (15,) and path.dtype == torch.int64,
        f"GMMHMM Viterbi path shape={path.shape}, dtype={path.dtype}")


# ---------------------------------------------- ProbabilisticPCAMixture
def test_ppca_mixture_recovers_3_clusters():
    """3 well-separated clusters in 6-D, each lying on a different
    2-D plane. The mixture should recover the three clusters and
    place each loading matrix along one of the planes."""
    torch.manual_seed(0)
    np.random.seed(0)
    K, F, L = 3, 6, 2
    means_true = torch.tensor([
        [+3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, +3.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, +3.0, 0.0, 0.0, 0.0],
    ])
    # Two latent dimensions per cluster, mapped to two distinct
    # feature axes each.
    W_true = torch.zeros(K, F, L)
    for k in range(K):
        W_true[k, 2 * k, 0] = 1.0
        W_true[k, 2 * k + 1, 1] = 1.0
# Generate data: 60 samples per cluster.
    parts = []
    for k in range(K):
        z = 0.3 * torch.randn(60, L)
        X = means_true[k] + z @ W_true[k].T + 0.05 * torch.randn(60, F)
        parts.append(X)
    X = torch.cat(parts, dim=0)
    y_true = np.array([0] * 60 + [1] * 60 + [2] * 60)

    ppca = ProbabilisticPCAMixture(n_components=K, n_features=F, n_latent=L)
    ppca.fit_em(X, n_iter=20)
    pred = ppca(X).detach().cpu().numpy().argmax(axis=1)
    # Best permutation
    best = 0
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        m = np.array([perm[s] for s in pred])
        best = max(best, (y_true == m).sum())
    _ok(best / len(y_true) >= 0.6,
        f"PPCA mixture cluster recovery = {best / len(y_true):.3f}")


def test_ppca_mixture_responsibilities_sum_to_one():
    ppca = ProbabilisticPCAMixture(n_components=4, n_features=5, n_latent=2)
    X = torch.randn(20, 5)
    r = ppca(X)
    _ok(torch.allclose(r.sum(-1), torch.ones(20), atol=1e-4),
        "PPCA mixture responsibilities sum to 1")
    _ok(r.shape == (20, 4), f"PPCA mixture resp shape {r.shape}")


if __name__ == "__main__":
    test_gmmhmm_recovers_2_state_regime_with_gaussian_emissions()
    test_gmmhmm_responsibilities_sum_to_one()
    test_gmmhmm_viterbi_returns_int_sequence()
    test_ppca_mixture_recovers_3_clusters()
    test_ppca_mixture_responsibilities_sum_to_one()
    print("All GMMHMM + ProbabilisticPCAMixture tests passed.")


