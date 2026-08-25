"""Smoke tests for tICA and NMF.

Run with:
    .venv/bin/python tests/test_tica_and_nmf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from DifferentiableMachineLearning.ClassicalML import tICA, NMF  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ----------------------------------------------------------------- tICA
def test_tica_recovers_slow_direction_in_2d_diffusion():
    """A 2-D process with a slow AR(1) x and an i.i.d. y should yield a
    tICA(1) projection that aligns with the x-axis."""
    torch.manual_seed(0)
    T = 5000
    # Slow: persistent AR(1) signal.
    slow = torch.zeros(T)
    slow[0] = 0.05 * torch.randn(1)
    for t in range(1, T):
        slow[t] = 0.999 * slow[t - 1] + 0.05 * torch.randn(1)
    # Fast: i.i.d. noise (zero autocorrelation at any lag > 0).
    fast = torch.randn(T)
    # Rotate by 30 deg to make sure the algorithm isn't picking
    # out a coordinate axis by accident.
    theta = float(torch.tensor(0.5).item())
    R = torch.tensor(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    X = torch.stack([slow, fast], dim=1) @ R.float().T
    tica = tICA(n_components=1, dim=2, lag=20)
    tica.fit(X)
    # The slow coordinate is the one that varies most slowly. Check
    # the autocorrelation of the projected signal is high.
    proj = tica.transform(X).squeeze().detach().cpu().numpy()
    # Compare to the *true* slow axis.
    slow_axis = R.float() @ torch.tensor([1.0, 0.0])  # rotated slow direction
    recovered = tica.components.detach().cpu().numpy().squeeze()
    cos = abs(float(recovered @ slow_axis.detach().cpu().numpy()) / (
        np.linalg.norm(recovered) * np.linalg.norm(slow_axis.detach().cpu().numpy())
    ))
    _ok(cos > 0.7, f"tICA slow direction alignment with truth = {cos:.3f}")


def test_tica_transform_shape():
    torch.manual_seed(0)
    tica = tICA(n_components=3, dim=5, lag=2)
    X = torch.randn(100, 5)
    tica.fit(X)
    proj = tica.transform(X)
    _ok(proj.shape == (100, 3), f"tICA transform shape {proj.shape}")


# ----------------------------------------------------------------- NMF
def test_nmf_reconstructs_synthetic_topics():
    """Two "topics" each of which is a sparse 10-D vector. Documents
    are random mixtures of the two topics; the learned W should
    recover something close to the two original topics."""
    torch.manual_seed(0)
    np.random.seed(0)
    n_features = 10
    topic_a = torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    topic_b = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    W_true = torch.stack([topic_a, topic_b], dim=0)        # [2, 10]
    # Random non-negative mixture coefficients.
    N = 100
    H_true = torch.abs(torch.randn(N, 2, dtype=torch.float32)) + 0.1
    X = H_true @ W_true + 0.01 * torch.abs(torch.randn(N, n_features))

    nmf = NMF(n_components=2, n_features=n_features, init="nndsvd")
    nmf.fit(X, n_iter=300)
    rec = nmf.reconstruct(nmf.H)
    rel_err = float(torch.mean(torch.sum((X - rec) ** 2, dim=1)) /
                    torch.mean(torch.sum(X ** 2, dim=1)))
    _ok(rel_err < 0.1, f"NMF rel reconstruction err = {rel_err:.4f}")

    # Check W recovers something close to the two original topics
    # (up to permutation).
    W = nmf.W.detach().detach().cpu().numpy()
    cos_a_direct = float(W[0] @ W_true[0].detach().cpu().numpy()) / (
        np.linalg.norm(W[0]) * np.linalg.norm(W_true[0].detach().cpu().numpy())
    )
    cos_b_direct = float(W[1] @ W_true[1].detach().cpu().numpy()) / (
        np.linalg.norm(W[1]) * np.linalg.norm(W_true[1].detach().cpu().numpy())
    )
    cos_a_swap = float(W[0] @ W_true[1].detach().cpu().numpy()) / (
        np.linalg.norm(W[0]) * np.linalg.norm(W_true[1].detach().cpu().numpy())
    )
    cos_b_swap = float(W[1] @ W_true[0].detach().cpu().numpy()) / (
        np.linalg.norm(W[1]) * np.linalg.norm(W_true[0].detach().cpu().numpy())
    )
    best = max(min(cos_a_direct, cos_b_direct), min(cos_a_swap, cos_b_swap))
    _ok(best > 0.85, f"NMF topic recovery (best abs cos) = {best:.3f}")


def test_nmf_w_h_stay_nonnegative():
    nmf = NMF(n_components=3, n_features=5)
    X = torch.abs(torch.randn(20, 5)) + 0.1
    nmf.fit(X, n_iter=50)
    _ok((nmf.W >= -1e-6).all().item(), "NMF W stays non-negative")
    _ok((nmf.H >= -1e-6).all().item(), "NMF H stays non-negative")


def test_nmf_transform_shape():
    nmf = NMF(n_components=3, n_features=6)
    X = torch.abs(torch.randn(10, 6)) + 0.1
    nmf.fit(X, n_iter=20)
    H_new = nmf.transform(X)
    _ok(H_new.shape == (10, 3), f"transform H shape {H_new.shape}")


if __name__ == "__main__":
    test_tica_recovers_slow_direction_in_2d_diffusion()
    test_tica_transform_shape()
    test_nmf_reconstructs_synthetic_topics()
    test_nmf_w_h_stay_nonnegative()
    test_nmf_transform_shape()
    print("All tICA + NMF tests passed.")

