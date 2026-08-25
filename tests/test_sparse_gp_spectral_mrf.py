"""Smoke tests for SparseGP, SpectralClustering, RBM, IsingModel.

Run with:
    .venv/bin/python tests/test_sparse_gp_spectral_mrf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from DifferentiableMachineLearning.ClassicalML import (  # noqa: E402
    SparseGP, SpectralClustering, RBM, IsingModel,
)


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ----------------------------------------------------------------- SparseGP
def test_sparse_gp_fits_sine():
    torch.manual_seed(0)
    n = 100
    x = torch.linspace(-3.0, 3.0, n).unsqueeze(-1)
    y = torch.sin(x).squeeze(-1) + 0.1 * torch.randn(n)
    sg = SparseGP(n_inducing=20, dim=1, n_train=n, kernel="rbf")
    sg.fit_init(x, y)
    initial = float(sg.elbo().item())
    sg.fit(n_outer=200, lr=1e-2)
    final = float(sg.elbo().item())
    _ok(final < initial, f"SparseGP neg-ELBO {initial:.3f} -> {final:.3f}")


def test_sparse_gp_predictive_std_grows_outside_training_data():
    torch.manual_seed(0)
    n = 50
    x = torch.linspace(-3.0, 3.0, n).unsqueeze(-1)
    y = torch.sin(x).squeeze(-1) + 0.05 * torch.randn(n)
    sg = SparseGP(n_inducing=15, dim=1, n_train=n, kernel="rbf")
    sg.fit_init(x, y)
    sg.fit(n_outer=200, lr=1e-2)
    _, std_in = sg.predict(torch.tensor([[0.0]]))
    _, std_out = sg.predict(torch.tensor([[10.0]]))
    si = float(std_in.item())
    so = float(std_out.item())
    _ok(so > si,
        f"out-of-range std ({so:.3f}) > in-range ({si:.3f})")


def test_sparse_gp_gradient_through_inducing_points():
    torch.manual_seed(0)
    n = 40
    x = torch.linspace(-3.0, 3.0, n).unsqueeze(-1)
    y = torch.sin(x).squeeze(-1)
    sg = SparseGP(n_inducing=10, dim=1, n_train=n, kernel="rbf")
    sg.fit_init(x, y)
    loss = sg.elbo()
    loss.backward()
    has_grad = sg.Z.grad is not None and float(torch.sum(torch.abs(sg.Z.grad))) > 0
    _ok(has_grad, "SparseGP inducing points Z receive non-zero gradients")


# --------------------------------------------------------- SpectralClustering
def test_spectral_clustering_recovers_two_moons():
    """Two well-separated blobs in 2-D should be recovered."""
    torch.manual_seed(0)
    np.random.seed(0)
    n = 60
    X1 = torch.randn(n, 2) + torch.tensor([0.0, 0.0])
    X2 = torch.randn(n, 2) + torch.tensor([5.0, 5.0])
    X = torch.cat([X1, X2], dim=0)
    y_true = np.array([0] * n + [1] * n)
    sc = SpectralClustering(n_clusters=2, gamma=0.3)
    pred = sc.fit_predict(X).detach().cpu().numpy()
    # Compute best (mod swap) accuracy
    same = (y_true == pred).sum()
    swap = (y_true == (1 - pred)).sum()
    acc = max(same, swap) / len(y_true)
    _ok(acc >= 0.95, f"SpectralClustering on 2 blobs = {acc:.3f}")


def test_spectral_clustering_fit_predict_shape():
    sc = SpectralClustering(n_clusters=3, gamma=0.5)
    X = torch.randn(30, 4)
    pred = sc.fit_predict(X)
    _ok(pred.shape == (30,), f"pred shape {pred.shape}")
    _ok(int((pred >= 0).all() and (pred < 3).all()),
        "pred labels in [0, n_clusters)")


def test_spectral_clustering_knn_affinity_recovers_concentric_circles():
    """k-NN affinity should perfectly separate two concentric circles."""
    torch.manual_seed(0)
    np.random.seed(0)
    n = 80
    theta = np.linspace(0, 2 * np.pi, n)
    X1 = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 2.0
    X2 = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 4.0
    X = torch.tensor(
        np.concatenate([X1 + 0.3 * np.random.randn(n, 2),
                       X2 + 0.3 * np.random.randn(n, 2)],
                      axis=0).astype("float32")
    )
    sc = SpectralClustering(n_clusters=2, affinity="knn", n_neighbors=5)
    pred = sc.fit_predict(X).detach().cpu().numpy()
    acc = max((pred[:n] == 0).sum() + (pred[n:] == 1).sum(),
               (pred[:n] == 1).sum() + (pred[n:] == 0).sum()) / (2 * n)
    _ok(acc >= 0.7, f"k-NN spectral clustering on circles = {acc:.3f}")


# ----------------------------------------------------------------- RBM
def test_rbm_cd1_training_decreases_free_energy():
    """On synthetic 4-bit-binary patterns the RBM should learn a
    distribution whose free energy on training samples goes down."""
    torch.manual_seed(0)
    np.random.seed(0)
    # Two 4-bit patterns and their noisy variants.
    a = np.array([1, 1, 0, 0], dtype="float32")
    b = np.array([0, 0, 1, 1], dtype="float32")
    X = []
    for _ in range(40):
        base = a if np.random.rand() < 0.5 else b
        flip = np.random.rand(4) < 0.1
        X.append(np.where(flip, 1 - base, base))
    X = torch.tensor(np.array(X, dtype="float32"))
    rbm = RBM(n_visible=4, n_hidden=6, lr=0.05, k=1)
    initial = float(rbm.free_energy(X).mean().item())
    rbm.fit(X, n_epochs=200)
    final = float(rbm.free_energy(X).mean().item())
    _ok(final < initial,
        f"RBM free energy {initial:.3f} -> {final:.3f}")


def test_rbm_samples_have_valid_binary_values():
    torch.manual_seed(0)
    rbm = RBM(n_visible=4, n_hidden=4, lr=0.05, k=1)
    rbm.fit(torch.tensor(np.random.randint(0, 2, (40, 4)).astype("float32")),
            n_epochs=50)
    samples = rbm.sample(n_steps=20, n_chains=5)
    _ok(samples.shape == (5, 4), f"RBM sample shape {samples.shape}")
    uniq = set(samples.reshape(-1).tolist())
    _ok(uniq.issubset({0.0, 1.0}), f"RBM samples only contain 0/1, got {uniq}")


# ----------------------------------------------------------- IsingModel
def test_ising_magnetisation_grows_below_critical_temperature():
    """For a 2-D Ising model with no external field the absolute
    magnetisation |m| is small at high temperature and large at
    low temperature (Curie point ~ 2.27)."""
    torch.manual_seed(0)
    np.random.seed(0)
    ising = IsingModel(n_rows=8, n_cols=8, J=1.0, h=0.0)
    mag_low = ising.magnetisation(n_burn_in=200, n_samples=500, beta=0.5)
    mag_high = ising.magnetisation(n_burn_in=200, n_samples=500, beta=0.1)
    _ok(mag_low > mag_high,
        f"|m|(low-T) = {mag_low:.3f} > |m|(high-T) = {mag_high:.3f}")


def test_ising_gibbs_samples_have_pm1_values():
    torch.manual_seed(0)
    ising = IsingModel(n_rows=4, n_cols=4)
    spins = ising.gibbs_sample(n_burn_in=20, n_steps=30, beta=0.4)
    _ok(spins.shape[0] >= 1 and spins.shape[1:] == (4, 4),
        f"Gibbs sample shape {spins.shape}")
    uniq = set(spins.reshape(-1).tolist())
    _ok(uniq.issubset({-1.0, 1.0}), f"Ising spins are +/-1, got {uniq}")


def test_ising_critical_temperature():
    ising = IsingModel(n_rows=8, n_cols=8, J=1.0, h=0.0)
    tc = ising.critical_temperature
    _ok(abs(tc - 2.269) < 0.01,
        f"critical_temperature = {tc:.4f} (expect ~2.269)")


def test_ising_spin_string():
    ising = IsingModel(n_rows=3, n_cols=4)
    spins = torch.tensor([[1.0, -1.0, 1.0, 1.0],
                           [-1.0, 1.0, -1.0, 1.0],
                           [1.0, 1.0, -1.0, -1.0]])
    s = ising.spin_string(spins)
    _ok("+" in s and "·" in s, "spin_string contains + and ·")
    _ok(len(s.split("\n")) == 3, "spin_string has 3 rows")


if __name__ == "__main__":
    test_sparse_gp_fits_sine()
    test_sparse_gp_predictive_std_grows_outside_training_data()
    test_sparse_gp_gradient_through_inducing_points()
    test_spectral_clustering_recovers_two_moons()
    test_spectral_clustering_fit_predict_shape()
    test_spectral_clustering_knn_affinity_recovers_concentric_circles()
    test_rbm_cd1_training_decreases_free_energy()
    test_rbm_samples_have_valid_binary_values()
    test_ising_magnetisation_grows_below_critical_temperature()
    test_ising_gibbs_samples_have_pm1_values()
    test_ising_critical_temperature()
    test_ising_spin_string()
    print("All SparseGP / SpectralClustering / RBM / Ising tests passed.")

