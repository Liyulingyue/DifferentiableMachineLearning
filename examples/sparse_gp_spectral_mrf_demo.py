"""Four end-to-end demos for the newer components:

1. SparseGP: variational sparse GP regression with Titsias 2009 ELBO,
   inducing points, and predictive uncertainty.
2. SpectralClustering: Ng-Jordan-Weiss spectral clustering on two
   moons / two circles synthetic data.
3. RBM: Bernoulli Restricted Boltzmann Machine with CD-k training,
   free-energy evaluation, and Gibbs sampling.
4. IsingModel: 2-D Ising model with single-spin Gibbs MCMC,
   demonstrating phase transition near the critical temperature.
"""

import torch
import numpy as np
from DifferentiableMachineLearning.ClassicalML import SparseGP, SpectralClustering, RBM, IsingModel


def sparse_gp_demo():
    print("=== Sparse GP (Titsias 2009) demo ===")
    torch.manual_seed(0)
    n = 80
    x = torch.linspace(-4.0, 4.0, n).unsqueeze(-1)
    y = torch.sin(x).squeeze(-1) + 0.15 * torch.randn(n)

    sg = SparseGP(n_inducing=15, dim=1, n_train=n, kernel="rbf")
    sg.fit_init(x, y)

    print(f"Initial neg-ELBO: {float(sg.elbo().detach().cpu().item()):.3f}")
    sg.fit(n_outer=300, lr=1e-2)
    print(f"Final   neg-ELBO: {float(sg.elbo().detach().cpu().item()):.3f}")

    x_test = torch.linspace(-6.0, 6.0, 15).unsqueeze(-1)
    mean, std = sg.predict(x_test)
    print("\nx_test     mean       std")
    print("(note: std grows outside training range)")
    for xi, m, s in zip(
        x_test.squeeze().detach().cpu().numpy(), mean.squeeze().detach().cpu().numpy(), std.squeeze().detach().cpu().numpy()
    ):
        print(f"{float(xi):+7.3f}   {float(m):+7.4f}   {float(s):.4f}")
    print()


def spectral_clustering_demo():
    print("=== Spectral Clustering (Ng-Jordan-Weiss) demo ===")
    torch.manual_seed(0)
    np.random.seed(0)

    n = 60
    X1 = torch.randn(n, 2) + torch.tensor([-1.5, -1.5])
    X2 = torch.randn(n, 2) + torch.tensor([1.5, 1.5])
    X = torch.cat([X1, X2], dim=0)
    y_true = np.array([0] * n + [1] * n)

    sc = SpectralClustering(n_clusters=2, gamma=0.3)
    pred = sc.fit_predict(X).numpy()

    same = (y_true == pred).sum()
    swap = (y_true == (1 - pred)).sum()
    acc = max(same, swap) / len(y_true)
    print(f"Two-blobs recovery accuracy: {acc:.3f}  (expect >= 0.95)")

    n = 80
    theta = np.linspace(0, 2 * np.pi, n)
    X1 = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 2.0
    X2 = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 4.0
    X = torch.tensor(np.concatenate([X1 + 0.3 * np.random.randn(n, 2),
                                    X2 + 0.3 * np.random.randn(n, 2)],
                                   axis=0).astype("float32"))
    sc2 = SpectralClustering(n_clusters=2, affinity="knn", n_neighbors=5)
    pred2 = sc2.fit_predict(X).numpy()
    acc2 = max((pred2[:n] == 0).sum() + (pred2[n:] == 1).sum(),
               (pred2[:n] == 1).sum() + (pred2[n:] == 0).sum()) / (2 * n)
    print(f"Two-circles recovery accuracy: {acc2:.3f}  (expect >= 0.90)")
    print()


def rbm_demo():
    print("=== Restricted Boltzmann Machine (Hinton 2002) demo ===")
    torch.manual_seed(0)
    np.random.seed(0)

    a = np.array([1, 1, 0, 0, 1, 0], dtype="float32")
    b = np.array([0, 0, 1, 1, 0, 1], dtype="float32")
    X = []
    for _ in range(60):
        base = a if np.random.rand() < 0.5 else b
        flip = np.random.rand(6) < 0.1
        X.append(np.where(flip, 1 - base, base))
    X = torch.tensor(np.array(X, dtype="float32"))

    rbm = RBM(n_visible=6, n_hidden=10, lr=0.05, k=1)
    initial_fe = float(rbm.free_energy(X).mean().item())
    print(f"Initial mean free energy: {initial_fe:.4f}")

    rbm.fit(X, n_epochs=300)
    final_fe = float(rbm.free_energy(X).mean().item())
    print(f"Final   mean free energy: {final_fe:.4f}  (expect lower)")
    print(f"Free energy reduction:   {initial_fe - final_fe:.4f}")

    samples = rbm.sample(n_steps=50, n_chains=3)
    print(f"\nGibbs samples (last step of 3 chains, 6 bits each):")
    for i, s in enumerate(samples.numpy()):
        print(f"  chain {i}: {s}")

    features = rbm.transform(X[:5])
    print(f"\nHidden activations for first 5 patterns: {features.shape}")
    print()


def ising_demo():
    print("=== 2-D Ising Model (MCMC) demo ===")
    print("Critical temperature T_c = 2/ln(1+sqrt(2)) ≈ 2.269 (J=1)")
    torch.manual_seed(0)
    np.random.seed(0)

    ising = IsingModel(n_rows=10, n_cols=10, J=1.0, h=0.0)

    for beta, label in [(0.3, "high-T (disordered)"),
                        (0.6, "low-T  (ferromagnetic)"),
                        (1.0, "very low-T (fully ordered)")]:
        mag = ising.magnetisation(n_burn_in=300, n_samples=500, beta=beta)
        print(f"  beta={beta:.1f}  |m|={mag:.4f}  ({label})")

    print("\nPhase transition near beta ~ 0.44:")
    for beta in [0.40, 0.42, 0.44, 0.46, 0.48]:
        mag = ising.magnetisation(n_burn_in=300, n_samples=400, beta=beta)
        print(f"  beta={beta:.2f}  |m|={mag:.4f}")


if __name__ == "__main__":
    sparse_gp_demo()
    spectral_clustering_demo()
    rbm_demo()
    ising_demo()
