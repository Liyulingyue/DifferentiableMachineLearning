"""Smoke tests for PCA, KernelRidge, and GMM in ClassicalML.

Run with:
    .venv/bin/python tests/test_classicalml2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from DifferentiableMachineLearning.ClassicalML import PCA, KernelRidge, GMM  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ------------------------------------------------------------------ PCA
def test_pca_initial_basis_is_orthonormal():
    pca = PCA(n_components=3, dim=5)
    B = pca._basis().detach().cpu().numpy()
    G = B @ B.T
    _ok(np.allclose(G, np.eye(3), atol=1e-5), f"initial basis orthonormal G={G}")


def test_pca_fit_recovers_axis_on_synthetic_line():
    """Data lying on a line in 5-D: PCA(1) should recover that direction."""
    torch.manual_seed(0)
    direction = torch.randn(5)
    direction = direction / torch.norm(direction)
    x = torch.linspace(-3.0, 3.0, 100).unsqueeze(-1) * direction
    x = x + 0.01 * torch.randn(100, 5)

    pca = PCA(n_components=1, dim=5).fit(x)
    rec = pca.reconstruct(pca.project(x))
    rel_err = float(torch.mean(torch.sum((x - rec) ** 2, dim=1)) / torch.mean(torch.sum(x ** 2, dim=1)))
    _ok(rel_err < 0.01, f"1-D PCA reconstruction rel err = {rel_err:.5f}")


def test_pca_gradient_flows_through_basis():
    pca = PCA(n_components=2, dim=4)
    x = torch.randn(5, 4)
    target = torch.randn(5, 4)
    pred = pca(x)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    has_grad = pca.components.grad is not None and torch.sum(pca.components.grad) != 0
    _ok(has_grad, "PCA basis receives non-zero gradients")


# ---------------------------------------------------------- KernelRidge
def test_kernel_ridge_rbf_fits_sine():
    torch.manual_seed(0)
    n = 200
    x = torch.linspace(-3.0, 3.0, n).unsqueeze(-1)
    y = torch.sin(x) + 0.05 * torch.randn(n, 1)
    kr = KernelRidge(n_support=n, dim_in=1, dim_out=1,
                     kernel="rbf", gamma=0.5, alpha=1e-2)
    kr.fit(x, y)
    pred = kr(x)
    mse = float(torch.mean((pred - y) ** 2))
    _ok(mse < 0.05, f"RBF kernel ridge sin-fit MSE = {mse:.5f}")


def test_kernel_ridge_linear_recovers_linear_function():
    torch.manual_seed(0)
    n = 50
    x = torch.randn(n, 2)
    w_true = torch.tensor([[1.5, -2.0]])
    y = x @ w_true.T
    kr = KernelRidge(n_support=n, dim_in=2, dim_out=1, kernel="linear", alpha=1e-6)
    kr.fit(x, y)
    pred = kr(x)
    mse = float(torch.mean((pred - y) ** 2))
    _ok(mse < 1e-3, f"linear kernel ridge MSE = {mse:.6f}")


# ----------------------------------------------------------------- GMM
def test_gmm_responsibilities_sum_to_one():
    gmm = GMM(k=3, dim=2, covariance_type="diag")
    x = torch.randn(7, 2)
    r = gmm(x)
    _ok(r.shape == (7, 3), f"responsibilities shape {r.shape}")
    _ok(torch.allclose(r.sum(-1), torch.ones(7), atol=1e-5),
        "responsibility rows sum to 1")


def test_gmm_em_separates_two_blobs():
    torch.manual_seed(0)
    blobs = [torch.randn(60, 2) * 0.2 + torch.tensor([0.0, 0.0]),
             torch.randn(60, 2) * 0.2 + torch.tensor([4.0, 4.0])]
    x = torch.cat(blobs, dim=0)
    gmm = GMM(k=2, dim=2, covariance_type="diag", reg=1e-4)
    # initialise means at extreme points
    gmm.means.data.copy_(torch.tensor([[-5.0, -5.0], [5.0, 5.0]]))
    gmm.fit_em(x, n_iter=40)
    r = gmm.responsibilities(x).detach().cpu().numpy()
    pred = r.argmax(axis=1)
    b0 = np.bincount(pred[:60], minlength=2).max()
    b1 = np.bincount(pred[60:], minlength=2).max()
    purity = b0 + b1
    _ok(purity >= 110,
        f"2-blob GMM purity = {purity} (b0={b0}, b1={b1}, expected >= 110)")


def test_gmm_gradient_path():
    gmm = GMM(k=2, dim=2, covariance_type="diag")
    x = torch.randn(5, 2)
    # responsibilities are differentiable
    r = gmm(x)
    loss = torch.nn.functional.mse_loss(r, torch.ones(5, 2) * 0.5)
    loss.backward()
    grads = [gmm.means.grad, gmm.log_vars.grad, gmm.log_weights.grad]
    has_grad = all(g is not None and torch.sum(torch.abs(g)) > 0 for g in grads)
    _ok(has_grad, "GMM parameters receive non-zero gradients through responsibilities")


if __name__ == "__main__":
    test_pca_initial_basis_is_orthonormal()
    test_pca_fit_recovers_axis_on_synthetic_line()
    test_pca_gradient_flows_through_basis()
    test_kernel_ridge_rbf_fits_sine()
    test_kernel_ridge_linear_recovers_linear_function()
    test_gmm_responsibilities_sum_to_one()
    test_gmm_em_separates_two_blobs()
    test_gmm_gradient_path()
    print("All PCA / KernelRidge / GMM tests passed.")

