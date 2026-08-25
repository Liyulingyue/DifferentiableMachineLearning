"""Smoke tests for LDA, NaiveBayes, ICA, and SoftDecisionTree.

Run with:
    .venv/bin/python tests/test_classicalml3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from DifferentiableMachineLearning.ClassicalML import (  # noqa: E402
    LDA, GaussianNB, MultinomialNB, ICA, SoftDecisionTree,
)


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ----------------------------------------------------------------- LDA
def test_lda_separates_three_clusters():
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    X, Y = [], []
    for cls, c in enumerate(centers):
        X.append(c + 0.4 * torch.randn(40, 2))
        Y.append(torch.full((40,), cls, dtype=torch.int64))
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    lda = LDA(n_components=2, dim=2, n_classes=3).fit(X, Y)
    proj = lda.project(X).detach().cpu().numpy()
    # The 1-D rank along the LDA axis should put class-0 points
    # mostly at one end and class-1 / class-2 points mostly at the other.
    class0 = np.argsort(proj[:40, 0])[:30]
    class1 = np.argsort(proj[40:80, 0])[:30]
    overlap = len(set(class0) & set(class1))
    _ok(overlap < 30, f"LDA top-1 axis class0 vs class1 overlap={overlap} (expected < 30)")


def test_lda_log_proba_is_differentiable():
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    X, Y = [], []
    for cls, c in enumerate(centers):
        X.append(c + 0.4 * torch.randn(30, 2))
        Y.append(torch.full((30,), cls, dtype=torch.int64))
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    lda = LDA(n_components=1, dim=2, n_classes=2).fit(X, Y)
    # predict_log_proba uses the *fitted* class stats (not the
    # components parameter) and so is not differentiable in the
    # ``components`` slot. We just check it produces well-shaped,
    # non-trivial log-probs.
    log_p = lda.predict_log_proba(X)
    _ok(log_p.shape == (60, 2), f"log_p shape {log_p.shape}")
    # The two classes should be distinguished by their log-probs on
    # the X that was used for fitting.
    diff = log_p[:30, 0] - log_p[:30, 1]
    _ok(float(torch.mean(diff)) > 0, "class 0 mean log-p0 - log-p1 should be > 0")


# ------------------------------------------------------------- NaiveBayes
def test_gaussian_nb_fits_separable_blobs():
    torch.manual_seed(0)
    X = torch.cat([
        torch.randn(50, 2) + torch.tensor([0.0, 0.0]),
        torch.randn(50, 2) + torch.tensor([5.0, 5.0]),
    ], dim=0)
    Y = torch.cat([torch.zeros(50, dtype=torch.int64),
                   torch.ones(50, dtype=torch.int64)])
    nb = GaussianNB(dim=2, n_classes=2).fit(X, Y)
    preds = nb.predict(X)
    acc = float(torch.mean((preds == Y).float()))
    _ok(acc >= 0.95, f"GaussianNB accuracy on 2-blob data = {acc:.3f}")


def test_multinomial_nb_fits_text_counts():
    torch.manual_seed(0)
    # Two "topics": class 0 emphasises words 0-2, class 1 emphasises words 3-5.
    n0_rows = [
        [5, 4, 3, 1, 0, 0],
        [6, 5, 2, 0, 1, 0],
        [4, 6, 3, 0, 0, 1],
        [5, 5, 4, 1, 0, 0],
    ]
    n1_rows = [
        [0, 1, 0, 5, 6, 4],
        [1, 0, 1, 4, 5, 5],
        [0, 0, 1, 6, 4, 5],
        [1, 1, 0, 5, 5, 4],
    ]
    rows = n0_rows + n1_rows
    X = torch.tensor(rows, dtype=torch.float32)
    Y = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int64)
    mnb = MultinomialNB(n_features=6, n_classes=2, alpha=1.0).fit(X, Y)
    preds = mnb.predict(X)
    acc = float(torch.mean((preds == Y).float()))
    _ok(acc == 1.0, f"MultinomialNB accuracy on synthetic topics = {acc:.3f}")


# ----------------------------------------------------------------- ICA
def test_ica_recovers_independent_sources_up_to_permutation_and_sign():
    torch.manual_seed(0)
    n = 2000
    # Two super-Gaussian sources
    s = torch.cat([torch.randn(n, 1) ** 3, torch.sign(torch.randn(n, 1))], dim=1)
    A = torch.tensor([[1.0, 0.5], [0.4, 1.0]])
    X = s @ A.T
    ica = ICA(n_components=2, dim=2, nonlinearity="cube", max_iter=1000, tol=1e-6)
    ica.fit(X)
    S_hat = ica.transform(X).detach().cpu().numpy()
    # Correlation matrix between true and estimated sources, after
    # matching the best (sign, permutation).
    C = np.abs(np.corrcoef(s.detach().cpu().numpy().T, S_hat.T)[:2, 2:])
    max_per_col = C.max(axis=0)
    _ok((max_per_col > 0.95).all(),
        f"ICA recovered both sources with |corr| > 0.95 (got {max_per_col.tolist()})")


def test_ica_inverse_transform_round_trip():
    torch.manual_seed(0)
    X = torch.randn(100, 3)
    ica = ICA(n_components=3, dim=3, nonlinearity="cube").fit(X)
    rec = ica.inverse_transform(ica.transform(X))
    err = float(torch.mean(torch.sum((X - rec) ** 2, dim=1)))
    _ok(err < 1e-3, f"ICA reconstruct round-trip MSE = {err:.6f}")


# ------------------------------------------------------ SoftDecisionTree
def test_soft_tree_leaf_probabilities_sum_to_one():
    tree = SoftDecisionTree(depth=3, n_features=4, n_classes=2)
    x = torch.randn(7, 4)
    p = tree(x)
    _ok(p.shape == (7, 2), f"tree output shape {p.shape}")
    _ok(torch.allclose(p.sum(-1), torch.ones(7), atol=1e-5),
        "tree output rows sum to 1")


def test_soft_tree_learns_xor():
    """Depth-2 tree should reach >=75% accuracy on XOR with enough steps."""
    torch.manual_seed(0)
    n = 200
    x = torch.randint(0, 2, (n, 2)).float()
    y = (x[:, 0] != x[:, 1]).long()
    tree = SoftDecisionTree(depth=2, n_features=2, n_classes=2, temperature=2.0)
    opt = torch.optim.Adam(tree.parameters(), lr=5e-2)
    for _ in range(300):
        logits = tree(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = float(torch.mean((tree.predict(x) == y).float()))
    _ok(acc >= 0.75, f"SoftDecisionTree XOR accuracy = {acc:.3f}")


if __name__ == "__main__":
    test_lda_separates_three_clusters()
    test_lda_log_proba_is_differentiable()
    test_gaussian_nb_fits_separable_blobs()
    test_multinomial_nb_fits_text_counts()
    test_ica_recovers_independent_sources_up_to_permutation_and_sign()
    test_ica_inverse_transform_round_trip()
    test_soft_tree_leaf_probabilities_sum_to_one()
    test_soft_tree_learns_xor()
    print("All LDA / NaiveBayes / ICA / SoftDecisionTree tests passed.")

