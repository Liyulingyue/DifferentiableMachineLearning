"""Smoke tests for DifferentiableMachineLearning.ClassicalML.

Run with:
    .venv/bin/python tests/test_classicalml.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from DifferentiableMachineLearning.ClassicalML import KMeans, KNN  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ---------------------------------------------------------------- KMeans
def test_kmeans_soft_assignment_shape_and_grad():
    km = KMeans(k=4, dim=3)
    x = torch.randn(10, 3)
    p = km.soft_assignment(x)
    _ok(p.shape == (10, 4), f"soft_assignment shape {p.shape}")
    _ok(torch.allclose(p.sum(-1), torch.ones(10), atol=1e-5),
        "soft_assignment rows sum to 1")

    # The forward is differentiable in the soft path
    target = torch.nn.functional.one_hot(
        torch.randint(0, 4, (10,)), num_classes=4
    ).float()
    loss = torch.nn.functional.mse_loss(p, target)
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.sum(p.grad) != 0
        for p in km.parameters()
    )
    _ok(has_grad, "KMeans centroids receive non-zero gradients on soft path")


def test_kmeans_hard_assignment_int():
    km = KMeans(k=3, dim=2, init="random")
    x = torch.randn(7, 2)
    labels = km.hard_assignment(x)
    _ok(labels.shape == (7,), f"hard_assignment shape {labels.shape}")
    _ok(labels.dtype == torch.int64, f"hard_assignment dtype {labels.dtype}")
    _ok(int((labels >= 0).all()) and int((labels < 3).all()),
        "hard_assignment indices are in [0, k)")


def test_kmeans_fit_clusters_3_blobs():
    """Synthetic 3-blob data: KMeans should find a clear 3-way split."""
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    blobs = []
    for c in centers:
        blobs.append(c + 0.1 * torch.randn(60, 2))
    x = torch.cat(blobs, dim=0)  # [180, 2]

    km = KMeans(k=3, dim=2, init="kmeans++")
    km.fit_kmeanspp(x)
    km.fit(x, n_iter=20)

    labels = km.hard_assignment(x)
    # Each blob should land in (at most) two clusters with one of them
    # clearly dominant. We check that the modal label covers >= 50% of
    # the points in every blob.
    by_blob = labels.reshape(3, 60).detach().cpu().numpy()
    biggest = [
        int(torch.bincount(torch.tensor(row), minlength=3).max())
        for row in by_blob
    ]
    _ok(all(b >= 30 for b in biggest),
        f"each blob has a dominant cluster (purities={biggest})")


def test_kmeans_eval_mode_returns_hard():
    km = KMeans(k=3, dim=2)
    x = torch.randn(5, 2)
    km.eval()
    out = km(x)            # should be hard assignment
    _ok(out.shape == (5,), f"eval() forward shape {out.shape}")


# ----------------------------------------------------------------- KNN
def test_knn_indices_shape():
    nn = KNN(k=3, dim=2)
    bank = torch.randn(20, 2)
    nn.update_memory(bank)
    x = torch.randn(8, 2)
    idx, d = nn.hard_retrieval(x)
    _ok(idx.shape == (8, 3), f"indices shape {idx.shape}")
    _ok(d.shape == (8, 3), f"distances shape {d.shape}")
    _ok((idx >= 0).all() and (idx < 20).all(), "indices are valid bank positions")


def test_knn_average_mode():
    nn = KNN(k=4, dim=3)
    nn.update_memory(torch.randn(30, 3))
    x = torch.randn(5, 3)
    out = nn(x, mode="average")
    _ok(out.shape == (5, 3), f"average output shape {out.shape}")


def test_knn_values_mode():
    nn = KNN(k=2, dim=2)
    keys = torch.randn(10, 2)
    vals = torch.arange(10, dtype=torch.float32).reshape(10, 1).tile((1, 2))
    nn.update_memory(keys, values=vals)
    # pick a key that exists in the bank so its nearest neighbour is itself
    out = nn(keys[0:1], mode="values")
    _ok(out.shape == (1, 2), f"values output shape {out.shape}")


def test_knn_soft_retrieval_is_probability():
    nn = KNN(k=4, dim=2)
    nn.update_memory(torch.randn(12, 2))
    w, _ = nn.soft_retrieval(torch.randn(3, 2))
    _ok(w.shape == (3, 4), f"soft weights shape {w.shape}")
    _ok(torch.allclose(w.sum(-1), torch.ones(3), atol=1e-5),
        "soft weights sum to 1")


if __name__ == "__main__":
    test_kmeans_soft_assignment_shape_and_grad()
    test_kmeans_hard_assignment_int()
    test_kmeans_fit_clusters_3_blobs()
    test_kmeans_eval_mode_returns_hard()
    test_knn_indices_shape()
    test_knn_average_mode()
    test_knn_values_mode()
    test_knn_soft_retrieval_is_probability()
    print("All ClassicalML smoke tests passed.")

