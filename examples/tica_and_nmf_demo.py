"""Two demos for the new ClassicalML additions.

1. tICA on a synthetic 2-D diffusion with one slow and one fast
   direction. We project the trajectory onto the recovered slow
   mode and inspect the autocorrelation at the requested lag.

2. NMF on a synthetic 3-topic document-term matrix. Each
   "document" is a random mixture of the 3 topics; we verify that
   the recovered dictionary is close to the true topics up to
   permutation and report the reconstruction error.
"""

import torch
import numpy as np
from DifferentiableMachineLearning.ClassicalML import tICA, NMF


def tica_demo():
    torch.manual_seed(0)
    T = 4000
    slow = torch.cumsum(0.05 * torch.randn(T), dim=0)
    fast = torch.cumsum(1.0 * torch.randn(T), dim=0)
    theta = 0.5
    R = torch.tensor(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    ).float()
    X = torch.stack([slow, fast], dim=1) @ R.T.float()
    tica = tICA(n_components=1, dim=2, lag=20)
    tica.fit(X)
    proj = tica.transform(X).squeeze().detach().cpu().numpy()
    # Compute autocorrelation of the projected slow mode at lag=20.
    proj_d = torch.tensor(proj)
    a = proj_d[:-20]
    b = proj_d[20:]
    corr = float((a - a.mean()) @ (b - b.mean()) /
                 (a.shape[0] * a.std() * b.std()))
    print(f"tICA slow-mode autocorrelation at lag=20: {corr:.3f}")
    # Compare to the *fast* mode (any vector orthogonal to slow).
    fast_axis = R @ torch.tensor([0.0, 1.0])
    other = X @ fast_axis.unsqueeze(-1)
    other = other.squeeze().detach().cpu().numpy()
    other_d = torch.tensor(other)
    a2 = other_d[:-20]
    b2 = other_d[20:]
    corr_fast = float((a2 - a2.mean()) @ (b2 - b2.mean()) /
                      (a2.shape[0] * a2.std() * b2.std()))
    print(f"fast mode autocorrelation at lag=20  : {corr_fast:.3f}")


def nmf_demo():
    torch.manual_seed(0)
    np.random.seed(0)
    n_features = 12
    topic_a = torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    topic_b = torch.tensor([0.0, 0.0, 0.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    topic_c = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    W_true = torch.stack([topic_a, topic_b, topic_c], dim=0)
    N = 200
    H_true = torch.abs(torch.randn(N, 3)) + 0.1
    X = H_true @ W_true + 0.01 * torch.abs(torch.randn(N, n_features))

    nmf = NMF(n_components=3, n_features=n_features, init="nndsvd")
    nmf.fit(X, n_iter=500)
    rec = nmf.reconstruct(nmf.H)
    rel_err = float(torch.mean(torch.sum((X - rec.detach()) ** 2, dim=1)) /
                    torch.mean(torch.sum(X ** 2, dim=1)))
    # Per-topic cosine (best over permutations)
    W = nmf.W.detach().numpy()
    best = []
    for i in range(3):
        c = []
        for j in range(3):
            cos = float(W[i] @ W_true[j].numpy()) / (
                np.linalg.norm(W[i]) * np.linalg.norm(W_true[j].numpy())
            )
            c.append(abs(cos))
        best.append(max(c))
    print(f"NMF 3-topic rel reconstruction err: {rel_err:.4f}")
    print(f"NMF per-topic best cosines       : {[round(c, 3) for c in best]}")


if __name__ == "__main__":
    print("=== tICA demo ===")
    tica_demo()
    print()
    print("=== NMF demo ===")
    nmf_demo()
