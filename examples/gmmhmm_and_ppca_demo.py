"""Two demos for the new ClassicalML additions.

1. GMMHMM on a synthetic 3-state chain emitting 4-D Gaussian
   observations. After Baum-Welch EM, the Viterbi path is
   compared to the true latent states (modulo permutation).

2. ProbabilisticPCAMixture on synthetic 4 clusters, each living
   in a different 2-D plane of a 10-D space. The mixture is fit
   with EM and the recovered cluster labels are reported.
"""

import torch
import numpy as np
from DifferentiableMachineLearning.ClassicalML import GMMHMM, ProbabilisticPCAMixture


def gmmhmm_demo():
    torch.manual_seed(0)
    np.random.seed(0)
    K, F = 3, 4
    trans = torch.tensor(
        [[0.85, 0.10, 0.05],
         [0.10, 0.85, 0.05],
         [0.05, 0.10, 0.85]]
    )
    means = torch.tensor([
        [+2.0, 0.0, 0.0, 0.0],
        [0.0, +2.0, 0.0, 0.0],
        [0.0, 0.0, +2.0, 0.0],
    ])

    def sample(T):
        z, x = [], []
        s = int(np.random.choice(K))
        for _ in range(T):
            z.append(s)
            x.append(np.random.normal(means[s].detach().cpu().numpy(), 0.2))
            s = int(np.random.choice(K, p=trans[s].detach().cpu().numpy()))
        return np.array(z), np.array(x)

    z_true, x_seq = sample(300)
    x = torch.tensor(x_seq, dtype=torch.float32)
    hmm = GMMHMM(n_states=K, n_components=1, n_features=F)
    hmm.fit_em(x, n_iter=50)
    pred_path = hmm.viterbi(x).detach().cpu().numpy()
    best = 0
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        m = np.array([perm[s] for s in pred_path])
        best = max(best, (z_true == m).sum())
    print(f"GMMHMM 3-state Viterbi accuracy (mod perm) = {best / len(z_true):.3f}")


def ppca_mixture_demo():
    torch.manual_seed(0)
    np.random.seed(0)
    K, F, L = 4, 10, 2
    means_true = torch.tensor(
        [[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    parts = []
    y_true = []
    for k in range(K):
        z = 0.3 * torch.randn(50, L)
        # Each cluster's loading is a different pair of feature axes.
        W_k = torch.zeros(F, L)
        W_k[2 * k, 0] = 1.0
        W_k[2 * k + 1, 1] = 1.0
        X = means_true[k] + z @ W_k.T + 0.05 * torch.randn(50, F)
        parts.append(X)
        y_true.extend([k] * 50)
    X = torch.cat(parts, dim=0)
    y_true = np.array(y_true)

    ppca = ProbabilisticPCAMixture(n_components=K, n_features=F, n_latent=L)
    ppca.fit_em(X, n_iter=30)
    pred = ppca(X).detach().cpu().numpy().argmax(axis=1)
    # Best permutation
    best = 0
    from itertools import permutations
    for perm in permutations(range(K)):
        m = np.array([perm[s] for s in pred])
        best = max(best, (y_true == m).sum())
    print(f"PPCA-mixture 4-cluster recovery (mod perm) = {best / len(y_true):.3f}")


if __name__ == "__main__":
    print("=== GMMHMM demo ===")
    gmmhmm_demo()
    print()
    print("=== ProbabilisticPCAMixture demo ===")
    ppca_mixture_demo()
