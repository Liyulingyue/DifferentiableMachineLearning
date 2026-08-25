"""Two end-to-end demos for the new ClassicalML additions.

1. KalmanFilter 2-D tracking: simulate a constant-velocity target
   observed through a noisy linear measurement, then run the
   smoother and print the recovered trajectory.
2. LinearChainCRF toy "POS" tagging: each token gets a small random
   feature vector; tag sequences follow a simple "color -> color"
   bias. Train the CRF for a few hundred steps and report the Viterbi
   decode accuracy on held-out data.
3. tSNE on a 3-cluster Gaussian mixture in 5-D; visualise the
   resulting 2-D embedding by reporting the cluster centroid
   distances.
"""

import torch
import numpy as np
from DifferentiableMachineLearning.ClassicalML import KalmanFilter, LinearChainCRF, tSNE


def kalman_demo():
    torch.manual_seed(0)
    T = 80
    A = torch.tensor([[1.0, 0.1], [0.0, 1.0]])
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x_true = torch.zeros(T, 2)
    for t in range(1, T):
        x_true[t] = A @ x_true[t - 1] + 0.1 * torch.randn(2)
    y = x_true @ C.T + 0.05 * torch.randn(T, 2)

    kf = KalmanFilter(state_dim=2, obs_dim=2)
    kf.A.data.copy_(torch.eye(2))
    kf.C.data.copy_(torch.eye(2))
    pred = kf(y)
    err = float(torch.mean((pred - x_true) ** 2).detach())
    print(f"Kalman 2-D tracking MSE = {err:.4f}")
    print("first 5 smoother outputs (vs truth):")
    for t in range(5):
        print(f"  t={t}  pred={pred[t].detach().cpu().numpy()}  truth={x_true[t].detach().cpu().numpy()}")


def crf_demo():
    torch.manual_seed(0)
    np.random.seed(0)
    n_tags = 3
    crf = LinearChainCRF(n_features=4, n_tags=n_tags)
    opt = torch.optim.Adam(crf.parameters(), lr=5e-2)

    def make_example():
        T = 8
        # Tags follow: prefer staying in the same tag (85%).
        tags = [int(np.random.randint(0, n_tags))]
        for _ in range(T - 1):
            if np.random.rand() < 0.85:
                tags.append(tags[-1])
            else:
                tags.append(int(np.random.randint(0, n_tags)))
        # Make the emission features correlated with the tag: each
        # tag has a different mean feature vector so the emitter
        # has signal to lock onto.
        tag_means = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32
        )
        feats = tag_means[tags] + 0.1 * torch.randn(T, 4)
        return feats, torch.tensor(tags, dtype=torch.int64)

    for step in range(400):
        feats, tags = make_example()
        loss = crf.nll(feats, tags)
        opt.zero_grad()
        loss.backward()
        opt.step()
    n_correct, n_total = 0, 0
    for _ in range(50):
        feats, tags = make_example()
        pred = crf.decode(feats)
        n_correct += int((pred == tags).sum())
        n_total += int(tags.shape[0])
    print(f"CRF Viterbi accuracy = {n_correct / n_total:.3f}")


def tsne_demo():
    torch.manual_seed(0)
    np.random.seed(0)
    parts = []
    for c in torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                               [5.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 5.0, 0.0, 0.0]]):
        parts.append(0.1 * torch.randn(25, 5) + c)
    X = torch.cat(parts, dim=0)
    tsne = tSNE(n_components=2, perplexity=15.0, n_iter=120, learning_rate=5.0)
    Y = tsne.fit_transform(X).detach().cpu().numpy()
    cent = [Y[i * 25:(i + 1) * 25].mean(axis=0) for i in range(3)]
    print("tSNE 2-D cluster centroids:")
    for i, c in enumerate(cent):
        print(f"  cluster {i}: {c}")
    # Distances
    d01 = float(np.linalg.norm(np.array(cent[0]) - np.array(cent[1])))
    d02 = float(np.linalg.norm(np.array(cent[0]) - np.array(cent[2])))
    d12 = float(np.linalg.norm(np.array(cent[1]) - np.array(cent[2])))
    print(f"  pairwise centroid distances: {d01:.2f} {d02:.2f} {d12:.2f}")


if __name__ == "__main__":
    print("=== KalmanFilter demo ===")
    kalman_demo()
    print()
    print("=== LinearChainCRF demo ===")
    crf_demo()
    print()
    print("=== tSNE demo ===")
    tsne_demo()
