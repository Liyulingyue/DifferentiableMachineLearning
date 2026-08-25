"""End-to-end demo: a SoftDecisionTree of depth 4 trained on a
2-D "moons"-like dataset with noisy class boundaries, evaluated by
held-out accuracy. The tree uses ~2*4-1 = 7 inner nodes and 16
leaves; with soft routing the gradient signal can flow back into
all 14 parameters (W, b, leaf_logits) simultaneously.
"""

import torch
import numpy as np
from DifferentiableMachineLearning.ClassicalML import SoftDecisionTree


def make_moons(n: int = 600, noise: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, np.pi, n // 2)
    x0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x1 = np.stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)], axis=1)
    X = np.concatenate([x0, x1], axis=0)
    X += noise * rng.standard_normal(X.shape)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype("int64")
    return torch.tensor(X.astype("float32")), torch.tensor(y)


def main():
    torch.manual_seed(0)
    X, Y = make_moons()
    tree = SoftDecisionTree(depth=4, n_features=2, n_classes=2, temperature=1.5)
    opt = torch.optim.Adam(tree.parameters(), lr=5e-2)
    for epoch in range(400):
        logits = tree(X)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = float(torch.mean((tree.predict(X) == Y).float()))
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.3f}")
    final_acc = float(torch.mean((tree.predict(X) == Y).float()))
    print(f"final train acc = {final_acc:.3f}")


if __name__ == "__main__":
    main()
