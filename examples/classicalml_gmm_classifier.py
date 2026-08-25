"""End-to-end demo: GMM produces K-dim soft responsibilities that feed
a linear classifier. The GMM parameters are *not* frozen, so they get
nudged toward class-separable positions during training, while still
behaving as a probabilistic mixture model.
"""

import torch
from DifferentiableMachineLearning.ClassicalML import GMM


def main():
    torch.manual_seed(0)
    # three Gaussian clusters, each assigned to its own class
    centers = torch.tensor([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    X, Y = [], []
    for cls, c in enumerate(centers):
        pts = c + 0.3 * torch.randn(80, 2)
        X.append(pts)
        Y.append(torch.full((80,), cls, dtype=torch.int64))
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)

    gmm = GMM(k=3, dim=2, covariance_type="diag", reg=1e-3)
    # warm-start means with the true centres, log-variances = log(0.3^2)
    gmm.means.data.copy_(centers.clone())
    gmm.log_vars.data.copy_(torch.log(torch.ones(3, 2) * 0.09))
    classifier = torch.nn.Linear(3, 3)

    opt = torch.optim.Adam(
        list(gmm.parameters()) + list(classifier.parameters()),
        lr=1e-2,
    )

    for epoch in range(200):
        feats = gmm(X)                # [240, 3] differentiable responsibilities
        logits = classifier(feats)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = torch.argmax(logits, dim=-1)
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={(acc==Y.squeeze()).float().mean().item():.3f}")

    print("final means:\n", gmm.means.detach().cpu().numpy())


if __name__ == "__main__":
    main()
