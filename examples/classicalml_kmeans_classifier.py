"""End-to-end demo: quantize a 2-D embedding with KMeans and feed the
soft cluster-assignment vector into a linear classifier. Both pieces
train jointly with backprop - a tiny example of "classical model +
deep model in one autograd graph".
"""

import torch
from DifferentiableMachineLearning.ClassicalML import KMeans


def main():
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    X, Y = [], []
    for cls, c in enumerate(centers):
        pts = c + 0.2 * torch.randn(80, 2)
        X.append(pts)
        Y.append(torch.full((80,), cls, dtype=torch.int64))
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)

    km = KMeans(k=3, dim=2, temperature=0.5, init="kmeans++")
    km.fit(X, n_iter=15)                       # warm-start centroids
    classifier = torch.nn.Linear(3, 3)        # 3 cluster-soft features -> 3 classes

    opt = torch.optim.Adam(
        list(km.parameters()) + list(classifier.parameters()),
        lr=1e-2,
    )

    for epoch in range(200):
        km.train()
        feats = km(X, hard=False)              # [240, 3] differentiable
        logits = classifier(feats)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = torch.argmax(logits, dim=-1)
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={(acc==Y.squeeze()).float().mean().item():.3f}")

    print("final centroids:\n", km.centroids.detach().cpu().numpy())


if __name__ == "__main__":
    main()
