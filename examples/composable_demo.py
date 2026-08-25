"""Compose DifferentiableMachineLearning components inside an ``nn.Module``
exactly the way you would with stock PyTorch layers.

Three small patterns:

* Pattern A —  PCA pre-fit, then freeze as a preprocessor.
* Pattern B —  KMeans soft assignments fed into a linear classifier,
  jointly fine-tuned with Adam.
* Pattern C —  MelSpectrogram (learnable window) feeding a small
  CNN, all trained end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from DifferentiableMachineLearning.ClassicalML import KMeans, PCA
from DifferentiableMachineLearning.SignalProcessing import MelSpectrogram


# ---------------------------------------------------------------------- A
class FrozenPCAHead(nn.Module):
    """PCA computed once on ``x``, then frozen as a fixed preprocessor."""

    def __init__(self, x, n_components, n_classes):
        super().__init__()
        self.pca = PCA(n_components=n_components, dim=x.shape[1]).fit(x)
        for p in self.pca.parameters():
            p.requires_grad_(False)              # freeze after fit
        self.head = nn.Linear(n_components, n_classes)

    def forward(self, x):
        return self.head(self.pca.project(x))


# ---------------------------------------------------------------------- B
class KMeansClassifier(nn.Module):
    """KMeans soft assignments + linear head, trained end-to-end."""

    def __init__(self, k, dim, n_classes, temperature=0.5):
        super().__init__()
        self.kmeans = KMeans(k=k, dim=dim, temperature=temperature)
        self.head = nn.Linear(k, n_classes)

    def init_centroids(self, x):
        """k-means++ on a sample of training data — call once before training."""
        self.kmeans.fit_kmeanspp(x)

    def forward(self, x):
        return self.head(self.kmeans(x))         # soft (N, k) -> logits


# ---------------------------------------------------------------------- C
class MelCNN(nn.Module):
    """Log-Mel with a learnable window feeding a tiny CNN classifier."""

    def __init__(self, n_mels, n_classes):
        super().__init__()
        self.mel = MelSpectrogram(
            n_mels=n_mels, sample_rate=16000,
            win_length=64, hop_length=32, n_fft=64,
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.cnn(self.mel(x))             # [B, n_mels, T]


# ---------------------------------------------------------------------- demo
def main():
    torch.manual_seed(0)

    # ----- A: 4 high-D blobs, project to 1-D, classify -------------------------
    blobs = torch.cat([torch.randn(40, 2) + t
                       for t in torch.tensor([[0., 0.], [5., 0.],
                                              [0., 5.], [5., 5.]])])
    y = torch.tensor(sum([[i] * 40 for i in range(4)], []))
    model_a = FrozenPCAHead(x=blobs, n_components=1, n_classes=4)
    train(model_a, blobs, y, lr=5e-2, label="A. FrozenPCAHead")

    # ----- B: same blobs, but PCA replaced by KMeans (joint training) --------
    model_b = KMeansClassifier(k=4, dim=2, n_classes=4)
    model_b.init_centroids(blobs)                # one-shot: k-means++ init
    train(model_b, blobs, y, lr=5e-2, label="B. KMeansClassifier")

    # ----- C: 1 kHz vs 4 kHz pure tones, classify by log-Mel spectrum --------
    t = torch.linspace(0, 0.05, 800)
    sig = torch.stack([torch.sin(2 * torch.pi * 1000 * t),
                       torch.sin(2 * torch.pi * 4000 * t)])
    y_snd = torch.tensor([0, 1])
    model_c = MelCNN(n_mels=16, n_classes=2)
    train(model_c, sig, y_snd, lr=1e-3, label="C. MelCNN", steps=100)


def train(model, x, y, lr, label, steps=200):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (model(x).argmax(1) == y).float().mean()
    n_pca = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{label:24s} loss={loss.item():.3f}   acc={acc.item():.2f}   "
          f"trainable params={n_pca}")


if __name__ == "__main__":
    main()