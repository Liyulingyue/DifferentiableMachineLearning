"""End-to-end demo: classify 1-D audio-like signals by their
log-Mel-spectrogram followed by a linear head.

The two "classes" are pure sine waves of different frequencies; the
log-Mel features make the frequency discrimination trivially linear
in the mel basis. The MelSpectrogram's STFT window is left as a
Hann buffer (non-learnable) so the run is fast.
"""

import torch
import numpy as np
from DifferentiableMachineLearning.SignalProcessing import MelSpectrogram


def synthesize(freqs, sr, duration):
    t = torch.arange(int(sr * duration), dtype=torch.float32) / sr
    sigs = [torch.sin(2 * np.pi * f * t) for f in freqs]
    return torch.stack(sigs, dim=0)  # [B, samples]


def main():
    torch.manual_seed(0)
    sr = 16000
    duration = 0.5
    freqs_low = [200.0, 250.0, 300.0]
    freqs_high = [800.0, 1000.0, 1200.0]
    X = torch.cat([synthesize(freqs_low, sr, duration),
                   synthesize(freqs_high, sr, duration)], dim=0)
    Y = torch.cat([torch.zeros(3, dtype=torch.int64),
                   torch.ones(3, dtype=torch.int64)])

    mel = MelSpectrogram(n_mels=32, sample_rate=sr,
                         win_length=400, hop_length=160, n_fft=512)
    feats = mel(X)                                  # [B, n_mels, T]
    feats_flat = feats.reshape([feats.shape[0], -1]) # [B, n_mels * T]

    head = torch.nn.Linear(feats_flat.shape[1], 2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for epoch in range(200):
        feats = mel(X)
        feats_flat = feats.reshape([feats.shape[0], -1])
        logits = head(feats_flat)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = float(torch.mean((torch.argmax(logits, -1) == Y).float()))
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.3f}")
    final_acc = float(torch.mean(
        (torch.argmax(head(mel(X).reshape([X.shape[0], -1])), -1) == Y).float()
    ))
    print(f"final acc = {final_acc:.3f}")


if __name__ == "__main__":
    main()
