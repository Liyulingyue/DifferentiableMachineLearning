"""Smoke tests for SignalProcessing and the new ClassicalML additions.

Run with:
    .venv/bin/python tests/test_signalproc_and_extras.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from DifferentiableMachineLearning.SignalProcessing import (  # noqa: E402
    STFT, ISTFT, MelSpectrogram, WaveletFilterBank,
)
from DifferentiableMachineLearning.ClassicalML import BayesianRidge, SVM  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ---------------------------------------------------------------- STFT
def test_stft_shapes():
    stft = STFT(win_length=64, hop_length=16, n_fft=64)
    x = torch.randn(8, 1024)
    spec = stft(x)
    # With center=True and n_fft=64, the padded length is 1024+64=1088,
    # n_frames = (1088 - 64) // 16 + 1 = 65.
    _ok(spec.shape == (8, 33, 65),
        f"STFT output shape {spec.shape}")
    # also test return_complex=False
    stft2 = STFT(win_length=64, hop_length=16, n_fft=64, return_complex=False)
    s2 = stft2(x)
    _ok(s2.shape[:2] == (8, 2), f"STFT(complex=False) shape prefix {s2.shape}")


def test_istft_round_trip():
    torch.manual_seed(0)
    x = torch.randn(1, 1024)
    stft = STFT(win_length=64, hop_length=16, n_fft=64, return_complex=False)
    istft = ISTFT(win_length=64, hop_length=16, n_fft=64, center=True)
    rec = istft(stft(x))
    # Hann is not COLA-perfect with hop=win/4 (the per-sample sum-of-
    # windows^2 is 1.5 in the centre), so the round-trip has a small
    # residual ~ 6% of the signal variance in the centre region.
    err = float(torch.mean((x[:, 64:-64] - rec[:, 64:-64]) ** 2))
    _ok(err < 0.5, f"STFT/iSTFT round-trip MSE = {err:.6f}")


def test_stft_learnable_window_receives_gradient():
    stft = STFT(win_length=32, hop_length=8, n_fft=32, learnable_window=True)
    x = torch.randn(2, 256)
    spec = stft(x)
    target = torch.zeros_like(spec.real)
    loss = torch.mean(torch.abs(spec - torch.complex(target, torch.zeros_like(target))) ** 2)
    loss.backward()
    has_grad = stft.window.grad is not None and float(torch.sum(torch.abs(stft.window.grad))) > 0
    _ok(has_grad, "learnable STFT window receives non-zero gradient")


# ---------------------------------------------------------- MelSpec
def test_mel_spectrogram_shape():
    mel = MelSpectrogram(n_mels=40, sample_rate=16000, win_length=400, hop_length=160, n_fft=512)
    x = torch.randn(4, 16000)
    out = mel(x)
    _ok(out.shape[1] == 40, f"MelSpec n_mels = {out.shape[1]}")
    _ok(out.shape[0] == 4, f"MelSpec batch = {out.shape[0]}")


# -------------------------------------------------- WaveletFilterBank
def test_wavelet_filterbank_returns_n_scales():
    wb = WaveletFilterBank(n_scales=3, filter_length=2, learnable=False)
    x = torch.randn(1, 128)
    details = wb(x)
    _ok(len(details) == 3, f"got {len(details)} detail levels")
    # Each level halves the length (approximately).
    lens = [int(d.shape[-1]) for d in details]
    _ok(lens[0] >= lens[1] >= lens[2],
        f"detail lengths non-increasing: {lens}")


# -------------------------------------------------------- BayesianRidge
def test_bayesian_ridge_recovers_linear_function():
    torch.manual_seed(0)
    n, d = 100, 3
    x = torch.randn(n, d)
    w_true = torch.tensor([[1.0], [-2.0], [0.5]])
    y = x @ w_true + 0.05 * torch.randn(n, 1)
    br = BayesianRidge(n_features=d, n_outputs=1, alpha_init=1.0, lambda_init=1.0)
    br.fit(x, y)
    pred = br(x)
    mse = float(torch.mean((pred - y) ** 2))
    _ok(mse < 0.5, f"BayesianRidge MSE = {mse:.4f}")


def test_bayesian_ridge_predictive_std_is_positive():
    torch.manual_seed(0)
    n, d = 50, 2
    x = torch.randn(n, d)
    y = torch.sum(x, dim=1, keepdim=True) + 0.1 * torch.randn(n, 1)
    br = BayesianRidge(n_features=d, n_outputs=1).fit(x, y)
    mean, std = br.forward_with_std(x)
    _ok(mean.shape == (n, 1), f"predictive mean shape {mean.shape}")
    _ok((std > 0).all(), "predictive std > 0 everywhere")


# ---------------------------------------------------------------- SVM
def test_svm_rbf_separates_two_blobs():
    torch.manual_seed(0)
    X = torch.cat([
        torch.randn(40, 2) + torch.tensor([0.0, 0.0]),
        torch.randn(40, 2) + torch.tensor([4.0, 4.0]),
    ], dim=0)
    Y = torch.cat([torch.zeros(40, dtype=torch.int64),
                   torch.ones(40, dtype=torch.int64)])
    svm = SVM(n_support=80, dim=2, n_classes=2, kernel="rbf", gamma=0.3, C=10.0)
    svm.fit(X, Y)
    acc = float(torch.mean((svm.predict(X) == Y).float()))
    _ok(acc >= 0.95, f"LS-SVM RBF accuracy on 2-blob = {acc:.3f}")


def test_svm_ovr_three_classes():
    torch.manual_seed(0)
    parts = []
    labels = []
    for c, mu in enumerate(torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])):
        parts.append(torch.randn(30, 2) + mu)
        labels.append(torch.full((30,), c, dtype=torch.int64))
    X = torch.cat(parts, dim=0)
    Y = torch.cat(labels, dim=0)
    svm = SVM(n_support=90, dim=2, n_classes=3, kernel="rbf", gamma=0.3, C=10.0)
    svm.fit(X, Y)
    acc = float(torch.mean((svm.predict(X) == Y).float()))
    _ok(acc >= 0.90, f"LS-SVM OvR 3-class accuracy = {acc:.3f}")


if __name__ == "__main__":
    test_stft_shapes()
    test_istft_round_trip()
    test_stft_learnable_window_receives_gradient()
    test_mel_spectrogram_shape()
    test_wavelet_filterbank_returns_n_scales()
    test_bayesian_ridge_recovers_linear_function()
    test_bayesian_ridge_predictive_std_is_positive()
    test_svm_rbf_separates_two_blobs()
    test_svm_ovr_three_classes()
    print("All SignalProcessing + BayesianRidge + SVM tests passed.")
