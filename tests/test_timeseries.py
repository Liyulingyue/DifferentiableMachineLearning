"""Smoke tests for DifferentiableMachineLearning.TimeSeries.

Run with:
    .venv/bin/python tests/test_timeseries.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from DifferentiableMachineLearning.TimeSeries import AR, ARMA, FIR, Autoregressive  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


def test_autoregressive_generic():
    block = Autoregressive(y_features=3, x_features=[2, 4], e_features=2)
    y = torch.randn(8, 3)
    u1 = torch.randn(8, 2)
    u2 = torch.randn(8, 4)
    v = torch.randn(8, 2)
    out = block(y, u1, u2, v)
    _ok(out.shape == (8, 1), f"generic block output shape {out.shape}")
    _ok(isinstance(out, torch.Tensor), "output is a torch.Tensor")

    # also accept single-sample (rank-1) inputs
    out1 = block(y[0], u1[0], u2[0], v[0])
    _ok(out1.shape == (1, 1), f"rank-1 inputs promoted to {out1.shape}")


def test_ar():
    p = 5
    ar = AR(p)
    y = torch.randn(16, p)
    out = ar(y)
    _ok(out.shape == (16, 1), f"AR({p}) output shape {out.shape}")


def test_arma():
    p, q = 4, 3
    arma = ARMA(p, q)
    y = torch.randn(16, p)
    v = torch.randn(16, q + 1)
    out = arma(y, v)
    _ok(out.shape == (16, 1), f"ARMA({p},{q}) output shape {out.shape}")


def test_fir():
    q = 4
    fir = FIR(q)
    u = torch.randn(16, q + 1)
    out = fir(u)
    _ok(out.shape == (16, 1), f"FIR({q}) output shape {out.shape}")


def test_gradient_flows():
    ar = AR(3)
    y = torch.randn(4, 3)
    target = torch.randn(4, 1)
    pred = ar(y)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.sum(p.grad) != 0
        for p in ar.parameters()
    )
    _ok(has_grad, "AR produces non-zero gradients on parameters")


if __name__ == "__main__":
    test_autoregressive_generic()
    test_ar()
    test_arma()
    test_fir()
    test_gradient_flows()
    print("All TimeSeries smoke tests passed.")
