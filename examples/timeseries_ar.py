"""Tiny end-to-end demo: fit an AR(2) model on a synthetic series.

The "true" process is   y_k = 0.6 y_{k-1} - 0.2 y_{k-2} + noise,
so after enough gradient steps the learned coefficients should be close.
"""

import torch
from DifferentiableMachineLearning.TimeSeries import AR


def main():
    torch.manual_seed(0)
    p = 2
    N = 4000

    # ground-truth series
    y = torch.zeros([N])
    a_true = torch.tensor([0.6, -0.2])
    for k in range(p, N):
        y[k] = a_true[0] * y[k - 1] + a_true[1] * y[k - 2] + 0.05 * torch.randn([1])

    # build windows: predict y[k] from y[k-p .. k-1]
    X = torch.stack([y[k - p : k] for k in range(p, N)], dim=0)  # [N-p, p]
    Y = y[p:].unsqueeze(-1)                                        # [N-p, 1]

    model = AR(p)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    for epoch in range(200):
        pred = model(X)
        loss = torch.nn.functional.mse_loss(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  loss={loss.item():.6f}")

    # the linear layer has weights of shape [1, total_in]
    w = model._block.linear.weight.detach().cpu().numpy().reshape(-1)
    print("learned AR coefficients:", w)
    print("ground truth            :", a_true.detach().cpu().numpy())


if __name__ == "__main__":
    main()
