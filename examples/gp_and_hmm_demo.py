"""Two end-to-end demos.

1. Gaussian Process regression: fit a noisy sine, tune kernel
   hyperparameters by minimising the negative log marginal
   likelihood, then plot the predictive mean ± 2 std on a dense
   grid (we just print numerical values here).
2. Gaussian HMM: fit on a synthetic 3-state regime-switching
   emission sequence, report the Viterbi path.
"""

import torch
import numpy as np
from DifferentiableMachineLearning.ClassicalML import GaussianProcess, GaussianHMM


def gp_demo():
    torch.manual_seed(0)
    n = 40
    x = torch.linspace(-3.0, 3.0, n).unsqueeze(-1)
    y = torch.sin(x).squeeze(-1) + 0.1 * torch.randn(n)

    gp = GaussianProcess(dim=1, n_train=n, kernel="rbf",
                         log_lengthscale=0.0, log_outputscale=0.0, log_noise=-1.0)
    gp.fit(x, y)

    opt = torch.optim.Adam(
        [gp.log_lengthscale, gp.log_outputscale, gp.log_noise],
        lr=5e-3,
    )
    for epoch in range(150):
        loss = gp.neg_log_marginal_likelihood()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 30 == 0:
            print(f"GP epoch {epoch:3d}  NLML={loss.item():.3f}  "
                  f"ls={torch.exp(gp.log_lengthscale).item():.3f}  "
                  f"noise={torch.exp(gp.log_noise).item():.3f}")

    x_test = torch.linspace(-5.0, 5.0, 20).unsqueeze(-1)
    mean, std = gp.forward_with_std(x_test)
    print("x_test      mean        std")
    for xi, m, s in zip(x_test.squeeze().detach().cpu().numpy(), mean.squeeze().detach().cpu().numpy(), std.squeeze().detach().cpu().numpy()):
        print(f"{xi:+5.2f}    {m:+5.2f}    {s:5.2f}")


def hmm_demo():
    torch.manual_seed(0)
    np.random.seed(0)
    trans = torch.tensor([[0.95, 0.03, 0.02],
                              [0.05, 0.90, 0.05],
                              [0.10, 0.10, 0.80]])
    emit = torch.tensor([[0.85, 0.10, 0.05],
                            [0.10, 0.80, 0.10],
                            [0.05, 0.10, 0.85]])
    start = torch.tensor([0.5, 0.3, 0.2])

    def sample(T):
        s = int(np.random.choice(3, p=start.detach().cpu().numpy()))
        seq, path = [], [s]
        for _ in range(T):
            seq.append(int(np.random.choice(3, p=emit[s].detach().cpu().numpy())))
            if _ < T - 1:
                s = int(np.random.choice(3, p=trans[s].detach().cpu().numpy()))
                path.append(s)
        return seq, path

    seq, path = sample(300)
    hmm = GaussianHMM(n_states=3, n_emissions=3)
    hmm.fit_em(torch.tensor(seq, dtype=torch.int64), n_iter=80)
    pred_path = hmm.viterbi(torch.tensor(seq, dtype=torch.int64)).detach().cpu().numpy()
    # Modulo permutation of state labels
    acc_direct = (np.array(path) == pred_path).mean()
    best = 0.0
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        m = np.array([perm[s] for s in pred_path])
        best = max(best, (np.array(path) == m).mean())
    print(f"HMM viterbi accuracy (mod perm) = {max(acc_direct, best):.3f}")


if __name__ == "__main__":
    print("=== GP demo ===")
    gp_demo()
    print()
    print("=== HMM demo ===")
    hmm_demo()
