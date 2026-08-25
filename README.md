# DifferentiableMachineLearning

A **classical statistics / scientific computing toolkit** built on top of PyTorch.

> Core idea: re-implement traditional ML, signal processing, and probabilistic
> inference methods as **`torch.nn.Module`** so they compose with neural
> networks, participate in autograd, and run on GPU like any other layer.

This is **not** just a deep-learning toolkit. The whole point of
`DifferentiableMachineLibrary` is: classical models are also parameters + structure,
and structure can be expressed with `torch.nn.Module` so it composes with
neural networks.

我们收录的方法都满足两条线：

1. **不与 PyTorch 内置功能重复**（CNN / RNN / Transformer / 各类优化器 / 损失函数 / 数据加载等都不在范围）。
2. **不是纯深度学习 pipeline**（图像分类、目标检测、分割等 end-to-end 模板不属于本工具包）。

具体收/不收规则见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## 子模块（按 8 大类组织）

### 1. 时序与状态空间模型
| Submodule | Contents |
| --- | --- |
| `TimeSeries` | `AR`, `ARMA`, `FIR`, `Autoregressive` |
| `ClassicalML` | `KalmanFilter`, `GaussianHMM`, `GMMHMM`, `tICA` |

### 2. 降维与流形学习
| Submodule | Contents |
| --- | --- |
| `ClassicalML` | `PCA`, `ICA`, `NMF`, `tSNE`, `ProbabilisticPCAMixture` |

### 3. 概率 / 生成模型
| Submodule | Contents |
| --- | --- |
| `ClassicalML` | `GMM`, `BayesianRidge`, `BayesianLinearVI`, `GaussianProcess`, `LinearChainCRF` |

### 4. 核方法
| Submodule | Contents |
| --- | --- |
| `ClassicalML` | `KernelRidge`, `SVM` |

### 5. 监督学习（分类 / 回归）
| Submodule | Contents |
| --- | --- |
| `ClassicalML` | `LDA`, `GaussianNB`, `MultinomialNB`, `SoftDecisionTree` |

### 6. 聚类
| Submodule | Contents |
| --- | --- |
| `ClassicalML` | `KMeans`, `KNN` |

### 7. 稀疏表示
| Submodule | Contents |
| --- | --- |
| `ClassicalML` | `SparseCoding` |

### 8. 信号处理
| Submodule | Contents |
| --- | --- |
| `SignalProcessing` | `STFT`/`ISTFT` (可学窗), `MelSpectrogram`, `WaveletFilterBank` |

每个组件的类 docstring 顶部都注明"对标方法"（sklearn / statsmodels / 经典文献），方便查证。

## 安装

```bash
git clone https://github.com/Liyulingyue/DifferentiableMachineLearning.git
cd DifferentiableMachineLearning
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -e .
```

依赖：`torch>=2.0`（CPU 版即可）。

## 快速上手

### 1. 时序与状态空间模型

```python
import torch
from DifferentiableMachineLearning.TimeSeries import AR, ARMA, FIR, Autoregressive

# 通用 A(p) y = B(q) u + C(o) v
model = Autoregressive(y_features=3, x_features=[2, 4], e_features=2)
y = torch.randn(8, 3)
u1, u2, v = torch.randn(8, 2), torch.randn(8, 4), torch.randn(8, 2)
out = model(y, u1, u2, v)                      # [8, 1]

ar = AR(p=5)
y_seq = torch.randn(8, 5)
print(ar(y_seq).shape)                         # [8, 1]
```

```python
# Kalman filter / smoother
from DifferentiableMachineLearning.ClassicalML import KalmanFilter
kf = KalmanFilter(state_dim=2, obs_dim=2)
x_smooth = kf(y_seq)                           # [T, state_dim]   smoother 输出

# HMM / GMM-HMM
from DifferentiableMachineLearning.ClassicalML import GaussianHMM, GMMHMM
hmm = GaussianHMM(n_states=3, n_emissions=10)
hmm.fit_em(seq, n_iter=80)
gamma = hmm(seq)                               # [T, n_states]
path  = hmm.viterbi(seq)                       # [T] int64

ghmm = GMMHMM(n_states=3, n_components=2, n_features=4)
ghmm.fit_em(x_seq, n_iter=50)                  # 连续观测
gamma = ghmm(x_seq)
path  = ghmm.viterbi(x_seq)

# tICA 慢模
from DifferentiableMachineLearning.ClassicalML import tICA
tica = tICA(n_components=3, dim=D, lag=20).fit(X)
slow = tica.transform(X)                        # [T, n_components]
```

### 2. 降维与流形学习

```python
from DifferentiableMachineLearning.ClassicalML import PCA, ICA, NMF, tSNE, ProbabilisticPCAMixture

# PCA: 正交基是 nn.Parameter，每步 QR 重新正交化
pca = PCA(n_components=2, dim=5).fit(X)
coords = pca.project(X)                        # [N, 2]
rec    = pca.reconstruct(coords)               # [N, 5]

# ICA: PCA 白化 + 对称 FastICA 定点迭代
ica = ICA(n_components=2, dim=2, nonlinearity="cube").fit(X)
sources = ica.transform(X)
recon   = ica.inverse_transform(sources)

# NMF: Lee-Seung 乘法更新；非负字典 W + 非负激活 H
nmf = NMF(n_components=10, n_features=D, init="nndsvd")
nmf.fit(X, n_iter=200)
H_new = nmf.transform(X)
rec   = nmf.reconstruct(H_new)

# t-SNE: Adam-based KL minimisation with early exaggeration
tsne = tSNE(n_components=2, perplexity=30.0, n_iter=500)
Y = tsne.fit_transform(X)                       # [N, 2]

# Mixture of PPCA
ppca = ProbabilisticPCAMixture(n_components=K, n_features=D, n_latent=L)
ppca.fit_em(X, n_iter=30)
gamma = ppca(X)                                # [N, K]
```

### 3. 概率 / 生成模型

```python
from DifferentiableMachineLearning.ClassicalML import GMM, BayesianRidge, BayesianLinearVI, GaussianProcess, LinearChainCRF

# GMM: 软责任可微，EM 闭式更新
gmm = GMM(k=3, dim=2, covariance_type="diag").fit_em(X, n_iter=20)
resp = gmm(X_test)                             # [N, 3]

# Bayesian Ridge: EM 更新 alpha + lambda，返回 predictive mean + std
br = BayesianRidge(n_features=3, n_outputs=1).fit(X, y)
mean, std = br.forward_with_std(X_test)

# Bayesian linear regression with mean-field VI
blr = BayesianLinearVI(n_features=3, n_outputs=1, prior_std=1.0, noise_std=0.1)
loss = blr.neg_elbo(x, y, n_samples=1)         # -ELBO
mean, std = blr.predict(x_test, n_samples=80)

# GP regression: RBF / Matérn / linear / poly
gp = GaussianProcess(dim=1, n_train=N, kernel="rbf").fit(X, y)
mean, std = gp.forward_with_std(X_test)
loss = gp.neg_log_marginal_likelihood()

# Linear-chain CRF
crf = LinearChainCRF(n_features=5, n_tags=3)
loss = crf.nll(features, tags)
gamma = crf(features)
path  = crf.decode(features)
```

### 4. 核方法

```python
from DifferentiableMachineLearning.ClassicalML import KernelRidge, SVM

# KernelRidge: 对偶形式 + 闭式解
kr = KernelRidge(n_support=200, dim_in=1, dim_out=1,
                 kernel="rbf", gamma=0.5, alpha=1e-2)
kr.fit(x_train, y_train)
pred = kr(x_test)

# LS-SVM: 对偶闭式线性系统；RBF / linear / poly；OvR 多分类
svm = SVM(n_support=80, dim=2, n_classes=2, kernel="rbf", gamma=0.3, C=10.0).fit(X, Y)
acc = (svm.predict(X) == Y).float().mean()
```

### 5. 监督学习（分类 / 回归）

```python
from DifferentiableMachineLearning.ClassicalML import LDA, GaussianNB, MultinomialNB, SoftDecisionTree

# LDA: 闭式特征分解；投影 + 可微高斯判别 log p(y|x)
lda = LDA(n_components=2, dim=2, n_classes=3).fit(X, Y)
logits = lda(X)                                # [N, 3]

# Naive Bayes
gnb = GaussianNB(dim=2, n_classes=2).fit(X, Y)
mnb = MultinomialNB(n_features=6, n_classes=2, alpha=1.0).fit(counts, Y)

# SoftDecisionTree
tree = SoftDecisionTree(depth=4, n_features=2, n_classes=2, temperature=1.5)
# train with Adam like any torch layer
```

### 6. 聚类

```python
from DifferentiableMachineLearning.ClassicalML import KMeans, KNN

# KMeans: 软分配路径可微
km = KMeans(k=4, dim=3, temperature=0.5)
km.fit_kmeanspp(X)
km.fit(X, n_iter=10)
soft = km(X_test)                              # [N, k]
hard = km(X_test, hard=True)                   # [N]

# KNN: 记忆层
nn = KNN(k=3, dim=4)
nn.update_memory(torch.randn(50, 4))
idx = nn(X_test, mode="indices")                # [N, 3]
avg = nn(X_test, mode="average")                # [N, dim]
```

### 7. 稀疏表示

```python
from DifferentiableMachineLearning.ClassicalML import SparseCoding

sc = SparseCoding(n_atoms=20, n_features=16, lmbda=0.05,
                  encoder="fista", n_iter=200)
sc.fit(X, n_outer=200, lr=5e-2)                # 训练字典
z, x_hat = sc(X)                              # 编码 + 重建
```

### 8. 信号处理

```python
import torch
from DifferentiableMachineLearning.SignalProcessing import STFT, ISTFT, MelSpectrogram, WaveletFilterBank

# STFT / iSTFT：分析窗默认 Hann，可学习
stft = STFT(win_length=400, hop_length=160, n_fft=512)
istft = ISTFT(win_length=400, hop_length=160, n_fft=512, center=True)
spec = stft(x)                                  # [B, F, T]   complex
x_rec = istft(spec)

# Log-Mel spectrogram
mel = MelSpectrogram(n_mels=40, sample_rate=16000,
                     win_length=400, hop_length=160, n_fft=512)
features = mel(x)                               # [B, 40, T]

# Wavelet filter bank
wfb = WaveletFilterBank(n_scales=3, filter_length=2, learnable=True)
details = wfb(x)                                # list of 3 tensors
```

## 设计理念

* **可微**：`KMeans` 在训练模式下输出软分配（带 temperature 的 softmax），
  质心作为 `nn.Parameter` 接收梯度。`KNN.soft_retrieval` 也提供
  基于距离的 softmax 邻居权重。
* **可组合**：`KMeans` 输出一个 K 维软分配向量，可以直接喂给
  `nn.Linear`；`KNN.soft_retrieval` 的权重可与注意力分数等价使用。
* **可降级**：所有可微路径都对应一个不可微的"硬"等价物（`hard=True`、
  `mode="indices"` / `"average"` / `"values"`），保证推理阶段行为
  与传统算法一致。

完整示例见 `examples/`：

* `examples/timeseries_ar.py` —— AR(2) 端到端回归
* `examples/classicalml_kmeans_classifier.py` —— KMeans 软特征 + 线性分类器联合训练
* `examples/classicalml_gmm_classifier.py` —— GMM 软责任 + 线性分类器联合训练
* `examples/classicalml_soft_tree_moons.py` —— 软决策树在 moons 数据集上训练
* `examples/signalproc_mel_classifier.py` —— log-Mel + 线性头区分纯音频率
* `examples/gp_and_hmm_demo.py` —— GP 回归调优超参数 + 3 状态 HMM 路径恢复
* `examples/lds_crf_tsne_demo.py` —— Kalman 2D 跟踪 + CRF 玩具标注 + tSNE 5D→2D
* `examples/sparse_coding_and_vi_demo.py` —— SparseCoding 字典学习 + BayesianLinearVI 噪声回归
* `examples/tica_and_nmf_demo.py` —— tICA 慢模投影 + NMF 3-主题分解
* `examples/gmmhmm_and_ppca_demo.py` —— GMMHMM 3-state 恢复 + PPCA 混合 4-聚类
* `examples/sparse_gp_spectral_mrf_demo.py` —— SparseGP 变分推断 + SpectralClustering 谱聚类 + RBM CD-k + IsingModel 相变

## 测试

```bash
.\.venv\Scripts\python.exe -m pytest tests/
```

78 个单元测试覆盖 26 个组件的形状、关键行为与数值正确性。

## 许可

Apache-2.0