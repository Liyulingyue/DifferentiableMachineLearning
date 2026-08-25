# Contributing to DifferentiableMachineLearning

## 项目定位

`DifferentiableMachineLearning` 是 **PyTorch 之上的"经典统计学习 / 科学计算"扩展**：

- 现有 PyTorch 已经提供了完整的深度学习栈（CNN / RNN / Transformer / 各类优化器 / 损失函数 / 数据加载 / 分布式 / 推理部署等）。
- 现有 PyTorch **不覆盖**（或没以可微 `nn.Module` 形式覆盖）大量经典方法（GMM / HMM / 谱聚类 / 流形学习 / t-SNE / GP / 贝叶斯推断 / RBM / Ising / ...）。
- 我们的价值 = 把这些"在 PyTorch 中不容易做"的方法以 **`torch.nn.Module`** 形式封装出来，**可以像普通算子一样参与组网、梯度反传、GPU 加速**。

## 收什么（in scope）

我们收的方法应当同时满足：

1. **经典方法**：在 `scikit-learn` / `statsmodels` / 经典教材（PRML / ESL / Bishop / Murphy / Barber 等）中有明确对标，或者引用了一篇公认经典论文。
2. **PyTorch 未覆盖**：PyTorch 自身的算子 / layer / module 中**没有等价物**。
3. **不平凡**：实现不是单行代码（除了 trivial 转发包装）。
4. **能 `torch.nn.Module`-化**：参数作为 `nn.Parameter`、buffer、`register_buffer`；可微路径用 torch ops；不可微路径（EM、MCMC、Viterbi、SMO、CD-k）也在 `forward / fit` 中明确标注。
5. **可验证**：能在 1-2 个 toy 合成数据集上 demo 验证，且有单元测试覆盖。

按以下 8 大类组织（README 顶部表格有详细列举）：

- **时序与状态空间模型**：AR / ARMA / FIR / Kalman / HMM / GMM-HMM / tICA ...
- **降维与流形学习**：PCA / ICA / NMF / t-SNE / PPCA-mixture / Isomap / LLE / LaplacianEigenmaps / SpectralClustering ...
- **概率 / 生成模型**：GMM / BayesianRidge / BayesianLinearVI / GP / SparseGP / LinearChainCRF ...
- **核方法**：KernelRidge / SVM / GP ...
- **监督学习**：LDA / Naive Bayes / SoftDecisionTree ...
- **聚类**：KMeans / KNN / SpectralClustering ...
- **稀疏表示**：SparseCoding / NMF / ...
- **信号处理**：STFT / ISTFT / MelSpectrogram / Wavelet / ...
- **图模型 / MCMC**：RBM / Ising / Sequential Monte Carlo / ...

## 不收什么（out of scope）

以下**一律不收**，即使它们可以用 `torch.nn.Module` 实现：

1. ❌ **直接包装 PyTorch 已有的算子 / layer**（`MyConv2D = torch.nn.Conv2d`、`MyLSTM = torch.nn.LSTM`）—— 包装增加抽象但没有新方法。
2. ❌ **经典深度学习模型**：ResNet / VGG / U-Net / Transformer / ViT / GAN / VAE / Normalizing Flow / DDPM / 任何主流 NN 架构 —— 这属于深度学习自身栈，请直接用 PyTorch 现有的 `torchvision` / 自定义 `nn.Module`。
3. ❌ **端到端 DL 任务模板**：图像分类 / 目标检测 / 语义分割 / 文本分类 / 命名实体识别 / 机器翻译 / 语音识别 / 推荐 end-to-end pipeline —— 这属于应用工程，不是"科学方法"。
4. ❌ **大规模数据预训练 / 自监督 / 对比学习 / RL 方法** —— 这类方法需要专门的数据集与训练流程，与本工具包定位不符。
5. ❌ **纯图论 / 离散优化 / 拓扑数据分析**（如最大流、TSP 求解、持续同调）—— 范畴太远。
6. ❌ **黑盒方法**：没有 toy 合成数据 demo 验证的、必须依赖外部特定数据集的。

> 一句话规则：**本项目只收"经典方法 + torch 不支持 + 我们用 `nn.Module` 重新实现一遍有学术/工程价值"的组件。**

## 如何贡献一个新组件

请按以下步骤：

1. **先开 issue 讨论**：提议要加的方法 + 对标（sklearn/论文）+ 为何属于 in scope + 与已有组件的关系。
2. **实现要求**：
   - 文件路径：`DifferentiableMachineLearning/<Submodule>/<ClassName>.py`，并在 `<Submodule>/__init__.py` 中 export。
   - 类 docstring 顶部必须有 `Analogue:` 一行，对标 `sklearn.X.Y` / `statsmodels.X.Y` / 论文标题+作者+年份。
   - 类继承 `torch.nn.Module`；参数（学习的 / 缓存的）按类型分清楚：`nn.Parameter` / `register_buffer` / 临时 tensor。
   - 提供 `forward`（可微路径）和 `fit_*`（不可微/EM/MCMC 路径）；两者最好并存。
   - **必须能在 1-2 个 toy 数据上跑通**并打印数值结果。
3. **测试**：`tests/test_<submodule>_<feature>.py`，每个新方法至少 3 个测试用例（形状、关键行为、梯度/数值正确性）。在 README 表格 + examples/ 中加一段用法。
4. **示例**：`examples/<feature>_demo.py` 必须可独立运行、打印关键结果。
5. **跑 `python -m pytest tests/`** 确保所有新旧测试通过。
6. **commit 之前** 跑一次 README 中的所有 `examples/*.py` 验证可执行。

## 命名约定

- 类名用 CamelCase，对应 sklearn / 论文里的方法名。
- 子模块名用 PascalCase（`ClassicalML`、`SignalProcessing`、`TimeSeries`）。
- 方法名 snake_case（`fit_em`, `forward_with_std`, `neg_log_marginal_likelihood`）。
- 不可微的拟合/EM/MCMC 方法用 `fit_<algo>` 前缀；可微的前向用 `forward`。

## 提问

不确定一个新方法该不该收？请开 issue 一起讨论。