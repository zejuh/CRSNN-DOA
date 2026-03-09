# Unified DOA Demo and Report

## 启动方式

在项目根目录安装依赖，然后启动本地演示页面：

```powershell
pip install -r requirements.txt
python demo_server.py --open
```

如果想使用 GPU 或指定端口：

```powershell
python demo_server.py --device cuda --port 8000 --open
```

## Demo 内容

页面支持切换以下模型：

- `ConvRecSNN (val-selected, lambda=3e-01)`
- `ConvRecSNN (noisy-best, lambda=3e-02)`
- `FlatLIFSNN (val-selected, lambda=1e-01)`
- `FlatLIFSNN (noisy-best, lambda=1e+00)`
- `CRNNBaseline`
- `GCCPHATLSBaseline`

点击监听者周围任意位置后，页面会：

1. 播放当前选中的 SpeechCommands 语音样本。
2. 将音频空间化到 4 麦克风阵列。
3. 用当前选中的模型做 DOA 预测。
4. 实时更新预测角度、置信度、绝对误差和角度分布。

## 实验结论

### 综合最优的 SNN

- 综合最强、最鲁棒的 SNN：`ConvRecSNN, lambda=0.3`
- 更强调效率和稀疏性的折中方案：`FlatLIFSNN, lambda=0.1`

如果只看含噪测试下的最优工作点：

- `ConvRecSNN`: `lambda=0.03`
- `FlatLIFSNN`: `lambda=1.0`

### 结果摘要

| 模型 | 选中 lambda | 含噪准确率 | 含噪角度 MAE (deg) | 说明 |
|---|---:|---:|---:|---|
| ConvRecSNN | 0.3 | 0.801 | 2.613 | 主基准实验中综合最强 |
| FlatLIFSNN | 0.1 | 0.742 | 3.628 | 更省算、更稀疏的 SNN |
| CRNNBaseline | 0.0 | 0.680 | 3.817 | ANN 基线 |
| GCCPHATLSBaseline | 0.0 | 0.683 | 3.868 | 传统方法基线 |

### 研究层面的解释

- `ConvRecSNN` 在含噪条件下准确率和角度误差都最好，鲁棒性最强。
- `FlatLIFSNN` 对发放率正则化（firing-rate regularization）的响应更明显，精度和效率之间的权衡也更清楚。
- 用干净验证集选出来的 `lambda`，不一定就是含噪测试集下最优的工作点，因此 demo 同时保留了 `val-selected` 和 `noisy-best` 两组模型。

<details>
<summary><strong>展开查看详细结论</strong></summary>

### 实验设置

- 音频设置：16 kHz，4 麦克风，0.32 秒窗口
- 标签空间：36 个 DOA 分箱
- 特征：log-mel + GCC-PHAT
- 随机种子：`274, 275, 276`
- SNN 的 lambda 扫描：`0, 0.03, 0.1, 0.3, 1.0`
- 验证集选型规则：先看角度 MAE，再看准确率；如果 SNN 接近，则偏向更低的 FR / SynOps

需要注意的是：

- 验证集是纯干净版本
- 含噪鲁棒性是在留出的含噪测试集上评估的
- 所以验证集选出来的 `lambda`，不一定就是含噪测试集下最优的 `lambda`

### 补充基准实验观察

- 学习式模型在干净测试划分上的表现已经比较饱和。
  - 真正拉开差距的是含噪测试划分，不是干净测试划分。
- 传统方法基线只能算有限竞争。
  - `GCCPHATLSBaseline` 在含噪准确率上和 `CRNNBaseline` 接近。
  - 但在角度误差上仍明显落后于 `ConvRecSNN`，也落后于 `FlatLIFSNN`。

### Lambda 结果

#### ConvRecSNN

含噪测试扫描：

| lambda | acc_mean | ang_mae_deg_mean | fr_mean | synops_per_sample_mean |
|---:|---:|---:|---:|---:|
| 0.00 | 0.798 | 2.691 | 0.0584 | 1.215e6 |
| 0.03 | 0.845 | 2.592 | 0.0608 | 1.266e6 |
| 0.10 | 0.804 | 2.831 | 0.0583 | 1.212e6 |
| 0.30 | 0.801 | 2.613 | 0.0564 | 1.172e6 |
| 1.00 | 0.823 | 2.610 | 0.0587 | 1.220e6 |

解释：

- `ConvRecSNN` 在这套设定里对发放率正则化的敏感度较弱。
- SynOps 在整个扫描里变化很小，基本只是几个百分点的量级。
- 含噪测试准确率在 `lambda = 0.03` 时最高，但基于验证集的选型选的是 `lambda = 0.3`。
- 这说明当前的验证规则更偏向一个“略微更正则化”的模型，而不是含噪测试上绝对最优的模型。

#### FlatLIFSNN

含噪测试扫描：

| lambda | acc_mean | ang_mae_deg_mean | fr_mean | synops_per_sample_mean |
|---:|---:|---:|---:|---:|
| 0.00 | 0.745 | 3.570 | 0.4002 | 2.589e5 |
| 0.03 | 0.757 | 3.279 | 0.3929 | 2.533e5 |
| 0.10 | 0.742 | 3.628 | 0.3757 | 2.418e5 |
| 0.30 | 0.735 | 3.620 | 0.3353 | 2.168e5 |
| 1.00 | 0.800 | 2.721 | 0.2423 | 1.553e5 |

解释：

- `FlatLIFSNN` 对 `lambda` 的稀疏性响应比 `ConvRecSNN` 明显得多。
- 从 `lambda = 0.0` 到 `lambda = 1.0`：
  - FR 下降约 `39.5%`
  - SynOps 下降约 `40.0%`
  - 含噪准确率提升约 `5.46` 个百分点
  - 含噪角度 MAE 改善约 `0.85 deg`
- 即便如此，基于验证集的选型仍然选的是 `lambda = 0.1`，而不是 `1.0`。

#### Lambda 结果的含义

最重要的方法学发现是：基于干净验证集的选型，与含噪测试鲁棒性并不完全一致。

- 对 `ConvRecSNN`，验证集选的是 `lambda = 0.3`，但含噪测试准确率的峰值出现在 `0.03`。
- 对 `FlatLIFSNN`，验证集选的是 `lambda = 0.1`，但含噪测试上最强的结果出现在 `1.0`。

这说明后续如果继续迭代，应该考虑：

1. 使用含噪开发集做模型选择，
2. 用包含鲁棒性的多目标验证标准，
3. 或者分别保留“优先干净条件”和“优先含噪鲁棒性”两种工作点。

### Research Extension

notebook 还补充了两项轻量研究分析：

- 帕累托风格的最优设置比较
- 从 `-10 dB` 到 `20 dB` 的固定 SNR 鲁棒性测试

#### 帕累托风格的效率视角

| Model | Selected lambda | fp32 模型大小 (KB) | 含噪准确率 | 含噪角度 MAE |
|---|---:|---:|---:|---:|
| ConvRecSNN | 0.3 | 774.9 | 0.801 | 2.613 |
| FlatLIFSNN | 0.1 | 388.6 | 0.742 | 3.628 |
| CRNNBaseline | 0.0 | 1642.1 | 0.680 | 3.817 |
| GCCPHATLSBaseline | 0.0 | 0.0 | 0.683 | 3.868 |

解释：

- `CRNNBaseline` 是最大的学习式模型，但表现仍不如 `ConvRecSNN`。
- `FlatLIFSNN` 比 `CRNNBaseline` 提供了更有吸引力的尺寸与效率平衡点。
- `ConvRecSNN` 不是最小模型，但它提供了最好的含噪准确率。

#### 固定 SNR 鲁棒性测试

这个扩展会用代表性随机种子（`274`）和选中的 lambda 重新训练学习式模型，然后在 `-10 dB` 到 `20 dB` 的固定 SNR 下评估。

| Model | Acc @ 0 dB | Acc @ 10 dB | Acc @ 20 dB | Mean acc across all SNRs |
|---|---:|---:|---:|---:|
| ConvRecSNN | 0.596 | 0.839 | 0.930 | 0.660 |
| FlatLIFSNN | 0.463 | 0.705 | 0.846 | 0.556 |
| CRNNBaseline | 0.458 | 0.646 | 0.795 | 0.527 |
| GCCPHATLSBaseline | 0.464 | 0.661 | 0.738 | 0.513 |

在中高 SNR 下，`ConvRecSNN` 也具有最低的角度误差：

- `2.73 deg` at `10 dB`
- `2.21 deg` at `15 dB`
- `1.87 deg` at `20 dB`

关键观察：

1. `ConvRecSNN` 在整个 SNR 扫描中最稳健。
2. `FlatLIFSNN` 在非负 SNR 下整体仍优于 `CRNNBaseline`。
3. `GCCPHATLSBaseline` 在 `0 dB` 附近还能作为参考，但随着 SNR 提升会明显落后于学习式 SNN。

### 局限性

- 数据仍然是合成的 4 麦克风空间化，不是真实阵列录音。
- 验证集只有干净版本，可能会让 lambda 选型偏离含噪鲁棒性。
- SynOps 只是近似指标，不是真实硬件能耗。
- SNR 鲁棒性扩展是单个随机种子的结果，更适合作为补充证据，而不是主基准实验表格本身。

</details>

## 目录说明

- [demo_server.py](./demo_server.py)：单文件 demo server，包含模型加载、特征提取、推理和 HTTP 接口
- [requirements.txt](./requirements.txt)：demo 运行依赖
- [demo/](./demo/)：前端页面资源
- [runs/unified_doa_notebook/checkpoints/](./runs/unified_doa_notebook/checkpoints/)：可切换的模型 checkpoint
- [snn_doa_unified_benchmark.ipynb](./snn_doa_unified_benchmark.ipynb)：主实验 notebook
