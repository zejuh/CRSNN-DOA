# Unified DOA Demo and Report

## 启动方式

在项目根目录安装依赖，然后启动本地 demo：

```powershell
pip install -r requirements.txt
python demo_server.py --open
```

如果想用 GPU 或指定端口：

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

如果只看 noisy 测试下的最优工作点：

- `ConvRecSNN`: `lambda=0.03`
- `FlatLIFSNN`: `lambda=1.0`

### 结果摘要

| 模型 | 选中 lambda | Noisy 准确率 | Noisy MAE (deg) | 说明 |
|---|---:|---:|---:|---|
| ConvRecSNN | 0.3 | 0.801 | 2.613 | 主 benchmark 中综合最强 |
| FlatLIFSNN | 0.1 | 0.742 | 3.628 | 更省算、更稀疏的 SNN |
| CRNNBaseline | 0.0 | 0.680 | 3.817 | ANN baseline |
| GCCPHATLSBaseline | 0.0 | 0.683 | 3.868 | 传统方法 baseline |

### 研究层面的解释

- `ConvRecSNN` 在 noisy 条件下准确率和角度误差都最好，鲁棒性最强。
- `FlatLIFSNN` 对 firing-rate regularization 的响应更明显，精度和效率之间的 trade-off 更清楚。
- 用 clean validation 选出来的 `lambda`，不一定就是 noisy test 下最优的工作点，因此 demo 同时保留了 `val-selected` 和 `noisy-best` 两组模型。

<details>
<summary><strong>展开查看详细结论</strong></summary>

### 实验设置

- 音频设置：16 kHz，4 麦克风，0.32 秒窗口
- 标签空间：36 个 DOA bins
- 特征：log-mel + GCC-PHAT
- 随机种子：`274, 275, 276`
- SNN 的 lambda sweep：`0, 0.03, 0.1, 0.3, 1.0`
- 验证集选型规则：先看 angular MAE，再看 accuracy，SNN 若接近则偏向更低 FR / SynOps

需要注意的是：

- validation 是 clean-only
- noisy robustness 是在 held-out noisy test split 上评估的
- 所以 validation 选出来的 `lambda`，不一定就是 noisy test 下最优的 `lambda`

### 主 benchmark 解释

1. `ConvRecSNN` 是这套 benchmark 中 noisy 条件下最强的模型。
   - 相比 `CRNNBaseline`，noisy accuracy 大约高 `12.2` 个百分点（`0.801` vs `0.680`）。
   - noisy angular MAE 相对下降约 `31.5%`（`2.61 deg` vs `3.82 deg`）。

2. `FlatLIFSNN` 是效率最好的 learned model。
   - 它被选中的 setting 只用了 `ConvRecSNN` 大约 `20.6%` 的 SynOps。
   - 也就是大约 `79.4%` 更低的 SynOps，同时保留了约 `92.6%` 的 noisy accuracy。

3. learned models 在 clean split 上已经比较饱和。
   - 真正拉开差距的是 noisy split，不是 clean split。

### Lambda 结果

#### ConvRecSNN

- noisy test accuracy 在 `lambda = 0.03` 时最高。
- 但 validation-based selection 选出来的是 `lambda = 0.3`。
- 整体上 SynOps 变化不大，说明这个模型对当前 FR regularizer 的敏感度较弱。

#### FlatLIFSNN

- `FlatLIFSNN` 对 `lambda` 的稀疏性响应明显得多。
- 从 `lambda = 0.0` 到 `lambda = 1.0`：
  - FR 下降约 `39.5%`
  - SynOps 下降约 `40.0%`
  - noisy accuracy 提升约 `5.46` 个百分点
  - noisy MAE 改善约 `0.85 deg`
- 但即便如此，validation-based selection 仍然选的是 `lambda = 0.1`，而不是 `1.0`。

### Research Extension

notebook 还补充了一个从 `-10 dB` 到 `20 dB` 的 fixed-SNR robustness sweep。

- `ConvRecSNN` 在整个 SNR sweep 中最稳健。
- `FlatLIFSNN` 在非负 SNR 下整体仍优于 `CRNNBaseline`。
- `GCCPHATLSBaseline` 在 `0 dB` 附近还能作为参考，但随着 SNR 变高会明显落后于 learned SNN。

### 局限性

- 数据仍然是合成的 4 麦克风空间化，不是真实阵列录音。
- validation 只有 clean，可能会让 lambda selection 偏离 noisy robustness。
- SynOps 是近似 proxy，不是真实硬件能耗。
- SNR robustness extension 是单 seed 结果，更适合作为补充证据。

</details>

## 目录说明

- [demo_server.py](./demo_server.py)：单文件 demo server，包含模型加载、特征提取、推理和 HTTP 接口
- [requirements.txt](./requirements.txt)：demo 运行依赖
- [demo/](./demo/)：前端页面资源
- [runs/unified_doa_notebook/checkpoints/](./runs/unified_doa_notebook/checkpoints/)：可切换的模型 checkpoint
- [snn_doa_unified_benchmark.ipynb](./snn_doa_unified_benchmark.ipynb)：主实验 notebook
