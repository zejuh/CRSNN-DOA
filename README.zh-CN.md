# Unified DOA Demo and Report

[English](./README.md) | [中文](./README.zh-CN.md)

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

## 目录说明

- [demo_server.py](./demo_server.py)：单文件 demo server，包含模型加载、特征提取、推理和 HTTP 接口
- [requirements.txt](./requirements.txt)：demo 运行依赖
- [demo/](./demo/)：前端页面资源
- [runs/unified_doa_notebook/checkpoints/](./runs/unified_doa_notebook/checkpoints/)：可切换的模型 checkpoint
- [snn_doa_unified_benchmark.ipynb](./snn_doa_unified_benchmark.ipynb)：主实验 notebook
