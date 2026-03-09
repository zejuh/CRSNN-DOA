# Unified DOA Demo and Report

## 启动方式

先自己决定是否创建虚拟环境，然后在项目根目录安装依赖并启动：

```powershell
pip install -r requirements.txt
python demo_server.py --open
```

如果想用 GPU 或指定端口：

```powershell
python demo_server.py --device cuda --port 8000 --open
```

## Demo 内容

页面支持直接切换这些模型：

- `ConvRecSNN (val-selected, lambda=3e-01)`
- `ConvRecSNN (noisy-best, lambda=3e-02)`
- `FlatLIFSNN (val-selected, lambda=1e-01)`
- `FlatLIFSNN (noisy-best, lambda=1e+00)`
- `CRNNBaseline`
- `GCCPHATLSBaseline`

点击声场中的任意方向后，页面会：

1. 播放所选 SpeechCommands 音频
2. 将其空间化到 4 麦克风阵列
3. 用当前选中的模型做 DOA 预测
4. 实时显示预测角度、误差、置信度和角度分布

## 实验结论

### 主结论

- 综合最强的 SNN 方案是 `ConvRecSNN, lambda=0.3`
- 如果更看重效率，最好的折中方案是 `FlatLIFSNN, lambda=0.1`
- 如果只看 noisy 条件下的运行点：
  - `ConvRecSNN` 的更优操作点是 `lambda=0.03`
  - `FlatLIFSNN` 的更优操作点是 `lambda=1.0`

### 主要结果

| Model | Selected lambda | Noisy acc | Noisy MAE (deg) | Comment |
|---|---:|---:|---:|---|
| ConvRecSNN | 0.3 | 0.801 | 2.613 | 主 benchmark 最强 |
| FlatLIFSNN | 0.1 | 0.742 | 3.628 | 更省算的 SNN |
| CRNNBaseline | 0.0 | 0.680 | 3.817 | ANN baseline |
| GCCPHATLSBaseline | 0.0 | 0.683 | 3.868 | classical baseline |

### 研究层面的解释

- `ConvRecSNN` 在 noisy 条件下准确率和角度误差都最好，鲁棒性最强
- `FlatLIFSNN` 的 firing-rate regularization 响应更明显，精度和效率之间的 trade-off 更清楚
- 当前 clean-validation 选出来的 `lambda`，不一定是 noisy test 最优，因此 demo 里同时保留了 `val-selected` 和 `noisy-best` 两组模型

## 目录说明

- [demo_server.py](./demo_server.py): 单文件 demo server，包含模型加载、特征提取、推理和 HTTP 接口
- [requirements.txt](./requirements.txt): demo 运行依赖
- [demo/](./demo/): 前端页面
- [runs/unified_doa_notebook/checkpoints/](./runs/unified_doa_notebook/checkpoints/): 可切换的模型 checkpoint
- [snn_doa_unified_benchmark.ipynb](./snn_doa_unified_benchmark.ipynb): 主实验 notebook
