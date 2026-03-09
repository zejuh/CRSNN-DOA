# Unified DOA Demo and Report

[English](./README.md) | [中文](./README.zh-CN.md)

## Quick Start

Install dependencies from the project root, then launch the local demo:

```powershell
pip install -r requirements.txt
python demo_server.py --open
```

To run on GPU or use a different port:

```powershell
python demo_server.py --device cuda --port 8000 --open
```

## Demo Overview

The demo supports switching between these models:

- `ConvRecSNN (val-selected, lambda=3e-01)`
- `ConvRecSNN (noisy-best, lambda=3e-02)`
- `FlatLIFSNN (val-selected, lambda=1e-01)`
- `FlatLIFSNN (noisy-best, lambda=1e+00)`
- `CRNNBaseline`
- `GCCPHATLSBaseline`

When you click anywhere around the listener, the page will:

1. Play the selected SpeechCommands sample.
2. Spatialize it to a 4-microphone array.
3. Run DOA prediction with the currently selected model.
4. Update the predicted angle, confidence, absolute error, and angle distribution live.

## Main Findings

### Best overall SNN

- Best robust SNN: `ConvRecSNN, lambda=0.3`
- Best efficiency-oriented SNN: `FlatLIFSNN, lambda=0.1`

If you only care about the best noisy operating point:

- `ConvRecSNN`: `lambda=0.03`
- `FlatLIFSNN`: `lambda=1.0`

### Summary Table

| Model | Selected lambda | Noisy acc | Noisy MAE (deg) | Comment |
|---|---:|---:|---:|---|
| ConvRecSNN | 0.3 | 0.801 | 2.613 | Strongest overall benchmark model |
| FlatLIFSNN | 0.1 | 0.742 | 3.628 | Better efficiency / sparsity trade-off |
| CRNNBaseline | 0.0 | 0.680 | 3.817 | ANN baseline |
| GCCPHATLSBaseline | 0.0 | 0.683 | 3.868 | Classical baseline |

### Interpretation

- `ConvRecSNN` is the most robust model under noisy conditions.
- `FlatLIFSNN` responds more clearly to firing-rate regularization and shows a cleaner accuracy-efficiency trade-off.
- The best `lambda` under clean validation is not always the best operating point under noisy test conditions, so the demo keeps both `val-selected` and `noisy-best` checkpoints.

## Repository Structure

- [demo_server.py](./demo_server.py): single-file demo server with model loading, feature extraction, inference, and HTTP endpoints
- [requirements.txt](./requirements.txt): runtime dependencies for the demo
- [demo/](./demo/): frontend assets
- [runs/unified_doa_notebook/checkpoints/](./runs/unified_doa_notebook/checkpoints/): switchable model checkpoints
- [snn_doa_unified_benchmark.ipynb](./snn_doa_unified_benchmark.ipynb): main experiment notebook
