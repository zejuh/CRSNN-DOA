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

<details>
<summary><strong>Detailed benchmark notes</strong></summary>

### Experimental setup

- Audio setup: 16 kHz, 4 microphones, 0.32 s window
- Label space: 36 DOA bins
- Features: log-mel + GCC-PHAT
- Seeds: `274, 275, 276`
- Lambda sweep for SNNs: `0, 0.03, 0.1, 0.3, 1.0`
- Validation selection rule: angular MAE first, then accuracy, then lower FR / SynOps for SNN tie-breaking

Important note:

- Validation is clean-only.
- Noisy robustness is evaluated on the held-out noisy test split.
- Therefore, the validation-selected `lambda` is not always the same `lambda` that maximizes noisy test performance.

### Main benchmark interpretation

1. `ConvRecSNN` is the strongest noisy-condition model in the benchmark.
   - It exceeds `CRNNBaseline` by about `12.2` percentage points in noisy accuracy (`0.801` vs `0.680`).
   - It reduces noisy angular MAE by about `31.5%` relative to `CRNNBaseline` (`2.61 deg` vs `3.82 deg`).

2. `FlatLIFSNN` is the most efficient learned model.
   - Its selected setting uses only about `20.6%` of `ConvRecSNN` SynOps.
   - That corresponds to about `79.4%` lower SynOps while retaining about `92.6%` of `ConvRecSNN` noisy accuracy.

3. Clean performance is largely saturated for learned models.
   - The real separation appears in the noisy split rather than the clean split.

### Lambda study

#### ConvRecSNN

- Noisy test accuracy peaks at `lambda = 0.03`.
- Validation-based selection chooses `lambda = 0.3`.
- SynOps changes are relatively small across the sweep, so this model is only weakly sensitive to the current FR regularizer.

#### FlatLIFSNN

- `FlatLIFSNN` shows a much clearer sparsity response to `lambda`.
- From `lambda = 0.0` to `lambda = 1.0`:
  - FR decreases by about `39.5%`
  - SynOps decreases by about `40.0%`
  - Noisy accuracy increases by about `5.46` percentage points
  - Noisy MAE improves by about `0.85 deg`
- Despite this, validation-based selection chooses `lambda = 0.1` rather than `1.0`.

### Research extension

The notebook also includes a fixed-SNR robustness sweep from `-10 dB` to `20 dB`.

- `ConvRecSNN` is the most robust model across the SNR sweep.
- `FlatLIFSNN` remains consistently better than `CRNNBaseline` at non-negative SNRs.
- `GCCPHATLSBaseline` can be competitive around `0 dB`, but it falls behind the learned SNNs as SNR improves.

### Limitations

- The data is synthetic 4-microphone spatialization rather than real array recordings.
- Validation is clean-only, which likely biases lambda selection away from noisy robustness.
- SynOps is an approximate compute proxy, not a hardware-measured energy metric.
- The SNR robustness extension is single-seed and should be treated as supporting evidence.

</details>

## Repository Structure

- [demo_server.py](./demo_server.py): single-file demo server with model loading, feature extraction, inference, and HTTP endpoints
- [requirements.txt](./requirements.txt): runtime dependencies for the demo
- [demo/](./demo/): frontend assets
- [runs/unified_doa_notebook/checkpoints/](./runs/unified_doa_notebook/checkpoints/): switchable model checkpoints
- [snn_doa_unified_benchmark.ipynb](./snn_doa_unified_benchmark.ipynb): main experiment notebook
