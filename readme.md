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

### Additional benchmark observations

- Clean performance is largely saturated for learned models.
  - The real separation appears in the noisy split rather than the clean split.
- The classical baseline is competitive only in a narrow sense.
  - `GCCPHATLSBaseline` stays close to `CRNNBaseline` on noisy accuracy.
  - It remains clearly below `ConvRecSNN` and below `FlatLIFSNN` in angular error.

### Lambda study

#### ConvRecSNN

Noisy sweep:

| lambda | acc_mean | ang_mae_deg_mean | fr_mean | synops_per_sample_mean |
|---:|---:|---:|---:|---:|
| 0.00 | 0.798 | 2.691 | 0.0584 | 1.215e6 |
| 0.03 | 0.845 | 2.592 | 0.0608 | 1.266e6 |
| 0.10 | 0.804 | 2.831 | 0.0583 | 1.212e6 |
| 0.30 | 0.801 | 2.613 | 0.0564 | 1.172e6 |
| 1.00 | 0.823 | 2.610 | 0.0587 | 1.220e6 |

Interpretation:

- `ConvRecSNN` is only weakly sensitive to FR regularization in this setup.
- SynOps changes are small, on the order of only a few percent across the sweep.
- Noisy test accuracy is highest at `lambda = 0.03`, but validation-based selection chooses `lambda = 0.3`.
- This means the current validation criterion prefers a slightly more regularized model than the one that maximizes noisy test accuracy.

#### FlatLIFSNN

Noisy sweep:

| lambda | acc_mean | ang_mae_deg_mean | fr_mean | synops_per_sample_mean |
|---:|---:|---:|---:|---:|
| 0.00 | 0.745 | 3.570 | 0.4002 | 2.589e5 |
| 0.03 | 0.757 | 3.279 | 0.3929 | 2.533e5 |
| 0.10 | 0.742 | 3.628 | 0.3757 | 2.418e5 |
| 0.30 | 0.735 | 3.620 | 0.3353 | 2.168e5 |
| 1.00 | 0.800 | 2.721 | 0.2423 | 1.553e5 |

Interpretation:

- `FlatLIFSNN` shows a much clearer sparsity response to `lambda` than `ConvRecSNN`.
- From `lambda = 0.0` to `lambda = 1.0`:
  - FR decreases by about `39.5%`
  - SynOps decreases by about `40.0%`
  - Noisy accuracy increases by about `5.46` percentage points
  - Noisy MAE improves by about `0.85 deg`
- Despite this, validation-based selection chooses `lambda = 0.1` rather than `1.0`.

#### Implication of the lambda results

The most important methodological finding is that clean-validation-based selection does not fully align with noisy-test robustness.

- For `ConvRecSNN`, validation chooses `lambda = 0.3` while noisy test accuracy peaks at `0.03`.
- For `FlatLIFSNN`, validation chooses `lambda = 0.1` while the strongest noisy test result appears at `1.0`.

This suggests that future iterations should consider:

1. a noisy development split for model selection,
2. a multi-objective validation criterion that includes robustness, or
3. separate clean-priority and noise-robustness-priority operating points.

### Research extension

The notebook also includes:

- a Pareto-style best-setting comparison,
- a fixed-SNR robustness sweep from `-10 dB` to `20 dB`.

#### Pareto-style efficiency view

| Model | Selected lambda | fp32 model size (KB) | Noisy acc | Noisy MAE |
|---|---:|---:|---:|---:|
| ConvRecSNN | 0.3 | 774.9 | 0.801 | 2.613 |
| FlatLIFSNN | 0.1 | 388.6 | 0.742 | 3.628 |
| CRNNBaseline | 0.0 | 1642.1 | 0.680 | 3.817 |
| GCCPHATLSBaseline | 0.0 | 0.0 | 0.683 | 3.868 |

Interpretation:

- `CRNNBaseline` is the largest learned model and still underperforms `ConvRecSNN`.
- `FlatLIFSNN` offers a more attractive size-efficiency point than `CRNNBaseline`.
- `ConvRecSNN` is not the smallest model, but it offers the best noisy accuracy.

#### Fixed-SNR robustness sweep

This extension retrains the learned models once on a representative seed (`274`) using the selected lambda and evaluates them at fixed SNR values from `-10 dB` to `20 dB`.

| Model | Acc @ 0 dB | Acc @ 10 dB | Acc @ 20 dB | Mean acc across all SNRs |
|---|---:|---:|---:|---:|
| ConvRecSNN | 0.596 | 0.839 | 0.930 | 0.660 |
| FlatLIFSNN | 0.463 | 0.705 | 0.846 | 0.556 |
| CRNNBaseline | 0.458 | 0.646 | 0.795 | 0.527 |
| GCCPHATLSBaseline | 0.464 | 0.661 | 0.738 | 0.513 |

At moderate to high SNR, `ConvRecSNN` also has the lowest angular error:

- `2.73 deg` at `10 dB`
- `2.21 deg` at `15 dB`
- `1.87 deg` at `20 dB`

Key observations:

1. `ConvRecSNN` is the most robust model across the full SNR sweep.
2. `FlatLIFSNN` remains consistently better than `CRNNBaseline` at non-negative SNRs.
3. `GCCPHATLSBaseline` can be competitive around `0 dB`, but it falls behind the learned SNNs as SNR improves.

### Limitations

- The data is synthetic 4-microphone spatialization rather than real array recordings.
- Validation is clean-only, which likely biases lambda selection away from noisy robustness.
- SynOps is an approximate compute proxy, not a hardware-measured energy metric.
- The SNR robustness extension is single-seed and should be treated as supporting evidence rather than the primary benchmark table.

</details>

## Repository Structure

- [demo_server.py](./demo_server.py): single-file demo server with model loading, feature extraction, inference, and HTTP endpoints
- [requirements.txt](./requirements.txt): runtime dependencies for the demo
- [demo/](./demo/): frontend assets
- [runs/unified_doa_notebook/checkpoints/](./runs/unified_doa_notebook/checkpoints/): switchable model checkpoints
- [snn_doa_unified_benchmark.ipynb](./snn_doa_unified_benchmark.ipynb): main experiment notebook
