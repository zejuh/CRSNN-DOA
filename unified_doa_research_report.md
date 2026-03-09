# Unified DOA Benchmark Research Report

## 1. Experimental Setup

This study benchmarks spiking and non-spiking models for synthetic 4-microphone direction-of-arrival (DOA) estimation built from SpeechCommands clips.

- Audio setup: 16 kHz, 4 microphones, 0.32 s window
- Label space: 36 DOA bins
- Features: log-mel + GCC-PHAT
- Models:
  - `ConvRecSNN`
  - `FlatLIFSNN`
  - `CRNNBaseline`
  - `GCCPHATLSBaseline`
- Training protocol:
  - 3 seeds: `274, 275, 276`
  - Lambda sweep for SNNs: `0, 0.03, 0.1, 0.3, 1.0`
  - Best setting chosen by aggregated validation metrics: angular MAE first, then accuracy, then lower FR / SynOps for SNN tie-breaking

Important protocol note:

- Validation is clean (`val_noise_prob = 0.0`)
- Noisy robustness is evaluated on the held-out noisy test split
- Therefore, validation-selected lambda is not necessarily the same lambda that maximizes noisy test accuracy

## 2. Main Benchmark Results

### 2.1 Best Validation-Selected Settings on the Noisy Test Split

| Model | Family | Selected lambda | Noisy acc (mean ± std) | Noisy MAE deg (mean ± std) | Tol@1 acc | SynOps / sample |
|---|---|---:|---:|---:|---:|---:|
| ConvRecSNN | snn | 0.3 | 0.801 ± 0.034 | 2.613 ± 0.268 | 0.990 | 1.172e6 |
| FlatLIFSNN | snn | 0.1 | 0.742 ± 0.030 | 3.628 ± 0.153 | 0.976 | 2.418e5 |
| CRNNBaseline | crnn | 0.0 | 0.680 ± 0.050 | 3.817 ± 0.639 | 0.961 | 0 |
| GCCPHATLSBaseline | classical | 0.0 | 0.683 ± 0.011 | 3.868 ± 0.116 | 0.981 | 0 |

### 2.2 Main Takeaways

1. `ConvRecSNN` is the strongest noisy-condition model in the main benchmark.
   - It outperforms `CRNNBaseline` by about `12.2` percentage points in noisy accuracy (`0.801` vs `0.680`).
   - It reduces noisy angular MAE by about `31.5%` relative to `CRNNBaseline` (`2.61 deg` vs `3.82 deg`).

2. `FlatLIFSNN` is the most efficient learned model.
   - Its selected setting uses only about `20.6%` of `ConvRecSNN` SynOps.
   - That is about `79.4%` lower SynOps while retaining about `92.6%` of `ConvRecSNN` noisy accuracy.

3. Clean performance is largely saturated for learned models.
   - `ConvRecSNN`, `FlatLIFSNN`, and `CRNNBaseline` all achieve roughly `0.967` to `0.985` clean accuracy.
   - The real separation appears in the noisy split, not in the clean split.

4. The classical baseline is competitive only in a narrow sense.
   - `GCCPHATLSBaseline` is close to `CRNNBaseline` on noisy accuracy.
   - However, it remains clearly below `ConvRecSNN` and below `FlatLIFSNN` in angular error.

## 3. Lambda Study

### 3.1 ConvRecSNN

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

### 3.2 FlatLIFSNN

Noisy sweep:

| lambda | acc_mean | ang_mae_deg_mean | fr_mean | synops_per_sample_mean |
|---:|---:|---:|---:|---:|
| 0.00 | 0.745 | 3.570 | 0.4002 | 2.589e5 |
| 0.03 | 0.757 | 3.279 | 0.3929 | 2.533e5 |
| 0.10 | 0.742 | 3.628 | 0.3757 | 2.418e5 |
| 0.30 | 0.735 | 3.620 | 0.3353 | 2.168e5 |
| 1.00 | 0.800 | 2.721 | 0.2423 | 1.553e5 |

Interpretation:

- `FlatLIFSNN` shows a much clearer sparsity response to lambda than `ConvRecSNN`.
- From `lambda = 0.0` to `lambda = 1.0`:
  - FR decreases by about `39.5%`
  - SynOps decreases by about `40.0%`
  - Noisy accuracy increases by about `5.46` percentage points
  - Noisy MAE improves by about `0.85 deg`
- Despite this, validation-based selection chooses `lambda = 0.1` rather than `1.0`.

### 3.3 Implication of the Lambda Results

The most important methodological finding is that clean-validation-based selection does not fully align with noisy-test robustness.

- For `ConvRecSNN`, validation chooses `lambda = 0.3` while noisy test accuracy peaks at `0.03`.
- For `FlatLIFSNN`, validation chooses `lambda = 0.1` while the strongest noisy test result appears at `1.0`.

This suggests that future iterations should consider one of the following:

1. a noisy development split for model selection,
2. a multi-objective validation criterion that includes robustness, or
3. separate “clean-priority” and “noise-robustness-priority” operating points.

## 4. Research Extension

The research extension adds two lightweight analyses:

- a Pareto-style best-setting comparison,
- a fixed-SNR robustness sweep.

### 4.1 Pareto-Style Efficiency View

Best-setting size estimates:

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

### 4.2 Fixed-SNR Robustness Sweep

This extension retrains the learned models once on a representative seed (`274`) using the selected lambda and evaluates them at fixed SNR values from `-10 dB` to `20 dB`.

#### Accuracy at Representative SNRs

| Model | Acc @ 0 dB | Acc @ 10 dB | Acc @ 20 dB | Mean acc across all SNRs |
|---|---:|---:|---:|---:|
| ConvRecSNN | 0.596 | 0.839 | 0.930 | 0.660 |
| FlatLIFSNN | 0.463 | 0.705 | 0.846 | 0.556 |
| CRNNBaseline | 0.458 | 0.646 | 0.795 | 0.527 |
| GCCPHATLSBaseline | 0.464 | 0.661 | 0.738 | 0.513 |

#### Angular MAE Trend

At moderate to high SNR, `ConvRecSNN` also has the lowest angular error:

- `2.73 deg` at `10 dB`
- `2.21 deg` at `15 dB`
- `1.87 deg` at `20 dB`

Key observations:

1. `ConvRecSNN` is the most robust model across the full SNR sweep.
2. `FlatLIFSNN` remains consistently better than `CRNNBaseline` at non-negative SNRs.
3. `GCCPHATLSBaseline` can be competitive around `0 dB`, but it falls behind the learned SNNs as SNR improves.

Important caveat:

- This SNR sweep is a single-seed extension, not a 3-seed aggregate.
- It should be interpreted as supporting evidence, not as the primary benchmark table.

## 5. Limitations

This project is strong as a course research prototype, but several limits remain:

1. The data is synthetic 4-microphone spatialization rather than real array recordings.
2. Validation is clean-only, which likely biases lambda selection away from noisy robustness.
3. SynOps is an approximate compute proxy, not a hardware-measured energy metric.
4. The research extension is single-seed for the SNR sweep.

## 6. Final Conclusion

The benchmark supports three main conclusions.

1. `ConvRecSNN` is the strongest model for noisy DOA estimation in this setup.
   - It achieves the best noisy accuracy and best noisy angular MAE among the compared models.

2. `FlatLIFSNN` is the best efficiency-oriented learned model.
   - It is substantially cheaper than `ConvRecSNN` in SynOps and model size while preserving much of the accuracy.

3. Lambda regularization matters, but the current model-selection protocol can be improved.
   - `FlatLIFSNN` shows a clear and meaningful sparsity-accuracy trade-off.
   - `ConvRecSNN` is less sensitive to the current FR regularizer.
   - A noisy validation split or a more explicitly multi-objective selector would likely produce better deployment choices.

Overall, the most defensible project-level claim is:

> In this synthetic 4-microphone DOA benchmark, `ConvRecSNN` is the most robust model under noise, while `FlatLIFSNN` provides the strongest accuracy-efficiency trade-off among the learned models.
