# LOCATA C-RSNN for Multi-Source DOA

## Goal
This project trains a spiking neural network (SNN) for 2-3 source direction-of-arrival (DOA) estimation with a practical robotics perspective:
- keep localization quality high,
- reduce spiking activity and compute cost,
- measure performance under both reference-array and robot-like conditions.

## Why This Setup
The notebook uses two evaluation conditions on top of LOCATA:
- `BK8 reference condition`: full 8-channel B&K array from LOCATA (high-quality upper bound).
- `Robotized condition`: 4 active microphones + lower SNR + channel jitter (deployment-oriented stress test).

This is intentional:
- B&K is excellent for reproducible algorithm benchmarking.
- Real robots usually use compact MEMS arrays with tighter cost/power/noise constraints.
- Reporting both conditions gives a clearer research story and better hardware relevance.

## Main Notebook
- `snn_ssl_demo.ipynb`

## Data Pipeline
- LOCATA dev set is downloaded automatically if missing.
- Extracted path after first run:
  - `./data/locata_dev.zip`
  - `./data/LOCATA_dev/dev/task*/recording*/benchmark2/*`
- The pipeline builds train/val/test examples from real LOCATA trajectories.
- For curriculum pretraining, simulated multi-source scenes are generated with `pyroomacoustics`.

## Model
The network is a C-RSNN:
- spiking `Conv2d` front-end for spatial-frequency patterns,
- recurrent spiking layer for temporal aggregation,
- DOA readout head,
- source-count head (2 or 3 sources).

Input features are STFT-based:
- log-magnitude,
- PHAT cross-power real part,
- PHAT cross-power imaginary part.

## Training Strategy
Three-stage sim-to-real curriculum:
1. 100% simulated training (geometry/acoustics pretraining),
2. 70% simulated + 30% real LOCATA mixed training,
3. 100% real LOCATA fine-tuning.

## Metrics
Results are saved to:
- `locata_crsnn_results.json`

Primary metrics:
- `micro_f1`
- `topk_hit` (uses predicted source count, no ground-truth `k` leakage)
- `count_acc`
- `mae_deg`
- `fr` (average firing rate)

## Quick Start
1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Open `snn_ssl_demo.ipynb` and run all cells.
3. If LOCATA is not present, the notebook will download and extract it automatically.

## Important Config Knobs
- Dataset scale:
  - `cfg.train_size`, `cfg.val_size`, `cfg.test_size`, `cfg.sim_train_size`
- Curriculum:
  - `cfg.stage1_sim_epochs`, `cfg.stage2_mix_epochs`, `cfg.stage3_real_epochs`, `cfg.stage2_sim_ratio`
- Task difficulty:
  - `cfg.doa_bins`, `cfg.min_sources`, `cfg.max_sources`
- Robotized stress:
  - `cfg.robot_active_mics`, `cfg.robot_snr_db_min/max`, `cfg.robot_channel_jitter_db_std`
