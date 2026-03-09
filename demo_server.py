import argparse
import csv
import json
import math
import socket
import webbrowser
from dataclasses import dataclass
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demo"
CHECKPOINT_DIR = ROOT / "runs" / "unified_doa_notebook" / "checkpoints"
BEST_SETTINGS_PATH = ROOT / "runs" / "unified_doa_notebook" / "best_settings.csv"
SUMMARY_AGG_PATH = ROOT / "runs" / "unified_doa_notebook" / "summary_agg.csv"
SNR_ROBUSTNESS_PATH = ROOT / "runs" / "unified_doa_notebook" / "research_extension" / "snr_robustness.csv"
DEMO_SAMPLES = {
    "left": {"label": "Left", "audio": DEMO_DIR / "assets" / "audio" / "left.wav"},
    "right": {"label": "Right", "audio": DEMO_DIR / "assets" / "audio" / "right.wav"},
    "yes": {"label": "Yes", "audio": DEMO_DIR / "assets" / "audio" / "yes.wav"},
    "no": {"label": "No", "audio": DEMO_DIR / "assets" / "audio" / "no.wav"},
}
FEATURE_CACHE: Dict[Tuple[int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
ANGLE_CACHE: Dict[int, torch.Tensor] = {}
PAIR_GEOMETRY_CACHE: Dict[Tuple[int, float], torch.Tensor] = {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def pick_host() -> str:
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return "127.0.0.1"


@dataclass
class ExpConfig:
    seed: int = 274
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate: int = 16000
    audio_channels: int = 4
    window_seconds: float = 0.32
    doa_bins: int = 36
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 48
    conv1_channels: int = 32
    conv2_channels: int = 64
    hidden_size: int = 192
    flat_hidden_size: int = 128
    crnn_hidden_size: int = 160
    beta: float = 0.95
    threshold: float = 1.0
    mic_radius: float = 0.04
    speed_of_sound: float = 343.0
    dropout: float = 0.15

    @property
    def device_type(self) -> str:
        return "cuda" if self.device.startswith("cuda") else "cpu"

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.sample_rate))

    @property
    def feature_frames(self) -> int:
        return self.window_samples // self.hop_length + 1

    @property
    def input_channels(self) -> int:
        pairs = self.audio_channels * (self.audio_channels - 1) // 2
        return self.audio_channels + pairs

    @property
    def bin_width_rad(self) -> float:
        return 2.0 * math.pi / self.doa_bins


def build_cfg(saved_cfg: Dict[str, object], device: str) -> ExpConfig:
    cfg = ExpConfig()
    for key, value in saved_cfg.items():
        setattr(cfg, key, value)
    cfg.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return cfg


def pad_or_trim(x: torch.Tensor, target: int, dim: int = -1) -> torch.Tensor:
    if x.shape[dim] == target:
        return x
    if x.shape[dim] > target:
        return x.narrow(dim, 0, target)
    shape = list(x.shape)
    shape[dim] = target - x.shape[dim]
    return torch.cat([x, x.new_zeros(shape)], dim=dim)


def feature_kernels(cfg: ExpConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.n_mels)
    if key not in FEATURE_CACHE:
        window = torch.hann_window(cfg.win_length)
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=cfg.n_fft // 2 + 1,
            f_min=0.0,
            f_max=cfg.sample_rate / 2.0,
            n_mels=cfg.n_mels,
            sample_rate=cfg.sample_rate,
        ).float()
        FEATURE_CACHE[key] = (window, mel_fb)
    return FEATURE_CACHE[key]


def extract_features(audio: torch.Tensor, cfg: ExpConfig) -> torch.Tensor:
    window, mel_fb = feature_kernels(cfg)
    stft = torch.stft(audio, cfg.n_fft, cfg.hop_length, cfg.win_length, window, return_complex=True)
    power = stft.abs().pow(2.0)
    mel = torch.einsum("fm,cft->cmt", mel_fb, power).clamp_min(1e-10)
    log_mel = (10.0 * torch.log10(mel)).transpose(1, 2)
    log_mel = (log_mel - log_mel.mean(dim=(1, 2), keepdim=True)) / log_mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)

    gcc_parts = []
    half = cfg.n_mels // 2
    for m in range(cfg.audio_channels):
        for n in range(m + 1, cfg.audio_channels):
            cross = stft[m].conj() * stft[n]
            phat = cross / cross.abs().clamp_min(1e-6)
            cc = torch.fft.irfft(phat, n=cfg.n_fft, dim=0)
            cc = torch.cat([cc[-half:], cc[:half]], dim=0).transpose(0, 1)
            gcc_parts.append(cc)
    gcc = torch.stack(gcc_parts, dim=0)
    gcc = (gcc - gcc.mean(dim=(1, 2), keepdim=True)) / gcc.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)

    feat = torch.cat([log_mel, gcc], dim=0)
    return pad_or_trim(feat, cfg.feature_frames, dim=1).contiguous()


def mic_positions(cfg: ExpConfig) -> np.ndarray:
    r = cfg.mic_radius
    return np.array([[r, 0.0], [0.0, r], [-r, 0.0], [0.0, -r]], dtype=np.float32)


def bin_centers(cfg: ExpConfig, device: Optional[torch.device] = None) -> torch.Tensor:
    if cfg.doa_bins not in ANGLE_CACHE:
        centers = torch.linspace(-math.pi, math.pi, cfg.doa_bins + 1)[:-1] + 0.5 * cfg.bin_width_rad
        ANGLE_CACHE[cfg.doa_bins] = centers
    centers = ANGLE_CACHE[cfg.doa_bins]
    return centers if device is None else centers.to(device)


def load_audio_mono(path: Path) -> Tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    waveform = torch.from_numpy(audio.T).float()
    return waveform.mean(dim=0), int(sample_rate)


def apply_fractional_delay(signal: np.ndarray, delay_s: float, sample_rate: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
    phase = np.exp(-2j * math.pi * freqs * delay_s)
    shifted = np.fft.irfft(np.fft.rfft(signal) * phase, n=signal.shape[0])
    return shifted.astype(np.float32)


def crop_active_speech(waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator, cfg: ExpConfig) -> torch.Tensor:
    x = waveform.float()
    if sample_rate != cfg.sample_rate:
        x = torchaudio.functional.resample(x, sample_rate, cfg.sample_rate)
    x = x - x.mean()
    peak = x.abs().amax()
    if peak > 0:
        x = x / peak

    mask = x.abs() > 0.05
    if mask.any():
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        margin = max(1, cfg.sample_rate // 50)
        left = max(0, int(idx[0]) - margin)
        right = min(x.numel(), int(idx[-1]) + 1 + margin)
        x = x[left:right]

    if x.numel() <= cfg.window_samples:
        pad_left = (cfg.window_samples - x.numel()) // 2
        pad_right = cfg.window_samples - x.numel() - pad_left
        return F.pad(x, (pad_left, pad_right))

    stride = max(1, cfg.window_samples // 8)
    windows = x.unfold(0, cfg.window_samples, stride)
    energy = windows.pow(2).mean(dim=-1)
    topk = min(3, int(energy.numel()))
    candidate = torch.topk(energy, k=topk).indices[int(rng.integers(0, topk))].item()
    start = int(candidate * stride)
    jitter = int(rng.integers(-stride // 2, stride // 2 + 1))
    start = max(0, min(start + jitter, x.numel() - cfg.window_samples))
    x = x[start : start + cfg.window_samples]
    return x / x.abs().amax().clamp_min(1e-6)


def spatialize_waveform(mono: torch.Tensor, azimuth: float, cfg: ExpConfig) -> torch.Tensor:
    pad = max(64, int(round(0.005 * cfg.sample_rate)))
    padded = F.pad(mono, (pad, pad)).cpu().numpy()
    direction = np.array([math.cos(azimuth), math.sin(azimuth)], dtype=np.float32)
    delays = -(mic_positions(cfg) @ direction) / cfg.speed_of_sound
    delays -= delays.mean()

    channels = []
    for delay_s in delays:
        shifted = apply_fractional_delay(padded, float(delay_s), cfg.sample_rate)
        channels.append(shifted[pad:-pad])
    audio = torch.from_numpy(np.stack(channels, axis=0)).float()
    return audio / audio.abs().amax().clamp_min(1e-6)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(voltage)
        return (voltage > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (voltage,) = ctx.saved_tensors
        return grad_output / (1.0 + voltage.abs()).pow(2)


def lif_step(current: torch.Tensor, memory: torch.Tensor, beta: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    memory = beta * memory + current
    spikes = SurrogateSpike.apply(memory - threshold)
    return spikes, memory - spikes * threshold


class DOASNN(nn.Module):
    def __init__(self, cfg: ExpConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv1d(cfg.input_channels, cfg.conv1_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(cfg.conv1_channels, cfg.conv2_channels, kernel_size=3, padding=1, bias=False)

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.input_channels, cfg.n_mels)
            flat_dim = F.avg_pool1d(self.conv2(F.avg_pool1d(self.conv1(dummy), 2)), 2).numel()

        self.in_fc = nn.Linear(flat_dim, cfg.hidden_size, bias=False)
        self.rec_fc = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.readout = nn.Linear(cfg.hidden_size, cfg.doa_bins)

        beta0 = math.log(cfg.beta / (1.0 - cfg.beta))
        self.beta1_logit = nn.Parameter(torch.tensor(beta0))
        self.beta2_logit = nn.Parameter(torch.tensor(beta0))
        self.beta_r_logit = nn.Parameter(torch.tensor(beta0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1, 3)
        batch_size, time_steps = x.shape[:2]

        mem1 = None
        mem2 = None
        mem_r = x.new_zeros(batch_size, self.cfg.hidden_size)
        spk_prev = x.new_zeros(batch_size, self.cfg.hidden_size)
        logits_sum = x.new_zeros(batch_size, self.cfg.doa_bins)
        total_spikes = x.new_zeros(())
        total_neurons = 0

        beta1 = torch.sigmoid(self.beta1_logit).clamp(0.5, 0.999)
        beta2 = torch.sigmoid(self.beta2_logit).clamp(0.5, 0.999)
        beta_r = torch.sigmoid(self.beta_r_logit).clamp(0.5, 0.999)

        for t in range(time_steps):
            current1 = self.conv1(x[:, t])
            if mem1 is None:
                mem1 = torch.zeros_like(current1)
            spk1, mem1 = lif_step(current1, mem1, beta1, self.cfg.threshold)
            pooled1 = F.avg_pool1d(spk1, 2)

            current2 = self.conv2(pooled1)
            if mem2 is None:
                mem2 = torch.zeros_like(current2)
            spk2, mem2 = lif_step(current2, mem2, beta2, self.cfg.threshold)
            pooled2 = F.avg_pool1d(spk2, 2)

            current_r = self.in_fc(pooled2.flatten(1)) + self.rec_fc(spk_prev)
            spk_r, mem_r = lif_step(current_r, mem_r, beta_r, self.cfg.threshold)
            spk_prev = spk_r

            logits_sum = logits_sum + self.readout(mem_r)
            total_spikes = total_spikes + spk1.sum() + spk2.sum() + spk_r.sum()
            total_neurons += spk1.numel() + spk2.numel() + spk_r.numel()

        return logits_sum / time_steps, total_spikes / max(total_neurons, 1)


class LIFLinearLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, beta: float = 0.95, threshold: float = 1.0) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.beta = beta
        self.threshold = threshold

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = x_seq.shape
        mem = x_seq.new_zeros(batch_size, self.fc.out_features)
        spikes = []
        total_spikes = x_seq.new_zeros(())

        for t in range(time_steps):
            spk, mem = lif_step(self.fc(x_seq[:, t]), mem, self.beta, self.threshold)
            spikes.append(spk)
            total_spikes = total_spikes + spk.sum()

        return torch.stack(spikes, dim=1), total_spikes


class FlatLIFSNN(nn.Module):
    def __init__(self, cfg: ExpConfig) -> None:
        super().__init__()
        d_in = cfg.input_channels * cfg.n_mels
        h = cfg.flat_hidden_size

        self.in_proj = nn.Linear(d_in, h)
        self.norm = nn.LayerNorm(h)
        self.lif1 = LIFLinearLayer(h, h, beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = LIFLinearLayer(h, h, beta=cfg.beta, threshold=cfg.threshold)
        self.drop = nn.Dropout(cfg.dropout)
        self.readout = nn.Linear(h, cfg.doa_bins)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, time_steps, channels * freq_bins)
        x = torch.tanh(self.norm(self.in_proj(x)))
        s1, _ = self.lif1(x)
        s2, _ = self.lif2(s1)
        logits = self.readout(self.drop(s2.mean(dim=1)))
        return logits, x.new_zeros(())


class CRNNBaseline(nn.Module):
    def __init__(self, cfg: ExpConfig) -> None:
        super().__init__()
        d_in = cfg.input_channels * cfg.n_mels
        h = cfg.crnn_hidden_size

        self.frontend = nn.Sequential(
            nn.Conv1d(d_in, h, kernel_size=3, padding=1),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, cfg.doa_bins),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, time_steps, channels * freq_bins)
        x = self.frontend(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(x)
        logits = self.head(out.mean(dim=1))
        return logits, x.new_zeros(())


def build_model(model_key: str, cfg: ExpConfig) -> nn.Module:
    if model_key == "conv_snn":
        return DOASNN(cfg)
    if model_key == "flat_snn":
        return FlatLIFSNN(cfg)
    if model_key == "crnn":
        return CRNNBaseline(cfg)
    raise ValueError(f"Unknown model_key: {model_key}")


def logits_to_azimuth(logits: torch.Tensor, cfg: ExpConfig) -> torch.Tensor:
    centers = bin_centers(cfg, logits.device)
    probs = logits.softmax(dim=-1)
    x = (probs * torch.cos(centers)).sum(dim=-1)
    y = (probs * torch.sin(centers)).sum(dim=-1)
    return torch.atan2(y, x)


def pair_geometry(cfg: ExpConfig, device: Optional[torch.device] = None) -> torch.Tensor:
    key = (cfg.audio_channels, cfg.mic_radius)
    if key not in PAIR_GEOMETRY_CACHE:
        positions = torch.as_tensor(mic_positions(cfg), dtype=torch.float32)
        deltas = []
        for m in range(cfg.audio_channels):
            for n in range(m + 1, cfg.audio_channels):
                deltas.append(positions[m] - positions[n])
        PAIR_GEOMETRY_CACHE[key] = torch.stack(deltas, dim=0)
    geometry = PAIR_GEOMETRY_CACHE[key]
    return geometry if device is None else geometry.to(device)


def azimuth_to_bins(azimuth: torch.Tensor, cfg: ExpConfig) -> torch.Tensor:
    centers = bin_centers(cfg, azimuth.device)
    diff = torch.atan2(
        torch.sin(azimuth.unsqueeze(-1) - centers),
        torch.cos(azimuth.unsqueeze(-1) - centers),
    ).abs()
    return diff.argmin(dim=-1)


def gccphat_ls_predict(features: torch.Tensor, cfg: ExpConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    gcc = features[:, cfg.audio_channels :, :, :]
    pair_scores = gcc.mean(dim=2)
    half = pair_scores.size(-1) // 2
    lag_idx = pair_scores.argmax(dim=-1)
    lag_samples = lag_idx - half
    tau = lag_samples.float() / cfg.sample_rate

    A = pair_geometry(cfg, device=features.device)
    direction = (cfg.speed_of_sound * tau) @ torch.linalg.pinv(A).T
    direction = F.normalize(direction, dim=-1, eps=1e-6)
    pred_azimuth = torch.atan2(direction[:, 1], direction[:, 0])
    pred_bin = azimuth_to_bins(pred_azimuth, cfg)
    return pred_bin, pred_azimuth


def smooth_distribution(pred_bin: int, cfg: ExpConfig, sigma_bins: float = 1.2) -> torch.Tensor:
    centers = torch.arange(cfg.doa_bins, dtype=torch.float32)
    diff = torch.minimum((centers - pred_bin).abs(), cfg.doa_bins - (centers - pred_bin).abs())
    probs = torch.exp(-0.5 * (diff / sigma_bins).pow(2))
    return probs / probs.sum().clamp_min(1e-8)


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


@lru_cache(maxsize=4)
def read_csv_rows(path_str: str) -> List[Dict[str, str]]:
    path = Path(path_str)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_checkpoint_metrics(model: str, lambda_fr: float, tag: Optional[str]) -> Dict[str, float]:
    best_rows = read_csv_rows(str(BEST_SETTINGS_PATH))
    summary_rows = read_csv_rows(str(SUMMARY_AGG_PATH))

    if tag == "val_selected" or model in {"CRNNBaseline", "GCCPHATLSBaseline"}:
        for row in best_rows:
            if row["model"] == model and row["split"] == "noisy":
                return {
                    "acc_mean": parse_float(row.get("acc_mean")),
                    "acc_std": parse_float(row.get("acc_std")),
                    "ang_mae_deg_mean": parse_float(row.get("ang_mae_deg_mean")),
                    "ang_mae_deg_std": parse_float(row.get("ang_mae_deg_std")),
                }

    for row in summary_rows:
        if row["model"] != model or row["split"] != "noisy":
            continue
        if abs(parse_float(row.get("lambda")) - lambda_fr) < 1e-12:
            return {
                "acc_mean": parse_float(row.get("acc_mean")),
                "acc_std": parse_float(row.get("acc_std")),
                "ang_mae_deg_mean": parse_float(row.get("ang_mae_deg_mean")),
                "ang_mae_deg_std": parse_float(row.get("ang_mae_deg_std")),
            }

    raise KeyError(f"Missing metrics for {model} lambda={lambda_fr}")


def load_best_noisy_rows() -> List[Dict[str, object]]:
    rows = []
    for row in read_csv_rows(str(BEST_SETTINGS_PATH)):
        if row["split"] != "noisy":
            continue
        rows.append(
            {
                "model": row["model"],
                "family": row["family"],
                "lambda": parse_float(row.get("lambda")),
                "acc_mean": parse_float(row.get("acc_mean")),
                "acc_std": parse_float(row.get("acc_std")),
                "ang_mae_deg_mean": parse_float(row.get("ang_mae_deg_mean")),
                "ang_mae_deg_std": parse_float(row.get("ang_mae_deg_std")),
            }
        )
    order = {"ConvRecSNN": 0, "FlatLIFSNN": 1, "CRNNBaseline": 2, "GCCPHATLSBaseline": 3}
    return sorted(rows, key=lambda item: order.get(str(item["model"]), 99))


def load_lambda_study_rows() -> Dict[str, List[Dict[str, float]]]:
    studies: Dict[str, List[Dict[str, float]]] = {"ConvRecSNN": [], "FlatLIFSNN": []}
    for row in read_csv_rows(str(SUMMARY_AGG_PATH)):
        model = row["model"]
        if row["split"] != "noisy" or model not in studies:
            continue
        studies[model].append(
            {
                "lambda": parse_float(row.get("lambda")),
                "acc_mean": parse_float(row.get("acc_mean")),
                "acc_std": parse_float(row.get("acc_std")),
                "synops_per_sample_mean": parse_float(row.get("synops_per_sample_mean")),
                "synops_per_sample_std": parse_float(row.get("synops_per_sample_std")),
            }
        )
    for model in studies:
        studies[model].sort(key=lambda item: item["lambda"])
    return studies


def load_snr_robustness_rows() -> Dict[str, List[Dict[str, float]]]:
    studies: Dict[str, List[Dict[str, float]]] = {
        "ConvRecSNN": [],
        "FlatLIFSNN": [],
        "CRNNBaseline": [],
        "GCCPHATLSBaseline": [],
    }
    if not SNR_ROBUSTNESS_PATH.exists():
        return studies

    for row in read_csv_rows(str(SNR_ROBUSTNESS_PATH)):
        model = row["model"]
        if model not in studies:
            continue
        studies[model].append(
            {
                "snr_db": parse_float(row.get("snr_db")),
                "acc": parse_float(row.get("acc")),
                "ang_mae_deg": parse_float(row.get("ang_mae_deg")),
            }
        )

    for model in studies:
        studies[model].sort(key=lambda item: item["snr_db"])
    return studies


def format_model_label(model: str, tag: Optional[str], lambda_fr: float) -> str:
    tag_text = {
        "val_selected": "val-selected",
        "noisy_best": "noisy-best",
        None: "checkpoint",
    }[tag]
    if model == "CRNNBaseline":
        return model
    if model == "GCCPHATLSBaseline":
        return model
    return f"{model} ({tag_text}, lambda={lambda_fr:.0e})"


def discover_model_entries() -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    seen = set()
    tagged_pairs = set()

    for path in CHECKPOINT_DIR.glob("*.pt"):
        payload = torch.load(path, map_location="cpu")
        tagged_pairs.add((payload["model"], float(payload["lambda"]), payload.get("tag")))

    for path in sorted(CHECKPOINT_DIR.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        model = str(payload["model"])
        model_key = str(payload["model_key"])
        lambda_fr = float(payload["lambda"])
        tag = payload.get("tag")

        if tag is None and (model, lambda_fr, "val_selected") in tagged_pairs:
            continue

        entry_id = path.stem
        if entry_id in seen:
            continue
        seen.add(entry_id)

        entries.append(
            {
                "id": entry_id,
                "label": format_model_label(model, tag, lambda_fr),
                "model": model,
                "model_key": model_key,
                "family": "learned",
                "tag": tag,
                "lambda": lambda_fr,
                "path": path,
                "config": payload["config"],
                "metrics": summarize_checkpoint_metrics(model, lambda_fr, tag),
            }
        )

    gcc_path = CHECKPOINT_DIR / "GCCPHATLSBaseline_metadata.json"
    if gcc_path.exists():
        gcc_meta = json.loads(gcc_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "id": "GCCPHATLSBaseline",
                "label": "GCCPHATLSBaseline",
                "model": "GCCPHATLSBaseline",
                "model_key": "gccphat_ls",
                "family": "classical",
                "tag": "val_selected",
                "lambda": 0.0,
                "path": gcc_path,
                "config": gcc_meta["config"],
                "metrics": summarize_checkpoint_metrics("GCCPHATLSBaseline", 0.0, "val_selected"),
            }
        )

    preferred = [
        "ConvRecSNN_val_selected_seed274_lambda3e-01",
        "ConvRecSNN_noisy_best_seed274_lambda3e-02",
        "FlatLIFSNN_val_selected_seed274_lambda1e-01",
        "FlatLIFSNN_noisy_best_seed274_lambda1e+00",
        "CRNNBaseline_val_selected_seed274_lambda0e+00",
        "GCCPHATLSBaseline",
    ]
    order = {key: idx for idx, key in enumerate(preferred)}
    return sorted(entries, key=lambda item: (order.get(str(item["id"]), 999), str(item["label"])))


class DemoModelZoo:
    def __init__(self, device: str) -> None:
        self.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.entries = discover_model_entries()
        if not self.entries:
            raise RuntimeError(f"No demo checkpoints found in {CHECKPOINT_DIR}")
        self.entry_map = {str(entry["id"]): entry for entry in self.entries}
        self.loaded_models: Dict[str, Dict[str, object]] = {}

    def _load_bundle(self, model_id: str) -> Dict[str, object]:
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        entry = self.entry_map[model_id]
        cfg = build_cfg(entry["config"], self.device)
        bundle: Dict[str, object] = {
            "cfg": cfg,
            "bin_centers_deg": [normalize_deg(math.degrees(x)) for x in bin_centers(cfg).tolist()],
        }

        if entry["family"] == "learned":
            checkpoint = torch.load(entry["path"], map_location=self.device)
            model = build_model(str(entry["model_key"]), cfg).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            with torch.inference_mode():
                dummy = torch.randn(1, cfg.input_channels, cfg.feature_frames, cfg.n_mels, device=self.device)
                model(dummy)

            bundle["model"] = model

        self.loaded_models[model_id] = bundle
        return bundle

    def summary(self) -> Dict[str, object]:
        cfg = build_cfg(self.entries[0]["config"], self.device)
        default_model = str(self.entries[0]["id"])
        return {
            "title": "Unified DOA Benchmark Demo",
            "device": self.device,
            "defaultModel": default_model,
            "config": {
                "audio_channels": cfg.audio_channels,
                "doa_bins": cfg.doa_bins,
                "window_seconds": cfg.window_seconds,
            },
            "samples": [
                {"key": key, "label": value["label"], "audio": f"assets/audio/{key}.wav"}
                for key, value in DEMO_SAMPLES.items()
            ],
            "models": [
                {
                    "id": str(entry["id"]),
                    "label": str(entry["label"]),
                    "model": str(entry["model"]),
                    "family": str(entry["family"]),
                    "lambda": float(entry["lambda"]),
                    "metrics": {
                        "noisy_acc": float(entry["metrics"]["acc_mean"]),
                        "noisy_mae_deg": float(entry["metrics"]["ang_mae_deg_mean"]),
                        "noisy_acc_std": float(entry["metrics"]["acc_std"]),
                        "noisy_mae_std": float(entry["metrics"]["ang_mae_deg_std"]),
                    },
                }
                for entry in self.entries
            ],
            "benchmark": {
                "best_noisy": load_best_noisy_rows(),
                "lambda_study": load_lambda_study_rows(),
                "snr_robustness": load_snr_robustness_rows(),
            },
        }

    def predict(self, model_id: str, sample_key: str, angle_deg: float) -> Dict[str, object]:
        if sample_key not in DEMO_SAMPLES:
            raise KeyError(f"Unknown sample: {sample_key}")
        if model_id not in self.entry_map:
            raise KeyError(f"Unknown model: {model_id}")

        bundle = self._load_bundle(model_id)
        cfg = bundle["cfg"]
        waveform, sample_rate = load_audio_mono(DEMO_SAMPLES[sample_key]["audio"])

        rng = np.random.default_rng(274)
        mono = crop_active_speech(waveform, sample_rate, rng, cfg)
        audio = spatialize_waveform(mono, math.radians(angle_deg), cfg)
        feat = extract_features(audio, cfg).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self.entry_map[model_id]["family"] == "classical":
                pred_bin, pred_azimuth = gccphat_ls_predict(feat, cfg)
                probs = smooth_distribution(int(pred_bin.item()), cfg).cpu()
                confidence = float(probs.max().item())
            else:
                logits, _ = bundle["model"](feat)
                probs = logits.softmax(dim=-1).squeeze(0).cpu()
                pred_azimuth = logits_to_azimuth(logits, cfg)
                confidence = float(probs.max().item())

        pred_angle_deg = normalize_deg(math.degrees(float(pred_azimuth.item())))
        error_deg = abs(normalize_deg(pred_angle_deg - angle_deg))
        topk = torch.topk(probs, k=min(5, probs.numel()))
        top_bins = [
            {"angle_deg": bundle["bin_centers_deg"][int(idx)], "prob": float(prob)}
            for prob, idx in zip(topk.values.tolist(), topk.indices.tolist())
        ]

        return {
            "modelId": model_id,
            "modelLabel": str(self.entry_map[model_id]["label"]),
            "sample": sample_key,
            "sampleLabel": DEMO_SAMPLES[sample_key]["label"],
            "targetAngleDeg": round(normalize_deg(angle_deg), 2),
            "predAngleDeg": round(pred_angle_deg, 2),
            "errorDeg": round(error_deg, 2),
            "confidence": round(confidence, 4),
            "distribution": [round(float(x), 6) for x in probs.tolist()],
            "binCentersDeg": bundle["bin_centers_deg"],
            "topBins": top_bins,
        }


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "DOADemo/2.0"

    def __init__(self, *args, directory: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, directory=str(DEMO_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/summary":
            self.respond_json(self.server.demo.summary())  # type: ignore[attr-defined]
            return
        if parsed.path == "/api/predict":
            params = parse_qs(parsed.query)
            sample = params.get("sample", ["yes"])[0]
            model_id = params.get("model", [self.server.demo.summary()["defaultModel"]])[0]  # type: ignore[attr-defined]
            angle = float(params.get("angle", ["20"])[0])
            try:
                payload = self.server.demo.predict(model_id, sample, angle)  # type: ignore[attr-defined]
            except Exception as exc:
                self.respond_json({"error": str(exc)}, status=400)
                return
            self.respond_json(payload)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[demo] {self.address_string()} - {fmt % args}")

    def respond_json(self, payload: Dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DOA benchmark demo server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--open", action="store_true", help="Open the demo page in a browser after startup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = DemoModelZoo(args.device)

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    server.demo = demo  # type: ignore[attr-defined]

    url = f"http://{args.host}:{args.port}"
    print(f"Serving demo on {url}")
    if args.host == "0.0.0.0":
        print(f"Local network hint: http://{pick_host()}:{args.port}")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
