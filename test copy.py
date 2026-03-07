import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)

FEATURE_CACHE: Dict[Tuple[int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
ANGLE_CACHE: Dict[int, torch.Tensor] = {}


@dataclass
class ExpConfig:
    seed: int = 274
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    data_root: str = "./data"
    speech_version: str = "speech_commands_v0.02"

    sample_rate: int = 16000
    audio_channels: int = 4
    window_seconds: float = 0.32
    doa_bins: int = 36

    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 48

    train_samples: int = 16000
    val_samples: int = 4000
    epochs: int = 15
    batch_size: int = 128
    num_workers: int = 2
    lr: float = 2e-3
    weight_decay: float = 1e-4
    lambda_fr: float = 5e-5
    label_smoothing: float = 0.05

    conv1_channels: int = 32
    conv2_channels: int = 64
    hidden_size: int = 192
    beta: float = 0.95
    threshold: float = 1.0

    mic_radius: float = 0.04
    speed_of_sound: float = 343.0
    output_dir: str = "./runs/snn_speechcommands_doa"

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def apply_fractional_delay(signal: np.ndarray, delay_s: float, sample_rate: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
    phase = np.exp(-2j * math.pi * freqs * delay_s)
    shifted = np.fft.irfft(np.fft.rfft(signal) * phase, n=signal.shape[0])
    return shifted.astype(np.float32)


def crop_active_speech(waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator, cfg: ExpConfig) -> torch.Tensor:
    x = waveform.mean(dim=0).float()
    if sample_rate != cfg.sample_rate:
        x = torchaudio.functional.resample(x, sample_rate, cfg.sample_rate)
    x = x - x.mean()
    peak = x.abs().amax()
    if peak > 0:
        x = x / peak

    if peak > 0:
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


class SpeechCommandsDOADataset(Dataset):
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        cfg: ExpConfig,
        max_samples: Optional[int],
        seed_offset: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.seed_offset = seed_offset

        rng = np.random.default_rng(cfg.seed + seed_offset)
        limit = len(base_dataset) if max_samples is None else min(max_samples, len(base_dataset))
        self.base_indices = rng.permutation(len(base_dataset))[:limit].tolist()
        self.label_offset = int(rng.integers(0, cfg.doa_bins))

    def __len__(self) -> int:
        return len(self.base_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.cfg.seed + self.seed_offset + index)
        waveform, sample_rate, *_ = self.base_dataset[self.base_indices[index]]
        mono = crop_active_speech(waveform, sample_rate, rng, self.cfg)

        label = (index * 5 + self.label_offset) % self.cfg.doa_bins
        center = -math.pi + (label + 0.5) * self.cfg.bin_width_rad
        jitter = rng.uniform(-0.35, 0.35) * self.cfg.bin_width_rad
        azimuth = math.atan2(
            math.sin(center + jitter),
            math.cos(center + jitter),
        )

        audio = spatialize_waveform(mono, azimuth, self.cfg)
        feat = extract_features(audio, self.cfg)
        return feat, torch.tensor(label, dtype=torch.long), torch.tensor(azimuth, dtype=torch.float32)


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

        mem1 = mem2 = None
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


def logits_to_azimuth(logits: torch.Tensor, cfg: ExpConfig) -> torch.Tensor:
    centers = bin_centers(cfg, logits.device)
    probs = logits.softmax(dim=-1)
    x = (probs * torch.cos(centers)).sum(dim=-1)
    y = (probs * torch.sin(centers)).sum(dim=-1)
    return torch.atan2(y, x)


def angular_error_deg(pred_azimuth: torch.Tensor, target_azimuth: torch.Tensor) -> torch.Tensor:
    diff = torch.atan2(torch.sin(pred_azimuth - target_azimuth), torch.cos(pred_azimuth - target_azimuth)).abs()
    return torch.rad2deg(diff)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    cfg: ExpConfig,
) -> Tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    total_loss = 0.0
    total_acc = 0.0
    total_ang = 0.0
    total_count = 0

    for xb, yb, ab in tqdm(loader, desc="train" if training else "val", leave=False):
        xb = xb.to(cfg.device, non_blocking=True)
        yb = yb.to(cfg.device, non_blocking=True)
        ab = ab.to(cfg.device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(cfg.device_type, enabled=(cfg.device_type == "cuda" and cfg.amp)):
            logits, avg_fr = model(xb)
            loss = criterion(logits, yb) + cfg.lambda_fr * avg_fr

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        pred = logits.argmax(dim=-1)
        pred_azimuth = logits_to_azimuth(logits.detach(), cfg)
        ang = angular_error_deg(pred_azimuth, ab).mean()
        acc = (pred == yb).float().mean()

        batch_size = xb.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_acc += float(acc.item()) * batch_size
        total_ang += float(ang.item()) * batch_size
        total_count += batch_size

    return total_loss / total_count, total_acc / total_count, total_ang / total_count


def save_curves(train_losses: Sequence[float], val_accs: Sequence[float], val_angles: Sequence[float], output_dir: Path) -> None:
    ensure_dir(output_dir)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, color="tab:blue")
    plt.title("Train Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(val_accs, color="tab:green")
    plt.title("Val Accuracy")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(val_angles, color="tab:red")
    plt.title("Val Angular Error")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "curves.png", dpi=180)
    plt.close()


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpeechCommands spatialized SNN DOA classifier")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExpConfig:
    cfg = ExpConfig()
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.train_samples is not None:
        cfg.train_samples = args.train_samples
    if args.val_samples is not None:
        cfg.val_samples = args.val_samples
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.smoke:
        cfg.epochs = 2
        cfg.batch_size = 64
        cfg.train_samples = 1024
        cfg.val_samples = 256
        cfg.output_dir = str(Path(cfg.output_dir) / "smoke")

    if cfg.device_type == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


def main() -> None:
    cfg = build_config(parse_args())
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    save_json(asdict(cfg), output_dir / "config.json")

    print("Loading SpeechCommands ...")
    train_base = torchaudio.datasets.SPEECHCOMMANDS(
        root=cfg.data_root,
        url=cfg.speech_version,
        download=True,
        subset="training",
    )
    val_base = torchaudio.datasets.SPEECHCOMMANDS(
        root=cfg.data_root,
        url=cfg.speech_version,
        download=True,
        subset="validation",
    )

    train_ds = SpeechCommandsDOADataset(train_base, cfg, cfg.train_samples, seed_offset=0)
    val_ds = SpeechCommandsDOADataset(val_base, cfg, cfg.val_samples, seed_offset=1_000_000)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device_type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(4 if cfg.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device_type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(4 if cfg.num_workers > 0 else None),
    )

    model = DOASNN(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)
    scaler = torch.amp.GradScaler(cfg.device_type, enabled=(cfg.device_type == "cuda" and cfg.amp))

    train_losses: List[float] = []
    val_accs: List[float] = []
    val_angles: List[float] = []
    best_val_angle = float("inf")
    best_val_acc = 0.0

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Device: {cfg.device}")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, train_ang = run_epoch(model, train_loader, optimizer, scaler, cfg)
        val_loss, val_acc, val_ang = run_epoch(model, val_loader, None, scaler, cfg)
        scheduler.step()

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_angles.append(val_ang)

        print(
            f"Epoch {epoch:02d} | lr={current_lr(optimizer):.2e} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | train_ang={train_ang:.2f}deg | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | val_ang={val_ang:.2f}deg"
        )

        if val_ang < best_val_angle:
            best_val_angle = val_ang
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, output_dir / "best.pt")

    save_curves(train_losses, val_accs, val_angles, output_dir)
    save_json(
        {
            "best_val_angle_deg": best_val_angle,
            "best_val_acc": best_val_acc,
            "final_train_loss": train_losses[-1],
            "final_val_acc": val_accs[-1],
            "final_val_angle_deg": val_angles[-1],
        },
        output_dir / "metrics.json",
    )
    print(f"Best validation angular error: {best_val_angle:.2f}deg | Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
