"""Analyze FastSpeech2 mel/prosody quality on the test set.

Runs inference with the best checkpoint, compares predicted mel/prosody/duration
against GT, and writes JSON statistics plus PNG visualizations.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate laugh
    cd ~/laughter-synthesis
    python scripts/analyze_mel_quality.py preprocess=laughter dataset=laughter
"""

import os
import sys
import json
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_module import DataModule
from lightning_module import BaselineLightningModule
import utils

seed_everything(1024)

CKPT_PATH = "/home/yamasawa/laughter-synthesis/pl_log/epoch=26-step=9288.ckpt"
OUTPUT_DIR = "/home/yamasawa/laughter-synthesis/mel_analysis_results"
N_SAMPLES_PLOT = 5


class MelCollector(pl.Callback):
    """Collects model predictions and GT from test_step outputs."""

    def __init__(self):
        self.results = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.results.append(outputs)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _metric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _safe_mse(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    min_len = min(len(a), len(b))
    diff = np.asarray(a[:min_len], dtype=np.float64) - np.asarray(b[:min_len], dtype=np.float64)
    return float(np.mean(diff ** 2))


def _expanded_feature(values, durations, text_length):
    values = np.asarray(values[:text_length])
    durations = np.asarray(durations[:text_length])
    return utils.expand(values, durations)


def _compute_sample_metrics(sample):
    pred_len = int(sample["mel_length"])
    gt_len = int(sample["mel_length_gt"])
    text_length = int(sample["text_length"])
    min_mel_len = min(pred_len, gt_len)

    mel_pred = sample["mel_pred"][:, :min_mel_len]
    mel_gt = sample["mel_gt"][:, :min_mel_len]
    mel_diff = mel_pred - mel_gt

    duration_pred = sample["duration_rounded_pred"][:text_length]
    duration_gt = sample["duration"][:text_length]

    pitch_pred = _expanded_feature(sample["pitch_pred"], duration_pred, text_length)
    pitch_gt = _expanded_feature(sample["pitch_gt"], duration_gt, text_length)
    energy_pred = _expanded_feature(sample["energy_pred"], duration_pred, text_length)
    energy_gt = _expanded_feature(sample["energy_gt"], duration_gt, text_length)

    return {
        "fid": sample["fid"],
        "mel_mse": float(np.mean(mel_diff ** 2)),
        "pitch_mse": _safe_mse(pitch_pred, pitch_gt),
        "energy_mse": _safe_mse(energy_pred, energy_gt),
        "duration_mse": float(np.mean((duration_pred.astype(np.float64) - duration_gt.astype(np.float64)) ** 2)),
        "duration_ratio": float(pred_len) / float(gt_len) if gt_len > 0 else float("nan"),
        "mel_error_by_band": {
            "low": float(np.mean(mel_diff[0:27, :] ** 2)),
            "mid": float(np.mean(mel_diff[27:54, :] ** 2)),
            "high": float(np.mean(mel_diff[54:80, :] ** 2)),
        },
    }


def _make_sample(batch, i):
    text_length = int(_to_numpy(batch["text_length"])[i])
    pred_len = int(_to_numpy(batch["mel_length"])[i])
    gt_len = int(_to_numpy(batch["mel_length_gt"])[i])

    return {
        "fid": batch["fid"][i],
        "raw_speaker": batch["raw_speaker"][i],
        "mel_pred": _to_numpy(batch["mel_pred"])[i, :, :pred_len],
        "mel_gt": _to_numpy(batch["mel_gt"])[i, :, :gt_len],
        "mel_length": pred_len,
        "mel_length_gt": gt_len,
        "pitch_pred": _to_numpy(batch["pitch_pred"])[i, :text_length],
        "pitch_gt": _to_numpy(batch["pitch_gt"])[i, :text_length],
        "energy_pred": _to_numpy(batch["energy_pred"])[i, :text_length],
        "energy_gt": _to_numpy(batch["energy_gt"])[i, :text_length],
        "duration_rounded_pred": _to_numpy(batch["duration_rounded_pred"])[i, :text_length],
        "duration": _to_numpy(batch["duration"])[i, :text_length],
        "text_length": text_length,
    }


def _plot_mel_comparison(sample, output_dir):
    fid = sample["fid"]
    pred_len = int(sample["mel_length"])
    gt_len = int(sample["mel_length_gt"])
    min_len = min(pred_len, gt_len)
    mel_gt = sample["mel_gt"][:, :min_len]
    mel_pred = sample["mel_pred"][:, :min_len]
    mel_error = np.abs(mel_gt - mel_pred)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    plots = [
        (mel_gt, "GT mel"),
        (mel_pred, "Predicted mel"),
        (mel_error, "|GT - pred|"),
    ]
    for ax, (data, title) in zip(axes, plots):
        image = ax.imshow(data, aspect="auto", origin="lower", interpolation="none")
        ax.set_title(f"{fid} - {title}")
        ax.set_ylabel("Mel bin")
        fig.colorbar(image, ax=ax)
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mel_comparison", f"{fid}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_duration_comparison(sample, output_dir):
    fid = sample["fid"]
    text_length = int(sample["text_length"])
    token_idx = np.arange(text_length)
    duration_gt = sample["duration"][:text_length]
    duration_pred = sample["duration_rounded_pred"][:text_length]
    width = 0.4

    fig, ax = plt.subplots(1, 1, figsize=(max(10, text_length * 0.25), 5))
    ax.bar(token_idx - width / 2, duration_gt, width=width, label="GT")
    ax.bar(token_idx + width / 2, duration_pred, width=width, label="Pred")
    ax.set_title(f"{fid} - Duration comparison")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Duration frames")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "duration_comparison", f"{fid}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_prosody_comparison(sample, output_dir):
    fid = sample["fid"]
    text_length = int(sample["text_length"])
    duration_gt = sample["duration"][:text_length]
    duration_pred = sample["duration_rounded_pred"][:text_length]
    pitch_gt = _expanded_feature(sample["pitch_gt"], duration_gt, text_length)
    pitch_pred = _expanded_feature(sample["pitch_pred"], duration_pred, text_length)
    energy_gt = _expanded_feature(sample["energy_gt"], duration_gt, text_length)
    energy_pred = _expanded_feature(sample["energy_pred"], duration_pred, text_length)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    axes[0].plot(pitch_gt, label="GT", linewidth=1.2)
    axes[0].plot(pitch_pred, label="Pred", linewidth=1.2, alpha=0.8)
    axes[0].set_title(f"{fid} - Pitch")
    axes[0].set_xlabel("Expanded frame")
    axes[0].set_ylabel("Pitch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(energy_gt, label="GT", linewidth=1.2)
    axes[1].plot(energy_pred, label="Pred", linewidth=1.2, alpha=0.8)
    axes[1].set_title(f"{fid} - Energy")
    axes[1].set_xlabel("Expanded frame")
    axes[1].set_ylabel("Energy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "prosody_comparison", f"{fid}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_aggregate_error_breakdown(aggregate, output_dir):
    metrics = ["mel_mse", "pitch_mse", "energy_mse", "duration_mse"]
    means = np.asarray([aggregate[metric]["mean"] for metric in metrics], dtype=np.float64)
    normalized = means / np.max(means) if np.max(means) > 0 else means

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(metrics, normalized)
    ax.set_title("Normalized mean error breakdown")
    ax.set_ylabel("Normalized mean error")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregate_error_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mel_error_by_band(aggregate, output_dir):
    bands = ["low", "mid", "high"]
    means = [aggregate["mel_error_by_band"][band]["mean"] for band in bands]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.bar(bands, means)
    ax.set_title("Mel error by frequency band")
    ax.set_xlabel("Mel band")
    ax.set_ylabel("Mean squared error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mel_error_by_band.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_aggregate(per_sample):
    metrics = ["mel_mse", "pitch_mse", "energy_mse", "duration_mse", "duration_ratio"]
    aggregate = {metric: _metric_stats([item[metric] for item in per_sample]) for metric in metrics}
    aggregate["duration_ratio"]["median"] = float(np.median([item["duration_ratio"] for item in per_sample]))
    aggregate["mel_error_by_band"] = {}
    for band in ["low", "mid", "high"]:
        aggregate["mel_error_by_band"][band] = _metric_stats(
            [item["mel_error_by_band"][band] for item in per_sample]
        )
    return aggregate


def _representative_samples(samples, per_sample, n_samples):
    if len(samples) <= n_samples:
        return samples

    order = np.argsort([item["mel_mse"] for item in per_sample])
    positions = np.linspace(0, len(order) - 1, n_samples).round().astype(int)
    indices = [int(order[pos]) for pos in positions]
    return [samples[index] for index in indices]


def _ensure_use_gst_key(cfg: DictConfig):
    if "use_gst" in cfg.model:
        return
    was_struct = OmegaConf.is_struct(cfg.model)
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.use_gst = False
    OmegaConf.set_struct(cfg.model, was_struct)


@hydra.main(config_path='../config', config_name='default')
def main(cfg: DictConfig):
    _ensure_use_gst_key(cfg)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for subdir in ["mel_comparison", "duration_comparison", "prosody_comparison"]:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    print(f"Loading checkpoint: {CKPT_PATH}")
    datamodule = DataModule(cfg)
    module = BaselineLightningModule(cfg)

    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    missing, unexpected = module.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)} (vocoder weights, expected)")

    collector = MelCollector()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        callbacks=[collector],
        logger=False,
    )

    # Disable test_epoch_end (it tries to log figures/audio which requires a logger)
    module.test_epoch_end = lambda outputs: None

    print("Running inference on test set...")
    trainer.test(module, datamodule=datamodule)

    print(f"Collected {len(collector.results)} batches")

    samples = []
    for batch in collector.results:
        batch_size = batch["mel_pred"].shape[0]
        for i in range(batch_size):
            samples.append(_make_sample(batch, i))

    per_sample = [_compute_sample_metrics(sample) for sample in samples]
    aggregate = _build_aggregate(per_sample)

    output = {
        "checkpoint": CKPT_PATH,
        "n_samples": len(per_sample),
        "per_sample": per_sample,
        "aggregate": aggregate,
    }

    json_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {json_path}")

    for sample in _representative_samples(samples, per_sample, N_SAMPLES_PLOT):
        _plot_mel_comparison(sample, OUTPUT_DIR)
        _plot_duration_comparison(sample, OUTPUT_DIR)
        _plot_prosody_comparison(sample, OUTPUT_DIR)
    _plot_aggregate_error_breakdown(aggregate, OUTPUT_DIR)
    _plot_mel_error_by_band(aggregate, OUTPUT_DIR)
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
