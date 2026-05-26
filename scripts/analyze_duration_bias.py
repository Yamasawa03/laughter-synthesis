"""EXP-006-01: Duration Predictor systematic shortening bias analysis.

Compares predicted vs GT duration at the token level using the EXP-004
FastSpeech2 checkpoint.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate laugh
    cd ~/laughter-synthesis
    python scripts/analyze_duration_bias.py preprocess=laughter dataset=laughter
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_module import DataModule
from lightning_module import BaselineLightningModule

seed_everything(1024)

OUTPUT_DIR = "/home/yamasawa/laughter-synthesis/exp006_01_results"
CKPT_PATH = "/home/yamasawa/laughter-synthesis/pl_log_exp004/epoch=21-step=7568.ckpt"


def ensure_use_gst_key(cfg: DictConfig):
    """EXP-004 checkpoints may use configs that predate model.use_gst."""
    if "use_gst" in cfg.model:
        return

    was_struct = OmegaConf.is_struct(cfg.model)
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.use_gst = False
    OmegaConf.set_struct(cfg.model, was_struct)


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def collect_duration_bias(module, dataloader, device):
    log_bias_values = []
    ratio_values = []
    gt_duration_values = []
    n_utterances = 0

    module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)

            pitch = batch.pop("pitch")
            energy = batch.pop("energy")
            duration = batch.pop("duration")
            mel = batch.pop("mel")
            mel_length = batch.pop("mel_length")
            mel_max_length = batch.pop("mel_max_length")

            output = module.model(batch)
            log_duration_pred = output["log_duration_pred"].detach().cpu()
            duration = duration.detach().cpu()
            text_length = batch["text_length"].detach().cpu()

            # Restore popped values to keep the batch dictionary consistent if reused.
            batch["pitch"], batch["energy"], batch["duration"] = pitch, energy, duration
            batch["mel"], batch["mel_length"], batch["mel_max_length"] = mel, mel_length, mel_max_length

            batch_size = log_duration_pred.shape[0]
            n_utterances += batch_size

            for i in range(batch_size):
                length = int(text_length[i].item())
                log_pred = log_duration_pred[i, :length]
                gt_duration = duration[i, :length].float()
                log_gt = torch.log(gt_duration + 1.0)
                bias = log_pred - log_gt
                pred_frames = torch.round(torch.exp(log_pred) - 1.0).clamp(min=0)
                ratio = pred_frames / gt_duration

                valid = torch.isfinite(ratio) & torch.isfinite(bias) & (gt_duration > 0)
                log_bias_values.extend(bias[valid].numpy().tolist())
                ratio_values.extend(ratio[valid].numpy().tolist())
                gt_duration_values.extend(gt_duration[valid].numpy().tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    return {
        "log_bias": np.asarray(log_bias_values, dtype=np.float64),
        "ratio": np.asarray(ratio_values, dtype=np.float64),
        "gt_duration": np.asarray(gt_duration_values, dtype=np.float64),
        "n_utterances": n_utterances,
    }


def bucket_stats(gt_duration, ratio):
    buckets = {
        "1-5": (gt_duration >= 1) & (gt_duration <= 5),
        "6-10": (gt_duration >= 6) & (gt_duration <= 10),
        "11-20": (gt_duration >= 11) & (gt_duration <= 20),
        "21+": gt_duration >= 21,
    }

    out = {}
    for name, mask in buckets.items():
        values = ratio[mask]
        out[name] = {
            "mean_ratio": float(np.mean(values)) if values.size else None,
            "count": int(values.size),
        }
    return out


def save_stats(path, log_bias, ratio, gt_duration, n_utterances):
    stats = {
        "n_tokens": int(log_bias.size),
        "n_utterances": int(n_utterances),
        "log_space_bias": {
            "mean": float(np.mean(log_bias)),
            "median": float(np.median(log_bias)),
            "std": float(np.std(log_bias)),
        },
        "frame_space": {
            "mean_ratio": float(np.mean(ratio)),
            "median_ratio": float(np.median(ratio)),
            "std_ratio": float(np.std(ratio)),
        },
        "by_gt_duration_bucket": bucket_stats(gt_duration, ratio),
    }

    with open(path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def save_bias_histogram(path, log_bias, stats):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(log_bias, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        stats["log_space_bias"]["mean"],
        color="red",
        linestyle="--",
        label=f"Mean: {stats['log_space_bias']['mean']:.3f}",
    )
    ax.axvline(
        stats["log_space_bias"]["median"],
        color="blue",
        linestyle="--",
        label=f"Median: {stats['log_space_bias']['median']:.3f}",
    )
    ax.set_xlabel("log_pred - log_gt")
    ax.set_ylabel("Count")
    ax.set_title("EXP-006-01: Duration Predictor Log-Space Bias")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_ratio_scatter(path, gt_duration, ratio):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(gt_duration, ratio, s=8, alpha=0.35)
    ax.axhline(1.0, color="red", linestyle="--", label="GT ratio: 1.0")
    ax.set_xlabel("GT duration (frames)")
    ax.set_ylabel("Predicted/GT ratio")
    ax.set_title("EXP-006-01: Token Duration Ratio by GT Duration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


@hydra.main(config_path="../config", config_name="default")
def main(cfg: DictConfig):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_use_gst_key(cfg)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this analysis script.")

    print(f"Loading checkpoint: {CKPT_PATH}")
    datamodule = DataModule(cfg)
    module = BaselineLightningModule(cfg)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)} (vocoder weights, expected)")

    module = module.to(device)
    dataloader = datamodule.test_dataloader()

    print("Running manual forward passes on test set...")
    data = collect_duration_bias(module, dataloader, device)
    log_bias = data["log_bias"]
    ratio = data["ratio"]
    gt_duration = data["gt_duration"]

    if log_bias.size == 0:
        raise RuntimeError("No valid duration tokens were collected.")

    stats_path = os.path.join(OUTPUT_DIR, "duration_bias_stats.json")
    stats = save_stats(stats_path, log_bias, ratio, gt_duration, data["n_utterances"])
    print(f"Stats saved to: {stats_path}")

    hist_path = os.path.join(OUTPUT_DIR, "duration_bias_histogram.png")
    save_bias_histogram(hist_path, log_bias, stats)
    print(f"Histogram saved to: {hist_path}")

    scatter_path = os.path.join(OUTPUT_DIR, "duration_ratio_by_token_length.png")
    save_ratio_scatter(scatter_path, gt_duration, ratio)
    print(f"Scatter plot saved to: {scatter_path}")

    print(f"  Tokens: {stats['n_tokens']}")
    print(f"  Utterances: {stats['n_utterances']}")
    print(f"  Mean log-space bias: {stats['log_space_bias']['mean']:.4f}")
    print(f"  Median duration ratio: {stats['frame_space']['median_ratio']:.4f}")


if __name__ == "__main__":
    main()
