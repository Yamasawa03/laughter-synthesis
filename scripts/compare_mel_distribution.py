"""Compare predicted mel and GT mel distributions."""

import argparse
import json
import os

import numpy as np
import torch


def read_file_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line.split("|", 1)[0].strip())
    return ids


def load_mel(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        mel = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        mel = obj
    elif isinstance(obj, dict):
        mel = None
        for key in ("mel", "mel_gt", "gt_mel", "mel_spectrogram"):
            if key in obj:
                value = obj[key]
                mel = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                break
        if mel is None:
            raise ValueError(f"cannot extract mel from keys={list(obj.keys())}")
    else:
        mel = np.asarray(obj)
    if mel.ndim != 2 or mel.shape[0] != 80:
        raise ValueError(f"expected shape [80, T], got {mel.shape}")
    return mel.astype(np.float32, copy=False)


def overall_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def split_stats(mels):
    pooled = np.concatenate(mels, axis=1)
    return {
        "channel_mean": np.mean(pooled, axis=1).astype(float).tolist(),
        "channel_std": np.std(pooled, axis=1).astype(float).tolist(),
        "channel_min": np.min(pooled, axis=1).astype(float).tolist(),
        "channel_max": np.max(pooled, axis=1).astype(float).tolist(),
        "overall": overall_stats(pooled.reshape(-1)),
    }


def save_plots(pred_all, gt_all, stats, output_dir, exp_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(-12, 3, 81)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gt_all.reshape(-1), bins=bins, alpha=0.55, density=True, label="GT", edgecolor="black")
    ax.hist(pred_all.reshape(-1), bins=bins, alpha=0.55, density=True, label="Pred", edgecolor="black")
    ax.axvline(stats["gt"]["overall"]["mean"], color="blue", linestyle="--", label="GT mean")
    ax.axvline(stats["pred"]["overall"]["mean"], color="red", linestyle="--", label="Pred mean")
    ax.set_xlabel("log-mel value")
    ax.set_ylabel("Density")
    ax.set_title(f"{exp_id}: Mel Value Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(output_dir, "mel_distribution_overall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    channels = np.arange(80)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(channels, stats["gt"]["channel_mean"], label="GT")
    axes[0].plot(channels, stats["pred"]["channel_mean"], label="Pred")
    axes[0].set_ylabel("Mean")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(channels, stats["gt"]["channel_std"], label="GT")
    axes[1].plot(channels, stats["pred"]["channel_std"], label="Pred")
    axes[1].set_xlabel("Mel channel")
    axes[1].set_ylabel("Std")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"{exp_id}: Channel-wise Mel Stats")
    fig.savefig(os.path.join(output_dir, "mel_channel_stats.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_mel_dir", required=True)
    parser.add_argument("--gt_mel_dir", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exp_id", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pred_mels = []
    gt_mels = []
    skipped = []
    for fid in read_file_ids(args.file_list):
        try:
            pred_mel = load_mel(os.path.join(args.pred_mel_dir, f"{fid}.pt"))
            gt_mel = load_mel(os.path.join(args.gt_mel_dir, f"{fid}.pt"))
        except Exception as exc:
            skipped.append({"fid": fid, "reason": str(exc)})
            continue
        pred_mels.append(pred_mel)
        gt_mels.append(gt_mel)

    pred_stats = split_stats(pred_mels)
    gt_stats = split_stats(gt_mels)
    result = {
        "exp_id": args.exp_id,
        "n_files": len(pred_mels),
        "skipped_fids": skipped,
        "pred": pred_stats,
        "gt": gt_stats,
    }
    with open(os.path.join(args.output_dir, "mel_distribution_stats.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    save_plots(np.concatenate(pred_mels, axis=1), np.concatenate(gt_mels, axis=1), result, args.output_dir, args.exp_id)
    print(f"Saved mel distribution stats for {args.exp_id}: n={len(pred_mels)}, skipped={len(skipped)}")


if __name__ == "__main__":
    main()
