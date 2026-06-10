"""Evaluate saved FastSpeech2 prosody predictions against GT features."""

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


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def corr(pred, gt):
    if pred.size < 2 or np.std(pred) == 0 or np.std(gt) == 0:
        return None
    return float(np.corrcoef(gt, pred)[0, 1])


def summarize_feature(per_fid, pooled_pred, pooled_gt):
    return {
        "rmse_pooled": rmse(pooled_pred, pooled_gt),
        "corr_pooled": corr(pooled_pred, pooled_gt),
        "rmse_per_fid_mean": float(np.mean(per_fid)),
        "rmse_per_fid_median": float(np.median(per_fid)),
    }


def save_scatter(path, gt, pred, title, xlabel="GT", ylabel="Pred"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt, pred, s=8, alpha=0.35)
    lo = float(min(np.min(gt), np.min(pred)))
    hi = float(max(np.max(gt), np.max(pred)))
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prosody_dir", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exp_id", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pools = {name: {"pred": [], "gt": [], "rmse": []} for name in ("pitch", "energy")}
    dur_pred = []
    dur_gt = []
    dur_log_pred = []
    dur_log_gt = []
    skipped = []

    for fid in read_file_ids(args.file_list):
        path = os.path.join(args.prosody_dir, f"{fid}.pt")
        try:
            data = torch.load(path, map_location="cpu")
            features = {}
            for name in ("pitch", "energy"):
                pred = to_numpy(data[f"{name}_pred"]).reshape(-1).astype(np.float64)
                gt = to_numpy(data[f"gt_{name}"]).reshape(-1).astype(np.float64)
                if len(pred) != len(gt):
                    raise ValueError(f"{name} length mismatch pred={len(pred)} gt={len(gt)}")
                features[name] = (pred, gt)

            d_pred = to_numpy(data["duration_rounded_pred"]).reshape(-1).astype(np.float64)
            d_gt = to_numpy(data["gt_duration"]).reshape(-1).astype(np.float64)
            if len(d_pred) != len(d_gt):
                raise ValueError(f"duration length mismatch pred={len(d_pred)} gt={len(d_gt)}")
        except Exception as exc:
            skipped.append({"fid": fid, "reason": str(exc)})
            continue

        for name, (pred, gt) in features.items():
            pools[name]["pred"].append(pred)
            pools[name]["gt"].append(gt)
            pools[name]["rmse"].append(rmse(pred, gt))
        dur_pred.append(d_pred)
        dur_gt.append(d_gt)
        dur_log_pred.append(np.log(d_pred + 1.0))
        dur_log_gt.append(np.log(d_gt + 1.0))

    pitch_pred = np.concatenate(pools["pitch"]["pred"])
    pitch_gt = np.concatenate(pools["pitch"]["gt"])
    energy_pred = np.concatenate(pools["energy"]["pred"])
    energy_gt = np.concatenate(pools["energy"]["gt"])
    duration_pred = np.concatenate(dur_pred)
    duration_gt = np.concatenate(dur_gt)
    duration_log_pred = np.concatenate(dur_log_pred)
    duration_log_gt = np.concatenate(dur_log_gt)

    result = {
        "exp_id": args.exp_id,
        "n_total": len(read_file_ids(args.file_list)),
        "n_evaluated": len(dur_pred),
        "skipped_fids": skipped,
        "pitch": summarize_feature(pools["pitch"]["rmse"], pitch_pred, pitch_gt),
        "energy": summarize_feature(pools["energy"]["rmse"], energy_pred, energy_gt),
        "duration": {
            "log_rmse_pooled": rmse(duration_log_pred, duration_log_gt),
            "ratio_mean": float(np.mean(duration_pred / np.maximum(duration_gt, 1.0))),
        },
    }
    with open(os.path.join(args.output_dir, "prosody_accuracy.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    save_scatter(os.path.join(args.output_dir, "scatter_pitch.png"), pitch_gt, pitch_pred, f"{args.exp_id}: pitch")
    save_scatter(os.path.join(args.output_dir, "scatter_energy.png"), energy_gt, energy_pred, f"{args.exp_id}: energy")
    save_scatter(os.path.join(args.output_dir, "scatter_duration.png"), duration_gt, duration_pred, f"{args.exp_id}: duration")
    print(f"Saved prosody accuracy for {args.exp_id}: n={result['n_evaluated']}, skipped={len(skipped)}")


if __name__ == "__main__":
    main()
