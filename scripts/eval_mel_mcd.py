"""Compute mel-stage MCD between FS2 predicted mel and GT mel.

Usage:
    python3 scripts/eval_mel_mcd.py \
        --pred_mel_dir /workspace/exp024_e2e_wav/pred_mel \
        --gt_mel_dir data/laughter/mel \
        --file_list filelists/laughter_train_subset200.txt \
        --output /workspace/exp024_eval_results/mel_mcd_results.json
"""
import argparse
import json
import os
import numpy as np
import torch

def mcd(mel_pred, mel_gt):
    """MCD in dB between two mel spectrograms (already in log-mel scale)."""
    diff = mel_pred - mel_gt
    return float(np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0))))


def load_mel_tensor(path):
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
            raise ValueError(f"Cannot extract mel from dict keys={list(obj.keys())}")
    else:
        mel = np.asarray(obj)

    if mel.ndim != 2 or mel.shape[0] != 80:
        raise ValueError(f"Expected mel shape [80, T], got {mel.shape}")
    return mel.astype(np.float32, copy=False)


def compute_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
    }

def dtw_align(pred, gt):
    """Simple DTW alignment for mel spectrograms. pred/gt shape: [n_mel, T]."""
    n = pred.shape[1]
    m = gt.shape[1]
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.sum((pred[:, i-1] - gt[:, j-1]) ** 2))
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    # Backtrack
    i, j = n, m
    pairs = []
    while i > 0 and j > 0:
        pairs.append((i-1, j-1))
        candidates = [(cost[i-1, j-1], i-1, j-1),
                       (cost[i-1, j], i-1, j),
                       (cost[i, j-1], i, j-1)]
        _, i, j = min(candidates, key=lambda x: x[0])
    pairs.reverse()
    pred_aligned = np.stack([pred[:, p] for p, _ in pairs], axis=1)
    gt_aligned = np.stack([gt[:, g] for _, g in pairs], axis=1)
    return pred_aligned, gt_aligned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_mel_dir", required=True)
    parser.add_argument("--gt_mel_dir", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no_dtw", action="store_true")
    args = parser.parse_args()

    with open(args.file_list) as f:
        fids = [line.strip().split("|")[0].strip() for line in f if line.strip() and not line.startswith("#")]

    results = []
    failed_fids = []
    alignment = "none_min_truncation" if args.no_dtw else "dtw"
    for i, fid in enumerate(fids):
        pred_path = os.path.join(args.pred_mel_dir, f"{fid}.pt")
        gt_path = os.path.join(args.gt_mel_dir, f"{fid}.pt")
        if not os.path.exists(pred_path):
            failed_fids.append({"fid": fid, "reason": f"missing pred: {pred_path}"})
            print(f"[{i+1}/{len(fids)}] {fid} SKIP (missing pred)")
            continue
        if not os.path.exists(gt_path):
            failed_fids.append({"fid": fid, "reason": f"missing gt: {gt_path}"})
            print(f"[{i+1}/{len(fids)}] {fid} SKIP (missing gt)")
            continue

        try:
            pred_mel = load_mel_tensor(pred_path)
            gt_mel = load_mel_tensor(gt_path)
            if args.no_dtw:
                min_len = min(pred_mel.shape[1], gt_mel.shape[1])
                pred_aligned = pred_mel[:, :min_len]
                gt_aligned = gt_mel[:, :min_len]
            elif pred_mel.shape[1] != gt_mel.shape[1]:
                pred_aligned, gt_aligned = dtw_align(pred_mel, gt_mel)
            else:
                pred_aligned, gt_aligned = pred_mel, gt_mel

            mcd_val = mcd(pred_aligned, gt_aligned)
            results.append({
                "fid": fid,
                "mcd": round(mcd_val, 4),
                "pred_frames": int(pred_mel.shape[1]),
                "gt_frames": int(gt_mel.shape[1]),
                "aligned_frames": int(pred_aligned.shape[1]),
                "frame_ratio": round(pred_mel.shape[1] / gt_mel.shape[1], 4),
            })
            print(f"[{i+1}/{len(fids)}] {fid} MCD={mcd_val:.2f} (pred={pred_mel.shape[1]} gt={gt_mel.shape[1]})")
        except Exception as exc:
            failed_fids.append({"fid": fid, "reason": str(exc)})
            print(f"[{i+1}/{len(fids)}] {fid} ERROR: {exc}")

    mcd_vals = [r["mcd"] for r in results]
    out = {
        "n_total": len(fids),
        "n_evaluated": len(results),
        "n_failed": len(failed_fids),
        "alignment": alignment,
        "stats": compute_stats(mcd_vals) if mcd_vals else {},
        "failed_fids": failed_fids,
        "items": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nMel-stage MCD ({alignment}): mean={out['stats'].get('mean', 'N/A')}")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
