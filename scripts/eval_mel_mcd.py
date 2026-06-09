"""Compute mel-level MCD between FS2 predicted mel and GT mel tensors.

Measures FS2 mel prediction quality directly, without vocoder influence.
Both inputs are .pt files with shape [80, T].

Usage:
    python scripts/eval_mel_mcd.py \
      --pred_mel_dir /workspace/exp024_e2e_wav/pred_mel \
      --gt_mel_dir data/laughter/mel \
      --file_list filelists/laughter_train_subset200.txt \
      --output /workspace/exp024_eval_results/mel_mcd_results.json
"""
import argparse
import json
import sys

import numpy as np
import torch
from scipy.fft import dct

N_MFCC = 13


def mel_to_mfcc(mel, n_mfcc=N_MFCC):
    mfcc = dct(mel, type=2, axis=0, norm="ortho")
    return mfcc[1 : n_mfcc + 1, :]


def compute_mel_mcd(pred_mel, gt_mel):
    min_len = min(pred_mel.shape[1], gt_mel.shape[1])
    mfcc_pred = mel_to_mfcc(pred_mel[:, :min_len])
    mfcc_gt = mel_to_mfcc(gt_mel[:, :min_len])
    diff = mfcc_pred - mfcc_gt
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))
    return float((10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(frame_dist))


def read_file_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split("|", 1)[0].strip())
    return ids


def compute_stats(values):
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
    }


def load_mel_tensor(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for key in ("mel", "mel_gt", "gt_mel", "mel_spectrogram"):
            if key in obj:
                v = obj[key]
                return v.numpy() if isinstance(v, torch.Tensor) else v
    raise ValueError(f"Cannot extract mel from {path}: type={type(obj)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_mel_dir", required=True)
    parser.add_argument("--gt_mel_dir", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fids = read_file_ids(args.file_list)
    print(f"File list: {len(fids)} entries")

    items = []
    for i, fid in enumerate(fids):
        pred_path = f"{args.pred_mel_dir}/{fid}.pt"
        gt_path = f"{args.gt_mel_dir}/{fid}.pt"

        entry = {"fid": fid, "status": "ok", "error": ""}
        try:
            pred_mel = load_mel_tensor(pred_path)
            gt_mel = load_mel_tensor(gt_path)

            mcd = compute_mel_mcd(pred_mel, gt_mel)
            entry["mel_mcd_db"] = round(mcd, 4)
            entry["pred_mel_frames"] = int(pred_mel.shape[1])
            entry["gt_mel_frames"] = int(gt_mel.shape[1])
            entry["frame_ratio"] = round(pred_mel.shape[1] / gt_mel.shape[1], 4)
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            print(f"  [{i+1}/{len(fids)}] {fid} ERROR: {e}")
            continue

        items.append(entry)
        if (i + 1) % 50 == 0 or i == len(fids) - 1:
            print(f"  [{i+1}/{len(fids)}] {fid} MCD={mcd:.2f} dB  frames={pred_mel.shape[1]}/{gt_mel.shape[1]}")

    valid = [it for it in items if it["status"] == "ok"]
    mcd_vals = [it["mel_mcd_db"] for it in valid]
    ratio_vals = [it["frame_ratio"] for it in valid]

    result = {
        "description": "mel-level MCD: FS2 predicted mel vs GT mel",
        "n_total": len(fids),
        "n_ok": len(valid),
        "n_failed": len(items) - len(valid),
        "mel_mcd_stats": compute_stats(mcd_vals) if mcd_vals else {},
        "frame_ratio_stats": compute_stats(ratio_vals) if ratio_vals else {},
        "items": items,
    }

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nMel MCD stats: mean={result['mel_mcd_stats'].get('mean', 'N/A')}, "
          f"median={result['mel_mcd_stats'].get('median', 'N/A')}")
    print(f"Frame ratio stats: mean={result['frame_ratio_stats'].get('mean', 'N/A')}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
