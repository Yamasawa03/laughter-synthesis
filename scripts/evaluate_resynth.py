"""Evaluate resynthesized wav quality against ground truth.

Computes MCD, PESQ, and STOI for each GT-wav / resynth-wav pair.
Also generates screening metadata and distribution plots.

Usage (inside Docker on GPU server):
    cd /workspace/laughter-synthesis
    python scripts/evaluate_resynth.py \
        --manifest /workspace/exp021_resynth_wav/output_manifest.csv \
        --output_dir /workspace/exp021_eval_results
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fftpack import dct

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("WARNING: pesq not installed. PESQ will be skipped.", file=sys.stderr)

try:
    from pystoi import stoi as stoi_fn
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("WARNING: pystoi not installed. STOI will be skipped.", file=sys.stderr)

SR = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000
N_MFCC = 13


def extract_mel(wav, sr=SR):
    if HAS_LIBROSA:
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        )
    else:
        from scipy.signal import stft as scipy_stft
        _, _, Zxx = scipy_stft(wav, fs=sr, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT)
        power = np.abs(Zxx) ** 2
        mel_basis = _mel_filterbank(sr, N_FFT, N_MELS, FMIN, FMAX)
        mel = mel_basis @ power
    return np.log(np.clip(mel, a_min=1e-5, a_max=None))


def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = mel_to_hz(mels)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lo, mid, hi = freqs[i], freqs[i + 1], freqs[i + 2]
        up = (fft_freqs - lo) / (mid - lo)
        down = (hi - fft_freqs) / (hi - mid)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


def mel_to_mfcc(mel, n_mfcc=N_MFCC):
    mfcc = dct(mel, type=2, axis=0, norm="ortho")
    return mfcc[1 : n_mfcc + 1, :]


def compute_mcd(wav_a, wav_b, sr=SR):
    mel_a = extract_mel(wav_a, sr)
    mel_b = extract_mel(wav_b, sr)
    min_len = min(mel_a.shape[1], mel_b.shape[1])
    mfcc_a = mel_to_mfcc(mel_a[:, :min_len])
    mfcc_b = mel_to_mfcc(mel_b[:, :min_len])
    diff = mfcc_a - mfcc_b
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))
    return float((10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(frame_dist))


def compute_rms(wav):
    return float(np.sqrt(np.mean(wav ** 2)))


def align_lengths(ref, deg):
    min_len = min(len(ref), len(deg))
    return ref[:min_len], deg[:min_len]


def compute_stats(values, name):
    arr = np.array(values)
    return {
        "name": name,
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
    }


def save_plots(stats_dict, eval_items, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots", file=sys.stderr)
        return

    valid = [item for item in eval_items if "error" not in item]
    metrics = [
        ("mcd_db", "MCD (dB)", "MCD"),
        ("pesq", "PESQ", "PESQ"),
        ("stoi", "STOI", "STOI"),
    ]

    for key, xlabel, title in metrics:
        values = [item[key] for item in valid if item.get(key) is not None]
        if not values:
            continue
        s = stats_dict.get(key.replace("_db", ""), {})
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=20, edgecolor="black", alpha=0.7)
        if s:
            ax.axvline(s["mean"], color="red", linestyle="--",
                       label=f"Mean: {s['mean']:.3f}")
            ax.axvline(s["median"], color="blue", linestyle="--",
                       label=f"Median: {s['median']:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"EXP-021-02: {title} Distribution (N={len(values)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        png_path = output_dir / f"distribution_{key}.png"
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {png_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gt_wav_dir", default=None)
    parser.add_argument("--resynth_wav_dir", default=None)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        entries = [row for row in csv.DictReader(f) if row["status"] == "ok"]
    print(f"Loaded {len(entries)} entries from manifest")
    print(f"PESQ available: {HAS_PESQ}, STOI available: {HAS_STOI}, "
          f"librosa available: {HAS_LIBROSA}")

    eval_items = []
    metadata_rows = []
    screening = {"silent": [], "extreme_duration": []}

    for i, entry in enumerate(entries):
        fid = entry["fid"]

        if args.resynth_wav_dir:
            resynth_path = os.path.join(args.resynth_wav_dir, f"{fid}.wav")
        else:
            resynth_path = entry["resynth_wav_path"]

        if args.gt_wav_dir:
            gt_path = os.path.join(args.gt_wav_dir, f"{fid}.wav")
        else:
            gt_path = entry["gt_wav_path"]
            if not os.path.isabs(gt_path):
                gt_path = os.path.join(os.getcwd(), gt_path)

        print(f"[{i + 1}/{len(entries)}] {fid}", end=" ")

        try:
            gt_wav, gt_sr = sf.read(gt_path)
            resynth_wav, resynth_sr = sf.read(resynth_path)
            assert gt_sr == SR, f"GT sr {gt_sr} != {SR}"
            assert resynth_sr == SR, f"Resynth sr {resynth_sr} != {SR}"

            gt_dur = len(gt_wav) / SR
            resynth_dur = len(resynth_wav) / SR
            dur_ratio = resynth_dur / gt_dur if gt_dur > 0 else 0.0
            gt_rms = compute_rms(gt_wav)
            resynth_rms = compute_rms(resynth_wav)

            metadata_rows.append({
                "fid": fid,
                "gt_duration_s": round(gt_dur, 4),
                "resynth_duration_s": round(resynth_dur, 4),
                "duration_ratio": round(dur_ratio, 4),
                "gt_rms": f"{gt_rms:.6e}",
                "resynth_rms": f"{resynth_rms:.6e}",
            })

            if resynth_rms < 1e-6:
                screening["silent"].append(fid)
            if dur_ratio < 0.5 or dur_ratio > 2.0:
                screening["extreme_duration"].append(
                    {"fid": fid, "ratio": round(dur_ratio, 4)})

            gt_al, resynth_al = align_lengths(gt_wav, resynth_wav)

            mcd = compute_mcd(gt_al, resynth_al)

            pesq_score = None
            if HAS_PESQ:
                try:
                    pesq_score = float(pesq_fn(SR, gt_al, resynth_al, "wb"))
                except Exception as e:
                    print(f"PESQ-err:{e}", end=" ")

            stoi_score = None
            if HAS_STOI:
                try:
                    stoi_score = float(stoi_fn(gt_al, resynth_al, SR, extended=False))
                except Exception as e:
                    print(f"STOI-err:{e}", end=" ")

            item = {
                "fid": fid,
                "mcd_db": round(mcd, 4),
                "pesq": round(pesq_score, 4) if pesq_score is not None else None,
                "stoi": round(stoi_score, 4) if stoi_score is not None else None,
                "gt_duration_s": round(gt_dur, 4),
                "resynth_duration_s": round(resynth_dur, 4),
                "duration_ratio": round(dur_ratio, 4),
            }
            eval_items.append(item)

            parts = [f"MCD={mcd:.2f}"]
            if pesq_score is not None:
                parts.append(f"PESQ={pesq_score:.2f}")
            if stoi_score is not None:
                parts.append(f"STOI={stoi_score:.3f}")
            print(" ".join(parts))

        except Exception as e:
            print(f"ERROR: {e}")
            eval_items.append({"fid": fid, "error": str(e)})
            metadata_rows.append({"fid": fid, "gt_duration_s": "", "resynth_duration_s": "",
                                  "duration_ratio": "", "gt_rms": "", "resynth_rms": ""})

    valid = [it for it in eval_items if "error" not in it]
    stats = {}

    mcd_vals = [it["mcd_db"] for it in valid]
    if mcd_vals:
        stats["mcd"] = compute_stats(mcd_vals, "MCD (dB)")

    pesq_vals = [it["pesq"] for it in valid if it["pesq"] is not None]
    if pesq_vals:
        stats["pesq"] = compute_stats(pesq_vals, "PESQ")

    stoi_vals = [it["stoi"] for it in valid if it["stoi"] is not None]
    if stoi_vals:
        stats["stoi"] = compute_stats(stoi_vals, "STOI")

    eval_results = {
        "experiment": "EXP-021-02",
        "description": "GT mel resynthesis quality evaluation (HiFi-GAN)",
        "n_total": len(entries),
        "n_evaluated": len(valid),
        "n_errors": len(entries) - len(valid),
        "stats": stats,
        "items": eval_items,
    }

    eval_path = output_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"\nEval results: {eval_path}")

    meta_path = output_dir / "output_metadata.csv"
    if metadata_rows:
        keys = ["fid", "gt_duration_s", "resynth_duration_s", "duration_ratio",
                "gt_rms", "resynth_rms"]
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata_rows)
    print(f"Metadata: {meta_path}")

    screen_path = output_dir / "screening_results.json"
    screening_out = {
        "n_total": len(entries),
        "silent": {
            "count": len(screening["silent"]),
            "threshold": "RMS < 1e-6",
            "fids": screening["silent"],
        },
        "extreme_duration": {
            "count": len(screening["extreme_duration"]),
            "threshold": "ratio < 0.5 or ratio > 2.0",
            "items": screening["extreme_duration"],
        },
    }
    with open(screen_path, "w") as f:
        json.dump(screening_out, f, indent=2, ensure_ascii=False)
    print(f"Screening: {screen_path}")

    if not args.no_plots:
        save_plots(stats, eval_items, output_dir)

    print("\n=== Summary ===")
    for metric, s in stats.items():
        print(f"  {s['name']}: mean={s['mean']:.4f}, median={s['median']:.4f}, "
              f"std={s['std']:.4f}, range=[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"  Screening: silent={screening_out['silent']['count']}, "
          f"extreme_duration={screening_out['extreme_duration']['count']}")


if __name__ == "__main__":
    main()
