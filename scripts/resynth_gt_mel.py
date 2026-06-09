"""Resynthesize wav files from ground-truth mel tensors with HiFi-GAN.

This script is for EXP-021-01. It loads GT mel `.pt` files, runs the
EXP-017 HiFi-GAN vocoder, writes PCM_16 wav files, and records an output
manifest for the next quality-evaluation experiment.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import hifigan


MANIFEST_FIELDS = [
    "fid",
    "status",
    "resynth_wav_path",
    "gt_wav_path",
    "mel_path",
    "file_size_bytes",
    "duration_seconds",
    "sample_rate",
    "n_samples",
    "mel_frames",
    "error",
]


def load_vocoder(config_path, ckpt_path, device="cuda"):
    with open(config_path) as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocoder.load_state_dict(ckpt["generator"], strict=True)
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder.to(device), int(config.get("sampling_rate", 16000))


def read_file_ids(file_list):
    ids = []
    with open(file_list) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split("|", 1)[0].strip())
    return ids


def extract_mel_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for key in ("mel", "mel_gt", "gt_mel", "mel_spectrogram"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
    raise TypeError(f"Unsupported mel object type: {type(obj)!r}")


def normalize_mel_shape(mel):
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel.squeeze(0)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel tensor, got shape {tuple(mel.shape)}")
    if mel.shape[0] == 80:
        return mel
    if mel.shape[1] == 80:
        return mel.transpose(0, 1)
    raise ValueError(f"Expected one mel dimension to be 80, got shape {tuple(mel.shape)}")


def synthesize(vocoder, mel, device):
    mel = normalize_mel_shape(mel).unsqueeze(0).to(device).float()
    with torch.no_grad():
        wav = vocoder(mel).squeeze().detach().cpu().numpy()
    return np.clip(wav, -1.0, 1.0)


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resynthesize wav files from GT mel tensors with HiFi-GAN."
    )
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--mel_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gt_wav_dir", default="data/laughter/wav")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by default. Use --device cpu for dry runs.")

    file_ids = read_file_ids(args.file_list)
    output_dir = Path(args.output_dir).expanduser()
    mel_dir = Path(args.mel_dir).expanduser()
    gt_wav_dir = Path(args.gt_wav_dir).expanduser()
    manifest_path = (
        Path(args.manifest_path).expanduser()
        if args.manifest_path
        else output_dir / "output_manifest.csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    vocoder, sample_rate = load_vocoder(args.config, args.checkpoint, device=device)
    rows = []
    generated = 0

    for index, fid in enumerate(file_ids, start=1):
        print(f"[{index}/{len(file_ids)}] {fid}")
        mel_path = mel_dir / f"{fid}.pt"
        wav_path = output_dir / f"{fid}.wav"
        gt_wav_path = gt_wav_dir / f"{fid}.wav"
        row = {
            "fid": fid,
            "status": "ok",
            "resynth_wav_path": str(wav_path),
            "gt_wav_path": str(gt_wav_path),
            "mel_path": str(mel_path),
            "file_size_bytes": "",
            "duration_seconds": "",
            "sample_rate": sample_rate,
            "n_samples": "",
            "mel_frames": "",
            "error": "",
        }

        try:
            mel = extract_mel_tensor(torch.load(mel_path, map_location="cpu"))
            mel = normalize_mel_shape(mel)
            row["mel_frames"] = int(mel.shape[1])
            wav = synthesize(vocoder, mel, device=device)
            sf.write(wav_path, wav, sample_rate, subtype="PCM_16")
            file_size = wav_path.stat().st_size
            row["file_size_bytes"] = file_size
            row["n_samples"] = int(wav.shape[0])
            row["duration_seconds"] = round(float(wav.shape[0]) / sample_rate, 6)
            generated += 1
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {row['error']}", file=sys.stderr)
            if args.fail_on_error:
                rows.append(row)
                write_manifest(manifest_path, rows)
                raise
        rows.append(row)
        write_manifest(manifest_path, rows)

    failed = len(rows) - generated
    print(f"Done. generated={generated}, failed={failed}, manifest={manifest_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
