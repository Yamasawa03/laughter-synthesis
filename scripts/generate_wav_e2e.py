"""FS2 free-run inference + HiFi-GAN wav generation.

This script removes GT pitch/energy/duration and GT mel targets before
forward, so FastSpeech2 predicts durations, prosody, and mel lengths. It
writes generated wavs, copied GT wavs, predicted mel tensors for mel-stage
MCD evaluation, and a manifest for downstream checks.

Environment variables:
    CKPT_PATH      - FS2 checkpoint (required)
    VOCODER_CKPT   - HiFi-GAN generator checkpoint path (required)
    VOCODER_CONFIG  - HiFi-GAN config.json path (required)
    OUT_DIR        - output directory (required)
    SPLIT          - dataset split (optional, default: test)
    FILE_LIST      - file list to filter fids (optional, default: all split samples)
    EXP_ID         - experiment id for eval_metadata.json (optional, default: EXP-024)
    SAVE_PROSODY   - save prosody predictions/GT to pred_prosody when set to 1
    MAX_SAMPLES    - limit number of samples (optional, default: all)

Usage:
    cd ~/laughter-synthesis
    CKPT_PATH=/models/EXP-024_.../epoch=49-step=17200.ckpt \
    VOCODER_CKPT=/models/EXP-017_.../g_00130000 \
    VOCODER_CONFIG=/work/hifigan/config_16k_320hop.json \
    OUT_DIR=/work/exp024_e2e \
    python scripts/generate_wav_e2e.py preprocess=laughter dataset=laughter +model.use_gst=false
"""
import csv
import os
import sys
import json
import numpy as np
import torch
import soundfile as sf
from os.path import join

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import hydra
from data_module import FSDataset
from lightning_module import BaselineLightningModule
import hifigan

CKPT_PATH = os.environ.get("CKPT_PATH")
VOCODER_CKPT = os.environ.get("VOCODER_CKPT")
VOCODER_CONFIG = os.environ.get("VOCODER_CONFIG")
OUT_DIR = os.environ.get("OUT_DIR")
SPLIT = os.environ.get("SPLIT", "test")
FILE_LIST = os.environ.get("FILE_LIST")
EXP_ID = os.environ.get("EXP_ID", "EXP-024")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None
SAVE_PROSODY = os.environ.get("SAVE_PROSODY") == "1"

MANIFEST_FIELDS = [
    "fid",
    "gen_wav_path",
    "gen_wav_len",
    "gt_wav_path",
    "gt_wav_len",
    "duration_ratio",
    "status",
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
    return vocoder.to(device)


def read_file_ids(file_list):
    ids = []
    with open(file_list) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split("|", 1)[0].strip())
    return ids


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def squeeze_feature(value):
    if value is None:
        return None
    if value.dim() == 3 and value.size(0) == 1 and value.size(-1) == 1:
        value = value.squeeze(0).squeeze(-1)
    elif value.dim() >= 2 and value.size(0) == 1:
        value = value.squeeze(0)
    return value.detach().cpu().float()


@hydra.main(config_path="../config", config_name="default")
def main(cfg):
    assert CKPT_PATH, "CKPT_PATH is required"
    assert VOCODER_CKPT, "VOCODER_CKPT is required"
    assert VOCODER_CONFIG, "VOCODER_CONFIG is required"
    assert OUT_DIR, "OUT_DIR is required"

    ocwd = hydra.utils.get_original_cwd()

    ckpt = CKPT_PATH if os.path.isabs(CKPT_PATH) else join(ocwd, CKPT_PATH)
    print(f"FS2 Checkpoint: {ckpt}")
    print(f"Vocoder ckpt:   {VOCODER_CKPT}")
    print(f"Vocoder config: {VOCODER_CONFIG}")
    print(f"Split:          {SPLIT}")

    model = BaselineLightningModule.load_from_checkpoint(ckpt, cfg=cfg, strict=False)
    model.eval()
    model.cuda()

    vocoder = load_vocoder(VOCODER_CONFIG, VOCODER_CKPT)
    model.vocoder = vocoder
    print("Vocoder loaded and injected into model")

    ds = FSDataset(SPLIT, cfg)
    sample_indices = list(range(len(ds)))
    file_list_path = None
    if FILE_LIST:
        file_list_path = FILE_LIST if os.path.isabs(FILE_LIST) else join(ocwd, FILE_LIST)
        requested_fids = read_file_ids(file_list_path)
        requested_set = set(requested_fids)
        sample_indices = [
            index for index, row in enumerate(ds.filelist) if row[0] in requested_set
        ]
        found_fids = {ds.filelist[index][0] for index in sample_indices}
        missing_fids = [fid for fid in requested_fids if fid not in found_fids]
        if missing_fids:
            raise ValueError(
                f"FILE_LIST contains {len(missing_fids)} fids not found in {SPLIT}: "
                f"{missing_fids[:5]}"
            )

    if MAX_SAMPLES:
        sample_indices = sample_indices[:MAX_SAMPLES]
    n = len(sample_indices)
    print(f"Dataset size: {len(ds)}, generating: {n}")
    if file_list_path:
        print(f"FILE_LIST:     {file_list_path}")

    out = OUT_DIR if os.path.isabs(OUT_DIR) else join(ocwd, OUT_DIR)
    os.makedirs(join(out, "gen"), exist_ok=True)
    os.makedirs(join(out, "gt"), exist_ok=True)
    os.makedirs(join(out, "pred_mel"), exist_ok=True)
    if SAVE_PROSODY:
        os.makedirs(join(out, "pred_prosody"), exist_ok=True)

    results = []
    manifest_rows = []
    for i, sample_index in enumerate(sample_indices):
        sample = ds[sample_index]
        fid = sample["fid"]
        print(f"[{i+1}/{n}] {fid}")

        batch = ds.collate_fn([sample])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            gt_pitch = batch.pop("pitch")
            gt_energy = batch.pop("energy")
            gt_duration = batch.pop("duration")
            mel_gt, mel_len_gt, mel_max_len = (
                batch.pop("mel"),
                batch.pop("mel_length"),
                batch.pop("mel_max_length"),
            )

            output = model(batch)
            pred_mel = output["mel_postnet_pred"]
            mel_len = output["mel_length"]
            wavs = model.synthesize(pred_mel, mel_len)
            gen_wav = wavs[0]

        torch.save(pred_mel[0, :, :mel_len.item()].cpu(), join(out, "pred_mel", f"{fid}.pt"))
        if SAVE_PROSODY:
            prosody = {
                "pitch_pred": squeeze_feature(output["pitch_pred"]),
                "energy_pred": squeeze_feature(output["energy_pred"]),
                "log_duration_pred": squeeze_feature(output["log_duration_pred"]),
                "duration_rounded_pred": squeeze_feature(output["duration_rounded_pred"]),
                "gt_pitch": squeeze_feature(gt_pitch),
                "gt_energy": squeeze_feature(gt_energy),
                "gt_duration": squeeze_feature(gt_duration),
            }
            torch.save(prosody, join(out, "pred_prosody", f"{fid}.pt"))

        gt_wav_path = join(cfg.preprocess.path.processed_path, "wav", f"{fid}.wav")
        if not os.path.isabs(gt_wav_path):
            gt_wav_path = join(ocwd, gt_wav_path)
        gt_wav, sr = sf.read(gt_wav_path)
        gt_wav = (gt_wav * 32768).astype(np.int16)

        gen_wav_path = join(out, "gen", f"{fid}.wav")
        gt_wav_out_path = join(out, "gt", f"{fid}.wav")
        sf.write(gen_wav_path, gen_wav, 16000, subtype="PCM_16")
        sf.write(gt_wav_out_path, gt_wav, 16000, subtype="PCM_16")

        duration_ratio = len(gen_wav) / len(gt_wav) if len(gt_wav) > 0 else 0
        results.append({
            "fid": fid,
            "gen_wav_len": len(gen_wav),
            "gt_wav_len": len(gt_wav),
            "mel_length_pred": mel_len.item(),
            "mel_length_gt": mel_len_gt.item(),
            "duration_ratio": duration_ratio,
        })
        manifest_rows.append({
            "fid": fid,
            "gen_wav_path": gen_wav_path,
            "gen_wav_len": len(gen_wav),
            "gt_wav_path": gt_wav_out_path,
            "gt_wav_len": len(gt_wav),
            "duration_ratio": duration_ratio,
            "status": "ok",
        })

    write_manifest(join(out, "output_manifest.csv"), manifest_rows)

    meta = {
        "exp_id": EXP_ID,
        "fs2_checkpoint": CKPT_PATH,
        "vocoder_checkpoint": VOCODER_CKPT,
        "vocoder_config": VOCODER_CONFIG,
        "split": SPLIT,
        "file_list": file_list_path,
        "save_prosody": SAVE_PROSODY,
        "n_samples": len(results),
        "samples": results,
    }
    with open(join(out, "eval_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Generated {len(results)} samples in {out}/")


if __name__ == "__main__":
    main()
