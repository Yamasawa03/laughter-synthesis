"""E2E inference: FS2 free-run + HiFi-GAN wav generation.

Free-run mode: GT pitch/energy/duration are removed from the batch
before forward, so FS2 predicts everything (like test_step in
lightning_module.py). Predicted mel is saved for mel-level MCD.

Environment variables:
    CKPT_PATH      - FS2 checkpoint (required)
    VOCODER_CKPT   - HiFi-GAN generator checkpoint path (required)
    VOCODER_CONFIG - HiFi-GAN config.json path (required)
    OUT_DIR        - output directory (required)
    FILE_LIST      - file list path relative to repo root (optional, default: test split from config)
    MAX_SAMPLES    - limit number of samples (optional, default: all)

Usage:
    cd ~/laughter-synthesis
    CKPT_PATH=... VOCODER_CKPT=... VOCODER_CONFIG=... OUT_DIR=... \
    FILE_LIST=filelists/laughter_train_subset200.txt \
    python scripts/generate_wav_e2e.py preprocess=laughter dataset=laughter +model.use_gst=false
"""
import csv
import json
import os
import sys
from os.path import join

import numpy as np
import soundfile as sf
import torch

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
FILE_LIST = os.environ.get("FILE_LIST")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None

MANIFEST_FIELDS = [
    "fid", "status", "gen_wav_path", "gt_wav_path", "pred_mel_path",
    "file_size_bytes", "gen_duration_sec", "gt_duration_sec",
    "duration_ratio", "sample_rate", "pred_mel_frames", "gt_mel_frames",
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
    return vocoder.to(device)


@hydra.main(config_path="../config", config_name="default")
def main(cfg):
    assert CKPT_PATH, "CKPT_PATH is required"
    assert VOCODER_CKPT, "VOCODER_CKPT is required"
    assert VOCODER_CONFIG, "VOCODER_CONFIG is required"
    assert OUT_DIR, "OUT_DIR is required"

    ocwd = hydra.utils.get_original_cwd()

    if FILE_LIST:
        cfg.dataset.test.filelist = join("./", FILE_LIST)

    ckpt = CKPT_PATH if os.path.isabs(CKPT_PATH) else join(ocwd, CKPT_PATH)
    print(f"FS2 Checkpoint: {ckpt}")
    print(f"Vocoder ckpt:   {VOCODER_CKPT}")
    print(f"Vocoder config: {VOCODER_CONFIG}")
    print(f"Mode:           free-run (predicted duration/pitch/energy)")

    model = BaselineLightningModule.load_from_checkpoint(ckpt, cfg=cfg, strict=False)
    model.eval()
    model.cuda()

    vocoder = load_vocoder(VOCODER_CONFIG, VOCODER_CKPT)
    model.vocoder = vocoder
    print("Vocoder loaded and injected into model")

    ds = FSDataset("test", cfg)
    n = len(ds)
    if MAX_SAMPLES:
        n = min(n, MAX_SAMPLES)
    print(f"Dataset size: {len(ds)}, generating: {n}")

    out = OUT_DIR if os.path.isabs(OUT_DIR) else join(ocwd, OUT_DIR)
    os.makedirs(join(out, "gen"), exist_ok=True)
    os.makedirs(join(out, "gt"), exist_ok=True)
    os.makedirs(join(out, "pred_mel"), exist_ok=True)

    manifest_path = join(out, "output_manifest.csv")
    manifest_file = open(manifest_path, "w", newline="")
    writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()

    results = []
    for i in range(n):
        sample = ds[i]
        fid = sample["fid"]
        print(f"[{i+1}/{n}] {fid}")

        batch = ds.collate_fn([sample])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        gt_mel_frames = batch["mel_length"].item()

        row = {"fid": fid, "status": "ok", "error": ""}
        try:
            with torch.no_grad():
                # free-run: remove GT targets so FS2 predicts everything
                batch.pop("pitch", None)
                batch.pop("energy", None)
                batch.pop("duration", None)
                gt_mel = batch.pop("mel", None)
                batch.pop("mel_length", None)
                batch.pop("mel_max_length", None)

                output = model(batch)
                pred_mel = output["mel_postnet_pred"]
                pred_mel_len = output["mel_length"]
                wavs = model.synthesize(pred_mel, pred_mel_len)
                gen_wav = wavs[0]

            # save predicted mel
            pred_mel_cpu = pred_mel[0, :, :pred_mel_len.item()].cpu()
            torch.save(pred_mel_cpu, join(out, "pred_mel", f"{fid}.pt"))

            # save GT wav
            gt_wav_path = join(cfg.preprocess.path.processed_path, "wav", f"{fid}.wav")
            if not os.path.isabs(gt_wav_path):
                gt_wav_path = join(ocwd, gt_wav_path)
            gt_wav, sr = sf.read(gt_wav_path)
            gt_wav = (gt_wav * 32768).astype(np.int16)

            gen_wav_out = join(out, "gen", f"{fid}.wav")
            gt_wav_out = join(out, "gt", f"{fid}.wav")
            sf.write(gen_wav_out, gen_wav, 16000, subtype="PCM_16")
            sf.write(gt_wav_out, gt_wav, 16000, subtype="PCM_16")

            gen_dur = len(gen_wav) / 16000
            gt_dur = len(gt_wav) / 16000
            row.update({
                "gen_wav_path": gen_wav_out,
                "gt_wav_path": gt_wav_out,
                "pred_mel_path": join(out, "pred_mel", f"{fid}.pt"),
                "file_size_bytes": os.path.getsize(gen_wav_out),
                "gen_duration_sec": f"{gen_dur:.4f}",
                "gt_duration_sec": f"{gt_dur:.4f}",
                "duration_ratio": f"{gen_dur / gt_dur:.4f}" if gt_dur > 0 else "inf",
                "sample_rate": 16000,
                "pred_mel_frames": pred_mel_len.item(),
                "gt_mel_frames": gt_mel_frames,
            })
            results.append(row)
        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
            results.append(row)
            print(f"  ERROR: {e}")

        writer.writerow(row)

    manifest_file.close()

    meta = {
        "exp_id": "EXP-024",
        "mode": "free-run",
        "fs2_checkpoint": CKPT_PATH,
        "vocoder_checkpoint": VOCODER_CKPT,
        "vocoder_config": VOCODER_CONFIG,
        "file_list": FILE_LIST or cfg.dataset.test.filelist,
        "n_total": len(results),
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
    }
    with open(join(out, "eval_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    ok = meta["n_ok"]
    fail = meta["n_failed"]
    print(f"\nDone. {ok} ok, {fail} failed out of {len(results)} in {out}/")


if __name__ == "__main__":
    main()
