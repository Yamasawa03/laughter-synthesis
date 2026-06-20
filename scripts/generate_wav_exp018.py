"""EXP-018: Teacher-forced inference + wav generation for Laughterscape test set.

Two-condition comparison: swap vocoder while keeping FS2 checkpoint fixed.

Environment variables:
    CKPT_PATH      - FS2 checkpoint (required)
    VOCODER_CKPT   - HiFi-GAN generator checkpoint path (required)
    VOCODER_CONFIG  - HiFi-GAN config.json path (required)
    OUT_DIR        - output directory (required)
    MAX_SAMPLES    - limit number of samples (optional, default: all)

Usage:
    cd ~/laughter-synthesis
    CKPT_PATH=/models/EXP-019_.../epoch=49-step=17200.ckpt \
    VOCODER_CKPT=/models/EXP-017_.../g_00130000 \
    VOCODER_CONFIG=/work/hifigan/config_16k_320hop.json \
    OUT_DIR=/work/exp018_wav_fmin0 \
    python scripts/generate_wav_exp018.py preprocess=laughter dataset=laughter +model.use_gst=false
"""
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
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None


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

    ckpt = CKPT_PATH if os.path.isabs(CKPT_PATH) else join(ocwd, CKPT_PATH)
    print(f"FS2 Checkpoint: {ckpt}")
    print(f"Vocoder ckpt:   {VOCODER_CKPT}")
    print(f"Vocoder config: {VOCODER_CONFIG}")

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
    print(f"Test set size: {len(ds)}, generating: {n}")

    out = OUT_DIR if os.path.isabs(OUT_DIR) else join(ocwd, OUT_DIR)
    os.makedirs(join(out, "gen"), exist_ok=True)
    os.makedirs(join(out, "gt"), exist_ok=True)

    results = []
    for i in range(n):
        sample = ds[i]
        fid = sample["fid"]
        print(f"[{i+1}/{n}] {fid}")

        batch = ds.collate_fn([sample])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            output = model(batch)
            pred_mel = output["mel_postnet_pred"]
            mel_len = batch["mel_length"]
            wavs = model.synthesize(pred_mel, mel_len)
            gen_wav = wavs[0]

        gt_wav_path = join(cfg.preprocess.path.processed_path, "wav", f"{fid}.wav")
        if not os.path.isabs(gt_wav_path):
            gt_wav_path = join(ocwd, gt_wav_path)
        gt_wav, sr = sf.read(gt_wav_path)
        gt_wav = (gt_wav * 32768).astype(np.int16)

        sf.write(join(out, "gen", f"{fid}.wav"), gen_wav, 16000, subtype="PCM_16")
        sf.write(join(out, "gt", f"{fid}.wav"), gt_wav, 16000, subtype="PCM_16")

        results.append({
            "fid": fid,
            "gen_wav_len": len(gen_wav),
            "gt_wav_len": len(gt_wav),
            "mel_length": mel_len.item(),
        })

    meta = {
        "exp_id": "EXP-018",
        "fs2_checkpoint": CKPT_PATH,
        "vocoder_checkpoint": VOCODER_CKPT,
        "vocoder_config": VOCODER_CONFIG,
        "n_samples": len(results),
        "samples": results,
    }
    with open(join(out, "eval_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Generated {len(results)} samples in {out}/")


if __name__ == "__main__":
    main()
