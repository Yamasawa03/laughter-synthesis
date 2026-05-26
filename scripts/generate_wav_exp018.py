"""EXP-018: Teacher-forced inference + wav generation for Laughterscape test set
using EXP-004 FS2 checkpoint + EXP-017 fmin=0 HiFi-GAN vocoder.

Usage:
    cd ~/laughter-synthesis
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

CKPT_PATH = os.environ.get(
    "CKPT_PATH", "pl_log_exp004/epoch=21-step=7568.ckpt"
)
OUT_DIR = os.environ.get("OUT_DIR", "exp018_wav_output")


@hydra.main(config_path="../config", config_name="default")
def main(cfg):
    ocwd = hydra.utils.get_original_cwd()
    ckpt = join(ocwd, CKPT_PATH)
    print(f"Checkpoint: {ckpt}")

    model = BaselineLightningModule.load_from_checkpoint(ckpt, cfg=cfg)
    model.eval()
    model.cuda()

    assert model.ensure_vocoder(verbose=True), "Vocoder not found"

    ds = FSDataset("test", cfg)
    n = len(ds)
    print(f"Test set size: {n}")

    out = join(ocwd, OUT_DIR)
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
        "checkpoint": CKPT_PATH,
        "n_samples": len(results),
        "vocoder": "hifigan/g_16k_320hop (EXP-017 fmin=0 130k)",
        "samples": results,
    }
    with open(join(out, "eval_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Generated {len(results)} samples in {out}/")


if __name__ == "__main__":
    main()
