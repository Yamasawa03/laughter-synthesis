"""EXP-016-04: Teacher-forced inference + wav generation for LJSpeech test set"""
import os
import sys
import json
import random
import numpy as np
import torch
import soundfile as sf
from os.path import join

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import hydra
from data_module import FSDataset
from lightning_module import BaselineLightningModule

@hydra.main(config_path="config", config_name="default")
def main(cfg):
    # Override for ljspeech
    ckpt_path = join(hydra.utils.get_original_cwd(), "pl_log_exp016_02/epoch26.ckpt")
    print(f"Loading model from {ckpt_path}")
    model = BaselineLightningModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()
    model.cuda()
    
    assert model.ensure_vocoder(verbose=True), "Vocoder not found"
    
    ds = FSDataset("test", cfg)
    print(f"Test set size: {len(ds)}")
    
    random.seed(42)
    indices = random.sample(range(len(ds)), min(20, len(ds)))
    
    ocwd = hydra.utils.get_original_cwd()
    out_dir = join(ocwd, "eval_exp016_04")
    os.makedirs(join(out_dir, "gen"), exist_ok=True)
    os.makedirs(join(out_dir, "gt"), exist_ok=True)
    
    results = []
    
    for i, idx in enumerate(indices):
        sample = ds[idx]
        fid = sample["fid"]
        print(f"[{i+1}/20] {fid}")
        
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
        
        gen_path = join(out_dir, "gen", f"{fid}.wav")
        gt_path = join(out_dir, "gt", f"{fid}.wav")
        sf.write(gen_path, gen_wav, 16000, subtype="PCM_16")
        sf.write(gt_path, gt_wav, 16000, subtype="PCM_16")
        
        results.append({
            "fid": fid,
            "gen_wav_len": len(gen_wav),
            "gt_wav_len": len(gt_wav),
            "mel_length": mel_len.item(),
        })
    
    with open(join(out_dir, "eval_metadata.json"), "w") as f:
        json.dump({"samples": results, "checkpoint": "epoch26.ckpt", "n_samples": len(results)}, f, indent=2)
    
    print(f"\nDone. Generated {len(results)} samples in {out_dir}/")

if __name__ == "__main__":
    main()
