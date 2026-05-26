"""EXP-005: FastSpeech2 単体品質評価 (MCD)

テストセット160件に対してFS2 best checkpointでmel spectrogramを生成し、
GT melとのMel Cepstral Distortion (MCD)を計算する。

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate laugh
    cd ~/laughter-synthesis
    python scripts/eval_mcd_exp005.py
"""

import os
import sys
import json
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.fftpack import dct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_module import DataModule
from lightning_module import BaselineLightningModule

seed_everything(1024)

OUTPUT_DIR = "/home/yamasawa/laughter-synthesis/exp005_results"
CKPT_PATH = "/home/yamasawa/laughter-synthesis/pl_log_exp004/epoch=21-step=7568.ckpt"
N_MFCC = 13


def mel_to_mfcc(mel_spectrogram, n_mfcc=N_MFCC):
    """mel spectrogram (80, T) -> MFCC (n_mfcc, T)

    Input is already log-mel + z-score normalized, so DCT is applied directly.
    """
    mfcc = dct(mel_spectrogram, type=2, axis=0, norm='ortho')
    return mfcc[1:n_mfcc + 1, :]


def compute_mcd(pred_mel, gt_mel):
    """Compute MCD between predicted and GT mel spectrograms.

    Both inputs: (80, T) numpy arrays.
    Returns MCD in dB.
    """
    min_len = min(pred_mel.shape[1], gt_mel.shape[1])
    pred_mel = pred_mel[:, :min_len]
    gt_mel = gt_mel[:, :min_len]

    pred_mfcc = mel_to_mfcc(pred_mel)
    gt_mfcc = mel_to_mfcc(gt_mel)

    diff = pred_mfcc - gt_mfcc
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))

    # MCD = (10 * sqrt(2) / ln(10)) * mean(frame_distances)
    mcd = (10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(frame_dist)
    return mcd


class MelCollector(pl.Callback):
    """Collects mel predictions and GT from test_step outputs."""

    def __init__(self):
        self.results = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.results.append(outputs)


@hydra.main(config_path='../config', config_name='default')
def main(cfg: DictConfig):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading checkpoint: {CKPT_PATH}")
    datamodule = DataModule(cfg)
    module = BaselineLightningModule(cfg)

    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    missing, unexpected = module.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)} (vocoder weights, expected)")

    collector = MelCollector()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        callbacks=[collector],
        logger=False,
    )

    # Disable test_epoch_end (it tries to log figures/audio which requires a logger)
    module.test_epoch_end = lambda outputs: None

    print("Running inference on test set (160 items)...")
    trainer.test(module, datamodule=datamodule)

    print(f"Collected {len(collector.results)} batches")

    # Compute MCD for each item
    mcd_results = []
    pred_mel_dir = os.path.join(OUTPUT_DIR, "pred_mel")
    gt_mel_dir = os.path.join(OUTPUT_DIR, "gt_mel")
    os.makedirs(pred_mel_dir, exist_ok=True)
    os.makedirs(gt_mel_dir, exist_ok=True)

    for batch in collector.results:
        mel_pred = batch['mel_pred'].numpy()  # (B, 80, T)
        mel_gt = batch['mel_gt'].numpy()      # (B, 80, T)
        mel_length = batch['mel_length'].numpy()
        mel_length_gt = batch['mel_length_gt'].numpy()
        fids = batch['fid']

        for i in range(mel_pred.shape[0]):
            fid = fids[i]
            pred = mel_pred[i, :, :mel_length[i]]
            gt = mel_gt[i, :, :mel_length_gt[i]]

            np.save(os.path.join(pred_mel_dir, f"{fid}.npy"), pred)
            np.save(os.path.join(gt_mel_dir, f"{fid}.npy"), gt)

            mcd = compute_mcd(pred, gt)
            mcd_results.append({
                "fid": fid,
                "mcd_db": float(mcd),
                "pred_frames": int(mel_length[i]),
                "gt_frames": int(mel_length_gt[i]),
                "duration_ratio": float(mel_length[i]) / float(mel_length_gt[i]),
            })

    mcd_values = [r["mcd_db"] for r in mcd_results]
    stats = {
        "n_items": len(mcd_values),
        "mean": float(np.mean(mcd_values)),
        "median": float(np.median(mcd_values)),
        "std": float(np.std(mcd_values)),
        "min": float(np.min(mcd_values)),
        "max": float(np.max(mcd_values)),
        "q25": float(np.percentile(mcd_values, 25)),
        "q75": float(np.percentile(mcd_values, 75)),
    }

    output = {
        "experiment": "EXP-005",
        "checkpoint": CKPT_PATH,
        "n_mfcc": N_MFCC,
        "stats": stats,
        "items": mcd_results,
    }

    json_path = os.path.join(OUTPUT_DIR, "mcd_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")
    print(f"  Mean MCD: {stats['mean']:.2f} dB")
    print(f"  Median MCD: {stats['median']:.2f} dB")
    print(f"  Std MCD: {stats['std']:.2f} dB")
    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}] dB")

    # Histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(mcd_values, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.2f} dB")
    ax.axvline(stats['median'], color='blue', linestyle='--', label=f"Median: {stats['median']:.2f} dB")
    ax.set_xlabel("MCD (dB)")
    ax.set_ylabel("Count")
    ax.set_title("EXP-005: FastSpeech2 MCD Distribution (N=160)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png_path = os.path.join(OUTPUT_DIR, "mcd_distribution.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distribution plot saved to: {png_path}")


if __name__ == '__main__':
    main()
