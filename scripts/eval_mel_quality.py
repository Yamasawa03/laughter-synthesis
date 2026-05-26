"""EXP-016-03: Evaluate mel quality of trained FS2 checkpoint.

Teacher-forced inference (GT duration/pitch/energy) to compute mel prediction
quality against ground truth. No vocoder needed.

Usage:
    python scripts/eval_mel_quality.py \
        --ckpt_path pl_log_exp016_02/epoch26.ckpt \
        --split test \
        --preprocess ljspeech \
        --dataset ljspeech \
        --output_json eval_results.json
"""
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from omegaconf import OmegaConf
import yaml

def load_config(preprocess_name, dataset_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, "config/default.yaml")) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(base_dir, f"config/preprocess/{preprocess_name}.yaml")) as f:
        preprocess = yaml.safe_load(f)
    with open(os.path.join(base_dir, f"config/dataset/{dataset_name}.yaml")) as f:
        dataset = yaml.safe_load(f)
    with open(os.path.join(base_dir, "config/model/default.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "config/train/default.yaml")) as f:
        train = yaml.safe_load(f)

    cfg = OmegaConf.create({
        "preprocess": preprocess, "dataset": dataset, "model": model_cfg,
        "train": train, "use_tb": False, "log_dir": "eval_tmp"
    })
    return cfg, base_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--preprocess", default="ljspeech")
    parser.add_argument("--dataset", default="ljspeech")
    parser.add_argument("--output_json", default="eval_results.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    cfg, base_dir = load_config(args.preprocess, args.dataset)

    import hydra.utils
    hydra.utils.get_original_cwd = lambda: base_dir

    from data_module import DataModule
    from lightning_module import BaselineLightningModule
    from model.loss import FastSpeech2Loss

    dm = DataModule(cfg)
    if args.split == "test":
        dl = dm.test_dataloader()
    else:
        dl = dm.val_dataloader()

    model_module = BaselineLightningModule.load_from_checkpoint(
        args.ckpt_path, cfg=cfg, map_location="cpu"
    )
    model = model_module.model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    criterion = FastSpeech2Loss(cfg)

    all_results = []
    total_mel_mse = 0.0
    total_mel_postnet_mse = 0.0
    total_pitch_mse = 0.0
    total_energy_mse = 0.0
    total_duration_mse = 0.0
    n_samples = 0

    print(f"Evaluating {args.ckpt_path} on {args.split} set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            output = model(batch)
            loss_dict = criterion(batch, output, torch.tensor(0))

            bs = batch["text_length"].shape[0]
            n_samples += bs

            for k in ["loss", "mel_loss", "mel_postnet_loss", "pitch_loss", "energy_loss", "duration_loss"]:
                val = loss_dict[k].item()
                if k == "mel_loss":
                    total_mel_mse += val * bs
                elif k == "mel_postnet_loss":
                    total_mel_postnet_mse += val * bs
                elif k == "pitch_loss":
                    total_pitch_mse += val * bs
                elif k == "energy_loss":
                    total_energy_mse += val * bs
                elif k == "duration_loss":
                    total_duration_mse += val * bs

            if batch_idx % 10 == 0:
                print(f"  batch {batch_idx}/{len(dl)}: mel_loss={loss_dict['mel_loss'].item():.4f}")

    results = {
        "checkpoint": args.ckpt_path,
        "split": args.split,
        "n_samples": n_samples,
        "mel_mae": total_mel_mse / n_samples,
        "mel_postnet_mae": total_mel_postnet_mse / n_samples,
        "pitch_mse": total_pitch_mse / n_samples,
        "energy_mse": total_energy_mse / n_samples,
        "duration_mse": total_duration_mse / n_samples,
        "total_loss": (total_mel_mse + total_mel_postnet_mse + total_pitch_mse + total_energy_mse + total_duration_mse) / n_samples,
    }

    print(f"\n=== Results ({args.split} set, {n_samples} samples) ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
