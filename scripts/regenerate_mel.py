"""Regenerate mel spectrograms with updated filterbank (norm=None).

Only mel is affected by the librosa.filters.mel norm change.
Duration, pitch, energy, code are unchanged.
"""
import os
import sys
import torch
import numpy as np
from os.path import join, exists
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from hifigan import meldataset

ROOT = os.path.expanduser("~/laughter-synthesis/data/laughter")
N_FFT = 1024
N_MEL = 80
SR = 16000
HOP_LENGTH = 320
WIN_LENGTH = 1024
FMIN = 0
FMAX = 8000


def regenerate_mel(root):
    mel_dir = join(root, "mel")
    wav_dir = join(root, "wav")
    dur_dir = join(root, "duration")

    fids = sorted([f[:-3] for f in os.listdir(mel_dir) if f.endswith(".pt")])
    print(f"Regenerating mel for {len(fids)} files")

    changed = 0
    skipped = 0
    for fid in tqdm(fids):
        wav_path = join(wav_dir, fid + ".wav")
        dur_path = join(dur_dir, fid + ".pt")
        mel_path = join(mel_dir, fid + ".pt")

        if not exists(wav_path) or not exists(dur_path):
            skipped += 1
            continue

        wav, sr = utils.read_audio(wav_path)
        assert sr == SR, f"{fid} has sr={sr}, expected {SR}"

        duration = torch.load(dur_path, map_location="cpu")
        nframe = int(duration.sum()) if isinstance(duration, torch.Tensor) else int(sum(duration))

        audio = torch.clip(torch.from_numpy(wav), -1, 1)
        _, mel_spec = meldataset.mel_spectrogram(
            audio.float().unsqueeze(0),
            N_FFT, N_MEL, SR, HOP_LENGTH, WIN_LENGTH, FMIN, FMAX,
        )
        mel_spec = mel_spec.squeeze(0)
        mel_spec = mel_spec[:, :nframe]

        torch.save(mel_spec, mel_path)
        changed += 1

    print(f"Done: {changed} regenerated, {skipped} skipped")


if __name__ == "__main__":
    regenerate_mel(ROOT)
