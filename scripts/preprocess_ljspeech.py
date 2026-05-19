import os
import random
import re
from os.path import exists, join

import hydra
import hydra.utils
import librosa
import numpy as np
import torch
from tqdm import tqdm

import utils
from dataset.ljspeech import LJSpeech
from preprocessor import Preprocessor


def resolve_path(orig_cwd, path):
    if os.path.isabs(path):
        return path
    return join(orig_cwd, path)


def ensure_dirs(root):
    for name in ["wav", "code", "duration", "speaker"]:
        os.makedirs(join(root, name), exist_ok=True)


def normalize_ljspeech_phone(phone):
    phone = phone.strip()
    if phone in ("", "sil", "spn"):
        return None
    return phone


def read_textgrid_intervals(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    phones_match = re.search(r'name\s*=\s*"phones".*?(?=item\s*\[|$)', content, re.DOTALL)
    if phones_match is None:
        return []
    phones_section = phones_match.group(0)
    intervals = []
    pattern = re.compile(
        r"xmin =\s*([0-9.]+)\s+xmax =\s*([0-9.]+)\s+text =\s*\"([^\"]*)\"",
        re.MULTILINE,
    )
    for start, end, label in pattern.findall(phones_section):
        phone = normalize_ljspeech_phone(label)
        if phone is not None:
            intervals.append((float(start), float(end), phone))
    return intervals


def write_ljspeech_item(fid, wav_path, textgrid_path, cfg, phone2id, processed_root, hop_length_sec):
    if not exists(textgrid_path):
        print(f"Skip {fid}: missing TextGrid")
        return False

    intervals = read_textgrid_intervals(textgrid_path)
    if not intervals:
        print(f"Skip {fid}: no valid phoneme intervals")
        return False

    phones = [phone for _, _, phone in intervals]
    unknown = sorted({phone for phone in phones if phone not in phone2id})
    if unknown:
        print(f"Skip {fid}: unknown phonemes {unknown}")
        return False

    start = intervals[0][0]
    end = intervals[-1][1]
    if end <= start:
        print(f"Skip {fid}: invalid trim region")
        return False

    wav, sr = utils.read_audio(wav_path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    wav = wav[int(start * sr): int(end * sr)]
    if wav.size == 0:
        print(f"Skip {fid}: empty wav after trim")
        return False

    if sr != cfg.audio.sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg.audio.sr)
    if wav.size == 0 or np.max(np.abs(wav)) == 0:
        print(f"Skip {fid}: silent wav after resample")
        return False

    wav = wav / np.max(np.abs(wav)) * cfg.audio.max_wav_value
    utils.write_audio(
        wav.astype(np.int16),
        join(processed_root, "wav", fid + ".wav"),
        cfg.audio.sr,
        "PCM_16",
    )

    with open(join(processed_root, "code", fid + ".txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(str(phone2id[phone]) for phone in phones))

    durations = [
        max(1, int(round((end - start) / hop_length_sec)))
        for start, end, _ in intervals
    ]
    torch.save(np.array(durations, dtype=np.int64), join(processed_root, "duration", fid + ".pt"))
    return True


def write_speakers(root, dataset, fids):
    speaker_path = join(root, "speaker.pt")
    if not exists(speaker_path):
        torch.save({dataset.SPEAKER: 0}, speaker_path)
    else:
        speakers = torch.load(speaker_path)
        if dataset.SPEAKER not in speakers:
            raise ValueError(f"speaker.pt does not contain {dataset.SPEAKER}")

    speaker_dir = join(root, "speaker")
    os.makedirs(speaker_dir, exist_ok=True)
    for fid in fids:
        path = join(speaker_dir, fid + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(dataset.get_speaker(fid))
    print("total speaker number: 1")


def split_filelists(processed_fids, cfg, orig_cwd):
    rng = random.Random(1234)
    fids = [(fid,) for fid in processed_fids]
    rng.shuffle(fids)

    test_size = 500
    val_size = 500
    test_filelist = fids[-test_size:] if len(fids) > test_size else fids[:]
    remaining = fids[:-test_size] if len(fids) > test_size else []
    val_filelist = remaining[-val_size:] if len(remaining) > val_size else remaining[:]
    train_filelist = remaining[:-val_size] if len(remaining) > val_size else []

    print(f"train: {len(train_filelist)}, val: {len(val_filelist)}, test: {len(test_filelist)}")
    for filelist, path in [
        (train_filelist, cfg.view.train_filelist),
        (val_filelist, cfg.view.val_filelist),
        (test_filelist, cfg.view.test_filelist),
    ]:
        utils.write_filelist(filelist, resolve_path(orig_cwd, path))


@hydra.main(config_path="../config", config_name="default")
def preprocess(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    cfg = cfg.preprocess
    dataset = LJSpeech(
        resolve_path(orig_cwd, cfg.path.ljspeech.wav_dir),
        resolve_path(orig_cwd, cfg.path.ljspeech.textgrid_dir),
    )
    processed_root = resolve_path(orig_cwd, cfg.path.processed_path)
    ensure_dirs(processed_root)

    phoneme_list = list(cfg.code.phoneme_list)
    phone2id = {phone: idx for idx, phone in enumerate(phoneme_list)}
    if len(phone2id) != int(cfg.code.nclusters):
        raise ValueError("code.nclusters must match the phoneme_list length")

    hop_length_sec = cfg.stft.hop_length / cfg.audio.sr
    print(f"Collected {len(dataset.filelist)} files")
    valid_fids = []
    for wav_path in tqdm(dataset.filelist):
        fid = dataset.get_fid(wav_path)
        if write_ljspeech_item(
            fid,
            wav_path,
            dataset.get_textgrid_path(fid),
            cfg,
            phone2id,
            processed_root,
            hop_length_sec,
        ):
            valid_fids.append(fid)
    print(f"TextGrid aligned: {len(valid_fids)} / {len(dataset.filelist)}")

    write_speakers(processed_root, dataset, valid_fids)

    print("Dump acoustic features")
    preprocessor = Preprocessor(cfg)
    processed_fids = preprocessor.process(processed_root, valid_fids)
    print(f"preprocess done, before: {len(valid_fids)}, after: {len(processed_fids)}")
    split_filelists(processed_fids, cfg, orig_cwd)


if __name__ == "__main__":
    preprocess()
