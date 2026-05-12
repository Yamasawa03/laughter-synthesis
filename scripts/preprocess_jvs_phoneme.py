import os
import random
import re
import subprocess
from collections import defaultdict
from os.path import exists, join

import hydra
import hydra.utils
import librosa
import numpy as np
import pyopenjtalk
import torch
from tqdm import tqdm

import utils
from dataset.jvs_phoneme import JVSPhoneme
from preprocessor import Preprocessor


def resolve_path(orig_cwd, path):
    if os.path.isabs(path):
        return path
    return join(orig_cwd, path)


def ensure_dirs(root):
    for name in ["wav", "code", "duration", "speaker"]:
        os.makedirs(join(root, name), exist_ok=True)


def text_to_phonemes(text):
    phones = []
    for phone in pyopenjtalk.g2p(text, kana=False).split():
        if phone in ("sil", "pau"):
            phone = "pau"
        if phone:
            phones.append(phone)
    return phones


def write_resampled_wav(src_path, tgt_path, cfg):
    if exists(tgt_path):
        return True
    wav, sr = utils.read_audio(src_path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    wav = wav if sr == cfg.audio.sr else librosa.resample(wav, orig_sr=sr, target_sr=cfg.audio.sr)
    wav, _ = librosa.effects.trim(
        wav,
        top_db=cfg.audio.top_db,
        frame_length=cfg.stft.window_length,
        hop_length=cfg.stft.hop_length,
    )
    if wav.size == 0 or np.max(np.abs(wav)) == 0:
        return False
    wav = wav / np.max(np.abs(wav)) * cfg.audio.max_wav_value
    utils.write_audio(wav.astype(np.int16), tgt_path, cfg.audio.sr, "PCM_16")
    return True


def write_phoneme_files(fid, text, phone2id, code_dir, lab_dir):
    code_path = join(code_dir, fid + ".txt")
    lab_path = join(lab_dir, fid + ".lab")
    if exists(code_path) and exists(lab_path):
        with open(lab_path, encoding="utf-8") as f:
            return f.read().strip().split()

    phones = text_to_phonemes(text)
    unknown = sorted({phone for phone in phones if phone not in phone2id})
    if unknown:
        raise ValueError(f"{fid} contains unknown phonemes: {unknown}")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(phone2id[phone]) for phone in phones))
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write(" ".join(phones))
    return phones


def write_mfa_dictionary(path, phonemes):
    if exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        for phone in phonemes:
            f.write(f"{phone}\t{phone}\n")


def run_mfa(wav_dir, dictionary_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if any(name.endswith(".TextGrid") for name in os.listdir(output_dir)):
        return
    acoustic_model = os.environ.get("MFA_ACOUSTIC_MODEL", "japanese_mfa")
    command = os.environ.get("MFA_COMMAND", "mfa")
    cmd = [command, "align", wav_dir, dictionary_path, acoustic_model, output_dir, "--clean"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def normalize_mfa_phone(phone):
    phone = phone.strip()
    if phone in ("", "sil", "sp", "spn"):
        return None
    if phone == "silB" or phone == "silE":
        return "pau"
    return phone


def read_textgrid_intervals(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    intervals = []
    pattern = re.compile(
        r"xmin =\s*([0-9.]+)\s+xmax =\s*([0-9.]+)\s+text =\s*\"([^\"]*)\"",
        re.MULTILINE,
    )
    for start, end, label in pattern.findall(content):
        phone = normalize_mfa_phone(label)
        if phone is not None:
            intervals.append((float(start), float(end), phone))
    return intervals


def write_duration_from_textgrid(fid, phones, textgrid_dir, duration_dir, hop_length_sec):
    duration_path = join(duration_dir, fid + ".pt")
    if exists(duration_path):
        return True
    textgrid_path = join(textgrid_dir, fid + ".TextGrid")
    if not exists(textgrid_path):
        return False
    intervals = read_textgrid_intervals(textgrid_path)
    aligned_phones = [phone for _, _, phone in intervals]
    if aligned_phones != phones:
        print(f"Skip {fid}: MFA phones do not match G2P phones")
        return False
    durations = [max(1, int(round((end - start) / hop_length_sec))) for start, end, _ in intervals]
    torch.save(np.array(durations, dtype=np.int64), duration_path)
    return True


def write_speakers(root, dataset, fids):
    speaker_path = join(root, "speaker.pt")
    if not exists(speaker_path):
        speakers = {speaker: idx for idx, speaker in enumerate(dataset.collect_speakers())}
        torch.save(speakers, speaker_path)
    else:
        speakers = torch.load(speaker_path)

    speaker_dir = join(root, "speaker")
    os.makedirs(speaker_dir, exist_ok=True)
    for fid in fids:
        path = join(speaker_dir, fid + ".txt")
        if exists(path):
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(dataset.get_speaker(fid))
    print(f"total speaker number: {len(speakers)}")


def split_filelists(dataset, processed_fids, cfg, orig_cwd):
    rng = random.Random(1234)
    processed = set(processed_fids)
    speaker2fids = defaultdict(list)
    for fid in processed_fids:
        speaker2fids[dataset.get_speaker(fid)].append((fid,))

    train_filelist, val_filelist, test_filelist = [], [], []
    val_speakers = int(cfg.view.val)
    test_wav_per_speaker = int(cfg.view.test_wav_per_speaker)
    for idx, speaker in enumerate(sorted(speaker2fids)):
        fids = speaker2fids[speaker]
        fids = [item for item in fids if item[0] in processed]
        rng.shuffle(fids)
        test, val = [], []
        if len(fids) > test_wav_per_speaker:
            fids, test = fids[:-test_wav_per_speaker], fids[-test_wav_per_speaker:]
        if idx < val_speakers and len(fids) > 1:
            fids, val = fids[:-1], fids[-1:]
        train_filelist.extend(fids)
        val_filelist.extend(val)
        test_filelist.extend(test)

    print(f"train: {len(train_filelist)}, val: {len(val_filelist)}, test: {len(test_filelist)}")
    for filelist, path in [
        (train_filelist, cfg.view.train_filelist),
        (val_filelist, cfg.view.val_filelist),
        (test_filelist, cfg.view.test_filelist),
    ]:
        out_path = resolve_path(orig_cwd, path)
        if not exists(out_path):
            utils.write_filelist(filelist, out_path)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def preprocess(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    cfg = cfg.preprocess
    dataset = JVSPhoneme(resolve_path(orig_cwd, cfg.path.jvs.path))
    processed_root = resolve_path(orig_cwd, cfg.path.processed_path)
    ensure_dirs(processed_root)

    phoneme_list = list(cfg.code.phoneme_list)
    phone2id = {phone: idx for idx, phone in enumerate(phoneme_list)}
    if len(phone2id) != int(cfg.code.nclusters):
        raise ValueError("code.nclusters must match the phoneme_list length")

    wav_dir = join(processed_root, "wav")
    code_dir = join(processed_root, "code")
    duration_dir = join(processed_root, "duration")
    mfa_dir = join(processed_root, "mfa")
    textgrid_dir = join(processed_root, "textgrid")
    os.makedirs(mfa_dir, exist_ok=True)
    os.makedirs(textgrid_dir, exist_ok=True)

    print(f"Collected {len(dataset.filelist)} files")
    fid2phones = {}
    valid_fids = []
    for wav_path in tqdm(dataset.filelist):
        fid = dataset.get_fid(wav_path)
        tgt_wav_path = join(wav_dir, fid + ".wav")
        if not write_resampled_wav(wav_path, tgt_wav_path, cfg):
            print(f"Skip {fid}: empty wav after trim")
            continue
        text = dataset.get_transcript(fid)
        try:
            phones = write_phoneme_files(fid, text, phone2id, code_dir, wav_dir)
        except ValueError as exc:
            print(exc)
            continue
        fid2phones[fid] = phones
        valid_fids.append(fid)

    dictionary_path = join(mfa_dir, "phoneme.dict")
    write_mfa_dictionary(dictionary_path, phoneme_list)
    run_mfa(wav_dir, dictionary_path, textgrid_dir)

    hop_length_sec = cfg.stft.hop_length / cfg.audio.sr
    aligned_fids = []
    for fid in tqdm(valid_fids):
        if write_duration_from_textgrid(fid, fid2phones[fid], textgrid_dir, duration_dir, hop_length_sec):
            aligned_fids.append(fid)
    print(f"MFA aligned: {len(aligned_fids)} / {len(valid_fids)}")

    write_speakers(processed_root, dataset, aligned_fids)

    print("Dump acoustic features")
    preprocessor = Preprocessor(cfg)
    processed_fids = preprocessor.process(processed_root, aligned_fids)
    print(f"preprocess done, before: {len(aligned_fids)}, after: {len(processed_fids)}")
    split_filelists(dataset, processed_fids, cfg, orig_cwd)


if __name__ == "__main__":
    preprocess()
