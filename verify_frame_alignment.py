"""Verify frame count alignment between HuBERT (16kHz/hop320) and mel (22050Hz/hop441).
Both should produce 50Hz frames, but resampling edge effects may cause 1-2 frame differences.
Threshold: <= 2 frames difference (preprocessor.py allows up to 3)."""

import sys
import torch
import librosa
from module.ssl import SSLWrapper
from utils import load_audio_with_resample

SAMPLES = [
    "/home/yamasawa/data/LJSpeech-1.1/wavs/LJ001-0001.wav",
    "/home/yamasawa/data/LJSpeech-1.1/wavs/LJ028-0205.wav",
    "/home/yamasawa/data/LJSpeech-1.1/wavs/LJ050-0269.wav",
]

def main():
    ssl_model = SSLWrapper("facebook/hubert-large-ll60k").cuda().eval()
    all_pass = True

    for wav_path in SAMPLES:
        audio_16k, _ = load_audio_with_resample(wav_path, to_torch=True)
        with torch.no_grad():
            feat = ssl_model(audio_16k.cuda())
        hubert_frames = feat[11].squeeze(0).shape[0]

        y, sr = librosa.load(wav_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=22050, hop_length=441, n_fft=1024, n_mels=80)
        mel_frames = mel.shape[1]

        diff = abs(hubert_frames - mel_frames)
        status = "OK" if diff <= 2 else "FAIL"
        if diff > 2:
            all_pass = False
        print(f"{wav_path}: hubert={hubert_frames}, mel={mel_frames}, diff={diff} [{status}]")

    if all_pass:
        print("\nALL PASS: frame alignment within tolerance (<=2)")
    else:
        print("\nFAIL: frame alignment exceeds tolerance. Return to EXP-039-02.")
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
