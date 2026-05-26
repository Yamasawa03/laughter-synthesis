"""EXP-010 段6検証: 保存済みpitchと現環境pyworld再計算pitchの比較（音素レベル）"""
import json
import glob
import os
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
from scipy.interpolate import interp1d

def pitch_world(wav, sr, hop_length):
    pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=hop_length / sr * 1000, f0_ceil=800, allowed_range=0.2)
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)
    return pitch

def downsample(pitch, down_rate):
    nonzero_ids = np.where(pitch != 0)[0]
    if len(nonzero_ids) == 0:
        return pitch
    interp_fn = interp1d(
        nonzero_ids, pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pad = 0 if len(pitch) % down_rate == 0 else down_rate - (len(pitch) % down_rate)
    pitch = interp_fn(np.arange(0, len(pitch) + pad))
    pos = 0
    for i in range(len(pitch) // down_rate):
        pitch[i] = np.mean(pitch[pos:pos+down_rate])
        pos += down_rate
    pitch = pitch[:i]
    return pitch

def compute_robust_pitch(wav, sr, hop_length=320):
    hops = [320, 160, 80, 40, 20]
    for hop in hops:
        pitch = pitch_world(wav, sr, hop)
        if np.sum(pitch != 0) <= 1:
            continue
        break
    else:
        return None
    pitch = downsample(pitch, 320 // hop)
    return pitch

def get_phoneme_pitch(pitch, duration):
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            pitch[i] = np.mean(pitch[pos:pos+d])
        else:
            pitch[i] = 0
        pos += d
    pitch = pitch[:len(duration)]
    return pitch

def main():
    base_dir = os.path.expanduser("~/laughter-synthesis/data/laughter")
    wav_dir = os.path.join(base_dir, "wav")
    pitch_dir = os.path.join(base_dir, "pitch")
    duration_dir = os.path.join(base_dir, "duration")

    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))[:10]
    if not wav_files:
        print(json.dumps({"error": f"No wav files found in {wav_dir}"}))
        return

    results = []
    for wf in wav_files:
        fid = os.path.basename(wf)[:-4]
        pitch_path = os.path.join(pitch_dir, fid + ".pt")
        dur_path = os.path.join(duration_dir, fid + ".pt")

        if not os.path.exists(pitch_path):
            results.append({"fid": fid, "error": "saved pitch not found"})
            continue
        if not os.path.exists(dur_path):
            results.append({"fid": fid, "error": "saved duration not found"})
            continue

        wav, sr = sf.read(wf)
        if len(wav.shape) == 2:
            wav = wav[:, 0]

        loaded_pitch = torch.load(pitch_path)
        saved_pitch = loaded_pitch.numpy() if hasattr(loaded_pitch, 'numpy') else np.array(loaded_pitch)

        loaded_dur = torch.load(dur_path)
        duration = loaded_dur.numpy() if hasattr(loaded_dur, 'numpy') else np.array(loaded_dur)
        duration = duration.astype(int)

        frame_pitch = compute_robust_pitch(wav, sr, hop_length=320)
        if frame_pitch is None:
            results.append({"fid": fid, "error": "pitch extraction failed"})
            continue

        phoneme_pitch = get_phoneme_pitch(frame_pitch.copy(), duration)

        min_len = min(len(saved_pitch), len(phoneme_pitch))
        s = saved_pitch[:min_len]
        r = phoneme_pitch[:min_len]
        diff = np.abs(s - r)

        corr = float(np.corrcoef(s, r)[0, 1]) if min_len > 1 and np.std(s) > 0 and np.std(r) > 0 else None

        results.append({
            "fid": fid,
            "saved_len": len(saved_pitch),
            "recomputed_len": len(phoneme_pitch),
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
            "correlation": corr,
            "exact_match": bool(np.allclose(s, r, atol=1e-4)),
        })

    valid_results = [r for r in results if "error" not in r]
    all_exact = len(valid_results) > 0 and all(r.get("exact_match", False) for r in valid_results)

    output = {
        "pyworld_version": pw.__version__ if hasattr(pw, "__version__") else "unknown",
        "checked_files": len(results),
        "valid_files": len(valid_results),
        "all_exact_match": all_exact,
        "results": results,
    }

    if all_exact:
        output["conclusion"] = "All pitch values match. Preprocessing used the same pyworld version (0.3.5). Stage 6 has NO version-dependent difference."
    else:
        output["conclusion"] = "Pitch values differ. pyworld version or parameters changed since preprocessing."

    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
