"""EXP-010 段2検証: リサンプリングが実際に発生するか、発生する場合の差分を測定"""
import json
import glob
import soundfile as sf
import os

def main():
    wav_dir = os.path.expanduser("~/laughter-synthesis/data/laughter/wav")
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))[:10]

    if not wav_files:
        print(json.dumps({"error": f"No wav files found in {wav_dir}"}))
        return

    results = []
    for wf in wav_files:
        fid = os.path.basename(wf)[:-4]
        info = sf.info(wf)
        results.append({"fid": fid, "samplerate": info.samplerate, "frames": info.frames})

    target_sr = 16000
    all_match = all(r["samplerate"] == target_sr for r in results)

    output = {
        "target_sr": target_sr,
        "checked_files": len(results),
        "all_already_target_sr": all_match,
        "files": results,
    }

    if all_match:
        output["conclusion"] = "All source wavs are already 16kHz. librosa.resample is never called. Stage 2 (resampling) has NO version-dependent risk."
    else:
        import librosa
        import numpy as np
        diffs = []
        for r in results:
            if r["samplerate"] != target_sr:
                wav, sr = sf.read(os.path.join(wav_dir, r["fid"] + ".wav"))
                resampled_kaiser = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
                resampled_soxr = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
                diff = np.abs(resampled_kaiser - resampled_soxr[:len(resampled_kaiser)])
                diffs.append({
                    "fid": r["fid"],
                    "orig_sr": r["samplerate"],
                    "max_abs_diff": float(np.max(diff)),
                    "mean_abs_diff": float(np.mean(diff)),
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                })
        output["resample_diffs"] = diffs
        output["conclusion"] = "Some files require resampling. See resample_diffs for version-dependent differences."

    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
