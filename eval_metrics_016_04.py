"""EXP-016-04: Compute MCD, PESQ, STOI for generated vs GT wavs"""
import os
import json
import numpy as np
import librosa
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from os.path import join

def compute_mcd(gen_wav, gt_wav, sr=16000, n_mfcc=13):
    """Compute Mel Cepstral Distortion between two waveforms"""
    gen_mfcc = librosa.feature.mfcc(y=gen_wav.astype(float), sr=sr, n_mfcc=n_mfcc)
    gt_mfcc = librosa.feature.mfcc(y=gt_wav.astype(float), sr=sr, n_mfcc=n_mfcc)
    min_len = min(gen_mfcc.shape[1], gt_mfcc.shape[1])
    gen_mfcc = gen_mfcc[:, :min_len]
    gt_mfcc = gt_mfcc[:, :min_len]
    diff = gen_mfcc - gt_mfcc
    mcd = np.mean(np.sqrt(2 * np.sum(diff[1:]**2, axis=0)))  # skip c0
    return float(mcd)

def main():
    base_dir = "eval_exp016_04"
    gen_dir = join(base_dir, "gen")
    gt_dir = join(base_dir, "gt")
    
    with open(join(base_dir, "eval_metadata.json")) as f:
        metadata = json.load(f)
    
    results = []
    for sample in metadata["samples"]:
        fid = sample["fid"]
        gen_wav, sr = sf.read(join(gen_dir, f"{fid}.wav"))
        gt_wav, sr2 = sf.read(join(gt_dir, f"{fid}.wav"))
        
        # Align lengths
        min_len = min(len(gen_wav), len(gt_wav))
        gen_aligned = gen_wav[:min_len]
        gt_aligned = gt_wav[:min_len]
        
        # MCD
        mcd_val = compute_mcd(gen_aligned, gt_aligned, sr=16000)
        
        # PESQ (needs 16kHz)
        try:
            pesq_val = pesq(16000, gt_aligned, gen_aligned, "wb")
        except Exception as e:
            pesq_val = None
            print(f"  PESQ failed for {fid}: {e}")
        
        # STOI
        try:
            stoi_val = stoi(gt_aligned, gen_aligned, 16000, extended=False)
        except Exception as e:
            stoi_val = None
            print(f"  STOI failed for {fid}: {e}")
        
        r = {"fid": fid, "mcd": mcd_val, "pesq": pesq_val, "stoi": stoi_val}
        results.append(r)
        pesq_str = f"{pesq_val:.3f}" if pesq_val is not None else "N/A"
        stoi_str = f"{stoi_val:.3f}" if stoi_val is not None else "N/A"
        print(f"{fid}: MCD={mcd_val:.2f}, PESQ={pesq_str}, STOI={stoi_str}")
    
    # Aggregate
    valid_mcd = [r["mcd"] for r in results if r["mcd"] is not None]
    valid_pesq = [r["pesq"] for r in results if r["pesq"] is not None]
    valid_stoi = [r["stoi"] for r in results if r["stoi"] is not None]
    
    summary = {
        "exp_id": "EXP-016-04",
        "checkpoint": "epoch26",
        "vocoder": "EXP-017 fmin=0 (g_00130000)",
        "n_samples": len(results),
        "mcd": {
            "mean": float(np.mean(valid_mcd)),
            "median": float(np.median(valid_mcd)),
            "std": float(np.std(valid_mcd)),
        },
        "pesq": {
            "mean": float(np.mean(valid_pesq)) if valid_pesq else None,
            "median": float(np.median(valid_pesq)) if valid_pesq else None,
            "std": float(np.std(valid_pesq)) if valid_pesq else None,
        },
        "stoi": {
            "mean": float(np.mean(valid_stoi)) if valid_stoi else None,
            "median": float(np.median(valid_stoi)) if valid_stoi else None,
            "std": float(np.std(valid_stoi)) if valid_stoi else None,
        },
        "per_sample": results,
    }
    
    out_path = join(base_dir, "eval_results_wav.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Summary ===")
    print(f"MCD:  mean={summary[mcd][mean]:.2f}, median={summary[mcd][median]:.2f}, std={summary[mcd][std]:.2f}")
    if valid_pesq:
        print(f"PESQ: mean={summary[pesq][mean]:.3f}, median={summary[pesq][median]:.3f}, std={summary[pesq][std]:.3f}")
    if valid_stoi:
        print(f"STOI: mean={summary[stoi][mean]:.3f}, median={summary[stoi][median]:.3f}, std={summary[stoi][std]:.3f}")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
