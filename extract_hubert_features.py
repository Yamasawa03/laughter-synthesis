"""Extract HuBERT-Large Layer 12 features for all files in a filelist.
Uses ssl_features_generator from speech2unit.py (saves .pt files to ./data/hubert/12/).
Existing .pt files are skipped automatically."""

import sys
from speech2unit import load_ssl_model, ssl_features_generator

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filelist_path>")
        sys.exit(1)

    filelist_path = sys.argv[1]
    filelist = [line.strip() for line in open(filelist_path) if line.strip()]
    print(f"Processing {len(filelist)} files from {filelist_path}")

    ssl_model = load_ssl_model("facebook/hubert-large-ll60k")
    count = 0
    for _ in ssl_features_generator(filelist, ssl_model, "hubert", "facebook/hubert-large-ll60k", 12):
        count += 1
    print(f"Done: {count}/{len(filelist)} files processed")

if __name__ == "__main__":
    main()
