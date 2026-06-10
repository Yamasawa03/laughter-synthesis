"""Check old/new predicted mel tensors for deterministic consistency."""

import argparse
import json
import os

import torch


def read_file_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line.split("|", 1)[0].strip())
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_dir", required=True)
    parser.add_argument("--new_dir", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []
    for fid in read_file_ids(args.file_list)[:args.n_samples]:
        old_path = os.path.join(args.old_dir, f"{fid}.pt")
        new_path = os.path.join(args.new_dir, f"{fid}.pt")
        item = {"fid": fid, "old_path": old_path, "new_path": new_path}
        try:
            old = torch.load(old_path, map_location="cpu")
            new = torch.load(new_path, map_location="cpu")
            if not isinstance(old, torch.Tensor) or not isinstance(new, torch.Tensor):
                raise TypeError(f"expected tensors, got {type(old)} / {type(new)}")
            item["old_shape"] = list(old.shape)
            item["new_shape"] = list(new.shape)
            item["shape_match"] = tuple(old.shape) == tuple(new.shape)
            item["allclose"] = bool(item["shape_match"] and torch.allclose(old, new, atol=1e-5, rtol=1e-4))
            item["max_abs_diff"] = float(torch.max(torch.abs(old - new)).item()) if item["shape_match"] else None
        except Exception as exc:
            item["shape_match"] = False
            item["allclose"] = False
            item["error"] = str(exc)
        results.append(item)

    output = {
        "n_checked": len(results),
        "all_allclose": all(item["allclose"] for item in results),
        "items": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Checked {len(results)} files, all_allclose={output['all_allclose']}")


if __name__ == "__main__":
    main()
