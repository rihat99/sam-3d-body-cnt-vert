#!/usr/bin/env python3
"""
One-time migration script: split combined MHR NPZ files into separate
contact and detect files.

Input (combined):
    hot_dca_trainval_mhr.npz  — keys: imgname, contact_label, bbox, cam_k
    hot_dca_test_mhr.npz      — keys: imgname, contact_label, bbox, cam_k

Output (split):
    hot_dca_trainval_contact_lod1.npz  — keys: imgname, contact_label
    hot_dca_trainval_detect.npz        — keys: imgname, bbox, cam_k
    hot_dca_test_contact_lod1.npz      — keys: imgname, contact_label
    hot_dca_test_detect.npz            — keys: imgname, bbox, cam_k

Usage:
    python dataset/split_existing_npz.py --dir ./dataset/damon_mhr_contact
"""

import argparse
import os
import numpy as np


LOD = 1  # existing data is LOD1 (18439 verts)


def split_file(npz_path: str, out_dir: str) -> None:
    print(f"\nProcessing: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    keys = list(data.keys())
    print(f"  Keys found: {keys}")

    imgnames = data["imgname"]
    num_samples = len(imgnames)
    print(f"  Samples: {num_samples}")

    # --- contact file ---
    base = os.path.basename(npz_path).replace("_mhr.npz", "")
    contact_path = os.path.join(out_dir, f"{base}_contact_lod{LOD}.npz")
    np.savez(contact_path, imgname=imgnames, contact_label=data["contact_label"])
    print(f"  Saved contact file: {contact_path}")
    print(f"    contact_label shape: {data['contact_label'].shape}")

    # --- detect file ---
    detect_path = os.path.join(out_dir, f"{base}_detect.npz")
    detect_kwargs = {"imgname": imgnames}
    if "bbox" in keys:
        detect_kwargs["bbox"] = data["bbox"]
        print(f"    bbox shape: {data['bbox'].shape}")
    else:
        print("  WARNING: no 'bbox' key found — detect file will lack bbox")
    if "cam_k" in keys:
        detect_kwargs["cam_k"] = data["cam_k"]
        print(f"    cam_k shape: {data['cam_k'].shape}")
    else:
        print("  WARNING: no 'cam_k' key found — detect file will lack cam_k")
    np.savez(detect_path, **detect_kwargs)
    print(f"  Saved detect file:  {detect_path}")

    # --- verify ---
    c = np.load(contact_path, allow_pickle=True)
    d = np.load(detect_path, allow_pickle=True)
    assert (c["imgname"] == imgnames).all(), "imgname mismatch in contact file"
    assert (d["imgname"] == imgnames).all(), "imgname mismatch in detect file"
    print(f"  Verification passed.")


def main():
    parser = argparse.ArgumentParser(description="Split combined MHR NPZ files")
    parser.add_argument(
        "--dir",
        type=str,
        default="./dataset/damon_mhr_contact",
        help="Directory containing the combined NPZ files",
    )
    args = parser.parse_args()

    files = [
        "hot_dca_trainval_mhr.npz",
        "hot_dca_test_mhr.npz",
    ]

    for fname in files:
        npz_path = os.path.join(args.dir, fname)
        if not os.path.exists(npz_path):
            print(f"Skipping (not found): {npz_path}")
            continue
        split_file(npz_path, args.dir)

    print("\nDone. You can now update config.yaml to use the new split files.")


if __name__ == "__main__":
    main()
