#!/usr/bin/env python3
"""
Convert DAMON/DECO dataset contact labels from SMPL format to MHR format.

Converts per-vertex contact labels from SMPL topology (6890 vertices) to MHR
topology using BodyConverter (barycentric interpolation). Supports any target
MHR LOD via --target_lod (default: 1).

Conversion chain:
  SMPL (6890) → MHR LOD1 (18439) [→ MHR LOD_N if target_lod != 1]

Usage:
  python mhr_smpl_conversion/convert_damon.py \\
    --input_path /path/to/hot_dca_trainval.npz \\
    --output_path /path/to/hot_dca_trainval_mhr.npz \\
    --smpl_model_path /path/to/SMPL_NEUTRAL.npz \\
    --target_lod 1 \\
    --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Allow running from repo root or from the mhr_smpl_conversion dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from better_human.smpl import SMPL
from mhr_smpl_conversion.body_converter import BodyConverter


def convert_dataset(
    input_path: str,
    output_path: str,
    smpl_model_path: str,
    target_lod: int = 1,
    device: str = "cuda",
    threshold: float = 0.5,
) -> None:
    """
    Convert DAMON dataset from SMPL to MHR format.

    Args:
        input_path:       Path to input .npz file (SMPL contacts).
        output_path:      Path to output .npz file (MHR contacts).
        smpl_model_path:  Path to SMPL_NEUTRAL.npz model file.
        target_lod:       Target MHR LOD level (0–6, default 1).
        device:           Torch device string ('cuda' or 'cpu').
        threshold:        Binarisation threshold for contact interpolation.
    """
    print(f"Loading dataset from: {input_path}")
    data = np.load(input_path, allow_pickle=True)

    print(f"Loading SMPL model from: {smpl_model_path}")
    smpl_model = SMPL(smpl_model_path).to(device)

    # BodyConverter needs SMPL face connectivity
    smpl_faces = smpl_model.faces  # Tensor or np.ndarray [13776, 3]
    if isinstance(smpl_faces, torch.Tensor):
        smpl_faces_np = smpl_faces.cpu().numpy()
    else:
        smpl_faces_np = np.asarray(smpl_faces)

    converter = BodyConverter(
        smpl_faces=smpl_faces_np,
        device=device,
        threshold=threshold,
    )

    imgnames = data["imgname"]
    contact_labels_smpl = data["contact_label"]
    contact_labels_objectwise_smpl = data["contact_label_objectwise"]

    num_samples = len(imgnames)
    n_target_verts = BodyConverter.LOD_VERTEX_COUNTS[target_lod]
    print(f"Processing {num_samples} samples  |  target: MHR LOD{target_lod} ({n_target_verts} verts)")

    contact_labels_mhr = []
    contact_labels_objectwise_mhr = []

    for i in tqdm(range(num_samples), desc="Converting"):
        # --- overall contact label -------------------------------------------
        smpl_contact = torch.from_numpy(
            contact_labels_smpl[i].astype(np.float32)
        ).to(device)

        result = converter.smpl_to_mhr(
            contacts=smpl_contact,
            target_lod=target_lod,
        )
        contact_labels_mhr.append(result.contacts.cpu().numpy())

        # --- objectwise contact labels ----------------------------------------
        objectwise_smpl: dict = contact_labels_objectwise_smpl[i]
        objectwise_mhr: dict = {}

        for obj_name, smpl_vertex_indices in objectwise_smpl.items():
            # Sparse indices → binary [6890]
            binary = np.zeros(6890, dtype=np.float32)
            binary[smpl_vertex_indices] = 1.0

            res = converter.smpl_to_mhr(
                contacts=torch.from_numpy(binary).to(device),
                target_lod=target_lod,
            )
            mhr_contacts = res.contacts  # long [V_tgt]
            mhr_indices = torch.where(mhr_contacts > 0)[0].cpu().numpy()
            objectwise_mhr[obj_name] = mhr_indices

        contact_labels_objectwise_mhr.append(objectwise_mhr)

    contact_labels_mhr_stacked = np.stack(contact_labels_mhr, axis=0)
    contact_labels_objectwise_arr = np.array(contact_labels_objectwise_mhr, dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving converted dataset to: {output_path}")
    np.savez(
        output_path,
        imgname=imgnames,
        contact_label=contact_labels_mhr_stacked,
        contact_label_objectwise=contact_labels_objectwise_arr,
    )

    print(f"Done. {num_samples} samples converted.")
    print(f"  SMPL vertices: 6890  →  MHR LOD{target_lod} vertices: {n_target_verts}")
    print(f"  Output keys: imgname, contact_label, contact_label_objectwise")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DAMON contact labels from SMPL to MHR topology."
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Path to input .npz file (SMPL-based contact labels).",
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Path to output .npz file.",
    )
    parser.add_argument(
        "--smpl_model_path", required=True,
        help="Path to SMPL_NEUTRAL.npz model file.",
    )
    parser.add_argument(
        "--target_lod", type=int, default=1, choices=list(range(7)),
        help="Target MHR LOD level (0–6). Default: 1 (18 439 verts).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default: cuda if available).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Binarisation threshold for barycentric contact interpolation. Default: 0.5.",
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    convert_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        smpl_model_path=args.smpl_model_path,
        target_lod=args.target_lod,
        device=args.device,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
