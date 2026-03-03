#!/usr/bin/env python3
"""
Convert Damon dataset from SMPL format to MHR format.
This script converts contact labels from SMPL topology (6890 vertices) 
to MHR topology (18439 vertices) using barycentric interpolation.
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import better_human
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from better_human.smpl import SMPL


def convert_contact_labels(
    source_contact_labels: torch.Tensor,
    smpl_2_mhr_path: str,
    smpl_model: SMPL,
    direction: str = "smpl2mhr",
    device: torch.device = "cuda",
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert per-vertex binary contact labels from one mesh topology to another
    using barycentric interpolation.

    For each target vertex, the contact label is computed by looking up the 3 source
    vertices of the mapped triangle, interpolating their binary labels using the
    barycentric coordinates, and thresholding the result.

    Args:
        source_contact_labels: Binary contact labels tensor of shape [V] or [B, V]
            where B is batch size and V is number of source vertices.
            Values should be 0 or 1.
        smpl_2_mhr_path: Path to the .npz file containing the surface mapping
            (triangle_ids and baryc_coords).
        smpl_model: SMPL model instance (used to get face connectivity).
        direction: Direction of conversion. Must be "smpl2mhr".
        device: Torch device to use for computation.
        threshold: Threshold for binarizing the interpolated labels.
            A value of 0.5 means majority-weighted vote. Lower values (e.g. 0.0)
            are more conservative (any contact in the triangle maps to contact).
            Default: 0.5.

    Returns:
        torch.Tensor: Binary contact labels for target vertices. Shape [V_target] or [B, V_target].
    """
    if direction == "smpl2mhr":
        mapping = np.load(smpl_2_mhr_path)
        mapped_face_id, baryc_coords = mapping["triangle_ids"], mapping["baryc_coords"]
        source_faces = smpl_model.faces
    else:
        raise ValueError(
            f"Direction '{direction}' not supported. Only 'smpl2mhr' is implemented."
        )

    # Move data to device
    mapped_face_id_tensor = torch.from_numpy(mapped_face_id).long().to(device)
    baryc_coords_tensor = torch.from_numpy(baryc_coords).float().to(device)  # [N_target, 3]
    source_faces_tensor = source_faces.long().to(device)

    source_contact_labels = source_contact_labels.float().to(device)  # [B, V_source]
    
    # Handle both single frame and batch
    if source_contact_labels.dim() == 1:
        source_contact_labels = source_contact_labels.unsqueeze(0)  # [1, V_source]
        squeeze_output = True
    else:
        squeeze_output = False

    # Get the triangle vertex indices for each target vertex
    tri_vertex_ids = source_faces_tensor[mapped_face_id_tensor]  # [N_target, 3]

    # Look up contact labels for the 3 vertices of each triangle
    # source_contact_labels: [B, V_source]
    # tri_vertex_ids: [N_target, 3]
    # We need to gather labels for each of the 3 vertices
    tri_labels = source_contact_labels[:, tri_vertex_ids]  # [B, N_target, 3]

    # Interpolate using barycentric coordinates
    # baryc_coords_tensor: [N_target, 3], tri_labels: [B, N_target, 3]
    interpolated = (tri_labels * baryc_coords_tensor.unsqueeze(0)).sum(dim=-1)  # [B, N_target]

    # Threshold to get binary labels
    target_labels = (interpolated >= threshold).long()

    if squeeze_output:
        target_labels = target_labels.squeeze(0)

    return target_labels


def convert_objectwise_labels(
    objectwise_dict: dict,
    smpl_2_mhr_path: str,
    smpl_model: SMPL,
    device: torch.device = "cuda",
    threshold: float = 0.5,
) -> dict:
    """
    Convert objectwise contact labels from SMPL to MHR topology.
    
    Args:
        objectwise_dict: Dictionary mapping object names to arrays of SMPL vertex indices.
        smpl_2_mhr_path: Path to the .npz file containing the surface mapping.
        smpl_model: SMPL model instance.
        device: Torch device to use for computation.
        threshold: Threshold for binarizing the interpolated labels.
    
    Returns:
        dict: Dictionary mapping object names to arrays of MHR vertex indices.
    """
    mhr_objectwise = {}
    
    for obj_name, smpl_vertex_indices in objectwise_dict.items():
        # Create binary contact label for this object on SMPL mesh
        smpl_contact_binary = np.zeros(6890, dtype=np.float32)
        smpl_contact_binary[smpl_vertex_indices] = 1.0
        
        # Convert to MHR using barycentric interpolation
        smpl_contact_tensor = torch.from_numpy(smpl_contact_binary).to(device)
        mhr_contact_binary = convert_contact_labels(
            smpl_contact_tensor,
            smpl_2_mhr_path,
            smpl_model,
            direction="smpl2mhr",
            device=device,
            threshold=threshold,
        )
        
        # Extract MHR vertex indices where contact is 1
        mhr_vertex_indices = torch.where(mhr_contact_binary > 0)[0].cpu().numpy()
        mhr_objectwise[obj_name] = mhr_vertex_indices
    
    return mhr_objectwise


def convert_dataset(
    input_path: str,
    output_path: str,
    smpl_model_path: str,
    smpl_2_mhr_path: str,
    device: str = "cuda",
):
    """
    Convert Damon dataset from SMPL to MHR format.
    
    Args:
        input_path: Path to input .npz file.
        output_path: Path to output .npz file.
        smpl_model_path: Path to SMPL model file.
        smpl_2_mhr_path: Path to SMPL to MHR mapping file.
        device: Device to use for computation.
    """
    print(f"Loading dataset from: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    
    print(f"Loading SMPL model from: {smpl_model_path}")
    smpl_model = SMPL(smpl_model_path).to(device)
    
    # Get the data
    imgnames = data["imgname"]
    contact_labels_smpl = data["contact_label"]
    contact_labels_objectwise_smpl = data["contact_label_objectwise"]
    
    num_samples = len(imgnames)
    print(f"Processing {num_samples} samples...")
    
    # Prepare output arrays
    contact_labels_mhr = []
    contact_labels_objectwise_mhr = []
    
    # Convert each sample
    for i in tqdm(range(num_samples), desc="Converting samples"):
        # Convert overall contact label
        smpl_contact = torch.from_numpy(contact_labels_smpl[i]).float().to(device)
        mhr_contact = convert_contact_labels(
            smpl_contact,
            smpl_2_mhr_path,
            smpl_model,
            direction="smpl2mhr",
            device=device,
            threshold=0.5,
        )
        contact_labels_mhr.append(mhr_contact.cpu().numpy())
        
        # Convert objectwise contact labels
        objectwise_smpl = contact_labels_objectwise_smpl[i]
        objectwise_mhr = convert_objectwise_labels(
            objectwise_smpl,
            smpl_2_mhr_path,
            smpl_model,
            device=device,
            threshold=0.5,
        )
        contact_labels_objectwise_mhr.append(objectwise_mhr)
    
    # Convert to numpy arrays
    # Stack contact labels as proper numpy array (all have same shape)
    contact_labels_mhr_stacked = np.stack(contact_labels_mhr, axis=0)
    contact_labels_objectwise_mhr = np.array(contact_labels_objectwise_mhr, dtype=object)
    
    # Save the converted data
    print(f"Saving converted dataset to: {output_path}")
    np.savez(
        output_path,
        imgname=imgnames,
        contact_label=contact_labels_mhr_stacked,
        contact_label_objectwise=contact_labels_objectwise_mhr,
    )
    
    print(f"✓ Successfully converted {num_samples} samples")
    print(f"  - SMPL vertices: 6890 -> MHR vertices: 18439")
    print(f"  - Output keys: imgname, contact_label, contact_label_objectwise")


def main():
    # Configuration
    dataset_base_path = "/media/rikhat/Hard/datasets/DECO/datasets/Release_Datasets/damon"
    smpl_model_path = "../better_human/models/smpl/SMPL_NEUTRAL.npz"
    smpl_2_mhr_path = "../MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz"
    output_dir = "./daimon_mhr_contact"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert trainval dataset
    print("=" * 80)
    print("Converting TRAINVAL dataset")
    print("=" * 80)
    convert_dataset(
        input_path=os.path.join(dataset_base_path, "hot_dca_trainval.npz"),
        output_path=os.path.join(output_dir, "hot_dca_trainval_mhr.npz"),
        smpl_model_path=smpl_model_path,
        smpl_2_mhr_path=smpl_2_mhr_path,
        device=device,
    )
    print()
    
    # Convert test dataset
    print("=" * 80)
    print("Converting TEST dataset")
    print("=" * 80)
    convert_dataset(
        input_path=os.path.join(dataset_base_path, "hot_dca_test.npz"),
        output_path=os.path.join(output_dir, "hot_dca_test_mhr.npz"),
        smpl_model_path=smpl_model_path,
        smpl_2_mhr_path=smpl_2_mhr_path,
        device=device,
    )
    print()
    
    print("=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print("  - hot_dca_trainval_mhr.npz")
    print("  - hot_dca_test_mhr.npz")


if __name__ == "__main__":
    main()
