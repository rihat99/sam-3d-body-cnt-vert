"""
PyTorch Dataset for DAMON with SMPL (6890-vertex) contact labels.

Ground-truth contacts come from the original DECO/DAMON NPZ files which
contain per-vertex binary labels in SMPL topology.  Bounding boxes and
camera intrinsics are loaded from the detect NPZ (imgname + bbox + cam_k)
produced by damon_append.py.

Interface is identical to DamonMHRDataset so the two can be swapped in
evaluate.py / inference_demo.py without changing training or collation code.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class DamonSMPLDataset(Dataset):
    """
    Dataset for DAMON images with SMPL contact labels (6 890 vertices).

    Args:
        smpl_npz_path: Path to the original DECO NPZ.
            Expected keys:
            - 'imgname':       [N] image filenames
            - 'contact_label': [N, 6890] float contact labels (binarised at 0.5)
            - 'cam_k':         [N, 3, 3] camera intrinsics (optional)
        detect_npz_path: Path to the detect NPZ (imgname + bbox + cam_k).
            Produced by damon_append.py.  If None, bbox falls back to
            full-image dimensions and cam_k to default focal length.
        data_root: Root directory for images.  If None, tries:
            1. DAMON_DATA_ROOT environment variable
            2. /data3/rikhat.akizhanov/DECO  (hard-coded fallback)

    Returns per item:
        (image, bbox, cam_k), contact_label
            image         — numpy uint8 RGB [H, W, 3]
            bbox          — float32 tensor [4]  (x1, y1, x2, y2)
            cam_k         — float32 tensor [3, 3]
            contact_label — int64  tensor [6890] binary {0, 1}
    """

    NUM_VERTICES = 6890

    def __init__(
        self,
        smpl_npz_path: str,
        detect_npz_path: Optional[str] = None,
        data_root: Optional[str] = None,
    ):
        super().__init__()

        self.smpl_npz_path = smpl_npz_path

        print(f"Loading Damon SMPL dataset from {smpl_npz_path}")
        smpl_data = np.load(smpl_npz_path, allow_pickle=True)

        self.imgnames      = smpl_data["imgname"]
        raw_labels         = smpl_data["contact_label"]          # [N, 6890] float
        self.contact_labels = (raw_labels > 0.5).astype(np.int64)

        num_samples = len(self.imgnames)
        assert self.contact_labels.shape == (num_samples, self.NUM_VERTICES), (
            f"Expected contact_labels shape ({num_samples}, {self.NUM_VERTICES}), "
            f"got {self.contact_labels.shape}"
        )

        # Camera intrinsics and bboxes from detect NPZ
        if detect_npz_path is not None:
            print(f"  Loading bbox and cam_k from {detect_npz_path}")
            detect_data = np.load(detect_npz_path, allow_pickle=True)
            self.bboxes  = detect_data["bbox"].astype(np.float32)   # [N, 4]
            self.cam_ks  = detect_data["cam_k"].astype(np.float32)  # [N, 3, 3]
            assert len(self.bboxes) == num_samples, (
                f"detect NPZ has {len(self.bboxes)} samples but SMPL NPZ has {num_samples}"
            )
            self.has_bboxes = True
            self.has_cam_ks = True
            print("  Bounding boxes and camera intrinsics available")
        else:
            self.bboxes = None
            self.cam_ks = None
            self.has_bboxes = False
            self.has_cam_ks = False
            print("  No detect NPZ — bbox will be full-image, cam_k will use default focal length")

        # Data root for image loading
        if data_root is None:
            data_root = os.environ.get("DAMON_DATA_ROOT", None)
            if data_root is None:
                data_root = "/data3/rikhat.akizhanov/DECO"
                print(
                    f"  Warning: data_root not specified. Using fallback: {data_root}\n"
                    f"  Set DAMON_DATA_ROOT env var or pass data_root explicitly."
                )
        self.data_root = data_root

        total_contacts = int(self.contact_labels.sum())
        print(
            f"Loaded {num_samples} SMPL samples | "
            f"total contact vertices: {total_contacts} | "
            f"avg per sample: {total_contacts / num_samples:.1f}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.imgnames)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[np.ndarray, torch.Tensor, torch.Tensor], torch.Tensor]:
        imgname  = self.imgnames[idx]
        img_path = os.path.join(self.data_root, imgname)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load image {img_path}\n"
                f"Image name: {imgname}\n"
                f"Data root: {self.data_root}\n"
                f"Error: {e}\n"
                f"Hint: Set data_root parameter or DAMON_DATA_ROOT env var."
            )

        width, height = image.size
        image = np.array(image, dtype=np.uint8)  # [H, W, 3]

        # Bounding box
        if self.has_bboxes:
            bbox = torch.from_numpy(self.bboxes[idx]).float()
        else:
            bbox = torch.tensor([0.0, 0.0, float(width), float(height)])

        # Camera intrinsics
        if self.has_cam_ks:
            cam_k = torch.from_numpy(self.cam_ks[idx]).float()
        else:
            focal = float(max(width, height))
            cam_k = torch.tensor([
                [focal, 0.0,   width  / 2.0],
                [0.0,   focal, height / 2.0],
                [0.0,   0.0,   1.0],
            ])

        contact_label = torch.from_numpy(self.contact_labels[idx]).long()

        return (image, bbox, cam_k), contact_label

    # ------------------------------------------------------------------
    # Train / val splitting — identical logic to DamonMHRDataset
    # ------------------------------------------------------------------

    @classmethod
    def split_train_val(
        cls,
        smpl_npz_path: str,
        detect_npz_path: Optional[str] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        data_root: Optional[str] = None,
    ):
        """
        Load the full dataset and split into train / val subsets.

        Uses the same 80/20 seed-42 split convention as DamonMHRDataset so
        metrics on both topologies are computed on the exact same images.

        Returns:
            (train_subset, val_subset) — torch.utils.data.Subset objects.
        """
        from torch.utils.data import Subset

        full_dataset = cls(
            smpl_npz_path=smpl_npz_path,
            detect_npz_path=detect_npz_path,
            data_root=data_root,
        )
        n = len(full_dataset)

        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(n)

        n_val         = max(1, int(round(n * val_ratio)))
        val_indices   = sorted(shuffled[:n_val].tolist())
        train_indices = sorted(shuffled[n_val:].tolist())

        print(
            f"Split (seed={seed}): {len(train_indices)} train, "
            f"{len(val_indices)} val (val_ratio={val_ratio})"
        )
        return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)

    # ------------------------------------------------------------------

    def get_sample_info(self, idx: int) -> dict:
        contact_label = self.contact_labels[idx]
        num_contacts  = int(contact_label.sum())
        info = {
            "idx":           idx,
            "imgname":       self.imgnames[idx],
            "num_contacts":  num_contacts,
            "contact_ratio": num_contacts / self.NUM_VERTICES,
        }
        if self.has_bboxes:
            bbox = self.bboxes[idx]
            info["bbox"]      = bbox.tolist()
            info["bbox_size"] = [float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
        if self.has_cam_ks:
            cam_k = self.cam_ks[idx]
            info["focal_length"]    = [float(cam_k[0, 0]), float(cam_k[1, 1])]
            info["principal_point"] = [float(cam_k[0, 2]), float(cam_k[1, 2])]
        return info
