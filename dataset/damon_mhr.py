"""
PyTorch Dataset for Damon with MHR contact labels.

This dataset loads images and binary per-vertex contact labels (18439 vertices for MHR topology),
along with bounding boxes and camera parameters.

Train / val / test splits are handled by passing separate npz files.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Tuple


class DamonMHRDataset(Dataset):
    """
    Dataset for Damon images with MHR contact labels, bounding boxes, and camera parameters.

    Args:
        npz_path: Path to the .npz file containing the dataset.
            Expected keys:
            - 'imgname':       Array of image filenames (e.g., 'datasets/HOT-Annotated/images/...')
            - 'contact_label': Binary contact labels  (N, 18439)
            - 'bbox':          Bounding boxes  (N, 4) in [x1, y1, x2, y2] — optional
            - 'cam_k':         Camera intrinsics (N, 3, 3) — optional
        data_root: Root directory for images. If None, tries:
            1. DAMON_DATA_ROOT environment variable
            2. Parent directory of npz_path
        transform: Optional transform applied to the PIL Image before returning.

    Returns per item:
        (image, bbox, cam_k):
            image  — numpy uint8 RGB array [H, W, 3]  (raw, SAM-3D-Body transforms applied later)
            bbox   — float32 tensor [4]  (x1, y1, x2, y2)
            cam_k  — float32 tensor [3, 3]
        contact_label — int64 tensor [18439] (binary)
    """

    NUM_VERTICES = 18439

    def __init__(
        self,
        npz_path: str,
        data_root: Optional[str] = None,
        transform=None,
    ):
        super().__init__()

        self.npz_path = npz_path
        self.transform = transform

        print(f"Loading Damon MHR dataset from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        self.imgnames = data['imgname']
        self.contact_labels = data['contact_label']  # (N, 18439)

        self.bboxes = data.get('bbox', None)     # (N, 4)  optional
        self.cam_ks = data.get('cam_k', None)    # (N, 3, 3)  optional

        num_samples = len(self.imgnames)
        assert self.contact_labels.shape == (num_samples, self.NUM_VERTICES), (
            f"Expected contact_labels shape ({num_samples}, {self.NUM_VERTICES}), "
            f"got {self.contact_labels.shape}"
        )

        if self.bboxes is not None:
            assert self.bboxes.shape == (num_samples, 4), (
                f"Expected bbox shape ({num_samples}, 4), got {self.bboxes.shape}"
            )
            print(f"  Bounding boxes available")
            self.has_bboxes = True
        else:
            print(f"  No bounding boxes — will use full image bbox")
            self.has_bboxes = False

        if self.cam_ks is not None:
            assert self.cam_ks.shape == (num_samples, 3, 3), (
                f"Expected cam_k shape ({num_samples}, 3, 3), got {self.cam_ks.shape}"
            )
            print(f"  Camera intrinsics available")
            self.has_cam_ks = True
        else:
            print(f"  No camera intrinsics — will use default focal length")
            self.has_cam_ks = False

        # Resolve data root
        if data_root is None:
            data_root = os.environ.get('DAMON_DATA_ROOT', None)
            if data_root is None:
                data_root = os.path.dirname(os.path.dirname(npz_path))
                print(
                    f"Warning: data_root not specified. Using default: {data_root}\n"
                    f"Set DAMON_DATA_ROOT env var or pass data_root explicitly if images are missing."
                )

        self.data_root = data_root

        total_contacts = int(self.contact_labels.sum())
        print(
            f"Loaded {num_samples} samples | "
            f"total contact vertices: {total_contacts} | "
            f"avg per sample: {total_contacts / num_samples:.1f}"
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.imgnames)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[np.ndarray, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns:
            (image, bbox, cam_k), contact_label
        """
        imgname = self.imgnames[idx]
        img_path = os.path.join(self.data_root, imgname)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load image {img_path}\n"
                f"Image name: {imgname}\n"
                f"Data root: {self.data_root}\n"
                f"Error: {e}\n"
                f"Hint: Set data_root parameter or DAMON_DATA_ROOT environment variable."
            )

        width, height = image.size

        if self.transform is not None:
            image = self.transform(image)

        # Return as numpy array (SAM-3D-Body transforms applied in data_utils.py)
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.uint8)  # [H, W, 3] RGB uint8

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
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ])

        contact_label = torch.from_numpy(self.contact_labels[idx]).long()

        return (image, bbox, cam_k), contact_label

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Train / val splitting
    # ------------------------------------------------------------------

    @classmethod
    def split_train_val(
        cls,
        npz_path: str,
        val_ratio: float = 0.2,
        seed: int = 42,
        data_root: Optional[str] = None,
    ):
        """
        Load the full dataset from *npz_path* and split it into train and val
        subsets using a fixed random seed.

        Args:
            npz_path:  Path to the combined train+val npz file.
            val_ratio: Fraction of samples to use for validation (default 0.2).
            seed:      Random seed for reproducible shuffling (default 42).
            data_root: Root directory for images.

        Returns:
            (train_subset, val_subset) — torch.utils.data.Subset objects that
            wrap the same underlying DamonMHRDataset instance.
        """
        from torch.utils.data import Subset

        full_dataset = cls(npz_path=npz_path, data_root=data_root)
        n = len(full_dataset)

        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(n)

        n_val = max(1, int(round(n * val_ratio)))
        val_indices   = sorted(shuffled[:n_val].tolist())
        train_indices = sorted(shuffled[n_val:].tolist())

        print(
            f"Split (seed={seed}): {len(train_indices)} train, "
            f"{len(val_indices)} val (val_ratio={val_ratio})"
        )
        return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)

    # ------------------------------------------------------------------

    def get_sample_info(self, idx: int) -> dict:
        """Return metadata dictionary for a sample (debugging / statistics)."""
        contact_label = self.contact_labels[idx]
        num_contacts = int(contact_label.sum())
        info = {
            'idx': idx,
            'imgname': self.imgnames[idx],
            'num_contacts': num_contacts,
            'contact_ratio': num_contacts / self.NUM_VERTICES,
        }
        if self.has_bboxes:
            bbox = self.bboxes[idx]
            info['bbox'] = bbox.tolist()
            info['bbox_size'] = [float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
        if self.has_cam_ks:
            cam_k = self.cam_ks[idx]
            info['focal_length'] = [float(cam_k[0, 0]), float(cam_k[1, 1])]
            info['principal_point'] = [float(cam_k[0, 2]), float(cam_k[1, 2])]
        return info


# ---------------------------------------------------------------------------
# Convenience helpers for building train / val / test datasets from config
# ---------------------------------------------------------------------------

def build_datasets(cfg):
    """
    Build train, val, and test dataset objects from a config node.

    Train and val come from the same combined npz file (TRAINVAL_NPZ); they
    are split by index using a fixed random seed so the split is perfectly
    reproducible across training and evaluation runs.

    Expected config keys under cfg.DATASET:
        TRAINVAL_NPZ  — path to the combined train+val npz file
        TEST_NPZ      — path to the held-out test npz file (optional)
        DATA_ROOT     — root directory for images (or use DAMON_DATA_ROOT env var)
        VAL_RATIO     — fraction of samples used for val (default 0.2)
        SEED          — random seed for reproducible split (default 42)
    """
    data_root = cfg.DATASET.get('DATA_ROOT', None)
    val_ratio = cfg.DATASET.get('VAL_RATIO', 0.2)
    seed      = cfg.DATASET.get('SEED', 42)

    train_ds, val_ds = DamonMHRDataset.split_train_val(
        npz_path=cfg.DATASET.TRAINVAL_NPZ,
        val_ratio=val_ratio,
        seed=seed,
        data_root=data_root,
    )

    test_ds = None
    if cfg.DATASET.get('TEST_NPZ', None):
        test_ds = DamonMHRDataset(
            npz_path=cfg.DATASET.TEST_NPZ,
            data_root=data_root,
        )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Damon MHR Dataset")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the .npz file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for images")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    print("=" * 80)
    print("Testing Damon MHR Dataset")
    print("=" * 80)

    dataset = DamonMHRDataset(npz_path=args.npz_path, data_root=args.data_root)
    print(f"\nDataset length: {len(dataset)}")

    for i in range(min(args.num_samples, len(dataset))):
        print(f"\nSample {i}:")
        info = dataset.get_sample_info(i)
        print(f"  Image: {info['imgname']}")
        print(f"  Contacts: {info['num_contacts']} ({info['contact_ratio']*100:.2f}%)")
        if 'bbox' in info:
            print(f"  BBox: {info['bbox']}  size: {info['bbox_size']}")
        if 'focal_length' in info:
            print(f"  fx={info['focal_length'][0]:.1f}  fy={info['focal_length'][1]:.1f}  "
                  f"cx={info['principal_point'][0]:.1f}  cy={info['principal_point'][1]:.1f}")

        try:
            (image, bbox, cam_k), contact_label = dataset[i]
            print(f"  Image shape: {image.shape}")
            print(f"  BBox: {bbox}")
            print(f"  cam_k shape: {cam_k.shape}")
            print(f"  contact_label shape: {contact_label.shape}, sum={contact_label.sum()}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
