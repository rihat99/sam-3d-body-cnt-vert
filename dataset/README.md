# Damon MHR Dataset - PyTorch Dataset

PyTorch Dataset for Damon with MHR contact labels, bounding boxes, and camera parameters.

## Overview

The `DamonMHRDataset` class provides a unified interface for loading:
- RGB images
- Per-vertex binary contact labels (18,439 MHR vertices)
- Human bounding boxes (predicted or full-image fallback)
- Camera intrinsics (predicted or default fallback)

## Quick Start

```python
from damon_mhr import DamonMHRDataset

# Create dataset
dataset = DamonMHRDataset(
    npz_path='../../datasets/daimon_mhr_contact/hot_dca_trainval_mhr_with_bbox.npz',
    data_root='/media/rikhat/Hard/datasets/DECO/'
)

# Get a sample - always returns (image, bbox, cam_k), contact_label
(image, bbox, cam_k), contact_label = dataset[0]
# image: PIL.Image or torch.Tensor (if transform applied)
# bbox: torch.Tensor (4,) - [x1, y1, x2, y2]
# cam_k: torch.Tensor (3, 3) - camera intrinsics matrix
# contact_label: torch.Tensor (18439,) - binary contact labels
```

## With Transform

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = DamonMHRDataset(
    npz_path='../../datasets/daimon_mhr_contact/hot_dca_trainval_mhr_with_bbox.npz',
    data_root='/media/rikhat/Hard/datasets/DECO/',
    transform=transform
)
```

## With DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for (images, bboxes, cam_ks), contact_labels in dataloader:
    # images: (B, 3, H, W)
    # bboxes: (B, 4) - [x1, y1, x2, y2]
    # cam_ks: (B, 3, 3) - camera intrinsics
    # contact_labels: (B, 18439) - binary contact per vertex
    pass
```

## Automatic Detection

The dataset automatically detects whether bboxes and camera parameters are available:

**With predicted values (after running `damon_append.py`):**
```
Loading Damon MHR dataset from hot_dca_trainval_mhr_with_bbox.npz
  ✓ Bounding boxes available (predicted)
  ✓ Camera parameters available (predicted)
```

**Without predicted values (original dataset):**
```
Loading Damon MHR dataset from hot_dca_trainval_mhr.npz
  ⚠ No bounding boxes - will use full image bbox
  ⚠ No camera parameters - will use default focal length
```

## Adding BBox and Camera Parameters

To add predicted bounding boxes and camera parameters to your dataset:

```bash
cd ../../datasets
python damon_append.py \
    --data_root /media/rikhat/Hard/datasets/DECO/ \
    --dataset_dir ./daimon_mhr_contact
```

This uses ViTDet for human detection and MoGe2 for camera estimation.
See `../../datasets/README_DAMON_APPEND.md` for details.

## Data Root Configuration

Three ways to specify the data root:

**1. Pass explicitly (recommended):**
```python
dataset = DamonMHRDataset(
    npz_path='path/to/data.npz',
    data_root='/media/rikhat/Hard/datasets/DECO/'
)
```

**2. Environment variable:**
```bash
export DAMON_DATA_ROOT=/media/rikhat/Hard/datasets/DECO/
```

**3. Auto-detection (may not work if images on external drive):**
```python
dataset = DamonMHRDataset(npz_path='path/to/data.npz')
# Uses parent directory of npz_path
```

## Sample Information

```python
info = dataset.get_sample_info(0)
print(info)
# {
#     'idx': 0,
#     'imgname': 'datasets/HOT-Annotated/images/vcoco_000000000589.jpg',
#     'num_contacts': 2702,
#     'contact_ratio': 0.1465,
#     'bbox': [347.5, 187.9, 493.6, 366.5],
#     'bbox_size': [146.1, 178.6],
#     'focal_length': [504.0, 504.0],
#     'principal_point': [320.0, 240.0]
# }
```

## Dataset Statistics

- **Training set**: 4,384 samples (~2,629 contact vertices per sample)
- **Test set**: 785 samples (~2,674 contact vertices per sample)
- **Vertices per mesh**: 18,439 (MHR topology)
- **Contact ratio**: ~14-15%

## Files

- `damon_mhr.py` - Main dataset implementation
- `__init__.py` - Package initialization
- `example_damon_mhr.py` - Usage example
- `README.md` - This file

## See Also

- `../../datasets/README_DAMON_APPEND.md` - How to add bboxes and camera parameters
- `../../datasets/DAMON_SUMMARY.md` - Complete overview
- `example_damon_mhr.py` - Working example
