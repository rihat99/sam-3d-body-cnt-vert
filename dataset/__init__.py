"""
Datasets for SAM-3D-Body training.

Available datasets:
- DamonMHRDataset: Damon dataset with MHR contact labels (18,439 vertices),
                   bounding boxes, and camera parameters
"""

from .damon_mhr import DamonMHRDataset

__all__ = [
    'DamonMHRDataset',
]
