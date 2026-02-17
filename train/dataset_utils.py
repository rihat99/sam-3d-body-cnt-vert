"""
Dataset utilities for preparing batches for SAM-3D-Body training.

Uses the same transform pipeline as the reference inference code
(GetBBoxCenterScale -> TopdownAffine -> ToTensor) to ensure batch format
is identical to what the model expects.
"""

import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision.transforms import ToTensor

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from sam_3d_body.data.utils.prepare_batch import NoCollate


# Build the same transform pipeline used by SAM3DBodyEstimator
# Body padding=1.25 (default), no UDP, input_size matches model config
_body_transform = Compose(
    [
        GetBBoxCenterScale(),  # padding=1.25 by default
        TopdownAffine(input_size=(896, 896), use_udp=False),
        VisionTransformWrapper(ToTensor()),
    ]
)


def prepare_training_batch(images, bboxes, cam_params, target_size=(896, 896), device='cuda'):
    """
    Prepare a batch for SAM-3D-Body model, using the same transform pipeline
    as the reference inference code.

    This ensures that bbox_center, bbox_scale, affine_trans, and the cropped
    images are computed identically to how the model was trained/evaluated.

    Args:
        images: list of numpy arrays (H, W, 3) in RGB, uint8
        bboxes: list of numpy arrays (4,) - [x1, y1, x2, y2]
        cam_params: dict with fx, fy, cx, cy
        target_size: tuple (H, W) - target image size (must match model config)
        device: torch device

    Returns:
        batch: dict with all required keys for SAM-3D-Body
    """
    # Build transform if target_size differs from default (896, 896)
    if target_size != (896, 896):
        transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=target_size, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
    else:
        transform = _body_transform

    # Keep the first original image for img_ori (needed for hand processing)
    orig_img = images[0] if isinstance(images[0], np.ndarray) else images[0].numpy()

    # Process each person using the same pipeline as prepare_batch
    data_list = []
    for img, bbox in zip(images, bboxes):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.numpy()

        height, width = img.shape[:2]

        data_info = dict(img=img)
        data_info["bbox"] = bbox.astype(np.float32)
        data_info["bbox_format"] = "xyxy"

        # No mask for training (same as reference when masks=None)
        data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
        data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        data_list.append(transform(data_info))

    # Collate into batch
    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]

    # Add batch dimension (1, N, ...) and convert to float, matching reference
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    # Camera intrinsics - use ORIGINAL (unadjusted) intrinsics.
    # The model's get_ray_condition() inverts affine_trans to map crop pixels
    # back to full-image space before applying cam_int, so cam_int must be
    # the raw camera parameters in full-image space.
    cam_int = torch.tensor(
        [
            [
                [cam_params['fx'], 0, cam_params['cx']],
                [0, cam_params['fy'], cam_params['cy']],
                [0, 0, 1],
            ]
        ],
        dtype=torch.float32,
    )
    batch["cam_int"] = cam_int

    # Store original image (needed for hand crop processing)
    batch["img_ori"] = [NoCollate(orig_img)]

    # Move everything to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    return batch
