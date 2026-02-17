"""
Video processing script for SAM 3D Body.

Features:
- Camera intrinsics estimated once on first frame and reused
- External bbox detection with largest-person selection
- Modular helper functions for clean code organization
"""

import os
import argparse
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

os.environ["MOMENTUM_ENABLED"] = "1"

from notebook.utils import setup_sam_3d_body


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "facebook/sam-3d-body-dinov3"

# Detection parameters
DETECTION_CATEGORY_ID = 0  # Human category
DETECTION_BBOX_THRESHOLD = 0.5
DETECTION_NMS_THRESHOLD = 0.3


# =============================================================================
# Helper Functions
# =============================================================================

def load_frames(video_path: str) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Yield frames from a video file or image directory.
    
    Args:
        video_path: Path to video file or directory of images
        
    Yields:
        Tuple of (frame as RGB numpy array, frame index)
    """
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_rgb, frame_idx
            
        cap.release()
        
    elif os.path.isdir(video_path):
        valid_extensions = ('.png', '.jpg', '.jpeg')
        frame_files = sorted([
            f for f in os.listdir(video_path) 
            if f.lower().endswith(valid_extensions)
        ])
        
        for frame_idx, frame_name in enumerate(frame_files):
            frame_path = os.path.join(video_path, frame_name)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb, frame_idx
    else:
        raise ValueError(f"Invalid video path: {video_path}")


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1: First bbox as [x1, y1, x2, y2]
        box2: Second bbox as [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


# Threshold for temporal bbox consistency (IoU below this triggers fallback)
BBOX_IOU_THRESHOLD = 0.7


def detect_largest_bbox(
    detector, 
    frame: np.ndarray,
    prev_bbox: Optional[np.ndarray] = None,
    det_cat_id: int = DETECTION_CATEGORY_ID,
    bbox_thr: float = DETECTION_BBOX_THRESHOLD,
    nms_thr: float = DETECTION_NMS_THRESHOLD,
    iou_threshold: float = BBOX_IOU_THRESHOLD,
) -> Optional[np.ndarray]:
    """
    Detect humans in frame and return the largest bounding box.
    
    Uses temporal consistency: if the current largest bbox differs too much
    from the previous frame's bbox (low IoU), returns the previous bbox instead.
    This handles cases where the human detector fails or picks a wrong detection.
    
    Args:
        detector: Human detector instance
        frame: RGB image as numpy array
        prev_bbox: Previous frame's bounding box for temporal consistency
        det_cat_id: Detection category ID (0 for humans)
        bbox_thr: Bounding box confidence threshold
        nms_thr: Non-maximum suppression threshold
        iou_threshold: Minimum IoU with prev_bbox to accept new detection
        
    Returns:
        Bounding box as numpy array [x1, y1, x2, y2], or None if no detection
    """
    # Detector expects BGR format
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    boxes = detector.run_human_detection(
        frame_bgr,
        det_cat_id=det_cat_id,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        default_to_full_image=False,
    )
    
    if len(boxes) == 0:
        # No detection - use previous bbox if available
        return prev_bbox
    
    # Compute areas and select largest bbox
    # boxes shape: (N, 4) with format [x1, y1, x2, y2]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)
    current_bbox = boxes[largest_idx:largest_idx+1]  # Keep as (1, 4) shape
    
    # Check temporal consistency if we have a previous bbox
    if prev_bbox is not None:
        iou = compute_iou(current_bbox[0], prev_bbox[0])
        if iou < iou_threshold:
            # Current bbox is too different - use previous bbox
            return prev_bbox
    
    return current_bbox


def get_camera_intrinsics(
    fov_estimator, 
    frame: np.ndarray, 
    device: str = "cuda"
) -> torch.Tensor:
    """
    Estimate camera intrinsics from a single frame.
    
    Args:
        fov_estimator: FOV estimator instance
        frame: RGB image as numpy array
        device: Target device for the tensor
        
    Returns:
        Camera intrinsics tensor
    """
    # Convert to tensor and normalize
    # frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    # frame_tensor = frame_tensor.to(device)
    
    cam_int = fov_estimator.get_cam_intrinsics(frame)
    return cam_int


def count_frames(video_path: str) -> int:
    """Count total frames in video file or image directory."""
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    elif os.path.isdir(video_path):
        valid_extensions = ('.png', '.jpg', '.jpeg')
        return len([
            f for f in os.listdir(video_path) 
            if f.lower().endswith(valid_extensions)
        ])
    return 0


# =============================================================================
# Main Processing
# =============================================================================

def process_video(video_path: str, output_path: str, sam3_pred: Optional[str] = None) -> None:
    """
    Process video and extract 3D body pose for the largest detected person.
    
    Args:
        video_path: Path to input video file or image directory
        output_path: Path for output numpy file
    """
    # Initialize estimator
    estimator = setup_sam_3d_body(hf_repo_id=MODEL_NAME)
    
    # Verify required components are available
    if estimator.detector is None:
        raise RuntimeError("Human detector is required for bbox detection")
    if estimator.fov_estimator is None:
        raise RuntimeError("FOV estimator is required for camera intrinsics")
    
    outputs = []
    cam_int = None  # Will be set on first valid frame
    prev_bbox = None  # Track previous bbox for temporal consistency
    total_frames = count_frames(video_path)

    if sam3_pred is not None:
        sam3_pred = np.load(sam3_pred, allow_pickle=True).item()

        print(sam3_pred['boxes'].shape[0], total_frames)    
        # assert sam3_pred['boxes'].shape[0] == total_frames
        total_frames = sam3_pred['boxes'].shape[0]
        
    
    # Process frames with progress bar
    frame_iterator = load_frames(video_path)
    pbar = tqdm(
        frame_iterator,
        total=total_frames,
        desc="Processing video",
        ncols=80,
        ascii=True,
    )
    
    for frame, frame_idx in pbar:
        # Detect largest person in frame (with temporal consistency)
        if sam3_pred is not None:
            bbox = sam3_pred['boxes'][frame_idx:frame_idx+1]
        else:
            bbox = detect_largest_bbox(estimator.detector, frame, prev_bbox=prev_bbox)
        
        if bbox is None:
            # No human detected, skip frame
            pbar.set_postfix({"status": "no detection"})
            continue
        
        # First valid frame: estimate camera intrinsics
        if cam_int is None:
            pbar.set_postfix({"status": "estimating intrinsics"})
            cam_int = get_camera_intrinsics(
                estimator.fov_estimator, 
                frame, 
                device=estimator.device
            )
            print(f"\nCamera intrinsics estimated from frame {frame_idx}")
        
        # Run pose estimation with pre-computed bbox and cached intrinsics
        output = estimator.process_one_image(
            frame,
            bboxes=bbox,
            cam_int=cam_int,
        )
        
        if output:
            outputs.append(output[0])
            prev_bbox = bbox  # Update tracked bbox for temporal consistency
            pbar.set_postfix({"detections": len(outputs)})
    
    # Save results
    if outputs:
        outputs_array = np.array(outputs)
        np.save(output_path, outputs_array)
        print(f"\nSaved {len(outputs)} frames to {output_path}")
    else:
        print("\nNo valid detections to save")


def main():
    """Parse arguments and run video processing."""
    parser = argparse.ArgumentParser(
        description="Process video for 3D body pose estimation"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input video file or image directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path for output numpy file"
    )
    parser.add_argument(
        "--sam3_pred",
        type=str,
        required=False,
        help="Path for output numpy file"
    )
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.sam3_pred)


if __name__ == "__main__":
    main()