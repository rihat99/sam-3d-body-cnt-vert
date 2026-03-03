#!/usr/bin/env python3
"""
Append predicted bounding boxes and camera parameters to Damon MHR dataset.

This script:
1. Loads images from the Damon dataset
2. Uses ViTDet to detect human bounding boxes
3. Uses MoGe2 to estimate camera intrinsics
4. Selects the largest bbox if multiple humans are detected
5. Appends bbox and cam_k to the NPZ files
"""

import os
import sys
import numpy as np
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

# Add sam-3d-body-cnt-vert to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sam-3d-body-cnt-vert"))

from tools.build_detector import HumanDetector
from tools.build_fov_estimator import FOVEstimator


def compute_bbox_area(bbox):
    """Compute area of a bounding box [x1, y1, x2, y2]"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def select_largest_bbox(bboxes):
    """Select the largest bbox from multiple detections"""
    if len(bboxes) == 0:
        return None
    if len(bboxes) == 1:
        return bboxes[0]
    
    # Compute areas and select largest
    areas = [compute_bbox_area(bbox) for bbox in bboxes]
    largest_idx = np.argmax(areas)
    return bboxes[largest_idx]


def process_dataset(
    npz_path: str,
    data_root: str,
    output_path: str,
    detector: HumanDetector,
    fov_estimator: FOVEstimator,
    device: str = "cuda"
):
    """
    Process a dataset NPZ file and append bbox and cam_k predictions.
    
    Args:
        npz_path: Path to input NPZ file
        data_root: Root directory for images
        output_path: Path to output NPZ file (can be same as input)
        detector: Human detector
        fov_estimator: FOV/camera parameter estimator
        device: Device to use
    """
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(npz_path)}")
    print(f"{'='*80}")
    
    # Load existing dataset
    data = np.load(npz_path, allow_pickle=True)
    imgnames = data['imgname']
    num_samples = len(imgnames)
    
    print(f"Number of samples: {num_samples}")
    
    # Check if already processed
    if 'bbox' in data.keys() and 'cam_k' in data.keys():
        print("Warning: Dataset already contains 'bbox' and 'cam_k' fields.")
        print("They will be overwritten.")
    
    # Initialize arrays for new data
    bboxes = np.zeros((num_samples, 4), dtype=np.float32)
    cam_ks = np.zeros((num_samples, 3, 3), dtype=np.float32)
    
    # Track statistics
    stats = {
        'no_detection': 0,
        'single_person': 0,
        'multiple_people': 0,
        'errors': 0
    }
    
    # Process each image
    print("\nProcessing images...")
    for idx in tqdm(range(num_samples)):
        imgname = imgnames[idx]
        img_path = os.path.join(data_root, imgname)
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
            
            H, W = img.shape[:2]
            
            # Detect humans
            detected_bboxes = detector.run_human_detection(img, bbox_thr=0.5)
            
            if len(detected_bboxes) == 0:
                # No detection - use full image
                bbox = np.array([0, 0, W, H], dtype=np.float32)
                stats['no_detection'] += 1
            elif len(detected_bboxes) == 1:
                # Single person
                bbox = detected_bboxes[0]
                stats['single_person'] += 1
            else:
                # Multiple people - select largest
                bbox = select_largest_bbox(detected_bboxes)
                stats['multiple_people'] += 1
            
            bboxes[idx] = bbox
            
            # Estimate camera intrinsics
            # Convert BGR to RGB for MoGe2
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cam_intrinsics = fov_estimator.get_cam_intrinsics(img_rgb)
            cam_ks[idx] = cam_intrinsics[0]  # Remove batch dimension
            
        except Exception as e:
            print(f"\nError processing sample {idx} ({imgname}): {e}")
            stats['errors'] += 1
            # Use full image bbox and default camera as fallback
            H, W = 480, 640  # Default dimensions
            bboxes[idx] = np.array([0, 0, W, H], dtype=np.float32)
            # Default camera (focal length ~= image width)
            cam_ks[idx] = np.array([
                [W, 0, W/2],
                [0, W, H/2],
                [0, 0, 1]
            ], dtype=np.float32)
    
    # Print statistics
    print(f"\n{'='*80}")
    print("Processing Statistics:")
    print(f"  Total samples: {num_samples}")
    print(f"  Single person detected: {stats['single_person']} ({stats['single_person']/num_samples*100:.1f}%)")
    print(f"  Multiple people (selected largest): {stats['multiple_people']} ({stats['multiple_people']/num_samples*100:.1f}%)")
    print(f"  No detection (used full image): {stats['no_detection']} ({stats['no_detection']/num_samples*100:.1f}%)")
    print(f"  Errors: {stats['errors']}")
    print(f"{'='*80}")
    
    # Sample statistics
    print("\nSample bbox and camera parameters (first 3 samples):")
    for i in range(min(3, num_samples)):
        print(f"\nSample {i} ({imgnames[i]}):")
        print(f"  BBox: [{bboxes[i][0]:.1f}, {bboxes[i][1]:.1f}, {bboxes[i][2]:.1f}, {bboxes[i][3]:.1f}]")
        print(f"  BBox size: {bboxes[i][2]-bboxes[i][0]:.1f} x {bboxes[i][3]-bboxes[i][1]:.1f}")
        print(f"  Focal length: fx={cam_ks[i][0,0]:.1f}, fy={cam_ks[i][1,1]:.1f}")
        print(f"  Principal point: cx={cam_ks[i][0,2]:.1f}, cy={cam_ks[i][1,2]:.1f}")
    
    # Create output dictionary with all data
    output_data = {}
    for key in data.keys():
        output_data[key] = data[key]
    
    # Add new fields
    output_data['bbox'] = bboxes
    output_data['cam_k'] = cam_ks
    
    # Save updated dataset
    print(f"\nSaving updated dataset to: {output_path}")
    np.savez(output_path, **output_data)
    
    # Verify saved data
    print("\nVerifying saved data...")
    verify_data = np.load(output_path, allow_pickle=True)
    print(f"  Keys: {list(verify_data.keys())}")
    print(f"  bbox shape: {verify_data['bbox'].shape}")
    print(f"  cam_k shape: {verify_data['cam_k'].shape}")
    
    print(f"\n✓ Successfully processed and saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Append predicted bboxes and camera parameters to Damon MHR dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/rikhat/Hard/datasets/DECO/",
        help="Root directory for images"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./damon_mhr_contact",
        help="Directory containing NPZ files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--detector_name",
        type=str,
        default="vitdet",
        help="Detector name (vitdet or sam3)"
    )
    parser.add_argument(
        "--fov_name",
        type=str,
        default="moge2",
        help="FOV estimator name (moge2)"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Update files in place (default: create _with_bbox suffix)"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("Damon MHR Dataset - Append BBox and Camera Parameters")
    print("="*80)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"\nDevice: {args.device}")
    print(f"Data root: {args.data_root}")
    print(f"Dataset directory: {args.dataset_dir}")
    
    # Initialize models
    print(f"\nInitializing models...")
    print(f"  Detector: {args.detector_name}")
    detector = HumanDetector(name=args.detector_name, device=args.device)
    
    print(f"  FOV Estimator: {args.fov_name}")
    fov_estimator = FOVEstimator(name=args.fov_name, device=args.device)
    
    print("✓ Models loaded successfully")
    
    # Find NPZ files
    dataset_files = [
        "hot_dca_trainval_mhr.npz",
        "hot_dca_test_mhr.npz"
    ]
    
    # Process each dataset file
    for dataset_file in dataset_files:
        npz_path = os.path.join(args.dataset_dir, dataset_file)
        
        if not os.path.exists(npz_path):
            print(f"\nSkipping {dataset_file} (not found)")
            continue
        
        # Determine output path
        if args.inplace:
            output_path = npz_path
        else:
            # Create backup with _with_bbox suffix
            base_name = dataset_file.replace('.npz', '')
            output_path = os.path.join(args.dataset_dir, f"{base_name}_with_bbox.npz")
        
        # Process dataset
        process_dataset(
            npz_path=npz_path,
            data_root=args.data_root,
            output_path=output_path,
            detector=detector,
            fov_estimator=fov_estimator,
            device=args.device
        )
    
    print("\n" + "="*80)
    print("All datasets processed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
