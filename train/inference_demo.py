"""
Demo script showing how to use a trained contact head for inference

This script shows how to:
1. Load a model with trained contact head
2. Run inference on ETH dataset samples
3. Visualize contact predictions
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset" / "eth"))

os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.utils.config import get_config
from dataset import ETHContactDataset
from dataset_utils import prepare_training_batch


def visualize_contact_prediction(image, bbox, contact_gt, contact_pred, contact_probs, save_path=None):
    """
    Visualize contact predictions on image
    
    Args:
        image: RGB image (H, W, 3)
        bbox: Bounding box [x1, y1, x2, y2]
        contact_gt: Ground truth contacts [left_hand, right_hand, left_foot, right_foot]
        contact_pred: Predicted contacts [left_hand, right_hand, left_foot, right_foot]
        contact_probs: Contact probabilities [left_hand, right_hand, left_foot, right_foot]
        save_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Display image
    ax.imshow(image)
    
    # Draw bounding box
    x1, y1, x2, y2 = bbox
    rect = Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='yellow', facecolor='none',
        label='Person BBox'
    )
    ax.add_patch(rect)
    
    # Contact labels
    contact_names = ['L Hand', 'R Hand', 'L Foot', 'R Foot']
    
    # Create text box with predictions
    text_lines = ["Contact Predictions:"]
    text_lines.append("-" * 30)
    
    for i, name in enumerate(contact_names):
        gt = "Yes" if contact_gt[i] else "No"
        pred = "Yes" if contact_pred[i] else "No"
        prob = contact_probs[i]
        
        # Color code by correctness
        if contact_gt[i] == contact_pred[i]:
            color = "✓"
        else:
            color = "✗"
        
        text_lines.append(f"{color} {name}: GT={gt:3s} Pred={pred:3s} ({prob:.2f})")
    
    # Add text box
    text = "\n".join(text_lines)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
        family='monospace'
    )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Contact head inference demo")
    parser.add_argument(
        "--config",
        type=str,
        default="train/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train/inference_samples",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print("Loading config...")
    cfg = get_config(args.config)
    
    # Load base model
    print("Loading SAM-3D-Body model...")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=cfg.MODEL.CHECKPOINT_PATH,
        device=args.device,
        mhr_path=cfg.MODEL.MHR_MODEL_PATH
    )
    
    # Load trained contact head
    print(f"Loading trained contact head from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    dataset = ETHContactDataset(
        data_path=cfg.DATASET.DATA_PATH,
        folders=cfg.DATASET.TRAIN_FOLDERS,
        sides=cfg.DATASET.SIDES,
        contact_threshold=cfg.DATASET.CONTACT_THRESHOLD,
        rebuild_cache=False
    )
    
    print(f"\nRunning inference on {args.num_samples} samples...")
    
    # Process samples
    for idx in range(min(args.num_samples, len(dataset))):
        print(f"\nProcessing sample {idx + 1}/{args.num_samples}")

        sample_id = np.random.randint(0, len(dataset))
        
        # Get sample
        (image, bbox), contact_gt = dataset[sample_id]
        
        # Prepare batch
        cam_params = {
            'fx': cfg.CAMERA.fx,
            'fy': cfg.CAMERA.fy,
            'cx': cfg.CAMERA.cx,
            'cy': cfg.CAMERA.cy
        }
        
        batch = prepare_training_batch(
            [image],
            [bbox],
            cam_params,
            target_size=(896, 896),  # Maximum resolution with DINOv3
            device=args.device
        )
        
        # Run inference
        with torch.no_grad():
            model._initialize_batch(batch)
            output = model.forward_step(batch, decoder_type="body")
            contact_logits = output["contact"]["contact_logits"]
            contact_probs = torch.sigmoid(contact_logits).cpu().numpy()[0]
            contact_pred = (contact_probs > 0.5).astype(bool)
        
        # Visualize
        save_path = output_dir / f"sample_{idx:03d}.png"
        visualize_contact_prediction(
            image, bbox,
            contact_gt, contact_pred, contact_probs,
            save_path=save_path
        )
        
        # Print results
        contact_names = ['Left Hand', 'Right Hand', 'Left Foot', 'Right Foot']
        print("  Contact predictions:")
        for i, name in enumerate(contact_names):
            gt = "Yes" if contact_gt[i] else "No "
            pred = "Yes" if contact_pred[i] else "No "
            prob = contact_probs[i]
            match = "✓" if contact_gt[i] == contact_pred[i] else "✗"
            print(f"    {match} {name:>11}: GT={gt} Pred={pred} (prob={prob:.3f})")
    
    print(f"\n✓ Done! Visualizations saved to {output_dir}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"  Samples processed: {args.num_samples}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
