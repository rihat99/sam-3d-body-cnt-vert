"""
Evaluation script for trained contact head

This script evaluates a trained contact head on the validation set and computes:
- Overall accuracy
- Per-contact accuracy (left/right hand, left/right foot)
- Precision, recall, F1 score
- Confusion matrix
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset" / "eth"))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.utils.config import get_config
from dataset import ETHContactDataset
from dataset_utils import prepare_training_batch


class ContactEvaluator:
    """Evaluator for contact head"""
    
    def __init__(self, config_path, checkpoint_path, split="val", device="cuda"):
        self.device = device
        self.split = split  # "train" or "val"

        # Output directory for figures
        self.figures_dir = Path(__file__).parent / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self.cfg = get_config(config_path)
        
        # Load base model
        print("Loading SAM-3D-Body model...")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=self.cfg.MODEL.CHECKPOINT_PATH,
            device=device,
            mhr_path=self.cfg.MODEL.MHR_MODEL_PATH
        )
        
        # Load trained contact head weights
        print(f"Loading trained contact head from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.eval()
        
        # Load dataset
        print(f"Loading {split} dataset...")
        self.setup_dataset()
        
        # Contact names for reporting
        self.contact_names = ['Left Hand', 'Right Hand', 'Left Foot', 'Right Foot']
    
    def setup_dataset(self):
        """Setup dataset using the same split strategy as training.

        Respects self.split ("train" or "val") so that either half can be
        evaluated independently.
        """
        train_videos = self.cfg.DATASET.get('TRAIN_VIDEOS') or None
        val_videos   = self.cfg.DATASET.get('VAL_VIDEOS')   or None

        if train_videos and val_videos:
            # Explicit per-video lists
            videos = train_videos if self.split == 'train' else val_videos
            dataset = ETHContactDataset(
                data_path=self.cfg.DATASET.DATA_PATH,
                folders=self.cfg.DATASET.TRAIN_FOLDERS,
                videos=videos,
                sides=self.cfg.DATASET.SIDES,
                contact_threshold=self.cfg.DATASET.CONTACT_THRESHOLD,
                rebuild_cache=False
            )
        elif self.cfg.DATASET.VAL_FOLDERS and self.split == 'val':
            # Folder-level split — only applies to val
            dataset = ETHContactDataset(
                data_path=self.cfg.DATASET.DATA_PATH,
                folders=self.cfg.DATASET.VAL_FOLDERS,
                sides=self.cfg.DATASET.SIDES,
                contact_threshold=self.cfg.DATASET.CONTACT_THRESHOLD,
                rebuild_cache=False
            )
        elif self.cfg.DATASET.VAL_FOLDERS and self.split == 'train':
            # Folder-level split — train side
            dataset = ETHContactDataset(
                data_path=self.cfg.DATASET.DATA_PATH,
                folders=self.cfg.DATASET.TRAIN_FOLDERS,
                sides=self.cfg.DATASET.SIDES,
                contact_threshold=self.cfg.DATASET.CONTACT_THRESHOLD,
                rebuild_cache=False
            )
        else:
            # Video-level split — reproduce the same split as training
            video_split_ratio = self.cfg.DATASET.get('VIDEO_SPLIT_RATIO', 0.8)
            train_dataset, val_dataset = ETHContactDataset.split_by_videos(
                val_ratio=1.0 - video_split_ratio,
                seed=self.cfg.DATASET.SEED,
                data_path=self.cfg.DATASET.DATA_PATH,
                folders=self.cfg.DATASET.TRAIN_FOLDERS,
                sides=self.cfg.DATASET.SIDES,
                contact_threshold=self.cfg.DATASET.CONTACT_THRESHOLD,
                rebuild_cache=False
            )
            dataset = train_dataset if self.split == 'train' else val_dataset

        print(f"{self.split.capitalize()} samples: {len(dataset)}")

        self.val_loader = DataLoader(
            dataset,
            batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True
        )
    
    def prepare_batch_for_model(self, images, bboxes):
        """Prepare batch for model"""
        cam_params = {
            'fx': self.cfg.CAMERA.fx,
            'fy': self.cfg.CAMERA.fy,
            'cx': self.cfg.CAMERA.cx,
            'cy': self.cfg.CAMERA.cy
        }
        
        images_list = [images[i] for i in range(len(images))]
        bboxes_list = [bboxes[i] for i in range(len(bboxes))]
        
        return prepare_training_batch(
            images_list,
            bboxes_list,
            cam_params,
            target_size=(896, 896),  # Maximum resolution with DINOv3
            device=self.device
        )
    
    @torch.no_grad()
    def evaluate(self, threshold=0.5):
        """Evaluate model on validation set"""
        all_predictions = []
        all_ground_truth = []
        all_probabilities = []
        
        print("\nRunning evaluation...")
        for (images, bboxes), contacts in tqdm(self.val_loader):
            # Prepare batch
            batch = self.prepare_batch_for_model(images, bboxes)
            
            # Forward pass
            self.model._initialize_batch(batch)
            output = self.model.forward_step(batch, decoder_type="body")
            contact_logits = output["contact"]["contact_logits"]
            contact_probs = torch.sigmoid(contact_logits)
            
            # Store predictions and ground truth
            all_predictions.append((contact_probs > threshold).cpu().numpy())
            all_ground_truth.append(contacts.cpu().numpy())
            all_probabilities.append(contact_probs.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)  # (N, 4)
        ground_truth = np.concatenate(all_ground_truth, axis=0)  # (N, 4)
        probabilities = np.concatenate(all_probabilities, axis=0)  # (N, 4)
        
        # Compute metrics
        self.print_metrics(predictions, ground_truth, probabilities)
        
        # Plot confusion matrices
        self.plot_confusion_matrices(predictions, ground_truth)
        
        # Plot probability distributions
        self.plot_probability_distributions(probabilities, ground_truth)
        
        # Plot ROC curves and compute AUC
        self.plot_roc_curves(probabilities, ground_truth)
        
        return predictions, ground_truth, probabilities
    
    def print_metrics(self, predictions, ground_truth, probabilities):
        """Print evaluation metrics"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Overall metrics
        overall_acc = (predictions == ground_truth).mean()
        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        
        # Per-contact metrics
        print("\nPer-Contact Metrics:")
        print("-" * 80)
        print(f"{'Contact':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 80)
        
        for i, name in enumerate(self.contact_names):
            pred = predictions[:, i]
            gt = ground_truth[:, i]
            
            # Accuracy
            acc = (pred == gt).mean()
            
            # Precision, Recall, F1
            tp = ((pred == 1) & (gt == 1)).sum()
            fp = ((pred == 1) & (gt == 0)).sum()
            fn = ((pred == 0) & (gt == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{name:<15} {acc:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        print("-" * 80)
        
        # Contact frequency
        print("\nContact Frequency (% of samples with contact):")
        for i, name in enumerate(self.contact_names):
            gt_freq = ground_truth[:, i].mean()
            pred_freq = predictions[:, i].mean()
            print(f"  {name:<15} GT: {gt_freq:.2%}  Pred: {pred_freq:.2%}")
    
    def plot_confusion_matrices(self, predictions, ground_truth):
        """Plot confusion matrix for each contact"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, ax) in enumerate(zip(self.contact_names, axes)):
            # Compute confusion matrix
            pred = predictions[:, i]
            gt = ground_truth[:, i]
            
            cm = np.zeros((2, 2), dtype=int)
            cm[0, 0] = ((pred == 0) & (gt == 0)).sum()  # TN
            cm[0, 1] = ((pred == 1) & (gt == 0)).sum()  # FP
            cm[1, 0] = ((pred == 0) & (gt == 1)).sum()  # FN
            cm[1, 1] = ((pred == 1) & (gt == 1)).sum()  # TP
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Contact', 'Contact'],
                       yticklabels=['No Contact', 'Contact'])
            ax.set_title(f'{name}\nAccuracy: {(pred == gt).mean():.2%}')
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')
        
        plt.tight_layout()
        out_path = self.figures_dir / f"{self.split}_confusion_matrices.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved confusion matrices to {out_path}")
        plt.close()
    
    def plot_probability_distributions(self, probabilities, ground_truth):
        """Plot probability distributions for positive and negative samples"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, ax) in enumerate(zip(self.contact_names, axes)):
            probs = probabilities[:, i]
            gt = ground_truth[:, i]
            
            # Plot histograms
            ax.hist(probs[gt == 0], bins=50, alpha=0.5, label='No Contact', color='blue')
            ax.hist(probs[gt == 1], bins=50, alpha=0.5, label='Contact', color='red')
            
            # Add threshold line
            ax.axvline(0.5, color='black', linestyle='--', label='Threshold')
            
            ax.set_title(name)
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = self.figures_dir / f"{self.split}_probability_distributions.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved probability distributions to {out_path}")
        plt.close()
    
    def plot_roc_curves(self, probabilities, ground_truth):
        """Plot ROC curves and compute AUC for each contact"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        auc_scores = []
        
        # Plot individual ROC curves
        for i, (name, ax) in enumerate(zip(self.contact_names, axes[:4])):
            probs = probabilities[:, i]
            gt = ground_truth[:, i]
            
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(gt, probs)
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{name}\nROC Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        
        # Plot all ROC curves together
        ax_all = axes[4]
        colors = ['red', 'blue', 'green', 'orange']
        for i, (name, color) in enumerate(zip(self.contact_names, colors)):
            probs = probabilities[:, i]
            gt = ground_truth[:, i]
            
            fpr, tpr, _ = roc_curve(gt, probs)
            roc_auc = auc(fpr, tpr)
            
            ax_all.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax_all.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random')
        ax_all.set_xlim([0.0, 1.0])
        ax_all.set_ylim([0.0, 1.05])
        ax_all.set_xlabel('False Positive Rate')
        ax_all.set_ylabel('True Positive Rate')
        ax_all.set_title('All Contacts - ROC Curves')
        ax_all.legend(loc="lower right")
        ax_all.grid(True, alpha=0.3)
        
        # Summary table
        ax_summary = axes[5]
        ax_summary.axis('off')
        
        summary_text = "AUC Scores Summary\n" + "="*30 + "\n\n"
        for i, name in enumerate(self.contact_names):
            summary_text += f"{name:>12}: {auc_scores[i]:.4f}\n"
        summary_text += "-"*30 + "\n"
        summary_text += f"{'Mean AUC':>12}: {np.mean(auc_scores):.4f}\n"
        
        ax_summary.text(0.1, 0.5, summary_text,
                       fontsize=14,
                       family='monospace',
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        out_path = self.figures_dir / f"{self.split}_roc_curves.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved ROC curves to {out_path}")
        
        # Print AUC summary
        print("\n" + "="*80)
        print("AUC SCORES")
        print("="*80)
        for i, name in enumerate(self.contact_names):
            print(f"{name:>15}: {auc_scores[i]:.4f}")
        print("-"*80)
        print(f"{'Mean AUC':>15}: {np.mean(auc_scores):.4f}")
        print("="*80)
        
        plt.close()
        
        return auc_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained contact head")
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
        help="Path to trained checkpoint (e.g., train/output/contact_head_eth/best_model.pth)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which data split to evaluate on (default: val)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for contact probability"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ContactEvaluator(
        args.config, args.checkpoint, split=args.split, device=args.device
    )
    
    # Run evaluation
    predictions, ground_truth, probabilities = evaluator.evaluate(threshold=args.threshold)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
