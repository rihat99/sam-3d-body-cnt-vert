"""
Training script for Contact Head on ETH dataset

This script trains only the contact head while keeping the rest of the SAM-3D-Body model frozen.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path

# Add parent directory to path to import sam_3d_body modules
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.utils.config import get_config

# Add current directory and dataset directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset" / "eth"))
from dataset import ETHContactDataset
from dataset_utils import prepare_training_batch


class ContactTrainer:
    """Trainer for contact head"""
    
    def __init__(self, config_path, device="cuda"):
        self.cfg = get_config(config_path)
        self.device = device
        
        # Create output directory
        self.output_dir = Path(self.cfg.OUTPUT.DIR) / self.cfg.OUTPUT.EXP_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(str(self.cfg))
        
        # Setup tensorboard
        if self.cfg.OUTPUT.USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        else:
            self.writer = None
        
        # Load model
        print("Loading SAM-3D-Body model...")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=self.cfg.MODEL.CHECKPOINT_PATH,
            device=device,
            mhr_path=self.cfg.MODEL.MHR_MODEL_PATH
        )
        
        # Freeze all parameters
        print("Freezing all model parameters except contact head...")
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze only contact head parameters
        for name, param in self.model.named_parameters():
            if "contact" in name.lower():
                param.requires_grad = True
                print(f"  Unfrozen: {name}")
        
        # CRITICAL: Keep decoder in train mode for gradient flow
        # Even though decoder parameters are frozen, we need gradients to flow
        # through it to reach the contact query tokens
        print("Setting decoder to train mode for gradient flow...")
        if hasattr(self.model, 'decoder'):
            self.model.decoder.train()
            # Ensure all layers are in train mode
            for module in self.model.decoder.modules():
                module.train()
        if hasattr(self.model, 'decoder_hand'):
            self.model.decoder_hand.train()
            for module in self.model.decoder_hand.modules():
                module.train()
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        
        # Verify gradient flow will work
        print("\nVerifying gradient flow setup...")
        decoder_train_mode = self.model.decoder.training if hasattr(self.model, 'decoder') else False
        decoder_hand_train_mode = self.model.decoder_hand.training if hasattr(self.model, 'decoder_hand') else False
        print(f"  Decoder train mode: {decoder_train_mode}")
        print(f"  Decoder (hand) train mode: {decoder_hand_train_mode}")
        if not decoder_train_mode or not decoder_hand_train_mode:
            print("  WARNING: Decoders not in train mode, gradients may not flow!")
        
        # Setup datasets
        print("Loading ETH dataset...")
        self.setup_datasets()
        
        # Log dataset statistics
        self.log_dataset_statistics()
        
        # Compute positive weights for imbalanced data
        self.pos_weight = self.compute_pos_weight()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )
        
        # Setup learning rate scheduler
        self.scheduler = self.setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint if specified
        if self.cfg.TRAIN.RESUME:
            self.load_checkpoint(self.cfg.TRAIN.RESUME)
    
    def setup_datasets(self):
        """Setup training and validation datasets"""
        # Determine train/val split
        if self.cfg.DATASET.VAL_FOLDERS:
            # Use separate folders for validation
            train_dataset = ETHContactDataset(
                data_path=self.cfg.DATASET.DATA_PATH,
                folders=self.cfg.DATASET.TRAIN_FOLDERS,
                sides=self.cfg.DATASET.SIDES,
                contact_threshold=self.cfg.DATASET.CONTACT_THRESHOLD,
                rebuild_cache=self.cfg.DATASET.REBUILD_CACHE
            )
            val_dataset = ETHContactDataset(
                data_path=self.cfg.DATASET.DATA_PATH,
                folders=self.cfg.DATASET.VAL_FOLDERS,
                sides=self.cfg.DATASET.SIDES,
                contact_threshold=self.cfg.DATASET.CONTACT_THRESHOLD,
                rebuild_cache=self.cfg.DATASET.REBUILD_CACHE
            )
        else:
            # Split single dataset
            full_dataset = ETHContactDataset(
                data_path=self.cfg.DATASET.DATA_PATH,
                folders=self.cfg.DATASET.TRAIN_FOLDERS,
                sides=self.cfg.DATASET.SIDES,
                contact_threshold=self.cfg.DATASET.CONTACT_THRESHOLD,
                rebuild_cache=self.cfg.DATASET.REBUILD_CACHE
            )
            
            # Compute split sizes
            total_size = len(full_dataset)
            train_size = int(total_size * self.cfg.DATASET.TRAIN_VAL_SPLIT)
            val_size = total_size - train_size
            
            # Set random seed for reproducibility
            torch.manual_seed(self.cfg.DATASET.SEED)
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Store datasets for statistics
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True
        )
    
    def log_dataset_statistics(self):
        """Log statistics about the dataset to identify potential issues"""
        print("\n=== Dataset Statistics ===")
        
        # Get the underlying dataset (handle random_split wrapper)
        if hasattr(self.train_dataset, 'dataset'):
            train_ds = self.train_dataset.dataset
        else:
            train_ds = self.train_dataset
            
        if hasattr(train_ds, 'samples'):
            samples = train_ds.samples
            
            # Count by folder (folder 1 has no feet labels, folder 2 has feet labels)
            folder_counts = {}
            for sample in samples:
                folder = sample.get('folder', 'unknown')
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            print(f"Samples by folder:")
            for folder, count in sorted(folder_counts.items()):
                print(f"  Folder {folder}: {count} samples")
            
            print(f"\nNote: Folder '1' has hand contacts only (feet=False)")
            print(f"      Folder '2' has both hand and feet contacts")
            
            if '1' in folder_counts and '2' in folder_counts:
                total = folder_counts['1'] + folder_counts['2']
                print(f"\nData split: {folder_counts['1']/total*100:.1f}% no-feet-labels, "
                      f"{folder_counts['2']/total*100:.1f}% with-feet-labels")
        
        print("=" * 40 + "\n")
    
    def compute_pos_weight(self):
        """Compute positive weights for imbalanced classes"""
        if self.cfg.TRAIN.POS_WEIGHT is not None:
            return torch.tensor(self.cfg.TRAIN.POS_WEIGHT).to(self.device)
        
        print("Computing positive weights from dataset...")
        contact_counts = torch.zeros(4)  # [left_hand, right_hand, left_foot, right_foot]
        total_samples = 0
        
        for batch in tqdm(self.train_loader, desc="Computing weights"):
            _, contacts = batch
            contact_counts += contacts.sum(dim=0)
            total_samples += contacts.shape[0]
        
        # Compute pos_weight as (num_negative / num_positive)
        neg_counts = total_samples - contact_counts
        pos_weight = neg_counts / (contact_counts + 1e-6)
        
        print(f"Contact counts: {contact_counts}")
        print(f"Positive weights: {pos_weight}")
        
        return pos_weight.to(self.device)
    
    def setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        warmup_epochs = self.cfg.TRAIN.get("LR_WARMUP_EPOCHS", 0)
        
        if self.cfg.TRAIN.LR_SCHEDULER == "cosine":
            # Cosine annealing after warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.TRAIN.EPOCHS - warmup_epochs,
                eta_min=self.cfg.TRAIN.LR_MIN
            )
            
            if warmup_epochs > 0:
                # Linear warmup from 0 to base LR
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,  # Start from 1% of base LR
                    total_iters=warmup_epochs
                )
                # Sequential: warmup then cosine
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            return cosine_scheduler
            
        elif self.cfg.TRAIN.LR_SCHEDULER == "step":
            step_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
            
            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    total_iters=warmup_epochs
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, step_scheduler],
                    milestones=[warmup_epochs]
                )
            return step_scheduler
        else:
            return None
    
    def prepare_batch_for_model(self, images, bboxes):
        """Prepare batch in format expected by SAM-3D-Body model"""
        cam_params = {
            'fx': self.cfg.CAMERA.fx,
            'fy': self.cfg.CAMERA.fy,
            'cx': self.cfg.CAMERA.cx,
            'cy': self.cfg.CAMERA.cy
        }
        
        # Convert to lists
        images_list = [images[i] for i in range(len(images))]
        bboxes_list = [bboxes[i] for i in range(len(bboxes))]
        
        batch = prepare_training_batch(
            images_list,
            bboxes_list,
            cam_params,
            target_size=(896, 896),  # Maximum resolution with DINOv3
            device=self.device
        )
        
        return batch
    
    def compute_loss(self, contact_logits, contact_gt):
        """
        Compute binary cross-entropy loss for contact prediction
        
        Args:
            contact_logits: (B, 4) - logits for each contact
            contact_gt: (B, 4) - ground truth contact labels
        """
        # Separate hand and foot contacts
        hand_logits = contact_logits[:, :2]  # left_hand, right_hand
        foot_logits = contact_logits[:, 2:]  # left_foot, right_foot
        
        hand_gt = contact_gt[:, :2].float()
        foot_gt = contact_gt[:, 2:].float()
        
        # Compute weighted BCE loss
        hand_loss = F.binary_cross_entropy_with_logits(
            hand_logits,
            hand_gt,
            pos_weight=self.pos_weight[:2]
        )
        
        foot_loss = F.binary_cross_entropy_with_logits(
            foot_logits,
            foot_gt,
            pos_weight=self.pos_weight[2:]
        )
        
        # Combine with weights
        total_loss = (
            self.cfg.TRAIN.LOSS_WEIGHTS.hand * hand_loss +
            self.cfg.TRAIN.LOSS_WEIGHTS.foot * foot_loss
        )
        
        return total_loss, hand_loss, foot_loss
    
    def compute_metrics(self, contact_logits, contact_gt):
        """Compute accuracy metrics"""
        contact_pred = (torch.sigmoid(contact_logits) > 0.5).float()
        
        # Per-contact accuracy
        accuracy = (contact_pred == contact_gt.float()).float().mean(dim=0)
        
        # Overall accuracy
        overall_acc = (contact_pred == contact_gt.float()).float().mean()
        
        return {
            'overall': overall_acc.item(),
            'left_hand': accuracy[0].item(),
            'right_hand': accuracy[1].item(),
            'left_foot': accuracy[2].item(),
            'right_foot': accuracy[3].item()
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # CRITICAL: Re-enable train mode for decoder after model.train()
        # This ensures gradients flow through the frozen decoder to contact tokens
        if hasattr(self.model, 'decoder'):
            self.model.decoder.train()
            for module in self.model.decoder.modules():
                module.train()
        if hasattr(self.model, 'decoder_hand'):
            self.model.decoder_hand.train()
            for module in self.model.decoder_hand.modules():
                module.train()
        
        total_loss = 0
        total_hand_loss = 0
        total_foot_loss = 0
        total_metrics = {
            'overall': 0,
            'left_hand': 0,
            'right_hand': 0,
            'left_foot': 0,
            'right_foot': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, ((images, bboxes), contacts) in enumerate(pbar):
            # Prepare batch
            batch = self.prepare_batch_for_model(images, bboxes)
            contacts = contacts.to(self.device)
            
            # Forward pass (no mixed precision due to MHR sparse operations)
            # Initialize batch size info
            self.model._initialize_batch(batch)
            output = self.model.forward_step(batch, decoder_type="body")
                
            # Get contact predictions
            if output["contact"] is None:
                raise RuntimeError("Model did not return contact predictions. "
                                    "Make sure DO_CONTACT_TOKENS is enabled in config.")
            
            contact_logits = output["contact"]["contact_logits"]
            
            # Compute loss
            loss, hand_loss, foot_loss = self.compute_loss(contact_logits, contacts)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.cfg.TRAIN.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.TRAIN.GRAD_CLIP
                )
            
            self.optimizer.step()
            
            # On first batch of first epoch, verify gradients are flowing to contact tokens
            if self.current_epoch == 0 and batch_idx == 0:
                print("\n=== Gradient Flow Check (First Batch) ===")
                for name, param in self.model.named_parameters():
                    if "contact" in name.lower() and param.requires_grad:
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            print(f"  ✓ {name}: grad_norm={grad_norm:.6f}")
                        else:
                            print(f"  ✗ {name}: NO GRADIENT!")
                print("=" * 50 + "\n")
            
            # Compute metrics
            with torch.no_grad():
                metrics = self.compute_metrics(contact_logits, contacts)
            
            # Update statistics
            total_loss += loss.item()
            total_hand_loss += hand_loss.item()
            total_foot_loss += foot_loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': metrics['overall']
            })
            
            # Log to tensorboard
            if self.writer and self.global_step % self.cfg.OUTPUT.LOG_FREQ == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/hand_loss', hand_loss.item(), self.global_step)
                self.writer.add_scalar('train/foot_loss', foot_loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', metrics['overall'], self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # Compute epoch averages
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_hand_loss = total_hand_loss / num_batches
        avg_foot_loss = total_foot_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_loss, avg_hand_loss, avg_foot_loss, avg_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_hand_loss = 0
        total_foot_loss = 0
        total_metrics = {
            'overall': 0,
            'left_hand': 0,
            'right_hand': 0,
            'left_foot': 0,
            'right_foot': 0
        }
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for (images, bboxes), contacts in pbar:
            # Prepare batch
            batch = self.prepare_batch_for_model(images, bboxes)
            contacts = contacts.to(self.device)
            
            # Forward pass
            self.model._initialize_batch(batch)
            output = self.model.forward_step(batch, decoder_type="body")
            contact_logits = output["contact"]["contact_logits"]
            
            # Compute loss and metrics
            loss, hand_loss, foot_loss = self.compute_loss(contact_logits, contacts)
            metrics = self.compute_metrics(contact_logits, contacts)
            
            # Update statistics
            total_loss += loss.item()
            total_hand_loss += hand_loss.item()
            total_foot_loss += foot_loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
        # Compute averages
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_hand_loss = total_hand_loss / num_batches
        avg_foot_loss = total_foot_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_loss, avg_hand_loss, avg_foot_loss, avg_metrics
    
    def save_checkpoint(self, filename):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': str(self.cfg)
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.current_epoch, self.cfg.TRAIN.EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_hand_loss, train_foot_loss, train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f} "
                  f"(Hand: {train_hand_loss:.4f}, Foot: {train_foot_loss:.4f})")
            print(f"  Train Accuracy: {train_metrics['overall']:.4f} "
                  f"(LH: {train_metrics['left_hand']:.4f}, RH: {train_metrics['right_hand']:.4f}, "
                  f"LF: {train_metrics['left_foot']:.4f}, RF: {train_metrics['right_foot']:.4f})")
            
            # Validate
            if epoch % self.cfg.TRAIN.VAL_FREQ == 0:
                val_loss, val_hand_loss, val_foot_loss, val_metrics = self.validate()
                
                print(f"  Val Loss: {val_loss:.4f} "
                      f"(Hand: {val_hand_loss:.4f}, Foot: {val_foot_loss:.4f})")
                print(f"  Val Accuracy: {val_metrics['overall']:.4f} "
                      f"(LH: {val_metrics['left_hand']:.4f}, RH: {val_metrics['right_hand']:.4f}, "
                      f"LF: {val_metrics['left_foot']:.4f}, RF: {val_metrics['right_foot']:.4f})")
                
                # Log to tensorboard
                if self.writer:
                    self.writer.add_scalar('val/loss', val_loss, epoch)
                    self.writer.add_scalar('val/hand_loss', val_hand_loss, epoch)
                    self.writer.add_scalar('val/foot_loss', val_foot_loss, epoch)
                    self.writer.add_scalar('val/accuracy', val_metrics['overall'], epoch)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
            
            # Save periodic checkpoint
            if epoch % self.cfg.TRAIN.SAVE_FREQ == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.2e}")
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        if self.writer:
            self.writer.close()
        
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Contact Head on ETH Dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="train/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training"
    )
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = ContactTrainer(args.config, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
