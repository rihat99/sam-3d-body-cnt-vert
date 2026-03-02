"""
Training script for per-vertex Contact Head on DAMON dataset.

Trains only the contact head (+ contact tokens + update layers) while keeping
the rest of SAM-3D-Body frozen.  Each run writes to a date-time-stamped folder
under OUTPUT.DIR so experiments are easy to track.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.utils.config import get_config

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
from damon_mhr import DamonMHRDataset
from dataset_utils import prepare_damon_batch


# ---------------------------------------------------------------------------
# Custom collate: images are numpy arrays of varying shape — keep as list
# ---------------------------------------------------------------------------

def damon_collate(batch):
    """
    batch: list of ((image, bbox, cam_k), contact_label)
    Returns:
        images    — list of B numpy arrays
        bboxes    — tensor [B, 4]
        cam_ks    — tensor [B, 3, 3]
        contact_labels — tensor [B, 18439]
    """
    images, bboxes, cam_ks, contact_labels = [], [], [], []
    for (img, bbox, cam_k), lbl in batch:
        images.append(img)
        bboxes.append(bbox)
        cam_ks.append(cam_k)
        contact_labels.append(lbl)
    return (
        images,
        torch.stack(bboxes),
        torch.stack(cam_ks),
        torch.stack(contact_labels),
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContactTrainer:
    """Trains per-vertex contact head on DAMON dataset."""

    def __init__(self, config_path: str, device: str = "cuda"):
        self.cfg = get_config(config_path)
        self.device = device

        # ---- Output directory: base/expname_YYYYMMDD_HHMMSS ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.cfg.OUTPUT.EXP_NAME}_{timestamp}"
        self.output_dir = Path(self.cfg.OUTPUT.DIR) / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(str(self.cfg))

        # TensorBoard
        if self.cfg.OUTPUT.USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        else:
            self.writer = None

        # ---- Load model ----
        print("Loading SAM-3D-Body model...")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=self.cfg.MODEL.CHECKPOINT_PATH,
            device=device,
            mhr_path=self.cfg.MODEL.MHR_MODEL_PATH,
        )

        # ---- Freeze all params; unfreeze contact-related ones ----
        print("Freezing all parameters except contact head & tokens...")
        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "contact" in name.lower():
                param.requires_grad = True
                print(f"  Unfrozen: {name}")

        # CRITICAL: decoder must stay in train mode so gradients flow to
        # contact query tokens even though decoder weights are frozen.
        for dec in [getattr(self.model, 'decoder', None),
                    getattr(self.model, 'decoder_hand', None)]:
            if dec is not None:
                for m in dec.modules():
                    m.train()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,}")

        # ---- Datasets ----
        print("Loading datasets...")
        data_root = self.cfg.DATASET.get('DATA_ROOT', None)
        val_ratio = self.cfg.DATASET.get('VAL_RATIO', 0.2)
        seed      = self.cfg.DATASET.get('SEED', 42)
        self.train_dataset, self.val_dataset = DamonMHRDataset.split_train_val(
            npz_path=self.cfg.DATASET.TRAINVAL_NPZ,
            val_ratio=val_ratio,
            seed=seed,
            data_root=data_root,
        )
        print(f"  Train: {len(self.train_dataset)}  Val: {len(self.val_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,
            drop_last=True,
            collate_fn=damon_collate,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,
            collate_fn=damon_collate,
        )

        # ---- Positive class weight ----
        self.pos_weight = self._compute_pos_weight()

        # ---- Optimizer & scheduler ----
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        self.scheduler = self._setup_scheduler()

        # ---- State ----
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        if self.cfg.TRAIN.RESUME:
            self._load_checkpoint(self.cfg.TRAIN.RESUME)

    # ------------------------------------------------------------------
    # Positive-weight computation
    # ------------------------------------------------------------------

    def _compute_pos_weight(self) -> torch.Tensor:
        if self.cfg.TRAIN.POS_WEIGHT is not None:
            pw = torch.tensor(self.cfg.TRAIN.POS_WEIGHT, dtype=torch.float32)
            return pw.to(self.device)

        print("Computing positive class weight from training set...")
        total_pos = 0
        total_neg = 0
        num_vertices = self.cfg.MODEL.CONTACT_HEAD.get('NUM_VERTICES', 18439)

        for _, _, _, contact_labels in tqdm(self.train_loader, desc="pos_weight"):
            pos = contact_labels.sum().item()
            total_pos += pos
            total_neg += contact_labels.numel() - pos

        pos_weight = total_neg / (total_pos + 1e-6)
        print(f"  pos/neg = {total_pos}/{total_neg}  pos_weight = {pos_weight:.2f}")
        return torch.tensor(pos_weight, dtype=torch.float32).to(self.device)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _setup_scheduler(self):
        warmup = self.cfg.TRAIN.get("LR_WARMUP_EPOCHS", 0)
        if self.cfg.TRAIN.LR_SCHEDULER == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.cfg.TRAIN.EPOCHS - warmup, 1),
                eta_min=self.cfg.TRAIN.LR_MIN,
            )
        elif self.cfg.TRAIN.LR_SCHEDULER == "step":
            main_sched = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            return None

        if warmup > 0:
            warm_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, total_iters=warmup
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warm_sched, main_sched], milestones=[warmup]
            )
        return main_sched

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(self, images, bboxes, cam_ks):
        return prepare_damon_batch(
            images,
            bboxes,
            cam_ks,
            target_size=tuple(self.cfg.MODEL.IMAGE_SIZE),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Per-vertex weighted binary cross-entropy.

        Args:
            logits:  [B, num_vertices]
            targets: [B, num_vertices] int64 binary
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
        )

    @torch.no_grad()
    def _compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5)
        gt = targets.bool()

        tp = (preds & gt).float().sum()
        fp = (preds & ~gt).float().sum()
        fn = (~preds & gt).float().sum()
        tn = (~preds & ~gt).float().sum()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item(),
        }

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()
        for dec in [getattr(self.model, 'decoder', None),
                    getattr(self.model, 'decoder_hand', None)]:
            if dec is not None:
                for m in dec.modules():
                    m.train()

        total_loss = 0.0
        total_metrics = {k: 0.0 for k in ('accuracy', 'precision', 'recall', 'f1', 'iou')}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (images, bboxes, cam_ks, contact_labels) in enumerate(pbar):
            contact_labels = contact_labels.to(self.device)

            batch = self._prepare_batch(images, bboxes, cam_ks)
            self.model._initialize_batch(batch)
            output = self.model.forward_step(batch, decoder_type="body")

            if output["contact"] is None:
                raise RuntimeError(
                    "No contact output — ensure DO_CONTACT_TOKENS: true in config."
                )

            logits = output["contact"]["contact_logits"]  # [B, 18439]
            loss = self._compute_loss(logits, contact_labels)

            self.optimizer.zero_grad()
            loss.backward()

            if self.cfg.TRAIN.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.GRAD_CLIP)

            self.optimizer.step()

            # Gradient check on first batch of first epoch
            if self.current_epoch == 0 and batch_idx == 0:
                print("\n=== Gradient Flow Check ===")
                for name, param in self.model.named_parameters():
                    if "contact" in name.lower() and param.requires_grad:
                        status = f"grad_norm={param.grad.norm().item():.6f}" if param.grad is not None else "NO GRAD"
                        print(f"  {name}: {status}")
                print("=" * 40 + "\n")

            metrics = self._compute_metrics(logits, contact_labels)

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'iou': f"{metrics['iou']:.4f}"})

            if self.writer and self.global_step % self.cfg.OUTPUT.LOG_FREQ == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/iou', metrics['iou'], self.global_step)
                self.writer.add_scalar('train/f1', metrics['f1'], self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in total_metrics.items()}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0.0
        total_metrics = {k: 0.0 for k in ('accuracy', 'precision', 'recall', 'f1', 'iou')}

        for images, bboxes, cam_ks, contact_labels in tqdm(self.val_loader, desc="Validation"):
            contact_labels = contact_labels.to(self.device)
            batch = self._prepare_batch(images, bboxes, cam_ks)
            self.model._initialize_batch(batch)
            output = self.model.forward_step(batch, decoder_type="body")
            logits = output["contact"]["contact_logits"]

            loss = self._compute_loss(logits, contact_labels)
            metrics = self._compute_metrics(logits, contact_labels)

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]

        n = len(self.val_loader)
        return total_loss / n, {k: v / n for k, v in total_metrics.items()}

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': str(self.cfg),
        }
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, checkpoint_path: str):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.current_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"\nStarting training — {self.cfg.TRAIN.EPOCHS} epochs")
        print(f"Output: {self.output_dir}\n")

        for epoch in range(self.current_epoch, self.cfg.TRAIN.EPOCHS):
            self.current_epoch = epoch

            train_loss, train_metrics = self.train_epoch()
            print(
                f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"IoU: {train_metrics['iou']:.4f}  F1: {train_metrics['f1']:.4f}  "
                f"Prec: {train_metrics['precision']:.4f}  Rec: {train_metrics['recall']:.4f}"
            )

            if epoch % self.cfg.TRAIN.VAL_FREQ == 0:
                val_loss, val_metrics = self.validate()
                print(
                    f"          Val  Loss: {val_loss:.4f} | "
                    f"IoU: {val_metrics['iou']:.4f}  F1: {val_metrics['f1']:.4f}  "
                    f"Prec: {val_metrics['precision']:.4f}  Rec: {val_metrics['recall']:.4f}"
                )

                if self.writer:
                    self.writer.add_scalar('val/loss', val_loss, epoch)
                    self.writer.add_scalar('val/iou', val_metrics['iou'], epoch)
                    self.writer.add_scalar('val/f1', val_metrics['f1'], epoch)
                    self.writer.add_scalar('val/precision', val_metrics['precision'], epoch)
                    self.writer.add_scalar('val/recall', val_metrics['recall'], epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best_model.pth')

            if epoch % self.cfg.TRAIN.SAVE_FREQ == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch:04d}.pth')

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            print(f"  lr = {lr:.2e}")

        self._save_checkpoint('final_model.pth')
        if self.writer:
            self.writer.close()
        print("\nTraining complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train per-vertex Contact Head on DAMON")
    parser.add_argument("--config", type=str, default="train/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    trainer = ContactTrainer(args.config, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
