"""
Evaluation script for the per-vertex Contact Head on DAMON dataset.

Evaluates a trained checkpoint on any of the three splits (train / val / test)
and reports:
  - Overall per-vertex accuracy, precision, recall, F1, IoU
  - Per-sample IoU histogram
  - Probability distribution (contact vs no-contact)
  - ROC curve + AUC

Usage:
    python train/evaluate.py \
        --config train/config.yaml \
        --checkpoint train/output/contact_vert_20260222_123456/best_model.pth \
        --split val
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mhr_smpl_conversion"))
os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.utils.config import get_config
from damon_mhr import DamonMHRDataset
from damon_smpl import DamonSMPLDataset
from dataset_utils import prepare_damon_batch
from train_contact import damon_collate
from body_converter import BodyConverter


# ---------------------------------------------------------------------------

class ContactEvaluator:
    """Evaluates per-vertex contact prediction on DAMON dataset."""

    def __init__(self, config_path: str, checkpoint_path: str,
                 split: str = "val", device: str = "cuda", mode: str = "smpl"):
        self.device = device
        self.split = split
        self.mode = mode  # "smpl" or "mhr"

        self.figures_dir = Path(__file__).parent / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = get_config(config_path)

        # Load model
        print("Loading SAM-3D-Body model...")
        self.model, _ = load_sam_3d_body(
            checkpoint_path=self.cfg.MODEL.CHECKPOINT_PATH,
            device=device,
            mhr_path=self.cfg.MODEL.MHR_MODEL_PATH,
        )

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # Contact converter (SMPL mode only — fast, CPU is fine)
        if self.mode == "smpl":
            print("Initialising MHR→SMPL contact converter...")
            self.converter = BodyConverter(device="cpu")

        # Dataset
        print(f"Loading {split} dataset...")
        self._setup_dataset()

    # ------------------------------------------------------------------

    def _setup_dataset(self):
        data_root = self.cfg.DATASET.get('DATA_ROOT', None)
        val_ratio = self.cfg.DATASET.get('VAL_RATIO', 0.2)
        seed      = self.cfg.DATASET.get('SEED', 42)

        if self.mode == "smpl":
            # Load GT contacts from original DECO SMPL NPZ
            smpl_trainval = self.cfg.DATASET.get('SMPL_TRAINVAL_NPZ', None)
            smpl_test     = self.cfg.DATASET.get('SMPL_TEST_NPZ', None)
            mhr_trainval  = self.cfg.DATASET.get('TRAINVAL_NPZ', None)
            mhr_test      = self.cfg.DATASET.get('TEST_NPZ', None)

            if self.split == 'test':
                if not smpl_test:
                    raise ValueError("DATASET.SMPL_TEST_NPZ not set in config.")
                dataset = DamonSMPLDataset(
                    smpl_npz_path=smpl_test,
                    mhr_npz_path=mhr_test,
                    data_root=data_root,
                )
            else:
                if not smpl_trainval:
                    raise ValueError("DATASET.SMPL_TRAINVAL_NPZ not set in config.")
                train_ds, val_ds = DamonSMPLDataset.split_train_val(
                    smpl_npz_path=smpl_trainval,
                    mhr_npz_path=mhr_trainval,
                    val_ratio=val_ratio,
                    seed=seed,
                    data_root=data_root,
                )
                dataset = train_ds if self.split == 'train' else val_ds
        else:
            # MHR mode — original behaviour
            if self.split == 'test':
                test_npz = self.cfg.DATASET.get('TEST_NPZ', None)
                if not test_npz:
                    raise ValueError("DATASET.TEST_NPZ not set in config.")
                dataset = DamonMHRDataset(npz_path=test_npz, data_root=data_root)
            else:
                # Reproduce the exact same train/val split used during training.
                train_ds, val_ds = DamonMHRDataset.split_train_val(
                    npz_path=self.cfg.DATASET.TRAINVAL_NPZ,
                    val_ratio=val_ratio,
                    seed=seed,
                    data_root=data_root,
                )
                dataset = train_ds if self.split == 'train' else val_ds

        print(f"  {self.split.capitalize()} samples: {len(dataset)}")

        self.loader = DataLoader(
            dataset,
            batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,
            collate_fn=damon_collate,
        )

    # ------------------------------------------------------------------

    def _prepare_batch(self, images, bboxes, cam_ks):
        return prepare_damon_batch(
            images, bboxes, cam_ks,
            target_size=tuple(self.cfg.MODEL.IMAGE_SIZE),
            device=self.device,
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, threshold: float = 0.5):
        all_probs = []   # [N, V_mhr=18439]
        all_gt = []      # [N, V_mhr or V_smpl]

        for images, bboxes, cam_ks, contact_labels in tqdm(self.loader, desc="Evaluating"):
            batch = self._prepare_batch(images, bboxes, cam_ks)
            self.model._initialize_batch(batch)
            output = self.model.forward_step(batch, decoder_type="body")
            logits = output["contact"]["contact_logits"]
            probs = torch.sigmoid(logits).cpu().float()

            all_probs.append(probs.numpy())
            all_gt.append(contact_labels.numpy())

        all_probs = np.concatenate(all_probs, axis=0)            # [N, 18439]
        all_gt    = np.concatenate(all_gt, axis=0).astype(bool)  # [N, V]

        if self.mode == "smpl":
            # Interpolate continuous MHR probs → SMPL space, then threshold once.
            # We use the raw _interpolate() path so we get float values for ROC/plots.
            print("Converting MHR predictions to SMPL space...")
            smpl_probs = self._interpolate_probs_to_smpl(all_probs)  # [N, 6890] float
            all_preds  = smpl_probs > threshold                       # [N, 6890] bool
            self._print_metrics(all_preds, all_gt, smpl_probs)
            self._plot_iou_histogram(all_preds, all_gt)
            self._plot_prob_distribution(smpl_probs, all_gt)
            self._plot_roc_curve(smpl_probs, all_gt)
            return smpl_probs, all_preds, all_gt
        else:
            all_preds = all_probs > threshold    # [N, 18439]
            self._print_metrics(all_preds, all_gt, all_probs)
            self._plot_iou_histogram(all_preds, all_gt)
            self._plot_prob_distribution(all_probs, all_gt)
            self._plot_roc_curve(all_probs, all_gt)
            return all_probs, all_preds, all_gt

    def _interpolate_probs_to_smpl(self, mhr_probs: np.ndarray) -> np.ndarray:
        """
        Interpolate continuous MHR probabilities [N, 18439] → SMPL [N, 6890].

        Uses the raw _interpolate() path so we get float values (not yet
        thresholded) suitable for ROC and probability-distribution plots.
        """
        probs_t = torch.from_numpy(mhr_probs)  # [N, 18439] float32
        interp = self.converter._interpolate(
            probs_t,
            self.converter._m2s_tri_ids,
            self.converter._m2s_baryc,
            self.converter._mhr_faces,
        )
        return interp.numpy()  # [N, 6890]

    # ------------------------------------------------------------------
    # Metric reporting
    # ------------------------------------------------------------------

    def _print_metrics(self, preds, gt, probs):
        tp = (preds & gt).sum()
        fp = (preds & ~gt).sum()
        fn = (~preds & gt).sum()
        tn = (~preds & ~gt).sum()

        accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = tp / (tp + fp + fn + 1e-8)

        # Per-sample IoU
        per_sample_tp = (preds & gt).sum(axis=1)
        per_sample_fp = (preds & ~gt).sum(axis=1)
        per_sample_fn = (~preds & gt).sum(axis=1)
        per_sample_iou = per_sample_tp / (per_sample_tp + per_sample_fp + per_sample_fn + 1e-8)

        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS  [{self.split}]")
        print("=" * 70)
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  IoU      : {iou:.4f}")
        print(f"  Mean per-sample IoU: {per_sample_iou.mean():.4f}  "
              f"(median: {np.median(per_sample_iou):.4f})")
        print(f"  GT contact rate : {gt.mean():.4f}")
        print(f"  Pred contact rate: {preds.mean():.4f}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_iou_histogram(self, preds, gt):
        tp = (preds & gt).sum(axis=1).astype(float)
        fp = (preds & ~gt).sum(axis=1).astype(float)
        fn = (~preds & gt).sum(axis=1).astype(float)
        iou = tp / (tp + fp + fn + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(iou, bins=50, color='steelblue', edgecolor='white')
        ax.axvline(iou.mean(), color='red', linestyle='--', label=f'Mean={iou.mean():.3f}')
        ax.set_xlabel('Per-sample IoU')
        ax.set_ylabel('Count')
        ax.set_title(f'Per-sample IoU distribution [{self.split}]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.figures_dir / f"{self.split}_iou_histogram.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved IoU histogram: {out}")

    def _plot_prob_distribution(self, probs, gt):
        flat_probs = probs.ravel()
        flat_gt = gt.ravel()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(flat_probs[~flat_gt], bins=100, alpha=0.5, label='No contact', color='royalblue',
                density=True)
        ax.hist(flat_probs[flat_gt], bins=100, alpha=0.5, label='Contact', color='crimson',
                density=True)
        ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Probability distribution [{self.split}]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.figures_dir / f"{self.split}_prob_distribution.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved probability distribution: {out}")

    def _plot_roc_curve(self, probs, gt):
        from sklearn.metrics import roc_curve, auc

        flat_probs = probs.ravel()
        flat_gt = gt.ravel().astype(int)

        fpr, tpr, _ = roc_curve(flat_gt, flat_probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve [{self.split}]')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.figures_dir / f"{self.split}_roc_curve.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved ROC curve: {out}  (AUC = {roc_auc:.4f})")
        return roc_auc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate per-vertex contact head on DAMON")
    parser.add_argument("--config", type=str, default="train/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binary threshold for contact probability")
    parser.add_argument("--mode", type=str, default="smpl",
                        choices=["smpl", "mhr"],
                        help="Mesh topology for metrics: 'smpl' (6890 verts, default) "
                             "or 'mhr' (18439 verts)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluator = ContactEvaluator(
        args.config, args.checkpoint,
        split=args.split, device=args.device, mode=args.mode,
    )
    evaluator.evaluate(threshold=args.threshold)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
