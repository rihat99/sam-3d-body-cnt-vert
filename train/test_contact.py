"""
Test-split evaluation for the per-vertex Contact Head.

A thin wrapper around evaluate.py that defaults to --split test.

Usage:
    python train/test_contact.py \
        --config train/config.yaml \
        --checkpoint train/output/contact_vert_20260222_123456/best_model.pth

All figures are saved to train/figures/ with a 'test_' prefix.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import ContactEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-vertex contact head on the DAMON test split"
    )
    parser.add_argument("--config", type=str, default="train/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (e.g., .../best_model.pth)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binary threshold for contact probability (default: 0.5)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluator = ContactEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split="test",
        device=args.device,
    )
    evaluator.evaluate(threshold=args.threshold)
    print("\nTest evaluation complete.")


if __name__ == "__main__":
    main()
