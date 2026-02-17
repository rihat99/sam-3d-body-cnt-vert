"""
Test script to verify training setup before running full training

This script:
1. Loads the model and config
2. Loads a small batch of data
3. Runs a single forward/backward pass
4. Verifies gradients are computed correctly
"""

import os
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset" / "eth"))

os.environ["MOMENTUM_ENABLED"] = "1"

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.utils.config import get_config
from dataset import ETHContactDataset
from torch.utils.data import DataLoader
from dataset_utils import prepare_training_batch


def test_setup(config_path="train/config.yaml"):
    """Test the training setup"""
    
    print("="*80)
    print("TESTING TRAINING SETUP")
    print("="*80)
    
    # 1. Load config
    print("\n1. Loading config...")
    try:
        cfg = get_config(config_path)
        print("   ✓ Config loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading config: {e}")
        return False
    
    # 2. Load model
    print("\n2. Loading model...")
    try:
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=cfg.MODEL.CHECKPOINT_PATH,
            device="cuda",
            mhr_path=cfg.MODEL.MHR_MODEL_PATH
        )
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        print(f"   → Make sure to update CHECKPOINT_PATH and MHR_MODEL_PATH in {config_path}")
        return False
    
    # 3. Freeze model except contact head
    print("\n3. Freezing model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    contact_params = []
    for name, param in model.named_parameters():
        if "contact" in name.lower():
            param.requires_grad = True
            contact_params.append(name)
    
    if len(contact_params) == 0:
        print("   ✗ No contact parameters found!")
        print("   → Make sure DO_CONTACT_TOKENS is enabled in model config")
        return False
    
    print(f"   ✓ Found {len(contact_params)} contact parameters:")
    for name in contact_params:
        print(f"     - {name}")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Trainable: {trainable:,} / {total:,} parameters")
    
    # 4. Load dataset
    print("\n4. Loading dataset...")
    try:
        dataset = ETHContactDataset(
            data_path=cfg.DATASET.DATA_PATH,
            folders=cfg.DATASET.TRAIN_FOLDERS,
            sides=cfg.DATASET.SIDES,
            contact_threshold=cfg.DATASET.CONTACT_THRESHOLD,
            rebuild_cache=False
        )
        print(f"   ✓ Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return False
    
    # 5. Create data loader
    print("\n5. Creating data loader...")
    loader = DataLoader(
        dataset,
        batch_size=2,  # Small batch for testing
        shuffle=False,
        num_workers=0
    )
    
    # 6. Test forward pass
    print("\n6. Testing forward pass...")
    try:
        (images, bboxes), contacts = next(iter(loader))
        print(f"   ✓ Batch loaded: {len(images)} samples")
        print(f"     - Image shapes: {[img.shape for img in images]}")
        print(f"     - Contact shape: {contacts.shape}")
        
        # Prepare batch using simple utility
        cam_params = {
            'fx': cfg.CAMERA.fx,
            'fy': cfg.CAMERA.fy,
            'cx': cfg.CAMERA.cx,
            'cy': cfg.CAMERA.cy
        }
        
        # Convert lists to match batch format
        images_list = [images[i] for i in range(len(images))]
        bboxes_list = [bboxes[i] for i in range(len(bboxes))]
        
        batch = prepare_training_batch(
            images_list,
            bboxes_list,
            cam_params,
            target_size=(896, 896),  # Maximum resolution with DINOv3
            device='cuda'
        )
        contacts = contacts.to("cuda")
        
        print("   ✓ Batch prepared for model")
        
        # Forward pass
        model.train()
        
        # Initialize batch size info (required by model)
        model._initialize_batch(batch)
        
        output = model.forward_step(batch, decoder_type="body")
        
        if output["contact"] is None:
            print("   ✗ Model did not return contact predictions!")
            print("   → Check that DO_CONTACT_TOKENS is enabled in model config")
            return False
        
        contact_logits = output["contact"]["contact_logits"]
        print(f"   ✓ Forward pass successful, contact_logits shape: {contact_logits.shape}")
        
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. Test backward pass
    print("\n7. Testing backward pass...")
    try:
        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            contact_logits,
            contacts.float()
        )
        print(f"   ✓ Loss computed: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                print(f"   ✓ Gradient for {name}: norm = {grad_norm:.6f}")
        
        if not has_grad:
            print("   ✗ No gradients computed!")
            return False
        
        print("   ✓ Backward pass successful")
        
    except Exception as e:
        print(f"   ✗ Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 8. Test optimizer step
    print("\n8. Testing optimizer step...")
    try:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )
        optimizer.step()
        print("   ✓ Optimizer step successful")
    except Exception as e:
        print(f"   ✗ Error in optimizer step: {e}")
        return False
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nYou can now run training with:")
    print(f"  python train/train_contact.py --config {config_path}")
    print()
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    success = test_setup(args.config)
    sys.exit(0 if success else 1)
