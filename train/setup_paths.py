"""
Interactive script to help set up paths in config.yaml

This script guides you through updating the model checkpoint paths in the config file.
"""

import yaml
from pathlib import Path
import os


def find_files(root_dir, pattern):
    """Search for files matching pattern"""
    root = Path(root_dir).expanduser()
    if not root.exists():
        return []
    
    matches = list(root.rglob(pattern))
    return [str(p) for p in matches]


def prompt_path(prompt_msg, pattern=None, default=None):
    """Prompt user for a file path"""
    while True:
        if default:
            path = input(f"{prompt_msg} [{default}]: ").strip()
            if not path:
                path = default
        else:
            path = input(f"{prompt_msg}: ").strip()
        
        path = Path(path).expanduser()
        
        if path.exists():
            return str(path)
        else:
            print(f"  ✗ Path not found: {path}")
            retry = input("  Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None


def main():
    print("="*80)
    print("SAM-3D-Body Contact Head Training - Path Setup")
    print("="*80)
    print("\nThis script will help you configure paths in config.yaml")
    print()
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Model checkpoint
    print("\n1. SAM-3D-Body Model Checkpoint")
    print("-" * 80)
    print("This is the main model checkpoint file (usually model.ckpt or model.pth)")
    print("\nSearching for checkpoints in common locations...")
    
    search_dirs = [
        "~/",
        str(Path.home() / "Downloads"),
        str(Path(__file__).parent.parent.parent),
    ]
    
    found_ckpts = []
    for search_dir in search_dirs:
        found = find_files(search_dir, "*.ckpt")
        found_ckpts.extend(found)
        if len(found_ckpts) > 10:  # Limit search
            break
    
    if found_ckpts:
        print("\nFound checkpoints:")
        for i, ckpt in enumerate(found_ckpts[:10], 1):
            print(f"  {i}. {ckpt}")
        
        choice = input("\nSelect a checkpoint (1-10) or enter custom path: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= min(10, len(found_ckpts)):
            checkpoint_path = found_ckpts[int(choice) - 1]
        else:
            checkpoint_path = prompt_path("Enter checkpoint path")
    else:
        checkpoint_path = prompt_path("Enter checkpoint path")
    
    if checkpoint_path:
        config['MODEL']['CHECKPOINT_PATH'] = checkpoint_path
        print(f"  ✓ Set checkpoint path: {checkpoint_path}")
    else:
        print("  ⚠ Skipping checkpoint path")
    
    # 2. MHR model
    print("\n2. MHR Model File")
    print("-" * 80)
    print("This is the MHR body model file (usually mhr_model.pt)")
    
    # Try to find it relative to checkpoint
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path).parent
        possible_mhr_paths = [
            checkpoint_dir / "assets" / "mhr_model.pt",
            checkpoint_dir / "mhr_model.pt",
            checkpoint_dir.parent / "assets" / "mhr_model.pt",
        ]
        
        for mhr_path in possible_mhr_paths:
            if mhr_path.exists():
                print(f"\nFound MHR model at: {mhr_path}")
                use_it = input("Use this path? (y/n): ").strip().lower()
                if use_it == 'y':
                    config['MODEL']['MHR_MODEL_PATH'] = str(mhr_path)
                    print(f"  ✓ Set MHR model path: {mhr_path}")
                    break
        else:
            mhr_path = prompt_path("Enter MHR model path")
            if mhr_path:
                config['MODEL']['MHR_MODEL_PATH'] = mhr_path
                print(f"  ✓ Set MHR model path: {mhr_path}")
    else:
        mhr_path = prompt_path("Enter MHR model path")
        if mhr_path:
            config['MODEL']['MHR_MODEL_PATH'] = mhr_path
            print(f"  ✓ Set MHR model path: {mhr_path}")
    
    # 3. Dataset path
    print("\n3. ETH Dataset Path")
    print("-" * 80)
    current_dataset_path = config['DATASET']['DATA_PATH']
    print(f"Current path: {current_dataset_path}")
    
    if Path(current_dataset_path).exists():
        print("  ✓ Dataset path exists")
        change = input("Change dataset path? (y/n): ").strip().lower()
        if change == 'y':
            dataset_path = prompt_path("Enter dataset path", default=current_dataset_path)
            if dataset_path:
                config['DATASET']['DATA_PATH'] = dataset_path
    else:
        print("  ✗ Dataset path not found")
        dataset_path = prompt_path("Enter dataset path")
        if dataset_path:
            config['DATASET']['DATA_PATH'] = dataset_path
    
    # 4. Output directory
    print("\n4. Output Directory")
    print("-" * 80)
    current_output_dir = config['OUTPUT']['DIR']
    print(f"Current output directory: {current_output_dir}")
    
    change = input("Change output directory? (y/n): ").strip().lower()
    if change == 'y':
        output_dir = input("Enter output directory: ").strip()
        if output_dir:
            config['OUTPUT']['DIR'] = output_dir
    
    # Save config
    print("\n" + "="*80)
    print("Saving configuration...")
    
    # Backup original config
    backup_path = config_path.with_suffix('.yaml.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Saved config to: {config_path}")
    
    # Summary
    print("\n" + "="*80)
    print("Configuration Summary:")
    print("="*80)
    print(f"Model Checkpoint:  {config['MODEL']['CHECKPOINT_PATH']}")
    print(f"MHR Model:        {config['MODEL']['MHR_MODEL_PATH']}")
    print(f"Dataset Path:     {config['DATASET']['DATA_PATH']}")
    print(f"Output Directory: {config['OUTPUT']['DIR']}")
    print()
    print("Next steps:")
    print("  1. Test setup:  python train/test_setup.py")
    print("  2. Start training: python train/train_contact.py")
    print("="*80)


if __name__ == "__main__":
    main()
