#!/usr/bin/env python3
"""
Example script showing how to use the Damon MHR dataset with PyTorch DataLoader.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from damon_mhr import DamonMHRDataset


def main():
    print("=" * 80)
    print("Damon MHR Dataset Example")
    print("=" * 80)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = DamonMHRDataset(
        npz_path='../../datasets/daimon_mhr_contact/hot_dca_trainval_mhr.npz',
        data_root='/media/rikhat/Hard/datasets/DECO/',
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Test loading a batch
    print("\nLoading first batch...")
    for batch_idx, ((images, bboxes, cam_ks), contact_labels) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Bboxes shape: {bboxes.shape}")
        print(f"  Bboxes dtype: {bboxes.dtype}")
        print(f"  Camera K shape: {cam_ks.shape}")
        print(f"  Contact labels shape: {contact_labels.shape}")
        print(f"  Contact labels dtype: {contact_labels.dtype}")
        print(f"  Contact labels unique values: {torch.unique(contact_labels)}")
        
        # Per-sample statistics
        for i in range(len(images)):
            num_contacts = contact_labels[i].sum().item()
            bbox = bboxes[i]
            focal = cam_ks[i, 0, 0]
            print(f"    Sample {i}:")
            print(f"      Contacts: {num_contacts} ({num_contacts/18439*100:.2f}%)")
            print(f"      BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            print(f"      Focal length: {focal:.1f}")
        
        # Only process first batch for demo
        break
    
    print("\n" + "=" * 80)
    print("Complete! Dataset is ready for training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
