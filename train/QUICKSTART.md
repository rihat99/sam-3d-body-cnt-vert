# Quick Start Guide - Contact Head Training

## Step 1: Set Up Paths

First, you need to configure the paths to your SAM-3D-Body checkpoint and MHR model.

### Option A: Interactive Setup (Recommended)

```bash
cd /home/rikhat/human_global_motion/sam-3d-body
python train/setup_paths.py
```

This script will guide you through setting up all necessary paths.

### Option B: Manual Setup

Edit `train/config.yaml` and update:

```yaml
MODEL:
  CHECKPOINT_PATH: "/path/to/your/sam-3d-body/model.ckpt"
  MHR_MODEL_PATH: "/path/to/your/sam-3d-body/assets/mhr_model.pt"
```

## Step 2: Test Setup

Before training, verify everything works:

```bash
python train/test_setup.py
```

This should output:
- ✓ Config loaded successfully
- ✓ Model loaded successfully
- ✓ Found contact parameters
- ✓ Dataset loaded
- ✓ Forward pass successful
- ✓ Backward pass successful
- ✓ Optimizer step successful

If all tests pass, you're ready to train!

## Step 3: Start Training

### Quick Test (Small Subset)

For initial testing with a small subset:

```bash
python train/train_contact.py
```

This uses the default config which is set for a small subset (folder "1", left camera only).

### Full Training

Once the quick test works, update `train/config.yaml`:

```yaml
DATASET:
  TRAIN_FOLDERS: ["1", "2"]  # Use both folders
  SIDES: ["left", "right"]   # Use both cameras
  
TRAIN:
  EPOCHS: 50                  # More epochs
  BATCH_SIZE: 16              # Larger batch size (adjust for your GPU)
```

Then start training:

```bash
python train/train_contact.py
```

## Step 4: Monitor Training

### TensorBoard

In a separate terminal:

```bash
tensorboard --logdir train/output/contact_head_eth/tensorboard/
```

Then open http://localhost:6006 in your browser.

### Training Output

Training outputs are saved to:
```
train/output/contact_head_eth/
├── best_model.pth              # Best model (lowest validation loss)
├── final_model.pth             # Final model after all epochs
├── checkpoint_epoch_N.pth      # Periodic checkpoints
├── config.yaml                 # Copy of config used for this run
└── tensorboard/                # TensorBoard logs
```

## Step 5: Evaluate

After training, evaluate your model:

```bash
python train/evaluate.py --checkpoint train/output/contact_head_eth/best_model.pth
```

This will output:
- Per-contact accuracy, precision, recall, F1 score
- Confusion matrices (saved as `confusion_matrices.png`)
- Probability distributions (saved as `probability_distributions.png`)

## Step 6: Run Inference

Try inference on sample images:

```bash
python train/inference_demo.py \
    --checkpoint train/output/contact_head_eth/best_model.pth \
    --num_samples 10 \
    --output_dir train/inference_samples
```

This saves visualizations showing:
- Input image with bounding box
- Ground truth vs predicted contacts
- Prediction probabilities

## Configuration Tips

### For Quick Testing (Small Subset)
```yaml
MODEL:
  IMAGE_SIZE: [512, 384]      # Maximum resolution
  BACKBONE:
    TYPE: vit_hmr_512_384     # High-res backbone
DATASET:
  TRAIN_FOLDERS: ["1"]        # Just one folder
  SIDES: ["left"]             # Just one camera
TRAIN:
  EPOCHS: 10                  # Fewer epochs
  BATCH_SIZE: 6               # Smaller batch for high-res
```

### For Full Training (All Data)
```yaml
MODEL:
  IMAGE_SIZE: [512, 384]      # Maximum resolution
  BACKBONE:
    TYPE: vit_hmr_512_384     # High-res backbone
DATASET:
  TRAIN_FOLDERS: ["1", "2"]   # All folders
  SIDES: ["left", "right"]    # All cameras
TRAIN:
  EPOCHS: 50                  # More epochs
  BATCH_SIZE: 8               # Adjust for GPU memory
```

### Adjusting Batch Size for GPU Memory

If you get OOM (Out of Memory) errors:
- Reduce `BATCH_SIZE` (try 6, 4, 2)
- Switch to standard backbone `TYPE: vit_hmr` (uses less memory)
- Reduce image resolution to [256, 192]
- Reduce `VAL_BATCH_SIZE` proportionally

If training is stable and you have memory to spare:
- Increase `BATCH_SIZE` (try 10, 12)

For reference, 4090 with 24GB at 896x896 with DINOv3 ViT-H/16+:
- Batch size 4 comfortably (fp32 training)
- Batch size 6 near limit
- Mixed precision disabled due to MHR model sparse operations
- DINOv3 at 896×896 captures maximum detail for contact detection
- Same resolution and backbone as the pretrained checkpoint

## Python Path

The scripts use the Python at:
```
/home/rikhat/miniconda3/envs/mhr/bin/python
```

If you need to use a different Python environment, activate it before running the scripts.

## Troubleshooting

### "No contact parameters found"

Your model config needs:
```yaml
DECODER:
  DO_CONTACT_TOKENS: true
```

Check the `model_config.yaml` that comes with your checkpoint.

### "CUDA out of memory"

Reduce batch size in `train/config.yaml`:
```yaml
TRAIN:
  BATCH_SIZE: 4  # or smaller
```

### "Dataset not found"

Check the dataset path:
```yaml
DATASET:
  DATA_PATH: "/home/rikhat/climbing/climb_forces/data/eth"
```

### Training is slow

- Increase `NUM_WORKERS` for faster data loading (try 4-8)
- Make sure `USE_FP16: true` for mixed precision training
- Use larger batch size if you have GPU memory

## Expected Training Time

With 4090 GPU:
- Small subset (folder 1, left only): ~5-10 minutes per epoch
- Full dataset (both folders, both cameras): ~15-30 minutes per epoch

For 20 epochs on small subset: ~2-3 hours total
For 50 epochs on full dataset: ~12-24 hours total

## Next Steps

After successful training:

1. **Evaluate on validation set**: See Step 5
2. **Visualize predictions**: See Step 6
3. **Integrate into your pipeline**: See `train/README.md` for how to use the trained model

## Getting Help

If you encounter issues:

1. Run `python train/test_setup.py` to verify setup
2. Check `train/README.md` for detailed documentation
3. Review config file: `train/config.yaml`
4. Check TensorBoard logs for training diagnostics

## Files Created

```
sam-3d-body/train/
├── config.yaml              # Main configuration file
├── train_contact.py         # Training script
├── test_setup.py           # Setup verification script
├── setup_paths.py          # Interactive path configuration
├── evaluate.py             # Evaluation script
├── inference_demo.py       # Inference demonstration
├── README.md               # Detailed documentation
└── QUICKSTART.md          # This file
```
