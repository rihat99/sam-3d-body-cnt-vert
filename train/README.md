# Contact Head Training for SAM-3D-Body

This directory contains scripts and configuration for training the contact prediction head on the ETH climbing dataset.

## Setup

### 1. Prerequisites

- SAM-3D-Body model checkpoint (`.ckpt` file)
- MHR model file (`mhr_model.pt`)
- ETH dataset (should be located at `/home/rikhat/climbing/climb_forces/data/eth`)

### 2. Configuration

Edit `config.yaml` and update the following paths:

```yaml
MODEL:
  CHECKPOINT_PATH: "/path/to/sam-3d-body/model.ckpt"  # UPDATE THIS
  MHR_MODEL_PATH: "/path/to/sam-3d-body/assets/mhr_model.pt"  # UPDATE THIS
```

### 3. Adjust Settings for Your Hardware

The default configuration is set for a 4090 GPU with 24GB memory:

```yaml
TRAIN:
  BATCH_SIZE: 16  # Adjust based on your GPU memory
  VAL_BATCH_SIZE: 8
```

If you encounter OOM errors, reduce the batch size.

## Training

### Step 1: Test Setup

Before starting full training, verify everything is working:

```bash
python train/test_setup.py --config train/config.yaml
```

This will:
- Load the model and verify contact head parameters are trainable
- Load a small batch of data
- Run a single forward/backward pass
- Verify gradients are computed correctly

### Step 2: Start Training

Once the setup test passes, start training:

```bash
python train/train_contact.py --config train/config.yaml
```

Or with custom config:

```bash
python train/train_contact.py --config path/to/your/config.yaml
```

### Training Progress

Training outputs will be saved to:
- Checkpoints: `train/output/contact_head_eth/`
- TensorBoard logs: `train/output/contact_head_eth/tensorboard/`

Monitor training with TensorBoard:

```bash
tensorboard --logdir train/output/contact_head_eth/tensorboard/
```

## Configuration Options

### Dataset Settings

```yaml
DATASET:
  TRAIN_FOLDERS: ["1"]  # Start with folder 1 for testing
  SIDES: ["left"]       # Use left camera only for initial testing
  TRAIN_VAL_SPLIT: 0.8  # 80% train, 20% validation
```

To use more data, update:
```yaml
DATASET:
  TRAIN_FOLDERS: ["1", "2"]  # Use both folders
  SIDES: ["left", "right"]   # Use both cameras
```

### Training Settings

```yaml
TRAIN:
  EPOCHS: 50           # Number of training epochs
  BATCH_SIZE: 16       # Batch size for training
  LR: 1e-4            # Learning rate
  LR_SCHEDULER: "cosine"  # Learning rate schedule
  SAVE_FREQ: 5        # Save checkpoint every N epochs
  VAL_FREQ: 1         # Validate every N epochs
```

### Loss Weights

Adjust the weights for hand vs foot contact:

```yaml
TRAIN:
  LOSS_WEIGHTS:
    hand: 1.0  # Weight for hand contact loss
    foot: 1.0  # Weight for foot contact loss
```

## Output Files

After training, you'll find:

- `best_model.pth` - Best model based on validation loss
- `final_model.pth` - Model after final epoch
- `checkpoint_epoch_N.pth` - Periodic checkpoints
- `tensorboard/` - Training logs

## Using the Trained Model

To use the trained contact head:

```python
import torch
from sam_3d_body.build_models import load_sam_3d_body

# Load base model
model, cfg = load_sam_3d_body(
    checkpoint_path="path/to/base/model.ckpt",
    mhr_path="path/to/mhr_model.pt"
)

# Load trained contact head weights
checkpoint = torch.load("train/output/contact_head_eth/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

model.eval()
model = model.to("cuda")

# Use for inference
# ... (prepare your batch)
output = model.forward_step(batch, decoder_type="body")
contact_probs = output["contact"]["contact_probs"]  # [B, 4] probabilities
```

## Troubleshooting

### "No contact parameters found"

Make sure your model config has:
```yaml
DECODER:
  DO_CONTACT_TOKENS: true
```

### Out of Memory (OOM)

Reduce batch size in config:
```yaml
TRAIN:
  BATCH_SIZE: 8  # or smaller
```

### "Dataset not found"

Check the data path in config:
```yaml
DATASET:
  DATA_PATH: "/home/rikhat/climbing/climb_forces/data/eth"
```

### Slow training

- Increase `NUM_WORKERS` for faster data loading
- Enable mixed precision training (already enabled by default)
- Use larger batch size if you have GPU memory

## Notes

- The model is trained with **only the contact head trainable**, all other parameters are frozen
- Contact predictions are: [left_hand, right_hand, left_foot, right_foot]
- For folder "1", only hand contacts have labels (feet are always False)
- For folder "2", both hand and foot contacts have labels
- Camera parameters are the same for all images in the ETH dataset (already configured)

## Camera Parameters (ETH Dataset)

The camera intrinsics are already configured in `config.yaml`:

```yaml
CAMERA:
  fx: 1419.27383
  fy: 1419.21058
  cx: 966.397807
  cy: 498.565941
```

These are the same for all images in the ETH dataset, as shown in `test.ipynb`.
