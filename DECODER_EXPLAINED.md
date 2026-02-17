# SAM-3D-Body Decoder Architecture Explained

## Overview

The SAM-3D-Body decoder is a **promptable transformer decoder** that uses cross-attention to estimate 3D human body pose and shape. The key innovation is that it performs **intermediate predictions after each decoder layer**, calling prediction heads at every layer to progressively refine the pose estimate.

## Architecture Components

### 1. PromptableDecoder (`promptable_decoder.py`)

The core decoder is a stack of Transformer decoder layers with special capabilities:
- **Type**: Cross-attention based Transformer decoder
- **Depth**: Multiple layers (typically 4-8 layers)
- **Special Feature**: Can output intermediate predictions at each layer

**Key Configuration:**
- `dims`: 1024 (decoder token dimension)
- `context_dims`: 1280 (image feature dimension from backbone)
- `depth`: Number of decoder layers
- `num_heads`: 8 (multi-head attention)
- `do_interm_preds`: True (enables intermediate predictions)
- `do_keypoint_tokens`: True (enables keypoint query tokens)
- `keypoint_token_update`: True (enables token updates between layers)

---

## Decoder Inputs

### 1. Token Embeddings (`token_embedding`)
**Shape**: `[B, N_tokens, 1024]`

The decoder receives multiple types of tokens concatenated together:

#### a) **Pose Token** (1 token)
- **Purpose**: Main token that predicts body pose and shape parameters
- **Initialization**: From learned embedding `init_pose` (407D) passed through linear layer
- **Contains**: Initial estimates for:
  - Global rotation (6D)
  - Body pose (260D continuous)
  - Shape parameters (45D)
  - Scale parameters (28D)
  - Hand parameters (108D, 54 per hand)
  - Face parameters (72D)
  - Camera parameters (3D: scale, tx, ty)

#### b) **Previous Estimate Token** (1 token, optional)
- **Purpose**: Provides previous iteration's prediction for refinement
- **Shape**: Same as pose token (407D)
- Used in iterative prompting scenarios

#### c) **Prompt Tokens** (1-70 tokens)
- **Purpose**: User-provided keypoint prompts to guide predictions
- **Encoding**: From `PromptEncoder`
  - 2D position encoding using Fourier features
  - Per-joint learned embeddings (70 different embeddings)
  - Special embeddings for invalid (-2) and incorrect (-1) points
- **Labels**:
  - `label >= 0`: Valid keypoint prompt for specific joint
  - `label = -1`: Incorrect/negative prompt
  - `label = -2`: Invalid/dummy prompt

#### d) **Keypoint Query Tokens** (70 tokens)
- **Purpose**: Learned queries for 70 body keypoints (2D projection)
- **Initialization**: From `keypoint_embedding` (learnable)
- **Updated**: After each layer with predicted 2D keypoint positions

#### e) **3D Keypoint Query Tokens** (70 tokens, optional)
- **Purpose**: Learned queries for 70 body keypoints (3D space)
- **Initialization**: From `keypoint3d_embedding` (learnable)
- **Updated**: After each layer with predicted 3D keypoint positions

**Total Token Count**: `1 + 1 + N_prompts + 70 + 70 = 142 + N_prompts` tokens

### 2. Image Embeddings (`image_embedding`)
**Shape**: `[B, H*W, 1280]` (flattened from `[B, 1280, H, W]`)

- **Source**: From backbone encoder (ViT/DinoV2)
- **Resolution**: Typically 16×16 spatial resolution
- **Enhanced with**: Ray conditioning (camera intrinsics encoded per pixel)

### 3. Token Augment (`token_augment`)
**Shape**: `[B, N_tokens, 1024]`

Position encodings for tokens:
- **Pose token**: Zero
- **Previous estimate**: Copy of previous embedding
- **Prompt tokens**: Copy of prompt embeddings
- **Keypoint tokens**: Updated with 2D position encoding from predictions
- **3D Keypoint tokens**: Updated with 3D position encoding from predictions

### 4. Image Augment (`image_augment`)
**Shape**: `[B, 1280, H, W]`

Dense positional encoding for image features:
- Generated using random Fourier features
- Provides spatial information to the decoder

---

## The Decoder Forward Pass: Layer-by-Layer Processing

### Standard Transformer Decoder Layer (`TransformerDecoderLayer`)

Each layer performs:

#### Step 1: Self-Attention on Tokens
```
tokens' = tokens + SelfAttention(LayerNorm(tokens) + PE)
```
- All tokens attend to each other
- Position encodings added if `repeat_pe=True`

#### Step 2: Cross-Attention to Image Features
```
tokens' = tokens' + CrossAttention(
    query=LayerNorm(tokens') + PE,
    key=LayerNorm(image_features) + PE,
    value=LayerNorm(image_features)
)
```
- Tokens attend to image features
- This is where visual information flows into predictions

#### Step 3: Feed-Forward Network (FFN)
```
tokens' = tokens' + FFN(LayerNorm(tokens'))
```
- Two-layer MLP with GELU activation
- Can use SwiGLU activation for better performance

---

## **🔥 THE TRICKY PART: Intermediate Predictions & Token Updates**

This is what makes the decoder special! After each layer (except the last), the decoder:

### 1. Calls Prediction Heads (`token_to_pose_output_fn`)

**Input**: Normalized first token (pose token) from current layer  
**Process**:
```python
def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
    pose_token = tokens[:, 0]  # Extract pose token
    
    # Run through MHR head
    pose_output = head_pose(pose_token, init_pose)
    # Output: body pose, shape, scale, hand, face params
    
    # Run through camera head  
    pred_cam = head_camera(pose_token, init_camera)
    # Output: camera translation (scale, tx, ty)
    
    # Project 3D keypoints to 2D
    pose_output = camera_project(pose_output, batch)
    
    return pose_output
```

**Outputs from heads:**
- **MHR Head**: Predicts 404D pose vector
  - Global rotation: 6D → converted to Euler angles
  - Body pose: 260D → converted to 133 joint angles
  - Shape: 45D PCA coefficients
  - Scale: 28D scale parameters
  - Hand: 108D (54 per hand, PCA)
  - Face: 72D expression parameters
  - Forwards through MHR model → 3D keypoints + mesh vertices

- **Camera Head**: Predicts 3D camera parameters
  - Scale factor, tx, ty → converted to camera translation
  
- **Camera Projection**: Projects 3D keypoints to 2D using perspective projection

### 2. Updates Keypoint Tokens (`keypoint_token_update_fn`)

After getting predictions, the decoder updates the keypoint query tokens:

#### 2D Keypoint Token Update:
```python
# Get predicted 2D keypoints (in crop space, -0.5 to 0.5)
pred_kps_2d = pose_output['pred_keypoints_2d_cropped']

# Update token position encodings
token_augment[keypoint_tokens] = positional_encoding(pred_kps_2d)

# Sample image features at predicted keypoint locations
kp_features = grid_sample(image_embeddings, pred_kps_2d)

# Add sampled features to token embeddings
token_embeddings[keypoint_tokens] += linear(kp_features)
```

**Why this is powerful:**
- Dynamically samples image features at predicted locations
- Creates feedback loop: predictions → better features → better predictions
- Invalid predictions (outside image) are masked out

#### 3D Keypoint Token Update:
```python
# Get predicted 3D keypoints (pelvis-normalized)
pred_kps_3d = pose_output['pred_keypoints_3d']
pred_kps_3d = pred_kps_3d - pelvis_center

# Update token position encodings with 3D positions
token_augment[keypoint3d_tokens] = positional_encoding_3d(pred_kps_3d)
```

---

## Complete Decoder Loop

```python
all_pose_outputs = []

for layer_idx, layer in enumerate(decoder_layers):
    # 1. Standard transformer layer
    tokens, image_features = layer(
        tokens, 
        image_features, 
        token_augment, 
        image_augment
    )
    
    # 2. 🔥 INTERMEDIATE PREDICTION (not on last layer)
    if layer_idx < num_layers - 1:
        # Run prediction heads
        pose_output = token_to_pose_output_fn(
            LayerNorm(tokens),
            prev_pose_output,
            layer_idx
        )
        all_pose_outputs.append(pose_output)
        
        # 3. 🔥 UPDATE KEYPOINT TOKENS
        tokens, token_augment = keypoint_token_update_fn(
            tokens, 
            token_augment, 
            pose_output, 
            layer_idx
        )

# 4. Final prediction after last layer
final_tokens = LayerNorm(tokens)
final_pose_output = token_to_pose_output_fn(
    final_tokens,
    all_pose_outputs[-1],
    layer_idx
)
all_pose_outputs.append(final_pose_output)

return final_tokens, all_pose_outputs
```

---

## Prediction Heads Details

### MHR Head (`mhr_head.py`)

**Input**: Pose token `[B, 1024]`  
**Architecture**: FFN (1024 → 1024/8 → 404)

**Output Dimensions** (404D total):
1. **Global Rotation**: 6D (6D rotation representation)
2. **Body Pose**: 260D (continuous representation)
   - Converted to 133 joint angles (3 DOF each, excluding hands/jaw)
3. **Shape**: 45D (PCA coefficients for body shape)
4. **Scale**: 28D (scale parameters for body parts)
5. **Hand Pose**: 108D (54D left + 54D right, PCA)
6. **Face Expression**: 72D (facial expression parameters)

**MHR Forward Pass:**
1. Converts 6D rotation → 3×3 rotation matrix → Euler angles
2. Converts continuous body pose → joint Euler angles
3. Runs through MHR (Momentum Human Representation) model
4. Outputs:
   - 3D mesh vertices (18,439 vertices)
   - 3D keypoints (70 keypoints from 308 total)
   - Joint coordinates (127 joints)
   - Joint rotations

### Camera Head (`camera_head.py`)

**Input**: Pose token `[B, 1024]`  
**Architecture**: FFN (1024 → 1024/8 → 3)

**Output**: 3D camera parameters `[B, 3]`
- `s`: Scale factor
- `tx`: Translation x
- `ty`: Translation y

**Perspective Projection:**
```python
# Convert to camera translation
tz = 2 * focal_length / (bbox_size * s)
tx_cam = tx + 2 * (bbox_center_x - img_center_x) / (bbox_size * s)
ty_cam = ty + 2 * (bbox_center_y - img_center_y) / (bbox_size * s)

# Project 3D points
keypoints_2d = project(keypoints_3d + camera_translation, intrinsics)
```

---

## Prompt Encoder Details

### Keypoint Prompt Encoding

**Input**: `[B, N_prompts, 3]` where last dim is `(x, y, label)`
- `x, y`: Normalized coordinates [0, 1]
- `label`: Joint index or special value

**Process**:
1. **Position Encoding**: 
   - Uses random Fourier features: `sin/cos(2π * coords @ random_matrix)`
   - Dimension: 1280 (640 features × 2 for sin/cos)

2. **Learned Embeddings** (added based on label):
   - 70 joint-specific embeddings (`point_embeddings[label]`)
   - `invalid_point_embed` (label = -2)
   - `not_a_point_embed` (label = -1)

3. **Linear Projection**: 1280 → 1024 to match decoder dim

### Keypoint Sampling Strategy

**KeypointSamplerV1** determines which keypoint to prompt:
- **80%** sample from key body joints (torso, hips, shoulders)
- **80%** sample worst prediction (largest error)
- **10%** use dummy prompt (no information)
- **0%** negative prompts (incorrect predictions)

---

## Final Outputs

### From Decoder
**Shape**: `[B, N_tokens, 1024]` + `List[Dict]` with L predictions + `Dict` contact output

### From Prediction Heads (at each layer)

**Pose Output Dictionary:**
```python
{
    # Raw predictions
    'pred_pose_raw': [B, 266],  # 6D rot + 260D body pose
    'global_rot': [B, 3],       # Euler angles
    'body_pose': [B, 133],      # Joint angles
    'shape': [B, 45],           # Shape PCA
    'scale': [B, 28],           # Scale params
    'hand': [B, 108],           # Hand PCA
    'face': [B, 72],            # Expression
    
    # 3D outputs
    'pred_keypoints_3d': [B, 70, 3],      # 3D keypoints
    'pred_vertices': [B, 18439, 3],       # Mesh vertices
    'pred_joint_coords': [B, 127, 3],     # Joint positions
    'joint_global_rots': [B, 127, 3, 3],  # Joint rotations
    
    # Camera & 2D projection
    'pred_cam': [B, 3],                   # Camera params
    'pred_cam_t': [B, 3],                 # Camera translation
    'focal_length': [B],                  # Focal length
    'pred_keypoints_2d': [B, 70, 2],      # Projected 2D kps
    'pred_keypoints_2d_depth': [B, 70],   # Depth values
    'pred_keypoints_2d_cropped': [B, 70, 2],  # In crop space
}
```

**Contact Output Dictionary (if `DO_CONTACT_TOKENS=True`):**
```python
{
    'contact_logits': [B, 4],  # Raw logits: [left_foot, right_foot, left_hand, right_hand]
    'contact_probs': [B, 4],   # Sigmoid probabilities (0-1)
}
```

**Full Model Output Dictionary:**
```python
{
    'mhr': pose_output,           # Body decoder pose output
    'mhr_hand': pose_output_hand, # Hand decoder pose output (if using hand decoder)
    'contact': contact_output,    # Contact predictions from body decoder
    'contact_hand': contact_output_hand,  # Contact predictions from hand decoder
    'condition_info': [B, 3],     # CLIFF condition info
    'image_embeddings': [B, 1280, H, W],  # Backbone features
}
```

---

## Key Dimensions Summary

| Component | Dimension | Description |
|-----------|-----------|-------------|
| **Decoder Token Dim** | 1024 | Internal decoder representation |
| **Image Feature Dim** | 1280 | From backbone (DinoV2/ViT) |
| **Pose Parameter Dim** | 404 | Total pose/shape output |
| - Global Rotation | 6 | 6D rotation representation |
| - Body Pose | 260 | Continuous representation |
| - Shape | 45 | PCA coefficients |
| - Scale | 28 | Body part scales |
| - Hand | 108 | Left (54) + Right (54) PCA |
| - Face | 72 | Expression parameters |
| **Camera Params** | 3 | Scale, tx, ty |
| **Contact Params** | 4 | Binary contact per body part |
| **Prompt Encoding** | 1280 → 1024 | Fourier features → decoder dim |
| **Num Joints** | 127 | Total skeletal joints |
| **Num Keypoints** | 70 | Sapiens keypoint set |
| **Num Vertices** | 18,439 | Mesh vertices |
| **Image Resolution** | 16×16 or 32×32 | Spatial feature resolution |
| **Num Contact Tokens** | 4 | Left/Right Foot, Left/Right Hand |

---

## Auxiliary Predictions

### Intermediate Supervision
- **Purpose**: Supervise predictions at each decoder layer
- **Count**: L predictions (one per layer)
- **Training**: Can compute loss at each layer for better gradients
- **Inference**: Only final prediction typically used

### Hand Detection Tokens (Optional)
If `DO_HAND_DETECT_TOKENS=True`:
- Additional 2 tokens for left/right hand bounding box prediction
- **Hand Box Output**: `[B, 2, 4]` - (x, y, w, h) for each hand
- **Hand Logits**: `[B, 2, 2]` - presence classification

### Contact Tokens (Optional)
If `DO_CONTACT_TOKENS=True`:
- Additional 4 learnable query tokens for contact prediction
- One token per body part: Left Foot, Right Foot, Left Hand, Right Hand

**Contact Head (`contact_head.py`):**

**Input**: 4 contact tokens `[B, 4, 1024]`  
**Architecture**: FFN (1024 → 256 → 1 per token)

**Output Dictionary:**
```python
{
    'contact_logits': [B, 4],  # Raw logits for each contact
    'contact_probs': [B, 4],   # Sigmoid probabilities
}
```

**Contact Index Mapping:**
| Index | Body Part |
|-------|-----------|
| 0 | Left Foot |
| 1 | Right Foot |
| 2 | Left Hand |
| 3 | Right Hand |

**Usage:**
- Contact predictions are added to final output dict under `"contact"` key
- For hand decoder, available under `"contact_hand"` key
- Not used for intermediate predictions (only final layer output)

---

## Why This Design Works

### 1. **Progressive Refinement**
- Each layer makes a prediction
- Subsequent layers can see and correct previous mistakes
- Similar to cascaded refinement in detection

### 2. **Feature-Prediction Alignment**
- Keypoint tokens sample features at predicted locations
- Creates tight coupling between predictions and visual features
- Handles occlusions naturally (invalid predictions masked)

### 3. **Multi-Modal Prompting**
- Supports keypoint prompts for user guidance
- Supports mask prompts (via mask encoder)
- Enables interactive refinement

### 4. **Hierarchical Tokens**
- Pose token: Global body representation
- Keypoint tokens: Local joint-specific features
- 3D tokens: Depth-aware representations

---

## Training Considerations

### Losses Applied
- **3D Keypoint Loss**: On predicted vs GT 3D keypoints
- **2D Reprojection Loss**: On projected 2D keypoints
- **Pose Parameter Loss**: On rotation matrices
- **Vertex Loss**: On mesh vertices (optional)
- **Intermediate Losses**: Applied at each decoder layer

### Auxiliary Losses
- **Hand Detection Loss**: If using hand detection tokens
- **Mask Loss**: If using mask prompts

---

## Inference Flow

```
1. Image → Backbone → Image Features [B, 1280, 16, 16]
2. Initialize tokens: pose + prompts + keypoint queries
3. For each decoder layer:
   a. Self-attention on tokens
   b. Cross-attention to image
   c. FFN
   d. 🔥 Run prediction heads → get 3D pose
   e. 🔥 Update keypoint tokens with predictions
4. Final output: refined 3D mesh + keypoints + camera
```

---

## Comparison to Standard Transformers

| Standard Decoder | SAM-3D-Body Decoder |
|------------------|---------------------|
| Single output at end | Outputs at every layer |
| Static query tokens | Dynamic query tokens updated with predictions |
| No mid-layer supervision | Intermediate supervision at each layer |
| Token-to-token attention | Token-image-prediction feedback loop |

This architecture enables the model to iteratively refine its predictions while maintaining spatial alignment with the input image, leading to more accurate and robust 3D pose estimation.
