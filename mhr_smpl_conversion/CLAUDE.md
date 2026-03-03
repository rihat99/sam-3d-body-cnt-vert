# mhr_smpl_conversion — Developer Notes for Claude

This document records the changes made to this module and provides usage examples
and guidance for future modifications.

---

## What this module does

Bidirectional conversion between the **SMPL** body model (6 890 vertices) and the
**MHR** body model (18 439 vertices).  Two kinds of data can be converted:

| Data | Direction | Method |
|------|-----------|--------|
| Pose / shape parameters | SMPL ↔ MHR | Optimization-based fitting |
| Per-vertex contact labels | SMPL ↔ MHR | Barycentric interpolation |
| Raw vertex positions | SMPL ↔ MHR | Barycentric interpolation |

Contact-label conversion is purely algebraic (no optimisation loop) and therefore
very fast.

---

## Key files

| File | Role |
|------|------|
| `conversion.py` | Main `Conversion` class — entry point for all conversions |
| `pytorch_fitting.py` | PyTorch optimisation back-end (GPU) |
| `pymomentum_fitting.py` | PyMomentum optimisation back-end (CPU only) |
| `utils.py` | `ConversionResult`, helper loaders, `BetterHumanSMPL` forward utils |
| `assets/smpl2mhr_mapping.npz` | Barycentric mapping SMPL → MHR |
| `assets/mhr2smpl_mapping.npz` | Barycentric mapping MHR → SMPL |
| `assets/mhr_face_mask.ply` | MHR mesh faces (used for contact interpolation) |

---

## Changes made (2026-03)

### 1. Replaced `smplx` with `better_human.smpl.SMPL`

`smplx` was removed as a dependency.  All three files (`conversion.py`,
`pytorch_fitting.py`, `utils.py`) now import:

```python
from better_human.smpl import SMPL as BetterHumanSMPL
```

**API differences from smplx:**

| smplx | better_human SMPL |
|-------|-------------------|
| `model.v_template` | `model.vertices_template` |
| `model(**params).vertices` | `model(betas=betas, q=q).vertices` |
| axis-angle params directly | must call `model.from_classic(betas, body_pose, translation, global_orient)` → `q` first |
| `model.use_pca` | `getattr(model, 'use_pca', False)` |

`from_classic` bridge:
```python
q = smpl_model.from_classic(
    betas=betas,           # [B, 10]
    body_pose=body_pose,   # [B, 69]  axis-angle
    translation=transl,    # [B, 3]
    global_orient=orient,  # [B, 3]
)  # → [B, 99] quaternion state vector
verts = smpl_model(betas=betas, q=q).vertices  # [B, 6890, 3]
```

### 2. Contact-label support added to `Conversion`

`convert_smpl2mhr()` and `convert_mhr2smpl()` now accept two new optional
parameters:

```
contacts: torch.Tensor | np.ndarray | None  — per-vertex binary labels
contact_threshold: float = 0.5              — binarisation threshold
```

`ConversionResult` (in `utils.py`) gained a new field:

```python
result_contacts: torch.Tensor | None  # Long [B, V_tgt], values in {0, 1}
```

The interpolation is purely barycentric — see `_interpolate_contacts()` in
`conversion.py` — and is independent of the pose-fitting optimiser.

Four new `@cached_property` attributes load and cache mapping tensors:

```
_smpl_faces_tensor         — SMPL triangle connectivity [13776, 3]
_mhr_faces_tensor          — MHR triangle connectivity  [36874, 3]
_smpl2mhr_contact_mapping  — (triangle_ids, baryc_coords) for SMPL→MHR
_mhr2smpl_contact_mapping  — (triangle_ids, baryc_coords) for MHR→SMPL
```

### 3. TorchScript MHR model support

The original code assumed `mhr.mhr.MHR` (the full Python class that requires
`pymomentum`).  The module now also supports the lighter TorchScript model
loaded via `torch.jit.load("mhr_model.pt")`.

**Structural difference:**

| Attribute | Full MHR | TorchScript |
|-----------|----------|-------------|
| character object | `model.character` | `model.character_torch` |
| mesh rest vertices | `character.mesh.vertices` | `character.mesh.rest_vertices` |
| mesh faces | `character.mesh.faces` | `character.mesh.faces` (same) |
| parameter names | `parameter_transform.names` (list of str) | not exposed — returns `None` |

**New compatibility properties added to `Conversion`:**

```python
_mhr_character       # returns model.character or model.character_torch
_mhr_rest_vertices   # rest-pose MHR verts as tensor, works for both model types
_mhr_mesh_faces      # MHR face connectivity as long tensor, works for both
_mhr_param_names     # list[str] or None (None for TorchScript)
```

All internal methods use these properties instead of accessing
`self._mhr_model.character` directly.

**Parameter-name masking:** `_s2m_get_mhr_param_mask()` uses joint names to
build masks that exclude hands or isolate head joints during optimisation.
When `_mhr_param_names` is `None` (TorchScript), the method falls back to
all-True masks for "no_hand" and all-False masks for "head" — i.e. no
joint-specific masking, which is safe but slightly less precise for
full pose fitting.

**`_smpl_faces_tensor`** was also patched to handle the case where
`smpl_model.faces` is already a CUDA tensor (raises `TypeError` with
`np.asarray()` otherwise).

---

## Usage examples

### Setup

```python
import torch
from better_human.smpl import SMPL as BetterHumanSMPL
from mhr_smpl_conversion.conversion import Conversion

# TorchScript MHR (no pymomentum required)
mhr_model = torch.jit.load("path/to/mhr_model.pt")

# better_human SMPL
smpl_model = BetterHumanSMPL("path/to/SMPL_NEUTRAL.npz")

converter = Conversion(
    mhr_model=mhr_model,
    smpl_model=smpl_model,
    method="pytorch",   # "pymomentum" requires the full MHR install
)
```

---

### Example 1 — Contacts only (fast, no optimisation)

```python
import torch

# SMPL → MHR
smpl_contacts = torch.randint(0, 2, (4, 6890)).float()  # [B, 6890]

tri_ids, baryc = converter._smpl2mhr_contact_mapping
mhr_contacts = converter._interpolate_contacts(
    smpl_contacts,
    tri_ids,
    baryc,
    converter._smpl_faces_tensor,
    threshold=0.5,
)
# mhr_contacts: LongTensor [4, 18439], values in {0, 1}

# MHR → SMPL
tri_ids2, baryc2 = converter._mhr2smpl_contact_mapping
smpl_contacts_back = converter._interpolate_contacts(
    mhr_contacts.float(),
    tri_ids2,
    baryc2,
    converter._mhr_faces_tensor,
    threshold=0.5,
)
# smpl_contacts_back: LongTensor [4, 6890]
```

---

### Example 2 — Contacts through the high-level API

```python
smpl_verts    = torch.randn(4, 6890, 3)          # pre-computed SMPL vertices
smpl_contacts = torch.randint(0, 2, (4, 6890)).float()

result = converter.convert_smpl2mhr(
    smpl_vertices=smpl_verts,
    contacts=smpl_contacts,
    return_mhr_parameters=False,
    return_mhr_vertices=False,
    return_fitting_errors=False,
    contact_threshold=0.5,
)
mhr_contacts = result.result_contacts   # LongTensor [4, 18439]
```

---

### Example 3 — Full pose conversion + contacts (SMPL → MHR)

```python
B = 8
smpl_params = {
    "betas":         torch.zeros(B, 10),
    "body_pose":     torch.zeros(B, 69),
    "global_orient": torch.zeros(B, 3),
    "translation":   torch.zeros(B, 3),
}
smpl_contacts = torch.randint(0, 2, (B, 6890)).float()

result = converter.convert_smpl2mhr(
    smpl_parameters=smpl_params,
    contacts=smpl_contacts,
    return_mhr_parameters=True,
    return_mhr_vertices=True,
    contact_threshold=0.5,
)

mhr_params   = result.result_parameters   # dict with MHR pose/shape params
mhr_verts    = result.result_vertices     # np.ndarray [B, 18439, 3]
mhr_contacts = result.result_contacts     # LongTensor [B, 18439]
```

---

### Example 4 — MHR → SMPL with contacts

```python
mhr_verts    = torch.randn(4, 18439, 3)
mhr_contacts = torch.randint(0, 2, (4, 18439)).float()

result = converter.convert_mhr2smpl(
    mhr_vertices=mhr_verts,
    contacts=mhr_contacts,
    return_smpl_parameters=True,
    contact_threshold=0.5,
)
smpl_contacts = result.result_contacts   # LongTensor [4, 6890]
```

---

## Adding support for a new body model

If you want to add a third body model (e.g. SMPL-X, STAR, …):

1. **Add mapping assets** — generate `newmodel2mhr_mapping.npz` and
   `mhr2newmodel_mapping.npz` with keys `triangle_ids` ([V_tgt]) and
   `baryc_coords` ([V_tgt, 3]).  See `dataset/convert_damon_to_mhr.py` for the
   barycentric computation recipe.

2. **Register in `load_surface_mapping()`** (`utils.py`) — add a branch for the
   new model type string.

3. **Add cached properties** in `Conversion` following the pattern of
   `_smpl2mhr_contact_mapping` / `_mhr2smpl_contact_mapping`.

4. **Forward wrapper** — add a `_newmodel_forward(params)` helper (analogous to
   `_smpl_forward`) that calls your model and returns vertices.

---

## Adding support for a new MHR variant

If Meta releases a new MHR version with a different Python interface:

1. Inspect the model with `dir(model)` and `dir(model.character_XX)`.
2. Extend `_mhr_character` to detect and return the correct sub-object.
3. Extend `_mhr_rest_vertices` / `_mhr_mesh_faces` if the mesh attribute names
   change again.
4. If parameter names are available under a new attribute, update
   `_mhr_param_names` to expose them.
5. The contact-interpolation path (`_interpolate_contacts`, `_mhr_faces_tensor`)
   is model-version-agnostic as long as the mesh topology (18 439 vertices,
   36 874 faces) does not change.
