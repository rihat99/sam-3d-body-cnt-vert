"""
Lightweight standalone bidirectional body-mesh contact/vertex converter.

Supports SMPL (6 890 verts) ↔ MHR LOD1 (18 439 verts) via precomputed barycentric
mapping files, plus MHR LOD1 → any other LOD (0, 2–6).

No dependency on pymomentum, smplx, or conversion.py.

LOD vertex counts:
  LOD 0: 73 639  LOD 1: 18 439  LOD 2: 10 661  LOD 3: 4 899
  LOD 4:  2 461  LOD 5:    971  LOD 6:    595  SMPL:  6 890

Conversion directions supported:
  SMPL  → MHR LOD1       (smpl_to_mhr, target_lod=1 default)
  SMPL  → MHR LOD_N      (smpl_to_mhr, target_lod=N, chains SMPL→LOD1→LOD_N)
  MHR LOD1 → SMPL        (mhr_to_smpl)
  MHR LOD1 → MHR LOD_N   (mhr_lod1_to_lod)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConversionOutput:
    """Holds the result of a BodyConverter conversion call."""
    vertices: Optional[Tensor] = None  # [B, V_tgt, 3] or [V_tgt, 3]
    contacts: Optional[Tensor] = None  # [B, V_tgt] long or [V_tgt] long


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

class BodyConverter:
    """
    Bidirectional SMPL ↔ MHR contact-label and vertex converter,
    with optional MHR LOD1 → LOD_N conversion.

    Uses precomputed barycentric mappings stored in the assets/ directory next
    to this file.  All heavy models (SMPL forward pass, MHR optimiser) are
    *not* required — just the mapping NPZ files and the MHR face PLY.

    Conversion directions:
        - SMPL → MHR LOD1 (default)
        - SMPL → MHR LOD_N  (chains SMPL→LOD1→LOD_N via smpl_to_mhr(target_lod=N))
        - MHR LOD1 → SMPL
        - MHR LOD1 → MHR LOD_N  (mhr_lod1_to_lod(target_lod=N))

    Note: only LOD1 ↔ LOD_N direction is supported; no lodN→lod1 reverse
    mappings exist.

    Args:
        smpl_faces: SMPL face connectivity [13776, 3].  Required only when
            calling smpl_to_mhr().  Can be a numpy array or torch Tensor.
        mhr_faces:  MHR LOD1 face connectivity [36874, 3].  Auto-loaded from
            assets/mhr_face_mask.ply (via trimesh) if not provided.
        assets_dir: Directory containing the mapping NPZ files and PLY.
            Defaults to <this_file>/assets (i.e. the module's own assets/).
        device:     Torch device string.  All cached tensors live here.
        threshold:  Default binarisation threshold for contact interpolation.
    """

    _ASSETS_SUBDIR = Path(__file__).parent / "assets"

    # Mapping from LOD number → expected vertex count (for validation)
    LOD_VERTEX_COUNTS: dict[int, int] = {
        0: 73639,
        1: 18439,
        2: 10661,
        3: 4899,
        4: 2461,
        5: 971,
        6: 595,
    }

    def __init__(
        self,
        smpl_faces: Optional[np.ndarray] = None,
        mhr_faces:  Optional[np.ndarray] = None,
        assets_dir: Optional[str] = None,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self._device = torch.device(device)
        self._threshold = threshold
        self._assets = Path(assets_dir) if assets_dir else self._ASSETS_SUBDIR

        # ── load barycentric mappings (SMPL ↔ MHR LOD1) ───────────────────
        m2s = np.load(self._assets / "mhr2smpl_mapping.npz")
        self._m2s_tri_ids = torch.from_numpy(m2s["triangle_ids"].astype(np.int64)).to(self._device)
        self._m2s_baryc   = torch.from_numpy(m2s["baryc_coords"].astype(np.float32)).to(self._device)

        s2m = np.load(self._assets / "smpl2mhr_mapping.npz")
        self._s2m_tri_ids = torch.from_numpy(s2m["triangle_ids"].astype(np.int64)).to(self._device)
        self._s2m_baryc   = torch.from_numpy(s2m["baryc_coords"].astype(np.float32)).to(self._device)

        # ── MHR LOD1 faces ────────────────────────────────────────────────
        if mhr_faces is not None:
            mhr_faces_np = np.asarray(mhr_faces, dtype=np.int64)
        else:
            import trimesh
            ply_path = self._assets / "mhr_face_mask.ply"
            mhr_mesh = trimesh.load(str(ply_path), process=False)
            mhr_faces_np = np.asarray(mhr_mesh.faces, dtype=np.int64)
        self._mhr_faces = torch.from_numpy(mhr_faces_np).long().to(self._device)

        # ── SMPL faces (optional — only needed for smpl_to_mhr) ───────────
        if smpl_faces is not None:
            if isinstance(smpl_faces, torch.Tensor):
                self._smpl_faces = smpl_faces.cpu().long().to(self._device)
            else:
                self._smpl_faces = (
                    torch.from_numpy(np.asarray(smpl_faces, dtype=np.int64))
                    .long().to(self._device)
                )
        else:
            self._smpl_faces = None

        # ── LOD mappings cache (LOD1 → LOD_N) ────────────────────────────
        # Keyed by target LOD int; each value is (tri_ids, baryc_coords)
        self._lod_mappings: dict[int, tuple[Tensor, Tensor]] = {}

    # ------------------------------------------------------------------
    # Public API — SMPL ↔ MHR
    # ------------------------------------------------------------------

    def mhr_to_smpl(
        self,
        vertices: Optional[Tensor | np.ndarray] = None,
        contacts: Optional[Tensor | np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> ConversionOutput:
        """
        Convert MHR LOD1 data to SMPL space.

        Args:
            vertices: [B, 18439, 3] or [18439, 3] — vertex positions.
            contacts: [B, 18439]   or [18439]     — contact labels (float or bool).
            threshold: Override the instance-level binarisation threshold.

        Returns:
            ConversionOutput with .vertices [B/·, 6890, 3] and/or .contacts [B/·, 6890].
        """
        thr = threshold if threshold is not None else self._threshold

        out_verts = None
        out_contacts = None

        if vertices is not None:
            verts_t, was_single = self._to_tensor_batched(vertices, ndim=3)
            out_verts = self._interpolate(verts_t, self._m2s_tri_ids, self._m2s_baryc, self._mhr_faces)
            if was_single:
                out_verts = out_verts.squeeze(0)

        if contacts is not None:
            cont_t, was_single = self._to_tensor_batched(contacts, ndim=2)
            interp = self._interpolate(cont_t, self._m2s_tri_ids, self._m2s_baryc, self._mhr_faces)
            out_contacts = (interp > thr).long()
            if was_single:
                out_contacts = out_contacts.squeeze(0)

        return ConversionOutput(vertices=out_verts, contacts=out_contacts)

    def smpl_to_mhr(
        self,
        vertices: Optional[Tensor | np.ndarray] = None,
        contacts: Optional[Tensor | np.ndarray] = None,
        threshold: Optional[float] = None,
        target_lod: int = 1,
    ) -> ConversionOutput:
        """
        Convert SMPL data to MHR space.

        If target_lod == 1 (default), converts directly to MHR LOD1 (18 439 verts).
        If target_lod != 1, chains SMPL → LOD1 → LOD_N using precomputed mappings.

        Requires ``smpl_faces`` to have been provided at construction time.

        Args:
            vertices:   [B, 6890, 3] or [6890, 3]
            contacts:   [B, 6890]   or [6890]
            threshold:  Override the instance-level binarisation threshold.
            target_lod: Target MHR LOD level (0–6).  Default 1.
        """
        if self._smpl_faces is None:
            raise ValueError(
                "smpl_to_mhr() requires SMPL face connectivity. "
                "Pass smpl_faces to BodyConverter.__init__()."
            )
        if target_lod not in self.LOD_VERTEX_COUNTS:
            raise ValueError(f"target_lod must be 0–6, got {target_lod}")

        thr = threshold if threshold is not None else self._threshold

        out_verts = None
        out_contacts = None

        if vertices is not None:
            verts_t, was_single = self._to_tensor_batched(vertices, ndim=3)
            out_verts = self._interpolate(verts_t, self._s2m_tri_ids, self._s2m_baryc, self._smpl_faces)
            if target_lod != 1:
                out_verts = self._apply_lod_mapping_verts(out_verts, target_lod)
            if was_single:
                out_verts = out_verts.squeeze(0)

        if contacts is not None:
            cont_t, was_single = self._to_tensor_batched(contacts, ndim=2)
            interp = self._interpolate(cont_t, self._s2m_tri_ids, self._s2m_baryc, self._smpl_faces)
            if target_lod != 1:
                interp = self._apply_lod_mapping_contacts(interp, target_lod)
            out_contacts = (interp > thr).long()
            if was_single:
                out_contacts = out_contacts.squeeze(0)

        return ConversionOutput(vertices=out_verts, contacts=out_contacts)

    # ------------------------------------------------------------------
    # Public API — MHR LOD1 → any LOD
    # ------------------------------------------------------------------

    def mhr_lod1_to_lod(
        self,
        vertices: Optional[Tensor | np.ndarray] = None,
        contacts: Optional[Tensor | np.ndarray] = None,
        target_lod: int = 1,
        threshold: Optional[float] = None,
    ) -> ConversionOutput:
        """
        Convert MHR LOD1 data (18 439 verts) to any target LOD.

        If target_lod == 1, returns the input unchanged (identity).

        Args:
            vertices:   [B, 18439, 3] or [18439, 3]
            contacts:   [B, 18439]   or [18439]
            target_lod: Target LOD level (0–6).
            threshold:  Override the instance-level binarisation threshold.

        Returns:
            ConversionOutput with .vertices [B/·, V_tgt, 3] and/or .contacts [B/·, V_tgt].
        """
        if target_lod not in self.LOD_VERTEX_COUNTS:
            raise ValueError(f"target_lod must be 0–6, got {target_lod}")

        thr = threshold if threshold is not None else self._threshold

        out_verts = None
        out_contacts = None

        if vertices is not None:
            verts_t, was_single = self._to_tensor_batched(vertices, ndim=3)
            if target_lod == 1:
                out_verts = verts_t
            else:
                out_verts = self._apply_lod_mapping_verts(verts_t, target_lod)
            if was_single:
                out_verts = out_verts.squeeze(0)

        if contacts is not None:
            cont_t, was_single = self._to_tensor_batched(contacts, ndim=2)
            if target_lod == 1:
                out_contacts = (cont_t > thr).long()
            else:
                interp = self._apply_lod_mapping_contacts(cont_t, target_lod)
                out_contacts = (interp > thr).long()
            if was_single:
                out_contacts = out_contacts.squeeze(0)

        return ConversionOutput(vertices=out_verts, contacts=out_contacts)

    # ------------------------------------------------------------------
    # LOD mapping helpers
    # ------------------------------------------------------------------

    def _get_lod_mapping(self, target_lod: int) -> tuple[Tensor, Tensor]:
        """Lazy-load and cache LOD1 → target_lod barycentric mapping."""
        if target_lod not in self._lod_mappings:
            npz_path = self._assets / f"lod1_to_lod{target_lod}_mapping.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"LOD mapping file not found: {npz_path}. "
                    f"Expected 'lod1_to_lod{target_lod}_mapping.npz' in {self._assets}."
                )
            lod_data = np.load(npz_path)
            tri_ids = torch.from_numpy(lod_data["triangle_ids"].astype(np.int64)).to(self._device)
            baryc   = torch.from_numpy(lod_data["baryc_coords"].astype(np.float32)).to(self._device)
            self._lod_mappings[target_lod] = (tri_ids, baryc)
        return self._lod_mappings[target_lod]

    def _apply_lod_mapping_verts(self, verts_lod1: Tensor, target_lod: int) -> Tensor:
        """Apply LOD1 → target_lod mapping to vertex positions [B, 18439, 3]."""
        tri_ids, baryc = self._get_lod_mapping(target_lod)
        return self._interpolate(verts_lod1, tri_ids, baryc, self._mhr_faces)

    def _apply_lod_mapping_contacts(self, cont_lod1: Tensor, target_lod: int) -> Tensor:
        """Apply LOD1 → target_lod mapping to contact values [B, 18439] → [B, V_tgt]."""
        tri_ids, baryc = self._get_lod_mapping(target_lod)
        return self._interpolate(cont_lod1, tri_ids, baryc, self._mhr_faces)

    # ------------------------------------------------------------------
    # Core interpolation
    # ------------------------------------------------------------------

    def _interpolate(
        self,
        source_data: Tensor,   # [B, V_src] or [B, V_src, 3]
        triangle_ids: Tensor,  # [V_tgt]
        baryc_coords: Tensor,  # [V_tgt, 3]
        source_faces: Tensor,  # [F, 3]
    ) -> Tensor:
        """
        Vectorised barycentric interpolation.

        For each target vertex i:
          1. Find the source triangle:  source_faces[triangle_ids[i]]  → 3 vert indices
          2. Gather source values at those 3 verts
          3. Weighted sum using baryc_coords[i]

        Returns [B, V_tgt] for 1-D data or [B, V_tgt, 3] for 3-D data.
        """
        source_data = source_data.to(self._device)

        # [V_tgt, 3]  — vertex indices in the source mesh
        tri_vertex_ids = source_faces[triangle_ids]  # [V_tgt, 3]

        if source_data.dim() == 2:
            # contacts: [B, V_src] → [B, V_tgt]
            tri_vals = source_data[:, tri_vertex_ids]          # [B, V_tgt, 3]
            return (tri_vals * baryc_coords).sum(-1)           # [B, V_tgt]
        else:
            # vertices: [B, V_src, 3] → [B, V_tgt, 3]
            tri_vals = source_data[:, tri_vertex_ids]          # [B, V_tgt, 3, 3]
            return (tri_vals * baryc_coords.unsqueeze(-1)).sum(-2)  # [B, V_tgt, 3]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor_batched(
        data: Tensor | np.ndarray,
        ndim: int,
    ) -> tuple[Tensor, bool]:
        """
        Ensure *data* is a float32 Tensor with a batch dimension.

        Returns:
            (batched_tensor, was_single)
            *was_single* is True if a batch dim was added (caller should squeeze output).
        """
        if isinstance(data, np.ndarray):
            t = torch.from_numpy(data.astype(np.float32))
        elif isinstance(data, torch.Tensor):
            t = data.float()
        else:
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(data)}")

        was_single = (t.dim() == ndim - 1)
        if was_single:
            t = t.unsqueeze(0)
        return t, was_single
