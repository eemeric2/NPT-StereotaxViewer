# NPT-StereotaxViewer/npt_stereotax_viewer/transform.py
"""
StereotaxTransform: Core coordinate transformation between voxel, RAS, and stereotaxic spaces.
Handles NIfTI affine conversions, HC (head-centered) NIfTI generation, and stereotaxic metadata.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import nibabel as nib

from .utils import VOI_Stereo, ChamberInfo


class StereotaxTransform:
    """
    Manages coordinate transformations for non-human primate stereotaxic MRI.
    
    Converts between:
    - Voxel indices (array coordinates)
    - RAS (Right-Anterior-Superior) physical coordinates
    - Stereotaxic coordinates (aligned to stereotaxic frame)
    
    Preserves original NIfTI affine and generates head-centered (HC) NIfTI variants.
    """

    def __init__(
        self,
        nifti_path: str,
        voxel_size: Tuple[float, float, float],
        voxel_origin: Tuple[float, float, float],
        stereotax_zero_ras: Optional[Tuple[float, float, float]] = None,
        vois_json: Optional[str] = None,
        chamber_json: Optional[str] = None,
    ):
        """
        Initialize stereotaxic transform.

        Parameters
        ----------
        nifti_path : str
            Path to NIfTI file (.nii or .nii.gz)
        voxel_size : tuple of float
            Voxel dimensions (X, Y, Z) in mm
        voxel_origin : tuple of float
            Origin voxel index [X, Y, Z] where stereotaxic [0,0,0] is located
        stereotax_zero_ras : tuple of float, optional
            RAS coordinates of stereotaxic origin. If None, computed from voxel_origin.
        vois_json : str, optional
            Path to VOI JSON file
        chamber_json : str, optional
            Path to chamber info JSON file
        """
        self.nifti_path = Path(nifti_path)
        self.voxel_size = np.array(voxel_size, dtype=np.float64)
        self.voxel_origin = np.array(voxel_origin, dtype=np.float64)

        # Load NIfTI and preserve original affine
        self.nifti = nib.load(self.nifti_path)
        self.data = self.nifti.get_fdata()
        self.original_affine = self.nifti.affine.copy()
        
        # Store original stereotax zero in RAS space
        if stereotax_zero_ras is None:
            # Compute RAS from voxel origin using original affine
            voxel_hom = np.hstack([self.voxel_origin, 1])
            ras = (self.original_affine @ voxel_hom)[:3]
            self.original_stereotax_zero_ras = ras
        else:
            self.original_stereotax_zero_ras = np.array(stereotax_zero_ras, dtype=np.float64)

        # Load VOIs and chamber info if provided
        self.vois_stereo = {}
        self.chamber_info = None

        if vois_json:
            self._load_vois(vois_json)
        if chamber_json:
            self._load_chamber(chamber_json)

    def _load_vois(self, vois_json: str):
        """Load VOIs from JSON file."""
        with open(vois_json) as f:
            vois_data = json.load(f)
        for name, voi_dict in vois_data.items():
            self.vois_stereo[name] = VOI_Stereo.from_dict(voi_dict)

    def _load_chamber(self, chamber_json: str):
        """Load chamber info from JSON file."""
        with open(chamber_json) as f:
            chamber_data = json.load(f)
        self.chamber_info = ChamberInfo.from_dict(chamber_data)

    # ========== Coordinate Conversion Methods ==========

    def voxel_to_ras(self, voxel_idx: np.ndarray) -> np.ndarray:
        """
        Convert voxel indices to RAS coordinates.

        Parameters
        ----------
        voxel_idx : np.ndarray
            Voxel indices [X, Y, Z]

        Returns
        -------
        np.ndarray
            RAS coordinates [R, A, S]
        """
        voxel_hom = np.hstack([voxel_idx.flatten(), 1])
        ras = (self.original_affine @ voxel_hom)[:3]
        return ras

    def ras_to_voxel(self, ras: np.ndarray) -> np.ndarray:
        """
        Convert RAS coordinates to voxel indices.

        Parameters
        ----------
        ras : np.ndarray
            RAS coordinates [R, A, S]

        Returns
        -------
        np.ndarray
            Voxel indices [X, Y, Z]
        """
        affine_inv = np.linalg.inv(self.original_affine)
        ras_hom = np.hstack([ras.flatten(), 1])
        voxel = (affine_inv @ ras_hom)[:3]
        return voxel

    def voxel_to_stereo(self, voxel_idx: np.ndarray) -> np.ndarray:
        """
        Convert voxel indices to stereotaxic coordinates.

        Formula (verified with MIPAV):
        - X, Y (flipped): stereo = (voxel_origin - voxel_idx) * voxel_size
        - Z: stereo = (voxel_idx - voxel_origin) * voxel_size

        Parameters
        ----------
        voxel_idx : np.ndarray
            Voxel indices [X, Y, Z]

        Returns
        -------
        np.ndarray
            Stereotaxic coordinates [X, Y, Z] in mm
        """
        voxel_idx = np.array(voxel_idx, dtype=np.float64).flatten()
        stereo = np.zeros(3, dtype=np.float64)
        stereo[0] = (self.voxel_origin[0] - voxel_idx[0]) * self.voxel_size[0]
        stereo[1] = (self.voxel_origin[1] - voxel_idx[1]) * self.voxel_size[1]
        stereo[2] = (voxel_idx[2] - self.voxel_origin[2]) * self.voxel_size[2]
        return stereo

    def stereo_to_voxel(self, stereo: np.ndarray) -> np.ndarray:
        """
        Convert stereotaxic coordinates to voxel indices.

        Parameters
        ----------
        stereo : np.ndarray
            Stereotaxic coordinates [X, Y, Z] in mm

        Returns
        -------
        np.ndarray
            Voxel indices [X, Y, Z]
        """
        stereo = np.array(stereo, dtype=np.float64).flatten()
        voxel = np.zeros(3, dtype=np.float64)
        voxel[0] = self.voxel_origin[0] - stereo[0] / self.voxel_size[0]
        voxel[1] = self.voxel_origin[1] - stereo[1] / self.voxel_size[1]
        voxel[2] = self.voxel_origin[2] + stereo[2] / self.voxel_size[2]
        return voxel

    def ras_to_stereo(self, ras: np.ndarray) -> np.ndarray:
        """Convert RAS to stereotaxic coordinates."""
        voxel = self.ras_to_voxel(ras)
        return self.voxel_to_stereo(voxel)

    def stereo_to_ras(self, stereo: np.ndarray) -> np.ndarray:
        """Convert stereotaxic to RAS coordinates."""
        voxel = self.stereo_to_voxel(stereo)
        return self.voxel_to_ras(voxel)

    # ========== HC (Head-Centered) NIfTI Methods ==========

    def _compute_hc_affine(self) -> np.ndarray:
        """
        Compute HC affine where stereotaxic origin becomes image origin.

        Returns
        -------
        np.ndarray
            4x4 affine matrix for HC NIfTI
        """
        affine_inv = np.linalg.inv(self.original_affine)
        ras_hom = np.hstack([self.original_stereotax_zero_ras.reshape(1, -1), np.ones((1, 1))])
        hc_origin_voxel = (affine_inv @ ras_hom.T).T[0, :3]
        
        affine_hc = self.original_affine.copy()
        affine_hc[:3, 3] = -self.original_affine[:3, :3] @ hc_origin_voxel
        
        return affine_hc

    def get_hc_nifti(self) -> nib.Nifti1Image:
        """
        Generate HC NIfTI image with stereotaxic origin at image center.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image with HC affine
        """
        affine_hc = self._compute_hc_affine()
        hc_nifti = nib.Nifti1Image(self.data, affine_hc)
        return hc_nifti

    def save_hc_nifti(self, output_path: str):
        """
        Save HC NIfTI to file.

        Parameters
        ----------
        output_path : str
            Output file path (.nii or .nii.gz)
        """
        hc_nifti = self.get_hc_nifti()
        nib.save(hc_nifti, output_path)

    def save_stereo_nifti(self, output_path: str):
        """
        Save original NIfTI (preserves original affine).

        Parameters
        ----------
        output_path : str
            Output file path
        """
        nib.save(self.nifti, output_path)

    # ========== Metadata Access ==========

    def get_voi_stereo(self, name: str) -> Optional[VOI_Stereo]:
        """Get VOI in stereotaxic coordinates."""
        return self.vois_stereo.get(name)

    def list_vois(self) -> list:
        """List all loaded VOI names."""
        return list(self.vois_stereo.keys())

    def get_chamber_info(self) -> Optional[ChamberInfo]:
        """Get chamber information."""
        return self.chamber_info