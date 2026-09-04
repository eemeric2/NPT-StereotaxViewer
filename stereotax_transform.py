# stereotax_transform.py
"""
Stereotaxic coordinate transformation module for NHP neurophysiology.

Provides transformations between:
- Voxel space (NIfTI array indices)
- RAS space (Right-Anterior-Superior, neuroimaging standard)
- Stereotaxic space (Horsley-Clarke, lab coordinates)
- Chamber space (relative to recording chamber)

Conventions:
- Stereotaxic coordinates: [ML, AP, DV] where positive ML=right, AP=anterior, DV=dorsal
- Chamber tilt: [roll, pitch, yaw] in degrees, right-hand rule
- Depth: 0 at chamber surface, positive into brain
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional
import xml.etree.ElementTree as ET
from matplotlib.widgets import Button
from matplotlib.patches import Patch
from typing import Optional, Union

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Contour:
    """A single 2D contour from a VOI."""
    slice_index: int  # k index in voxel space
    points_ij: np.ndarray  # Nx2 array of [i, j] coordinates


@dataclass
class VOI:
    """Volume of Interest in voxel space."""
    name: str
    contours: List[Contour]
    color: Tuple[int, int, int, int] = (255, 0, 0, 255)  # RGBA
    
    @property
    def all_points_ijk(self) -> np.ndarray:
        """Get all points as Nx3 array of [i, j, k] voxel coordinates."""
        all_points = []
        for contour in self.contours:
            k = contour.slice_index
            for ij in contour.points_ij:
                all_points.append([ij[0], ij[1], k])
        return np.array(all_points) if all_points else np.empty((0, 3))


@dataclass
class VOI_Stereo:
    """VOI with coordinates in stereotaxic space."""
    name: str
    contours_stereo: List[Dict]  # List of {'slice_dv': float, 'points_stereo': np.ndarray}
    original_voi: VOI
    color: Tuple[int, int, int, int] = (255, 0, 0, 255)
    
    @property
    def all_points_stereo(self) -> np.ndarray:
        """Get all points as Nx3 array of [ML, AP, DV] coordinates."""
        all_points = [c['points_stereo'] for c in self.contours_stereo]
        return np.vstack(all_points) if all_points else np.empty((0, 3))
    
    @property
    def contours_hc(self) -> List[Dict]:
        """Alias for backwards compatibility."""
        return self.contours_stereo
    
    def get_contours_at_dv(self, dv: float, tolerance: float = 0.5) -> List[Dict]:
        """Get contours near a specific DV level."""
        return [c for c in self.contours_stereo 
                if abs(c['slice_dv'] - dv) <= tolerance]


@dataclass
class ChamberInfo:
    """
    Recording chamber with position and orientation.
    
    Attributes
    ----------
    chamber_id : str
        Unique identifier (e.g., "chamber_01")
    name : str
        Descriptive name (e.g., "Left PFC")
    stereo_location : np.ndarray
        Chamber center in stereotaxic coords [ML, AP, DV]
    tilt_degrees : np.ndarray
        Rotation angles [roll, pitch, yaw] in degrees
    radius : float
        Chamber radius in mm (default 10.0)
    grid_type : str
        Positioning system type ("plug_system", "cartesian")
    active : bool
        Whether chamber is currently in use
    """
    chamber_id: str
    name: str
    stereo_location: np.ndarray
    tilt_degrees: np.ndarray
    radius: float = 10.0
    grid_type: str = "cartesian"
    lookup_file: Optional[str] = None
    notes: str = ""
    active: bool = True
    
    # Computed rotation matrix and axes (set in __post_init__)
    rotation_matrix: np.ndarray = field(default=None, repr=False)
    x_axis: np.ndarray = field(default=None, repr=False)
    y_axis: np.ndarray = field(default=None, repr=False)
    depth_axis: np.ndarray = field(default=None, repr=False)
    normal: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compute rotation matrix and chamber axes."""
        self.stereo_location = np.asarray(self.stereo_location, dtype=np.float64)
        self.tilt_degrees = np.asarray(self.tilt_degrees, dtype=np.float64)
        
        # Build rotation matrices (right-hand rule)
        roll_rad = np.radians(self.tilt_degrees[0])   # about X (ML)
        pitch_rad = np.radians(self.tilt_degrees[1])  # about Y (AP)
        yaw_rad = np.radians(self.tilt_degrees[2])    # about Z (DV)
        
        # Rotation about X (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        # Rotation about Y (pitch)
        Ry = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        # Rotation about Z (yaw)
        Rz = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = Rz @ Ry @ Rx (applied right to left)
        self.rotation_matrix = Rz @ Ry @ Rx
        
        # Chamber local axes in stereotaxic coordinates
        # Before rotation: X=ML, Y=AP, Z=down (into brain)
        self.x_axis = self.rotation_matrix @ np.array([1, 0, 0])
        self.y_axis = self.rotation_matrix @ np.array([0, 1, 0])
        self.depth_axis = self.rotation_matrix @ np.array([0, 0, -1])  # Into brain
        self.normal = -self.depth_axis  # Outward from brain
    
    @property
    def tilt(self) -> np.ndarray:
        """Alias for tilt_degrees."""
        return self.tilt_degrees
    
    def chamber_to_stereo(
        self, 
        x: float, 
        y: float, 
        depth: float
    ) -> np.ndarray:
        """
        Convert chamber coordinates to stereotaxic coordinates.
        
        Parameters
        ----------
        x : float
            X position in chamber plane (mm), positive = right in chamber view
        y : float
            Y position in chamber plane (mm), positive = anterior in chamber view
        depth : float
            Depth from chamber surface (mm), positive = into brain
        
        Returns
        -------
        np.ndarray
            Stereotaxic coordinates [ML, AP, DV]
        """
        # Position in chamber local frame
        local_pos = x * self.x_axis + y * self.y_axis + depth * self.depth_axis
        
        # Add chamber location
        return self.stereo_location + local_pos
    
    def stereo_to_chamber(
        self, 
        stereo_coords: Union[list, np.ndarray]
    ) -> Tuple[bool, np.ndarray, float]:
        """
        Convert stereotaxic coordinates to chamber coordinates.
        
        Parameters
        ----------
        stereo_coords : array-like
            Stereotaxic coordinates [ML, AP, DV]
        
        Returns
        -------
        reachable : bool
            True if target is within chamber radius
        chamber_xy : np.ndarray
            [x, y] position in chamber plane
        depth : float
            Depth from chamber surface (positive = into brain)
        """
        stereo_coords = np.asarray(stereo_coords, dtype=np.float64)
        
        # Vector from chamber to target
        delta = stereo_coords - self.stereo_location
        
        # Project onto chamber axes
        x = np.dot(delta, self.x_axis)
        y = np.dot(delta, self.y_axis)
        depth = np.dot(delta, self.depth_axis)
        
        # Check if within chamber radius
        radial_dist = np.sqrt(x**2 + y**2)
        reachable = radial_dist <= self.radius
        
        return reachable, np.array([x, y]), depth
    
    def find_entry_point(
        self, 
        target_stereo: Union[list, np.ndarray]
    ) -> Dict:
        """
        Find the chamber entry point to reach a target location.
        
        Parameters
        ----------
        target_stereo : array-like
            Target location in stereotaxic coordinates [ML, AP, DV]
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'reachable': bool
            - 'chamber_x': float, X position in chamber (mm)
            - 'chamber_y': float, Y position in chamber (mm)
            - 'depth': float, depth to target (mm)
            - 'radial_distance': float, distance from chamber center (mm)
            - 'entry_point_hc': np.ndarray, entry point in stereotaxic coordinates
            - 'target_hc': np.ndarray, target in stereotaxic coordinates
        """
        target_stereo = np.asarray(target_stereo, dtype=np.float64)
        reachable, chamber_xy, depth = self.stereo_to_chamber(target_stereo)
        
        radial_distance = np.linalg.norm(chamber_xy)
        entry_point = self.chamber_to_stereo(chamber_xy[0], chamber_xy[1], 0)
        
        return {
            'reachable': reachable,
            'chamber_x': chamber_xy[0],
            'chamber_y': chamber_xy[1],
            'depth': depth,
            'radial_distance': radial_distance,
            'entry_point_hc': entry_point,
            'target_hc': target_stereo
        }
    
    def get_trajectory(
        self, 
        x: float, 
        y: float, 
        depth_range: Tuple[float, float] = (-5, 50),
        n_points: int = 100
    ) -> np.ndarray:
        """
        Get points along an electrode trajectory.
        
        Parameters
        ----------
        x : float
            X position in chamber plane (mm)
        y : float
            Y position in chamber plane (mm)
        depth_range : tuple
            (min_depth, max_depth) in mm. Negative = above chamber.
        n_points : int
            Number of points along trajectory
        
        Returns
        -------
        np.ndarray
            Nx3 array of stereotaxic coordinates along trajectory
        """
        depths = np.linspace(depth_range[0], depth_range[1], n_points)
        points = np.array([self.chamber_to_stereo(x, y, d) for d in depths])
        return points

    # In ChamberInfo class, add this method:
    def get_disk_edge(self, n_points: int = 100) -> np.ndarray:
        """
        Get points along the chamber disk edge for visualization.
        
        Returns
        -------
        np.ndarray
            Nx3 array of stereotaxic coordinates along disk edge.
        """
        theta = np.linspace(0, 2 * np.pi, n_points)
        edge_points = np.array([
            self.chamber_to_stereo(
                self.radius * np.cos(t),
                self.radius * np.sin(t),
                0
            ) for t in theta
        ])
        return edge_points

    def get_disk_points(
        self, 
        n_radial: int = 10, 
        n_angular: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get points on the chamber disk surface for visualization.
        
        Returns
        -------
        X, Y, Z : np.ndarray
            Meshgrid arrays of stereotaxic coordinates for the disk surface.
        """
        r = np.linspace(0, self.radius, n_radial)
        theta = np.linspace(0, 2 * np.pi, n_angular)
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        # Points in chamber local coordinates
        local_x = r_grid * np.cos(theta_grid)
        local_y = r_grid * np.sin(theta_grid)
        
        # Transform each point to stereotaxic
        shape = local_x.shape
        X = np.zeros(shape)
        Y = np.zeros(shape)
        Z = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                stereo_point = self.chamber_to_stereo(local_x[i, j], local_y[i, j], 0)
                X[i, j] = stereo_point[0]
                Y[i, j] = stereo_point[1]
                Z[i, j] = stereo_point[2]
        
        return X, Y, Z


# =============================================================================
# Main Transform Class
# =============================================================================

class StereotaxTransform:
    """
    Coordinate transformation handler for NHP neurophysiology.
    
    Loads subject configuration and provides methods to convert between
    voxel, RAS, and stereotaxic coordinate systems.
    """
    
    def __init__(
        self,
        nifti_path: Union[str, Path],
        stereotax_zero_ras: Union[list, np.ndarray],
        voi_dir: Optional[Union[str, Path]] = None,
        voi_files: Optional[List[str]] = None,
        chambers: Optional[List[Dict]] = None,
        subject_id: str = "unknown"
    ):
        """
        Initialize transform from parameters.
        
        Parameters
        ----------
        nifti_path : str or Path
            Path to anatomical NIfTI file
        stereotax_zero_ras : array-like
            RAS coordinates of stereotaxic origin [R, A, S]
        voi_dir : str or Path, optional
            Directory containing VOI files
        voi_files : list of str, optional
            VOI filenames
        chambers : list of dict, optional
            Chamber configurations
        subject_id : str
            Subject identifier
        """
        
        """
        Initialize transform from parameters.
        ...
        """
        self.nifti_path = Path(nifti_path)
        self.voi_dir = Path(voi_dir) if voi_dir else None
        self.voi_files = voi_files or []
        self.subject_id = subject_id

        # Store the ORIGINAL stereotaxic zero BEFORE it gets changed
        self.original_stereotax_zero_ras = np.asarray(stereotax_zero_ras, dtype=np.float64)

        # Load ORIGINAL NIfTI first to get its affine
        original_img = nib.load(str(self.nifti_path))
        self.original_affine = original_img.affine.copy()

        # Check for or create *_HC.nii file
        hc_path = self._get_or_create_stereo_nifti()
        self.nifti_path = hc_path  # Use the HC version for display
        
        # For *_HC.nii files, stereotaxic zero is always at [0, 0, 0]
        if self.nifti_path.stem.endswith('_HC'):
            self.stereotax_zero_ras = np.array([0, 0, 0])
            print("Using *_HC.nii file - stereotaxic_zero_ras set to [0, 0, 0]")
        else:
            # For non-HC files, use the provided value
            self.stereotax_zero_ras = np.asarray(stereotax_zero_ras, dtype=np.float64)
        
        # Load NIfTI
        self._load_nifti()
        
        # Compute stereotax zero in voxel space
        self.stereotax_zero_voxel = self.ras_to_voxel(self.stereotax_zero_ras)
        
        # Parse chambers
        self.chambers: List[ChamberInfo] = []
        if chambers:
            for c in chambers:
                chamber = ChamberInfo(
                    chamber_id=c.get('chamber_id', 'chamber_01'),
                    name=c.get('name', c.get('chamber_id', 'Unknown')),
                    stereo_location=np.array(c['stereo_location']),
                    tilt_degrees=np.array(c.get('tilt', [0, 0, 0])),
                    radius=c.get('radius', 10.0),
                    grid_type=c.get('grid_type', 'cartesian'),
                    lookup_file=c.get('lookup_file'),
                    notes=c.get('notes', ''),
                    active=c.get('active', True)
                )
                self.chambers.append(chamber)

    def _get_or_create_stereo_nifti(self) -> Path:
        """
        Get or create the stereotaxic-aligned NIfTI file.
        
        Returns
        -------
        Path
            Path to the *_HC.nii file
        """
        hc_path = self.nifti_path.parent / f"{self.nifti_path.stem}_HC.nii"
        
        # If HC version already exists, use it
        if hc_path.exists():
            print(f"Found existing stereotaxic NIfTI: {hc_path}")
            return hc_path
        
        # Otherwise, create it from the original
        print(f"Creating stereotaxic NIfTI from {self.nifti_path}")
        print(f"  (This may take a moment...)")
        
        # Load original NIfTI temporarily
        img = nib.load(str(self.nifti_path))
        data = img.get_fdata()
        affine = img.affine
        voxel_size = np.abs(np.diag(affine)[:3])
        
        # Calculate stereotaxic coordinates of voxel [0, 0, 0]
        origin_ras = (affine @ np.array([0, 0, 0, 1]))[:3]
        origin_stereo = origin_ras - self.stereotax_zero_ras
        
        # Build new affine for stereotaxic space
        new_affine = np.eye(4)
        new_affine[0, 0] = voxel_size[0]
        new_affine[1, 1] = voxel_size[1]
        new_affine[2, 2] = voxel_size[2]
        new_affine[:3, 3] = origin_stereo
        
        # Create and save new NIfTI
        new_img = nib.Nifti1Image(data, new_affine)
        nib.save(new_img, str(hc_path))
        
        print(f"Saved stereotaxic NIfTI to: {hc_path}")
        return hc_path

    def _load_nifti(self):
        """Load NIfTI file and extract geometry."""
        if not self.nifti_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {self.nifti_path}")
        
        img = nib.load(str(self.nifti_path))
        self.nifti_data = img.get_fdata()
        self.affine = img.affine
        self.shape = self.nifti_data.shape
        self.voxel_size = np.abs(np.diag(self.affine)[:3])
        
        # Store header for saving
        self._nifti_header = img.header.copy()
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'StereotaxTransform':
        """
        Create transform from subject JSON configuration.
        
        Parameters
        ----------
        json_path : str or Path
            Path to subject JSON file
        
        Returns
        -------
        StereotaxTransform
            Configured transform object
        """
        json_path = Path(json_path)
        subject_dir = json_path.parent
        
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # Find NIfTI file
        nifti_name = config['anatomical_file']
        nifti_path = subject_dir / 'Imaging' / nifti_name
        if not nifti_path.exists():
            nifti_path = subject_dir / nifti_name
        
        # VOI directory
        voi_dir = subject_dir / 'VOIs'
        if not voi_dir.exists():
            voi_dir = subject_dir / 'Imaging'
        
        return cls(
            nifti_path=nifti_path,
            stereotax_zero_ras=config['stereotax_zero_ras'],
            voi_dir=voi_dir,
            voi_files=config.get('voi_files', []),
            chambers=config.get('chambers', []),
            subject_id=config.get('nhp_id', 'unknown')
        )
    
    # =========================================================================
    # Coordinate Conversions
    # =========================================================================
    
    def voxel_to_ras(self, voxel_coords: np.ndarray) -> np.ndarray:
        """
        Convert voxel coordinates to RAS coordinates.
        
        Parameters
        ----------
        voxel_coords : np.ndarray
            Voxel coordinates, shape (3,) or (N, 3)
        
        Returns
        -------
        np.ndarray
            RAS coordinates, same shape as input
        """
        voxel_coords = np.asarray(voxel_coords, dtype=np.float64)
        single = voxel_coords.ndim == 1
        if single:
            voxel_coords = voxel_coords.reshape(1, -1)
        
        # Apply affine: RAS = affine @ [i, j, k, 1]^T
        ones = np.ones((voxel_coords.shape[0], 1))
        voxel_hom = np.hstack([voxel_coords, ones])
        ras = (self.affine @ voxel_hom.T).T[:, :3]
        
        return ras[0] if single else ras
    
    def ras_to_voxel(self, ras_coords: np.ndarray) -> np.ndarray:
        """
        Convert RAS coordinates to voxel coordinates.
        
        Parameters
        ----------
        ras_coords : np.ndarray
            RAS coordinates, shape (3,) or (N, 3)
        
        Returns
        -------
        np.ndarray
            Voxel coordinates, same shape as input
        """
        ras_coords = np.asarray(ras_coords, dtype=np.float64)
        single = ras_coords.ndim == 1
        if single:
            ras_coords = ras_coords.reshape(1, -1)
        
        # Inverse affine
        affine_inv = np.linalg.inv(self.affine)
        ones = np.ones((ras_coords.shape[0], 1))
        ras_hom = np.hstack([ras_coords, ones])
        voxel = (affine_inv @ ras_hom.T).T[:, :3]
        
        return voxel[0] if single else voxel
    
    def ras_to_stereo(self, ras_coords: np.ndarray) -> np.ndarray:
        """
        Convert RAS coordinates to stereotaxic coordinates.
        
        Parameters
        ----------
        ras_coords : np.ndarray
            RAS coordinates, shape (3,) or (N, 3)
        
        Returns
        -------
        np.ndarray
            Stereotaxic coordinates [ML, AP, DV], same shape as input
        """
        ras_coords = np.asarray(ras_coords, dtype=np.float64)
        return ras_coords - self.stereotax_zero_ras
    
    def stereo_to_ras(self, stereo_coords: np.ndarray) -> np.ndarray:
        """
        Convert stereotaxic coordinates to RAS coordinates.
        
        Parameters
        ----------
        stereo_coords : np.ndarray
            Stereotaxic coordinates [ML, AP, DV], shape (3,) or (N, 3)
        
        Returns
        -------
        np.ndarray
            RAS coordinates, same shape as input
        """
        stereo_coords = np.asarray(stereo_coords, dtype=np.float64)
        return stereo_coords + self.stereotax_zero_ras
    
    def voxel_to_stereo(self, voxel_coords: np.ndarray) -> np.ndarray:
        """
        Convert voxel coordinates to stereotaxic coordinates.
        
        Parameters
        ----------
        voxel_coords : np.ndarray
            Voxel coordinates, shape (3,) or (N, 3)
        
        Returns
        -------
        np.ndarray
            Stereotaxic coordinates [ML, AP, DV], same shape as input
        """
        ras = self.voxel_to_ras(voxel_coords)
        return self.ras_to_stereo(ras)
    
    def stereo_to_voxel(self, stereo_coords: np.ndarray) -> np.ndarray:
        """
        Convert stereotaxic coordinates to voxel coordinates.
        
        Parameters
        ----------
        stereo_coords : np.ndarray
            Stereotaxic coordinates [ML, AP, DV], shape (3,) or (N, 3)
        
        Returns
        -------
        np.ndarray
            Voxel coordinates, same shape as input
        """
        ras = self.stereo_to_ras(stereo_coords)
        return self.ras_to_voxel(ras)
    
    # Aliases for backwards compatibility
    def voxel_to_hc(self, voxel_coords: np.ndarray) -> np.ndarray:
        """Alias for voxel_to_stereo."""
        return self.voxel_to_stereo(voxel_coords)
    
    def hc_to_voxel(self, hc_coords: np.ndarray) -> np.ndarray:
        """Alias for stereo_to_voxel."""
        return self.stereo_to_voxel(hc_coords)

    def _compute_hc_affine(self) -> np.ndarray:
        """
        Compute affine matrix for HC coordinate space.
        
        The HC affine has the same rotation/scaling as the original,
        but with translation adjusted so the HC origin maps to [0, 0, 0].
        """
        import sys
           
        # Use the stored original stereotaxic zero (before it was changed to [0,0,0])
        affine_inv = np.linalg.inv(self.original_affine)
        ones = np.ones((1, 1))
        ras_hom = np.hstack([self.original_stereotax_zero_ras.reshape(1, -1), ones])
        hc_origin_voxel = (affine_inv @ ras_hom.T).T[0, :3]
        
        affine_hc = self.original_affine.copy()
        affine_hc[:3, 3] = -self.original_affine[:3, :3] @ hc_origin_voxel
        
        return affine_hc

    def get_hc_nifti(self) -> nib.Nifti1Image:
        """
        Create a NIfTI image with HC coordinate system.
        
        Returns
        -------
        nib.Nifti1Image
            NIfTI image with affine set to HC coordinates.
        """
        img = nib.load(str(self.nifti_path))
        header = img.header.copy()
        header['descrip'] = b'Horsley-Clarke coordinates (origin at interaural)'
        
        affine_hc = self._compute_hc_affine()
        
        return nib.Nifti1Image(
            img.get_fdata(),
            affine_hc,
            header
        )

    def save_hc_nifti(
        self,
        output_path: Optional[Union[str, Path]] = None,
        suffix: str = "_HC"
    ) -> Path:
        """
        Save NIfTI with HC coordinate system.
        
        Parameters
        ----------
        output_path : str or Path, optional
            Output file path. If not provided, saves to same directory
            as input with suffix appended.
        suffix : str, default="_HC"
            Suffix to append to filename (if output_path not provided).
        
        Returns
        -------
        Path
            Path to saved file.
        """
        if output_path is None:
            stem = self.nifti_path.stem
            if stem.endswith('.nii'):
                stem = stem[:-4]
            output_path = self.nifti_path.parent / f"{stem}{suffix}.nii"
        
        output_path = Path(output_path)
        
        img_hc = self.get_hc_nifti()
        nib.save(img_hc, output_path)
        
        return output_path
    # =========================================================================
    # VOI Handling
    # =========================================================================
    
    def load_voi(self, name_or_path: Union[str, Path]) -> VOI:
        """
        Load a VOI from XML file.
        
        Parameters
        ----------
        name_or_path : str or Path
            VOI filename, partial name, or full path
        
        Returns
        -------
        VOI
            Loaded VOI in voxel coordinates
        """
        path = Path(name_or_path)
        
        # If no suffix, search by partial name match
        if not path.suffix and self.voi_dir:
            name_lower = str(name_or_path).lower()
            for voi_file in self.voi_files:
                if name_lower in voi_file.lower():
                    path = self.voi_dir / voi_file
                    break
            else:
                raise FileNotFoundError(
                    f"No VOI file matching '{name_or_path}' found in {self.voi_files}"
                )
        # If has suffix but not absolute, prepend voi_dir
        elif not path.is_absolute() and self.voi_dir:
            path = self.voi_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"VOI file not found: {path}")
        
        return self._parse_voi_xml(path)
    
    def _parse_voi_xml(self, xml_path: Path) -> VOI:
        """Parse MIPAV VOI XML format."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get VOI name from filename
        voi_name = xml_path.stem
        
        # Parse color if present
        color = (255, 0, 0, 255)
        color_elem = root.find('.//Color')
        if color_elem is not None and color_elem.text:
            parts = color_elem.text.split(',')
            if len(parts) >= 4:
                color = tuple(int(p) for p in parts[:4])
        
        # Parse contours
        contours = []
        for contour_elem in root.findall('.//Contour'):
            slice_elem = contour_elem.find('Slice-number')
            if slice_elem is None or slice_elem.text is None:
                continue
            
            slice_idx = int(slice_elem.text)
            
            points = []
            for pt_elem in contour_elem.findall('Pt'):
                if pt_elem.text:
                    coords = pt_elem.text.split(',')
                    if len(coords) >= 2:
                        i, j = float(coords[0]), float(coords[1])
                        points.append([i, j])
            
            if points:
                contours.append(Contour(
                    slice_index=slice_idx,
                    points_ij=np.array(points)
                ))
        
        return VOI(name=voi_name, contours=contours, color=color)
    
    def convert_voi_to_stereo(self, voi: VOI) -> VOI_Stereo:
        """
        Convert VOI from voxel space to stereotaxic space.
        
        Parameters
        ----------
        voi : VOI
            VOI in voxel coordinates
        
        Returns
        -------
        VOI_Stereo
            VOI in stereotaxic coordinates
        """
        contours_stereo = []
        for contour in voi.contours:
            # Build ijk coordinates
            k = contour.slice_index
            ijk = np.column_stack([
                contour.points_ij,
                np.full(len(contour.points_ij), k)
            ])
            
            # Convert to stereotaxic
            stereo_pts = self.voxel_to_stereo(ijk)
            slice_dv = stereo_pts[0, 2] if len(stereo_pts) > 0 else 0
            
            contours_stereo.append({
                'slice_dv': slice_dv,
                'points_stereo': stereo_pts,
                'points_hc': stereo_pts  # Alias for compatibility
            })
        
        return VOI_Stereo(
            name=voi.name,
            contours_stereo=contours_stereo,
            original_voi=voi,
            color=voi.color
        )
    
    # Alias for backwards compatibility
    def convert_voi_to_hc(self, voi: VOI) -> VOI_Stereo:
        """Alias for convert_voi_to_stereo."""
        return self.convert_voi_to_stereo(voi)
    
    # =========================================================================
    # NIfTI Export
    # =========================================================================
    
    @property
    def stereo_nifti_path(self) -> Optional[Path]:
        """Path to stereotaxic-aligned NIfTI file."""
        if self.nifti_path:
            stem = self.nifti_path.stem
            if stem.endswith('_HC'):
                return self.nifti_path
            return self.nifti_path.parent / f"{stem}_HC.nii"
        return None
    
    @property
    def hc_nifti_path(self) -> Optional[Path]:
        """Alias for stereo_nifti_path."""
        return self.stereo_nifti_path
    
    def save_stereo_nifti(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save NIfTI with stereotaxic-aligned coordinates.
        
        The output NIfTI has an affine where voxel [0,0,0] maps to the
        minimum stereotaxic extent, and voxel * voxel_size gives
        stereotaxic coordinates directly.
        
        Parameters
        ----------
        output_path : str or Path, optional
            Output path. If None, uses default *_HC.nii naming.
        
        Returns
        -------
        Path
            Path to saved file
        """
        if output_path is None:
            output_path = self.stereo_nifti_path
        output_path = Path(output_path)
        
        # Calculate stereotaxic coordinates of voxel [0,0,0]
        origin_stereo = self.voxel_to_stereo(np.array([0, 0, 0]))
        
        # Build new affine: translation is origin in stereotaxic space
        # Keep same voxel size but make it positive (direct mapping)
        new_affine = np.eye(4)
        new_affine[0, 0] = self.voxel_size[0]
        new_affine[1, 1] = self.voxel_size[1]
        new_affine[2, 2] = self.voxel_size[2]
        new_affine[:3, 3] = origin_stereo
        
        # Create and save new NIfTI
        new_img = nib.Nifti1Image(self.nifti_data, new_affine)
        nib.save(new_img, str(output_path))
        
        print(f"Saved stereotaxic NIfTI to: {output_path}")
        return output_path
    
    # Alias
    # def save_hc_nifti(self, output_path: Optional[Union[str, Path]] = None) -> Path:
    #     """Alias for save_stereo_nifti."""
    #     return self.save_stereo_nifti(output_path)
    
    # =========================================================================
    # Chamber Access
    # =========================================================================
    
    def get_chamber(self, chamber_id: str) -> Optional[ChamberInfo]:
        """Get chamber by ID."""
        for chamber in self.chambers:
            if chamber.chamber_id == chamber_id:
                return chamber
        return None
    
    def get_chamber_by_name(self, name: str) -> Optional[ChamberInfo]:
        """Get chamber by name."""
        for chamber in self.chambers:
            if chamber.name == name:
                return chamber
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def load_subject(json_path: Union[str, Path]) -> StereotaxTransform:
    """
    Load subject configuration and return transform object.
    
    Parameters
    ----------
    json_path : str or Path
        Path to subject JSON file
    
    Returns
    -------
    StereotaxTransform
        Configured transform object
    """
    return StereotaxTransform.from_json(json_path)

# =============================================================================
# Visualization: Stereotax Triplanar Viewer
# =============================================================================

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox, Button
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class StereotaxTriplanarViewer:
    """
    Interactive triplanar viewer using stereotaxic (HC) coordinates.
    
    Uses *_HC.nii files directly where voxel * voxel_size = stereotaxic coords.
    
    Features:
    - Click on any view to set target position
    - Manual stereotaxic coordinate entry
    - VOI overlay from *_HC.xml files
    - Chamber disk visualization
    - Trajectory line drawing
    - Chamber coordinate calculation with reachability status
    
    Parameters
    ----------
    transform : StereotaxTransform
        Configured transform object with loaded NIfTI and chambers
    on_position_change : callable, optional
        Callback function(ml, ap, dv) called when position changes
    
    Example
    -------
    >>> transform = StereotaxTransform.from_json('58x.json')
    >>> viewer = StereotaxTriplanarViewer(transform)
    >>> viewer.show()
    """
    
    def __init__(self, transform: 'StereotaxTransform', on_position_change=None):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for StereotaxTriplanarViewer")

        self.transform = transform
        self.on_position_change = on_position_change
        
        # Ensure HC NIfTI exists
        if not transform.stereo_nifti_path.exists():
            print(f"Creating stereotaxic NIfTI: {transform.stereo_nifti_path}")
            transform.save_stereo_nifti()
        
        # Load HC-aligned NIfTI
        self._load_stereo_nifti()
        
        # Current position in stereotaxic coords [ML, AP, DV]
        self.position = np.array([0.0, 0.0, 0.0])
        
        # Load VOIs if available
        self.vois_stereo = []
        for voi_file in transform.voi_files:
            try:
                voi = transform.load_voi(voi_file)
                voi_stereo = transform.convert_voi_to_stereo(voi)
                self.vois_stereo.append(voi_stereo)
            except Exception as e:
                print(f"Warning: Could not load VOI {voi_file}: {e}")
        
        # Selected chamber (if any)
        self.selected_chamber = transform.chambers[0] if transform.chambers else None
        
        # Trajectory points (in stereo coords)
        self.trajectory = None
        
        # Initialize figure
        self._setup_figure()
    
    def _load_stereo_nifti(self):
        """Load the HC-aligned NIfTI file."""
        import nibabel as nib
        
        nii = nib.load(self.transform.stereo_nifti_path)
        self.volume = nii.get_fdata()
        self.voxel_size = np.array(nii.header.get_zooms()[:3])
        
        # In HC NIfTI: voxel * voxel_size = stereotaxic coords
        # Calculate coordinate bounds
        self.shape = self.volume.shape
        self.stereo_min = np.array([0, 0, 0]) * self.voxel_size  # Origin at voxel [0,0,0]
        self.stereo_max = (np.array(self.shape) - 1) * self.voxel_size

        # Get the true origin from the NIfTI affine
        self.nifti_origin = nii.affine[:3, 3]
        # Actually, the HC NIfTI has stereotax zero at a specific voxel
        # The origin in stereo coords depends on where stereotax_zero is
        # For simplicity, we'll work in voxel space and convert
        
    def _stereo_to_voxel_hc(self, stereo):
        """Convert stereotaxic coords to HC NIfTI voxel indices."""
        # In HC NIfTI, the stereotax zero is at the same voxel location
        # stereo = (voxel - stereo_zero_voxel) * voxel_size
        # voxel = stereo / voxel_size + stereo_zero_voxel
        stereo = np.asarray(stereo)
        voxel = stereo / self.voxel_size + self.transform.stereotax_zero_voxel
        return voxel
    
    def _voxel_hc_to_stereo(self, voxel):
        """Convert HC NIfTI voxel indices to stereotaxic coords."""
        voxel = np.asarray(voxel)
        stereo = (voxel - self.transform.stereotax_zero_voxel) * self.voxel_size
        return stereo
    
    def _setup_figure(self):
        """Create the figure with triplanar views and controls."""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(f'Stereotax Viewer: {self.transform.subject_id}', fontsize=14)
        
        # Create grid for plots and controls
        # Top row: 3 slice views
        # Bottom row: controls and info
        
        self.ax_axial = self.fig.add_subplot(2, 3, 1)      # Axial (top-down)
        self.ax_coronal = self.fig.add_subplot(2, 3, 2)    # Coronal (front view)
        self.ax_sagittal = self.fig.add_subplot(2, 3, 3)   # Sagittal (side view)
        
        # Info panel
        self.ax_info = self.fig.add_subplot(2, 3, 4)
        self.ax_info.axis('off')
        
        # Controls panel
        self.ax_controls = self.fig.add_subplot(2, 3, 5)
        self.ax_controls.axis('off')
        
        # 3D view (optional)
        self.ax_3d = self.fig.add_subplot(2, 3, 6, projection='3d')
        
        # Set up slice views
        self._setup_slice_views()
        
        # Set up coordinate entry - but do it AFTER figure is fully set up
        self.text_ml = None
        self.text_ap = None
        self.text_dv = None
        self.radio_chamber = None
        
        # Connect click events BEFORE setting up controls
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Delay control setup to avoid resize event issues
        self.fig.canvas.draw()
        self._setup_controls()
        
        # Initial update
        self._update_views()
    
    def _setup_slice_views(self):
        """Configure the slice view axes."""
        # Axial view: ML (x) vs AP (y), slice along DV
        self.ax_axial.set_xlabel('ML (mm, +right)')
        self.ax_axial.set_ylabel('AP (mm, +anterior)')
        self.ax_axial.set_title('Axial (top-down)')
        
        # Coronal view: ML (x) vs DV (y), slice along AP
        self.ax_coronal.set_xlabel('ML (mm, +right)')
        self.ax_coronal.set_ylabel('DV (mm, +dorsal)')
        self.ax_coronal.set_title('Coronal (front)')
        
        # Sagittal view: AP (x) vs DV (y), slice along ML
        self.ax_sagittal.set_xlabel('AP (mm, +anterior)')
        self.ax_sagittal.set_ylabel('DV (mm, +dorsal)')
        self.ax_sagittal.set_title('Sagittal (side)')
    
    def _setup_controls(self):
        """Set up coordinate entry text boxes."""
        # Use a safer approach - create axes but catch any errors
        try:
            box_width = 0.08
            box_height = 0.04
            
            # Position in lower left corner
            left_start = 0.08
            bottom_start = 0.08
            
            # Temporarily suppress warnings from tight_layout
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # ML entry
                ax_ml = self.fig.add_axes([left_start, bottom_start + 0.08, box_width, box_height])
                ax_ml.set_xticks([])
                ax_ml.set_yticks([])
                self.text_ml = TextBox(ax_ml, 'ML:', initial=f'{self.position[0]:.1f}')
                
                # AP entry
                ax_ap = self.fig.add_axes([left_start, bottom_start + 0.04, box_width, box_height])
                ax_ap.set_xticks([])
                ax_ap.set_yticks([])
                self.text_ap = TextBox(ax_ap, 'AP:', initial=f'{self.position[1]:.1f}')
                
                # DV entry
                ax_dv = self.fig.add_axes([left_start, bottom_start, box_width, box_height])
                ax_dv.set_xticks([])
                ax_dv.set_yticks([])
                self.text_dv = TextBox(ax_dv, 'DV:', initial=f'{self.position[2]:.1f}')

                # Submit button (goes after the DV entry)
                ax_submit = self.fig.add_axes([left_start + 0.10, bottom_start + 0.04, 0.06, box_height])
                self.btn_submit = Button(ax_submit, 'Submit')
                self.btn_submit.on_clicked(self._on_submit_coords)

                # Chamber selector (if multiple chambers)
                if len(self.transform.chambers) > 1:
                    ax_chamber = self.fig.add_axes([left_start + 0.12, bottom_start, 0.15, box_height])
                    from matplotlib.widgets import RadioButtons
                    chamber_names = [c.name for c in self.transform.chambers]
                    self.radio_chamber = RadioButtons(ax_chamber, chamber_names)
                    self.radio_chamber.on_clicked(self._on_chamber_select)
        except Exception as e:
            print(f"Warning: Could not set up all controls: {e}")
            self.text_ml = None
            self.text_ap = None
            self.text_dv = None
    
    def _on_ml_change(self, text):
        try:
            self.position[0] = float(text)
            self._update_views()
        except ValueError:
            pass
    
    def _on_ap_change(self, text):
        try:
            self.position[1] = float(text)
            self._update_views()
        except ValueError:
            pass
    
    def _on_dv_change(self, text):
        try:
            self.position[2] = float(text)
            self._update_views()
        except ValueError:
            pass
    
    def _on_chamber_select(self, label):
        for chamber in self.transform.chambers:
            if chamber.name == label:
                self.selected_chamber = chamber
                break
        self._update_views()
    
    def _on_click(self, event):
        """Handle click on slice views to update position."""
        # Ignore events that don't have inaxes (like ResizeEvent)
        if not hasattr(event, 'inaxes') or event.inaxes is None:
            return
        
        if event.inaxes == self.ax_axial:
            # Axial: x=ML, y=AP
            if event.xdata is not None and event.ydata is not None:
                self.position[0] = event.xdata  # ML
                self.position[1] = event.ydata  # AP
        elif event.inaxes == self.ax_coronal:
            # Coronal: x=ML, y=DV
            if event.xdata is not None and event.ydata is not None:
                self.position[0] = event.xdata  # ML
                self.position[2] = event.ydata  # DV
        elif event.inaxes == self.ax_sagittal:
            # Sagittal: x=AP, y=DV
            if event.xdata is not None and event.ydata is not None:
                self.position[1] = event.xdata  # AP
                self.position[2] = event.ydata  # DV
        else:
            return  # Click not on a slice view
        
        self._update_views()
        
        # Call callback if provided
        if self.on_position_change:
            self.on_position_change(*self.position)

    def _on_submit_coords(self, event):
        """Handle submit button click - read all three textboxes and update once."""
        try:
            ml = float(self.text_ml.text)
            ap = float(self.text_ap.text)
            dv = float(self.text_dv.text)
            self.position = np.array([ml, ap, dv])
            self._update_views()
        except ValueError:
            pass
    
    def _get_slice(self, axis, position_stereo):
        """Get a 2D slice from the volume at the given stereotaxic position.
        
        The _HC.nii file uses a coordinate system where:
        - voxel_origin = [88, 142, 21] corresponds to stereotaxic [0, 0, 0]
        - X and Y axes are flipped relative to voxel indices
        - Z axis is normal
        
        Conversion formulas:
        stereotaxic_x = (voxel_origin[0] - voxel_x) * voxel_size[0]
        stereotaxic_y = (voxel_origin[1] - voxel_y) * voxel_size[1]
        stereotaxic_z = (voxel_z - voxel_origin[2]) * voxel_size[2]
        """
        voxel_size = self.transform.voxel_size
        voxel_origin = np.array([88, 142, 21])
        
        # Convert stereotaxic position to voxel index
        if axis == 0:  # ML axis
            voxel_idx_float = voxel_origin[0] - (position_stereo / voxel_size[0])
        elif axis == 1:  # AP axis
            voxel_idx_float = voxel_origin[1] - (position_stereo / voxel_size[1])
        else:  # DV axis (axis == 2)
            voxel_idx_float = voxel_origin[2] + (position_stereo / voxel_size[2])
        
        idx = int(np.clip(voxel_idx_float, 0, self.shape[axis] - 1))
        
        if axis == 0:  # ML: slice shows [1]/AP x [2]/DV
            slice_data = self.volume[idx, :, :]
            
            # Rotate 90° counter-clockwise
            slice_data = np.rot90(slice_data, k=-1)
            
            # Create stereotaxic coordinates for the slice dimensions
            ap_voxels = np.arange(self.shape[1])
            dv_voxels = np.arange(self.shape[2])
            ap_stereo = (voxel_origin[1] - ap_voxels) * voxel_size[1]
            dv_stereo = (dv_voxels - voxel_origin[2]) * voxel_size[2]
            
            # extent = [xmin, xmax, ymin, ymax]
            extent = [ap_stereo[-1], ap_stereo[0], dv_stereo[0], dv_stereo[-1]]
            
        elif axis == 1:  # AP: slice shows [0]/ML x [2]/DV
            slice_data = self.volume[:, idx, :]
            
            # Rotate 90° counter-clockwise
            slice_data = np.rot90(slice_data, k=-1)
            
            # Create stereotaxic coordinates for the slice dimensions
            ml_voxels = np.arange(self.shape[0])
            dv_voxels = np.arange(self.shape[2])
            ml_stereo = (voxel_origin[0] - ml_voxels) * voxel_size[0]
            dv_stereo = (dv_voxels - voxel_origin[2]) * voxel_size[2]
            
            # extent = [xmin, xmax, ymin, ymax]
            extent = [ml_stereo[-1], ml_stereo[0], dv_stereo[0], dv_stereo[-1]]
            
        else:  # DV (axis == 2): slice shows [0]/ML x [1]/AP
            slice_data = self.volume[:, :, idx]
            
            # Rotate 90° counter-clockwise
            slice_data = np.rot90(slice_data, k=1)
            
            # Create stereotaxic coordinates for the slice dimensions
            ml_voxels = np.arange(self.shape[0])
            ap_voxels = np.arange(self.shape[1])
            ml_stereo = (voxel_origin[0] - ml_voxels) * voxel_size[0]
            ap_stereo = (voxel_origin[1] - ap_voxels) * voxel_size[1]
            
            # extent = [xmin, xmax, ymin, ymax]
            extent = [ml_stereo[-1], ml_stereo[0], ap_stereo[-1], ap_stereo[0]]
        
        return slice_data, extent
    
    def _update_views(self):
        """Update all slice views and info panel."""
        # Update text boxes safely
        self._update_textbox_values()
        # Update text boxes
        self.text_ml.set_val(f'{self.position[0]:.1f}')
        self.text_ap.set_val(f'{self.position[1]:.1f}')
        self.text_dv.set_val(f'{self.position[2]:.1f}')
        
        # Clear and redraw slice views
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.clear()
        
        self._setup_slice_views()
        # Draw axial slice (at current DV)
        slice_data, extent = self._get_slice(2, self.position[2])
        self.ax_axial.imshow(slice_data, cmap='gray', origin='lower', extent=extent, aspect='equal')
        self.ax_axial.axhline(self.position[1], color='yellow', linewidth=0.5, alpha=0.7)
        self.ax_axial.axvline(self.position[0], color='yellow', linewidth=0.5, alpha=0.7)
        self.ax_axial.plot(self.position[0], self.position[1], 'r+', markersize=10)
        
        # Draw coronal slice (at current AP)
        slice_data, extent = self._get_slice(1, self.position[1])
        self.ax_coronal.imshow(slice_data, cmap='gray', origin='lower', extent=extent, aspect='equal')
        self.ax_coronal.axhline(self.position[2], color='yellow', linewidth=0.5, alpha=0.7)
        self.ax_coronal.axvline(self.position[0], color='yellow', linewidth=0.5, alpha=0.7)
        self.ax_coronal.plot(self.position[0], self.position[2], 'r+', markersize=10)
        
        # Draw sagittal slice (at current ML)
        slice_data, extent = self._get_slice(0, self.position[0])
        self.ax_sagittal.imshow(slice_data, cmap='gray', origin='lower', extent=extent, aspect='equal')
        self.ax_sagittal.axhline(self.position[2], color='yellow', linewidth=0.5, alpha=0.7)
        self.ax_sagittal.axvline(self.position[1], color='yellow', linewidth=0.5, alpha=0.7)
        self.ax_sagittal.plot(self.position[1], self.position[2], 'r+', markersize=10)
        
        # Draw VOI contours
        # self._draw_vois()
        
        # Draw chamber
        self._draw_chamber()
        
        # Draw trajectory if set
        if self.trajectory is not None:
            self._draw_trajectory()
        
        # Update info panel
        self._update_info()
        
        # Update 3D view
        self._update_3d()
        
        self.fig.canvas.draw_idle()
    
    def _draw_vois(self):
        """Draw VOI contours on slice views."""
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.vois_stereo)))
        
        for voi_stereo, color in zip(self.vois_stereo, colors):
            for contour in voi_stereo.contours_stereo:
                points = contour['points_stereo']  # Nx3 array [ML, AP, DV]
                
                # Draw on axial if contour is near current DV
                dv_vals = points[:, 2]
                if np.min(dv_vals) <= self.position[2] <= np.max(dv_vals):
                    # Find points close to current DV slice
                    mask = np.abs(dv_vals - self.position[2]) < self.voxel_size[2]
                    if np.any(mask):
                        self.ax_axial.plot(points[mask, 0], points[mask, 1], 
                                          '.', color=color, markersize=2, alpha=0.7)
                
                # Draw on coronal if contour is near current AP
                ap_vals = points[:, 1]
                if np.min(ap_vals) <= self.position[1] <= np.max(ap_vals):
                    mask = np.abs(ap_vals - self.position[1]) < self.voxel_size[1]
                    if np.any(mask):
                        self.ax_coronal.plot(points[mask, 0], points[mask, 2],
                                            '.', color=color, markersize=2, alpha=0.7)
                
                # Draw on sagittal if contour is near current ML
                ml_vals = points[:, 0]
                if np.min(ml_vals) <= self.position[0] <= np.max(ml_vals):
                    mask = np.abs(ml_vals - self.position[0]) < self.voxel_size[0]
                    if np.any(mask):
                        self.ax_sagittal.plot(points[mask, 1], points[mask, 2],
                                             '.', color=color, markersize=2, alpha=0.7)
    
  
    def _draw_chamber(self):
        """Draw chamber disk on slice views."""
        if self.selected_chamber is None:
            return
        
        chamber = self.selected_chamber
        
        # Get chamber disk edge points
        disk_edge = chamber.get_disk_edge(n_points=100)
        
        # Draw chamber disk edge on axial view
        self.ax_axial.plot(disk_edge[:, 0], disk_edge[:, 1], 'c-', linewidth=1, alpha=0.7)
        
        # Draw chamber center
        self.ax_axial.plot(chamber.stereo_location[0], chamber.stereo_location[1], 
                        'co', markersize=8, markerfacecolor='none', linewidth=2)
        
        # Draw on coronal view
        self.ax_coronal.plot(disk_edge[:, 0], disk_edge[:, 2], 'c-', linewidth=1, alpha=0.7)
        self.ax_coronal.plot(chamber.stereo_location[0], chamber.stereo_location[2],
                            'co', markersize=8, markerfacecolor='none', linewidth=2)
        
        # Draw on sagittal view
        self.ax_sagittal.plot(disk_edge[:, 1], disk_edge[:, 2], 'c-', linewidth=1, alpha=0.7)
        self.ax_sagittal.plot(chamber.stereo_location[1], chamber.stereo_location[2],
                            'co', markersize=8, markerfacecolor='none', linewidth=2)
    
    def _draw_trajectory(self):
        """Draw trajectory line on slice views."""
        if self.trajectory is None:
            return
        
        traj = self.trajectory  # Nx3 array [ML, AP, DV]
        
        # Draw on each view
        self.ax_axial.plot(traj[:, 0], traj[:, 1], 'g-', linewidth=1, alpha=0.7)
        self.ax_coronal.plot(traj[:, 0], traj[:, 2], 'g-', linewidth=1, alpha=0.7)
        self.ax_sagittal.plot(traj[:, 1], traj[:, 2], 'g-', linewidth=1, alpha=0.7)
    
    def _update_info(self):
        """Update the info panel with current position and chamber info."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_lines = [
            f"Position (stereo): ML={self.position[0]:.1f}, AP={self.position[1]:.1f}, DV={self.position[2]:.1f}",
        ]
        
        if self.selected_chamber:
            chamber = self.selected_chamber
            # Calculate chamber coordinates
            result = chamber.find_entry_point(self.position)
            
            if result['reachable']:
                info_lines.append(f"")
                info_lines.append(f"Chamber: {chamber.name}")
                info_lines.append(f"  Grid X: {result['chamber_x']:.2f} mm")
                info_lines.append(f"  Grid Y: {result['chamber_y']:.2f} mm")
                info_lines.append(f"  Depth:  {result['depth']:.2f} mm")
                info_lines.append(f"  Status: REACHABLE ✓")
                
                # Set trajectory using chamber coordinates
                self.trajectory = chamber.get_trajectory(
                    result['chamber_x'], 
                    result['chamber_y'], 
                    depth_range=(-5, result['depth'] + 5),
                    n_points=50
                )
            else:
                info_lines.append(f"")
                info_lines.append(f"Chamber: {chamber.name}")
                info_lines.append(f"  Status: NOT REACHABLE ✗")
                info_lines.append(f"  Distance from center: {result['radial_distance']:.1f} mm")
                self.trajectory = None
        
        text = '\n'.join(info_lines)
        self.ax_info.text(0.05, 0.95, text, transform=self.ax_info.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

    def _update_textbox_values(self):
        """Safely update text box values without triggering events."""
        if self.text_ml is not None:
            try:
                self.text_ml.set_val(f'{self.position[0]:.1f}')
            except:
                pass
        if self.text_ap is not None:
            try:
                self.text_ap.set_val(f'{self.position[1]:.1f}')
            except:
                pass
        if self.text_dv is not None:
            try:
                self.text_dv.set_val(f'{self.position[2]:.1f}')
            except:
                pass
    
    def _update_3d(self):
        """Update the 3D view with current position and trajectory."""
        self.ax_3d.clear()
        self.ax_3d.set_xlabel('ML (mm)')
        self.ax_3d.set_ylabel('AP (mm)')
        self.ax_3d.set_zlabel('DV (mm)')
        self.ax_3d.set_title('3D View')
        
        legend_elements = []
        
        # Draw current position
        self.ax_3d.scatter(*self.position, c='red', s=100, marker='o')
        
        # Draw chamber
        if self.selected_chamber:
            chamber = self.selected_chamber
            X, Y, Z = chamber.get_disk_points(n_radial=10, n_angular=24)
            self.ax_3d.plot_surface(X, Y, Z, alpha=0.3, color='cyan')
        
        # Draw trajectory
        if self.trajectory is not None:
            self.ax_3d.plot(self.trajectory[:, 0], self.trajectory[:, 1], 
                            self.trajectory[:, 2], 'g-', linewidth=2)
        
        for voi_stereo in self.vois_stereo:
            # Get color from VOI XML (RGBA format 0-255)
            if hasattr(voi_stereo, 'color') and voi_stereo.color:
                # Convert from 0-255 to 0-1
                color = tuple(c / 255.0 for c in voi_stereo.color)
            else:
                # Fallback to default color if not available
                color = (0.5, 0.5, 0.5, 1.0)  # gray
            
            points = voi_stereo.all_points_stereo
            if len(points) > 100:
                idx = np.random.choice(len(points), 100, replace=False)
                points = points[idx]
            
            if len(points) > 0:
                # Draw VOI points
                voi_scatter = self.ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                    s=2, alpha=0.6, c=[color])
                
                # Create legend label from VOI name
                filename = voi_stereo.name
                voi_name = filename.rsplit('_', 1)[-1]
                legend_elements.append(Patch(facecolor=color, label=voi_name))

        # Add legend with only VOI entries
        if legend_elements:
            self.ax_3d.legend(
                handles=legend_elements,
                loc='upper left',
                fontsize=9,
                framealpha=0.9,
                edgecolor='black'
            )
    
    def set_position(self, ml, ap, dv):
        """Set the current stereotaxic position.
        
        Parameters
        ----------
        ml : float
            Mediolateral position (mm, +right)
        ap : float
            Anteroposterior position (mm, +anterior)
        dv : float
            Dorsoventral position (mm, +dorsal)
        """
        self.position = np.array([ml, ap, dv])
        self._update_views()
    
    def show(self):
        """Display the viewer."""
        # Use subplots_adjust instead of tight_layout to avoid widget issues
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12, 
                                hspace=0.4, wspace=0.35)
        plt.show()
    
    def get_figure(self):
        """Return the matplotlib figure for embedding in other GUIs."""
        return self.fig
    
# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test with 58x
    json_path = Path(r'C:\Users\eemeric2\OneDrive - Johns Hopkins\Documents\GitHub\NPT\database\58x\58x.json')
    
    print("Loading subject configuration...")
    transform = StereotaxTransform.from_json(json_path)
    
    print(f"\nSubject: {transform.subject_id}")
    print(f"NIfTI shape: {transform.shape}")
    print(f"Voxel size: {transform.voxel_size}")
    print(f"Stereotax zero (RAS): {transform.stereotax_zero_ras}")
    print(f"Stereotax zero (voxel): {transform.stereotax_zero_voxel}")
    
    # Test coordinate conversion
    print("\n--- Coordinate Conversion Tests ---")
    origin_voxel = transform.stereotax_zero_voxel
    origin_stereo = transform.voxel_to_stereo(origin_voxel)
    print(f"Origin voxel {origin_voxel} -> stereo {origin_stereo}")
    print(f"  (Should be [0, 0, 0])")
    
    # Test chamber
    if transform.chambers:
        chamber = transform.chambers[0]
        print(f"\n--- Chamber: {chamber.name} ---")
        print(f"Location (stereo): {chamber.stereo_location}")
        print(f"Tilt: {chamber.tilt_degrees}")
        
        # Test forward transform
        test_chamber = [0, 0, 20]  # Center, 20mm deep
        stereo = chamber.chamber_to_stereo(*test_chamber)
        print(f"\nChamber {test_chamber} -> stereo {stereo}")
        
        # Test inverse transform
        result = chamber.find_entry_point(stereo)
        print(f"Inverse: chamber_xy=[{result['chamber_x']:.3f}, {result['chamber_y']:.3f}], depth={result['depth']:.3f}")
        print(f"Reachable: {result['reachable']}")
    
    # Test VOI loading
    vois_stereo = []
    if transform.voi_files:
        print(f"\n--- VOI Loading ---")
        for voi_file in transform.voi_files:
            voi = transform.load_voi(voi_file)
            voi_stereo = transform.convert_voi_to_stereo(voi)
            vois_stereo.append(voi_stereo)
            print(f"{voi.name}: {len(voi.contours)} contours, {len(voi_stereo.all_points_stereo)} points")
    
    print("\n--- All coordinate tests passed! ---")
    
    # Launch triplanar viewer
    print("\n--- Launching Triplanar Viewer ---")
    
    def on_position_change(ml, ap, dv):
        print(f"Position: ML={ml:.2f}, AP={ap:.2f}, DV={dv:.2f}")
    
    # Create viewer - it manages its own window
    viewer = StereotaxTriplanarViewer(
        transform=transform,
        on_position_change=on_position_change
    )
    # DON'T call tight_layout - use viewer.show() which handles layout
    viewer.show()