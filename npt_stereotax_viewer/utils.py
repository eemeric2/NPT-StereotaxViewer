"""
Utility data classes for stereotaxic coordinate system.

Defines data structures for VOIs (Volumes of Interest), contours, and recording chambers.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import xml.etree.ElementTree as ET


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