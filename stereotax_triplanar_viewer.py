# neurophys_tracker/visualization/stereotax_triplanar_viewer.py
"""
Triplanar MRI viewer using HC-transformed (stereotaxic) NIfTI files.

This viewer expects:
- *_HC.nii files where voxel coordinates map directly to stereotaxic space
- *_HC.xml VOI files already in stereotaxic coordinates
- Origin at stereotaxic [0, 0, 0]
"""

import numpy as np
import nibabel as nib
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Union

# Import our coordinate transform module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from stereotax_transform import HCTransform, ChamberInfo, VOI_HC


@dataclass
class CrosshairPosition:
    """Stereotaxic coordinates [ML, AP, DV]."""
    ml: float  # medial-lateral (X)
    ap: float  # anterior-posterior (Y)
    dv: float  # dorsal-ventral (Z)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.ml, self.ap, self.dv])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CrosshairPosition':
        return cls(ml=arr[0], ap=arr[1], dv=arr[2])


class StereotaxTriplanarViewer:
    """
    Interactive triplanar MRI viewer in stereotaxic coordinates.
    
    Uses HC-transformed NIfTI files where:
    - Voxel [i, j, k] * voxel_size = stereotaxic [ML, AP, DV]
    - Origin is at stereotaxic [0, 0, 0]
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        transform: StereotaxTransform,
        chamber: Optional[ChamberInfo] = None,
        vois: Optional[List[VOI_HC]] = None,
        on_position_change: Optional[Callable[[CrosshairPosition], None]] = None,
        title: str = "Stereotaxic Triplanar Viewer"
    ):
        """
        Initialize the viewer.
        
        Parameters
        ----------
        parent : tk.Widget
            Parent Tkinter widget
        transform : StereotaxTransform
            Loaded transform object with HC NIfTI data
        chamber : ChamberInfo, optional
            Chamber for trajectory calculations
        vois : list of VOI_HC, optional
            VOIs to overlay (already in stereotaxic coords)
        on_position_change : callable, optional
            Callback when crosshair position changes
        title : str
            Window title
        """
        self.parent = parent
        self.transform = transform
        self.chamber = chamber
        self.vois = vois or []
        self.on_position_change = on_position_change
        
        # Load HC NIfTI data
        self._load_hc_nifti()
        
        # Initialize crosshair at origin [0, 0, 0]
        self.current_position = CrosshairPosition(ml=0.0, ap=0.0, dv=0.0)
        
        # Trajectory line storage
        self.trajectory_lines = {'sagittal': None, 'coronal': None, 'axial': None}
        self.crosshair_lines = {
            'sagittal': {'h': None, 'v': None},
            'coronal': {'h': None, 'v': None},
            'axial': {'h': None, 'v': None}
        }
        
        # Chamber disk lines
        self.chamber_disk_lines = {'sagittal': None, 'coronal': None, 'axial': None}
        
        # Build UI
        self._create_widgets(title)
        self._plot_initial()
        self._draw_crosshairs()
        self._draw_vois()
        if self.chamber:
            self._draw_chamber()
        
        # Trigger initial callback
        if self.on_position_change:
            self.on_position_change(self.current_position)
    
    def _load_hc_nifti(self):
        """Load HC-transformed NIfTI file."""
        hc_path = self.transform.hc_nifti_path
        
        if hc_path is None or not hc_path.exists():
            # Create HC NIfTI if it doesn't exist
            print("HC NIfTI not found, creating...")
            hc_path = self.transform.save_hc_nifti()
        
        img = nib.load(str(hc_path))
        self.mri_data = img.get_fdata()
        
        # For HC NIfTI, the affine should give us direct stereotaxic coordinates
        # Voxel [0,0,0] should be at some stereotaxic coordinate
        # Voxel spacing from affine diagonal
        affine = img.affine
        self.voxel_size = np.abs(np.diag(affine)[:3])
        
        # Origin: where does voxel [0,0,0] map to in stereotaxic?
        # For HC-aligned, this should be the minimum stereotaxic extent
        self.origin = affine[:3, 3]
        
        # Calculate stereotaxic extent
        shape = np.array(self.mri_data.shape)
        self.stereo_min = self.origin
        self.stereo_max = self.origin + shape * self.voxel_size
        
        print(f"HC NIfTI loaded: shape={self.mri_data.shape}")
        print(f"Voxel size: {self.voxel_size}")
        print(f"Stereotaxic extent: [{self.stereo_min}] to [{self.stereo_max}]")
    
    def _stereotax_to_voxel(self, stereo: np.ndarray) -> np.ndarray:
        """Convert stereotaxic [ML, AP, DV] to voxel indices."""
        voxel = (stereo - self.origin) / self.voxel_size
        return np.round(voxel).astype(int)
    
    def _voxel_to_stereotax(self, voxel: np.ndarray) -> np.ndarray:
        """Convert voxel indices to stereotaxic [ML, AP, DV]."""
        return self.origin + voxel * self.voxel_size
    
    def _create_widgets(self, title: str):
        """Create the viewer widgets."""
        # Main frame
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with 3 subplots
        self.fig = Figure(figsize=(15, 5))
        self.axes = {
            'sagittal': self.fig.add_subplot(131),
            'coronal': self.fig.add_subplot(132),
            'axial': self.fig.add_subplot(133)
        }
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect click handler
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Toolbar
        toolbar_frame = ttk.Frame(self.main_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)
        
        # Coordinate display/entry frame
        self._create_coord_frame()
        
        # Chamber results frame (if chamber provided)
        if self.chamber:
            self._create_chamber_results_frame()
    
    def _create_coord_frame(self):
        """Create coordinate display and entry widgets."""
        coord_frame = ttk.LabelFrame(self.main_frame, text="Stereotaxic Coordinates", padding=10)
        coord_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # ML entry
        ttk.Label(coord_frame, text="ML:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.ml_var = tk.StringVar(value=f"{self.current_position.ml:.2f}")
        self.ml_entry = ttk.Entry(coord_frame, textvariable=self.ml_var, width=10)
        self.ml_entry.grid(row=0, column=1, padx=5)
        self.ml_entry.bind('<Return>', self._on_manual_input)
        
        # AP entry
        ttk.Label(coord_frame, text="AP:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.ap_var = tk.StringVar(value=f"{self.current_position.ap:.2f}")
        self.ap_entry = ttk.Entry(coord_frame, textvariable=self.ap_var, width=10)
        self.ap_entry.grid(row=0, column=3, padx=5)
        self.ap_entry.bind('<Return>', self._on_manual_input)
        
        # DV entry
        ttk.Label(coord_frame, text="DV:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.dv_var = tk.StringVar(value=f"{self.current_position.dv:.2f}")
        self.dv_entry = ttk.Entry(coord_frame, textvariable=self.dv_var, width=10)
        self.dv_entry.grid(row=0, column=5, padx=5)
        self.dv_entry.bind('<Return>', self._on_manual_input)
        
        ttk.Label(coord_frame, text="mm").grid(row=0, column=6, sticky=tk.W, padx=5)
        
        # Go button
        ttk.Button(coord_frame, text="Go", command=self._on_manual_input).grid(
            row=0, column=7, padx=10
        )
    
    def _create_chamber_results_frame(self):
        """Create frame showing chamber coordinate results."""
        self.chamber_frame = ttk.LabelFrame(
            self.main_frame, 
            text=f"Chamber: {self.chamber.name}", 
            padding=10
        )
        self.chamber_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Chamber coordinates
        ttk.Label(self.chamber_frame, text="Chamber X:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.chamber_x_var = tk.StringVar(value="---")
        ttk.Label(self.chamber_frame, textvariable=self.chamber_x_var, 
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.chamber_frame, text="Chamber Y:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.chamber_y_var = tk.StringVar(value="---")
        ttk.Label(self.chamber_frame, textvariable=self.chamber_y_var,
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=3, padx=5)
        
        ttk.Label(self.chamber_frame, text="Depth:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.depth_var = tk.StringVar(value="---")
        ttk.Label(self.chamber_frame, textvariable=self.depth_var,
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=5, padx=5)
        
        ttk.Label(self.chamber_frame, text="mm").grid(row=0, column=6, sticky=tk.W)
        
        # Status
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self.chamber_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=7, padx=20)
    
    def _plot_initial(self):
        """Plot initial MRI slices."""
        # Get voxel indices for current position
        voxel = self._stereotax_to_voxel(self.current_position.to_array())
        voxel = np.clip(voxel, 0, np.array(self.mri_data.shape) - 1)
        
        # Calculate extents for imshow
        self.extents = {
            'sagittal': [self.stereo_min[1], self.stereo_max[1], 
                         self.stereo_min[2], self.stereo_max[2]],  # AP, DV
            'coronal': [self.stereo_min[0], self.stereo_max[0], 
                        self.stereo_min[2], self.stereo_max[2]],   # ML, DV
            'axial': [self.stereo_min[0], self.stereo_max[0], 
                      self.stereo_min[1], self.stereo_max[1]]      # ML, AP
        }
        
        # Sagittal (YZ plane, fixed X/ML)
        ax = self.axes['sagittal']
        slice_data = self.mri_data[voxel[0], :, :].T
        self.images = {}
        self.images['sagittal'] = ax.imshow(
            slice_data, cmap='gray', aspect='auto', origin='lower',
            extent=self.extents['sagittal']
        )
        ax.set_xlabel('AP (mm)')
        ax.set_ylabel('DV (mm)')
        ax.set_title(f'Sagittal (ML={self.current_position.ml:.1f})')
        
        # Coronal (XZ plane, fixed Y/AP)
        ax = self.axes['coronal']
        slice_data = self.mri_data[:, voxel[1], :].T
        self.images['coronal'] = ax.imshow(
            slice_data, cmap='gray', aspect='auto', origin='lower',
            extent=self.extents['coronal']
        )
        ax.set_xlabel('ML (mm)')
        ax.set_ylabel('DV (mm)')
        ax.set_title(f'Coronal (AP={self.current_position.ap:.1f})')
        
        # Axial (XY plane, fixed Z/DV)
        ax = self.axes['axial']
        slice_data = self.mri_data[:, :, voxel[2]].T
        self.images['axial'] = ax.imshow(
            slice_data, cmap='gray', aspect='auto', origin='lower',
            extent=self.extents['axial']
        )
        ax.set_xlabel('ML (mm)')
        ax.set_ylabel('AP (mm)')
        ax.set_title(f'Axial (DV={self.current_position.dv:.1f})')
        
        self.fig.tight_layout()
    
    def _update_slices(self):
        """Update MRI slices for current position."""
        voxel = self._stereotax_to_voxel(self.current_position.to_array())
        voxel = np.clip(voxel, 0, np.array(self.mri_data.shape) - 1)
        
        # Update image data
        self.images['sagittal'].set_data(self.mri_data[voxel[0], :, :].T)
        self.images['coronal'].set_data(self.mri_data[:, voxel[1], :].T)
        self.images['axial'].set_data(self.mri_data[:, :, voxel[2]].T)
        
        # Update titles
        self.axes['sagittal'].set_title(f'Sagittal (ML={self.current_position.ml:.1f})')
        self.axes['coronal'].set_title(f'Coronal (AP={self.current_position.ap:.1f})')
        self.axes['axial'].set_title(f'Axial (DV={self.current_position.dv:.1f})')
    
    def _draw_crosshairs(self):
        """Draw crosshairs on all views."""
        pos = self.current_position
        
        # Remove old crosshairs
        for view in self.crosshair_lines:
            for line in self.crosshair_lines[view].values():
                if line is not None:
                    line.remove()
        
        # Sagittal view (AP-DV plane)
        ax = self.axes['sagittal']
        self.crosshair_lines['sagittal']['h'] = ax.axhline(
            y=pos.dv, color='red', linewidth=0.8, alpha=0.7
        )
        self.crosshair_lines['sagittal']['v'] = ax.axvline(
            x=pos.ap, color='red', linewidth=0.8, alpha=0.7
        )
        
        # Coronal view (ML-DV plane)
        ax = self.axes['coronal']
        self.crosshair_lines['coronal']['h'] = ax.axhline(
            y=pos.dv, color='red', linewidth=0.8, alpha=0.7
        )
        self.crosshair_lines['coronal']['v'] = ax.axvline(
            x=pos.ml, color='red', linewidth=0.8, alpha=0.7
        )
        
        # Axial view (ML-AP plane)
        ax = self.axes['axial']
        self.crosshair_lines['axial']['h'] = ax.axhline(
            y=pos.ap, color='red', linewidth=0.8, alpha=0.7
        )
        self.crosshair_lines['axial']['v'] = ax.axvline(
            x=pos.ml, color='red', linewidth=0.8, alpha=0.7
        )
        
        self._update_slices()
        self.canvas.draw_idle()
    
    def _draw_vois(self):
        """Draw VOI contours on all views."""
        # Color map for different VOIs
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        
        for i, voi in enumerate(self.vois):
            color = colors[i % len(colors)]
            
            for contour in voi.contours_hc:
                pts = np.array(contour['points_hc'])
                if len(pts) < 2:
                    continue
                
                # Plot on each view
                # Sagittal (AP-DV)
                self.axes['sagittal'].plot(
                    pts[:, 1], pts[:, 2], color=color, alpha=0.5, linewidth=0.5
                )
                # Coronal (ML-DV)
                self.axes['coronal'].plot(
                    pts[:, 0], pts[:, 2], color=color, alpha=0.5, linewidth=0.5
                )
                # Axial (ML-AP)
                self.axes['axial'].plot(
                    pts[:, 0], pts[:, 1], color=color, alpha=0.5, linewidth=0.5
                )
    
    def _draw_chamber(self):
        """Draw chamber disk on all views."""
        if not self.chamber:
            return
        
        # Get chamber disk edge points
        X, Y, Z = self.chamber.get_disk_points(n_radial=2, n_angular=50)
        edge_ml = X[-1, :]
        edge_ap = Y[-1, :]
        edge_dv = Z[-1, :]
        
        # Plot on each view
        self.axes['sagittal'].plot(edge_ap, edge_dv, 'purple', linewidth=2, alpha=0.7)
        self.axes['coronal'].plot(edge_ml, edge_dv, 'purple', linewidth=2, alpha=0.7)
        self.axes['axial'].plot(edge_ml, edge_ap, 'purple', linewidth=2, alpha=0.7)
        
        # Mark chamber center
        loc = self.chamber.stereo_location
        self.axes['sagittal'].plot(loc[1], loc[2], 'p', color='purple', markersize=8)
        self.axes['coronal'].plot(loc[0], loc[2], 'p', color='purple', markersize=8)
        self.axes['axial'].plot(loc[0], loc[1], 'p', color='purple', markersize=8)
    
    def _draw_trajectory(self, entry: np.ndarray, target: np.ndarray):
        """Draw trajectory line from entry to target."""
        # Remove old trajectory
        for view, line in self.trajectory_lines.items():
            if line is not None:
                line.remove()
        
        # Sagittal (AP-DV)
        self.trajectory_lines['sagittal'], = self.axes['sagittal'].plot(
            [entry[1], target[1]], [entry[2], target[2]],
            'g-', linewidth=2, alpha=0.8
        )
        
        # Coronal (ML-DV)
        self.trajectory_lines['coronal'], = self.axes['coronal'].plot(
            [entry[0], target[0]], [entry[2], target[2]],
            'g-', linewidth=2, alpha=0.8
        )
        
        # Axial (ML-AP)
        self.trajectory_lines['axial'], = self.axes['axial'].plot(
            [entry[0], target[0]], [entry[1], target[1]],
            'g-', linewidth=2, alpha=0.8
        )
        
        self.canvas.draw_idle()
    
    def _on_click(self, event):
        """Handle mouse click to move crosshairs."""
        if event.inaxes is None:
            return
        
        pos = self.current_position
        
        if event.inaxes == self.axes['sagittal']:
            # Sagittal: AP and DV change, ML stays
            self.current_position = CrosshairPosition(
                ml=pos.ml, ap=event.xdata, dv=event.ydata
            )
        elif event.inaxes == self.axes['coronal']:
            # Coronal: ML and DV change, AP stays
            self.current_position = CrosshairPosition(
                ml=event.xdata, ap=pos.ap, dv=event.ydata
            )
        elif event.inaxes == self.axes['axial']:
            # Axial: ML and AP change, DV stays
            self.current_position = CrosshairPosition(
                ml=event.xdata, ap=event.ydata, dv=pos.dv
            )
        else:
            return
        
        self._update_display()
    
    def _on_manual_input(self, event=None):
        """Handle manual coordinate entry."""
        try:
            ml = float(self.ml_var.get())
            ap = float(self.ap_var.get())
            dv = float(self.dv_var.get())
            
            # Validate bounds
            if not (self.stereo_min[0] <= ml <= self.stereo_max[0] and
                    self.stereo_min[1] <= ap <= self.stereo_max[1] and
                    self.stereo_min[2] <= dv <= self.stereo_max[2]):
                messagebox.showwarning("Out of Bounds", 
                    "Coordinates are outside MRI volume.")
                return
            
            self.current_position = CrosshairPosition(ml=ml, ap=ap, dv=dv)
            self._update_display()
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers.")
    
    def _update_display(self):
        """Update all display elements for current position."""
        # Update coordinate entries
        self.ml_var.set(f"{self.current_position.ml:.2f}")
        self.ap_var.set(f"{self.current_position.ap:.2f}")
        self.dv_var.set(f"{self.current_position.dv:.2f}")
        
        # Update crosshairs and slices
        self._draw_crosshairs()
        
        # Update chamber calculations if chamber present
        if self.chamber:
            self._update_chamber_coords()
        
        # Trigger callback
        if self.on_position_change:
            self.on_position_change(self.current_position)
    
    def _update_chamber_coords(self):
        """Calculate and display chamber coordinates for current position."""
        target = self.current_position.to_array()
        result = self.chamber.find_entry_point(target)
        
        if result['reachable']:
            self.chamber_x_var.set(f"{result['chamber_x']:.2f}")
            self.chamber_y_var.set(f"{result['chamber_y']:.2f}")
            self.depth_var.set(f"{result['depth']:.2f}")
            self.status_var.set("✓ Reachable")
            self.status_label.config(foreground="green")
            
            # Draw trajectory
            self._draw_trajectory(result['entry_point_hc'], target)
        else:
            self.chamber_x_var.set("---")
            self.chamber_y_var.set("---")
            self.depth_var.set("---")
            self.status_var.set(f"✗ Not reachable ({result['radial_distance']:.1f}mm from center)")
            self.status_label.config(foreground="red")
            
            # Clear trajectory
            for view, line in self.trajectory_lines.items():
                if line is not None:
                    line.remove()
                    self.trajectory_lines[view] = None
            self.canvas.draw_idle()
    
    def set_position(self, position: CrosshairPosition):
        """Programmatically set crosshair position."""
        self.current_position = position
        self._update_display()
    
    def get_result(self) -> Optional[Dict]:
        """Get current chamber coordinates if reachable."""
        if not self.chamber:
            return None
        
        result = self.chamber.find_entry_point(self.current_position.to_array())
        if result['reachable']:
            return {
                'chamber_x': result['chamber_x'],
                'chamber_y': result['chamber_y'],
                'depth': result['depth'],
                'target_stereotaxic': self.current_position.to_array(),
                'entry_point': result['entry_point_hc']
            }
        return None


# Standalone test
if __name__ == "__main__":
    from pathlib import Path
    
    # Test with 58x
    json_path = Path(r'C:\Users\eemeric2\OneDrive - Johns Hopkins\Documents\GitHub\NPT\database\58x\58x.json')
    
    # Load transform
    transform = StereotaxTransform.from_json(json_path)
    
    # Load VOIs
    vois_hc = [transform.convert_voi_to_hc(transform.load_voi(f)) for f in transform.voi_files]
    
    # Get chamber
    chamber = transform.chambers[0] if transform.chambers else None
    
    # Create Tk window
    root = tk.Tk()
    root.title("Stereotax Triplanar Viewer Test")
    root.geometry("1400x800")
    
    def on_position_change(pos):
        print(f"Position: ML={pos.ml:.2f}, AP={pos.ap:.2f}, DV={pos.dv:.2f}")
    
    viewer = StereotaxTriplanarViewer(
        parent=root,
        transform=transform,
        chamber=chamber,
        vois=vois_hc,
        on_position_change=on_position_change
    )
    
    root.mainloop()