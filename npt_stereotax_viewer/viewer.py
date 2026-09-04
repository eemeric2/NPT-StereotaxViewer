# NPT-StereotaxViewer/npt_stereotax_viewer/viewer.py
"""
StereotaxTriplanarViewer: Interactive 3D triplanar viewer for stereotaxic MRI.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple

from .transform import StereotaxTransform


class StereotaxTriplanarViewer:
    """
    Interactive triplanar viewer using stereotaxic coordinates.
    
    Features:
    - Click on any view to set target position
    - Manual stereotaxic coordinate entry
    - Chamber disk visualization
    - VOI overlay
    - Synchronized crosshairs
    """
    
    def __init__(self, transform: StereotaxTransform, init_stereo: Tuple[float, float, float] = (0, 0, 0)):
        self.transform = transform
        
        # Current position in stereotaxic coords [ML, AP, DV]
        self.position = np.array(init_stereo, dtype=np.float64)
        
        # Load volume
        self._load_stereo_volume()
        
        # Initialize figure
        self._setup_figure()
    
    def _load_stereo_volume(self):
        """Load the stereotaxic-aligned volume."""
        self.volume = self.transform.data
        self.voxel_size = np.array(self.transform.voxel_size)
        self.voxel_origin = np.array(self.transform.voxel_origin)
        self.shape = self.volume.shape
    
    def _stereo_to_voxel(self, stereo):
        """Convert stereotaxic coords to voxel indices."""
        stereo = np.asarray(stereo)
        voxel = np.zeros(3)
        voxel[0] = self.voxel_origin[0] - stereo[0] / self.voxel_size[0]
        voxel[1] = self.voxel_origin[1] - stereo[1] / self.voxel_size[1]
        voxel[2] = self.voxel_origin[2] + stereo[2] / self.voxel_size[2]
        return voxel
    
    def _voxel_to_stereo(self, voxel):
        """Convert voxel indices to stereotaxic coords."""
        voxel = np.asarray(voxel)
        stereo = np.zeros(3)
        stereo[0] = (self.voxel_origin[0] - voxel[0]) * self.voxel_size[0]
        stereo[1] = (self.voxel_origin[1] - voxel[1]) * self.voxel_size[1]
        stereo[2] = (voxel[2] - self.voxel_origin[2]) * self.voxel_size[2]
        return stereo
    
    def _setup_figure(self):
        """Create the figure with triplanar views and controls."""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Stereotaxic MRI Viewer', fontsize=14)
        
        # Create axes
        self.ax_axial = self.fig.add_subplot(2, 3, 1)
        self.ax_coronal = self.fig.add_subplot(2, 3, 2)
        self.ax_sagittal = self.fig.add_subplot(2, 3, 3)
        
        # Info panel
        self.ax_info = self.fig.add_subplot(2, 3, 4)
        self.ax_info.axis('off')
        
        # 3D view
        self.ax_3d = self.fig.add_subplot(2, 3, 5, projection='3d')
        
        # Setup slice views
        self._setup_slice_views()
        
        # Setup controls
        self.text_ml = None
        self.text_ap = None
        self.text_dv = None
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)  # ADD THIS
        
        # Draw and setup controls
        self.fig.canvas.draw()
        self._setup_controls()
        
        # Initial update
        self._update_views()
    
    def _setup_slice_views(self):
        """Configure the slice view axes labels."""
        self.ax_axial.set_xlabel('ML (mm, +right)')
        self.ax_axial.set_ylabel('AP (mm, +anterior)')
        self.ax_axial.set_title('Axial (top-down)')
        
        self.ax_coronal.set_xlabel('ML (mm, +right)')
        self.ax_coronal.set_ylabel('DV (mm, +dorsal)')
        self.ax_coronal.set_title('Coronal (front)')
        
        self.ax_sagittal.set_xlabel('AP (mm, +anterior)')
        self.ax_sagittal.set_ylabel('DV (mm, +dorsal)')
        self.ax_sagittal.set_title('Sagittal (side)')
    
    def _setup_controls(self):
        """Set up coordinate entry text boxes."""
        try:
            box_width = 0.08
            box_height = 0.04
            left_start = 0.08
            bottom_start = 0.08
            
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
            
            # Submit button - STORE AS INSTANCE VARIABLE
            ax_submit = self.fig.add_axes([left_start + 0.10, bottom_start + 0.04, 0.06, box_height])
            self.btn_submit = Button(ax_submit, 'Submit')
            self.btn_submit.on_clicked(self._on_submit_coords)
            
        except Exception as e:
            print(f"Warning: Could not set up controls: {e}")
    
    def _on_click(self, event):
        """Handle click on slice views to update position."""
        if not hasattr(event, 'inaxes') or event.inaxes is None:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        if event.inaxes == self.ax_axial:
            # Axial: x=ML, y=AP
            self.position[0] = event.xdata  # ML
            self.position[1] = event.ydata  # AP
        elif event.inaxes == self.ax_coronal:
            # Coronal: x=ML, y=DV
            self.position[0] = event.xdata  # ML
            self.position[2] = event.ydata  # DV
        elif event.inaxes == self.ax_sagittal:
            # Sagittal: x=AP, y=DV
            self.position[1] = event.xdata  # AP
            self.position[2] = event.ydata  # DV
        else:
            return
        
        self._update_views()
    
    def _on_submit_coords(self, event):
        """Handle submit button - read all three textboxes and update position."""
        try:
            if self.text_ml is None or self.text_ap is None or self.text_dv is None:
                print("Warning: textboxes not initialized")
                return
            
            ml = float(self.text_ml.text)
            ap = float(self.text_ap.text)
            dv = float(self.text_dv.text)
            
            self.position = np.array([ml, ap, dv])
            self._update_views()
            print(f"Position updated: ML={ml:.1f}, AP={ap:.1f}, DV={dv:.1f}")
        except ValueError as e:
            print(f"Error parsing coordinates: {e}")
        except Exception as e:
            print(f"Error in submit: {e}")

    def _on_scroll(self, event):
        """Handle mouse scroll to navigate through slices.
        
        - Scroll on Axial: change DV
        - Scroll on Coronal: change AP
        - Scroll on Sagittal: change ML
        """
        if event.inaxes is None:
            return
        
        direction = 1 if event.button == 'up' else -1
        step = 0.5  # mm per scroll
        
        if event.inaxes == self.ax_axial:
            # Axial view: scroll changes DV
            self.position[2] += step * direction
        elif event.inaxes == self.ax_coronal:
            # Coronal view: scroll changes AP
            self.position[1] += step * direction
        elif event.inaxes == self.ax_sagittal:
            # Sagittal view: scroll changes ML
            self.position[0] += step * direction
        else:
            return
        
        self._update_views()
        
    def _get_slice(self, axis, position_stereo):
        """Get a 2D slice from the volume at the given stereotaxic position.
        
        axis: 0=ML, 1=AP, 2=DV (which axis to slice along)
        position_stereo: stereotaxic coordinate value along that axis
        
        Returns: slice_data (2D), extent (for imshow)
        """
        # Convert stereotaxic position to voxel index
        if axis == 0:  # ML axis
            voxel_idx_float = self.voxel_origin[0] - (position_stereo / self.voxel_size[0])
        elif axis == 1:  # AP axis
            voxel_idx_float = self.voxel_origin[1] - (position_stereo / self.voxel_size[1])
        else:  # DV axis (axis == 2)
            voxel_idx_float = self.voxel_origin[2] + (position_stereo / self.voxel_size[2])
        
        idx = int(np.clip(voxel_idx_float, 0, self.shape[axis] - 1))
        
        if axis == 0:  # ML: slice shows AP (y) x DV (z)
            slice_data = self.volume[idx, :, :]
            slice_data = np.rot90(slice_data, k=-1)
            
            ap_voxels = np.arange(self.shape[1])
            dv_voxels = np.arange(self.shape[2])
            ap_stereo = (self.voxel_origin[1] - ap_voxels) * self.voxel_size[1]
            dv_stereo = (dv_voxels - self.voxel_origin[2]) * self.voxel_size[2]
            
            extent = [ap_stereo[-1], ap_stereo[0], dv_stereo[0], dv_stereo[-1]]
            
        elif axis == 1:  # AP: slice shows ML (x) x DV (z)
            slice_data = self.volume[:, idx, :]
            slice_data = np.rot90(slice_data, k=-1)
            
            ml_voxels = np.arange(self.shape[0])
            dv_voxels = np.arange(self.shape[2])
            ml_stereo = (self.voxel_origin[0] - ml_voxels) * self.voxel_size[0]
            dv_stereo = (dv_voxels - self.voxel_origin[2]) * self.voxel_size[2]
            
            extent = [ml_stereo[-1], ml_stereo[0], dv_stereo[0], dv_stereo[-1]]
            
        else:  # DV (axis == 2): slice shows ML (x) x AP (y)
            slice_data = self.volume[:, :, idx]
            slice_data = np.rot90(slice_data, k=1)
            
            ml_voxels = np.arange(self.shape[0])
            ap_voxels = np.arange(self.shape[1])
            ml_stereo = (self.voxel_origin[0] - ml_voxels) * self.voxel_size[0]
            ap_stereo = (self.voxel_origin[1] - ap_voxels) * self.voxel_size[1]
            
            extent = [ml_stereo[-1], ml_stereo[0], ap_stereo[-1], ap_stereo[0]]
        
        return slice_data, extent
    
    def _update_views(self):
        """Update all slice views and info panel."""
        # Update textboxes
        if self.text_ml is not None:
            self.text_ml.set_val(f'{self.position[0]:.1f}')
        if self.text_ap is not None:
            self.text_ap.set_val(f'{self.position[1]:.1f}')
        if self.text_dv is not None:
            self.text_dv.set_val(f'{self.position[2]:.1f}')
        
        # Clear axes
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
        
        # Update info panel
        self._update_info()
        
        # Update 3D view
        self._update_3d()
        
        self.fig.canvas.draw_idle()
    
    def _update_info(self):
        """Update the info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = (
            f"Stereotaxic Position:\n"
            f"  ML: {self.position[0]:7.2f} mm\n"
            f"  AP: {self.position[1]:7.2f} mm\n"
            f"  DV: {self.position[2]:7.2f} mm\n\n"
            f"Voxel Indices:\n"
        )
        
        voxel = self._stereo_to_voxel(self.position)
        info_text += f"  X: {voxel[0]:7.1f}\n"
        info_text += f"  Y: {voxel[1]:7.1f}\n"
        info_text += f"  Z: {voxel[2]:7.1f}\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _update_3d(self):
        """Update the 3D view."""
        self.ax_3d.clear()
        self.ax_3d.set_xlabel('ML (mm)')
        self.ax_3d.set_ylabel('AP (mm)')
        self.ax_3d.set_zlabel('DV (mm)')
        self.ax_3d.set_title('3D View')
        
        # Draw current position
        self.ax_3d.scatter(*self.position, c='red', s=100, marker='o')
    
    def set_position(self, ml, ap, dv):
        """Set the current stereotaxic position."""
        self.position = np.array([ml, ap, dv])
        self._update_views()
    
    def show(self):
        """Display the viewer."""
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12,
                                hspace=0.4, wspace=0.35)
        plt.show()