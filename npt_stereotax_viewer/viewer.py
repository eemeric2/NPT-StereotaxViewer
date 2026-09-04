# NPT-StereotaxViewer/npt_stereotax_viewer/viewer.py
"""
StereotaxTriplanarViewer: Interactive 3D triplanar viewer for stereotaxic MRI.

Displays synchronized Axial, Coronal, and Sagittal views with overlays:
- Crosshairs at current stereotaxic position
- VOI contours (optional)
- Chamber visualization (optional)
- Manual coordinate entry
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.patches import Circle
from typing import Optional, Dict, Tuple
from pathlib import Path

from .transform import StereotaxTransform
from .utils import VOI_Stereo


class StereotaxTriplanarViewer:
    """
    Interactive triplanar viewer for stereotaxic MRI with coordinate overlays.

    Displays Axial, Coronal, Sagittal slices synchronized to a single stereotaxic
    coordinate. User can navigate via:
    - Mouse scroll on any view
    - Manual coordinate entry (X, Y, Z textboxes)
    - Slider controls
    """

    def __init__(
        self,
        transform: StereotaxTransform,
        init_stereo: Tuple[float, float, float] = (0, 0, 0),
        show_chamber: bool = True,
        show_voi_contours: bool = False,
    ):
        """
        Initialize triplanar viewer.

        Parameters
        ----------
        transform : StereotaxTransform
            Coordinate transform object with loaded MRI data
        init_stereo : tuple of float
            Initial stereotaxic coordinates [X, Y, Z]
        show_chamber : bool
            Display chamber outline
        show_voi_contours : bool
            Display VOI contours on slices
        """
        self.transform = transform
        self.show_chamber = show_chamber
        self.show_voi_contours = show_voi_contours

        # Current stereotaxic position
        self.stereo = np.array(init_stereo, dtype=np.float64)
        self.voxel = self.transform.stereo_to_voxel(self.stereo)

        # Create figure and subplots
        self.fig = plt.figure(figsize=(16, 12))
        
        # Adjust layout for controls
        self.fig.subplots_adjust(
            left=0.1, right=0.85, top=0.93, bottom=0.25, hspace=0.3, wspace=0.3
        )

        # Axial (XY), Coronal (XZ), Sagittal (YZ)
        self.ax_axial = self.fig.add_subplot(2, 3, 1)
        self.ax_coronal = self.fig.add_subplot(2, 3, 2)
        self.ax_sagittal = self.fig.add_subplot(2, 3, 3)

        # 3D VOI legend
        self.ax_legend = self.fig.add_subplot(2, 3, 5)

        # Info panel
        self.ax_info = self.fig.add_subplot(2, 3, 6)

        # Initial image data
        self._update_slices()
        self._draw_legend()

        # Crosshair artists (updated dynamically)
        self.h_axial = None
        self.v_axial = None
        self.h_coronal = None
        self.v_coronal = None
        self.h_sagittal = None
        self.v_sagittal = None

        self.im_axial = None
        self.im_coronal = None
        self.im_sagittal = None

        # Initial plot
        self._plot_slices()

        # Mouse scroll connection
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Textbox controls
        self._setup_controls()

    def _update_slices(self):
        """Extract slices at current voxel position."""
        x, y, z = [int(np.round(v)) for v in self.voxel]
        data = self.transform.data

        # Clamp to valid ranges
        x = np.clip(x, 0, data.shape[0] - 1)
        y = np.clip(y, 0, data.shape[1] - 1)
        z = np.clip(z, 0, data.shape[2] - 1)

        self.current_x, self.current_y, self.current_z = x, y, z

        self.axial = data[x, :, z]  # XY slice
        self.coronal = data[x, y, :]  # XZ slice
        self.sagittal = data[:, y, z]  # YZ slice

    def _plot_slices(self):
        """Plot image slices with crosshairs."""
        # Axial (XY)
        if self.im_axial is None:
            self.im_axial = self.ax_axial.imshow(
                np.rot90(self.axial, k=1), cmap="gray", origin="lower"
            )
            self.ax_axial.set_title("Axial (XY)")
            self.ax_axial.set_xlabel("Y (voxel)")
            self.ax_axial.set_ylabel("X (voxel)")
        else:
            self.im_axial.set_data(np.rot90(self.axial, k=1))

        # Coronal (XZ)
        if self.im_coronal is None:
            self.im_coronal = self.ax_coronal.imshow(
                np.rot90(self.coronal, k=-1), cmap="gray", origin="lower"
            )
            self.ax_coronal.set_title("Coronal (XZ)")
            self.ax_coronal.set_xlabel("Z (voxel)")
            self.ax_coronal.set_ylabel("X (voxel)")
        else:
            self.im_coronal.set_data(np.rot90(self.coronal, k=-1))

        # Sagittal (YZ)
        if self.im_sagittal is None:
            self.im_sagittal = self.ax_sagittal.imshow(
                np.rot90(self.sagittal, k=-1), cmap="gray", origin="lower"
            )
            self.ax_sagittal.set_title("Sagittal (YZ)")
            self.ax_sagittal.set_xlabel("Z (voxel)")
            self.ax_sagittal.set_ylabel("Y (voxel)")
        else:
            self.im_sagittal.set_data(np.rot90(self.sagittal, k=-1))

        self._draw_crosshairs()
        self._draw_info()

    def _draw_crosshairs(self):
        """Draw crosshairs at current position."""
        # Remove old crosshairs
        for artist in [
            self.h_axial,
            self.v_axial,
            self.h_coronal,
            self.v_coronal,
            self.h_sagittal,
            self.v_sagittal,
        ]:
            if artist is not None:
                artist.remove()

        # Axial crosshairs (XY plane, viewed from superior)
        self.h_axial = self.ax_axial.axhline(
            y=self.current_x, color="red", linewidth=1, alpha=0.7
        )
        self.v_axial = self.ax_axial.axvline(
            x=self.current_y, color="red", linewidth=1, alpha=0.7
        )

        # Coronal crosshairs (XZ plane, viewed from anterior)
        self.h_coronal = self.ax_coronal.axhline(
            y=self.current_x, color="green", linewidth=1, alpha=0.7
        )
        self.v_coronal = self.ax_coronal.axvline(
            x=self.current_z, color="green", linewidth=1, alpha=0.7
        )

        # Sagittal crosshairs (YZ plane, viewed from right)
        self.h_sagittal = self.ax_sagittal.axhline(
            y=self.current_y, color="blue", linewidth=1, alpha=0.7
        )
        self.v_sagittal = self.ax_sagittal.axvline(
            x=self.current_z, color="blue", linewidth=1, alpha=0.7
        )

    def _draw_info(self):
        """Update info panel with coordinates."""
        self.ax_info.clear()
        self.ax_info.axis("off")

        info_text = (
            f"Stereotaxic [X, Y, Z]:\n"
            f"  X: {self.stereo[0]:7.2f} mm\n"
            f"  Y: {self.stereo[1]:7.2f} mm\n"
            f"  Z: {self.stereo[2]:7.2f} mm\n\n"
            f"Voxel [X, Y, Z]:\n"
            f"  X: {self.current_x:3d}\n"
            f"  Y: {self.current_y:3d}\n"
            f"  Z: {self.current_z:3d}\n\n"
            f"Image intensity:\n"
            f"  {self.transform.data[self.current_x, self.current_y, self.current_z]:.1f}"
        )

        self.ax_info.text(
            0.05,
            0.95,
            info_text,
            transform=self.ax_info.transAxes,
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    def _draw_legend(self):
        """Draw 3D VOI legend."""
        self.ax_legend.clear()
        self.ax_legend.axis("off")

        vois = self.transform.list_vois()
        if not vois:
            self.ax_legend.text(0.5, 0.5, "No VOIs loaded", ha="center", va="center")
            return

        legend_text = "Regions of Interest (VOI):\n\n"
        for i, voi_name in enumerate(vois):
            voi = self.transform.get_voi_stereo(voi_name)
            if voi and hasattr(voi, 'color_rgba'):
                rgba = voi.color_rgba
                color = tuple(rgba[:3] / 255.0) if rgba is not None else (0.5, 0.5, 0.5)
            else:
                color = (0.5, 0.5, 0.5)

            circle = Circle((0.05, 0.9 - i * 0.1), 0.02, color=color, transform=self.ax_legend.transAxes)
            self.ax_legend.add_patch(circle)
            legend_text += f"  • {voi_name}\n"

        self.ax_legend.text(
            0.15,
            0.9,
            legend_text,
            transform=self.ax_legend.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
        )

    def _setup_controls(self):
        """Setup textbox and button controls."""
        # Textboxes for manual coordinate entry
        ax_x = self.fig.add_axes([0.15, 0.15, 0.08, 0.03])
        ax_y = self.fig.add_axes([0.15, 0.10, 0.08, 0.03])
        ax_z = self.fig.add_axes([0.15, 0.05, 0.08, 0.03])

        self.text_x = TextBox(ax_x, "X (mm):", initial=f"{self.stereo[0]:.2f}")
        self.text_y = TextBox(ax_y, "Y (mm):", initial=f"{self.stereo[1]:.2f}")
        self.text_z = TextBox(ax_z, "Z (mm):", initial=f"{self.stereo[2]:.2f}")

        # Submit button
        ax_submit = self.fig.add_axes([0.30, 0.10, 0.06, 0.04])
        btn_submit = Button(ax_submit, "Go to Coordinates")
        btn_submit.on_clicked(self._on_submit)

    def _on_submit(self, event):
        """Handle submit button click."""
        try:
            x = float(self.text_x.text)
            y = float(self.text_y.text)
            z = float(self.text_z.text)
            self.set_stereo_position(np.array([x, y, z]))
        except ValueError:
            pass

    def _on_scroll(self, event):
        """Handle mouse scroll to navigate slices."""
        if event.inaxes is None:
            return

        # Determine which axis was scrolled
        if event.inaxes == self.ax_axial:
            direction = 1 if event.button == "up" else -1
            self.stereo[2] += 0.5 * direction
        elif event.inaxes == self.ax_coronal:
            direction = 1 if event.button == "up" else -1
            self.stereo[2] += 0.5 * direction
        elif event.inaxes == self.ax_sagittal:
            direction = 1 if event.button == "up" else -1
            self.stereo[2] += 0.5 * direction

        self.voxel = self.transform.stereo_to_voxel(self.stereo)
        self._update_slices()
        self._plot_slices()
        self._update_textboxes()
        self.fig.canvas.draw()

    def _on_click(self, event):
        """Handle mouse click to set position."""
        if event.inaxes is None or event.xdata is None:
            return

        # Determine clicked position and update accordingly
        if event.inaxes == self.ax_axial:
            y_click = int(np.round(event.xdata))
            x_click = int(np.round(event.ydata))
            self.voxel[0] = x_click
            self.voxel[1] = y_click
        elif event.inaxes == self.ax_coronal:
            z_click = int(np.round(event.xdata))
            x_click = int(np.round(event.ydata))
            self.voxel[0] = x_click
            self.voxel[2] = z_click
        elif event.inaxes == self.ax_sagittal:
            z_click = int(np.round(event.xdata))
            y_click = int(np.round(event.ydata))
            self.voxel[1] = y_click
            self.voxel[2] = z_click
        else:
            return

        self.stereo = self.transform.voxel_to_stereo(self.voxel)
        self._update_slices()
        self._plot_slices()
        self._update_textboxes()
        self.fig.canvas.draw()

    def _update_textboxes(self):
        """Update coordinate textboxes with current position."""
        self.text_x.set_val(f"{self.stereo[0]:.2f}")
        self.text_y.set_val(f"{self.stereo[1]:.2f}")
        self.text_z.set_val(f"{self.stereo[2]:.2f}")

    def set_stereo_position(self, stereo: np.ndarray):
        """
        Set viewer to stereotaxic coordinates.

        Parameters
        ----------
        stereo : np.ndarray
            Stereotaxic coordinates [X, Y, Z]
        """
        self.stereo = np.array(stereo, dtype=np.float64)
        self.voxel = self.transform.stereo_to_voxel(self.stereo)
        self._update_slices()
        self._plot_slices()
        self._update_textboxes()
        self.fig.canvas.draw()

    def show(self):
        """Display viewer."""
        plt.show()