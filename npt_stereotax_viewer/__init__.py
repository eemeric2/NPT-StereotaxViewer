# NPT-StereotaxViewer/npt_stereotax_viewer/__init__.py
"""
NPT-StereotaxViewer: Reusable stereotaxic MRI viewer for non-human primate neurophysiology.

Core modules:
- transform: Coordinate conversions (voxel, RAS, stereotaxic)
- viewer: Interactive triplanar viewer with overlays
- utils: Data structures (VOI, ChamberInfo, etc.)
"""

from .transform import StereotaxTransform
from .viewer import StereotaxTriplanarViewer
from .utils import VOI, VOI_Stereo, Contour, ChamberInfo

__version__ = "0.1.0"
__author__ = "NPT Team"

__all__ = [
    "StereotaxTransform",
    "StereotaxTriplanarViewer",
    "VOI",
    "VOI_Stereo",
    "Contour",
    "ChamberInfo",
]