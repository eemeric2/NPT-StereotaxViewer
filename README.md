# NPT-StereotaxViewer

**Interactive stereotaxic MRI viewer for non-human primate neurophysiology**

A modular, reusable Python package for viewing and manipulating NHP brain MRI data in stereotaxic coordinates. Features synchronized triplanar display, coordinate transformations, and Horsley-Clarke stereotaxic reference frame Horsley-Clarke stereotaxic reference frame (HC) NIfTI generation.

## Features

✅ **Triplanar viewer**: Synchronized Axial, Coronal, Sagittal views with crosshairs  
✅ **Coordinate systems**: Voxel ↔ RAS ↔ Stereotaxic conversions  
✅ **HC NIfTI generation**: Transform MRI data to Horsley-Clarke stereotaxic reference frame reference frame  
✅ **Interactive navigation**: Mouse scroll, click-to-position, manual coordinate entry  
✅ **VOI support**: Load and visualize regions of interest  
✅ **Chamber visualization**: Display electrode chamber outlines  
✅ **Modular design**: Clean separation of concerns for easy extension  

## Installation

### From source

```bash
git clone https://github.com/your-org/NPT-StereotaxViewer.git
cd NPT-StereotaxViewer
pip install -e .
```
## Dependencies
- Python ≥ 3.8
- numpy ≥ 1.19.0
- nibabel ≥ 3.0.0
- matplotlib ≥ 3.3.0

## Quick Start
### 1. **Load and view MRI with triplanar viewer**
```python
from npt_stereotax_viewer import StereotaxTransform, StereotaxTriplanarViewer

# Initialize transform with MRI file and known stereotaxic parameters
transform = StereotaxTransform(
    nifti_path="T1_58x.nii",
    voxel_size=(0.7, 0.625, 0.625),      # mm
    voxel_origin=(88, 142, 21),           # voxel indices at stereo [0,0,0]
)

# Create and display interactive viewer
viewer = StereotaxTriplanarViewer(
    transform=transform,
    init_stereo=(0, 0, 0),        # Start at stereotaxic origin
    show_chamber=True,
    show_voi_contours=True,
)
viewer.show()
```
## Navigation:

- Scroll on any view to move through Z planes
- Click on a view to jump to that coordinate
- Manual entry: Type stereotaxic coordinates (mm) and click "Go to Coordinates"

### 2. **Generate Horsley-Clarke stereotaxic reference frame (HC) NIfTI**
```python
# Generate HC NIfTI where stereotaxic origin becomes image origin
hc_nifti = transform.get_hc_nifti()

# Save to file
transform.save_hc_nifti("T1_58x_HC.nii")
```

The HC transform aligns the image so that stereotaxic coordinate [0, 0, 0] is at the image center, making it easier to work with aligned electrode arrays and anatomical landmarks.

### 3. **Coordinate conversions**
```python
import numpy as np

# Convert between coordinate systems
voxel = np.array([100, 150, 30])
stereo = transform.voxel_to_stereo(voxel)        # → [-8.4, 4.375, 5.625]
ras = transform.voxel_to_ras(voxel)              # → RAS physical coordinates

# Inverse conversions
voxel_back = transform.stereo_to_voxel(stereo)
voxel_back = transform.ras_to_voxel(ras)
```

4. Load VOIs and chamber information
```python
# Load from JSON files
transform = StereotaxTransform(
    nifti_path="T1_58x.nii",
    voxel_size=(0.7, 0.625, 0.625),
    voxel_origin=(88, 142, 21),
    vois_json="58x.json",                 # VOIs in stereotaxic coordinates
    chamber_json="chamber_info.json",     # Chamber geometry
)

# Access VOI data
vois = transform.list_vois()
voi_data = transform.get_voi_stereo("arcuates")

# Chamber info
chamber = transform.get_chamber_info()
```

## Core Modules
### transform.py - Coordinate Transformations
#### Class: StereotaxTransform

Core functionality for coordinate conversions and HC NIfTI generation.

##### Key methods:

- voxel_to_stereo() / stereo_to_voxel() - Stereotaxic conversions
- voxel_to_ras() / ras_to_voxel() - RAS physical conversions
- get_hc_nifti() - Generate Horsley-Clarke stereotaxic reference frame NIfTI
- save_hc_nifti() - Save HC NIfTI to file
- get_voi_stereo() - Retrieve VOI in stereotaxic space
- list_vois() - List all loaded VOIs

##### Coordinate System Details:

Stereotaxic coordinates are computed from voxel indices using:

```makefile
stereo_X = (voxel_origin_X - voxel_X) × voxel_size_X
stereo_Y = (voxel_origin_Y - voxel_Y) × voxel_size_Y
stereo_Z = (voxel_Z - voxel_origin_Z) × voxel_size_Z
```
This formula (verified against MIPAV) accounts for flipped X,Y axes relative to standard array indexing.

### viewer.py - Interactive Triplanar Display
**Class: StereotaxTriplanarViewer**

Interactive matplotlib-based viewer with synchronized slice navigation.

**Key methods:**

- set_stereo_position() - Jump to stereotaxic coordinate
- show() - Display interactive viewer

#### Features:

- Real-time crosshair synchronization across views
- Intensity value display at cursor
- VOI legend with color coding
- Manual coordinate input with "Go to Coordinates" button
- Mouse wheel navigation through Z-axis
- Click-to-position on any view
- utils.py - Data Structures

### Classes:

- Contour - 2D contour with vertices and color
- VOI - Volume of interest with name, color, contours
- VOI_Stereo - VOI in stereotaxic coordinates with rotation/translation
- ChamberInfo - Electrode chamber geometry and positioning

## Coordinate System Transformation
### Stereotaxic vs. Voxel Coordinates
The stereotaxic coordinate system is fixed to the subject's brain, while voxel indices reference the image array. The transform requires:

1. Voxel origin: The voxel [X, Y, Z] that corresponds to stereotaxic [0, 0, 0]
2. Voxel size: Physical dimensions (mm) of each voxel in each direction
3. NIfTI affine: Maps voxel indices to RAS (DICOM standard) physical space

**Example:**

- Voxel [88, 142, 21] → Stereotaxic [0, 0, 0]
- Voxel size: [0.7, 0.625, 0.625] mm
- Then voxel [100, 150, 30] → Stereotaxic [-8.4, 4.375, 5.625] mm

## Horsley-Clarke stereotaxic reference frame (HC) NIfTI
The HC transform recomputes the NIfTI affine so that stereotaxic [0, 0, 0] becomes the image origin (RAS [0, 0, 0]). This is useful for:

- Electrode trajectory visualization: Trajectories defined in stereotaxic coordinates align naturally with image axes
- Population statistics: Multiple subjects' HC images can be directly compared
- Anatomical landmarks: Recording sites defined relative to stereotaxic frame are preserved

#### Formula:

```css
affine_HC = affine_original
affine_HC[:3, 3] = -affine_original[:3, :3] @ ras_to_voxel(stereotax_zero_ras)
```

### Testing
Run tests with pytest:

```bash
pytest tests/
```

#### Test modules:

- test_imports.py - Module import and initialization
- test_hc_creation.py - HC NIfTI generation accuracy
- test_coordinate_conversions.py - Coordinate system round-trips

Example test result:

```yaml
tests/test_hc_creation.py::test_hc_nifti_matches_reference PASSED
✓ HC NIfTI generation verified
  Affine matches: 0.00e+00
  Data max diff: 0.00e+00
  Shape: (174, 210, 99)
```

### Example Data
Test data is included in **examples/data/58x/**:

- Imaging/58x_wip_mprageax_.70mm_sense_4_1_cropped.nii - Original T1 MRI
- Imaging/58x_wip_mprageax_.70mm_sense_4_1_cropped_HC.nii - HC NIfTI (reference)
- VOIs/ - ROI contours in XML format
- 58x.json - Stereotaxic metadata

### Project Structure
```python
NPT-StereotaxViewer/
├── npt_stereotax_viewer/
│   ├── __init__.py           # Public API
│   ├── transform.py          # Coordinate transformations
│   ├── viewer.py             # Interactive viewer
│   └── utils.py              # Data structures
├── tests/
│   ├── test_imports.py
│   ├── test_hc_creation.py
│   └── test_coordinate_conversions.py
├── examples/
│   └── data/
│       └── 58x/              # Example subject data
├── docs/
├── README.md
├── setup.py
├── requirements.txt
└── .gitignore
```
## API Reference
### StereotaxTransform
```python
class StereotaxTransform:
    def __init__(
        self,
        nifti_path: str,
        voxel_size: Tuple[float, float, float],
        voxel_origin: Tuple[float, float, float],
        stereotax_zero_ras: Optional[Tuple[float, float, float]] = None,
        vois_json: Optional[str] = None,
        chamber_json: Optional[str] = None,
    )
    
    def voxel_to_stereo(self, voxel_idx: np.ndarray) -> np.ndarray
    def stereo_to_voxel(self, stereo: np.ndarray) -> np.ndarray
    def voxel_to_ras(self, voxel_idx: np.ndarray) -> np.ndarray
    def ras_to_voxel(self, ras: np.ndarray) -> np.ndarray
    def ras_to_stereo(self, ras: np.ndarray) -> np.ndarray
    def stereo_to_ras(self, stereo: np.ndarray) -> np.ndarray
    
    def get_hc_nifti(self) -> nib.Nifti1Image
    def save_hc_nifti(self, output_path: str)
    def save_stereo_nifti(self, output_path: str)
    
    def get_voi_stereo(self, name: str) -> Optional[VOI_Stereo]
    def list_vois(self) -> list
    def get_chamber_info(self) -> Optional[ChamberInfo]
```

### StereotaxTriplanarViewer
```python
class StereotaxTriplanarViewer:
    def __init__(
        self,
        transform: StereotaxTransform,
        init_stereo: Tuple[float, float, float] = (0, 0, 0),
        show_chamber: bool = True,
        show_voi_contours: bool = False,
    )
    
    def set_stereo_position(self, stereo: np.ndarray)
    def show()
```
## References
- MIPAV: https://mipav.cit.nih.gov/ (used for coordinate verification)
- NiBabel: https://nipy.org/nibabel/ (NIfTI file handling)
- NIfTI specification: https://nifti.nimh.nih.gov/

Contributing
Contributions welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-thing)
Commit changes (git commit -am 'Add amazing thing')
Push to branch (git push origin feature/amazing-thing)
Open a Pull Request
License
MIT License - See LICENSE file for details

## Authors

**Questions?** Open an issue or contact the maintainers.

Erik E. Emeric - eee@jhu.edu
