# NPT-StereotaxViewer/tests/test_imports.py
"""
Test that all modules import correctly and core functionality works.
"""

import pytest
from pathlib import Path


def test_imports():
    """Test that all public modules import without error."""
    from npt_stereotax_viewer import (
        StereotaxTransform,
        StereotaxTriplanarViewer,
        VOI,
        VOI_Stereo,
        Contour,
        ChamberInfo,
    )
    assert StereotaxTransform is not None
    assert StereotaxTriplanarViewer is not None
    assert VOI is not None
    assert VOI_Stereo is not None
    assert Contour is not None
    assert ChamberInfo is not None


def test_transform_initialization():
    """Test StereotaxTransform initializes with test data."""
    from npt_stereotax_viewer import StereotaxTransform
    
    data_root = Path(__file__).parent.parent / "examples" / "data" / "58x"
    nifti_path = data_root / "Imaging" / "58x_wip_mprageax_.70mm_sense_4_1_cropped.nii"
    
    if not nifti_path.exists():
        pytest.skip(f"Test data not found: {nifti_path}")
    
    transform = StereotaxTransform(
        nifti_path=str(nifti_path),
        voxel_size=(0.7, 0.625, 0.625),
        voxel_origin=(88, 142, 21),
    )
    
    assert transform.data.shape == (174, 210, 99)
    assert transform.voxel_size[0] == 0.7
    assert transform.voxel_origin[0] == 88


def test_coordinate_conversions():
    """Test basic coordinate conversion formulas."""
    from npt_stereotax_viewer import StereotaxTransform
    
    data_root = Path(__file__).parent.parent / "examples" / "data" / "58x"
    nifti_path = data_root / "Imaging" / "58x_wip_mprageax_.70mm_sense_4_1_cropped.nii"
    
    if not nifti_path.exists():
        pytest.skip(f"Test data not found: {nifti_path}")
    
    transform = StereotaxTransform(
        nifti_path=str(nifti_path),
        voxel_size=(0.7, 0.625, 0.625),
        voxel_origin=(88, 142, 21),
    )
    
    # Test known conversion: voxel origin should map to stereo [0,0,0]
    import numpy as np
    stereo_origin = transform.voxel_to_stereo(np.array([88, 142, 21]))
    np.testing.assert_array_almost_equal(stereo_origin, [0, 0, 0], decimal=5)
    
    # Test roundtrip
    test_voxel = np.array([100, 150, 30])
    stereo = transform.voxel_to_stereo(test_voxel)
    voxel_back = transform.stereo_to_voxel(stereo)
    np.testing.assert_array_almost_equal(test_voxel, voxel_back, decimal=5)