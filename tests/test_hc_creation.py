# NPT-StereotaxViewer/tests/test_hc_creation.py
"""
Test HC NIfTI generation against known-good reference files.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import pytest

from npt_stereotax_viewer import StereotaxTransform


@pytest.fixture
def sample_data():
    """Locate test data directory."""
    data_root = Path(__file__).parent.parent / "examples" / "data" / "58x"
    if not data_root.exists():
        pytest.skip(f"Test data not found at {data_root}")
    return data_root


def test_hc_nifti_matches_reference(sample_data):
    """
    Verify newly-generated HC NIfTI matches existing reference file.
    
    Checks:
    - Affine matrix matches
    - Data arrays identical
    - Shapes consistent
    """
    nifti_path = sample_data / "Imaging" / "58x_wip_mprageax_.70mm_sense_4_1_cropped.nii"
    ref_hc_path = sample_data / "Imaging" / "58x_wip_mprageax_.70mm_sense_4_1_cropped_HC.nii"
    
    assert nifti_path.exists(), f"Original NIfTI not found: {nifti_path}"
    assert ref_hc_path.exists(), f"Reference HC NIfTI not found: {ref_hc_path}"
    
    # Initialize transform with known parameters
    transform = StereotaxTransform(
        nifti_path=str(nifti_path),
        voxel_size=(0.7, 0.625, 0.625),
        voxel_origin=(88, 142, 21),
    )
    
    # Generate new HC NIfTI
    new_hc = transform.get_hc_nifti()
    
    # Load reference
    ref_hc = nib.load(ref_hc_path)
    
    # Test affine
    assert new_hc.affine.shape == ref_hc.affine.shape, "Affine shape mismatch"
    np.testing.assert_array_almost_equal(
        new_hc.affine, ref_hc.affine, decimal=5,
        err_msg="Affine matrices don't match"
    )
    
    # Test data
    np.testing.assert_array_almost_equal(
        new_hc.get_fdata(), ref_hc.get_fdata(), decimal=5,
        err_msg="Data arrays don't match"
    )
    
    # Test shape
    assert new_hc.shape == ref_hc.shape, "Shape mismatch"
    
    print(f"✓ HC NIfTI generation verified")
    print(f"  Affine matches: {np.max(np.abs(new_hc.affine - ref_hc.affine)):.2e}")
    print(f"  Data max diff: {np.max(np.abs(new_hc.get_fdata() - ref_hc.get_fdata())):.2e}")
    print(f"  Shape: {new_hc.shape}")


def test_stereo_to_voxel_roundtrip(sample_data):
    """Test that stereo → voxel → stereo conversions are consistent."""
    nifti_path = sample_data / "Imaging" / "58x_wip_mprageax_.70mm_sense_4_1_cropped.nii"
    
    if not nifti_path.exists():
        pytest.skip(f"Test data not found: {nifti_path}")
    
    transform = StereotaxTransform(
        nifti_path=str(nifti_path),
        voxel_size=(0.7, 0.625, 0.625),
        voxel_origin=(88, 142, 21),
    )
    
    # Test known points
    test_stereo = np.array([[0, 0, 0], [-5, 3, 2], [2, -1, -4]], dtype=float)
    
    for stereo in test_stereo:
        voxel = transform.stereo_to_voxel(stereo)
        stereo_roundtrip = transform.voxel_to_stereo(voxel)
        
        np.testing.assert_array_almost_equal(
            stereo, stereo_roundtrip, decimal=5,
            err_msg=f"Roundtrip failed for {stereo}"
        )


def test_hc_nifti_save(sample_data, tmp_path):
    """Test HC NIfTI can be saved and reloaded."""
    nifti_path = sample_data / "Imaging" / "58x_wip_mprageax_.70mm_sense_4_1_cropped.nii"
    
    if not nifti_path.exists():
        pytest.skip(f"Test data not found: {nifti_path}")
    
    transform = StereotaxTransform(
        nifti_path=str(nifti_path),
        voxel_size=(0.7, 0.625, 0.625),
        voxel_origin=(88, 142, 21),
    )
    
    # Save HC NIfTI
    output_path = tmp_path / "test_output_HC.nii"
    transform.save_hc_nifti(str(output_path))
    
    # Reload and verify
    assert output_path.exists(), "Output file not created"
    reloaded = nib.load(output_path)
    
    hc_nifti = transform.get_hc_nifti()
    np.testing.assert_array_equal(reloaded.get_fdata(), hc_nifti.get_fdata())
    np.testing.assert_array_almost_equal(reloaded.affine, hc_nifti.affine, decimal=5)