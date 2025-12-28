"""
Unit tests for src/warp_sphere.py - Spherical inverse mapping.
"""

import numpy as np
import pytest

from src.config import CalibrationData, OutputConfig
from src.warp_sphere import (
    compute_warp_maps,
    create_equirectangular_grid,
    estimate_panorama_coverage,
    spherical_to_world_directions,
    warp_all_images,
    warp_image_to_equirectangular,
)


class TestCreateEquirectangularGrid:
    """Tests for create_equirectangular_grid function."""
    
    def test_grid_shape(self):
        """Test grid has correct shape."""
        theta, phi = create_equirectangular_grid(1024, 512)
        
        assert theta.shape == (512, 1024)
        assert phi.shape == (512, 1024)
    
    def test_theta_range(self):
        """Test theta covers -pi to +pi."""
        theta, _ = create_equirectangular_grid(1024, 512)
        
        assert theta.min() >= -np.pi - 0.01
        assert theta.max() <= np.pi + 0.01
    
    def test_phi_range(self):
        """Test phi covers -pi/2 to +pi/2."""
        _, phi = create_equirectangular_grid(1024, 512)
        
        assert phi.min() >= -np.pi/2 - 0.01
        assert phi.max() <= np.pi/2 + 0.01
    
    def test_float32_dtype(self):
        """Test grids are float32."""
        theta, phi = create_equirectangular_grid(1024, 512)
        
        assert theta.dtype == np.float32
        assert phi.dtype == np.float32
    
    def test_center_values(self):
        """Test center pixel values."""
        theta, phi = create_equirectangular_grid(101, 51)  # Odd for exact center
        
        # Center should be approximately theta=0, phi=0
        center_theta = theta[25, 50]
        center_phi = phi[25, 50]
        
        assert abs(center_theta) < 0.1
        assert abs(center_phi) < 0.1


class TestSphericalToWorldDirections:
    """Tests for spherical_to_world_directions function."""
    
    def test_unit_vectors(self):
        """Test output are unit vectors."""
        theta = np.array([[0, np.pi/2]])
        phi = np.array([[0, 0]])
        
        x, y, z = spherical_to_world_directions(theta, phi)
        
        # Should be unit vectors
        norms = np.sqrt(x**2 + y**2 + z**2)
        assert np.allclose(norms, 1.0)
    
    def test_forward_direction(self):
        """Test theta=0, phi=0 points forward (z=1)."""
        theta = np.array([[0.0]])
        phi = np.array([[0.0]])
        
        x, y, z = spherical_to_world_directions(theta, phi)
        
        assert abs(x[0, 0]) < 1e-6
        assert abs(y[0, 0]) < 1e-6
        assert abs(z[0, 0] - 1.0) < 1e-6
    
    def test_right_direction(self):
        """Test theta=pi/2 points right (x=1)."""
        theta = np.array([[np.pi/2]])
        phi = np.array([[0.0]])
        
        x, y, z = spherical_to_world_directions(theta, phi)
        
        assert abs(x[0, 0] - 1.0) < 1e-6
        assert abs(y[0, 0]) < 1e-6
        assert abs(z[0, 0]) < 1e-6
    
    def test_up_direction(self):
        """Test phi=pi/2 points up (y=1)."""
        theta = np.array([[0.0]])
        phi = np.array([[np.pi/2]])
        
        x, y, z = spherical_to_world_directions(theta, phi)
        
        assert abs(x[0, 0]) < 1e-6
        assert abs(y[0, 0] - 1.0) < 1e-6
        assert abs(z[0, 0]) < 1e-6


class TestComputeWarpMaps:
    """Tests for compute_warp_maps function."""
    
    def test_map_shape(self, sample_calibration):
        """Test warp maps have correct shape."""
        theta, phi = create_equirectangular_grid(512, 256)
        R = np.eye(3)
        
        map_x, map_y, mask = compute_warp_maps(
            theta, phi, R, sample_calibration, 800, 600
        )
        
        assert map_x.shape == (256, 512)
        assert map_y.shape == (256, 512)
        assert mask.shape == (256, 512)
    
    def test_float32_maps(self, sample_calibration):
        """Test maps are float32."""
        theta, phi = create_equirectangular_grid(512, 256)
        R = np.eye(3)
        
        map_x, map_y, _ = compute_warp_maps(
            theta, phi, R, sample_calibration, 800, 600
        )
        
        assert map_x.dtype == np.float32
        assert map_y.dtype == np.float32
    
    def test_valid_mask_in_fov(self, sample_calibration):
        """Test that center region has valid mask."""
        theta, phi = create_equirectangular_grid(512, 256)
        R = np.eye(3)
        
        _, _, mask = compute_warp_maps(
            theta, phi, R, sample_calibration, 800, 600
        )
        
        # Center of panorama (theta=0, phi=0) should be valid for identity R
        # This is forward direction which maps to image center
        center_region = mask[100:156, 230:282]  # Near center
        assert np.sum(center_region) > 0  # Some valid pixels
    
    def test_invalid_outside_bounds(self, sample_calibration):
        """Test pixels outside image bounds are invalid."""
        theta, phi = create_equirectangular_grid(512, 256)
        R = np.eye(3)
        
        map_x, map_y, mask = compute_warp_maps(
            theta, phi, R, sample_calibration, 800, 600
        )
        
        # Invalid pixels should have map value of -1
        invalid_x = map_x[~mask]
        invalid_y = map_y[~mask]
        
        assert np.all(invalid_x == -1)
        assert np.all(invalid_y == -1)


class TestWarpImageToEquirectangular:
    """Tests for warp_image_to_equirectangular function."""
    
    def test_output_shape(self, sample_image, sample_calibration):
        """Test warped image has correct shape."""
        theta, phi = create_equirectangular_grid(512, 256)
        R = np.eye(3)
        
        warped, mask = warp_image_to_equirectangular(
            sample_image, R, sample_calibration, theta, phi, (256, 512)
        )
        
        assert warped.shape == (256, 512, 3)
        assert mask.shape == (256, 512)
    
    def test_mask_dtype(self, sample_image, sample_calibration):
        """Test mask is uint8."""
        theta, phi = create_equirectangular_grid(512, 256)
        R = np.eye(3)
        
        _, mask = warp_image_to_equirectangular(
            sample_image, R, sample_calibration, theta, phi, (256, 512)
        )
        
        assert mask.dtype == np.uint8
    
    def test_mask_values(self, sample_image, sample_calibration):
        """Test mask values are 0 or 255."""
        theta, phi = create_equirectangular_grid(512, 256)
        R = np.eye(3)
        
        _, mask = warp_image_to_equirectangular(
            sample_image, R, sample_calibration, theta, phi, (256, 512)
        )
        
        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values)


class TestWarpAllImages:
    """Tests for warp_all_images function."""
    
    def test_warps_multiple_images(self, sample_image_sequence, sample_calibration):
        """Test warping multiple images."""
        n_images = 3
        images = sample_image_sequence[:n_images]
        rotations = [np.eye(3) for _ in range(n_images)]
        
        output_config = OutputConfig(pano_width=512)
        
        warped, masks = warp_all_images(
            images, rotations, sample_calibration, output_config
        )
        
        assert len(warped) == n_images
        assert len(masks) == n_images
    
    def test_consistent_output_shape(self, sample_image_sequence, sample_calibration):
        """Test all outputs have same shape."""
        n_images = 3
        images = sample_image_sequence[:n_images]
        rotations = [np.eye(3) for _ in range(n_images)]
        
        output_config = OutputConfig(pano_width=512)
        
        warped, masks = warp_all_images(
            images, rotations, sample_calibration, output_config
        )
        
        shapes = [img.shape for img in warped]
        mask_shapes = [m.shape for m in masks]
        
        assert all(s == shapes[0] for s in shapes)
        assert all(s == mask_shapes[0] for s in mask_shapes)


class TestEstimatePanoramaCoverage:
    """Tests for estimate_panorama_coverage function."""
    
    def test_empty_masks(self):
        """Test with empty masks list."""
        coverage = estimate_panorama_coverage([])
        assert coverage == {}
    
    def test_full_coverage(self):
        """Test 100% coverage."""
        mask = np.ones((256, 512), dtype=np.uint8) * 255
        
        coverage = estimate_panorama_coverage([mask])
        
        assert coverage["coverage_percent"] == 100.0
    
    def test_partial_coverage(self, sample_mask):
        """Test partial coverage calculation."""
        coverage = estimate_panorama_coverage([sample_mask])
        
        assert 0 < coverage["coverage_percent"] < 100
        assert coverage["covered_pixels"] < coverage["total_pixels"]
    
    def test_overlap_detection(self):
        """Test overlap detection between masks."""
        H, W = 256, 512
        
        # Two overlapping masks
        mask1 = np.zeros((H, W), dtype=np.uint8)
        mask1[:, :300] = 255
        
        mask2 = np.zeros((H, W), dtype=np.uint8)
        mask2[:, 200:] = 255
        
        coverage = estimate_panorama_coverage([mask1, mask2])
        
        assert coverage["multi_coverage_pixels"] > 0
        assert coverage["max_overlap"] >= 2
    
    def test_returns_pixel_counts(self, sample_mask):
        """Test that pixel counts are returned."""
        coverage = estimate_panorama_coverage([sample_mask])
        
        assert "total_pixels" in coverage
        assert "covered_pixels" in coverage
        assert "single_coverage_pixels" in coverage
        assert "multi_coverage_pixels" in coverage

