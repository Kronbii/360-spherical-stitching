"""
Unit tests for src/blend.py - Image blending algorithms.
"""

import numpy as np
import pytest
import cv2

from src.blend import (
    blend_panorama,
    create_distance_weight,
    create_seam_visualization,
    feather_blend,
    fill_gaps,
    gaussian_pyramid,
    laplacian_pyramid,
    multiband_blend,
    reconstruct_from_laplacian,
)
from src.config import BlendingConfig


class TestCreateDistanceWeight:
    """Tests for create_distance_weight function."""
    
    def test_output_shape(self, sample_mask):
        """Test output has same shape as input."""
        weight = create_distance_weight(sample_mask)
        assert weight.shape == sample_mask.shape
    
    def test_float32_output(self, sample_mask):
        """Test output is float32."""
        weight = create_distance_weight(sample_mask)
        assert weight.dtype == np.float32
    
    def test_zero_outside_mask(self, sample_mask):
        """Test weight is zero outside mask."""
        weight = create_distance_weight(sample_mask)
        
        # Where mask is 0, weight should be 0 or very small
        outside_mask = sample_mask == 0
        assert np.max(weight[outside_mask]) < 0.1
    
    def test_nonzero_inside_mask(self, sample_mask):
        """Test weight is nonzero inside mask."""
        weight = create_distance_weight(sample_mask)
        
        inside_mask = sample_mask > 127
        assert np.max(weight[inside_mask]) > 0
    
    def test_max_at_center(self, sample_mask):
        """Test weight is maximum near center of mask."""
        weight = create_distance_weight(sample_mask, sigma=10)
        
        # Find center of masked region
        inside = sample_mask > 127
        if np.any(inside):
            y_coords, x_coords = np.where(inside)
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            
            # Weight at center should be relatively high
            center_weight = weight[center_y, center_x]
            assert center_weight > 0.3
    
    def test_handles_empty_mask(self):
        """Test handling of empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        weight = create_distance_weight(mask)
        
        assert np.all(weight == 0)


class TestGaussianPyramid:
    """Tests for gaussian_pyramid function."""
    
    def test_pyramid_length(self, sample_image):
        """Test pyramid has correct number of levels."""
        img = sample_image.astype(np.float32)
        levels = 4
        
        pyramid = gaussian_pyramid(img, levels)
        
        assert len(pyramid) == levels
    
    def test_decreasing_size(self, sample_image):
        """Test each level is smaller than previous."""
        img = sample_image.astype(np.float32)
        
        pyramid = gaussian_pyramid(img, 4)
        
        for i in range(len(pyramid) - 1):
            assert pyramid[i].shape[0] > pyramid[i+1].shape[0]
            assert pyramid[i].shape[1] > pyramid[i+1].shape[1]
    
    def test_first_level_is_input(self, sample_image):
        """Test first level is original image."""
        img = sample_image.astype(np.float32)
        
        pyramid = gaussian_pyramid(img, 4)
        
        assert np.allclose(pyramid[0], img)


class TestLaplacianPyramid:
    """Tests for laplacian_pyramid function."""
    
    def test_pyramid_length(self, sample_image):
        """Test pyramid has correct number of levels."""
        img = sample_image.astype(np.float32)
        levels = 4
        
        pyramid = laplacian_pyramid(img, levels)
        
        assert len(pyramid) == levels
    
    def test_last_level_is_low_frequency(self, sample_image):
        """Test last level contains low frequency."""
        img = sample_image.astype(np.float32)
        
        lap_pyr = laplacian_pyramid(img, 4)
        gauss_pyr = gaussian_pyramid(img, 4)
        
        # Last level should be same as last Gaussian level
        assert np.allclose(lap_pyr[-1], gauss_pyr[-1])


class TestReconstructFromLaplacian:
    """Tests for reconstruct_from_laplacian function."""
    
    def test_reconstruction_accuracy(self, sample_image):
        """Test image can be reconstructed from Laplacian pyramid."""
        img = sample_image.astype(np.float32)
        
        lap_pyr = laplacian_pyramid(img, 4)
        reconstructed = reconstruct_from_laplacian(lap_pyr)
        
        # Should be very close to original
        assert np.allclose(reconstructed, img, atol=1.0)
    
    def test_output_shape(self, sample_image):
        """Test reconstructed image has same shape."""
        img = sample_image.astype(np.float32)
        
        lap_pyr = laplacian_pyramid(img, 4)
        reconstructed = reconstruct_from_laplacian(lap_pyr)
        
        assert reconstructed.shape == img.shape


class TestFeatherBlend:
    """Tests for feather_blend function."""
    
    def test_output_shape(self, sample_warped_images):
        """Test output shape matches input."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="feather")
        
        result = feather_blend(images, masks, config)
        
        assert result.shape == images[0].shape
    
    def test_output_dtype(self, sample_warped_images):
        """Test output is uint8."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="feather")
        
        result = feather_blend(images, masks, config)
        
        assert result.dtype == np.uint8
    
    def test_valid_pixel_range(self, sample_warped_images):
        """Test pixel values are in valid range."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="feather")
        
        result = feather_blend(images, masks, config)
        
        assert result.min() >= 0
        assert result.max() <= 255
    
    def test_raises_on_empty_input(self):
        """Test raises error with no images."""
        config = BlendingConfig(method="feather")
        
        with pytest.raises(ValueError):
            feather_blend([], [], config)


class TestMultibandBlend:
    """Tests for multiband_blend function."""
    
    def test_output_shape(self, sample_warped_images):
        """Test output shape matches input."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="multiband", multiband_levels=3)
        
        result = multiband_blend(images, masks, config)
        
        assert result.shape == images[0].shape
    
    def test_output_dtype(self, sample_warped_images):
        """Test output is uint8."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="multiband", multiband_levels=3)
        
        result = multiband_blend(images, masks, config)
        
        assert result.dtype == np.uint8
    
    def test_valid_pixel_range(self, sample_warped_images):
        """Test pixel values are in valid range."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="multiband", multiband_levels=3)
        
        result = multiband_blend(images, masks, config)
        
        assert result.min() >= 0
        assert result.max() <= 255
    
    def test_raises_on_empty_input(self):
        """Test raises error with no images."""
        config = BlendingConfig(method="multiband")
        
        with pytest.raises(ValueError):
            multiband_blend([], [], config)


class TestBlendPanorama:
    """Tests for blend_panorama dispatcher function."""
    
    def test_dispatches_feather(self, sample_warped_images):
        """Test dispatches to feather blending."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="feather")
        
        result = blend_panorama(images, masks, config)
        
        assert result is not None
        assert result.shape == images[0].shape
    
    def test_dispatches_multiband(self, sample_warped_images):
        """Test dispatches to multiband blending."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="multiband", multiband_levels=3)
        
        result = blend_panorama(images, masks, config)
        
        assert result is not None
        assert result.shape == images[0].shape
    
    def test_raises_on_unknown_method(self, sample_warped_images):
        """Test raises error on unknown method."""
        images, masks = sample_warped_images
        config = BlendingConfig(method="unknown")
        
        with pytest.raises(ValueError, match="Unknown"):
            blend_panorama(images, masks, config)


class TestFillGaps:
    """Tests for fill_gaps function."""
    
    def test_no_change_when_no_gaps(self):
        """Test no change when panorama is fully covered."""
        H, W = 256, 512
        panorama = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        mask = np.ones((H, W), dtype=np.uint8) * 255
        
        result = fill_gaps(panorama, [mask])
        
        assert np.array_equal(result, panorama)
    
    def test_fills_gap_regions(self):
        """Test that gap regions are filled."""
        H, W = 256, 512
        panorama = np.zeros((H, W, 3), dtype=np.uint8)
        panorama[:, :W//2, 0] = 255  # Red on left half
        
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[:, :W//2] = 255  # Only left half valid
        
        result = fill_gaps(panorama, [mask])
        
        # Right half (gap) should no longer be black
        right_half = result[:, W//2:, :]
        assert np.mean(right_half) > 0  # Some pixels filled
    
    def test_output_shape(self):
        """Test output has same shape as input."""
        H, W = 256, 512
        panorama = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.ones((H, W), dtype=np.uint8) * 255
        
        result = fill_gaps(panorama, [mask])
        
        assert result.shape == panorama.shape


class TestCreateSeamVisualization:
    """Tests for create_seam_visualization function."""
    
    def test_output_shape(self, sample_warped_images):
        """Test output has correct shape."""
        images, masks = sample_warped_images
        
        vis = create_seam_visualization(images, masks)
        
        assert vis.shape == images[0].shape
    
    def test_color_image(self, sample_warped_images):
        """Test output is color image."""
        images, masks = sample_warped_images
        
        vis = create_seam_visualization(images, masks)
        
        assert len(vis.shape) == 3
        assert vis.shape[2] == 3
    
    def test_handles_empty_input(self):
        """Test handles empty input."""
        vis = create_seam_visualization([], [])
        
        assert vis is not None
        assert len(vis.shape) == 3
    
    def test_different_colors_per_image(self, sample_warped_images):
        """Test each image gets different color."""
        images, masks = sample_warped_images
        
        vis = create_seam_visualization(images, masks)
        
        # Check that we have multiple colors
        unique_colors = np.unique(vis.reshape(-1, 3), axis=0)
        assert len(unique_colors) >= 2  # At least background + one image

