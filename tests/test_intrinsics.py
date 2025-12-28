"""
Unit tests for src/intrinsics.py - Camera intrinsics estimation.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import CalibrationData, IntrinsicsConfig
from src.intrinsics import (
    estimate_from_35mm_equivalent,
    estimate_from_calib_json,
    estimate_from_focal_length,
    estimate_from_hfov,
    estimate_intrinsics,
    save_intrinsics_report,
    undistort_images,
)
from src.io_utils import ImageInfo


class TestEstimateFromHFOV:
    """Tests for estimate_from_hfov function."""
    
    def test_calculates_focal_length(self):
        """Test focal length calculation from HFOV."""
        calib = estimate_from_hfov(65.0, 1920, 1080)
        
        # fx = W / (2 * tan(hfov/2))
        expected_fx = 1920 / (2 * math.tan(math.radians(65) / 2))
        assert abs(calib.fx - expected_fx) < 0.01
    
    def test_principal_point_at_center(self):
        """Test principal point is at image center."""
        calib = estimate_from_hfov(65.0, 1920, 1080)
        
        assert calib.cx == 960.0  # W/2
        assert calib.cy == 540.0  # H/2
    
    def test_square_pixels_assumption(self):
        """Test that fx == fy (square pixels)."""
        calib = estimate_from_hfov(65.0, 1920, 1080)
        assert calib.fx == calib.fy
    
    def test_source_string(self):
        """Test source describes HFOV fallback."""
        calib = estimate_from_hfov(65.0, 1920, 1080)
        assert "HFOV" in calib.source
        assert "65" in calib.source
    
    def test_different_fov_values(self):
        """Test with different FOV values."""
        calib_narrow = estimate_from_hfov(45.0, 1920, 1080)
        calib_wide = estimate_from_hfov(90.0, 1920, 1080)
        
        # Narrower FOV = longer focal length
        assert calib_narrow.fx > calib_wide.fx


class TestEstimateFrom35mmEquivalent:
    """Tests for estimate_from_35mm_equivalent function."""
    
    def test_calculates_from_35mm(self):
        """Test calculation from 35mm equivalent."""
        exif_data = {'FocalLengthIn35mmFilm': 28}
        calib = estimate_from_35mm_equivalent(exif_data, 4000, 3000)
        
        # fx_pixels = (f35 / 36.0) * image_width
        expected_fx = (28 / 36.0) * 4000
        assert calib is not None
        assert abs(calib.fx - expected_fx) < 0.01
    
    def test_returns_none_on_missing(self):
        """Test returns None when tag missing."""
        exif_data = {}
        calib = estimate_from_35mm_equivalent(exif_data, 4000, 3000)
        assert calib is None
    
    def test_returns_none_on_zero(self):
        """Test returns None when focal length is zero."""
        exif_data = {'FocalLengthIn35mmFilm': 0}
        calib = estimate_from_35mm_equivalent(exif_data, 4000, 3000)
        assert calib is None
    
    def test_source_includes_focal_length(self):
        """Test source string includes focal length."""
        exif_data = {'FocalLengthIn35mmFilm': 28}
        calib = estimate_from_35mm_equivalent(exif_data, 4000, 3000)
        assert "28" in calib.source
        assert "35mm" in calib.source.lower()


class TestEstimateFromFocalLength:
    """Tests for estimate_from_focal_length function."""
    
    def test_returns_none_without_sensor_size(self):
        """Test returns None when sensor size unavailable."""
        exif_data = {'FocalLength': 4.5}
        calib = estimate_from_focal_length(exif_data, 4000, 3000)
        assert calib is None
    
    def test_returns_none_on_missing_focal(self):
        """Test returns None when focal length missing."""
        exif_data = {}
        calib = estimate_from_focal_length(exif_data, 4000, 3000)
        assert calib is None


class TestEstimateFromCalibJson:
    """Tests for estimate_from_calib_json function."""
    
    def test_loads_calibration(self, temp_dir):
        """Test loading calibration from JSON."""
        calib_data = {
            "fx": 1500.0,
            "fy": 1500.0,
            "cx": 960.0,
            "cy": 540.0
        }
        calib_path = temp_dir / "calib.json"
        with open(calib_path, 'w') as f:
            json.dump(calib_data, f)
        
        calib = estimate_from_calib_json(calib_path, 1920, 1080)
        
        assert calib is not None
        assert calib.fx == 1500.0
        assert calib.fy == 1500.0
    
    def test_loads_distortion(self, temp_dir):
        """Test loading distortion coefficients."""
        calib_data = {
            "fx": 1500.0,
            "fy": 1500.0,
            "dist_coeffs": [0.1, -0.2, 0.0, 0.0, 0.05]
        }
        calib_path = temp_dir / "calib.json"
        with open(calib_path, 'w') as f:
            json.dump(calib_data, f)
        
        calib = estimate_from_calib_json(calib_path, 1920, 1080)
        
        assert calib.dist_coeffs == [0.1, -0.2, 0.0, 0.0, 0.05]
    
    def test_scales_normalized_values(self, temp_dir):
        """Test scaling of normalized calibration values."""
        calib_data = {
            "fx": 0.8,  # Normalized
            "fy": 0.8
        }
        calib_path = temp_dir / "calib.json"
        with open(calib_path, 'w') as f:
            json.dump(calib_data, f)
        
        calib = estimate_from_calib_json(calib_path, 1920, 1080)
        
        # Should be scaled by image dimensions
        assert calib.fx == 0.8 * 1920
        assert calib.fy == 0.8 * 1080
    
    def test_returns_none_on_missing_file(self, temp_dir):
        """Test returns None for non-existent file."""
        calib = estimate_from_calib_json(temp_dir / "missing.json", 1920, 1080)
        assert calib is None
    
    def test_returns_none_on_none_path(self):
        """Test returns None when path is None."""
        calib = estimate_from_calib_json(None, 1920, 1080)
        assert calib is None


class TestEstimateIntrinsics:
    """Tests for estimate_intrinsics function."""
    
    def test_uses_hfov_fallback(self, temp_dir):
        """Test fallback to HFOV when no EXIF."""
        # Create image info without EXIF
        info = ImageInfo(path=temp_dir / "test.jpg", exif_data={})
        config = IntrinsicsConfig(hfov_deg=70.0)
        
        calib = estimate_intrinsics([info], 1920, 1080, config)
        
        assert "HFOV" in calib.source
        assert "70" in calib.source
    
    def test_uses_calib_json_first(self, temp_dir):
        """Test calibration JSON takes priority."""
        # Create calibration file
        calib_data = {"fx": 2000.0, "fy": 2000.0}
        calib_path = temp_dir / "calib.json"
        with open(calib_path, 'w') as f:
            json.dump(calib_data, f)
        
        info = ImageInfo(
            path=temp_dir / "test.jpg",
            exif_data={'FocalLengthIn35mmFilm': 28}
        )
        config = IntrinsicsConfig(calib_json=calib_path)
        
        calib = estimate_intrinsics([info], 1920, 1080, config)
        
        assert calib.fx == 2000.0
        assert "calibration" in calib.source.lower()
    
    def test_uses_35mm_equivalent(self, temp_dir):
        """Test uses EXIF 35mm equivalent when available."""
        info = ImageInfo(
            path=temp_dir / "test.jpg",
            exif_data={'FocalLengthIn35mmFilm': 28}
        )
        config = IntrinsicsConfig()
        
        calib = estimate_intrinsics([info], 4000, 3000, config)
        
        assert "35mm" in calib.source.lower()


class TestSaveIntrinsicsReport:
    """Tests for save_intrinsics_report function."""
    
    def test_creates_report_file(self, temp_dir, sample_calibration):
        """Test report file creation."""
        info = ImageInfo(path=temp_dir / "test.jpg", exif_data={})
        output_path = temp_dir / "intrinsics.json"
        
        save_intrinsics_report(sample_calibration, [info], 1920, 1080, output_path)
        
        assert output_path.exists()
    
    def test_report_contains_intrinsics(self, temp_dir, sample_calibration):
        """Test report contains intrinsics data."""
        info = ImageInfo(path=temp_dir / "test.jpg", exif_data={})
        output_path = temp_dir / "intrinsics.json"
        
        save_intrinsics_report(sample_calibration, [info], 1920, 1080, output_path)
        
        with open(output_path) as f:
            report = json.load(f)
        
        assert "intrinsics" in report
        assert report["intrinsics"]["fx"] == 1000.0
    
    def test_report_contains_fov(self, temp_dir, sample_calibration):
        """Test report contains FOV calculation."""
        info = ImageInfo(path=temp_dir / "test.jpg", exif_data={})
        output_path = temp_dir / "intrinsics.json"
        
        save_intrinsics_report(sample_calibration, [info], 1920, 1080, output_path)
        
        with open(output_path) as f:
            report = json.load(f)
        
        assert "field_of_view" in report
        assert "horizontal_deg" in report["field_of_view"]


class TestUndistortImages:
    """Tests for undistort_images function."""
    
    def test_skips_without_distortion(self, sample_image):
        """Test skips undistortion when no coefficients."""
        calib = CalibrationData(
            fx=1000.0, fy=1000.0, cx=400.0, cy=300.0,
            dist_coeffs=None
        )
        
        images, new_calib = undistort_images([sample_image], calib)
        
        # Should return original images
        assert len(images) == 1
        assert new_calib == calib
    
    def test_skips_zero_distortion(self, sample_image):
        """Test skips when distortion coefficients are zero."""
        calib = CalibrationData(
            fx=1000.0, fy=1000.0, cx=400.0, cy=300.0,
            dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0]
        )
        
        images, new_calib = undistort_images([sample_image], calib)
        
        assert len(images) == 1
    
    def test_applies_undistortion(self, sample_image):
        """Test applies undistortion with coefficients."""
        calib = CalibrationData(
            fx=600.0, fy=600.0, 
            cx=sample_image.shape[1] / 2, 
            cy=sample_image.shape[0] / 2,
            dist_coeffs=[0.1, -0.1, 0.0, 0.0, 0.0]
        )
        
        images, new_calib = undistort_images([sample_image], calib)
        
        assert len(images) == 1
        assert images[0].shape == sample_image.shape
        # New calibration should have no distortion
        assert new_calib.dist_coeffs is None

