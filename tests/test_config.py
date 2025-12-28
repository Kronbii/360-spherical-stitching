"""
Unit tests for src/config.py - Configuration dataclasses.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import (
    BlendingConfig,
    CalibrationData,
    DebugConfig,
    IntrinsicsConfig,
    MatchingConfig,
    OutputConfig,
    PipelineConfig,
)


class TestMatchingConfig:
    """Tests for MatchingConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MatchingConfig()
        assert config.match_width == 1600
        assert config.orb_nfeatures == 3000
        assert config.ratio_test_threshold == 0.75
        assert config.ransac_reproj_threshold == 3.0
        assert config.min_inliers == 60
        assert config.use_clahe is False
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = MatchingConfig(
            match_width=1200,
            orb_nfeatures=5000,
            ratio_test_threshold=0.8,
            min_inliers=40,
            use_clahe=True
        )
        assert config.match_width == 1200
        assert config.orb_nfeatures == 5000
        assert config.ratio_test_threshold == 0.8
        assert config.min_inliers == 40
        assert config.use_clahe is True


class TestIntrinsicsConfig:
    """Tests for IntrinsicsConfig dataclass."""
    
    def test_default_hfov(self):
        """Test default HFOV value."""
        config = IntrinsicsConfig()
        assert config.hfov_deg == 65.0
        assert config.calib_json is None
    
    def test_custom_calib_path(self, temp_dir):
        """Test with custom calibration path."""
        calib_path = temp_dir / "calib.json"
        config = IntrinsicsConfig(calib_json=calib_path)
        assert config.calib_json == calib_path


class TestBlendingConfig:
    """Tests for BlendingConfig dataclass."""
    
    def test_default_multiband(self):
        """Test default multiband blending."""
        config = BlendingConfig()
        assert config.method == "multiband"
        assert config.multiband_levels == 5
    
    def test_feather_config(self):
        """Test feather blending configuration."""
        config = BlendingConfig(method="feather", feather_sigma=30.0)
        assert config.method == "feather"
        assert config.feather_sigma == 30.0


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""
    
    def test_default_dimensions(self):
        """Test default panorama dimensions."""
        config = OutputConfig()
        assert config.pano_width == 4096
        assert config.pano_height == 2048  # W/2 for equirectangular
    
    def test_custom_dimensions(self):
        """Test custom panorama dimensions."""
        config = OutputConfig(pano_width=8192)
        assert config.pano_width == 8192
        assert config.pano_height == 4096
    
    def test_output_format(self):
        """Test output format options."""
        config_jpg = OutputConfig(output_format="jpg")
        config_png = OutputConfig(output_format="png")
        assert config_jpg.output_format == "jpg"
        assert config_png.output_format == "png"


class TestCalibrationData:
    """Tests for CalibrationData dataclass."""
    
    def test_camera_matrix_K(self, sample_calibration):
        """Test camera matrix K property."""
        K = sample_calibration.K
        assert K.shape == (3, 3)
        assert K[0, 0] == 1000.0  # fx
        assert K[1, 1] == 1000.0  # fy
        assert K[0, 2] == 400.0   # cx
        assert K[1, 2] == 300.0   # cy
        assert K[2, 2] == 1.0
    
    def test_to_dict(self, sample_calibration):
        """Test conversion to dictionary."""
        d = sample_calibration.to_dict()
        assert d["fx"] == 1000.0
        assert d["fy"] == 1000.0
        assert d["cx"] == 400.0
        assert d["cy"] == 300.0
        assert d["source"] == "test"
    
    def test_from_json(self, temp_dir):
        """Test loading from JSON file."""
        calib_data = {
            "fx": 1500.0,
            "fy": 1500.0,
            "cx": 640.0,
            "cy": 480.0,
            "dist_coeffs": [0.1, -0.2, 0.0, 0.0, 0.05]
        }
        calib_path = temp_dir / "calib.json"
        with open(calib_path, 'w') as f:
            json.dump(calib_data, f)
        
        calib = CalibrationData.from_json(calib_path)
        assert calib.fx == 1500.0
        assert calib.fy == 1500.0
        assert calib.dist_coeffs == [0.1, -0.2, 0.0, 0.0, 0.05]


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_path_conversion(self, temp_dir):
        """Test automatic path conversion."""
        config = PipelineConfig(
            input_dir=str(temp_dir / "input"),
            output_dir=str(temp_dir / "output")
        )
        assert isinstance(config.input_dir, Path)
        assert isinstance(config.output_dir, Path)
    
    def test_to_dict(self, temp_dir):
        """Test conversion to dictionary."""
        config = PipelineConfig(
            input_dir=temp_dir / "input",
            output_dir=temp_dir / "output"
        )
        d = config.to_dict()
        
        assert "input_dir" in d
        assert "output_dir" in d
        assert "matching" in d
        assert "intrinsics" in d
        assert "blending" in d
        assert "output" in d
        assert "debug" in d
    
    def test_nested_configs(self, temp_dir):
        """Test nested configuration objects."""
        config = PipelineConfig(
            input_dir=temp_dir,
            output_dir=temp_dir,
            matching=MatchingConfig(min_inliers=40),
            blending=BlendingConfig(method="feather")
        )
        assert config.matching.min_inliers == 40
        assert config.blending.method == "feather"

