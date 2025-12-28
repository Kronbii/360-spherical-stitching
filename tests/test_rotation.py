"""
Unit tests for src/rotation.py - Rotation extraction from homography.
"""

import numpy as np
import pytest

from src.config import CalibrationData
from src.features import MatchResult
from src.rotation import (
    chain_rotations,
    compute_relative_rotations,
    estimate_total_rotation_coverage,
    extract_rotation_from_homography,
    orthonormalize_rotation,
    rotation_angle_degrees,
    rotation_axis,
)


class TestOrthonormalizeRotation:
    """Tests for orthonormalize_rotation function."""
    
    def test_preserves_valid_rotation(self, sample_rotation_matrix):
        """Test that valid rotation is preserved."""
        R_out = orthonormalize_rotation(sample_rotation_matrix)
        
        # Should be very close to input
        assert np.allclose(R_out, sample_rotation_matrix, atol=1e-10)
    
    def test_orthonormalizes_noisy_matrix(self):
        """Test orthonormalization of noisy rotation."""
        # Create a noisy rotation matrix
        R_noisy = np.array([
            [0.99, -0.15, 0.02],
            [0.14, 0.98, 0.03],
            [-0.01, -0.04, 1.01]
        ])
        
        R_out = orthonormalize_rotation(R_noisy)
        
        # Should be orthonormal: R @ R^T = I
        assert np.allclose(R_out @ R_out.T, np.eye(3), atol=1e-10)
        
        # Should have det = 1
        assert np.allclose(np.linalg.det(R_out), 1.0, atol=1e-10)
    
    def test_handles_reflection(self):
        """Test handling of reflection (det = -1)."""
        # Reflection matrix
        R_reflect = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        R_out = orthonormalize_rotation(R_reflect)
        
        # Should be proper rotation (det = +1)
        assert np.linalg.det(R_out) > 0
    
    def test_identity_preserved(self):
        """Test identity matrix is preserved."""
        R_out = orthonormalize_rotation(np.eye(3))
        assert np.allclose(R_out, np.eye(3))


class TestExtractRotationFromHomography:
    """Tests for extract_rotation_from_homography function."""
    
    def test_extracts_rotation(self, sample_calibration):
        """Test extraction of rotation from homography."""
        # Create homography for pure rotation
        angle = np.radians(15)
        R_true = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        
        K = sample_calibration.K
        H = K @ R_true @ np.linalg.inv(K)
        
        R_extracted, det = extract_rotation_from_homography(H, K)
        
        # Should recover the rotation
        assert R_extracted.shape == (3, 3)
        assert np.allclose(R_extracted, R_true, atol=1e-6)
    
    def test_returns_determinant(self, sample_calibration):
        """Test that determinant is returned."""
        H = np.eye(3)
        K = sample_calibration.K
        
        R, det = extract_rotation_from_homography(H, K)
        
        assert isinstance(det, float)
    
    def test_handles_scale_factor(self, sample_calibration):
        """Test handling of scale factor."""
        H = np.eye(3)
        K = sample_calibration.K
        
        R1, _ = extract_rotation_from_homography(H, K, scale_factor=1.0)
        R2, _ = extract_rotation_from_homography(H, K, scale_factor=0.5)
        
        # Both should be valid rotations
        assert np.allclose(np.linalg.det(R1), 1.0)
        assert np.allclose(np.linalg.det(R2), 1.0)


class TestComputeRelativeRotations:
    """Tests for compute_relative_rotations function."""
    
    def test_computes_rotations(self, sample_calibration):
        """Test computation of relative rotations."""
        K = sample_calibration.K
        H = np.eye(3)  # Identity homography = identity rotation
        
        results = [
            MatchResult(0, 1, H, 100, 150, True),
            MatchResult(1, 2, H, 80, 120, True),
        ]
        
        rotations, diags = compute_relative_rotations(results, K)
        
        assert len(rotations) == 2
        assert len(diags) == 2
        assert all(R.shape == (3, 3) for R in rotations)
    
    def test_handles_failed_homography(self, sample_calibration):
        """Test handling of failed homography."""
        K = sample_calibration.K
        
        results = [
            MatchResult(0, 1, None, 10, 20, False),  # Failed
        ]
        
        rotations, diags = compute_relative_rotations(results, K)
        
        # Should use identity for failed matches
        assert len(rotations) == 1
        assert np.allclose(rotations[0], np.eye(3))
        assert diags[0]["status"] == "failed"
    
    def test_diagnostics_include_angle(self, sample_calibration):
        """Test that diagnostics include rotation angle."""
        K = sample_calibration.K
        H = np.eye(3)
        
        results = [MatchResult(0, 1, H, 100, 150, True)]
        
        _, diags = compute_relative_rotations(results, K)
        
        assert "angle_deg" in diags[0]


class TestChainRotations:
    """Tests for chain_rotations function."""
    
    def test_chains_identity(self):
        """Test chaining identity rotations."""
        R_rel = [np.eye(3), np.eye(3)]
        
        R_global = chain_rotations(R_rel)
        
        assert len(R_global) == 3  # n+1 global rotations
        assert all(np.allclose(R, np.eye(3)) for R in R_global)
    
    def test_first_rotation_is_identity(self, sample_rotation_matrix):
        """Test that first global rotation is identity."""
        R_rel = [sample_rotation_matrix]
        
        R_global = chain_rotations(R_rel)
        
        assert np.allclose(R_global[0], np.eye(3))
    
    def test_accumulates_rotations(self):
        """Test proper accumulation of rotations."""
        # Two 10-degree rotations around Y
        angle = np.radians(10)
        c, s = np.cos(angle), np.sin(angle)
        R_10 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        R_rel = [R_10, R_10]
        R_global = chain_rotations(R_rel)
        
        # Last rotation should be ~20 degrees
        total_angle = rotation_angle_degrees(R_global[-1])
        assert abs(total_angle - 20.0) < 0.1
    
    def test_maintains_orthonormality(self):
        """Test that chained rotations remain orthonormal."""
        # Create several rotations
        np.random.seed(42)
        R_rel = []
        for _ in range(10):
            angle = np.radians(np.random.uniform(-20, 20))
            c, s = np.cos(angle), np.sin(angle)
            R_rel.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
        
        R_global = chain_rotations(R_rel)
        
        # All should be orthonormal
        for R in R_global:
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestRotationAngleDegrees:
    """Tests for rotation_angle_degrees function."""
    
    def test_identity_angle(self):
        """Test identity has zero angle."""
        angle = rotation_angle_degrees(np.eye(3))
        assert abs(angle) < 1e-10
    
    def test_known_angle(self):
        """Test known rotation angle."""
        angle_deg = 30.0
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        computed_angle = rotation_angle_degrees(R)
        assert abs(computed_angle - angle_deg) < 0.01
    
    def test_90_degree_rotation(self):
        """Test 90 degree rotation."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        
        angle = rotation_angle_degrees(R)
        assert abs(angle - 90.0) < 0.01
    
    def test_180_degree_rotation(self):
        """Test 180 degree rotation."""
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        
        angle = rotation_angle_degrees(R)
        assert abs(angle - 180.0) < 0.01


class TestRotationAxis:
    """Tests for rotation_axis function."""
    
    def test_z_axis_rotation(self):
        """Test rotation around Z axis."""
        angle = np.radians(30)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        axis = rotation_axis(R)
        
        # Should be [0, 0, ±1]
        assert abs(axis[2]) > 0.99
        assert abs(axis[0]) < 0.01
        assert abs(axis[1]) < 0.01
    
    def test_unit_vector(self, sample_rotation_matrix):
        """Test that axis is unit vector."""
        axis = rotation_axis(sample_rotation_matrix)
        
        assert np.allclose(np.linalg.norm(axis), 1.0)


class TestEstimateTotalRotationCoverage:
    """Tests for estimate_total_rotation_coverage function."""
    
    def test_single_image(self):
        """Test with single image."""
        coverage = estimate_total_rotation_coverage([np.eye(3)])
        
        # With single image, returns minimal info
        assert coverage["total_deg"] == 0
        assert coverage["per_image_avg_deg"] == 0
    
    def test_calculates_total_rotation(self):
        """Test total rotation calculation."""
        # Create rotations: 0, 10, 20 degrees
        rotations = []
        for angle in [0, 10, 20]:
            angle_rad = np.radians(angle)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            rotations.append(R)
        
        coverage = estimate_total_rotation_coverage(rotations)
        
        assert abs(coverage["total_deg"] - 20.0) < 0.1
        assert coverage["n_images"] == 3
    
    def test_per_image_angles(self):
        """Test per-image angle calculation."""
        # Two 15-degree rotations
        angle = np.radians(15)
        c, s = np.cos(angle), np.sin(angle)
        R_15 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        rotations = [np.eye(3), R_15, R_15 @ R_15]
        
        coverage = estimate_total_rotation_coverage(rotations)
        
        assert len(coverage["per_image_angles"]) == 2
        assert all(abs(a - 15.0) < 0.1 for a in coverage["per_image_angles"])

