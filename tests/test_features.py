"""
Unit tests for src/features.py - ORB feature matching.
"""

import numpy as np
import pytest
import cv2

from src.config import MatchingConfig
from src.features import (
    MatchResult,
    apply_clahe,
    check_matching_quality,
    draw_matches_visualization,
    extract_orb_features,
    find_homography_ransac,
    interpolate_homography,
    match_features_knn,
    match_image_pair,
    match_sequential_pairs,
)


class TestApplyCLAHE:
    """Tests for apply_clahe function."""
    
    def test_returns_grayscale(self, sample_image):
        """Test CLAHE returns grayscale image."""
        result = apply_clahe(sample_image)
        assert len(result.shape) == 2  # Grayscale
    
    def test_same_dimensions(self, sample_image):
        """Test output has same dimensions as input."""
        result = apply_clahe(sample_image)
        assert result.shape == sample_image.shape[:2]
    
    def test_enhances_contrast(self, sample_image):
        """Test that CLAHE modifies the image."""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        result = apply_clahe(sample_image)
        
        # Should be different from simple grayscale
        assert not np.array_equal(result, gray)


class TestExtractORBFeatures:
    """Tests for extract_orb_features function."""
    
    def test_extracts_keypoints(self, sample_image):
        """Test extraction of keypoints."""
        kp, desc = extract_orb_features(sample_image)
        
        assert len(kp) > 0
        assert desc is not None
    
    def test_descriptor_shape(self, sample_image):
        """Test descriptor array shape."""
        kp, desc = extract_orb_features(sample_image)
        
        # ORB descriptors are 32 bytes (256 bits)
        assert desc.shape[1] == 32
        assert desc.shape[0] == len(kp)
    
    def test_respects_nfeatures(self, sample_image):
        """Test nfeatures parameter."""
        kp, _ = extract_orb_features(sample_image, nfeatures=100)
        assert len(kp) <= 100
    
    def test_clahe_option(self, sample_image):
        """Test CLAHE preprocessing option."""
        kp_normal, _ = extract_orb_features(sample_image, use_clahe=False)
        kp_clahe, _ = extract_orb_features(sample_image, use_clahe=True)
        
        # Both should extract features, may have different counts
        assert len(kp_normal) > 0
        assert len(kp_clahe) > 0
    
    def test_handles_grayscale_input(self, sample_image):
        """Test handling of grayscale input."""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        kp, desc = extract_orb_features(gray)
        
        assert len(kp) > 0


class TestMatchFeaturesKNN:
    """Tests for match_features_knn function."""
    
    def test_finds_matches(self, sample_image_pair):
        """Test finding matches between images."""
        img1, img2 = sample_image_pair
        _, desc1 = extract_orb_features(img1)
        _, desc2 = extract_orb_features(img2)
        
        matches = match_features_knn(desc1, desc2)
        assert len(matches) > 0
    
    def test_ratio_test_filters(self, sample_image_pair):
        """Test that ratio test reduces matches."""
        img1, img2 = sample_image_pair
        _, desc1 = extract_orb_features(img1)
        _, desc2 = extract_orb_features(img2)
        
        matches_strict = match_features_knn(desc1, desc2, ratio_threshold=0.5)
        matches_loose = match_features_knn(desc1, desc2, ratio_threshold=0.9)
        
        assert len(matches_strict) <= len(matches_loose)
    
    def test_handles_empty_descriptors(self):
        """Test handling empty descriptors."""
        matches = match_features_knn(np.array([]), np.array([]))
        assert len(matches) == 0
    
    def test_handles_none_descriptors(self):
        """Test handling None descriptors."""
        matches = match_features_knn(None, None)
        assert len(matches) == 0


class TestFindHomographyRANSAC:
    """Tests for find_homography_ransac function."""
    
    def test_finds_homography(self, sample_image_pair):
        """Test finding homography with RANSAC."""
        img1, img2 = sample_image_pair
        kp1, desc1 = extract_orb_features(img1)
        kp2, desc2 = extract_orb_features(img2)
        matches = match_features_knn(desc1, desc2)
        
        H, mask, inliers = find_homography_ransac(kp1, kp2, matches)
        
        assert H is not None
        assert H.shape == (3, 3)
        assert inliers > 0
    
    def test_returns_inlier_mask(self, sample_image_pair):
        """Test that inlier mask is returned."""
        img1, img2 = sample_image_pair
        kp1, desc1 = extract_orb_features(img1)
        kp2, desc2 = extract_orb_features(img2)
        matches = match_features_knn(desc1, desc2)
        
        H, mask, inliers = find_homography_ransac(kp1, kp2, matches)
        
        assert mask is not None
        assert mask.sum() == inliers
    
    def test_handles_insufficient_matches(self):
        """Test handling fewer than 4 matches."""
        kp1 = [cv2.KeyPoint(0, 0, 1)]
        kp2 = [cv2.KeyPoint(0, 0, 1)]
        matches = [cv2.DMatch(0, 0, 0)]
        
        H, mask, inliers = find_homography_ransac(kp1, kp2, matches)
        
        assert H is None
        assert inliers == 0


class TestMatchImagePair:
    """Tests for match_image_pair function."""
    
    def test_returns_match_result(self, sample_image_pair):
        """Test that MatchResult is returned."""
        img1, img2 = sample_image_pair
        config = MatchingConfig(min_inliers=10)
        
        result = match_image_pair(img1, img2, 0, 1, config)
        
        assert isinstance(result, MatchResult)
        assert result.src_idx == 0
        assert result.dst_idx == 1
    
    def test_successful_match(self, sample_image_pair):
        """Test successful matching result."""
        img1, img2 = sample_image_pair
        config = MatchingConfig(min_inliers=10)
        
        result = match_image_pair(img1, img2, 0, 1, config)
        
        assert result.success is True
        assert result.homography is not None
        assert result.inliers >= config.min_inliers
    
    def test_unsuccessful_match_insufficient_inliers(self, sample_image):
        """Test unsuccessful match with unrelated images."""
        # Create completely different images
        img1 = sample_image.copy()
        np.random.seed(100)
        img2 = np.random.randint(0, 255, sample_image.shape, dtype=np.uint8)
        
        config = MatchingConfig(min_inliers=100)
        result = match_image_pair(img1, img2, 0, 1, config)
        
        # May or may not find homography, but unlikely to have many inliers
        assert result.inliers < 100


class TestInterpolateHomography:
    """Tests for interpolate_homography function."""
    
    def test_interpolates_identity(self):
        """Test interpolation with identity."""
        H1 = np.eye(3)
        H2 = np.array([[1, 0, 100], [0, 1, 50], [0, 0, 1]], dtype=np.float64)
        
        H_interp = interpolate_homography(H1, H2)
        
        # Should be between H1 and H2
        assert H_interp[0, 2] == 50  # Half of 100
        assert H_interp[1, 2] == 25  # Half of 50
    
    def test_returns_3x3_matrix(self):
        """Test output shape."""
        H1 = np.eye(3)
        H2 = np.eye(3) * 2
        
        H_interp = interpolate_homography(H1, H2)
        
        assert H_interp.shape == (3, 3)


class TestMatchSequentialPairs:
    """Tests for match_sequential_pairs function."""
    
    def test_matches_all_pairs(self, sample_image_sequence):
        """Test matching all sequential pairs."""
        config = MatchingConfig(min_inliers=10)
        
        results = match_sequential_pairs(sample_image_sequence, config)
        
        # Should have n-1 results for n images
        assert len(results) >= len(sample_image_sequence) - 1
    
    def test_results_are_sequential(self, sample_image_sequence):
        """Test that results cover sequential pairs."""
        config = MatchingConfig(min_inliers=10)
        
        results = match_sequential_pairs(sample_image_sequence, config)
        
        # Check indices are sequential
        covered_pairs = set()
        for r in results:
            covered_pairs.add((r.src_idx, r.dst_idx))
        
        # All adjacent pairs should be covered
        for i in range(len(sample_image_sequence) - 1):
            assert (i, i + 1) in covered_pairs


class TestCheckMatchingQuality:
    """Tests for check_matching_quality function."""
    
    def test_all_successful(self):
        """Test with all successful matches."""
        results = [
            MatchResult(0, 1, np.eye(3), 100, 150, True),
            MatchResult(1, 2, np.eye(3), 80, 120, True),
        ]
        
        success, message = check_matching_quality(results, 60)
        
        assert success is True
        assert "successfully" in message.lower()
    
    def test_with_failures(self):
        """Test with some failed matches."""
        results = [
            MatchResult(0, 1, np.eye(3), 100, 150, True),
            MatchResult(1, 2, None, 20, 50, False),
        ]
        
        success, message = check_matching_quality(results, 60)
        
        assert success is False
        assert "failed" in message.lower()
        assert "(1,2)" in message
    
    def test_provides_suggestions(self):
        """Test that failure message includes suggestions."""
        results = [
            MatchResult(0, 1, None, 20, 50, False),
        ]
        
        success, message = check_matching_quality(results, 60)
        
        assert "suggestions" in message.lower() or "Suggestions" in message


class TestDrawMatchesVisualization:
    """Tests for draw_matches_visualization function."""
    
    def test_returns_image(self, sample_image_pair):
        """Test that visualization image is returned."""
        img1, img2 = sample_image_pair
        result = MatchResult(0, 1, np.eye(3), 50, 100, True)
        
        vis = draw_matches_visualization(img1, img2, result)
        
        assert vis is not None
        assert len(vis.shape) == 3  # Color image
    
    def test_wider_than_input(self, sample_image_pair):
        """Test that visualization is wider than single input."""
        img1, img2 = sample_image_pair
        result = MatchResult(0, 1, np.eye(3), 50, 100, True)
        
        vis = draw_matches_visualization(img1, img2, result)
        
        # Should be at least as wide as both images combined
        assert vis.shape[1] >= img1.shape[1] + img2.shape[1]

