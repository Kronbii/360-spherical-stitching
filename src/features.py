"""
Feature extraction and matching module using ORB.

Uses:
- ORB features for detection and description
- BFMatcher with Hamming distance
- kNN matching with Lowe's ratio test
- RANSAC for robust homography estimation
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import MatchingConfig

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of feature matching between two images."""
    src_idx: int  # Source image index
    dst_idx: int  # Destination image index
    homography: Optional[np.ndarray]  # 3x3 homography matrix (if successful)
    inliers: int  # Number of RANSAC inliers
    total_matches: int  # Total matches after ratio test
    success: bool  # Whether matching was successful
    src_points: Optional[np.ndarray] = None  # Source keypoints for inliers
    dst_points: Optional[np.ndarray] = None  # Destination keypoints for inliers
    inlier_mask: Optional[np.ndarray] = None  # RANSAC inlier mask


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to image.
    
    This is used ONLY for feature extraction, not for final colors.
    
    Args:
        image: Input BGR image.
        
    Returns:
        CLAHE-enhanced grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def extract_orb_features(
    image: np.ndarray,
    nfeatures: int = 3000,
    use_clahe: bool = False
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract ORB features from an image.
    
    Args:
        image: Input BGR image.
        nfeatures: Number of features to detect.
        use_clahe: Whether to apply CLAHE before detection.
        
    Returns:
        Tuple of (keypoints, descriptors).
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        if use_clahe:
            gray = apply_clahe(image)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create ORB detector with specified settings
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    
    # Detect and compute
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    if descriptors is None:
        return [], np.array([])
    
    return keypoints, descriptors


def match_features_knn(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.7,
    symmetric: bool = True
) -> List[cv2.DMatch]:
    """
    Match features using BFMatcher with kNN and ratio test.
    
    Args:
        desc1: Descriptors from first image.
        desc2: Descriptors from second image.
        ratio_threshold: Lowe's ratio test threshold (lower = stricter).
        symmetric: If True, perform symmetric matching (cross-check) for better quality.
        
    Returns:
        List of good matches after ratio test.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    # Create BFMatcher with Hamming distance (for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # kNN match with k=2
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except cv2.error as e:
        logger.warning(f"kNN matching failed: {e}")
        return []
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    # Symmetric matching (cross-check): match in both directions and keep only consistent matches
    if symmetric and len(good_matches) > 0:
        try:
            # Match in reverse direction
            matches_reverse = bf.knnMatch(desc2, desc1, k=2)
            
            # Create set of reverse matches for fast lookup
            reverse_match_map = {}
            for match_pair in matches_reverse:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        # Reverse match: trainIdx -> queryIdx
                        reverse_match_map[m.trainIdx] = m.queryIdx
            
            # Keep only matches that are consistent in both directions
            symmetric_matches = []
            for match in good_matches:
                # Check if reverse match exists and points to same keypoint
                if match.trainIdx in reverse_match_map:
                    if reverse_match_map[match.trainIdx] == match.queryIdx:
                        symmetric_matches.append(match)
            
            logger.debug(f"Symmetric matching: {len(good_matches)} -> {len(symmetric_matches)} matches")
            return symmetric_matches
        except cv2.error as e:
            logger.debug(f"Symmetric matching failed, using one-way matches: {e}")
            return good_matches
    
    return good_matches


def find_homography_ransac(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    reproj_threshold: float = 3.0,
    max_iters: int = 3000,
    refine: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Find homography between two images using RANSAC.
    
    Args:
        kp1: Keypoints from first image.
        kp2: Keypoints from second image.
        matches: Good matches from ratio test.
        reproj_threshold: RANSAC reprojection threshold in pixels.
        max_iters: Maximum RANSAC iterations.
        refine: If True, refine homography using all inliers after RANSAC.
        
    Returns:
        Tuple of (homography matrix, inlier mask, number of inliers).
    """
    if len(matches) < 4:
        return None, None, 0
    
    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography with RANSAC
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
        maxIters=max_iters,
        confidence=0.995
    )
    
    if H is None or mask is None:
        return None, None, 0
    
    inliers = int(mask.sum())
    
    # Refinement: recompute homography using all inliers with least squares for better accuracy
    if refine and inliers >= 4:
        inlier_pts1 = src_pts[mask.ravel() == 1]
        inlier_pts2 = dst_pts[mask.ravel() == 1]
        
        # Use all inliers to compute refined homography (direct least squares, no RANSAC)
        try:
            H_refined = cv2.findHomography(
                inlier_pts1, inlier_pts2,
                method=0  # 0 = all points (least squares), faster and more accurate for inliers
            )[0]
            
            if H_refined is not None:
                # Verify refined homography by checking reprojection error
                # Project points and check how many are still inliers
                projected = cv2.perspectiveTransform(inlier_pts1, H_refined)
                errors = np.linalg.norm(projected - inlier_pts2, axis=2).ravel()
                refined_inlier_mask = errors < reproj_threshold * 1.5  # Slightly more lenient
                refined_inliers = int(refined_inlier_mask.sum())
                
                # Use refined homography if it maintains most inliers
                if refined_inliers >= inliers * 0.9:
                    H = H_refined
                    logger.debug(f"Homography refined: {refined_inliers} inliers (was {inliers})")
        except Exception as e:
            logger.debug(f"Homography refinement failed: {e}, using original")
    
    return H, mask, inliers


def match_image_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    idx1: int,
    idx2: int,
    config: MatchingConfig,
    scale: float = 1.0
) -> MatchResult:
    """
    Match features between two images and compute homography.
    
    Args:
        img1: First image (BGR).
        img2: Second image (BGR).
        idx1: Index of first image.
        idx2: Index of second image.
        config: Matching configuration.
        scale: Scale factor (for adjusting RANSAC threshold).
        
    Returns:
        MatchResult with homography and statistics.
    """
    # Extract features
    kp1, desc1 = extract_orb_features(img1, config.orb_nfeatures, config.use_clahe)
    kp2, desc2 = extract_orb_features(img2, config.orb_nfeatures, config.use_clahe)
    
    logger.debug(f"Image {idx1}: {len(kp1)} keypoints, Image {idx2}: {len(kp2)} keypoints")
    
    if len(kp1) < 10 or len(kp2) < 10:
        logger.warning(f"Too few keypoints: img{idx1}={len(kp1)}, img{idx2}={len(kp2)}")
        return MatchResult(
            src_idx=idx1, dst_idx=idx2,
            homography=None, inliers=0, total_matches=0, success=False
        )
    
    # Match features
    matches = match_features_knn(desc1, desc2, config.ratio_test_threshold, config.symmetric_matching)
    logger.debug(f"Pair ({idx1},{idx2}): {len(matches)} matches after ratio test")
    
    if len(matches) < 4:
        return MatchResult(
            src_idx=idx1, dst_idx=idx2,
            homography=None, inliers=0, total_matches=len(matches), success=False
        )
    
    # Find homography
    # Scale RANSAC threshold if images were downscaled
    # Also scale by image size for better adaptation (base threshold assumes ~1600px width)
    img_width = img1.shape[1]
    base_width = 1600.0
    size_scale = img_width / base_width
    scaled_threshold = config.ransac_reproj_threshold * scale * max(0.5, min(2.0, size_scale))
    H, mask, inliers = find_homography_ransac(
        kp1, kp2, matches, 
        scaled_threshold, 
        max_iters=config.ransac_max_iters,
        refine=config.ransac_refinement
    )
    
    if H is None:
        return MatchResult(
            src_idx=idx1, dst_idx=idx2,
            homography=None, inliers=0, total_matches=len(matches), success=False
        )
    
    success = inliers >= config.min_inliers
    
    # Extract inlier points if successful
    src_pts = None
    dst_pts = None
    if success and mask is not None:
        inlier_matches = [m for m, flag in zip(matches, mask.ravel()) if flag]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in inlier_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in inlier_matches])
    
    logger.debug(f"Pair ({idx1},{idx2}): {inliers} inliers, success={success}")
    
    return MatchResult(
        src_idx=idx1, dst_idx=idx2,
        homography=H, inliers=inliers, total_matches=len(matches),
        success=success,
        src_points=src_pts, dst_points=dst_pts,
        inlier_mask=mask
    )


def interpolate_homography(H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
    """
    Interpolate between two homographies (approximate midpoint).
    
    This is a simple approximation using matrix square root concept.
    For pure rotations, this gives reasonable intermediate rotation.
    
    Args:
        H1: First homography (identity or previous).
        H2: Second homography (skipped pair).
        
    Returns:
        Interpolated homography.
    """
    # Simple linear interpolation in matrix space (approximate)
    # For pure rotation homographies, this is a reasonable approximation
    # More sophisticated: use Lie algebra (matrix logarithm)
    return (H1 + H2) / 2.0


def match_sequential_pairs(
    images: List[np.ndarray],
    config: MatchingConfig,
    scale: float = 1.0
) -> List[MatchResult]:
    """
    Match sequential pairs of images (i, i+1).
    
    If matching fails for a pair, attempts (i, i+2) for stability and
    interpolates the intermediate homography.
    
    Args:
        images: List of images (BGR, already downscaled for matching).
        config: Matching configuration.
        scale: Scale factor relative to original images.
        
    Returns:
        List of MatchResult for each pair (length = len(images) - 1).
    """
    n_images = len(images)
    results = []
    skip_next = False
    
    logger.info(f"Matching {n_images - 1} sequential pairs...")
    
    i = 0
    while i < n_images - 1:
        if skip_next:
            skip_next = False
            i += 1
            continue
            
        # Try adjacent pair first
        result = match_image_pair(images[i], images[i + 1], i, i + 1, config, scale)
        
        if result.success:
            results.append(result)
            logger.info(f"Pair ({i},{i+1}): {result.inliers} inliers ✓")
            i += 1
        else:
            # Try skipping one image for stability
            if i + 2 < n_images:
                logger.warning(f"Pair ({i},{i+1}) weak ({result.inliers} inliers), trying ({i},{i+2})")
                skip_result = match_image_pair(images[i], images[i + 2], i, i + 2, config, scale)
                
                if skip_result.success:
                    logger.info(f"Pair ({i},{i+2}): {skip_result.inliers} inliers ✓ (skip match)")
                    
                    # Create interpolated result for (i, i+1)
                    # Approximate: use sqrt of skip homography
                    if skip_result.homography is not None:
                        # Use matrix square root approximation for interpolation
                        H_skip = skip_result.homography
                        # Simple approximation: identity interpolation
                        H_half = interpolate_homography(np.eye(3), H_skip)
                        
                        interp_result = MatchResult(
                            src_idx=i,
                            dst_idx=i + 1,
                            homography=H_half,
                            inliers=skip_result.inliers // 2,  # Approximate
                            total_matches=skip_result.total_matches,
                            success=True,  # Mark as success since skip worked
                        )
                        results.append(interp_result)
                        logger.info(f"Pair ({i},{i+1}): interpolated from skip match")
                        
                        # Also add result for (i+1, i+2) as interpolated
                        interp_result2 = MatchResult(
                            src_idx=i + 1,
                            dst_idx=i + 2,
                            homography=H_half,
                            inliers=skip_result.inliers // 2,
                            total_matches=skip_result.total_matches,
                            success=True,
                        )
                        results.append(interp_result2)
                        logger.info(f"Pair ({i+1},{i+2}): interpolated from skip match")
                        
                        i += 2  # Skip both pairs since we handled them
                        continue
            
            # No skip match available or it failed - add weak result
            results.append(result)
            logger.warning(f"Pair ({i},{i+1}): {result.inliers} inliers - WEAK")
            i += 1
    
    # Summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"Matching complete: {successful}/{len(results)} pairs successful")
    
    return results


def draw_matches_visualization(
    img1: np.ndarray,
    img2: np.ndarray,
    result: MatchResult,
    max_matches: int = 100
) -> np.ndarray:
    """
    Draw match visualization for debugging.
    
    Args:
        img1: First image.
        img2: Second image.
        result: MatchResult containing match information.
        max_matches: Maximum number of matches to draw.
        
    Returns:
        Visualization image.
    """
    # Re-extract features to get keypoints for visualization
    kp1, desc1 = extract_orb_features(img1)
    kp2, desc2 = extract_orb_features(img2)
    
    # Match again for visualization
    matches = match_features_knn(desc1, desc2)
    
    # Draw matches
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_matches],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Add text overlay
    cv2.putText(vis, f"Pair ({result.src_idx},{result.dst_idx}): {result.inliers} inliers",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return vis


def save_match_visualization(
    img1: np.ndarray,
    img2: np.ndarray,
    result: MatchResult,
    output_path: Path
) -> None:
    """
    Save match visualization to file.
    
    Args:
        img1: First image.
        img2: Second image.
        result: MatchResult.
        output_path: Path to save visualization.
    """
    vis = draw_matches_visualization(img1, img2, result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    logger.debug(f"Saved match visualization to {output_path}")


def check_matching_quality(results: List[MatchResult], min_inliers: int) -> Tuple[bool, str]:
    """
    Check overall matching quality and provide diagnostic message.
    
    Args:
        results: List of MatchResult from sequential matching.
        min_inliers: Minimum inliers threshold.
        
    Returns:
        Tuple of (success, diagnostic message).
    """
    failed_pairs = [(r.src_idx, r.dst_idx, r.inliers) for r in results if not r.success]
    
    if not failed_pairs:
        return True, "All pairs matched successfully"
    
    msg_parts = [f"Matching failed for {len(failed_pairs)} pair(s):"]
    for src, dst, inliers in failed_pairs:
        msg_parts.append(f"  - Pair ({src},{dst}): {inliers} inliers (need {min_inliers})")
    
    msg_parts.append("\nSuggestions:")
    msg_parts.append("  1. Ensure sufficient overlap between images (30-50%)")
    msg_parts.append("  2. Try --clahe flag for better feature detection in low contrast")
    msg_parts.append("  3. Check if images are in correct order")
    msg_parts.append("  4. Reduce movement between shots")
    
    return False, "\n".join(msg_parts)

