"""
Rotation extraction from homography matrices.

For pure rotation (no translation), the homography between two views is:
    H = K @ R @ K^{-1}
    
Therefore:
    R = K^{-1} @ H @ K
    
We orthonormalize R using SVD to ensure it's a valid rotation matrix.
"""

import logging
from typing import List, Tuple

import numpy as np

from .config import CalibrationData
from .features import MatchResult

logger = logging.getLogger(__name__)


def orthonormalize_rotation(R_raw: np.ndarray) -> np.ndarray:
    """
    Orthonormalize a 3x3 matrix to be a valid rotation matrix using SVD.
    
    R = U @ V^T
    If det(R) < 0, flip sign of last column of U.
    
    Args:
        R_raw: Approximate rotation matrix.
        
    Returns:
        Orthonormalized rotation matrix.
    """
    U, S, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    
    # Ensure det(R) = +1 (proper rotation, not reflection)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    return R


def extract_rotation_from_homography(
    H: np.ndarray,
    K: np.ndarray,
    scale_factor: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Extract rotation matrix from homography using camera intrinsics.
    
    For pure rotation: H = K @ R @ K^{-1}
    Therefore: R = K^{-1} @ H @ K
    
    Args:
        H: 3x3 homography matrix.
        K: 3x3 camera intrinsic matrix.
        scale_factor: Scale factor if homography was computed on scaled images.
        
    Returns:
        Tuple of (rotation matrix R, determinant of raw R for diagnostics).
    """
    # If homography was computed on scaled images, we need to adjust K
    # K_scaled = diag(s, s, 1) @ K for uniform scaling
    if scale_factor != 1.0:
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_factor  # fx
        K_scaled[1, 1] *= scale_factor  # fy
        K_scaled[0, 2] *= scale_factor  # cx
        K_scaled[1, 2] *= scale_factor  # cy
        K = K_scaled
    
    K_inv = np.linalg.inv(K)
    
    # R_raw = K^{-1} @ H @ K
    R_raw = K_inv @ H @ K
    
    # Log raw determinant for diagnostics
    det_raw = np.linalg.det(R_raw)
    
    # Orthonormalize
    R = orthonormalize_rotation(R_raw)
    
    return R, det_raw


def compute_relative_rotations(
    match_results: List[MatchResult],
    K: np.ndarray,
    scale_factor: float = 1.0
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Compute relative rotations from homographies for all matched pairs.
    
    Args:
        match_results: List of MatchResult with homographies.
        K: 3x3 camera intrinsic matrix.
        scale_factor: Scale factor if matching was on scaled images.
        
    Returns:
        Tuple of (list of relative rotations R_rel, list of diagnostic dicts).
    """
    relative_rotations = []
    diagnostics = []
    
    logger.info("Computing relative rotations from homographies...")
    
    for result in match_results:
        if result.homography is None or not result.success:
            # Use identity rotation if homography failed or matching was unsuccessful
            # For failed matches, assume no rotation (identity) - this allows pipeline to continue
            if result.inliers == 0:
                logger.debug(f"Pair ({result.src_idx},{result.dst_idx}): No matches, using identity rotation")
            else:
                logger.debug(f"Pair ({result.src_idx},{result.dst_idx}): Weak match ({result.inliers} inliers), using identity rotation")
            R_rel = np.eye(3)
            diag = {
                "pair": (result.src_idx, result.dst_idx),
                "status": "failed",
                "det_raw": 1.0,
                "inliers": result.inliers,
            }
        else:
            R_rel, det_raw = extract_rotation_from_homography(
                result.homography, K, scale_factor
            )
            
            # Compute rotation angle for logging
            angle = rotation_angle_degrees(R_rel)
            
            diag = {
                "pair": (result.src_idx, result.dst_idx),
                "status": "success",
                "det_raw": det_raw,
                "det_final": np.linalg.det(R_rel),
                "angle_deg": angle,
                "inliers": result.inliers,
            }
            
            logger.debug(f"Pair ({result.src_idx},{result.dst_idx}): "
                        f"rotation {angle:.2f}°, det_raw={det_raw:.4f}, "
                        f"inliers={result.inliers}")
        
        relative_rotations.append(R_rel)
        diagnostics.append(diag)
    
    return relative_rotations, diagnostics


def chain_rotations(relative_rotations: List[np.ndarray]) -> List[np.ndarray]:
    """
    Chain relative rotations to get global rotations.
    
    R_global[0] = I (identity - first image is reference)
    R_global[i+1] = R_rel[i] @ R_global[i]
    
    Args:
        relative_rotations: List of relative rotations R_rel[i] (from image i to i+1).
        
    Returns:
        List of global rotations R_global (length = len(relative_rotations) + 1).
    """
    n = len(relative_rotations) + 1
    global_rotations = [np.eye(3)]  # R_global[0] = I
    
    for i, R_rel in enumerate(relative_rotations):
        R_global_new = R_rel @ global_rotations[-1]
        
        # Re-orthonormalize to prevent drift
        R_global_new = orthonormalize_rotation(R_global_new)
        
        global_rotations.append(R_global_new)
    
    logger.info(f"Chained {len(relative_rotations)} relative rotations into {n} global rotations")
    
    return global_rotations


def rotation_angle_degrees(R: np.ndarray) -> float:
    """
    Compute rotation angle in degrees from rotation matrix.
    
    angle = arccos((trace(R) - 1) / 2)
    
    Args:
        R: 3x3 rotation matrix.
        
    Returns:
        Rotation angle in degrees.
    """
    trace = np.trace(R)
    # Clamp to valid range for arccos
    cos_angle = np.clip((trace - 1) / 2, -1, 1)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def rotation_axis(R: np.ndarray) -> np.ndarray:
    """
    Extract rotation axis from rotation matrix.
    
    Args:
        R: 3x3 rotation matrix.
        
    Returns:
        Unit vector rotation axis.
    """
    # For small angles, use approximation
    angle = rotation_angle_degrees(R)
    if angle < 1e-6:
        return np.array([0, 1, 0])  # Arbitrary axis for identity
    
    # Rotation axis is eigenvector with eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(R)
    
    # Find eigenvector with eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    axis = np.real(eigenvectors[:, idx])
    
    return axis / np.linalg.norm(axis)


def estimate_total_rotation_coverage(global_rotations: List[np.ndarray]) -> dict:
    """
    Estimate total rotation coverage for logging/diagnostics.
    
    Assumes camera primarily rotates around vertical (Y) axis for panorama.
    
    Args:
        global_rotations: List of global rotation matrices.
        
    Returns:
        Dictionary with coverage statistics.
    """
    if len(global_rotations) < 2:
        return {"total_deg": 0, "per_image_avg_deg": 0}
    
    # Calculate total rotation from first to last
    R_total = global_rotations[-1] @ global_rotations[0].T
    total_angle = rotation_angle_degrees(R_total)
    
    # Calculate average per-image rotation
    per_image_angles = []
    for i in range(1, len(global_rotations)):
        R_step = global_rotations[i] @ global_rotations[i-1].T
        per_image_angles.append(rotation_angle_degrees(R_step))
    
    avg_angle = np.mean(per_image_angles) if per_image_angles else 0
    
    return {
        "total_deg": total_angle,
        "per_image_avg_deg": avg_angle,
        "per_image_angles": per_image_angles,
        "n_images": len(global_rotations),
        "estimated_fov_coverage": total_angle,
    }


def log_rotation_summary(
    global_rotations: List[np.ndarray],
    diagnostics: List[dict]
) -> None:
    """
    Log summary of rotation estimation.
    
    Args:
        global_rotations: List of global rotation matrices.
        diagnostics: List of diagnostic dictionaries from relative rotation computation.
    """
    coverage = estimate_total_rotation_coverage(global_rotations)
    
    logger.info("=" * 50)
    logger.info("ROTATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Number of images: {coverage['n_images']}")
    logger.info(f"Total rotation coverage: {coverage['total_deg']:.1f}°")
    logger.info(f"Average rotation per step: {coverage['per_image_avg_deg']:.1f}°")
    
    if coverage['per_image_angles']:
        logger.info(f"Rotation range: {min(coverage['per_image_angles']):.1f}° - "
                   f"{max(coverage['per_image_angles']):.1f}°")
    
    # Log any issues
    failed = [d for d in diagnostics if d.get('status') == 'failed']
    if failed:
        logger.warning(f"WARNING: {len(failed)} pair(s) had failed homographies")
    
    # Check for suspicious rotations
    suspicious = [d for d in diagnostics 
                  if d.get('angle_deg', 0) > 60 or 
                  abs(d.get('det_raw', 1) - 1) > 0.5]
    if suspicious:
        logger.warning(f"WARNING: {len(suspicious)} pair(s) have suspicious rotation estimates")
        for d in suspicious:
            logger.warning(f"  Pair {d['pair']}: angle={d.get('angle_deg', 'N/A')}°, "
                          f"det_raw={d.get('det_raw', 'N/A')}")
    
    logger.info("=" * 50)

