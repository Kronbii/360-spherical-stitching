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


def interpolate_rotation_from_neighbors(
    match_results: List[MatchResult],
    relative_rotations: List[np.ndarray],
    failed_idx: int
) -> np.ndarray:
    """
    Interpolate rotation for a failed match using neighboring successful rotations.
    
    Args:
        match_results: List of all match results.
        relative_rotations: List of computed relative rotations (may have identity for failed ones).
        failed_idx: Index of the failed match in match_results.
        
    Returns:
        Interpolated rotation matrix, or identity if no neighbors available.
    """
    # Try previous neighbor first (most reliable for temporal sequences)
    if failed_idx > 0 and match_results[failed_idx - 1].success:
        prev_R = relative_rotations[failed_idx - 1]
        if not np.allclose(prev_R, np.eye(3)):
            logger.debug(f"Pair ({match_results[failed_idx].src_idx},{match_results[failed_idx].dst_idx}): "
                        f"Using previous successful rotation (pair {failed_idx - 1})")
            return prev_R.copy()
    
    # Try next neighbor
    if failed_idx < len(match_results) - 1 and match_results[failed_idx + 1].success:
        next_R = relative_rotations[failed_idx + 1]
        if not np.allclose(next_R, np.eye(3)):
            logger.debug(f"Pair ({match_results[failed_idx].src_idx},{match_results[failed_idx].dst_idx}): "
                        f"Using next successful rotation (pair {failed_idx + 1})")
            return next_R.copy()
    
    # If both neighbors exist and are successful, average them (in rotation space)
    if (failed_idx > 0 and failed_idx < len(match_results) - 1 and
        match_results[failed_idx - 1].success and match_results[failed_idx + 1].success):
        prev_R = relative_rotations[failed_idx - 1]
        next_R = relative_rotations[failed_idx + 1]
        
        if not np.allclose(prev_R, np.eye(3)) and not np.allclose(next_R, np.eye(3)):
            # Average rotations using matrix logarithm (Lie algebra)
            # For small rotations, this is approximately: (log(R1) + log(R2)) / 2
            # Simplified: use direct average and re-orthonormalize (good enough for small rotations)
            R_avg = (prev_R + next_R) / 2
            R_interp = orthonormalize_rotation(R_avg)
            logger.debug(f"Pair ({match_results[failed_idx].src_idx},{match_results[failed_idx].dst_idx}): "
                        f"Interpolated from neighbors (pairs {failed_idx - 1} and {failed_idx + 1})")
            return R_interp
    
    # Fallback to identity
    logger.debug(f"Pair ({match_results[failed_idx].src_idx},{match_results[failed_idx].dst_idx}): "
                f"No successful neighbors, using identity")
    return np.eye(3)


def compute_relative_rotations(
    match_results: List[MatchResult],
    K: np.ndarray,
    scale_factor: float = 1.0
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Compute relative rotations from homographies for all matched pairs.
    Uses neighborhood interpolation for failed matches (better for video sequences).
    
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
    
    # First pass: compute rotations for successful matches
    for result in match_results:
        if result.homography is None or not result.success:
            # Placeholder - will be interpolated in second pass
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
    
    # Second pass: interpolate failed rotations from neighbors
    for i, result in enumerate(match_results):
        if result.homography is None or not result.success:
            R_interp = interpolate_rotation_from_neighbors(match_results, relative_rotations, i)
            relative_rotations[i] = R_interp
            
            # Update diagnostics
            if not np.allclose(R_interp, np.eye(3)):
                angle = rotation_angle_degrees(R_interp)
                diagnostics[i]["status"] = "interpolated"
                diagnostics[i]["angle_deg"] = angle
                logger.info(f"Pair ({result.src_idx},{result.dst_idx}): "
                           f"Interpolated rotation {angle:.2f}° from neighbors")
    
    return relative_rotations, diagnostics


def smooth_rotations_temporal(
    global_rotations: List[np.ndarray],
    window_size: int = 3
) -> List[np.ndarray]:
    """
    Apply temporal smoothing to global rotations using a moving average.
    
    For videos with smooth camera motion, this helps reduce jitter from
    failed matches or estimation errors. Uses direct matrix averaging with
    re-orthonormalization (works well for small rotations in smooth sequences).
    
    Args:
        global_rotations: List of global rotation matrices.
        window_size: Size of smoothing window (must be odd, default 3).
        
    Returns:
        Smoothed list of global rotation matrices.
    """
    if len(global_rotations) <= 2:
        return global_rotations
    
    if window_size % 2 == 0:
        window_size += 1  # Make odd
    if window_size < 3:
        window_size = 3
    
    half_window = window_size // 2
    smoothed = [global_rotations[0]]  # First rotation stays the same (reference)
    
    logger.info(f"Applying temporal smoothing to rotations (window size: {window_size})...")
    
    for i in range(1, len(global_rotations) - 1):
        # Collect rotations in window
        start_idx = max(0, i - half_window)
        end_idx = min(len(global_rotations), i + half_window + 1)
        
        # Collect all rotations in the window
        window_rots = []
        for j in range(start_idx, end_idx):
            window_rots.append(global_rotations[j])
        
        # Average rotation matrices directly (works for small rotations)
        # Re-orthonormalize to ensure valid rotation
        R_avg = np.mean(window_rots, axis=0)
        R_smooth = orthonormalize_rotation(R_avg)
        smoothed.append(R_smooth)
    
    # Last rotation stays the same
    smoothed.append(global_rotations[-1])
    
    logger.info(f"Applied temporal smoothing to {len(global_rotations)} rotations (window: {window_size})")
    return smoothed


def chain_rotations(relative_rotations: List[np.ndarray], apply_smoothing: bool = True, smoothing_window: int = 3) -> List[np.ndarray]:
    """
    Chain relative rotations to get global rotations.
    
    R_global[0] = I (identity - first image is reference)
    R_global[i+1] = R_rel[i] @ R_global[i]
    
    Args:
        relative_rotations: List of relative rotations R_rel[i] (from image i to i+1).
        apply_smoothing: If True, apply temporal smoothing to global rotations (good for videos).
        smoothing_window: Size of smoothing window (odd, 3-15 recommended, larger = straighter lines).
        
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
    
    # Apply temporal smoothing for video sequences
    if apply_smoothing and len(global_rotations) > 3:
        global_rotations = smooth_rotations_temporal(global_rotations, window_size=smoothing_window)
    
    return global_rotations


def yaw_degrees(R: np.ndarray) -> float:
    """
    Yaw (azimuth) of a global rotation, in degrees, wrapped to [-180, +180].

    This is the direction the camera faces around the vertical axis, taken from the
    rotated optical axis: R @ [0, 0, 1].

    Args:
        R: 3x3 global rotation matrix (camera to world).

    Returns:
        Yaw angle in degrees.
    """
    return float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))


def unwrapped_yaw_degrees(global_rotations: List[np.ndarray]) -> np.ndarray:
    """
    Yaw of every frame, unwrapped so a sweep past ±180° keeps counting.

    Needed because rotation_angle_degrees() saturates at 180° and cannot express how
    far a panorama sweep has actually travelled.

    Args:
        global_rotations: List of global rotation matrices.

    Returns:
        Array of unwrapped yaw angles in degrees, starting from the first frame's yaw.
    """
    if not global_rotations:
        return np.zeros(0)
    yaws = np.array([yaw_degrees(R) for R in global_rotations])
    return np.degrees(np.unwrap(np.radians(yaws)))


def sweep_span_degrees(global_rotations: List[np.ndarray]) -> float:
    """
    Total yaw travelled from the first frame to the last, in degrees.

    Unlike rotation_angle_degrees(global_rotations[-1] @ global_rotations[0].T), this
    does not saturate: a 332° sweep reports 332°, and a 400° sweep reports 400°.

    Args:
        global_rotations: List of global rotation matrices.

    Returns:
        Absolute yaw span in degrees.
    """
    y = unwrapped_yaw_degrees(global_rotations)
    return float(abs(y[-1] - y[0])) if len(y) > 1 else 0.0


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
    
    For panoramas, we use the direct rotation from first to last frame as the primary metric.
    This correctly handles 360° panoramas where the last frame is close to the first.
    We also calculate cumulative rotation for comparison (which can exceed 360° due to error accumulation).
    
    Args:
        global_rotations: List of global rotation matrices.
        
    Returns:
        Dictionary with coverage statistics.
    """
    if len(global_rotations) < 2:
        return {"total_deg": 0, "per_image_avg_deg": 0, "n_images": 1}
    
    # Calculate per-step rotation angles
    per_image_angles = []
    for i in range(1, len(global_rotations)):
        R_step = global_rotations[i] @ global_rotations[i-1].T
        per_image_angles.append(rotation_angle_degrees(R_step))
    
    # Calculate cumulative total by summing per-step angles
    # This can exceed 360° due to error accumulation or if video captures >360°
    cumulative_total = sum(per_image_angles)
    
    # Calculate direct rotation from first to last
    # For a 360° panorama, this should be close to 360° (or 0° if exactly closed)
    R_total = global_rotations[-1] @ global_rotations[0].T
    direct_total = rotation_angle_degrees(R_total)
    
    # Determine primary metric based on the relationship between direct and cumulative
    # Strategy:
    # 1. If cumulative is reasonable (≤ 450°), use it (represents actual path traveled)
    # 2. If cumulative is excessive (> 450°), there's likely error accumulation:
    #    - If direct is small (< 30°), it's likely a closed 360° loop - use 360°
    #    - If direct is moderate (30-180°), it's likely a 360° loop with error - use 360° + small offset
    #    - If direct is large (> 180°), use direct (not a closed loop, actual rotation)
    if cumulative_total <= 450.0:
        # Cumulative is reasonable - use it as it represents the actual path traveled
        total_angle = cumulative_total
    else:
        # Cumulative is excessive (> 450°) - likely error accumulation
        if direct_total < 30.0:
            # Direct is very small - likely a closed 360° loop with error accumulation
            total_angle = 360.0
            logger.debug(f"Direct rotation ({direct_total:.1f}°) suggests closed 360° loop, "
                        f"but cumulative ({cumulative_total:.1f}°) indicates error accumulation. "
                        f"Using 360° as estimate.")
        elif direct_total < 180.0:
            # Direct is moderate - likely a 360° loop with error accumulation
            # For a 360° panorama, report 360° (error accumulation causes cumulative overestimate)
            # Only add a small offset if direct suggests significant misalignment (> 90°)
            if direct_total < 90.0:
                # Small to moderate direct rotation - very likely exactly 360°
                total_angle = 360.0
            else:
                # Larger direct (> 90°) - some misalignment, use 360° + small offset
                offset = min((direct_total - 90.0) * 0.15, 20.0)  # Max 20° offset
                total_angle = 360.0 + offset
            
            logger.debug(f"Direct rotation ({direct_total:.1f}°) with excessive cumulative "
                        f"({cumulative_total:.1f}°) suggests 360° loop with error accumulation. "
                        f"Using estimate: {total_angle:.1f}°")
        else:
            # Direct is large - not a closed loop, use direct rotation
            total_angle = direct_total
            logger.debug(f"Direct rotation ({direct_total:.1f}°) is large, using it despite "
                        f"high cumulative ({cumulative_total:.1f}°).")
    
    avg_angle = np.mean(per_image_angles) if per_image_angles else 0
    
    return {
        "total_deg": total_angle,
        "direct_total_deg": direct_total,  # Direct rotation from first to last
        "cumulative_total_deg": cumulative_total,  # Sum of all per-step angles
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

