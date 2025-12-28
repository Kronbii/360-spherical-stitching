"""
Circular matching module for detecting wrap-around overlap in 360° panoramas.

Detects when a video captures more than 360°, causing the end to overlap with the beginning.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .config import MatchingConfig
from .features import MatchResult, match_image_pair
from .rotation import rotation_angle_degrees

logger = logging.getLogger(__name__)


def detect_circular_closure(
    images: List[np.ndarray],
    match_results: List[MatchResult],
    global_rotations: List[np.ndarray],
    config: MatchingConfig,
    scale: float = 1.0,
    check_last_n_frames: int = 15,
    check_first_n_frames: int = 20,
    min_closure_rotation: float = 330.0,
    min_closure_inliers: int = 30
) -> Optional[int]:
    """
    Detect where a 360° loop closes by using rotation data + feature matching.
    
    When capturing more than 360°, the last frames overlap with early frames.
    This function finds the best closure point where the loop should end.
    
    Strategy:
    1. Use rotation data to find where 360° is reached
    2. Validate with feature matching around that area
    3. Find earliest good closure point
    
    Args:
        images: List of images (downscaled for matching).
        match_results: List of sequential match results.
        global_rotations: List of global rotation matrices.
        config: Matching configuration.
        scale: Scale factor for matching.
        check_last_n_frames: Number of last frames to check for closure.
        check_first_n_frames: Number of first frames to check against.
        min_closure_rotation: Minimum rotation coverage (degrees) before checking closure.
        min_closure_inliers: Minimum inliers required for a valid closure match.
        
    Returns:
        Index to trim at (frames after this index should be removed), or None if no closure detected.
    """
    n_images = len(images)
    
    if n_images < max(check_last_n_frames, check_first_n_frames) + 5:
        logger.debug(f"Too few images ({n_images}) for circular closure detection")
        return None
    
    # Estimate total rotation coverage
    if len(global_rotations) < 2:
        return None
    
    # Calculate total rotation angle (from first to last frame)
    R_total = global_rotations[-1] @ global_rotations[0].T
    total_rotation_deg = rotation_angle_degrees(R_total)
    
    logger.debug(f"Total rotation coverage: {total_rotation_deg:.1f}°")
    
    # Only check for closure if we've rotated more than minimum threshold
    if total_rotation_deg < min_closure_rotation:
        logger.debug(f"Rotation coverage ({total_rotation_deg:.1f}°) below threshold ({min_closure_rotation}°), no closure check")
        return None
    
    logger.info(f"Checking for circular closure (total rotation: {total_rotation_deg:.1f}°)...")
    
    # STEP 1: Find where 360° rotation is reached using rotation data
    estimated_360_idx = None
    for i in range(1, len(global_rotations)):
        # Calculate rotation from frame 0 to frame i
        R_i = global_rotations[i] @ global_rotations[0].T
        angle_i = rotation_angle_degrees(R_i)
        
        if angle_i >= 360.0:
            estimated_360_idx = i
            logger.debug(f"Rotation data suggests 360° reached at frame {estimated_360_idx} ({angle_i:.1f}°)")
            break
    
    # If no exact 360° point found, find the closest one
    if estimated_360_idx is None:
        closest_to_360 = None
        closest_diff = float('inf')
        for i in range(1, len(global_rotations)):
            R_i = global_rotations[i] @ global_rotations[0].T
            angle_i = rotation_angle_degrees(R_i)
            diff = abs(angle_i - 360.0)
            if diff < closest_diff:
                closest_diff = diff
                closest_to_360 = i
        if closest_to_360 is not None and closest_diff < 30.0:  # Within 30° of 360
            estimated_360_idx = closest_to_360
            logger.debug(f"Closest to 360° is frame {estimated_360_idx} ({rotation_angle_degrees(global_rotations[estimated_360_idx] @ global_rotations[0].T):.1f}°)")
    
    # STEP 2: Search around estimated 360° point with feature matching
    # Search window: frames from (estimated_360_idx - 10) to (estimated_360_idx + 10), or last N frames if no estimate
    if estimated_360_idx is not None:
        search_start = max(check_first_n_frames + 5, estimated_360_idx - 10)
        search_end = min(n_images, estimated_360_idx + 10)
    else:
        # No rotation estimate, check last frames
        search_start = max(check_first_n_frames + 5, n_images - check_last_n_frames)
        search_end = n_images
    
    logger.debug(f"Searching for closure between frames {search_start} and {search_end}")
    
    best_match_idx = None
    best_match_inliers = 0
    best_match_early_idx = None
    
    # Work backwards from search_end to find earliest good closure
    for last_idx in reversed(range(search_start, search_end)):
        # Try matching this frame against early frames
        for early_idx in range(min(check_first_n_frames, last_idx - 3)):
            # Skip if too close
            if last_idx - early_idx < 3:
                continue
            
            # Match last frame against early frame
            result = match_image_pair(
                images[last_idx],
                images[early_idx],
                last_idx,
                early_idx,
                config,
                scale
            )
            
            if result.success and result.inliers >= min_closure_inliers:
                # Found a match - prefer earlier frames (lower last_idx) to maximize trimming
                if best_match_idx is None or last_idx < best_match_idx:
                    best_match_idx = last_idx
                    best_match_inliers = result.inliers
                    best_match_early_idx = early_idx
                    logger.debug(f"Found closure: frame {last_idx} matches frame {early_idx} ({result.inliers} inliers)")
    
    # STEP 3: If we found a match, verify it's reasonable (within ±20 frames of rotation estimate)
    if best_match_idx is not None:
        if estimated_360_idx is not None:
            offset = abs(best_match_idx - estimated_360_idx)
            if offset > 20:
                logger.warning(f"Closure match at frame {best_match_idx} is {offset} frames away from rotation estimate ({estimated_360_idx}), may be incorrect")
            else:
                logger.info(f"Closure validated: frame {best_match_idx} matches frame {best_match_early_idx} "
                          f"({best_match_inliers} inliers, {offset} frames from rotation estimate)")
        else:
            logger.info(f"Detected circular closure: trim after frame {best_match_idx} "
                       f"(frame {best_match_idx} matches frame {best_match_early_idx} with {best_match_inliers} inliers)")
        return best_match_idx
    
    # STEP 4: If no match found but we have rotation estimate, use it anyway (rotation data is reliable)
    if estimated_360_idx is not None:
        logger.warning(f"No feature matches found, but rotation data indicates 360° at frame {estimated_360_idx}, using rotation-based estimate")
        return estimated_360_idx
    
    logger.debug("No circular closure detected")
    return None


def trim_excess_frames(
    images: List,
    image_infos: List,
    match_results: List[MatchResult],
    global_rotations: List[np.ndarray],
    trim_idx: int
) -> Tuple[List, List, List[MatchResult], List[np.ndarray]]:
    """
    Trim excess frames after circular closure point.
    
    Args:
        images: List of images to trim.
        image_infos: List of image info objects to trim.
        match_results: List of match results to trim.
        global_rotations: List of global rotations to trim.
        trim_idx: Index to trim at (frames after this index are removed).
        
    Returns:
        Tuple of (trimmed_images, trimmed_image_infos, trimmed_match_results, trimmed_global_rotations).
    """
    # Trim images and infos (keep frames 0 to trim_idx inclusive)
    trimmed_images = images[:trim_idx + 1]
    trimmed_image_infos = image_infos[:trim_idx + 1]
    
    # Trim match_results (keep pairs that don't involve frames after trim_idx)
    # A match result (i, i+1) should be kept if i+1 <= trim_idx
    trimmed_match_results = [r for r in match_results if r.dst_idx <= trim_idx]
    
    # Trim global_rotations (keep rotations for frames 0 to trim_idx inclusive)
    trimmed_global_rotations = global_rotations[:trim_idx + 1]
    
    logger.info(f"Trimmed {len(images) - len(trimmed_images)} excess frames "
               f"(kept {len(trimmed_images)} frames, removed frames {trim_idx + 1} to {len(images) - 1})")
    
    return trimmed_images, trimmed_image_infos, trimmed_match_results, trimmed_global_rotations
