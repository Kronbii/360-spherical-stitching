"""
Circular matching module for detecting wrap-around overlap in 360° panoramas.

Detects when a video captures more than 360°, causing the end to overlap with the beginning.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .config import MatchingConfig
from .features import MatchResult, match_image_pair
from .rotation import rotation_angle_degrees, sweep_span_degrees, unwrapped_yaw_degrees

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
    
    # Total yaw travelled. This must use the unwrapped sweep span, not the geodesic
    # angle between the first and last rotation: rotation_angle_degrees() is
    # arccos((trace-1)/2), which saturates at 180°, so a 332° sweep measures as 28°
    # and any threshold above 180° could never be reached.
    total_rotation_deg = sweep_span_degrees(global_rotations)

    logger.debug(f"Total rotation coverage: {total_rotation_deg:.1f}°")

    # Only check for closure if we've rotated more than minimum threshold
    if total_rotation_deg < min_closure_rotation:
        logger.debug(f"Rotation coverage ({total_rotation_deg:.1f}°) below threshold ({min_closure_rotation}°), no closure check")
        return None
    
    logger.info(f"Checking for circular closure (total rotation: {total_rotation_deg:.1f}°)...")
    
    # STEP 1: Find where 360° of yaw is reached, again from the unwrapped sweep
    yaw = unwrapped_yaw_degrees(global_rotations)
    travelled = np.abs(yaw - yaw[0])

    estimated_360_idx = None
    past = np.where(travelled >= 360.0)[0]
    if len(past):
        estimated_360_idx = int(past[0])
        logger.debug(f"Yaw reaches 360° at frame {estimated_360_idx} ({travelled[estimated_360_idx]:.1f}°)")
    else:
        # Not a full turn: fall back to the closest approach, if it is close at all
        closest = int(np.argmin(np.abs(travelled - 360.0)))
        if abs(travelled[closest] - 360.0) < 30.0:
            estimated_360_idx = closest
            logger.debug(f"Closest to 360° is frame {closest} ({travelled[closest]:.1f}°)")
    
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
    
    # STEP 4: Rotation data alone is not evidence of closure. It is exactly what drifts,
    # and trimming on it would delete real frames because of an estimation error. Report
    # no closure and let enforce_open_sweep() absorb the drift instead.
    if estimated_360_idx is not None:
        logger.warning(
            f"Rotation data suggests 360° near frame {estimated_360_idx}, but no frame there matches "
            f"any early frame. Treating the sweep as open rather than trimming on drift alone."
        )

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

def enforce_open_sweep(
    global_rotations: List[np.ndarray],
    hfov_deg: float,
    margin_deg: float = 2.0
) -> Tuple[List[np.ndarray], Optional[dict]]:
    """
    Stop an over-reaching rotation chain from wrapping the last frames onto the first.

    Every pair's rotation is estimated independently, so the chain accumulates error.
    When it over-estimates, the sweep can be reported as long enough to close the circle
    even though the camera never returned to its starting view. The warp stage then
    places the final frames on the same arc as the first ones, and since their content is
    unrelated, one paints over the other — a hard seam of two different scenes.

    Call this only after a feature-level closure check has failed, i.e. the tail and head
    provably do not see the same thing. The correction squeezes the whole sweep about the
    vertical axis so the two ends stop just short of touching, spreading the accumulated
    error evenly over the sequence instead of dumping it at the join.

    Args:
        global_rotations: Chained global rotations.
        hfov_deg: Horizontal field of view of one frame, in degrees.
        margin_deg: Extra gap to leave between the two ends.

    Returns:
        Tuple of (rotations, report). Report is None when no correction was needed,
        otherwise a dict describing what was applied.
    """
    if len(global_rotations) < 3:
        return global_rotations, None

    span = sweep_span_degrees(global_rotations)
    limit = 360.0 - hfov_deg - margin_deg

    if span <= limit:
        logger.debug(f"Sweep span {span:.1f}° fits within {limit:.1f}°, no wrap correction needed")
        return global_rotations, None

    scale = limit / span
    yaw = unwrapped_yaw_degrees(global_rotations)
    corrected = []
    for R, y in zip(global_rotations, yaw - yaw[0]):
        a = np.radians((scale - 1.0) * y)          # negative: pull each frame back
        Ry = np.array([[np.cos(a), 0.0, np.sin(a)],
                       [0.0,       1.0, 0.0      ],
                       [-np.sin(a), 0.0, np.cos(a)]])
        corrected.append(Ry @ R)                   # spin about world vertical only

    report = {
        "span_before": span,
        "span_after": sweep_span_degrees(corrected),
        "limit": limit,
        "scale": scale,
        "hfov_deg": hfov_deg,
    }
    logger.warning(
        f"Sweep measured {span:.1f}° of yaw, which with a {hfov_deg:.1f}° field of view would "
        f"wrap the last frames onto the first."
    )
    logger.warning(
        f"No feature match confirms the loop closes, so the chain over-reached. Squeezing the "
        f"sweep to {report['span_after']:.1f}° (x{scale:.4f}) to keep the ends apart."
    )
    return corrected, report


def head_tail_overlap(
    images: List,
    global_rotations: List[np.ndarray],
    hfov_deg: float
) -> Optional[Tuple[int, int]]:
    """
    Frames whose placement collides with the first frame's arc, if any.

    Args:
        images: Frames, only used for its length.
        global_rotations: Chained global rotations.
        hfov_deg: Horizontal field of view of one frame, in degrees.

    Returns:
        (first colliding index, last index) or None when nothing collides.
    """
    yaw = unwrapped_yaw_degrees(global_rotations)
    travelled = np.abs(yaw - yaw[0])
    colliding = np.where(travelled > 360.0 - hfov_deg)[0]
    if not len(colliding):
        return None
    return int(colliding[0]), len(global_rotations) - 1
