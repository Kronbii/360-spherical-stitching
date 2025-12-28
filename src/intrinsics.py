"""
Camera intrinsics estimation module.

Supports:
- Loading from calibration JSON file
- Estimation from EXIF FocalLengthIn35mmFilm
- Estimation from EXIF FocalLength (if sensor size available)
- Fallback to horizontal FOV
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import CalibrationData, IntrinsicsConfig
from .io_utils import ImageInfo

logger = logging.getLogger(__name__)


def estimate_from_calib_json(calib_path: Path, image_width: int, image_height: int) -> Optional[CalibrationData]:
    """
    Load calibration from a JSON file.
    
    Expected JSON format:
    {
        "fx": float,
        "fy": float,
        "cx": float,
        "cy": float,
        "dist_coeffs": [k1, k2, p1, p2, k3]  // optional
    }
    
    If fx/fy are normalized (0-1 range), they will be scaled by image dimensions.
    
    Args:
        calib_path: Path to calibration JSON file.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        
    Returns:
        CalibrationData or None if loading fails.
    """
    if not calib_path or not calib_path.exists():
        return None
    
    try:
        with open(calib_path, 'r') as f:
            data = json.load(f)
        
        fx = data['fx']
        fy = data['fy']
        cx = data.get('cx', image_width / 2)
        cy = data.get('cy', image_height / 2)
        
        # Check if values are normalized (0-1 range suggests normalized)
        if fx < 10:  # Likely normalized
            fx *= image_width
            fy *= image_height
            cx *= image_width
            cy *= image_height
            logger.info("Calibration values appear normalized, scaling by image dimensions")
        
        dist_coeffs = data.get('dist_coeffs')
        
        calib = CalibrationData(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist_coeffs=dist_coeffs
        )
        
        logger.info(f"Loaded calibration from {calib_path}")
        logger.info(f"  fx={calib.fx:.2f}, fy={calib.fy:.2f}, cx={calib.cx:.2f}, cy={calib.cy:.2f}")
        if dist_coeffs:
            logger.info(f"  distortion: {dist_coeffs}")
        
        return calib
        
    except Exception as e:
        logger.warning(f"Could not load calibration from {calib_path}: {e}")
        return None


def estimate_from_35mm_equivalent(exif_data: dict, image_width: int, image_height: int) -> Optional[CalibrationData]:
    """
    Estimate intrinsics from EXIF FocalLengthIn35mmFilm.
    
    The 35mm equivalent focal length is relative to a 36mm sensor width.
    fx_pixels = (f35 / 36.0) * image_width
    
    Args:
        exif_data: Dictionary with EXIF data.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        
    Returns:
        CalibrationData or None if tag not available.
    """
    f35 = exif_data.get('FocalLengthIn35mmFilm')
    
    if f35 is None or f35 <= 0:
        return None
    
    try:
        f35 = float(f35)
    except (ValueError, TypeError):
        return None
    
    # 35mm film has a width of 36mm
    # The 35mm equivalent focal length tells us the FOV relative to 35mm film
    fx_pixels = (f35 / 36.0) * image_width
    fy_pixels = fx_pixels  # Assume square pixels
    
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Calculate corresponding HFOV for logging
    hfov = 2 * math.atan(image_width / (2 * fx_pixels)) * 180 / math.pi
    
    calib = CalibrationData(
        fx=fx_pixels,
        fy=fy_pixels,
        cx=cx,
        cy=cy
    )
    
    logger.info(f"Estimated intrinsics from 35mm equivalent focal length")
    logger.info(f"  FocalLengthIn35mmFilm: {f35}mm")
    logger.info(f"  fx={calib.fx:.2f} pixels, HFOV≈{hfov:.1f}°")
    
    return calib


def estimate_from_focal_length(exif_data: dict, image_width: int, image_height: int) -> Optional[CalibrationData]:
    """
    Attempt to estimate intrinsics from EXIF FocalLength.
    
    This requires knowing the sensor size, which is typically not in EXIF.
    We do NOT use a phone database; just log that it's unavailable.
    
    Args:
        exif_data: Dictionary with EXIF data.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        
    Returns:
        CalibrationData or None (almost always None as sensor size unavailable).
    """
    focal_length = exif_data.get('FocalLength')
    
    if focal_length is None:
        return None
    
    # Check if we have sensor size info (very rare in phone EXIF)
    # Standard EXIF tags for sensor size are not commonly set by phones
    sensor_width = exif_data.get('SensorWidth')  # This is almost never present
    
    if sensor_width is None:
        logger.debug(f"FocalLength found ({focal_length}mm) but sensor size unavailable in EXIF")
        logger.debug("Cannot estimate intrinsics without sensor dimensions (not using phone database)")
        return None
    
    try:
        sensor_width = float(sensor_width)
        focal_length = float(focal_length)
    except (ValueError, TypeError):
        return None
    
    fx_pixels = (focal_length / sensor_width) * image_width
    fy_pixels = fx_pixels
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    calib = CalibrationData(
        fx=fx_pixels,
        fy=fy_pixels,
        cx=cx,
        cy=cy
    )
    
    return calib


def estimate_from_hfov(hfov_deg: float, image_width: int, image_height: int) -> CalibrationData:
    """
    Estimate intrinsics from horizontal field of view.
    
    fx = W / (2 * tan(hfov/2))
    
    Args:
        hfov_deg: Horizontal field of view in degrees.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        
    Returns:
        CalibrationData.
    """
    hfov_rad = math.radians(hfov_deg)
    fx = image_width / (2 * math.tan(hfov_rad / 2))
    fy = fx  # Assume square pixels
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    calib = CalibrationData(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    
    logger.info(f"Using HFOV fallback for intrinsics")
    logger.info(f"  HFOV: {hfov_deg}°")
    logger.info(f"  fx={calib.fx:.2f} pixels")
    
    return calib


def estimate_intrinsics(
    image_infos: List[ImageInfo],
    image_width: int,
    image_height: int,
    config: IntrinsicsConfig
) -> CalibrationData:
    """
    Estimate camera intrinsics using the best available method.
    
    Priority:
    1. Calibration JSON file
    2. EXIF FocalLengthIn35mmFilm
    3. EXIF FocalLength + sensor size (rarely available)
    4. User-specified HFOV fallback
    
    Args:
        image_infos: List of ImageInfo with EXIF data.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        config: IntrinsicsConfig with settings.
        
    Returns:
        CalibrationData with estimated intrinsics.
    """
    # Method 1: Calibration JSON
    if config.calib_json:
        calib = estimate_from_calib_json(config.calib_json, image_width, image_height)
        if calib:
            return calib
        logger.warning(f"Could not load calibration from {config.calib_json}, trying other methods")
    
    # Collect EXIF data from all images to find the best estimate
    exif_samples = [info.exif_data for info in image_infos if info.exif_data]
    
    # Try to find FocalLengthIn35mmFilm from any image
    for exif_data in exif_samples:
        if exif_data:
            # Method 2: 35mm equivalent
            calib = estimate_from_35mm_equivalent(exif_data, image_width, image_height)
            if calib:
                return calib
    
    # Method 3: FocalLength with sensor size (rarely works)
    for exif_data in exif_samples:
        if exif_data:
            calib = estimate_from_focal_length(exif_data, image_width, image_height)
            if calib:
                return calib
    
    # Method 4: HFOV fallback
    logger.warning("No EXIF focal length information available")
    return estimate_from_hfov(config.hfov_deg, image_width, image_height)


def save_intrinsics_report(
    calib: CalibrationData,
    image_infos: List[ImageInfo],
    image_width: int,
    image_height: int,
    output_path: Path
) -> None:
    """
    Save detailed intrinsics report to JSON.
    
    Args:
        calib: Estimated calibration data.
        image_infos: List of ImageInfo with EXIF data.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        output_path: Path to save the report.
    """
    # Collect EXIF focal length info from all images
    focal_info = []
    for info in image_infos:
        if info.exif_data:
            entry = {"filename": info.path.name}
            if 'FocalLengthIn35mmFilm' in info.exif_data:
                entry['FocalLengthIn35mmFilm'] = info.exif_data['FocalLengthIn35mmFilm']
            if 'FocalLength' in info.exif_data:
                entry['FocalLength'] = info.exif_data['FocalLength']
            if len(entry) > 1:
                focal_info.append(entry)
    
    # Calculate HFOV from estimated intrinsics
    hfov = 2 * math.atan(image_width / (2 * calib.fx)) * 180 / math.pi
    vfov = 2 * math.atan(image_height / (2 * calib.fy)) * 180 / math.pi
    
    report = {
        "intrinsics": {
            "fx": calib.fx,
            "fy": calib.fy,
            "cx": calib.cx,
            "cy": calib.cy,
            "dist_coeffs": calib.dist_coeffs,
        },
        "camera_matrix_K": calib.K.tolist(),
        "estimation_source": "estimated",
        "image_dimensions": {
            "width": image_width,
            "height": image_height,
        },
        "field_of_view": {
            "horizontal_deg": round(hfov, 2),
            "vertical_deg": round(vfov, 2),
        },
        "exif_focal_length_samples": focal_info[:5],  # First 5 samples
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved intrinsics report to {output_path}")


def undistort_images(
    images: List[np.ndarray],
    calib: CalibrationData
) -> Tuple[List[np.ndarray], CalibrationData]:
    """
    Undistort images if distortion coefficients are available.
    
    Args:
        images: List of images as numpy arrays.
        calib: CalibrationData with distortion coefficients.
        
    Returns:
        Tuple of (undistorted images, new calibration data).
    """
    if calib.dist_coeffs is None or all(d == 0 for d in calib.dist_coeffs):
        logger.info("No distortion coefficients provided, skipping undistortion")
        return images, calib
    
    logger.info("Undistorting images...")
    
    dist_coeffs = np.array(calib.dist_coeffs, dtype=np.float64)
    K = calib.K
    
    # Get optimal new camera matrix (alpha=1 keeps all pixels)
    h, w = images[0].shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0)
    
    # Compute undistortion maps once
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
    
    # Undistort all images
    undistorted = []
    for img in images:
        undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        undistorted.append(undist)
    
    # Create new calibration with updated intrinsics
    new_calib = CalibrationData(
        fx=new_K[0, 0],
        fy=new_K[1, 1],
        cx=new_K[0, 2],
        cy=new_K[1, 2],
        dist_coeffs=None  # No distortion after undistortion
    )
    
    logger.info(f"Undistorted {len(images)} images")
    logger.info(f"New intrinsics: fx={new_calib.fx:.2f}, fy={new_calib.fy:.2f}")
    
    return undistorted, new_calib


def undistort_image_single(
    image: np.ndarray,
    calib: CalibrationData,
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Undistort a single image if distortion coefficients are available.
    
    This is a memory-efficient version for sequential processing.
    
    Args:
        image: Single image as numpy array.
        calib: CalibrationData with distortion coefficients.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        
    Returns:
        Undistorted image.
    """
    if calib.dist_coeffs is None or all(d == 0 for d in calib.dist_coeffs):
        return image
    
    dist_coeffs = np.array(calib.dist_coeffs, dtype=np.float64)
    K = calib.K
    
    # Get optimal new camera matrix
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0)
    
    # Compute undistortion maps
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
    
    # Undistort image
    undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    
    return undistorted

