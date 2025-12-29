"""
Spherical inverse mapping for equirectangular projection.

Converts images to equirectangular panorama coordinates using inverse mapping
and cv2.remap for efficient warping.

Coordinate system:
- theta: azimuth angle, -pi to +pi (wraps at ±180°)
- phi: elevation angle, -pi/2 to +pi/2 (±90°)
- World coordinate: [sin(theta)*cos(phi), sin(phi), cos(theta)*cos(phi)]
- Equirectangular: theta maps to U (horizontal), phi maps to V (vertical)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import cv2
import numpy as np

from .config import CalibrationData, OutputConfig

logger = logging.getLogger(__name__)


def create_equirectangular_grid(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create theta and phi grids for equirectangular projection.
    
    Args:
        width: Panorama width (W).
        height: Panorama height (H = W/2 for equirectangular).
        
    Returns:
        Tuple of (theta grid, phi grid) as float32 arrays of shape (H, W).
    """
    # Create normalized coordinate grids
    u = np.linspace(0, 1, width, dtype=np.float32)
    v = np.linspace(0, 1, height, dtype=np.float32)
    
    # Meshgrid
    U, V = np.meshgrid(u, v)
    
    # Convert to spherical angles
    # theta: 0 -> -pi, 1 -> +pi (azimuth)
    # phi: 0 -> -pi/2, 1 -> +pi/2 (elevation, top to bottom)
    # Top of equirectangular (V=0) = looking down (-pi/2)
    # Bottom of equirectangular (V=1) = looking up (+pi/2)
    theta = 2 * np.pi * U - np.pi  # [-pi, +pi]
    phi = -np.pi / 2 + np.pi * V   # [-pi/2, +pi/2]
    
    return theta, phi


def spherical_to_world_directions(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to world direction vectors.
    
    r_world = [sin(theta)*cos(phi), sin(phi), cos(theta)*cos(phi)]
    
    Args:
        theta: Azimuth angle grid (H, W) in radians.
        phi: Elevation angle grid (H, W) in radians.
        
    Returns:
        Tuple of (x, y, z) world direction components, each of shape (H, W).
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(phi)
    z = np.cos(theta) * np.cos(phi)
    
    return x, y, z


def compute_warp_maps(
    theta: np.ndarray,
    phi: np.ndarray,
    R: np.ndarray,
    calib: CalibrationData,
    image_width: int,
    image_height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute warp maps for a single image using inverse mapping.
    
    For each pixel (U, V) in panorama:
    1. Convert to world direction r_world
    2. Transform to camera frame: r_cam = R^T @ r_world
    3. Project to image: u = fx*(x/z)+cx, v = fy*(y/z)+cy
    4. Valid where z > 0 and (u, v) within image bounds
    
    Args:
        theta: Azimuth grid (H, W).
        phi: Elevation grid (H, W).
        R: Global rotation matrix for this image.
        calib: Camera calibration data.
        image_width: Source image width.
        image_height: Source image height.
        
    Returns:
        Tuple of (map_x, map_y, valid_mask) for cv2.remap.
    """
    # Get world directions
    x_world, y_world, z_world = spherical_to_world_directions(theta, phi)
    
    # Stack into (H, W, 3)
    r_world = np.stack([x_world, y_world, z_world], axis=-1)
    
    # Transform to camera frame: r_cam = R @ r_world
    # R_global transforms world directions to camera frame
    # (camera rotated by R, so world point appears at R @ world_dir in camera)
    R_mat = R.astype(np.float32)
    
    # Efficient batch matrix-vector multiplication
    # Reshape r_world to (H*W, 3), multiply, reshape back
    H, W = theta.shape
    r_flat = r_world.reshape(-1, 3)
    r_cam_flat = (R_mat @ r_flat.T).T  # (H*W, 3)
    r_cam = r_cam_flat.reshape(H, W, 3)
    
    x_cam = r_cam[:, :, 0]
    y_cam = r_cam[:, :, 1]
    z_cam = r_cam[:, :, 2]
    
    # Valid mask: z > 0 (point is in front of camera)
    valid_z = z_cam > 1e-6
    
    # Project to image coordinates
    # Avoid division by zero
    z_safe = np.where(valid_z, z_cam, 1.0)
    
    map_x = calib.fx * (x_cam / z_safe) + calib.cx
    map_y = calib.fy * (y_cam / z_safe) + calib.cy
    
    # Valid mask: within image bounds
    margin = 1.0  # Small margin to avoid edge artifacts
    valid_bounds = (
        (map_x >= margin) & (map_x < image_width - margin) &
        (map_y >= margin) & (map_y < image_height - margin)
    )
    
    valid_mask = valid_z & valid_bounds
    
    return map_x, map_y, valid_mask


def warp_image_to_equirectangular(
    image: np.ndarray,
    R: np.ndarray,
    calib: CalibrationData,
    theta: np.ndarray,
    phi: np.ndarray,
    output_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp a single image to equirectangular panorama coordinates.
    
    Args:
        image: Source image (BGR).
        R: Global rotation matrix.
        calib: Camera calibration.
        theta: Precomputed azimuth grid.
        phi: Precomputed elevation grid.
        output_shape: (height, width) of output panorama.
        
    Returns:
        Tuple of (warped image, valid mask).
    """
    H, W = output_shape
    img_h, img_w = image.shape[:2]
    
    # Compute warp maps
    map_x, map_y, valid_mask = compute_warp_maps(
        theta, phi, R, calib, img_w, img_h
    )
    
    # Warp using cv2.remap
    warped = cv2.remap(
        image,
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Convert mask to uint8
    mask = (valid_mask.astype(np.uint8) * 255)
    
    return warped, mask


def warp_all_images(
    images: List[np.ndarray],
    global_rotations: List[np.ndarray],
    calib: CalibrationData,
    output_config: OutputConfig
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Warp all images to equirectangular coordinates.
    
    Args:
        images: List of source images (BGR).
        global_rotations: List of global rotation matrices.
        calib: Camera calibration.
        output_config: Output configuration with panorama dimensions.
        
    Returns:
        Tuple of (list of warped images, list of masks).
    """
    W = output_config.pano_width
    H = output_config.pano_height
    
    logger.info(f"Warping {len(images)} images to equirectangular ({W}x{H})...")
    
    # Precompute theta/phi grids once
    theta, phi = create_equirectangular_grid(W, H)
    
    warped_images = []
    masks = []
    
    for i, (image, R) in enumerate(zip(images, global_rotations)):
        # Progress update every image (or every 5 for large batches)
        if len(images) > 50:
            if (i + 1) % 5 == 0 or i == 0:
                logger.info(f"Warping image {i+1}/{len(images)}...")
        else:
            logger.info(f"Warping image {i+1}/{len(images)}...")
        
        warped, mask = warp_image_to_equirectangular(
            image, R, calib, theta, phi, (H, W)
        )
        
        warped_images.append(warped)
        masks.append(mask)
        
        # Log coverage
        coverage = np.sum(mask > 0) / (W * H) * 100
        if len(images) <= 50:  # Only log coverage for smaller batches
            logger.debug(f"  Image {i+1}: {coverage:.1f}% panorama coverage")
    
    logger.info(f"Warped all {len(images)} images")
    
    return warped_images, masks


def warp_and_blend_sequential(
    image_infos: List,
    global_rotations: List[np.ndarray],
    calib: CalibrationData,
    output_config: OutputConfig,
    blend_callback: Callable[[np.ndarray, np.ndarray, int], None],
    image_width: int,
    image_height: int,
    undistort_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> List[np.ndarray]:
    """
    Warp images sequentially and call blend_callback for each warped image.
    
    This is memory-efficient as it only keeps one image in memory at a time.
    
    Args:
        image_infos: List of ImageInfo objects.
        global_rotations: List of global rotation matrices.
        calib: Camera calibration.
        output_config: Output configuration.
        blend_callback: Function(warped_image, mask, index) called for each warped image.
        image_width: Source image width.
        image_height: Source image height.
        undistort_func: Optional function to undistort images (if provided).
        
    Returns:
        List of masks (for coverage statistics).
    """
    from .io_utils import load_image
    
    W = output_config.pano_width
    H = output_config.pano_height
    
    logger.info(f"Sequentially warping {len(image_infos)} images to equirectangular ({W}x{H})...")
    
    # Precompute theta/phi grids once
    theta, phi = create_equirectangular_grid(W, H)
    
    masks = []
    
    for i, (image_info, R) in enumerate(zip(image_infos, global_rotations)):
        # Progress update: every image for small batches, every 5 for large batches
        if len(image_infos) > 50:
            if (i + 1) % 5 == 0 or i == 0:
                logger.info(f"Processing image {i+1}/{len(image_infos)}...")
        else:
            logger.info(f"Processing image {i+1}/{len(image_infos)}...")
        
        # Load image
        image = load_image(image_info.path, max_width=None)
        
        # Undistort if needed
        if undistort_func:
            image = undistort_func(image)
        
        # Warp image
        warped, mask = warp_image_to_equirectangular(
            image, R, calib, theta, phi, (H, W)
        )
        
        # Call blend callback
        blend_callback(warped, mask, i)
        
        # Store mask for coverage statistics
        masks.append(mask)
        
        # Log coverage for smaller batches
        if len(image_infos) <= 50:
            coverage = np.sum(mask > 0) / (W * H) * 100
            logger.debug(f"  Image {i+1}: {coverage:.1f}% panorama coverage")
        
        # Free memory explicitly
        del image, warped
    
    logger.info(f"Sequentially warped all {len(image_infos)} images")
    
    return masks


def estimate_panorama_coverage(masks: List[np.ndarray]) -> dict:
    """
    Estimate panorama coverage from masks.
    
    Args:
        masks: List of valid masks (uint8, 0 or 255).
        
    Returns:
        Dictionary with coverage statistics.
    """
    if not masks:
        return {"coverage_percent": 0.0, "total_pixels": 0, "covered_pixels": 0}
    
    # Combine all masks
    combined = np.zeros_like(masks[0])
    for mask in masks:
        combined = np.maximum(combined, mask)
    
    H, W = combined.shape
    total_pixels = H * W
    covered_pixels = np.sum(combined > 0)
    coverage_percent = (covered_pixels / total_pixels) * 100
    
    return {
        "coverage_percent": coverage_percent,
        "total_pixels": total_pixels,
        "covered_pixels": covered_pixels,
    }


def save_debug_warps(
    warped_images: List[np.ndarray],
    masks: List[np.ndarray],
    output_dir: Path,
    num_to_save: int = 5
) -> None:
    """
    Save sample warped images for debugging.
    
    Args:
        warped_images: List of warped images.
        masks: List of masks.
        output_dir: Output directory.
        num_to_save: Number of images to save (evenly spaced).
    """
    if not warped_images:
        return
    
    debug_dir = output_dir / "debug" / "warped"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    n = len(warped_images)
    step = max(1, n // num_to_save)
    indices = list(range(0, n, step))[:num_to_save]
    
    for idx in indices:
        # Combine warped image with mask overlay
        warped = warped_images[idx].copy()
        mask = masks[idx]
        
        # Overlay mask as semi-transparent red
        mask_overlay = np.zeros_like(warped)
        mask_overlay[:, :, 2] = mask  # Red channel
        warped_overlay = cv2.addWeighted(warped, 0.7, mask_overlay, 0.3, 0)
        
        output_path = debug_dir / f"warped_{idx:04d}.jpg"
        cv2.imwrite(str(output_path), warped_overlay)
    
    logger.info(f"Saved {len(indices)} warped images to {debug_dir}")
