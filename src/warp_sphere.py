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
import os
from concurrent.futures import ThreadPoolExecutor
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


def world_direction_grid(
    theta: np.ndarray,
    phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    World direction for every panorama pixel, as float32 and computed once.

    Same result as spherical_to_world_directions, but kept in float32 and intended to be
    hoisted out of a per-frame loop: the grid depends only on the output size.

    Args:
        theta: Azimuth grid (H, W).
        phi: Elevation grid (H, W).

    Returns:
        Tuple of (x, y, z) direction components, float32, shape (H, W).
    """
    cos_phi = np.cos(phi)
    x = np.multiply(np.sin(theta), cos_phi, dtype=np.float32)
    y = np.sin(phi).astype(np.float32, copy=False)
    z = np.multiply(np.cos(theta), cos_phi, dtype=np.float32)
    return x, y, z


def compute_warp_maps(
    theta: np.ndarray,
    phi: np.ndarray,
    R: np.ndarray,
    calib: CalibrationData,
    image_width: int,
    image_height: int,
    world_dirs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
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
    # World directions depend only on the panorama grid, never on R, so a caller warping
    # a whole sequence should compute them once and pass them in. Recomputing four
    # transcendentals over every output pixel per frame dominates the warp otherwise.
    if world_dirs is None:
        x_world, y_world, z_world = world_direction_grid(theta, phi)
    else:
        x_world, y_world, z_world = world_dirs

    # Transform to camera frame: r_cam = R @ r_world
    # R_global transforms world directions to camera frame
    # (camera rotated by R, so world point appears at R @ world_dir in camera)
    R_mat = R.astype(np.float32)

    # Written as three explicit rows rather than a reshape-and-matmul: the same
    # arithmetic without materialising the (H, W, 3) stack and its two transposes,
    # which at 4096x2048 is 100 MB of traffic per frame.
    x_cam = R_mat[0, 0] * x_world + R_mat[0, 1] * y_world + R_mat[0, 2] * z_world
    y_cam = R_mat[1, 0] * x_world + R_mat[1, 1] * y_world + R_mat[1, 2] * z_world
    z_cam = R_mat[2, 0] * x_world + R_mat[2, 1] * y_world + R_mat[2, 2] * z_world

    # Valid mask: z > 0 (point is in front of camera)
    valid_mask = z_cam > 1e-6

    # Avoid division by zero. Pixels behind the camera get a placeholder depth and are
    # masked out below, so the values they produce never matter.
    np.copyto(z_cam, 1.0, where=~valid_mask)

    # Project to image coordinates, in place to keep the temporaries down
    map_x = x_cam
    map_x /= z_cam
    map_x *= calib.fx
    map_x += calib.cx

    map_y = y_cam
    map_y /= z_cam
    map_y *= calib.fy
    map_y += calib.cy

    # Valid mask: within image bounds
    margin = 1.0  # Small margin to avoid edge artifacts
    valid_mask &= map_x >= margin
    valid_mask &= map_x < image_width - margin
    valid_mask &= map_y >= margin
    valid_mask &= map_y < image_height - margin

    return map_x, map_y, valid_mask


def warp_image_to_equirectangular(
    image: np.ndarray,
    R: np.ndarray,
    calib: CalibrationData,
    theta: np.ndarray,
    phi: np.ndarray,
    output_shape: Tuple[int, int],
    world_dirs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
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
        theta, phi, R, calib, img_w, img_h, world_dirs
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
    n = len(image_infos)

    # Load and remap are the expensive part and both release the GIL, so frames warp
    # concurrently. The blend callback still runs in index order, since blending modes
    # like 'none' and 'sharp' depend on which frame reaches a pixel first.
    workers = getattr(output_config, "warp_workers", 0)
    if workers <= 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    workers = min(workers, n)

    logger.info(f"Warping {n} images to equirectangular ({W}x{H}) on {workers} worker(s)...")

    # Precompute theta/phi grids once, and the world directions they imply
    theta, phi = create_equirectangular_grid(W, H)
    world_dirs = world_direction_grid(theta, phi)

    def warp_one(idx: int):
        image = load_image(image_infos[idx].path, max_width=None)
        if undistort_func:
            image = undistort_func(image)
        return warp_image_to_equirectangular(image, global_rotations[idx], calib, theta, phi,
                                            (H, W), world_dirs)

    def report(idx: int, mask: np.ndarray) -> None:
        if n > 50:
            if (idx + 1) % 5 == 0 or idx == 0:
                logger.info(f"Processing image {idx+1}/{n}...")
        else:
            logger.info(f"Processing image {idx+1}/{n}...")
            coverage = np.sum(mask > 0) / (W * H) * 100
            logger.debug(f"  Image {idx+1}: {coverage:.1f}% panorama coverage")

    masks = []

    if workers == 1:
        for i in range(n):
            warped, mask = warp_one(i)
            report(i, mask)
            blend_callback(warped, mask, i)
            masks.append(mask)
            del warped
    else:
        # Keep only a small number of warped frames in flight: each one is
        # W*H*3 bytes, which is 24 MB at 4096x2048.
        in_flight = workers + 2
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            submitted = 0
            for i in range(n):
                while submitted < n and len(pending) < in_flight:
                    pending[submitted] = pool.submit(warp_one, submitted)
                    submitted += 1
                warped, mask = pending.pop(i).result()
                report(i, mask)
                blend_callback(warped, mask, i)
                masks.append(mask)
                del warped

    logger.info(f"Warped all {n} images")

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
