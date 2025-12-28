"""
Image blending module for panorama stitching.

Implements:
- Feather blending (fast): distance-based soft blending
- Multiband blending (default): Laplacian pyramid blending
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from .config import BlendingConfig

logger = logging.getLogger(__name__)


def create_distance_weight(mask: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """
    Create distance-based weight from mask using distance transform.
    
    Args:
        mask: Binary mask (uint8, 0 or 255).
        sigma: Gaussian blur sigma for smoothing.
        
    Returns:
        Soft weight map (float32, 0 to 1).
    """
    # Distance transform from edge of mask
    binary_mask = (mask > 127).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    
    # Compute distance from mask boundary
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Normalize to 0-1 range
    max_dist = np.max(dist)
    if max_dist > 0:
        dist = dist / max_dist
    
    # Apply Gaussian blur for smoother transitions
    if sigma > 0:
        ksize = int(sigma * 6) | 1  # Ensure odd kernel size
        dist = cv2.GaussianBlur(dist, (ksize, ksize), sigma)
    
    return dist.astype(np.float32)


def feather_blend(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    config: BlendingConfig
) -> np.ndarray:
    """
    Blend images using feather (weighted average) blending.
    
    Creates soft weights based on distance from mask edges,
    then computes weighted average.
    
    Args:
        images: List of warped images (BGR, uint8).
        masks: List of masks (uint8, 0 or 255).
        config: Blending configuration.
        
    Returns:
        Blended panorama (BGR, uint8).
    """
    if not images:
        raise ValueError("No images to blend")
    
    H, W = images[0].shape[:2]
    
    logger.info(f"Feather blending {len(images)} images...")
    
    # Accumulate weighted sum
    result = np.zeros((H, W, 3), dtype=np.float64)
    total_weight = np.zeros((H, W), dtype=np.float64)
    
    for i, (img, mask) in enumerate(zip(images, masks)):
        # Create soft weight from mask
        weight = create_distance_weight(mask, config.feather_sigma)
        
        # Accumulate
        for c in range(3):
            result[:, :, c] += img[:, :, c].astype(np.float64) * weight
        total_weight += weight
    
    # Normalize by total weight
    # Avoid division by zero
    nonzero = total_weight > 1e-6
    for c in range(3):
        result[:, :, c] = np.where(nonzero, result[:, :, c] / total_weight, 0)
    
    # Convert to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    logger.info("Feather blending complete")
    
    return result


def gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build Gaussian pyramid for an image.
    
    Args:
        image: Input image (float32).
        levels: Number of pyramid levels.
        
    Returns:
        List of images at decreasing resolutions.
    """
    pyramid = [image]
    current = image
    
    for i in range(levels - 1):
        # Downsample
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def laplacian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build Laplacian pyramid for an image.
    
    Args:
        image: Input image (float32).
        levels: Number of pyramid levels.
        
    Returns:
        List of Laplacian images (high-frequency details) plus lowest Gaussian level.
    """
    gaussian = gaussian_pyramid(image, levels)
    laplacian = []
    
    for i in range(levels - 1):
        # Upsample next level
        upsampled = cv2.pyrUp(gaussian[i + 1])
        
        # Match sizes (pyrUp might have different size)
        h, w = gaussian[i].shape[:2]
        upsampled = cv2.resize(upsampled, (w, h))
        
        # Laplacian = difference
        lap = gaussian[i] - upsampled
        laplacian.append(lap)
    
    # Add the lowest level of Gaussian
    laplacian.append(gaussian[-1])
    
    return laplacian


def reconstruct_from_laplacian(laplacian: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct image from Laplacian pyramid.
    
    Args:
        laplacian: Laplacian pyramid (from laplacian_pyramid function).
        
    Returns:
        Reconstructed image.
    """
    # Start from lowest level
    current = laplacian[-1]
    
    for i in range(len(laplacian) - 2, -1, -1):
        # Upsample
        upsampled = cv2.pyrUp(current)
        
        # Match sizes
        h, w = laplacian[i].shape[:2]
        upsampled = cv2.resize(upsampled, (w, h))
        
        # Add detail
        current = upsampled + laplacian[i]
    
    return current


def multiband_blend(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    config: BlendingConfig
) -> np.ndarray:
    """
    Blend images using multiband (Laplacian pyramid) blending.
    
    Memory-optimized version: processes images one at a time instead of storing all pyramids.
    This provides better seam blending by blending different frequency bands separately.
    
    Args:
        images: List of warped images (BGR, uint8).
        masks: List of masks (uint8, 0 or 255).
        config: Blending configuration.
        
    Returns:
        Blended panorama (BGR, uint8).
    """
    if not images:
        raise ValueError("No images to blend")
    
    levels = config.multiband_levels
    H, W = images[0].shape[:2]
    
    logger.info(f"Multiband blending {len(images)} images with {levels} levels (memory-optimized)...")
    
    # Compute weights from masks (distance-based for smooth transitions)
    weights = []
    for mask in masks:
        w = create_distance_weight(mask, sigma=30.0)
        weights.append(w)
    
    # Normalize weights (they should sum to 1 where at least one mask is valid)
    weight_sum = sum(weights)
    weight_sum = np.maximum(weight_sum, 1e-6)  # Avoid division by zero
    weights_normalized = [w / weight_sum for w in weights]
    del weights, weight_sum  # Free memory
    
    # Initialize blended pyramid (will accumulate contributions)
    # First determine sizes for each level
    blended_pyramid = []
    for level in range(levels):
        # Calculate size for this level
        current_h, current_w = H, W
        for l in range(level):
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2
        blended_pyramid.append(np.zeros((current_h, current_w, 3), dtype=np.float32))
    
    # Process each image one at a time to minimize memory usage
    for i, (img, w_norm) in enumerate(zip(images, weights_normalized)):
        if (i + 1) % 10 == 0:
            logger.debug(f"Processing image {i + 1}/{len(images)}...")
        
        # Convert to float32 for this image
        img_f = img.astype(np.float32)
        
        # Build Laplacian pyramid for this image
        lap_pyramid = laplacian_pyramid(img_f, levels)
        
        # Build Gaussian pyramid for weight (3 channels)
        w_3ch = np.stack([w_norm, w_norm, w_norm], axis=-1).astype(np.float32)
        w_pyramid = gaussian_pyramid(w_3ch, levels)
        
        # Accumulate weighted contribution to each level
        for level in range(levels):
            lap_level = lap_pyramid[level]
            w_level = w_pyramid[level]
            
            # Ensure sizes match
            h_lap, w_lap = lap_level.shape[:2]
            h_w, w_w = w_level.shape[:2]
            if h_lap != h_w or w_lap != w_w:
                w_level = cv2.resize(w_level, (w_lap, h_lap))
            
            # Accumulate
            blended_pyramid[level] += lap_level * w_level
        
        # Free memory for this image's pyramids immediately
        del img_f, lap_pyramid, w_3ch, w_pyramid
    
    # Reconstruct from blended pyramid
    result = reconstruct_from_laplacian(blended_pyramid)
    del blended_pyramid  # Free memory
    
    # Clip and convert to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    logger.info("Multiband blending complete")
    
    return result


def blend_panorama(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    config: BlendingConfig
) -> np.ndarray:
    """
    Blend warped images into final panorama.
    
    Dispatches to appropriate blending method based on config.
    
    Args:
        images: List of warped images (BGR, uint8).
        masks: List of masks (uint8, 0 or 255).
        config: Blending configuration.
        
    Returns:
        Blended panorama (BGR, uint8).
    """
    if config.method == "feather":
        return feather_blend(images, masks, config)
    elif config.method == "multiband":
        return multiband_blend(images, masks, config)
    else:
        raise ValueError(f"Unknown blending method: {config.method}")


def fill_gaps(panorama: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """
    Fill gaps in panorama using inpainting.
    
    Args:
        panorama: Blended panorama.
        masks: List of masks (to determine uncovered regions).
        
    Returns:
        Panorama with gaps filled.
    """
    # Create combined mask of covered regions
    H, W = panorama.shape[:2]
    covered = np.zeros((H, W), dtype=np.uint8)
    for mask in masks:
        covered = np.maximum(covered, mask)
    
    # Find uncovered regions
    uncovered = (covered == 0).astype(np.uint8) * 255
    uncovered_pixels = np.sum(uncovered > 0)
    
    if uncovered_pixels == 0:
        logger.info("No gaps to fill in panorama")
        return panorama
    
    coverage = (1 - uncovered_pixels / (H * W)) * 100
    logger.info(f"Filling {uncovered_pixels} gap pixels ({100 - coverage:.1f}% of panorama)")
    
    # Use inpainting to fill gaps
    result = cv2.inpaint(panorama, uncovered, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result


def create_seam_visualization(
    images: List[np.ndarray],
    masks: List[np.ndarray]
) -> np.ndarray:
    """
    Create visualization showing which image contributes to each pixel.
    
    Args:
        images: List of warped images.
        masks: List of masks.
        
    Returns:
        Visualization image with color-coded regions.
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    H, W = images[0].shape[:2]
    
    # Generate distinct colors for each image
    n = len(images)
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color = cv2.cvtColor(
            np.array([[[hue, 255, 200]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR
        )[0, 0]
        colors.append(color)
    
    # Create visualization
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    
    for i, (mask, color) in enumerate(zip(masks, colors)):
        mask_bool = mask > 127
        vis[mask_bool] = color
    
    return vis

