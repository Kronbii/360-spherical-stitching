"""
Shared pytest fixtures for panorama stitching tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample BGR image with random features for testing."""
    np.random.seed(42)
    H, W = 600, 800
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Add random circles for features
    for _ in range(100):
        cx, cy = np.random.randint(50, W-50), np.random.randint(50, H-50)
        r = np.random.randint(5, 25)
        color = tuple(int(x) for x in np.random.randint(50, 255, 3))
        cv2.circle(img, (cx, cy), r, color, -1)
    
    # Add some rectangles
    for _ in range(20):
        x1, y1 = np.random.randint(0, W-100), np.random.randint(0, H-100)
        x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
        color = tuple(int(x) for x in np.random.randint(50, 255, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    return img


@pytest.fixture
def sample_image_pair(sample_image) -> Tuple[np.ndarray, np.ndarray]:
    """Create two overlapping images for matching tests."""
    H, W = sample_image.shape[:2]
    
    # Create a wider base image
    wide = np.zeros((H, W * 2, 3), dtype=np.uint8)
    wide[:, :W] = sample_image
    
    # Add some unique features to the right side
    np.random.seed(43)
    for _ in range(50):
        cx = np.random.randint(W, W * 2 - 50)
        cy = np.random.randint(50, H - 50)
        r = np.random.randint(5, 25)
        color = tuple(int(x) for x in np.random.randint(50, 255, 3))
        cv2.circle(wide, (cx, cy), r, color, -1)
    
    # Extract two overlapping views (50% overlap)
    overlap = int(W * 0.5)
    img1 = wide[:, :W].copy()
    img2 = wide[:, W - overlap:2*W - overlap].copy()
    
    return img1, img2


@pytest.fixture
def sample_image_sequence(sample_image) -> List[np.ndarray]:
    """Create a sequence of overlapping images for pipeline tests."""
    H, W = sample_image.shape[:2]
    
    # Create a wide panoramic scene
    wide_W = W * 4
    wide = np.zeros((H, wide_W, 3), dtype=np.uint8)
    
    # Fill with random features
    np.random.seed(42)
    for _ in range(500):
        cx = np.random.randint(50, wide_W - 50)
        cy = np.random.randint(50, H - 50)
        r = np.random.randint(5, 30)
        color = tuple(int(x) for x in np.random.randint(50, 255, 3))
        cv2.circle(wide, (cx, cy), r, color, -1)
    
    # Add section markers
    for i in range(4):
        cv2.putText(wide, f"S{i}", (i * W + 100, H // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
    
    # Extract overlapping views
    images = []
    overlap = 0.5
    step = int(W * (1 - overlap))
    
    for i in range(5):
        start = i * step
        end = start + W
        if end <= wide_W:
            images.append(wide[:, start:end].copy())
    
    return images


@pytest.fixture
def test_images_dir(temp_dir, sample_image_sequence) -> Path:
    """Create a directory with test images saved as files."""
    images_dir = temp_dir / "images"
    images_dir.mkdir()
    
    for i, img in enumerate(sample_image_sequence):
        cv2.imwrite(str(images_dir / f"image_{i:03d}.jpg"), img)
    
    return images_dir


@pytest.fixture
def sample_homography() -> np.ndarray:
    """Create a sample homography matrix (small rotation + translation)."""
    # Approximate homography for small rotation
    angle = np.radians(10)
    c, s = np.cos(angle), np.sin(angle)
    H = np.array([
        [c, -s, 50],
        [s, c, 20],
        [0, 0, 1]
    ], dtype=np.float64)
    return H


@pytest.fixture
def sample_rotation_matrix() -> np.ndarray:
    """Create a sample rotation matrix (10 degrees around Y axis)."""
    angle = np.radians(10)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float64)
    return R


@pytest.fixture
def sample_calibration():
    """Create sample camera calibration data."""
    from src.config import CalibrationData
    return CalibrationData(
        fx=1000.0,
        fy=1000.0,
        cx=400.0,
        cy=300.0,
        source="test"
    )


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Create a sample binary mask for blending tests."""
    H, W = 512, 1024
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Create a rectangular valid region
    mask[50:H-50, 100:W//2] = 255
    
    return mask


@pytest.fixture
def sample_warped_images() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create sample warped images and masks for blending tests."""
    H, W = 512, 1024
    
    images = []
    masks = []
    
    # Create 3 overlapping warped images
    for i in range(3):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # Each image covers a different horizontal region
        start_x = i * W // 4
        end_x = start_x + W // 2
        
        # Fill with a color gradient
        for x in range(start_x, min(end_x, W)):
            color = ((i + 1) * 80, 100 + x % 50, 150 - i * 30)
            img[:, x] = color
        
        mask[:, start_x:min(end_x, W)] = 255
        
        images.append(img)
        masks.append(mask)
    
    return images, masks

