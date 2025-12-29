# Technical Deep Dive

This document provides a comprehensive technical explanation of how the 360° spherical panorama stitching pipeline works.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Coordinate Systems](#coordinate-systems)
3. [Image Ordering](#image-ordering)
4. [Camera Intrinsics Estimation](#camera-intrinsics-estimation)
5. [Feature Detection and Matching](#feature-detection-and-matching)
6. [Homography Estimation](#homography-estimation)
7. [Rotation Extraction](#rotation-extraction)
8. [Rotation Chaining and Smoothing](#rotation-chaining-and-smoothing)
9. [Spherical Warping](#spherical-warping)
10. [Blending](#blending)
11. [Video Frame Extraction](#video-frame-extraction)
12. [Optimizations](#optimizations)

## Pipeline Overview

The complete pipeline consists of the following stages:

```
1. Image Loading & Sorting
   ↓
2. Camera Intrinsics Estimation
   ↓
3. Feature Matching (ORB + RANSAC)
   ↓
4. Rotation Extraction (Homography → Rotation Matrix)
   ↓
5. Rotation Chaining & Temporal Smoothing
   ↓
6. Spherical Warping (Equirectangular Projection)
   ↓
7. Blending
   ↓
8. Output (Panorama + Viewer)
```

## Coordinate Systems

### Equirectangular Coordinates

Equirectangular projection maps spherical coordinates (longitude, latitude) to a 2D rectangle:

- **θ (theta)**: Azimuth angle, ranges from -π to +π (-180° to +180°)
  - Maps to horizontal coordinate U in panorama (0 to width)
- **φ (phi)**: Elevation angle, ranges from -π/2 to +π/2 (-90° to +90°)
  - Maps to vertical coordinate V in panorama (0 to height)

### World Direction Vectors

For a given (θ, φ), the corresponding 3D world direction vector is:

```
r_world = [sin(θ)cos(φ), sin(φ), cos(θ)cos(φ)]
```

This represents a unit vector pointing from the origin in the direction specified by the spherical angles.

### Camera Coordinates

The camera is rotated by rotation matrix R. A world direction vector r_world is transformed to camera coordinates as:

```
r_cam = R^T @ r_world
```

Or equivalently:
```
r_cam = R^{-1} @ r_world  (since R is orthogonal, R^T = R^{-1})
```

### Image Projection

Given a camera direction vector r_cam = [x, y, z], the 2D image coordinates are:

```
u = fx * (x/z) + cx
v = fy * (y/z) + cy
```

Where:
- `fx, fy`: Focal length in pixels
- `cx, cy`: Principal point (optical center) in pixels

## Image Ordering

The pipeline uses a robust multi-stage sorting strategy:

1. **EXIF Timestamp**: Primary method - extracts `EXIF DateTimeOriginal` or `EXIF CreateDate`
2. **File Modification Time**: Fallback if EXIF data is missing
3. **Natural Filename Sort**: Final fallback using natural (human-friendly) string comparison

This ensures images are processed in the correct temporal order, which is crucial for rotation chaining.

## Camera Intrinsics Estimation

### Intrinsic Matrix K

The camera intrinsic matrix is:

```
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
```

### Estimation Methods

1. **EXIF-Based (Primary)**:
   - Extracts `EXIF FocalLengthIn35mmFilm` 
   - Converts to pixel focal length: `fx = (focal_length_35mm / 36mm) * sensor_width_px`
   - Assumes square pixels: `fy = fx`
   - Principal point: `cx = width/2, cy = height/2`

2. **HFOV Fallback**:
   - If EXIF unavailable, uses horizontal field of view (HFOV)
   - `fx = width / (2 * tan(HFOV/2))`
   - Common phone HFOVs: 65-85°

3. **Calibration File (Optional)**:
   - Can provide custom `fx, fy, cx, cy, dist_coeffs` via JSON
   - Useful for calibrated cameras or when EXIF is unreliable

### Distortion Correction

If distortion coefficients are provided (from calibration file), images are undistorted using OpenCV's `initUndistortRectifyMap` before processing.

## Feature Detection and Matching

### ORB Features

**ORB (Oriented FAST and Rotated BRIEF)** is used for feature detection:

- **FAST detector**: Finds corner-like features
- **BRIEF descriptor**: Binary descriptor (256 bits)
- **Rotation invariance**: ORB includes orientation computation
- **Configurable**: Default 3000 features per image

### Matching Strategy

1. **BFMatcher (Brute Force)**: Compares all descriptors
2. **kNN matching** (k=2): Finds 2 nearest neighbors for each feature
3. **Lowe's Ratio Test**: Filters ambiguous matches
   - Ratio = distance(1st) / distance(2nd)
   - Accept if ratio < threshold (default 0.75)
   - Lower threshold = stricter matching

### Symmetric Matching (Optional)

When enabled, matches are validated in both directions:
- Match A→B AND B→A
- Only consistent matches are kept
- More robust but stricter (can reduce match count)

## Homography Estimation

### Pure Rotation Model

For a camera rotating around its optical center (no translation), the relationship between two views is a **homography**:

```
H = K @ R @ K^{-1}
```

Where:
- `K`: Camera intrinsic matrix
- `R`: 3×3 rotation matrix between the two views

### RANSAC Homography Estimation

**RANSAC (Random Sample Consensus)** robustly estimates the homography:

1. **Random sampling**: Select 4 point correspondences
2. **Homography computation**: Solve for H using DLT (Direct Linear Transform)
3. **Inlier counting**: Count points that fit the model (reprojection error < threshold)
4. **Iteration**: Repeat many times, keep best model
5. **Refinement** (optional): Re-estimate using all inliers for better accuracy

### Adaptive Threshold

RANSAC reprojection threshold scales with image size:
```
threshold = base_threshold * (match_width / reference_width)
```

This ensures consistent behavior across different resolutions.

## Rotation Extraction

### From Homography to Rotation

Given homography H and intrinsics K:

```
R_raw = K^{-1} @ H @ K
```

This extracts the rotation matrix from the homography.

### Orthonormalization

Since R_raw may not be a perfect rotation matrix (due to noise/errors), it's orthonormalized using **SVD**:

```
U, S, Vt = SVD(R_raw)
R = U @ Vt
```

If `det(R) < 0` (reflection instead of rotation), flip the sign of the last column of U.

This ensures R is a valid rotation matrix: `R^T @ R = I` and `det(R) = 1`.

## Rotation Chaining and Smoothing

### Global Rotation Chaining

Starting with identity for the first image:

```
R_global[0] = I (identity matrix)

For i = 0 to n-1:
    R_global[i+1] = R_rel[i] @ R_global[i]
    R_global[i+1] = orthonormalize(R_global[i+1])
```

Where `R_rel[i]` is the relative rotation from image i to image i+1.

The orthonormalization at each step prevents error accumulation.

### Neighbor Interpolation for Failed Matches

When a pair fails to match:

1. **Try previous neighbor**: Use rotation from pair (i-1, i)
2. **Try next neighbor**: Use rotation from pair (i+1, i+2)
3. **Average neighbors**: If both exist, average them in rotation space
4. **Identity fallback**: If no neighbors available, use identity

This ensures the pipeline continues even with some failed matches.

### Temporal Smoothing

For video sequences, global rotations are smoothed using a **moving average**:

1. For each rotation R[i], collect rotations in a window: `[R[i-w/2], ..., R[i+w/2]]`
2. Average the rotation matrices: `R_avg = mean(R_window)`
3. Orthonormalize: `R_smooth = orthonormalize(R_avg)`

**Window size**: Configurable (default 3, can increase to 7-15 for straighter lines)

This reduces jitter and makes camera motion appear smoother, resulting in straighter lines in the panorama.

## Spherical Warping

### Inverse Mapping

The warping uses **inverse mapping** (panorama → source image) for efficiency:

For each pixel (U, V) in the panorama:

1. **Convert to spherical angles**:
   ```
   θ = 2π * (U/width) - π
   φ = -π/2 + π * (V/height)
   ```

2. **Convert to world direction**:
   ```
   r_world = [sin(θ)cos(φ), sin(φ), cos(θ)cos(φ)]
   ```

3. **Transform to camera frame**:
   ```
   r_cam = R^T @ r_world
   ```

4. **Project to image**:
   ```
   u = fx * (x_cam/z_cam) + cx
   v = fy * (y_cam/z_cam) + cy
   ```

5. **Check validity**:
   - z_cam > 0 (point in front of camera)
   - (u, v) within image bounds

6. **Sample using cv2.remap**: Uses bilinear interpolation

### Efficiency

- Pre-compute θ and φ grids once (don't recompute per image)
- Use OpenCV's optimized `remap` function for sampling
- Process images sequentially for memory efficiency (optional)

## Blending

### Blending Methods

#### 1. No Blending (Hard Seam Cutting)
- For each pixel, select image with highest mask value
- Sharpest result, but may show seams
- Memory-efficient: processes one image at a time

#### 2. Sharp Blending (Minimal Blur)
- Uses distance transform from mask edge
- Linear weight falloff: `weight = clip(distance / blend_width, 0, 1)`
- No Gaussian blur - only blends in small zone (1-3 pixels)
- Configurable `sharp_blend_width` parameter

#### 3. Feather Blending
- Distance transform + Gaussian blur
- Smooth weight transitions
- Configurable `feather_sigma` parameter

#### 4. Multiband Blending
- Uses Laplacian pyramid decomposition
- Blends different frequency bands separately
- Best for handling exposure differences
- Memory-optimized: processes one image at a time

### Distance-Based Weights

All blending methods use distance from mask edge:

1. Compute distance transform: `dist = cv2.distanceTransform(mask)`
2. Create weights based on distance:
   - **Sharp**: `weight = dist / blend_width` (linear, clamped)
   - **Feather**: Apply Gaussian blur to distance
   - **Multiband**: Use blurred distance for pyramid weights

3. Normalize weights across all images: `w_norm = w / sum(all_weights)`

## Video Frame Extraction

### Extraction Methods

#### 1. Uniform Extraction
- Distributes N frames evenly across video duration
- Simple and predictable

#### 2. Interval Extraction
- Extract every Nth frame
- Good for constant rotation speed

#### 3. FPS-Based Extraction
- Extract at target FPS (e.g., 2 fps from 30 fps video)
- Calculates frame interval: `interval = round(video_fps / target_fps)`

#### 4. Motion-Based Keyframes
- Two-pass algorithm:
  - **Pass 1**: Compute motion scores (frame differences)
  - **Pass 2**: Select frames evenly spaced in cumulative motion space
- Adapts to variable rotation speed
- Configurable: `min_frames`, `max_frames`, `motion_threshold`

## Optimizations

### Memory Efficiency

1. **Sequential Processing**: For "none" blending, process images one at a time
2. **Downscaled Matching**: Match at lower resolution (e.g., 1600px), warp at full resolution
3. **Pyramid Blending**: Process images sequentially in multiband blending
4. **Explicit Memory Management**: Delete large arrays when no longer needed

### Performance

1. **Precomputed Grids**: Spherical coordinate grids computed once, reused
2. **Optimized OpenCV Operations**: Uses `cv2.remap` for efficient warping
3. **Vectorized Operations**: NumPy vectorization for distance transforms, masking
4. **Progress Logging**: Progress updates to track long operations

### Robustness

1. **Multiple Fallbacks**: Image ordering, intrinsics estimation, match interpolation
2. **Error Handling**: Continues with failed matches using interpolation
3. **Validation**: Checks for minimum inliers, rotation validity
4. **Adaptive Thresholds**: RANSAC threshold scales with image size

## Mathematical Foundations

### Rotation Matrices

Rotation matrices in SO(3) (Special Orthogonal Group):
- **Orthogonal**: `R^T @ R = I`
- **Determinant**: `det(R) = 1`
- **Composition**: Rotations compose via matrix multiplication
- **3 degrees of freedom**: Can represent any 3D rotation

### Homography for Pure Rotation

For pure rotation (no translation), points are related by:
```
x' = K @ R @ K^{-1} @ x
```

This is a homography (projective transformation in 2D), not a fundamental matrix.

### Equirectangular Projection

Equirectangular is a simple cylindrical projection:
- Preserves angles (conformal) along meridians and parallels
- Distortion increases toward poles
- Standard format for 360° panoramas (used by YouTube, Facebook, etc.)

## Limitations and Future Work

### Current Limitations

1. **Parallax**: Assumes pure rotation (no translation) - close objects may show ghosting
2. **Loop Closure**: Doesn't explicitly handle 360° loop closure optimization
3. **Vertical Coverage**: Limited by camera FOV (typically captures horizontal band)
4. **Moving Objects**: Will appear as artifacts or ghosts

### Potential Improvements

1. **Bundle Adjustment**: Global optimization of all rotations simultaneously
2. **Loop Closure**: Match first and last images, distribute error
3. **Essential Matrix**: Use for scenes with small translation
4. **Line Straightening**: Post-process to detect and straighten lines
5. **Exposure Compensation**: Automatic exposure/white balance alignment
6. **GPU Acceleration**: CUDA/OpenCL for faster warping and blending

