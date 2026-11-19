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
8. [Rotation Chaining and Smoothing](#rotation-chaining-and-smoothing) (the `chain` method)
9. [Global Orientation](#global-orientation) (the default `bundle` method)
10. [Spherical Warping](#spherical-warping)
11. [Blending](#blending)
12. [Video Frame Extraction](#video-frame-extraction)
13. [Optimizations](#optimizations)

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
5. Orientation, one of:
   chain  (v1)       Rotation Chaining & Temporal Smoothing
   bundle (default)  the chain (or a gyroscope) as a prior → pair selection → guided
                     rotation-only matching → focal search → rotation averaging →
                     bundle adjustment
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

Every frame's orientation is a rotation matrix R that maps world directions into that camera's
frame (world → camera):

```
r_cam = R @ r_world
```

Its transpose goes the other way (camera → world): `r_world = Rᵀ @ r_cam`, and the camera's optical
axis in world coordinates is the third row of R. Frame 0 is the reference: its R is the identity
unless a gyroscope prior sets it.

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

In the `chain` method, global rotations are smoothed using a **moving average**:

1. Pad the sequence at both ends by reflection, so every frame gets a full, centred window
2. For each rotation R[i], average the matrices in `[R[i-w/2], ..., R[i+w/2]]`
3. Orthonormalize the average by SVD
4. Apply one correction to every frame so that frame 0 is exactly where it was

**Window size**: Configurable (`matching.rotation_smoothing_window`; 17 in the example config)

This reduces jitter, at the cost of also averaging real motion. [TEMPORAL_SMOOTHING.md](TEMPORAL_SMOOTHING.md)
has the details. The `bundle` method below does not smooth.

## Global Orientation

The default method, `orientation.method: bundle` ([src/orientation.py](src/orientation.py)). It
replaces chaining and smoothing with one solve over many frame pairs.

### Prior

A first guess at every frame's rotation, used only to choose pairs, place search windows and start
the solve:

- **Without a gyroscope**: the v1 chain above, unsmoothed. Links the chain did not measure (failed
  pairs that fell back to no rotation, or interpolated ones) are marked as doubtful, so pairs that
  span them are searched without a window.
- **With a gyroscope**: the gyro integrated between frame timestamps ([src/imu.py](src/imu.py)):
  `C(t + dt) = C(t) · exp([ω dt]ₓ)` for the camera → world rotation C, with ω held constant between
  samples. Before that, two things are estimated from the video itself:
  - *Clock offset*: the angle turned over each frame interval is the same for both sensors and
    independent of axes, so the gyro's `∫|ω| dt` over each interval is correlated with the angle of
    the matching image pair over a range of offsets.
  - *Axes*: per interval, the camera rotation vector `v` and the gyro rotation vector `g` satisfy
    `v ≈ R_ci g`; Kabsch solves for `R_ci`, snapped to the nearest axis permutation.

How far the prior can drift between two frames sets the search window and the sanity bound:
`0.1°` per chain link, or `0.5°` per second of gyro.

### Pair Selection

- every frame with its next three;
- one pair per frame whose prior separation is closest to half the field of view;
- per frame, the two nearest frames the sweep reaches again after turning at least 180°: the loop
  closures.

After the first solve, pairs are selected again from the solved orientations and the new ones
matched.

### Guided, Rotation-Only Matching

Keypoints of frame i are projected through the prior's relative rotation `R_ij = R_j Rᵢᵀ` and only
matched against keypoints of frame j within the window around that prediction. Two bearing vectors
fix a rotation, so RANSAC draws two-point samples, solves each by Kabsch (all samples batched), and
counts inliers by the angle between `R bᵢ` and `bⱼ`. A pair that fails inside its window is retried
over the whole frame, still checked against the prior.

### Focal Length

A rotation fitted with the wrong focal length is scaled by roughly `f_true / f`. Along a sweep that
comes back round, that breaks the loop by about `(scale − 1) × 360°`. For candidate focal lengths
the pairs are refitted on their inliers and averaged by plain least squares, and the mean remaining
disagreement is minimised (coarse grid, then golden-section search). With a loop the minimum is
sharp and the result is used; without one it is shallow, and the configured focal length is kept.

### Rotation Averaging

Unknowns are a rotation vector per frame (frame 0 fixed). For every pair the residual is

```
e_ij = log( R_ijᵀ · R_j Rᵢᵀ )        (a rotation vector, radians)
```

minimised with a soft-L1 loss. Pairs still more than 3° off after a solve are dropped, a few at a
time and never a pair that is the only link across a stretch of frames; stretches that no pair
links are tied by the prior at full weight.

### Bundle Adjustment

Each verified correspondence `xᵢ ↔ xⱼ` gives a symmetric transfer residual

```
xⱼ ≈ π( K(f) · R_j Rᵢᵀ · K(f)⁻¹ xᵢ )      and the same from j to i
```

over all rotations and, when a loop pins it, one shared focal length (as `log f`), with a soft-L1
loss and an analytic Jacobian (left Jacobians of SO(3) for the rotation vectors). The solve starts
from the averaged rotations, so residuals start within a few pixels.

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
   r_cam = R @ r_world
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
2. **Open sweeps**: Without a loop closure the focal length, and with it the total sweep, comes from
   the configuration
3. **Vertical Coverage**: Limited by camera FOV (typically captures horizontal band)
4. **Moving Objects**: Will appear as artifacts or ghosts
5. **Rolling shutter and stabilisation**: Not modelled; they are part of the remaining residual

### Potential Improvements

1. **Rolling-shutter model**: A per-row rotation in bundle adjustment
2. **Horizon levelling**: Use the gyroscope's companion accelerometer, or the sweep's axis, to set
   the world's up direction instead of frame 0's
3. **Essential Matrix**: Use for scenes with small translation
4. **Line Straightening**: Post-process to detect and straighten lines
5. **Exposure Compensation**: Automatic exposure/white balance alignment
6. **GPU Acceleration**: CUDA/OpenCL for faster warping and blending

