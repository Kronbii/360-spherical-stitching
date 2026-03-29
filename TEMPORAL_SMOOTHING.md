# Temporal Smoothing Algorithm - Exact Implementation

This document describes **exactly** how temporal smoothing is applied to rotation matrices to reduce jitter between frames.

## Overview

Temporal smoothing uses a **moving average filter** on rotation matrices. The key insight is that for small rotations in smooth sequences, averaging rotation matrices directly and then re-orthonormalizing produces good results.

## Algorithm Steps

### Input
- `global_rotations`: List of N 3×3 rotation matrices `[R₀, R₁, R₂, ..., Rₙ₋₁]`
- `window_size`: Odd integer (default 3, typically 3-15)

### Step-by-Step Process

#### 1. **Window Size Validation**
```python
if window_size % 2 == 0:
    window_size += 1  # Make odd
if window_size < 3:
    window_size = 3

half_window = window_size // 2  # e.g., window_size=5 → half_window=2
```

#### 2. **Initialize Output**
```python
smoothed = [global_rotations[0]]  # First rotation stays unchanged (reference frame)
```

**Important**: The first rotation matrix is kept as-is because it's the reference frame (typically identity).

#### 3. **Process Middle Frames (i = 1 to N-2)**

For each frame index `i` from 1 to N-2:

```python
for i in range(1, len(global_rotations) - 1):
    # Step 3a: Define window boundaries
    start_idx = max(0, i - half_window)
    end_idx = min(len(global_rotations), i + half_window + 1)
    
    # Step 3b: Collect all rotations in the window
    window_rots = []
    for j in range(start_idx, end_idx):
        window_rots.append(global_rotations[j])
    
    # Step 3c: Average the rotation matrices element-wise
    R_avg = np.mean(window_rots, axis=0)  # Shape: (3, 3)
    
    # Step 3d: Re-orthonormalize to ensure valid rotation matrix
    R_smooth = orthonormalize_rotation(R_avg)
    
    smoothed.append(R_smooth)
```

#### 4. **Final Frame**
```python
smoothed.append(global_rotations[-1])  # Last rotation stays unchanged
```

**Important**: The last rotation matrix is also kept as-is.

### Example: Window Size = 5

For a sequence of 10 rotations with `window_size=5` (half_window=2):

- **Frame 0**: Unchanged (reference)
- **Frame 1**: Average of frames [0, 1, 2, 3] (window clipped at start)
- **Frame 2**: Average of frames [0, 1, 2, 3, 4]
- **Frame 3**: Average of frames [1, 2, 3, 4, 5]
- **Frame 4**: Average of frames [2, 3, 4, 5, 6]
- **Frame 5**: Average of frames [3, 4, 5, 6, 7]
- **Frame 6**: Average of frames [4, 5, 6, 7, 8]
- **Frame 7**: Average of frames [5, 6, 7, 8, 9] (window clipped at end)
- **Frame 8**: Average of frames [6, 7, 8, 9] (window clipped at end)
- **Frame 9**: Unchanged (last frame)

## Orthonormalization (Critical Step)

After averaging, the matrix `R_avg` is no longer guaranteed to be a valid rotation matrix. It must be orthonormalized using **SVD (Singular Value Decomposition)**:

```python
def orthonormalize_rotation(R_raw: np.ndarray) -> np.ndarray:
    # Step 1: Compute SVD
    U, S, Vt = np.linalg.svd(R_raw)
    
    # Step 2: Reconstruct rotation matrix
    R = U @ Vt
    
    # Step 3: Ensure proper rotation (det = +1, not reflection)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1  # Flip last column of U
        R = U @ Vt
    
    return R
```

**Why SVD?**
- SVD decomposes `R_raw = U @ diag(S) @ Vt`
- For a valid rotation matrix, we want `R = U @ Vt` (ignoring singular values)
- This gives the closest valid rotation matrix to `R_raw` in the Frobenius norm

## Mathematical Details

### Matrix Averaging

For a window of rotation matrices `[R₁, R₂, ..., Rₖ]`, the average is:

```
R_avg = (1/k) * (R₁ + R₂ + ... + Rₖ)
```

This is **element-wise averaging** of the 3×3 matrices.

### Why This Works for Small Rotations

For small rotations, the rotation matrices are close to identity:
- `R ≈ I + ε` where `ε` is small
- Averaging: `(I + ε₁ + I + ε₂ + ...)/k ≈ I + (ε₁ + ε₂ + ...)/k`
- After orthonormalization, this gives a smooth interpolation

**Note**: This is an approximation. For large rotations, you might want to use:
- Lie algebra (matrix logarithm): `log(R_avg) = mean([log(R₁), log(R₂), ...])`
- Quaternion averaging
- But for typical panorama sequences with small per-frame rotations, direct averaging works well.

## Implementation in Pipeline

The smoothing is applied **after** rotation chaining:

```python
# 1. Chain relative rotations to get global rotations
global_rotations = chain_rotations(relative_rotations, apply_smoothing=False)

# 2. Apply temporal smoothing
if apply_smoothing and len(global_rotations) > 3:
    global_rotations = smooth_rotations_temporal(
        global_rotations, 
        window_size=smoothing_window
    )
```

## Key Properties

1. **Boundary Conditions**: First and last rotations are never smoothed (kept as reference)
2. **Symmetric Window**: Window is centered on each frame (when possible)
3. **Edge Handling**: At sequence boundaries, window is clipped (not padded)
4. **Orthonormalization**: Every smoothed rotation is re-orthonormalized via SVD
5. **Determinant Check**: Ensures `det(R) = +1` (proper rotation, not reflection)

## Replication in Another Pipeline

To replicate this in another pipeline:

```python
import numpy as np

def orthonormalize_rotation(R_raw):
    """Orthonormalize matrix to valid rotation using SVD."""
    U, S, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def smooth_rotations_temporal(global_rotations, window_size=3):
    """
    Apply temporal smoothing to rotation matrices.
    
    Args:
        global_rotations: List of 3×3 rotation matrices
        window_size: Odd integer, size of smoothing window
    
    Returns:
        List of smoothed rotation matrices
    """
    n = len(global_rotations)
    if n <= 2:
        return global_rotations
    
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    if window_size < 3:
        window_size = 3
    
    half_window = window_size // 2
    smoothed = [global_rotations[0]]  # First unchanged
    
    # Process middle frames
    for i in range(1, n - 1):
        # Define window
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        
        # Collect rotations in window
        window_rots = global_rotations[start_idx:end_idx]
        
        # Average element-wise
        R_avg = np.mean(window_rots, axis=0)
        
        # Re-orthonormalize
        R_smooth = orthonormalize_rotation(R_avg)
        smoothed.append(R_smooth)
    
    # Last unchanged
    smoothed.append(global_rotations[-1])
    
    return smoothed
```

## Configuration

In your config, the smoothing window is controlled by:
```yaml
matching:
  rotation_smoothing_window: 3  # Default, can increase to 7-15 for smoother results
```

**Guidelines**:
- **window_size = 3**: Minimal smoothing, preserves more detail
- **window_size = 5-7**: Good balance for most videos
- **window_size = 9-15**: Strong smoothing, straighter lines, may oversmooth fast motion

