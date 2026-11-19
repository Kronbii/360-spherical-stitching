# Temporal Smoothing (v1 `chain` method)

This document describes exactly how the `chain` orientation method smooths its rotations,
as implemented in `smooth_rotations_temporal` and `reflect_pad_rotations` in
[src/rotation.py](src/rotation.py).

> **Applies to `orientation.method: chain` only.** The default method since 2.0, `bundle`,
> does not smooth: it solves every rotation against every matched pair at once, which
> removes the jitter at its source instead of averaging real motion away with it. With
> `bundle`, `rotation_smoothing_window` is ignored. See [README.md](README.md) and
> [USAGE.md](USAGE.md#orientation).

## Why smooth at all

Each pairwise rotation is estimated independently, so every frame carries a few tenths of a
degree of pitch and roll error. Chained together, that shows up as a staircase along any long
straight edge. A moving average over the chained rotations damps it. It does not remove the
error that accumulates along the chain; only a global solve does that.

## Algorithm

### Input

- `global_rotations`: the chained rotations `[R₀, R₁, ..., Rₙ₋₁]` (world → camera).
- `window_size`: odd integer; even values are rounded up, values below 3 become 3.

### 1. Pad by reflection

Every frame gets a full, centred window, including the first and last. The sequence is
extended by `half = window_size // 2` frames at each end, reflected so that it *continues*
the motion rather than repeating the end pose:

```python
left  = [R₀ @ R_k.T @ R₀           for k = half, ..., 1]
right = [Rₙ₋₁ @ R_{n-1-k}.T @ Rₙ₋₁ for k = 1, ..., half]
padded = left + global_rotations + right
```

For a yaw-only sweep this is `yaw[-k] = 2·yaw[0] − yaw[k]` at the start, and the mirror of
that at the end. Clamping the window instead would average the tail mostly against earlier
frames and drag it backwards; leaving the last frame unsmoothed would make it snap away from
its smoothed neighbours, a visible tear on a clip that ends mid-motion.

### 2. Average and project back onto a rotation

```python
for i in range(n):
    window = padded[i : i + 2 * half + 1]
    smoothed[i] = orthonormalize_rotation(np.mean(window, axis=0))
```

Averaging the matrices element-wise is accurate for the small rotations between nearby video
frames; the average is not a rotation, so SVD projects it back onto the closest one:

```python
def orthonormalize_rotation(R_raw):
    U, S, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:        # a reflection, not a rotation
        U[:, -1] *= -1
        R = U @ Vt
    return R
```

### 3. Put frame 0 back

Frame 0 is the reference every other frame is expressed against, so the correction that
returns it exactly to its original orientation is applied to the whole sequence:

```python
C = global_rotations[0] @ smoothed[0].T
smoothed = [orthonormalize_rotation(C @ R) for R in smoothed]
```

## Where it runs

In the `chain` method, after the relative rotations are chained and before the wrap check
([src/pipeline.py](src/pipeline.py), `orient_with_chain`):

```python
global_rotations = chain_rotations(
    relative_rotations,
    apply_smoothing=config.matching.rotation_smoothing_window > 1,
    smoothing_window=config.matching.rotation_smoothing_window,
)
```

## Configuration

```yaml
orientation:
  method: chain
matching:
  rotation_smoothing_window: 17   # 1 turns smoothing off
```

- **3**: minimal smoothing.
- **5–9**: a balance for most videos.
- **11–17**: straighter long edges on slow sweeps; on fast or uneven motion it starts to bend
  real movement, which is what bundle adjustment avoids.

On the sample run a window of 17 makes pitch wobble about 16× and roll wobble about 15× quieter
(wobble measured as the standard deviation of the second difference, sweep removed).
