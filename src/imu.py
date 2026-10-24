"""
Gyroscope integration and orientation priors.

The camera turns about its own centre, so a gyroscope alone predicts how every frame
is oriented relative to the first. That prediction drifts — bias, scale error and noise
all integrate — but over a few frames it is accurate to a fraction of a degree, which
is enough to tell the matcher where to look and to start the bundle adjustment close to
the answer.

Conventions match the rest of the pipeline (see warp_sphere.py):
- R (world -> camera): a world direction d appears in the camera as R @ d.
- C = R.T (camera -> world).
- The gyro measures the camera's angular velocity w in the camera frame, so
  dC/dt = C @ [w]x and, over a short step, C(t + dt) = C(t) @ exp([w dt]x).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix [v]x, so that skew(a) @ b == np.cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation matrix for the rotation vector w (Rodrigues' formula)."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    K = skew(w)
    if theta < 1e-8:
        # Second-order Taylor expansion keeps it accurate near identity
        return np.eye(3) + K + 0.5 * K @ K
    return (np.eye(3) + np.sin(theta) / theta * K
            + (1.0 - np.cos(theta)) / theta ** 2 * K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (inverse of so3_exp)."""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * v
    if np.pi - theta < 1e-5:
        # Near 180 degrees the antisymmetric part vanishes; take the axis from R + I
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
        k = int(np.argmax(axis))
        axis = M[:, k] / max(axis[k], 1e-12)
        return theta * axis / np.linalg.norm(axis)
    return theta / (2.0 * np.sin(theta)) * v


def rotation_error_degrees(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Angle of the rotation separating an estimate from the truth, in degrees."""
    return float(np.degrees(np.linalg.norm(so3_log(R_est @ R_true.T))))


@dataclass
class GyroLog:
    """A gyroscope recording: timestamps (s) and angular velocity (rad/s, camera frame)."""
    t: np.ndarray       # (M,)
    omega: np.ndarray   # (M, 3)


def integrate_gyro(
    gyro: GyroLog,
    frame_times: np.ndarray,
    R0: np.ndarray,
    bias: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Dead-reckon a world -> camera rotation for every frame timestamp.

    Each gyro sample is held constant until the next one (zero-order hold) and the
    interval is split exactly at frame timestamps, so frames need not line up with
    samples.

    Args:
        gyro: Gyro samples, sorted by time and spanning the frame timestamps.
        frame_times: Frame timestamps in seconds, ascending.
        R0: Orientation (world -> camera) at frame_times[0]. In practice this comes
            from the accelerometer (pitch and roll) with yaw set to zero.
        bias: Optional bias estimate (rad/s) to subtract from every sample.

    Returns:
        List of world -> camera rotations, one per frame.
    """
    t = np.asarray(gyro.t, dtype=np.float64)
    w = np.asarray(gyro.omega, dtype=np.float64)
    if bias is not None:
        w = w - np.asarray(bias)[None, :]
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if len(frame_times) == 0:
        return []
    if t[0] > frame_times[0] + 1e-9 or t[-1] < frame_times[-1] - 1e-9:
        raise ValueError("Gyro log must cover every frame timestamp")

    C = R0.T.copy()  # camera -> world
    out = [R0.copy()]
    k = int(np.searchsorted(t, frame_times[0], side="right") - 1)
    now = frame_times[0]
    for target in frame_times[1:]:
        while now < target - 1e-12:
            next_sample = t[k + 1] if k + 1 < len(t) else np.inf
            step_end = min(next_sample, target)
            dt = step_end - now
            if dt > 0:
                C = C @ so3_exp(w[k] * dt)
            now = step_end
            if step_end >= next_sample - 1e-12:
                k += 1
        # Keep C on the manifold over long integrations
        U, _, Vt = np.linalg.svd(C)
        C = U @ Vt
        out.append(C.T.copy())
    return out


def relative_rotation(R_i: np.ndarray, R_j: np.ndarray) -> np.ndarray:
    """Rotation taking camera-i bearings into camera j: b_j = R_ij @ b_i."""
    return R_j @ R_i.T


def predicted_overlap_pairs(
    rotations: List[np.ndarray],
    max_angle_deg: float,
    min_gap: int = 1
) -> List[tuple]:
    """
    Frame pairs whose optical axes the prior puts within max_angle_deg of each other.

    This is how loop closures are found without any image search: after a full turn the
    gyro, drift and all, still says the last frames look at the same wall as the first.

    Args:
        rotations: Prior world -> camera rotations.
        max_angle_deg: Largest angle between optical axes for a pair to be tried.
        min_gap: Smallest index difference to consider.

    Returns:
        Sorted list of (i, j) with i < j.
    """
    axes = np.array([R[2, :] for R in rotations])  # optical axis in world, per frame
    cos_max = np.cos(np.radians(max_angle_deg))
    pairs = []
    n = len(rotations)
    for i in range(n):
        dots = axes[i + min_gap:] @ axes[i]
        for off in np.nonzero(dots >= cos_max)[0]:
            pairs.append((i, i + min_gap + int(off)))
    return pairs
