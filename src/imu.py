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


# ---------------------------------------------------------------------------------------
# Reading real logs
# ---------------------------------------------------------------------------------------

_TIME_COLUMNS = ("t", "time_s", "seconds_elapsed", "time", "timestamp", "seconds", "sec")
_AXIS_COLUMNS = {
    0: ("wx", "x", "gx", "gyro_x", "gyrox", "rotationratex", "rotation_rate_x", "omega_x"),
    1: ("wy", "y", "gy", "gyro_y", "gyroy", "rotationratey", "rotation_rate_y", "omega_y"),
    2: ("wz", "z", "gz", "gyro_z", "gyroz", "rotationratez", "rotation_rate_z", "omega_z"),
}


def _time_to_seconds(t: np.ndarray) -> np.ndarray:
    """
    Convert a timestamp column to seconds, whatever unit the logger used.

    Gyros sample at tens to thousands of hertz, so the median step identifies the unit:
    ~0.001-0.05 in seconds, ~1-50 in milliseconds, and so on.
    """
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    for scale in (1.0, 1e-3, 1e-6, 1e-9):
        if step * scale <= 0.1:
            return t * scale
    raise ValueError(f"Cannot tell the unit of gyro timestamps (median step {step})")


def load_gyro_csv(path, units: str = "rad/s") -> GyroLog:
    """
    Read a gyroscope log from CSV.

    Expects a header row. The time column is the first of t, time_s, seconds_elapsed,
    time, timestamp, seconds; its unit (s, ms, us or ns) is inferred from the sample
    spacing. Angular-rate columns are matched by common names (wx/x/gx/gyro_x/
    rotationRateX ...), case-insensitively. Rates are in the device's own axes; the
    pipeline works out how those map onto the camera.

    Args:
        path: CSV file.
        units: "rad/s" or "deg/s".

    Returns:
        GyroLog sorted by time, with duplicate timestamps dropped.
    """
    import csv
    from pathlib import Path

    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        rows = [r for r in reader if r and any(c.strip() for c in r)]

    def column(candidates):
        for name in candidates:
            if name in header:
                return header.index(name)
        return None

    t_col = column(_TIME_COLUMNS)
    w_cols = [column(_AXIS_COLUMNS[k]) for k in range(3)]
    if t_col is None or any(c is None for c in w_cols):
        raise ValueError(f"{path}: need a time column and three angular-rate columns, got {header}")

    data = np.array([[float(r[t_col])] + [float(r[c]) for c in w_cols] for r in rows])
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    keep = np.concatenate([[True], np.diff(data[:, 0]) > 0])
    data = data[keep]

    t = _time_to_seconds(data[:, 0])
    omega = data[:, 1:4]
    if units in ("deg/s", "deg_s", "dps"):
        omega = np.radians(omega)
    elif units not in ("rad/s", "rad_s"):
        raise ValueError(f"Unknown gyro units {units!r}; use 'rad/s' or 'deg/s'")
    logger.info(f"Gyro log {path.name}: {len(t)} samples over {t[-1] - t[0]:.1f}s "
                f"(~{(len(t) - 1) / (t[-1] - t[0]):.0f} Hz)")
    return GyroLog(t=t, omega=omega)


def write_frame_times(path, names: List[str], times: List[float]) -> None:
    """Write a frame_times.csv: one row per frame, file name and time in seconds."""
    from pathlib import Path
    with Path(path).open("w") as f:
        f.write("file,time_s\n")
        for name, t in zip(names, times):
            f.write(f"{name},{t:.6f}\n")


def load_frame_times(path, names: List[str]) -> np.ndarray:
    """
    Timestamps (s) for the named frames, in the order given, from a frame_times.csv.

    Raises:
        ValueError: If a frame has no timestamp.
    """
    import csv
    from pathlib import Path

    with Path(path).open(newline="") as f:
        table = {row["file"].strip(): float(row["time_s"]) for row in csv.DictReader(f)}
    missing = [n for n in names if n not in table]
    if missing:
        raise ValueError(f"{path}: no timestamp for {len(missing)} frame(s), e.g. {missing[0]}")
    return np.array([table[n] for n in names])


# ---------------------------------------------------------------------------------------
# Calibrating a gyro against the video: time offset and axes
# ---------------------------------------------------------------------------------------

def parse_axes(spec: List[str]) -> np.ndarray:
    """
    Rotation taking gyro axes to camera axes from a list like ["x", "-y", "-z"].

    Entry k names the gyro axis (with sign) that becomes camera axis k. Camera axes are
    x right, y down, z forward. ["x", "-y", "-z"] is a phone's rear camera in the
    upright orientation of the video, for the iOS / Android device frame (x right,
    y up, z out of the screen).
    """
    if len(spec) != 3:
        raise ValueError(f"axes needs three entries, got {spec}")
    R = np.zeros((3, 3))
    for row, s in enumerate(spec):
        s = str(s).strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        axis = s.lstrip("+-")
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Bad axis {s!r} in {spec}")
        R[row, "xyz".index(axis)] = sign
    if not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError(f"axes {spec} is not a rotation (each axis once, determinant +1)")
    return R


def _cumulative_angle(gyro: GyroLog) -> np.ndarray:
    """∫|ω|dt at every gyro sample (trapezoid rule)."""
    speed = np.linalg.norm(gyro.omega, axis=1)
    return np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(gyro.t))])


def estimate_time_offset(
    gyro: GyroLog,
    frame_times: np.ndarray,
    vision_angles: np.ndarray,
    coarse_step_s: float = 0.005,
) -> tuple:
    """
    Offset between the gyro clock and the video clock: gyro_time = video_time + offset.

    The angle the camera turns between two frames is the same whichever sensor measures
    it and whichever axes it is measured in, so it can be lined up before anything else
    is known. For every candidate offset, the gyro's turn over each frame interval is
    correlated with the turn the images saw; the offset with the highest correlation
    wins, then gets refined on a finer grid.

    Args:
        gyro: Gyro log on its own clock and axes.
        frame_times: Frame timestamps on the video clock.
        vision_angles: Angle (rad) turned over each frame interval, NaN where unknown.

    Returns:
        (offset_s, correlation at the best offset).
    """
    frame_times = np.asarray(frame_times, dtype=np.float64)
    valid = np.isfinite(vision_angles)
    if valid.sum() < 5:
        raise ValueError("Too few frame intervals with a vision rotation to align the gyro clock")
    lo = gyro.t[0] - frame_times[0]
    hi = gyro.t[-1] - frame_times[-1]
    if hi < lo:
        raise ValueError("The gyro log is shorter than the video; it cannot cover every frame")

    A = _cumulative_angle(gyro)
    t0, t1 = frame_times[:-1][valid], frame_times[1:][valid]
    v = vision_angles[valid]
    v = (v - v.mean()) / (v.std() + 1e-12)

    def scores(offsets):
        out = np.empty(len(offsets))
        for s in range(0, len(offsets), 2048):
            o = offsets[s:s + 2048, None]
            g = np.interp(t1[None, :] + o, gyro.t, A) - np.interp(t0[None, :] + o, gyro.t, A)
            g = (g - g.mean(axis=1, keepdims=True)) / (g.std(axis=1, keepdims=True) + 1e-12)
            out[s:s + 2048] = (g * v[None, :]).mean(axis=1)
        return out

    coarse = np.arange(lo, hi + coarse_step_s / 2, coarse_step_s) if hi > lo else np.array([lo])
    c = scores(coarse)
    best = coarse[int(np.argmax(c))]
    fine = np.clip(np.arange(best - 2 * coarse_step_s, best + 2 * coarse_step_s, 1e-4), lo, hi)
    f = scores(fine)
    k = int(np.argmax(f))
    return float(fine[k]), float(f[k])


def gyro_interval_rotvecs(gyro: GyroLog, frame_times: np.ndarray) -> np.ndarray:
    """Rotation vector of the body increment over each frame interval, in the gyro's axes."""
    R = integrate_gyro(gyro, frame_times, np.eye(3))
    return np.array([so3_log(R[k] @ R[k + 1].T) for k in range(len(R) - 1)])


def estimate_axis_alignment(
    gyro_rotvecs: np.ndarray,
    vision_rotvecs: np.ndarray,
    snap_deg: float = 10.0,
) -> tuple:
    """
    Rotation taking gyro axes to camera axes, from per-interval rotations of both.

    Each frame interval gives the same rotation twice, once in the gyro's axes and once
    in the camera's: v_k ≈ R g_k. Kabsch finds R. Phones mount the IMU square to the
    camera, so a result within snap_deg of an axis permutation is snapped onto it.

    Args:
        gyro_rotvecs: (N, 3) gyro rotation vectors per interval.
        vision_rotvecs: (N, 3) camera rotation vectors for the same intervals (NaN rows
            are skipped).

    Returns:
        (R, snapped: bool, residual_deg: angle between the raw fit and the snapped one).
    """
    ok = np.all(np.isfinite(vision_rotvecs), axis=1) & (np.linalg.norm(vision_rotvecs, axis=1) > 1e-3)
    if ok.sum() < 5:
        raise ValueError("Too few frame intervals with enough rotation to align gyro axes")
    H = gyro_rotvecs[ok].T @ vision_rotvecs[ok]
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # Nearest signed permutation with determinant +1
    import itertools
    best, best_err = None, np.inf
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            P = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                P[row, col] = s
            if np.linalg.det(P) < 0:
                continue
            err = rotation_error_degrees(R, P)
            if err < best_err:
                best, best_err = P, err
    if best_err <= snap_deg:
        return best, True, float(best_err)
    return R, False, float(best_err)


def gyro_to_camera(gyro: GyroLog, offset_s: float, R_cam_imu: np.ndarray) -> GyroLog:
    """A gyro log moved onto the video clock and into camera axes."""
    return GyroLog(t=gyro.t - offset_s, omega=gyro.omega @ R_cam_imu.T)
