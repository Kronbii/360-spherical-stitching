"""
Synthetic sweep generator: camera frames and a gyroscope log with known ground truth.

The "world" is an existing equirectangular panorama. A pinhole camera turning about its
own centre sees exactly a reprojection of that panorama, so rendering a frame is the
warp in warp_sphere.py run backwards: for every camera pixel, take its ray, rotate it
into the world, look up the panorama.

What is simulated, and what is not:
- Simulated: a handheld-style sweep (yaw past 360° with pitch and roll wobble), per-frame
  exposure changes, sensor noise, and a MEMS-like gyroscope with constant-plus-random-walk
  bias, scale-factor error and white noise, sampled faster than the camera.
- Not simulated: translation of the camera (parallax), lens distortion, rolling shutter,
  motion blur, camera-IMU time offset or misalignment. Those are real-world effects
  this simulation leaves out on purpose so the ground truth stays exact.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from .config import CalibrationData
from .imu import GyroLog, so3_exp, so3_log

logger = logging.getLogger(__name__)


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class SweepSpec:
    """Parameters of the simulated capture."""
    duration_s: float = 24.0
    frame_rate_hz: float = 3.0
    gyro_rate_hz: float = 200.0
    yaw_span_deg: float = 396.0        # a little past a full turn, so the ends overlap
    pitch_amp_deg: float = 4.0
    pitch_period_s: float = 9.0
    roll_amp_deg: float = 2.5
    roll_period_s: float = 6.5
    image_width: int = 800
    image_height: int = 600
    hfov_deg: float = 65.0
    exposure_jitter: float = 0.06      # relative per-frame gain variation
    pixel_noise_sigma: float = 2.0     # grey levels
    # Gyroscope error model (per axis, rad/s unless noted)
    gyro_bias_const_dps: float = 0.25  # constant bias magnitude, deg/s
    gyro_bias_walk_dps_rt_s: float = 0.02  # bias random walk, deg/s/sqrt(s)
    gyro_noise_dps_rt_hz: float = 0.01     # white noise density, deg/s/sqrt(Hz)
    gyro_scale_error: float = 0.005    # scale-factor error (fraction)
    seed: int = 7


@dataclass
class SimulatedSweep:
    """Everything the simulator produced, including the ground truth."""
    spec: SweepSpec
    calib: CalibrationData
    frame_times: np.ndarray
    frames: List[np.ndarray]
    gt_rotations: List[np.ndarray]       # world -> camera, per frame
    gyro: GyroLog
    gyro_bias_true: np.ndarray           # (M, 3) the bias actually applied
    exposure_gains: np.ndarray = field(default_factory=lambda: np.zeros(0))


def true_orientation(t: float, spec: SweepSpec) -> np.ndarray:
    """
    World -> camera rotation at time t for the simulated sweep.

    Yaw follows a smooth ease-in/ease-out profile so the camera starts and ends at rest;
    pitch and roll wobble the way a hand-held phone does.
    """
    s = np.clip(t / spec.duration_s, 0.0, 1.0)
    # Smoothstep-like profile with non-zero speed in the middle
    ease = s - np.sin(2 * np.pi * s) / (2 * np.pi)
    yaw = np.radians(spec.yaw_span_deg) * ease
    pitch = np.radians(spec.pitch_amp_deg) * np.sin(2 * np.pi * t / spec.pitch_period_s)
    roll = np.radians(spec.roll_amp_deg) * np.sin(2 * np.pi * t / spec.roll_period_s + 1.0)
    # Camera -> world: yaw about world vertical, then pitch, then roll about the optical axis
    C = rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)
    return C.T


def body_rates(spec: SweepSpec, t: np.ndarray) -> np.ndarray:
    """True angular velocity in the camera frame, by central differences of orientation."""
    h = 1e-4
    out = np.empty((len(t), 3))
    for k, tk in enumerate(t):
        C_minus = true_orientation(tk - h, spec).T
        C_plus = true_orientation(tk + h, spec).T
        out[k] = so3_log(C_minus.T @ C_plus) / (2 * h)
    return out


def simulate_gyro(spec: SweepSpec, rng: np.random.Generator):
    """
    Gyro samples with bias, scale error and noise, over the whole sweep.

    Returns (GyroLog of measurements, true bias per sample).
    """
    dt = 1.0 / spec.gyro_rate_hz
    t = np.arange(0.0, spec.duration_s + dt / 2, dt)
    w_true = body_rates(spec, t)

    bias0 = rng.normal(size=3)
    bias0 = bias0 / np.linalg.norm(bias0) * np.radians(spec.gyro_bias_const_dps)
    walk = rng.normal(size=(len(t), 3)) * np.radians(spec.gyro_bias_walk_dps_rt_s) * np.sqrt(dt)
    bias = bias0[None, :] + np.cumsum(walk, axis=0)
    noise = rng.normal(size=(len(t), 3)) * np.radians(spec.gyro_noise_dps_rt_hz) * np.sqrt(spec.gyro_rate_hz)
    scale = 1.0 + spec.gyro_scale_error * np.array([1.0, -0.6, 0.4])

    w_meas = w_true * scale[None, :] + bias + noise
    return GyroLog(t=t, omega=w_meas), bias


def render_frame(
    world: np.ndarray,
    R: np.ndarray,
    calib: CalibrationData,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Pinhole view of an equirectangular world for a camera with world -> camera rotation R.

    Uses the same angle conventions as warp_sphere.create_equirectangular_grid, so a frame
    rendered here and warped back by the pipeline lands exactly where it came from.
    """
    Hw, Ww = world.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays = np.stack([(u - calib.cx) / calib.fx, (v - calib.cy) / calib.fy, np.ones_like(u)], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    d = rays @ R  # world direction = Rᵀ r, written for row vectors
    theta = np.arctan2(d[..., 0], d[..., 2])
    phi = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    map_x = ((theta + np.pi) / (2 * np.pi) * (Ww - 1)).astype(np.float32)
    map_y = ((phi + np.pi / 2) / np.pi * (Hw - 1)).astype(np.float32)
    return cv2.remap(world, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def simulate_sweep(world: np.ndarray, spec: SweepSpec = SweepSpec()) -> SimulatedSweep:
    """Render every frame and record the gyro for one simulated sweep."""
    rng = np.random.default_rng(spec.seed)
    W, H = spec.image_width, spec.image_height
    f = (W / 2) / np.tan(np.radians(spec.hfov_deg) / 2)
    calib = CalibrationData(fx=f, fy=f, cx=(W - 1) / 2, cy=(H - 1) / 2,
                            source=f"simulation: {spec.hfov_deg:.1f}° horizontal FOV")

    n_frames = int(round(spec.duration_s * spec.frame_rate_hz)) + 1
    frame_times = np.linspace(0.0, spec.duration_s, n_frames)
    gt = [true_orientation(t, spec) for t in frame_times]

    gains = 1.0 + rng.uniform(-spec.exposure_jitter, spec.exposure_jitter, n_frames)
    frames = []
    for R, g in zip(gt, gains):
        img = render_frame(world, R, calib, W, H).astype(np.float32) * g
        img += rng.normal(0.0, spec.pixel_noise_sigma, img.shape).astype(np.float32)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))

    gyro, bias = simulate_gyro(spec, rng)
    logger.info(f"Simulated {n_frames} frames at {spec.frame_rate_hz} Hz and "
                f"{len(gyro.t)} gyro samples at {spec.gyro_rate_hz} Hz")
    return SimulatedSweep(spec=spec, calib=calib, frame_times=frame_times, frames=frames,
                          gt_rotations=gt, gyro=gyro, gyro_bias_true=bias,
                          exposure_gains=gains)
