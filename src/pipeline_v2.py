"""
v2 orientation pipeline on simulated data: the gyro and no-gyro paths side by side.

Both paths run the shared stage in orientation.py (pair selection, guided matching,
rotation averaging, bundle adjustment). They differ only in the prior:

- run_orientation_v2: the integrated gyroscope.
- run_orientation_without_imu: the v1 homography chain.

Pairwise chaining of the v2 rotation-only matches is kept for comparison, as is the
v1 chain itself. Warping and blending reuse the v1 modules unchanged.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .blend import blend_panorama
from .bundle_adjust import BundleResult, reprojection_rms
from .config import BlendingConfig, CalibrationData, MatchingConfig
from .guided_match import PairMatch
from .imu import GyroLog, integrate_gyro, relative_rotation, rotation_error_degrees
from .orientation import (OrientationOptions, OrientationResult, detect_all_features,
                          estimate_orientation, gyro_prior, homography_chain_prior)
from .rotation import orthonormalize_rotation
from .warp_sphere import create_equirectangular_grid, warp_image_to_equirectangular, world_direction_grid

logger = logging.getLogger(__name__)

# The simulation entry points take the shared options unchanged
V2Options = OrientationOptions


@dataclass
class V2Result:
    """Orientations from each stage plus diagnostics."""
    imu_rotations: List[np.ndarray]
    chained_rotations: List[np.ndarray]
    ba_rotations: List[np.ndarray]
    focal_initial: float
    focal_ba: float
    pairs: List[PairMatch]
    bundle: BundleResult
    diagnostics: Dict = field(default_factory=dict)
    averaged_rotations: List[np.ndarray] = field(default_factory=list)


def chain_adjacent(
    R0: np.ndarray,
    pairs: List[PairMatch],
    fallback: List[np.ndarray],
    n: int,
) -> List[np.ndarray]:
    """
    Multiply adjacent pairwise rotations into global orientations (no global refinement).

    A failed adjacent pair falls back to the prior's relative rotation for that step.
    """
    adjacent = {(p.i, p.j): p for p in pairs if p.j == p.i + 1 and p.success}
    out = [R0.copy()]
    for i in range(n - 1):
        p = adjacent.get((i, i + 1))
        R_ij = p.R_ij if p is not None else relative_rotation(fallback[i], fallback[i + 1])
        out.append(orthonormalize_rotation(R_ij @ out[-1]))
    return out


def run_orientation_v2(
    frames: List[np.ndarray],
    frame_times: np.ndarray,
    gyro: GyroLog,
    calib_init: CalibrationData,
    R0: np.ndarray,
    options: OrientationOptions = OrientationOptions(),
    progress=None,
) -> V2Result:
    """
    Estimate every frame's orientation from images and a gyroscope.

    Args:
        frames: BGR frames in capture order.
        frame_times: Frame timestamps (s), on the same clock as the gyro.
        gyro: Gyroscope log covering the frames, in the camera frame.
        calib_init: Starting intrinsics (fx = fy assumed). The focal length is refined.
        R0: World -> camera orientation of the first frame (defines the output frame).
        options: Settings.
        progress: Optional callable(stage: str, done: int, total: int).

    Returns:
        V2Result with the gyro-only, chained and bundle-adjusted orientations.
    """
    n = len(frames)
    feats = detect_all_features(frames, options.orb_nfeatures, options.workers)
    prior_rots = integrate_gyro(gyro, frame_times, R0)
    prior = gyro_prior(prior_rots, frame_times, options.gyro_drift_deg_per_s)
    h, w = frames[0].shape[:2]
    res = estimate_orientation(feats, calib_init.K, (w, h), prior, options, progress)
    return V2Result(
        imu_rotations=prior_rots,
        chained_rotations=chain_adjacent(R0, res.pairs, prior_rots, n),
        ba_rotations=res.rotations,
        focal_initial=float(calib_init.fx),
        focal_ba=res.focal,
        pairs=res.pairs,
        bundle=res.bundle,
        diagnostics=res.diagnostics,
        averaged_rotations=res.averaged,
    )


def run_orientation_without_imu(
    frames: List[np.ndarray],
    calib_init: CalibrationData,
    R0: np.ndarray,
    options: OrientationOptions = OrientationOptions(),
    matching: Optional[MatchingConfig] = None,
) -> Tuple[OrientationResult, List[np.ndarray]]:
    """
    Same stage as run_orientation_v2, with the v1 homography chain as the prior.

    Returns:
        (OrientationResult, the unsmoothed v1 chain used as the prior).
    """
    matching = matching or MatchingConfig(match_width=None, min_inliers=60, rotation_smoothing_window=1)
    prior = homography_chain_prior(frames, calib_init.K, matching, options.chain_drift_deg_per_link, R0=R0)
    feats = detect_all_features(frames, options.orb_nfeatures, options.workers)
    h, w = frames[0].shape[:2]
    res = estimate_orientation(feats, calib_init.K, (w, h), prior, options)
    return res, prior.rotations


def per_frame_errors(estimated: List[np.ndarray], truth: List[np.ndarray]) -> np.ndarray:
    """Rotation error (degrees) of each frame against ground truth."""
    return np.array([rotation_error_degrees(E, T) for E, T in zip(estimated, truth)])


def evaluate_against_truth(
    result: V2Result,
    truth: List[np.ndarray],
    true_focal: float,
    cx: float,
    cy: float,
    extra: Optional[Dict[str, Union[List[np.ndarray], Tuple[List[np.ndarray], float]]]] = None,
) -> Dict:
    """
    Per-frame rotation error and reprojection RMS for each stage.

    Reprojection RMS is measured on the same verified correspondences for every method,
    each with the focal length it would actually use (initial for gyro and chaining,
    refined for bundle adjustment). An extra method is either a list of rotations (scored
    with the initial focal) or a (rotations, focal) tuple.
    """
    ok = [p for p in result.pairs if p.success]
    methods = {
        "imu_only": (result.imu_rotations, result.focal_initial),
        "pairwise_chain": (result.chained_rotations, result.focal_initial),
        "bundle_adjusted": (result.ba_rotations, result.focal_ba),
    }
    for name, value in (extra or {}).items():
        methods[name] = value if isinstance(value, tuple) else (value, result.focal_initial)

    report = {}
    for name, (rots, f) in methods.items():
        err = per_frame_errors(rots, truth)
        loop_gap = rotation_error_degrees(rots[-1] @ rots[0].T, truth[-1] @ truth[0].T)
        report[name] = {
            "mean_deg": float(err.mean()),
            "median_deg": float(np.median(err)),
            "max_deg": float(err.max()),
            "final_frame_deg": float(err[-1]),
            "end_to_start_deg": float(loop_gap),
            "reprojection_rms_px": reprojection_rms(rots, f, cx, cy, ok),
            "per_frame_deg": [float(e) for e in err],
        }
    report["focal"] = {
        "true_px": float(true_focal),
        "initial_px": result.focal_initial,
        "initial_error_pct": 100.0 * (result.focal_initial - true_focal) / true_focal,
        "bundle_adjusted_px": result.focal_ba,
        "bundle_adjusted_error_pct": 100.0 * (result.focal_ba - true_focal) / true_focal,
    }
    return report


def stitch_equirectangular(
    frames: List[np.ndarray],
    rotations: List[np.ndarray],
    calib: CalibrationData,
    width: int,
    blending: BlendingConfig = BlendingConfig(method="feather"),
):
    """Warp every frame with the given rotations and blend, using the v1 warp and blend."""
    H, W = width // 2, width
    theta, phi = create_equirectangular_grid(W, H)
    dirs = world_direction_grid(theta, phi)
    warped, masks = [], []
    for img, R in zip(frames, rotations):
        w, m = warp_image_to_equirectangular(img, R, calib, theta, phi, (H, W), dirs)
        warped.append(w)
        masks.append(m)
    pano = blend_panorama(warped, masks, blending)
    coverage = np.zeros((H, W), np.uint8)
    for m in masks:
        np.maximum(coverage, m, out=coverage)
    return pano, coverage
