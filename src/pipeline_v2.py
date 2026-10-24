"""
v2 orientation pipeline: gyro prior -> guided pairwise rotations -> bundle adjustment.

Stages:
1. Gyro prior — integrate the gyroscope between frame timestamps to get a dead-reckoned
   orientation per frame.
2. Pair selection — adjacent pairs, skip-one pairs, and any pair the prior says should
   overlap. The last group is where loop closures come from: after a full turn the
   prior still knows the last frames face the same wall as the first ones.
3. Guided matching — ORB matches searched only near the position the prior predicts,
   then a rotation-only RANSAC (guided_match.py).
4. Chaining (for comparison) — adjacent pairwise rotations multiplied together, the way
   v1 builds its orientations.
5. Bundle adjustment — all rotations and the focal length refined together over every
   pair (bundle_adjust.py), starting from the gyro prior.

Warping and blending reuse the v1 modules unchanged.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .blend import blend_panorama
from .bundle_adjust import BundleResult, bundle_adjust, reprojection_rms
from .config import BlendingConfig, CalibrationData
from .guided_match import FrameFeatures, PairMatch, detect_features, match_pair
from .imu import GyroLog, integrate_gyro, predicted_overlap_pairs, relative_rotation, rotation_error_degrees
from .rotation import orthonormalize_rotation
from .warp_sphere import create_equirectangular_grid, warp_image_to_equirectangular, world_direction_grid

logger = logging.getLogger(__name__)


@dataclass
class V2Options:
    """Settings for the v2 orientation pipeline."""
    orb_nfeatures: int = 3000
    base_search_radius_px: float = 25.0   # window around the gyro prediction, before drift
    gyro_drift_deg_per_s: float = 0.5     # assumed worst-case prior drift rate, widens the window
    ransac_threshold_px: float = 2.0
    min_inliers: int = 30
    overlap_angle_fraction: float = 0.5   # try pairs whose axes are within this × HFOV
    max_pair_gap_local: int = 3           # always try (i, i+1) .. (i, i+gap)
    min_loop_gap: int = 12                # prior-predicted pairs at least this many frames apart
    optimize_focal: bool = True
    ba_loss_scale_px: float = 1.5
    ba_max_per_pair: int = 150
    seed: int = 0


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


def select_pairs(prior: List[np.ndarray], hfov_deg: float, options: V2Options) -> List[tuple]:
    """Local pairs plus every pair the prior predicts to overlap substantially."""
    n = len(prior)
    pairs = {(i, i + g) for g in range(1, options.max_pair_gap_local + 1)
             for i in range(n - g)}
    # Beyond the local window, only pairs far apart in time: those are the loop closures
    # chaining can never see, and the only ones that add information the local pairs lack
    pairs.update(predicted_overlap_pairs(prior, options.overlap_angle_fraction * hfov_deg,
                                         min_gap=max(options.min_loop_gap,
                                                     options.max_pair_gap_local + 1)))
    return sorted(pairs)


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
    options: V2Options = V2Options(),
    progress=None,
) -> V2Result:
    """
    Estimate every frame's orientation from images and a gyroscope.

    Args:
        frames: BGR frames in capture order.
        frame_times: Frame timestamps (s), on the same clock as the gyro.
        gyro: Gyroscope log covering the frames.
        calib_init: Starting intrinsics (fx = fy assumed). The focal length is refined.
        R0: World -> camera orientation of the first frame (defines the output frame).
        options: Pipeline settings.
        progress: Optional callable(stage: str, done: int, total: int).

    Returns:
        V2Result with the gyro-only, chained and bundle-adjusted orientations.
    """
    n = len(frames)
    K = calib_init.K
    hfov_deg = float(np.degrees(2 * np.arctan(frames[0].shape[1] / 2 / calib_init.fx)))
    rng = np.random.default_rng(options.seed)

    # 1. Gyro prior
    prior = integrate_gyro(gyro, frame_times, R0)

    # 2. Pairs to try
    candidates = select_pairs(prior, hfov_deg, options)
    logger.info(f"v2: {n} frames, {len(candidates)} candidate pairs "
                f"(HFOV {hfov_deg:.1f}° from initial focal)")

    # 3. Features and guided matching
    feats: List[FrameFeatures] = []
    for k, img in enumerate(frames):
        feats.append(detect_features(img, options.orb_nfeatures))
        if progress:
            progress("features", k + 1, n)

    pairs: List[PairMatch] = []
    for k, (i, j) in enumerate(candidates):
        dt = abs(frame_times[j] - frame_times[i])
        drift_deg = options.gyro_drift_deg_per_s * dt
        radius = options.base_search_radius_px + calib_init.fx * np.radians(drift_deg)
        pm = match_pair(
            i, j, feats[i], feats[j], K, relative_rotation(prior[i], prior[j]),
            search_radius_px=radius,
            ransac_threshold_px=options.ransac_threshold_px,
            max_prior_disagreement_deg=2.0 + drift_deg,
            min_inliers=options.min_inliers,
            rng=rng,
        )
        pairs.append(pm)
        if progress:
            progress("matching", k + 1, len(candidates))

    ok = [p for p in pairs if p.success]
    loop = [p for p in ok if p.j - p.i > options.max_pair_gap_local]
    logger.info(f"v2: {len(ok)}/{len(pairs)} pairs matched, {len(loop)} beyond the local window")

    # 4. Pairwise chaining (what you get without global refinement)
    chained = chain_adjacent(R0, pairs, prior, n)

    # 5. Bundle adjustment, initialised from the gyro prior
    bundle = bundle_adjust(
        prior, ok, calib_init.fx, calib_init.cx, calib_init.cy,
        optimize_focal=options.optimize_focal,
        loss_scale_px=options.ba_loss_scale_px,
        max_per_pair=options.ba_max_per_pair,
        seed=options.seed,
    )

    diagnostics = {
        "n_frames": n,
        "n_candidate_pairs": len(candidates),
        "n_matched_pairs": len(ok),
        "n_long_range_pairs": len(loop),
        "mean_inliers": float(np.mean([p.inliers for p in ok])) if ok else 0.0,
        "ba_rms_before_px": bundle.rms_before_px,
        "ba_rms_after_px": bundle.rms_after_px,
        "ba_observations": bundle.n_observations,
    }
    return V2Result(
        imu_rotations=prior,
        chained_rotations=chained,
        ba_rotations=bundle.rotations,
        focal_initial=float(calib_init.fx),
        focal_ba=bundle.focal,
        pairs=pairs,
        bundle=bundle,
        diagnostics=diagnostics,
    )


def per_frame_errors(estimated: List[np.ndarray], truth: List[np.ndarray]) -> np.ndarray:
    """Rotation error (degrees) of each frame against ground truth."""
    return np.array([rotation_error_degrees(E, T) for E, T in zip(estimated, truth)])


def evaluate_against_truth(result: V2Result, truth: List[np.ndarray], true_focal: float,
                           cx: float, cy: float, extra: Optional[Dict[str, List[np.ndarray]]] = None) -> Dict:
    """
    Per-frame rotation error and reprojection RMS for each stage.

    Reprojection RMS is measured on the same verified correspondences for every method,
    each with the focal length it would actually use (initial for gyro and chaining,
    refined for bundle adjustment).
    """
    ok = [p for p in result.pairs if p.success]
    methods = {
        "imu_only": (result.imu_rotations, result.focal_initial),
        "pairwise_chain": (result.chained_rotations, result.focal_initial),
        "bundle_adjusted": (result.ba_rotations, result.focal_ba),
    }
    for name, rots in (extra or {}).items():
        methods[name] = (rots, result.focal_initial)

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
