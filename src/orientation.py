"""
Global orientation for a rotating camera, with or without a gyroscope.

Both modes run the same stages once they have a first guess at every frame's
orientation (the prior):

    prior -> pair selection -> guided matching -> rotation averaging
          -> pair re-selection from the averaged orientations -> bundle adjustment

Only the prior differs:

- With a gyroscope it is the integrated gyro (imu.py). Its error grows with elapsed time.
- Without one it is the v1 homography chain (rotation.py). Its error grows with the
  number of links between two frames.

The prior is never part of the answer. It says which frames should overlap, where a
keypoint should land (so matching can search a small window), and where the solver
starts. Every output orientation comes from image correspondences — including the
loop-closure pairs between the end and the start of the sweep, which a chain never
sees.

Rotation averaging comes before bundle adjustment because the prior can be far off
across the loop: a chain that over-counts 4% of a 330° sweep is 13° wrong at the join,
which is hundreds of pixels. Pixel residuals that large are what a robust loss is
built to ignore, so bundle adjustment alone would keep the drift. Averaging works on
pair rotations instead, where 13° is just a large residual to be spread around the
loop, and hands bundle adjustment a start that is already within a pixel or two.
"""

import dataclasses
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .bundle_adjust import BundleResult, bundle_adjust
from .guided_match import (FrameFeatures, PairMatch, detect_features, kabsch_rotation, match_pair,
                           pixels_to_bearings)
from .imu import (GyroLog, estimate_axis_alignment, estimate_time_offset, gyro_interval_rotvecs,
                  gyro_to_camera, relative_rotation, so3_log)

logger = logging.getLogger(__name__)


@dataclass
class OrientationOptions:
    """Settings shared by the gyro and no-gyro paths."""
    orb_nfeatures: int = 3000
    # Matching
    base_search_radius_deg: float = 2.0     # window around the prior's prediction, before drift
    base_disagreement_deg: float = 2.0      # allowed prior/vision disagreement, before drift
    ransac_threshold_px: float = 2.0
    min_inliers: int = 30
    # Pair selection
    overlap_angle_fraction: float = 0.5     # pairs whose optical axes are within this x HFOV
    max_pair_gap_local: int = 3             # always try (i, i+1) .. (i, i+gap)
    long_pairs_per_frame: int = 1           # extra non-loop pairs per frame, chosen ~fraction x HFOV apart
    loop_pairs_per_frame: int = 2           # loop-closure pairs per frame, nearest first
    refine_rounds: int = 1                  # re-select pairs from the averaged orientations
    # How fast each prior drifts (sets search windows and sanity bounds)
    gyro_drift_deg_per_s: float = 0.5
    chain_drift_deg_per_link: float = 0.1
    # Rotation averaging
    averaging_robust_deg: float = 2.0
    averaging_reject_deg: float = 3.0
    # Bundle adjustment
    optimize_focal: bool = True
    ba_loss_scale_px: float = 1.5
    ba_max_per_pair: int = 150
    ba_max_observations: int = 60_000       # caps problem size on long videos
    ba_max_nfev: int = 200
    seed: int = 0
    workers: int = 0                        # threads for features and matching (0 = auto)


@dataclass
class OrientationPrior:
    """A first guess at every frame's orientation, and how quickly it goes stale."""
    rotations: List[np.ndarray]             # world -> camera
    position: np.ndarray                    # where each frame sits along what the drift grows with
    drift_deg_per_unit: float
    source: str

    def drift_deg(self, i: int, j: int) -> float:
        """Plausible prior error between frames i and j, in degrees."""
        return float(self.drift_deg_per_unit * abs(self.position[j] - self.position[i]))


def gyro_prior(rotations: Sequence[np.ndarray], frame_times: np.ndarray,
               drift_deg_per_s: float) -> OrientationPrior:
    """Prior from integrated gyro orientations: error grows with elapsed time."""
    return OrientationPrior(list(rotations), np.asarray(frame_times, dtype=np.float64),
                            drift_deg_per_s, "gyroscope")


def chain_prior(
    rotations: Sequence[np.ndarray],
    drift_deg_per_link: float,
    link_doubt_deg: Optional[Sequence[float]] = None,
) -> OrientationPrior:
    """
    Prior from chained pairwise rotations: error grows with the number of links.

    Args:
        rotations: Chained orientations.
        drift_deg_per_link: Error each well-measured link may add.
        link_doubt_deg: Optional extra error per link (len(rotations) - 1 values), for
            links the chain did not actually measure: a pair that failed and fell back
            to no rotation can be off by a whole step. Any frame pair spanning such a
            link then gets a correspondingly wide search and a loose sanity check.
    """
    steps = np.ones(len(rotations) - 1)
    if link_doubt_deg is not None and drift_deg_per_link > 0:
        steps = steps + np.asarray(link_doubt_deg, dtype=np.float64) / drift_deg_per_link
    position = np.concatenate([[0.0], np.cumsum(steps)])
    return OrientationPrior(list(rotations), position, drift_deg_per_link, "homography chain")


@dataclass
class AveragingResult:
    """Output of average_rotations."""
    rotations: List[np.ndarray]
    used: List[PairMatch]
    rejected: List[PairMatch]
    residual_deg: np.ndarray                # per used pair, after the solve
    bridges: List[Tuple[int, int]] = field(default_factory=list)  # gaps no pair spans


@dataclass
class OrientationResult:
    """Every stage's orientations plus diagnostics."""
    rotations: List[np.ndarray]             # final, bundle adjusted
    focal: float                            # bundle-adjusted focal length (px)
    focal_initial: float
    prior: OrientationPrior
    averaged: List[np.ndarray]
    pairs: List[PairMatch]                  # every pair tried
    used_pairs: List[PairMatch]             # pairs that went into bundle adjustment
    bundle: BundleResult
    diagnostics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------

def optical_axes(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Each camera's optical axis in world coordinates (row 2 of world -> camera R)."""
    return np.array([R[2, :] for R in rotations])


def unwrapped_heading_deg(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Heading of each optical axis about the world vertical, unwrapped so a sweep keeps counting."""
    axes = optical_axes(rotations)
    return np.degrees(np.unwrap(np.arctan2(axes[:, 0], axes[:, 2])))


def is_loop_pair(heading_deg: np.ndarray, i: int, j: int) -> bool:
    """True when the sweep went at least half way round between frames i and j."""
    return bool(abs(heading_deg[j] - heading_deg[i]) >= 180.0)


def _workers(requested: int, n: int) -> int:
    if requested <= 0:
        requested = max(1, min(8, (os.cpu_count() or 2) - 1))
    return max(1, min(requested, max(n, 1)))


# ---------------------------------------------------------------------------------------
# Features and matching
# ---------------------------------------------------------------------------------------

def detect_all_features(frames: Sequence[np.ndarray], nfeatures: int = 3000,
                        workers: int = 0) -> List[FrameFeatures]:
    """
    ORB features for every frame, detected once and reused by every pair.

    Frames are indexed inside the workers, so a lazily loaded sequence (io_utils.
    LazyImages) is read one frame per worker rather than all at once.
    """
    with ThreadPoolExecutor(max_workers=_workers(workers, len(frames))) as pool:
        return list(pool.map(lambda k: detect_features(frames[k], nfeatures), range(len(frames))))


def select_pairs(prior: OrientationPrior, hfov_deg: float, options: OrientationOptions,
                 exclude: Optional[set] = None) -> List[Tuple[int, int]]:
    """
    Frame pairs worth matching, given the prior.

    - Local: (i, i+1) .. (i, i+max_pair_gap_local), always.
    - Long: per frame, the pair beyond the local window whose prior separation is closest
      to overlap_angle_fraction x HFOV. These tie frames together over tens of degrees,
      which is what pins down the focal length and stops slow drift between loop ends.
    - Loop: per frame, the nearest frames the sweep reaches again after going at least
      half way round. These are the pairs chaining never sees.

    A pair qualifies when the prior puts the two optical axes within the overlap angle,
    widened by how far the prior could have drifted between the two frames.
    """
    n = len(prior.rotations)
    exclude = exclude or set()
    pairs = {(i, i + g) for g in range(1, options.max_pair_gap_local + 1) for i in range(n - g)}

    axes = optical_axes(prior.rotations)
    heading = unwrapped_heading_deg(prior.rotations)
    target = options.overlap_angle_fraction * hfov_deg
    for i in range(n):
        js = np.arange(i + options.max_pair_gap_local + 1, n)
        if len(js) == 0:
            continue
        angle = np.degrees(np.arccos(np.clip(axes[js] @ axes[i], -1.0, 1.0)))
        drift = prior.drift_deg_per_unit * np.abs(prior.position[js] - prior.position[i])
        overlaps = angle <= target + drift
        loop = overlaps & (np.abs(heading[js] - heading[i]) >= 180.0)
        near = overlaps & ~loop

        if options.long_pairs_per_frame > 0 and near.any():
            order = np.argsort(np.abs(angle[near] - target))
            pairs.update((i, int(j)) for j in js[near][order[:options.long_pairs_per_frame]])
        if options.loop_pairs_per_frame > 0 and loop.any():
            order = np.argsort(angle[loop])
            pairs.update((i, int(j)) for j in js[loop][order[:options.loop_pairs_per_frame]])

    return sorted(p for p in pairs if p not in exclude)


def match_candidates(
    features: Sequence[FrameFeatures],
    K: np.ndarray,
    prior: OrientationPrior,
    candidates: Sequence[Tuple[int, int]],
    options: OrientationOptions,
    label: str = "matching",
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> List[PairMatch]:
    """Guided matching plus rotation-only RANSAC for every candidate pair."""
    total = len(candidates)
    done = [0]
    lock = threading.Lock()
    step = max(1, total // 10)

    def one(ij):
        i, j = ij
        drift = prior.drift_deg(i, j)
        kwargs = dict(
            search_radius_px=K[0, 0] * np.radians(options.base_search_radius_deg + drift),
            ransac_threshold_px=options.ransac_threshold_px,
            max_prior_disagreement_deg=options.base_disagreement_deg + drift,
            min_inliers=options.min_inliers,
        )
        R_prior = relative_rotation(prior.rotations[i], prior.rotations[j])
        result = match_pair(i, j, features[i], features[j], K, R_prior,
                            rng=np.random.default_rng([options.seed, i, j]), **kwargs)
        if not result.success:
            # The prior can be locally wrong (a chain step that fell back to no rotation,
            # a gyro glitch), which puts the true matches outside the window. Search the
            # whole frame instead; the prior still vets the fitted rotation.
            result = match_pair(i, j, features[i], features[j], K, R_prior, use_window=False,
                                rng=np.random.default_rng([options.seed, i, j, 1]), **kwargs)
        with lock:
            done[0] += 1
            if progress:
                progress(label, done[0], total)
            elif done[0] % step == 0 or done[0] == total:
                logger.info(f"  {label}: {done[0]}/{total} pairs")
        return result

    if total == 0:
        return []
    with ThreadPoolExecutor(max_workers=_workers(options.workers, total)) as pool:
        return list(pool.map(one, candidates))


def adjacent_vision_rotations(
    features: Sequence[FrameFeatures],
    K: np.ndarray,
    options: OrientationOptions,
) -> List[Optional[np.ndarray]]:
    """
    Unguided rotation-only estimate for every adjacent pair, or None where it failed.

    Used to calibrate a gyroscope against the video (time offset and axes) before the
    gyro can be trusted as a prior.
    """
    n = len(features)

    def one(i):
        p = match_pair(i, i + 1, features[i], features[i + 1], K, None,
                       ransac_threshold_px=options.ransac_threshold_px,
                       min_inliers=options.min_inliers,
                       rng=np.random.default_rng([options.seed, i, i + 1]))
        return p.R_ij

    with ThreadPoolExecutor(max_workers=_workers(options.workers, n - 1)) as pool:
        return list(pool.map(one, range(n - 1)))


# ---------------------------------------------------------------------------------------
# Rotation averaging
# ---------------------------------------------------------------------------------------

def connected_components(n: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Component label of every frame, for the graph whose edges are the given pairs."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(k) for k in range(n)])


def bridging_gaps(n: int, pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Adjacent frames (k, k+1) to tie together so that every frame is connected.

    Walking the sequence in order, a frame that no matched pair links (directly or
    through other frames) to everything before it gets tied to its predecessor.
    """
    labels = connected_components(n, pairs)
    bridges = []
    joined = set(pairs)
    for k in range(n - 1):
        if labels[k] != labels[k + 1]:
            bridges.append((k, k + 1))
            joined.add((k, k + 1))
            labels = connected_components(n, list(joined))
    return bridges


def average_rotations(
    initial: Sequence[np.ndarray],
    pairs: Sequence[PairMatch],
    robust_deg: float = 2.0,
    reject_deg: float = 3.0,
    max_rounds: int = 12,
    tether_weight: float = 0.01,
    loss: str = "soft_l1",
) -> AveragingResult:
    """
    Orientations most consistent with every pairwise rotation at once.

    Minimises, over all matched pairs, the angle between the measured relative rotation
    R_ij and the one implied by the solution, R_j R_iᵀ, under a soft-L1 loss. Frame 0
    is held fixed.

    Pairs that still disagree by more than reject_deg after a solve (a wrong match, or
    repeated texture that fooled RANSAC) are dropped and the solve repeated, but
    cautiously: a robust loss tends to pile a conflict onto a few pairs, which are not
    necessarily the wrong ones, so each round drops only the worst offenders, and never
    a pair that is the only link between two stretches of frames.

    Where no pair links a stretch of frames to the rest, the initial guess holds them
    together (at full weight); a very weak tether to the initial adjacent rotations keeps
    the problem well posed everywhere else.
    """
    n = len(initial)
    R_init = np.asarray(initial, dtype=np.float64)
    R_fixed = R_init[0]
    ok = [p for p in pairs if p.success]
    if n < 2 or not ok:
        return AveragingResult([R.copy() for R in R_init], [], [], np.zeros(0))

    I_all = np.array([p.i for p in ok])
    J_all = np.array([p.j for p in ok])
    Rij_all = np.array([p.R_ij for p in ok])
    w_all = np.sqrt(np.clip(np.array([p.inliers for p in ok], float), 0, 100) / 100.0)

    # Tether: adjacent relative rotations of the initial guess, nearly weightless
    T_i = np.arange(n - 1)
    T_j = T_i + 1
    T_R = np.einsum("nab,ncb->nac", R_init[1:], R_init[:-1])

    n_par = 3 * (n - 1)

    def unpack(x):
        rots = np.empty((n, 3, 3))
        rots[0] = R_fixed
        rots[1:] = Rotation.from_rotvec(x.reshape(-1, 3)).as_matrix() @ R_init[1:]
        return rots

    def angle_errors(rots, I, J, Rij):
        E = np.transpose(Rij, (0, 2, 1)) @ rots[J] @ np.transpose(rots[I], (0, 2, 1))
        return Rotation.from_matrix(E).as_rotvec()

    def solve(active, x0):
        I, J, Rij, w = I_all[active], J_all[active], Rij_all[active], w_all[active]
        bridges = bridging_gaps(n, list(zip(I.tolist(), J.tolist())))
        tether_w = np.full(n - 1, tether_weight)
        for k, _ in bridges:
            tether_w[k] = 1.0
        I_c = np.concatenate([I, T_i])
        J_c = np.concatenate([J, T_j])
        R_c = np.concatenate([Rij, T_R])
        w_c = np.concatenate([w, tether_w])

        def fun(params):
            return (angle_errors(unpack(params), I_c, J_c, R_c) * w_c[:, None]).ravel()

        rows, cols = [], []
        m = len(I_c)
        for cams in (I_c, J_c):
            moving = cams > 0
            for k in range(3):
                for a in range(3):
                    rows.append(3 * np.nonzero(moving)[0] + a)
                    cols.append(3 * (cams[moving] - 1) + k)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        sparsity = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                              shape=(3 * m, n_par)).tocsr()
        sparsity.data[:] = 1
        sol = least_squares(fun, x0, jac_sparsity=sparsity, method="trf", loss=loss,
                            f_scale=np.radians(robust_deg), x_scale=1.0, max_nfev=200)
        return sol.x, bridges

    active = np.ones(len(ok), bool)
    x = np.zeros(n_par)
    for _ in range(max_rounds):
        x, bridges = solve(active, x)
        err_deg = np.degrees(np.linalg.norm(angle_errors(unpack(x), I_all, J_all, Rij_all), axis=1))
        over = active & (err_deg > reject_deg)
        if not over.any():
            break
        # Only the worst group this round: their removal changes the solve, and pairs
        # that only looked bad because of them come back within tolerance
        cut = max(reject_deg, 0.5 * err_deg[over].max())
        dropped = False
        for k in np.argsort(-err_deg):
            if not (over[k] and err_deg[k] >= cut):
                continue
            trial = active.copy()
            trial[k] = False
            if len(bridging_gaps(n, list(zip(I_all[trial].tolist(), J_all[trial].tolist())))) > len(bridges):
                continue  # the only link across a stretch: keep it
            active = trial
            dropped = True
        if not dropped:
            break

    rots = unpack(x)
    err_deg = np.degrees(np.linalg.norm(angle_errors(rots, I_all, J_all, Rij_all), axis=1))
    used = [p for p, a in zip(ok, active) if a]
    rejected = [p for p, a in zip(ok, active) if not a]
    return AveragingResult([R for R in rots], used, rejected, err_deg[active], bridges)


# ---------------------------------------------------------------------------------------
# Focal length
# ---------------------------------------------------------------------------------------

# A focal 10% off must make the pairs disagree at least this much more for the search
# to count as sharp. On phone video that takes a loop closure; a narrow field of view,
# rolling shutter and stabilisation warps leave an open sweep's minimum shallow
FOCAL_SHARPNESS_FOR_LOOP = 1.5


def refit_pair_rotations(pairs: Sequence[PairMatch], K: np.ndarray) -> List[PairMatch]:
    """Every matched pair's rotation fitted again to its inliers, with intrinsics K."""
    out = []
    for p in pairs:
        if p.success and len(p.pts_i) >= 2:
            R = kabsch_rotation(pixels_to_bearings(p.pts_i, K), pixels_to_bearings(p.pts_j, K))
            out.append(dataclasses.replace(p, R_ij=R))
        else:
            out.append(p)
    return out


def _with_focal(K: np.ndarray, f: float) -> np.ndarray:
    K = K.copy()
    K[0, 0] = K[1, 1] = f
    return K


def estimate_focal(
    pairs: Sequence[PairMatch],
    K: np.ndarray,
    initial: Sequence[np.ndarray],
    bounds: Tuple[float, float] = (0.7, 1.4),
    grid: int = 9,
    refine_steps: int = 10,
) -> Tuple[float, Dict]:
    """
    The focal length at which the pairwise rotations agree best.

    A rotation fitted with the wrong focal length is scaled by roughly true/assumed.
    When the sweep comes back round, that breaks the loop: the chain of small rotations
    no longer meets the loop-closure pairs, by about (scale - 1) x 360°. For each
    candidate the pairs are refitted and averaged by plain least squares, which spreads
    such a conflict over every pair instead of hiding it in a few, and scored by the
    mean disagreement left (the worst 5% of pairs left out, for stray wrong matches).
    A coarse grid over [bounds] x the starting focal finds the basin, golden-section
    search the minimum.

    With a loop the minimum is sharp. Without one, a uniform scale error barely shows,
    the curve is shallow, and the result is only as good as the perspective effects in
    the pairs allow; the report says how sharp the minimum was.

    This runs before rotation averaging rejects anything: with a focal length 10% off,
    good long-range pairs disagree with the short ones by several degrees and would
    otherwise be thrown out as outliers. Bundle adjustment refines the result.

    Returns:
        (focal length in pixels, report with the cost at each candidate tried).
    """
    f0 = float(K[0, 0])
    ok = [p for p in pairs if p.success]
    tried: Dict[float, float] = {}

    def cost(log_ratio: float) -> float:
        key = round(float(log_ratio), 6)
        if key in tried:
            return tried[key]
        f = f0 * np.exp(log_ratio)
        refit = refit_pair_rotations(ok, _with_focal(K, f))
        avg = average_rotations(initial, refit, reject_deg=np.inf, max_rounds=1, loss="linear")
        r = np.sort(avg.residual_deg)
        tried[key] = float(np.mean(r[:max(1, int(0.95 * len(r)))]))
        return tried[key]

    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    xs = np.linspace(lo, hi, grid)
    costs = [cost(x) for x in xs]
    k = int(np.argmin(costs))
    a, b = xs[max(k - 1, 0)], xs[min(k + 1, grid - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(refine_steps):
        if cost(c) < cost(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    best = min(tried, key=tried.get)
    f = f0 * float(np.exp(best))
    report = {
        "focal_start_px": f0,
        "focal_px": f,
        "cost_deg_at_start": cost(0.0),
        "cost_deg_at_best": tried[best],
        # How much worse 10% either side is: large with a loop, near 1 without one
        "sharpness": float(min(cost(best + np.log(1.1)), cost(best - np.log(1.1))) / tried[best]),
        "tried": {f"{f0 * np.exp(x):.1f}": v for x, v in sorted(tried.items())},
    }
    return f, report


# ---------------------------------------------------------------------------------------
# The whole stage
# ---------------------------------------------------------------------------------------

def estimate_orientation(
    features: Sequence[FrameFeatures],
    K: np.ndarray,
    image_size: Tuple[int, int],
    prior: OrientationPrior,
    options: OrientationOptions = OrientationOptions(),
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> OrientationResult:
    """
    Orientations of every frame from a prior and image correspondences.

    Args:
        features: Per-frame features (detect_all_features), in capture order.
        K: Intrinsics at the resolution the features were detected at (fx = fy assumed).
        image_size: (width, height) of those images.
        prior: Starting orientations and their drift model (gyro_prior / chain_prior).
        options: Settings.
        progress: Optional callable(stage, done, total).

    Returns:
        OrientationResult. Frame 0 keeps the prior's orientation; it is the reference
        every other frame is expressed against.
    """
    t_start = time.time()
    n = len(features)
    width, _ = image_size
    focal0 = float(K[0, 0])
    hfov_deg = float(np.degrees(2 * np.arctan(width / 2 / focal0)))

    # Round 1: pairs and matches from the prior
    candidates = select_pairs(prior, hfov_deg, options)
    logger.info(f"Orientation ({prior.source} prior): {n} frames, {len(candidates)} candidate pairs, "
                f"HFOV {hfov_deg:.1f}°")
    pairs = match_candidates(features, K, prior, candidates, options, "matching", progress)
    _log_matches(pairs, prior.rotations, "prior")

    focal_report = None
    refine_focal_in_ba = options.optimize_focal
    if options.optimize_focal:
        f_search, focal_report = estimate_focal(pairs, K, prior.rotations)
        sharp = focal_report["sharpness"] >= FOCAL_SHARPNESS_FOR_LOOP
        if sharp:
            focal_report["decision"] = "searched (sharp minimum)"
            logger.info(f"Focal length: {focal0:.1f}px -> {f_search:.1f}px, where the pairs agree best "
                        f"(mean disagreement {focal_report['cost_deg_at_start']:.3f}° -> "
                        f"{focal_report['cost_deg_at_best']:.3f}°; 10% off costs x{focal_report['sharpness']:.1f})")
        else:
            # Without a loop, a uniform scale error barely shows in the pairs: the shallow
            # minimum moves with the starting point, and bundle adjustment's own estimate
            # drifts with rolling shutter and stabilisation warps. Neither can be trusted,
            # so keep the configured focal length, which a calibration can make right.
            refine_focal_in_ba = False
            focal_report["pairs_prefer_px"] = f_search
            focal_report["decision"] = "kept the configured focal (no loop closure to pin it)"
            f_search = focal0
            logger.warning(f"Focal length kept at {focal0:.1f}px (the pairs lean towards "
                           f"{focal_report['pairs_prefer_px']:.0f}px, too weakly to trust). Without a loop "
                           f"closure the images do not pin the focal length, and with it the total sweep, "
                           f"so the configured one is used as is. It should come from a calibration, such "
                           f"as intrinsics_refined.json from a capture that comes back round.")
        focal_report["focal_used_px"] = f_search
        K = _with_focal(K, f_search)
        pairs = refit_pair_rotations(pairs, K)
        hfov_deg = float(np.degrees(2 * np.arctan(width / 2 / K[0, 0])))

    averaged = average_rotations(prior.rotations, pairs, options.averaging_robust_deg,
                                 options.averaging_reject_deg)
    _log_averaging(averaged)

    # Later rounds: the averaged orientations are a far better prior (drift spread round
    # the loop), so pairs it predicts to overlap are worth trying even if the first prior
    # placed them too far apart.
    for _ in range(options.refine_rounds):
        refined = OrientationPrior(averaged.rotations, prior.position, 0.0, "averaged")
        tried = {(p.i, p.j) for p in pairs}
        extra = select_pairs(refined, hfov_deg, options, exclude=tried)
        if not extra:
            break
        logger.info(f"Orientation: {len(extra)} more pairs predicted by the averaged orientations")
        new_pairs = match_candidates(features, K, refined, extra, options, "re-matching", progress)
        pairs = pairs + new_pairs
        _log_matches(new_pairs, averaged.rotations, "re-selection")
        averaged = average_rotations(averaged.rotations, pairs, options.averaging_robust_deg,
                                     options.averaging_reject_deg)
        _log_averaging(averaged)

    if not averaged.used:
        raise ValueError("No frame pair could be matched; cannot estimate orientations")

    per_pair = int(np.clip(options.ba_max_observations // max(len(averaged.used), 1),
                           15, options.ba_max_per_pair))
    bundle = bundle_adjust(
        averaged.rotations, averaged.used, float(K[0, 0]), float(K[0, 2]), float(K[1, 2]),
        optimize_focal=refine_focal_in_ba,
        loss_scale_px=options.ba_loss_scale_px,
        max_per_pair=per_pair,
        max_nfev=options.ba_max_nfev,
        seed=options.seed,
    )

    final = bundle.rotations
    if averaged.bridges:
        final = reanchor_components(final, averaged.rotations, [(p.i, p.j) for p in averaged.used])
        logger.warning(f"Orientation: no matched pair spans {len(averaged.bridges)} gap(s) between frames "
                       f"{', '.join(f'{a}-{b}' for a, b in averaged.bridges[:8])}"
                       f"{' ...' if len(averaged.bridges) > 8 else ''}; the prior holds them together there")
    heading_final = unwrapped_heading_deg(final)
    loop_used = [p for p in averaged.used if is_loop_pair(heading_final, p.i, p.j)]
    diagnostics = {
        "prior": prior.source,
        "n_frames": n,
        "n_pairs_tried": len(pairs),
        "n_pairs_matched": sum(p.success for p in pairs),
        "n_pairs_used": len(averaged.used),
        "n_pairs_rejected_by_averaging": len(averaged.rejected),
        "bridged_gaps": [list(b) for b in averaged.bridges],
        "n_loop_pairs": len(loop_used),
        "n_long_pairs": sum(1 for p in averaged.used if p.j - p.i > options.max_pair_gap_local
                            and not is_loop_pair(heading_final, p.i, p.j)),
        "mean_inliers": float(np.mean([p.inliers for p in averaged.used])),
        "averaging_residual_deg_median": float(np.median(averaged.residual_deg)),
        "averaging_residual_deg_max": float(np.max(averaged.residual_deg)),
        "ba_observations": bundle.n_observations,
        "ba_rms_before_px": bundle.rms_before_px,
        "ba_rms_after_px": bundle.rms_after_px,
        "focal_initial_px": focal0,
        "focal_search_px": float(K[0, 0]),
        "focal_px": bundle.focal,
        "focal_refined_by_ba": refine_focal_in_ba,
        "hfov_deg": float(np.degrees(2 * np.arctan(width / 2 / bundle.focal))),
        "sweep_prior_deg": float(abs(unwrapped_heading_deg(prior.rotations)[-1]
                                     - unwrapped_heading_deg(prior.rotations)[0])),
        "sweep_deg": float(abs(heading_final[-1] - heading_final[0])),
        "seconds": time.time() - t_start,
    }
    if focal_report:
        diagnostics["focal_search"] = focal_report
    logger.info(f"Orientation: sweep {diagnostics['sweep_prior_deg']:.1f}° ({prior.source}) -> "
                f"{diagnostics['sweep_deg']:.1f}° (bundle adjusted), {len(loop_used)} loop-closure pairs, "
                f"focal {focal0:.1f} -> {bundle.focal:.1f}px")

    return OrientationResult(
        rotations=final,
        focal=bundle.focal,
        focal_initial=focal0,
        prior=prior,
        averaged=averaged.rotations,
        pairs=pairs,
        used_pairs=averaged.used,
        bundle=bundle,
        diagnostics=diagnostics,
    )


def reanchor_components(
    rotations: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
    pairs: Sequence[Tuple[int, int]],
) -> List[np.ndarray]:
    """
    Put each separately connected stretch of frames back where the reference had it.

    Bundle adjustment only sees pixels, so a stretch no pair links to frame 0 can turn
    freely as a whole. Its shape is still right; this restores its placement from the
    reference (the averaged orientations, which hold the bridges), as the rotation that
    best maps the stretch onto its reference poses.
    """
    rotations = [np.asarray(R).copy() for R in rotations]
    labels = connected_components(len(rotations), pairs)
    for label in np.unique(labels):
        if label == labels[0]:
            continue
        members = np.nonzero(labels == label)[0]
        M = sum(reference[k] @ rotations[k].T for k in members)
        U, _, Vt = np.linalg.svd(M)
        C = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        for k in members:
            rotations[k] = C @ rotations[k]
    return rotations


def _log_matches(pairs: Sequence[PairMatch], rotations: Sequence[np.ndarray], stage: str) -> None:
    heading = unwrapped_heading_deg(rotations)
    ok = [p for p in pairs if p.success]
    loops = sum(1 for p in ok if is_loop_pair(heading, p.i, p.j))
    logger.info(f"Orientation ({stage}): {len(ok)}/{len(pairs)} pairs matched, "
                f"{loops} of them close the loop")


def _log_averaging(result: AveragingResult) -> None:
    if len(result.residual_deg):
        logger.info(f"Rotation averaging: {len(result.used)} pairs, residual median "
                    f"{np.median(result.residual_deg):.3f}°, max {np.max(result.residual_deg):.3f}°, "
                    f"{len(result.rejected)} rejected as inconsistent")


# ---------------------------------------------------------------------------------------
# Priors from data
# ---------------------------------------------------------------------------------------

# How far off a chain link can be when the chain did not measure it
FAILED_LINK_DOUBT_DEG = 90.0         # fell back to no rotation at all
INTERPOLATED_LINK_DOUBT_DEG = 10.0   # borrowed from neighbouring pairs


def homography_chain_prior(
    frames: Sequence[np.ndarray],
    K: np.ndarray,
    matching_config,
    drift_deg_per_link: float,
    R0: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> OrientationPrior:
    """
    The v1 orientation path, unsmoothed, as a prior: homography per adjacent pair,
    R = K⁻¹HK, chained.

    Smoothing is left out on purpose. It is a cosmetic fix for chain jitter that also
    bends real motion; bundle adjustment removes the jitter at its source. Links the
    chain could not measure (failed or interpolated pairs) are marked as doubtful, so
    the matching stage does not trust them to say where to look.
    """
    from .features import match_sequential_pairs
    from .rotation import chain_rotations, compute_relative_rotations, log_rotation_summary

    results = match_sequential_pairs(frames, matching_config, scale=scale)
    relative, diags = compute_relative_rotations(results, K, 1.0)
    chained = chain_rotations(relative, apply_smoothing=False)
    log_rotation_summary(chained, diags)
    if R0 is not None:
        chained = [R @ R0 for R in chained]
    doubt = [FAILED_LINK_DOUBT_DEG if d["status"] == "failed"
             else INTERPOLATED_LINK_DOUBT_DEG if d["status"] == "interpolated" else 0.0
             for d in diags]
    n_doubt = sum(1 for x in doubt if x > 0)
    if n_doubt:
        logger.info(f"Chain prior: {n_doubt} of {len(doubt)} links were not measured directly; "
                    f"pairs across them are searched without a window")
    return chain_prior(chained, drift_deg_per_link, doubt)


def calibrate_gyro(
    features: Sequence[FrameFeatures],
    K: np.ndarray,
    frame_times: np.ndarray,
    gyro: GyroLog,
    options: OrientationOptions,
    time_offset_s: Optional[float] = None,
    R_cam_imu: Optional[np.ndarray] = None,
) -> Tuple[GyroLog, Dict]:
    """
    Put a raw gyro log on the video clock and in camera axes.

    Whatever is not given is estimated from the video: adjacent frames are matched
    without any prior, and the rotation each interval shows is compared with the gyro.
    The clock offset comes first (it only needs how far the camera turned, which does
    not depend on axes); the axes follow from the same intervals once the clocks agree.

    Returns:
        (gyro log in camera axes on the video clock, calibration report).
    """
    report: Dict = {}
    vision = None
    if time_offset_s is None or R_cam_imu is None:
        logger.info("Gyro calibration: matching adjacent frames without a prior...")
        rel = adjacent_vision_rotations(features, K, options)
        # Body increment over each interval, in camera axes: Δ = R_ijᵀ
        vision = np.array([so3_log(R.T) if R is not None else np.full(3, np.nan) for R in rel])
        report["vision_intervals"] = int(np.all(np.isfinite(vision), axis=1).sum())

    if time_offset_s is None:
        angles = np.linalg.norm(vision, axis=1)
        time_offset_s, corr = estimate_time_offset(gyro, frame_times, angles)
        report["time_offset_s"] = time_offset_s
        report["time_offset_correlation"] = corr
        level = logging.INFO if corr >= 0.8 else logging.WARNING
        logger.log(level, f"Gyro calibration: clock offset {time_offset_s:+.4f}s "
                          f"(correlation {corr:.3f}{'' if corr >= 0.8 else ', weak: check the log matches this video'})")
    else:
        report["time_offset_s"] = float(time_offset_s)

    if R_cam_imu is None:
        g = gyro_interval_rotvecs(gyro_to_camera(gyro, time_offset_s, np.eye(3)), frame_times)
        R_cam_imu, snapped, err = estimate_axis_alignment(g, vision)
        report["axes_estimated"] = True
        report["axes_snapped"] = snapped
        report["axes_fit_deg_from_nearest_permutation"] = err
        logger.info(f"Gyro calibration: gyro -> camera axes {R_cam_imu.round(3).tolist()}"
                    f"{' (snapped to an axis permutation, raw fit ' + f'{err:.1f}° away)' if snapped else ''}")
    report["R_cam_imu"] = np.asarray(R_cam_imu).tolist()
    return gyro_to_camera(gyro, time_offset_s, R_cam_imu), report


def rotation_report(result: OrientationResult) -> Dict:
    """JSON-serialisable summary of an orientation result."""
    d = dict(result.diagnostics)
    d["per_pair"] = [
        {"i": p.i, "j": p.j, "inliers": p.inliers,
         "angle_deg": float(np.degrees(np.linalg.norm(so3_log(p.R_ij))))}
        for p in result.used_pairs
    ]
    return d
