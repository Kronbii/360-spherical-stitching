"""
IMU-guided feature matching and a rotation-only model.

v1 fits a full 8-degree-of-freedom homography to each pair and then forces it into a
rotation. When the camera only turns, the pair has three degrees of freedom, so v2 fits
exactly that: a rotation between the two cameras' bearing vectors, from two-point
samples inside RANSAC.

The gyro prior is used twice:
1. Search window — each keypoint is only allowed to match keypoints near where the
   prior rotation says it should land. That removes most repeated-texture mistakes
   before the ratio test even runs.
2. Sanity check — a fitted rotation that disagrees with the prior by more than the
   prior's plausible drift is rejected.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .features import extract_orb_features
from .imu import so3_log

logger = logging.getLogger(__name__)


@dataclass
class FrameFeatures:
    """ORB keypoint positions (pixels) and descriptors for one frame."""
    points: np.ndarray       # (N, 2) float32
    descriptors: np.ndarray  # (N, 32) uint8


@dataclass
class PairMatch:
    """Inlier correspondences and the fitted rotation for one frame pair."""
    i: int
    j: int
    pts_i: np.ndarray        # (M, 2) inlier pixels in frame i
    pts_j: np.ndarray        # (M, 2) matching pixels in frame j
    R_ij: Optional[np.ndarray]  # rotation with b_j = R_ij @ b_i, or None if failed
    candidates: int          # matches that passed the window and ratio test
    inliers: int

    @property
    def success(self) -> bool:
        return self.R_ij is not None


def detect_features(image: np.ndarray, nfeatures: int = 3000) -> FrameFeatures:
    """ORB features for a frame, using the same detector settings as v1."""
    kps, desc = extract_orb_features(image, nfeatures=nfeatures, use_clahe=False)
    if len(kps) == 0:
        return FrameFeatures(np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8))
    pts = np.float32([kp.pt for kp in kps])
    return FrameFeatures(pts, desc)


def pixels_to_bearings(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unit bearing vectors in the camera frame for pixel coordinates."""
    homog = np.hstack([points.astype(np.float64), np.ones((len(points), 1))])
    rays = homog @ np.linalg.inv(K).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def project_bearings(bearings: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixels for bearings, and a mask of those in front of the camera."""
    front = bearings[:, 2] > 1e-6
    z = np.where(front, bearings[:, 2], 1.0)
    uv = np.stack([K[0, 0] * bearings[:, 0] / z + K[0, 2],
                   K[1, 1] * bearings[:, 1] / z + K[1, 2]], axis=1)
    return uv, front


def kabsch_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares rotation R with b ≈ R @ a for rows of unit vectors (Kabsch/Wahba)."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def batched_kabsch(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Kabsch rotations for many two-point samples at once.

    Sample k pairs bearings (a1[k], b1[k]) and (a2[k], b2[k]); the result R[k] is the
    rotation with b ≈ R[k] @ a for both, exactly as kabsch_rotation would give.
    """
    H = a1[:, :, None] * b1[:, None, :] + a2[:, :, None] * b2[:, None, :]
    U, _, Vt = np.linalg.svd(H)
    V, Ut = np.transpose(Vt, (0, 2, 1)), np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ Ut))
    V[:, :, 2] *= d[:, None]
    return V @ Ut


def ransac_rotation(
    b_i: np.ndarray,
    b_j: np.ndarray,
    threshold_rad: float,
    iterations: int = 400,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Robustly fit b_j = R @ b_i from two-point samples.

    Two non-parallel bearings fix a rotation, so the minimal sample is two
    correspondences — far fewer than a homography's four, which means fewer iterations
    for the same confidence.

    Args:
        b_i, b_j: (N, 3) unit bearings.
        threshold_rad: Inlier threshold on the angle between R @ b_i and b_j.
        iterations: RANSAC iterations.
        rng: Random generator (seeded for reproducibility).

    Returns:
        (R, inlier mask). R is None when fewer than 5 inliers support any sample.
    """
    n = len(b_i)
    if n < 5:
        return None, np.zeros(n, bool)
    rng = rng or np.random.default_rng(0)
    cos_thr = np.cos(threshold_rad)

    # Every sample is drawn up front and scored in batches: the per-sample work is a
    # 3x3 SVD and one pass over the bearings, far too small to run one at a time.
    first = rng.integers(0, n, iterations)
    second = rng.integers(0, n - 1, iterations)
    second += second >= first                      # two distinct indices per sample
    usable = np.abs(np.einsum("ij,ij->i", b_i[first], b_i[second])) <= 0.9999
    first, second = first[usable], second[usable]
    if len(first) == 0:
        return None, np.zeros(n, bool)
    models = batched_kabsch(b_i[first], b_j[first], b_i[second], b_j[second])

    best_mask = np.zeros(n, bool)
    best_count = 0
    for start in range(0, len(models), 64):
        chunk = models[start:start + 64]
        cosines = np.einsum("kab,nb,na->kn", chunk, b_i, b_j, optimize=True)
        masks = cosines >= cos_thr
        counts = masks.sum(axis=1)
        k = int(np.argmax(counts))
        if counts[k] > best_count:
            best_count, best_mask = int(counts[k]), masks[k]
    if best_count < 5:
        return None, best_mask
    # Refit on all inliers, then re-select once with the refined model
    R = kabsch_rotation(b_i[best_mask], b_j[best_mask])
    mask = np.einsum("ij,ij->i", b_i @ R.T, b_j) >= cos_thr
    if mask.sum() >= 5:
        R = kabsch_rotation(b_i[mask], b_j[mask])
    return R, mask


def guided_match(
    feat_i: FrameFeatures,
    feat_j: FrameFeatures,
    K: np.ndarray,
    R_prior_ij: Optional[np.ndarray],
    search_radius_px: float = 40.0,
    ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Descriptor matches, restricted to a window around the prior's prediction.

    Args:
        feat_i, feat_j: Features of the two frames.
        K: Intrinsics used to turn the prior rotation into a pixel prediction.
        R_prior_ij: Prior rotation (b_j = R @ b_i). None falls back to unguided matching.
        search_radius_px: Window radius around each predicted position.
        ratio: Lowe's ratio threshold, applied among candidates inside the window.

    Returns:
        (idx_i, idx_j) index arrays of accepted matches.
    """
    if len(feat_i.points) < 2 or len(feat_j.points) < 2:
        return np.zeros(0, int), np.zeros(0, int)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    mask = None
    # Share of the frame a window covers. Past half, it restricts little and is not
    # worth building; matching then searches the whole frame.
    width, height = 2 * K[0, 2], 2 * K[1, 2]
    window_share = np.pi * search_radius_px ** 2 / (width * height)
    if R_prior_ij is not None and window_share < 0.5:
        b = pixels_to_bearings(feat_i.points, K) @ R_prior_ij.T
        pred, front = project_bearings(b, K)
        pred = pred.astype(np.float32)
        pts_j = feat_j.points
        mask = np.zeros((len(pred), len(pts_j)), np.uint8)
        if window_share < 0.02:
            # Small windows: keypoints of frame j inside each one, from two k-d trees
            rows_front = np.nonzero(front)[0]
            if len(rows_front):
                near = cKDTree(pred[front]).sparse_distance_matrix(
                    cKDTree(pts_j), search_radius_px, output_type="ndarray")
                mask[rows_front[near["i"]], near["j"]] = 1
        else:
            # Large windows hold most keypoints, and listing every hit would cost far
            # more memory than the mask itself, so test distances in row blocks instead
            r2 = np.float32(search_radius_px ** 2)
            for s in range(0, len(pred), 256):
                dx = pred[s:s + 256, None, 0] - pts_j[None, :, 0]
                dy = pred[s:s + 256, None, 1] - pts_j[None, :, 1]
                mask[s:s + 256] = (dx * dx + dy * dy <= r2) & front[s:s + 256, None]

    knn = matcher.knnMatch(feat_i.descriptors, feat_j.descriptors, k=2, mask=mask)
    idx_i, idx_j = [], []
    for m in knn:
        if len(m) == 0:
            continue
        if len(m) == 1 or m[0].distance < ratio * m[1].distance:
            # With a window, a lone candidate is accepted; RANSAC decides the rest
            if m[0].distance <= 64:
                idx_i.append(m[0].queryIdx)
                idx_j.append(m[0].trainIdx)
    return np.array(idx_i, int), np.array(idx_j, int)


def match_pair(
    i: int,
    j: int,
    feat_i: FrameFeatures,
    feat_j: FrameFeatures,
    K: np.ndarray,
    R_prior_ij: Optional[np.ndarray],
    search_radius_px: float = 40.0,
    ransac_threshold_px: float = 2.0,
    max_prior_disagreement_deg: float = 5.0,
    min_inliers: int = 30,
    rng: Optional[np.random.Generator] = None,
    use_window: bool = True,
) -> PairMatch:
    """
    Guided matching plus rotation-only RANSAC for one frame pair.

    With use_window False the prior does not restrict where matches are searched, but a
    fitted rotation that disagrees with it by more than max_prior_disagreement_deg is
    still rejected.
    """
    ii, jj = guided_match(feat_i, feat_j, K, R_prior_ij if use_window else None, search_radius_px)
    empty = np.zeros((0, 2), np.float32)
    if len(ii) < min_inliers:
        return PairMatch(i, j, empty, empty, None, len(ii), 0)

    p_i, p_j = feat_i.points[ii], feat_j.points[jj]
    b_i, b_j = pixels_to_bearings(p_i, K), pixels_to_bearings(p_j, K)
    R, mask = ransac_rotation(b_i, b_j, ransac_threshold_px / K[0, 0], rng=rng)
    inliers = int(mask.sum())
    if R is None or inliers < min_inliers:
        return PairMatch(i, j, empty, empty, None, len(ii), inliers)

    if R_prior_ij is not None:
        disagreement = np.degrees(np.linalg.norm(so3_log(R @ R_prior_ij.T)))
        if disagreement > max_prior_disagreement_deg:
            logger.debug(f"Pair ({i},{j}): rotation disagrees with the prior by "
                         f"{disagreement:.1f}°, rejected")
            return PairMatch(i, j, empty, empty, None, len(ii), inliers)

    return PairMatch(i, j, p_i[mask], p_j[mask], R, len(ii), inliers)


def draw_pair(img_i: np.ndarray, img_j: np.ndarray, pair: PairMatch, max_lines: int = 200) -> np.ndarray:
    """Side-by-side image of a matched pair with its verified inlier correspondences."""
    n = min(len(pair.pts_i), max_lines)
    step = max(1, len(pair.pts_i) // max(n, 1))
    idx = np.arange(0, len(pair.pts_i), step)[:n]
    kp_i = [cv2.KeyPoint(float(x), float(y), 8) for x, y in pair.pts_i[idx]]
    kp_j = [cv2.KeyPoint(float(x), float(y), 8) for x, y in pair.pts_j[idx]]
    matches = [cv2.DMatch(k, k, 0.0) for k in range(len(idx))]
    vis = cv2.drawMatches(img_i, kp_i, img_j, kp_j, matches, None, matchColor=(0, 255, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    angle = np.degrees(np.linalg.norm(so3_log(pair.R_ij))) if pair.R_ij is not None else float("nan")
    cv2.putText(vis, f"Pair ({pair.i},{pair.j}): {pair.inliers} inliers, {angle:.2f} deg",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return vis
