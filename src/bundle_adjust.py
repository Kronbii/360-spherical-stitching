"""
Global bundle adjustment for a rotating camera.

Pairwise estimates are each good to a fraction of a degree, but chaining them adds the
errors up, so after a full turn the last frame no longer lines up with the first. Bundle
adjustment solves for every camera's rotation at once, against every correspondence from
every pair — including the loop-closure pairs between the end and the start of the
sweep — so error that chaining would push to the end is spread out where it is
smallest.

Model: each match is a pixel x_i in frame i and x_j in frame j. The residual is the
pixel distance between x_j and x_i transferred through the rotations,
    x_j ≈ K(f) · R_j R_iᵀ · K(f)⁻¹ x_i,
evaluated in both directions. Unknowns are a rotation vector per camera (frame 0 is
held fixed to remove the global gauge) and, optionally, one shared focal length.
A soft-L1 loss keeps the occasional wrong match from pulling the solution.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .guided_match import PairMatch

logger = logging.getLogger(__name__)


@dataclass
class BundleResult:
    """Output of bundle_adjust."""
    rotations: List[np.ndarray]
    focal: float
    rms_before_px: float
    rms_after_px: float
    n_observations: int
    n_pairs: int
    cost_history: dict = field(default_factory=dict)
    success: bool = True


def _gather_observations(pairs: List[PairMatch], max_per_pair: int, seed: int):
    rng = np.random.default_rng(seed)
    cam_i, cam_j, pi, pj = [], [], [], []
    used = 0
    for p in pairs:
        if not p.success or len(p.pts_i) == 0:
            continue
        idx = np.arange(len(p.pts_i))
        if len(idx) > max_per_pair:
            idx = rng.choice(idx, max_per_pair, replace=False)
        cam_i.append(np.full(len(idx), p.i))
        cam_j.append(np.full(len(idx), p.j))
        pi.append(p.pts_i[idx])
        pj.append(p.pts_j[idx])
        used += 1
    if not cam_i:
        raise ValueError("No successful pairs to adjust")
    return (np.concatenate(cam_i), np.concatenate(cam_j),
            np.concatenate(pi).astype(np.float64), np.concatenate(pj).astype(np.float64),
            used)


def transfer_residuals(
    rotations: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    cam_i: np.ndarray,
    cam_j: np.ndarray,
    pts_i: np.ndarray,
    pts_j: np.ndarray,
) -> np.ndarray:
    """
    Symmetric transfer residuals in pixels, shape (4 * n_obs,).

    rotations is an (N, 3, 3) array of world -> camera rotations.
    """
    R_rel = np.einsum("nab,ncb->nac", rotations[cam_j], rotations[cam_i])  # R_j R_iᵀ

    def transfer(R, pts_src, pts_dst):
        rays = np.stack([(pts_src[:, 0] - cx) / focal,
                         (pts_src[:, 1] - cy) / focal,
                         np.ones(len(pts_src))], axis=1)
        r = np.einsum("nab,nb->na", R, rays)
        z = np.where(np.abs(r[:, 2]) < 1e-6, 1e-6, r[:, 2])
        u = focal * r[:, 0] / z + cx
        v = focal * r[:, 1] / z + cy
        return np.concatenate([u - pts_dst[:, 0], v - pts_dst[:, 1]])

    fwd = transfer(R_rel, pts_i, pts_j)
    bwd = transfer(np.transpose(R_rel, (0, 2, 1)), pts_j, pts_i)
    return np.concatenate([fwd, bwd])


def reprojection_rms(
    rotations: List[np.ndarray],
    focal: float,
    cx: float,
    cy: float,
    pairs: List[PairMatch],
    max_per_pair: int = 10_000,
) -> float:
    """RMS symmetric transfer error (px) of a set of rotations over the given pairs."""
    ci, cj, pi, pj, _ = _gather_observations(pairs, max_per_pair, seed=0)
    r = transfer_residuals(np.asarray(rotations), focal, cx, cy, ci, cj, pi, pj)
    # Each correspondence contributes two 2D errors; report per 2D error
    return float(np.sqrt(np.mean(r.reshape(4, -1) ** 2) * 2))


def bundle_adjust(
    initial_rotations: List[np.ndarray],
    pairs: List[PairMatch],
    focal: float,
    cx: float,
    cy: float,
    optimize_focal: bool = True,
    loss: str = "soft_l1",
    loss_scale_px: float = 1.5,
    max_per_pair: int = 150,
    max_nfev: int = 100,
    seed: int = 0,
) -> BundleResult:
    """
    Jointly refine all camera rotations (and the focal length) over all matched pairs.

    Args:
        initial_rotations: Starting world -> camera rotations, one per frame (the gyro
            prior is a good start: it is within a few degrees everywhere).
        pairs: Matched pairs, including non-adjacent and loop-closure pairs.
        focal: Initial focal length in pixels (fx = fy assumed).
        cx, cy: Principal point, held fixed.
        optimize_focal: Also solve for a shared focal length.
        loss: scipy robust loss name.
        loss_scale_px: Residual scale (pixels) at which the robust loss starts to bend.
        max_per_pair: Correspondences sampled per pair, to bound the problem size.
        max_nfev: Iteration cap for the solver.
        seed: Seed for the per-pair subsampling.

    Returns:
        BundleResult with refined rotations (frame 0 unchanged) and focal length.
    """
    n = len(initial_rotations)
    ci, cj, pi, pj, used_pairs = _gather_observations(pairs, max_per_pair, seed)
    m = len(ci)
    R0 = np.asarray(initial_rotations, dtype=np.float64)
    R_fixed = R0[0]

    # Parameterize frames 1..n-1 as rotation-vector corrections applied on the left
    n_rot = 3 * (n - 1)
    n_params = n_rot + (1 if optimize_focal else 0)

    def unpack(x):
        rots = np.empty((n, 3, 3))
        rots[0] = R_fixed
        rots[1:] = Rotation.from_rotvec(x[:n_rot].reshape(-1, 3)).as_matrix() @ R0[1:]
        f = focal * np.exp(x[n_rot]) if optimize_focal else focal
        return rots, f

    def fun(x):
        rots, f = unpack(x)
        return transfer_residuals(rots, f, cx, cy, ci, cj, pi, pj)

    # Each residual touches two cameras (and the focal length)
    sp_rows, sp_cols = [], []
    rows = np.arange(m)
    for block in range(4):
        r = rows + block * m
        for cams in (ci, cj):
            moving = cams > 0
            for k in range(3):
                sp_rows.append(r[moving])
                sp_cols.append(3 * (cams[moving] - 1) + k)
        if optimize_focal:
            sp_rows.append(r)
            sp_cols.append(np.full(m, n_rot))
    sp_rows, sp_cols = np.concatenate(sp_rows), np.concatenate(sp_cols)
    sparsity = coo_matrix((np.ones(len(sp_rows), np.uint8), (sp_rows, sp_cols)),
                          shape=(4 * m, n_params)).tocsr()
    sparsity.data[:] = 1

    x0 = np.zeros(n_params)
    r0 = fun(x0)
    rms_before = float(np.sqrt(np.mean(r0.reshape(4, -1) ** 2) * 2))
    logger.info(f"Bundle adjustment: {n} cameras, {used_pairs} pairs, {m} correspondences, "
                f"initial RMS {rms_before:.2f}px")

    result = least_squares(
        fun, x0, jac_sparsity=sparsity, method="trf", loss=loss,
        f_scale=loss_scale_px, x_scale="jac", max_nfev=max_nfev, verbose=0,
    )
    rots, f = unpack(result.x)
    r1 = result.fun
    rms_after = float(np.sqrt(np.mean(r1.reshape(4, -1) ** 2) * 2))
    logger.info(f"Bundle adjustment: RMS {rms_before:.2f}px -> {rms_after:.2f}px, "
                f"focal {focal:.1f} -> {f:.1f}px, {result.nfev} evaluations "
                f"({result.message})")

    return BundleResult(
        rotations=[R for R in rots],
        focal=float(f),
        rms_before_px=rms_before,
        rms_after_px=rms_after,
        n_observations=m,
        n_pairs=used_pairs,
        cost_history={"initial_cost": float(0.5 * np.sum(r0 ** 2)), "final_cost": float(result.cost),
                      "nfev": int(result.nfev)},
        success=bool(result.success),
    )
