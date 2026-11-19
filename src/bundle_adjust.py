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
    observations_used: dict = field(default_factory=dict)  # (i, j) -> indices into that pair's points


def _gather_observations(pairs: List[PairMatch], max_per_pair: int, seed: int, chosen=None):
    """
    Correspondences for the solve: up to max_per_pair per successful pair, sampled.

    When chosen is a dict, the sampled indices of each pair are recorded in it under
    (i, j), so callers can tell which correspondences the solve never saw.
    """
    rng = np.random.default_rng(seed)
    cam_i, cam_j, pi, pj = [], [], [], []
    used = 0
    for p in pairs:
        if not p.success or len(p.pts_i) == 0:
            continue
        idx = np.arange(len(p.pts_i))
        if len(idx) > max_per_pair:
            idx = rng.choice(idx, max_per_pair, replace=False)
        if chosen is not None:
            chosen[(p.i, p.j)] = idx
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


def _skew_batch(v: np.ndarray) -> np.ndarray:
    """[v]x for each row of v, shape (N, 3, 3)."""
    S = np.zeros((len(v), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -v[:, 2], v[:, 1]
    S[:, 1, 0], S[:, 1, 2] = v[:, 2], -v[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -v[:, 1], v[:, 0]
    return S


def left_jacobians(phi: np.ndarray) -> np.ndarray:
    """
    Left Jacobian of SO(3) for each rotation vector: exp(phi + d) ≈ exp(J_l d) exp(phi).
    """
    theta = np.linalg.norm(phi, axis=1)
    S = _skew_batch(phi)
    small = theta < 1e-6
    t = np.where(small, 1.0, theta)
    a = np.where(small, 0.5, (1.0 - np.cos(t)) / t ** 2)
    b = np.where(small, 1.0 / 6.0, (t - np.sin(t)) / t ** 3)
    return np.eye(3)[None] + a[:, None, None] * S + b[:, None, None] * (S @ S)


def transfer_jacobian(
    rotations: np.ndarray,
    jl: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    cam_i: np.ndarray,
    cam_j: np.ndarray,
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    optimize_focal: bool,
):
    """
    Analytic Jacobian of transfer_residuals.

    Parameters are a rotation vector per camera 1..N-1 (applied on the left of that
    camera's starting rotation, so jl holds each camera's left Jacobian, zero for the
    fixed camera 0) and, optionally, log(focal) as the last column.

    For a transfer r = R_dst R_srcᵀ u with u = K⁻¹x_src:
        ∂r/∂φ_dst = -[r]x J_l(φ_dst)
        ∂r/∂φ_src = R_dst R_srcᵀ [u]x J_l(φ_src)
        ∂r/∂log f = R_dst R_srcᵀ (-u_x, -u_y, 0)
    and the pixel is p = f r_xy / r_z + c.

    Returns:
        (rows, cols, values) of the non-zero entries, in transfer_residuals' row order.
    """
    m = len(cam_i)
    R_rel = np.einsum("nab,ncb->nac", rotations[cam_j], rotations[cam_i])
    rows, cols, vals = [], [], []
    n_rot = 3 * (len(rotations) - 1)
    obs = np.arange(m)

    for direction, (Rr, src, dst, pts) in enumerate((
            (R_rel, cam_i, cam_j, pts_i),
            (np.transpose(R_rel, (0, 2, 1)), cam_j, cam_i, pts_j))):
        u = np.stack([(pts[:, 0] - cx) / focal, (pts[:, 1] - cy) / focal, np.ones(m)], axis=1)
        r = np.einsum("nab,nb->na", Rr, u)
        z = np.where(np.abs(r[:, 2]) < 1e-6, 1e-6, r[:, 2])
        # dp/dr, shape (m, 2, 3)
        P = np.zeros((m, 2, 3))
        P[:, 0, 0] = focal / z
        P[:, 1, 1] = focal / z
        P[:, 0, 2] = -focal * r[:, 0] / z ** 2
        P[:, 1, 2] = -focal * r[:, 1] / z ** 2

        d_dst = P @ (-_skew_batch(r) @ jl[dst])
        d_src = P @ (Rr @ _skew_batch(u) @ jl[src])
        if optimize_focal:
            dr_df = np.einsum("nab,nb->na", Rr, np.stack([-u[:, 0], -u[:, 1], np.zeros(m)], 1))
            d_f = focal * r[:, :2] / z[:, None] + np.einsum("nab,nb->na", P, dr_df)

        for comp in range(2):
            row = (2 * direction + comp) * m + obs
            for cams, D in ((dst, d_dst), (src, d_src)):
                moving = cams > 0
                for k in range(3):
                    rows.append(row[moving])
                    cols.append(3 * (cams[moving] - 1) + k)
                    vals.append(D[moving, comp, k])
            if optimize_focal:
                rows.append(row)
                cols.append(np.full(m, n_rot))
                vals.append(d_f[:, comp])

    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


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
    chosen = {}
    ci, cj, pi, pj, used_pairs = _gather_observations(pairs, max_per_pair, seed, chosen)
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

    def jac(x):
        rots, f = unpack(x)
        jl = np.zeros((n, 3, 3))
        jl[1:] = left_jacobians(x[:n_rot].reshape(-1, 3))
        rows, cols, vals = transfer_jacobian(rots, jl, f, cx, cy, ci, cj, pi, pj, optimize_focal)
        return coo_matrix((vals, (rows, cols)), shape=(4 * m, n_params)).tocsr()

    x0 = np.zeros(n_params)
    r0 = fun(x0)
    rms_before = float(np.sqrt(np.mean(r0.reshape(4, -1) ** 2) * 2))
    logger.info(f"Bundle adjustment: {n} cameras, {used_pairs} pairs, {m} correspondences, "
                f"initial RMS {rms_before:.2f}px")

    result = least_squares(
        fun, x0, jac=jac, method="trf", loss=loss, tr_solver="lsmr",
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
        observations_used=chosen,
    )
