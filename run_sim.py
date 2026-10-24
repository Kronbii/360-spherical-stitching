#!/usr/bin/env python3
"""
Validate the v2 orientation pipeline on a simulated sweep with known ground truth.

    python run_sim.py --world showcase/panorama.jpg --out output/sim

Renders frames from an existing panorama, simulates a gyroscope, runs the gyro-only,
pairwise-chained and bundle-adjusted estimates (plus the v1 homography chain for
reference), and writes:

    metrics.json        per-method rotation error, end-to-start gap, reprojection RMS, focal
    sweep.npz           frames' timestamps, ground truth and every method's rotations
    frames/             the simulated frames
    pano_<method>.jpg   panoramas stitched with each method's rotations
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import BlendingConfig, CalibrationData, MatchingConfig
from src.features import match_sequential_pairs
from src.pipeline_v2 import V2Options, evaluate_against_truth, run_orientation_v2, stitch_equirectangular
from src.rotation import chain_rotations, compute_relative_rotations
from src.simulate import SweepSpec, simulate_sweep


def v1_homography_chain(frames, calib, smoothing_window):
    """Orientations from the shipped v1 path: homography per pair, R = K⁻¹HK, chained."""
    cfg = MatchingConfig(match_width=None, min_inliers=60,
                         rotation_smoothing_window=smoothing_window)
    results = match_sequential_pairs(frames, cfg, scale=1.0)
    rel, _ = compute_relative_rotations(results, calib.K, 1.0)
    return chain_rotations(rel, apply_smoothing=smoothing_window > 1,
                           smoothing_window=smoothing_window)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=Path, default=Path("showcase/panorama.jpg"),
                    help="Equirectangular image used as the simulated environment")
    ap.add_argument("--out", type=Path, default=Path("output/sim"))
    ap.add_argument("--focal-error", type=float, default=-0.035,
                    help="Relative error of the starting focal length (e.g. -0.035 = 3.5%% short)")
    ap.add_argument("--pano-width", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-v1", action="store_true", help="Skip the v1 homography baseline")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_sim")
    args.out.mkdir(parents=True, exist_ok=True)

    world = cv2.imread(str(args.world))
    if world is None:
        raise SystemExit(f"Cannot read {args.world}")

    spec = SweepSpec(seed=args.seed)
    t0 = time.time()
    sim = simulate_sweep(world, spec)
    log.info(f"Simulation: {len(sim.frames)} frames in {time.time() - t0:.1f}s")

    frames_dir = args.out / "frames"
    frames_dir.mkdir(exist_ok=True)
    for k, img in enumerate(sim.frames):
        cv2.imwrite(str(frames_dir / f"frame_{k:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    f0 = sim.calib.fx * (1.0 + args.focal_error)
    calib_init = CalibrationData(fx=f0, fy=f0, cx=sim.calib.cx, cy=sim.calib.cy,
                                 source="simulation: perturbed starting focal")

    t0 = time.time()
    result = run_orientation_v2(sim.frames, sim.frame_times, sim.gyro, calib_init,
                                sim.gt_rotations[0], V2Options(seed=args.seed))
    log.info(f"v2 orientation: {time.time() - t0:.1f}s")

    extra = {}
    if not args.skip_v1:
        # v1 does not use the gyro; anchor its first frame like the others
        v1 = v1_homography_chain(sim.frames, calib_init, smoothing_window=1)
        extra["v1_homography_chain"] = [R @ sim.gt_rotations[0] for R in v1]

    metrics = evaluate_against_truth(result, sim.gt_rotations, sim.calib.fx,
                                     sim.calib.cx, sim.calib.cy, extra)
    metrics["simulation"] = {
        "world_image": args.world.name,
        "frames": len(sim.frames),
        "frame_rate_hz": spec.frame_rate_hz,
        "gyro_rate_hz": spec.gyro_rate_hz,
        "duration_s": spec.duration_s,
        "yaw_span_deg": spec.yaw_span_deg,
        "image_size": [spec.image_width, spec.image_height],
        "hfov_deg": spec.hfov_deg,
        "gyro_bias_const_dps": spec.gyro_bias_const_dps,
        "gyro_bias_walk_dps_rt_s": spec.gyro_bias_walk_dps_rt_s,
        "gyro_noise_dps_rt_hz": spec.gyro_noise_dps_rt_hz,
        "gyro_scale_error": spec.gyro_scale_error,
        "seed": spec.seed,
    }
    metrics["pipeline"] = result.diagnostics
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    np.savez_compressed(
        args.out / "sweep.npz",
        frame_times=sim.frame_times,
        gt=np.array(sim.gt_rotations),
        imu=np.array(result.imu_rotations),
        chain=np.array(result.chained_rotations),
        ba=np.array(result.ba_rotations),
        v1=np.array(extra["v1_homography_chain"]) if extra else np.zeros(0),
        focal_true=sim.calib.fx, focal_init=result.focal_initial, focal_ba=result.focal_ba,
        cx=sim.calib.cx, cy=sim.calib.cy,
        pair_ij=np.array([[p.i, p.j] for p in result.pairs if p.success]),
        gyro_t=sim.gyro.t, gyro_w=sim.gyro.omega,
    )

    # Panoramas with each method's orientations
    for name, rots, f in [("ba", result.ba_rotations, result.focal_ba),
                          ("chain", result.chained_rotations, result.focal_initial),
                          ("imu", result.imu_rotations, result.focal_initial)]:
        calib = CalibrationData(fx=f, fy=f, cx=sim.calib.cx, cy=sim.calib.cy)
        pano, _ = stitch_equirectangular(sim.frames, rots, calib, args.pano_width)
        cv2.imwrite(str(args.out / f"pano_{name}.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("\nRotation error against ground truth (degrees), simulated data")
    print(f"{'method':24s} {'mean':>7s} {'median':>7s} {'max':>7s} {'last':>7s} {'end-start':>9s} {'reproj px':>9s}")
    for name in ["imu_only", "v1_homography_chain", "pairwise_chain", "bundle_adjusted"]:
        if name not in metrics:
            continue
        m = metrics[name]
        print(f"{name:24s} {m['mean_deg']:7.3f} {m['median_deg']:7.3f} {m['max_deg']:7.3f} "
              f"{m['final_frame_deg']:7.3f} {m['end_to_start_deg']:9.3f} {m['reprojection_rms_px']:9.2f}")
    fm = metrics["focal"]
    print(f"focal: true {fm['true_px']:.1f}px, start {fm['initial_px']:.1f}px "
          f"({fm['initial_error_pct']:+.2f}%), after BA {fm['bundle_adjusted_px']:.1f}px "
          f"({fm['bundle_adjusted_error_pct']:+.3f}%)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
