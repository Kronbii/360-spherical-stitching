#!/usr/bin/env python3
"""
Validate the v2 orientation pipeline on a simulated sweep with known ground truth.

    python run_sim.py --world showcase/panorama.jpg --out output/sim

Renders frames from an existing panorama, simulates a gyroscope, and scores against
ground truth: the gyro alone, the v1 homography chain, the v2 pairwise chain, and
bundle adjustment twice, once started from the gyro and once from the v1 chain (no
gyro). Writes:

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

from src.config import CalibrationData
from src.pipeline_v2 import (V2Options, evaluate_against_truth, run_orientation_v2,
                             run_orientation_without_imu, stitch_equirectangular)
from src.simulate import SweepSpec, simulate_sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=Path, default=Path("showcase/panorama.jpg"),
                    help="Equirectangular image used as the simulated environment")
    ap.add_argument("--out", type=Path, default=Path("output/sim"))
    ap.add_argument("--focal-error", type=float, default=-0.035,
                    help="Relative error of the starting focal length (e.g. -0.035 = 3.5%% short)")
    ap.add_argument("--pano-width", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-v1", action="store_true",
                    help="Skip the v1 homography baseline and bundle adjustment without the gyro")
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
    no_imu = None
    if not args.skip_v1:
        # Bundle adjustment without the gyro, started from the unsmoothed v1 chain and
        # anchored at the same first frame as the gyro path
        t0 = time.time()
        no_imu, v1 = run_orientation_without_imu(sim.frames, calib_init, sim.gt_rotations[0],
                                                 V2Options(seed=args.seed))
        log.info(f"Bundle adjustment without gyro: {time.time() - t0:.1f}s")
        extra["v1_homography_chain"] = v1
        extra["bundle_adjusted_no_imu"] = (no_imu.rotations, no_imu.focal)

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
    if no_imu is not None:
        metrics["pipeline_no_imu"] = no_imu.diagnostics
        metrics["focal"]["bundle_adjusted_no_imu_px"] = no_imu.focal
        metrics["focal"]["bundle_adjusted_no_imu_error_pct"] = 100.0 * (no_imu.focal - sim.calib.fx) / sim.calib.fx
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    np.savez_compressed(
        args.out / "sweep.npz",
        frame_times=sim.frame_times,
        gt=np.array(sim.gt_rotations),
        imu=np.array(result.imu_rotations),
        chain=np.array(result.chained_rotations),
        ba=np.array(result.ba_rotations),
        v1=np.array(extra["v1_homography_chain"]) if extra else np.zeros(0),
        ba_no_imu=np.array(no_imu.rotations) if no_imu is not None else np.zeros(0),
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
    for name in ["imu_only", "v1_homography_chain", "pairwise_chain", "bundle_adjusted",
                 "bundle_adjusted_no_imu"]:
        if name not in metrics:
            continue
        m = metrics[name]
        print(f"{name:24s} {m['mean_deg']:7.3f} {m['median_deg']:7.3f} {m['max_deg']:7.3f} "
              f"{m['final_frame_deg']:7.3f} {m['end_to_start_deg']:9.3f} {m['reprojection_rms_px']:9.2f}")
    fm = metrics["focal"]
    print(f"focal: true {fm['true_px']:.1f}px, start {fm['initial_px']:.1f}px "
          f"({fm['initial_error_pct']:+.2f}%), after BA {fm['bundle_adjusted_px']:.1f}px "
          f"({fm['bundle_adjusted_error_pct']:+.3f}%)"
          + (f", after BA without gyro {fm['bundle_adjusted_no_imu_px']:.1f}px "
             f"({fm['bundle_adjusted_no_imu_error_pct']:+.3f}%)" if no_imu is not None else ""))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
