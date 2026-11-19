#!/usr/bin/env python3
"""
Compare the orientations of pipeline runs on real footage, where there is no ground truth.

    python scripts/compare_orientation.py output/run_chain output/run_bundle

Every run directory needs the orientation.npz the pipeline writes. One of them must come
from orientation.method 'bundle': its verified correspondences are the common yardstick.
They are held out: a sample per matched pair, in full-resolution pixels, of the RANSAC
inliers that bundle adjustment itself did not fit, so the bundle-adjusted run is not
scored on the points it was solved against. Each run's rotations and focal
length transfer every correspondence from one frame into the other, and the distance to
where it was actually seen is the error. Pairs are split by how far apart the two frames
are along the sweep:

    near   frames within the local window (mostly adjacent)
    long   further apart, but less than half a turn
    loop   at least half a turn apart: the end of the sweep against its start

Frames a run trimmed (circular closure) are left out of that run's numbers.

Writes compare.json next to the first run unless --out is given.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bundle_adjust import transfer_residuals  # noqa: E402


def heading_deg(rotations):
    axes = np.array([R[2, :] for R in rotations])
    return np.degrees(np.unwrap(np.arctan2(axes[:, 0], axes[:, 2])))


def load_runs(path: Path):
    """
    The orientations a run produced. A bundle run that had to squeeze its sweep apart
    (no loop closure, ends colliding) yields two entries: what bundle adjustment solved,
    and what the warp actually used.
    """
    d = np.load(path / "orientation.npz")
    base = {"path": path, "method": str(d["method"]), "focal": float(d["focal_px"]), "data": d}
    out = [dict(base, label=path.name, rotations=d["rotations"])]
    if "solved_rotations" in d and not np.allclose(d["solved_rotations"], d["rotations"]):
        out[0]["label"] = f"{path.name} (as warped, squeezed)"
        out.insert(0, dict(base, label=f"{path.name} (solved)", rotations=d["solved_rotations"]))
    return out


def principal_point(run):
    intr = json.loads((run["path"] / "intrinsics.json").read_text())
    return float(intr["intrinsics"]["cx"]), float(intr["intrinsics"]["cy"])


def errors_px(run, cx, cy, ci, cj, pi, pj):
    """Per-correspondence transfer error (px), averaged over both directions."""
    r = transfer_residuals(run["rotations"], run["focal"], cx, cy, ci, cj, pi, pj).reshape(4, -1)
    fwd = np.hypot(r[0], r[1])
    bwd = np.hypot(r[2], r[3])
    return 0.5 * (fwd + bwd)


def summarise(err):
    if len(err) == 0:
        return None
    return {
        "n": int(len(err)),
        "rms_px": float(np.sqrt(np.mean(err ** 2))),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95)),
        "over_3px_pct": float(100.0 * np.mean(err > 3.0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--local-gap", type=int, default=3)
    args = ap.parse_args()

    runs = [r for p in args.runs for r in load_runs(p)]
    refs = [r for r in runs if r["method"] == "bundle" and "pair_ij" in r["data"]]
    if not refs:
        raise SystemExit("Need at least one run with orientation.method 'bundle'")
    ref = refs[0]
    d = ref["data"]
    counts = d["pair_counts"]
    ij = d["pair_ij"]
    ci = np.repeat(ij[:, 0], counts)
    cj = np.repeat(ij[:, 1], counts)
    pi, pj = d["pts_i"].astype(np.float64), d["pts_j"].astype(np.float64)
    cx, cy = principal_point(ref)

    # Classify pairs by the reference run's own sweep
    h = heading_deg(ref["rotations"])
    travelled = np.abs(h[cj] - h[ci])
    gap = cj - ci
    kind = np.where(travelled >= 180.0, "loop", np.where(gap <= args.local_gap, "near", "long"))

    report = {"reference": str(ref["path"]), "pairs": {k: int(np.sum((kind == k)[np.r_[0, np.cumsum(counts)[:-1]]]))
                                                       for k in ("near", "long", "loop")},
              "runs": {}}
    for run in runs:
        n = len(run["rotations"])
        keep = (ci < n) & (cj < n)
        err = errors_px(run, cx, cy, ci[keep], cj[keep], pi[keep], pj[keep])
        k = kind[keep]
        hr = heading_deg(run["rotations"])
        entry = {
            "method": run["method"],
            "frames": n,
            "focal_px": run["focal"],
            "hfov_deg": float(np.degrees(2 * np.arctan(cx / run["focal"]))),
            "sweep_deg": float(abs(hr[-1] - hr[0])),
            "all": summarise(err),
        }
        for name in ("near", "long", "loop"):
            entry[name] = summarise(err[k == name])
        report["runs"][run["label"]] = entry

    out = args.out or (args.runs[0].parent / "compare.json")
    out.write_text(json.dumps(report, indent=2))

    print(f"Yardstick: held-out verified correspondences from {ref['path']} "
          f"({report['pairs']['near']} near, {report['pairs']['long']} long, {report['pairs']['loop']} loop pairs)")
    print(f"{'run':40s} {'frames':>6s} {'focal':>7s} {'sweep':>7s} "
          f"{'all rms':>8s} {'near':>7s} {'long':>7s} {'loop':>7s} {'>3px':>6s}")
    for label, e in report["runs"].items():
        def rms(x):
            return f"{e[x]['rms_px']:7.2f}" if e[x] else "      -"
        print(f"{label:40s} {e['frames']:6d} {e['focal_px']:7.1f} {e['sweep_deg']:6.1f}° "
              f"{e['all']['rms_px']:8.2f} {rms('near')} {rms('long')} {rms('loop')} {e['all']['over_3px_pct']:5.1f}%")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
