#!/usr/bin/env python3
"""
Figures and a short video from a run_sim.py output folder.

    python scripts/make_v2_media.py --sim output/sim --out output/sim/media

Writes (prefix configurable):
    v2-rotation-error.png   per-frame rotation error against ground truth, per method
    v2-loop-seam.webp       the end-to-start seam with and without bundle adjustment
    v2-loop-closure.webp    a loop-closure pair found through the gyro prior
    v2-sweep.mp4            the simulated sweep, panorama assembling, then bundle adjustment
    v2-sweep-poster.jpg     a still from the video

Everything is labelled as simulated on the image itself.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CalibrationData  # noqa: E402
from src.guided_match import detect_features, match_pair  # noqa: E402
from src.imu import relative_rotation, so3_exp, so3_log  # noqa: E402
from src.warp_sphere import (create_equirectangular_grid, warp_image_to_equirectangular,  # noqa: E402
                             world_direction_grid)

# The figure shows the three estimates being compared; the v2 rotation-only chain is in metrics.json
FIGURE_METHODS = ("imu_only", "v1_homography_chain", "bundle_adjusted")

FONT_DIRS = [Path.home() / ".local/share/fonts/ibm-plex", Path("/usr/share/fonts/truetype/dejavu")]

# Shared method palette (figures use the light-card versions, the video the dark ones)
METHODS = [
    # key, label, light colour, dark colour
    ("imu_only", "Gyro only (dead reckoning)", "#c7791a", "#e8a94f"),
    ("pairwise_chain", "Pairwise chain, v2 rotation model", "#7a6fb0", "#a99be0"),
    ("v1_homography_chain", "Pairwise chain, v1 homography", "#b4407f", "#e07ab5"),
    ("bundle_adjusted", "Bundle adjusted", "#2f7d4f", "#7fd3a0"),
]


def font(size, bold=False, mono=False):
    from PIL import ImageFont
    names = (["IBMPlexMono-Bold.ttf" if bold else "IBMPlexMono-Regular.ttf", "DejaVuSansMono.ttf"] if mono else
             ["IBMPlexSans-Bold.ttf" if bold else "IBMPlexSans-Regular.ttf",
              "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"])
    for name in names:
        for d in FONT_DIRS:
            if (d / name).exists():
                return ImageFont.truetype(str(d / name), size)
    return ImageFont.load_default()


def save_figure(fig, path: Path):
    """PNG directly; WebP through Pillow (matplotlib has no WebP writer)."""
    if path.suffix.lower() == ".webp":
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        buf.seek(0)
        Image.open(buf).convert("RGB").save(path, "WEBP", quality=86, method=6)
    else:
        fig.savefig(path, facecolor="white")


def load_run(sim_dir: Path):
    d = np.load(sim_dir / "sweep.npz")
    metrics = json.loads((sim_dir / "metrics.json").read_text())
    frames = [cv2.imread(str(p)) for p in sorted((sim_dir / "frames").glob("frame_*.jpg"))]
    return d, metrics, frames


def calib(f, cx, cy):
    return CalibrationData(fx=float(f), fy=float(f), cx=float(cx), cy=float(cy))


def axis_theta_deg(R):
    a = R[2]
    return float(np.degrees(np.arctan2(a[0], a[2])))


# --------------------------------------------------------------------------- figures

def figure_rotation_error(metrics, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    for key, label, colour, _ in METHODS:
        if key not in metrics or key not in FIGURE_METHODS:
            continue
        e = np.array(metrics[key]["per_frame_deg"])
        # Frame 0 is the reference every method is anchored to, so its error is zero
        ax.plot(np.arange(1, len(e)), np.maximum(e[1:], 1e-3), color=colour, lw=2.2,
                label=f"{label}  (mean {metrics[key]['mean_deg']:.2f}°)")
    ax.set_yscale("log")
    ax.set_ylim(0.002, 10)
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.01°", "0.1°", "1°", "10°"])
    ax.set_xlim(0, None)
    ax.set_xlabel("Frame in sweep (73 frames over 396° of yaw; frame 0 is the shared reference)")
    ax.set_ylabel("Rotation error vs ground truth")
    ax.grid(True, which="major", color="#dddddd", lw=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fm = metrics["focal"]
    fig.suptitle("Orientation error per frame on a simulated sweep", x=0.07, ha="left",
                 fontsize=14, fontweight="bold")
    ax.set_title(f"Simulated data. Starting focal length {fm['initial_error_pct']:+.1f}% off; "
                 f"bundle adjustment recovers it to {fm['bundle_adjusted_error_pct']:+.2f}%.",
                 loc="left", fontsize=10, color="#555555")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def warp_set(frames, rots, cal, idx, W):
    """Average-blend a subset of frames into a W x W/2 panorama, return (pano, coverage)."""
    H = W // 2
    theta, phi = create_equirectangular_grid(W, H)
    dirs = world_direction_grid(theta, phi)
    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for k in idx:
        w, m = warp_image_to_equirectangular(frames[k], rots[k], cal, theta, phi, (H, W), dirs)
        wt = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
        acc += w.astype(np.float32) * wt[..., None]
        wsum += wt
    pano = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)
    return pano, wsum > 0


def loop_seam_panel(frames, rots, cal, gt, W=4096, cut_deg=14.0, half_width_deg=24.0,
                    phi_band=(-20.0, 24.0)):
    """
    The end of the sweep laid over the start, split along one vertical line.

    Left of the line: the first frames. Right of it: the last frames, which cover the
    same wall after a full turn. Any rotation error left at the end shows as a break.
    """
    yaw_true = np.degrees(np.unwrap(np.radians([axis_theta_deg(R) for R in gt])))
    first = [k for k in range(len(frames)) if yaw_true[k] < 45]
    last = [k for k in range(len(frames)) if yaw_true[k] > 360 - 5]
    pa, ca = warp_set(frames, rots, cal, first, W)
    pb, cb = warp_set(frames, rots, cal, last, W)
    H = W // 2
    cut = int((cut_deg + 180) / 360 * (W - 1))
    comp = pa.copy()
    comp[:, cut:] = pb[:, cut:]
    x0 = int((cut_deg - half_width_deg + 180) / 360 * (W - 1))
    x1 = int((cut_deg + half_width_deg + 180) / 360 * (W - 1))
    y0 = int((phi_band[0] + 90) / 180 * (H - 1))
    y1 = int((phi_band[1] + 90) / 180 * (H - 1))
    crop = comp[y0:y1, x0:x1].copy()
    return crop, cut - x0


def figure_loop_seam(d, metrics, frames, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = list(d["gt"])
    cx, cy = d["cx"], d["cy"]
    panels = [
        ("Pairwise chain (v1), no BA", list(d["v1"]), d["focal_init"],
         metrics["v1_homography_chain"]["end_to_start_deg"]),
        ("Bundle adjusted", list(d["ba"]), d["focal_ba"],
         metrics["bundle_adjusted"]["end_to_start_deg"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), dpi=150)
    fig.patch.set_facecolor("white")
    for ax, (title, rots, f, gap) in zip(axes, panels):
        crop, cut = loop_seam_panel(frames, rots, calib(f, cx, cy), gt)
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.axvline(cut, color="#ffffff", lw=1.0, ls=(0, (4, 4)), alpha=0.9)
        ax.set_title(f"{title}\nend-to-start error {gap:.2f}°", loc="left", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for x, txt, ha in ((cut + 10, "last frames", "left"), (cut - 10, "first frames", "right")):
            ax.text(x, 18, txt, color="white", fontsize=9, va="top", ha=ha,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#000000", ec="none", alpha=0.55))
    fig.suptitle("Where the sweep meets its start (simulated data)", x=0.02, y=0.985, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.02, 0.02, "Left of the dashed line: frames from the start of the sweep. Right: frames "
             "from after a full turn, over the same wall.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save_figure(fig, out_path)
    plt.close(fig)


def figure_loop_closure(d, frames, out_path: Path, i=None, j=None):
    """A first-and-last frame pair matched through the gyro prior, with its search window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt, imu = list(d["gt"]), list(d["imu"])
    n = len(frames)
    if i is None or j is None:
        yaw = np.degrees(np.unwrap(np.radians([axis_theta_deg(R) for R in gt])))
        i = int(np.argmin(np.abs(yaw - 12.0)))
        j = int(np.argmin(np.abs(yaw - (360 + 24.0))))
    cal = calib(d["focal_init"], d["cx"], d["cy"])
    K = cal.K
    fi, fj = detect_features(frames[i]), detect_features(frames[j])
    t = d["frame_times"]
    drift = 0.5 * abs(t[j] - t[i])
    radius = 25.0 + cal.fx * np.radians(drift)
    R_prior = relative_rotation(imu[i], imu[j])
    pm = match_pair(i, j, fi, fj, K, R_prior, search_radius_px=radius,
                    max_prior_disagreement_deg=2.0 + drift, rng=np.random.default_rng(0))
    gyro_err = np.degrees(np.linalg.norm(so3_log(R_prior @ relative_rotation(gt[i], gt[j]).T)))

    a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(frames[j], cv2.COLOR_BGR2RGB)
    gap = 24
    canvas = np.full((a.shape[0], a.shape[1] * 2 + gap, 3), 255, np.uint8)
    canvas[:, :a.shape[1]] = a
    canvas[:, a.shape[1] + gap:] = b
    fig, ax = plt.subplots(figsize=(10, 4.9), dpi=150)
    fig.patch.set_facecolor("white")
    ax.imshow(canvas)
    off = a.shape[1] + gap
    rng = np.random.default_rng(1)
    if pm.success:
        sel = rng.choice(len(pm.pts_i), min(60, len(pm.pts_i)), replace=False)
        # Where the gyro prior predicted those points would land
        from src.guided_match import pixels_to_bearings, project_bearings
        pred, _ = project_bearings(pixels_to_bearings(pm.pts_i[sel], K) @ R_prior.T, K)
        for k, ((x0, y0), (x1, y1), (px, py)) in enumerate(zip(pm.pts_i[sel], pm.pts_j[sel], pred)):
            ax.plot([x0, x1 + off], [y0, y1], color="#3fbf6f", lw=0.8, alpha=0.85)
            if k < 6:
                ax.add_patch(plt.Circle((px + off, py), radius, fill=False, color="#ff9f2e",
                                        lw=1.4, alpha=0.9))
                ax.plot(px + off, py, "+", color="#ff9f2e", ms=7, mew=1.6)
    ax.set_xlim(0, canvas.shape[1])
    ax.set_ylim(canvas.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Simulated frames {i} and {j}, {t[j] - t[i]:.0f} s and one full turn apart. "
                 f"{pm.inliers} rotation inliers.\nGreen: a sample of 60 matches. Orange: the gyro's "
                 f"predicted position for six of them, and the {radius:.0f} px window searched around it.",
                 loc="left", fontsize=9.5, color="#333333")
    fig.suptitle("Loop closure found through the gyro prior", x=0.02, y=0.985, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.02, 0.02, f"The gyro's relative rotation for this pair was {gyro_err:.2f}° off; "
             f"the window is sized for that drift.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_figure(fig, out_path)
    plt.close(fig)
    return i, j, pm.inliers, gyro_err


# --------------------------------------------------------------------------- video

BG = (16, 19, 17)          # near-black, slightly green
INK = (232, 236, 232)
MUTED = (150, 160, 152)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def slerp_rotations(Ra, Rb, s):
    return so3_exp(s * so3_log(Rb @ Ra.T)) @ Ra


class StripStitcher:
    """Incremental distance-weighted blend into a cropped equirectangular strip."""

    def __init__(self, W, phi_band_deg=(-36.0, 36.0)):
        self.W, self.H = W, W // 2
        theta, phi = create_equirectangular_grid(self.W, self.H)
        self.y0 = int((phi_band_deg[0] + 90) / 180 * (self.H - 1))
        self.y1 = int((phi_band_deg[1] + 90) / 180 * (self.H - 1))
        self.theta, self.phi = theta[self.y0:self.y1], phi[self.y0:self.y1]
        self.dirs = world_direction_grid(self.theta, self.phi)
        self.reset()

    def reset(self):
        h = self.y1 - self.y0
        self.acc = np.zeros((h, self.W, 3), np.float32)
        self.wsum = np.zeros((h, self.W), np.float32)

    def add(self, img, R, cal):
        h = self.y1 - self.y0
        w, m = warp_image_to_equirectangular(img, R, cal, self.theta, self.phi, (h, self.W), self.dirs)
        wt = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
        self.acc += w.astype(np.float32) * wt[..., None]
        self.wsum += wt
        return m

    def image(self):
        out = self.acc / np.maximum(self.wsum, 1e-6)[..., None]
        out[self.wsum <= 0] = (28, 32, 29)
        return out.astype(np.uint8)


def draw_plot(draw, box, metrics, show, n, ymax=5.0, colours=None):
    """Per-frame error lines; `show` maps method -> last frame index to draw."""
    x0, y0, x1, y1 = box
    f_small = font(15)
    # Axes and grid
    for v in range(0, int(ymax) + 1):
        y = y1 - (y1 - y0) * v / ymax
        draw.line([(x0, y), (x1, y)], fill=(44, 50, 46), width=1)
        draw.text((x0 - 10, y), f"{v}°", font=f_small, fill=MUTED, anchor="rm")
    draw.text((x0, y1 + 10), "frame 0", font=f_small, fill=MUTED, anchor="lt")
    draw.text((x1, y1 + 10), f"frame {n - 1}", font=f_small, fill=MUTED, anchor="rt")
    draw.text(((x0 + x1) / 2, y1 + 10), "rotation error vs ground truth, per frame",
              font=f_small, fill=MUTED, anchor="mt")
    for key, upto in show.items():
        if key not in metrics:
            continue
        e = np.array(metrics[key]["per_frame_deg"])
        m = max(1, int(round(upto)))
        pts = [(x0 + (x1 - x0) * k / (n - 1), y1 - (y1 - y0) * min(e[k], ymax) / ymax)
               for k in range(min(m + 1, n))]
        if len(pts) > 1:
            draw.line(pts, fill=colours[key], width=3, joint="curve")


def render_video(d, metrics, frames, out_mp4: Path, poster: Path, fps=30):
    from PIL import Image, ImageDraw

    gt = list(d["gt"])
    imu, ba = list(d["imu"]), list(d["ba"])
    n = len(frames)
    cx, cy = d["cx"], d["cy"]
    cal_init = calib(d["focal_init"], cx, cy)
    cal_ba = calib(d["focal_ba"], cx, cy)
    colours = {k: hex_rgb(dark) for k, _, _, dark in METHODS}
    labels = {k: lab for k, lab, _, _ in METHODS}
    pairs = metrics["pipeline"]["n_matched_pairs"]
    obs = metrics["pipeline"]["ba_observations"]

    Wv, Hv = 1280, 720
    strip = StripStitcher(1280)
    sh = strip.y1 - strip.y0
    strip_y = Hv - sh - 14

    f_title, f_body, f_small = font(26, bold=True), font(19), font(15)
    f_big = font(22, bold=True)

    tmp = Path(tempfile.mkdtemp(prefix="v2video_"))
    count = [0]

    def emit(img):
        img.save(tmp / f"f{count[0]:05d}.png")
        count[0] += 1

    def base(subtitle):
        img = Image.new("RGB", (Wv, Hv), BG)
        dr = ImageDraw.Draw(img)
        dr.text((24, 20), "360° stitcher v2: gyro prior + bundle adjustment", font=f_title, fill=INK)
        dr.text((24, 56), subtitle, font=f_body, fill=MUTED)
        badge = "SIMULATED DATA"
        bw = dr.textlength(badge, font=font(16, bold=True))
        dr.rounded_rectangle([Wv - bw - 48, 20, Wv - 24, 50], radius=6, outline=(232, 169, 79), width=2)
        dr.text((Wv - 36 - bw, 35), badge, font=font(16, bold=True), fill=(232, 169, 79), anchor="lm")
        return img, dr

    def paste_strip(img, dr, pano_bgr, outline_mask=None, colour=(127, 211, 160)):
        rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        if outline_mask is not None:
            cnts, _ = cv2.findContours(outline_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, cnts, -1, colour, 2)
        img.paste(Image.fromarray(rgb), (0, strip_y))
        dr.text((24, strip_y - 8), "equirectangular panorama, 360° wide (band shown: ±36° elevation)",
                font=f_small, fill=MUTED, anchor="lb")

    frame_box = (24, 100, 24 + 400, 100 + 300)
    plot_box = (520, 128, 1250, 372)

    def legend(dr, keys):
        y = plot_box[1] - 22
        x = plot_box[0]
        for k in keys:
            dr.line([(x, y), (x + 22, y)], fill=colours[k], width=4)
            dr.text((x + 30, y), labels[k], font=f_small, fill=INK, anchor="lm")
            x += 40 + dr.textlength(labels[k], font=f_small)

    # Phase 1: sweep. Frames placed with the gyro prior as they arrive.
    per = 3  # video frames per simulated frame -> 73 * 3 / 30 ≈ 7.3 s
    strip.reset()
    for k in range(n):
        m = strip.add(frames[k], imu[k], cal_init)
        for rep in range(per):
            img, dr = base("A camera turns 396° on the spot; a 200 Hz gyro is integrated between 3 Hz frames.")
            small = cv2.resize(frames[k], (400, 300), interpolation=cv2.INTER_AREA)
            img.paste(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), frame_box[:2])
            dr.text((frame_box[0], frame_box[3] + 8), f"simulated camera frame {k + 1}/{n}",
                    font=f_small, fill=MUTED)
            legend(dr, ["imu_only", "v1_homography_chain"])
            draw_plot(dr, plot_box, metrics, {"imu_only": k + rep / per, "v1_homography_chain": k + rep / per},
                      n, colours=colours)
            paste_strip(img, dr, strip.image(), m > 0, colours["imu_only"])
            emit(img)

    gyro_pano = strip.image()
    # Hold: the gyro-only panorama, drift visible at the loop
    for rep in range(int(1.2 * fps)):
        img, dr = base("Placed with the gyro alone, the end of the sweep does not meet its start.")
        dr.text((frame_box[0], frame_box[1] + 10), "Gyro only", font=f_big, fill=colours["imu_only"])
        for r, line in enumerate([
            f"error at last frame: {metrics['imu_only']['final_frame_deg']:.2f}°",
            f"mean error: {metrics['imu_only']['mean_deg']:.2f}°",
            "bias 0.25°/s, scale error 0.5%, noise",
        ]):
            dr.text((frame_box[0], frame_box[1] + 52 + 32 * r), line, font=f_body, fill=INK)
        legend(dr, ["imu_only", "v1_homography_chain"])
        draw_plot(dr, plot_box, metrics, {"imu_only": n - 1, "v1_homography_chain": n - 1}, n, colours=colours)
        paste_strip(img, dr, gyro_pano)
        emit(img)

    # Phase 2: bundle adjustment — rotations move from the gyro prior to the solution
    steps = int(3.0 * fps)
    for s in range(steps + 1):
        a = s / steps
        ease = a * a * (3 - 2 * a)
        f = d["focal_init"] + ease * (d["focal_ba"] - d["focal_init"])
        cal = calib(f, cx, cy)
        strip.reset()
        for k in range(n):
            strip.add(frames[k], slerp_rotations(imu[k], ba[k], ease), cal)
        img, dr = base(f"Bundle adjustment: all {n} rotations and the focal length, solved together.")
        dr.text((frame_box[0], frame_box[1] + 10), "Bundle adjustment", font=f_big,
                fill=colours["bundle_adjusted"])
        for r, line in enumerate([
            f"{pairs} matched pairs, gyro-guided",
            f"incl. loop closures from end to start",
            f"{obs:,} correspondences, soft-L1 loss",
            f"reprojection RMS {metrics['pipeline']['ba_rms_before_px']:.1f} → "
            f"{metrics['pipeline']['ba_rms_after_px']:.2f} px",
            f"focal {d['focal_init']:.0f} → {f:.0f} px (true {d['focal_true']:.0f})",
        ]):
            dr.text((frame_box[0], frame_box[1] + 52 + 32 * r), line, font=f_body, fill=INK)
        legend(dr, ["imu_only", "v1_homography_chain", "bundle_adjusted"])
        draw_plot(dr, plot_box, metrics,
                  {"imu_only": n - 1, "v1_homography_chain": n - 1, "bundle_adjusted": (n - 1) * ease},
                  n, colours=colours)
        paste_strip(img, dr, strip.image())
        emit(img)

    final = strip.image()
    # Phase 3: hold with the numbers
    hold = int(4.0 * fps)
    for rep in range(hold):
        img, dr = base("Same frames, same features: only the orientations differ.")
        dr.text((frame_box[0], frame_box[1] + 10), "Mean rotation error", font=f_big, fill=INK)
        rows = [("imu_only", "gyro only"), ("v1_homography_chain", "pairwise chain (v1)"),
                ("bundle_adjusted", "bundle adjusted")]
        for r, (key, lab) in enumerate(rows):
            y = frame_box[1] + 56 + 40 * r
            dr.line([(frame_box[0], y + 12), (frame_box[0] + 22, y + 12)], fill=colours[key], width=4)
            dr.text((frame_box[0] + 32, y), lab, font=f_body, fill=INK)
            dr.text((frame_box[0] + 440, y), f"{metrics[key]['mean_deg']:.2f}°", font=f_body,
                    fill=INK, anchor="ra")
        dr.text((frame_box[0], frame_box[1] + 200),
                f"end-to-start: {metrics['v1_homography_chain']['end_to_start_deg']:.2f}° → "
                f"{metrics['bundle_adjusted']['end_to_start_deg']:.3f}°", font=f_body, fill=MUTED)
        dr.text((frame_box[0], frame_box[1] + 232),
                "pure rotation, no lens distortion", font=f_body, fill=MUTED)
        legend(dr, ["imu_only", "v1_homography_chain", "bundle_adjusted"])
        draw_plot(dr, plot_box, metrics,
                  {"imu_only": n - 1, "v1_homography_chain": n - 1, "bundle_adjusted": n - 1},
                  n, colours=colours)
        paste_strip(img, dr, final)
        emit(img)
        if rep == hold // 2:
            img.convert("RGB").save(poster, quality=88)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "24",
           "-movflags", "+faststart", "-an", str(out_mp4)]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp)
    return count[0] / fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", type=Path, default=Path("output/sim"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--prefix", default="v2-")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()
    out = args.out or (args.sim / "media")
    out.mkdir(parents=True, exist_ok=True)

    d, metrics, frames = load_run(args.sim)
    p = args.prefix
    figure_rotation_error(metrics, out / f"{p}rotation-error.png")
    figure_loop_seam(d, metrics, frames, out / f"{p}loop-seam.webp")
    info = figure_loop_closure(d, frames, out / f"{p}loop-closure.webp")
    print(f"loop-closure figure: frames {info[0]} and {info[1]}, {info[2]} inliers, "
          f"gyro relative error {info[3]:.2f}°")
    if not args.skip_video:
        secs = render_video(d, metrics, frames, out / f"{p}sweep.mp4", out / f"{p}sweep-poster.jpg")
        print(f"video: {secs:.1f} s")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
