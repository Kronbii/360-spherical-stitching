"""
Tests for the shared orientation stage (orientation.py) in both of its modes, the gyro
I/O and self-calibration it relies on, and its wiring into the pipeline.
"""

import cv2
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from src.bundle_adjust import left_jacobians, transfer_jacobian, transfer_residuals
from src.config import (CalibrationData, ImuConfig, IntrinsicsConfig, OrientationConfig,
                        OutputConfig, PipelineConfig, load_config_from_yaml)
from src.guided_match import PairMatch, batched_kabsch, kabsch_rotation
from src.imu import (GyroLog, estimate_axis_alignment, estimate_time_offset, gyro_interval_rotvecs,
                     load_frame_times, load_gyro_csv, parse_axes, relative_rotation, so3_exp,
                     so3_log, write_frame_times)
from src.io_utils import ImageInfo, LazyImages, load_image
from src.orientation import (OrientationOptions, average_rotations, calibrate_gyro, chain_prior,
                             detect_all_features, select_pairs, unwrapped_heading_deg)
from src.pipeline_v2 import V2Options, per_frame_errors, run_orientation_without_imu
from src.simulate import SweepSpec, rot_y, simulate_sweep
from tests.test_imu_v2 import textured_world


def yaw_sweep(n, step_deg, wobble_deg=1.5):
    """World -> camera rotations for a sweep about the vertical with pitch/roll wobble."""
    return [(rot_y(np.radians(step_deg * k))
             @ so3_exp(np.radians([wobble_deg * np.sin(k / 3), 0, wobble_deg * np.cos(k / 4)]))).T
            for k in range(n)]


def rotation_pair(truth, i, j, noise_deg=0.0, rng=None, inliers=100):
    R = relative_rotation(truth[i], truth[j])
    if noise_deg:
        R = so3_exp(np.radians(rng.normal(0, noise_deg, 3))) @ R
    empty = np.zeros((0, 2), np.float32)
    return PairMatch(i, j, empty, empty, R, inliers, inliers)


# ---------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------

class TestBatchedKabsch:
    def test_matches_single_kabsch(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(30, 2, 3))
        b = rng.normal(size=(30, 2, 3))
        a /= np.linalg.norm(a, axis=2, keepdims=True)
        b /= np.linalg.norm(b, axis=2, keepdims=True)
        batched = batched_kabsch(a[:, 0], b[:, 0], a[:, 1], b[:, 1])
        single = np.array([kabsch_rotation(a[k], b[k]) for k in range(30)])
        assert np.allclose(batched, single, atol=1e-12)
        assert np.allclose(np.linalg.det(batched), 1.0)


class TestBundleJacobian:
    def test_analytic_matches_finite_differences(self):
        rng = np.random.default_rng(1)
        n, m = 5, 40
        R0 = Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.3).as_matrix()
        ci = rng.integers(0, n - 1, m)
        cj = np.minimum(ci + rng.integers(1, 3, m), n - 1)
        f0, cx, cy = 600.0, 400.0, 300.0
        pi = rng.uniform(0, 800, (m, 2))
        pj = rng.uniform(0, 600, (m, 2))
        n_rot = 3 * (n - 1)

        def unpack(x):
            rots = R0.copy()
            rots[1:] = Rotation.from_rotvec(x[:n_rot].reshape(-1, 3)).as_matrix() @ R0[1:]
            return rots, f0 * np.exp(x[n_rot])

        def fun(x):
            rots, f = unpack(x)
            return transfer_residuals(rots, f, cx, cy, ci, cj, pi, pj)

        x = rng.normal(size=n_rot + 1) * 0.2
        rots, f = unpack(x)
        jl = np.zeros((n, 3, 3))
        jl[1:] = left_jacobians(x[:n_rot].reshape(-1, 3))
        r, c, v = transfer_jacobian(rots, jl, f, cx, cy, ci, cj, pi, pj, True)
        J = coo_matrix((v, (r, c)), shape=(4 * m, n_rot + 1)).toarray()

        J_num = np.zeros_like(J)
        h = 1e-6
        for k in range(n_rot + 1):
            e = np.zeros_like(x)
            e[k] = h
            J_num[:, k] = (fun(x + e) - fun(x - e)) / (2 * h)
        assert np.abs(J - J_num).max() < 1e-7 * np.abs(J_num).max()

    def test_left_jacobian_small_angle(self):
        assert np.allclose(left_jacobians(np.zeros((1, 3)))[0], np.eye(3))


class TestRotationAveraging:
    def make_problem(self, outlier=False):
        rng = np.random.default_rng(3)
        truth = yaw_sweep(40, 9.5)  # 370° in 40 frames: the ends overlap
        pairs = [rotation_pair(truth, i, i + g, 0.05, rng) for g in (1, 2) for i in range(40 - g)]
        pairs += [rotation_pair(truth, i, j, 0.05, rng) for i, j in [(0, 37), (0, 38), (1, 38), (1, 39)]]
        if outlier:
            bad = rotation_pair(truth, 5, 20)
            bad.R_ij = so3_exp(np.radians([0, 20, 0])) @ bad.R_ij
            pairs.append(bad)
        # Chained start that over-counts every step by 0.3°: 11° wrong by the end
        init = [so3_exp(np.radians([0, 0.3 * k, 0])).T @ R for k, R in enumerate(truth)]
        init[0] = truth[0]
        return truth, pairs, init

    def test_closes_a_drifting_loop(self):
        truth, pairs, init = self.make_problem()
        assert per_frame_errors(init, truth).max() > 10
        res = average_rotations(init, pairs)
        assert per_frame_errors(res.rotations, truth).max() < 0.2
        assert np.allclose(res.rotations[0], truth[0])
        assert not res.rejected

    def test_rejects_an_inconsistent_pair(self):
        truth, pairs, init = self.make_problem(outlier=True)
        res = average_rotations(init, pairs)
        assert [(p.i, p.j) for p in res.rejected] == [(5, 20)]
        assert per_frame_errors(res.rotations, truth).max() < 0.2

    def test_gap_no_pair_spans_is_held_by_the_prior(self):
        """
        The failure seen on real footage: a low-texture stretch where no pair links the
        frames before it to the frames after it. The pairs after the gap disagree with
        the prior's step size, which without a bridge swings that whole stretch.
        """
        truth = yaw_sweep(30, 5.0)
        pairs = [rotation_pair(truth, i, i + g) for g in (1, 2) for i in range(30 - g)
                 if not (i <= 14 < i + g)]
        # Prior over-counts each step by 0.4°
        init = [so3_exp(np.radians([0, 0.4 * k, 0])).T @ R for k, R in enumerate(truth)]
        res = average_rotations(init, pairs)
        assert res.bridges == [(14, 15)]
        h_init = unwrapped_heading_deg(init)
        h_res = unwrapped_heading_deg(res.rotations)
        # Across the gap the step is the prior's; either side matches the pairs
        assert (h_res[15] - h_res[14]) == pytest.approx(h_init[15] - h_init[14], abs=0.05)
        assert per_frame_errors(res.rotations[:15], truth[:15]).max() < 0.05
        assert np.allclose(np.diff(h_res[15:]), np.diff(unwrapped_heading_deg(truth[15:])), atol=0.05)

    def test_reanchor_restores_placement_of_a_loose_stretch(self):
        from src.orientation import reanchor_components
        truth = yaw_sweep(10, 4.0)
        turned = [R if k < 5 else so3_exp(np.radians([0, 12.0, 0])) @ R for k, R in enumerate(truth)]
        pairs = [(i, i + 1) for i in range(9) if i != 4]
        fixed = reanchor_components(turned, truth, pairs)
        assert per_frame_errors(fixed, truth).max() < 1e-9

    def test_unreached_frame_stays_with_its_start(self):
        truth = yaw_sweep(6, 10.0)
        pairs = [rotation_pair(truth, i, i + 1) for i in range(3)]  # frames 4, 5 unreached
        res = average_rotations(truth, pairs)
        assert per_frame_errors(res.rotations, truth).max() < 1e-6


class TestFocalSearch:
    def pairs_for(self, truth, K, loop):
        from tests.test_imu_v2 import synthetic_pairs
        n = len(truth)
        extra = [(0, n - 3), (0, n - 2), (1, n - 2), (1, n - 1), (2, n - 1)] if loop else []
        return synthetic_pairs(truth, K, n_pts=60, noise_px=0.3, gaps=(1, 2, 4), extra=extra)

    def test_loop_pins_the_focal(self):
        from src.orientation import estimate_focal
        from tests.test_imu_v2 import calib_for
        truth = yaw_sweep(40, 9.5)  # 370°: the ends overlap
        K = calib_for(640, 480, 60).K
        pairs = self.pairs_for(truth, K, loop=True)
        K_short = K.copy()
        K_short[0, 0] = K_short[1, 1] = K[0, 0] * 0.9
        f, report = estimate_focal(pairs, K_short, truth)
        assert f == pytest.approx(K[0, 0], rel=0.01)
        assert report["sharpness"] > 1.5

    def test_a_loop_sharpens_the_minimum(self):
        """Same sweep with and without its loop-closure pairs: the loop pins the focal harder."""
        from src.orientation import estimate_focal
        from tests.test_imu_v2 import calib_for
        truth = yaw_sweep(40, 9.5)
        K = calib_for(640, 480, 60).K
        _, open_report = estimate_focal(self.pairs_for(truth, K, loop=False), K, truth)
        _, loop_report = estimate_focal(self.pairs_for(truth, K, loop=True), K, truth)
        assert loop_report["sharpness"] > 1.5 * open_report["sharpness"]


class TestPairSelection:
    def test_local_long_and_loop_pairs(self):
        prior = chain_prior(yaw_sweep(80, 5.0), drift_deg_per_link=0.1)  # 395°
        options = OrientationOptions()
        pairs = select_pairs(prior, hfov_deg=60.0, options=options)
        assert len(pairs) == len(set(pairs)) and pairs == sorted(pairs)
        assert all((i, i + 1) in pairs for i in range(79))
        heading = unwrapped_heading_deg(prior.rotations)
        loops = [(i, j) for i, j in pairs if abs(heading[j] - heading[i]) >= 180]
        longs = [(i, j) for i, j in pairs if j - i > options.max_pair_gap_local and (i, j) not in loops]
        assert loops and all(heading[j] - heading[i] > 300 for i, j in loops)
        assert longs and all(abs(heading[j] - heading[i]) < 60 for i, j in longs)
        # Excluded pairs are left out
        assert not set(select_pairs(prior, 60.0, options, exclude=set(loops))) & set(loops)

    def test_short_sweep_has_no_loop_pairs(self):
        prior = chain_prior(yaw_sweep(30, 5.0), drift_deg_per_link=0.1)  # 145°
        heading = unwrapped_heading_deg(prior.rotations)
        pairs = select_pairs(prior, 60.0, OrientationOptions())
        assert all(abs(heading[j] - heading[i]) < 180 for i, j in pairs)


class TestLazyImages:
    @pytest.fixture
    def frames(self, temp_dir):
        infos = []
        for k in range(5):
            img = np.full((20, 30, 3), 40 * k, np.uint8)
            path = temp_dir / f"f{k}.png"
            cv2.imwrite(str(path), img)
            infos.append(ImageInfo(path=path))
        return infos

    def test_behaves_like_a_list(self, frames):
        lazy = LazyImages(frames)
        assert len(lazy) == 5
        assert np.array_equal(lazy[2], load_image(frames[2].path))
        assert np.array_equal(lazy[-1], load_image(frames[4].path))
        tail = lazy[3:]
        assert isinstance(tail, LazyImages) and len(tail) == 2
        assert np.array_equal(tail[0], lazy[3])
        with pytest.raises(IndexError):
            lazy[5]

    def test_cache_stays_within_budget(self, frames):
        lazy = LazyImages(frames, cache_bytes=20 * 30 * 3)  # room for one frame
        for k in range(5):
            lazy[k]
        assert len(lazy._cache) == 1 and 4 in lazy._cache


# ---------------------------------------------------------------------------------------
# Gyro I/O and calibration
# ---------------------------------------------------------------------------------------

class TestGyroIO:
    def test_sensor_logger_style_columns(self, temp_dir):
        path = temp_dir / "Gyroscope.csv"
        t = 1.7e18 + np.arange(5) * 1e7  # nanoseconds, 100 Hz
        rows = [f"{int(a)},{k * 0.01:.2f},{k * 0.3},{k * 0.2},{k * 0.1}" for k, a in enumerate(t)]
        path.write_text("time,seconds_elapsed,z,y,x\n" + "\n".join(rows) + "\n")
        log = load_gyro_csv(path)
        assert np.allclose(np.diff(log.t), 0.01)
        assert np.allclose(log.omega[3], [0.3, 0.6, 0.9])  # x, y, z regardless of column order

    def test_milliseconds_degrees_unsorted_duplicates(self, temp_dir):
        path = temp_dir / "gyro.csv"
        path.write_text("timestamp,gx,gy,gz\n20,0,0,90\n0,0,0,0\n10,0,0,45\n10,0,0,45\n")
        log = load_gyro_csv(path, units="deg/s")
        assert np.allclose(log.t, [0.0, 0.01, 0.02])
        assert np.allclose(log.omega[:, 2], np.radians([0, 45, 90]))

    def test_missing_columns(self, temp_dir):
        path = temp_dir / "gyro.csv"
        path.write_text("t,wx,wy\n0,0,0\n")
        with pytest.raises(ValueError, match="angular-rate"):
            load_gyro_csv(path)

    def test_parse_axes(self):
        assert np.allclose(parse_axes(["x", "-y", "-z"]), np.diag([1.0, -1.0, -1.0]))
        assert np.allclose(parse_axes(["-y", "x", "z"]) @ [1.0, 2.0, 3.0], [-2.0, 1.0, 3.0])
        for bad in (["x", "x", "z"], ["x", "y", "-z"], ["x", "y"], ["x", "q", "z"]):
            with pytest.raises(ValueError):
                parse_axes(bad)

    def test_frame_times_roundtrip(self, temp_dir):
        path = temp_dir / "frame_times.csv"
        write_frame_times(path, ["b.jpg", "a.jpg"], [0.5, 0.25])
        assert np.allclose(load_frame_times(path, ["a.jpg", "b.jpg"]), [0.25, 0.5])
        with pytest.raises(ValueError, match="no timestamp"):
            load_frame_times(path, ["c.jpg"])


class TestGyroCalibration:
    def sweep_gyro(self, offset=0.0, seed=0):
        """A gyro with a speed profile that varies, so the clock offset is identifiable."""
        rng = np.random.default_rng(seed)
        t = np.arange(-2.0, 12.0, 0.005)
        speed = 0.6 + 0.4 * np.sin(t * 1.3) + 0.2 * np.sin(t * 3.7)
        omega = np.stack([0.05 * np.sin(t * 2.1), speed, 0.04 * np.cos(t * 1.7)], 1)
        omega += rng.normal(0, 0.002, omega.shape)
        return GyroLog(t=t + offset, omega=omega)

    def test_time_offset(self):
        cam = self.sweep_gyro()
        frame_times = np.arange(0.0, 10.0, 1 / 15)
        rv = gyro_interval_rotvecs(cam, frame_times)
        shifted = GyroLog(t=cam.t + 3.21, omega=cam.omega)
        offset, corr = estimate_time_offset(shifted, frame_times, np.linalg.norm(rv, axis=1))
        assert offset == pytest.approx(3.21, abs=5e-3)
        assert corr > 0.99

    def test_axis_alignment(self):
        rng = np.random.default_rng(4)
        P = parse_axes(["-z", "x", "-y"])
        g = rng.normal(size=(60, 3)) * 0.05
        v = g @ P.T + rng.normal(0, 1e-3, g.shape)
        R, snapped, err = estimate_axis_alignment(g, v)
        assert snapped and np.allclose(R, P) and err < 3

    def test_offset_and_axes_from_rendered_video(self):
        world = textured_world(width=2048)
        spec = SweepSpec(duration_s=6.0, frame_rate_hz=4.0, yaw_span_deg=150.0,
                         image_width=320, image_height=240, gyro_pad_s=1.0)
        sim = simulate_sweep(world, spec)
        device_to_camera = parse_axes(["x", "-y", "-z"])
        raw = GyroLog(t=sim.gyro.t + 7.5, omega=sim.gyro.omega @ device_to_camera)
        feats = detect_all_features(sim.frames, 1500)
        cam, report = calibrate_gyro(feats, sim.calib.K, sim.frame_times, raw, OrientationOptions())
        assert report["time_offset_s"] == pytest.approx(7.5, abs=5e-3)
        assert np.allclose(report["R_cam_imu"], device_to_camera)
        assert np.allclose(cam.omega, sim.gyro.omega, atol=1e-9)


# ---------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------

def render_sweep(seed=7):
    world = textured_world(width=2048)
    spec = SweepSpec(duration_s=12.0, frame_rate_hz=3.0, yaw_span_deg=380.0,
                     image_width=400, image_height=300, hfov_deg=70.0, gyro_pad_s=1.0, seed=seed)
    return simulate_sweep(world, spec)


@pytest.mark.slow
class TestWithoutImu:
    def test_ba_from_the_homography_chain(self):
        sim = render_sweep()
        f0 = sim.calib.fx * 0.97
        init = CalibrationData(fx=f0, fy=f0, cx=sim.calib.cx, cy=sim.calib.cy)
        res, chain = run_orientation_without_imu(sim.frames, init, sim.gt_rotations[0],
                                                 V2Options(orb_nfeatures=1500))
        e_chain = per_frame_errors(chain, sim.gt_rotations)
        e_ba = per_frame_errors(res.rotations, sim.gt_rotations)
        assert e_ba.mean() < 0.2 and e_ba.mean() < e_chain.mean()
        assert res.diagnostics["n_loop_pairs"] > 0
        assert res.focal == pytest.approx(sim.calib.fx, rel=0.01)


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineBundle:
    """run_pipeline with orientation.method 'bundle', without and with a gyro log."""

    def write_frames(self, sim, folder):
        folder.mkdir()
        names = []
        for k, img in enumerate(sim.frames):
            names.append(f"frame_{k:04d}.png")
            cv2.imwrite(str(folder / names[-1]), img)
        return names

    def check(self, config, sim):
        from src.pipeline import run_pipeline
        pano = run_pipeline(config)
        assert pano.exists()
        saved = np.load(config.output_dir / "orientation.npz")
        assert str(saved["method"]) == "bundle"
        # The pipeline expresses everything relative to frame 0
        truth = [R @ sim.gt_rotations[0].T for R in sim.gt_rotations]
        assert per_frame_errors(list(saved["rotations"]), truth).mean() < 0.3
        assert float(saved["focal_px"]) == pytest.approx(sim.calib.fx, rel=0.015)
        refined = CalibrationData.from_json(config.output_dir / "intrinsics_refined.json")
        assert refined.fx == pytest.approx(float(saved["focal_px"]))
        import json
        return json.loads((config.output_dir / "orientation.json").read_text())

    def test_v1_chain_on_frames_narrower_than_match_width(self, temp_dir):
        """match_width only ever shrinks frames; it must not rescale K for small ones."""
        from src.pipeline import run_pipeline
        sim = render_sweep()
        self.write_frames(sim, temp_dir / "frames")
        config = PipelineConfig(
            input_dir=temp_dir / "frames", output_dir=temp_dir / "out",
            intrinsics=IntrinsicsConfig(hfov_deg=70.0),
            output=OutputConfig(pano_width=512, warp_workers=1),
            orientation=OrientationConfig(method="chain"),
        )
        config.matching.match_width = 1600  # wider than the 400 px frames
        config.matching.min_inliers = 40
        config.matching.rotation_smoothing_window = 1
        config.blending.method = "none"
        run_pipeline(config)
        saved = np.load(config.output_dir / "orientation.npz")
        truth = [R @ sim.gt_rotations[0].T for R in sim.gt_rotations]
        # The sweep passes 360°, so v1 trims at the closure; what it keeps is chained
        # without any squeeze. With K scaled for 1600 px the first ten frames were
        # already tens of degrees off; v1's own failures only show near the end.
        rots = list(saved["rotations"])
        assert len(rots) < len(truth)
        assert per_frame_errors(rots[:15], truth[:15]).max() < 2.0

    def test_without_gyro(self, temp_dir):
        sim = render_sweep()
        self.write_frames(sim, temp_dir / "frames")
        config = PipelineConfig(
            input_dir=temp_dir / "frames", output_dir=temp_dir / "out",
            intrinsics=IntrinsicsConfig(hfov_deg=72.0),
            output=OutputConfig(pano_width=512, warp_workers=1),
            orientation=OrientationConfig(method="bundle"),
        )
        config.matching.min_inliers = 40
        config.blending.method = "none"
        config.debug.enabled = True
        config.debug.save_seams = False
        report = self.check(config, sim)
        assert report["prior"] == "homography chain" and report["n_loop_pairs"] > 0
        debug = config.output_dir / "debug" / "matches"
        assert list(debug.glob("match_*.jpg")) and list(debug.glob("loop_*.jpg"))

    def test_with_gyro_on_its_own_clock_and_axes(self, temp_dir):
        sim = render_sweep()
        names = self.write_frames(sim, temp_dir / "frames")
        write_frame_times(temp_dir / "frames" / "frame_times.csv", names, list(sim.frame_times))
        device = sim.gyro.omega @ parse_axes(["x", "-y", "-z"])
        t = sim.gyro.t + 42.0
        (temp_dir / "gyro.csv").write_text(
            "t,wx,wy,wz\n" + "\n".join(f"{a:.6f},{w[0]:.8f},{w[1]:.8f},{w[2]:.8f}"
                                       for a, w in zip(t, device)) + "\n")
        config = PipelineConfig(
            input_dir=temp_dir / "frames", output_dir=temp_dir / "out",
            intrinsics=IntrinsicsConfig(hfov_deg=72.0),
            output=OutputConfig(pano_width=512, warp_workers=1),
            orientation=OrientationConfig(method="bundle"),
            imu=ImuConfig(gyro_csv=temp_dir / "gyro.csv"),
        )
        config.blending.method = "none"
        report = self.check(config, sim)
        assert report["prior"] == "gyroscope"
        assert report["gyro_calibration"]["time_offset_s"] == pytest.approx(42.0, abs=5e-3)


class TestOrientationConfig:
    def write(self, temp_dir, extra):
        path = temp_dir / "c.yaml"
        path.write_text(f"input_dir: {temp_dir}\noutput_dir: {temp_dir / 'o'}\n{extra}")
        return path

    def test_defaults_to_bundle(self, temp_dir):
        config = load_config_from_yaml(self.write(temp_dir, ""))
        assert config.orientation.method == "bundle" and config.imu.gyro_csv is None
        chain = load_config_from_yaml(self.write(temp_dir, "orientation:\n  method: chain\n"))
        assert chain.orientation.method == "chain"

    def test_bundle_with_gyro(self, temp_dir):
        config = load_config_from_yaml(self.write(temp_dir, (
            "orientation:\n  method: bundle\n  loop_pairs_per_frame: 4\n"
            "imu:\n  gyro_csv: g.csv\n  time_offset_s: auto\n  axes: [x, -y, -z]\n")))
        assert config.orientation.method == "bundle"
        assert config.orientation.loop_pairs_per_frame == 4
        assert str(config.imu.gyro_csv) == "g.csv"
        assert config.imu.time_offset_s is None and config.imu.axes == ["x", "-y", "-z"]
        assert config.to_dict()["imu"]["axes"] == ["x", "-y", "-z"]

    def test_fixed_offset_and_auto_axes(self, temp_dir):
        config = load_config_from_yaml(self.write(temp_dir, (
            "orientation:\n  method: bundle\nimu:\n  gyro_csv: g.csv\n  time_offset_s: 0.25\n  axes: auto\n")))
        assert config.imu.time_offset_s == 0.25 and config.imu.axes is None

    def test_rejects_bad_settings(self, temp_dir):
        with pytest.raises(ValueError, match="method"):
            load_config_from_yaml(self.write(temp_dir, "orientation:\n  method: magic\n"))
        with pytest.raises(ValueError, match="Unknown orientation"):
            load_config_from_yaml(self.write(temp_dir, "orientation:\n  loop_pairs: 3\n"))
        with pytest.raises(ValueError, match="gyro"):
            load_config_from_yaml(self.write(temp_dir, "orientation:\n  method: chain\nimu:\n  gyro_csv: g.csv\n"))


class TestFrameTimesFromVideo:
    def test_extraction_writes_timestamps(self, temp_dir):
        from src.video_utils import FRAME_TIMES_FILENAME, extract_frames_from_video
        video = temp_dir / "clip.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        for k in range(20):
            writer.write(np.full((48, 64, 3), k * 10, np.uint8))
        writer.release()
        paths = extract_frames_from_video(video, temp_dir / "frames", method="interval", frame_interval=4)
        times = load_frame_times(temp_dir / "frames" / FRAME_TIMES_FILENAME, [p.name for p in paths])
        assert len(paths) == 5
        assert np.allclose(times, np.arange(5) * 0.4, atol=1e-3)
