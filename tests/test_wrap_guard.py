"""Tests for sweep measurement and the wrap-overlap guard."""

import numpy as np
import pytest

from src.circular_match import enforce_open_sweep, head_tail_overlap
from src.rotation import (
    rotation_angle_degrees,
    sweep_span_degrees,
    unwrapped_yaw_degrees,
    yaw_degrees,
)


def ry(deg):
    """Rotation about the vertical axis."""
    a = np.radians(deg)
    return np.array([[np.cos(a), 0.0, np.sin(a)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(a), 0.0, np.cos(a)]])


def sweep(total_deg, n=309):
    """A clean yaw-only sweep of n frames covering total_deg."""
    return [ry(a) for a in np.linspace(0.0, total_deg, n)]


class TestYawMeasurement:
    def test_yaw_of_identity_is_zero(self):
        assert yaw_degrees(np.eye(3)) == pytest.approx(0.0)

    def test_yaw_matches_construction(self):
        for deg in (-170, -90, -12.5, 0, 33, 90, 179):
            assert yaw_degrees(ry(deg)) == pytest.approx(deg, abs=1e-6)

    def test_unwrapped_yaw_passes_180_without_jumping(self):
        y = unwrapped_yaw_degrees(sweep(-332.5))
        assert y[0] == pytest.approx(0.0, abs=1e-6)
        assert y[-1] == pytest.approx(-332.5, abs=1e-3)
        # a wrapped signal would show a ~360° step somewhere
        assert np.abs(np.diff(y)).max() < 5.0

    def test_span_does_not_saturate_at_180(self):
        """The bug this guards: rotation_angle_degrees is arccos-based and caps at 180."""
        rots = sweep(-332.5)
        assert rotation_angle_degrees(rots[-1] @ rots[0].T) < 180.0
        assert sweep_span_degrees(rots) == pytest.approx(332.5, abs=1e-3)

    def test_span_reports_more_than_a_full_turn(self):
        assert sweep_span_degrees(sweep(400.0)) == pytest.approx(400.0, abs=1e-3)

    def test_span_of_single_rotation_is_zero(self):
        assert sweep_span_degrees([np.eye(3)]) == 0.0
        assert sweep_span_degrees([]) == 0.0


class TestHeadTailOverlap:
    def test_no_overlap_for_a_short_sweep(self):
        rots = sweep(-200.0)
        assert head_tail_overlap(rots, rots, 40.0) is None

    def test_detects_frames_landing_on_the_first_frame_arc(self):
        rots = sweep(-332.5)
        hit = head_tail_overlap(rots, rots, 40.0)
        assert hit is not None
        first, last = hit
        assert last == len(rots) - 1
        # 360 - 40 = 320°, reached at ~96% of a 332.5° sweep
        assert first == pytest.approx(int(0.962 * len(rots)), abs=6)


class TestEnforceOpenSweep:
    def test_short_sweep_is_left_alone(self):
        rots = sweep(-250.0)
        out, report = enforce_open_sweep(rots, 40.0)
        assert report is None
        assert all(np.allclose(a, b) for a, b in zip(out, rots))

    def test_overreaching_sweep_is_squeezed_below_the_limit(self):
        rots = sweep(-332.5)
        out, report = enforce_open_sweep(rots, 40.0, margin_deg=2.0)
        assert report is not None
        assert report["span_before"] == pytest.approx(332.5, abs=1e-3)
        assert report["span_after"] == pytest.approx(318.0, abs=0.5)
        assert 0.9 < report["scale"] < 1.0

    def test_squeezing_removes_the_collision(self):
        rots = sweep(-332.5)
        assert head_tail_overlap(rots, rots, 40.0) is not None
        out, _ = enforce_open_sweep(rots, 40.0)
        assert head_tail_overlap(out, out, 40.0) is None

    def test_first_frame_stays_put(self):
        """The first frame is the reference; correction must not move it."""
        rots = sweep(-332.5)
        out, _ = enforce_open_sweep(rots, 40.0)
        assert np.allclose(out[0], rots[0], atol=1e-9)

    def test_output_stays_a_rotation(self):
        out, _ = enforce_open_sweep(sweep(-332.5), 40.0)
        for R in out:
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)

    def test_correction_is_monotonic_in_frame_index(self):
        """Error is spread along the sweep, not dumped at the join."""
        rots = sweep(-332.5)
        out, _ = enforce_open_sweep(rots, 40.0)
        shift = np.abs(unwrapped_yaw_degrees(out) - unwrapped_yaw_degrees(rots))
        assert np.all(np.diff(shift) >= -1e-6)
        assert shift[0] == pytest.approx(0.0, abs=1e-6)
        assert shift[-1] == pytest.approx(14.5, abs=1.0)

    def test_pitch_is_preserved(self):
        """Squeezing spins about the vertical axis, so elevation must not change."""
        rx = lambda d: np.array([[1, 0, 0],
                                 [0, np.cos(np.radians(d)), -np.sin(np.radians(d))],
                                 [0, np.sin(np.radians(d)), np.cos(np.radians(d))]])
        rots = [ry(a) @ rx(6.0) for a in np.linspace(0, -332.5, 309)]
        pitch = lambda R: np.degrees(np.arcsin(np.clip(-R[1, 2], -1, 1)))
        out, _ = enforce_open_sweep(rots, 40.0)
        for a, b in zip(rots, out):
            assert pitch(b) == pytest.approx(pitch(a), abs=1e-6)

    def test_too_few_frames_is_a_no_op(self):
        rots = [np.eye(3), ry(-300.0)]
        out, report = enforce_open_sweep(rots, 40.0)
        assert report is None
        assert out is rots
