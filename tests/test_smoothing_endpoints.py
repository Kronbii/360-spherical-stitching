"""Tests for endpoint handling in temporal smoothing and neighbour interpolation."""

import numpy as np
import pytest

from src.features import MatchResult
from src.rotation import (
    interpolate_rotation_from_neighbors,
    reflect_pad_rotations,
    smooth_rotations_temporal,
    unwrapped_yaw_degrees,
)


def ry(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0.0, np.sin(a)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(a), 0.0, np.cos(a)]])


def steps(rots):
    """Per-frame rotation magnitude in degrees."""
    return np.array([
        np.degrees(np.arccos(np.clip((np.trace(rots[i] @ rots[i - 1].T) - 1) / 2, -1, 1)))
        for i in range(1, len(rots))
    ])


def slow_then_fast(n=309):
    """A sweep that idles at the start and is still moving at the end, like a handheld clip."""
    return list(np.concatenate([np.linspace(0, -1, 12), np.linspace(-1, -332.5, n - 12)]))


class TestReflectPad:
    def test_pad_length(self):
        rots = [ry(d) for d in (0, -1, -2, -3, -4)]
        assert len(reflect_pad_rotations(rots, 2)) == len(rots) + 4

    def test_zero_pad_is_a_copy(self):
        rots = [ry(0), ry(-1)]
        assert reflect_pad_rotations(rots, 0) == rots

    def test_reflection_continues_the_motion(self):
        """Padding must extrapolate, not repeat: yaw[-k] == 2*yaw[0] - yaw[k]."""
        rots = [ry(-2.0 * i) for i in range(6)]
        padded = reflect_pad_rotations(rots, 3)
        y = unwrapped_yaw_degrees(padded)
        # first three entries continue the ramp backwards: +6, +4, +2
        assert y[0] == pytest.approx(6.0, abs=1e-6)
        assert y[1] == pytest.approx(4.0, abs=1e-6)
        assert y[2] == pytest.approx(2.0, abs=1e-6)
        # and forwards past the end: -12, -14, -16
        assert y[-1] == pytest.approx(-16.0, abs=1e-6)

    def test_padded_entries_are_rotations(self):
        for R in reflect_pad_rotations([ry(-3.0 * i) for i in range(5)], 3):
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


class TestSmoothingEndpoints:
    def test_no_tear_at_the_final_frame(self):
        """The bug: clamping plus exempt endpoints snapped the last frame several degrees."""
        rots = [ry(a) for a in slow_then_fast()]
        raw_max = steps(rots).max()
        sm = smooth_rotations_temporal(list(rots), window_size=17)
        assert steps(sm)[-1] <= raw_max * 1.2
        assert steps(sm).max() <= raw_max * 1.2

    def test_first_frame_is_preserved_exactly(self):
        rots = [ry(a) for a in slow_then_fast()]
        sm = smooth_rotations_temporal(list(rots), window_size=17)
        assert np.allclose(sm[0], rots[0], atol=1e-12)

    def test_sweep_span_is_kept(self):
        rots = [ry(a) for a in slow_then_fast()]
        sm = smooth_rotations_temporal(list(rots), window_size=17)
        y_raw, y_sm = unwrapped_yaw_degrees(rots), unwrapped_yaw_degrees(sm)
        assert abs(y_sm[-1] - y_raw[-1]) < 1.0

    def test_tail_is_not_dragged_backwards(self):
        """Clamped windows lagged the last frames by degrees; a full window must not."""
        rots = [ry(a) for a in slow_then_fast()]
        sm = smooth_rotations_temporal(list(rots), window_size=17)
        dev = np.abs(unwrapped_yaw_degrees(sm) - unwrapped_yaw_degrees(rots))
        assert dev[-10:].max() < 1.0

    def test_jitter_is_still_removed(self):
        """The point of smoothing must survive the endpoint fix."""
        rng = np.random.default_rng(0)
        clean = np.linspace(0, -300, 200)
        noisy = clean + rng.normal(0, 0.25, clean.size)
        sm = smooth_rotations_temporal([ry(a) for a in noisy], window_size=15)
        wobble = lambda y: np.std(np.diff(y, 2))
        assert wobble(unwrapped_yaw_degrees(sm)) < wobble(noisy) / 3

    def test_output_length_matches_input(self):
        for n in (3, 5, 50):
            rots = [ry(-a) for a in range(n)]
            assert len(smooth_rotations_temporal(list(rots), 17)) == n

    def test_window_larger_than_sequence_is_clamped(self):
        rots = [ry(-a) for a in range(5)]
        out = smooth_rotations_temporal(list(rots), window_size=101)
        assert len(out) == 5
        for R in out:
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_two_frames_are_returned_untouched(self):
        rots = [np.eye(3), ry(-5)]
        assert smooth_rotations_temporal(rots, 17) is rots


class TestNeighbourInterpolation:
    @staticmethod
    def result(i, ok):
        return MatchResult(src_idx=i, dst_idx=i + 1, homography=None if not ok else np.eye(3),
                           inliers=500 if ok else 0, total_matches=600, success=ok)

    def test_averages_both_neighbours_when_available(self):
        """This path used to be unreachable: the previous-neighbour branch always won."""
        results = [self.result(0, True), self.result(1, False), self.result(2, True)]
        rots = [ry(-2.0), np.eye(3), ry(-4.0)]
        out = interpolate_rotation_from_neighbors(results, rots, 1)
        yaw = np.degrees(np.arctan2(out[0, 2], out[2, 2]))
        assert yaw == pytest.approx(-3.0, abs=0.05)      # midway, not -2 or -4

    def test_falls_back_to_previous_when_next_failed(self):
        results = [self.result(0, True), self.result(1, False), self.result(2, False)]
        rots = [ry(-2.0), np.eye(3), np.eye(3)]
        out = interpolate_rotation_from_neighbors(results, rots, 1)
        assert np.allclose(out, ry(-2.0))

    def test_falls_back_to_next_when_previous_failed(self):
        results = [self.result(0, False), self.result(1, False), self.result(2, True)]
        rots = [np.eye(3), np.eye(3), ry(-4.0)]
        out = interpolate_rotation_from_neighbors(results, rots, 1)
        assert np.allclose(out, ry(-4.0))

    def test_identity_when_no_neighbour_is_usable(self):
        results = [self.result(0, False), self.result(1, False), self.result(2, False)]
        rots = [np.eye(3)] * 3
        assert np.allclose(interpolate_rotation_from_neighbors(results, rots, 1), np.eye(3))

    def test_stationary_neighbour_is_not_treated_as_usable(self):
        """An identity rotation carries no information, so it should not be copied."""
        results = [self.result(0, True), self.result(1, False), self.result(2, True)]
        rots = [np.eye(3), np.eye(3), ry(-4.0)]
        out = interpolate_rotation_from_neighbors(results, rots, 1)
        assert np.allclose(out, ry(-4.0))

    def test_handles_the_first_pair(self):
        results = [self.result(0, False), self.result(1, True)]
        rots = [np.eye(3), ry(-3.0)]
        assert np.allclose(interpolate_rotation_from_neighbors(results, rots, 0), ry(-3.0))

    def test_handles_the_last_pair(self):
        results = [self.result(0, True), self.result(1, False)]
        rots = [ry(-3.0), np.eye(3)]
        assert np.allclose(interpolate_rotation_from_neighbors(results, rots, 1), ry(-3.0))
