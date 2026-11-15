"""Unit tests for :mod:`adjustment_blend.core`.

These run anywhere — no Maya required — which is the whole point of keeping the
algorithm pure. ``pytest`` from the repo root.
"""

from __future__ import annotations

import math

import pytest

from adjustment_blend import core


# ---------------------------------------------------------------------------
# Attribute name helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("short, long", [
    ("tx", "translateX"),
    ("ry", "rotateY"),
    ("sz", "scaleZ"),
])
def test_normalize_attr_name_expands_short_form(short, long):
    assert core.normalize_attr_name(short) == long


def test_normalize_attr_name_passes_through_long_form():
    assert core.normalize_attr_name("translateX") == "translateX"
    assert core.normalize_attr_name("unknown") == "unknown"


def test_get_other_axis():
    assert core.get_other_axis("translateX") == ["translateY", "translateZ"]
    assert core.get_other_axis("rotateY") == ["rotateX", "rotateZ"]
    assert core.get_other_axis("scaleZ") == ["scaleX", "scaleY"]


def test_get_other_channel():
    assert core.get_other_channel("rotateX") == ["translate", "scale"]
    assert core.get_other_channel("translateZ") == ["rotate", "scale"]


# ---------------------------------------------------------------------------
# is_equal
# ---------------------------------------------------------------------------

def test_is_equal_flat_and_empty_are_true():
    assert core.is_equal([])
    assert core.is_equal([5.0, 5.0, 5.0])
    assert core.is_equal([0.0])


def test_is_equal_detects_motion():
    assert not core.is_equal([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# is_still
# ---------------------------------------------------------------------------

def test_is_still_ignores_rounding_noise():
    assert core.is_still([])
    assert core.is_still([0.0, 0.0, 0.0])
    assert core.is_still([0.0, 3e-10, 8e-10, 1e-9])


def test_is_still_detects_motion():
    assert not core.is_still([0.0, 0.0, 0.01])
    assert not core.is_still([0.0, 2.0, 2.0, 2.0])   # constant speed is motion


# ---------------------------------------------------------------------------
# get_velocity_graph
# ---------------------------------------------------------------------------

def test_velocity_graph_leads_with_zero_and_uses_absolute_delta():
    assert core.get_velocity_graph([10, 10, 14, 20]) == [0.0, 0.0, 4.0, 6.0]


def test_velocity_graph_is_unsigned():
    # Moving down counts the same as moving up.
    assert core.get_velocity_graph([0, -5, -5, 0]) == [0.0, 5.0, 0.0, 5.0]


def test_velocity_graph_length_matches_input():
    values = [1.0, 3.0, 2.0, 9.0, 9.0]
    assert len(core.get_velocity_graph(values)) == len(values)


# ---------------------------------------------------------------------------
# normalize_values
# ---------------------------------------------------------------------------

def test_normalize_values_sums_to_normal():
    out = core.normalize_values([1.0, 2.0, 1.0])
    assert math.isclose(sum(out), 100.0)
    assert math.isclose(out[1], 50.0)


def test_normalize_values_flat_returns_zeros():
    assert core.normalize_values([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_normalize_values_custom_normal():
    out = core.normalize_values([1.0, 1.0], normal=1.0)
    assert math.isclose(sum(out), 1.0)


# ---------------------------------------------------------------------------
# map_from_to
# ---------------------------------------------------------------------------

def test_map_from_to_endpoints_and_midpoint():
    assert core.map_from_to(0, 0, 100, 10, 20) == 10
    assert core.map_from_to(100, 0, 100, 10, 20) == 20
    assert core.map_from_to(50, 0, 100, 10, 20) == 15


def test_map_from_to_degenerate_input_range():
    # a == b would divide by zero; we return the low end instead.
    assert core.map_from_to(5, 3, 3, 10, 20) == 10


# ---------------------------------------------------------------------------
# get_float_range
# ---------------------------------------------------------------------------

def test_get_float_range_fills_whole_frames():
    assert core.get_float_range([1, 4]) == [1.0, 2.0, 3.0, 4.0]


def test_get_float_range_includes_subframe_keys():
    out = core.get_float_range([1.0, 2.5, 4.0])
    assert 2.5 in out
    assert out == sorted(out)
    assert out[0] == 1.0 and out[-1] == 4.0


# ---------------------------------------------------------------------------
# distribute_adjustment — the algorithm
# ---------------------------------------------------------------------------

def _single_segment(velocity, adj_start, adj_end):
    """Helper: run a single 0..N segment over a contiguous integer grid."""
    n = len(velocity)
    calc = [float(i) for i in range(n)]
    adjustment = [adj_start] + [0.0] * (n - 2) + [adj_end]
    return core.distribute_adjustment(velocity, adjustment, [(0.0, float(n - 1))], calc)


def test_distribute_endpoints_are_exact():
    # Whatever the motion looks like, the artist's keyed ends must be honored.
    velocity = [0.0, 1.0, 5.0, 2.0, 1.0]
    out = _single_segment(velocity, 0.0, 90.0)
    assert math.isclose(out[0], 0.0)
    assert math.isclose(out[-1], 90.0)


def test_distribute_constant_velocity_is_linear():
    # Uniform motion -> evenly spread adjustment -> a straight ramp.
    velocity = [0.0, 1.0, 1.0, 1.0, 1.0]
    out = _single_segment(velocity, 0.0, 40.0)
    assert [round(v, 6) for v in out] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_distribute_front_loaded_motion_rises_early():
    # All motion happens up front: the value should reach its end almost
    # immediately, then hold.
    velocity = [0.0, 10.0, 0.0, 0.0, 0.0]
    out = _single_segment(velocity, 0.0, 100.0)
    assert math.isclose(out[1], 100.0)
    assert math.isclose(out[-1], 100.0)


def test_distribute_back_loaded_motion_rises_late():
    # All motion happens at the end: the value holds at the start, then jumps.
    velocity = [0.0, 0.0, 0.0, 0.0, 10.0]
    out = _single_segment(velocity, 0.0, 100.0)
    assert math.isclose(out[0], 0.0)
    assert math.isclose(out[3], 0.0)
    assert math.isclose(out[-1], 100.0)


def test_distribute_is_monotonic_for_monotonic_target():
    # Speed is unsigned, so accumulation never goes backwards: a 0 -> 100
    # adjustment should produce a non-decreasing curve.
    velocity = [0.0, 3.0, 1.0, 4.0, 1.0, 5.0]
    out = _single_segment(velocity, 0.0, 100.0)
    assert all(b >= a - 1e-9 for a, b in zip(out, out[1:]))


def test_distribute_multi_segment_lands_on_each_key():
    # Two segments: 0 -> 50 over frames 0..2, then 50 -> 60 over frames 2..4.
    calc = [0.0, 1.0, 2.0, 3.0, 4.0]
    velocity = [0.0, 1.0, 1.0, 1.0, 1.0]
    adjustment = [0.0, 0.0, 50.0, 0.0, 60.0]
    key_ranges = [(0.0, 2.0), (2.0, 4.0)]
    out = core.distribute_adjustment(velocity, adjustment, key_ranges, calc)

    # One value per whole frame, shared boundary emitted once.
    assert len(out) == 5
    assert math.isclose(out[0], 0.0)
    assert math.isclose(out[2], 50.0)   # boundary key
    assert math.isclose(out[-1], 60.0)


def test_distribute_flat_segment_spreads_evenly():
    # No motion below -> nothing to distribute against -> an even spread that
    # still lands on the end key. (It used to hold at the segment start and
    # overwrite the end key with the start value.)
    velocity = [0.0, 0.0, 0.0, 0.0, 0.0]
    out = _single_segment(velocity, 0.0, 40.0)
    assert [round(v, 6) for v in out] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_distribute_second_segment_ignores_motion_before_its_first_key():
    # Constant speed, keys at 0, 4 and 8. The speed sample at frame 4 is the
    # motion from frame 3 to 4: it belongs to the first segment. Counting it in
    # the second gave 10, 14, 16, 18, 20 instead of even steps.
    calc = [float(i) for i in range(9)]
    velocity = [0.0] + [1.0] * 8
    adjustment = [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 20.0]
    out = core.distribute_adjustment(velocity, adjustment, [(0.0, 4.0), (4.0, 8.0)], calc)
    assert [round(v, 6) for v in out] == [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]


def test_distribute_hold_in_second_segment_does_not_pop():
    # The motion stops at the middle key. The second segment has nothing to
    # follow, so it spreads evenly. It used to put the whole change on the
    # frame after the key: 10, 20, 20, 20, 20.
    calc = [float(i) for i in range(9)]
    velocity = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    adjustment = [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 20.0]
    out = core.distribute_adjustment(velocity, adjustment, [(0.0, 4.0), (4.0, 8.0)], calc)
    assert [round(v, 6) for v in out[4:]] == [10.0, 12.5, 15.0, 17.5, 20.0]


def test_distribute_hold_in_first_segment_keeps_the_middle_key():
    # Nothing moves before the middle key. That key was set to 10 and used to
    # come out as 0.
    calc = [float(i) for i in range(9)]
    velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    adjustment = [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 20.0]
    out = core.distribute_adjustment(velocity, adjustment, [(0.0, 4.0), (4.0, 8.0)], calc)
    assert [round(v, 6) for v in out] == [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]


def test_distribute_noise_on_a_still_segment_is_not_followed():
    # A still channel with rounding noise: following the noise would hand out
    # the change in uneven steps. It must spread evenly instead.
    velocity = [0.0, 9e-10, 1e-10, 7e-10, 2e-10]
    out = _single_segment(velocity, 0.0, 40.0)
    assert [round(v, 6) for v in out] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_segment_weights_sum_to_100_and_skip_the_first_sample():
    weights = core.segment_weights([7.0, 1.0, 3.0])
    assert weights[0] == 0.0
    assert math.isclose(sum(weights), 100.0)
    assert [round(w, 6) for w in weights] == [0.0, 25.0, 75.0]
