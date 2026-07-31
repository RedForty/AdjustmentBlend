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


def test_distribute_flat_segment_holds_at_start():
    # No motion below -> nothing to distribute against -> the curve holds at the
    # segment start (a documented limitation handled upstream by smart fallback).
    velocity = [0.0, 0.0, 0.0, 0.0]
    out = _single_segment(velocity, 0.0, 100.0)
    assert all(math.isclose(v, 0.0) for v in out[:-1])
