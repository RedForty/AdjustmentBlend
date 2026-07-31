"""Tests for the experimental vector-speed core.

Focus: the case that motivated the smart-neighbor fallback — a flat adjustment
axis while a sibling axis is moving below.
"""

from __future__ import annotations

import math

import pytest

from adjustment_blend import core, vector_core as vc


def _smoothstep_ramp(start, end, n):
    """Ease-in/ease-out ramp: slow-fast-slow, so speed is non-constant."""
    out = []
    for i in range(n):
        u = i / (n - 1)
        s = 3 * u * u - 2 * u * u * u
        out.append(start + (end - start) * s)
    return out


# ---------------------------------------------------------------------------
# Quaternion / angular speed
# ---------------------------------------------------------------------------

def test_euler_to_quat_identity():
    assert vc.euler_to_quat(0, 0, 0) == (1.0, 0.0, 0.0, 0.0)


def test_single_axis_angular_speed_equals_delta_radians():
    # Pure rotateX ramp, 10 deg/frame -> angular speed ~ 10 deg in radians.
    rx = [0.0, 10.0, 20.0, 30.0]
    flat = [0.0, 0.0, 0.0, 0.0]
    sp = vc.angular_speed(rx, flat, flat, "xyz")
    assert sp[0] == 0.0
    for s in sp[1:]:
        assert math.isclose(s, math.radians(10.0), rel_tol=1e-9)


def test_angular_speed_nonzero_when_sibling_axis_moves():
    # rotateY/Z flat, rotateX moving -> orientation IS changing -> speed > 0.
    rx = _smoothstep_ramp(0, 60, 11)
    flat = [0.0] * 11
    sp = vc.angular_speed(rx, flat, flat, "xyz")
    assert not core.is_equal(sp)
    assert max(sp) > 0.0


def test_rotate_order_changes_orientation():
    # xyz vs zyx should differ for a genuinely 3-axis rotation.
    q_xyz = vc.euler_to_quat(30, 40, 50, "xyz")
    q_zyx = vc.euler_to_quat(30, 40, 50, "zyx")
    assert not all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(q_xyz, q_zyx))


# ---------------------------------------------------------------------------
# Translation speed
# ---------------------------------------------------------------------------

def test_translation_speed_is_pythagorean():
    tx = [0.0, 3.0]
    ty = [0.0, 4.0]
    tz = [0.0, 0.0]
    sp = vc.translation_speed(tx, ty, tz)
    assert sp == [0.0, 5.0]


def test_translation_speed_rides_any_axis():
    # tx flat, tz moving -> combined speed tracks tz.
    tx = [0.0] * 5
    tz = [0.0, 1.0, 3.0, 6.0, 10.0]
    sp = vc.translation_speed(tx, tx, tz)
    assert not core.is_equal(sp)


# ---------------------------------------------------------------------------
# distribute_by_speed
# ---------------------------------------------------------------------------

def _single_segment(speed, adj_start, adj_end):
    n = len(speed)
    calc = [float(i) for i in range(n)]
    adjustment = [adj_start] + [0.0] * (n - 2) + [adj_end]
    return vc.distribute_by_speed(speed, adjustment, [(0.0, float(n - 1))], calc)


def test_distribute_endpoints_exact():
    out = _single_segment([0.0, 1.0, 5.0, 2.0, 1.0], 0.0, 90.0)
    assert math.isclose(out[0], 0.0)
    assert math.isclose(out[-1], 90.0)


def test_flat_segment_distributes_linearly_not_held():
    # The old behavior collapsed flat segments (held at start, jumped at the
    # boundary). The new fallback spreads them linearly with exact endpoints.
    out = _single_segment([0.0, 0.0, 0.0, 0.0, 0.0], 0.0, 40.0)
    assert [round(v, 6) for v in out] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_constant_speed_is_linear():
    out = _single_segment([0.0, 1.0, 1.0, 1.0, 1.0], 0.0, 40.0)
    assert [round(v, 6) for v in out] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_channel_of():
    assert vc.channel_of("translateX") == "translate"
    assert vc.channel_of("rotateY") == "rotate"
    assert vc.channel_of("scaleZ") == "scale"
    assert vc.channel_of("visibility") == ""


def test_hottest_group_picks_the_mover():
    speeds = {
        "rotate": [0.0, 0.0, 0.0],          # static
        "translate": [0.0, 2.0, 5.0],       # moving
        "scale": [0.0, 0.0, 0.0],           # static
    }
    picked = vc.hottest_group(speeds, exclude="rotate")
    assert picked is not None
    assert picked[0] == "translate"
    assert picked[1] == [0.0, 2.0, 5.0]


def test_hottest_group_excludes_self_and_flats():
    speeds = {"rotate": [0.0, 1.0, 2.0], "translate": [0.0, 0.0, 0.0]}
    # Everything other than 'rotate' is flat -> nothing to borrow.
    assert vc.hottest_group(speeds, exclude="rotate") is None


def test_hottest_group_prefers_larger_peak():
    speeds = {
        "rotate": [0.0, 0.0, 0.0],
        "translate": [0.0, 1.0, 1.0],
        "scale": [0.0, 9.0, 3.0],
    }
    picked = vc.hottest_group(speeds, exclude="rotate")
    assert picked[0] == "scale"


def test_the_smart_fallback_scenario():
    """THE motivating case: adjustment introduces rotateY, but rotateY is flat
    below while rotateX sweeps with ease-in/ease-out.

    Expectation: the new rotateY curve eases the same way rotateX does
    (rides the orientation's motion) — monotonic, exact endpoints, and clearly
    NOT linear — with no fallback heuristic involved.
    """
    n = 11
    rx_below = _smoothstep_ramp(0, 60, n)      # the sibling axis that's moving
    flat = [0.0] * n                            # rotateY / rotateZ flat below
    speed = vc.angular_speed(rx_below, flat, flat, "xyz")

    calc = [float(i) for i in range(n)]
    adjustment = [0.0] + [0.0] * (n - 2) + [45.0]   # introduce rotateY 0 -> 45
    out = vc.distribute_by_speed(speed, adjustment, [(0.0, float(n - 1))], calc)

    # Endpoints exact.
    assert math.isclose(out[0], 0.0)
    assert math.isclose(out[-1], 45.0)
    # Monotonic (speed is unsigned).
    assert all(b >= a - 1e-9 for a, b in zip(out, out[1:]))
    # Eased, not linear: the midpoint should be near the linear midpoint (the
    # ramp is symmetric) but the quarter point should lag a linear quarter,
    # because rotateX is moving slowly early.
    linear_quarter = 45.0 * 0.25
    assert out[n // 4] < linear_quarter

    # And crucially: the old per-axis signal (rotateY's own flat motion) would
    # have had nothing to ride.
    own_axis_velocity = core.get_velocity_graph(flat)
    assert core.is_equal(own_axis_velocity)  # flat -> the fallback's whole reason to exist
