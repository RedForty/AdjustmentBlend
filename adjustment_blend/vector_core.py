"""
EXPERIMENTAL — vector-speed signal for adjustment distribution.

This explores replacing the per-scalar-channel velocity signal with a single
*group* speed:

* translate -> Euclidean speed of (tx, ty, tz):  |d position / dt|
* rotate    -> geodesic angular speed of the orientation built from
               (rx, ry, rz) + rotation order:  |d orientation / dt|
* scale     -> Euclidean speed of (sx, sy, sz)

All three axes of a channel group then share one speed signal, so a flat axis
on the adjustment layer still rides whatever motion the *other* axes of that
group are doing below. That removes the need for the smart-neighbor fallback in
the common case (a sibling axis is moving), and falls back to a clean linear
distribution only when the whole group is genuinely static.

Pure Python, no Maya — so it's testable in isolation. Not wired into the Maya
pipeline yet; this is a sandbox for the algorithm.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .core import is_equal, map_from_to, normalize_values

# Maya rotateOrder enum -> axis order string.
ROTATE_ORDER_NAMES = {0: "xyz", 1: "yzx", 2: "zxy", 3: "xzy", 4: "yxz", 5: "zyx"}


# =============================================================================
# Quaternion helpers (w, x, y, z)
# =============================================================================

def _axis_quat(axis: str, angle_rad: float) -> Tuple[float, float, float, float]:
    """Unit quaternion for a rotation of `angle_rad` about a principal axis."""
    half = angle_rad * 0.5
    c, s = math.cos(half), math.sin(half)
    if axis == "x":
        return (c, s, 0.0, 0.0)
    if axis == "y":
        return (c, 0.0, s, 0.0)
    return (c, 0.0, 0.0, s)  # z


def _qmul(a, b):
    """Hamilton product a*b (apply b first, then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def euler_to_quat(rx_deg, ry_deg, rz_deg, order="xyz") -> Tuple[float, float, float, float]:
    """Build a unit quaternion from Euler angles (degrees) and a rotation order.

    `order` may be a Maya rotateOrder int (0..5) or an axis string like "xyz".
    The first axis in the order is applied first, matching Maya.
    """
    if isinstance(order, int):
        order = ROTATE_ORDER_NAMES[order]

    angles = {"x": math.radians(rx_deg), "y": math.radians(ry_deg), "z": math.radians(rz_deg)}

    q = (1.0, 0.0, 0.0, 0.0)  # identity
    for axis in order:  # first axis applied first
        q = _qmul(_axis_quat(axis, angles[axis]), q)
    return q


def _quat_geodesic_angle(q1, q2) -> float:
    """Shortest rotation angle (radians) between two orientation quaternions."""
    dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    # abs() handles quaternion double-cover; clamp guards float error.
    dot = min(1.0, abs(dot))
    return 2.0 * math.acos(dot)


# =============================================================================
# Per-group speed signals
# =============================================================================

def translation_speed(tx: Sequence[float], ty: Sequence[float], tz: Sequence[float]) -> List[float]:
    """Per-frame Euclidean speed of the (tx, ty, tz) point, leading 0."""
    speed = [0.0]
    for i in range(1, len(tx)):
        dx, dy, dz = tx[i] - tx[i - 1], ty[i] - ty[i - 1], tz[i] - tz[i - 1]
        speed.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return speed


def angular_speed(rx, ry, rz, order="xyz") -> List[float]:
    """Per-frame geodesic angular speed of the orientation, leading 0.

    Non-zero whenever *any* contributing axis moves, so a flat axis still sees
    motion if a sibling axis is rotating.
    """
    quats = [euler_to_quat(rx[i], ry[i], rz[i], order) for i in range(len(rx))]
    speed = [0.0]
    for i in range(1, len(quats)):
        speed.append(_quat_geodesic_angle(quats[i - 1], quats[i]))
    return speed


def scale_speed(sx, sy, sz) -> List[float]:
    """Per-frame Euclidean speed of the (sx, sy, sz) scale, leading 0."""
    return translation_speed(sx, sy, sz)


def channel_group_speed(group: str, axes: Dict[str, Sequence[float]], order="xyz") -> List[float]:
    """Dispatch to the right speed signal for a channel group.

    `axes` maps "X"/"Y"/"Z" -> the below-motion value graph for that axis.
    """
    x, y, z = axes["X"], axes["Y"], axes["Z"]
    if group == "translate":
        return translation_speed(x, y, z)
    if group == "rotate":
        return angular_speed(x, y, z, order)
    if group == "scale":
        return scale_speed(x, y, z)
    raise ValueError(f"Unknown channel group: {group}")


def channel_of(attr: str) -> str:
    """Return the channel group for an attribute (``rotateY`` -> ``rotate``)."""
    for group in ("translate", "rotate", "scale"):
        if attr.startswith(group):
            return group
    return ""


def hottest_group(speeds: Dict[str, Sequence[float]], exclude: str = "") -> Optional[Tuple[str, list]]:
    """Pick the moving group with the most motion, for cross-channel fallback.

    Returns ``(group_name, speed)`` for the group (other than ``exclude``) with
    the largest peak speed, or ``None`` if no other group is moving.

    Note: peak speed is compared across groups with different units (cm vs
    radians), so this is a heuristic — in practice it answers "borrow the
    channel that's clearly doing something," which is exactly the rotate-from-
    translate case it's meant for.
    """
    best: Optional[Tuple[str, list, float]] = None
    for name, speed in speeds.items():
        if name == exclude or is_equal(speed):
            continue
        peak = max(speed)
        if best is None or peak > best[2]:
            best = (name, list(speed), peak)
    if best is None:
        return None
    return best[0], best[1]


# =============================================================================
# Distribution driven by a speed signal (with linear fallback)
# =============================================================================

def _segment_weights(speed_slice: Sequence[float]) -> List[float]:
    """Normalized per-frame weights (summing to 100) for one segment.

    If the segment carries motion, weights follow the speed. If it's flat
    (genuinely no motion to ride), fall back to an even *linear* spread with a
    leading zero so the endpoints stay exact -- which also fixes the old
    "hold then jump at the boundary" artifact for motionless segments.
    """
    if not is_equal(speed_slice) and sum(speed_slice) > 0.0:
        return normalize_values(speed_slice)

    n = len(speed_slice)
    if n <= 1:
        return [0.0] * n
    step = 100.0 / (n - 1)
    return [0.0] + [step] * (n - 1)


def distribute_by_speed(
    speed: Sequence[float],
    adjustment_values: Sequence[float],
    key_ranges: Sequence[Tuple[float, float]],
    calculation_range: Sequence[float],
) -> List[float]:
    """Distribute the adjustment across frames, weighted by a group speed signal.

    Same arc-length idea as :func:`adjustment_blend.core.distribute_adjustment`,
    but the weighting comes from a shared per-group speed and flat segments
    distribute linearly instead of collapsing.
    """
    calc = list(calculation_range)
    new_values: List[float] = []
    emitted_frames = set()

    for start_frame, end_frame in key_ranges:
        start_idx = calc.index(start_frame)
        end_idx = calc.index(end_frame) + 1

        weights = _segment_weights(speed[start_idx:end_idx])

        adj_start = adjustment_values[start_idx]
        adj_end = adjustment_values[end_idx - 1]

        running = 0.0
        for i, frame in enumerate(range(int(start_frame), int(end_frame) + 1)):
            running += weights[i]
            value = map_from_to(running, 0.0, 100.0, adj_start, adj_end)
            if frame not in emitted_frames:
                new_values.append(value)
                emitted_frames.add(frame)

    return new_values
