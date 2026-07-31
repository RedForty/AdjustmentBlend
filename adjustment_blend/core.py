"""
Pure, Maya-independent core of Adjustment Blend.

Everything in this module is plain Python: no ``maya.cmds``, no OpenMaya, no
scene state. That keeps the actual *algorithm* — turning a velocity graph plus
a sparse adjustment into a dense, motion-matched curve — testable in isolation
and reusable from any context (unit tests, CI, headless pipelines).

The Maya-facing layers (:mod:`adjustment_blend.maya_layers`,
:mod:`adjustment_blend.maya_scene`, :mod:`adjustment_blend.pipeline`) are
responsible for reading values out of the scene and writing keyframes back;
they hand plain lists of floats to the functions here.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

# =============================================================================
# Attribute constants
# =============================================================================

CHANNELS = ["translate", "rotate", "scale"]

ATTRIBUTES = [
    "translateX", "translateY", "translateZ",
    "rotateX", "rotateY", "rotateZ",
    "scaleX", "scaleY", "scaleZ",
]

# Short <-> long attribute name mapping.
SHORT_TO_LONG = {
    "tx": "translateX", "ty": "translateY", "tz": "translateZ",
    "rx": "rotateX", "ry": "rotateY", "rz": "rotateZ",
    "sx": "scaleX", "sy": "scaleY", "sz": "scaleZ",
}

ROTATION_ATTRS = {"rotateX", "rotateY", "rotateZ", "rx", "ry", "rz"}
SCALE_ATTRS = {"scaleX", "scaleY", "scaleZ", "sx", "sy", "sz"}


def normalize_attr_name(attr: str) -> str:
    """Return the long form of an attribute name (``tx`` -> ``translateX``)."""
    return SHORT_TO_LONG.get(attr, attr)


# =============================================================================
# Small numeric helpers
# =============================================================================

def is_equal(values: Sequence[float]) -> bool:
    """Return ``True`` if every element is identical (or the list is empty).

    Used throughout to detect "flat" graphs that carry no motion and therefore
    cannot drive a velocity-matched distribution.
    """
    return not values or list(values).count(values[0]) == len(values)


def map_from_to(x: float, a: float, b: float, c: float, d: float) -> float:
    """Linearly remap ``x`` from the range ``[a, b]`` onto ``[c, d]``.

    If the input range is degenerate (``a == b``) the low end ``c`` is returned
    rather than dividing by zero.
    """
    if b == a:
        return c
    return (x - a) / (b - a) * (d - c) + c


def get_velocity_graph(values: Sequence[float]) -> List[float]:
    """Return the per-frame *speed* graph for a value graph.

    The result is the absolute frame-to-frame delta, with a leading ``0.0`` so
    the output lines up index-for-index with the input::

        [10, 10, 14, 20]  ->  [0.0, 0.0, 4.0, 6.0]

    Speed (not signed velocity) is what we want: a control moving fast in any
    direction should soak up more of the adjustment, regardless of sign.
    """
    velocity_graph = [0.0]
    for i in range(1, len(values)):
        velocity_graph.append(abs(values[i] - values[i - 1]))
    return velocity_graph


def normalize_values(values: Sequence[float], normal: float = 100.0) -> List[float]:
    """Scale ``values`` so their absolute sum equals ``normal``.

    Returns a list of zeros when the input sums to zero (a flat graph cannot be
    normalized into proportions).
    """
    total = abs(sum(values))
    if total > 0.0:
        mult = normal / total
        return [x * mult for x in values]
    return [0.0 for _ in values]


def get_float_range(keys: Sequence[float]) -> List[float]:
    """Return the sorted, de-duplicated set of every whole frame spanned by
    ``keys`` plus the key times themselves.

    This is the per-frame sampling grid used to evaluate composite motion
    between the first and last adjustment key.
    """
    whole_frames = [float(x) for x in range(int(min(keys)), int(max(keys)) + 1)]
    grid = set(whole_frames)
    grid.update(keys)
    return sorted(grid)


def get_other_axis(attribute: str) -> List[str]:
    """Return the same attribute on the two *other* axes.

    ``translateX`` -> ``[translateY, translateZ]``. Used by the smart fallback
    to borrow motion from a sibling axis when an attribute's own composite is
    flat.
    """
    axes = ["X", "Y", "Z"]
    this_axis = next(a for a in axes if a in attribute)
    others = [a for a in axes if a != this_axis]
    return [attribute.replace(this_axis, others[0]),
            attribute.replace(this_axis, others[1])]


def get_other_channel(channel: str) -> List[str]:
    """Return the two channels other than the one named in ``channel``.

    ``rotateX`` -> ``[translate, scale]``. Used as a last-resort fallback to
    borrow motion from an entirely different channel.
    """
    channels = list(CHANNELS)
    this_channel = next(c for c in channels if c in channel)
    channels.remove(this_channel)
    return channels


# =============================================================================
# The algorithm: velocity-matched distribution
# =============================================================================

def distribute_adjustment(
    composite_velocity: Sequence[float],
    adjustment_values: Sequence[float],
    key_ranges: Sequence[Tuple[float, float]],
    calculation_range: Sequence[float],
) -> List[float]:
    """Distribute a sparse adjustment across frames, weighted by motion.

    This is the heart of adjustment blending. Given:

    * ``composite_velocity`` — the per-frame speed of the layers *below* the
      adjustment layer, sampled over ``calculation_range``;
    * ``adjustment_values`` — the adjustment layer's value at each frame of
      ``calculation_range`` (typically only the segment endpoints are
      meaningful, since the artist keys just the ends);
    * ``key_ranges`` — the ``(start, end)`` segments between adjustment keys;
    * ``calculation_range`` — the sorted frame grid the two value lists are
      sampled on;

    we walk each segment and hand out the segment's total value change in
    proportion to where motion actually happens. Frames where the lower layers
    move fast absorb a large share of the change; held frames barely move. The
    endpoints always land exactly on the artist's keyed values.

    The returned list contains one value per whole frame from the first key to
    the last, with the shared frame between adjacent segments emitted once.
    """
    new_values: List[float] = []
    emitted_frames = set()

    for start_frame, end_frame in key_ranges:
        start_idx = list(calculation_range).index(start_frame)
        end_idx = list(calculation_range).index(end_frame) + 1

        velocity_slice = composite_velocity[start_idx:end_idx]
        normalized_velocity = normalize_values(velocity_slice)

        adj_start = adjustment_values[start_idx]
        adj_end = adjustment_values[end_idx - 1]

        # Accumulate normalized motion (0 -> 100%) across the segment and map
        # that running percentage onto the segment's value change.
        sum_percentage = 0.0
        for i, frame in enumerate(range(int(start_frame), int(end_frame) + 1)):
            sum_percentage += normalized_velocity[i]
            new_value = map_from_to(sum_percentage, 0.0, 100.0, adj_start, adj_end)

            # The boundary frame is shared by two segments; emit it only once.
            if frame not in emitted_frames:
                new_values.append(new_value)
                emitted_frames.add(frame)

    return new_values
