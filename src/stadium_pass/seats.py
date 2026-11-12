"""Tool 1: turn a section, row and seat into a camera position in court coordinates.

Court frame used everywhere in this project:
    origin  centre court, on the floor
    +x      toward the east hoop
    +y      toward the north sideline
    +z      up
All distances are in meters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_CHART = Path(__file__).resolve().parents[2] / "data" / "park_seating.json"

# For each side of the court: the direction pointing away from the court, and
# the axis (0 = x, 1 = y) that a section on that side runs along.
_SIDES = {
    "north": {"outward": np.array([0.0, 1.0, 0.0]), "along": 0},
    "south": {"outward": np.array([0.0, -1.0, 0.0]), "along": 0},
    "east": {"outward": np.array([1.0, 0.0, 0.0]), "along": 1},
    "west": {"outward": np.array([-1.0, 0.0, 0.0]), "along": 1},
}

LOOK_AT_CHOICES = ("center_court", "east_hoop", "west_hoop", "near_hoop", "far_hoop")


@dataclass
class CourtPose:
    """Where a seated fan's eyes are and what they look at, in court coordinates.

    Attributes:
        section: Section id from the seating chart, e.g. "112".
        row: Row label, e.g. "F".
        seat_fraction: Position along the section, 0.0 (first seat) to 1.0 (last seat).
        eye: Eye position [x, y, z] in meters.
        target: Point the fan is looking at [x, y, z] in meters.
        look_at: Name of the look-at choice used, e.g. "near_hoop".
    """

    section: str
    row: str
    seat_fraction: float
    eye: list[float]
    target: list[float]
    look_at: str

    def to_dict(self) -> dict:
        """Return the pose as a plain dictionary, ready for JSON or a tool result."""
        return asdict(self)


def load_chart(path: str | Path = DEFAULT_CHART) -> dict:
    """Read the seating chart JSON (court size, rows and sections).

    Args:
        path: Path to the chart file. Defaults to data/park_seating.json.

    Returns:
        The chart as a dictionary.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def court_landmarks(chart: dict) -> dict[str, list[float]]:
    """List named points on the court that are easy to find in a 3D scan.

    These are the reference points used to line the court up with a scan
    (see align.load_landmarks): rims, free-throw line centres, corners and
    centre court.

    Args:
        chart: The seating chart, used for court length, width and rim position.

    Returns:
        A mapping from landmark name to its [x, y, z] position in meters.
    """
    c = chart["court"]
    hx, hy = c["length"] / 2, c["width"] / 2
    rim_x = hx - c["rim_from_baseline"]
    ft_x = hx - c["free_throw_from_baseline"]
    return {
        "center_court": [0.0, 0.0, 0.0],
        "east_rim": [rim_x, 0.0, c["rim_height"]],
        "west_rim": [-rim_x, 0.0, c["rim_height"]],
        "east_free_throw_center": [ft_x, 0.0, 0.0],
        "west_free_throw_center": [-ft_x, 0.0, 0.0],
        "corner_northeast": [hx, hy, 0.0],
        "corner_southeast": [hx, -hy, 0.0],
        "corner_northwest": [-hx, hy, 0.0],
        "corner_southwest": [-hx, -hy, 0.0],
    }


def _find_section(chart: dict, section: str) -> dict:
    """Look up one section in the chart by its id (case and whitespace ignored).

    Args:
        chart: The seating chart.
        section: Section id to find, e.g. "112".

    Returns:
        The section's entry from the chart.

    Raises:
        ValueError: If no section has that id. The message lists valid ids so
            the agent can correct itself.
    """
    for s in chart["sections"]:
        if s["id"].lower() == str(section).strip().lower():
            return s
    ids = ", ".join(s["id"] for s in chart["sections"])
    raise ValueError(f"Unknown section '{section}'. Valid sections: {ids}")


def _row_index(chart: dict, row: str) -> int:
    """Convert a row label into its position from the front (A = 0).

    Args:
        chart: The seating chart.
        row: Row label, e.g. "F" or "f".

    Returns:
        Zero-based row index.

    Raises:
        ValueError: If the row is not in the chart. The message lists valid rows.
    """
    labels = [r.upper() for r in chart["rows"]["labels"]]
    r = str(row).strip().upper()
    if r not in labels:
        raise ValueError(f"Unknown row '{row}'. Valid rows: {', '.join(labels)}")
    return labels.index(r)


def _look_at_point(chart: dict, side: str, look_at: str) -> np.ndarray:
    """Work out the point a fan in a given seat is looking at.

    "near_hoop" and "far_hoop" depend on where the seat is: for a seat behind
    the east baseline the near hoop is the east one. Sideline seats have no
    near or far hoop, so those fall back to centre court.

    Args:
        chart: The seating chart, used for rim position and height.
        side: Which side of the court the seat is on: north, south, east or west.
        look_at: One of LOOK_AT_CHOICES.

    Returns:
        The look-at point [x, y, z] in meters.

    Raises:
        ValueError: If look_at is not a known choice.
    """
    c = chart["court"]
    rim_x = c["length"] / 2 - c["rim_from_baseline"]
    if look_at == "center_court":
        return np.array([0.0, 0.0, 1.0])  # about chest height of a player
    if look_at == "east_hoop":
        return np.array([rim_x, 0.0, c["rim_height"]])
    if look_at == "west_hoop":
        return np.array([-rim_x, 0.0, c["rim_height"]])
    if look_at in ("near_hoop", "far_hoop"):
        if side in ("east", "west"):
            near = "east_hoop" if side == "east" else "west_hoop"
            far = "west_hoop" if side == "east" else "east_hoop"
            return _look_at_point(chart, side, near if look_at == "near_hoop" else far)
        return _look_at_point(chart, side, "center_court")
    raise ValueError(f"Unknown look_at '{look_at}'. Choose one of {LOOK_AT_CHOICES}")


def seat_to_court_pose(
    section: str,
    row: str,
    seat_fraction: float = 0.5,
    look_at: str = "center_court",
    chart: dict | None = None,
) -> CourtPose:
    """Compute where a fan's eyes are for a given seat, and what they face.

    The seat starts on the court edge for its section's side, moves outward by
    the front setback plus one row depth per row, and rises by the row rise per
    row (capped at the chart's maximum eye height, so every seat stays inside
    the height range captured in the scan).

    Args:
        section: Section id, e.g. "112".
        row: Row label, e.g. "F". Row A is courtside.
        seat_fraction: Position along the section: 0.0 = first seat,
            0.5 = middle, 1.0 = last seat. Values outside 0-1 are clamped.
        look_at: One of LOOK_AT_CHOICES.
        chart: The seating chart. Loaded from the default file if omitted.

    Returns:
        The seat's CourtPose.

    Raises:
        ValueError: For an unknown section, row or look_at choice.
    """
    chart = chart or load_chart()
    sec = _find_section(chart, section)
    idx = _row_index(chart, row)
    rows = chart["rows"]
    court = chart["court"]
    side = _SIDES[sec["side"]]

    frac = float(np.clip(seat_fraction, 0.0, 1.0))
    along_frac = sec["from"] + frac * (sec["to"] - sec["from"])
    side_len = court["length"] if side["along"] == 0 else court["width"]

    # Start on the court edge for this side, at the right spot along it.
    half = np.array([court["length"] / 2, court["width"] / 2, 0.0])
    eye = side["outward"] * half
    eye[side["along"]] = along_frac * side_len

    # Move back from the court and up, one step per row.
    setback = rows["front_setback"] + idx * rows["row_depth"]
    eye = eye + side["outward"] * setback
    eye[2] = min(rows["base_eye_height"] + idx * rows["row_rise"], rows["max_eye_height"])

    target = _look_at_point(chart, sec["side"], look_at)
    return CourtPose(
        section=sec["id"],
        row=rows["labels"][idx],
        seat_fraction=frac,
        eye=[round(float(v), 4) for v in eye],
        target=[round(float(v), 4) for v in target],
        look_at=look_at,
    )
