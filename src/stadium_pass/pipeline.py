"""Glue between the three tools: seat -> court pose -> scan pose -> camera files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .align import Similarity
from .colmap import write_colmap
from .seats import CourtPose

UP = np.array([0.0, 0.0, 1.0])


def seat_cameras(pose: CourtPose, sim: Similarity, name: str, drift_m: float = 0.15) -> list[dict]:
    """Turn one seat into camera keyframes in scan coordinates.

    Produces two keyframes: the seat itself, and the same seat with the head
    moved slightly sideways. Rendering between them gives a 5-10 second clip
    that feels like a seated fan rather than a frozen photo.

    Args:
        pose: The seat's pose in court coordinates.
        sim: Court-to-scan transform (identity before you have a scan).
        name: Base name for the keyframes; "_a" and "_b" are appended.
        drift_m: Sideways head movement in meters. 0 gives one still camera.

    Returns:
        A list of {"name", "eye", "target", "up"} camera dictionaries.
    """
    eye = np.array(pose.eye)
    target = np.array(pose.target)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    lateral = np.cross(UP, forward)
    lateral /= np.linalg.norm(lateral)

    up_scan = sim.apply_direction(UP).tolist()
    tgt_scan = sim.apply_point(target).tolist()
    cams = [{"name": f"{name}_a", "eye": sim.apply_point(eye).tolist(), "target": tgt_scan, "up": up_scan}]
    if drift_m > 0:
        cams.append({
            "name": f"{name}_b",
            "eye": sim.apply_point(eye + lateral * drift_m).tolist(),
            "target": tgt_scan,
            "up": up_scan,
        })
    return cams


def write_seat_files(out_dir: str | Path, seats: list[tuple[str, CourtPose]],
                     sim: Similarity, drift_m: float = 0.15) -> dict:
    """Write camera files for every seat: one folder per seat, plus one with all seats.

    Import one seat's folder at a time into SuperSplat to render that seat's
    clip; the "_all_seats" folder is handy for previewing every seat at once.

    Args:
        out_dir: Folder to write into.
        seats: List of (name, CourtPose) pairs.
        sim: Court-to-scan transform.
        drift_m: Sideways head movement in meters for each seat's second keyframe.

    Returns:
        A mapping from seat name (and "_all_seats") to its images.txt path.
    """
    out = Path(out_dir)
    written = {}
    every = []
    for name, pose in seats:
        cams = seat_cameras(pose, sim, name, drift_m)
        every.extend(cams)
        written[name] = str(write_colmap(out / name, cams))
    written["_all_seats"] = str(write_colmap(out / "_all_seats", every))
    return written
