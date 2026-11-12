"""Tool 2: line the court up with your 3D scan.

A Gaussian Splat from Scaniverse or Polycam has its own scale, position and
rotation. Read 3-5 landmarks off the scan in SuperSplat (the rim, the
free-throw line centre, a court corner), pair them with the same points in
court coordinates, and this module fits the transform between the two:

    scan_point = scale * R @ court_point + t
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .seats import court_landmarks


@dataclass
class Similarity:
    """A scale, rotation and translation that maps court coordinates into a scan.

    Attributes:
        scale: Scan units per meter.
        R: 3x3 rotation matrix.
        t: Translation vector, in scan units.
        rms_error: Root-mean-square distance between the fitted and the clicked
            landmarks, in scan units. Small means the landmarks agree.
    """

    scale: float
    R: np.ndarray
    t: np.ndarray
    rms_error: float

    def apply_point(self, p) -> np.ndarray:
        """Move a point from court coordinates into the scan (scale, rotate, shift).

        Args:
            p: A point [x, y, z] in meters.

        Returns:
            The same point in scan coordinates.
        """
        return self.scale * self.R @ np.asarray(p, dtype=float) + self.t

    def apply_direction(self, d) -> np.ndarray:
        """Rotate a direction into the scan. Directions ignore scale and shift.

        Args:
            d: A direction vector in court coordinates, e.g. [0, 0, 1] for "up".

        Returns:
            The unit-length direction in scan coordinates.
        """
        v = self.R @ np.asarray(d, dtype=float)
        return v / np.linalg.norm(v)

    @classmethod
    def identity(cls) -> "Similarity":
        """Return a transform that changes nothing.

        Used before you have a scan: camera files then stay in court coordinates,
        which is enough to test the agent end to end.
        """
        return cls(1.0, np.eye(3), np.zeros(3), 0.0)


def fit_similarity(court_pts, splat_pts) -> Similarity:
    """Find the scale, rotation and translation that best map one point set onto another.

    Uses the closed-form least-squares method by Umeyama (1991), so small
    errors in where you clicked the landmarks average out.

    Args:
        court_pts: N x 3 landmark positions in court coordinates (meters).
        splat_pts: N x 3 positions of the same landmarks in the scan.

    Returns:
        The fitted Similarity, including its RMS error.

    Raises:
        ValueError: If the arrays don't match, there are fewer than 3 points,
            or the points are all in one spot or one straight line.
    """
    src = np.asarray(court_pts, dtype=float)
    dst = np.asarray(splat_pts, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Need two N x 3 arrays of matching points")
    if len(src) < 3:
        raise ValueError("Need at least 3 landmark pairs")

    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs**2).sum() / len(src)
    if var_s < 1e-12:
        raise ValueError("Landmarks are all at the same point")

    cov = xd.T @ xs / len(src)
    if np.linalg.matrix_rank(cov) < 2:
        raise ValueError("Landmarks are in a straight line; pick points that form a triangle")
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1  # keep a proper rotation, never a mirror image
    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - scale * R @ mu_s

    pred = (scale * (R @ src.T)).T + t
    rms = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))
    return Similarity(scale, R, t, rms)


def load_landmarks(path: str | Path, chart: dict) -> Similarity:
    """Read a landmarks file and fit the court-to-scan transform.

    File format (see data/landmarks.example.json):
        {"points": [{"name": "east_rim", "splat": [x, y, z]}, ...]}
    "name" must be one of seats.court_landmarks; for any other point give
    "court": [x, y, z] in meters instead.

    Args:
        path: Path to the landmarks JSON.
        chart: The seating chart, used to look up named court landmarks.

    Returns:
        The fitted Similarity.

    Raises:
        ValueError: If a landmark name is unknown and has no explicit "court" position.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    named = court_landmarks(chart)
    court_pts, splat_pts = [], []
    for p in data["points"]:
        if "court" in p:
            court_pts.append(p["court"])
        elif p.get("name") in named:
            court_pts.append(named[p["name"]])
        else:
            raise ValueError(f"Landmark '{p.get('name')}' is not a known court point; add 'court': [x, y, z]")
        splat_pts.append(p["splat"])
    return fit_similarity(court_pts, splat_pts)
