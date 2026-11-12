"""Tests for the three geometry tools. None of these call Claude or need an API key."""

import numpy as np
import pytest

from stadium_pass.align import Similarity, fit_similarity
from stadium_pass.colmap import look_at_rotation, rotation_to_quaternion, write_colmap
from stadium_pass.pipeline import seat_cameras
from stadium_pass.seats import court_landmarks, load_chart, seat_to_court_pose

CHART = load_chart()


# ---------- Tool 1: seats ----------

def test_row_a_is_courtside_and_low():
    """Row A sits one front-setback behind the baseline, at seated eye height."""
    p = seat_to_court_pose("112", "A", chart=CHART)
    half_len = CHART["court"]["length"] / 2
    assert p.eye[0] == pytest.approx(half_len + CHART["rows"]["front_setback"])
    assert p.eye[2] == pytest.approx(CHART["rows"]["base_eye_height"])


def test_row_f_is_further_back_and_higher():
    """Row F is five row-depths behind row A, and higher."""
    a = seat_to_court_pose("112", "A", chart=CHART)
    f = seat_to_court_pose("112", "F", chart=CHART)
    assert f.eye[0] - a.eye[0] == pytest.approx(5 * CHART["rows"]["row_depth"])
    assert f.eye[2] > a.eye[2]


def test_height_never_exceeds_cap():
    """No seat is placed above the highest point the scan captured."""
    for row in CHART["rows"]["labels"]:
        assert seat_to_court_pose("110", row, chart=CHART).eye[2] <= CHART["rows"]["max_eye_height"] + 1e-9


def test_sideline_seats_on_correct_side():
    """North and south sections land on the right sideline, and the right half of it."""
    assert seat_to_court_pose("110", "C", chart=CHART).eye[1] > 0  # north
    assert seat_to_court_pose("113", "C", chart=CHART).eye[1] < 0  # south
    assert seat_to_court_pose("110", "C", chart=CHART).eye[0] < 0  # west half
    assert seat_to_court_pose("111", "C", chart=CHART).eye[0] > 0  # east half


def test_near_hoop_for_baseline_seat():
    """A seat behind the east baseline looking at the near hoop faces the east rim."""
    p = seat_to_court_pose("112", "B", look_at="near_hoop", chart=CHART)
    assert p.target == pytest.approx(court_landmarks(CHART)["east_rim"])


def test_bad_section_and_row_raise():
    """Unknown sections and rows raise clear errors instead of guessing."""
    with pytest.raises(ValueError):
        seat_to_court_pose("999", "A", chart=CHART)
    with pytest.raises(ValueError):
        seat_to_court_pose("112", "Z", chart=CHART)


# ---------- Tool 2: alignment ----------

def _random_similarity(seed=0):
    """Make a known scale, rotation and translation to test the fit against.

    Args:
        seed: Random seed, so the test is repeatable.

    Returns:
        (scale, rotation matrix, translation).
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return 0.37, q, np.array([1.5, -2.0, 0.7])


def test_fit_recovers_known_transform():
    """With perfect landmarks, the fit recovers the exact transform."""
    s, R, t = _random_similarity()
    court = np.array(list(court_landmarks(CHART).values()))[:4]
    splat = (s * (R @ court.T)).T + t
    sim = fit_similarity(court, splat)
    assert sim.scale == pytest.approx(s, rel=1e-6)
    assert np.allclose(sim.R, R, atol=1e-6)
    assert sim.rms_error < 1e-6


def test_fit_tolerates_small_click_errors():
    """Slightly mis-clicked landmarks still give a close fit and a small error."""
    s, R, t = _random_similarity(1)
    court = np.array(list(court_landmarks(CHART).values()))[:5]
    splat = (s * (R @ court.T)).T + t + np.random.default_rng(2).normal(scale=0.01, size=(5, 3))
    sim = fit_similarity(court, splat)
    assert sim.scale == pytest.approx(s, rel=0.05)
    assert sim.rms_error < 0.05


def test_collinear_landmarks_rejected():
    """Landmarks in a straight line can't fix a rotation, so they're refused."""
    court = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    with pytest.raises(ValueError):
        fit_similarity(court, court)


# ---------- Tool 3: COLMAP ----------

def test_camera_looks_at_target():
    """The look-at point lands dead centre in front of the camera."""
    eye, target = np.array([16.0, 0.0, 1.6]), np.array([0.0, 0.0, 1.0])
    R = look_at_rotation(eye, target)
    cam_pt = R @ target - R @ eye
    assert cam_pt[2] > 0
    assert np.allclose(cam_pt[:2], 0, atol=1e-9)


def test_world_up_is_image_up():
    """The horizon stays level and the picture isn't upside down."""
    R = look_at_rotation([16.0, 0.0, 1.6], [0.0, 0.0, 1.0])
    # COLMAP image y points down, so world up maps to negative camera y.
    assert (R @ np.array([0, 0, 1.0]))[1] < 0


def test_quaternion_roundtrip():
    """Converting to a quaternion and back gives the same rotation."""
    R = look_at_rotation([3.0, -7.0, 2.0], [0.0, 0.0, 1.0])
    w, x, y, z = rotation_to_quaternion(R)
    back = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(back, R, atol=1e-9)


def test_write_colmap_files(tmp_path):
    """A seat produces two keyframes (seat plus head drift) and a lens file."""
    pose = seat_to_court_pose("112", "F", chart=CHART)
    cams = seat_cameras(pose, Similarity.identity(), "sec112_rowF")
    path = write_colmap(tmp_path, cams)
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == 2
    assert lines[0].split()[-1] == "sec112_rowF_a"
    assert (tmp_path / "cameras.txt").exists()
