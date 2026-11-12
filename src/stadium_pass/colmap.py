"""Tool 3: write camera poses as COLMAP text files that SuperSplat can import.

SuperSplat's Timeline imports a COLMAP images.txt file and turns each camera
into a keyframe. COLMAP stores world-to-camera poses, and its cameras look
along +Z with +X to the right and +Y down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def look_at_rotation(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Build the rotation for a camera at `eye` pointing at `target`.

    Args:
        eye: Camera position [x, y, z].
        target: Point the camera faces [x, y, z].
        up: Which way is up in the world, so the horizon stays level.

    Returns:
        A 3x3 world-to-camera rotation. Its rows are the camera's right, down
        and forward directions, expressed in world coordinates.

    Raises:
        ValueError: If the camera position and the target are the same point.
    """
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    n = np.linalg.norm(z)
    if n < 1e-9:
        raise ValueError("Camera position and look-at point are the same")
    z /= n
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:  # looking straight up or down: pick any sideways axis
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a unit quaternion, the form COLMAP stores.

    Uses the numerically stable branch for whichever diagonal term is largest.

    Args:
        R: A 3x3 rotation matrix.

    Returns:
        Quaternion [w, x, y, z] with w >= 0.
    """
    m = np.asarray(R, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def pose_line(image_id: int, name: str, eye, target, up) -> str:
    """Format one camera as an entry in COLMAP's images.txt.

    Args:
        image_id: 1-based camera number; SuperSplat orders keyframes by it.
        name: Label for the camera, e.g. "sec112_rowF_a".
        eye: Camera position.
        target: Point the camera faces.
        up: World up direction.

    Returns:
        Two lines of text: the pose, then an empty line (COLMAP's slot for
        2D feature points, which we don't need).
    """
    R = look_at_rotation(eye, target, up)
    t = -R @ np.asarray(eye, dtype=float)
    qw, qx, qy, qz = rotation_to_quaternion(R)
    vals = " ".join(f"{v:.6f}" for v in (qw, qx, qy, qz, *t))
    return f"{image_id} {vals} 1 {name}\n\n"


def write_colmap(out_dir: str | Path, cameras: list[dict], width: int = 1920,
                 height: int = 1080, fov_deg: float = 60.0) -> Path:
    """Write images.txt (the poses) and cameras.txt (the lens) into a folder.

    Args:
        out_dir: Folder to write into; created if missing.
        cameras: List of {"name", "eye", "target", "up"}, in scan coordinates.
        width: Image width in pixels.
        height: Image height in pixels.
        fov_deg: Horizontal field of view in degrees. 60 is close to a phone's main lens.

    Returns:
        Path to the images.txt file to import into SuperSplat.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    focal = (width / 2) / np.tan(np.radians(fov_deg) / 2)
    (out / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {focal:.3f} {focal:.3f} {width / 2:.1f} {height / 2:.1f}\n",
        encoding="utf-8",
    )
    lines = [
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "# POINTS2D[] (empty)\n",
    ]
    for i, cam in enumerate(cameras, start=1):
        lines.append(pose_line(i, cam["name"], cam["eye"], cam["target"], cam["up"]))
    path = out / "images.txt"
    path.write_text("".join(lines), encoding="utf-8")
    return path
