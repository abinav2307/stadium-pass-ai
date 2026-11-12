"""The seat-mapping agent.

Claude reads the seating chart image and plain-language seat requests, then
calls three local tools to turn them into camera files for SuperSplat. Claude
decides what to do; the Python tools do all the geometry, so the numbers are exact.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import anthropic

from .align import Similarity, load_landmarks
from .chart import render_chart
from .pipeline import write_seat_files
from .seats import LOOK_AT_CHOICES, CourtPose, load_chart, seat_to_court_pose

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM = """You are the seat-mapping agent for Stadium Pass AI.
A fan asks for a seat in plain language. Pick the exact section, row and seat
from the seating chart, then call the tools to produce camera files.

Rules:
- Only use sections and rows that exist in the chart. Call list_sections first.
- For vague requests ("courtside near the hoop", "somewhere cheap behind the basket"),
  choose the best matching seat and say in one sentence why you chose it.
- Row A is courtside; later rows are further back and higher.
- Choose look_at sensibly: baseline seats usually watch the near hoop; sideline seats watch centre court.
- Call plan_seat once per requested seat, giving each a short snake_case name (e.g. sec112_rowF).
- When every seat is planned, call write_camera_files exactly once.
- Finish with a short summary: each seat, where it is, and the camera folder to import.
"""

TOOLS = [
    {
        "name": "list_sections",
        "description": "List the sections, rows and court size from the seating chart.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "plan_seat",
        "description": "Compute the camera position for one seat, in court and scan coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short snake_case name for this seat"},
                "section": {"type": "string"},
                "row": {"type": "string"},
                "seat_fraction": {
                    "type": "number",
                    "description": "Position along the section: 0 = first seat, 0.5 = middle, 1 = last seat",
                },
                "look_at": {"type": "string", "enum": list(LOOK_AT_CHOICES)},
            },
            "required": ["name", "section", "row"],
        },
    },
    {
        "name": "write_camera_files",
        "description": "Write COLMAP camera files for every planned seat, ready to import into SuperSplat.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


class Session:
    """State for one agent run: the chart, the scan alignment, planned seats and the log."""

    def __init__(self, chart: dict, sim: Similarity, out_dir: Path, drift_m: float, log_path: Path):
        """Set up a run.

        Args:
            chart: The seating chart.
            sim: Court-to-scan transform (identity before you have a scan).
            out_dir: Folder for the camera files.
            drift_m: Sideways head movement per seat, in meters.
            log_path: JSON Lines file that records every step of the run.
        """
        self.chart = chart
        self.sim = sim
        self.out_dir = out_dir
        self.drift_m = drift_m
        self.log_path = log_path
        self.seats: list[tuple[str, CourtPose]] = []

    def log(self, kind: str, payload) -> None:
        """Record one step of the run in the log file and print a short line to the console.

        The log is the agent's trace: what Claude said, which tools it called with
        what inputs, and what came back. It's what you show in the demo video.

        Args:
            kind: Step type, e.g. "request", "claude", "tool_call", "tool_result".
            payload: Anything JSON-serialisable describing the step.
        """
        line = {"t": round(time.time(), 2), "kind": kind, "data": payload}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"[{kind}] {json.dumps(payload, default=str)[:300]}")

    def run_tool(self, name: str, args: dict):
        """Run the local tool Claude asked for and return its result.

        Args:
            name: Tool name from TOOLS.
            args: The inputs Claude supplied.

        Returns:
            A JSON-serialisable result to send back to Claude.

        Raises:
            ValueError: For an unknown tool, or bad inputs such as a section that
                doesn't exist. The caller passes the message back to Claude so it
                can correct itself.
        """
        if name == "list_sections":
            return {
                "court": self.chart["court"],
                "rows": self.chart["rows"]["labels"],
                "sections": [{k: s[k] for k in ("id", "side", "label")} for s in self.chart["sections"]],
            }
        if name == "plan_seat":
            pose = seat_to_court_pose(
                args["section"], args["row"], args.get("seat_fraction", 0.5),
                args.get("look_at", "center_court"), self.chart,
            )
            self.seats.append((args["name"], pose))
            return {
                "name": args["name"],
                "court_pose": pose.to_dict(),
                "scan_eye": [round(v, 4) for v in self.sim.apply_point(pose.eye).tolist()],
            }
        if name == "write_camera_files":
            if not self.seats:
                raise ValueError("No seats planned yet; call plan_seat first")
            return write_seat_files(self.out_dir, self.seats, self.sim, self.drift_m)
        raise ValueError(f"Unknown tool {name}")


def _image_block(path: Path, media_type: str = "image/png") -> dict:
    """Package an image file as a content block Claude can see.

    Args:
        path: Image file to send.
        media_type: The image's MIME type, e.g. "image/png" or "image/jpeg".

    Returns:
        A base64 image content block for the Messages API.
    """
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def run_agent(requests: list[str], landmarks: str | None = None, out_root: str = "runs",
              drift_m: float = 0.15, model: str = DEFAULT_MODEL, max_turns: int = 12) -> Path:
    """Run the agent on one or more seat requests and write camera files for each seat.

    The loop: send Claude the chart image and the requests; whenever Claude asks
    for a tool, run it locally and send the result back; stop when Claude gives
    its final answer or after max_turns rounds.

    Args:
        requests: Plain-language seat requests, e.g. ["Section 112, Row F"].
        landmarks: Optional landmarks JSON to line up with a scan. Without it,
            poses stay in court coordinates, which is fine for testing.
        out_root: Parent folder for runs; each run gets a timestamped subfolder.
        drift_m: Sideways head movement per seat, in meters.
        model: Claude model to use.
        max_turns: Safety limit on request/response rounds.

    Returns:
        The run folder, containing the chart image, log.jsonl and cameras/.
    """
    chart = load_chart()
    sim = load_landmarks(landmarks, chart) if landmarks else Similarity.identity()
    run_dir = Path(out_root) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    session = Session(chart, sim, run_dir / "cameras", drift_m, run_dir / "log.jsonl")

    if landmarks:
        session.log("alignment", {"scale": sim.scale, "rms_error": sim.rms_error})
    else:
        session.log("alignment", "none: poses are in court coordinates (fine for testing)")

    chart_png = render_chart(run_dir / "seating_chart.png", chart)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    user_text = "Seat requests:\n" + "\n".join(f"- {r}" for r in requests)
    messages = [{"role": "user", "content": [_image_block(chart_png), {"type": "text", "text": user_text}]}]
    session.log("request", requests)

    for _ in range(max_turns):
        resp = client.messages.create(model=model, max_tokens=2000, system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                session.log("claude", block.text)
        if resp.stop_reason != "tool_use":
            break
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            session.log("tool_call", {"tool": block.name, "input": block.input})
            try:
                out = session.run_tool(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
                session.log("tool_result", out)
            except Exception as e:  # send the error back so Claude can fix its next call
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True})
                session.log("tool_error", str(e))
        messages.append({"role": "user", "content": results})

    print(f"\nRun saved to {run_dir}")
    return run_dir


def verify_render(image_path: str, request: str, model: str = DEFAULT_MODEL) -> str:
    """Ask Claude whether a rendered frame matches the seat that was requested.

    This is the agent's quality check: after you render a seat in SuperSplat,
    pass the frame here. Claude answers PASS or FAIL with a short reason and,
    on FAIL, what to change.

    Args:
        image_path: A PNG or JPEG frame from the render.
        request: The original seat request, e.g. "Section 112, Row F".
        model: Claude model to use.

    Returns:
        Claude's verdict as text.
    """
    path = Path(image_path)
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    client = anthropic.Anthropic()
    prompt = (
        f"This frame was rendered for the seat request: '{request}'. "
        "Does the viewpoint match (side of the court, distance, height, what it faces)? "
        "Answer PASS or FAIL on the first line, then one or two sentences on why, "
        "and if FAIL, what to change (for example: move back two rows, look at the near hoop)."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": [_image_block(path, media), {"type": "text", "text": prompt}]}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
