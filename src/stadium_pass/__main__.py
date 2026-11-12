"""Command line: python -m stadium_pass <command> ...

  chart                          draw the seating chart PNG
  pose    --section 112 --row F  compute one seat without calling Claude (no API key needed)
  agent   "Section 112, Row F"   let Claude plan seats and write camera files
  verify  --image f.png --request "..."   ask Claude if a render matches the seat
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from .align import Similarity, load_landmarks
from .chart import render_chart
from .pipeline import write_seat_files
from .seats import LOOK_AT_CHOICES, load_chart, seat_to_court_pose


def _build_parser() -> argparse.ArgumentParser:
    """Define the four commands and their options.

    Returns:
        The configured argument parser.
    """
    p = argparse.ArgumentParser(prog="stadium_pass", description="Stadium Pass AI seat-mapping agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chart", help="Draw the seating chart PNG")
    c.add_argument("--out", default="data/park_seating.png")

    q = sub.add_parser("pose", help="Compute one seat's camera with no API call")
    q.add_argument("--section", required=True)
    q.add_argument("--row", required=True)
    q.add_argument("--seat", type=float, default=0.5, help="0 = first seat, 1 = last, 0.5 = middle")
    q.add_argument("--look-at", default="center_court", choices=LOOK_AT_CHOICES)
    q.add_argument("--landmarks", help="Landmarks JSON to line up with your scan")
    q.add_argument("--out", default="runs")

    a = sub.add_parser("agent", help="Let Claude plan seats from plain-language requests")
    a.add_argument("requests", nargs="+", help='e.g. "Section 112, Row F" "courtside near the east hoop"')
    a.add_argument("--landmarks", help="Landmarks JSON to line up with your scan")
    a.add_argument("--out", default="runs")
    a.add_argument("--drift", type=float, default=0.15, help="Sideways head drift in meters (0 = still)")

    v = sub.add_parser("verify", help="Ask Claude if a rendered frame matches the requested seat")
    v.add_argument("--image", required=True)
    v.add_argument("--request", required=True)
    return p


def _run_pose(args: argparse.Namespace) -> None:
    """Handle the `pose` command: one seat, computed locally, written as camera files.

    Useful for checking the geometry without an API key or any cost.

    Args:
        args: Parsed command-line options.
    """
    chart = load_chart()
    sim = load_landmarks(args.landmarks, chart) if args.landmarks else Similarity.identity()
    pose = seat_to_court_pose(args.section, args.row, args.seat, args.look_at, chart)
    name = f"sec{pose.section}_row{pose.row}"
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    files = write_seat_files(out / "cameras", [(name, pose)], sim)
    print(json.dumps({"court_pose": pose.to_dict(), "alignment_rms": sim.rms_error, "files": files}, indent=2))


def main() -> None:
    """Entry point: load the API key from .env, parse the command, and run it."""
    load_dotenv()
    args = _build_parser().parse_args()

    if args.cmd == "chart":
        print(render_chart(args.out))
    elif args.cmd == "pose":
        _run_pose(args)
    elif args.cmd == "agent":
        from .agent import run_agent  # imported here so `chart` and `pose` work without the API package configured
        run_agent(args.requests, args.landmarks, args.out, args.drift)
    elif args.cmd == "verify":
        from .agent import verify_render
        print(verify_render(args.image, args.request))


if __name__ == "__main__":
    main()
