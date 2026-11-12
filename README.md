# Stadium Pass AI: seat-mapping prototype

**Goal:** show that one 3D reconstruction of a venue can be rendered from any specific seat, so a team
could sell unlimited, seat-priced virtual tickets beyond the arena's physical capacity.

This repository is the first working slice of that pipeline: a real Gaussian Splat captured on a basketball
court, lined up with real court measurements, and an agent that turns a seat request into an exact camera
position for rendering.

```
"Section 112, Row F"
        |
   Claude (vision) reads the seating chart + request, chooses the seat and the tools to call
        |
   Tool 1  seats.py    section / row / seat  ->  eye position + look-at point (court meters)
   Tool 2  align.py    court meters          ->  scan coordinates (fitted from measured landmarks)
   Tool 3  colmap.py   camera pose           ->  COLMAP images.txt, imported into SuperSplat
        |
   SuperSplat renders that seat's view  ->  `verify` asks Claude whether the render matches the seat
```

---

## Status (as of November 11, 2026)

| Stage | Status |
| --- | --- |
| Reference footage of real seat views (Chase Center) | Done: 7 phone clips, 4 distinct seat views |
| 3D capture of a court | Done: Polycam Gaussian Splat, about 1.9 million Gaussians |
| Court measured on site | Done: 69.1 ft x 43.8 ft (21.06 m x 13.35 m) |
| Scan lined up with court coordinates | Done: 5 landmarks, about 17 cm average error |
| Seat-mapping agent and geometry tools | Done: 14 automated tests passing |
| Camera import into SuperSplat, seat renders | **In progress** |
| Render check (`verify`), seat-map page, demo video | Next |

---

## Build log

### 1. Real seat views as ground truth
Seven phone clips filmed at Chase Center (Golden State Warriors) across two games, covering courtside,
lower bowl and an elevated corner about 20 rows up. They show how differently one game looks from
different seats, which is the product's core promise, and serve as the reference the reconstructions
are compared against.

### 2. Ruled out broadcast and highlight footage
Public NBA and YouTube footage was considered and rejected for two reasons:
- **Technical:** a broadcast shows one moving camera at a time, while rebuilding a new viewpoint needs
  several synchronized cameras at known positions.
- **Rights:** game footage belongs to the league and its broadcast partners.

All footage used here was shot by the founder.

### 3. Capturing a court in 3D
A public park court, with one person holding a still basketball pose as the "player."

| Attempt | Method | Result |
| --- | --- | --- |
| 1 | Scaniverse, continuous video walk-around at seated, standing and overhead heights | On-device splat built and shared for review; not sharp enough |
| 2 | Overlapping still photos, Gaussian Splat trained in the cloud by Polycam | **Two usable splats.** The better one was kept |

The kept scan was exported as a Splat PLY (461 MB, about 1.9 million Gaussians), which is not the same
as an ordinary point-cloud PLY. It was checked in SuperSplat Editor v3.7 (browser-based, local only, not
published). Rendered views from new angles stay sharp on the subject and the court lines; stray "floater"
splats in the sky have been identified for cleanup.

### 4. Measuring the court and lining up the scan
The scan has its own arbitrary scale and orientation. To place a seat accurately, court coordinates
(meters) have to be mapped into scan coordinates.

- **Court size,** measured on site: 21.06 m x 13.35 m.
- **Landmarks** read in SuperSplat with its Sphere Selection tool: the rim (the average of its left and
  right edges) and the four corners of the lane. See `data/landmarks.json`.
- **Fit:** a least-squares similarity transform (scale, rotation, translation; Umeyama, 1991) in
  `align.py`.

**Cross-checks on the measurements**

| Check | Result |
| --- | --- |
| Rim equidistant from both free-throw line ends | 1.559 vs 1.559 scan units |
| Free-throw line at right angles to the lane | Confirmed (dot product ~0) |
| Scale from rim height vs. scale from lane width | 0.326 vs 0.326 units per meter |
| Lane width worked out from the scan | **3.66 m (12-ft lane),** not the NBA's 4.88 m |
| Court "up" maps to scan "up" | (-0.005, 1.000, 0.002) |
| Average landmark error | **17 cm** (6.7 cm without the weakest point) |
| Same fit with left and right swapped | 2.9 m error, so the labelling is confirmed correct |

The weakest landmark is the far baseline corner, which sits near the edge of the captured area. It's kept
for now and will be re-measured.

### 5. The seat-mapping agent
Built and tested before the capture, so the pipeline could be checked end to end in court coordinates.
See "Design decisions" below.

---

## Design decisions

- **Claude decides; Python calculates.** Claude reads the seating chart image and interprets the request.
  All geometry runs in deterministic, tested Python, so camera positions are exact and repeatable.
- **Ticket holders don't need an LLM.** For a purchased seat, the `pose` path goes straight from section
  and row to a camera with no model call. The agent is for seat discovery before purchase ("courtside near
  the hoop," "show me row A vs row F").
- **Errors are fed back to the agent.** An invalid section or row raises a clear error, which goes back to
  Claude so it can correct itself. A test covers this.
- **Every run is logged.** Each step goes to `log.jsonl`: requests, Claude's reasoning, tool calls and results.
- **COLMAP camera format,** a widely supported standard that SuperSplat imports onto its timeline.
- **Head drift.** Each seat gets two keyframes 15 cm apart, so renders look like a seated person rather
  than a frozen photo.
- **Seats stay inside the captured area.** Row heights are capped at the highest point captured (about
  2.4 m), because a splat looks right only near the angles it was captured from.

---

## Known limitations

- **One frozen moment.** The scan is a still pose; moving players need synchronized multi-camera capture.
  That's the next technical milestone.
- **Fence placement.** On this court, a fence sits just behind the baselines, so the baseline sections
  fall outside the captured area. Demo seats will be chosen along the sidelines and inside the fence.
- **The seating chart is a demo layout** around a park court, not a real arena chart.
- **Seat position along a row** is set as a fraction (0 to 1), not a seat number, and 0 means the
  lower-numbered end on the court axis rather than the fan's left.
- **The 3D reconstruction itself** comes from Polycam. This project's own code covers seat mapping,
  alignment, camera generation and render checking.

## Next steps

1. Import a test camera into SuperSplat and confirm its orientation.
2. Choose three seats inside the captured area; render them; check each with `verify`.
3. Seat-map web page: real Chase Center view next to the reconstructed view for each seat.
4. A 2-minute demo video.
5. Later: synchronized multi-camera capture for motion, then live reconstruction.

---

## Project layout

| Path | What it is |
| --- | --- |
| `src/stadium_pass/seats.py` | Tool 1: seat to court position |
| `src/stadium_pass/align.py` | Tool 2: line the court up with a scan |
| `src/stadium_pass/colmap.py` | Tool 3: write camera files |
| `src/stadium_pass/pipeline.py` | Connects the three tools; adds the head drift |
| `src/stadium_pass/agent.py` | Claude agent loop, logging, render check |
| `src/stadium_pass/chart.py` | Draws the seating chart image Claude reads |
| `src/stadium_pass/__main__.py` | Command line |
| `data/park_seating.json` | Measured court size, rows and sections |
| `data/landmarks.json` | Measured landmarks: court and scan positions |
| `tests/` | 14 tests (geometry, alignment, camera files, agent loop); none call the API |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env               # then add your Anthropic API key
pytest
```

## Commands

```bash
python -m stadium_pass chart                                               # draw the seating chart
python -m stadium_pass pose --section 111 --row A --landmarks data/landmarks.json    # one seat, no API call
python -m stadium_pass agent "Section 111, Row C" "courtside near the east hoop" --landmarks data/landmarks.json
python -m stadium_pass verify --image render.png --request "Section 111, Row C"
```

Each agent run writes `runs/<timestamp>/` with the chart Claude saw, `log.jsonl`, and
`cameras/<seat>/images.txt` to import into SuperSplat.

Defaults to `claude-sonnet-5-5`; set `CLAUDE_MODEL` in `.env` to override.

## How this was built

The code was written with AI assistance (Claude), working from the design decisions above. The capture,
court measurements and landmark readings were done by hand and checked numerically, as shown in the
build log.
