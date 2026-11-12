"""Draw the seating chart as a PNG: a top-down court with labelled sections.

Claude reads this image alongside each seat request, so labels are large and clear.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .seats import load_chart

W, H = 1400, 1000  # image size in pixels
SCALE = 26  # pixels per meter
CX, CY = W // 2, H // 2  # centre court, in pixels

_COLORS = [(66, 133, 244), (52, 168, 83), (234, 67, 53), (251, 188, 5), (155, 89, 182), (0, 150, 136)]


def _font(size: int):
    """Load a bold font at the given size, falling back to Pillow's built-in font.

    Args:
        size: Font size in pixels.

    Returns:
        A Pillow font object.
    """
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _px(x: float, y: float) -> tuple[float, float]:
    """Convert court coordinates (meters) to image pixels.

    East is to the right and north is up the page, like a map.

    Args:
        x: Court x in meters.
        y: Court y in meters.

    Returns:
        (column, row) in pixels.
    """
    return CX + x * SCALE, CY - y * SCALE


def _section_box(section: dict, court: dict, rows: dict) -> list[tuple[float, float]]:
    """Work out the pixel rectangle a section covers, from its front row to its back row.

    Args:
        section: The section's entry from the chart.
        court: The chart's court dimensions.
        rows: The chart's row settings.

    Returns:
        [top-left, bottom-right] corners in pixels.
    """
    hx, hy = court["length"] / 2, court["width"] / 2
    front = rows["front_setback"] - rows["row_depth"] / 2
    back = rows["front_setback"] + len(rows["labels"]) * rows["row_depth"]
    f, t = section["from"], section["to"]
    if section["side"] in ("north", "south"):
        sign = 1 if section["side"] == "north" else -1
        x0, x1 = f * court["length"], t * court["length"]
        y0, y1 = sign * (hy + front), sign * (hy + back)
        return [_px(x0, max(y0, y1)), _px(x1, min(y0, y1))]
    sign = 1 if section["side"] == "east" else -1
    y0, y1 = f * court["width"], t * court["width"]
    x0, x1 = sign * (hx + front), sign * (hx + back)
    return [_px(min(x0, x1), y1), _px(max(x0, x1), y0)]


def render_chart(out_path: str | Path, chart: dict | None = None) -> Path:
    """Draw the court, both hoops and every section, and save it as a PNG.

    Args:
        out_path: Where to save the image.
        chart: The seating chart. Loaded from the default file if omitted.

    Returns:
        The path the image was saved to.
    """
    chart = chart or load_chart()
    c, rows = chart["court"], chart["rows"]
    hx, hy = c["length"] / 2, c["width"] / 2

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    big, mid, small = _font(30), _font(22), _font(16)

    # Court, half-court line and hoops
    d.rectangle([_px(-hx, hy), _px(hx, -hy)], fill=(232, 205, 160), outline="black", width=3)
    d.line([_px(0, hy), _px(0, -hy)], fill="black", width=2)
    rim_x = hx - c["rim_from_baseline"]
    for sx in (1, -1):
        rx, ry = _px(sx * rim_x, 0)
        d.ellipse([rx - 8, ry - 8, rx + 8, ry + 8], outline=(200, 60, 0), width=4)
    d.text(_px(rim_x - 1.2, -1.0), "EAST HOOP", fill="black", font=small, anchor="mm")
    d.text(_px(-rim_x + 1.2, -1.0), "WEST HOOP", fill="black", font=small, anchor="mm")

    # Sections
    for i, s in enumerate(chart["sections"]):
        col = _COLORS[i % len(_COLORS)]
        tint = tuple(int(v + (255 - v) * 0.75) for v in col)
        box = _section_box(s, c, rows)
        d.rectangle(box, fill=tint, outline=col, width=4)
        centre = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
        d.text(centre, f"SECTION {s['id']}", fill="black", font=mid, anchor="mm")

    labels = rows["labels"]
    d.text((CX, 24), f"{chart['venue']}  |  rows {labels[0]} (courtside) to {labels[-1]} (back)",
           fill="black", font=big, anchor="mm")
    d.text((CX, H - 24), "N at top  |  Row A is closest to the court", fill="black", font=small, anchor="mm")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
