"""Where the garden boundaries are messy: plain counts per area, worst first (PLAN K).

    python boundary_score.py <knoxmap map dir> [--top N] [--pictures]

Reads the boundaries plots2.py wrote (<map>_fences.geojson, source=plots; a
front is one of FRONT_STYLES, which plots2 gives only to fronts) and the
plots (<map>_plots.npz: street, buildings). Counts, for every boundary line:

- zigzags: length of line laid in short steps (pieces under ZIGZAG_MIN)
  where it runs at an angle Build cannot draw as a line (between 1:5 and
  6:7, not 2:1 or 45 degrees within LATTICE_TOL): a staircase of straight
  pieces, a zigzag in game (a front along a street at 1:3, or curving).
  Shallower is a straight run with a jog now and then, as vanilla's.
  Long lattice runs along such a line (plots2.lattice_runs) are not;
- bends: a vertex where the line turns (more than BEND_DEG), on the line
  simplified to STEP tiles (a staircase - even with steps of 2 or 3 - is
  the line it follows, counted as a zigzag if off the lattice, not a bend a
  step);
- short pieces: a stretch between bends under SHORT tiles;
- loose ends: an end meeting no other line, building or street (a fence
  stopping in the middle of a garden);
- doubles: length of line within DOUBLE tiles of another, parallel line;
- fronts off the street: length of front line more than OFF tiles from the
  street (a front stepping back and forth instead of following the street);
- in the street: length of front line inside the street's own shape by more
  than INSIDE tiles (<map>_street.wkb, plots2.py: the road graph's outline),
  and of any boundary on the carriageway's core (the road graph's own road,
  <map>_roadgraph_tiles.npz, a square in from its edge). Both should be 0.

The street is plots2.py's (the road graph's outline on a road-graph map),
not the painted ground.

Each count is summed into AREA x AREA tile squares; an area's mess is the
sum of the counts (doubles and off-street in tiles / 4) per 100 tiles of
boundary in it. The worst areas are listed with their world coordinates (to
visit in game) and, with --pictures, drawn top-down (ground, lines by kind,
the counted faults marked) into Building_types/out/boundaries/.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, Point

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from buildings import blocks as B  # noqa: E402
from ground.plotlib import FRONT_STYLES  # noqa: E402

BEND_DEG = 10.0
STEP = 1.6
LATTICE_TOL = 0.12          # a line's slope within this share of 0, 1/2, 1 or 2 (either way) is on the lattice
ZIGZAG_MIN = 3.0            # tiles: shorter stretches are counted as bends, not zigzags
SHORT = 2.0
DOUBLE = 1.2
OFF = 1.5
INSIDE = 0.3
AREA = 20
CELL = 300
OUT = os.path.join(HERE, "..", "..", "out", "boundaries")


def turn(a, b, c) -> float:
    """Degrees the line a-b-c turns at b."""
    d1 = math.atan2(b[1] - a[1], b[0] - a[0])
    d2 = math.atan2(c[1] - b[1], c[0] - b[0])
    return abs((math.degrees(d2 - d1) + 180) % 360 - 180)


def on_lattice(dx: float, dy: float) -> bool:
    """A direction Build draws as a line: along the grid, 45 degrees or 2:1."""
    lo, hi = sorted((dx, dy))
    r = lo / hi if hi else 0.0
    return any(abs(r - v) <= LATTICE_TOL * max(v, 0.5) for v in (0.0, 0.5, 1.0))


def score(map_dir: str) -> dict:
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    bb = info["bbox"]
    proj = B.Projector.build(bb["south"], bb["west"], bb["north"], bb["east"], info["meters_per_tile"],
                             info.get("rotation") or 0.0)
    z = np.load(os.path.join(map_dir, f"{name}_plots.npz"))
    street, building = z["street"], z["building"]
    H, W = street.shape
    lines = []                                         # (LineString, style)
    for ft in json.load(open(os.path.join(map_dir, f"{name}_fences.geojson"), encoding="utf-8"))["features"]:
        if ft["properties"].get("source") != "plots":
            continue
        pts = [proj.to_px(lat, lon) for lon, lat in ft["geometry"]["coordinates"]]
        if len(pts) >= 2:
            lines.append((LineString(pts), ft["properties"].get("knox_style", "")))
    tree = shapely.STRtree([ln for ln, _ in lines])
    # Distance (tiles) from each square to the street, and to anything an end may meet.
    to_street = ndimage.distance_transform_edt(~street)
    to_solid = ndimage.distance_transform_edt(~(street | building))

    def at(grid, p):
        x, y = min(max(int(p[0]), 0), W - 1), min(max(int(p[1]), 0), H - 1)
        return grid[y, x]

    faults = {"zigzag": [], "bend": [], "short": [], "end": [], "double": [], "off": [], "inside": [], "core": []}
    shape_path = os.path.join(map_dir, f"{name}_street.wkb")
    street_shape = shapely.from_wkb(open(shape_path, "rb").read()) if os.path.exists(shape_path) else None
    if street_shape is not None:
        street_shape = street_shape.buffer(-INSIDE)
        shapely.prepare(street_shape)
    tiles_path = os.path.join(map_dir, f"{name}_roadgraph_tiles.npz")
    core = None
    if os.path.exists(tiles_path):
        core = ndimage.binary_erosion(np.load(tiles_path)["owner"][:H, :W] != 0, structure=np.ones((3, 3), bool))   # (x, y, weight)
    total = []                                                              # (x, y, length)
    for i, (ln, st) in enumerate(lines):
        c = list(ln.coords)
        for a, b in zip(c, c[1:]):
            n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1])))
            for k in range(n):
                t = (k + 0.5) / n
                total.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, math.hypot(b[0] - a[0], b[1] - a[1]) / n))
        c = list(shapely.simplify(ln, STEP).coords)
        cut = [0] + [j for j in range(1, len(c) - 1) if turn(c[j - 1], c[j], c[j + 1]) > BEND_DEG] + [len(c) - 1]
        for j in cut[1:-1]:
            faults["bend"].append((*c[j], 1.0))
        raw = list(ln.coords)
        trend = list(shapely.simplify(ln, STEP).coords)
        for a, b in zip(trend, trend[1:]):
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            lo_, hi_ = sorted((dx, dy))
            if math.hypot(dx, dy) < ZIGZAG_MIN or on_lattice(dx, dy) or not (0.2 <= lo_ / max(hi_, 1e-9) <= 0.85):
                continue                       # shallower than 1:5 is a jog now and then, as vanilla's fences
            stretch = LineString([a, b]).buffer(1.6)
            for p0, p1 in zip(raw, raw[1:]):
                n = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
                if 0 < n < ZIGZAG_MIN and stretch.contains(Point(mid)):
                    faults["zigzag"].append((*mid, n))
        for j0, j1 in zip(cut, cut[1:]):
            piece = LineString(c[j0:j1 + 1])
            if piece.length < SHORT:
                mid = piece.interpolate(0.5, normalized=True)
                faults["short"].append((mid.x, mid.y, 1.0))
        if not ln.is_ring:
            for e in (c[0], c[-1]):
                pe = Point(e)
                near = [int(j) for j in tree.query(pe.buffer(0.3)) if int(j) != i]
                if not any(lines[j][0].distance(pe) < 0.3 for j in near) and at(to_solid, e) > 1.5:
                    faults["end"].append((*e, 1.0))
        # Doubles: sample points with another line within DOUBLE running the same way.
        for d in np.arange(0.5, ln.length, 1.0):
            p = ln.interpolate(d)
            q = ln.interpolate(min(ln.length, d + 0.5))
            heading = math.atan2(q.y - p.y, q.x - p.x) if (q.x, q.y) != (p.x, p.y) else 0.0
            for j in tree.query(p.buffer(DOUBLE)):
                j = int(j)
                if j == i:
                    continue
                o = lines[j][0]
                dd = o.distance(p)
                if 0.3 < dd < DOUBLE:
                    s = o.project(p)
                    r0, r1 = o.interpolate(max(0, s - 0.5)), o.interpolate(min(o.length, s + 0.5))
                    h2 = math.atan2(r1.y - r0.y, r1.x - r0.x)
                    if abs(math.sin(h2 - heading)) < 0.26:            # within 15 degrees
                        faults["double"].append((p.x, p.y, 0.5))      # each pair is seen from both lines
                        break
            if st in FRONT_STYLES and at(to_street, (p.x, p.y)) > OFF:
                faults["off"].append((p.x, p.y, 1.0))
            if st in FRONT_STYLES and street_shape is not None and shapely.contains_xy(street_shape, p.x, p.y):
                faults["inside"].append((p.x, p.y, 1.0))
            if core is not None and at(core, (p.x, p.y)):
                faults["core"].append((p.x, p.y, 1.0))
    # Into areas.
    gh, gw = -(-H // AREA), -(-W // AREA)
    length = np.zeros((gh, gw))
    for x, y, w in total:
        length[min(int(y) // AREA, gh - 1), min(int(x) // AREA, gw - 1)] += w
    count = {k: np.zeros((gh, gw)) for k in faults}
    for k, fl in faults.items():
        for x, y, w in fl:
            count[k][min(max(int(y), 0) // AREA, gh - 1), min(max(int(x), 0) // AREA, gw - 1)] += w
    raw = count["bend"] + count["short"] + count["end"] + (count["zigzag"] + count["double"] + count["off"]) / 4         + count["inside"] + count["core"]
    mess = np.where(length >= 20, raw / np.maximum(length, 1) * 100, 0)
    return {"name": name, "lines": lines, "faults": faults, "length": length, "count": count,
            "mess": mess, "H": H, "W": W, "origin": origin(map_dir, name)}


def origin(map_dir: str, name: str) -> tuple[int, int]:
    """The map's world origin in tiles (the .pzw's worldOrigin, cells)."""
    import re
    pzw = open(os.path.join(map_dir, f"{name}.pzw"), encoding="utf-8").read()
    m = re.search(r'<worldOrigin origin="(\d+),(\d+)"', pzw)
    return (int(m.group(1)) * CELL, int(m.group(2)) * CELL) if m else (0, 0)


def picture(res: dict, map_dir: str, gy: int, gx: int, path: str, pad: int = 6, px: int = 16) -> None:
    """Top-down: the ground, boundary lines (fronts red, sides orange), faults marked."""
    name = res["name"]
    g = Image.open(os.path.join(map_dir, f"{name}.bmp")).convert("RGB")
    x0, y0 = max(0, gx * AREA - pad), max(0, gy * AREA - pad)
    x1, y1 = min(res["W"], (gx + 1) * AREA + pad), min(res["H"], (gy + 1) * AREA + pad)
    img = g.crop((x0, y0, x1, y1)).resize(((x1 - x0) * px, (y1 - y0) * px), Image.NEAREST)
    img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), 0.35)
    d = ImageDraw.Draw(img)
    f = lambda p: ((p[0] - x0) * px, (p[1] - y0) * px)  # noqa: E731
    for ln, st in res["lines"]:
        c = list(ln.coords)
        if all(not (x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2) for x, y in c):
            continue
        d.line([f(p) for p in c], fill=(220, 30, 30) if st in FRONT_STYLES else (240, 150, 20), width=3)
    marks = {"zigzag": (120, 120, 0), "bend": (30, 30, 220), "short": (160, 0, 200), "end": (0, 0, 0), "double": (0, 170, 170), "off": (240, 0, 160), "inside": (255, 0, 0), "core": (255, 120, 0)}
    for k, fl in res["faults"].items():
        for x, y, _w in fl:
            if x0 <= x <= x1 and y0 <= y <= y1:
                cx, cy = f((x, y))
                r = 5 if k in ("bend", "short", "end") else 2
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=marks[k], width=2)
    d.rectangle((0, 0, 260, 16 * len(marks) + 4), fill=(255, 255, 255))
    for i, (k, c) in enumerate(marks.items()):
        d.text((4, 2 + 16 * i), {"off": "front off street", "end": "loose end"}.get(k, k), fill=c)
    img.save(path)


def main() -> int:
    map_dir = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
    res = score(map_dir)
    ox, oy = res["origin"]
    tot = {k: int(round(v.sum())) for k, v in res["count"].items()}
    print(f"{res['name']}: {len(res['lines'])} boundary lines, {int(res['length'].sum())} tiles; "
          f"zigzag tiles {tot['zigzag']}, bends {tot['bend']}, short pieces {tot['short']}, loose ends {tot['end']}, "
          f"doubled tiles {tot['double']}, front tiles off the street {tot['off']}; "
          f"in the street: front {tot['inside']}, on the carriageway {tot['core']}")
    order = np.dstack(np.unravel_index(np.argsort(-res["mess"], axis=None), res["mess"].shape))[0][:top]
    rows = []
    if "--pictures" in sys.argv:
        os.makedirs(OUT, exist_ok=True)
    for rank, (gy, gx) in enumerate(order, 1):
        c = {k: int(round(v[gy, gx])) for k, v in res["count"].items()}
        cx, cy = gx * AREA + AREA // 2, gy * AREA + AREA // 2
        rows.append({"rank": rank, "mess": round(float(res["mess"][gy, gx]), 1), "map": [int(cx), int(cy)],
                     "world": [int(ox + cx), int(oy + cy)], "length": int(res["length"][gy, gx]), **c})
        print(f"{rank:2d}. mess {res['mess'][gy, gx]:5.1f}  world {ox + cx},{oy + cy} (map {cx},{cy})  "
              f"{int(res['length'][gy, gx])} tiles: zigzag {c['zigzag']} bends {c['bend']} short {c['short']} ends {c['end']} "
              f"doubled {c['double']} off-street {c['off']}")
        if "--pictures" in sys.argv:
            picture(res, map_dir, gy, gx, os.path.join(OUT, f"{res['name']}_worst_{rank:02d}.png"))
    json.dump({"totals": tot, "worst": rows}, open(os.path.join(map_dir, f"{res['name']}_boundary_score.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
