"""Stand building outlines upright on a KnoxMap map's tile grid, before Build.

    python tidy_outlines.py <knoxmap map dir> [--min 1.5] [--max-turn 40]

KnoxMap squares up only near-rectangular buildings (replacing them with an
upright rectangle); an L-shaped house or a shop with a rear wing is drawn at
its real angle, so its walls become staircases and its roof splits into
slivers that are roofed flat. This turns the outline itself instead:

  1. outlines that touch (terraces, rows of shops) form a group,
  2. each group turns about its centre by its own angle to the tile grid
     (area-weighted, folded to -45..45 degrees), so party walls stay shared,
  3. each outline loses notches and spurs narrower than --min tiles and has
     every edge snapped level or upright,
  4. corners are rounded to whole tiles, so neighbours' walls meet exactly.

Groups turned by more than --max-turn degrees are left alone. The original
outlines are kept as <map>_buildings.orig.geojson, and each run starts again
from them.
"""
from __future__ import annotations

import cmath
import json
import math
import os
import shutil
import sys

from shapely import affinity
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator.renderer import Projector  # noqa: E402

TOUCH = 0.3        # tiles: outlines this close count as touching
MAX_CHANGE = 0.25  # refuse a tidy that changes more than this share of the area


def main_angle(poly: Polygon) -> float:
    r = poly.minimum_rotated_rectangle
    (x0, y0), (x1, y1), (x2, y2) = list(r.exterior.coords)[:3]
    if math.dist((x0, y0), (x1, y1)) >= math.dist((x1, y1), (x2, y2)):
        return math.degrees(math.atan2(y1 - y0, x1 - x0))
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def fold(angle: float) -> float:
    """Angle to the nearest grid axis, -45..45."""
    return (angle + 45) % 90 - 45


def group_turn(polys) -> float:
    """Area-weighted mean angle to the grid, averaged on the 90-degree circle."""
    z = sum(p.area * cmath.exp(4j * math.radians(main_angle(p))) for p in polys)
    return fold(math.degrees(cmath.phase(z)) / 4) if abs(z) > 0 else 0.0


def snap_rectilinear(poly: Polygon) -> Polygon:
    pts = list(poly.exterior.coords)[:-1]
    n = len(pts)
    if n < 4:
        return poly
    level = [abs(pts[(i + 1) % n][0] - pts[i][0]) >= abs(pts[(i + 1) % n][1] - pts[i][1])
             for i in range(n)]
    out = []
    for i in range(n):
        if level[i - 1] == level[i]:
            continue
        lev_i, up_i = (i - 1, i) if level[i - 1] else (i, i - 1)
        la, lb = pts[lev_i % n], pts[(lev_i + 1) % n]
        ua, ub = pts[up_i % n], pts[(up_i + 1) % n]
        out.append((round((ua[0] + ub[0]) / 2), round((la[1] + lb[1]) / 2)))
    # Rounding can leave repeated or collinear points; drop them.
    clean = []
    for p in out:
        if not clean or p != clean[-1]:
            clean.append(p)
    if len(clean) >= 2 and clean[0] == clean[-1]:
        clean.pop()
    if len(clean) < 4:
        return poly
    snapped = Polygon(clean)
    return snapped if snapped.is_valid else snapped.buffer(0)


def tidy(poly: Polygon, min_t: float) -> Polygon:
    half = min_t / 2
    opened = poly.buffer(-half, join_style=2).buffer(half, join_style=2)
    if opened.is_empty or opened.geom_type != "Polygon":
        opened = poly
    closed = opened.buffer(half, join_style=2).buffer(-half, join_style=2)
    if closed.geom_type != "Polygon":
        closed = opened
    square = snap_rectilinear(closed.simplify(min_t / 3))
    return square if square.geom_type == "Polygon" and not square.is_empty else poly


def groups_of(polys):
    tree = STRtree(polys)
    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, p in enumerate(polys):
        for j in tree.query(p.buffer(TOUCH)):
            j = int(j)
            if j != i and p.buffer(TOUCH).intersects(polys[j]):
                parent[find(i)] = find(j)
    out = {}
    for i in range(len(polys)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def main() -> int:
    map_dir = sys.argv[1]
    min_t = float(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 1.5
    max_turn = float(sys.argv[sys.argv.index("--max-turn") + 1]) if "--max-turn" in sys.argv else 40
    name = os.path.basename(os.path.normpath(map_dir))
    with open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    b = info["bbox"]
    proj = Projector.build(b["south"], b["west"], b["north"], b["east"],
                           info["meters_per_tile"], info.get("rotation") or 0.0)
    path = os.path.join(map_dir, f"{name}_buildings.geojson")
    orig = os.path.join(map_dir, f"{name}_buildings.orig.geojson")
    if not os.path.exists(orig):
        shutil.copy(path, orig)
    with open(orig, encoding="utf-8") as f:
        gj = json.load(f)

    idx, polys = [], []
    for k, ft in enumerate(gj["features"]):
        if ft["geometry"]["type"] != "Polygon":
            continue
        ring = [proj.to_px(lat, lon) for lon, lat in ft["geometry"]["coordinates"][0]]
        p = Polygon(ring)
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type == "Polygon" and p.area >= 4:
            idx.append(k)
            polys.append(p)

    turned = tidied = refused = skipped = 0
    before, after = [], []
    for members in groups_of(polys):
        turn = group_turn([polys[i] for i in members])
        if abs(turn) > max_turn:
            skipped += len(members)
            continue
        centre = unary_union([polys[i] for i in members]).centroid
        for i in members:
            p = polys[i]
            before.append(abs(fold(main_angle(p))))
            upright = affinity.rotate(p, -turn, origin=centre) if abs(turn) > 0.5 else p
            new = tidy(upright, min_t)
            if new.symmetric_difference(upright).area > MAX_CHANGE * upright.area:
                new = upright
                refused += 1
            else:
                tidied += 1
            turned += abs(turn) > 0.5
            after.append(abs(fold(main_angle(new))))
            ring = [proj.to_latlon(x, y) for x, y in new.exterior.coords]
            gj["features"][idx[i]]["geometry"] = mapping(Polygon([(lon, lat) for lat, lon in ring]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gj, f)

    def share(v, lo):
        return sum(a <= lo for a in v)
    print(f"outlines {len(polys)}: turned {turned}, squared {tidied}, kept unsquared {refused}, "
          f"left alone (turn > {max_turn:g}) {skipped}")
    print(f"  within 2 deg of the grid: before {share(before, 2)}/{len(before)}, "
          f"after {share(after, 2)}/{len(after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
