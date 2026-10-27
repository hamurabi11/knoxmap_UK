"""V shapes in the garden boundaries: a straight boundary that dips out and back -
angled pieces leaving a lattice line and coming back to it, the line running on
past both ends - as the corner of a front strip square to an angled street cut
into a plot cut on the grid did (user, 2026-10-16). Counts them, on fronts apart
from sides and backs, and lists the first few.

    python v_check.py <knoxmap map dir> [--list N]

Reads <map>_fences.geojson (source=plots, the exact `tiles`), joined into one
network first (a V can span two lines where the style changes or a junction
splits them). A V: corners p0 ... p2 (up to V_CORNERS between them) with p0-p2
on one of the 16 lattice directions, the corners between all on one side of it,
the deepest within V_DEPTH and at least V_MIN off it, p0-p2 no longer than
V_WIDE, at least one of the pieces between angled to it (a square notch is not
a V), and the line carrying straight on along p0-p2 past both p0 and p2.
"""
from __future__ import annotations

import json
import math
import os
import sys

import shapely
from shapely.geometry import LineString

V_DEPTH = 4.0      # tiles
V_MIN = 0.5        # tiles: shallower is a nub, not a V
V_WIDE = 10.0      # tiles
V_CORNERS = 3      # corners in the dip at most
DIRS = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1)]


def corners(pts):
    out = [tuple(pts[0])]
    for p in pts[1:]:
        p = tuple(p)
        if math.dist(p, out[-1]) < 1e-6:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            if abs((b[0] - a[0]) * (p[1] - b[1]) - (b[1] - a[1]) * (p[0] - b[0])) < 1e-6 and \
                    (b[0] - a[0]) * (p[0] - b[0]) + (b[1] - a[1]) * (p[1] - b[1]) > 0:
                out[-1] = p
                continue
        out.append(p)
    return out


def _on_lattice(dx, dy):
    return any(abs(dx * b - dy * a) < 1e-6 for a, b in DIRS)


def _parallel(u, v):
    return abs(u[0] * v[1] - u[1] * v[0]) < 1e-6 and u[0] * v[0] + u[1] * v[1] > 0


def vees(pts):
    """The V shapes in one line: their deepest corners."""
    c = corners(pts)
    found = []
    for i0 in range(1, len(c) - 2):
        for i2 in range(i0 + 2, min(len(c) - 1, i0 + V_CORNERS + 2)):
            p0, p2 = c[i0], c[i2]
            base = (p2[0] - p0[0], p2[1] - p0[1])
            L = math.hypot(*base)
            if L < 1e-6 or L > V_WIDE or not _on_lattice(*base):
                continue
            # The line carries straight on along the base past both ends.
            if not (_parallel((p0[0] - c[i0 - 1][0], p0[1] - c[i0 - 1][1]), base) and
                    _parallel((c[i2 + 1][0] - p2[0], c[i2 + 1][1] - p2[1]), base)):
                continue
            off = [((q[0] - p0[0]) * base[1] - (q[1] - p0[1]) * base[0]) / L for q in c[i0 + 1:i2]]
            if not (all(o > 1e-6 for o in off) or all(o < -1e-6 for o in off)):
                continue
            deep = max(abs(o) for o in off)
            if deep < V_MIN or deep > V_DEPTH:
                continue
            seg = list(zip(c[i0:i2], c[i0 + 1:i2 + 1]))
            angled = any(abs((b[0] - a[0]) * base[0] + (b[1] - a[1]) * base[1]) > 1e-6 and
                         abs((b[0] - a[0]) * base[1] - (b[1] - a[1]) * base[0]) > 1e-6 for a, b in seg)
            if angled:
                found.append(c[i0 + 1 + max(range(len(off)), key=lambda n: abs(off[n]))])
    return found


def main() -> int:
    map_dir = sys.argv[1]
    n_list = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 5
    name = os.path.basename(os.path.normpath(map_dir))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from ground.plotlib import FRONT_STYLES
    feats = json.load(open(os.path.join(map_dir, f"{name}_fences.geojson"), encoding="utf-8"))["features"]
    lines, fronts = [], []
    for ft in feats:
        t = ft["properties"].get("tiles")
        if t and ft["properties"].get("source") == "plots" and len(t) >= 2:
            ln = LineString(t)
            lines.append(ln)
            if ft["properties"].get("knox_style") in FRONT_STYLES:
                fronts.append(ln)
    net = shapely.line_merge(shapely.union_all(lines, grid_size=1e-4))
    front_net = shapely.union_all(fronts).buffer(0.05) if fronts else None
    found = {"side or back": [], "front": []}
    for ln in getattr(net, "geoms", [net]):
        for p in vees(list(ln.coords)):
            on_front = front_net is not None and front_net.contains(shapely.Point(p))
            found["front" if on_front else "side or back"].append(p)
    print(f"{name}: V shapes in the plot boundaries: {len(found['side or back'])} on sides and backs, "
          f"{len(found['front'])} on fronts")
    for x, y in found["side or back"][:n_list]:
        print(f"  side map {x:.1f},{y:.1f}")
    for x, y in found["front"][:n_list]:
        print(f"  front map {x:.1f},{y:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
