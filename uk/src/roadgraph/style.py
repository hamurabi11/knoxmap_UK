"""How a town's roads are drawn, measured from their centrelines: vanilla
B42's from the game's own streets.xml (every street as the map's makers drew
it: a name, a width and a line of points in world tiles), ours from the road
graph's routed paths - one measure for both.

    python style.py vanilla <name> [x0,y0,x1,y1] <out.json>   (the streets with a point in the box)
    python style.py roads <roadgraph out dir> <out.json>      (paths.json, written by run.py)
    python style.py score <ours.json> <vanilla.json> ...
    python style.py shapes <out.json>                         (vanilla's stretches between junctions, by shape)

Each centreline is cut at its points into segments, each put on the nearest
of the sixteen directions (along the grid, 2:1, 45) within SNAP_DEG, else
"other". A segment at least STRAIGHT tiles long is a straight; the shorter
ones between two straights are a bend's pieces. The profile:
- share of length on each slope family;
- straights' lengths on each;
- bends per 100 tiles of road, by how far they turn (22.5, 45, 90) and
  whether sharp (no pieces: a corner) or curved (pieces between);
- curved bends' radius (the pieces' length over the turn), by road width.

What vanilla does (all of it, LOG 2026-10-10): 85% of its street length on
the grid, 9% at 45, 5.5% at 2:1, 0.4% other; a one-notch turn is a sharp
corner (98%); a 45 is sharp (72%) or one 2:1 piece; a right angle is sharp
(56%) or three pieces - 2:1, 45, 1:2, near equal - at a radius of about 12
(roads 6 wide or less) or 20 (8 and more), or one 45 piece cutting a tight
corner (radius 4-9).
"""
from __future__ import annotations

import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRAIGHT = 20.0              # tiles: a segment this long is a straight, shorter ones between are a bend
SNAP_DEG = 3.0               # a segment within this of one of the sixteen directions is on it
FAMILIES = {"grid": (0.0, 90.0), "45": (45.0, 135.0), "2:1": (26.57, 63.43, 116.57, 153.43)}


def streets_xml() -> str:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
    import ukpaths  # noqa: E402
    cfg = json.load(open(os.path.join(ukpaths.KNOX, "knoxmap_config.json"), encoding="utf-8"))
    return os.path.join(cfg["pz_install"], "media", "maps", "Muldraugh, KY", "streets.xml")


def vanilla_streets(box=None) -> list:
    """[(width, [(x, y), ...])] from streets.xml, those with a point in box (world tiles)."""
    out = []
    for st in ET.parse(streets_xml()).getroot().iter("street"):
        pts = [(float(p.get("x")), float(p.get("y"))) for p in st.iter("point")]
        if box is None or any(box[0] <= x < box[2] and box[1] <= y < box[3] for x, y in pts):
            out.append((int(st.get("width")), pts))
    return out


def family(angle: float) -> str:
    a = angle % 180
    for name, angles in FAMILIES.items():
        if min(min(abs(a - x), 180 - abs(a - x)) for x in angles) <= SNAP_DEG:
            return name
    return "other"


def _segments(pts):
    return [(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), math.dist(a, b))
            for a, b in zip(pts, pts[1:]) if math.dist(a, b) > 1e-6]


def bends(pts) -> list:
    """[(turn in degrees, curved?, radius or None)] between a centreline's straights."""
    segs = _segments(pts)
    out = []
    i = 0
    while i < len(segs) - 1:
        if segs[i][1] < STRAIGHT:
            i += 1
            continue
        j = i + 1
        while j < len(segs) and segs[j][1] < STRAIGHT:
            j += 1
        if j >= len(segs):
            break
        turn = abs((segs[j][0] - segs[i][0] + 540) % 360 - 180)
        if turn > 5:
            arc = sum(s[1] for s in segs[i + 1:j])
            out.append((turn, j > i + 1, arc / math.radians(turn) if j > i + 1 else None))
        i = j
    return out


def profile(streets: list) -> dict:
    total = 0.0
    by_family = {"grid": 0.0, "2:1": 0.0, "45": 0.0, "other": 0.0}
    straights = {"grid": [], "2:1": [], "45": []}
    kinds: dict = {}
    radii: dict = {}
    for width, pts in streets:
        for angle, L in _segments(pts):
            total += L
            f = family(angle)
            by_family[f] += L
            if L >= STRAIGHT and f in straights:
                straights[f].append(L)
        for turn, curved, r in bends(pts):
            size = min((22.5, 45.0, 90.0), key=lambda t: abs(t - turn)) if turn < 112 else 180.0
            key = f"{'curved' if curved else 'sharp'} {size:g}"
            kinds[key] = kinds.get(key, 0) + 1
            if curved and size == 90.0:
                radii.setdefault("w<=6" if width <= 6 else "w>=7", []).append(r)
    pct = lambda v: [round(float(np.percentile(v, q)), 1) for q in (10, 50, 90)] if v else None
    return {
        "length": round(total),
        "share by slope": {k: round(v / max(total, 1), 3) for k, v in by_family.items()},
        "straight lengths (10/50/90th)": {k: pct(v) for k, v in straights.items()},
        "bends per 100 tiles": {k: round(100 * v / max(total, 1), 3) for k, v in sorted(kinds.items())},
        "right-angle curve radius (10/50/90th)": {k: pct(v) for k, v in sorted(radii.items())},
    }


def stretches(streets: list, min_len: float = 8.0) -> list:
    """[(width, pts)]: the streets cut where another street meets or crosses
    them - the stretches between junctions."""
    import shapely
    from shapely.ops import substring
    lines = [shapely.LineString(p) for _w, p in streets]
    tree = shapely.STRtree(lines)
    out = []
    for i, (w, pts) in enumerate(streets):
        ln = lines[i]
        cuts = {0.0, ln.length}
        for j in tree.query(ln.buffer(1.0)):
            if j == i:
                continue
            for end in (streets[j][1][0], streets[j][1][-1]):
                if ln.distance(shapely.Point(end)) <= 1.0:
                    cuts.add(round(ln.project(shapely.Point(end)), 1))
            inter = ln.intersection(lines[j])
            for g in getattr(inter, "geoms", [inter]):
                if g.geom_type == "Point":
                    cuts.add(round(ln.project(g), 1))
        ds = sorted(cuts)
        out += [(w, list(substring(ln, a, b).coords)) for a, b in zip(ds, ds[1:]) if b - a >= min_len]
    return out


def shape(pts) -> str:
    """A stretch's shape: its bends between straights (STRAIGHT tiles and
    more), each S (sharp) or C (curved) and its turn in notches of the
    sixteen, mirrored so the first turns positive - "" for a straight."""
    angles = [0, 26.57, 45, 63.43, 90, 116.57, 135, 153.43, 180, 206.57, 225, 243.43, 270, 296.57, 315, 333.43]
    runs = []
    for a, L in _segments(pts):
        h = min(range(16), key=lambda k: min(abs(a % 360 - angles[k]), 360 - abs(a % 360 - angles[k])))
        if runs and runs[-1][0] == h:
            runs[-1][1] += L
        else:
            runs.append([h, L])
    longs = [i for i, r in enumerate(runs) if r[1] >= STRAIGHT]
    toks = []
    for a, b in zip(longs, longs[1:]):
        t = (runs[b][0] - runs[a][0]) % 16
        toks.append(("C" if b > a + 1 else "S", t - 16 if t > 8 else t))
    if toks and toks[0][1] < 0:
        toks = [(k, -t) for k, t in toks]
    return " ".join(f"{k}{t:+d}" for k, t in toks)


def score(target: list, ours: dict) -> dict:
    """How far `ours` is from the vanilla profiles in `target` (their mean):
    the gaps in slope shares (0-2) and in bends per 100 tiles of each kind
    (summed, over vanilla's total)."""
    fams = ("grid", "2:1", "45", "other")
    want = {f: float(np.mean([t["share by slope"][f] for t in target])) for f in fams}
    keys = sorted({k for t in target for k in t["bends per 100 tiles"]} | set(ours["bends per 100 tiles"]))
    want_b = {k: float(np.mean([t["bends per 100 tiles"].get(k, 0) for t in target])) for k in keys}
    return {"slope gap": round(sum(abs(ours["share by slope"][f] - want[f]) for f in fams), 3),
            "bend gap": round(sum(abs(ours["bends per 100 tiles"].get(k, 0) - want_b[k]) for k in keys)
                              / max(sum(want_b.values()), 1e-9), 3),
            "vanilla shares": {f: round(v, 3) for f, v in want.items()},
            "vanilla bends per 100": {k: round(v, 3) for k, v in want_b.items()}}


def main() -> int:
    mode = sys.argv[1]
    if mode == "vanilla":
        box = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 4 else None
        prof = profile(vanilla_streets(box))
        prof["source"] = f"vanilla {sys.argv[2]} {box or 'whole map'}"
        json.dump(prof, open(sys.argv[-1], "w"), indent=1)
    elif mode == "roads":
        paths = json.load(open(os.path.join(sys.argv[2], "paths.json")))
        prof = profile([(p["width"], [tuple(c) for c in p["path"]]) for p in paths])
        prof["source"] = f"road graph {sys.argv[2]}"
        json.dump(prof, open(sys.argv[3], "w"), indent=1)
    elif mode == "shapes":                # shapes <out.json>: vanilla's stretch shapes, counted
        counts: dict = {}
        for _w, pts in stretches(vanilla_streets()):
            k = shape(pts)
            counts[k] = counts.get(k, 0) + 1
        json.dump(dict(sorted(counts.items(), key=lambda kv: -kv[1])), open(sys.argv[2], "w"), indent=1)
        print(sum(counts.values()), "stretches,", len(counts), "shapes")
        return 0
    elif mode == "score":
        print(json.dumps(score([json.load(open(p)) for p in sys.argv[3:]], json.load(open(sys.argv[2]))), indent=1))
        return 0
    print(json.dumps(prof, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
