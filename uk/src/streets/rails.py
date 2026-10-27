"""Railway track along OSM's railways, laid the way vanilla lays its own.

    python rails.py <knoxmap map dir> [--rebuild-table]   # after blocks.py, before plots2.py
    python rails.py <knoxmap map dir> --apply             # after sports.py, before kerbs.py

Two runs. The first lays the track out (<map>_rails.json) before plots2.py,
which takes its bed as the railway (gardens back onto it; no fence crosses
it). The second puts it into the map once sports.py has rebuilt
structures.json from Generate's: its tiles ahead of kerbs.py's (kerbs.py trims
its own off the end on a rerun), and the ground it paints is what kerbs.py
then takes as its starting point.

Vanilla's track (Muldraugh, KY: 213,000 squares of industry_railroad_*) is
three squares of sleepers and rails on a dirt bed a square wider each side,
in four steady patterns - along y, along x and the two diagonals - and a
fixed set of pieces at each 45-degree bend between them; it never turns 90
degrees in one go and never runs up a ramp. Across a road the sleepers carry
straight on (about 370 crossing squares that way, a handful with special
pieces), with a crossing signal each side. So here:

- Each OSM line (ways joined end to end, the straightest way on through a
  junction) is redrawn on the grid and the diagonals: legs in the 8
  directions, turning 45 degrees at a time, at least LEG_MIN squares between
  bends, steered back towards the real line as it drifts (within about
  half a jog of it).
- The legs get the steady patterns (STEADY) and each bend vanilla's own
  pieces (data/vanilla_rail.json, learnt from vanilla's lots: the pieces
  that differ from the two legs' steady patterns round each of its bends, as
  most of its bends of that shape have them).
- The bed is dirt under the track and a square round it, never on a road,
  pavement, building or water; what Generate painted as the bed (gravel,
  where the real line ran) and no longer is goes back to the ground beside it.
  Nothing grows on the bed.
- Where a straight leg crosses a road, a crossing signal stands each side,
  on diagonally opposite corners as vanilla's do (US crossbucks: a stand-in).
- The railway's own bridges Generate lifted come back to the ground (user,
  2026-10-18): deck, ramps, railings and posts out, the road beneath a level
  crossing. A road's bridge over the line stays, the track under it.
- Not laid: squares under a building, on water, on a road bridge's ramp, and
  where an earlier line's track already is (a junction: the branch stops
  short).

The ground files it changes are kept as they were beside them
(*.prerails.bmp), taken afresh whenever another step has written them since,
so it can be rerun at any point.
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
sys.path.insert(0, os.path.join(ukpaths.KNOX, "tools"))
sys.path.insert(0, ukpaths.assets("Tile_creator", "src") or "")

from generator import pz_colors as C  # noqa: E402

TABLE = os.path.normpath(os.path.join(HERE, "..", "..", "data", "vanilla_rail.json"))
SHEET = "industry_railroad_"
TRACK_LAYER, SIGNAL_LAYER = "FloorFurniture", "Furniture"
LEG_MIN = 7          # squares between two bends (vanilla's bend pieces reach 2-3 squares from it)
LOOK = 12            # squares ahead along the real line for its direction
GAIN = 0.1           # radians of steer back per square of drift
WINDOW = 7           # squares round a bend looked at when the table is built
MIN_SEP = 6          # squares between the centre lines of two tracks run side by side (vanilla's
                     # double track: 9 along y, 5 or 10 along x; the "/" diagonal is six squares a row)
SEP_NEED = {"NS": 4, "EW": 4, "BS": 5, "FS": 6}   # ...the least at which their patterns do not overlap
DEV_MAX = 8          # squares: run beside another line no further than this off its own real line -
THROAT = 60          # ...past it within this of its own end (a station's throat, lines closer than
                     # tracks can be) it joins that line; further from its end it steers on its own
PAR_NEAR = 14        # squares: a line laid this near, running the same way, is run beside
JOIN_NEAR = 2.0      # a line's end this near another line joins it: a branch
DUP_M = 2.5          # a stretch of line this near one kept already is the same track drawn twice
                     # (two tracks side by side are 3.2 m apart or more)...
DUP_LEN = 20         # metres: ...for this long at least
CORRIDOR_M = 30      # a line this near the corridor's spine, running its way, is one of its tracks
SAMPLE_M = 2.0       # metres between the points a line's offset from the spine is read at
CROSS_M = 100        # metres: a line shorter than this between two others is a crossover
JOIN_M = 2.0         # metres: a line ending this near another starts or ends on it
SIGNAL_OUT = (4, 3, 5, 6, 7, 8)   # squares from the centre line, first that is free (vanilla: 3-5, mostly 4)
LAND_GAP = 11        # squares: land narrower than this between lines (or in a bend) is the railway's

# The four steady patterns, from vanilla: a line's squares at one step along it,
# from its centre square. NS: along y; EW: along x; BS: the diagonal "\" (x - y
# the same); FS: the diagonal "/" (x + y the same). Checked against vanilla's
# bends whenever the table is built.
STEADY = {
    "NS": [(-1, 0, "01_0"), (0, 0, "01_1"), (1, 0, "01_2")],
    "EW": [(0, -1, "01_5"), (0, 0, "01_4"), (0, 1, "01_3")],
    "BS": [(-2, 0, "01_16"), (-1, 0, "01_17"), (0, 0, "01_18"), (1, 0, "01_19"), (2, 0, "01_20")],
    "FS": [(-2, 0, "01_12"), (-1, 0, "01_13"), (0, 0, "01_10"), (1, 0, "01_11"), (2, 0, "01_8"), (3, 0, "01_9")],
}
FAMILY = {(0, 1): "NS", (0, -1): "NS", (1, 0): "EW", (-1, 0): "EW",
          (1, 1): "BS", (-1, -1): "BS", (1, -1): "FS", (-1, 1): "FS"}
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]   # by angle, y down
CENTRE = {"01_1": "NS", "01_4": "EW", "01_18": "BS", "01_10": "FS"}
# Crossing signals (industry_railroad_05_48-51), by the side of the line they
# stand on: vanilla puts 48 east of a line along y, 50 west; 49 south of a
# line along x, 51 north.
SIGNAL = {("NS", 1): "05_48", ("NS", -1): "05_50", ("EW", 1): "05_49", ("EW", -1): "05_51"}
ROAD = (C.DARK_ASPHALT, C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE)
STREET = ROAD + (C.PALE_CONCRETE,)


def steady(anchor, d, s0, s1):
    """{square: sprite} of a leg from anchor along d, steps s0..s1 (both included)."""
    out = {}
    for s in range(s0, s1 + 1):
        px, py = anchor[0] + s * d[0], anchor[1] + s * d[1]
        for ox, oy, spr in STEADY[FAMILY[d]]:
            out[(px + ox, py + oy)] = spr
    return out


def bend_steady(key, reach):
    """The two legs of a bend at (0, 0), drawn in key order (the later wins)."""
    out = {}
    for d in key:
        out.update(steady((0, 0), d, 0, reach))
    return out


def joint_key(dirs) -> str:
    """A joint's legs (directions out from it), as the table keys them."""
    return "|".join(f"{a},{b}" for a, b in sorted({tuple(d) for d in dirs}))


def bend_key(d1, d2) -> str:
    return joint_key([d1, d2])


def parse_key(k: str):
    return tuple(tuple(int(v) for v in part.split(",")) for part in k.split("|"))


# --- the table, from vanilla's lots ----------------------------------------------------------------

def vanilla_track() -> dict:
    """{(x, y): [sprite suffixes]} of vanilla's track at ground level."""
    import pack
    from cell_check import read_cell
    at = {}
    maps = os.path.join(pack.game_dir(), "media", "maps")
    for m in sorted(os.listdir(maps)):
        for h in glob.glob(os.path.join(maps, m, "*.lotheader")):
            if SHEET.encode() not in open(h, "rb").read():
                continue
            cx, cy = map(int, re.findall(r"(\d+)_(\d+)\.lotheader", os.path.basename(h))[0])
            for (x, y, z), names in read_cell(os.path.join(maps, m), cx, cy).items():
                rs = [n[len(SHEET):] for n in names if n.startswith(SHEET)]
                if z == 0 and rs:
                    at[(cx * 256 + x, cy * 256 + y)] = rs
    return at


def build_table() -> dict:
    """Vanilla's pieces at each kind of joint: where two legs meet (a bend) or
    three (a branch leaving a line through its points). A joint is where the
    centre lines meet: a run of one pattern ending there and another ending
    or passing through it. Each kind's pieces are the squares round it that
    differ from its legs' steady patterns, as most of its joints have them."""
    at = vanilla_track()
    along = {"NS": lambda x, y: (x, y), "EW": lambda x, y: (y, x), "BS": lambda x, y: (x - y, y),
             "FS": lambda x, y: (x + y, y)}
    line_eq = {"NS": (1, 0), "EW": (0, 1), "BS": (1, -1), "FS": (1, 1)}
    point = {"NS": lambda c, t: (c, t), "EW": lambda c, t: (t, c), "BS": lambda c, t: (c + t, t),
             "FS": lambda c, t: (c - t, t)}
    forward = {"NS": (0, 1), "EW": (1, 0), "BS": (1, 1), "FS": (-1, 1)}     # t growing
    centres = collections.defaultdict(set)
    for (x, y), rs in at.items():
        for r in rs:
            if r in CENTRE:
                c, t = along[CENTRE[r]](x, y)
                centres[(CENTRE[r], c)].add(t)
    runs = []
    for (f, c), ts in centres.items():
        ts = sorted(ts)
        start = prev = ts[0]
        for t in ts[1:] + [None]:
            if t is None or t != prev + 1:
                if prev - start >= 4:
                    runs.append((f, c, start, prev))
                start = t
            prev = t if t is not None else prev
    by_family = collections.defaultdict(list)
    for r in runs:
        by_family[r[0]].append(r)
    legs_at = collections.defaultdict(set)
    for f, c, t0, t1 in runs:
        for t_end, sign in ((t0, -1), (t1, 1)):
            ex, ey = point[f](c, t_end)
            for g, others in by_family.items():
                if g == f:
                    continue
                (a1, b1), (a2, b2) = line_eq[f], line_eq[g]
                det = a1 * b2 - a2 * b1
                for _g, c2, u0, u1 in others:
                    x, y = (c * b2 - c2 * b1) / det, (a1 * c2 - a2 * c) / det
                    if x != int(x) or y != int(y) or abs(x - ex) > 4 or abs(y - ey) > 4:
                        continue
                    _c, u = along[g](int(x), int(y))
                    if not (u0 - 4 <= u <= u1 + 4):
                        continue
                    here = legs_at[(int(x), int(y))]
                    v = forward[f]
                    here.add((-sign * v[0], -sign * v[1]))
                    w = forward[g]
                    if u > u0 + 3:
                        here.add((-w[0], -w[1]))
                    if u < u1 - 3:
                        here.add(w)
    votes = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    count = collections.Counter()
    for (x, y), legs in legs_at.items():
        if len(legs) not in (2, 3):
            continue
        key = joint_key(legs)
        count[key] += 1
        for dx in range(-WINDOW, WINDOW + 1):
            for dy in range(-WINDOW, WINDOW + 1):
                votes[key][(dx, dy)][tuple(sorted(at.get((x + dx, y + dy), [])))] += 1
    joints = {}
    for key, n in count.items():
        dirs = parse_key(key)
        if len(dirs) == 2:
            d1, d2 = dirs
            cos = (d1[0] * d2[0] + d1[1] * d2[1]) / math.hypot(*d1) / math.hypot(*d2)
            if n < 8 or abs(cos - math.cos(math.radians(135))) > 0.01:
                continue          # the odd 90-degree turn: too few to learn from
        voted = {o: c.most_common(1)[0][0] for o, c in votes[key].items()}
        base = bend_steady(dirs, WINDOW)
        diff, far = [], 0
        for (dx, dy), spr in voted.items():
            want = spr[0] if spr else None
            if base.get((dx, dy)) != want:
                if max(abs(dx), abs(dy)) >= WINDOW - 1:
                    far += 1
                    continue
                diff.append([dx, dy, want])
        # A bend's window must end on its legs' own patterns; a branch's, seen
        # a few times only, may have something else at its edge (left out).
        if (len(dirs) == 2 and far) or not diff:
            continue
        joints[key] = {"seen": n, "radius": max(max(abs(dx), abs(dy)) for dx, dy, _ in diff), "pieces": diff}
    table = {"bends": joints}
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    json.dump(table, open(TABLE, "w"), indent=0)
    print(f"table: {sum(len(parse_key(k)) == 2 for k in joints)} bends and "
          f"{sum(len(parse_key(k)) == 3 for k in joints)} branch joints from vanilla ({dict(count)})")
    return table


# --- the lines ---------------------------------------------------------------------------------

def osm_lines(map_dir, info, proj):
    """[(points in tiles, tags)] of the map's railways, ways joined end to end;
    and the ways on bridges, as they are."""
    from generator.renderer import _is_polygon, classify
    from ground import plotlib as P
    ways = [([proj.to_px(la, lo) for la, lo in f.geometry], f.tags or {})
            for f in P.osm_features(map_dir, info, proj)
            if f.kind == "way" and len(f.geometry) >= 2 and not _is_polygon(f)
            and classify(f.tags or {}) == "railway"
            and (f.tags or {}).get("tunnel", "no") == "no"]        # a tunnel's track is underground
    bridges = [(pts, tags) for pts, tags in ways if tags.get("bridge", "no") != "no"]
    key = lambda p: (round(p[0], 2), round(p[1], 2))  # noqa: E731
    # Main lines first, then sidings; longest first.
    ways.sort(key=lambda w: ("service" in w[1], -sum(math.dist(a, b) for a, b in zip(w[0], w[0][1:]))))
    chains = [[list(pts), tags] for pts, tags in ways]
    joined = True
    while joined:
        joined = False
        ends = collections.defaultdict(list)
        for i, (pts, _t) in enumerate(chains):
            ends[key(pts[0])].append((i, 0))
            ends[key(pts[-1])].append((i, -1))
        for k, here in ends.items():
            if len(here) < 2:
                continue
            # The straightest pair on through this node.
            best = None
            for a in range(len(here)):
                for b in range(a + 1, len(here)):
                    (i, ei), (j, ej) = here[a], here[b]
                    if i == j:
                        continue
                    va = _away(chains[i][0], ei)
                    vb = _away(chains[j][0], ej)
                    cos = (va[0] * vb[0] + va[1] * vb[1]) / (math.hypot(*va) * math.hypot(*vb) or 1)
                    if cos < -0.85 and (best is None or cos < best[0]):
                        best = (cos, i, ei, j, ej)
            if best:
                _, i, ei, j, ej = best
                a = chains[i][0] if ei == -1 else chains[i][0][::-1]
                b = chains[j][0] if ej == 0 else chains[j][0][::-1]
                chains[i] = [a + b[1:], chains[i][1]]
                del chains[j]
                joined = True
                break
    return [(pts, tags) for pts, tags in chains], bridges


def _away(pts, end):
    """Direction leaving a chain's end, into the chain, over its first few tiles."""
    seq = pts if end == 0 else pts[::-1]
    p0 = seq[0]
    for p in seq[1:]:
        if math.dist(p0, p) >= 5:
            return (p[0] - p0[0], p[1] - p0[1])
    return (seq[-1][0] - p0[0], seq[-1][1] - p0[1])


def _perp(k):
    """The lattice step across a leg of direction k: a line run beside it is
    this many steps over (along x for a leg along y or a diagonal)."""
    return (0, 1) if DIRS[k][1] == 0 else (1, 0)


def _sep(v, k) -> int:
    """How far apart, in its pattern's own terms, a shift v leaves a leg of direction k."""
    fam = FAMILY[DIRS[k]]
    return abs({"NS": v[0], "EW": v[1], "BS": v[0] - v[1], "FS": v[0] + v[1]}[fam])


def _turn_by(a, b) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def snap(pts, laid=None, start_on=None, end_on=None):
    """The line on the grid and the diagonals: [(anchor, direction, steps)] legs,
    and its joints with other lines [(square, other line, direction out along
    this one)].

    Steered back to the real line as it drifts, LEG_MIN squares at least
    between bends. With the lines laid so far (`laid`, {square: (line, k)} of
    their centre squares): a line that starts on another (`start_on`, a
    branch) leaves it from its centre line at 45 degrees and holds that a leg;
    one that ends on another (`end_on`) turns in at 45 degrees near its end and
    stops on its centre line; and where it runs beside one laid already, the
    same way within PAR_NEAR, it copies that line's legs a fixed shift over (at
    least MIN_SEP), so the two bend together and never cross."""
    from shapely.geometry import LineString, Point
    laid = laid or {}
    line = LineString(pts)
    total = line.length

    def heading(prog):
        a = line.interpolate(prog)
        b = line.interpolate(min(total, prog + LOOK))
        if a.distance(b) < 1e-6:
            b, a = a, line.interpolate(max(0, prog - LOOK))
        return a, math.atan2(b.y - a.y, b.x - a.x)

    def nearest(q, lid, r):
        best = None
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                got = laid.get((q[0] + dx, q[1] + dy))
                if got and got[0] == lid and (best is None or dx * dx + dy * dy < best[0]):
                    best = (dx * dx + dy * dy, (q[0] + dx, q[1] + dy), got[1])
        return best

    def crowded(q, k, skip):
        """Squares ahead along k come within a pattern's width of another line."""
        need = SEP_NEED[FAMILY[DIRS[k]]] - 1
        for j in (3, 6, 9):
            cx, cy = q[0] + j * DIRS[k][0], q[1] + j * DIRS[k][1]
            for dx in range(-need, need + 1):
                for dy in range(-need, need + 1):
                    got = laid.get((cx + dx, cy + dy))
                    if got and got[0] not in skip:
                        return True
        return False

    joints = []
    _, th0 = heading(0)
    k, q, hold = round(th0 / (math.pi / 4)) % 8, (round(pts[0][0]), round(pts[0][1])), 0
    if start_on is not None:
        hit = nearest(q, start_on, 8)
        if hit:
            _d, q, kt = hit
            k = min(((kt + 1) % 8, (kt - 1) % 8, (kt + 3) % 8, (kt + 5) % 8),
                    key=lambda c: _turn_by(c * math.pi / 4, th0))
            joints.append((q, start_on, k))
            hold = LEG_MIN
    end_pts = {sq for sq, (lid, _k) in laid.items() if lid == end_on} if end_on is not None else set()
    legs, start, n, steps = [], q, 0, 0
    follow, converging, conv_d, alone = None, False, 99.0, -1

    def turn_to(k2):
        nonlocal start, n, k
        if k2 != k:
            legs.append((start, DIRS[k], n))
            start, n, k = q, 0, k2

    while steps < 4 * total + 80:
        prog = line.project(Point(q))
        a, th = heading(prog)
        if converging:
            if q in end_pts and n > 0:
                joints.append((q, end_on, (k + 4) % 8))
                break
            near = nearest(q, end_on, 12)
            d_now = math.sqrt(near[0]) if near else 99.0
            if n > 40 or d_now > conv_d + 1.5:
                break                       # missed it: the line ends here
            conv_d = min(conv_d, d_now)
        else:
            if end_on is not None and n >= max(LEG_MIN, hold):
                hit = nearest(q, end_on, 12)
                if hit and total - prog <= math.sqrt(hit[0]) + 3:
                    _d, target, kt = hit
                    towards = [c for c in ((kt + 1) % 8, (kt - 1) % 8, (kt + 3) % 8, (kt + 5) % 8)
                               if math.dist((q[0] + DIRS[c][0], q[1] + DIRS[c][1]), target) < math.dist(q, target)]
                    if towards:
                        follow = None
                        converging, conv_d = True, math.sqrt(hit[0])
                        turn_to(max(towards, key=lambda c: math.cos(_turn_by(c * math.pi / 4, th))))
            if not converging and prog >= total - 0.5:
                break
            if not converging and follow:
                v, lid, flip = follow
                got = laid.get((q[0] - v[0], q[1] - v[1]))
                kg = (got[1] + (4 if flip else 0)) % 8 if got and got[0] == lid else None
                e = math.cos(th) * (q[1] - a.y) - math.sin(th) * (q[0] - a.x)
                if (kg is None or _turn_by(kg * math.pi / 4, th) > math.radians(50) or abs(e) > PAR_NEAR
                        or _sep(v, kg) < SEP_NEED[FAMILY[DIRS[kg]]]):
                    follow = None
                elif abs(e) > DEV_MAX and n >= LEG_MIN:
                    follow = None
                    alone = steps + 3 * LEG_MIN
                    if total - prog <= THROAT:
                        # Pushed this far off its course near its end by the
                        # line it runs beside: it joins that line (a throat).
                        towards = [c for c in ((kg + 1) % 8, (kg - 1) % 8)
                                   if _sep((v[0] + DIRS[c][0], v[1] + DIRS[c][1]), kg) < _sep(v, kg)]
                        if towards:
                            end_on, end_pts = lid, {sq for sq, (l2, _k) in laid.items() if l2 == lid}
                            converging, conv_d = True, _sep(v, kg)
                            turn_to(towards[0])
                else:
                    turn_to(kg)
            if not converging and not follow:
                e = math.cos(th) * (q[1] - a.y) - math.sin(th) * (q[0] - a.x)
                want = th + max(-1.0, min(1.0, -GAIN * e))
                kw = round(want / (math.pi / 4)) % 8
                turn = (kw - k) % 8
                skip = {start_on} if n < hold else set()
                if n >= max(LEG_MIN, hold) and turn:
                    k2 = (k + (1 if turn < 4 else -1 if turn > 4 else (1 if e < 0 else -1))) % 8
                    if not crowded(q, k2, skip | {end_on}):
                        turn_to(k2)
                # Beside a line laid already, the same way: run with it.
                if laid and n >= hold and steps >= alone:
                    pv = _perp(k)
                    for d in sorted(range(-PAR_NEAR, PAR_NEAR + 1), key=abs):
                        if abs(d) < MIN_SEP:
                            continue
                        got = laid.get((q[0] - d * pv[0], q[1] - d * pv[1]))
                        if got and got[1] % 4 == k % 4 and _turn_by(k * math.pi / 4, th) < math.radians(30):
                            follow = ((d * pv[0], d * pv[1]), got[0], got[1] != k)
                            break
        q = (q[0] + DIRS[k][0], q[1] + DIRS[k][1])
        n += 1
        steps += 1
    legs.append((start, DIRS[k], n))
    return legs, joints


def dedupe(lines):
    """The lines without the stretches that run within DUP_M of a line kept
    before them for DUP_LEN or more (the same track drawn twice - Alton's
    Watercress Line south of the station is two ways 0-0.4 m apart - or two
    tracks drawn as one where they share a stretch before parting). A line
    broken that way comes back as its pieces, each end where it was cut
    recorded as on the line it was cut against: [(points, tags, (start on,
    end on))], an index into the list returned, or None."""
    from shapely.geometry import LineString
    kept = []                       # (line, tags, [start on, end on])
    for pts, tags in lines:
        line = LineString(pts)
        n = max(2, int(line.length / SAMPLE_M) + 1)
        samples = [line.interpolate(i * line.length / (n - 1)) for i in range(n)]
        near = []
        for p_ in samples:
            hits = [(k_[0].distance(p_), idx) for idx, k_ in enumerate(kept)]
            hit = min(hits) if hits else (99.0, None)
            near.append(hit[1] if hit[0] <= DUP_M else None)
        # Only a long stretch is a double: a branch starts on its parent, and
        # its first metres are near it whatever it is.
        dup = [x is not None for x in near]
        i0 = 0
        while i0 < len(dup):
            if not dup[i0]:
                i0 += 1
                continue
            i1 = i0
            while i1 + 1 < len(dup) and dup[i1 + 1]:
                i1 += 1
            if (i1 - i0) * line.length / (n - 1) < DUP_LEN:
                dup[i0:i1 + 1] = [False] * (i1 - i0 + 1)
            i0 = i1 + 1
        run, run_start_on, prev_dup_on = [], None, None
        for idx, (p_, d) in enumerate(zip(samples, dup)):
            if not d:
                if not run:
                    run_start_on = prev_dup_on
                run.append((p_.x, p_.y))
                continue
            prev_dup_on = near[idx]
            if len(run) >= 2:
                kept.append((LineString(run), tags, [run_start_on, near[idx]]))
            run = []
        if len(run) >= 2:
            kept.append((LineString(run), tags, [run_start_on, None]))
    return [(list(ln.coords), tags, tuple(on)) for ln, tags, on in kept]


def spine_points(legs):
    """Every centre square of a snapped line in order, with the step taken from it."""
    out = []
    for anchor, d, n in legs:
        for st in range(n):
            out.append(((anchor[0] + st * d[0], anchor[1] + st * d[1]), d))
    if legs:
        a, d, n = legs[-1]
        out.append(((a[0] + n * d[0], a[1] + n * d[1]), d))
    return out


def levels(seq, tol=2, min_run=3 * LEG_MIN):
    """A track's wanted offsets, square by square, as a few flat stretches:
    runs within `tol` of their median, a run shorter than `min_run` folded
    into the neighbour nearer its value. A line drifting away from another
    moves across once, in one diagonal, not a square at a time."""
    runs = []
    i = 0
    while i < len(seq):
        j, vals = i, [seq[i]]
        while j + 1 < len(seq) and abs(seq[j + 1] - float(np.median(vals + [seq[j + 1]]))) <= tol:
            j += 1
            vals.append(seq[j])
        runs.append([i, j, int(round(float(np.median(vals))))])
        i = j + 1
    while len(runs) > 1:
        short = [r for r in range(len(runs)) if runs[r][1] - runs[r][0] + 1 < min_run]
        if not short:
            break
        r = min(short, key=lambda r: runs[r][1] - runs[r][0])
        a, b, v = runs[r]
        if r == 0:
            nb = 1
        elif r == len(runs) - 1:
            nb = r - 1
        else:
            nb = r - 1 if abs(runs[r - 1][2] - v) <= abs(runs[r + 1][2] - v) else r + 1
        runs[nb] = [min(runs[nb][0], a), max(runs[nb][1], b), runs[nb][2]]
        del runs[r]
    out = list(seq)
    for a, b, v in runs:
        out[a:b + 1] = [v] * (b - a + 1)
    return out


def to_legs(path):
    """A path of squares one step apart as [(anchor, direction, steps)] legs."""
    legs = []
    for p0, p1 in zip(path, path[1:]):
        d = (p1[0] - p0[0], p1[1] - p0[1])
        if legs and legs[-1][1] == d:
            a, _d, n = legs[-1]
            legs[-1] = (a, d, n + 1)
        else:
            legs.append((p0, d, 1))
    return legs


def bundle(spine_line, spine_legs, others, w, h, cut_on=None):
    """The corridor's other tracks laid with its spine (the line fitted to the
    grid on its own): each the spine moved across by a whole number of squares
    along `perp`. In passes, so the order the lines come in does not matter:

    1. each line's real offset from the spine, square by square along it;
    2. what each line starts and ends on: the spine or a line whose real
       offset is within JOIN_NEAR there (its parent at that end), else nothing;
    3. a crossover - shorter than CROSS_M, a different parent at each end - is
       a diagonal from one to the other, not a track of its own;
    4. the other tracks spread out from the spine in their real order, as far
       as their real offsets say and MIN_SEP at least from the next track in;
    5. each track's offsets as a few flat stretches (levels): a line drifting
       away moves across once;
    6. a track starting or ending on a parent starts or ends on its centre line;
    7. a track moves from one offset to the next on a 45-degree diagonal where
       the spine runs straight.

    Returns, per line of `others`, its path of squares (None if it is not one
    of the corridor's tracks) and its joints [(square, parent, direction out)];
    parent -1 is the spine."""
    from shapely.geometry import LineString, Point
    pts = spine_points(spine_legs)
    n_lines = len(others)
    if not pts:
        return [None] * n_lines, [[] for _ in others]
    straight = sum(1 for _p, d in pts if d[0] == 0) >= sum(1 for _p, d in pts if d[1] == 0)
    perp = (1, 0) if straight else (0, 1)
    along = np.array([spine_line.project(Point(p_)) for p_, _d in pts])
    J = len(pts)
    # 1. Real offsets (metres, + along perp).
    real = np.full((n_lines, J), np.nan)
    length = []
    for i, (opts, _tags) in enumerate(others):
        line = LineString(opts)
        length.append(line.length)
        n = max(2, int(line.length / SAMPLE_M) + 1)
        ss, oo = [], []
        for k in range(n):
            q = line.interpolate(k * line.length / (n - 1))
            sp = spine_line.project(q)
            b0 = spine_line.interpolate(max(0.0, sp - 1.0))
            b1 = spine_line.interpolate(min(spine_line.length, sp + 1.0))
            tx, ty = b1.x - b0.x, b1.y - b0.y
            tl = math.hypot(tx, ty) or 1.0
            base = spine_line.interpolate(sp)
            o = (tx * (q.y - base.y) - ty * (q.x - base.x)) / tl
            side = (tx * perp[1] - ty * perp[0]) / tl
            ss.append(sp)
            oo.append(o if side >= 0 else -o)
        ss, oo = np.array(ss), np.array(oo)
        order_ = np.argsort(ss)
        ss, oo = ss[order_], oo[order_]
        if (np.abs(oo) <= CORRIDOR_M).mean() < 0.7:
            continue
        here = (along >= ss[0]) & (along <= ss[-1])
        real[i, here] = np.interp(along[here], ss, oo)
    span = [(int(np.flatnonzero(~np.isnan(real[i]))[0]), int(np.flatnonzero(~np.isnan(real[i]))[-1]))
            if (~np.isnan(real[i])).any() else None for i in range(n_lines)]

    # 2. Parents at each end: the line its real end point lies on (JOIN_M).
    shapes = [LineString(o[0]) for o in others]

    def parent_of(i, end):
        e = Point(end)
        best, who = JOIN_M, None
        if spine_line.distance(e) <= best:
            best, who = spine_line.distance(e), -1
        for k, sh in enumerate(shapes):
            if k != i and span[k] is not None and sh.distance(e) < best:
                best, who = sh.distance(e), k
        return who
    # Which end of the line is at the start of its span along the spine; an
    # end dedupe cut is on the line it was cut against.
    ends = []
    for i, sp_ in enumerate(span):
        if sp_ is None:
            ends.append((None, None))
            continue
        a_, b_ = others[i][0][0], others[i][0][-1]
        ca, cb = cut_on[i] if cut_on else (None, None)
        pa = ca if ca is not None else parent_of(i, a_)
        pb = cb if cb is not None else parent_of(i, b_)
        if spine_line.project(Point(a_)) > spine_line.project(Point(b_)):
            pa, pb = pb, pa
        ends.append((pa, pb))
    # 3. Crossovers.
    cross = [span[i] is not None and length[i] < CROSS_M and None not in ends[i] and ends[i][0] != ends[i][1]
             for i in range(n_lines)]
    # 4. The tracks spread out from the spine.
    want = np.full((n_lines, J), np.nan)
    for j in range(J):
        present = [i for i in range(n_lines) if not np.isnan(real[i, j]) and not cross[i]]
        for sign in (1, -1):
            last = 0
            for i in sorted((i for i in present if sign * real[i, j] >= 0), key=lambda i: sign * real[i, j]):
                last = max(last + MIN_SEP, int(round(sign * real[i, j])))
                want[i, j] = sign * last
    # 5. Flat stretches.
    held = {}
    for i in range(n_lines):
        if span[i] is None or cross[i]:
            continue
        j0, j1 = span[i]
        held[i] = levels([int(want[i, j]) for j in range(j0, j1 + 1)])

    def offset_of(k, j):
        """Track k's offset at spine square j (or the nearest of its own within
        a leg or so: a crossover's end can fall just short of its parent's span)."""
        if k == -1:
            return 0
        if k in held and span[k][0] - 2 * LEG_MIN <= j <= span[k][1] + 2 * LEG_MIN:
            jj = min(max(j, span[k][0]), span[k][1])
            return held[k][jj - span[k][0]]
        return None
    # 6. Ends on their parents (7, crossovers: one parent's offset, then the other's).
    start_c = {}
    for i in range(n_lines):
        if span[i] is None:
            continue
        j0, j1 = span[i]
        ps, pe = ends[i]
        c0 = offset_of(ps, j0) if ps is not None else None
        c1 = offset_of(pe, j1) if pe is not None else None
        if cross[i]:
            if c0 is None or c1 is None:
                continue
            mid = (j1 - j0) // 2
            first = max(0, mid - abs(c1 - c0))     # the diagonal ends at the middle of its span
            held[i] = [c0] * first + [c1] * (j1 - j0 + 1 - first)
        elif i in held:
            if c1 is not None:
                for idx in range(max(0, len(held[i]) - abs(held[i][-1] - c1) - 2 * LEG_MIN), len(held[i])):
                    held[i][idx] = c1
            else:
                pe = None
        start_c[i] = c0 if c0 is not None else (held[i][0] if i in held else None)
        ends[i] = (ps if c0 is not None else None, pe if c1 is not None else None)
    if os.environ.get("RAILS_DEBUG"):
        for i in range(n_lines):
            hv = held.get(i)
            print("  bundle", i, "span", span[i], "ends", ends[i], "cross", cross[i], "start_c", start_c.get(i),
                  "levels", sorted(set(hv)) if hv else None, "first", hv[:3] if hv else None, "last", hv[-3:] if hv else None)
    # Where the spine runs straight: steps back to its last bend and straight
    # steps on from each square (a track moves across only inside such a run,
    # LEG_MIN clear of either end: no 90-degree turns, no legs of 1-3 squares).
    straight_j = [(d[0] == 0) if perp == (1, 0) else (d[1] == 0) for _p, d in pts]
    run_on = [0] * J
    for j in range(J - 1, -1, -1):
        run_on[j] = (run_on[j + 1] + 1 if j + 1 < J else 1) if straight_j[j] else 0
    since = [0] * J
    for j in range(J):
        since[j] = (since[j - 1] + 1 if j else 1) if straight_j[j] else 0
    # 7. Paths.
    paths, joints = [], []
    for i in range(n_lines):
        if i not in held or start_c.get(i) is None:
            paths.append(None)
            joints.append([])
            continue
        j0, j1 = span[i]
        c, moved, path = start_c[i], -LEG_MIN, []
        for idx, j in enumerate(range(j0, j1 + 1)):
            p_, d = pts[j]
            path.append((p_[0] + c * perp[0], p_[1] + c * perp[1]))
            need = abs(held[i][idx] - c)
            starting = moved != idx - 1
            if (j < j1 and held[i][idx] != c and straight_j[j]
                    and (not starting or (idx - moved >= LEG_MIN and since[j] >= LEG_MIN
                                          and run_on[j] >= need + LEG_MIN))):
                c += 1 if held[i][idx] > c else -1
                moved = idx
        legs_i = to_legs(path)
        mine = []
        ps, pe = ends[i]
        if legs_i and ps is not None and legs_i[0][1][0] and legs_i[0][1][1]:
            mine.append((path[0], ps, DIRS.index(legs_i[0][1])))
        if legs_i and pe is not None and legs_i[-1][1][0] and legs_i[-1][1][1]:
            d_last = legs_i[-1][1]
            mine.append((path[-1], pe, DIRS.index((-d_last[0], -d_last[1]))))
        paths.append(path)
        joints.append(mine)
    return paths, joints


def lay(legs, table):
    """{square: sprite} of a snapped line: steady legs, vanilla's pieces at each bend."""
    out = {}
    for anchor, d, n in legs:
        out.update(steady(anchor, d, 0, n))
    for (a1, d1, n1), (a2, d2, _n2) in zip(legs, legs[1:]):
        anchor = a2
        back = (-d1[0], -d1[1])
        key = bend_key(back, d2)
        bend = table["bends"].get(key)
        if bend is None:
            continue
        kd = parse_key(key)
        for sq, spr in bend_steady(kd, bend["radius"]).items():
            out[(anchor[0] + sq[0], anchor[1] + sq[1])] = spr
        for dx, dy, spr in bend["pieces"]:
            at = (anchor[0] + dx, anchor[1] + dy)
            if spr is None:
                out.pop(at, None)
            else:
                out[at] = spr
    return out


# --- the map -----------------------------------------------------------------------------------

def _digest(path: str) -> str:
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def is_colour(img, colours):
    m = np.zeros(img.shape[:2], bool)
    for c in colours:
        m |= (img == np.array(c, np.uint8)).all(axis=2)
    return m


def _setup(map_dir):
    from generator.renderer import Projector
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    b = info["bbox"]
    proj = Projector.build(b["south"], b["west"], b["north"], b["east"], info["meters_per_tile"],
                           info.get("rotation") or 0.0)
    return name, info, proj


def _buildings(map_dir, name, proj, h, w):
    """Squares under a building, as plots2.py rasters them."""
    from shapely.geometry import Polygon
    from ground import plotlib as P
    out = np.zeros((h, w), bool)
    feats = json.load(open(os.path.join(map_dir, f"{name}_buildings.geojson"), encoding="utf-8"))["features"]
    for ft in feats:
        if ft["geometry"]["type"] != "Polygon" or (ft["properties"].get("building") or "").lower() in P.NOT_BUILDINGS:
            continue
        poly = Polygon([proj.to_px(la, lo) for lo, la in ft["geometry"]["coordinates"][0]])
        rr, cc = P.raster(poly if poly.is_valid else poly.buffer(0), w, h)
        out[rr, cc] = True
    return out


def layout(map_dir) -> int:
    """Where the track goes (run after blocks.py, before plots2.py, which takes
    its bed as the railway): <map>_rails.json."""
    name, info, proj = _setup(map_dir)
    table = build_table() if ("--rebuild-table" in sys.argv or not os.path.exists(TABLE)) \
        else json.load(open(TABLE))
    # The ground as the hand-over left it, as plots2.py reads it (later steps' changes not in it).
    gb = os.path.join(map_dir, f"{name}_ground_base.pretrees.bmp")
    gb = gb if os.path.exists(gb) else os.path.join(map_dir, f"{name}_ground_base.bmp")
    ground = np.array(Image.open(gb).convert("RGB"))
    h, w = ground.shape[:2]
    building = _buildings(map_dir, name, proj, h, w)
    water = is_colour(ground, [C.WATER])
    street = is_colour(ground, STREET)
    road = is_colour(ground, ROAD)

    # Generate's bridges (its own structures.json, kept as .orig once sports.py
    # has run). The railway's own come back to the ground, the road under them
    # becoming a level crossing as vanilla has them (user, 2026-10-18): their
    # decks, ramps, railings and posts go (apply). A road's bridge over the
    # line stays, the track under it.
    lines, bridges = osm_lines(map_dir, info, proj)
    sp = os.path.join(map_dir, f"{name}_structures.json")
    orig = os.path.join(map_dir, f"{name}_structures.orig.json")
    src = orig if os.path.exists(orig) else sp
    rows = (json.load(open(src, encoding="utf-8")).get("tiles") or []) if os.path.exists(src) else []
    deck = np.zeros((h, w), bool)
    ramp = np.zeros((h, w), bool)
    for x, y, z, layer, tile in rows:
        if 0 <= x < w and 0 <= y < h and layer == "Floor":
            if z >= 1:
                deck[y, x] = True
            elif tile.startswith("ramps_01"):
                ramp[y, x] = True
    parts, _n = ndimage.label(deck | ramp, np.ones((3, 3), bool))
    rail_decks = set(np.unique(parts[deck & _near_lines(bridges, h, w)]).tolist()) - {0}
    lowered = np.isin(parts, list(rail_decks))
    overhead = (deck | ramp) & ~lowered

    track, owner, legs_of = {}, {}, []
    not_laid = collections.Counter()
    worst = 0.0
    from shapely.geometry import LineString, Point
    # The tracks: OSM's lines without the stretches drawn twice; the corridor's
    # spine (the line longest on the map) fitted to the grid on its own, and the
    # lines running with it laid as its tracks, side by side (bundle); a line
    # off the corridor fitted on its own, joining what it meets.
    pieces = dedupe(lines)
    if not pieces:
        # No railway on this map: an empty layout, so plots2.py and --apply have one to read.
        json.dump({"track": [], "signals": [], "bed": [], "land": [], "lowered": [], "lines": []},
                  open(os.path.join(map_dir, f"{name}_rails.json"), "w"))
        print(f"{name}: no railway")
        return 0
    lines = [(p_, t_) for p_, t_, _on in pieces]
    cut_on = [on for _p, _t, on in pieces]

    def on_map(pts_):
        ln = LineString(pts_)
        from shapely.geometry import box as _box
        return ln.intersection(_box(0, 0, w, h)).length
    spine = max(range(len(lines)), key=lambda i: on_map(lines[i][0]))
    spine_legs, _j = snap(lines[spine][0])
    others = [i for i in range(len(lines)) if i != spine]
    as_bundle = {spine: -1, **{i: k for k, i in enumerate(others)}}
    paths, bjoints = bundle(LineString(lines[spine][0]), spine_legs, [lines[i] for i in others], w, h,
                            [tuple(as_bundle.get(x) if x is not None else None for x in cut_on[i]) for i in others])
    legs_by = {spine: spine_legs}
    laid, branch_joints = {}, []

    def note(i, legs):
        for anchor, d, n in legs:
            kk = DIRS.index(d)
            for st in range(n + 1):
                laid.setdefault((anchor[0] + st * d[0], anchor[1] + st * d[1]), (i, kk))
    note(spine, spine_legs)
    alone = []
    for k, i in enumerate(others):
        if paths[k] is None or len(paths[k]) < 2:
            alone.append(i)
            continue
        legs_by[i] = to_legs(paths[k])
        note(i, legs_by[i])
        branch_joints += [(sq, spine if par == -1 else others[par], kb) for sq, par, kb in bjoints[k]]
    shapes = [LineString(p_) for p_, _t in lines]

    def meets(i, end):
        if not (0 <= end[0] < w and 0 <= end[1] < h):
            return None                     # off the map: nothing to join there
        for j, sh in enumerate(shapes):
            if j != i and j in legs_by and sh.distance(Point(end)) <= JOIN_NEAR:
                return j
        return None
    for i in alone:
        pts_ = lines[i][0]
        legs, joints = snap(pts_, laid, meets(i, pts_[0]), meets(i, pts_[-1]))
        legs_by[i] = legs
        note(i, legs)
        branch_joints += joints
    stats = {"tracks with the spine": len(legs_by) - 1 - len(alone), "on their own": len(alone)}
    for i, legs in legs_by.items():
        line = LineString(lines[i][0])
        for anchor, d, n in legs:
            for s_ in range(0, n + 1, 3):
                p = (anchor[0] + s_ * d[0], anchor[1] + s_ * d[1])
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    worst = max(worst, line.distance(Point(p)))
        legs_of.append(legs)
        for sq, spr in lay(legs, table).items():
            x, y = sq
            if not (0 <= x < w and 0 <= y < h):
                continue
            why = ("building" if building[y, x] else "water" if water[y, x]
                   else "a road bridge's ramp" if overhead[y, x] and ramp[y, x]
                   else "another line" if sq in track and owner[sq] != i else None)
            if why:
                not_laid[why] += 1
                continue
            track[sq] = spr
            owner[sq] = i

    # Where a branch leaves or joins a line: vanilla's points there where it
    # has that kind of joint, else the line it meets kept whole over it.
    with_points = 0
    for sq, other, kb in branch_joints:
        got = laid.get(sq)
        if not got or got[0] != other:
            continue
        kt = got[1]
        key = joint_key([DIRS[kt], DIRS[(kt + 4) % 8], DIRS[kb]])
        jt = table["bends"].get(key)
        if jt:
            # Only where the track round it is the joint's own legs: a branch
            # leaving where the spine bends is not vanilla's kind of joint.
            r = jt["radius"]
            dt, db = DIRS[kt], DIRS[kb]
            fits = all(laid.get((sq[0] + t * dt[0], sq[1] + t * dt[1]), (None,))[0] == other
                       and laid.get((sq[0] - t * dt[0], sq[1] - t * dt[1]), (None,))[0] == other
                       and (sq[0] + t * db[0], sq[1] + t * db[1]) in laid for t in range(1, r + 1))
            if not fits:
                jt = None
        if jt:
            with_points += 1
            put = dict((((sq[0] + dx, sq[1] + dy), spr) for (dx, dy), spr in
                        bend_steady(parse_key(key), jt["radius"]).items()))
            for dx, dy, spr in jt["pieces"]:
                put[(sq[0] + dx, sq[1] + dy)] = spr
        else:
            put = steady(sq, DIRS[kt], -3, 3)
        for (x, y), spr in put.items():
            if not (0 <= x < w and 0 <= y < h) or building[y, x] or water[y, x]:
                continue
            if spr is None:
                track.pop((x, y), None)
            else:
                track[(x, y)] = spr
                owner[(x, y)] = other

    # The bed: dirt under the track and a square round it, off roads, buildings and water.
    on = np.zeros((h, w), bool)
    for x, y in track:
        on[y, x] = True
    bed = ndimage.binary_dilation(on, np.ones((3, 3), bool)) & ~street & ~building & ~water
    # The railway's land: the bed and any strip too narrow for a garden between
    # its lines (plots2.py: no garden, so no fence, between two tracks).
    pad = LAND_GAP
    land = ndimage.binary_closing(np.pad(on | bed, pad), np.ones((LAND_GAP, LAND_GAP), bool))[pad:-pad, pad:-pad]
    land &= ~street & ~building & ~water

    # Crossing signals where a straight leg crosses a road at its own level.
    signals = []
    for legs in legs_of:
        for anchor, d, n in legs:
            fam = FAMILY[d]
            if fam not in ("NS", "EW"):
                continue
            centre = [(anchor[0] + s * d[0], anchor[1] + s * d[1]) for s in range(n + 1)]
            inside = [0 <= x < w and 0 <= y < h for x, y in centre]
            on_road = [ok and road[y, x] and not overhead[y, x] for ok, (x, y) in zip(inside, centre)]
            s = 0
            while s < len(centre):
                if not on_road[s]:
                    s += 1
                    continue
                # The road and the pavements either side of it.
                e = s
                while e + 1 < len(centre) and inside[e + 1] and (
                        on_road[e + 1] or street[centre[e + 1][1], centre[e + 1][0]]):
                    e += 1
                while s - 1 >= 0 and inside[s - 1] and street[centre[s - 1][1], centre[s - 1][0]]:
                    s -= 1
                ends = sorted([centre[max(0, s - 1)], centre[min(len(centre) - 1, e + 1)]])
                # Diagonally opposite corners: the + side at the low end, the - side at the high end.
                for (cx, cy), side in ((ends[0], 1), (ends[1], -1)):
                    for out in SIGNAL_OUT:
                        x, y = (cx + side * out, cy) if fam == "NS" else (cx, cy + side * out)
                        if (0 <= x < w and 0 <= y < h and not street[y, x] and not building[y, x]
                                and not water[y, x] and (x, y) not in track):
                            signals.append([x, y, 0, SIGNAL_LAYER, SHEET + SIGNAL[(fam, side)]])
                            break
                s = e + 1

    out = {"track": [[x, y, spr] for (x, y), spr in sorted(track.items())], "signals": signals,
           "bed": np.argwhere(bed)[:, ::-1].tolist(), "land": np.argwhere(land)[:, ::-1].tolist(),
           "lowered": np.argwhere(lowered)[:, ::-1].tolist(),
           "lines": [[list(p) for p in pts] for pts, _t in lines]}
    json.dump(out, open(os.path.join(map_dir, f"{name}_rails.json"), "w"))
    bends = sum(max(0, len(legs) - 1) for legs in legs_of)
    print(f"{name}: {len(lines)} lines after the doubles; {stats}; branches joined {len(branch_joints)} "
          f"(with vanilla's points {with_points})")
    print(f"{name}: {len(legs_of)} lines, {bends} bends; track squares {len(track)}, bed {int(bed.sum())}; "
          f"signals {len(signals)}; railway bridges brought to the ground {len(rail_decks)} "
          f"({int(lowered.sum())} squares); furthest from the real line {worst:.1f} tiles; "
          f"not laid {dict(not_laid)}")
    return 0


def apply(map_dir) -> int:
    """The track into the map (run after sports.py, before kerbs.py): its tiles
    in structures.json ahead of kerbs.py's, the railway's bridges out, the bed
    painted."""
    name = os.path.basename(os.path.normpath(map_dir))
    lay_path = os.path.join(map_dir, f"{name}_rails.json")
    if not os.path.exists(lay_path):
        print(f"{name}: no {os.path.basename(lay_path)} - run rails.py before plots2.py")
        return 1
    rl = json.load(open(lay_path))
    # Ground as it was before this step (taken afresh when another step wrote it since).
    files = [os.path.join(map_dir, f"{name}_ground_base.bmp"), os.path.join(map_dir, f"{name}_veg_base.bmp")]
    state_path = os.path.join(map_dir, f"{name}_rails_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    # kerbs.py writes these after this step: a file it wrote last is no start
    # for this one (the old bed is in it) - this step's own copy is, or on a map
    # never railed before, the ground as it was before kerbs.py.
    kp = os.path.join(map_dir, f"{name}_kerbs_state.json")
    kerbs_state = json.load(open(kp)) if os.path.exists(kp) else {}
    for f in files:
        keep = f.replace(".bmp", ".prerails.bmp")
        now = _digest(f)
        if kerbs_state.get(os.path.basename(f)) == now:
            pre = f.replace(".bmp", ".prekerbs.bmp")
            # kerbs.py took a fresh pre-kerbs copy after this step's own (another
            # step rewrote the file in between): that copy is the newer start.
            if not os.path.exists(keep) or (os.path.exists(pre) and os.path.getmtime(pre) > os.path.getmtime(keep)):
                shutil.copy(pre, keep)
        elif not os.path.exists(keep) or state.get(os.path.basename(f)) != now:
            shutil.copy(f, keep)
    ground, veg = (np.array(Image.open(f.replace(".bmp", ".prerails.bmp")).convert("RGB")) for f in files)
    h, w = ground.shape[:2]
    bed = np.zeros((h, w), bool)
    for x, y in rl["bed"]:
        bed[y, x] = True
    bed &= ~is_colour(ground, STREET + (C.WATER,))
    old = is_colour(ground, [C.LIGHT_ASPHALT]) & _near_lines([(p, None) for p in rl["lines"]], h, w) & ~bed
    if old.any():
        # From the nearest square that is neither old bed nor new (the new bed
        # is still Generate's gravel until painted below).
        _d, (iy, ix) = ndimage.distance_transform_edt(old | bed, return_indices=True)
        ground[old] = ground[iy[old], ix[old]]
    ground[bed] = C.DIRT
    veg[bed] = C.VEG_NOTHING

    sp = os.path.join(map_dir, f"{name}_structures.json")
    data = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {"bridges": 0, "monuments": 0, "tiles": []}
    rows = data.get("tiles") or []
    kerbs = data.get("kerbs_added") or 0
    mine = data.get("rails_added") or 0
    if kerbs + mine > len(rows):
        raise ValueError(f"{sp}: rails_added={mine} + kerbs_added={kerbs} but only {len(rows)} tiles - "
                         "rerun from sports.py")
    head, tail = rows[:len(rows) - kerbs - mine], rows[len(rows) - kerbs:]
    lowered = {tuple(p) for p in rl["lowered"]}
    # A bridge's railings and posts can stand a square or two off its deck.
    near = {(x + dx, y + dy) for x, y in lowered for dx in range(-2, 3) for dy in range(-2, 3)}

    def bridge_bit(r):
        if r[4].startswith("fencing_01"):
            return (r[0], r[1]) in near
        return (r[0], r[1]) in lowered and ((r[3] == "Floor" and r[2] >= 1) or r[4].startswith("ramps_01"))
    kept = [r for r in head if not bridge_bit(r)]
    new = [[x, y, 0, TRACK_LAYER, SHEET + spr] for x, y, spr in rl["track"]] + rl["signals"]
    data["tiles"] = kept + new + tail
    data["rails_added"] = len(new)
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    for f, img in zip(files, (ground, veg)):
        Image.fromarray(img).save(f, format="BMP")
        state[os.path.basename(f)] = _digest(f)
    json.dump(state, open(state_path, "w"))
    print(f"{name}: track squares {len(rl['track'])}, signals {len(rl['signals'])}, bed painted {int(bed.sum())}, "
          f"old bed returned to the ground beside it {int(old.sum())}; bridge tiles taken out {len(head) - len(kept)}")
    return 0


def main() -> int:
    return apply(sys.argv[1]) if "--apply" in sys.argv else layout(sys.argv[1])


def _near_lines(lines, h, w, reach=4.0):
    """Squares within reach of the real lines (where Generate painted the bed)."""
    from PIL import ImageDraw
    img = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(img)
    for pts, _tags in lines:
        dr.line([(x, y) for x, y in pts], fill=1, width=int(2 * reach))
    return np.array(img, bool)


if __name__ == "__main__":
    sys.exit(main())
