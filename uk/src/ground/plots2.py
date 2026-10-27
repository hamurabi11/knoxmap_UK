"""Garden plots as shapes, their fronts on the street's own line.

    python plots2.py <knoxmap map dir> <area data dir> [--picture out.png] [--box x,y,w,h]
                     [--inspire-attrs <titles.geojson>]

Writes <map>_fences.geojson (from its .orig: the plots' boundaries, source=plots,
each with its exact line in tiles as `tiles`) and <map>_plots.npz (plot, street,
building, drive, home, forecourt, frames), read by trees.py, sports.py, kerbs.py
and Build. Reads the ground as the road graph's hand-over left it
(<map>_ground_base.pretrees.bmp, else _ground_base.bmp), never Build's own
<map>.bmp, so a second run gives the same plots.

1. The street, one shape: on a road-graph map the graph's own outlines
   (roadside.outlines: carriageway and bays, and verge and pavement where the
   hand-over painted them - in town), on their 45-degree and 2:1 lattice;
   otherwise the painted street. Footpaths from OSM, where Generate moved
   them (plotlib.footpaths), laid on the lattice; water, fields, car parks:
   no plot land either. Drives (the narrow dark tarmac) are plot land, kept
   as their own mask for Build.
2. Each block (a piece of plot land) is cut on the grid through the widest
   gaps between buildings until each piece holds one property (cut).
   With --inspire-attrs, HM Land Registry's titles are evidence
   (ground/inspire.py attributes): a garage or shed in a home's title,
   near it and smaller, is that home's (one property, one plot, the home
   leading it); a home alone in its title has its plot trimmed to the
   title's reach from the house (N, E, S, W, plus REACH_SLACK), never on its
   street side, nor where the title is tight on the house (under
   REACH_GARDEN of garden: the garden is a title of its own), a narrow
   strip it gives up going to its neighbour; a home whose title reaches no
   road gets no made-up drive; a road running into a title and stopping
   there is its drive (Build leaves the front open where it meets the
   street).
3. The front of each block is shared out house by house (frontage): along
   the street face a house is nearest, each house takes the strip from the
   street back to its rear wall, between the midpoints to its neighbours,
   its sides square to the face (so on the lattice too). A grid cut that
   met an angled street off square left a wedge of the next plot in front of
   the house; this is the one rule that replaces the old shadow, ribbon and
   gap passes. Land no plot got, under GAP_TILES across between plots, goes
   to the plot it borders most; a piece of a plot under 2 squares across
   likewise, and a strip along the street under STRIP_TILES across (a
   plot's arm in front of other land, or land no plot got) to the plot
   behind it (never a plot's only frontage).
4. Boundaries: every plot outline (holes too) as one noded set of lines,
   each piece classed by what lies either side (plot, street, nothing), a
   drive's mouth left open; one kind per boundary by LiDAR vote, one style
   per boundary from its lower-numbered home plot; a stretch running off the
   lattice laid as lattice runs (lattice_runs), never across the street.
"""
from __future__ import annotations

import csv
import json
import math
import os
import pickle
import re
import shutil
import sys
from collections import Counter

import numpy as np
import shapely
import shapely.affinity
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from area import area_conf  # noqa: E402
from buildings import blocks as B  # noqa: E402
from ground import inspire as INS  # noqa: E402
from ground import plotlib as P  # noqa: E402
from ground.plotlib import WALL_SHARE, Heights, raster  # noqa: E402
from generator import pz_colors as C  # noqa: E402
from knoxbuild.areas import AreaIndex  # noqa: E402

MIN_LINE = 0.5             # tiles: a boundary piece shorter than this is left out
FRONT_PROBE = 0.6          # tiles past a side its far side is read at
GRID = 1e-3                # every plot shape is snapped to this grid, so neighbours' outlines coincide
FRONT_REACH = 12           # tiles: a house further than this from the street has no frontage strip
FRONT_STYLES = P.FRONT_STYLES
TREND = 1.6                # tiles: a line simplified this much shows the way it runs (a 2:1 staircase stays 2:1)
LATTICE = (0.0, 0.5, 1.0)  # minor / major slopes Build draws as a line: along the grid, 2:1, 45 degrees
LATTICE_TOL = 0.04         # a trend this near a lattice slope is on it: left exactly as it is
OFF_LATTICE = (0.2, 0.85)  # trends between these slopes (1:3, 2:3...) are laid as lattice runs; shallower is jogs
LATTICE_DRIFT = 1.5        # tiles the lattice runs may stray from the line (all to the garden side on a front)
VERGE_REACH = 3            # tiles: open ground this near the street is its verge (a boundary behind it is a front)
JOIN_MAX_TILES = 120.0     # tiles: ...if it is no bigger than this, nor than the home (a school is no one's garage)
JOIN_TILES = 10.0          # tiles: --inspire-attrs joins a garage or shed to the home in its title this near it (further: across a lane, the next block)
HOUSE_TAGS = {"house", "detached", "semidetached_house", "terrace", "bungalow", "residential", "cottage",
              "farm", "apartments", "dwelling_house"}   # ...never one OSM calls a house
REACH_SLACK = 2.0          # tiles: --inspire-attrs trims a plot no closer to its house than its title's reach plus this
REACH_GARDEN = 4.0         # tiles: ...and only where that leaves this much garden past the house's wall
REACH_STREET = 6.0         # tiles: ...and not on a side where the street is within its reach plus this (the front)
DRIVE_TILES = 3.0          # tiles: a drive from OSM is this wide
STUB_TILES = 2.0           # tiles: a plot boundary shorter than this with an end on a building goes (a hedge piece between a house's corner and its front wall)
OSM_MIN_TILES = 6.0        # tiles: an OSM fence kept off the plots shorter than this goes (bits of a way cut by paths and roads; a short wall stays)
NUB = 0.3                  # tiles: a boundary's points this near the straight line past them go (a 0.2-tile nub in a hedge)
TAKE_FILL = 0.75           # frontage takes a piece of a neighbour's plot only if it fills this much of its turned bounding rectangle
STRIP_TILES = 4            # tiles: a part of a plot (or land no plot got) thinner than this, along the street, goes to the plot behind



# Boundaries by group (see the plots' step 4): the weight on each kind's
# evidence when a group's kind is voted, fronts and sides/backs apart - raise
# one to see more of it (FRONT_BIAS["fence"] 1.5: more low fences along the
# street) - and each (footing, kind, setting) its style.
FRONT_JOIN = 2.5          # tiles: a front run carries on across a gap this wide (a drive's mouth)
BLOCK_PLOTS = 12          # plots at most in a block whose sides and backs share a kind
FRONT_BIAS = {"hedge": 1.0, "wall": 1.0, "fence": 1.0}
BACK_BIAS = {"hedge": 1.0, "wall": 1.0, "fence": 1.0}
BOUNDARY_STYLE = {
    ("front", "hedge", "old"): "hedge_low", ("front", "hedge", "town"): "hedge_low", ("front", "hedge", "country"): "hedge_low",
    ("front", "wall", "old"): "uk_brick_low", ("front", "wall", "town"): "uk_brick_low", ("front", "wall", "country"): "uk_brick_low",
    ("front", "fence", "old"): "uk_brick_low", ("front", "fence", "town"): "uk_brick_low", ("front", "fence", "country"): "short_wooden",
    ("back", "hedge", "old"): "hedge_tall", ("back", "hedge", "town"): "hedge_tall", ("back", "hedge", "country"): "hedge_tall",
    ("back", "wall", "old"): "uk_brick_tall", ("back", "wall", "town"): "uk_brick_tall", ("back", "wall", "country"): "uk_brick_tall",
    ("back", "fence", "old"): "tall_wooden", ("back", "fence", "town"): "hedge_tall", ("back", "fence", "country"): "tall_wooden",
}

def mask_shape(mask: np.ndarray):
    """A boolean tile mask as one shape: the union of its rows' runs."""
    boxes = []
    for y in range(mask.shape[0]):
        row = mask[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            boxes.append(box(int(a), y, int(b), y + 1))
    return unary_union(boxes) if boxes else Polygon()


def graph_street(map_dir: str, name: str):
    """The road graph's street, exact, or None on a map without one: each
    link's carriageway and bays, and its verge and pavement only where the
    hand-over painted them (handover.repaint: in town). Outside town the
    graph still has a pavement band that was never painted - taken as street
    it left 13,000 tiles of grass in no plot and fronts 2 tiles back from the
    kerb (2026-10-13 review)."""
    path = os.path.join(map_dir, f"{name}_roadgraph_state.pkl")
    if not os.path.exists(path):
        return None
    from roadgraph import raster as R, roadside
    links, paths, proj, _owner, _half, town, chosen = pickle.load(open(path, "rb"))
    bands = roadside.outlines(paths, links, chosen, roadside.handedness(proj))
    town_shape = mask_shape(np.asarray(town, bool))
    parts = []
    for road_, _verge, pave in bands.values():
        parts.append(road_)
        if not pave.is_empty:
            parts.append(pave.intersection(town_shape))
    parts += [t for _, t in R.junction_cuts(paths, links)]
    return unary_union([p_ for p_ in parts if not p_.is_empty])


class Boxes:
    """Each plot's bounding box, kept up to date: the plots near a shape
    without testing all of them (Alton's 4,500 plots against each other took
    plots2 from a minute and a half to over twelve)."""

    def __init__(self, shapes):
        self.b = np.array([s_.bounds if not s_.is_empty else (np.inf, np.inf, -np.inf, -np.inf)
                           for s_ in shapes], float).reshape(-1, 4)

    def update(self, i, s_):
        self.b[i] = s_.bounds if not s_.is_empty else (np.inf, np.inf, -np.inf, -np.inf)

    def near(self, g) -> list:
        x0, y0, x1, y1 = g.bounds
        b = self.b
        return np.nonzero((b[:, 0] <= x1) & (b[:, 2] >= x0) & (b[:, 1] <= y1) & (b[:, 3] >= y0))[0].tolist()


def give_away(q, shapes, exclude=-1, boxes: Boxes | None = None) -> int:
    """Join bit q to the plot whose outline it shares most (not `exclude`):
    its index, or -1 if it touches none."""
    best, most = -1, 0.0
    qb = q.buffer(0.05)
    for o in (boxes.near(qb) if boxes is not None else range(len(shapes))):
        s_ = shapes[o]
        if o == exclude or s_.is_empty or not s_.intersects(qb):
            continue
        shared = qb.intersection(s_).area
        if shared > most:
            best, most = o, shared
    if best >= 0:
        shapes[best] = areal(shapely.union(areal(shapes[best]), areal(q), grid_size=GRID))
        if boxes is not None:
            boxes.update(best, shapes[best])
    return best


def polys(g) -> list:
    if g.is_empty:
        return []
    return [g] if g.geom_type == "Polygon" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]


def areal(g):
    """Only the polygons of an overlay's result (it can return lines and points beside them)."""
    ps = polys(g)
    return ps[0] if len(ps) == 1 else shapely.MultiPolygon(ps) if ps else Polygon()


def rings_of(g) -> list:
    """Every ring of a shape - outer and holes, of every part - as lines."""
    return [LineString(r.coords) for p_ in polys(g) for r in [p_.exterior, *p_.interiors]]


def cut(piece, members, bbox, seed):
    """A block cut along the grid through the widest gaps between buildings
    until each piece holds one property: [(seed building, shape)]."""
    out, stack = [], [(piece, members)]
    while stack:
        shp, mem = stack.pop()
        if shp.is_empty:
            continue
        here = [m for m in mem if seed[m]]
        if len(here) == 1:
            out.append((here[0], shp))
            continue
        if not here:
            continue
        best = None
        for axis in (0, 1):
            for gap, c, left, right in P.cuts_on(mem, bbox, axis):
                nl, nr = sum(seed[m] for m in left), sum(seed[m] for m in right)
                if nl and nr:
                    key = (gap, -abs(nl - nr))
                    if best is None or key > best[0]:
                        best = (key, axis, c, left, right)
        x0, y0, x1, y1 = shp.bounds
        if best is not None:
            _k, axis, c, left, right = best
            lo = box(x0 - 1, y0 - 1, c, y1 + 1) if axis == 0 else box(x0 - 1, y0 - 1, x1 + 1, c)
            stack.append((shp.intersection(lo), left))
            stack.append((shp.difference(lo), right))
            continue
        # No cut through a gap: a straight cut along the grid anyway, between
        # the middles of the two middle buildings along the axis they spread
        # most on (a split by nearest building gave slanted sides).
        mids = {m: ((bbox[m][0] + bbox[m][2]) / 2, (bbox[m][1] + bbox[m][3]) / 2) for m in mem}
        spread = [max(mids[m][a_] for m in here) - min(mids[m][a_] for m in here) for a_ in (0, 1)]
        axis = 0 if spread[0] >= spread[1] else 1
        order = sorted(here, key=lambda m: mids[m][axis])
        h = len(order) // 2
        c = round((mids[order[h - 1]][axis] + mids[order[h]][axis]) / 2)
        lo = box(x0 - 1, y0 - 1, c, y1 + 1) if axis == 0 else box(x0 - 1, y0 - 1, x1 + 1, c)
        left = set(order[:h]) | {m for m in mem if not seed[m] and mids[m][axis] < c}
        stack.append((shp.intersection(lo), [m for m in mem if m in left]))
        stack.append((shp.difference(lo), [m for m in mem if m not in left]))
    return out


def _face_segments(street) -> list:
    """The street's edge as straight segments, each (a, b) in tiles."""
    out = []
    for ring in rings_of(street):
        cs = list(ring.coords)
        out += [(a, b) for a, b in zip(cs, cs[1:]) if math.dist(a, b) > 1e-6]
    return out


def _lattice_step(u) -> float:
    """The distance between tile corners along direction u (unit): 1 along
    the grid, sqrt 2 at 45 degrees, sqrt 5 at 2:1; 0 off the lattice."""
    lo, hi = sorted((abs(u[0]), abs(u[1])))
    r = lo / hi if hi else 0.0
    for v, step in ((0.0, 1.0), (0.5, math.sqrt(5)), (1.0, math.sqrt(2))):
        if abs(r - v) < 1e-3:
            return step
    return 0.0


def frontage(shapes, plots, blds, street, buildings_shape, land) -> int:
    """Each house's strip of front: along the street face it is nearest, from
    the street back to its rear wall, between the midpoints to its
    neighbours on that face line, its sides square to the face (on the
    lattice when the face is). Land of another plot in it moves to the house
    - only in rectangles (TAKE_FILL: never a strip's corner as a wedge),
    not land in that plot's own front garden (its strip before its house's
    front wall: the deeper corners of a strip square to an angled street cut
    Vs into plots cut on the grid), never its last frontage, and never
    so that the neighbour's plot is cut in two (its far part was then given
    away whole: 67,000 tiles moved on the first run).
    Returns the tiles moved."""
    segs = _face_segments(street)
    if not segs:
        return 0
    for i_ in range(len(shapes)):
        shapes[i_] = areal(shapes[i_])
    land, street, buildings_shape = areal(land), areal(street), areal(buildings_shape)
    seg_tree = shapely.STRtree([LineString(s) for s in segs])
    count = Counter(plots)
    homes = []                                    # (line key, k, a, u, n, s0, s1, depth)
    for k, j in enumerate(plots):
        bd = blds[j]
        if not bd["home"]:
            continue
        x0, y0, x1, y1 = bd["bbox"]
        fp = (shapes[k] if count[j] > 1 else box(x0, y0, x1, y1)).intersection(buildings_shape)
        if fp.is_empty or fp.distance(street) > FRONT_REACH:
            continue
        near = [int(i) for i in seg_tree.query(fp.buffer(FRONT_REACH))]
        if not near:
            continue
        i = min(near, key=lambda i: (LineString(segs[i]).distance(fp), -math.dist(*segs[i])))
        a, b = segs[i]
        L = math.dist(a, b)
        u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        if not (u[0] > 1e-9 or (abs(u[0]) <= 1e-9 and u[1] > 0)):
            u = (-u[0], -u[1])                       # one way along every line, so positions compare
        nc = (-u[1], u[0])
        n = nc
        c = fp.centroid
        if (c.x - a[0]) * n[0] + (c.y - a[1]) * n[1] < 0:
            n = (-n[0], -n[1])                       # into the land, towards the house
        pts = [p_ for g in polys(fp) for p_ in g.exterior.coords]
        s = [(p_[0] - a[0]) * u[0] + (p_[1] - a[1]) * u[1] for p_ in pts]
        d = [(p_[0] - a[0]) * n[0] + (p_[1] - a[1]) * n[1] for p_ in pts]
        if min(d) < -0.5:
            continue                                 # the house stands on the face's line: no front
        # The face's line: its direction and its offset (one key for every
        # segment of a straight street).
        key = (round(u[0], 3), round(u[1], 3), round(a[0] * nc[0] + a[1] * nc[1], 1))
        homes.append([key, k, a, u, n, min(s), max(s), max(d), min(d)])
    # Neighbours along each face line: the strip ends midway between them,
    # snapped to tile corners along the face so its sides run through them.
    by_line: dict = {}
    for h in homes:
        by_line.setdefault(h[0], []).append(h)
    strips, fronts = {}, {}                      # each strip, and its part before the house's front wall
    for line in by_line.values():
        a, u = line[0][2], line[0][3]
        for h in line:                               # one origin per line
            off = (h[2][0] - a[0]) * u[0] + (h[2][1] - a[1]) * u[1]
            h[5] += off
            h[6] += off
        line.sort(key=lambda h: (h[5] + h[6]) / 2)
        step = _lattice_step(u)
        snap = (lambda v: round(v / step) * step) if step else (lambda v: v)
        for i, h in enumerate(line):
            lo = snap((line[i - 1][6] + h[5]) / 2) if i else h[5]
            hi = snap((h[6] + line[i + 1][5]) / 2) if i + 1 < len(line) else h[6]
            if hi - lo < 1.0:
                continue
            n, depth = h[4], h[7]
            corners = [(a[0] + u[0] * s_ + n[0] * d_, a[1] + u[1] * s_ + n[1] * d_)
                       for s_, d_ in ((lo, -1.0), (hi, -1.0), (hi, depth), (lo, depth))]
            q = shapely.set_precision(Polygon(corners), GRID)
            strips[h[1]] = areal(shapely.difference(areal(shapely.intersection(q, land, grid_size=GRID)),
                                                    street, grid_size=GRID))
            front_ = [(a[0] + u[0] * s_ + n[0] * d_, a[1] + u[1] * s_ + n[1] * d_)
                      for s_, d_ in ((lo, -1.0), (hi, -1.0), (hi, h[8] + 1.0), (lo, h[8] + 1.0))]
            fronts[h[1]] = areal(shapely.intersection(strips[h[1]], shapely.set_precision(Polygon(front_), GRID),
                                                      grid_size=GRID))
    moved = 0
    boxes = Boxes(shapes)
    for k, strip in strips.items():
        if strip.is_empty:
            continue
        for o in boxes.near(strip):
            if o == k or plots[o] == plots[k] or shapes[o].is_empty or not shapes[o].intersects(strip):
                continue
            take = areal(shapely.difference(areal(shapely.intersection(shapes[o], strip, grid_size=GRID)),
                                            buildings_shape, grid_size=GRID))       # land, never a house
            if o in fronts:                          # the neighbour's own front garden stays its
                take = areal(shapely.difference(take, fronts[o], grid_size=GRID))
            # Only rectangles (turned with an angled face, or not): the corner
            # of a strip square to a 45-degree street, cutting into a plot cut
            # on the grid, took a triangle - a V in the neighbour's straight side.
            take = areal(shapely.union_all([q_ for q_ in polys(take)
                                            if q_.area >= TAKE_FILL * shapely.oriented_envelope(q_).area]))
            if take.is_empty or take.area < 0.5:
                continue
            rest = areal(shapely.difference(shapes[o], take, grid_size=GRID))
            had_front = shapes[o].distance(street) < 1.0
            if rest.is_empty or (had_front and rest.distance(street) >= 1.0):
                continue
            if len(polys(rest)) > len(polys(shapes[o])):
                continue                             # across the neighbour's plot: it would be cut in two
            shapes[o] = rest
            shapes[k] = areal(shapely.union(shapes[k], take, grid_size=GRID))
            boxes.update(o, shapes[o])
            boxes.update(k, shapes[k])
            moved += int(round(take.area))
    return moved


def street_strips_to_behind(shapes, land, street, buildings_shape, boxes, drives=None) -> int:
    """Strips along the street to the plot behind (user, 2026-10-14: fences
    on the road or pavement's edge): a part of a plot under STRIP_TILES
    across, wholly within STRIP_TILES of the street and along it for 3 tiles
    or more - and land no plot got between the street and a plot, wholly
    within twice STRIP_TILES of the street (a wedge between an angled street
    and a plot square to the grid; wider open ground - a green - stays open)
    - goes to the plot behind it: the one sharing most of its outline that
    runs along the street (a plot beside it shares an edge running away from
    the street: never that one), at least half its length on the street.
    Each was a second front a strip back from the first (80% and 19% of
    Alton's fences set back from the street). Never a plot's only frontage
    (it keeps a tile of edge on the street), nor a part whose going would
    split its plot (the plot-in-pieces pass then gave the far piece away,
    building and all), nor a part holding a building or a drive (`drives`).
    `shapes` and `boxes` are edited; returns the strips moved."""
    moved = 0
    reach = street.buffer(STRIP_TILES + 0.05)
    wedge = street.buffer(2 * STRIP_TILES + 0.05)
    edge = street.buffer(0.1)
    shapely.prepare(reach)
    shapely.prepare(wedge)

    def behind(q, exclude=-1) -> int:
        """The plot behind q, or -1."""
        on_street = q.boundary.intersection(edge).length
        best, most = -1, 0.5 * on_street
        qb = q.buffer(0.05)
        for o in boxes.near(qb):
            if o == exclude or shapes[o].is_empty:
                continue
            shared = q.boundary.intersection(shapes[o].buffer(0.05))
            along = 0.0                              # the shared outline's length along the street
            for seg in getattr(shared, "geoms", [shared]):
                if seg.geom_type != "LineString":
                    continue
                c = list(seg.coords)
                for p0, p1 in zip(c, c[1:]):
                    n = math.dist(p0, p1)
                    rise = abs(street.distance(Point(p1)) - street.distance(Point(p0)))
                    along += math.sqrt(max(0.0, n * n - rise * rise))
            if along > most:
                best, most = o, along
        return best

    def take(q, o) -> None:
        shapes[o] = areal(shapely.union(shapes[o], areal(q), grid_size=GRID))
        boxes.update(o, shapes[o])
    for o in range(len(shapes)):
        s_ = shapes[o]
        if s_.is_empty or s_.distance(street) > 0.5:
            continue
        core = s_.buffer(-STRIP_TILES / 2, join_style=2).buffer(STRIP_TILES / 2, join_style=2)
        for q in polys(shapely.set_precision(s_.difference(core), GRID)):
            if (q.area < 2.0 or not reach.contains(q) or q.boundary.intersection(edge).length < 3.0
                    or q.intersects(buildings_shape) or (drives is not None and q.intersects(drives))):
                continue
            rest = areal(shapely.set_precision(shapes[o].difference(q), GRID))
            if rest.is_empty or rest.boundary.intersection(edge).length < 1.0:
                continue                             # its only frontage
            if len(polys(rest)) > len(polys(shapes[o])):
                continue                             # the plot would come apart
            to = behind(q, o)
            if to >= 0:
                shapes[o] = rest
                boxes.update(o, rest)
                take(q, to)
                moved += 1
    left = land.difference(unary_union(shapes + [buildings_shape]))
    for q in polys(shapely.set_precision(left, GRID)):
        if (q.area < 2.0 or not wedge.contains(q) or q.boundary.intersection(edge).length < 3.0
                or (drives is not None and q.intersects(drives))):
            continue                                 # (tarmac before a row - frontage parking, a shared drive: review, 2026-10-14)
        to = behind(q)
        if to >= 0:
            take(q, to)
            moved += 1
    return moved


def lattice_runs(a, b, side: int, drift: float = LATTICE_DRIFT) -> list:
    """Points from a to b in long runs Build draws as lines (LATTICE slopes),
    staying within LATTICE_DRIFT of the line a-b: on the side +1 / -1 of it
    along the world's minor axis (y when the line runs mostly along x; a
    front keeps to its garden) or either side (0)."""
    ax, ay = round(a[0]), round(a[1])               # runs from a tile corner: their corners are corners
    dx, dy = b[0] - ax, b[1] - ay
    major_x = abs(dx) >= abs(dy)
    L, m = (abs(dx), abs(dy)) if major_x else (abs(dy), abs(dx))
    sM, sm = (1 if (dx if major_x else dy) >= 0 else -1), (1 if (dy if major_x else dx) >= 0 else -1)
    r = m / L
    lo = max(v for v in LATTICE if v <= r)
    hi = min(v for v in LATTICE if v >= r)
    rel = side * sm                                  # the side in the line's own minor direction
    band = (0.0, drift) if rel > 0 else (-drift, 0.0) if rel < 0 else (-drift / 2, drift / 2)
    moves = {v: ((2, 1) if v == 0.5 else (1, int(v))) for v in (lo, hi)}       # (major, minor) per move
    t = e = 0.0
    slope = hi if band[0] + band[1] > 0 else lo
    pts = [(ax, ay)]
    while True:
        # Stay on the slope while the next move keeps within the band; else
        # change slope (checked before the move, so the band holds).
        for cand in (slope, lo if slope == hi else hi):
            mv = moves[cand]
            if band[0] - 1e-9 <= e + mv[1] - r * mv[0] <= band[1] + 1e-9:
                slope = cand
                break
        else:
            slope = min((lo, hi), key=lambda v: abs(e + moves[v][1] - r * moves[v][0] - sum(band) / 2))
        mv = moves[slope]
        if t + mv[0] > L:
            break
        t += mv[0]
        e += mv[1] - r * mv[0]
        u, w = t * sM, (r * t + e) * sm
        pts.append((ax + u, ay + w) if major_x else (ax + w, ay + u))
    pts.append((b[0], b[1]))
    out = [pts[0]]
    for q in pts[1:]:                                # one segment a run
        if len(out) >= 2:
            p0, p1 = out[-2], out[-1]
            if abs((p1[0] - p0[0]) * (q[1] - p1[1]) - (p1[1] - p0[1]) * (q[0] - p1[0])) < 1e-9:
                out[-1] = q
                continue
        out.append(q)
    return out


_INNER: dict = {}


def _inner(street):
    """The street a hair inside its edge, made once a street (buffering a
    town's whole street per line was most of plots2's time on Alton)."""
    if street is None or street.is_empty:
        return None
    if id(street) not in _INNER:
        _INNER.clear()
        _INNER[id(street)] = (street, street.buffer(-0.05))
        shapely.prepare(_INNER[id(street)][1])
    return _INNER[id(street)][1]


def on_lattice_line(coords, front: bool, street) -> list:
    """A line with each stretch whose trend runs off the lattice (between
    OFF_LATTICE slopes, at least 3 tiles; not within LATTICE_TOL of a lattice
    slope) laid as lattice runs - never where they would cross the street;
    the rest exactly as it was."""
    ln = LineString(coords)
    trend = list(shapely.simplify(ln, TREND).coords)
    if len(trend) < 2:
        return coords
    index = {tuple(round(v, 6) for v in c): i for i, c in enumerate(coords)}
    inner = _inner(street)
    out = [tuple(coords[0])]
    for p_, q_ in zip(trend, trend[1:]):
        i0, i1 = index.get(tuple(round(v, 6) for v in p_)), index.get(tuple(round(v, 6) for v in q_))
        original = [tuple(c) for c in coords[i0 + 1:i1 + 1]] if i0 is not None and i1 is not None and i1 > i0 else None
        dx, dy = q_[0] - p_[0], q_[1] - p_[1]
        lo_, hi_ = sorted((abs(dx), abs(dy)))
        r = lo_ / hi_ if hi_ else 0.0
        runs = None
        if hi_ >= 3 and OFF_LATTICE[0] <= r <= OFF_LATTICE[1] and min(abs(r - v) for v in LATTICE) > LATTICE_TOL:
            side = 0
            if front and inner is not None:
                L = math.hypot(dx, dy)
                mx, my = (p_[0] + q_[0]) / 2, (p_[1] + q_[1]) / 2
                nx, ny = -dy / L, dx / L
                gx, gy = (-nx, -ny) if shapely.contains_xy(street, mx + nx, my + ny) else (nx, ny)
                side = 1 if (gy if abs(dx) >= abs(dy) else gx) > 0 else -1
            runs = lattice_runs(p_, q_, side)
            if inner is not None and LineString(runs).intersects(inner):
                runs = None                          # across the street: as it was
        if runs:
            out += [r_ for r_ in runs if math.dist(r_, out[-1]) > 1e-9][:1] + runs[1:]   # the corner it starts from too
        else:
            out += original or [tuple(q_)]
    return out


PATH_SIMPLIFY = 3.0       # tiles: a footpath's wiggles under this go before it is laid on the lattice
PATH_JOG = 7              # tiles: a footpath stretch straying less than this from the grid is laid square, with one jog (user, 2026-10-16: wedges along an alley's bend)
PATH_DRIFT = 6.0           # tiles: a footpath's lattice runs keep within half this of its line (long runs)
DIRECTIONS16 = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
                (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)]


def smooth_path(coords, avoid=None) -> list:
    """A footpath laid like the roads: its line simplified by PATH_SIMPLIFY,
    then each stretch from the corner it has reached along the nearest of
    vanilla's 16 directions (grid, 45 degrees, 2:1), a whole number of steps
    towards the next point - tile corners all the way, so the path and the
    garden boundaries along it are long straight lattice lines (user,
    2026-10-13: footpaths too zig-zagged, the boundaries messy beside them).
    A stretch straying less than PATH_JOG from the grid is laid square with
    one jog (half way, else at an end), and each end meets square, where that
    crosses no house."""
    pts = list(shapely.simplify(LineString(coords), PATH_SIMPLIFY).coords)
    out = [(round(pts[0][0]), round(pts[0][1]))]
    for q in pts[1:]:
        p_ = out[-1]
        d = (round(q[0]) - p_[0], round(q[1]) - p_[1])
        if d == (0, 0):
            continue
        square = None
        on_dir = any(d[0] * v[1] == d[1] * v[0] and d[0] * v[0] + d[1] * v[1] > 0 for v in DIRECTIONS16)
        prev = (out[-1][0] - out[-2][0], out[-1][1] - out[-2][1]) if len(out) >= 2 else None
        if not on_dir and min(abs(d[0]), abs(d[1])) < PATH_JOG:
            # Nearly along the grid: straight runs with one square jog - half
            # way, else at the start or the end - where that crosses no house;
            # else as below.
            q_r = (p_[0] + d[0], p_[1] + d[1])
            ax_x = abs(d[0]) >= abs(d[1])
            for f_ in (0.5, 0.0, 1.0):
                h_ = (round(d[0] * f_), 0) if ax_x else (0, round(d[1] * f_))
                a_ = (p_[0] + h_[0], p_[1] + h_[1])
                b_ = (a_[0], q_r[1]) if ax_x else (q_r[0], a_[1])
                cand = [c_ for c_ in (a_, b_, q_r) if c_ != p_]
                if _doubles_back(prev, (cand[0][0] - p_[0], cand[0][1] - p_[1])):
                    continue                                # back over the leg before: a spur
                if avoid is None or avoid.intersection(LineString([p_] + cand)).length <= 0.5:
                    square = cand
                    break
        if square:
            legs = square
        elif on_dir:
            legs = [(p_[0] + d[0], p_[1] + d[1])]            # on one of the 16 directions: as it is
        else:                                               # between two: long runs, within PATH_DRIFT / 2
            # ...tighter where long runs would cut through a house the line
            # itself passes (beside houses the path keeps to its line).
            q_r = (p_[0] + d[0], p_[1] + d[1])
            over = avoid.intersection(LineString([p_, q_r])).length if avoid is not None else 0.0
            for drift in (PATH_DRIFT, PATH_DRIFT / 2, PATH_DRIFT / 4):
                legs = [(round(x), round(y)) for x, y in lattice_runs(p_, q_r, 0, drift)[1:]]
                if avoid is None or avoid.intersection(LineString([p_] + legs)).length <= over + 0.5:
                    break
        for nxt in legs:
            if nxt == out[-1]:
                continue
            if len(out) >= 2 and _same_way(out[-2], out[-1], nxt):
                out[-1] = nxt
            else:
                out.append(nxt)
    # Each end meets square: a short angled last piece (into the street, or a
    # junction) as an L, its last run along the grid, where that crosses no
    # house (user, 2026-10-16: an alley's mouth cut a wedge out of a corner).
    for end in (0, -1):
        if len(out) < 2:
            break
        i = 0 if end == 0 else len(out) - 1                 # the end; a last step under 2 tiles (rounding) is passed over
        step = 1 if end == 0 else -1
        if len(out) >= 3 and math.dist(out[i], out[i + step]) < 2:
            i += step
        a, b_ = out[i + step], out[i]
        d = (b_[0] - a[0], b_[1] - a[1])
        on_dir = any(d[0] * v[1] == d[1] * v[0] and d[0] * v[0] + d[1] * v[1] > 0 for v in DIRECTIONS16)
        if d[0] and d[1] and not on_dir and min(abs(d[0]), abs(d[1])) < PATH_JOG:
            corner = (a[0], b_[1]) if abs(d[0]) >= abs(d[1]) else (b_[0], a[1])
            j = i + 2 * step                                 # the leg before: carry straight on along it if it is on the grid
            if 0 <= j < len(out):
                before = (a[0] - out[j][0], a[1] - out[j][1])
                for c_ in ((b_[0], a[1]), (a[0], b_[1])):
                    if (c_[0] - a[0]) * before[1] == (c_[1] - a[1]) * before[0] and                             (c_[0] - a[0]) * before[0] + (c_[1] - a[1]) * before[1] > 0:
                        corner = c_                          # straight on, never back over it
                if _doubles_back(before, (corner[0] - a[0], corner[1] - a[1])):
                    corner = (a[0], b_[1]) if corner == (b_[0], a[1]) else (b_[0], a[1])
            if avoid is None or avoid.intersection(LineString([a, corner, b_])).length <= 0.5:
                out.insert(i + 1 if end == 0 else i, corner)
    return out if len(out) >= 2 else [tuple(c) for c in coords]


def _doubles_back(u, v) -> bool:
    """Whether leg v runs straight back along leg u."""
    return u is not None and u[0] * v[1] == u[1] * v[0] and u[0] * v[0] + u[1] * v[1] < 0


def _same_way(a, b, c) -> bool:
    return abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) < 1e-9         and (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) > 0


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    picture = sys.argv[sys.argv.index("--picture") + 1] if "--picture" in sys.argv else None
    view = [int(v) for v in sys.argv[sys.argv.index("--box") + 1].split(",")] if "--box" in sys.argv else None
    attrs_path = sys.argv[sys.argv.index("--inspire-attrs") + 1] if "--inspire-attrs" in sys.argv else None
    if attrs_path and area_conf.is_extra(attrs_path):
        print(f"--inspire-attrs left out: {area} is at the osm_lidar data level")
        attrs_path = None
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    b = info["bbox"]
    mpt = info["meters_per_tile"]
    proj = B.Projector.build(b["south"], b["west"], b["north"], b["east"], mpt, info.get("rotation") or 0.0)
    W, H = proj.width, proj.height

    # --- buildings ------------------------------------------------------------------------------
    areas = AreaIndex.load(map_dir, name, proj)
    types, homes_in = {}, {}
    if os.path.exists(os.path.join(area, "types.csv")):
        types = {r["osm_id"]: (r.get("type") or "", r.get("era") or "")
                 for r in csv.DictReader(open(os.path.join(area, "types.csv"), encoding="utf-8"))}
    if os.path.exists(os.path.join(area, "buildings.csv")):
        homes_in = {r["osm_id"]: (int(r.get("uprn_count") or 0), r.get("form") or "")
                    for r in csv.DictReader(open(os.path.join(area, "buildings.csv"), encoding="utf-8"))}
    feats = json.load(open(os.path.join(map_dir, f"{name}_buildings.geojson"), encoding="utf-8"))["features"]
    bid = np.zeros((H, W), np.int32)
    blds = []
    for ft in feats:
        p = ft["properties"]
        if ft["geometry"]["type"] != "Polygon" or (p.get("building") or "").lower() in P.NOT_BUILDINGS:
            continue
        poly = Polygon([proj.to_px(la, lo) for lo, la in ft["geometry"]["coordinates"][0]])
        if not poly.is_valid:
            poly = poly.buffer(0)
        rr, cc = raster(poly, W, H)
        rr, cc = rr[bid[rr, cc] == 0], cc[bid[rr, cc] == 0]
        if not len(rr):
            continue
        pid = str(p.get("parent_osm_id") or p.get("osm_id"))
        typ, era = types.get(pid, ("", ""))
        btag = (p.get("building") or "").lower()
        cat = areas.category_at(float(cc.mean()) + 0.5, float(rr.mean()) + 0.5)
        blds.append({
            "pid": pid, "frame": p.get("lidar_frame"), "osm": str(p.get("osm_id")),
            "prop": re.sub(r"p\d+$", "", str(p.get("osm_id"))),
            "bbox": (int(cc.min()), int(rr.min()), int(cc.max()) + 1, int(rr.max()) + 1),
            "out": not p.get("parent_osm_id") and (len(rr) * mpt * mpt <= P.OUTBUILDING_M2
                                                   or btag in P.SHED_VALUES or typ == "outbuilding"),
            "home": not (btag in P.NOT_HOME_TAGS or (p.get("amenity") or "") in P.NOT_HOME_AMENITY
                         or cat in P.NOT_HOME_AREAS or typ == "non_residential"),
            "old": any(w in f"{typ} {era}" for w in P.OLD_TYPES),
            "shop": p.get("knox_dwelling") in ("shop", "shop_flats"),
            "front_side": p.get("knox_front"), "poly": poly,
            "dwelling": (p.get("knox_homes") or 0) > 0, "tag": btag})
        bid[rr, cc] = len(blds)
    # INSPIRE as evidence: what each building's title says (--inspire-attrs).
    attrs, joined = {}, 0
    if attrs_path:
        items = []
        for bd in blds:
            fr = bd["frame"]
            real = INS.poly_map(bd["poly"], lambda x, y, fr=fr: P.frame_to_real(x, y, fr)) if fr else None
            c_ = bd["poly"].centroid
            items.append({"real": real, "home": bd["dwelling"], "prop": bd["prop"], "centre": (c_.x, c_.y),
                          "frame": fr})
        attrs = INS.attributes(attrs_path, proj, items, INS.real_ways(map_dir, info, proj), P.frame_to_real)
        for j_, a_ in attrs.items():           # a garage or shed in a home's title is that home's
            o_ = a_["owner"]
            if o_ is not None and not blds[j_]["dwelling"] and blds[j_]["tag"] not in HOUSE_TAGS and \
                    blds[j_]["poly"].distance(blds[o_]["poly"]) <= JOIN_TILES and \
                    blds[j_]["poly"].area <= min(blds[o_]["poly"].area, JOIN_MAX_TILES):
                blds[j_]["prop"] = blds[a_["owner"]]["prop"]
                blds[j_]["home"] = False
                joined += 1
    whole = np.array([not bd["out"] for bd in blds])
    seed = whole.copy()
    by_prop: dict = {}
    for j, bd in enumerate(blds):
        if whole[j]:
            by_prop.setdefault(bd["prop"], []).append(j)
    members_of = {}
    for js in by_prop.values():
        # A home leads its group (its seed, its plot's facts), never its garage.
        js.sort(key=lambda j: (not blds[j]["dwelling"], -blds[j]["poly"].area, j))
        bx = (min(blds[j]["bbox"][0] for j in js), min(blds[j]["bbox"][1] for j in js),
              max(blds[j]["bbox"][2] for j in js), max(blds[j]["bbox"][3] for j in js))
        for j in js:
            blds[j]["bbox"] = bx
            seed[j] = j == js[0]
        blds[js[0]]["home"] = any(blds[j]["home"] for j in js)
        members_of[js[0]] = js

    # --- 1: street, separators, plot land -------------------------------------------------------
    # The ground as the hand-over (or Generate) left it: Build rewrites <map>.bmp
    # with drives, paths and beds, which a second run read as street (review D).
    for f_ in (f"{name}_ground_base.pretrees.bmp", f"{name}_ground_base.bmp", f"{name}.bmp"):
        if os.path.exists(os.path.join(map_dir, f_)):
            g = ground_rgb = np.array(Image.open(os.path.join(map_dir, f_)).convert("RGB"))[:H, :W]
            break

    def colour(*cols):
        m = np.zeros((H, W), bool)
        for c in cols:
            m |= np.all(g == c, axis=2)
        return m
    carriage = colour(C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE)
    tarmac = colour(C.DARK_ASPHALT)
    car_park = ndimage.binary_opening(tarmac, iterations=2)
    drive_t = tarmac & ~car_park                     # plot land, a feature of its own (Build: open mouths)
    water_t = colour(C.WATER)
    rail_t = np.zeros((H, W), bool)
    laid = os.path.join(map_dir, f"{name}_rails.json")
    if os.path.exists(laid):                         # a railway parts gardens as water does: the track
        rl = json.load(open(laid))                   # as streets/rails.py laid it out, bed and all...
        for x, y in rl.get("land", []) + rl["bed"] + [t[:2] for t in rl["track"]]:
            if 0 <= x < W and 0 <= y < H:
                water_t[y, x] = rail_t[y, x] = True
    rail = Polygon() if os.path.exists(laid) else P.railway_shape(map_dir, info, proj)   # ...else OSM's line
    if not rail.is_empty:
        rr, cc = raster(rail, W, H)
        water_t[rr, cc] = True
        rail_t[rr, cc] = True
    sites = [p_ for p_ in P.tagged_polygons(map_dir, info, proj, P.SITE_TAGS) if p_.area >= P.SITE_MIN_TILES]
    fields_t = car_park | colour(C.LIGHT_GRASS, C.SAND) | P.field_mask(map_dir, info, proj, W, H)
    for p_ in sites:                                 # a building site: no gardens on it
        rr, cc = raster(p_, W, H)
        fields_t[rr, cc] = True
    residential = np.zeros((H, W), bool)
    for shp, props in areas._items:
        if props.get("category") in P.RESIDENTIAL_AREAS:
            rr, cc = raster(shp, W, H)
            residential[rr, cc] = True
    buildings_shape = mask_shape(bid > 0)
    graph = graph_street(map_dir, name)
    if graph is not None:
        street = graph.difference(buildings_shape)
    else:
        street_t = carriage | (colour(C.PALE_CONCRETE) & ndimage.binary_dilation(carriage, iterations=P.PAVEMENT_REACH))
        rs_path = os.path.join(map_dir, f"{name}_roadside.npz")
        if os.path.exists(rs_path):
            rs = np.load(rs_path)
            for key in ("verge", "pavement"):
                m = rs[key][:H, :W]
                street_t[:m.shape[0], :m.shape[1]] |= m
        street = mask_shape(street_t & (bid == 0))
    street = shapely.set_precision(street.buffer(0), GRID)
    shapely.prepare(street)
    rr, cc = raster(street, W, H)
    street_t = np.zeros((H, W), bool)
    street_t[rr, cc] = True
    # A level crossing's squares (track over the road) stay street.
    crossing = rail_t & street_t
    rail_t &= ~crossing
    water_t &= ~crossing
    # Footpaths: OSM's, where Generate moved them, laid on the lattice; a
    # garden boundary runs along each side and no plot crosses one (user,
    # 2026-10-13: the path between Runnymede Drive and Waverley Close).
    # On tile centres (the lattice is on corners): PATH_HALF either side then
    # ends on tile edges, with no row of tiles split between path and garden
    # (each was neither: a strip of nothing beside every path, fenced twice).
    houses_ = shapely.affinity.translate(buildings_shape, -0.5, -0.5)   # checked where the path will lie
    path_lines = [(shapely.affinity.translate(LineString(smooth_path(list(ln.coords), houses_)), 0.5, 0.5), paved)
                  for ln, paved in P.footpaths(map_dir, info, proj)]
    path_shapes = {pv: unary_union([ln.buffer(P.PATH_HALF, cap_style=2, join_style=2) for ln, p_ in path_lines if p_ == pv])
                   for pv in (True, False)}
    paths = unary_union(list(path_shapes.values())).difference(buildings_shape)
    rr, cc = raster(paths, W, H) if not paths.is_empty else (np.zeros(0, int), np.zeros(0, int))
    path_t = np.zeros((H, W), bool)
    path_t[rr, cc] = True
    path_paved = np.zeros((H, W), bool)               # which of them are paved (kerbs.py paints them)
    if not path_shapes[True].is_empty:
        rr, cc = raster(path_shapes[True], W, H)
        path_paved[rr, cc] = True
    path_paved &= path_t
    blocked = unary_union([mask_shape(water_t), mask_shape(fields_t & ~(bid > 0)), paths])
    land = box(0, 0, W, H).difference(unary_union([street, blocked])).union(buildings_shape)
    land = shapely.set_precision(land.buffer(0), GRID)

    # --- 2: cut each block ----------------------------------------------------------------------
    centre = {j: ((bd["bbox"][0] + bd["bbox"][2]) / 2, (bd["bbox"][1] + bd["bbox"][3]) / 2)
              for j, bd in enumerate(blds)}
    tree = shapely.STRtree([Point(centre[j]) for j in range(len(blds))])
    pieces = []
    blocks_ = polys(land)
    for blk in blocks_:
        members = [int(i) for i in tree.query(blk, predicate="contains")]
        if any(seed[m] for m in members):
            pieces += cut(blk, members, {m: blds[m]["bbox"] for m in members}, seed)

    # --- 3: each piece as a plot ----------------------------------------------------------------
    OPEN_TILES = P.OPEN_M / mpt
    plots, shapes, spare = [], [], []
    for j, shp in pieces:
        x0, y0, x1, y1 = blds[j]["bbox"]
        here = box(x0, y0, x1, y1)
        cy, cx = min(H - 1, (y0 + y1) // 2), min(W - 1, (x0 + x1) // 2)
        if not residential[cy, cx]:
            shp = shp.intersection(here.buffer(OPEN_TILES, join_style=2))
        parts = polys(shp)
        if not parts:
            continue
        own = max(parts, key=lambda q: q.intersection(here.buffer(0.5)).area)
        spare += [q for q in parts if q is not own and q.area >= 1]
        plots.append(j)
        shapes.append(shapely.set_precision(own, GRID))
    given, lost = 0, 0
    spare_boxes = Boxes(shapes)
    for q in spare:                                  # bits a lane or a cut left away from their building
        if give_away(shapely.set_precision(q, GRID), shapes, boxes=spare_boxes) >= 0:
            given += int(q.area)
        else:
            lost += int(q.area)
    # A terrace or semi pair with more homes than units: a strip per home.
    strips = 0
    lengths: dict = {}
    for j in members_of:
        x0, y0, x1, y1 = blds[j]["bbox"]
        lengths.setdefault(blds[j]["pid"], []).append(max(x1 - x0, y1 - y0))
    for k in range(len(plots)):
        j = plots[k]
        n_all, form = homes_in.get(blds[j]["pid"], (0, ""))
        if form not in P.ROW_FORMS or not blds[j]["home"]:
            continue
        x0, y0, x1, y1 = blds[j]["bbox"]
        along_x = (x1 - x0) >= (y1 - y0)
        lo, hi = (x0, x1) if along_x else (y0, y1)
        n = int(round(n_all * (hi - lo) / max(1, sum(lengths.get(blds[j]["pid"], [hi - lo])))))
        n = min(n, (hi - lo) // P.RECT_TILES)
        if n < 2:
            continue
        cuts = [lo + round(i * (hi - lo) / n) for i in range(n + 1)]
        bx0, by0, bx1, by1 = shapes[k].bounds
        parts = []
        for a, c in zip(cuts, cuts[1:]):
            band = box(a if a != lo else bx0 - 1, by0 - 1, c if c != hi else bx1 + 1, by1 + 1) if along_x \
                else box(bx0 - 1, a if a != lo else by0 - 1, bx1 + 1, c if c != hi else by1 + 1)
            parts.append(shapes[k].intersection(band))
        if not all(pt.distance(street) < 1.0 for pt in parts):
            continue
        shapes[k] = shapely.set_precision(parts[0], GRID)
        for pt in parts[1:]:
            plots.append(j)
            shapes.append(shapely.set_precision(pt, GRID))
            strips += 1

    # A home alone in its title: its plot no further from the house than the
    # title reaches (N, E, S, W, plus REACH_SLACK, on whole tiles), except
    # towards its street (the street's own line makes the front). What it
    # gives up is open ground, or a gap a neighbour takes below.
    trimmed = trim_tiles = 0
    if attrs:
        once = Counter(plots)
        street_edge = street.buffer(0.5)
        prop_poly: dict = {}
        for b_ in blds:
            prop_poly.setdefault(b_["prop"], []).append(b_["poly"])
        for k, j in enumerate(plots):
            a_ = attrs.get(j)
            if a_ is None or once[j] > 1 or a_["homes"] != 1 or not blds[j]["dwelling"] or shapes[k].is_empty:
                continue
            c_ = blds[j]["poly"].centroid
            bx0, by0, bx1, by1 = blds[j]["poly"].bounds
            wall = {"N": c_.y - by0, "E": bx1 - c_.x, "S": by1 - c_.y, "W": c_.x - bx0}
            lim = {"W": -1e9, "N": -1e9, "E": 1e9, "S": 1e9}
            for d_, (dx, dy) in {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.items():
                r_ = a_["reach"][d_]
                if r_ <= 0 or d_ == blds[j]["front_side"]:
                    continue
                far = r_ + REACH_SLACK                          # reach is in tiles already
                if far - wall[d_] < REACH_GARDEN:
                    continue                                  # a title tight on the house: its garden is another title
                probe = LineString([(c_.x, c_.y), (c_.x + dx * (far + REACH_STREET), c_.y + dy * (far + REACH_STREET))])
                if probe.intersects(street):
                    continue                                  # its street side
                v = (c_.x + dx * far) if dx else (c_.y + dy * far)
                lim[d_] = math.floor(v) if d_ in ("W", "N") else math.ceil(v)
            if all(abs(v) == 1e9 for v in lim.values()):
                continue
            keep = box(lim["W"], lim["N"], lim["E"], lim["S"])
            own_ = unary_union(prop_poly[blds[j]["prop"]])
            new_ = areal(shapely.set_precision(shapes[k].intersection(keep).union(own_.intersection(shapes[k])), GRID))
            # What it gives up at the street stays: the land between a house and
            # its street is its front (the ray test misses diagonal streets).
            cut_ = areal(shapes[k].difference(new_))
            at_street = [q_ for q_ in polys(cut_)
                         if q_.boundary.intersection(street_edge).length >= 3.0]   # along the street, not a side strip touching it
            if at_street:
                new_ = areal(shapely.set_precision(unary_union([new_] + at_street), GRID))
            gone = shapes[k].area - new_.area
            if gone >= 1.0 and not new_.is_empty:
                old_ = shapes[k]
                shapes[k] = polys(new_)[0] if len(polys(new_)) == 1 else max(polys(new_), key=lambda q: q.area)
                trimmed += 1
                trim_tiles += int(gone)
                # A narrow strip it gave up goes to the neighbour (the gap rule
                # below would as often hand it straight back).
                for q_ in polys(shapely.set_precision(old_.difference(shapes[k]), GRID)):
                    if q_.buffer(-P.GAP_TILES / 2, join_style=2).is_empty:
                        give_away(q_, shapes, exclude=k)

    # Every home reaches a street: a plot with no frontage gets a straight drive
    # RECT_TILES wide to the nearest (on the tiles; its shape then joins the plot).
    # With --inspire-attrs not where the home's title reaches no road (its way
    # out is someone else's land: a shared drive, a courtyard).
    plot = np.zeros((H, W), np.int32)
    for k, s_ in enumerate(shapes, 1):
        rr, cc = raster(s_, W, H)
        free = plot[rr, cc] == 0
        plot[rr[free], cc[free]] = k
    # A home on a footpath has its way out (English estates front paths, not roads).
    touch = ndimage.binary_dilation(street_t | path_t)
    touch_town = ndimage.binary_dilation(street_t, iterations=P.TOWN_FRONT + 1)
    passable = ~(street_t | water_t | fields_t | path_t) & (bid == 0)
    made = np.zeros((H, W), bool)
    drive_boxes = Boxes(shapes)
    drives, stranded, no_road = 0, [], 0
    r_dense = int(round(P.DENSITY_M / mpt))
    coverage = ndimage.uniform_filter((bid > 0).astype(float), size=2 * r_dense + 1, mode="constant")
    for k, j in enumerate(plots, 1):
        if not blds[j]["home"]:
            continue
        m = plot == k
        if not m.any() or (m & touch).any():
            continue
        if attrs.get(j) is not None and not attrs[j]["road"]:
            no_road += 1
            continue
        same = np.isin(bid, [i + 1 for i, b_ in enumerate(blds) if b_["pid"] == blds[j]["pid"]])
        x0, y0, x1, y1 = blds[j]["bbox"]
        in_town = coverage[min(H - 1, (y0 + y1) // 2), min(W - 1, (x0 + x1) // 2)] >= (P.DENSE_LO + P.DENSE_HI) / 2
        if (same & touch).any() or (in_town and (same & touch_town).any()):
            continue
        best = None
        for rot in range(4):
            got = P.drive_to_street(np.rot90(m, rot), np.rot90(plot, rot), np.rot90(street_t, rot),
                                    np.rot90(passable, rot), k)
            if got and (best is None or got[0] < best[0]):
                best = (got[0], rot, got[1], got[2])
        if best is None:
            stranded.append(blds[j]["pid"])
            continue
        _score, rot, corr, _mouth = best
        corr = np.rot90(corr, -rot)
        plot[corr] = k
        made |= corr
        cs = mask_shape(corr)
        for o in drive_boxes.near(cs):
            if o != k - 1 and shapes[o].intersects(cs):
                shapes[o] = shapely.set_precision(shapes[o].difference(cs), GRID)
                drive_boxes.update(o, shapes[o])
        shapes[k - 1] = shapely.set_precision(shapes[k - 1].union(cs), GRID)
        drive_boxes.update(k - 1, shapes[k - 1])
        drives += 1
    drive_t = drive_t | made
    # Drives from INSPIRE: a road running into a home's title and stopping
    # there, carried in with the house's frame, DRIVE_TILES wide, on its own
    # plot and off buildings; Build leaves its mouth open.
    osm_drives = 0
    if attrs:
        for k, j in enumerate(plots, 1):
            a_ = attrs.get(j)
            if a_ is None or not a_["drives"] or not blds[j]["frame"]:
                continue
            for ln in a_["drives"]:
                g_ = LineString([INS.frame_to_game(x, y, blds[j]["frame"]) for x, y in ln.coords])
                strip = g_.buffer(DRIVE_TILES / 2, cap_style=2).intersection(shapes[k - 1]).difference(buildings_shape)
                if strip.area < 2 * DRIVE_TILES or strip.distance(street) > 1.0:
                    continue
                rr, cc = raster(strip, W, H)
                drive_t[rr, cc] = True
                osm_drives += 1

    # Each house's front strip (frontage), then land no plot got and thin bits.
    fronts_moved = frontage(shapes, plots, blds, street, buildings_shape, land)
    filled = 0
    late = Boxes(shapes)
    leftover = land.difference(unary_union(shapes + [buildings_shape]))
    for q in polys(shapely.set_precision(leftover, GRID)):
        if q.area < 0.5 or not q.buffer(-P.GAP_TILES / 2, join_style=2).is_empty:
            continue                                 # wider than a gap: open ground
        if give_away(q, shapes, boxes=late) >= 0:
            filled += int(round(q.area))
    slivers = 0
    for o in range(len(shapes)):
        s_ = shapes[o]
        if s_.is_empty:
            continue
        core = s_.buffer(-0.99, join_style=2).buffer(0.99, join_style=2)
        for q in polys(s_.difference(core.buffer(0.01, join_style=2))):
            if q.area < 1.0 or q.intersects(buildings_shape):
                continue
            q = shapely.set_precision(q, GRID)
            rest = shapely.set_precision(shapes[o].difference(q), GRID)
            if rest.is_empty or (shapes[o].distance(street) < 1.0 and rest.distance(street) >= 1.0):
                continue                             # its only frontage
            if give_away(q, shapes, exclude=o, boxes=late) >= 0:
                shapes[o] = rest
                late.update(o, rest)
                slivers += 1
    street_strips = street_strips_to_behind(shapes, land, street, buildings_shape, late,
                                            mask_shape(drive_t) if drive_t.any() else None)
    # A plot left in pieces keeps its piece by its building.
    for o, s_ in enumerate(shapes):
        ps = polys(s_)
        if len(ps) > 1:
            x0, y0, x1, y1 = blds[plots[o]]["bbox"]
            keep = max(ps, key=lambda q: q.intersection(box(x0, y0, x1, y1).buffer(0.5)).area)
            shapes[o] = keep
            late.update(o, keep)
            for q in ps:
                if q is not keep:
                    if give_away(q, shapes, exclude=o, boxes=late) >= 0:
                        given += int(q.area)
                    else:
                        lost += int(q.area)
    plot[:] = 0
    for k, s_ in enumerate(shapes, 1):
        rr, cc = raster(s_, W, H)
        free = plot[rr, cc] == 0
        plot[rr[free], cc[free]] = k
    town = []
    for j in plots:
        x0, y0, x1, y1 = blds[j]["bbox"]
        c = coverage[min(H - 1, (y0 + y1) // 2), min(W - 1, (x0 + x1) // 2)]
        town.append(float(np.clip((c - P.DENSE_LO) / (P.DENSE_HI - P.DENSE_LO), 0.0, 1.0)))

    # --- 4: boundaries: every plot's outline, piece by piece ----------------------------------
    corners = [proj.to_latlon(x, y) for x, y in ((0, 0), (W, 0), (0, H), (W, H))]
    es, ns = zip(*[B.TO_BNG(lon, lat) for lat, lon in corners])
    heights = Heights(area, (int(min(es)) - 2, int(min(ns)) - 2, int(max(es)) + 2, int(max(ns)) + 2))
    lane = colour(C.DARKEST_ASPHALT) | drive_t

    def to_bng(x, y, fr):
        rx, ry = P.frame_to_real(x, y, fr)
        lat, lon = proj.to_latlon(rx, ry)
        return B.TO_BNG(lon, lat)

    def tile_at(x, y):
        ix, iy = int(math.floor(x)), int(math.floor(y))
        return (ix, iy) if 0 <= ix < W and 0 <= iy < H else None

    def lane_end_on(px, py, nx, ny) -> bool:
        """A lane running straight away from the side here for LANE_END,
        narrower than LANE_WIDE: the plot's drive."""
        t = tile_at(px + nx * FRONT_PROBE, py + ny * FRONT_PROBE)
        if t is None or not lane[t[1], t[0]]:
            return False
        sx, sy = (int(round(nx)), int(round(ny)))
        if abs(sx) + abs(sy) != 1:
            return False
        run, x, y = 0, t[0], t[1]
        while run < P.LANE_END and 0 <= x < W and 0 <= y < H and lane[y, x]:
            run, x, y = run + 1, x + sx, y + sy
        across = 1
        for d in (1, -1):
            x, y = t[0] + sy * d, t[1] + sx * d
            while 0 <= x < W and 0 <= y < H and lane[y, x] and across < P.LANE_WIDE:
                across, x, y = across + 1, x + sy * d, y + sx * d
        return run >= P.LANE_END and across < P.LANE_WIDE

    # Open ground a few squares wide between a plot and the street (a rural
    # road's grass verge) is the street's: the boundary at its back is a front.
    verge = street.buffer(VERGE_REACH).difference(land)
    shapely.prepare(verge)
    # All the plots' outlines - holes and every part too - as one noded set of
    # lines: a side two plots share is one line.
    plot_tree = shapely.STRtree(shapes)
    built = {k: shapes[k].intersection(buildings_shape).area for k in range(len(shapes))}
    no_garden = {k: shapes[k].area - built[k] < P.MIN_GARDEN for k in range(len(shapes))}
    noded = unary_union([r_ for s_ in shapes for r_ in rings_of(s_)])
    segs = []
    for ln in getattr(noded, "geoms", [noded]):
        cs = list(ln.coords)
        segs += list(zip(cs, cs[1:]))

    def plot_at(x, y):
        hit = [int(o) for o in plot_tree.query(Point(x, y), predicate="intersects")]
        return hit[0] if hit else -1

    def what(x, y):
        """What lies at a point beside a side: ("plot", k), ("street",),
        ("none",), or ("stop",) - a building or water: no boundary. The
        railway is "none": a garden backing onto the line has its back
        fence (review, 2026-10-14: as water, it had none)."""
        t = tile_at(x, y)
        if t is not None and rail_t[t[1], t[0]] and not bid[t[1], t[0]]:
            return ("none",)
        if t is None or bid[t[1], t[0]] or water_t[t[1], t[0]]:
            return ("stop",)
        k = plot_at(x, y)
        if k >= 0:
            return ("plot", k)
        if shapely.contains_xy(street, x, y) or shapely.contains_xy(verge, x, y):
            return ("street",)
        return ("none",)

    def classify(mx, my, nx, ny):
        """(home plot, other plot or -1, front) for a stretch, or None."""
        l_ = what(mx + nx * FRONT_PROBE, my + ny * FRONT_PROBE)
        r_ = what(mx - nx * FRONT_PROBE, my - ny * FRONT_PROBE)
        if "stop" in (l_[0], r_[0]) or l_ == r_:
            return None
        sides_ = [l_, r_]
        homes_ = [x_[1] for x_ in sides_ if x_[0] == "plot" and blds[plots[x_[1]]]["home"]]
        if not homes_:
            return None
        k = homes_[0]
        o = sides_[1] if sides_[0] == ("plot", k) else sides_[0]
        sgn = 1 if sides_[1] == ("plot", k) else -1          # +n points away from k
        front = o[0] == "street"
        tin = tile_at(mx - sgn * nx * FRONT_PROBE, my - sgn * ny * FRONT_PROBE)
        if front and tin and drive_t[tin[1], tin[0]]:
            return None                                      # a drive's mouth
        # A lane end-on (a painted lane the graph did not draw), only on a
        # side along the grid, never the street.
        if not front and (abs(nx) < 1e-9 or abs(ny) < 1e-9) and lane_end_on(mx, my, sgn * nx, sgn * ny):
            return None
        o_k = o[1] if o[0] == "plot" else -1
        if no_garden[k] and (o_k < 0 or no_garden[o_k]) and not front:
            return None
        if front and blds[plots[k]]["shop"]:
            return None                                      # a shop's forecourt is open
        return (k, o_k, front)

    pieces_ = []                                 # ((p0, p1), home plot, other or -1, front, kind)
    for (ax, ay), (bx, by) in segs:
        L = math.hypot(bx - ax, by - ay)
        if L < 1e-6:
            continue
        nx, ny = (by - ay) / L, -(bx - ax) / L
        n = max(1, int(math.ceil(L)))
        cls = []
        for i_ in range(n):
            t0, t1 = i_ / n, (i_ + 1) / n
            mx, my = ax + (bx - ax) * (t0 + t1) / 2, ay + (by - ay) * (t0 + t1) / 2
            cls.append((t0, t1, classify(mx, my, nx, ny)))
        runs_, cur = [], None
        for t0, t1, c in cls:
            if cur is not None and c == cur[2]:
                cur = (cur[0], t1, c)
            else:
                if cur is not None:
                    runs_.append(cur)
                cur = (t0, t1, c)
        runs_.append(cur)
        for t0, t1, c in runs_:
            if c is None:
                continue
            k, o_k, front = c
            p0 = (ax + (bx - ax) * t0, ay + (by - ay) * t0)
            p1 = (ax + (bx - ax) * t1, ay + (by - ay) * t1)
            bd = blds[plots[k]]
            if bd["frame"]:
                ln = LineString([to_bng(*p0, bd["frame"]), to_bng(*p1, bd["frame"])])
                hedge, wall, _top = heights.along(ln, corridor=P.SIDE_CORRIDOR_M)
                kind = "hedge" if hedge >= P.PLOT_HEDGE_SHARE else "wall" if wall >= WALL_SHARE else "fence"
            else:
                kind = "fence"
            pieces_.append(((p0, p1), k, o_k, front, kind))

    # One kind of boundary a group, not a boundary at a time (user,
    # 2026-10-14: a hedge on one side, a fence on the next, the neighbour's
    # different again, looked random): a street's run of fronts (each front
    # line joined to the next across a drive's mouth, FRONT_JOIN) takes one
    # kind, and a block of plots backing onto each other (plots sharing a
    # side, up to BLOCK_PLOTS) one kind for all its sides and backs - each by the LiDAR's
    # evidence along the whole group (by length, times FRONT_BIAS / BACK_BIAS:
    # raise a weight to see more of that kind), its style by the group's
    # setting too (old houses or new, in town or out, by length), from
    # BOUNDARY_STYLE.
    parent = {}

    def root(a):
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a

    # Neighbours joined by their longest shared sides first, a block no bigger
    # than BLOCK_PLOTS (a row of a dozen gardens, not a whole estate).
    shared = Counter()
    for (pp, k, o_k, front, _kind) in pieces_:
        if not front and o_k >= 0:
            shared[(min(k, o_k), max(k, o_k))] += math.hypot(pp[1][0] - pp[0][0], pp[1][1] - pp[0][1])
    size = Counter()
    for (a, b), _len in shared.most_common():
        ra, rb = root(a), root(b)
        if ra != rb and size.get(ra, 1) + size.get(rb, 1) <= BLOCK_PLOTS:
            parent[ra] = rb
            size[rb] = size.get(ra, 1) + size.get(rb, 1)
    fronts = [n for n, pc in enumerate(pieces_) if pc[3]]
    front_runs = unary_union([LineString(pieces_[n][0]).buffer(FRONT_JOIN) for n in fronts]) if fronts else None
    run_parts = list(getattr(front_runs, "geoms", [front_runs])) if front_runs is not None else []
    run_tree = shapely.STRtree(run_parts) if run_parts else None

    def group_of(n):
        (p0, p1), k, o_k, front, _kind = pieces_[n]
        if front and run_tree is not None:
            mid = Point((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            hit = [int(h) for h in run_tree.query(mid, predicate="intersects")]
            return ("front", hit[0] if hit else -1 - n)
        return ("back", root(k))

    kinds, olds, towns = {}, {}, {}
    for n, ((p0, p1), k, o_k, front, kind) in enumerate(pieces_):
        g = group_of(n)
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        bias = FRONT_BIAS if front else BACK_BIAS
        kinds.setdefault(g, Counter())[kind] += length * bias.get(kind, 1.0)
        owner = min(x_ for x_ in (k, o_k) if x_ >= 0 and blds[plots[x_]]["home"])
        olds.setdefault(g, Counter())[bool(blds[plots[owner]]["old"])] += length
        towns.setdefault(g, Counter())[bool(town[owner] >= 0.5)] += length
    by_style = {}
    for n, ((p0, p1), k, o_k, front, _kind) in enumerate(pieces_):
        g = group_of(n)
        kind = kinds[g].most_common(1)[0][0]
        old, in_town = olds[g].most_common(1)[0][0], towns[g].most_common(1)[0][0]
        st = BOUNDARY_STYLE[("front" if front else "back", kind, "town" if in_town else "old" if old else "country")]
        by_style.setdefault(st, []).append(LineString([p0, p1]))
    lines = []
    for st, ls in by_style.items():
        merged = shapely.line_merge(unary_union(ls))
        for ln in getattr(merged, "geoms", [merged]):
            if ln.length >= MIN_LINE:
                lines.append((on_lattice_line(list(shapely.simplify(ln, NUB).coords), st in FRONT_STYLES, street), st))

    # A boundary under STUB_TILES with one end on a building is a stub against
    # a wall (between a house's corner and its front wall): it goes. With both
    # ends on walls it closes a passage between two houses: it stays.
    walls_ = buildings_shape.buffer(0.3)
    shapely.prepare(walls_)
    def stub(pts_):
        ends = [walls_.contains(Point(pts_[0])), walls_.contains(Point(pts_[-1]))]
        return LineString(pts_).length < STUB_TILES and any(ends) and not all(ends)
    kept_ = [(pts_, st) for pts_, st in lines if not stub(pts_)]
    stubs = len(lines) - len(kept_)
    lines = kept_

    # A building site's outline: its site fence (off the street, as any line).
    for p_ in sites:
        for ring in getattr(p_, "geoms", [p_]):
            off = shapely.simplify(LineString(ring.exterior.coords), 0.5).difference(street)
            for ln in getattr(off, "geoms", [off]):
                if ln.geom_type == "LineString" and ln.length >= MIN_LINE:
                    lines.append((on_lattice_line(list(ln.coords), False, street), P.SITE_STYLE))

    # --- 5: the fences file and the plots -------------------------------------------------------
    out = []
    for pts_, st in lines:
        ring = [proj.to_latlon(px, py) for px, py in pts_]
        barrier = "hedge" if st.startswith("hedge") else "wall" if "brick" in st else "fence"
        out.append({"type": "Feature",
                    "properties": {"barrier": barrier, "knox_style": st, "source": "plots",
                                   # The exact line, in tiles: Build reads this, not
                                   # lon/lat back (which came back 0.01 tile off the lattice).
                                   "tiles": [[round(px, 4), round(py, 4)] for px, py in pts_]},
                    "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in ring]}})
    fp = os.path.join(map_dir, f"{name}_fences.geojson")
    orig = os.path.join(map_dir, f"{name}_fences.orig.geojson")
    if os.path.exists(fp) and not os.path.exists(orig):
        shutil.copy(fp, orig)
    base = json.load(open(orig, encoding="utf-8")) if os.path.exists(orig) else {"type": "FeatureCollection", "features": []}
    kept_osm, dropped_osm, dropped_short = [], 0, 0
    for ft in base.get("features", []):
        coords = (ft.get("geometry") or {}).get("coordinates") or []
        if ft.get("geometry", {}).get("type") != "LineString" or len(coords) < 2:
            kept_osm.append(ft)
            continue
        pts = [proj.to_px(lat, lon) for lon, lat in coords]
        on = tot = 0
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            n = max(1, int(math.hypot(bx - ax, by - ay)))
            for t in range(n + 1):
                x, y = int(ax + (bx - ax) * t / n), int(ay + (by - ay) * t / n)
                tot += 1
                on += 0 <= x < W and 0 <= y < H and plot[y, x] > 0
        if on * 2 > tot:
            dropped_osm += 1
        elif LineString(pts).length < OSM_MIN_TILES and ft.get("properties", {}).get("barrier") == "fence":
            dropped_short += 1                       # a bit of a way, cut by paths and roads
        else:
            kept_osm.append(ft)
    base["features"] = kept_osm + out
    json.dump(base, open(fp, "w", encoding="utf-8"))
    frames = np.array([blds[j]["frame"] or [np.nan] * 5 for j in plots], dtype=float).reshape(-1, 5)
    forecourt = np.zeros((H, W), bool)
    ys_, xs_ = np.mgrid[0:H, 0:W]
    for k, j in enumerate(plots, 1):
        bd = blds[j]
        if not bd["shop"] or bd["front_side"] not in ("N", "S", "E", "W"):
            continue
        x0, y0, x1, y1 = bd["bbox"]
        mine = (plot == k) & (bid == 0) & ~street_t
        beyond = {"E": (xs_ >= x1) & (ys_ >= y0) & (ys_ < y1), "W": (xs_ < x0) & (ys_ >= y0) & (ys_ < y1),
                  "S": (ys_ >= y1) & (xs_ >= x0) & (xs_ < x1), "N": (ys_ < y0) & (xs_ >= x0) & (xs_ < x1)}
        forecourt |= mine & beyond[bd["front_side"]]
    np.savez_compressed(os.path.join(map_dir, f"{name}_plots.npz"), plot=plot, frames=frames,
                        building=bid > 0, street=street_t, drive=drive_t, path=path_t, path_paved=path_paved,
                        home=np.array([blds[j]["home"] for j in plots]), forecourt=forecourt)
    open(os.path.join(map_dir, f"{name}_street.wkb"), "wb").write(shapely.to_wkb(street))

    # --- report ---------------------------------------------------------------------------------
    homes = [k for k, j in enumerate(plots) if blds[j]["home"]]
    no_front = sum(1 for k in homes if shapes[k].distance(street) > VERGE_REACH)
    angled = sum(1 for pts_, _ in lines for a_, b_ in zip(pts_, pts_[1:])
                 if abs(b_[0] - a_[0]) > 1e-6 and abs(b_[1] - a_[1]) > 1e-6)
    print(f"plots {len(plots)} ({len(homes)} homes; terrace strips {strips}); footpaths {len(path_lines)}; "
          f"drives made {drives}, homes with no way to a street {len(stranded)}, not on a street {no_front}")
    if attrs_path:
        print(f"  inspire-attrs: buildings with a title {len(attrs)}; outbuildings joined to their home {joined}; "
              f"plots trimmed to their title's reach {trimmed} ({trim_tiles} tiles); no drive made (title reaches no road) "
              f"{no_road}; drives from OSM roads into a title {osm_drives}")
    print(f"  front strips moved {fronts_moved} tiles; gaps filled {filled}; slivers moved {slivers}; strips along the street to the plot behind {street_strips}; "
          f"bits given to a neighbour {given} tiles, to none {lost}")
    print(f"  boundary lines {len(lines)} ({angled} angled pieces); total length {int(sum(LineString(p_).length for p_, _ in lines))} tiles; "
          f"OSM walls/fences on plot land left out: {dropped_osm}, fences under {OSM_MIN_TILES:.0f} tiles left out: {dropped_short}; "
          f"stubs against a wall dropped: {stubs}")
    print("  by style:", dict(Counter(st for _, st in lines).most_common()))
    sizes = Counter(g for g in (group_of(n_) for n_ in range(len(pieces_))) if g[0] == "back")
    blocks = Counter(root(k_) for (_pp, k_, _o, front_, _kd) in pieces_ if not front_)
    plots_in = Counter(root(k_) for k_ in {pc[1] for pc in pieces_})
    print(f"  boundary groups: {len(run_parts)} front runs, {len(blocks)} blocks of back gardens "
          f"(largest {max(plots_in.values()) if plots_in else 0} plots)")

    if picture:
        x0, y0, w, h = view or (0, 0, W, H)
        K = 4
        img = Image.fromarray(ground_rgb[y0:y0 + h, x0:x0 + w]).resize((w * K, h * K), Image.NEAREST)
        rng = np.random.default_rng(5)
        tint = np.vstack([[0, 0, 0], rng.integers(60, 255, (len(plots), 3))]).astype(np.uint8)
        base_ = np.array(img).astype(np.float32)
        lab_ = np.repeat(np.repeat(plot[y0:y0 + h, x0:x0 + w], K, 0), K, 1)
        blt = np.repeat(np.repeat(bid[y0:y0 + h, x0:x0 + w] > 0, K, 0), K, 1)
        mix = np.where((lab_ > 0)[..., None], base_ * 0.6 + tint[lab_] * 0.4, base_)
        mix[blt] = (40, 40, 40)
        img = Image.fromarray(mix.astype(np.uint8))
        dr = ImageDraw.Draw(img)
        for pts_, st in lines:
            dr.line([((px - x0) * K, (py - y0) * K) for px, py in pts_], fill=P.COLOURS.get(st, (90, 90, 90)), width=2)
        img.save(picture)
        print("wrote", picture)
    return 0


if __name__ == "__main__":
    sys.exit(main())
