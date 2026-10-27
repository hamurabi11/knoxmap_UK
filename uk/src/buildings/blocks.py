"""Carry real building blocks into a KnoxMap map whole, before Build.

    python blocks.py <knoxmap map dir> <area data dir> [--streets "High Street" ...] [--report]
                     [--rect N] [--rect-count N]

--streets defaults to the streets street_widths.py set from their frontages
(the High Street first: a corner block goes with the first street listed).

Replaces tidy_outlines.py. KnoxMap's Knox County roads move every corner of
every building by the road move at that corner (octilinear.py), then Build
squares each building up on its own and nudges it off the road. A terrace or
a row of shops came out as separate boxes with gaps and forecourts: only 35%
of the High Street's real building ground was built on in the game (LOG
2026-10-06). Here everything is decided on the real outlines, then each block
is moved as one piece:

In real coordinates (British National Grid):
  1. Blocks: outlines that touch (TOUCH_M) are one block - a terrace, a row of
     shops with their back ranges. OSM often draws a row as separate boxes
     with slivers between them; two outlines under GAP_M apart whose sliver is
     mostly roof in the LiDAR (GAP_ROOF) are attached too, and the sliver is
     shared out between them.
  2. Each block is registered onto the LiDAR: the shift (up to REG_M, in
     REG_STEP steps) that best lays it on the LiDAR's buildings - building-class
     points from the National LiDAR Programme cloud where it has them, else
     anything REG_HEIGHT_M above the ground. Kept only if it is clearly better
     (REG_GAIN) than none.
  3. Outline check: how much of each outline the LiDAR sees as building
     (`lidar_cover`), and LiDAR buildings no outline covers. Those the point
     cloud classes as building, not on an outline (NEW_CLEAR_M) and big enough
     for a building (NEW_MIN_M2) become buildings of their own: OSM lacks
     whole estates. Each is cut where a thin link joins two (NEW_SPLIT_M: a
     pair of semis through a hedge), and each part's outline is up to RECT_MAX
     solid rectangles at least RECT_MIN wide on its own axis (house, wing,
     garage; rectify); a part no such rectangles fit is left out. Moved from the LiDAR's frame
     into OSM's by the registration of the blocks around them, and they move
     into the game like their neighbours. osm_id NEW_ID + k, source=lidar.
     The rest (extensions, frontage strips) go to lidar_only.geojson.

Into the game, block by block, biggest first:
  4. A block fronting one of the --streets (within FRONT_NEAR_M of its centre
     line, or nearer it than any other named road and within FRONT_M) is
     carried along the street: every point keeps its
     distance along the street (as a share of its length) and its offset from
     the midline between the two frontages, and is put at that distance and
     offset from the straightened street in the game, pinned at every
     junction (the OSM node it shares with each side road, which the
     straightening keeps shared). Each building is then moved rigidly to the turn and shift that best fit where the carry
     puts it, squared to the tiles, so its shape is its real
     shape; hairline gaps that leaves between neighbours close (CLOSE). A
     terrace on a gentle curve comes out straight along the game's straight
     street, at the back of its pavement, and party walls stay shared. Only
     blocks within a street's length (END_REACH past its ends) are carried. Any other block is
     moved rigidly: turned square to the tiles about its centre (the way its
     walls run, wall_angle), then shifted by the
     move KnoxMap gave its buildings (the area-weighted mean of their centres'
     moves). Either way it then moves by under half a tile so its main walls
     fall on tile edges. With FRONT_ANCHOR in the map's folder (LOG 2026-10-25
     #3), each building is turned about the middle of its street-facing edge
     (front_point) instead of its centre, so that edge stays on the building
     line and the back takes the error; a block carried along its street then
     has its houses' fronts on each side put on one line (one_line), where
     they are within LINE_SNAP of it (its slivers stay where the carry put
     them); a block moved rigidly is anchored on the road nearest it.
  5. Kept off the roads: every road within ROAD_NEAR of the block, in
     reality or in the game, that is not tagged to run under a building
     (layer < 0, tunnel, covered) must stay off the block in the game - its carriageway always
     (plus REPEL), its pavement too where the real gap is that wide. And a
     block that stands at least GAP_KEEP from a neighbour in reality keeps a
     tile from it. The smallest whole-tile shift (up to NUDGE) that does that
     is taken - for a block moved rigidly, the one that also brings it
     nearest its real distance from the named street nearest it (within
     SETBACK_NEAR); a block carried along its street slides along it, and steps
     back from it (STEP_BACK at most, at a higher price) only where the real
     street is narrower than any the game can draw. Tiles still on a carriageway after that are
     trimmed off: a building a little short beats one standing in the road.
     A service lane (no pavement, often a guess at a yard's drive) only
     pushes a block where that is easy and is never cut back for.
     Once every block has its tiles, a long non-home building (LONG_M,
     LONG_ASPECT, LONG_FILL; not LONG_HOMES, an unnamed building=yes or the
     LiDAR's own) that came out a staircase (LONG_STAIR) or that a road cut
     (LONG_TRIM) is lifted and laid as one straight rectangle of its real
     length and width, at the nearest place up to LONG_REACH tiles away that
     is off every road, on the real building's side of each road near it, and
     neither on nor beside another building (straighten_pass); with nowhere to
     go it stays as it was.
     Then, on a road-graph map, a block still with squares on the street as
     the hand-over painted it (carriageway, bays, verge, pavement: wider than
     the roads above) is moved to the nearest whole-tile shift up to
     STREET_REACH where it has none, on free ground and not newly beside
     another building (street_pass, `off_street`). A block whose buildings
     stand apart on the tiles moves piece by piece, so a row keeps its
     frontage where it is clear; with nowhere clear it stays as it was.
  6. Tiles: each tile whose middle is inside one of the block's outlines
     belongs to that building. Each building is then rebuilt from at most
     RECT_COUNT rectangles at least RECT_TILES tiles each way (rect_cover):
     bits sticking out narrower than that go. Where another block got there first, this one
     tries a one-tile shift, then gives way.
  7. Lots: the game builds each building on a rectangle that may not overlap
     another's. A building whose rectangle would take in a neighbour's tiles
     (an L reaching behind the next shop) is cut into pieces whose rectangles
     do not; pieces are separate buildings with the same parent.

Before all this, homes OSM draws one by one that share a party wall, in a
row near the grid, are joined into one building each (join_rows; `row_of`
their ids): the unit splitter cuts them back into homes at their addresses,
under one roof. A terrace on a 45-degree street is left to step house by house.

Written to the map's buildings file: each piece as an outline on tile edges,
`placed` (so Build draws it exactly, without squaring or nudging), `block`,
`laid_straight` (straighten_pass), `off_street` (street_pass), `row_of` (join_rows),
and `lidar_frame` - the exact turn and shift back to the real building,
with the registration folded in, so the roof planner and unit splitter read
the LiDAR on these tiles. Trees under the new tiles are cleared from the
vegetation maps, and service lanes under them from the ground maps (with any
stretch of lane then joined to no street; clear_lanes). The original buildings file is kept as *.orig.geojson and
each run starts again from it.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

import numpy as np
import shapely
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import nearest_points, transform, unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator import pz_colors as C  # noqa: E402
from generator.renderer import Projector  # noqa: E402
from knoxbuild.footprint import MIN_TILES, _clear_box, _hug, _largest_component  # noqa: E402
from area.roofs import mosaic  # noqa: E402
from buildings.tidy_outlines import fold  # noqa: E402

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True).transform
TO_WGS = Transformer.from_crs(27700, 4326, always_xy=True).transform

TOUCH_M = 0.3          # outlines this close are one block
GAP_M = 2.5            # outlines closer than this may be one row with a sliver between
GAP_ROOF = 0.5         # ...and are, when this share of the sliver is roof in the LiDAR
REG_M = 2.0            # registration: search shifts up to this, metres
REG_STEP = 0.5
REG_RES = 0.5          # grid the registration is measured on, metres
REG_HEIGHT_M = 2.5     # above the ground: building, where the cloud has no class
REG_GAIN = 0.03        # a shift must raise the overlap by this much to be used
NUDGE = 4              # tiles a block may move to stand off the road
STEP_BACK = 2          # ...and a block on its street, back from it (a pinch point)
ROAD_NEAR = 20         # roads this near a block in reality are kept their distance
REPEL = 0.5            # ...at least this clear of the carriageway
GAP_KEEP = 0.8         # blocks this far apart in reality stay a tile apart in the game
COVER_LOW = 0.3        # an outline less than this covered: not in the LiDAR
LIDAR_ONLY_M2 = 20     # a LiDAR building this big that no outline covers is reported
NEW_MIN_M2 = 30        # ...this big, and
NEW_CLEAR_M = 0.5      # ...not on an outline, is a building OSM lacks
NEW_CLOUD = 0.9        # ...where the point cloud covers this much of it
NEW_SPLIT_M = 1.0      # links this thin between two LiDAR buildings are cut (a pair of semis)
# ...but not a tree whose crown the point cloud classed as building. Of the
# points more than 2 m up in a real roof most are building class (OSM houses
# 0.56-0.96 where checked); a crown is half vegetation (0.47-0.49). And a
# crown stands well above the houses round it (+5.5 to +17.5 m; houses at
# most +3.4 m). Checked by eye on the point cloud (out/lidar/lidar_added_trees.png).
TREE_BUILDING_SHARE = 0.5   # fewer of its raised points building class than this: a tree
TREE_ABOVE_M = 5.0          # ...or this much taller than the buildings within TREE_AROUND_M
TREE_ALONE_M = 12.0         # ...or this tall with none round it
TREE_AROUND_M = 60.0
RECT_TILES = 2         # every building is rectangles at least this many tiles each way
                       # (no 1x1 or 2x1 bits; a later fine-tuning pass may add them back)
                       # - plotlib.py's RECT_TILES (3) is not this value out of sync: a
                       # plot's minimum is PLAN.md principle 13 (square plots), a building's
                       # is principle 12, and the two are allowed to differ
RECT_COUNT = 0         # ...and at most this many of them (0: as many as it takes - a block
                       # of shops drawn as one outline with a dozen rear wings needs a dozen;
                       # capped at 6, one lost half its ground)
SETBACK_NEAR = 30      # a rigid block is drawn to its real distance from the street this near
NEW_ID = 9_000_000_000_000
MIN_SIDE = 3           # Build's min_size: narrower pieces are not built
SNAP_DEG = 45          # carried buildings are always squared to the tiles: the game
                       # cannot turn a building, and one left at its street's angle
                       # (45 degrees on a straightened side street) is a staircase
CLOSE = 0.6            # gaps this wide (each side) between a block's buildings close
COURT_MIN = MIN_TILES  # a hole in a building this big is a courtyard, left open (smaller: built over)
BIG_TILES = 400        # a building this big (a school, a works), once turned to the grid,
SQUARE_DEG = 12        # ...has every wall this near the grid made square to it: OSM draws
                       # its walls a few degrees apart, and a 60 m wall 3 degrees off is a
                       # staircase of 2-tile steps (PLAN A4 item 9; Alton College, 2026-10-14)
SQUARE_IOU_LOSS = 0.01 # ...only where the squared outline fits the LiDAR's building ground
                       # (IoU, registered) no worse than this below the outline as drawn:
                       # the LiDAR is the truth for shape (principle 1); 0.01 is the IoU's
                       # own noise on its 0.5 m grid
LONG_M = 15.0          # a building this long (its real outline's long side)...
LONG_ASPECT = 2.5      # ...and this many times as long as it is wide (a range, a terrace row)
LONG_FILL = 0.65       # ...filling this much of its own rectangle (a range, not an L or a works)...
LONG_STAIR = 0.8       # ...laid as a staircase (filling less than this of its box), or
LONG_TRIM = 0.1        # ...that would lose this share of its tiles to a road, is laid as one
                       # straight rectangle - its real length, its real area's width - and
                       # put whole inside its block (straighten_long): the nearest place up to
                       # LONG_REACH tiles away that is off the roads and every other building's
                       # tiles, over garden ground (the plots are drawn after). User, 2026-10-19,
                       # on the Quaker meeting house, Alton, a staircase carried across Old
                       # Odiham Road: "place it fully in the block at the expense of other
                       # plots as long as it doesn't touch the other polygons ... prioritise
                       # these really long polygons where stepping doesn't make sense"
LONG_HOMES = {"terrace", "house", "houses", "semidetached_house", "detached", "residential", "apartments",
              "bungalow", "cottage", "dormitory", "farm", "static_caravan"}
                       # ...but never homes: a terrace steps house by house along its street, which
                       # looks right on the map (user, 2026-10-19); nor a building=yes with no name or
                       # use (most are terraces), nor the LiDAR's own (rows OSM lacks)
LONG_REACH = 20
LONG_TURN_COST = 6     # ...turned the other way round only where that is this much nearer
SIDE_NEAR = 10         # ...and stays on its side of each road this near the real building
STREET_REACH = 3       # a block left with squares on the street as the road graph painted it
                       # (carriageway, bays, verge, pavement: wider than the roads step 5 keeps
                       # off) is moved up to this many tiles to where it has none, if the ground
                       # there is free (street_pass). User, 2026-10-20: "corners still end up on
                       # roads, with clear space behind them to be nudged back"
ROW_WALL_M = 2.0       # homes OSM draws one by one (semis, a terrace) sharing a party wall this
                       # long are one building, a row (join_rows): Build gives it one roof and the
                       # unit splitter cuts it back into homes at its addresses - not a cluster of
                       # small separately roofed boxes (Normandy Street, Alton; user, 2026-10-19)...
ROW_DEPTH = 1.6        # ...homes no more than this many times deeper than each other...
ROW_GRID_DEG = 20      # ...in a row running this near the grid (on a 45-degree street a terrace
                       # steps house by house, which looks right: user, 2026-10-19)
ROW_MAX = 12           # ...and at most this many in one row
FRONT_ANCHOR = "front_anchor.on"      # in a map's folder: buildings squared about their street-facing edge
ANCHOR_FRONT = False                  # ...its switch (main sets it; roadgraph/mode_test.py too)
LINE_SNAP = 1.5                       # tiles: a house's front this near its run's line is put on it
FRONT_TURN = 45.0                     # degrees: an edge this near the street's way faces it
ROWS_ANY_ANGLE = "rows_any_angle.on"   # in a map's folder: rows joined at any angle, laid square in their own
                       # frame, for the turned-buildings step to turn onto their road (rotate/; LOG 2026-10-21 #D6)
FRONT_M = 45           # blocks this near a street (and nearest it) are carried along it
END_REACH = 8          # a street carries nothing further than this past its ends
FRONT_NEAR_M = 15      # ...and any block this near its centre line fronts it, whatever
                       # lane runs closer behind
MID_WINDOW_M = 15      # the midline between frontages, smoothed over this much street


def ring_lonlat(ft):
    return ft["geometry"]["coordinates"][0]


class Ground:
    """What the LiDAR says is building, on a 1 m BNG grid."""

    def __init__(self, area: str, box_):
        self.box = box_
        e0, n0, e1, n1 = box_
        cache = os.path.join(area, "lidar")
        dtm = mosaic("dtm", box_, cache)
        ndsm = mosaic("dsm", box_, cache) - dtm
        self.ndsm = ndsm
        built = np.nan_to_num(ndsm, nan=0.0) > REG_HEIGHT_M
        self.up_n = np.zeros(built.shape, np.int32)   # points more than 2 m up, per 1 m cell
        self.up_b = np.zeros(built.shape, np.int32)   # ...of them building class
        cloud = os.path.join(area, "nlp", "core.npz")
        self.source = "height"
        self.cloud = np.zeros_like(built)
        if os.path.exists(cloud):
            d = np.load(cloud)
            x, y, cls = d["x"], d["y"], d["cls"]
            c = ((x - e0)).astype(int)
            r = ((n1 - y)).astype(int)
            ok = (c >= 0) & (c < e1 - e0) & (r >= 0) & (r < n1 - n0)
            up = ok.copy()
            up[ok] = d["z"][ok] - np.nan_to_num(dtm[r[ok], c[ok]], nan=1e9) > 2.0
            np.add.at(self.up_n, (r[up], c[up]), 1)
            np.add.at(self.up_b, (r[up & (cls == 6)], c[up & (cls == 6)]), 1)
            seen = np.zeros_like(built)
            seen[r[ok], c[ok]] = True                 # where the cloud reaches
            cls6 = np.zeros_like(built)
            cls6[r[ok & (cls == 6)], c[ok & (cls == 6)]] = True
            # Building points, gaps between them closed, trimmed to what
            # stands up: trees (classes 3-5) drop out, roofs do not.
            near6 = cls6.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    near6 |= np.roll(np.roll(cls6, dy, 0), dx, 1)
            reach = seen.copy()
            for dy in (-2, 0, 2):
                for dx in (-2, 0, 2):
                    reach |= np.roll(np.roll(seen, dy, 0), dx, 1)
            built = np.where(reach, near6 & (np.nan_to_num(ndsm, nan=0.0) > 2.0), built)
            self.cloud = reach
            self.source = "cloud"
        self.built = built

    def window(self, bounds, res=REG_RES):
        """The built grid resampled to `res` over bounds (BNG), and its origin."""
        e0, n0, e1, n1 = self.box
        minx, miny, maxx, maxy = bounds
        xs = np.arange(minx, maxx, res) + res / 2
        ys = np.arange(maxy, miny, -res) - res / 2
        c = np.clip((xs - e0).astype(int), 0, self.built.shape[1] - 1)
        r = np.clip((n1 - ys).astype(int), 0, self.built.shape[0] - 1)
        return self.built[np.ix_(r, c)]


def raster_bng(poly, bounds, res=REG_RES):
    minx, miny, maxx, maxy = bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, res) + res / 2, np.arange(maxy, miny, -res) - res / 2)
    return shapely.contains_xy(poly, xs, ys)


def register(ground: Ground, block_bng) -> tuple[float, float, float, float]:
    """(dx, dy, overlap before, overlap after) laying the block on the LiDAR."""
    pad = REG_M + 1
    minx, miny, maxx, maxy = block_bng.bounds
    bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
    lid = ground.window(bounds)

    def iou_at(sx, sy):
        m = raster_bng(affinity.translate(block_bng, sx, sy), bounds)
        return (m & lid).sum() / max(1, (m | lid).sum())

    zero = iou_at(0.0, 0.0)          # no shift: how it sits as generated
    best = None
    for sx in np.arange(-REG_M, REG_M + 1e-9, REG_STEP):
        for sy in np.arange(-REG_M, REG_M + 1e-9, REG_STEP):
            iou = iou_at(sx, sy)
            key = (round(iou, 4), -(abs(sx) + abs(sy)))
            if best is None or key > best[0]:
                best = (key, sx, sy, iou)
    _k, sx, sy, iou = best
    if iou - zero < REG_GAIN:
        return 0.0, 0.0, zero, zero
    return float(sx), float(sy), zero, iou


def groups_of(n, pairs, size):
    """Groups of 0..n-1 joined by `pairs`, biggest first."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in pairs:
        parent[find(i)] = find(j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return sorted(out.values(), key=lambda m: -sum(size(i) for i in m))


def neighbours_touching(mine):
    from shapely.strtree import STRtree
    polys = [m["bng"] for m in mine]
    tree = STRtree(polys)
    for i, a in enumerate(polys):
        for j in tree.query(a.buffer(TOUCH_M)):
            j = int(j)
            if j > i and a.distance(polys[j]) <= TOUCH_M:
                yield i, j


def closing(poly, r):
    return poly.buffer(r, join_style=2).buffer(-r, join_style=2)


def neighbours(mine, ground, shift_of):
    """Pairs of outlines that are one block, and the slivers of roof between
    the ones OSM drew apart: [(i, j)], [(i, j, sliver in BNG)], real gaps."""
    from shapely.strtree import STRtree
    polys = [m["bng"] for m in mine]
    tree = STRtree(polys)
    pairs, slivers, open_gaps = [], [], 0
    for i, a in enumerate(polys):
        for j in tree.query(a.buffer(GAP_M)):
            j = int(j)
            if j <= i:
                continue
            b = polys[j]
            d = a.distance(b)
            if d >= GAP_M:
                continue
            if d <= TOUCH_M:
                pairs.append((i, j))
                continue
            r = d / 2 + 0.25
            gap = closing(unary_union([a, b]), r).difference(closing(a, r)).difference(closing(b, r))
            if gap.area < 0.5:
                pairs.append((i, j))
                continue
            # The LiDAR in the sliver, read where this pair's LiDAR sits.
            sx = (shift_of(i)[0] + shift_of(j)[0]) / 2
            sy = (shift_of(i)[1] + shift_of(j)[1]) / 2
            g = affinity.translate(gap, sx, sy)
            bounds = g.bounds
            rm = raster_bng(g, bounds, 0.25)
            if rm.sum() == 0:
                continue
            roof = (rm & ground.window(bounds, 0.25)).sum() / rm.sum()
            if roof >= GAP_ROOF:
                pairs.append((i, j))
                slivers.append((i, j, gap))
            else:
                open_gaps += 1
    return pairs, slivers, open_gaps


def front_point(poly, line) -> tuple[float, float]:
    """The middle of the building's edge that faces `line` (a street's centre line):
    of its outline's edges running within FRONT_TURN of the street there, the one
    whose middle is nearest it; with none, the outline's point nearest the street."""
    best = None
    for g in getattr(poly, "geoms", [poly]):
        pts = list(g.exterior.coords)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            ln = math.hypot(x1 - x0, y1 - y0)
            if ln < 1.0:
                continue
            mid = shapely.Point((x0 + x1) / 2, (y0 + y1) / 2)
            s_ = line.project(mid)
            a, b = line.interpolate(max(0.0, s_ - 1.0)), line.interpolate(min(line.length, s_ + 1.0))
            way = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
            edge = math.degrees(math.atan2(y1 - y0, x1 - x0))
            if abs((edge - way + 90) % 180 - 90) > FRONT_TURN:
                continue                                 # an end wall, not a front
            d = line.distance(mid)
            if best is None or d < best[0]:
                best = (d, (mid.x, mid.y))
    if best is None:
        q = nearest_points(poly, line)[0]
        return q.x, q.y
    return best[1]


class Street:
    """A street's centre line in the real map frame and in the game (both in
    tiles), to carry buildings along it. A point keeps its place along the
    street and its offset from the midline between the street's frontages
    (`mid`, a function of the distance along).

    Its place along is pinned at every junction: the straightened street is
    not stretched evenly (a stretch between two side streets can come out a
    dozen tiles longer or shorter), so the distance is interpolated between
    where each side street meets it, really and in the game (`anchors`)."""

    def __init__(self, name, real, game, mid, anchors=None):
        self.name, self.real, self.game, self.mid = name, real, game, mid
        self.Lr, self.Lg = real.length, game.length
        sr, sg = zip(*(anchors or [(0.0, 0.0), (self.Lr, self.Lg)]))
        self.sr, self.sg = np.array(sr), np.array(sg)

    def along(self, s):
        """Distance along the game street for a real distance along."""
        if s <= self.sr[0]:
            return float(self.sg[0] + (s - self.sr[0]))
        if s >= self.sr[-1]:
            return float(self.sg[-1] + (s - self.sr[-1]))
        return float(np.interp(s, self.sr, self.sg))

    @staticmethod
    def _dir(line, s):
        L = line.length
        a = line.interpolate(max(0.0, min(L, s - 1.0)))
        b = line.interpolate(max(0.0, min(L, s + 1.0)))
        t = np.array([b.x - a.x, b.y - a.y])
        t /= np.hypot(*t)
        return t, np.array([-t[1], t[0]])

    def locate(self, x, y):
        """(distance along, offset across) of a real point."""
        s = self.real.project(shapely.Point(x, y))
        t, n = self._dir(self.real, s)
        foot = self.real.interpolate(s)
        d = np.array([x - foot.x, y - foot.y])
        return s + float(d @ t), float(d @ n)

    def map(self, x, y):
        s, off = self.locate(x, y)
        sg = self.along(s)
        sc = max(0.0, min(self.Lg, sg))
        t, n = self._dir(self.game, sc)
        foot = self.game.interpolate(sc)
        q = np.array([foot.x, foot.y]) + t * (sg - sc) + n * (off - self.mid(s))
        return float(q[0]), float(q[1])

    def turn_at(self, x, y):
        """Degrees the street turns between the real map and the game here."""
        s, _off = self.locate(x, y)
        tr, _ = self._dir(self.real, max(0.0, min(self.Lr, s)))
        tg, _ = self._dir(self.game, max(0.0, min(self.Lg, self.along(s))))
        a = math.degrees(math.atan2(tg[1], tg[0]) - math.atan2(tr[1], tr[0]))
        return (a + 180) % 360 - 180

    def carry(self, poly):
        poly = shapely.segmentize(poly, 2.0)
        out = Polygon([self.map(x, y) for x, y in poly.exterior.coords],
                      [[self.map(x, y) for x, y in r.coords] for r in poly.interiors])
        return out if out.is_valid else out.buffer(0)

    def fit(self, poly):
        """The street's carry applied to one building: it moves rigidly to the
        turn and shift that best fit where the carry puts its outline (least
        squares), squared to the tiles within SNAP_DEG - so its shape stays its
        real shape, and its neighbours, carried by the same smooth mapping,
        land against it. (outline, turn, where its fit centre went, the fit
        centre)."""
        pts = np.array(shapely.segmentize(poly, 2.0).exterior.coords[:-1])
        tgt = np.array([self.map(x, y) for x, y in pts])
        pc, qc = pts.mean(0), tgt.mean(0)
        a, b = pts - pc, tgt - qc
        turn = math.degrees(math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(),
                                       (a * b).sum()))
        off = fold(wall_angle(poly) + turn)
        if abs(off) <= SNAP_DEG:
            turn -= off
        if ANCHOR_FRONT:
            # Turned about its street-facing edge's middle, that point put where the
            # carry puts it: the front stays on the building line, the back moves.
            a = front_point(poly, self.real)
            qa = self.map(*a)
            out = affinity.translate(affinity.rotate(poly, turn, origin=a), qa[0] - a[0], qa[1] - a[1])
            return out, turn, (float(qa[0]), float(qa[1])), (float(a[0]), float(a[1]))
        out = affinity.translate(affinity.rotate(poly, turn, origin=tuple(pc)),
                                 qc[0] - pc[0], qc[1] - pc[1])
        return out, turn, (float(qc[0]), float(qc[1])), (float(pc[0]), float(pc[1]))


def load_streets(map_dir, info, proj, names, polys_px, game_lines=None, passage_ways=frozenset()):
    """Street objects for `names`, and every road's real centre line (tiles)
    with its name, for telling which street a block fronts. `game_lines`
    ({OSM way id: (its line in tiles, tags to set)}): the roads as some routing
    drew them, in place of the map's own straightening (roadgraph/mode_test.py:
    each road mode's routing, with the width and roadside tags the hand-over
    gives a way; a way it has no line for left out). `passage_ways` (OSM way
    ids): ways to run under a building as if OSM tagged them a passage
    (mode_test.py's passage prototype, LOG 2026-10-25 #6)."""
    from shapely.ops import linemerge
    from generator import osm
    from generator.octilinear import ROAD_CATEGORIES, straighten_roads
    from generator.renderer import _is_polygon, classify
    cache = os.path.join(map_dir, info["osm_cache"])
    real_feats = osm.load_cache(cache, tuple(info["osm_bbox"]))
    game_feats = osm.load_cache(cache, tuple(info["osm_bbox"]))
    def is_road(f):
        return f.kind == "way" and len(f.geometry) >= 2 and not _is_polygon(f) and classify(f.tags) in ROAD_CATEGORIES
    if game_lines is not None:
        from dataclasses import replace
        game_feats = [f if not is_road(f) else replace(f, geometry=[proj.to_latlon(x, y)
                                                                    for x, y in game_lines[f.osm_id][0]],
                                                       tags={**f.tags, **game_lines[f.osm_id][1]})
                      for f in game_feats if not is_road(f) or f.osm_id in game_lines]
    elif info.get("straight_roads"):
        straighten_roads(game_feats, proj, classify, _is_polygon, map_dir=map_dir)

    def roads(feats):
        return [f for f in feats if is_road(f)]
    all_roads = [(f.tags.get("name") or "", shapely.LineString([proj.to_px(la, lo) for la, lo in f.geometry]))
                 for f in roads(real_feats)]
    # Every road as drawn in the game (centre line, half its carriageway,
    # its pavement), beside where it really is.
    from generator.renderer import SIDEWALK_M, _way_width_m

    def _sidewalk_width_m(feat, cat):
        """Pavement each side as the maps were generated (KnoxMap 1.4/1.5): the
        way's own sidewalk width tag, else its class's. Upstream 1.6 sizes
        pavements by road hierarchy; our maps' ground was drawn before that, and
        buildings must stand off the pavement as it is drawn."""
        raw = feat.tags.get("sidewalk:both:width") or feat.tags.get("sidewalk:width")
        if raw:
            try:
                return max(0.0, min(6.0, float(str(raw).split()[0].replace(",", "."))))
            except (ValueError, IndexError):
                pass
        return SIDEWALK_M.get(cat, 0.0)
    game_by_id = {f.osm_id: f for f in roads(game_feats)}
    road_list = []
    for f in roads(real_feats):
        g = game_by_id.get(f.osm_id)
        if g is None:
            continue
        cat = classify(f.tags)
        road_list.append({
            "name": f.tags.get("name") or "",
            "real": shapely.LineString([proj.to_px(la, lo) for la, lo in f.geometry]),
            "game": shapely.LineString([proj.to_px(la, lo) for la, lo in g.geometry]),
            "half": _way_width_m(g, cat) / 2 / info["meters_per_tile"],
            "pave": _sidewalk_width_m(g, cat) / info["meters_per_tile"],
            # A service lane is paint on the ground: a building is pushed off
            # it where that is easy, never cut back for it.
            "minor": cat == "road_service",
            # Under a building on purpose: a basement ramp, an arch.
            "under": (str(f.tags.get("layer", "0")).lstrip("-").isdigit() and int(f.tags.get("layer", "0")) < 0)
            or f.tags.get("tunnel") in ("building_passage", "yes") or f.tags.get("covered") == "yes"
            or f.osm_id in (passage_ways or ())})
    # The railways too, as a carriageway with no pavement: the track's bed
    # (rail_kit; the railway's own buildings are let stand on it, roads_to_clear).
    from buildings import rail_kit
    game_rail = {f.osm_id: f for f in game_feats if rail_kit.is_rail_line(f, classify, _is_polygon)}
    for f in real_feats:
        g = game_rail.get(f.osm_id) if rail_kit.is_rail_line(f, classify, _is_polygon) else None
        if g is None:
            continue
        road_list.append({
            "name": "", "rail": True,
            "real": shapely.LineString([proj.to_px(la, lo) for la, lo in f.geometry]),
            "game": shapely.LineString([proj.to_px(la, lo) for la, lo in g.geometry]),
            "half": rail_kit.RAIL_HALF_M / info["meters_per_tile"], "pave": 0.0, "minor": False,
            "under": f.tags.get("tunnel", "no") != "no"})
    streets = []
    from shapely.strtree import STRtree
    tree = STRtree(polys_px)
    pairs = []
    for nm in names:
        real = linemerge([ln for n, ln in all_roads if n == nm])
        game = linemerge([shapely.LineString([proj.to_px(la, lo) for la, lo in f.geometry])
                          for f in roads(game_feats) if f.tags.get("name") == nm])
        if real.is_empty or game.is_empty:
            continue
        if real.geom_type == "LineString" and game.geom_type == "LineString":
            pairs.append((nm, real, game, True))
        else:
            # A street that forks: each of its ways is carried on its own
            # (one way's real and game lines already run the same way).
            for r in road_list:
                if r["name"] == nm:
                    pairs.append((nm, r["real"], r["game"], False))
    for nm, real, game, merged in pairs:
        g0, g1 = shapely.Point(game.coords[0]), shapely.Point(game.coords[-1])
        if merged and shapely.Point(real.coords[0]).distance(g1) < shapely.Point(real.coords[0]).distance(g0):
            game = shapely.LineString(list(game.coords)[::-1])
        # The midline between the frontages either side, every 5 tiles.
        samples = []
        for d in np.arange(0.0, real.length, 5.0):
            foot = real.interpolate(d)
            t, n = Street._dir(real, d)
            sides = []
            for sgn in (1, -1):
                ray = shapely.LineString([(foot.x, foot.y), (foot.x + sgn * n[0] * 30, foot.y + sgn * n[1] * 30)])
                ds = [foot.distance(ray.intersection(polys_px[k])) for k in tree.query(ray)
                      if not ray.intersection(polys_px[k]).is_empty]
                sides.append(min(ds) if ds else None)
            if None not in sides:
                samples.append((d, (sides[0] - sides[1]) / 2))
        if samples:
            sd, sm = np.array(samples).T
            centre = float(np.median(sm))

            def mid(s, sd=sd, sm=sm, centre=centre):
                near = np.abs(sd - s) <= MID_WINDOW_M
                return float(np.median(sm[near])) if near.sum() >= 3 else centre
        else:
            def mid(s):
                return 0.0
        # Junctions: the OSM nodes this street shares with each other road.
        # The straightening keeps a shared node shared, so the same junction
        # is the point the two game lines have in common.
        def shared(a, b):
            ka = {(round(x, 3), round(y, 3)) for x, y in a.coords}
            return [pt for pt in ((round(x, 3), round(y, 3)) for x, y in b.coords) if pt in ka]
        pins = []
        for r in road_list:
            if r["name"] == nm:
                continue
            jr, jg = shared(real, r["real"]), shared(game, r["game"])
            if not jr or len(jr) != len(jg):
                continue
            sr_ = sorted(real.project(shapely.Point(pt)) for pt in set(jr))
            sg_ = sorted(game.project(shapely.Point(pt)) for pt in set(jg))
            if len(sr_) == len(sg_):
                pins += list(zip(sr_, sg_))
        anchors = []
        for a in sorted([(0.0, 0.0)] + pins + [(real.length, game.length)]):
            if not anchors or (a[0] > anchors[-1][0] + 1.0 and a[1] > anchors[-1][1] + 1.0):
                anchors.append(a)
        worst = max((abs(g - r_ / real.length * game.length) for r_, g in anchors), default=0.0)
        print(f"  {nm}: pinned at {len(pins)} junctions; an even stretch would be up to {worst:.0f} tiles out")
        st = Street(nm, real, game, mid, anchors)
        streets.append(st)
        print(f"  {nm}: {real.length:.0f} tiles real, {game.length:.0f} in the game; "
              f"midline {np.median([m for _d, m in samples]) if samples else 0:+.1f} tiles off the centre line "
              f"({len(samples)} samples)")
    return streets, all_roads, road_list


def tile_mask(polys, x0, y0, w, h):
    """Owner index (1-based, 0 none) of each tile, by the tile's middle."""
    xs, ys = np.meshgrid(np.arange(x0, x0 + w) + 0.5, np.arange(y0, y0 + h) + 0.5)
    own = np.zeros((h, w), np.int32)
    for k, p in enumerate(polys, 1):
        inside = shapely.contains_xy(p, xs, ys) & (own == 0)
        own[inside] = k
    return own


def _best_box(held, inside, side):
    """The rectangle, both sides at least `side`, wholly within `inside`,
    holding the most `held` tiles: (tiles, y0, y1, x0, x1), half open."""
    h, w = inside.shape
    acc = np.zeros((h + 1, w + 1), dtype=np.int32)
    acc[1:, 1:] = held.cumsum(0).cumsum(1)
    best = (0, 0, 0, 0, 0)
    heights = np.zeros(w, dtype=np.int32)
    for y in range(h):
        heights = np.where(inside[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x, tall in enumerate(heights.tolist() + [0]):
            start = x
            while stack and stack[-1][1] >= tall:
                start, high = stack.pop()
                if high >= side and x - start >= side:
                    top = y + 1 - high
                    n = int(acc[y + 1, x] - acc[top, x] - acc[y + 1, start] + acc[top, start])
                    if n > best[0]:
                        best = (n, top, y + 1, start, x)
            stack.append((start, tall))
    return best


def rect_cover(mask, side=RECT_TILES, most=RECT_COUNT):
    """The building's tiles as rectangles at least `side` tiles each way
    (at most `most` of them; 0 for no limit), chosen greedily, each adding
    at least side x side tiles not yet covered; what no such rectangle
    reaches (a spur, a strip narrower than `side`) is dropped."""
    keep = np.zeros_like(mask, dtype=bool)
    count = 0
    while not most or count < most:
        todo = mask & ~keep
        if todo.sum() < side * side:
            break
        n, y0, y1, x0, x1 = _best_box(todo, mask, side)
        if n < side * side:   # nothing new worth a rectangle
            break
        keep[y0:y1, x0:x1] = True
        count += 1
    return keep


def clear_lanes(map_dir, name, built, road_list) -> int:
    """Take service lanes off the ground where a building now stands on them,
    and with them any stretch of lane that then leads nowhere (left joined to
    no other road): a lane drawn through a house came out of its far wall.
    The ground under a lane taken off is the ground beside it. Both ground
    maps (the drawn one and the base Build dresses from) are edited."""
    from scipy import ndimage
    carriage_colours = (C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE)
    total = 0
    for bmp_name in (f"{name}.bmp", f"{name}_ground_base.bmp"):
        path = os.path.join(map_dir, bmp_name)
        if not os.path.exists(path):
            continue
        g = np.array(Image.open(path).convert("RGB"))
        H, W = g.shape[:2]
        carriage = np.zeros((H, W), bool)
        for c in carriage_colours:
            carriage |= np.all(g == c, axis=2)
        # The service lanes' own carriageway, by their drawn lines.
        lane_img = Image.new("1", (W, H), 0)
        dr = ImageDraw.Draw(lane_img)
        for r in road_list:
            if r["minor"] and not r["under"] and len(r["game"].coords) >= 2:
                dr.line([tuple(c) for c in r["game"].coords], fill=1, width=max(1, int(r["half"] * 2)))
        lane = np.array(lane_img, dtype=bool) & carriage
        other = carriage & ~lane                    # the streets, which stay
        gone = lane & built
        kept = lane & ~gone
        # Lane pieces now joined to no street go too.
        lab, n = ndimage.label(kept | other)
        if n:
            joined = np.zeros(n + 1, bool)
            joined[np.unique(lab[other])] = True
            joined[0] = False
            gone |= kept & ~joined[lab]
        if not gone.any():
            continue
        # Each taken-off tile takes the colour of the nearest tile that is
        # neither carriageway nor taken off.
        keep_px = ~(carriage | gone)
        _d, (iy, ix) = ndimage.distance_transform_edt(~keep_px, return_indices=True)
        g[gone] = g[iy[gone], ix[gone]]
        Image.fromarray(g).save(path, format="BMP")
        total = max(total, int(gone.sum()))
    return total


def mask_shape(mask, x0, y0):
    """The tiles of a mask as one (multi)polygon, in tile coords."""
    rows = []
    for y in range(mask.shape[0]):
        xs = np.nonzero(mask[y])[0]
        if len(xs) == 0:
            continue
        start = xs[0]
        for a, b in zip(xs, list(xs[1:]) + [None]):
            if b is None or b != a + 1:
                rows.append(box(x0 + start, y0 + y, x0 + a + 1, y0 + y + 1))
                if b is not None:
                    start = b
    return unary_union(rows)


def wall_angle(poly) -> float:
    """The direction its walls run, -45..45 degrees from the grid: the
    length-weighted mean of its edges' angles on the 90-degree circle. (The
    tightest rectangle round an L can sit at 45 degrees to its walls.)"""
    z = 0j
    for g in getattr(poly, "geoms", [poly]):
        if g.geom_type != "Polygon":
            continue
        pts = list(g.exterior.coords)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            ln = math.hypot(x1 - x0, y1 - y0)
            if ln > 0:
                z += ln * complex(math.cos(4 * math.atan2(y1 - y0, x1 - x0)),
                                  math.sin(4 * math.atan2(y1 - y0, x1 - x0)))
    return fold(math.degrees(math.atan2(z.imag, z.real)) / 4) if abs(z) > 0 else 0.0


def squared_walls(poly, tol=SQUARE_DEG):
    """An outline already turned to the grid with every wall within `tol`
    degrees of it made square to it: a run of such walls one way becomes one
    line, at their length-weighted mean; other walls (a real diagonal wing, a
    curve) stay as they are, joined where the lines meet. The outline as it
    was if the result is not one valid polygon (whether it is kept is the
    LiDAR's to say: square_big)."""
    def ring(coords):
        pts = list(coords)[:-1]
        n = len(pts)
        edges = []
        for i in range(n):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
            ln = math.hypot(x1 - x0, y1 - y0)
            if ln == 0:
                continue
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
            kind = "h" if min(ang, 180 - ang) <= tol else "v" if abs(ang - 90) <= tol else None
            edges.append((kind, (x0, y0), (x1, y1), ln))
        if len(edges) < 3 or all(e[0] is None for e in edges):
            return None
        for _ in range(len(edges)):                   # start at a change of kind
            if edges[0][0] is None or edges[0][0] != edges[-1][0]:
                break
            edges = edges[1:] + edges[:1]
        lines = []                                    # (kind, value or (p0, p1))
        for kind, p0, p1, ln in edges:
            if kind is not None and lines and lines[-1][0] == kind:
                k, (s, w) = lines[-1]
                lines[-1] = (k, (s + ln * ((p0[1] + p1[1]) / 2 if k == "h" else (p0[0] + p1[0]) / 2), w + ln))
            elif kind is not None:
                lines.append((kind, (ln * ((p0[1] + p1[1]) / 2 if kind == "h" else (p0[0] + p1[0]) / 2), ln)))
            else:
                lines.append((None, (p0, p1)))
        if len(lines) < 3:
            return None

        def meet(a, b):
            (ka, va), (kb, vb) = a, b
            if ka is not None:
                va = va[0] / va[1]
            if kb is not None:
                vb = vb[0] / vb[1]
            if ka == "h" and kb == "v":
                return (vb, va)
            if ka == "v" and kb == "h":
                return (va, vb)
            if ka is None and kb is None:
                return va[1]
            if ka is None:
                (x0, y0), (x1, y1) = va
                c, k = vb, kb
                end = (x1, y1)
            else:
                (x0, y0), (x1, y1) = vb
                c, k = va, ka
                end = (x0, y0)
            if k == "h":
                return (x0 + (c - y0) * (x1 - x0) / (y1 - y0), c) if y1 != y0 else end
            return (c, y0 + (c - x0) * (y1 - y0) / (x1 - x0)) if x1 != x0 else end
        return [meet(lines[i], lines[(i + 1) % len(lines)]) for i in range(len(lines))]
    outer = ring(poly.exterior.coords)
    if outer is None:
        return poly
    holes = [ring(r.coords) or list(r.coords) for r in poly.interiors]   # an off-grid courtyard as drawn
    out = Polygon(outer, holes)
    if not out.is_valid:
        out = out.buffer(0)
    if out.geom_type != "Polygon":
        return poly
    return out


RECT_RES = 0.5         # grid a LiDAR shape is cut into rectangles on, metres
RECT_MIN = 3.0         # a rectangle of a new building at least this wide
RECT_MAX = 4           # ...and at most this many of them
RECT_KEEP = 0.5        # the rectangles must hold this share of the shape


def rectify(poly):
    """A LiDAR shape as buildings: cut where thin links join two, each part as
    a few solid rectangles on its own axis (the house, a wing, a garage).
    Returns the parts that fit, maybe none."""
    parts = poly.buffer(-NEW_SPLIT_M / 2, join_style=2).buffer(NEW_SPLIT_M / 2, join_style=2)
    out = []
    for part in getattr(parts, "geoms", [parts]):
        if part.geom_type == "Polygon" and part.area >= NEW_MIN_M2 * 0.6:
            fit = _rectify_one(part)
            if fit is not None and fit.area >= NEW_MIN_M2:
                out.append(fit)
    return out


def _rectify_one(poly):
    rect = poly.minimum_rotated_rectangle
    rc = list(rect.exterior.coords)
    e1 = (rc[1][0] - rc[0][0], rc[1][1] - rc[0][1])
    e2 = (rc[2][0] - rc[1][0], rc[2][1] - rc[1][1])
    long_edge = e1 if math.hypot(*e1) >= math.hypot(*e2) else e2
    angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    c = poly.centroid
    up = affinity.rotate(poly, -angle, origin=c)
    # Spurs and notches under a metre off first.
    up = closing(up.buffer(-0.5, join_style=2).buffer(0.5, join_style=2), 0.5)
    if up.is_empty:
        return None
    minx, miny, maxx, maxy = up.bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, RECT_RES) + RECT_RES / 2,
                         np.arange(miny, maxy, RECT_RES) + RECT_RES / 2)
    mask = shapely.contains_xy(up, xs, ys)
    total = int(mask.sum())
    if total == 0:
        return None
    rects, left = [], mask.copy()
    need = int(RECT_MIN / RECT_RES)
    for _ in range(RECT_MAX):
        y0, y1, x0, x1 = _clear_box(np.ones_like(left), ~left)
        if y1 - y0 < need or x1 - x0 < need:
            break
        rects.append(box(minx + x0 * RECT_RES, miny + y0 * RECT_RES,
                         minx + x1 * RECT_RES, miny + y1 * RECT_RES))
        left[y0:y1, x0:x1] = False
    if not rects or (total - int(left.sum())) < RECT_KEEP * total:
        return None
    shape_ = unary_union(rects)
    if shape_.geom_type != "Polygon":
        shape_ = max(shape_.geoms, key=lambda g: g.area)
    return affinity.rotate(Polygon(shape_.exterior), angle, origin=c)


def bng_to_px(proj, poly):
    def one(x, y):
        lon, lat = TO_WGS(x, y)
        return proj.to_px(lat, lon)
    return unary_union([Polygon([one(x, y) for x, y in g.exterior.coords]).buffer(0)
                        for g in getattr(poly, "geoms", [poly])
                        if g.geom_type == "Polygon" and g.area > 0.05])


def _holes(mask):
    from scipy import ndimage as ndi
    filled = ndi.binary_fill_holes(mask)
    return filled & ~mask


def mask_outline(mask, x0, y0):
    """The outline of a mask on tile edges, exterior only, in tile coords."""
    u = mask_shape(mask, x0, y0)
    if u.geom_type != "Polygon":
        u = max(u.geoms, key=lambda g: g.area)
    return Polygon(u.exterior).simplify(0.01)


# --- the steps, in the order main() runs them ---------------------------------------------


def streets_measured(map_dir, info):
    """The streets street_widths.py measured (the ones with frontages), the
    High Street first: a corner block goes with the first street listed."""
    import gzip
    with gzip.open(os.path.join(map_dir, info["osm_cache"]), "rt", encoding="utf-8") as f:
        names = sorted({ft["tags"]["name"] for ft in json.load(f)["features"]
                        if ft["kind"] == "way" and ft["tags"].get("width:source") and ft["tags"].get("name")},
                       key=lambda nm: nm != "High Street")
    return names or ["High Street"]


def relation_outlines(map_dir, info):
    """Buildings OSM maps as multipolygon relations (a school round a
    courtyard), which the area's buildings.geojson (ways only) lacks: their
    real outline in lon/lat from the map's OSM download, inner rings kept as
    holes. {osm_id: Polygon}; a relation of several outer parts keeps its
    biggest."""
    import gzip
    with gzip.open(os.path.join(map_dir, info["osm_cache"]), "rt", encoding="utf-8") as f:
        feats = json.load(f)["features"]
    out = {}
    for ft in feats:
        if ft["kind"] != "relation" or "building" not in ft["tags"]:
            continue
        rings = {"outer": [], "inner": []}
        for role, ring in ft.get("role_geoms") or []:
            if len(ring) >= 4:
                rings["inner" if role == "inner" else "outer"].append(Polygon([(lon, lat) for lat, lon in ring]))
        if not rings["outer"]:
            continue
        poly = unary_union([p.buffer(0) for p in rings["outer"]])
        if rings["inner"]:
            poly = poly.difference(unary_union([p.buffer(0) for p in rings["inner"]]))
        if poly.geom_type != "Polygon":
            poly = max(getattr(poly, "geoms", []), key=lambda g: g.area, default=None)
        if poly is not None and not poly.is_empty:
            out[str(ft["osm_id"])] = poly
    return out


def lonlat_to_px(proj, poly):
    """A lon/lat polygon in tiles, holes kept."""
    def ring(r):
        return [proj.to_px(lat, lon) for lon, lat in r.coords]
    return Polygon(ring(poly.exterior), [ring(r) for r in poly.interiors])


def outlines_to_place(gj, real, proj, W, H, relations=None):
    """KnoxMap's buildings, split into those left to it (`keep`) and those
    placed here (`mine`): one with a real outline in the area (a way in
    `real`, or a relation in `relations`: relation_outlines) that is on the
    map, valid and at least 4 tiles. Each of `mine` is a record holding its
    real outline in BNG (`bng`) and in tiles (`px`), courtyards as holes, and
    KnoxMap's moved one (`moved`); the steps below add to it."""
    relations = relations or {}
    keep, mine = [], []
    for ft in gj["features"]:
        oid = str(ft["properties"].get("osm_id"))
        if oid in real and real[oid]["geometry"]["type"] == "Polygon":
            outline = shape(real[oid]["geometry"])
        else:
            outline = relations.get(oid)
        if outline is not None and ft["geometry"]["type"] in ("Polygon", "MultiPolygon"):
            bng = transform(TO_BNG, outline)
            px = lonlat_to_px(proj, outline)
            given = shape(ft["geometry"])
            given = max(getattr(given, "geoms", [given]), key=lambda g: g.area)
            moved = Polygon([proj.to_px(lat, lon) for lon, lat in given.exterior.coords])
            inside = moved.intersects(box(0, 0, W, H))
            if inside and bng.is_valid and px.is_valid and moved.is_valid and px.area >= 4:
                mine.append({"ft": ft, "id": oid, "bng": bng, "px": px, "moved": moved})
                continue
        keep.append(ft)
    return keep, mine


def join_rows(mine, proj):
    """Homes OSM draws one by one that share a party wall (ROW_WALL_M) and are
    about as deep as each other, in a row near the grid: each row one record,
    the union of its homes (the first one's id and tags, `row_of` its ids),
    which the unit splitter cuts back into homes at their addresses. Returns
    (records, rows joined, homes in them)."""
    from shapely.strtree import STRtree
    homes = [k for k, m in enumerate(mine) if not not_homes(m)]
    polys = [mine[k]["bng"] for k in homes]
    tree = STRtree(polys)
    pairs = []
    for a, k in enumerate(homes):
        pa = polys[a]
        for b in tree.query(pa.buffer(TOUCH_M)):
            b = int(b)
            if b <= a:
                continue
            pb = polys[b]
            wall = pa.buffer(TOUCH_M).intersection(pb.buffer(TOUCH_M)).area / (2 * TOUCH_M)
            if wall < ROW_WALL_M:
                continue
            da, db = long_size(pa)[1], long_size(pb)[1]
            if max(da, db) > ROW_DEPTH * max(min(da, db), 1e-6):
                continue
            pairs.append((a, b))
    out, joined, used = [], 0, set()
    for group in groups_of(len(homes), pairs, lambda a: polys[a].area):
        if len(group) < 2 or len(group) > ROW_MAX:
            continue
        idx = [homes[a] for a in group]
        union = unary_union([mine[k]["bng"].buffer(TOUCH_M) for k in idx]).buffer(-TOUCH_M)
        if union.geom_type != "Polygon":
            continue
        upx = unary_union([mine[k]["px"].buffer(0.3) for k in idx]).buffer(-0.3)
        if upx.geom_type != "Polygon" or abs(fold(wall_angle(upx))) > ROW_GRID_DEG:
            continue
        first = min(idx, key=lambda k: str(mine[k]["id"]))
        moved = unary_union([mine[k]["moved"] for k in idx])
        moved = moved if moved.geom_type == "Polygon" else moved.convex_hull
        ft = dict(mine[first]["ft"], properties=dict(mine[first]["ft"]["properties"],
                                                       row_of=[str(mine[k]["id"]) for k in idx]))
        out.append({"ft": ft, "id": mine[first]["id"], "bng": union, "px": upx, "moved": moved})
        used.update(idx)
        joined += 1
    return [m for k, m in enumerate(mine) if k not in used] + out, joined, len(used)


def register_blocks(ground, mine, blocks):
    """Step 2: each block registered onto the LiDAR as one piece, every
    member's shift put on its record (`reg`). Returns each block's shift and
    its (overlap before, overlap after)."""
    shifts, gains = [], []
    for members in blocks:
        sx, sy, before, after = register(ground, unary_union([mine[i]["bng"] for i in members]))
        shifts.append((sx, sy))
        gains.append((before, after))
        for i in members:
            mine[i]["reg"] = (sx, sy)
    return shifts, gains


def px_to_bng(proj, poly):
    """A polygon in tiles (the real map's frame) in BNG, holes kept."""
    def ring(r):
        return [TO_BNG(*proj.to_latlon(x, y)[::-1]) for x, y in r.coords]
    return Polygon(ring(poly.exterior), [ring(r) for r in poly.interiors])


def lidar_iou(ground, poly, bounds):
    """Overlap (IoU) of an outline (BNG, registered) with the LiDAR's building
    ground in `bounds` (the same window for every outline compared: a window
    of its own counted different neighbours)."""
    m = raster_bng(poly, bounds)
    lid = ground.window(bounds)
    return float((m & lid).sum() / max(1, (m | lid).sum()))


def square_big(ground, mine, proj):
    """Which big buildings place_block squares (squared_walls): each of
    BIG_TILES or more, its walls squared as it stands turned to the grid,
    is marked `square` if that fits the LiDAR no worse than SQUARE_IOU_LOSS
    below its outline. Returns (squared, kept as drawn by the LiDAR)."""
    squared = refused = 0
    for m in mine:
        px = m["px"]
        if px.area < BIG_TILES:
            continue
        c = px.centroid
        a = fold(wall_angle(px))
        turned = affinity.rotate(px, -a, origin=c)
        sq = squared_walls(turned)
        if sq is turned:
            continue
        reg = m.get("reg", (0.0, 0.0))
        drawn = affinity.translate(m["bng"], *reg)
        sq_bng = affinity.translate(px_to_bng(proj, affinity.rotate(sq, a, origin=c)), *reg)
        minx, miny, maxx, maxy = unary_union([drawn, sq_bng]).bounds
        window = (minx - 1, miny - 1, maxx + 1, maxy + 1)
        before, after = lidar_iou(ground, drawn, window), lidar_iou(ground, sq_bng, window)
        m["square"] = after >= before - SQUARE_IOU_LOSS
        squared += m["square"]
        refused += not m["square"]
    return squared, refused


def lidar_buildings(ground, mine, keep, proj, W, H, rail=None):
    """Step 3: LiDAR buildings no outline covers - OSM's gaps. Those that pass
    (big enough, clear of the outlines, in the cloud, not a tree) come back as
    new records for `mine`, placed by the registration and KnoxMap's moves of
    the outlines round them (so those must be registered first); the rest,
    with why, as features for lidar_only.geojson. (new, only)"""
    from scipy import ndimage
    from shapely.strtree import STRtree
    e0, n0, e1, n1 = ground.box
    cover_img = Image.new("1", (e1 - e0, n1 - n0), 0)
    draw = ImageDraw.Draw(cover_img)

    def draw_bng(poly):
        poly = poly.buffer(1.0)
        for g in getattr(poly, "geoms", [poly]):
            draw.polygon([(x - e0, n1 - y) for x, y in g.exterior.coords], fill=1)
    for m in mine:
        draw_bng(affinity.translate(m["bng"], *m["reg"]))
    for ft in keep:
        if ft["geometry"]["type"] == "Polygon":
            draw_bng(transform(TO_BNG, shape(ft["geometry"])))
    covered = np.array(cover_img, dtype=bool)
    lab, n_lab = ndimage.label(ground.built & ~covered)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n_lab + 1))
    outlines = [affinity.translate(m["bng"], *m["reg"]) for m in mine] + \
        [transform(TO_BNG, shape(ft["geometry"])) for ft in keep if ft["geometry"]["type"] == "Polygon"]
    out_tree = STRtree(outlines)
    cent = np.array([m["bng"].centroid.coords[0] for m in mine])
    heights = {}

    def height_of(i):
        """95th-percentile height above ground over a registered outline."""
        if i not in heights:
            poly_ = affinity.translate(mine[i]["bng"], *mine[i]["reg"])
            minx, miny, maxx, maxy = poly_.bounds
            gx = np.arange(int(minx), int(maxx) + 1) + 0.5
            gy = np.arange(int(miny), int(maxy) + 1) + 0.5
            X, Y = np.meshgrid(gx, gy)
            inside = shapely.contains_xy(poly_, X, Y)
            r_ = np.clip((n1 - Y[inside]).astype(int), 0, ground.ndsm.shape[0] - 1)
            c_ = np.clip((X[inside] - e0).astype(int), 0, ground.ndsm.shape[1] - 1)
            v = ground.ndsm[r_, c_]
            v = v[~np.isnan(v)]
            heights[i] = float(np.percentile(v, 95)) if len(v) else float("nan")
        return heights[i]
    new, only = [], []
    for k, sl_ in enumerate(ndimage.find_objects(lab), 1):
        if sizes[k - 1] < LIDAR_ONLY_M2:
            continue
        here = lab[sl_] == k
        rr, cc = np.nonzero(here)
        rr, cc = rr + sl_[0].start, cc + sl_[1].start
        poly = unary_union([box(e0 + c, n1 - r - 1, e0 + c + 1, n1 - r) for r, c in zip(rr, cc)])
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        lon, lat = TO_WGS(poly.centroid.x, poly.centroid.y)
        cx_, cy_ = proj.to_px(lat, lon)
        if not (0 <= cx_ < W and 0 <= cy_ < H):
            continue
        clear = min((poly.distance(outlines[j]) for j in out_tree.query(poly.buffer(NEW_CLEAR_M))),
                    default=99.0)
        in_cloud = ground.cloud[rr, cc].mean() if len(rr) else 0.0
        # A tree's crown the cloud classed as building: half vegetation, or
        # standing well above the houses round it.
        share = ground.up_b[rr, cc].sum() / max(1, ground.up_n[rr, cc].sum())
        tall = float(np.nanpercentile(ground.ndsm[rr, cc], 95)) if len(rr) else 0.0
        cx_b, cy_b = poly.centroid.x, poly.centroid.y
        near_h = [height_of(i) for i in range(len(cent))
                  if math.hypot(cent[i][0] - cx_b, cent[i][1] - cy_b) <= TREE_AROUND_M]
        near_h = [v for v in near_h if v == v]
        above = (tall - float(np.median(near_h)) >= TREE_ABOVE_M) if len(near_h) >= 3 else tall >= TREE_ALONE_M
        why = [w for w, bad in (("small", sizes[k - 1] < NEW_MIN_M2),
                                ("on an outline", clear <= NEW_CLEAR_M),
                                ("outside the cloud", in_cloud < NEW_CLOUD),
                                ("a tree", share < TREE_BUILDING_SHARE or above),
                                # Parked carriages, canopies, kit (rail_kit; trains later).
                                ("on the railway", rail is not None and rail.lidar_on(poly))) if bad]
        if not why and len(cent):
            # The LiDAR's frame back into OSM's, by the registration of the
            # buildings around it; and KnoxMap's move of them, likewise.
            d = np.hypot(*(cent - np.array(poly.centroid.coords[0])).T)
            wts = 1.0 / (d * d + 25.0)
            near = np.argsort(d)[:8]
            reg = tuple(float(v) for v in
                        (np.array([mine[i]["reg"] for i in near]) * wts[near, None]).sum(0) / wts[near].sum())
            fits = rectify(poly)
            if not fits:
                only.append({"type": "Feature", "properties": {"m2": int(sizes[k - 1]), "why": "no rectangles fit"},
                             "geometry": mapping(transform(TO_WGS, poly))})
                continue
            mv = np.array([[mine[i]["moved"].centroid.x - mine[i]["px"].centroid.x,
                            mine[i]["moved"].centroid.y - mine[i]["px"].centroid.y] for i in near])
            dmove = (mv * wts[near, None]).sum(0) / wts[near].sum()
            for outline in fits:
                bng = affinity.translate(outline, -reg[0], -reg[1])
                px_ = bng_to_px(proj, bng)
                oid = NEW_ID + len(new)
                area_m2 = float(bng.area)
                ft = {"type": "Feature", "geometry": mapping(transform(TO_WGS, bng)),
                      "properties": {"osm_id": oid, "building": "house" if area_m2 <= 250 else "yes",
                                     "source": "lidar"}}
                new.append({"ft": ft, "id": str(oid), "bng": bng, "px": px_,
                            "moved": affinity.translate(px_, *dmove), "reg": reg, "new": True})
            continue
        only.append({"type": "Feature", "properties": {"m2": int(sizes[k - 1]), "why": ", ".join(why)},
                     "geometry": mapping(transform(TO_WGS, poly))})
    return new, only


def outline_cover(ground, poly):
    """How much of an outline (BNG, registered) the LiDAR sees as building."""
    bounds = poly.bounds
    rm = raster_bng(poly, bounds)
    return float((rm & ground.window(bounds)).sum() / max(1, rm.sum()))


class Placement:
    """The game's tiles as the blocks are placed on them, biggest first, and
    the tallies the report prints."""

    def __init__(self, H, W):
        self.owner = np.zeros((H, W), np.int32)        # 1-based index into `mine`
        self.block_owner = np.zeros((H, W), np.int32)  # 1-based block index
        self.placed_real = {}       # block index -> its real outline (tiles)
        self.block_of = {}          # index into `mine` -> its block
        self.moves = []             # how far each building moved, tiles
        self.along_street = self.nudged = self.kept_apart = self.given_way = 0
        self.trimmed = self.squared_off = self.corner_blocks = 0
        self.trim_by = {}           # (road name, half width) -> tiles trimmed for it
        self.pulled_in = []         # (osm id, (dx, dy)): long buildings laid straight (straighten_pass)
        self.long_todo = []         # ...the ones to try, noted by place_block
        self.off_street = []        # (block, (dx, dy), squares): moved off the painted street (street_pass)


def front_street(streets, all_roads, whole_real, centre):
    """The street a block is carried along (step 4), or None: one of ours, the
    road nearest it - and only within the street's length (past its end the
    carry guesses)."""
    if not streets:
        return None

    def within(st):
        s_, _o = st.locate(centre.x, centre.y)
        # And only where the game street runs the way the real one does:
        # where the straightening reversed a stretch (London Road off the
        # north end of the High Street) the carry mirrored the frontage onto
        # the far side of the street.
        return -END_REACH <= s_ <= st.Lr + END_REACH and abs(st.turn_at(centre.x, centre.y)) <= 90
    ours = sorted((st.real.distance(whole_real), k) for k, st in enumerate(streets) if within(st))
    nearest = min((r for r in all_roads if r[0]), key=lambda r: r[1].distance(whole_real))
    if ours and ours[0][0] <= FRONT_NEAR_M:
        return streets[ours[0][1]]
    return next((streets[k] for d, k in ours if streets[k].name == nearest[0] and d <= FRONT_M), None)


def building_streets(streets, all_roads, pxs, street):
    """Step 4, a block round a corner: each building carried along its own
    street (front_street of it alone), one with none along the block's - the
    street carrying most of the block's area, else `street`. A block fronts
    two streets at a corner; carried whole along one, its other frontage
    landed on the other street (Alton's High Street terrace, carried along
    Turk Street, was cut back to the kerb: LOG 2026-10-14). (each building's
    street, the block's street)"""
    own = [front_street(streets, all_roads, p, p.centroid) for p in pxs]
    area: dict = {}
    for st, p in zip(own, pxs):
        if st is not None:
            area[id(st)] = area.get(id(st), 0.0) + p.area
    if area:
        best = max(area, key=area.get)
        street = next(st for st in own if st is not None and id(st) == best)
    return [st or street for st in own], street


def carried(street, pxs, raw_gaps, each=None):
    """Step 4, a block carried along its street, building by building
    (Street.fit; `each`, a street per building, for a block round a corner).
    Returns the placed outlines, the slivers, each building's turn, where its
    frame's centre went and that centre (the point of the real outline the
    fit turned about)."""
    sts = each or [street] * len(pxs)
    fits = [st.fit(p) for st, p in zip(sts, pxs)]
    if ANCHOR_FRONT:
        fits = one_line(fits, sts)
    gaps = [street.carry(g) for g in raw_gaps if g.geom_type == "Polygon"]
    return [f[0] for f in fits], gaps, [f[1] for f in fits], [f[2] for f in fits], [f[3] for f in fits]


def one_line(fits, sts):
    """FRONT_ANCHOR, a block carried along its street: its houses' fronts on each side of
    each street (a run) put on one line - the run's median distance from the street in
    the game - where they are within LINE_SNAP of it (a house really set back stays)."""
    groups: dict = {}
    for k, (fit, st) in enumerate(zip(fits, sts)):
        s_, off = st.locate(*fit[3])
        groups.setdefault((id(st), off - st.mid(s_) > 0), []).append(k)
    fits = list(fits)
    for ks in groups.values():
        if len(ks) < 2:
            continue
        st = sts[ks[0]]
        at = []
        for k in ks:
            qa = fits[k][2]
            sg = st.game.project(shapely.Point(qa))
            _t, n = Street._dir(st.game, sg)
            foot = st.game.interpolate(sg)
            at.append((k, n, float((np.array(qa) - (foot.x, foot.y)) @ n)))
        line = float(np.median([o for _k, _n, o in at]))
        for k, n, o in at:
            if 0 < abs(line - o) <= LINE_SNAP:
                dx, dy = (line - o) * n
                out, turn, qa, a = fits[k]
                fits[k] = (affinity.translate(out, dx, dy), turn, (qa[0] + dx, qa[1] + dy), a)
    return fits


def moved_rigidly(pxs, moved, raw_gaps, whole_real, road=None):
    """Step 4, any other block: turned square to the tiles about its centre,
    each building then square on its own (two houses joined into one block
    can stand at different angles; the block's mean would leave both askew),
    then shifted by the area-weighted mean of KnoxMap's moves of them
    (`moved`, its outlines). With FRONT_ANCHOR and `road` (the real centre line
    of the road nearest the block), turned about the middle of the block's edge
    facing it instead, each building about its own, and the front point then
    moved by the area-weighted mean of the real buildings' moves - so the backs
    take the turn. Returns what carried() does."""
    front = ANCHOR_FRONT and road is not None
    centre = shapely.Point(front_point(whole_real, road)) if front else whole_real.centroid
    turn = -fold(wall_angle(whole_real))
    turned = [affinity.rotate(p, turn, origin=centre) for p in pxs]
    road_t = affinity.rotate(road, turn, origin=centre) if front else None
    turns = []
    for k, t in enumerate(turned):
        off = fold(wall_angle(t))
        if abs(off) > 0.5:
            turned[k] = affinity.rotate(t, -off, origin=front_point(t, road_t) if front else t.centroid)
        turns.append(turn - off if abs(off) > 0.5 else turn)
    areas = np.array([p.area for p in pxs])
    if front:
        d = np.array([[mv.centroid.x - p.centroid.x, mv.centroid.y - p.centroid.y] for mv, p in zip(moved, pxs)])
    else:
        d = np.array([[mv.centroid.x - t.centroid.x, mv.centroid.y - t.centroid.y]
                      for mv, t in zip(moved, turned)])
    tx, ty = (d * areas[:, None]).sum(0) / areas.sum()
    placed = [affinity.translate(p, tx, ty) for p in turned]
    gaps = [affinity.translate(affinity.rotate(g, turn, origin=centre), tx, ty) for g in raw_gaps]
    where = [(t.centroid.x + tx, t.centroid.y + ty) for t in turned]
    return placed, gaps, turns, where, [(p.centroid.x, p.centroid.y) for p in pxs]


EDGE_STEPS = 10        # onto_tile_edges: shifts of a tenth of a tile, under half a tile each way
EDGE_FINE = 20         # ...the outline sampled this many times a tile each way (a multiple of
                       # EDGE_STEPS: 20 picks the old polygon search's shift but in near-ties,
                       # a few times faster)


def onto_tile_edges(placed, gaps):
    """Under half a tile more, so the block's walls sit on tile edges: the
    (fx, fy), in tenths of a tile, that best lays its outline on the tiles.
    The outline is sampled once, EDGE_FINE times finer than the tiles; each
    shift moves the samples, and each tile's share of them is its cover; a
    shift's misfit is the cover outside the tiles it would get (by their
    middles, as tile_mask) plus the uncovered part of those - the area between
    outline and tiles - with no polygons built (it was 42% of the block
    step's time)."""
    u = unary_union(placed + gaps)
    if u.is_empty:
        return 0.0, 0.0
    F = EDGE_FINE
    minx, miny, maxx, maxy = u.bounds
    x0, y0 = int(math.floor(minx)) - 1, int(math.floor(miny)) - 1
    w, h = int(math.ceil(maxx)) - x0 + 1, int(math.ceil(maxy)) - y0 + 1
    xs, ys = np.meshgrid(x0 + (np.arange(w * F) + 0.5) / F, y0 + (np.arange(h * F) + 0.5) / F)
    fine = shapely.contains_xy(u, xs, ys).astype(np.float32)
    mx, my = np.meshgrid(x0 + np.arange(w) + 0.5, y0 + np.arange(h) + 0.5)
    best = None
    S, step = EDGE_STEPS, EDGE_FINE // EDGE_STEPS
    for ix in range(-S // 2, S - S // 2):
        for iy in range(-S // 2, S - S // 2):
            kx, ky = ix * step, iy * step
            moved = np.zeros_like(fine)                  # the outline moved kx, ky samples
            sy0, sy1 = max(0, -ky), fine.shape[0] - max(0, ky)
            sx0, sx1 = max(0, -kx), fine.shape[1] - max(0, kx)
            moved[sy0 + ky:sy1 + ky, sx0 + kx:sx1 + kx] = fine[sy0:sy1, sx0:sx1]
            cover = moved.reshape(h, F, w, F).mean(axis=(1, 3))
            mk = shapely.contains_xy(u, mx - ix / S, my - iy / S)      # a tile is the building's by its middle
            if not mk.any():
                continue
            err = float((1.0 - cover)[mk].sum() + cover[~mk].sum())
            if best is None or err < best[0] - 1e-6:
                best = (err, ix / S, iy / S)
    return (best[1], best[2]) if best else (0.0, 0.0)


def block_tiles(placed, gaps, close, W, H):
    """Step 6's tiles: each to the building whose outline holds its middle,
    the slivers' to the nearest building, and a block over the map's edge
    keeps the part inside. `close`: first close the hairline gaps between
    buildings fitted one by one. (own, x0, y0), own 1-based into the block's
    buildings, or None when nothing of it is on the map."""
    whole = unary_union(placed + gaps)
    if close:
        whole = closing(whole, CLOSE)
    minx, miny, maxx, maxy = whole.bounds
    x0, y0 = int(math.floor(minx)), int(math.floor(miny))
    w, h = int(math.ceil(maxx)) - x0, int(math.ceil(maxy)) - y0
    own = tile_mask(placed, x0, y0, w, h)
    fill = (tile_mask([whole], x0, y0, w, h) > 0) & (own == 0)
    for y, x in zip(*np.nonzero(fill)):
        pt = shapely.Point(x0 + x + 0.5, y0 + y + 0.5)
        own[y, x] = 1 + int(np.argmin([p.distance(pt) for p in placed]))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    own = own[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    if not (own > 0).any():
        return None
    return own, cx0, cy0


def roads_to_clear(road_list, whole_real, here_game, street, station=False):
    """Step 5, the roads a block must keep off and how far: off the
    carriageway always, off the pavement too where it really stands that far
    back. A road the block is carried along already has it at the right
    distance. [(road, distance)]"""
    need = []
    for r in road_list:
        # A station's buildings stand on the railway's land as they really do.
        if station and r.get("rail"):
            continue
        # Under a building on purpose (layer < 0: an arch, a ramp) - but only
        # the building it really passes under may stand over it: one tag on a
        # car park's whole loop let every building near it stand on it, and
        # left its way out a tile wide (user, 2026-10-11).
        if r["under"] and r["real"].intersects(whole_real):
            continue
        c_real = r["real"].distance(whole_real)
        # Near it really, or brought near it by the straightening.
        if c_real > ROAD_NEAR and r["game"].distance(here_game) > ROAD_NEAR:
            continue
        if c_real <= REPEL:
            # Drawn along its wall in OSM: off the carriageway, no more.
            c_req = r["half"] if not r["minor"] else 0.0
        elif street is not None and r["name"] == street.name:
            c_req = r["half"] + REPEL
        elif r["minor"]:
            c_req = min(c_real, r["half"] + REPEL)
        else:
            c_req = max(r["half"] + REPEL, min(c_real, r["half"] + r["pave"]))
        if c_req > 0:
            need.append((r, c_req))
    return need


def beside_neighbours(at, whole_real):
    """Step 5: the tiles next to blocks already placed that really stand
    GAP_KEEP or more (and under 3) from this one, which keeps a tile apart."""
    apart = np.zeros(at.owner.shape, bool)
    for bj, other in at.placed_real.items():
        gap = other.distance(whole_real)
        if GAP_KEEP <= gap < 3.0:
            theirs = at.block_owner == bj + 1
            grown = theirs.copy()
            grown[1:, :] |= theirs[:-1, :]
            grown[:-1, :] |= theirs[1:, :]
            grown[:, 1:] |= theirs[:, :-1]
            grown[:, :-1] |= theirs[:, 1:]
            apart |= grown
    return apart


def setback_street(road_list, whole_real):
    """Step 5, a block moved rigidly keeps its real distance from its street:
    the named road nearest it really (within SETBACK_NEAR), and the distance
    to draw it at in the game; None if there is none."""
    named = [(r["real"].distance(whole_real), r) for r in road_list
             if r["name"] and not r["under"] and r["real"].distance(whole_real) <= SETBACK_NEAR]
    if not named:
        return None
    c_real, fr = min(named, key=lambda t: t[0])
    return fr, max(c_real, fr["half"] + REPEL)


def nudge(at, mk, x0, y0, shape_, need, apart, front, street, centre):
    """Step 5, the smallest whole-tile shift (up to NUDGE) that keeps the
    block (mk, its corner at x0, y0; shape_ its tiles as a shape) off the
    roads it needs to clear, off tiles already taken, a tile from neighbours
    it is really apart from, and at its setback from `front` - a block on its
    street slides only along it (off a side road at a corner), so its
    frontage stays on the line, and steps back at most STEP_BACK. (ox, oy),
    or None when it does not fit on the map at all."""
    H, W = at.owner.shape
    h, w = mk.shape

    def cost(ox, oy):
        ax0, ay0 = x0 + ox, y0 + oy
        if ax0 < 0 or ay0 < 0 or ax0 + w > W or ay0 + h > H:
            return None
        sl = (slice(ay0, ay0 + h), slice(ax0, ax0 + w))
        moved = affinity.translate(shape_, ox, oy)
        short = sum(max(0.0, c_req - moved.distance(r["game"])) for r, c_req in need)
        off = abs(moved.distance(front[0]["game"]) - front[1]) if front else 0.0
        return (short, int((at.owner[sl] > 0)[mk].sum()), int(apart[sl][mk].sum()), off)
    here = cost(0, 0)
    if here is None:
        return None
    ox = oy = 0
    along_x = False
    if street is not None:
        gx_, gy_ = street.map(centre.x, centre.y)
        tdir, _n = Street._dir(street.game, max(0.0, min(street.Lg, street.game.project(shapely.Point(gx_, gy_)))))
        along_x = abs(tdir[0]) >= abs(tdir[1])
    if here[0] > 0.01 or here[1] or here[2] or here[3] > 1.0:
        reach = NUDGE if (here[0] > 0.01 or here[3] > 1.0) else 1
        best = (here[0] * 10 + here[1] * 2 + here[2] + here[3], 0, 0)
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                across = (dy if along_x else dx) if street is not None else 0
                if abs(across) > STEP_BACK:
                    continue
                c = cost(dx, dy)
                if c is None:
                    continue
                score = c[0] * 10 + c[1] * 2 + c[2] + c[3] + (abs(dx) + abs(dy)) * 0.5 + abs(across) * 1.5
                if score < best[0]:
                    best = (score, dx, dy)
        _s, ox, oy = best
        if ox or oy:
            at.nudged += 1
        if here[2] and not cost(ox, oy)[2]:
            at.kept_apart += 1
    return ox, oy


def long_size(real):
    """(length, width) of a building's real outline (tiles, a metre each): the
    tightest rectangle's long side, and the area over it (an L or a ragged
    outline is narrower than its rectangle)."""
    if real.is_empty:
        return 0.0, 0.0
    r = real.minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return 0.0, 0.0
    xs, ys = r.exterior.coords.xy
    a = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
    b = math.hypot(xs[2] - xs[1], ys[2] - ys[1])
    long_ = max(a, b)
    return long_, real.area / max(long_, 1e-6)


def is_long(real) -> bool:
    """A range or a row: at least LONG_M long and LONG_ASPECT times its width,
    and nearly a rectangle (LONG_FILL)."""
    long_, wide = long_size(real)
    rect = real.minimum_rotated_rectangle
    return (long_ >= LONG_M and long_ / max(wide, 1e-6) >= LONG_ASPECT
            and rect.area > 0 and real.area / rect.area >= LONG_FILL)


def on_roads(own_k, need, ax0, ay0):
    """The tiles of own_k (its corner at ax0, ay0) on a carriageway trim_off_roads would cut."""
    h, w = own_k.shape
    xs, ys = np.meshgrid(np.arange(ax0, ax0 + w) + 0.5, np.arange(ay0, ay0 + h) + 0.5)
    on = np.zeros(own_k.shape, bool)
    for r, _c_req in need:
        if not r["minor"]:
            on |= shapely.contains_xy(r["game"].buffer(r["half"], cap_style=2), xs, ys) & own_k
    return on


def side_of(line, pt) -> int:
    """Which side of a road's centre line a point is: +1 or -1; 0 on it, or
    past either end of it (no side)."""
    d = line.project(pt)
    if d <= 0.0 or d >= line.length:
        return 0
    a, b = line.interpolate(max(0.0, d - 0.5)), line.interpolate(min(line.length, d + 0.5))
    cross = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x)
    return (cross > 0) - (cross < 0)


def not_homes(m) -> bool:
    """A building straighten_long may lay straight: not a home, a row of homes,
    or one that may be either (see LONG_HOMES)."""
    tags = m["ft"]["properties"]
    try:
        if int(str(m["id"]).split("u")[0].split("p")[0]) >= NEW_ID:
            return False
    except ValueError:
        pass
    b = (tags.get("building") or "yes").strip().lower()
    if b in LONG_HOMES:
        return False
    if b == "yes" and not any(tags.get(k) for k in ("name", "amenity", "shop", "office", "leisure", "tourism",
                                                     "craft", "healthcare", "religion")):
        return False
    return True


def straighten_long(at, real, placed, cx, cy, need):
    """A long building as one straight rectangle on the grid, its real length
    and width, moved from where it was laid (its middle at cx, cy) to the
    nearest place (up to LONG_REACH) that is clear of every road in `need` by
    what it needs and on the real building's side of each road near it, and
    neither on nor beside a tile another building has (`at.owner`, every
    block placed: straighten_pass runs last). It lies the way it was laid
    (`placed`, squared); the other way round too for one near 45 degrees, at
    LONG_TURN_COST. (mask, x0, y0, (dx, dy) its middle moved, turned) or None."""
    H, W = at.owner.shape
    long_, wide = long_size(real)
    L, Wd = max(MIN_SIDE, int(round(long_))), max(MIN_SIDE, int(round(wide)))
    off = fold(wall_angle(placed))
    sq = affinity.rotate(placed, -off, origin=placed.centroid)
    bx0_, by0_, bx1_, by1_ = sq.bounds
    laid_x = (bx1_ - bx0_) >= (by1_ - by0_)
    ways = [((L, Wd) if laid_x else (Wd, L), 0, False)]
    if abs(off) >= 30:
        ways.append(((Wd, L) if laid_x else (L, Wd), LONG_TURN_COST, True))
    tries = sorted(((dx, dy) for dx in range(-LONG_REACH, LONG_REACH + 1) for dy in range(-LONG_REACH, LONG_REACH + 1)),
                   key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), t))
    # the side of each road the real building stands on (a staircase carried
    # across a road may have its middle on the wrong one)
    sides = [(r, side_of(r["real"], real.centroid)) for r, _c in need
             if not r["minor"] and r["real"].distance(real) < SIDE_NEAR]
    sides = [(r, s_) for r, s_ in sides if s_]
    held = at.owner > 0
    best = None
    for (w, h), extra, turned in ways:
        bx0, by0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        mask = np.ones((h, w), bool)
        shape0 = shapely.box(bx0, by0, bx0 + w, by0 + h)
        for dx, dy in tries:
            if best is not None and abs(dx) + abs(dy) + extra >= best[0]:
                break
            x0, y0 = bx0 + dx, by0 + dy
            if x0 < 1 or y0 < 1 or x0 + w > W - 1 or y0 + h > H - 1:
                continue
            if held[y0 - 1:y0 + h + 1, x0 - 1:x0 + w + 1].any():
                continue                            # on or beside another building
            moved = affinity.translate(shape0, dx, dy)
            if any(moved.distance(r["game"]) < c_req - 0.01 for r, c_req in need if not r["minor"]):
                continue
            if on_roads(mask, need, x0, y0).any():
                continue
            mid = shapely.Point(x0 + w / 2, y0 + h / 2)
            if any(side_of(r["game"], mid) == -s_ for r, s_ in sides):
                continue                            # over a road: another block
            best = (abs(dx) + abs(dy) + extra, mask, x0, y0,
                    (int(round(x0 + w / 2 - cx)), int(round(y0 + h / 2 - cy))), turned)
            break
    return None if best is None else best[1:]


def straighten_pass(at, mine, show=False):
    """After every block has its tiles: each long building place_block noted
    (`at.long_todo`, biggest first) lifted off its tiles and laid straight
    (straighten_long); if there is no place for it, it stays as it was. Its
    record's frame follows it: its middle, the rest of its turn to the grid,
    and a quarter turn when it was laid the other way round."""
    for t in sorted(at.long_todo, key=lambda t: -t["real"].area):
        i = t["i"]
        mine_tiles = at.owner == i + 1
        if not mine_tiles.any():
            continue
        ys_, xs_ = np.nonzero(mine_tiles)
        cx, cy = xs_.mean() + 0.5, ys_.mean() + 0.5
        bi = int(at.block_owner[ys_[0], xs_[0]]) - 1
        at.owner[mine_tiles] = 0
        got = straighten_long(at, t["real"], t["placed"], cx, cy, t["need"])
        if got is None:
            at.owner[mine_tiles] = i + 1
            continue
        mask, x0, y0, d, turned = got
        at.block_owner[mine_tiles] = 0
        at.owner[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = i + 1
        at.block_owner[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = bi + 1
        gx, gy = x0 + mask.shape[1] / 2, y0 + mask.shape[0] / 2
        turn = t["turn"] - fold(wall_angle(t["placed"])) + (90.0 if turned else 0.0)
        c, (rx, ry) = t["c"], t["reg"]
        mine[i]["frame"] = [turn, gx - c[0] - rx, gy - c[1] - ry, gx, gy]
        mine[i]["how"] = dict(mine[i].get("how") or {}, laid_straight=list(d), turned=bool(turned))
        at.pulled_in.append((mine[i]["id"], d))
        if show:
            print(f"    laid straight in its block: {mine[i]['ft']['properties'].get('name') or mine[i]['id']} "
                  f"{mask.shape[1]}x{mask.shape[0]} at {x0},{y0} (moved {d}{', turned' if turned else ''})")


def painted_street(map_dir, name, shape_):
    """The street as the road graph's hand-over painted it, before this step:
    carriageway, parking bays, verge and pavement (<map>_roadgraph_tiles.npz,
    <map>_roadside.npz). None on a map with no road graph."""
    tiles = os.path.join(map_dir, f"{name}_roadgraph_tiles.npz")
    side = os.path.join(map_dir, f"{name}_roadside.npz")
    if not (os.path.exists(tiles) and os.path.exists(side)):
        return None
    st = np.load(tiles)["owner"] != 0
    rs = np.load(side)
    st = st | rs["parking"] | rs["verge"] | rs["pavement"]
    return st if st.shape == shape_ else None


def carriageways(road_list, W, H):
    """Every road's and railway's carriageway as step 5 keeps blocks off it
    (not service lanes, nor those meant to run under a building)."""
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    out = np.zeros((H, W), bool)
    for r in road_list:
        if r["minor"] or r["under"]:
            continue
        x0, y0, x1, y1 = r["game"].buffer(r["half"]).bounds
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(math.ceil(x1)) + 1), min(H, int(math.ceil(y1)) + 1)
        if x1 > x0 and y1 > y0:
            sl = (slice(y0, y1), slice(x0, x1))
            out[sl] |= shapely.contains_xy(r["game"].buffer(r["half"], cap_style=2), xs[sl], ys[sl])
    return out


def street_pass(at, mine, street, carr, show=False):
    """After straighten_pass: each block with squares on the painted `street`
    lifted and put back at the nearest whole-tile shift (up to STREET_REACH)
    that has none, is on no other building's tiles nor newly beside one, and
    is on no more of `carr` (roads and railways, step 5's) than before. A
    block whose buildings stand apart on the tiles moves piece by piece (the
    touching ones together), so a row keeps its frontage where it is clear.
    A piece with nowhere clear stays as it was. Its buildings' frames follow it."""
    from scipy import ndimage as ndi
    H, W = at.owner.shape
    R = STREET_REACH
    tries = sorted(((dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1) if dx or dy),
                   key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), t))
    blocks = np.unique(at.block_owner[street & (at.block_owner > 0)])
    sizes = np.bincount(at.block_owner.ravel())
    tiles_of = np.bincount(at.owner.ravel())        # each building's tiles (a move keeps them)
    for b in sorted(blocks, key=lambda b: -sizes[b]):
        ys_, xs_ = np.nonzero(at.block_owner == b)
        lab, n_lab = ndi.label(at.block_owner[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1] == b)
        for p in range(1, n_lab + 1):
            py, px = np.nonzero(lab == p)
            y0, x0 = ys_.min() + py.min(), xs_.min() + px.min()
            mk = lab[py.min():py.max() + 1, px.min():px.max() + 1] == p
            h, w = mk.shape
            n_st = int(street[y0:y0 + h, x0:x0 + w][mk].sum())
            if not n_st:
                continue
            ids = at.owner[y0:y0 + h, x0:x0 + w][mk]
            if (np.bincount(ids, minlength=len(tiles_of))[ids] != tiles_of[ids]).any():
                continue                            # a building with tiles in another piece: not torn
            got = clear_shift(at, street, carr, mk, x0, y0, R, tries)
            if got is None:
                continue
            dx, dy = got
            at.owner[y0:y0 + h, x0:x0 + w][mk] = 0
            at.block_owner[y0:y0 + h, x0:x0 + w][mk] = 0
            sl = (slice(y0 + dy, y0 + dy + h), slice(x0 + dx, x0 + dx + w))
            at.owner[sl][mk] = ids
            at.block_owner[sl][mk] = b
            for i in np.unique(ids) - 1:
                fr = mine[i]["frame"]
                mine[i]["frame"] = [fr[0], fr[1] + dx, fr[2] + dy, fr[3] + dx, fr[4] + dy]
                mine[i]["how"] = dict(mine[i].get("how") or {}, off_street=[int(dx), int(dy)])
            at.off_street.append((int(b) - 1, (dx, dy), n_st))
            if show:
                nm_ = mine[int(ids[0]) - 1]["ft"]["properties"].get("name") or mine[int(ids[0]) - 1]["id"]
                print(f"    off the street: block {int(b) - 1} ({nm_}) {n_st} squares, moved {dx},{dy}")


def clear_shift(at, street, carr, mk, x0, y0, R, tries):
    """street_pass for one piece (mk, its corner at x0, y0): the first shift
    in `tries` that takes it off `street` onto free ground, (dx, dy), or None."""
    H, W = at.owner.shape
    h, w = mk.shape
    here = (slice(y0, y0 + h), slice(x0, x0 + w))
    n_carr = int(carr[here][mk].sum())
    # a window round the piece, R + 1 wider each way
    wy0, wx0 = max(0, y0 - R - 1), max(0, x0 - R - 1)
    win = (slice(wy0, min(H, y0 + h + R + 1)), slice(wx0, min(W, x0 + w + R + 1)))
    this = np.zeros(at.owner.shape, bool)
    this[here] = mk
    others = (at.owner[win] > 0) & ~this[win]
    near = others.copy()
    near[1:, :] |= others[:-1, :]
    near[:-1, :] |= others[1:, :]
    near[:, 1:] |= others[:, :-1]
    near[:, :-1] |= others[:, 1:]
    touched = bool((near & this[win]).any())
    for dx, dy in tries:
        ax0, ay0 = x0 + dx, y0 + dy
        if ax0 < 0 or ay0 < 0 or ax0 + w > W or ay0 + h > H:
            continue
        sl = (slice(ay0, ay0 + h), slice(ax0, ax0 + w))
        if (street[sl] & mk).any() or int((carr[sl] & mk).sum()) > n_carr:
            continue
        wsl = (slice(ay0 - wy0, ay0 - wy0 + h), slice(ax0 - wx0, ax0 - wx0 + w))
        if (others[wsl] & mk).any():
            continue
        if not touched and (near[wsl] & mk).any():
            continue                            # beside a building it stood apart from
        return dx, dy
    return None


def trim_off_roads(own, need, ax0, ay0):
    """Step 5, whatever of the block (own, its corner now at ax0, ay0) still
    stands on a carriageway comes off `own`: a building a little short beats
    one standing in the road. A service lane is never cut back for.
    [(road, distance it needed, tiles taken off)]"""
    if not need:
        return []
    h, w = own.shape
    mk = own > 0
    xs, ys = np.meshgrid(np.arange(ax0, ax0 + w) + 0.5, np.arange(ay0, ay0 + h) + 0.5)
    out = []
    for r, c_req in need:
        if r["minor"]:
            continue
        on = shapely.contains_xy(r["game"].buffer(r["half"], cap_style=2), xs, ys) & mk
        if on.any():
            out.append((r, c_req, int(on.sum())))
            own[on] = 0
            mk = own > 0
    return out


def square_up(own, n, rect_tiles, rect_count):
    """Step 6: each of the block's n buildings in `own` as rectangles at
    least rect_tiles each way (rect_cover), in place. Returns the tiles
    dropped."""
    dropped = 0
    for k in range(1, n + 1):
        mine_k = own == k
        if mine_k.any():
            kept = rect_cover(mine_k, rect_tiles, rect_count)
            dropped += int((mine_k & ~kept).sum())
            own[mine_k & ~kept] = 0
    return dropped


def place_block(bi, members, mine, gaps_bng, shift, streets, all_roads, road_list, proj, at,
                rect_tiles, rect_count, show):
    """Steps 4-6 for one block (members: indexes into `mine`; gaps_bng: the
    slivers shared out between them; shift: its registration): carried along
    its street or moved rigidly, laid on tile edges, given its tiles, nudged
    off the roads and apart from its neighbours, trimmed, squared up, and its
    tiles claimed on `at` where no earlier block got there first (else it
    gives way). Each building's record gets its frame back to the real one
    (`frame`) and how it was placed (`how`)."""
    H, W = at.owner.shape
    pxs = [mine[i]["px"] for i in members]
    whole_real = unary_union(pxs)
    centre = whole_real.centroid
    raw_gaps = [bng_to_px(proj, g) for g in gaps_bng]
    street = front_street(streets, all_roads, whole_real, centre)
    each = [street] * len(pxs)
    if street is not None:
        at.along_street += 1
        each, street = building_streets(streets, all_roads, pxs, street)
        at.corner_blocks += len({id(st) for st in each}) > 1
        placed, gaps, turns, where, fit_centres = carried(street, pxs, raw_gaps, each)
    else:
        # anchored on the road nearest it, of any kind (a yard's block on its lane): tried
        # on its named street instead (step 5's - a lane behind it is no front), and with the
        # nearest road only where there is none, on Alton's centre it put 1,879-1,914
        # building squares on the street against 1,738, the High Street no better at its
        # line (LOG 2026-10-26 #1)
        near_road = min((r[1] for r in all_roads), key=lambda ln: ln.distance(whole_real), default=None)             if ANCHOR_FRONT else None
        placed, gaps, turns, where, fit_centres = moved_rigidly(
            pxs, [mine[i]["moved"] for i in members], raw_gaps, whole_real, near_road)
    placed = [squared_walls(p) if mine[i].get("square") else p for p, i in zip(placed, members)]
    fx, fy = onto_tile_edges(placed, gaps)
    placed = [affinity.translate(p, fx, fy) for p in placed]
    gaps = [affinity.translate(g, fx, fy) for g in gaps]
    got = block_tiles(placed, gaps, street is not None and len(placed) > 1, W, H)
    if got is None:
        return
    own, x0, y0 = got
    mk = own > 0
    shape_ = mask_shape(mk, x0, y0)
    need = roads_to_clear(road_list, whole_real, shape_, street,
                          station=any(mine[i].get("rail") == "station" for i in members))
    apart = beside_neighbours(at, whole_real)
    front = setback_street(road_list, whole_real) if street is None else None
    shifted = nudge(at, mk, x0, y0, shape_, need, apart, front, street, centre)
    if shifted is None:
        return
    ox, oy = shifted
    # A long building at an angle to the grid, or that a road would cut, is
    # noted, to be laid straight and whole inside its block once every block
    # has its tiles (straighten_pass)
    long_k = {}
    for k, i in enumerate(members, 1):
        own_k = own == k
        if not own_k.any() or not is_long(pxs[k - 1]) or not not_homes(mine[i]):
            continue
        ys_k, xs_k = np.nonzero(own_k)
        askew = own_k.sum() < LONG_STAIR * (np.ptp(xs_k) + 1) * (np.ptp(ys_k) + 1)
        cut = on_roads(own_k, need, x0 + ox, y0 + oy).sum() > LONG_TRIM * own_k.sum()
        if askew or cut:
            long_k[k] = i
    for r, c_req, n in trim_off_roads(own, need, x0 + ox, y0 + oy):
        if show:
            nm_ = mine[members[0]]["ft"]["properties"].get("name") or mine[members[0]]["id"]
            print(f"    trim {n:3d} tiles of block {bi} ({nm_}, {len(members)} bldgs, "
                  f"on {street.name if street else 'rigid'}) for {r['name'] or '(unnamed)'}: real gap "
                  f"{r['real'].distance(whole_real):.1f}, game {affinity.translate(shape_, ox, oy).distance(r['game']):.1f}, "
                  f"needs {c_req:.1f}; shift {ox},{oy}")
        at.trimmed += n
        key = (r["name"] or "(unnamed)", r["half"])
        at.trim_by[key] = at.trim_by.get(key, 0) + n
    at.squared_off += square_up(own, len(members), rect_tiles, rect_count)
    mk = own > 0
    h, w = own.shape
    sl = (slice(y0 + oy, y0 + oy + h), slice(x0 + ox, x0 + ox + w))
    taken = at.owner[sl] > 0
    if (mk & taken).any():
        at.given_way += 1
    for k, i in enumerate(members, 1):
        at.owner[sl][(own == k) & ~taken] = i + 1
        at.block_of[i] = bi
    at.block_owner[sl][mk & ~taken] = bi + 1
    at.placed_real[bi] = whole_real
    # The frame back to each real building, for the LiDAR: tile point g is
    # real point c + rot(-turn)(g - g_c), where c is the building's real
    # centre and g_c where it went; the registration is added on.
    sx, sy = shift
    for k, i in enumerate(members):
        c = shapely.Point(fit_centres[k])
        gx, gy = where[k][0] + fx + ox, where[k][1] + fy + oy
        turn_k = turns[k]
        at.moves.append(math.hypot(gx - c.x, gy - c.y))
        cb = mine[i]["bng"].centroid
        lon0, lat0 = TO_WGS(cb.x, cb.y)
        lon1, lat1 = TO_WGS(cb.x + sx, cb.y + sy)
        r0, r1 = proj.to_px(lat0, lon0), proj.to_px(lat1, lon1)
        mine[i]["frame"] = [turn_k, gx - c.x - (r1[0] - r0[0]), gy - c.y - (r1[1] - r0[1]), gx, gy]
        if k + 1 in long_k:
            # every road within its reach, not only the block's own
            reach = affinity.translate(shape_, ox, oy).buffer(LONG_REACH + long_size(pxs[k])[0])
            at.long_todo.append({"i": i, "real": pxs[k], "placed": placed[k], "turn": turns[k],
                                 "c": (c.x, c.y), "reg": (r1[0] - r0[0], r1[1] - r0[1]),
                                 "need": roads_to_clear(road_list, whole_real, reach, street)})
        mine[i]["how"] = {"moved_by": each[k].name if each[k] is not None else "rigid",
                          "nudge": [int(ox), int(oy)],
                          "gave_way": int((mk & taken & (own == k + 1)).sum())}


def cut_into_lots(owner, n):
    """Step 7: the game builds each building on a rectangle that may not
    overlap another's, so a building whose rectangle would take in a
    neighbour's tiles is cut into pieces whose rectangles do not, biggest
    first. A building's own courtyard (COURT_MIN tiles or more) is kept out
    of its rectangles too, so it is cut into pieces round it: Build fills a
    piece's outline in. Pieces too small to build lose their tiles (`owner`
    is edited). Returns the pieces [(index into mine, piece number, x0, y0,
    mask)], the tiles each kept (1-based), tiles lost, and pieces per
    building."""
    from scipy import ndimage as ndi
    H, W = owner.shape
    pieces = []
    lots = np.zeros((H, W), bool)
    court = np.zeros((H, W), bool)
    lost = 0
    queue = []
    for i in range(n):
        ys_, xs_ = np.nonzero(owner == i + 1)
        if len(ys_):
            y0, x0 = ys_.min(), xs_.min()
            mk = owner[y0:ys_.max() + 1, x0:xs_.max() + 1] == i + 1
            queue.append((i, x0, y0, mk))
            lab, n_lab = ndi.label(_holes(mk))
            for k2 in range(1, n_lab + 1):
                if (lab == k2).sum() >= COURT_MIN:
                    # (not a building standing in it: the review, 2026-10-14)
                    court[y0:y0 + mk.shape[0], x0:x0 + mk.shape[1]] |= (lab == k2) & (owner[y0:y0 + mk.shape[0], x0:x0 + mk.shape[1]] == 0)
    queue.sort(key=lambda q: -q[3].sum())
    count = {}
    while queue:
        i, x0, y0, mk = queue.pop(0)
        h, w = mk.shape
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        if mk.sum() < MIN_TILES:
            lost += int(mk.sum())
            owner[sl][mk] = 0
            continue
        blocked = ((owner[sl] > 0) & (owner[sl] != i + 1)) | lots[sl] | court[sl]
        by0, by1, bx0, bx1 = _clear_box(mk, blocked)
        part = np.zeros_like(mk)
        part[by0:by1, bx0:bx1] = mk[by0:by1, bx0:bx1]
        piece, px0, py0 = _hug(part, x0, y0)
        if piece.sum() < MIN_TILES or min(piece.shape) < MIN_SIDE:
            lost += int(mk.sum())
            owner[sl][mk] = 0
            continue
        k = count.get(i, 0)
        count[i] = k + 1
        pieces.append((i, k, px0, py0, piece))
        lots[py0:py0 + piece.shape[0], px0:px0 + piece.shape[1]] = True
        # What is left of this building becomes more pieces.
        rest = mk.copy()
        rest[py0 - y0:py0 - y0 + piece.shape[0], px0 - x0:px0 - x0 + piece.shape[1]] &= ~piece
        lab, n_lab = ndi.label(rest)
        for k2 in range(1, n_lab + 1):
            part = lab == k2
            rr, cc = np.nonzero(part)
            sub = part[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            queue.append((i, x0 + cc.min(), y0 + rr.min(), sub))
        queue.sort(key=lambda q: -q[3].sum())
    # Tiles no piece kept are free again.
    final = np.zeros((H, W), np.int32)
    for i, k, x0, y0, piece in pieces:
        final[y0:y0 + piece.shape[0], x0:x0 + piece.shape[1]][piece] = i + 1
    return pieces, final, lost, count


def placed_features(keep, mine, pieces, block_of, proj):
    """The buildings file's features: those left to KnoxMap, then each piece
    as an outline on tile edges with what Build and the later steps read
    (`placed`, `block`, `lidar_frame` ...); a building cut for lots is pieces
    `<id>p<k>` with the same parent."""
    out = list(keep)
    for i, k, x0, y0, piece in pieces:
        m = mine[i]
        outline = mask_outline(piece, x0, y0)
        ring = [proj.to_latlon(x, y) for x, y in outline.exterior.coords]
        props = dict(m["ft"]["properties"])
        props.update({"placed": "block", "block": block_of[i], "lidar_frame": m["frame"],
                      "lidar_cover": round(m["cover"], 2),
                      "lidar_shift": [round(v, 1) for v in m["reg"]], **m.get("how", {})})
        holes = int(np.sum(_holes(piece)))
        if holes:
            props["courtyard_tiles"] = holes
        if k:
            props.update({"osm_id": f"{m['id']}p{k}", "parent_osm_id": m["id"], "piece": k})
        out.append({"type": "Feature", "properties": props,
                    "geometry": mapping(Polygon([(lon, lat) for lat, lon in ring]))})
    return out


def clear_trees(map_dir, name, built):
    """Trees off the new ground, in both vegetation maps."""
    for veg_name in (f"{name}_veg.bmp", f"{name}_veg_base.bmp"):
        vp = os.path.join(map_dir, veg_name)
        if os.path.exists(vp):
            veg = np.array(Image.open(vp).convert("RGB"))
            veg[built] = C.VEG_NOTHING
            Image.fromarray(veg).save(vp, format="BMP")


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    show = "--report" in sys.argv
    rect_tiles = int(sys.argv[sys.argv.index("--rect") + 1]) if "--rect" in sys.argv else RECT_TILES
    rect_count = int(sys.argv[sys.argv.index("--rect-count") + 1]) if "--rect-count" in sys.argv else RECT_COUNT
    names = []
    if "--streets" in sys.argv:
        for a in sys.argv[sys.argv.index("--streets") + 1:]:
            if a.startswith("--"):
                break
            names.append(a)
    name = os.path.basename(os.path.normpath(map_dir))
    with open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    names = names or streets_measured(map_dir, info)
    b = info["bbox"]
    proj = Projector.build(b["south"], b["west"], b["north"], b["east"],
                           info["meters_per_tile"], info.get("rotation") or 0.0)
    W, H = proj.width, proj.height
    path = os.path.join(map_dir, f"{name}_buildings.geojson")
    orig = os.path.join(map_dir, f"{name}_buildings.orig.geojson")
    if not os.path.exists(orig):
        shutil.copy(path, orig)
    with open(orig, encoding="utf-8") as f:
        gj = json.load(f)
    with open(os.path.join(area, "buildings.geojson"), encoding="utf-8") as f:
        real = {str(ft["properties"]["osm_id"]): ft for ft in json.load(f)["features"]}
    keep, mine = outlines_to_place(gj, real, proj, W, H, relation_outlines(map_dir, info))
    # Buildings on the railway (rail_kit): its own kept where they stand, a
    # footbridge left out (on the ground it would block the track).
    from buildings import rail_kit
    from generator import osm as _osm
    from generator.renderer import _is_polygon, classify
    rail = rail_kit.RailLand(_osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])),
                             area, classify, _is_polygon)
    for m in mine:
        tags = (real.get(m["id"], {}).get("properties", {}).get("tags") or m["ft"]["properties"])
        m["rail"] = rail.kind(tags, m["bng"])
    footbridges = [m["id"] for m in mine if m["rail"] == "footbridge"]
    mine = [m for m in mine if m["rail"] != "footbridge"]
    global ROW_GRID_DEG, ANCHOR_FRONT
    if os.path.exists(os.path.join(map_dir, FRONT_ANCHOR)):
        ANCHOR_FRONT = True
        print(f"{FRONT_ANCHOR}: buildings squared about their street-facing edge")
    if os.path.exists(os.path.join(map_dir, ROWS_ANY_ANGLE)):
        ROW_GRID_DEG = 90
        print(f"{ROWS_ANY_ANGLE}: rows of homes joined at any angle")
    mine, rows_joined, homes_joined = join_rows(mine, proj)
    for m in mine:
        m.setdefault("rail", None)

    def kept_bridge(ft):
        t = real.get(str(ft["properties"].get("osm_id")), {}).get("properties", {}).get("tags") or ft["properties"]
        return (ft["geometry"]["type"] == "Polygon"
                and rail.kind(t, transform(TO_BNG, shape(ft["geometry"]))) == "footbridge")
    footbridges += [str(ft["properties"].get("osm_id")) for ft in keep if kept_bridge(ft)]
    keep = [ft for ft in keep if not kept_bridge(ft)]
    corners = [proj.to_latlon(x, y) for x, y in ((0, 0), (W, 0), (0, H), (W, H))]
    es, ns = zip(*[TO_BNG(lon, lat) for lat, lon in corners])
    ground = Ground(area, (int(min(es)) - 5, int(min(ns)) - 5, int(max(es)) + 5, int(max(ns)) + 5))

    # --- 1-3: blocks, registration, outline check (real coordinates) -----------------------
    def size(i):
        return mine[i]["bng"].area
    # First the outlines that touch, registered; the LiDAR's buildings OSM
    # lacks placed by them; then the slivers between near outlines read in
    # that registration, and the blocks registered again.
    register_blocks(ground, mine, groups_of(len(mine), list(neighbours_touching(mine)), size))
    new, only = lidar_buildings(ground, mine, keep, proj, W, H, rail if rail else None)
    mine += new
    with open(os.path.join(map_dir, "lidar_only.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": only}, f)
    pairs, slivers, open_gaps = neighbours(mine, ground, lambda i: mine[i]["reg"])
    blocks = groups_of(len(mine), pairs, size)
    shifts, gains = register_blocks(ground, mine, blocks)
    where = {i: bi for bi, members in enumerate(blocks) for i in members}
    block_slivers = {}
    for i, _j, gap in slivers:
        block_slivers.setdefault(where[i], []).append(gap)
    for m in mine:
        m["cover"] = outline_cover(ground, affinity.translate(m["bng"], *m["reg"]))
    squared, refused = square_big(ground, mine, proj)

    # --- 4-6: into the game, block by block, biggest first ---------------------------------
    # the lanes the hand-over made passages (--lanes): run under their building, kept over them
    from regional.passages import made_passages
    streets, all_roads, road_list = load_streets(map_dir, info, proj, names, [m["px"] for m in mine],
                                                 passage_ways=made_passages(map_dir, info))
    at = Placement(H, W)
    for bi, members in enumerate(blocks):
        place_block(bi, members, mine, block_slivers.get(bi, []), shifts[bi], streets, all_roads,
                    road_list, proj, at, rect_tiles, rect_count, show)
    straighten_pass(at, mine, show)
    painted = painted_street(map_dir, name, (H, W))
    if painted is not None:
        street_pass(at, mine, painted, carriageways(road_list, W, H), show)

    # --- 7: lots that do not overlap; written -----------------------------------------------
    pieces, final, lost, count = cut_into_lots(at.owner, len(mine))
    gj["features"] = placed_features(keep, mine, pieces, at.block_of, proj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gj, f)
    clear_trees(map_dir, name, final > 0)
    lanes_cut = clear_lanes(map_dir, name, final > 0, road_list)

    # --- report ------------------------------------------------------------------------------
    real_tiles = sum(int(round(m["px"].area)) for m in mine)
    built = int((final > 0).sum())
    multi = sum(1 for v in count.values() if v > 1)
    sh = np.array(shifts)
    print(f"buildings {len(mine)} in {len(blocks)} blocks ({sum(len(b) > 1 for b in blocks)} of 2+); "
          f"{len(keep)} left to KnoxMap")
    print(f"  near outlines: {len(slivers)} slivers of roof closed, {open_gaps} real gaps kept")
    print(f"  LiDAR ({ground.source}): registered {int((np.abs(sh).sum(1) > 0).sum())} blocks, "
          f"median shift {np.median(np.abs(sh[:, 0])):.1f},{np.median(np.abs(sh[:, 1])):.1f} m; "
          f"overlap {np.mean([g[0] for g in gains]):.2f} -> {np.mean([g[1] for g in gains]):.2f}")
    low = [m for m in mine if m["cover"] < COVER_LOW]
    print(f"  outlines the LiDAR barely sees (<{COVER_LOW:.0%}): {len(low)}; "
          f"buildings added from the LiDAR: {len(new)}; other LiDAR bits without an outline: "
          f"{len(only)} (lidar_only.geojson)")
    on_rail = sum(1 for f in only if "on the railway" in f["properties"]["why"])
    print(f"  railway: {sum(1 for m in mine if m.get('rail') == 'station')} of its own buildings kept on its land; "
          f"footbridges left out {len(footbridges)}; LiDAR shapes on it not made buildings {on_rail}")
    print(f"  carried along a street: {at.along_street} blocks ({at.corner_blocks} round a corner, each building along its own); the rest moved rigidly")
    print(f"  moved: median {np.median(at.moves):.1f} tiles; nudged {at.nudged} (off roads, or apart from a "
          f"neighbour: {at.kept_apart}); gave way to a neighbour {at.given_way}; tiles trimmed off a road {at.trimmed}")
    print(f"  long buildings laid straight and whole in their block: {len(at.pulled_in)}")
    if painted is not None:
        on_st = np.unique(at.block_owner[painted & (final > 0)])
        print(f"  blocks (or their pieces) moved off the painted street (up to {STREET_REACH} tiles): "
              f"{len(at.off_street)} ({sum(n for _b, _d, n in at.off_street)} squares); "
              f"still on it: {len(on_st[on_st > 0])} blocks, "
              f"{int((painted & (final > 0)).sum())} squares")
    print(f"  rows of homes drawn one by one joined into one building each: {rows_joined} "
          f"({homes_joined} homes)")
    if show:
        from collections import Counter
        big = [f["properties"]["why"] for f in only if f["properties"]["m2"] >= NEW_MIN_M2]
        print(f"    LiDAR bits >= {NEW_MIN_M2} m2 not made buildings, why:",
              Counter(w for ws in big for w in ws.split(", ")).most_common())
        print("    trimmed by road:", sorted(at.trim_by.items(), key=lambda kv: -kv[1])[:12])
    print(f"  service lanes under buildings taken off the ground: {lanes_cut} tiles "
          f"(with the bits of lane that no longer lead anywhere)")
    print(f"  big buildings' walls squared: {squared}; left as drawn, squared they fit the LiDAR worse: {refused}")
    print(f"  squared to rectangles >= {rect_tiles}x{rect_tiles}: {at.squared_off} tiles of spurs and strips dropped")
    print(f"  tiles: real {real_tiles}, built {built} ({built / max(1, real_tiles):.0%}); "
          f"lost to lots/size {lost}")
    print(f"  pieces {len(pieces)}; buildings cut for lots {multi}")
    if show:
        for m in low:
            print(f"    low cover {m['id']} {m['ft']['properties'].get('name') or ''}: {m['cover']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
