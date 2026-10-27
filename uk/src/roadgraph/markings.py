"""UK road markings and signals on a road-graph map (user, 2026-10-11), from
the graph's own geometry: the lines vanilla's (street_trafficlines_01;
Erika's pieces wait for its licence check), the signals, beacons, signs and
street furniture ours (asset_creator, laid by furniture.py).

Each marking is a loose tile by name (x, y, 0, layer, tile), written by the
hand-over to <map>_markings.json; kerbs.py adds them to the map with its
kerbs (a square that is no longer road by then loses its road markings).
Thin lines lie on a tile edge (street_trafficlines_01: faded white 0-7,
fresh white 8-15, faded yellow 16-23; W N E S at +0, +2, +4, +6, measured
from the sprites); 48 and 49 are bright bars a fifth of a tile wide through
the middle, running N-S and E-W. None run at 45 degrees, so a diagonal
stretch gets no lines.

- Centre lines: white, broken - 4 tiles on and 8 off, 6 on and 3 off within
  HAZARD tiles of a junction (the UK warning line) - on A, B and minor roads
  and town streets at least CENTRE_MIN wide; none in a junction's mouth.
  KnoxMap's American solid yellow is taken off (handover.redress).
- Edge lines (Generate's): kept only on A and B roads out of town.
- Give-way lines: at each junction, every arm that is not part of the road
  through it (the two arms of highest rank, then the straightest pair) gets
  a double broken white line across it at the mouth - across its approach
  half where it has a centre line; every arm at a mini roundabout, and at a
  roundabout (OSM junction=roundabout or OS Open Roads' form_of_way
  Roundabout) every arm but the two circulating.
- Yellow lines: double yellows tight to the kerb (ours, road_markings_uk_01,
  with an L where the kerb turns; 2026-10-14 - before, vanilla's single
  fresh yellow stood in) the whole length of town through roads (A, B, minor)
  and of a street with parking bays where it has none, and on other town
  streets within YELLOW_REACH of a junction (the Highway Code's 10 m).
- Bay lines: a white line along the carriageway beside parking bays.
Only where the pieces draw a clean line (user, 2026-10-11: markings on
stepped stretches came out staggered): lines along a kerb or the bays on
straight runs of YELLOW_RUN / BAY_RUN squares, centre lines on stretches
along the grid CENTRE_RUN tiles long, give-way lines where the side road
leaves square-on and straight and its mouth row is whole; nothing beside a
45-degree edge.
- Zebra crossings: OSM crossing=zebra / crossing_ref=zebra, or a marked
  crossing (crossing=marked|uncontrolled, not crossing:markings=no) where the
  LiDAR intensity shows paint across the road; ZEBRA_DEPTH deep, the UK's
  white bands, give-way lines and zig-zags (decals.zebra; vanilla's
  bright bars until 2026-10-14).
- Double white lines (solid) instead of the centre line within
  decals.BEND_REACH of a sharp bend out of town (decals.bends).
- The one- and two-tile markings - give-way triangles, speed roundels, SLOW
  and deflection arrows at bends, lane arrows, mini roundabout discs, yellow
  boxes: decals.py.
- Signal crossings (crossing=traffic_signals: pelican, puffin, toucan): two
  broken lines across (the studs), a stop line across each approach lane
  STOP_BACK tiles before it, and a signal head on the kerb beside each stop
  line facing the traffic (ours, on green: furniture.signal).
- Belisha beacons, junction signals, signs and the street furniture:
  furniture.py, after the markings.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import shapely

from . import decals, furniture
from .network import RANK
from .raster import vertices

FADED_WHITE, FRESH_WHITE, FRESH_YELLOW = 0, 8, 24
EDGE = {"W": 0, "N": 2, "E": 4, "S": 6}
BAR_NS, BAR_EW = 48, 49
LINES = "street_trafficlines_01_"
CENTRE_CLASSES = ("motorway", "a_road", "b_road", "minor")
CENTRE_MIN = 6               # a town street this wide or more has a centre line
DASH, GAP = 4, 8             # tiles: a centre line's marks and gaps
HAZARD_DASH, HAZARD_GAP = 6, 3
HAZARD = 20                  # tiles from a junction the warning line runs
YELLOW_REACH = 10            # tiles from a junction the kerb has yellow lines
THROUGH = ("a_road", "b_road", "minor")   # town roads with yellow lines all along
YELLOW_RUN = 3               # squares in a straight run along the kerb, at least, to carry a yellow line
BAY_RUN = 3                  # ...and along the bays, to carry their line
CENTRE_RUN = 12              # tiles a stretch along the grid runs, at least, to carry a centre line
STOP_BACK = 3                # tiles from a signal crossing to its stop lines
ZEBRA_DEPTH = 3
ZEBRA_BRIGHT = 1.4           # the crossing's paint this much brighter than the road round it
NODE_REACH = 6               # tiles: an OSM point this near a road is on it
COMPASS = {(0, -1): "N", (1, 0): "E", (0, 1): "S", (-1, 0): "W"}   # map tiles: x east, y south


def _unit(a, b):
    d = (b[0] - a[0], b[1] - a[1])
    n = math.hypot(*d)
    return (d[0] / n, d[1] / n) if n else (0.0, 0.0)


def _axis(d):
    """d as one of the four grid directions, or None if it is not one."""
    for v in COMPASS:
        if abs(d[0] - v[0]) < 1e-6 and abs(d[1] - v[1]) < 1e-6:
            return v
    return None


class Marks:
    """The tiles laid so far, one per square and layer."""

    def __init__(self):
        self.tiles: dict = {}
        self.count: dict = {}
        self.above: dict = {}             # (x, y, z, layer) -> tile: a storey up or more (a telegraph pole's top, its wires)

    def put(self, x, y, layer, tile, kind):
        if (x, y, layer) not in self.tiles:
            self.tiles[(x, y, layer)] = tile
            self.count[kind] = self.count.get(kind, 0) + 1

    def put_above(self, x, y, z, layer, tile, kind):
        if (x, y, z, layer) not in self.above:
            self.above[(x, y, z, layer)] = tile
            self.count[kind] = self.count.get(kind, 0) + 1

    def rows(self):
        return [[int(x), int(y), 0, layer, t] for (x, y, layer), t in sorted(self.tiles.items())] + \
            [[int(x), int(y), int(z), layer, t] for (x, y, z, layer), t in sorted(self.above.items())]


def junctions(paths: dict, links) -> dict:
    """{corner: [(link, unit direction out, half width)]} at every path end."""
    arms: dict = {}
    for i, path in paths.items():
        vs = vertices(path)
        if len(vs) < 2 or vs[0] == vs[-1]:
            continue
        for end, nxt in ((vs[0], vs[1]), (vs[-1], vs[-2])):
            arms.setdefault(tuple(end), []).append((i, _unit(end, nxt), links[i].width / 2))
    return arms


def through_pair(arms, links):
    """The two arms of the road through a junction: highest rank first, then
    the straightest pair."""
    best = None
    for a in range(len(arms)):
        for b in range(a + 1, len(arms)):
            (ia, da, _), (ib, db, _) = arms[a], arms[b]
            straight = -(da[0] * db[0] + da[1] * db[1])          # 1 when opposite
            key = (RANK[links[ia].cls] + RANK[links[ib].cls], -straight)
            if best is None or key < best[0]:
                best = (key, (a, b))
    return best[1] if best else ()


def has_centre(link) -> bool:
    """Whether the road has a centre line: A, B, minor, wide town streets - not a one-way street."""
    one_way = link.tags.get("oneway") in ("yes", "1", "-1") and not link.tags.get("dual_carriageway")
    return not one_way and (link.cls in CENTRE_CLASSES or (link.cls == "local" and link.width >= CENTRE_MIN))


def centre_lines(m: Marks, paths, links, owner, arms, bent=None):
    degree = {c: len(a) for c, a in arms.items()}
    mouth = {c: max(h for _i, _d, h in a) + 2 for c, a in arms.items()}
    for i, path in paths.items():
        l = links[i]
        if not has_centre(l):
            continue
        vs = vertices(path)
        total = sum(math.dist(a, b) for a, b in zip(vs, vs[1:]))
        ends = [tuple(vs[0]), tuple(vs[-1])]
        skip0 = mouth.get(ends[0], 0) if degree.get(ends[0], 0) >= 3 else 0
        skip1 = mouth.get(ends[1], 0) if degree.get(ends[1], 0) >= 3 else 0
        solid_at = (bent or {}).get(i, ())
        s = 0.0
        for a, b in zip(vs, vs[1:]):
            d = _axis(_unit(a, b))
            n = int(round(math.dist(a, b)))
            by_bend = any(s - decals.BEND_REACH <= sb <= s + n + decals.BEND_REACH for sb in solid_at)
            if n < CENTRE_RUN and not (by_bend and n >= 3):
                d = None                                  # a short stretch between bends: no line
            for k in range(n):
                at = s + k + 0.5
                if d is not None and skip0 <= at <= total - skip1:
                    near = min(at - skip0 if skip0 else 1e9, total - skip1 - at if skip1 else 1e9)
                    solid = any(abs(at - sb) <= decals.BEND_REACH for sb in solid_at)
                    on = solid or ((at % (HAZARD_DASH + HAZARD_GAP)) < HAZARD_DASH if near < HAZARD
                                   else (at % (DASH + GAP)) < DASH)
                    if on:
                        # The tile whose west (or north) edge is the centre line here.
                        x = int(a[0] + d[0] * k + (0 if d[0] >= 0 else -1)) if d[0] else int(a[0])
                        y = int(a[1] + d[1] * k + (0 if d[1] >= 0 else -1)) if d[1] else int(a[1])
                        if 0 <= y < owner.shape[0] and 0 <= x < owner.shape[1] and owner[y, x] == i + 1:
                            side = "W" if d[0] == 0 else "N"
                            if solid:
                                m.put(x, y, "FloorOverlay", decals.tile(f"double_white_{side}"), "double white line")
                            else:
                                m.put(x, y, "FloorOverlay", LINES + str(FADED_WHITE + EDGE[side]), "centre line")
            s += math.dist(a, b)


def _across(owner, link_i, row_pt, d):
    """The road squares of link_i in the row (or column) across direction d
    through row_pt."""
    H, W = owner.shape
    x, y = int(math.floor(row_pt[0])), int(math.floor(row_pt[1]))
    out = []
    if d[0] == 0:                       # road runs N-S: a row across
        if 0 <= y < H:
            out = [(xx, y) for xx in range(max(0, x - 12), min(W, x + 13)) if owner[y, xx] == link_i + 1]
    else:
        if 0 <= x < W:
            out = [(x, yy) for yy in range(max(0, y - 12), min(H, y + 13)) if owner[yy, x] == link_i + 1]
    return out


def _left(heading, tile, centre, hand):
    """Whether the square lies left of the road's centre, facing `heading`."""
    o = (tile[0] + 0.5 - centre[0], tile[1] + 0.5 - centre[1])
    return (heading[0] * o[1] - heading[1] * o[0]) * hand > 0


def give_ways(m: Marks, paths, links, owner, arms, minis, hand, drawn=None):
    eligible = 0
    for corner, arm in arms.items():
        if len(arm) < 3 and not (len(arm) >= 2 and corner in minis):
            continue
        circulating = tuple(k for k, (i, _d, _h) in enumerate(arm) if decals._is_roundabout(links[i]))
        keep = () if corner in minis else circulating if len(circulating) == 2 else through_pair(arm, links)
        major = max((arm[k][2] for k in keep), default=max(h for _i, _d, h in arm))
        for k, (i, d, _h) in enumerate(arm):
            if k in keep:
                continue
            vs_i = vertices(paths[i])
            fl = decals.flow(links[i], paths[i])
            if fl and (fl == 1) == (tuple(vs_i[0]) == corner):     # one-way, its traffic leaving the junction
                m.count["one-way arm leaving the junction (no give-way)"] = \
                    m.count.get("one-way arm leaving the junction (no give-way)", 0) + 1
                continue
            eligible += 1
            ax = _axis(d)
            if ax is None:
                m.count["give-way on a diagonal (not drawn)"] = m.count.get("give-way on a diagonal (not drawn)", 0) + 1
                continue
            s0 = math.ceil(major) + 1
            # Only where the arm leaves square-on and runs on straight past
            # the line, and the line's row is the arm's full width: at a
            # skewed or stepped mouth the line came out in bits (user, 2026-10-11).
            vs = vertices(paths[i])
            first = vs[:2] if tuple(vs[0]) == corner else vs[::-1][:2]
            if math.dist(*first) < s0 + 3:
                m.count["give-way at a bent mouth (not drawn)"] = m.count.get("give-way at a bent mouth (not drawn)", 0) + 1
                continue
            pt = (corner[0] + ax[0] * (s0 + 0.5), corner[1] + ax[1] * (s0 + 0.5))
            row = _across(owner, i, pt, ax)
            gaps = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(row, row[1:])]
            if len(row) < links[i].width - 1 or (gaps and max(gaps) > 1):
                m.count["give-way at a ragged mouth (not drawn)"] = m.count.get("give-way at a ragged mouth (not drawn)", 0) + 1
                continue
            if has_centre(links[i]):
                towards = (-ax[0], -ax[1])            # a driver coming up to the junction
                row = [t for t in row if _left(towards, t, pt, hand)]
            sides = ("N", "S") if ax[0] == 0 else ("W", "E")
            for t in row:
                if (t[0] + t[1]) % 2 == 0:
                    for layer, side in zip(("FloorOverlay", "FloorGrime"), sides):
                        m.put(t[0], t[1], layer, LINES + str(FRESH_WHITE + EDGE[side]), "give-way line")
            if drawn is not None and row:
                drawn.append((i, pt, ax))
    return eligible


STEP = {"W": (-1, 0), "E": (1, 0), "N": (0, -1), "S": (0, 1)}


def _shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """a moved so out[y, x] = a[y + dy, x + dx] (False off the edge)."""
    H, W = a.shape
    out = np.zeros_like(a)
    out[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = \
        a[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
    return out


def straight_runs(a: np.ndarray, side: str, n: int) -> np.ndarray:
    """The squares of a in a straight run of n or more along an edge on
    `side` (W/E edges run N-S, N/S edges E-W): a line laid square by square
    only reads as one where its squares follow on (user, 2026-10-11: lines
    on stepped stretches came out staggered)."""
    ax, ay = (0, 1) if side in ("W", "E") else (1, 0)
    run = np.zeros_like(a)
    for start in range(-(n - 1), 1):
        window = a.copy()
        for k in range(start, start + n):
            window &= _shift(a, ax * k, ay * k)
        run |= window
    return run


def beside(a: np.ndarray, side: str) -> np.ndarray:
    """Squares whose neighbour on `side` is in a."""
    dx, dy = STEP[side]
    return _shift(a, dx, dy)


DOUBLE_YELLOW = "road_markings_uk_01_"
DY_EDGE = {"W": 0, "N": 1, "E": 2, "S": 3}             # double yellows along one edge...
DY_CORNER = {("N", "W"): 4, ("E", "N"): 5, ("E", "S"): 6, ("S", "W"): 7}   # ...and the L where a kerb turns


def yellow_lines(m: Marks, paths, links, owner, kerbside, arms, bayed: set, town, half: dict):
    """Double yellow lines along the kerb (ours, road_markings_uk_01; user,
    2026-10-14 - until then vanilla's single yellow stood in): the whole length of town through roads
    (THROUGH: A, B and minor roads - user, 2026-10-11, more roads not
    fewer), and on other town streets within YELLOW_REACH of a junction;
    never beside bays; straight kerb runs only."""
    H, W = owner.shape
    near = np.zeros((H, W), bool)
    for corner, arm in arms.items():
        if len(arm) >= 3:
            cx, cy = int(corner[0]), int(corner[1])
            r = YELLOW_REACH + int(max(h for _i, _d, h in arm))
            ys, xs = np.ogrid[max(0, cy - r):min(H, cy + r), max(0, cx - r):min(W, cx + r)]
            near[max(0, cy - r):min(H, cy + r), max(0, cx - r):min(W, cx + r)] |= \
                (xs + 0.5 - corner[0]) ** 2 + (ys + 0.5 - corner[1]) ** 2 <= r * r
    lab = owner - 1
    town_street = np.zeros((H, W), bool)
    bayed_street = np.zeros((H, W), bool)
    paved = {i for i, l in enumerate(links) if l.cls in ("a_road", "b_road", "minor", "local")}
    named_bays = {links[i].tags.get("name") for i in bayed} - {None}
    for i in paved:
        mine = lab == i
        town_street |= mine
        if links[i].cls in THROUGH or i in bayed or links[i].tags.get("name") in named_bays:
            bayed_street |= mine
    # Not beside a 45-degree edge: its half squares (raster.py) and the road round them.
    slant = np.zeros((H, W), bool)
    for x, y in half:
        slant[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    want = (owner > 0) & town_street & town & (near | bayed_street) & ~slant
    sides: dict = {}
    for side in STEP:
        for y, x in zip(*np.nonzero(straight_runs(want & beside(kerbside, side), side, YELLOW_RUN))):
            sides.setdefault((int(x), int(y)), set()).add(side)
    for (x, y), here in sorted(sides.items()):
        pair = tuple(sorted(here))
        if len(here) == 2 and pair in DY_CORNER:
            m.put(x, y, "FloorGrime2", DOUBLE_YELLOW + str(DY_CORNER[pair]), "double yellow corner")
        else:
            m.put(x, y, "FloorGrime2", DOUBLE_YELLOW + str(DY_EDGE[sorted(here)[0]]), "double yellow")


def bay_lines(m: Marks, owner, bays):
    """A white line along the carriageway's edge where parking bays run
    beside it; straight runs only."""
    for side in STEP:
        for y, x in zip(*np.nonzero(straight_runs((owner > 0) & beside(bays, side), side, BAY_RUN))):
            m.put(int(x), int(y), "FloorGrime2", LINES + str(FADED_WHITE + EDGE[side]), "bay line")


def load_nodes(area: str):
    path = os.path.join(area, "road_nodes.geojson")
    if not os.path.exists(path):
        return []
    return [(f["geometry"]["coordinates"], f["properties"])
            for f in json.load(open(path, encoding="utf-8"))["features"]]


def on_graph(lonlat, proj, links, paths):
    """(link, snapped point, unit direction along it) for an OSM point on a
    road, carried as the shift samples are (its share along the link)."""
    x, y = proj.to_px(lonlat[1], lonlat[0])
    p = shapely.Point(x, y)
    best = None
    for i, path in paths.items():
        line = shapely.LineString(links[i].pts)
        d = line.distance(p)
        if d <= NODE_REACH and (best is None or d < best[0]):
            best = (d, i, line)
    if best is None:
        return None
    _d, i, line = best
    t = line.project(p) / max(line.length, 1e-9)
    snapped = shapely.LineString(paths[i])
    q = snapped.interpolate(t, normalized=True)
    a = snapped.interpolate(max(0.0, t - 0.02), normalized=True)
    b = snapped.interpolate(min(1.0, t + 0.02), normalized=True)
    return i, (q.x, q.y), _unit((a.x, a.y), (b.x, b.y))


def painted(lidar, proj, lonlat, width) -> bool:
    """Whether the 2019 intensity at a crossing is brighter than the road round it (paint)."""
    from .widths import TO_BNG
    e, n = TO_BNG.transform(lonlat[0], lonlat[1])
    r, c = int(lidar.box[3] - n), int(e - lidar.box[0])
    here = lidar.inten[r - 1:r + 2, c - 1:c + 2]                      # the crossing's middle, 3 m square
    round_ = lidar.inten[r - 15:r + 16, c - 15:c + 16]                # 30 m round it: its 25th percentile is the tarmac
    if here.size == 0 or np.isnan(here).all() or np.isnan(round_).all():
        return False
    return float(np.nanmean(here)) >= ZEBRA_BRIGHT * float(np.nanpercentile(round_, 25))


def crossings(m: Marks, nodes, proj, links, paths, owner, ground_ok, hand, lidar, zebras=None, signals=None):
    heads = []
    for lonlat, tags in nodes:
        kind = (tags.get("crossing") or "").lower()
        ref = (tags.get("crossing_ref") or "").lower()
        if tags.get("highway") == "traffic_signals" and kind != "traffic_signals":
            continue                                    # a crossing's own lights, or a junction's
        zebra = kind == "zebra" or ref == "zebra"
        signal = kind == "traffic_signals" or ref in ("pelican", "puffin", "toucan", "pegasus")
        marked = kind in ("marked", "uncontrolled") and (tags.get("crossing:markings") or "").lower() != "no"
        if not (zebra or signal or marked):
            continue
        got = on_graph(lonlat, proj, links, paths)
        if got is None:
            m.count["crossing off the roads"] = m.count.get("crossing off the roads", 0) + 1
            continue
        i, q, d = got
        ax = _axis((round(d[0]), round(d[1]))) if abs(abs(d[0]) - abs(d[1])) > 0.5 else None
        if ax is None:
            m.count["crossing on a diagonal (not drawn)"] = m.count.get("crossing on a diagonal (not drawn)", 0) + 1
            continue
        if signal:
            side = "N" if ax[0] == 0 else "W"           # the studs: a broken line each side of the crossing
            for k in (-1, 1):
                pt = (q[0] + ax[0] * (k + 0.5 * (k > 0)), q[1] + ax[1] * (k + 0.5 * (k > 0)))
                for t in _across(owner, i, pt, ax):
                    if (t[0] + t[1]) % 2 == 0:
                        m.put(t[0], t[1], "FloorOverlay", LINES + str(FRESH_WHITE + EDGE[side]), "crossing studs")
            for sgn in (-1, 1):                         # traffic coming from either way
                heading = (ax[0] * -sgn, ax[1] * -sgn)  # towards the crossing
                stop = (q[0] + ax[0] * sgn * STOP_BACK, q[1] + ax[1] * sgn * STOP_BACK)
                lane = [t for t in _across(owner, i, stop, ax) if _left(heading, t, stop, hand)]
                bar = BAR_EW if ax[0] == 0 else BAR_NS
                for t in lane:
                    m.put(t[0], t[1], "FloorOverlay", LINES + str(bar), "stop line")
                if lane:
                    # The head on the first kerbside square beyond the lane, facing the traffic.
                    out_ = max(lane, key=lambda t: abs((t[0] + 0.5 - stop[0]) * ax[1] - (t[1] + 0.5 - stop[1]) * ax[0]))
                    step = (int(np.sign(out_[0] + 0.5 - stop[0])) if ax[0] == 0 else 0,
                            int(np.sign(out_[1] + 0.5 - stop[1])) if ax[1] == 0 else 0)
                    x, y = out_
                    for _ in range(6):
                        x, y = x + step[0], y + step[1]
                        if not (0 <= y < owner.shape[0] and 0 <= x < owner.shape[1]):
                            break
                        if owner[y, x] == 0 and ground_ok[y, x]:
                            facing = COMPASS[(-heading[0], -heading[1])]
                            m.put(x, y, "Furniture", furniture.signal("green", facing), "signal head")
                            heads.append((x, y))
                            break
            m.count["signal crossings"] = m.count.get("signal crossings", 0) + 1
            if signals is not None:
                signals.append((q, ax))
            continue
        if not zebra and not (lidar is not None and painted(lidar, proj, lonlat, links[i].width)):
            m.count["marked crossing without paint in the LiDAR"] = \
                m.count.get("marked crossing without paint in the LiDAR", 0) + 1
            continue
        q = decals.zebra(m, owner, i, q, ax, ZEBRA_DEPTH)
        if q is None:
            continue
        m.count["zebra crossings"] = m.count.get("zebra crossings", 0) + 1
        if zebras is not None:
            zebras.append((q, ax))
    return heads


def lay(paths, links, owner, half, town, bayed, bays, kerbside, ground_ok, hand, proj, area, lidar=None):
    """Every marking: (tile rows, counts, signal head squares)."""
    m = Marks()
    arms = junctions(paths, links)
    nodes = load_nodes(area)
    minis = set()
    for lonlat, tags in nodes:
        if tags.get("highway") == "mini_roundabout":
            x, y = proj.to_px(lonlat[1], lonlat[0])
            # The junction it is at: the nearest with three arms or more (it moved with the roads).
            near = min((c for c, a in arms.items() if len(a) >= 3),
                       key=lambda c: math.dist(c, (x, y)), default=None)
            if near is not None and math.dist(near, (x, y)) <= 3 * NODE_REACH:
                minis.add(near)
    bent = decals.bends(paths, links, arms, town)
    centre_lines(m, paths, links, owner, arms, bent)
    drawn: list = []
    eligible = give_ways(m, paths, links, owner, arms, minis, hand, drawn)
    yellow_lines(m, paths, links, owner, kerbside, arms, bayed, town, half)
    bay_lines(m, owner, bays)
    zebras: list = []
    signals: list = []
    heads = crossings(m, nodes, proj, links, paths, owner, ground_ok, hand, lidar, zebras, signals)
    m.count["junction arms giving way"] = eligible
    # Beacons, junction signals, signs, post boxes, benches... (furniture.py),
    # off the road: the half squares are road too (painted, owned by no link;
    # kerbs.py took a sign off each that stood on one).
    road = owner.copy()
    for x, y in half:
        if 0 <= y < road.shape[0] and 0 <= x < road.shape[1] and road[y, x] == 0:
            road[y, x] = len(links) + 1
    limits: list = []
    heads = list(heads) + sorted(furniture.lay(m, nodes, proj, links, paths, road, ground_ok, hand, arms, zebras,
                                               area, town, through_pair, on_graph, limits, signals))
    decals.lay(m, owner, paths, links, arms, minis, drawn, limits, bent, nodes, proj, hand)
    return m.rows(), m.count, heads


def edge_keep(owner, links, town) -> np.ndarray:
    """Where Generate's edge lines stay: on A and B roads out of town."""
    cls = np.array([l.cls for l in links] + [""])
    lab = np.where(owner > 0, owner - 1, len(links))
    return np.isin(cls[lab], ("motorway", "a_road", "b_road")) & ~town[:owner.shape[0], :owner.shape[1]]
