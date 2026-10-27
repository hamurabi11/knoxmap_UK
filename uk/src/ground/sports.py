"""Sports pitches, courts and playgrounds from OSM, with the game's own kit.

    python sports.py <knoxmap map dir> [--picture out.png] [--box x,y,w,h]

Run after trees.py (it clears the trees and tufts off the pitches) and before
Build. Rerun it after plots2.py, trees.py or a new Generate.

KnoxMap paints a pitch as grass and nothing more. Everything here was read
off the vanilla map (Build 42, the sports park at world 13650-13750,
1800-1845 and every playground in the game), not guessed:

- Tennis (vanilla's own courts): the court recreational_sports_01_22 (green)
  inside a surround of _23 (red), bright white lines street_trafficlines_01
  8-15 on the tile edges, the net _49/_50/_51 along a row (west post, net,
  east post) or _54/_53/_52 down a column (north to south), tall chain-link
  all round with a gate. A court is COURT_W x COURT_L tiles (singles 8 m,
  doubles 10, 24 long: the real court is 8.23 / 10.97 x 23.77 m) in a
  module of MODULE_W x MODULE_L (real 18.3 x 36.6 m); as many side by side
  as the pitch holds, smaller when it holds none.
- Football: lines on the grass (vanilla's pitch has none; an English one
  does), goals _18/_17/_16 down a column or _19/_20/_21 along a row,
  penalty and goal areas and the centre circle scaled from 105 x 68 m.
- Basketball, netball, multi-sport: a tarmac court with faded white lines
  (0-7), vanilla's 4 x 4 centre circle (_108-_119) and keys, a hoop at each
  end: the pole _33 on the ground, the board a floor up, _25 / _27 at the
  west / east end, _24 / _26 at the north / south.
- Cricket nets: green matting (_22), chain-link round three sides and
  between the lanes, the bowlers' end open.
- Bowls and any other grass pitch: a lawn, cleared.
- Playgrounds: vanilla's swings, slides, climbing frames and roundabouts
  (location_community_park_01) as vanilla puts them together, each on a
  pad of sand, inside railings with a gate.

Each pitch is squared to the grid like a plot: the rectangle with its real
centre and size, turned to the nearer axis, then shrunk off buildings,
streets and the pitches already placed.

Written as loose tiles into <map>_structures.json, which Build already lays
into lots (Generate's own file is kept as <map>_structures.orig.json; each
run starts again from it), and fences as lines into <map>_fences.geojson
(source=sports, replaced each run; OSM's own fences round a pitch that gets
one are left out). Trees and tufts on the pitches are cleared from
<map>_veg_base.bmp and <map>_veg.bmp.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from buildings import blocks as B  # noqa: E402

SPORTS = "recreational_sports_01_"
PARK = "location_community_park_01_"
LINES_BRIGHT = 8           # street_trafficlines_01: W NW N NE E SE S SW from here
LINES_FADED = 0
EDGE_ORDER = ("W", "NW", "N", "NE", "E", "SE", "S", "SW")
# A square can carry more than one line piece: they go on these layers in turn.
LINE_LAYERS = ("FloorOverlay", "FloorGrime", "FloorGrime2")
GREEN, RED = SPORTS + "22", SPORTS + "23"
TARMAC = "blends_street_01_64"    # Rules.txt street3, the smooth tarmac
SAND = "blends_natural_01_0"      # Rules.txt sand, as under vanilla's playgrounds
SAND_RGB, GRASS_RGB = (210, 200, 160), (117, 117, 47)   # the same on the ground bitmap; medium grass

COURT_W, COURT_L = 10, 24          # tennis: doubles width, length
SERVICE = 6                        # tennis: service line from the net
MODULE_W, MODULE_L = 18, 36        # tennis: a court and its run-off
MIN_MODULE_W, MIN_MODULE_L = 14, 30
MIN_COURT_W, MIN_COURT_L = 10, 16  # a pitch smaller than this holds no court (mini tennis: a lawn)
MIN_SIDE = 6                       # a pitch squared smaller than this either way is left out
SHRINK_KEEP = 0.6                  # ...or shrunk below this share of its size

# Playground kit as vanilla puts it together: (x, y) -> tiles bottom up.
KIT = {
    "swings_x": {(0, 0): [1], (0, 1): [0], (1, 0): [3], (2, 0): [4], (3, 0): [2, 1], (3, 1): [0]},
    "swings_y": {(0, 0): [13], (1, 0): [14], (0, 1): [11], (0, 2): [10], (0, 3): [12, 13], (1, 3): [14]},
    "frame": {(0, 0): [51], (1, 0): [52], (0, 1): [49], (1, 1): [50]},
    "roundabout": {(0, 0): [23], (1, 0): [22], (0, 1): [20], (1, 1): [21]},
    "slide_x": {(0, 0): [18], (1, 0): [19]},
    "slide_y": {(0, 0): [17], (0, 1): [16]},
    "tower_x": {(0, 0): [38], (1, 0): [39]},
    "tower_y": {(0, 0): [31], (0, 1): [30]},
}
KIT_ORDER = ("swings_x", "frame", "roundabout", "slide_y", "swings_y", "tower_x", "slide_x", "tower_y")
KIT_LAYERS = ("Furniture", "Furniture2")


def kind_of(tags: dict) -> str | None:
    """What a leisure feature is built as, or None to leave it."""
    leisure = tags.get("leisure")
    if leisure == "playground":
        return "playground"
    if leisure != "pitch":
        return None
    sport = (tags.get("sport") or "").lower()
    name = (tags.get("name") or "").lower()
    if "tennis" in sport or (not sport and "tennis" in name):
        return "tennis"
    if any(s in sport for s in ("soccer", "football")):
        return "football"
    if any(s in sport for s in ("basketball", "netball", "multi")):
        return "court"
    if "cricket_nets" in sport:
        return "nets"
    return "lawn"


def hard(tags: dict) -> bool:
    return (tags.get("surface") or "") in {"asphalt", "concrete", "paved", "tartan", "acrylic"}


# --- the frame of a pitch ----------------------------------------------------------------
class Frame:
    """A rectangle x0..x1, y0..y1 with its own axes: u along its length, v
    across. Lines and kit are laid out in (u, v) and turned onto the map."""

    def __init__(self, x0, y0, x1, y1, along_x=None):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.along_x = (x1 - x0) >= (y1 - y0) if along_x is None else along_x
        self.L = (x1 - x0) if self.along_x else (y1 - y0)
        self.W = (y1 - y0) if self.along_x else (x1 - x0)

    def tile(self, u, v):
        return (self.x0 + u, self.y0 + v) if self.along_x else (self.x0 + v, self.y0 + u)

    def across(self, u, v):
        """The edge across the length at u (u fixed), beside v: (x, y, 'V'|'H')."""
        return (self.x0 + u, self.y0 + v, "V") if self.along_x else (self.x0 + v, self.y0 + u, "H")

    def along(self, u, v):
        """The edge along the length at v (v fixed), beside u."""
        return (self.x0 + u, self.y0 + v, "H") if self.along_x else (self.x0 + v, self.y0 + u, "V")

    def corner(self, u, v):
        return self.tile(u, v)


def rect_lines(f: Frame, u0, v0, u1, v1, edges: set):
    for v in range(v0, v1):
        edges.add(f.across(u0, v))
        edges.add(f.across(u1, v))
    for u in range(u0, u1):
        edges.add(f.along(u, v0))
        edges.add(f.along(u, v1))


def line_pieces(edges: set, box, base: int) -> list:
    """Line pieces for edges on tile corners: (x, y, 'V') is the line down
    x from y to y+1, (x, y, 'H') along y from x to x+1. Each goes to the tile
    inside the box (x0, y0, x1, y1) - west and north edges of the tile past
    it, the east and south edges of the last row and column - and a tile's
    edges become as few pieces as the sheet has (corners first)."""
    x0, y0, x1, y1 = box
    sides: dict[tuple[int, int], set] = {}
    for x, y, d in edges:
        if d == "V":
            key, side = ((x - 1, y), "E") if x >= x1 else ((x, y), "W")
        else:
            key, side = ((x, y - 1), "S") if y >= y1 else ((x, y), "N")
        sides.setdefault(key, set()).add(side)
    out = []
    for (x, y), s in sides.items():
        pieces = []
        for pair in ("NW", "NE", "SE", "SW"):
            if set(pair) <= s:
                pieces.append(pair)
                s = s - set(pair)
        pieces += sorted(s)
        for layer, p in zip(LINE_LAYERS, pieces):
            out.append((x, y, 0, layer, f"street_trafficlines_01_{base + EDGE_ORDER.index(p)}"))
    return out


def fill(f: Frame, tile: str, u0=0, v0=0, u1=None, v1=None) -> list:
    u1 = f.L if u1 is None else u1
    v1 = f.W if v1 is None else v1
    return [(*f.tile(u, v), 0, "Floor", tile) for u in range(u0, u1) for v in range(v0, v1)]


def run_along_v(f: Frame, u, v0, v1, west_mid_east, north_mid_south) -> list:
    """A row of kit across the pitch at u, v0..v1-1, ends and middle: on the
    map a row (west, middle, east) or a column (north, middle, south)."""
    ends = north_mid_south if f.along_x else west_mid_east
    out = []
    for v in range(v0, v1):
        t = ends[0] if v == v0 else ends[2] if v == v1 - 1 else ends[1]
        out.append((*f.tile(u, v), 0, "Furniture", t))
    return out


# --- what goes on each kind -------------------------------------------------------------------
def courts_across(length, width) -> int:
    """Tennis courts side by side across a pitch this long (along the
    courts) and wide: one per real module width, as many as fit at the
    least; 0 if a court does not fit this way round."""
    if length < MIN_COURT_L or width < MIN_COURT_W:
        return 0
    return max(1, min(round(width / 18.3), width // MIN_MODULE_W))


def tennis_fit(f: Frame) -> tuple[bool, int]:
    """How well courts fit a frame: full length first, then how many."""
    return f.L >= COURT_L + 4, courts_across(f.L, f.W)


def tennis(f: Frame, gate_side: str) -> tuple[list, list]:
    """Courts side by side across the pitch, each in its module."""
    tiles = fill(f, RED)
    edges = set()
    n = courts_across(f.L, f.W)
    mw = f.W // n
    cw = COURT_W if mw >= COURT_W + 4 else 8
    cl = min(COURT_L, (f.L - 4) // 2 * 2)
    s = min(SERVICE, round(cl * 6.4 / 23.77))
    u0 = (f.L - cl) // 2
    mid = u0 + cl // 2
    nets = []
    for k in range(n):
        v0 = k * mw + (mw - cw) // 2
        tiles += fill(f, GREEN, u0, v0, u0 + cl, v0 + cw)
        rect_lines(f, u0, v0, u0 + cl, v0 + cw, edges)
        for v in (v0 + 1, v0 + cw - 1):                         # singles sidelines
            for u in range(u0, u0 + cl):
                edges.add(f.along(u, v))
        for u in (mid - s, mid + s):                            # service lines
            for v in range(v0 + 1, v0 + cw - 1):
                edges.add(f.across(u, v))
        for u in range(mid - s, mid + s):                       # centre service line
            edges.add(f.along(u, v0 + cw // 2))
        nets += run_along_v(f, mid, v0, v0 + cw, (SPORTS + "49", SPORTS + "50", SPORTS + "51"),
                            (SPORTS + "54", SPORTS + "53", SPORTS + "52"))
    tiles += line_pieces(edges, (f.x0, f.y0, f.x1, f.y1), LINES_BRIGHT) + nets
    return tiles, [("tall_chainlink", ring_with_gate(f, gate_side))]


def football(f: Frame, tags: dict) -> tuple[list, list]:
    tiles = []
    if (tags.get("surface") or "") == "artificial_turf":
        tiles += fill(f, GREEN)
    elif hard(tags):
        tiles += fill(f, TARMAC)
    m = 1 if min(f.L, f.W) >= 20 else 0                          # run-off inside the mapped pitch
    L, Wd = f.L - 2 * m, f.W - 2 * m
    edges = set()
    rect_lines(f, m, m, m + L, m + Wd, edges)
    half = m + L // 2
    for v in range(m, m + Wd):
        edges.add(f.across(half, v))
    cv = m + Wd // 2
    for depth_m, width_m in ((16.5, 40.3), (5.5, 18.3)):          # penalty and goal areas
        d = max(1, round(L * depth_m / 105))
        w = min(Wd - 2, max(2, round(Wd * width_m / 68)))
        a, b = cv - w // 2, cv - w // 2 + w
        if L < 30 and depth_m < 10:
            continue
        rect_lines(f, m, a, m + d, b, edges)
        rect_lines(f, m + L - d, a, m + L, b, edges)
    r = max(2, round(Wd * 9.15 / 68))
    edges |= circle_edges(f, half, cv, r)
    tiles += line_pieces(edges, (f.x0, f.y0, f.x1, f.y1), LINES_BRIGHT)
    g = 6 if Wd >= 24 else 4
    for u in (m, m + L - 1):
        tiles += run_along_v(f, u, cv - g // 2, cv - g // 2 + g, (SPORTS + "19", SPORTS + "20", SPORTS + "21"),
                             (SPORTS + "18", SPORTS + "17", SPORTS + "16"))
    return tiles, []


def circle_edges(f: Frame, cu, cv, r) -> set:
    """A ring of radius r round the tile corner (cu, cv), stepped along tile edges."""
    inside = set()
    for u in range(cu - r - 1, cu + r + 1):
        for v in range(cv - r - 1, cv + r + 1):
            if (u + 0.5 - cu) ** 2 + (v + 0.5 - cv) ** 2 <= r * r:
                inside.add((u, v))
    edges = set()
    for u, v in inside:
        if (u - 1, v) not in inside:
            edges.add(f.across(u, v))
        if (u + 1, v) not in inside:
            edges.add(f.across(u + 1, v))
        if (u, v - 1) not in inside:
            edges.add(f.along(u, v))
        if (u, v + 1) not in inside:
            edges.add(f.along(u, v + 1))
    return edges


# Vanilla's centre circle, 4 x 4 round the tile corner (2, 2) of the stamp, as
# it lies on a court running along x; the circle is the same either way.
CIRCLE = {(0, 0): 112, (1, 0): 113, (2, 0): 114, (0, 1): 108, (2, 1): 115, (3, 1): 116,
          (0, 2): 109, (1, 2): 110, (2, 2): 118, (3, 2): 117, (1, 3): 111, (2, 3): 119}


def court(f: Frame) -> tuple[list, list]:
    tiles = fill(f, TARMAC)
    m = 1 if min(f.L, f.W) >= 9 else 0
    L, Wd = min(28, f.L - 2 * m), min(15, f.W - 2 * m)
    u0, v0 = (f.L - L) // 2, (f.W - Wd) // 2
    edges = set()
    rect_lines(f, u0, v0, u0 + L, v0 + Wd, edges)
    half, cv = u0 + L // 2, v0 + Wd // 2
    for v in range(v0, v0 + Wd):
        edges.add(f.across(half, v))
    kd, kw = max(2, round(L * 5.8 / 28)), max(2, round(Wd * 4.9 / 15))
    rect_lines(f, u0, cv - kw // 2, u0 + kd, cv - kw // 2 + kw, edges)
    rect_lines(f, u0 + L - kd, cv - kw // 2, u0 + L, cv - kw // 2 + kw, edges)
    tiles += line_pieces(edges, (f.x0, f.y0, f.x1, f.y1), LINES_FADED)
    for (a, b), t in CIRCLE.items():
        x, y = (f.tile(half - 2 + a, cv - 2 + b) if f.along_x else f.tile(half - 2 + b, cv - 2 + a))
        tiles.append((x, y, 0, "FloorGrime2", SPORTS + str(t)))
    # A hoop at each end, in the run-off past the end line where there is one.
    for u, end in ((u0 - 1 if u0 else 0, "start"), (u0 + L if u0 + L < f.L else f.L - 1, "end")):
        x, y = f.tile(u, cv)
        board = {(True, "start"): 25, (True, "end"): 27, (False, "start"): 24, (False, "end"): 26}[(f.along_x, end)]
        tiles += [(x, y, 0, "Furniture", SPORTS + "33"), (x, y, 1, "Furniture", SPORTS + str(board))]
    return tiles, []


def nets(f: Frame) -> tuple[list, list]:
    tiles = fill(f, GREEN)
    lanes = max(1, round(f.W / 3.5))
    runs = []
    pts = [f.corner(f.L, 0), f.corner(0, 0), f.corner(0, f.W), f.corner(f.L, f.W)]
    runs.append(("tall_chainlink", [pts]))
    for k in range(1, lanes):
        v = round(k * f.W / lanes)
        runs.append(("tall_chainlink", [[f.corner(0, v), f.corner(f.L, v)]]))
    return tiles, runs


def playground(f: Frame, gate_side: str) -> tuple[list, list]:
    tiles = []
    taken = np.zeros((f.y1 - f.y0, f.x1 - f.x0), bool)
    placed = 0
    k = 0
    # Kit in turn, each where it first fits with a tile clear all round,
    # scanning in from the corner; at most one of each kind per 150 m².
    want = max(2, min(len(KIT_ORDER), (f.L * f.W) // 60))
    for name in KIT_ORDER[:want]:
        stamp = KIT[name]
        sw = max(a for a, _ in stamp) + 1
        sh = max(b for _, b in stamp) + 1
        spot = None
        for y in range(1, taken.shape[0] - sh):
            for x in range(1, taken.shape[1] - sw):
                if not taken[y - 1:y + sh + 1, x - 1:x + sw + 1].any():
                    spot = (x, y)
                    break
            if spot:
                break
        if not spot:
            continue
        x, y = spot
        taken[y - 1:y + sh + 1, x - 1:x + sw + 1] = True
        for yy in range(y - 1, y + sh + 1):            # the pad
            for xx in range(x - 1, x + sw + 1):
                tiles.append((f.x0 + xx, f.y0 + yy, 0, "Floor", SAND))
        for (a, b), ids in stamp.items():
            for layer, t in zip(KIT_LAYERS, ids):
                tiles.append((f.x0 + x + a, f.y0 + y + b, 0, layer, PARK + str(t)))
        placed += 1
        k += 1
    return tiles, [("black_metal", ring_with_gate(f, gate_side))]


def ring_with_gate(f: Frame, side: str) -> list:
    """The pitch's outline as one open line, a tile left out as the gate two
    tiles in from a corner on the given side (map side: N, S, W, E)."""
    x0, y0, x1, y1 = f.x0, f.y0, f.x1, f.y1
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    start = {"N": 0, "E": 1, "S": 2, "W": 3}[side]
    a, b = ring[start], ring[(start + 1) % 4]
    dx, dy = (b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1])
    g0 = (a[0] + 2 * dx, a[1] + 2 * dy)
    g1 = (g0[0] + dx, g0[1] + dy)
    rest = [ring[(start + k) % 4] for k in range(1, 5)]
    return [[g1] + rest + [g0]]


# --- squaring ---------------------------------------------------------------------------------
def square_up(poly: Polygon, blocked, W, H):
    """The pitch as a grid rectangle: its own centre and size, turned to the
    nearer axis, then shrunk side by side off blocked tiles. None if it ends
    up too small."""
    mr = poly.minimum_rotated_rectangle
    xs, ys = mr.exterior.coords.xy
    e0 = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
    e1 = math.hypot(xs[2] - xs[1], ys[2] - ys[1])
    ang = math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0])) % 180
    first_along_x = ang < 45 or ang > 135
    wx, wy = (e0, e1) if first_along_x else (e1, e0)
    c = poly.centroid
    x0, y0 = int(round(c.x - wx / 2)), int(round(c.y - wy / 2))
    x1, y1 = x0 + int(round(wx)), y0 + int(round(wy))
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    want = (x1 - x0) * (y1 - y0)
    while x1 - x0 >= MIN_SIDE and y1 - y0 >= MIN_SIDE:
        sub = blocked[y0:y1, x0:x1]
        if not sub.any():
            break
        counts = {"W": sub[:, 0].sum(), "E": sub[:, -1].sum(), "N": sub[0, :].sum(), "S": sub[-1, :].sum()}
        side = max(counts, key=counts.get)
        if counts[side] == 0:
            # Blocked only inside: take the side nearest the blocked tiles.
            ry, rx = np.nonzero(sub)
            near = {"W": rx.min(), "E": sub.shape[1] - 1 - rx.max(),
                    "N": ry.min(), "S": sub.shape[0] - 1 - ry.max()}
            side = min(near, key=near.get)
        if side == "W":
            x0 += 1
        elif side == "E":
            x1 -= 1
        elif side == "N":
            y0 += 1
        else:
            y1 -= 1
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE or (x1 - x0) * (y1 - y0) < SHRINK_KEEP * want:
        return None
    return x0, y0, x1, y1


def gate_side(box, street) -> str:
    """The side of a box facing the nearest street."""
    x0, y0, x1, y1 = box
    H, W = street.shape
    best = None
    for side, (dx, dy), (sx, sy) in (("N", (0, -1), ((x0 + x1) // 2, y0)), ("S", (0, 1), ((x0 + x1) // 2, y1 - 1)),
                                     ("W", (-1, 0), (x0, (y0 + y1) // 2)), ("E", (1, 0), (x1 - 1, (y0 + y1) // 2))):
        for k in range(1, 80):
            x, y = sx + dx * k, sy + dy * k
            if not (0 <= x < W and 0 <= y < H):
                break
            if street[y, x]:
                if best is None or k < best[0]:
                    best = (k, side)
                break
    return best[1] if best else "S"


# --- main ---------------------------------------------------------------------------------------
def load_pitches(map_dir, info, proj):
    from generator import osm
    from generator.octilinear import straighten_roads
    from generator.renderer import _is_polygon, classify
    feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"]))
    if info.get("straight_roads"):
        straighten_roads(feats, proj, classify, _is_polygon, map_dir=map_dir)
    out = []
    for f in feats:
        tags = f.tags or {}
        kind = kind_of(tags)
        if not kind:
            continue
        if f.kind == "way" and _is_polygon(f):
            ring = f.geometry
        elif f.kind == "relation":
            rings = [r for role, r in (f.role_geoms or []) if role in ("outer", "")]
            ring = rings[0] if rings else None
        else:
            ring = None
        if not ring or len(ring) < 4:
            continue
        poly = Polygon([proj.to_px(la, lo) for la, lo in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        out.append((f.osm_id, kind, tags, poly))
    return out


def main() -> int:
    map_dir = sys.argv[1]
    picture = sys.argv[sys.argv.index("--picture") + 1] if "--picture" in sys.argv else None
    view = [int(v) for v in sys.argv[sys.argv.index("--box") + 1].split(",")] if "--box" in sys.argv else None
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    b = info["bbox"]
    proj = B.Projector.build(b["south"], b["west"], b["north"], b["east"], info["meters_per_tile"],
                             info.get("rotation") or 0.0)
    W, H = proj.width, proj.height
    P = np.load(os.path.join(map_dir, f"{name}_plots.npz"))
    building, street = P["building"], P["street"]

    pitches = load_pitches(map_dir, info, proj)
    # Biggest first: a playground drawn inside a park's pitch, or courts
    # inside a club's ground, give way to what is already there.
    pitches.sort(key=lambda p: -p[3].area)
    used = np.zeros((H, W), bool)
    tiles, fences, boxes = [], [], []
    beds = []                     # (real outline, box, ground colour) to repaint
    count = {}
    left_out = []
    for osm_id, kind, tags, poly in pitches:
        c = poly.centroid
        if not (0 <= c.x < W and 0 <= c.y < H):
            continue
        box = square_up(poly, building | street | used, W, H)
        if box is None:
            left_out.append((osm_id, kind))
            continue
        x0, y0, x1, y1 = box
        f = Frame(x0, y0, x1, y1)
        gate = gate_side(box, street)
        if kind == "tennis":
            # Courts along the pitch, or across it where more fit that way
            # (a club's three courts side by side along its longer side);
            # full-length courts first, so a single court's 36 x 18 m is one
            # court, not two short ones across it.
            other = Frame(x0, y0, x1, y1, along_x=not f.along_x)
            if tennis_fit(other) > tennis_fit(f):
                f = other
            if not courts_across(f.L, f.W):
                kind = "lawn"
            else:
                t, fl = tennis(f, gate)
        if kind == "football":
            t, fl = football(f, tags)
        elif kind == "court":
            t, fl = court(f)
        elif kind == "nets":
            t, fl = nets(f)
        elif kind == "playground":
            t, fl = playground(f, gate)
        elif kind == "lawn":
            t, fl = (fill(f, TARMAC) if hard(tags) else []), []
        tiles += t
        fences += [(st, line, osm_id) for st, lines in fl for line in lines]
        used[y0:y1, x0:x1] = True
        boxes.append((kind, box, osm_id))
        if kind == "playground":
            beds.append((poly, box, SAND_RGB))
        elif kind in ("football", "lawn") and not hard(tags) and tags.get("surface") != "artificial_turf":
            beds.append((poly, box, GRASS_RGB))
        count[kind] = count.get(kind, 0) + 1

    # --- written out ----------------------------------------------------------------------
    sp = os.path.join(map_dir, f"{name}_structures.json")
    orig = os.path.join(map_dir, f"{name}_structures.orig.json")
    base = {"bridges": 0, "monuments": 0, "tiles": []}
    if os.path.exists(sp):
        cur = json.load(open(sp, encoding="utf-8"))
        if not cur.get("sports_added"):
            # Generate's own file (a new one after a new Generate): keep it.
            shutil.copy(sp, orig)
            base = cur
        elif os.path.exists(orig):
            base = json.load(open(orig, encoding="utf-8"))
    out = dict(base)
    out["tiles"] = list(base.get("tiles") or []) + [list(t) for t in tiles]
    out["sports_added"] = len(tiles)
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(out, fh)

    fp = os.path.join(map_dir, f"{name}_fences.geojson")
    fc = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {"type": "FeatureCollection",
                                                                              "features": []}
    fenced = [box for kind, box, _ in boxes if kind in ("tennis", "playground", "nets")]

    def inside_fenced(ft) -> bool:
        coords = (ft.get("geometry") or {}).get("coordinates") or []
        if ft.get("geometry", {}).get("type") != "LineString" or len(coords) < 2:
            return False
        pts = [proj.to_px(lat, lon) for lon, lat in coords]
        on = sum(any(bx0 - 2 <= x <= bx1 + 2 and by0 - 2 <= y <= by1 + 2 for bx0, by0, bx1, by1 in fenced)
                 for x, y in pts)
        return on * 2 > len(pts)

    kept, dropped = [], 0
    for ft in fc.get("features", []):
        src = (ft.get("properties") or {}).get("source")
        if src == "sports":
            continue
        if src not in ("plots", "inspire") and inside_fenced(ft):
            dropped += 1
            continue
        kept.append(ft)
    for st, line, osm_id in fences:
        ring = [proj.to_latlon(px, py) for px, py in line]
        kept.append({"type": "Feature",
                     "properties": {"barrier": "fence", "knox_style": st, "source": "sports",
                                    "pitch": str(osm_id)},
                     "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in ring]}})
    fc["features"] = kept
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)

    # Trees and tufts off the pitches: nothing grows on a court or a pitch.
    cleared = 0
    for vp in (os.path.join(map_dir, f"{name}_veg_base.bmp"), os.path.join(map_dir, f"{name}_veg.bmp")):
        if not os.path.exists(vp):
            continue
        veg = np.array(Image.open(vp).convert("RGB"))
        for _k, (x0, y0, x1, y1), _o in boxes:
            sub = veg[y0:y1, x0:x1]
            cleared += int(sub.any(axis=2).sum())
            sub[:] = 0
        Image.fromarray(veg).save(vp, format="BMP")

    # Generate paints a pitch's real outline (sand for a playground, grass for
    # a pitch), which the squared box no longer matches: a grass pitch on a
    # school's paved yard came out half concrete. The box gets its own ground,
    # and a playground's sand that spilled past its railings goes back to grass.
    resanded = 0
    for gp in (os.path.join(map_dir, f"{name}.bmp"), os.path.join(map_dir, f"{name}_ground_base.bmp")):
        if not os.path.exists(gp) or not beds:
            continue
        ground = np.array(Image.open(gp).convert("RGB"))
        for poly, (x0, y0, x1, y1), rgb in beds:
            if rgb == SAND_RGB:
                mask = Image.new("1", (W, H), 0)
                ImageDraw.Draw(mask).polygon(list(poly.buffer(1).exterior.coords), fill=1)
                spill = np.array(mask)[:H, :W]
                spill[y0:y1, x0:x1] = False
                spill &= (ground[:H, :W] == SAND_RGB).all(axis=2)
                ground[:H, :W][spill] = GRASS_RGB
                resanded += int(spill.sum())
            ground[y0:y1, x0:x1] = rgb
        Image.fromarray(ground).save(gp, format="BMP")

    print(f"pitches on the map: {len(boxes)} {count}")
    print(f"playground sand outside the railings back to grass: {resanded // 2} tiles")
    print(f"left out (no room once squared): {len(left_out)} {left_out}")
    print(f"tiles: {len(tiles)}; fence lines: {len(fences)}; OSM fences round them left out: {dropped}")
    print(f"vegetation cleared: {cleared // 2} tiles")
    for kind, box, osm_id in boxes:
        x0, y0, x1, y1 = box
        print(f"  {kind:10} {osm_id:>11} at {x0},{y0} {x1 - x0}x{y1 - y0}")

    if picture:
        ground = np.array(Image.open(os.path.join(map_dir, f"{name}.bmp")).convert("RGB"))[:H, :W]
        img = Image.fromarray(ground)
        d = ImageDraw.Draw(img)
        colours = {"tennis": (200, 60, 40), "football": (255, 255, 255), "court": (60, 60, 200),
                   "nets": (20, 120, 20), "playground": (230, 200, 0), "lawn": (0, 200, 200)}
        for kind, (x0, y0, x1, y1), _o in boxes:
            d.rectangle((x0, y0, x1 - 1, y1 - 1), outline=colours[kind])
        for x, y, z, layer, t in tiles:
            if layer.startswith("Floor") and "trafficlines" in t:
                img.putpixel((x, y), (255, 255, 255))
        if view:
            vx, vy, vw, vh = view
            img = img.crop((vx, vy, vx + vw, vy + vh))
        img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
        img.save(picture)
        print(f"wrote {picture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
