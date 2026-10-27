"""Tiles from recipes: furniture, wall pieces, windows and floors described in
JSON and drawn in the game's projection. This is how a landmark gets the
tiles its listing and photos call for (a carved pulpit, a font, an arcade, a
traceried window) without anyone drawing them by hand: a recipe says what
the thing is made of and where, in metres, and this draws it.

    python recipe_kit.py [recipes/<sheet>.json ...]       (KnoxMap's Python; all of recipes/ by default)
    python recipe_kit.py recipes/x.json --preview out.png  (also a picture of every piece assembled;
                                                            --wall <wall_kit material>: the wall windows stand in)
    python recipe_kit.py --materials out.png               (every material and pattern on a sheet)

Writes, per recipe file (one sheet):
    Tile_creator/build/sheets/<group>/<sheet>.png, DEPTH_<sheet>.png, <sheet>.slots.json   (pack.py, register.py)
    recipes/built/<sheet>.json   what Build needs to place them: furniture roles, window entries, floors

A recipe file:

    {"sheet": "church_furnishings_uk_01",
     "about": "...", "evidence": ["the list text's words", "photos looked at (never copied)"],
     "items": [ {item}, ... ]}

Items (all lengths in metres; x east, y south, z up; a square is 1 m, a storey 3 m):

    {"name": "pew", "kind": "object", "size": [3, 1], "facings": ["N", "S", "E", "W"],
     "like": {"S": {"0,0": "location_community_church_small_01_51", ...}, ...} | "<vanilla tile>",
     "props": [["CustomName", "Pew"]], "parts": [ {part}, ... ]}
        Drawn facing south (its front towards +y) on squares [0, size[0]) x [0, size[1]); the
        other facings are that turned. "like" lends a vanilla tile's properties (sitting,
        collision, a pew's bed) square by square, or one tile for every square; "props" adds
        to them, or stands alone. Whatever rises above 3 m is cut into the storeys above
        (roles <name>@1, @2: Build lays them a floor up).
    {"name": "arcade", "kind": "edge", "length": 1, "layer": "WallOverlay", "parts": [...]}
        Stands on a square's west edge (the wall plane x = 0, along y), the north edge's
        made by mirroring; `length` squares along the edge.
    {"name": "perp3", "kind": "window", "shapes": [2, 1], "tracery": "perpendicular",
     "lights": 3, "glass": "glass_leaded", "frame": "ashlar"}
        A window across len(shapes) squares, each square's vanilla WindowShape (2 then 1:
        the two halves of an arch). Closed, smashed and glass gone, at vanilla church
        windows' offsets, so the game breaks it as it breaks theirs. Window sheets hold only
        windows ("windows": true on the file). tracery: "perpendicular", "intersecting",
        "reticulated", "square", "open", or "perp_square" (square-headed: a flat transom at each
        of "transoms": [m up from the hole's foot], cusped heads under the head - and under the
        transoms with "cusped_transoms" - its spandrels sunk, a label lit). "glass_line": a dark
        line where glass meets stone. "sub_arch"/"sub_head": pointed heads to the lights
        (radius x light width, height m). On a north edge the squares take the shapes reversed
        (vanilla mirrors its half-window shapes there): the record's "shapes_n". "cover":
        [[vanilla sheet, W piece, N piece], ...]: wall pieces whose hole the window also fills
        with its frame (Build lays the plain window piece walls_exterior_house_01_12/13 under a
        recipe window).
    {"name": "flags", "kind": "floor", "like": "floors_interior_tilesandwood_01_42", "mat": "flags"}

Parts:
    {"box": [x0, x1, y0, y1, z0, z1], "mat": ...}
    {"prism": {"c": [x, y], "r": r, "n": 8, "z": [z0, z1], "r_top": r, "rot": deg}, "mat": ...}
    {"pyramid": {"c": [x, y], "r": r, "n": 4, "z": [z0, z1], "rot": deg}, "mat": ...}
    {"spire": {... as pyramid, "crockets": k, "crocket": m, "finial": m, "crocket_mat": .., "finial_mat": ..}, "mat": ...}
        a pinnacle's or spire's cap: k crockets up each arris, a finial bud on its point
    {"gable": [x0, x1, y0, y1, z0, z1], "ridge": "x" | "y", "mat": ..., "end": ...}
    {"poly": [[x, y, z], ...], "mat": ...}                       one flat face, seen from either side
    {"panel": {"o": [x, y, z], "u": [..], "v": [..]}, "pattern": {"name": ..., ...}}
        a picture on a flat face: s along u, t along v (materials.PATTERNS)
    {"ruin_wall": {"a": [x, y], "b": [x, y], "thick": t, "top": [[s, z], ...], "arches": [[s0, s1, spring, rise]],
                   "rounds": [[s, z, r]], "putlogs": [z, ...], "seed": k}, "mat": ...}
        a ruined wall a -> b: its top height along it (s from 0 to 1), breaches and windows
        through it, rows of putlog holes; drawn both faces, its broken top and ends
  any part may carry "repeat": {"n": k, "step": [dx, dy, dz]} and "edges": false (no dark outline).
"mat" is a materials.MATERIALS name (or "kit:<wall_kit material>"), or {"name": ..., "rgb": [...], ...}.

The drawing: every face is rasterised into each square's sprite with a depth
buffer (walls.py's camera: x = 63 + 64 (wx - wy), y = 192 + 32 (wx + wy) - 64 z;
light by the face's normal, walls.py's EAST_LIGHT and TOP_LIGHT), and a pixel
belongs to the square and storey its surface point stands on - so a pew three
squares long is three sprites that meet, as vanilla's are. Each sprite's depth
picture is its own (walls.depth_tile).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import materials as M  # noqa: E402
import wall_kit as WK  # noqa: E402
import window_kit as WIN  # noqa: E402

W, F = WK.W, WK.F
RECIPES = os.path.normpath(os.path.join(HERE, "..", "recipes"))
BUILT = os.path.join(RECIPES, "built")
STOREY_M = 3.0
J = np.array([[64.0, -64.0, 0.0], [32.0, 32.0, -64.0]])
VIEW = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
EDGE_M = 0.012                  # the dark outline round a solid's faces
EDGE_DARK = 0.80
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
# Window sheets: vanilla church windows' layout (fixtures_windows_church: smashed +64, glass gone +96).
WINDOW_STATES = {"closed": 0, "smashed": 64, "noglass": 96}
WINDOW_SLOTS = 128


# ---- geometry: parts -> faces (canonical coordinates) ----

class Face:
    __slots__ = ("verts", "mat", "solid", "edges", "centre", "panel", "light")

    def __init__(self, verts, mat=None, solid=True, edges=True, centre=None, panel=None, light=1.0):
        self.verts = np.asarray(verts, float)
        self.mat, self.solid, self.edges = mat, solid, edges
        self.centre = None if centre is None else np.asarray(centre, float)
        self.panel, self.light = panel, light


def _box_faces(b, mat, edges, light):
    x0, x1, y0, y1, z0, z1 = b
    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    P = lambda x, y, z: (x, y, z)  # noqa: E731
    quads = [
        [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)],      # top
        [P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z0), P(x0, y1, z0)],      # bottom
        [P(x1, y0, z1), P(x1, y1, z1), P(x1, y1, z0), P(x1, y0, z0)],      # east
        [P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0), P(x0, y0, z0)],      # west
        [P(x0, y1, z1), P(x1, y1, z1), P(x1, y1, z0), P(x0, y1, z0)],      # south
        [P(x0, y0, z1), P(x1, y0, z1), P(x1, y0, z0), P(x0, y0, z0)],      # north
    ]
    return [Face(q, mat, True, edges, c, light=light) for q in quads]


def _ring(cx, cy, r, n, rot, z):
    return [(cx + r * math.cos(math.radians(rot + 360 * k / n)), cy + r * math.sin(math.radians(rot + 360 * k / n)), z)
            for k in range(n)]


def _prism_faces(p, mat, edges, light):
    cx, cy = p["c"]
    n, r = int(p.get("n", 8)), p["r"]
    rt = p.get("r_top", r)
    z0, z1 = p["z"]
    rot = p.get("rot", 180 / n)
    lo, hi = _ring(cx, cy, r, n, rot, z0), _ring(cx, cy, rt, n, rot, z1)
    c = (cx, cy, (z0 + z1) / 2)
    out = [Face([hi[k], hi[(k + 1) % n], lo[(k + 1) % n], lo[k]], mat, True, edges and n <= 12, c, light=light)
           for k in range(n)]
    if rt > 0:
        out.append(Face(hi, mat, True, edges, c, light=light))
    out.append(Face(lo, mat, True, edges, c, light=light))
    return out


def _pyramid_faces(p, mat, edges, light):
    cx, cy = p["c"]
    n, r = int(p.get("n", 4)), p["r"]
    z0, z1 = p["z"]
    rot = p.get("rot", 45 if n == 4 else 180 / n)
    lo = _ring(cx, cy, r, n, rot, z0)
    apex = (cx, cy, z1)
    c = (cx, cy, z0 + (z1 - z0) / 4)
    return [Face([apex, lo[(k + 1) % n], lo[k]], mat, True, edges, c, light=light) for k in range(n)]


def _spire_faces(p, mat, edges, light):
    """A pyramid (a pinnacle's or a spire's cap) with `crockets` knobs up each
    arris, `crocket` m across, and a finial bud `finial` m high on its point."""
    cx, cy = p["c"]
    n, r = int(p.get("n", 4)), p["r"]
    z0, z1 = p["z"]
    rot = p.get("rot", 45 if n == 4 else 180 / n)
    out = _pyramid_faces(dict(p, rot=rot), mat, edges, light)
    k = int(p.get("crockets", 0))
    cm = p.get("crocket_mat", mat)
    size = p.get("crocket", r * 0.3)
    for (bx, by, _bz) in _ring(cx, cy, r, n, rot, z0):
        for i in range(k):
            f = (i + 0.7) / (k + 0.6)                     # up the arris, none at its foot or point
            s = size * (1 - 0.4 * f)
            px, py, pz = bx + (cx - bx) * f, by + (cy - by) * f, z0 + (z1 - z0) * f
            ox, oy = (bx - cx) / r * s * 0.25, (by - cy) / r * s * 0.25        # a little proud of the arris
            # a leaf curling up off the arris: a small pyramid, its foot sunk in the slope
            out += _pyramid_faces({"c": [px + ox, py + oy], "r": s * 0.5, "n": 4, "z": [pz - s * 0.35, pz + s * 0.65]},
                                  cm, edges, light)
    fin = p.get("finial", 0)
    if fin:
        fm = p.get("finial_mat", cm)
        out += _prism_faces({"c": [cx, cy], "r": fin * 0.28, "r_top": fin * 0.2, "n": 8,
                             "z": [z1 - fin * 0.15, z1 + fin * 0.55]}, fm, edges, light)
        out += _pyramid_faces({"c": [cx, cy], "r": fin * 0.2, "n": 8, "z": [z1 + fin * 0.55, z1 + fin]},
                              fm, edges, light)
    return out


def _gable_faces(g, ridge, mat, end, edges, light):
    x0, x1, y0, y1, z0, z1 = g
    c = ((x0 + x1) / 2, (y0 + y1) / 2, z0 + (z1 - z0) / 3)
    if ridge == "x":
        ym = (y0 + y1) / 2
        slopes = [[(x0, ym, z1), (x1, ym, z1), (x1, y0, z0), (x0, y0, z0)],
                  [(x0, ym, z1), (x1, ym, z1), (x1, y1, z0), (x0, y1, z0)]]
        ends = [[(x0, y0, z0), (x0, ym, z1), (x0, y1, z0)], [(x1, y0, z0), (x1, ym, z1), (x1, y1, z0)]]
    else:
        xm = (x0 + x1) / 2
        slopes = [[(xm, y0, z1), (xm, y1, z1), (x0, y1, z0), (x0, y0, z0)],
                  [(xm, y0, z1), (xm, y1, z1), (x1, y1, z0), (x1, y0, z0)]]
        ends = [[(x0, y0, z0), (xm, y0, z1), (x1, y0, z0)], [(x0, y1, z0), (xm, y1, z1), (x1, y1, z0)]]
    return ([Face(s, mat, True, edges, c, light=light) for s in slopes]
            + [Face(e, end or mat, True, edges, c, light=light) for e in ends])


def _ruin_wall_faces(r, mat, light):
    """A ruined wall from a to b, `thick` through: both faces a picture of the
    wall (materials.ruin: its ragged top, breaches, putlog holes), its broken
    top across the thickness step by step, and its two broken ends. Given the
    `centre` of the polygon it is a side of and its `mitre` (half the corner's
    turn, 22.5 for an octagon), its faces are mitred to meet the next side's."""
    a, b = np.array(r["a"], float), np.array(r["b"], float)
    d = b - a
    length = float(np.hypot(*d))
    u = d / length
    n = np.array([d[1], -d[0]]) / length                 # across the wall
    half = r["thick"] / 2
    outer = 1
    if "centre" in r and np.dot((a + b) / 2 - np.array(r["centre"], float), n) < 0:
        outer = -1                                       # n points in: the other side is the outside
    ext = half * math.tan(math.radians(r.get("mitre", 0.0)))
    top = sorted(r["top"])                               # [[s 0..1, z], ...]
    zmax = max(z for _s, z in top)
    pat = {"name": "ruin", "top": top, "arches": r.get("arches", []), "rounds": r.get("rounds", []),
           "putlogs": r.get("putlogs", []), "mat": mat, "seed": r.get("seed", 0), "length": length}

    def p(s, z, side):
        grow = ext * (2 * s - 1) * (1 if side == outer else -1)
        q = a + d * s + u * grow + n * half * side
        return (q[0], q[1], z)
    faces = []
    for side in (1, -1):
        o = np.array(p(0, zmax, side))
        along = np.array(p(1, zmax, side)) - o
        grow = ext / length * (1 if side == outer else -1)        # this face's ends along the centre line
        faces.append(Face([o, o + along, o + along + [0, 0, -zmax], o + [0, 0, -zmax]], None, False, False,
                          panel={"o": o, "u": along, "v": np.array([0, 0, -zmax]),
                                 "pattern": dict(pat, s0=-grow, s1=1 + grow)}, light=light))
    for (s0, z0), (s1, z1) in zip(top, top[1:]):         # the broken top, across the thickness
        faces.append(Face([p(s0, z0, 1), p(s1, z1, 1), p(s1, z1, -1), p(s0, z0, -1)], mat, False, True, light=light))
    for s_end, z_end in (top[0], top[-1]):               # the broken ends
        if z_end > 0.05:
            faces.append(Face([p(s_end, z_end, 1), p(s_end, z_end, -1), p(s_end, 0, -1), p(s_end, 0, 1)],
                              mat, False, True, light=light))
    return faces


def faces_of(parts) -> list[Face]:
    out = []
    for part in parts:
        reps = part.get("repeat") or {"n": 1, "step": [0, 0, 0]}
        for k in range(int(reps["n"])):
            d = np.array(reps["step"], float) * k
            out += [_shift(f, d) for f in _part_faces(part)]
    return out


def _shift(f: Face, d) -> Face:
    if not d.any():
        return f
    panel = None
    if f.panel:
        panel = dict(f.panel, o=list(np.array(f.panel["o"], float) + d))
    return Face(f.verts + d, f.mat, f.solid, f.edges, None if f.centre is None else f.centre + d, panel, f.light)


def _part_faces(part):
    mat = part.get("mat", "stone")
    edges = part.get("edges", True)
    light = part.get("light", 1.0)
    if "box" in part:
        return _box_faces(part["box"], mat, edges, light)
    if "prism" in part:
        return _prism_faces(part["prism"], mat, edges, light)
    if "cyl" in part:
        return _prism_faces(dict(part["cyl"], n=part["cyl"].get("n", 16)), mat, False, light)
    if "pyramid" in part:
        return _pyramid_faces(part["pyramid"], mat, edges, light)
    if "spire" in part:
        return _spire_faces(part["spire"], mat, edges, light)
    if "gable" in part:
        return _gable_faces(part["gable"], part.get("ridge", "x"), mat, part.get("end"), edges, light)
    if "poly" in part:
        return [Face(part["poly"], mat, False, edges, light=light)]
    if "ruin_wall" in part:
        return _ruin_wall_faces(part["ruin_wall"], mat, light)
    if "panel" in part:
        p = part["panel"]
        o, u, v = (np.array(p[k], float) for k in ("o", "u", "v"))
        return [Face([o, o + u, o + u + v, o + v], None, False, False,
                     panel={"o": o, "u": u, "v": v, "pattern": part["pattern"]}, light=light)]
    raise ValueError(f"part with no shape: {sorted(part)}")


# ---- transforms ----

def facing_transform(facing: str, size):
    """Canonical (facing south) -> this facing: (3x3 matrix, offset), and the turned footprint."""
    w, d = size
    if facing == "S":
        return np.eye(3), np.zeros(3), (w, d)
    if facing == "N":
        return np.diag([-1.0, -1.0, 1.0]), np.array([w, d, 0.0]), (w, d)
    if facing == "E":                                   # front (+y) -> +x
        return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]), np.array([0, w, 0.0]), (d, w)
    if facing == "W":                                   # front (+y) -> -x
        return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]), np.array([d, 0, 0.0]), (d, w)
    raise ValueError(facing)


def side_transform(side: str):
    """An edge piece: drawn on the west edge; the north edge's is it mirrored (x <-> y)."""
    if side == "W":
        return np.eye(3), np.zeros(3)
    return np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]]), np.zeros(3)


# ---- rasterising ----

def _newell(v):
    n = np.zeros(3)
    for a, b in zip(v, np.roll(v, -1, 0)):
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else n


def _basis(n):
    """A face's texture axes: u along it (level), v down it; on a level face u east, v south."""
    if abs(n[2]) < 0.5:
        a = np.cross([0.0, 0.0, 1.0], n)
        a /= np.linalg.norm(a)
        b = np.cross(n, a)                     # down the face, in its plane (straight down on an upright one)
        return a, (b if b[2] <= 0 else -b)
    a = np.array([1.0, 0.0, 0.0]) - n * n[0]
    if np.linalg.norm(a) < 1e-6:
        a = np.array([0.0, 1.0, 0.0]) - n * n[1]
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    if b[1] < 0 and abs(n[2]) > 0.9:
        b = -b
    return a, b


def _light(n):
    return n[0] ** 2 * W.EAST_LIGHT + n[1] ** 2 * 1.0 + n[2] ** 2 * W.TOP_LIGHT


def _inside(px, py, poly):
    """Even-odd: are points (px, py) inside the polygon (k x 2)?"""
    inside = np.zeros(px.shape, bool)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, 0)):
        crosses = (y0 > py) != (y1 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (px < xi)
    return inside


def _edge_dist(px, py, poly):
    best = np.full(px.shape, 9.0)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, 0)):
        dx, dy = x1 - x0, y1 - y0
        L = dx * dx + dy * dy
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / L, 0, 1) if L > 0 else 0
        best = np.minimum(best, np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))
    return best


class Sprite:
    """One square's sprite on one storey, with its depth and the owner of each pixel."""

    def __init__(self):
        self.rgba = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
        self.depth = np.full((F.TILE_H, F.TILE_W), np.nan)
        self.z = np.full((F.TILE_H, F.TILE_W), np.inf)
        self.owner = np.full((F.TILE_H, F.TILE_W, 3), -999, np.int64)


def draw_faces(faces, R, t, local, rng, sprite: Sprite, footprint=None, storeys=1):
    """Rasterise faces (canonical) turned by R, t into the sprite of the square
    and storey at `local` (i, j, k): each pixel nearest the camera wins."""
    i, j, k = local
    shift = np.array([i, j, k * STOREY_M], float)
    ys, xs = np.mgrid[0:F.TILE_H, 0:F.TILE_W] + 0.5
    for f in faces:
        vw = f.verts @ R.T + t - shift                         # this sprite's square's coordinates
        n = _newell(vw)
        if not n.any():
            continue
        if f.solid and f.centre is not None:
            cw = f.centre @ R.T + t - shift
            if np.dot(n, vw.mean(0) - cw) < 0:
                n = -n
        facing_cam = np.dot(n, VIEW)
        if f.solid and facing_cam <= 1e-6:
            continue
        if not f.solid and facing_cam < 0:
            n = -n
        if f.panel:
            pan = f.panel
            ac, bc = pan["u"] / np.linalg.norm(pan["u"]), pan["v"] / np.linalg.norm(pan["v"])
        else:
            nc = _newell(f.verts)
            if f.solid and f.centre is not None and np.dot(nc, f.verts.mean(0) - f.centre) < 0:
                nc = -nc
            ac, bc = _basis(nc)
        o = f.verts[0]
        aw, bw = R @ ac, R @ bc
        ow = R @ o + t - shift
        S0 = np.array([63.0, 192.0]) + J @ ow
        Sa, Sb = J @ aw, J @ bw
        det = Sa[0] * Sb[1] - Sa[1] * Sb[0]
        if abs(det) < 1e-6:
            continue
        sv = (vw @ J.T) + np.array([63.0, 192.0])
        x0, x1 = int(max(0, np.floor(sv[:, 0].min()))), int(min(F.TILE_W, np.ceil(sv[:, 0].max()) + 1))
        y0, y1 = int(max(0, np.floor(sv[:, 1].min()))), int(min(F.TILE_H, np.ceil(sv[:, 1].max()) + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        px, py = xs[y0:y1, x0:x1] - S0[0], ys[y0:y1, x0:x1] - S0[1]
        al = (px * Sb[1] - py * Sb[0]) / det
        be = (py * Sa[0] - px * Sa[1]) / det
        rel = f.verts - o
        poly = np.stack([rel @ ac, rel @ bc], 1)
        inside = _inside(al, be, poly)
        if not inside.any():
            continue
        pw = ow[None, None, :] + al[..., None] * aw + be[..., None] * bw      # local world point
        dv = W.DEPTH_V0 - W.DEPTH_ALONG * (pw[..., 0] + pw[..., 1]) - W.DEPTH_UP * pw[..., 2]
        zb = sprite.z[y0:y1, x0:x1]
        win = inside & (dv < zb - 1e-6)
        if not win.any():
            continue
        a_, b_ = al[win], be[win]
        pc = o + a_[:, None] * ac + b_[:, None] * bc                        # canonical point (textures)
        if f.panel:
            lu, lv = np.linalg.norm(f.panel["u"]), np.linalg.norm(f.panel["v"])
            ss = a_ / lu
            if Sa[0] < -1e-6:
                # u runs right to left on screen (a west edge's +y, seen from the east):
                # the picture is read the other way, so it is the right way round
                ss = 1 - ss
            rgba = M.pattern(f.panel["pattern"])(ss, b_ / lv, lu, lv, f.panel["pattern"], rng)
            keep = rgba[:, 3] > 127
            col = rgba[:, :3]
        else:
            fn, spec = M.material(f.mat)
            col = fn(pc @ ac, pc @ bc, R.T @ n, spec)       # the canonical normal
            keep = np.ones(len(a_), bool)
            if f.edges:
                near = _edge_dist(a_, b_, poly) < EDGE_M
                col = np.where(near[:, None], col * EDGE_DARK, col)
        col = np.clip(col * _light(n) * f.light, 0, 255)
        idx = np.nonzero(win)
        idx = (idx[0][keep], idx[1][keep])
        if not len(idx[0]):
            continue
        yy, xx = idx[0] + y0, idx[1] + x0
        sprite.rgba[yy, xx, :3] = col[keep].astype(np.uint8)
        sprite.rgba[yy, xx, 3] = 255
        sprite.z[yy, xx] = dv[win][keep]
        sprite.depth[yy, xx] = dv[win][keep]
        p = pw[win][keep] + shift - n * 1e-4
        own = np.stack([np.floor(p[:, 0]), np.floor(p[:, 1]), np.floor(p[:, 2] / STOREY_M)], 1).astype(np.int64)
        if footprint is not None:                       # (i0, i1, j0, j1): the squares the piece covers
            own[:, 0] = np.clip(own[:, 0], footprint[0], footprint[1] - 1)
            own[:, 1] = np.clip(own[:, 1], footprint[2], footprint[3] - 1)
        own[:, 2] = np.clip(own[:, 2], 0, storeys - 1)
        sprite.owner[yy, xx] = own


def render_piece(faces, R, t, footprint, storeys, rng):
    """{(i, j, k): (rgba, depth)} for every square and storey of a piece: the
    squares it is declared on, and any its parts reach beyond them (an eaves, a
    pier straddling a wall line): a sprite only holds its own square's column,
    so an overhang drawn by the edge square was cut off."""
    out = {}
    pts = np.concatenate([f.verts @ R.T + t for f in faces])
    i0, j0 = min(0, int(math.floor(pts[:, 0].min() + 1e-6))), min(0, int(math.floor(pts[:, 1].min() + 1e-6)))
    i1 = max(footprint[0], int(math.ceil(pts[:, 0].max() - 1e-6)))
    j1 = max(footprint[1], int(math.ceil(pts[:, 1].max() - 1e-6)))
    span = (i0, i1, j0, j1)
    # each face's screen box at square (0, 0), storey 0: a sprite (i, j, k) only draws what lands in its 128 x 256
    boxes = []
    for f in faces:
        sv = (f.verts @ R.T + t) @ J.T + np.array([63.0, 192.0])
        boxes.append((sv[:, 0].min(), sv[:, 0].max(), sv[:, 1].min(), sv[:, 1].max()))
    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(storeys):
                dx, dy = 64 * (i - j), 32 * (i + j) - 192 * k
                near = [f for f, (x0, x1, y0, y1) in zip(faces, boxes)
                        if x1 - dx >= 0 and x0 - dx < F.TILE_W and y1 - dy >= 0 and y0 - dy < F.TILE_H]
                if not near:
                    continue
                s = Sprite()
                draw_faces(near, R, t, (i, j, k), rng, s, span, storeys)
                mine = (s.owner[..., 0] == i) & (s.owner[..., 1] == j) & (s.owner[..., 2] == k)
                # a pixel whose own square's sprite cannot hold it (it lies a column past that
                # canvas: a face meeting the square's corner edge-on) stays in this one, or it
                # is lost from both and leaves a see-through slit
                oi, oj, ok = s.owner[..., 0], s.owner[..., 1], s.owner[..., 2]
                drawn = s.rgba[..., 3] > 0
                ys_, xs_ = np.mgrid[0:F.TILE_H, 0:F.TILE_W]
                px_ = xs_ + 64 * ((i - j) - (oi - oj))
                py_ = ys_ + 32 * ((i + j) - (oi + oj)) - 192 * (k - ok)
                if mine.any():                         # (only into a sprite there is anyway: no new tiles)
                    mine |= drawn & ~mine & ((px_ < 0) | (px_ >= F.TILE_W) | (py_ < 0) | (py_ >= F.TILE_H))
                s.rgba[~mine] = 0
                s.depth[~mine] = np.nan
                if s.rgba[..., 3].any():
                    out[(i, j, k)] = (s.rgba, s.depth)
    return out


def _storeys(faces):
    top = max(f.verts[:, 2].max() for f in faces)
    return max(1, int(math.ceil(top / STOREY_M - 1e-6)))


# ---- windows ----

def window_texture(item, state, rng, side="W"):
    """The window across its squares as one wall texture (64 px a metre, 192 rows a storey),
    cut to the holes of its shapes on that side's wall."""
    shapes = item["shapes"]
    n = len(shapes)
    mask = np.zeros((192, 64 * n), bool)
    for k, sh in enumerate(shapes):
        mask[:, 64 * k:64 * k + 64] = WIN.outline(sh, side)
    if n > 1:
        # the squares' holes stop a few pixels short of their shared edge, and the
        # shaped wall pieces have no wall there either (vanilla's big windows draw
        # their own mullion over it): the window covers the slit, top to bottom
        from scipy import ndimage
        near = np.zeros(mask.shape, bool)
        for k in range(1, n):
            near[:, 64 * k - 10:64 * k + 10] = True
        mask |= ndimage.binary_closing(mask, structure=np.ones((1, 21))) & near
    ys, xs = np.nonzero(mask)
    c0, c1, r0, r1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rows, cols = np.mgrid[0:192, 0:64 * n] + 0.5
    u, v = cols / 64.0, rows / 64.0
    span = (c1 - c0) / 64.0
    mid = (c0 + c1) / 2 / 64.0
    x = u - mid
    tracery = item.get("tracery", "perpendicular")
    lights = int(item.get("lights", 3))
    fw = 0.045                                              # the stone frame's face
    # A window two storeys tall is two window items in the same squares: its `part`
    # "lower" (straight lights, its sill `sill` m up, the wall below that) on one storey,
    # "upper" (the lights running on from below, the head) on the storey above.
    part = item.get("part")
    sill = float(item.get("sill", 0.0))
    y = (r1 / 64.0) - v                                     # up from the bottom of the hole
    foot = 0.0 if part == "upper" else sill + fw
    if part == "lower":
        head_y = 99.0                                       # no head on this storey
        opening = (np.abs(x) <= span / 2 - fw) & (y >= foot)
    elif tracery == "open":
        # one round-headed opening, no lights or tracery (a louvred bell opening);
        # the wall-hole's own head round it stays stone: the voussoirs
        ro = span / 2 - 2 * fw
        head_y = (r1 - r0) / 64.0 - span / 2 - fw         # the springing
        opening = (np.abs(x) <= ro) & (y >= foot) \
            & (y <= head_y + np.sqrt(np.clip(ro * ro - x * x, 0, None)))
    elif tracery in ("square", "perp_square"):
        head_y = (r1 - r0) / 64.0                          # no head: a square frame
        opening = (np.abs(x) <= span / 2 - fw) & (y >= foot) & (y <= head_y - fw)
    else:
        ratio = item.get("arch", 0.75 if lights >= 3 else 1.0)
        r = span * ratio
        c = r - span / 2
        rise = math.sqrt(max(r * r - c * c, 0))
        head_y = (r1 - r0) / 64.0 - rise                  # springing, up from the bottom of the hole
        opening = _pointed(x, y, span - 2 * fw, head_y, r - fw, foot)
    frame_fn, fspec = M.material(item.get("frame", "ashlar"))
    glass_fn, gspec = M.material(item.get("glass", "glass_leaded"))
    nrm = np.array([1.0, 0.0, 0.0])
    tex = np.zeros((192, 64 * n, 4))
    tex[mask, :3] = frame_fn(u[mask], v[mask], nrm, fspec)
    tex[mask, 3] = 255
    if sill > 0:                                            # the wall under the sill, then the sill
        wall_fn, wspec = M.material(item.get("wall", "kit:flint_church"))
        under = mask & (y < sill - 0.06)
        tex[under, :3] = wall_fn(u[under], v[under], nrm, wspec)
    # mullions, a transom at the springing in Perpendicular work, sub-arches heading each light
    lw = (span - 2 * fw) / lights
    stone = np.zeros(mask.shape, bool)
    sunk = np.zeros(mask.shape, bool)                      # stone set back (spandrels): a shade darker
    lit = np.zeros(mask.shape, bool)                       # stone catching the light (a label)
    xl = x + span / 2 - fw                                 # across the opening, from its left
    for m in range(1, lights):
        stone |= np.abs(xl - m * lw) < 0.022
    k = np.clip(np.floor(xl / lw), 0, lights - 1)
    lx = xl - (k + 0.5) * lw                               # across this light, from its middle
    if tracery == "perp_square":
        # Barry's square-headed Perpendicular lights: a flat transom at each of
        # `transoms` (m up from the hole's foot on this storey; "cusped_transoms": the
        # lights under them headed too), and under the square head (not on a lower
        # part) each light's cusped four-centred head, the spandrels sunk, a label over
        tops = [float(tz) for tz in item.get("transoms", [])]
        for tz in tops:
            stone |= np.abs(y - tz) < 0.025
        heads = tops if item.get("cusped_transoms") else []
        if part != "lower":
            heads = heads + [head_y - fw]
        hh = float(item.get("head", 0.22))
        for tz in heads:
            sub = M.arch_opening(lx, y - (tz - hh), lw - 0.044, 0.0, "tudor", hh - 0.03)
            spandrel = (y < tz) & (y > tz - hh) & ~sub & opening
            stone |= spandrel
            sunk |= spandrel & (np.abs(lx) < lw / 2 - 0.03) & (y < tz - 0.02)
        if part != "lower":
            label = mask & (y > head_y - fw - 0.01)       # the square head's label moulding, lit
            lit |= label
    elif tracery not in ("square", "open") and part != "lower":
        hs = 0.38
        if "sub_arch" in item:                             # pointed heads to the lights: radius sub_arch x light
            rs, half = lw * float(item["sub_arch"]), (lw - 0.044) / 2
            hs = float(item.get("sub_head", 0.38))
            sub = _pointed(lx, y - (head_y - hs), lw - 0.044, hs - math.sqrt(max(rs * rs - (rs - half) ** 2, 0)), rs, 0.0)
        else:
            sub = _pointed(lx, y - (head_y - 0.38), lw - 0.044, 0.38 - lw * 0.55, lw * 0.6, 0.0)
        below = y < head_y
        stone |= below & (y > head_y - hs) & ~sub
        if tracery == "perpendicular":
            stone |= (np.abs(y - head_y) < 0.02)           # the transom
            panel = lw / 2
            xp = x + span / 2 - fw
            stone |= (y > head_y) & (np.abs(np.mod(xp + panel / 2, panel) - panel / 2) < 0.016) \
                & (np.abs(np.mod(xp, lw) - lw / 2) > 0.02)   # supermullions, half-panels
            cusp = (y > head_y + 0.02) & (np.mod(y - head_y, 0.32) < 0.025)
            stone |= cusp
        elif tracery == "intersecting":
            for m in range(0, lights + 1):
                off = m * lw - (span / 2 - fw)
                d = np.hypot(np.abs(x - off) + (r - fw - span / 2), np.maximum(y - head_y, 0))
                stone |= (y > head_y) & (np.abs(d - (r - fw)) < 0.022)
        elif tracery == "reticulated":
            cy = head_y + 0.28
            for m in range(lights):
                cx = -span / 2 + fw + (m + 0.5) * lw
                d = np.hypot(x - cx, y - cy)
                stone |= (y > head_y) & (np.abs(d - lw * 0.38) < 0.02)
    else:
        sub = (np.abs(lx) < lw / 2 - 0.022) & (y < head_y - fw - 0.06) | (y < head_y - fw) & (y > head_y - fw - 0.08) \
            & (np.abs(lx) < lw / 2 - 0.022) & ((np.abs(lx) < lw / 2 - 0.06) | (y < head_y - fw - 0.04))
        stone |= ~sub
    glass = opening & ~stone & mask
    tex[sunk & mask, :3] *= 0.78
    tex[lit & ~glass, :3] = np.clip(tex[lit & ~glass, :3] * 1.12, 0, 255)
    if item.get("glass_line", False):                      # a dark line where the glass meets the stone
        from scipy import ndimage
        line = mask & ~glass & ndimage.binary_dilation(glass, iterations=1)
        tex[line, :3] *= 0.62
    if state == "noglass":
        tex[glass, 3] = 0
    else:
        g = glass_fn(u[glass], v[glass], nrm, gspec)
        tex[glass, :3] = g
        if state == "smashed":
            keep = (rng.random(glass.sum()) < 0.10)
            edge = np.zeros(mask.shape, bool)
            from scipy import ndimage
            edge = glass & ndimage.binary_dilation(~glass & mask, iterations=2)
            gone = ~(keep | edge[glass])
            vals = tex[glass]
            vals[gone, 3] = 0
            tex[glass] = vals
    # "cover": [[vanilla sheet, W piece, N piece], ...] - wall pieces whose hole the window must
    # also fill (Build lays the plain window piece, walls_exterior_house_01_12/13, under a recipe
    # window: its hole reaches past a shape's at the jambs and showed the room through): each
    # square's part of their hole outside the window's own is the stone frame
    if item.get("cover"):
        extra = np.zeros(mask.shape, bool)
        for sheet, iw, i_n in item["cover"]:
            hole = WIN.piece_hole(sheet, iw if side == "W" else i_n, side)
            for k in range(n):
                extra[:, 64 * k:64 * k + 64] |= hole
        extra &= ~mask
        tex[extra, :3] = frame_fn(u[extra], v[extra], nrm, fspec)
        tex[extra, 3] = 255
    # reveal: a shade where the frame meets the glass
    tex[..., :3] = np.clip(tex[..., :3], 0, 255)
    return tex.astype(np.uint8)


def _pointed(x, y, span, spring, r, foot):
    """Inside a two-centred arch of radius r over an opening span wide, springing at `spring` (y up from the sill)."""
    half = span / 2
    c = r - half
    return (np.abs(x) <= half) & (y >= foot) & ((y <= spring) | ((np.abs(x) + c) ** 2 + (y - spring) ** 2 <= r * r))


# ---- items -> sprites ----

def build_item(item, rng):
    """[(slot name, rgba, depth, slot entry)] and the item's Build record."""
    kind = item.get("kind", "object")
    out, record = [], {"kind": kind}
    if kind == "floor":
        faces = [Face([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], item["mat"], False, False)]
        s = Sprite()
        draw_faces(faces, np.eye(3), np.zeros(3), (0, 0, 0), rng, s, (0, 1, 0, 1), 1)
        out.append(("floor", s.rgba, s.depth, {"like": item["like"], "props": item.get("props", [])}))
        record["tile"] = 0
        return out, record
    if kind == "object":
        faces = faces_of(item["parts"])
        storeys = _storeys(faces)
        record.update(layer=item.get("layer", "Furniture"), tiles={}, storeys=storeys)
        for facing in item.get("facings", ["S"]):
            R, t, fp = facing_transform(facing, item["size"])
            orient = OPPOSITE[facing]
            sprites = render_piece(faces, R, t, fp, storeys, rng)
            like = item.get("like")
            for (i, j, k), (rgba, depth) in sorted(sprites.items()):
                key = f"{i},{j}"
                if isinstance(like, dict):
                    lk = (like.get(facing) or {}).get(key) if k == 0 else None
                else:
                    lk = like if k == 0 else None
                entry = _slot(lk, item, k)
                record["tiles"].setdefault(str(k), {}).setdefault(orient, {})[key] = len(out)
                out.append((f"{facing}{key}@{k}", rgba, depth, entry))
        return out, record
    if kind == "edge":
        faces = faces_of(item["parts"])
        storeys = _storeys(faces)
        n = int(item.get("length", 1))
        record.update(layer=item.get("layer", "WallOverlay"), tiles={}, storeys=storeys)
        for side in item.get("sides", ["W", "N"]):
            R, t = side_transform(side)
            fp = (1, n) if side == "W" else (n, 1)
            sprites = render_piece(faces, R, t, fp, storeys, rng)
            for (i, j, k), (rgba, depth) in sorted(sprites.items()):
                key = f"{i},{j}"
                record["tiles"].setdefault(str(k), {}).setdefault(side, {})[key] = len(out)
                out.append((f"{side}{key}@{k}", rgba, depth, _slot(item.get("like") if k == 0 else None, item, k)))
        return out, record
    raise ValueError(f"{item.get('name')}: kind {kind!r}")


def _slot(like, item, storey):
    props = [list(p) for p in item.get("props", [])]
    if storey and not like:
        props = [p for p in props if p[0] not in ("solid", "solidtrans")]
    if like:
        return {"like": like, "props": props, "drop": item.get("drop", [])}
    return {"props": props}


def build_windows(items, rng):
    """A window sheet: per window and square, W and N, closed / smashed / glass gone."""
    tiles = [None] * WINDOW_SLOTS
    slots = [None] * WINDOW_SLOTS
    records = {}
    base = 0
    for item in items:
        shapes = item["shapes"]
        # Vanilla's half-window shapes are mirrored on a north wall (shape 12's hole is at
        # the square's east end on a west wall, its west end on a north one), and Build lays
        # square k at x + k: so along a north edge the squares take the shapes in reverse
        # order (11 then 12), their picture drawn to match.
        shapes_n = list(reversed(shapes))
        rec = {"kind": "window", "squares": [], "shapes": list(shapes), "shapes_n": shapes_n}
        texs = {st: window_texture(item, st, rng) for st in WINDOW_STATES}
        texs_n = texs if shapes_n == list(shapes) else \
            {st: window_texture(dict(item, shapes=shapes_n), st, rng, "N") for st in WINDOW_STATES}
        for k in range(len(shapes)):
            pair = {}
            for jj, side in enumerate(("W", "N")):
                sh = shapes[k] if side == "W" else shapes_n[k]
                slot = base + 2 * k + jj
                if slot + WINDOW_STATES["noglass"] >= WINDOW_SLOTS:
                    raise ValueError(f"{item['name']}: a window sheet holds 16 window squares (start another sheet)")
                for st, off in WINDOW_STATES.items():
                    tex = (texs if side == "W" else texs_n)[st][:, 64 * k:64 * k + 64]
                    tiles[slot + off] = WIN.project(np.ascontiguousarray(tex), side)
                    slots[slot + off] = {"like": f"fixtures_windows_church_{2 * (sh - 1) + jj + off}",
                                         "WindowShape": sh, "name": item["name"]}
                pair[side] = slot
            rec["squares"].append(pair)
        base += 2 * len(shapes)
        records[item["name"]] = rec
    return tiles, slots, records


def build_sheet(recipe) -> dict:
    sheet = recipe["sheet"]
    rng = np.random.default_rng(sum(map(ord, sheet)))
    blank = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
    if recipe.get("windows"):
        tiles, slots, records = build_windows(recipe["items"], rng)
        tiles = [t if t is not None else blank for t in tiles]
        depths = [np.full((F.TILE_H, F.TILE_W), np.nan)] * len(tiles)
        for k, t in enumerate(tiles):
            if slots[k]:
                depths[k] = _wall_depth(slots[k]["like"])
    else:
        tiles, depths, slots, records = [], [], [], {}
        for item in recipe["items"]:
            made, rec = build_item(item, rng)
            first = len(tiles)
            for _name, rgba, depth, entry in made:
                tiles.append(rgba)
                depths.append(depth)
                slots.append(entry)
            rec = _offset(rec, first)
            records[item["name"]] = rec
    WK.F.write_sheet(WK.sheet_file(sheet + ".png"), tiles)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{sheet}.png"),
                     [W.depth_tile(t, d) for t, d in zip(tiles, depths)])
    with open(WK.sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=0)
    built = {"sheet": sheet, "items": _named(records, sheet)}
    os.makedirs(BUILT, exist_ok=True)
    with open(os.path.join(BUILT, sheet + ".json"), "w", encoding="utf-8") as f:
        json.dump(built, f, indent=1)
    return {"sheet": sheet, "tiles": tiles, "built": built}


def _wall_depth(like):
    """A window's depth: the wall plane's (it lies in the west or north face)."""
    side = "N" if like and int(like.rsplit("_", 1)[1]) % 2 else "W"
    _u, _v, panel, depth = WK.FACES[side]
    return np.where(panel, depth, np.nan)


def _offset(rec, first):
    if "tile" in rec:
        rec["tile"] += first
    for lvl in (rec.get("tiles") or {}).values():
        for orient in lvl.values():
            for key in orient:
                orient[key] += first
    return rec


def _named(records, sheet):
    """Slot numbers -> tile names."""
    def nm(k):
        return f"{sheet}_{k}"
    out = {}
    for name, rec in records.items():
        r = dict(rec)
        if "tile" in r:
            r["tile"] = nm(r["tile"])
        if "tiles" in r:
            r["tiles"] = {lvl: {o: {k: nm(v) for k, v in sq.items()} for o, sq in orients.items()}
                          for lvl, orients in r["tiles"].items()}
        if "squares" in r:
            r["squares"] = [{s: nm(v) for s, v in sq.items()} for sq in r["squares"]]
        out[name] = r
    return out


# ---- previews ----

def preview(built, path, wall="walls_house_uk_01_flint"):
    """Every item put together: objects in each facing, edges W and N, windows on a wall, floors."""
    sheet = Image.open(WK.sheet_file(built["sheet"] + ".png")).convert("RGBA")
    cols = sheet.width // 128

    def tile(name):
        k = int(name.rsplit("_", 1)[1])
        r, c = divmod(k, cols)
        return sheet.crop((c * 128, r * 256, c * 128 + 128, r * 256 + 256))

    # a window stands in its wall: the wall kit's piece cut to its WindowShape
    wall_png = WK.sheet_file(wall + ".png")
    walls = Image.open(wall_png).convert("RGBA") if os.path.exists(wall_png) else None

    def shaped(shape, side):
        k = WK.FIRST["shaped"] + 2 * (shape - 1) + (0 if side == "W" else 1)
        r, c = divmod(k, walls.width // 128)
        return walls.crop((c * 128, r * 256, c * 128 + 128, r * 256 + 256))

    groups = []
    for name, rec in built["items"].items():
        placed = []
        if rec["kind"] == "floor":
            for i in range(2):
                for j in range(2):
                    placed.append((i, j, 0, rec["tile"]))
        elif rec["kind"] == "window":
            for side in ("W", "N"):
                for k, sq in enumerate(rec["squares"]):
                    at = (0, k) if side == "W" else (k + len(rec["squares"]) + 1, 0)
                    if walls is not None:
                        placed.append(at + (0, ("wall", rec.get("shapes_n" if side == "N" else "shapes",
                                                                 rec["shapes"])[k], side)))
                    placed.append(at + (0, sq[side]))
        else:
            off = 0
            for orient in ("S", "W", "N", "E"):
                for lvl, orients in rec["tiles"].items():
                    for key, tname in (orients.get(orient) or {}).items():
                        i, j = map(int, key.split(","))
                        placed.append((i + off, j, int(lvl), tname))
                if any(orient in o for o in rec["tiles"].values()):
                    off += 4
        groups.append((name, placed))
    pics = []
    for name, placed in groups:
        if not placed:
            continue
        xs = [64 * (i - j) for i, j, _k, _t in placed]
        ys = [32 * (i + j) - 192 * k for i, j, k, _t in placed]
        img = Image.new("RGBA", (max(xs) - min(xs) + 128, max(ys) - min(ys) + 256), (74, 84, 72, 255))
        for (i, j, k, tname), x, y in sorted(zip(placed, xs, ys), key=lambda p: (p[0][2], p[0][0] + p[0][1], not isinstance(p[0][3], tuple))):
            img.alpha_composite(shaped(*tname[1:]) if isinstance(tname, tuple) else tile(tname),
                                (x - min(xs), y - min(ys)))
        bg = np.asarray(img)[..., :3].astype(int) != np.array([74, 84, 72])
        yy, xx = np.nonzero(bg.any(-1))
        if len(yy):
            img = img.crop((max(0, xx.min() - 6), max(0, yy.min() - 6), xx.max() + 7, yy.max() + 7))
        pics.append((name, img))
    # packed in rows, each piece over its name
    from PIL import ImageDraw
    width, rows, row, x, h = 1100, [], [], 0, 0
    for name, img in pics:
        if row and x + img.width > width:
            rows.append((row, h))
            row, x, h = [], 0, 0
        row.append((x, name, img))
        x += img.width + 10
        h = max(h, img.height + 14)
    if row:
        rows.append((row, h))
    out = Image.new("RGBA", (width, sum(h for _r, h in rows) + 4), (40, 44, 40, 255))
    d = ImageDraw.Draw(out)
    y = 0
    for row, h in rows:
        for x, name, img in row:
            d.text((x + 2, y), name, fill=(230, 230, 220, 255))
            out.alpha_composite(img, (x, y + 14))
        y += h
    out.save(path)
    return path


def materials_sheet(path):
    n = len(M.MATERIALS) + len(M.PATTERNS)
    img = Image.new("RGBA", (128 * 8, 256 * math.ceil(n / 8)), (60, 60, 60, 255))
    rng = np.random.default_rng(0)
    for k, name in enumerate(list(M.MATERIALS) + list(M.PATTERNS)):
        s = Sprite()
        if name in M.MATERIALS:
            faces = _box_faces([0.1, 0.9, 0.1, 0.9, 0, 1.6], name, True, 1.0)
        else:
            faces = [Face([(0.05, 0.5, 2.6), (0.95, 0.5, 2.6), (0.95, 0.5, 0.2), (0.05, 0.5, 0.2)], None, False, False,
                          panel={"o": np.array([0.05, 0.5, 2.6]), "u": np.array([0.9, 0, 0]),
                                 "v": np.array([0, 0, -2.4]), "pattern": {"name": name}})]
        draw_faces(faces, np.eye(3), np.zeros(3), (0, 0, 0), rng, s)
        r, c = divmod(k, 8)
        img.alpha_composite(Image.fromarray(s.rgba), (c * 128, r * 256))
    img.save(path)
    return path


def recipe_files(paths=None):
    if paths:
        return paths
    # KnoxMapUK_Tiles' recipe sheets, then KnoxMapUK_Westminster's (Tile_creator's pack.py)
    order = [s for f in ("sheets.order.json", "westminster.order.json")
             for s in json.load(open(os.path.join(RECIPES, f), encoding="utf-8"))]
    # a landmark's own wall kit sits in the order too (its tileset id is its place: Parliament's
    # ashlar, 2026-10-22), but wall_kit.py draws it, not a recipe
    return [os.path.join(RECIPES, s + ".json") for s in order if not s.startswith(("walls_house_", "roofs_"))]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recipes", nargs="*")
    ap.add_argument("--preview", help="write a picture of the pieces (one recipe) here")
    ap.add_argument("--materials", help="write a sheet of every material and pattern here")
    ap.add_argument("--wall", default="flint", help="the wall kit windows stand in on the preview (wall_kit material)")
    a = ap.parse_args(argv)
    if a.materials:
        print("wrote", materials_sheet(a.materials))
        if not a.recipes:
            return
    for path in recipe_files(a.recipes):
        recipe = json.load(open(path, encoding="utf-8"))
        made = build_sheet(recipe)
        print("wrote", made["sheet"], len(made["tiles"]), "sprites", list(made["built"]["items"]))
        if a.preview:
            out = a.preview if len(recipe_files(a.recipes)) == 1 else a.preview.replace(".png", f"_{made['sheet']}.png")
            print("preview", preview(made["built"], out, f"walls_house_uk_01_{a.wall}"))
        else:
            preview(made["built"], os.path.join(WK.PREVIEW, f"recipe_{made['sheet']}.png"))


if __name__ == "__main__":
    main()
