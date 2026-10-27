"""The last American pieces of the street, and the countryside's: rendered at
the game's camera (street_objects.py's), as its sheets are.

    "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b -P streetscape.py -- [sheet ...] [slot ...]

- street_lighting_uk_01: the 1990s street lamp - a grey concrete column, a
  bracket arm, a low-pressure sodium lantern (its bowl the orange of SOX) -
  its arm reaching N, E, S, W as vanilla's lighting_outdoor_01_8-11, which
  register.py swaps it for in the map tools' rules.
- road_markings_uk_01: double yellow lines on a square's W, N, E, S edge, and
  the L where a kerb turns (NW, NE, SE, SW): two lines 0.075 m wide, 0.075 m
  apart, the outer 0.06 m from the kerb (TSRGD 1994, drawn at vanilla's scale).
- road_markings_uk_02: the UK's road markings of one and two tiles
  (marking_faces.py draws them; each a flat picture on the road, its
  pieces one a square): a give-way triangle, the 30 and 40 roundels, lane
  arrows (ahead, left, right, ahead or left, ahead or right) and the
  deflection arrow, 1 x 2 each, and SLOW, 2 x 2 - each for a driver heading
  N, E, S, W, its pieces (gx, gy) from its north-west square; a zebra's
  black and white bands (the road N-S or E-W), its zig-zags along a W, N, E
  or S kerb, a yellow box (hatching, an edge, a corner), double white lines
  (W or N edge, as vanilla's centre line) and a mini roundabout's disc
  (2 x 2), as they lie.
- street_drains_uk_01: a road gully's square iron grate in the gutter by the
  kerb on the W, N, E, S edge (marking_faces.py draws it).
- countryside_uk_01: a five-bar field gate, wooden and galvanised (closed W,
  closed N, open W, open N, as vanilla's fence gates); a stile (W, N); a
  kissing gate (W, N); a telegraph pole in two storeys (its foot, and its
  top with the crossarm across wires running E-W or N-S) and the wires
  between poles (E-W, N-S), a storey up.
- street_curbs_uk_01, _diag, _diag_2: UK concrete kerbs in exactly vanilla's
  224 shapes, so kerbs.py's matching (measure_kerbs.py's kerbs_topdown.json)
  carries over: each piece's outline from its measured points (kerb_shapes.py:
  traced, thinned to its centre line, a 60 mm band along it) raised 0.11 m
  (a kerb's upstand) as a heightfield. Vanilla's sprites give only where the kerb lies (measured,
  not traced): the kerb itself is ours.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import street_objects as so  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # asset_creator: acpaths
import acpaths  # noqa: E402
KERBS = acpaths.KERBS_TOPDOWN
LEVEL = 3.0                                   # m a storey: a sprite a level up starts this high
COLUMN = (150, 150, 144)                      # weathered concrete
SOX = (255, 150, 60)                          # a low-pressure sodium lantern's bowl
YELLOW = (238, 196, 40)                       # road-marking yellow, a little worn
KERB = (168, 166, 158)
TIMBER = (128, 104, 72)                       # weathered oak / tanalised softwood
GALV = (160, 166, 168)


# --- the street lamp ------------------------------------------------------------------------------
def street_lamp():
    """Built with its arm reaching south (turned to each facing)."""
    grey = so.material("sl_concrete", COLUMN, grime=0.12)
    c = 0.5
    so.cylinder(c, c, 0.0, 0.085, 0.10, grey, sides=8)                 # the root
    so.cylinder(c, c, 0.10, 0.070, 3.00, grey, sides=8, r2=0.055)      # the column, tapering
    so.box(c, c + 0.06, 0.55, 0.06, 0.012, 0.18, so.material("sl_door", (120, 120, 116)))   # its door
    so.box(c, c + 0.33, 3.02, 0.05, 0.70, 0.05, grey)                  # the bracket arm, out over the road
    so.box(c, c + 0.08, 2.88, 0.04, 0.04, 0.16, grey)                  # its stay
    dark = so.material("sl_canopy", (70, 72, 74))
    so.box(c, c + 0.66, 2.98, 0.20, 0.52, 0.06, dark)                  # the lantern's canopy
    so.sphere(c, c + 0.66, 2.96, 0.13, so.material("sl_bowl", SOX, glow=0.55), segments=10, rings=6, squash_z=0.45)
    return {"geometry": [("cylinder", 0.5, 0.5, 0.0, 0.08, 3.1)]}


# --- double yellow lines --------------------------------------------------------------------------
LINE_W, LINE_GAP, LINE_IN = 0.075, 0.075, 0.06


def _strip(edge, d0, d1, mat, a0=0.0, a1=1.0):
    """A painted strip along a square's edge, d0..d1 in from it, a0..a1 along it."""
    z = 0.002
    if edge == "W":
        so.quad((d0, a0, z), (d1 - d0, 0, 0), (0, a1 - a0, 0), mat, (0, 0, 1))
    elif edge == "E":
        so.quad((1 - d1, a0, z), (d1 - d0, 0, 0), (0, a1 - a0, 0), mat, (0, 0, 1))
    elif edge == "N":
        so.quad((a0, d0, z), (a1 - a0, 0, 0), (0, d1 - d0, 0), mat, (0, 0, 1))
    else:
        so.quad((a0, 1 - d1, z), (a1 - a0, 0, 0), (0, d1 - d0, 0), mat, (0, 0, 1))


def double_yellow(edges):
    """The two lines along each edge in `edges` (one, or two meeting at a corner)."""
    mat = so.material("dy_yellow", YELLOW, grime=0.18)
    lines = [(LINE_IN, LINE_IN + LINE_W), (LINE_IN + LINE_W + LINE_GAP, LINE_IN + 2 * LINE_W + LINE_GAP)]
    for e in edges:
        for k, (d0, d1) in enumerate(lines):
            a0, a1 = 0.0, 1.0
            if len(edges) == 2:                       # an L: each line stops where its partner crosses
                other = [o for o in edges if o != e][0]
                start = lines[k][0]
                if other in ("W", "N"):
                    a0 = start
                else:
                    a1 = 1 - start
            _strip(e, d0, d1, mat, a0, a1)
    return {"geometry": []}


# --- the road markings of one and two tiles -------------------------------------------------------
MARKING_FACES = os.path.join(so.RENDERS, "marking_faces")       # marking_faces.py's pictures


def _marking(name):
    """Its picture, loaded afresh (each slot starts from an empty Blender)."""
    return bpy.data.images.load(os.path.join(MARKING_FACES, name + ".png"))


def _picture_at(x, y, heading, w, l):
    """Map (x, y) in a marking's squares (from its NW corner) -> its picture's
    (u, v), v from the top: the picture drawn for a driver heading north
    (marking_faces.py), its top the way they go, their left its left."""
    if heading in (None, "N"):
        return x / w, y / l
    if heading == "E":
        return y / w, (l - x) / l
    if heading == "S":
        return (w - x) / w, (l - y) / l
    return (w - y) / w, x / l


DECAL_OVER = 0.03                                   # tiles a marking's piece runs past its square


def decal(name, heading, size, grid):
    """Square `grid` (from the marking's north-west square) of the flat
    picture `name`, `size` (across, along) tiles, laid for `heading` (None:
    as drawn, north up)."""
    w, l = size
    gx, gy = grid
    img = _marking(name)
    mat = so.material(f"rm_{name}", image=img, interp="Linear", grime=0.12, alpha=img)
    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE":
            node.extension = "CLIP"                   # past the picture's edge: nothing, not its other side
    # A little past the square all round: pieces drawn edge to edge left a
    # seam of the road showing through their antialiased edges (a dark
    # cross through the mini roundabout's disc, 2026-10-14).
    e = DECAL_OVER
    uv = []
    for cx, cy in ((-e, -e), (1 + e, -e), (1 + e, 1 + e), (-e, 1 + e)):
        u, v = _picture_at(gx + cx, gy + cy, heading, w, l)
        uv.append((u, 1 - v))
    so.quad((-e, -e, 0.002), (1 + 2 * e, 0, 0), (0, 1 + 2 * e, 0), mat, (0, 0, 1), uv=uv)
    return {"geometry": []}


def _pieces(heading, size):
    """A marking's squares on the map, (gx, gy) from its north-west one."""
    w, l = size
    mw, mh = (w, l) if heading in (None, "N", "S") else (l, w)
    return [(gx, gy) for gy in range(mh) for gx in range(mw)]


# What a driver reads: (picture, size across x along), for each heading.
READ = [("give_way_triangle", (1, 2)), ("limit_30", (1, 2)), ("limit_40", (1, 2))] + \
    [(f"arrow_{k}", (1, 2)) for k in ("ahead", "left", "right", "ahead_left", "ahead_right", "deflect")] + \
    [("slow", (2, 2))]
BOXES = ("fill", "W", "N", "E", "S", "NW", "NE", "SE", "SW")


def markings_sheet():
    out = []
    for name, size in READ:
        for h in "NESW":
            out += [so.slot(f"{name}_{h}_{gx}{gy}", decal, (name, h, size, (gx, gy))) for gx, gy in _pieces(h, size)]
    out += [so.slot("zebra_NS", decal, ("zebra", "N", (1, 1), (0, 0))),
            so.slot("zebra_EW", decal, ("zebra", "E", (1, 1), (0, 0)))]
    # A zig-zag on the driver's left: heading N it runs along the W kerb, E the N...
    out += [so.slot(f"zigzag_{e}", decal, ("zigzag", h, (1, 1), (0, 0))) for e, h in zip("WNES", "NESW")]
    out += [so.slot(f"box_{k}", decal, (f"box_{k}", None, (1, 1), (0, 0))) for k in BOXES]
    out += [so.slot(f"double_white_{e}", decal, (f"double_white_{e}", None, (1, 1), (0, 0))) for e in "WN"]
    out += [so.slot(f"mini_disc_{gx}{gy}", decal, ("mini_disc", None, (2, 2), (gx, gy))) for gy in (0, 1) for gx in (0, 1)]
    return out + [so.slot("empty")] * (-len(out) % 8)


# --- the countryside -----------------------------------------------------------------------------
def _leaf(o, a, t, width, height, mat, metal):
    """A five-bar gate's leaf: from hinge corner o along a (unit), t its thickness direction."""
    o, a, t = (np.array(v, float) for v in (o, a, t))
    up = np.array((0, 0, 1.0))
    bar = 0.035 if metal else 0.045
    so.slab(o, a * 0.06, t * 0.06, up * height, mat)                          # hanging stile
    so.slab(o + a * (width - 0.05), a * 0.05, t * 0.05, up * height, mat)     # shutting stile
    for k in range(5):                                                        # the five bars
        z = 0.18 + k * (height - 0.24) / 4
        so.slab(o + up * z, a * width, t * bar * 0.8, up * bar, mat)
    # the brace, from the hinge's foot up towards the shutting end (a slanting slab)
    p0 = o + a * 0.06 + up * 0.18
    p1 = o + a * (width * 0.6) + up * (height - 0.06)
    d = p1 - p0
    so.slab(p0, d, t * bar * 0.8, up * bar, mat)


def field_gate(metal, state):
    """As vanilla's fence gates lie: shut in the W or N edge, or open 90 degrees into the square."""
    mat = so.material("fg_metal" if metal else "fg_timber", GALV if metal else TIMBER, grime=0.15)
    post = so.material("fg_post", (110, 92, 66), grime=0.15)
    o, a, t = {"closed_W": ((0.01, 0.02), (0, 1), (1, 0)), "closed_N": ((0.02, 0.01), (1, 0), (0, 1)),
               "open_W": ((0.03, 0.02), (1, 0), (0, 1)), "open_N": ((0.02, 0.03), (0, 1), (1, 0))}[state]
    o, a, t = o + (0,), a + (0,), t + (0,)
    _leaf(o, a, t, 0.94, 1.15, mat, metal)
    # the posts it hangs on and shuts against, on the line
    if state.endswith("W") and state.startswith("closed"):
        so.box(0.03, 0.02, 0.0, 0.08, 0.08, 1.25, post)
    elif state.startswith("closed"):
        so.box(0.02, 0.03, 0.0, 0.08, 0.08, 1.25, post)
    else:
        so.box(0.03, 0.03, 0.0, 0.08, 0.08, 1.25, post)
    return {"geometry": []}


def stile(edge):
    """A step stile: two posts, two rails, a step either side."""
    wood = so.material("st_timber", TIMBER, grime=0.15)
    if edge == "W":
        for y in (0.15, 0.85):
            so.box(0.04, y, 0.0, 0.08, 0.08, 1.05, wood)
        so.box(0.04, 0.5, 0.95, 0.05, 0.78, 0.07, wood)
        so.box(0.04, 0.5, 0.50, 0.05, 0.78, 0.07, wood)
        so.box(0.12, 0.5, 0.25, 0.22, 0.60, 0.05, wood)                    # the step
        so.box(0.12, 0.5, 0.0, 0.05, 0.05, 0.25, wood)
    else:
        for x in (0.15, 0.85):
            so.box(x, 0.04, 0.0, 0.08, 0.08, 1.05, wood)
        so.box(0.5, 0.04, 0.95, 0.78, 0.05, 0.07, wood)
        so.box(0.5, 0.04, 0.50, 0.78, 0.05, 0.07, wood)
        so.box(0.5, 0.12, 0.25, 0.60, 0.22, 0.05, wood)
        so.box(0.5, 0.12, 0.0, 0.05, 0.05, 0.25, wood)
    return {"geometry": []}


def kissing_gate(edge):
    """A kissing gate: a small gate swinging in a U of rails, on the W or N edge."""
    wood = so.material("kg_timber", TIMBER, grime=0.15)
    if edge == "W":
        for x, y in ((0.03, 0.12), (0.03, 0.88), (0.42, 0.12), (0.42, 0.88)):
            so.box(x, y, 0.0, 0.07, 0.07, 1.1, wood)
        for z in (0.45, 1.0):
            so.box(0.225, 0.12, z, 0.40, 0.05, 0.06, wood)               # the U's arms
            so.box(0.225, 0.88, z, 0.40, 0.05, 0.06, wood)
            so.box(0.42, 0.5, z, 0.05, 0.76, 0.06, wood)                 # ...and its back
        so.slab((0.06, 0.12, 0.12), (0.30, 0.42, 0), (0.03, -0.03, 0), (0, 0, 0.9), wood)   # the gate, half open
    else:
        for x, y in ((0.12, 0.03), (0.88, 0.03), (0.12, 0.42), (0.88, 0.42)):
            so.box(x, y, 0.0, 0.07, 0.07, 1.1, wood)
        for z in (0.45, 1.0):
            so.box(0.12, 0.225, z, 0.05, 0.40, 0.06, wood)
            so.box(0.88, 0.225, z, 0.05, 0.40, 0.06, wood)
            so.box(0.5, 0.42, z, 0.76, 0.05, 0.06, wood)
        so.slab((0.12, 0.06, 0.12), (0.42, 0.30, 0), (-0.03, 0.03, 0), (0, 0, 0.9), wood)
    return {"geometry": []}


POLE_R = 0.085
WIRE_Z = 1.55                                     # above the upper storey's floor
WIRE_OFF = (-0.14, 0.14)                          # the wires' two lines, either side of the pole


def pole(part, run=None):
    """A creosoted telegraph pole: its foot (this storey) or its top (the one
    above, with its crossarm across wires running `run`)."""
    wood = so.material("tp_wood", (82, 66, 50), grime=0.2)
    c = 0.5
    if part == "foot":
        so.cylinder(c, c, 0.0, POLE_R, LEVEL, wood, sides=10)
        so.box(c, c + POLE_R, 2.3, 0.05, 0.01, 0.07, so.material("tp_plate", (60, 60, 60)))   # its number plate
        return {"geometry": [("cylinder", 0.5, 0.5, 0.0, POLE_R + 0.01, LEVEL)]}
    so.cylinder(c, c, 0.0, POLE_R, WIRE_Z + 0.35, wood, sides=10, r2=POLE_R * 0.85)
    arm = so.material("tp_arm", (70, 58, 46), grime=0.2)
    if run == "EW":
        so.box(c, c, WIRE_Z - 0.06, 0.06, 0.50, 0.06, arm)                # across the wires
    else:
        so.box(c, c, WIRE_Z - 0.06, 0.50, 0.06, 0.06, arm)
    white = so.material("tp_insulator", (220, 220, 214))
    for o in WIRE_OFF:
        x, y = (c, c + o) if run == "EW" else (c + o, c)
        so.cylinder(x, y, WIRE_Z, 0.018, 0.04, white, sides=6)
    return {"geometry": []}


def wires(run):
    """Two wires across the square, a storey up, running E-W or N-S."""
    black = so.material("tp_wire", (30, 30, 30))
    for o in WIRE_OFF:
        if run == "EW":
            so.box(0.5, 0.5 + o, WIRE_Z + 0.03, 1.0, 0.012, 0.012, black)
        else:
            so.box(0.5 + o, 0.5, WIRE_Z + 0.03, 0.012, 1.0, 0.012, black)
    return {"geometry": []}


# --- kerbs from vanilla's measured shapes --------------------------------------------------------
GRID = 96                                         # cells across the square, the heightfield's
KERB_H = 0.11


MASKS = os.path.join(so.RENDERS, "kerb_masks.json")       # kerb_shapes.py's outlines


def kerb(name):
    """A kerb raised over our outline for vanilla piece `name` (kerb_shapes.py:
    cells GRID a metre from the square's north corner, some past its edge)."""
    cells = {tuple(c) for c in _masks[name]}
    verts, faces = [], []
    s = 1.0 / GRID

    def v(x, y, z):
        verts.append((x * s, y * s, z))
        return len(verts) - 1

    for i, j in cells:
        inside = ((i + 0.5) * s, (j + 0.5) * s, KERB_H / 2)              # this cell's middle
        a, b, c, d = v(i, j, KERB_H), v(i + 1, j, KERB_H), v(i + 1, j + 1, KERB_H), v(i, j + 1, KERB_H)
        faces.append(so._outward(verts, [a, b, c, d], inside))            # the top
        for (di, dj), (p, q) in (((0, -1), ((i, j), (i + 1, j))), ((1, 0), ((i + 1, j), (i + 1, j + 1))),
                                  ((0, 1), ((i + 1, j + 1), (i, j + 1))), ((-1, 0), ((i, j + 1), (i, j)))):
            if (i + di, j + dj) in cells:
                continue
            t0, t1 = v(p[0], p[1], KERB_H), v(q[0], q[1], KERB_H)
            b0, b1 = v(p[0], p[1], 0.0), v(q[0], q[1], 0.0)
            faces.append(so._outward(verts, [t0, t1, b1, b0], inside))    # a face down to the road
    mat = so.material("kb_concrete", KERB, grime=0.10)
    so._mesh(verts, faces, [[(0, 0), (1, 0), (1, 1), (0, 1)] for _ in faces], mat, "kerb")
    return {"geometry": []}


slot = so.slot
so.SHEETS["street_lighting_uk_01"] = [slot(f"street_lamp_{f}", street_lamp, (), f) for f in "NESW"] + \
    [slot("empty")] * 4
so.SHEETS["road_markings_uk_01"] = [slot(f"double_yellow_{e}", double_yellow, ((e,),)) for e in "WNES"] + \
    [slot(f"double_yellow_{a}{b}", double_yellow, ((a, b),)) for a, b in (("N", "W"), ("N", "E"), ("S", "E"), ("S", "W"))]
so.SHEETS["road_markings_uk_02"] = markings_sheet()
# Road gullies (user, 2026-10-14): square British grates by the kerb on the W,
# N, E, S edge; W and N stand in for vanilla's street_decoration_01_13 / _14
# in the map tools' storm drain rule (register.py DRAIN_SWAP).
so.SHEETS["street_drains_uk_01"] = [slot(f"gully_{e}", decal, (f"gully_{e}", None, (1, 1), (0, 0))) for e in "WNES"] +     [slot("empty")] * 4
cs = []
for metal in (False, True):
    for state in ("closed_W", "closed_N", "open_W", "open_N"):
        cs.append(slot(f"{'metal' if metal else 'wooden'}_gate_{state}", field_gate, (metal, state)))
cs += [slot("stile_W", stile, ("W",)), slot("stile_N", stile, ("N",)),
       slot("kissing_gate_W", kissing_gate, ("W",)), slot("kissing_gate_N", kissing_gate, ("N",)),
       slot("pole_foot", pole, ("foot",)), slot("pole_top_EW", pole, ("top", "EW")), slot("pole_top_NS", pole, ("top", "NS")),
       slot("wires_EW", wires, ("EW",)), slot("wires_NS", wires, ("NS",))]
cs += [slot("empty")] * (24 - len(cs))
so.SHEETS["countryside_uk_01"] = cs
_measured = json.load(open(KERBS))
_masks = json.load(open(MASKS)) if os.path.exists(MASKS) else {}
for sheet, ours, n in (("street_curbs_01", "street_curbs_uk_01", 64), ("street_curbs_01_diag", "street_curbs_uk_01_diag", 80),
                       ("street_curbs_01_diag_2", "street_curbs_uk_01_diag_2", 80)):
    so.SHEETS[ours] = [slot(f"kerb_{i}", kerb, (f"{sheet}_{i}",)) if f"{sheet}_{i}" in _measured else slot("empty")
                       for i in range(n)]
OURS = ["street_lighting_uk_01", "road_markings_uk_01", "countryside_uk_01",
        "street_curbs_uk_01", "street_curbs_uk_01_diag", "street_curbs_uk_01_diag_2", "road_markings_uk_02",
        "street_drains_uk_01"]

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sheets = [a for a in args if a in OURS] or OURS
    only = {a for a in args if a not in OURS}
    for sheet in sheets:
        so.render_sheet(sheet, only)
