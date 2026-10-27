"""A 1990s English town centre's street furniture, and its pubs' signs:
rendered at the game's camera (street_objects.py's), as its sheets are.

    "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b -P townscape.py -- [sheet ...] [slot ...]

- street_furniture_uk_03:
  - a bus stop: its pole and flag (TSRGD diagram 974; sign_faces.py), the
    flag reaching out over the kerb, a timetable case on the pole - its
    flag towards the road, N, E, S, W;
  - a bus shelter (the 1990s glass box: aluminium frame, glazed back and
    part ends, a flat roof, a perch seat, a lit advert at one end), two
    squares long, open to the road - as vanilla's park bench lies, each
    facing's two pieces at its SpriteGridPos;
  - a litter bin (black, a gold band, LITTER), round;
  - a grit bin (yellow, its lid sloping, GRIT), its front N, E, S, W;
  - a street cabinet (BT's green), its doors N, E, S, W;
  - a Sheffield cycle stand, running E-W or N-S;
  - pedestrian guard railing (galvanised, the 1990s at crossings and
    junctions) along a square's W, N, E or S edge.
- pub_signs_uk_01: each pub's hanging sign (pubs.py) on its post, its
  board facing N-S or E-W (painted both sides): a pub's sprites are its
  place in pubs.PUBS times two.
"""
from __future__ import annotations

import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pubs  # noqa: E402
import street_objects as so  # noqa: E402

ALUMINIUM = (176, 180, 182)
GLASS = (150, 176, 186)
BT_GREEN = (36, 78, 52)
GRIT_YELLOW = (226, 186, 30)
GALV = (158, 164, 166)
POST_BLACK = (34, 32, 30)


def glass(name, see=0.55):
    """A pane: the glass's tint, `see` of what lies behind showing through."""
    m = so.material(name, GLASS, grime=0.05)
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    shader = out.inputs["Surface"].links[0].from_socket
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 1 - see
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(shader, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


# --- the objects, built facing south (the road) at the square's middle; turned after ------------
def bus_stop():
    """The pole, its flag out over the kerb (south), the flag's plane across the kerb."""
    pole = so.material("bs_pole", so.POST)
    c = 0.5
    so.cylinder(c, 0.35, 0.0, 0.04, 2.75, pole, sides=10)
    so.box(c, 0.54, 2.62, 0.03, 0.40, 0.03, pole)                       # the bracket
    so.sign_plate(c, 0.72, 2.36, 0.40, 0.48, "E", "bus_flag", out=0.008, back=False)   # (proud of its edge)
    so.sign_plate(c, 0.72, 2.36, 0.40, 0.48, "W", "bus_flag", out=0.008, back=False)
    so.box(c, 0.72, 2.12, 0.012, 0.41, 0.49, so.material("bs_edge", (60, 60, 60)))   # the plate's edge, between faces
    so.box(c, 0.28, 1.25, 0.22, 0.05, 0.42, so.material("bs_case", (40, 44, 48)))      # the timetable case
    so.panel(c, 0.28 - 0.026, 1.29, 0.17, 0.32, "N", so.material("bs_table", (226, 226, 220)))
    return {"geometry": [("cylinder", 0.5, 0.35, 0.0, 0.05, 2.75)]}


def bus_shelter():
    """Two squares long (x 0..2), its back to the north, open to the south."""
    frame = so.material("sh_frame", ALUMINIUM, grime=0.10)
    pane = glass("sh_glass")
    x0, x1, yb, yf, top = 0.06, 1.94, 0.20, 0.66, 2.25
    for x, y in ((x0, yb), (x1, yb), (x0, yf), (x1, yf)):
        so.box(x, y, 0.0, 0.05, 0.05, top, frame)                        # the posts
    so.box((x0 + x1) / 2, (yb + 0.80) / 2, top, x1 - x0 + 0.10, 0.80 - yb + 0.08, 0.07,
           so.material("sh_roof", (110, 114, 118)))                      # the flat roof, over the kerb a little
    so.slab((x0, yb - 0.01, 0.12), (x1 - x0, 0, 0), (0, 0.02, 0), (0, 0, 1.95), pane)   # the glazed back
    so.slab((x0 - 0.01, yb, 0.12), (0.02, 0, 0), (0, yf - yb, 0), (0, 0, 1.95), pane)   # the west end
    for z in (0.12, 1.20, 2.07):                                        # the back's rails
        so.box((x0 + x1) / 2, yb, z, x1 - x0, 0.03, 0.04, frame)
    # The east end: a lit advert (a poster's colour, glowing).
    so.box(x1, (yb + yf) / 2, 0.30, 0.06, yf - yb, 1.75, frame)
    so.panel(x1 - 0.031, (yb + yf) / 2, 0.40, yf - yb - 0.08, 1.55, "W",
             so.material("sh_advert", (214, 200, 160), glow=0.85), out=0.0)
    so.box((x0 + x1) / 2, yb + 0.12, 0.62, 1.20, 0.10, 0.05, so.material("sh_perch", (60, 64, 68)))   # the perch seat
    return {"geometry": [("box", x0, yb - 0.03, 0.0, x1, yb + 0.03, top), ("box", x1 - 0.04, yb, 0.0, x1 + 0.04, yf, top),
                         ("box", x0 - 0.02, yb, 0.0, x0 + 0.02, yf, top)]}


def litter_bin():
    """Black, round, a gold band and LITTER towards the viewer (south)."""
    black = so.material("lb_black", (28, 28, 30), grime=0.10)
    c = 0.5
    so.cylinder(c, c, 0.0, 0.20, 0.06, black, sides=14)
    so.cylinder(c, c, 0.06, 0.22, 0.74, black, sides=14, r2=0.235)
    so.cylinder(c, c, 0.62, 0.238, 0.05, so.material("lb_gold", so.GOLD), sides=14)
    so.cylinder(c, c, 0.80, 0.25, 0.08, black, sides=14, r2=0.20)       # its domed lid, the opening under it
    so.cylinder(c, c, 0.74, 0.225, 0.06, so.material("lb_mouth", (12, 12, 12)), sides=14)
    so.sign_plate(c, c + 0.236, 0.48, 0.22, 0.06, "S", "litter_label", back=False)
    return {"geometry": [("cylinder", 0.5, 0.5, 0.0, 0.24, 0.88)]}


def grit_bin():
    """A yellow grit bin, its front (GRIT) south, its lid sloping up to the back."""
    yellow = so.material("gb_yellow", GRIT_YELLOW, grime=0.14)
    c = 0.5
    so.box(c, c, 0.0, 0.86, 0.52, 0.56, yellow)
    so.slab((c - 0.45, c + 0.29, 0.56), (0.90, 0, 0), (0, -0.60, 0.16), (0, 0, 0.05), yellow)   # the lid
    so.sign_plate(c, c + 0.262, 0.36, 0.40, 0.13, "S", "grit_label", back=False)
    return {"geometry": [("box", 0.07, 0.24, 0.0, 0.93, 0.76, 0.75)]}


def street_cabinet():
    """BT's green cabinet, its doors south (towards the road)."""
    green = so.material("sc_green", BT_GREEN, grime=0.12)
    c = 0.5
    so.box(c, c, 0.0, 0.80, 0.40, 0.10, so.material("sc_plinth", (90, 90, 86)))
    so.box(c, c, 0.10, 0.76, 0.36, 1.05, green)
    so.box(c, c, 1.15, 0.80, 0.40, 0.04, green)                          # its lid, over the doors
    so.box(c, c + 0.182, 0.15, 0.006, 0.006, 0.95, so.material("sc_seam", (20, 40, 28)))   # the doors' meeting
    so.panel(c - 0.22, c + 0.181, 0.92, 0.10, 0.06, "S", so.material("sc_plate", (220, 220, 214)))
    return {"geometry": [("box", 0.10, 0.30, 0.0, 0.90, 0.70, 1.19)]}


def cycle_stand(run):
    """A Sheffield stand: an inverted U of steel tube, running E-W or N-S."""
    steel = so.material("cs_steel", GALV, grime=0.08)
    a, b = 0.16, 0.84
    pts = ((a, 0.5), (b, 0.5)) if run == "EW" else ((0.5, a), (0.5, b))
    for x, y in pts:
        so.cylinder(x, y, 0.0, 0.026, 0.74, steel, sides=8)
    if run == "EW":
        so.box(0.5, 0.5, 0.71, b - a + 0.05, 0.05, 0.05, steel)
    else:
        so.box(0.5, 0.5, 0.71, 0.05, b - a + 0.05, 0.05, steel)
    return {"geometry": [("cylinder", x, y, 0.0, 0.03, 0.76) for x, y in pts]}


def guard_rail(edge):
    """Galvanised pedestrian guard railing along a square's edge: rails, bars, a post each end."""
    steel = so.material("gr_steel", GALV, grime=0.10)
    d = 0.05                                                            # in from the edge
    along_x = edge in ("N", "S")
    off = d if edge in ("W", "N") else 1 - d

    def bar(t, z0, h, w=0.025):
        x, y = (t, off) if along_x else (off, t)
        so.box(x, y, z0, w, w, h, steel)

    def rail(z, h=0.05):
        x, y = (0.5, off) if along_x else (off, 0.5)
        so.box(x, y, z, 1.0 if along_x else 0.04, 0.04 if along_x else 1.0, h, steel)

    for t in (0.03, 0.97):
        bar(t, 0.0, 1.05, 0.05)
    rail(0.12)
    rail(0.95, 0.07)
    for k in range(1, 7):
        bar(0.03 + k * 0.94 / 7, 0.12, 0.84)
    x0, y0, x1, y1 = (0, off - 0.03, 1, off + 0.03) if along_x else (off - 0.03, 0, off + 0.03, 1)
    return {"geometry": [("box", x0, y0, 0.0, x1, y1, 1.05)]}


def pub_sign(key):
    """A pub's board hanging from its post's arm, painted both sides (north and south)."""
    wood = so.material("ps_post", POST_BLACK, grime=0.12)
    so.box(0.14, 0.5, 0.0, 0.12, 0.12, 3.0, wood)                       # the post
    so.box(0.53, 0.5, 2.85, 0.80, 0.08, 0.08, wood)                     # the arm
    so.slab((0.20, 0.47, 2.48), (0.32, 0, 0.38), (0, 0.06, 0), (0, 0, 0.05), wood)   # its brace
    for x in (0.25, 0.85):                                              # the chains
        so.box(x, 0.5, 2.48, 0.012, 0.012, 0.37, so.material("ps_iron", (40, 40, 40)))
    so.box(0.55, 0.5, 1.53, 0.78, 0.05, 0.98, so.material("ps_frame", (50, 40, 30)))   # the board's frame
    so.sign_plate(0.55, 0.5, 2.02, 0.72, 0.92, "S", f"pub_{key}", out=0.027, back=False)
    so.sign_plate(0.55, 0.5, 2.02, 0.72, 0.92, "N", f"pub_{key}", out=0.027, back=False)
    return {"geometry": [("box", 0.08, 0.44, 0.0, 0.20, 0.56, 3.0)]}


slot = so.slot
t3 = [slot(f"bus_stop_{f}", bus_stop, (), f) for f in so.FACINGS]
t3 += [slot(f"bus_shelter_{f}_{gx}{gy}", bus_shelter, (), f, (gx, gy)) for f, (gx, gy) in so.BENCH_PIECES]
t3 += [slot("litter_bin", litter_bin)]
t3 += [slot(f"grit_bin_{f}", grit_bin, (), f) for f in so.FACINGS]
t3 += [slot(f"street_cabinet_{f}", street_cabinet, (), f) for f in so.FACINGS]
t3 += [slot("cycle_stand_EW", cycle_stand, ("EW",)), slot("cycle_stand_NS", cycle_stand, ("NS",))]
t3 += [slot(f"guard_rail_{e}", guard_rail, (e,)) for e in "WNES"]
so.SHEETS["street_furniture_uk_03"] = t3 + [slot("empty")] * (-len(t3) % 8)
ps = []
for key in pubs.PUBS:
    ps += [slot(f"pub_{key}_S", pub_sign, (key,), "S"), slot(f"pub_{key}_E", pub_sign, (key,), "E")]
so.SHEETS["pub_signs_uk_01"] = ps + [slot("empty")] * (-len(ps) % 8)
OURS = ["street_furniture_uk_03", "pub_signs_uk_01"]

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sheets = [a for a in args if a in OURS] or OURS
    only = {a for a in args if a not in OURS}
    for sheet in sheets:
        so.render_sheet(sheet, only)
