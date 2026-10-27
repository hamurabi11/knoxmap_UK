"""Farmyard props as they stood in 1993 (PLAN.md step 4): one square each.

    python farm_props.py   (KnoxMap's Python)

Sheet commercial_farm_uk_01 (its DEPTH_ sheet, .slots.json and a preview), for the
user's own yard templates (as the car park's props are: made here, not placed):

    0  round straw bale lying, its axis east-west
    1  round straw bale lying, its axis north-south
    2  silage bale in black wrap, on its end
    3  a stack of small square straw bales
    4  galvanised water trough, along north-south
    5  galvanised water trough, along east-west
    6  galvanised feed bin with its lid
    7  three old tyres, stacked (they hold a silage clamp's sheet down)
    8  muck heap (straw and dung)

Drawn as structure_kit.py draws: shapes in metres on the square (wx east, wy south,
z up) through walls.py's camera and light, every pixel's depth into the DEPTH_ sheet.
A shape is a set of flat faces (a cylinder: a 12-sided prism and its two discs; a
heap: a 10-sided cone); each face facing the camera is drawn, far ones first, lit by
which way it faces (the top 1.12, south 1.0, east 0.8: walls.py's light).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import supermarket as SM  # noqa: E402

WK, F = SM.WK, SM.F
W = WK.W
SHEET = "commercial_farm_uk_01"
VIEW = np.array([1.0, 1.0, 1.35])        # towards the camera (walls.py's depth: wx + wy, z at 42.4 / 62.8)
N_SIDES = 12

# (review r1: the straw lemony, the muck too light - sand)
STRAW, WRAP, GALV, WATER = (196, 168, 100), (36, 38, 36), (150, 154, 152), (54, 66, 64)
TYRE, MUCK = (34, 34, 34), (62, 46, 30)


def _tex(rgb, rng, n=32, grain=6.0, streak=None):
    """A face's texture: a colour, its grain; streaks along s (straw) or t."""
    t = np.ones((n, n, 4)) * 255.0
    t[..., :3] = np.array(rgb, float) + rng.normal(0, grain, (n, n, 1))
    if streak == "s":
        t[..., :3] *= (1 + 0.12 * rng.normal(0, 1, (n, 1, 1)))
    elif streak == "t":
        t[..., :3] *= (1 + 0.12 * rng.normal(0, 1, (1, n, 1)))
    return t


def _light(n):
    """A face's light by which way it faces: each direction's light weighted by the square
    of its share of the normal (the shares' squares sum to 1: a south-east facet lies
    between the south's and the east's light, not brighter than either - code review)."""
    n = n / np.linalg.norm(n)
    w = n ** 2
    return float(W.TOP_LIGHT * w[2] * (n[2] > 0) + W.EAST_LIGHT * w[0] * (n[0] > 0) + 1.0 * w[1] * (n[1] > 0)
                 + 0.55 * w[0] * (n[0] < 0) + 0.55 * w[1] * (n[1] < 0) + 0.4 * w[2] * (n[2] < 0))


class Sprite:
    """Faces for one square's sprite, drawn back to front."""

    def __init__(self):
        self.out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
        self.depth = np.full((F.TILE_H, F.TILE_W), np.nan)
        self.faces = []

    def face(self, origin, a, b, tex, normal=None):
        o, a, b = (np.array(v, float) for v in (origin, a, b))
        n = np.cross(a, b) if normal is None else np.array(normal, float)
        if n @ VIEW < 0:
            n = -n
            if normal is not None:
                return                       # a face given its outward normal, turned away: unseen
        centre = o + a / 2 + b / 2
        self.faces.append((centre @ VIEW, o, a, b, tex, _light(n)))

    def draw(self):
        for _k, o, a, b, tex, light in sorted(self.faces, key=lambda f: f[0]):
            try:
                W._draw(self.out, o, a, b, tex, light, depth=self.depth)
            except np.linalg.LinAlgError:
                pass                         # edge-on to the camera: a line, nothing to paint
        return self.out, self.depth

    # -- shapes ------------------------------------------------------------------------
    def box(self, x0, x1, y0, y1, z0, z1, tex_side, tex_top=None):
        top = tex_top if tex_top is not None else tex_side
        self.face((x0, y0, z1), (x1 - x0, 0, 0), (0, y1 - y0, 0), top, (0, 0, 1))
        self.face((x0, y1, z1), (x1 - x0, 0, 0), (0, 0, z0 - z1), tex_side, (0, 1, 0))     # south
        self.face((x1, y0, z1), (0, y1 - y0, 0), (0, 0, z0 - z1), tex_side, (1, 0, 0))     # east

    def cylinder(self, centre, axis, r, length, tex_side, tex_end, hole=0.0, rim=True):
        """A cylinder: its centre (of one end), axis (unit), radius, length; `hole` a
        fraction of r at the ends painted as the dark inside (a tyre's); `rim` its edge darkened."""
        c, ax = np.array(centre, float), np.array(axis, float)
        u = np.cross(ax, (0, 0, 1) if abs(ax[2]) < 0.9 else (1, 0, 0))
        u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        for k in range(N_SIDES):
            a0, a1 = 2 * np.pi * k / N_SIDES, 2 * np.pi * (k + 1) / N_SIDES
            p0 = c + r * (np.cos(a0) * u + np.sin(a0) * v)
            p1 = c + r * (np.cos(a1) * u + np.sin(a1) * v)
            mid = (a0 + a1) / 2
            self.face(p0, p1 - p0, ax * length, tex_side, np.cos(mid) * u + np.sin(mid) * v)
        # the ends: a square face, its texture cut to a disc (or a ring)
        n = tex_end.shape[0]
        yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1
        rr = np.hypot(xx, yy)
        disc = tex_end.copy()
        disc[..., 3] = np.where(rr <= 1, 255, 0)
        if rim:
            disc[..., :3] *= np.where(rr > 0.86, 0.85, 1.0)[..., None]   # its rim
        if hole:
            # the tyre's inside, its inner wall shading to dark (review r1: the hole showed
            # the ground through it)
            inner = rr < hole
            disc[inner, :3] = np.array((15, 15, 15)) + (rr[inner, None] / hole) * 22
        for end, nrm in ((c, -ax), (c + ax * length, ax)):
            self.face(end - r * u - r * v, 2 * r * u, 2 * r * v, disc, nrm)

    def cone(self, centre, r, h, tex):
        """A heap: a cone of N sides on the ground, each side a triangle (a face cut in half)."""
        c = np.array(centre, float)
        apex = c + (0, 0, h)
        n = tex.shape[0]
        tri = tex.copy()
        ss, tt = np.mgrid[0:n, 0:n] + 0.5
        tri[..., 3] = np.where(ss + tt <= n, 255, 0)                   # rows t, columns s: s + t < 1
        sides = 10
        rs = r * (1 + 0.14 * np.sin(np.arange(sides + 1) * 2.7))          # lumpy, not a neat cone
        rs[-1] = rs[0]
        for k in range(sides):
            a0, a1 = 2 * np.pi * k / sides, 2 * np.pi * (k + 1) / sides
            p0 = c + rs[k] * np.array([np.cos(a0), np.sin(a0), 0])
            p1 = c + rs[k + 1] * np.array([np.cos(a1), np.sin(a1), 0])
            mid = (a0 + a1) / 2
            nrm = np.array([np.cos(mid) * h, np.sin(mid) * h, r])
            self.face(apex, p0 - apex, p1 - apex, tri, nrm)


def _twine(tex, every=8, along="t"):
    if along == "t":
        tex[::every, :, :3] *= 0.72
    else:
        tex[:, ::every, :3] *= 0.72
    return tex


def bale_lying(rng, axis):
    s = Sprite()
    r, ln = 0.55, 1.1
    ax = (1, 0, 0) if axis == "x" else (0, 1, 0)
    start = (0.5 - ln / 2, 0.5, r) if axis == "x" else (0.5, 0.5 - ln / 2, r)
    side = _twine(_tex(STRAW, rng, streak="t"), 6, "s")                 # net wrap lines round it
    end = _tex(STRAW, rng, grain=9)
    yy, xx = (np.mgrid[0:32, 0:32] + 0.5) / 16 - 1
    end[..., :3] *= (0.9 + 0.1 * np.sin(np.hypot(xx, yy) * 18))[..., None]   # the rolled spiral
    s.cylinder(start, ax, r, ln, side, end)
    return s.draw()


def bale_wrapped(rng):
    s = Sprite()
    side = _tex(WRAP, rng, grain=1)
    side[:, 8:12, :3] *= np.array([1.4, 1.8, 1.8, 1.4])[None, :, None]  # one soft sheen down it
    end = _tex(WRAP, rng, grain=1)
    end[6:10, :, :3] *= 1.35
    s.cylinder((0.5, 0.5, 0.0), (0, 0, 1), 0.6, 1.2, side, end, rim=False)
    return s.draw()


def bale_stack(rng):
    s = Sprite()
    for z, (x0, x1, y0, y1) in enumerate([(0.05, 0.95, 0.05, 0.95), (0.05, 0.95, 0.05, 0.95),
                                          (0.15, 0.85, 0.15, 0.85)]):
        for k in range(2):                                              # two bales a course, crossed
            if z % 2 == 0:
                bx = (x0, x1, y0 + k * (y1 - y0) / 2, y0 + (k + 1) * (y1 - y0) / 2)
            else:
                bx = (x0 + k * (x1 - x0) / 2, x0 + (k + 1) * (x1 - x0) / 2, y0, y1)
            side = _twine(_tex(STRAW, rng, streak="s" if z % 2 else "t"), 11, "s")
            side[:, :1, :3] *= 0.7
            s.box(bx[0], bx[1], bx[2], bx[3], 0.36 * z, 0.36 * (z + 1), side,
                  _twine(_tex(STRAW, rng, grain=8), 11, "s"))
    return s.draw()


def trough(rng, along):
    s = Sprite()
    rim = _tex(GALV, rng, grain=4)
    rim[:2, :, :3] *= 1.2
    rim[-3:, :, :3] *= 0.85
    water = _tex(WATER, rng, grain=2)
    water[:, :, :3] *= np.linspace(0.9, 1.15, 32)[None, :, None]
    if along == "y":
        x0, x1, y0, y1 = 0.25, 0.75, 0.04, 0.96
    else:
        x0, x1, y0, y1 = 0.04, 0.96, 0.25, 0.75
    s.box(x0, x1, y0, y1, 0.0, 0.5, rim, water)
    # (review r1: a closed chest) the water 5 cm down, a bright rolled rim round it,
    # the far inside walls showing above the water
    lip = _tex(np.array(GALV) * 1.2, rng, grain=2)
    s.faces.pop(0)                                                      # not the box's flat top
    s.face((x0, y0, 0.45), (x1 - x0, 0, 0), (0, y1 - y0, 0), water, (0, 0, 1))
    s.face((x0 + 0.03, y0 + 0.03, 0.5), (x1 - x0 - 0.03, 0, 0), (0, 0, -0.05), rim, (0, 1, 0))   # inside, far
    s.face((x0 + 0.03, y0 + 0.03, 0.5), (0, y1 - y0 - 0.03, 0), (0, 0, -0.05), rim, (1, 0, 0))
    for (ox, oy, ax, ay) in ((x0, y0, x1 - x0, 0.03), (x0, y1 - 0.03, x1 - x0, 0.03),
                             (x0, y0, 0.03, y1 - y0), (x1 - 0.03, y0, 0.03, y1 - y0)):
        s.face((ox, oy, 0.5), (ax, 0, 0), (0, ay, 0), lip, (0, 0, 1))
    return s.draw()


def feed_bin(rng):
    s = Sprite()
    side = _tex(GALV, rng, grain=4)
    side[4::9, :, :3] *= 0.8                                            # its corrugations
    lid = _tex(np.array(GALV) * 1.05, rng, grain=3)
    s.cylinder((0.5, 0.5, 0.0), (0, 0, 1), 0.32, 0.78, side, lid)
    s.cylinder((0.5, 0.5, 0.78), (0, 0, 1), 0.35, 0.06, _tex(np.array(GALV) * 0.9, rng, grain=3), lid)
    s.box(0.47, 0.53, 0.47, 0.53, 0.84, 0.92, _tex((60, 60, 60), rng, grain=2))    # the handle
    return s.draw()


def tyres(rng):
    s = Sprite()
    for k in range(3):
        tread = _tex(TYRE, rng, grain=3)
        tread[:, ::4, :3] *= 0.6
        tread[:4, :, :3] *= 0.45                                        # the joins between the tyres:
        tread[-4:, :, :3] *= 0.45                                       # each one's rounded shoulders
        wall = _tex(np.array(TYRE) * 1.3, rng, grain=3)
        s.cylinder((0.5, 0.5, 0.22 * k), (0, 0, 1), 0.38, 0.22, tread, wall, hole=0.52)   # (review: in line)
    return s.draw()


def muck_heap(rng):
    s = Sprite()
    t = _tex(MUCK, rng, n=48, grain=6)
    for _ in range(14):                                                 # dull straw streaks, not sparkle
        y, x = rng.integers(0, 46), rng.integers(0, 40)
        t[y:y + 1, x:x + 7, :3] = np.array(STRAW) * 0.55
    s.cone((0.5, 0.5, 0.0), 0.52, 0.5, t)
    return s.draw()


PROPS = [  # (maker, the game's name for it); appended only: a slot's place fixes its id
    (lambda r: bale_lying(r, "x"), "Round Bale"), (lambda r: bale_lying(r, "y"), "Round Bale"),
    (bale_wrapped, "Silage Bale"), (bale_stack, "Straw Bales"), (lambda r: trough(r, "y"), "Water Trough"),
    (lambda r: trough(r, "x"), "Water Trough"), (feed_bin, "Feed Bin"), (tyres, "Old Tyres"),
    (muck_heap, "Muck Heap"),
]


def main() -> int:
    rng = np.random.default_rng(1993)
    pieces = [make(rng) for make, _n in PROPS]
    tiles = [t for t, _d in pieces]
    F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"), [W.depth_tile(t, d) for t, d in pieces])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump([{"props": [["solidtrans", ""], ["CustomName", name]]} for _m, name in PROPS], f)
    # preview: each prop on a grass square at 1x, a vanilla-sized row
    os.makedirs(SM.PREVIEW, exist_ok=True)
    img = Image.new("RGBA", (len(tiles) * 136 + 8, 280), (96, 100, 96, 255))
    for k, t in enumerate(tiles):
        img.alpha_composite(Image.fromarray(t), (8 + 136 * k, 12))
    img.save(os.path.join(SM.PREVIEW, "farm_props.png"))
    order = os.path.join(HERE, "sheets.order.json")
    have = json.load(open(order, encoding="utf-8"))
    if SHEET not in have:
        have.append(SHEET)
    json.dump(have, open(order, "w", encoding="utf-8"), indent=1)
    print("wrote", SHEET, len(tiles))
    return 0


if __name__ == "__main__":
    sys.exit(main())
