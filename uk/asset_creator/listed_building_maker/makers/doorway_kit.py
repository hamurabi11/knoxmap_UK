"""Doorcases, door hoods and porches for the listed buildings' front doors.

    python doorway_kit.py   (KnoxMap's Python)

Writes Tile_creator/build/sheets/<group>/doorways_uk_01.png, its DEPTH_ sheet and a
preview. Four pieces, each for a door in a west wall (slot 2 k) and in a north
wall (2 k + 1), laid by Build on the front door's own square
(knoxbuild/tbx.py `door_surround`, regional_uk SURROUNDS):

    0  doorcase   pilasters, an entablature and an open pediment (Georgian)
    2  hood       a flat hood on two carved brackets (C18)
    4  porch      a flat-roofed porch on two columns (Greek Doric, Regency)
    6  porch_timber  an oak-posted porch with a little tiled gable (cottages)

Each is built of boxes and planes in metres on the square (wx east, wy
south, z up) and drawn with walls.py's camera and light; every pixel's depth
goes into the DEPTH_ sheet (walls.py's encoding). The door it frames is
house_walls.DOOR's opening (0.10-0.87 along the wall, 2.36 m high). A door in
a west wall is seen from the east, so its pieces stand out east of the wall
(wx > 0); in a north wall, south of it (wy > 0).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wall_kit as WK  # noqa: E402

W, F = WK.W, WK.F
SHEET = "doorways_uk_01"
WHITE, OAK, TILE = (232, 230, 222), (78, 62, 46), (132, 70, 54)
D0, D1, DH = 0.10, 0.87, 2.36          # the door's opening: along the wall, and its height
WALL = 2 / 64                          # the wall's face (house wall thickness)


def _flat(rgb, h=4, w=4):
    return np.dstack([np.full((h, w, 3), rgb, float), np.full((h, w), 255.0)])


class Piece:
    """Boxes and planes for one sprite, drawn back to front."""

    def __init__(self, north: bool):
        self.north = north
        self.out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
        self.depth = np.full((F.TILE_H, F.TILE_W), np.nan)
        self.items = []

    def _p(self, out, along, z):
        """Wall coordinates (out from the wall, along it, up) -> world (wx, wy, z)."""
        return (along, out, z) if self.north else (out, along, z)

    def box(self, o0, o1, a0, a1, z0, z1, rgb):
        self.items.append((o0 + a0, "box", (o0, o1, a0, a1, z0, z1, rgb)))

    def gable(self, o0, o1, a0, a1, z0, z1, rgb_roof, rgb_end):
        self.items.append((o0 + a0, "gable", (o0, o1, a0, a1, z0, z1, rgb_roof, rgb_end)))

    def pediment(self, o, a0, a1, z0, z1, rgb):
        self.items.append((o + a0 - 0.01, "ped", (o, a0, a1, z0, z1, rgb)))

    def _face(self, origin, a, b, tex, light):
        W._draw(self.out, origin, a, b, tex, light, depth=self.depth)

    def draw(self):
        for _key, kind, args in sorted(self.items, key=lambda t: t[0]):
            getattr(self, "_" + kind)(*args)
        return self.out, self.depth

    def _vec(self, out, along, z):
        return np.array(self._p(out, along, z), float)

    def _box(self, o0, o1, a0, a1, z0, z1, rgb):
        tex = _flat(rgb)
        lo = self._vec(o0, a0, z1)
        da = self._vec(0, a1 - a0, 0)
        do = self._vec(o1 - o0, 0, 0)
        dz = self._vec(0, 0, -(z1 - z0))
        # the face outward from the wall, the end face along it, the top
        out_light = 1.0 if self.north else W.EAST_LIGHT
        end_light = W.EAST_LIGHT if self.north else 1.0
        self._face(lo + do, da, dz, tex, out_light)
        self._face(lo + da, do, dz, tex, end_light)
        self._face(lo, do, da, tex, W.TOP_LIGHT)

    def _ped(self, o, a0, a1, z0, z1, rgb):
        """A triangular pediment flat on the wall, its raking cornices darker."""
        n = 64
        tri = np.zeros((n, n, 4))
        for r in range(n):                              # r: down from the apex
            half = (r + 1) / n / 2
            c0, c1 = int((0.5 - half) * n), int(np.ceil((0.5 + half) * n))
            tri[r, c0:c1, :3] = rgb
            tri[r, c0:c1, 3] = 255
            tri[r, c0:c0 + 3, :3] = np.array(rgb) * 0.82
            tri[r, max(c0, c1 - 3):c1, :3] = np.array(rgb) * 0.82
        light = 1.0 if self.north else W.EAST_LIGHT
        self._face(self._vec(o, a0, z1), self._vec(0, a1 - a0, 0), self._vec(0, 0, -(z1 - z0)), tri, light)

    def _gable(self, o0, o1, a0, a1, z0, z1, rgb_roof, rgb_end):
        """A little pitched roof running out from the wall: two slopes, a gable end."""
        mid = (a0 + a1) / 2
        n = 32
        courses = np.zeros((n, n, 4))
        courses[..., :3] = rgb_roof
        courses[..., 3] = 255
        courses[::4, :, :3] *= 0.7                      # tile courses along the slope's foot
        far_light = 0.85 if self.north else 0.7
        near_light = 1.0 if self.north else 0.95
        # far slope (towards a0), then the near one, then the end
        self._face(self._vec(o0, mid, z1), self._vec(o1 - o0, 0, 0), self._vec(0, a0 - mid, z0 - z1),
                   courses, far_light)
        self._face(self._vec(o0, mid, z1), self._vec(o1 - o0, 0, 0), self._vec(0, a1 - mid, z0 - z1),
                   courses, near_light)
        tri = np.zeros((n, n, 4))
        for r in range(n):
            half = (r + 1) / n / 2
            c0, c1 = int((0.5 - half) * n), int(np.ceil((0.5 + half) * n))
            tri[r, c0:c1, :3] = rgb_end
            tri[r, c0:c1, 3] = 255
        light = W.EAST_LIGHT if not self.north else 1.0
        self._face(self._vec(o1, a0, z1), self._vec(0, a1 - a0, 0), self._vec(0, 0, -(z1 - z0)), tri, light)


def doorcase(north):
    p = Piece(north)
    for a0 in (D0 - 0.09, D1 + 0.01):                   # pilasters
        p.box(WALL, WALL + 0.05, a0, a0 + 0.08, 0.0, DH + 0.06, WHITE)
    p.box(WALL, WALL + 0.09, D0 - 0.12, D1 + 0.12, DH + 0.06, DH + 0.20, WHITE)   # entablature
    p.pediment(WALL + 0.09, D0 - 0.12, D1 + 0.12, DH + 0.20, DH + 0.50, WHITE)
    return p.draw()


def hood(north):
    p = Piece(north)
    for a0 in (D0 - 0.05, D1 - 0.02):                   # carved brackets
        p.box(WALL, WALL + 0.36, a0, a0 + 0.07, DH + 0.02, DH + 0.20, (212, 210, 202))
    p.box(WALL, WALL + 0.48, D0 - 0.10, D1 + 0.10, DH + 0.20, DH + 0.30, WHITE)
    return p.draw()


def porch(north):
    p = Piece(north)
    for a0 in (D0 - 0.06, D1 - 0.04):                   # pilasters on the wall
        p.box(WALL, WALL + 0.04, a0, a0 + 0.10, 0.0, DH + 0.10, WHITE)
    p.box(WALL, 0.92, D0 - 0.12, D1 + 0.12, 0.0, 0.08, (180, 176, 168))           # a stone step
    for a0 in (D0 - 0.08, D1 - 0.02):                   # columns
        p.box(0.72, 0.82, a0, a0 + 0.10, 0.08, DH + 0.10, WHITE)
    p.box(WALL, 0.88, D0 - 0.14, D1 + 0.14, DH + 0.10, DH + 0.30, WHITE)           # entablature and flat roof
    p.box(WALL + 0.02, 0.86, D0 - 0.12, D1 + 0.12, DH + 0.30, DH + 0.34, (96, 98, 104))   # lead
    return p.draw()


def porch_timber(north):
    p = Piece(north)
    for a0 in (D0 - 0.06, D1 - 0.02):                   # oak posts
        p.box(0.70, 0.78, a0, a0 + 0.08, 0.0, DH, OAK)
    p.box(WALL, 0.80, D0 - 0.08, D1 + 0.08, DH, DH + 0.08, OAK)                   # the plate
    p.gable(WALL, 0.86, D0 - 0.14, D1 + 0.14, DH + 0.08, DH + 0.62, TILE, OAK)
    return p.draw()


PIECES = [doorcase, hood, porch, porch_timber]


def main() -> None:
    tiles, depths = [], []
    for make in PIECES:
        for north in (False, True):
            t, d = make(north)
            tiles.append(t)
            depths.append(d)
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"),
                     [W.depth_tile(t, d) for t, d in zip(tiles, depths)])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        f.write("[" + ", ".join(["null"] * len(tiles)) + "]")      # no vanilla piece: seen only
    # A preview: each piece on a brick wall with a door, west and north.
    wall = WK.vanilla_tile("walls_exterior_house_01", 14), WK.vanilla_tile("walls_exterior_house_01", 15)
    img = Image.new("RGBA", (128 * 8, 256), (80, 100, 80, 255))
    for k, t in enumerate(tiles):
        img.alpha_composite(Image.fromarray(np.ascontiguousarray(wall[k % 2])), (k * 128, 0))
        img.alpha_composite(Image.fromarray(t), (k * 128, 0))
    img.save(os.path.join(WK.PREVIEW, f"kit_{SHEET}.png"))
    print("wrote", SHEET)


if __name__ == "__main__":
    main()
