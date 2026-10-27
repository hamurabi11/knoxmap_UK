"""Boundaries along 45-degree and 2:1 lines, to follow angled roads.

    python diag_walls.py   -> build/sheets/<group>/boundaries_diag_uk_01.png
                              build/preview/diag_boundaries.png

The game's boundaries stand on a square's west or north edge only (WallW,
WallN; Build 42 has no diagonal wall), so along an angled street a garden
front is a staircase. The road graph draws angled edges on lines through tile
corners (raster.lattice_k), so a boundary along one crosses each square in
one of CUTS (14 ways: 45 degrees corner to corner or mid-edge to mid-edge,
2:1 corner to mid-edge, shallow or steep). For each boundary style (STYLES, in Build's names,
knoxbuild/fences.py) one picture per cut, drawn along the line by walls.py's
projector, and three invisible pieces carrying the style's collision on a
west edge, a north edge or both (properties copied by pack.py from the
style's straight pieces: hoppable, climbable...). Build keeps a staircase's
edges as the invisible pieces and lays the pictures along the line
(fences.py, DIAG_*).

Sheet: 24 slots a style: 0-13 the cuts, 16-18 invisible W, N, NW, the rest
empty. (Walls north of a road are moved forward by Build, picture and
collision together, a whole square: no pictures of their own.)
Heights and colours measured on the straight pieces each style meets
(vanilla's fences and hedges, our brick walls), 2026-10-12.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import flags
import floor_tiles as F
import walls as W
from sheet_paths import sheet_file

SHEET = "boundaries_diag_uk_01"
# (p0, p1) on the square, wx wy 0..1; the same list in knoxbuild/fences.py.
CUTS = [((0, 1), (1, 0)), ((0, 0), (1, 1)),                              # 45 degrees, corner to corner
        ((0, 1), (1, .5)), ((0, .5), (1, 0)), ((0, 0), (1, .5)), ((0, .5), (1, 1)),   # 2:1 shallow
        ((0, 1), (.5, 0)), ((.5, 1), (1, 0)), ((0, 0), (.5, 1)), ((.5, 0), (1, 1)),   # 1:2 steep
        # 45 degrees, mid-edge to mid-edge: a boundary drawn down the middle of
        # its collision staircase cuts a square's corner off (user, 2026-10-12:
        # through the corners it stood up to 0.7 of a square off its collision).
        ((.5, 0), (0, .5)), ((1, .5), (.5, 1)), ((.5, 0), (1, .5)), ((0, .5), (.5, 1))]
SLOTS = 24                   # a style's slots: 0-13 the cuts, 16-18 invisible W, N, NW
SHADOW_W, SHADOW_ALPHA = 0.12, 110   # the contact shadow: squares wide, darkest alpha
STYLE_ORDER = ["uk_brick_low", "uk_brick_tall", "hedge_low", "hedge_tall",
               "short_wooden", "tall_wooden", "black_metal"]


def _noise(shape, rng, scales=((1, 1.0), (3, 0.7), (8, 0.5))) -> np.ndarray:
    out = np.zeros(shape)
    for s, w in scales:
        out += w * ndimage.gaussian_filter(rng.normal(0, 1, shape), s)
    return out / (out.std() or 1)


def hedge_face(px: int, hpx: int, rng, base, ragged: bool = True) -> np.ndarray:
    """Clipped privet or beech: blotchy leaves, darker hollows, a ragged edge
    (not on a face under a top: the paving showed through)."""
    n = _noise((hpx, px), rng)
    rgb = np.array(base) * (1 + 0.22 * n)[..., None]
    rgb += rng.normal(0, 6, (hpx, px, 1))
    a = np.full((hpx, px), 255.0)
    top = (2.5 + 1.8 * ndimage.gaussian_filter1d(rng.normal(0, 1, px), 2)).astype(int)
    for x in range(px if ragged else 0):
        a[:max(0, top[x]), x] = 0
    return np.dstack([rgb, a])


def wood_face(px: int, hpx: int, rng, base) -> np.ndarray:
    """Close-board fence: vertical boards 6 px wide, a post every metre, a gravel board at the foot."""
    rgb = np.zeros((hpx, px, 3)) + base
    for x0 in range(0, px, 6):
        rgb[:, x0:x0 + 6] *= 1 + rng.normal(0, 0.07)
        rgb[:, x0] *= 0.72                                       # the board's edge
    rgb *= (1 + 0.05 * _noise((hpx, px), rng, ((1, 1.0),)))[..., None]
    for x0 in range(0, px, W.PX):
        rgb[:, x0:x0 + 4] = np.array(base) * 0.85                # post
    rgb[-6:] = np.array(base) * 0.6                              # gravel board
    return np.dstack([rgb, np.full((hpx, px), 255.0)])


def railing_face(px: int, hpx: int, rng, base) -> np.ndarray:
    """Black railings: a bar every 8 px, top and bottom rails, spear tops."""
    a = np.zeros((hpx, px))
    rgb = np.zeros((hpx, px, 3)) + base
    for x0 in range(2, px, 8):
        a[2:, x0:x0 + 2] = 255
        a[0:2, x0] = 255                                         # spear tip
    a[6:8] = 255
    a[-4:-2] = 255
    rgb[:, :, :] *= 1 + 0.25 * (np.arange(px) % 8 == 2)[None, :, None]   # a highlight on each bar
    return np.dstack([rgb, a])


def brick_face(faces, mortar):
    def face(px, hpx, rng, base, courses):
        return np.dstack([W.face_texture(px + W.BRICK, courses, faces, mortar, rng)[:, :px], np.full((hpx, px), 255.0)])
    return face


def styles():
    rng = np.random.default_rng(5)
    faces, _m = W.brick_library(rng)
    faces, mortar = W.styled(faces, flags.LEVEL), W.MORTAR
    brick = brick_face(faces, mortar)
    top = lambda px, tpx, rng: np.dstack([W.top_texture(px, faces, mortar, rng)[:tpx], np.full((tpx, px), 255.0)])
    green = lambda px, tpx, rng, base: hedge_face(px, tpx, rng, np.array(base) * 1.15)[..., :]
    # name: (height px, thickness px, face(px, hpx, rng), top(px, tpx, rng) or None)
    return {
        "uk_brick_low": (W.LOW * W.COURSE + W.COPING_H, W.T, lambda px, h, r: brick(px, h, r, None, W.LOW), top),
        "uk_brick_tall": (W.TALL * W.COURSE + W.COPING_H, W.T, lambda px, h, r: brick(px, h, r, None, W.TALL), top),
        "hedge_low": (60, 30, lambda px, h, r: hedge_face(px, h, r, (52, 70, 29), False), lambda px, t, r: green(px, t, r, (52, 70, 29))),
        "hedge_tall": (108, 34, lambda px, h, r: hedge_face(px, h, r, (49, 66, 27), False), lambda px, t, r: green(px, t, r, (49, 66, 27))),
        "short_wooden": (66, 4, lambda px, h, r: wood_face(px, h, r, (88, 64, 42)), None),
        "tall_wooden": (146, 4, lambda px, h, r: wood_face(px, h, r, (80, 46, 27)), None),
        "black_metal": (70, 2, lambda px, h, r: railing_face(px, h, r, (30, 30, 31)), None),
    }


def piece(p0, p1, spec, rng, depth=None) -> np.ndarray:
    """One style's picture along p0-p1 on the square: face (unless edge-on), then
    top; each pixel's depth value into `depth`, if given (walls._draw)."""
    hpx, tpx, face, top = spec
    out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
    H, t = hpx / W.PX, tpx / W.PX
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    length = math.hypot(*d)
    n = np.array([d[1], -d[0]]) / length
    if n.sum() < 0:
        n = -n                                                   # the side facing the viewer
    light = W.EAST_LIGHT + (1 - W.EAST_LIGHT) * max(0.0, n[1]) / (abs(n[0]) + abs(n[1]))
    px = max(1, round(length * W.PX))
    grow = d / length * (2.0 / W.PX)                             # two pixels past each end: no seam between pieces
    p0, d = p0 - grow, d + 2 * grow
    front = p0 + n * t / 2                                       # the box centred on the line
    back = -n * (t + 1.5 / W.PX)                                 # the top a pixel and a half over the face: no seam
    if abs(d[0] - d[1]) > 1e-9:                                  # along the view a face is edge-on
        W._draw(out, (front[0], front[1], H), (d[0], d[1], 0), (0, 0, -H), face(px, hpx, rng), light, depth=depth)
    if top is not None and tpx > 1:
        W._draw(out, (front[0] + back[0], front[1] + back[1], H), (d[0], d[1], 0), (-back[0], -back[1], 0),
                top(px, tpx, rng), W.TOP_LIGHT, depth=depth)
    elif abs(d[0] - d[1]) <= 1e-9:                               # a thin fence seen end-on: a line
        W._draw(out, (front[0], front[1], H), (d[0], d[1], 0), (-n[0] * 0.05, -n[1] * 0.05, 0),
                face(px, max(2, hpx // 20), rng)[:2], 1.0, depth=depth)
    # A soft shadow on the ground along the foot, both sides: ties the
    # picture to the floor it crosses at an angle (user, 2026-10-12: some
    # 45-degree walls looked floaty). Drawn last, only where nothing is.
    shade = np.zeros((6, px, 4))
    shade[..., 3] = (np.linspace(1, 0, 6) ** 1.5 * SHADOW_ALPHA)[:, None]
    for side in (1, -1):
        base = front if side == 1 else front + back
        W._draw(out, (base[0], base[1], 0.001), (d[0], d[1], 0),
                (side * n[0] * SHADOW_W, side * n[1] * SHADOW_W, 0), shade, 1.0, soft=True, depth=depth)
    return out


def invisible() -> np.ndarray:
    """A sprite that draws nothing: one pixel at alpha 1 (an empty one is dropped from the pack)."""
    out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
    out[F.TILE_H - 1, 63] = (0, 0, 0, 1)
    return out


depth_tile = W.depth_tile           # a sprite's depth picture (walls.py)


def build() -> tuple[list, list]:
    """The sheet's sprites and their depth pictures (DEPTH_<sheet>.png)."""
    specs, rng = styles(), np.random.default_rng(9)
    tiles, depths = [], []
    for name in STYLE_ORDER:
        for a, b in CUTS:
            dep = np.full((F.TILE_H, F.TILE_W), np.nan)
            tiles.append(piece(a, b, specs[name], rng, dep))
            depths.append(depth_tile(tiles[-1], dep))
        tiles += [None] * (16 - len(CUTS))
        depths += [None] * (16 - len(CUTS))
        for _k in range(3):
            tiles.append(invisible())
            depths.append(depth_tile(tiles[-1], np.full((F.TILE_H, F.TILE_W), np.nan)))
        tiles += [None] * (SLOTS - 19)
        depths += [None] * (SLOTS - 19)
    return tiles, depths


def preview(tiles) -> None:
    """Each style along a 45-degree and a 2:1 line, both ways, on our flags."""
    fl = F.sheet_tile(sheet_file("blends_street_uk_01.png"), 0)
    n, rows = 14, len(STYLE_ORDER)
    img = Image.new("RGBA", (n * 128 + 40, rows * 520), (40, 40, 40, 255))
    for r, name in enumerate(STYLE_ORDER):
        base = r * SLOTS
        things = {}
        for k in range(3):
            things[(1 + k, 4 - k)] = base + 0                    # 45, one way
            things[(5 + k, 2 + k)] = base + 1                    # 45, the other
        for k in range(2):
            things[(8 + 2 * k, 4 - k)] = base + 2
            things[(9 + 2 * k, 4 - k)] = base + 3
        sub = Image.new("RGBA", (n * 128, 12 * 64 + 192), (0, 0, 0, 0))
        cells = sorted({(x, y) for x in range(n) for y in range(7)}, key=lambda p: (p[0] + p[1], p[0]))
        for x, y in cells:
            sub.alpha_composite(Image.fromarray(fl), ((x - y) * 64 + 5 * 64, (x + y) * 32))
        for x, y in cells:
            if (x, y) in things:
                sub.alpha_composite(Image.fromarray(tiles[things[(x, y)]]), ((x - y) * 64 + 5 * 64, (x + y) * 32))
        img.alpha_composite(sub.crop((0, 160, n * 128, 680)), (0, r * 520))
    img.save(os.path.join(flags.BUILD, "preview", "diag_boundaries.png"))


def main() -> None:
    tiles, depths = build()
    F.write_sheet(sheet_file(SHEET + ".png"), tiles)
    F.write_sheet(sheet_file("DEPTH_" + SHEET + ".png"), depths)
    preview(tiles)
    print("wrote", SHEET, sum(t is not None for t in tiles), "pieces")


if __name__ == "__main__":
    main()
