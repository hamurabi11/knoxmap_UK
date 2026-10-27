"""Drains: black cast-iron downpipes with hopper heads, into gully grates.

    python drain_kit.py   (KnoxMap's Python)

Writes Tile_creator/build/sheets/<group>/drains_uk_01.png, its DEPTH_ sheet,
.slots.json and a preview. A downpipe is one storey a piece, on a west or a
north wall, near either end of the wall's square (ALONG: clear of the window
opening, 0.23-0.77 along - user, 2026-10-12: one stood in front of a window):

    slot  4 k + 0  foot: the pipe and its shoe over a gully grate on the ground
          4 k + 1  middle storey
          4 k + 2  top: the pipe and its hopper head under the eaves
          4 k + 3  a one-storey building's: hopper, pipe and grate together
    k = 0 west wall, near the square's start;  1 west, near its end;
        2 north, near the start;               3 north, near the end

Build lays them (knoxbuild/tbx.py `drains`) at each end of every outside
wall run the camera sees, from the ground floor to the eaves, on the
WallFurniture2 layer (beside the trims on WallFurniture). From
Tile_creator/src/house_walls.py's prototype (PLAN I), drawn here with depth.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wall_kit as WK  # noqa: E402

W, F = WK.W, WK.F
SHEET = "drains_uk_01"
PIPE = (28, 28, 30)
ALONG = (0.08, 0.92)
OFF = 0.06                     # stands off the wall's face
H = 3.0                        # a storey


def _solid(h, w, rgb, lit_col=None):
    t = np.zeros((h, w, 4))
    t[..., :3] = rgb
    t[..., 3] = 255
    if lit_col is not None:
        t[:, lit_col, :3] *= 1.6
    return t


def piece(side: str, along: float, part: str):
    out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
    depth = np.full((F.TILE_H, F.TILE_W), np.nan)
    draw = lambda o, a, b, tex: W._draw(out, o, a, b, tex, 1.0, depth=depth)
    pt = (lambda a_, o_, z: (o_, a_, z)) if side == "W" else (lambda a_, o_, z: (a_, o_, z))
    vec = (lambda a_, o_, z: np.array(pt(a_, o_, z), float))
    top = part in ("top", "single")
    foot = part in ("foot", "single")
    z_hi = H - 0.22 if top else H                       # under the hopper
    # the pipe: a 4 px strip, its lit edge
    draw(pt(along - 2 / 64, OFF, z_hi), vec(4 / 64, 0, 0), vec(0, 0, -z_hi), _solid(64, 4, PIPE, 0))
    if top:                                              # the hopper head: a wider box under the eaves
        draw(pt(along - 5 / 64, OFF + 0.02, H - 0.06), vec(10 / 64, 0, 0), vec(0, 0, -0.18), _solid(12, 10, PIPE))
        draw(pt(along - 5 / 64, OFF - 0.02, H - 0.06), vec(10 / 64, 0, 0), vec(0, 0.04, 0),
             _solid(4, 10, tuple(c * 1.8 for c in PIPE)))
    if foot:                                             # the shoe and the gully grate on the ground
        draw(pt(along - 2 / 64, OFF, 0.12), vec(4 / 64, 0, 0), vec(0, 0.06, -0.10), _solid(8, 4, PIPE))
        g = _solid(10, 10, (40, 40, 42))
        g[::3, :, :3] = 88
        draw(pt(along - 0.09, 0.02, 0.002), vec(0.18, 0, 0), vec(0, 0.18, 0), g)
    return out, depth


def main() -> None:
    pieces = []
    for side in ("W", "N"):
        for along in ALONG:
            for part in ("foot", "mid", "top", "single"):
                pieces.append(piece(side, along, part))
    tiles = [t for t, _d in pieces]
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"), [W.depth_tile(t, d) for t, d in pieces])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump([None] * len(tiles), f)                 # seen only: no properties
    # Preview: a two-storey corner of brick, pipes at both ends of each wall.
    wall = WK.vanilla_tile("walls_exterior_house_01", 4), WK.vanilla_tile("walls_exterior_house_01", 5)
    img = Image.new("RGBA", (128 * 8, 256 + 192), (80, 100, 80, 255))
    for k in range(8):
        side, along = k // 4, (k // 2) % 2
        base = 8 * side + 4 * along
        for z, part in ((1, 2), (0, 0)):
            img.alpha_composite(Image.fromarray(np.ascontiguousarray(wall[side])), (k * 128, (1 - z) * 192))
            img.alpha_composite(Image.fromarray(tiles[base + part]), (k * 128, (1 - z) * 192))
    img.save(os.path.join(WK.PREVIEW, f"kit_{SHEET}.png"))
    print("wrote", SHEET)


if __name__ == "__main__":
    main()
