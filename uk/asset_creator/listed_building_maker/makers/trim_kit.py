"""Wall trims for the listed buildings: a plinth, a first-floor band, an eaves cornice.

    python trim_kit.py   (KnoxMap's Python)

Writes Tile_creator/build/sheets/<group>/wall_trim_uk_01.png, its DEPTH_ sheet,
.slots.json and a preview. A wall piece serves every storey, so what belongs
to one storey only is laid over it, square by square along the building's
outside walls (knoxbuild/tbx.py `wall_trim`, regional_uk TRIMS): the plinth
along the ground floor's foot, the band along the first floor's, the cornice
along the top storey's head. Each a west and a north piece (slot 2 k, 2 k + 1):

    0  plinth, tarred black (render, painted brick)
    2  plinth, a projecting brick plinth (brick, flint)
    4  band, painted (render, painted brick)
    6  band, stone (brick, flint)
    8  cornice, painted
    10 cornice, stone
    12 parapet, brick with a stone coping (stood on the roof's floor, along the eaves)
    14 parapet, painted with a stone coping

Thin boxes on the wall's face (doorway_kit.Piece, metres), projecting a few
centimetres; a cornice in two steps, its underside in shadow.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import doorway_kit as DK  # noqa: E402
import wall_kit as WK  # noqa: E402

W = WK.W
SHEET = "wall_trim_uk_01"
WHITE, STONE, TAR, BRICK = (230, 228, 220), (196, 190, 172), (52, 50, 48), (118, 66, 52)
F = DK.WALL                                              # the wall's face


def piece(boxes, north):
    p = DK.Piece(north)
    for o0, o1, z0, z1, rgb in boxes:
        p.box(o0, o1, 0.0, 1.0, z0, z1, rgb)
    return p.draw()


def plinth(rgb):
    return [(F, F + 0.04, 0.0, 0.36, rgb), (F, F + 0.05, 0.36, 0.40, tuple(min(255, c * 1.15) for c in rgb))]


def band(rgb):
    return [(F, F + 0.035, 0.00, 0.13, rgb)]


def cornice(rgb):
    shade = tuple(c * 0.62 for c in rgb)
    return [(F, F + 0.06, 2.70, 2.82, shade), (F, F + 0.13, 2.82, 2.96, rgb), (F, F + 0.16, 2.96, 3.0, rgb)]


def parapet(rgb):
    return [(F - 0.02, F + 0.10, 0.0, 0.62, rgb), (F - 0.03, F + 0.13, 0.62, 0.72, STONE)]


TRIMS = [plinth(TAR), plinth(BRICK), band(WHITE), band(STONE), cornice(WHITE), cornice(STONE),
         parapet(BRICK), parapet(WHITE)]
NAMES = ["plinth_black", "plinth_brick", "band_white", "band_stone", "cornice_white", "cornice_stone",
         "parapet_brick", "parapet_white"]


def main() -> None:
    pieces = [piece(b, north) for b in TRIMS for north in (False, True)]
    tiles = [t for t, _d in pieces]
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"), [W.depth_tile(t, d) for t, d in pieces])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump([None] * len(tiles), f)                     # seen only: no properties
    wall = WK.vanilla_tile("walls_exterior_house_01", 4), WK.vanilla_tile("walls_exterior_house_01", 5)
    img = Image.new("RGBA", (128 * len(tiles), 256), (80, 100, 80, 255))
    for k, t in enumerate(tiles):
        img.alpha_composite(Image.fromarray(np.ascontiguousarray(wall[k % 2])), (k * 128, 0))
        img.alpha_composite(Image.fromarray(t), (k * 128, 0))
    img.save(os.path.join(WK.PREVIEW, f"kit_{SHEET}.png"))
    print("wrote", SHEET, NAMES)


if __name__ == "__main__":
    main()
