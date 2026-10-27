"""Listed structures that are not buildings: a war memorial, stocks, a milestone.

    python structure_kit.py   (KnoxMap's Python)

Writes Tile_creator/build/sheets/<group>/listed_structures_uk_01.png, its DEPTH_
sheet, .slots.json and a preview:

    0  war memorial, its lower storey: three steps, the pedestal, the shaft
    1  war memorial, its upper storey: the shaft and the wheel-head cross
       (Odiham's, 1920: a limestone wheel-head cross on a stepped base)
    2  village stocks and whipping post (oak)
    3  milestone (painted stone, a dark plate)

Built from boxes on the square (doorway_kit.Piece: wx east, wy south, z up,
metres), drawn with walls.py's camera; what rises above a storey (3 m) is
drawn a storey up as its own sprite, laid at z 1 (as the telegraph pole's
top is). apply.py stands each at its list entry's point (STRUCTURE_TILES).
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
SHEET = "listed_structures_uk_01"
STONE, PALE, OAK = (176, 172, 160), (196, 192, 180), (92, 74, 56)
STOREY_M = 3.0


def built(boxes, level: int):
    """The sprite of these boxes' part on storey `level` (boxes: o0 o1 a0 a1 z0 z1 rgb)."""
    p = DK.Piece(north=False)
    z_lo, z_hi = level * STOREY_M, (level + 1) * STOREY_M
    for o0, o1, a0, a1, z0, z1, rgb in boxes:
        lo, hi = max(z0, z_lo), min(z1, z_hi)
        if hi > lo:
            p.box(o0, o1, a0, a1, lo - z_lo, hi - z_lo, rgb)
    return p.draw()


def memorial():
    b = [(0.05, 0.95, 0.05, 0.95, 0.00, 0.15, STONE), (0.15, 0.85, 0.15, 0.85, 0.15, 0.30, STONE),
         (0.25, 0.75, 0.25, 0.75, 0.30, 0.45, STONE),                               # three steps
         (0.32, 0.68, 0.32, 0.68, 0.45, 1.45, PALE),                                # the pedestal
         (0.34, 0.66, 0.34, 0.66, 1.45, 1.60, STONE),
         (0.43, 0.57, 0.43, 0.57, 1.60, 4.85, PALE),                                # the shaft
         (0.45, 0.55, 0.28, 0.72, 4.05, 4.25, PALE), (0.28, 0.72, 0.45, 0.55, 4.05, 4.25, PALE),   # the arms
         # the wheel: short rings between the arms
         (0.45, 0.55, 0.33, 0.38, 3.85, 4.05, PALE), (0.45, 0.55, 0.62, 0.67, 3.85, 4.05, PALE),
         (0.45, 0.55, 0.33, 0.38, 4.25, 4.45, PALE), (0.45, 0.55, 0.62, 0.67, 4.25, 4.45, PALE),
         (0.33, 0.38, 0.45, 0.55, 3.85, 4.05, PALE), (0.62, 0.67, 0.45, 0.55, 3.85, 4.05, PALE),
         (0.33, 0.38, 0.45, 0.55, 4.25, 4.45, PALE), (0.62, 0.67, 0.45, 0.55, 4.25, 4.45, PALE)]
    return [built(b, 0), built(b, 1)]


def stocks():
    b = [(0.44, 0.56, 0.18, 0.26, 0.0, 1.00, OAK), (0.44, 0.56, 0.74, 0.82, 0.0, 1.00, OAK),   # posts
         (0.45, 0.55, 0.18, 0.82, 0.32, 0.50, OAK), (0.45, 0.55, 0.18, 0.82, 0.52, 0.68, OAK),   # boards
         (0.46, 0.54, 0.88, 0.95, 0.0, 1.85, OAK)]                                    # the whipping post
    return [built(b, 0)]


def milestone():
    b = [(0.40, 0.60, 0.36, 0.64, 0.0, 0.72, (214, 212, 204)),
         (0.60, 0.61, 0.40, 0.60, 0.25, 0.62, (40, 40, 44))]                          # the plate, facing east
    return [built(b, 0)]


def main() -> None:
    pieces = memorial() + stocks() + milestone()
    tiles = [t for t, _d in pieces]
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"), [W.depth_tile(t, d) for t, d in pieces])
    props = [{"props": [["solid", ""], ["CustomName", "War Memorial"]]},
             {"props": [["CustomName", "War Memorial"]]},
             {"props": [["solidtrans", ""], ["CustomName", "Village Stocks"]]},
             {"props": [["solidtrans", ""], ["CustomName", "Milestone"]]}]
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(props, f)
    img = Image.new("RGBA", (128 * 3, 512), (80, 100, 80, 255))
    img.alpha_composite(Image.fromarray(tiles[1]), (0, 0))
    img.alpha_composite(Image.fromarray(tiles[0]), (0, 192))
    img.alpha_composite(Image.fromarray(tiles[2]), (128, 192))
    img.alpha_composite(Image.fromarray(tiles[3]), (256, 192))
    img.save(os.path.join(WK.PREVIEW, f"kit_{SHEET}.png"))
    print("wrote", SHEET)


if __name__ == "__main__":
    main()
