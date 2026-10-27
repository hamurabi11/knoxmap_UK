"""A pub's beer garden as it stood in 1993 (PLAN.md step 5): the picnic bench.

    python pub_props.py [--preview]   (KnoxMap's Python)

Sheet commercial_pub_uk_01 (its DEPTH_ sheet, .slots.json) and built/pub_props.json
(the tiles by part, inlined in regional.json for Build's beer_gardens.py):

    bench_x_0, bench_x_1   the A-frame picnic bench along west-east, its west and east squares
    bench_y_0, bench_y_1   ...along north-south, its north and south squares

The pub garden's slatted softwood table with its seats fixed either side, 1.8 m long
and 0.9 m across (a square: drawn wider, its ends were cut), weathered to a grey-brown; drawn as farm_props.py draws (faces in
metres through walls.py's camera, with depth), the whole bench in metres and each
square's part cut from it. Appended only: a part's place fixes its ids.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import farm_props as FP  # noqa: E402
import fascias as FA  # noqa: E402

WK, W = FP.WK, FP.W
SHEET = "commercial_pub_uk_01"
WOOD = (132, 106, 78)
LONG, WIDE = (0.1, 1.9), 0.46            # m: along the bench (over two squares), half its width across
                                         # (within the square: wider, its seats' ends were cut by the tile's edge)


def _slats(rng, along):
    t = FP._tex(WOOD, rng, n=32, grain=5)
    if along == "s":
        t[::8, :, :3] *= 0.6                                   # the gaps between its boards
    else:
        t[:, ::8, :3] *= 0.6
    return t


def bench_boxes():
    """The bench along x in metres: (x0, x1, y0, y1, z0, z1, part); y centred on 0.5."""
    c, (a0, a1) = 0.5, LONG
    out = [
        (a0, a1, c - 0.24, c + 0.24, 0.70, 0.75, "top"),       # the table
        (a0, a1, c - WIDE, c - 0.32, 0.40, 0.45, "seat"),      # its seats either side
        (a0, a1, c + 0.32, c + WIDE, 0.40, 0.45, "seat"),
    ]
    for x in (a0 + 0.25, a1 - 0.3):                            # the frames at each end
        out.append((x, x + 0.06, c - WIDE + 0.02, c + WIDE - 0.02, 0.36, 0.40, "frame"))
        for y in (c - 0.40, c - 0.14, c + 0.08, c + 0.34):
            out.append((x, x + 0.06, y, y + 0.06, 0.0, 0.70 if abs(y - c) < 0.2 else 0.40, "frame"))
    return out


def piece(rng, along, k):
    """Square k (0, 1) of the bench along `along` (x or y)."""
    s = FP.Sprite()
    grain = "s" if along == "x" else "t"                       # its boards run along it
    tex = {"top": _slats(rng, grain), "seat": _slats(rng, grain), "frame": FP._tex(np.array(WOOD) * 0.85, rng, n=16, grain=4)}
    for x0, x1, y0, y1, z0, z1, part in bench_boxes():
        lo, hi = max(x0, k), min(x1, k + 1)                    # its part on square k
        if hi <= lo:
            continue
        lo, hi = lo - k, hi - k
        if along == "x":
            s.box(lo, hi, y0, y1, z0, z1, tex[part], tex[part])
        else:
            s.box(y0, y1, lo, hi, z0, z1, tex[part], tex[part])
    out, depth = s.draw()
    return out, depth


PARTS = [("bench_x_0", "x", 0), ("bench_x_1", "x", 1), ("bench_y_0", "y", 0), ("bench_y_1", "y", 1)]


def main(argv) -> int:
    rng = np.random.default_rng(1993)
    tiles, depths, built = [], [], {"sheet": SHEET, "parts": {}}
    for name, along, k in PARTS:
        t, d = piece(rng, along, k)
        built["parts"][name] = f"{SHEET}_{len(tiles)}"
        tiles.append(t)
        depths.append(d)
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"), [W.depth_tile(t, d) for t, d in zip(tiles, depths)])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump([{"props": [["solidtrans", ""], ["CustomName", "Picnic Bench"]]}] * len(tiles), f)
    os.makedirs(FA.BUILT, exist_ok=True)
    with open(os.path.join(FA.BUILT, "pub_props.json"), "w", encoding="utf-8") as f:
        json.dump(built, f, indent=1)
    if "--preview" in argv:
        img = Image.new("RGBA", (520, 300), (96, 110, 80, 255))
        for k in range(2):                                     # along x at left, along y at right
            img.alpha_composite(Image.fromarray(tiles[k]), (60 + 64 * k, 20 + 32 * k))
        for k in range(2):
            img.alpha_composite(Image.fromarray(tiles[2 + k]), (360 - 64 * k, 20 + 32 * k))
        img.save(os.path.join(FP.SM.PREVIEW, "pub_bench.png"))
    order = os.path.join(HERE, "sheets.order.json")
    have = json.load(open(order, encoding="utf-8"))
    if SHEET not in have:
        have.append(SHEET)
    json.dump(have, open(order, "w", encoding="utf-8"), indent=1)
    print("wrote", SHEET, len(tiles))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
