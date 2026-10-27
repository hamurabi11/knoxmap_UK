"""Measure B42's kerb and verge tiles on their square, seen from above.

    python measure_kerbs.py [<out dir>]       (default ../../data/kerbs)

Writes
- kerbs_topdown.json: for each kerb tile (street_curbs_01, _diag, _diag_2),
  up to 200 points (u east, v south, 0..1 on its square) where the kerb stands;
- verge_cover.json: for each verge tile (street_curbs_01_blend_medium_grass;
  the other grounds share its layout), which of 8 x 8 sample points it paints.

The 2x sprite is 128 x 256; the floor diamond is its bottom 64 rows, and a
point (u, v) on the square is drawn at
    sx = 64 + (u - v) * 64,   sy = 192 + (u + v) * 32.
A kerb stands about 7 px tall, so each of its pixels is taken 3 px lower (its
mid-height) before projecting back. kerb_use.txt (how many vanilla cells use
each tile) is counted from the game's lotheaders, see LOG 2026-10-08.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

TILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                     "Knoxmap", "vendor", "PZMappingTools", "Tiles", "2x")
KERB_SHEETS = {"street_curbs_01": 64, "street_curbs_01_diag": 80, "street_curbs_01_diag_2": 80}
VERGE_SHEET = "street_curbs_01_blend_medium_grass"
LIFT = 3.0          # px: half a kerb's height
GRID = 8


def to_square(sx, sy):
    d, s = (sx - 64) / 64, (sy - 192) / 32
    return (s + d) / 2, (s - d) / 2


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "kerbs")
    rng = np.random.default_rng(0)
    kerbs = {}
    for sheet, n in KERB_SHEETS.items():
        im = np.array(Image.open(os.path.join(TILES, sheet + ".png")).convert("RGBA"))
        for i in range(n):
            r, c = divmod(i, 8)
            ys, xs = np.nonzero(im[r * 256:(r + 1) * 256, c * 128:(c + 1) * 128, 3] > 40)
            if len(xs) < 20:
                continue
            u, v = to_square(xs + 0.5, ys + 0.5 + LIFT)
            pts = np.stack([u, v], 1)
            if len(pts) > 200:
                pts = pts[rng.choice(len(pts), 200, replace=False)]
            kerbs[f"{sheet}_{i}"] = pts.round(3).tolist()
    verges = {}
    im = np.array(Image.open(os.path.join(TILES, VERGE_SHEET + ".png")).convert("RGBA"))
    for i in range((im.shape[0] // 256) * 8):
        r, c = divmod(i, 8)
        a = im[r * 256:(r + 1) * 256, c * 128:(c + 1) * 128, 3]
        cov = np.zeros((GRID, GRID), bool)
        for y in range(GRID):
            for x in range(GRID):
                u, v = (x + .5) / GRID, (y + .5) / GRID
                sx, sy = int(64 + (u - v) * 64), int(192 + (u + v) * 32)
                cov[y, x] = a[min(sy, 255), min(sx, 127)] > 100
        if cov.any():
            verges[str(i)] = cov.astype(int).tolist()
    os.makedirs(out, exist_ok=True)
    json.dump(kerbs, open(os.path.join(out, "kerbs_topdown.json"), "w"))
    json.dump(verges, open(os.path.join(out, "verge_cover.json"), "w"))
    print(f"{len(kerbs)} kerb tiles, {len(verges)} verge tiles -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
