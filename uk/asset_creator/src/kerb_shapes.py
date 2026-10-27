"""The outlines of our UK kerbs, one per vanilla kerb piece, for streetscape.py.

    python kerb_shapes.py          (the Knoxmap venv's Python: numpy, scipy)

kerbs.py chooses kerb pieces by matching the road's edge against each
vanilla piece's measured points (measure_kerbs.py's kerbs_topdown.json: up
to 200 points on its square where the kerb stands). Our kerbs must lie
exactly there, so each is drawn from those points - but the points cover
the kerb's top and its face as the sprite shows them (twice a kerb's width)
and are a random sample (ragged). So: the points traced (splatted, closed),
thinned to their centre line (Zhang-Suen) and its spurs pruned, its ends carried to the square's
edge where they stop just short of it (so neighbouring pieces meet), and a
band of a kerb's width (BAND) drawn along it. Measured, not traced: only
where the kerb runs comes from vanilla.

Writes asset_creator/renders/kerb_masks.json: {piece: [[i, j], ...]}, its
cells on a raster of GRID a metre with (0, 0) at the square's north corner (i
east, j south; a cell past the square's edge, negative or GRID and beyond).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # asset_creator: acpaths
import acpaths  # noqa: E402
KERBS = acpaths.KERBS_TOPDOWN
OUT = os.path.join(os.path.dirname(HERE), "renders", "kerb_masks.json")
GRID = 96                     # cells a metre
PAD = 48                      # cells round the square: a piece's sprite may draw past its square (diag_2_70)
N = GRID + 2 * PAD
SPLAT = 0.035                 # m round each measured point
BAND = 0.03                   # m: half a kerb's width, as vanilla draws it (a 60 mm top)
SPUR = 8                      # cells: side branches of the centre line shorter than this come off
TRIM = 14                     # cells off each end before it is carried on (a trace's round end bends the line)
EDGE_NEAR = 0.26              # m: a centre line ending this near the square's edge runs on to it


def thin(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: the mask's one-cell centre line."""
    img = img.copy().astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            ns = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(ns[:8])
            A = sum(((ns[k] == 0) & (ns[k + 1] == 1)).astype(int) for k in range(8))
            if step == 0:
                c = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                c = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            kill = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            return img.astype(bool)


def _ends(skel):
    """The line's end cells: those whose neighbours form one run round them
    (a diagonal step's end has two neighbours, touching, and is an end)."""
    p = np.pad(skel.astype(np.uint8), 1)
    ring = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    runs = sum(((ring[k] == 0) & (ring[(k + 1) % 8] == 1)).astype(int) for k in range(8))
    n = sum(r.astype(int) for r in ring)
    return skel & ((runs == 1) | (n == 0)) & (n <= 3)


def prune(skel: np.ndarray, n: int = SPUR) -> np.ndarray:
    """The centre line without the short spurs a bulge in the trace leaves:
    its ends eaten back n cells, then grown back along it from what is left
    (so the main line keeps its length, a spur does not come back)."""
    cut = skel.copy()
    for _ in range(n):
        cut &= ~_ends(cut)
    if not cut.any():
        return skel
    grown = cut.copy()
    for _ in range(n):
        ends = _ends(grown) & cut | _ends(grown)
        grown |= ndimage.binary_dilation(ends, np.ones((3, 3), bool)) & skel
    return grown


SNAP = [np.array(v) / np.linalg.norm(v) for v in
        [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
         (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)]]


def extend_ends(skel: np.ndarray) -> np.ndarray:
    """Carry each end of the centre line that stops near the square's edge on to it."""
    out = skel.copy()
    near = EDGE_NEAR * GRID
    for j, i in zip(*np.nonzero(_ends(skel))):
        dists = {"W": i - PAD, "E": PAD + GRID - 1 - i, "N": j - PAD, "S": PAD + GRID - 1 - j}
        side = min(dists, key=dists.get)
        if dists[side] > near or dists[side] <= 0:
            continue                                 # (an end past the square's edge stays where it is)
        # The way the line runs at this end: from the cells just behind it.
        w = 14                                       # the run of the line behind its end
        jj, ii = np.nonzero(skel[max(0, j - w):j + w + 1, max(0, i - w):i + w + 1])
        d = np.array([i - (ii.mean() + max(0, i - w)), j - (jj.mean() + max(0, j - w))])
        if np.linalg.norm(d) < 1e-6:
            d = {"W": (-1, 0), "E": (1, 0), "N": (0, -1), "S": (0, 1)}[side]
        d = np.asarray(d, float) / np.linalg.norm(d)
        # Snapped to the nearest of the kerbs' 16 directions (along the grid,
        # 45 degrees, 2:1): a trace's end drifts, a kerb does not.
        d = max(SNAP, key=lambda q: float(np.dot(q, d)))
        x, y = float(i), float(j)
        for _ in range(int(near * 2)):
            x, y = x + d[0], y + d[1]
            xi, yi = int(round(x)), int(round(y))
            if not (PAD <= xi < PAD + GRID and PAD <= yi < PAD + GRID):
                break
            out[yi, xi] = True
    return out


def outline(points) -> np.ndarray:
    yy, xx = np.mgrid[0:N, 0:N]
    fat = np.zeros((N, N), bool)
    r = SPLAT * GRID
    for u, v in points:
        fat |= (xx + 0.5 - (u * GRID + PAD)) ** 2 + (yy + 0.5 - (v * GRID + PAD)) ** 2 <= r * r
    fat = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(fat, 6), iterations=5)[6:-6, 6:-6])   # (a hole thins to a loop)
    fat = ndimage.gaussian_filter(fat.astype(float), 3.0, mode="nearest") > 0.5     # smoothed: a ragged trace thins to a wobbly line
    line = prune(thin(fat))
    trimmed = line.copy()
    for _ in range(TRIM):
        trimmed &= ~_ends(trimmed)
    line = extend_ends(trimmed if trimmed.sum() > 2 * TRIM else line)
    if not line.any():
        return fat
    return ndimage.distance_transform_edt(~line) <= BAND * GRID


def main() -> int:
    measured = json.load(open(KERBS))
    masks = {name: [[int(i) - PAD, int(j) - PAD] for j, i in zip(*np.nonzero(outline(pts)))]
             for name, pts in measured.items()}
    json.dump(masks, open(OUT, "w"))
    print(len(masks), "kerb outlines ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
