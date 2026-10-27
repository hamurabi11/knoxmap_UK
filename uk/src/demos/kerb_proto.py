"""Prototype: kerbs chosen to follow a smooth road edge, drawn with the game's sprites.

    python kerb_proto.py        (pictures to out/roads/kerb_proto*.png)

A road outline is drawn at SS sub-tiles per tile. Its edge is cut into the
squares it crosses; each becomes a road square with the kerb tile whose
measured line (measure_kerbs.py) lies closest to the edge there, near-ties
going to the piece vanilla uses most (kerb_use.txt), and the verge tile whose
grass covers the part of the square outside the road best. Straight edges
must lie on tile boundaries, as vanilla's do.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


_SRC = os.path.dirname(os.path.abspath(__file__))
HERE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_SRC, "..", "..", "data", "kerbs")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
TILES = os.path.join(ukpaths.KNOX, "vendor", "PZMappingTools", "Tiles", "2x")
OUT = os.path.join(_SRC, "..", "..", "out", "roads")
SS = 8
W = H = 48

lines = {k: np.array(v) for k, v in json.load(open(HERE + "/kerbs_topdown.json")).items()}
# Only real kerb lines: drop near-empty pieces (caps, stubs).
lines = {k: v for k, v in lines.items() if len(v) >= 12}
# End caps (a kerb that stops) only where a run ends: not chosen mid-run.
CAPS = {f"street_curbs_01_{i}" for i in range(32, 40)}
lines = {k: v for k, v in lines.items() if k not in CAPS}
# How many vanilla cells use each piece: near-ties go to the common one (a
# plain straight kerb, not the one with an end cap).
use = {}
for row in open(HERE + "/kerb_use.txt"):
    n, name = row.split()
    use[name] = int(n)
top_use = max(use.values())
verges = {k: np.array(v, bool) for k, v in json.load(open(HERE + "/verge_cover.json")).items()}
penalty = {k: 0.10 * (1 - math.log(use.get(k, 0) + 1) / math.log(top_use + 1)) for k in lines}


def road_mask():
    """A two-lane road with a rounded 90-degree bend into a roundabout."""
    img = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(img)
    s = SS
    half = 3.5
    # roundabout: ring between r 4 (island) and r 10, centre (34, 30)
    cx, cy, R, r = 34.5, 30.5, 10, 4
    d.ellipse(((cx - R) * s, (cy - R) * s, (cx + R) * s, (cy + R) * s), fill=255)
    # arm west from the roundabout, then bending north with radius 9
    d.rectangle((14 * s, (cy - half) * s, cx * s - 1, (cy + half) * s - 1), fill=255)
    bx, by, br = 14, cy - 9.5, 9.5                      # bend centre: road turns from west to north
    d.pieslice(((bx - br - half) * s, (by - br - half) * s, (bx + br + half) * s, (by + br + half) * s),
               90, 180, fill=255)
    d.pieslice(((bx - br + half) * s, (by - br + half) * s, (bx + br - half) * s, (by + br - half) * s),
               90, 180, fill=0)
    d.rectangle(((bx - br - half) * s, 0, (bx - br + half) * s - 1, by * s - 1), fill=255)
    # arm south
    d.rectangle(((cx - half) * s, cy * s, (cx + half) * s - 1, H * s - 1), fill=255)
    d.ellipse(((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s), fill=0)
    return np.array(img) > 127


def kerbs_for(mask):
    # Edge, a hair inside the road: contour of the mask at 0.5, moved toward the road.
    dist_in = 0.05 * SS                          # a hair into the road: only to settle a point on a square's side
    from scipy import ndimage
    inside = ndimage.distance_transform_edt(mask)
    gy, gx = np.gradient(ndimage.gaussian_filter(inside, 2))
    # The edge, exactly: the boundary between a road sub-pixel and the
    # non-road one beside it. The nudge into the road only picks the square
    # that carries the kerb; the line keeps its true place on that square, so
    # a kerb along a square's side is matched to the straight piece on it.
    pts_all = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = np.roll(np.roll(mask, -dy, 0), -dx, 1)
        ys, xs = np.nonzero(mask & ~nb)
        pts_all.append(np.stack([ys + 0.5 + dy * 0.5, xs + 0.5 + dx * 0.5], 1))
    per_square: dict = {}
    for c in [np.concatenate(pts_all)]:
        for py, px in c:
            if not (SS <= px < mask.shape[1] - SS and SS <= py < mask.shape[0] - SS):
                continue
            iy, ix = int(min(py, mask.shape[0] - 1)), int(min(px, mask.shape[1] - 1))
            g = np.array([gx[iy, ix], gy[iy, ix]]); n = np.linalg.norm(g)
            qx, qy = (px + g[0] / n * dist_in, py + g[1] / n * dist_in) if n > 0 else (px, py)
            sq = (int(qx / SS), int(qy / SS))
            per_square.setdefault(sq, []).append((px / SS - sq[0], py / SS - sq[1]))
    road = mask.reshape(H, SS, W, SS).mean(axis=(1, 3)) > 0.5
    chosen = {}
    for sq, pts in per_square.items():
        if not (0 <= sq[0] < W and 0 <= sq[1] < H):
            continue
        p = np.array(pts)
        if len(p) < 3:
            continue
        best = None
        for name, q in lines.items():
            # symmetric mean nearest distance
            d1 = np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1))
            score = d1.min(1).mean() + d1.min(0).mean() + penalty[name]
            if best is None or score < best[0]:
                best = (score, name)
        # The verge: the part of the square outside the road, painted over by
        # the blend piece whose grass covers the most like it.
        sub = ~mask[sq[1] * SS:(sq[1] + 1) * SS, sq[0] * SS:(sq[0] + 1) * SS]
        verge = None
        if sub.mean() > 0.08:
            vbest = None
            for vk, cov in verges.items():
                inter = (cov & sub).sum(); union = (cov | sub).sum()
                iou = inter / union if union else 0
                if vbest is None or iou > vbest[0]:
                    vbest = (iou, vk)
            verge = vbest[1]
        chosen[sq] = best + (verge,)
        road[sq[1], sq[0]] = True                 # a kerb stands on road
    return road, chosen


def sprite(name, cache={}):
    sheet, idx = name.rsplit("_", 1)
    if sheet not in cache:
        cache[sheet] = Image.open(os.path.join(TILES, sheet + ".png")).convert("RGBA")
    r, c = divmod(int(idx), 8)
    return cache[sheet].crop((c * 128, r * 256, (c + 1) * 128, (r + 1) * 256))


def draw(road, chosen, out):
    img = Image.new("RGBA", ((W + H) * 64 + 128, (W + H) * 32 + 256), (25, 25, 28, 255))
    street, grass = sprite("blends_street_01_0"), sprite("blends_natural_01_16")
    for s in range(W + H):
        for x in range(W):
            y = s - x
            if not 0 <= y < H:
                continue
            sx, sy = (x - y) * 64 + H * 64, (x + y) * 32
            img.alpha_composite(street if road[y, x] else grass, (sx, sy))
    for s in range(W + H):
        for x in range(W):
            y = s - x
            if (x, y) in chosen:
                sx, sy = (x - y) * 64 + H * 64, (x + y) * 32
                if chosen[(x, y)][2] is not None:
                    img.alpha_composite(sprite("street_curbs_01_blend_medium_grass_" + str(chosen[(x, y)][2])), (sx, sy))
                img.alpha_composite(sprite(chosen[(x, y)][1]), (sx, sy))
    img = img.resize((img.width // 2, img.height // 2))
    img.save(out)


mask = road_mask()
road, chosen = kerbs_for(mask)
scores = [c[0] for c in chosen.values()]
print(len(chosen), "kerb squares; fit score median", round(float(np.median(scores)), 3), "worst", round(max(scores), 3))
draw(road, chosen, OUT + "/kerb_proto.png")
# top-down check: road squares and the chosen tiles' lines
top = Image.new("RGB", (W * 16, H * 16), (60, 90, 50))
d = ImageDraw.Draw(top)
for y in range(H):
    for x in range(W):
        if road[y, x]:
            d.rectangle((x * 16, y * 16, x * 16 + 15, y * 16 + 15), fill=(80, 80, 85))
for (x, y), (s, name, _v) in chosen.items():
    for u, v in lines[name]:
        d.point((x * 16 + u * 16, y * 16 + v * 16), fill=(255, 255, 255))
top.save(OUT + "/kerb_proto_top.png")
