"""Does each road keep one width along its length? A check, after kerbs.py.

    python road_width_check.py <knoxmap map dir> [--before] [--list N]

For every straight stretch of road (the straightened lines Generate drew),
the tarmac is counted across it at each step along its middle - per column
for a road along x, per row for one along y or on the diagonal - leaving
END_TILES plus its width clear of each end (junctions and bends). A stretch
whose count changes for more than a couple of steps is reported: by how
much, how often, and where. Forks and long junction mouths (two roads side by
side for a while) show up as changes too: read the list, don't chase the total.
--before checks the ground as Generate drew it (<map>_ground_base.prekerbs.bmp).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # src: our packages
from streets import diagonals as D  # noqa: E402
from generator.renderer import _pixel_width, _way_width_m  # noqa: E402
from streets import kerbs as K  # noqa: E402

MIN_STEPS = 12          # stretches with fewer steps measured are not judged
END_TILES = 3


def main() -> int:
    map_dir = sys.argv[1]
    before = "--before" in sys.argv
    show = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 8
    name = os.path.basename(os.path.normpath(map_dir))
    path = os.path.join(map_dir, f"{name}_ground_base{'.prekerbs' if before else ''}.bmp")
    ground = np.array(Image.open(path).convert("RGB"))
    road = K.is_colour(ground, K.ROAD)
    H, W = road.shape
    from buildings import blocks as B
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    bb = info["bbox"]
    proj = B.Projector.build(bb["south"], bb["west"], bb["north"], bb["east"], info["meters_per_tile"],
                             info.get("rotation") or 0.0)
    judged = varied = 0
    worst = []
    by_kind = {"along x": [0, 0], "along y": [0, 0], "diagonal": [0, 0]}
    for cat, f, a, b, diagonal in D._segments(map_dir, info, proj):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if diagonal:
            kind = "diagonal"
        elif abs(dy) <= 0.05 * length:
            kind = "along x"
        elif abs(dx) <= 0.05 * length:
            kind = "along y"
        else:
            continue
        guess = _pixel_width(_way_width_m(f, cat), info["meters_per_tile"])
        reach = int(math.ceil(guess * (math.sqrt(2) if kind == "diagonal" else 1))) + 3
        counts, where = [], []
        steps = int(length)
        for i in range(steps + 1):
            t = i / max(1, steps)
            x, y = a[0] + dx * t, a[1] + dy * t
            ix, iy = int(x), int(y)
            if not (0 <= ix < W and 0 <= iy < H) or not road[iy, ix]:
                continue
            # The tarmac across the road at this step, within its own reach:
            # one that runs to the edge of the reach is something else
            # joining (a side road, a car park) and is not counted.
            if kind == "along x":
                line = road[max(0, iy - reach):iy + reach + 1, ix]
            else:
                line = road[iy, max(0, ix - reach):ix + reach + 1]
            centre = min(reach, iy if kind == "along x" else ix)
            lo = hi = centre
            while lo > 0 and line[lo - 1]:
                lo -= 1
            while hi < len(line) - 1 and line[hi + 1]:
                hi += 1
            if lo == 0 or hi == len(line) - 1:
                continue
            counts.append(hi - lo + 1)
            where.append((ix, iy))
        if not counts:
            continue
        typical = int(np.median(counts))
        trim = typical + END_TILES
        counts, where = counts[trim:len(counts) - trim], where[trim:len(where) - trim]
        if len(counts) < MIN_STEPS:
            continue
        judged += 1
        by_kind[kind][0] += 1
        # A side road or a drive joining makes the count jump for a step or
        # two; a change that lasts is what is counted.
        c = np.array(counts)
        off = c != typical
        runs = np.diff(np.concatenate([[0], off.astype(int), [0]]))
        starts, ends = np.nonzero(runs == 1)[0], np.nonzero(runs == -1)[0]
        lasting = [(s, e) for s, e in zip(starts, ends) if e - s >= 3]
        if lasting:
            varied += 1
            by_kind[kind][1] += 1
            spread = int(c.max() - c.min())
            s0, e0 = lasting[0]
            worst.append((spread, len(lasting), kind, f.tags.get("name") or str(f.osm_id), where[s0]))
    print(f"stretches judged: {judged}; width changes along them: {varied}")
    for k, (j, v) in by_kind.items():
        print(f"  {k:9} {v} of {j}")
    for spread, n, kind, nm, at in sorted(worst, reverse=True)[:show]:
        print(f"  {kind:9} {nm[:28]:28} spread {spread}, {n} changes, first at {at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
