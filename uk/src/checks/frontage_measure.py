"""How well a built map keeps a street's frontages, against the real ones.

    python frontage_measure.py <knoxmap map dir> <area data dir> [--streets "High Street" ...]

Run after Build (reads <map>_footprints.npz, the tiles the game really gets).
Every STEP tiles along the street, square to it both sides, the distance from
the centre line to the first building, in the game (straightened street) and
in reality (the real street and outlines, in tiles). Per side:

  at line   share of the street with a building on the frontage line (within
            TOL tiles of that side's median distance) - continuity
  spread    interquartile range of the distance - how straight the line is
  none      share with no building within REACH tiles - gaps
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
from buildings import blocks as B  # noqa: E402
from knoxbuild.population import load_footprints  # noqa: E402

STEP = 2.0
REACH = 30.0
TOL = 1.5


def distances(line, built, side_sign):
    out = []
    H, W = built.shape
    for d in np.arange(0.0, line.length, STEP):
        foot = line.interpolate(d)
        _t, n = B.Street._dir(line, d)
        hit = None
        for r in np.arange(0.5, REACH, 0.5):
            x = int(foot.x + side_sign * n[0] * r)
            y = int(foot.y + side_sign * n[1] * r)
            if 0 <= x < W and 0 <= y < H and built[y, x]:
                hit = r
                break
        out.append(hit)
    return out


def summary(ds):
    got = np.array([v for v in ds if v is not None])
    if len(got) == 0:
        return {"at line": 0.0, "spread": float("nan"), "none": 1.0, "median": float("nan")}
    med = float(np.median(got))
    q1, q3 = np.percentile(got, [25, 75])
    return {"at line": float(np.mean([v is not None and abs(v - med) <= TOL for v in ds])),
            "spread": float(q3 - q1), "none": float(np.mean([v is None for v in ds])), "median": med}


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    names = sys.argv[sys.argv.index("--streets") + 1:] if "--streets" in sys.argv else ["High Street"]
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json")))
    b = info["bbox"]
    proj = B.Projector.build(b["south"], b["west"], b["north"], b["east"], 1.0, info.get("rotation") or 0.0)
    W, H = proj.width, proj.height
    game = np.zeros((H, W), bool)
    for x0, y0, m, _lv, _k in load_footprints(os.path.join(map_dir, f"{name}_footprints.npz")):
        h, w = m.shape
        game[y0:y0 + h, x0:x0 + w] |= m
    real_feats = json.load(open(os.path.join(area, "buildings.geojson"), encoding="utf-8"))["features"]
    real_px = [B.Polygon([proj.to_px(la, lo) for lo, la in f["geometry"]["coordinates"][0]]) for f in real_feats]
    img = Image.new("1", (W, H), 0)
    dr = ImageDraw.Draw(img)
    for p in real_px:
        dr.polygon(list(p.exterior.coords), fill=1)
    real = np.array(img, dtype=bool)
    streets, _roads, _list = B.load_streets(map_dir, info, proj, names, [p for p in real_px if p.is_valid])
    print(f"{name}")
    for st in streets:
        for sgn, label in ((1, "side A"), (-1, "side B")):
            r = summary(distances(st.real, real, sgn))
            g = summary(distances(st.game, game, sgn))
            print(f"  {st.name} {label}: real  at line {r['at line']:.0%}, spread {r['spread']:.1f}, "
                  f"none {r['none']:.0%}, median {r['median']:.1f}")
            print(f"  {' ' * len(st.name)} {' ' * len(label)}  game  at line {g['at line']:.0%}, "
                  f"spread {g['spread']:.1f}, none {g['none']:.0%}, median {g['median']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
