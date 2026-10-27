"""Pictures to check blocks.py by eye.

    python blocks_check.py <knoxmap map dir> <area data dir> <out prefix> [--box x0,y0,w,h]

<prefix>_game.png  the map's ground, each block's tiles in its own colour
                   (outlined), so gaps, forecourts and frontages show
<prefix>_real.png  real coordinates: the LiDAR's buildings (grey), the OSM
                   outlines as registered (white), LiDAR buildings no outline
                   covers (red)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import shapely
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import shape
from shapely.ops import transform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
from buildings import blocks as B  # noqa: E402

K = 4


def main() -> int:
    map_dir, area, prefix = sys.argv[1:4]
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json")))
    b = info["bbox"]
    proj = B.Projector.build(b["south"], b["west"], b["north"], b["east"], 1.0, info["rotation"])
    W, H = proj.width, proj.height
    bx = [int(v) for v in sys.argv[sys.argv.index("--box") + 1].split(",")] if "--box" in sys.argv \
        else [0, 0, W, H]
    x0, y0, w, h = bx
    ground = np.array(Image.open(os.path.join(map_dir, f"{name}_ground_base.bmp")).convert("RGB"))
    feats = json.load(open(os.path.join(map_dir, f"{name}_buildings.geojson"), encoding="utf-8"))["features"]
    img = Image.fromarray(ground[y0:y0 + h, x0:x0 + w]).resize((w * K, h * K), Image.NEAREST)
    dr = ImageDraw.Draw(img, "RGBA")
    rng = np.random.default_rng(3)
    colours = {}
    for ft in feats:
        p = ft["properties"]
        if not p.get("placed"):
            continue
        blk = p["block"]
        if blk not in colours:
            colours[blk] = tuple(int(v) for v in rng.integers(60, 230, 3))
        ring = [proj.to_px(lat, lon) for lon, lat in ft["geometry"]["coordinates"][0]]
        pts = [((x - x0) * K, (y - y0) * K) for x, y in ring]
        dr.polygon(pts, fill=colours[blk] + (200,), outline=(0, 0, 0, 255))
    img.save(prefix + "_game.png")

    # Real coordinates around the same tiles.
    corners = [proj.to_latlon(x, y) for x, y in ((x0, y0), (x0 + w, y0), (x0, y0 + h), (x0 + w, y0 + h))]
    es, ns = zip(*[B.TO_BNG(lon, lat) for lat, lon in corners])
    e0, n0, e1, n1 = int(min(es)), int(min(ns)), int(max(es)) + 1, int(max(ns)) + 1
    g = B.Ground(area, (e0 - 5, n0 - 5, e1 + 5, n1 + 5))
    built = g.built[5:-5, 5:-5]
    img2 = Image.fromarray(np.where(built, 110, 30).astype(np.uint8)).convert("RGB")
    img2 = img2.resize((img2.width * K, img2.height * K), Image.NEAREST)
    d2 = ImageDraw.Draw(img2)
    real = {str(f["properties"]["osm_id"]): f for f in
            json.load(open(os.path.join(area, "buildings.geojson"), encoding="utf-8"))["features"]}

    def draw(poly, colour, width=2):
        for gg in getattr(poly, "geoms", [poly]):
            d2.line([((x - e0) * K, (n1 - y) * K) for x, y in gg.exterior.coords], fill=colour, width=width)
    seen = set()
    for ft in feats:
        p = ft["properties"]
        oid = str(p.get("parent_osm_id") or p.get("osm_id"))
        if oid in seen or oid not in real or not p.get("placed"):
            continue
        seen.add(oid)
        poly = affinity.translate(transform(B.TO_BNG, shape(real[oid]["geometry"])), *p.get("lidar_shift", (0, 0)))
        draw(poly, (255, 255, 255))
    only_path = os.path.join(map_dir, "lidar_only.geojson")
    if os.path.exists(only_path):
        for ft in json.load(open(only_path))["features"]:
            draw(transform(B.TO_BNG, shape(ft["geometry"])), (255, 60, 60), 3)
    img2.save(prefix + "_real.png")
    print("wrote", prefix + "_game.png", prefix + "_real.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
