"""Snap a map's roads in a test square and report (new_road_approach.md, build order 1).

    python run.py <knoxmap map dir> <area data dir> <out dir> [--box x0,y0,x1,y1] [--no-os] [--no-lidar]
                  [--roads graph|grid|cost]

--roads: how built-up streets are routed (modes.py; graph, the default, as
before; --grid is --roads grid).

Classes are checked against OS Open Roads (osroads.py; the area's
os_roads.geojson) unless --no-os, then widths measured from the LiDAR already
downloaded (widths.py) unless --no-lidar. Writes <out>/checks.json, <out>/paths.json (each road's routed centreline, for style.py),
<out>/os_roads.json (the OS report), <out>/widths.json (each link's
measures), <out>/snapped.png (the snapped tiles - full grey,
half tiles lighter - with the real OSM centrelines over them in red and the
junctions in yellow) and <out>/now.png (the same square of the map's ground
as the current pipeline draws it), at 2 pixels a tile.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from roadgraph import network, osroads, raster, snap, widths  # noqa: E402
from generator import osm  # noqa: E402

TOWN_REACH = 40          # tiles from a building: in town (the narrower band)
PAD = 64                 # tiles round the map the roads are drawn on too (a road
                         # leaving the map and coming back is still joined)


def town_mask(map_dir: str, proj, H: int, W: int) -> np.ndarray:
    """Within TOWN_REACH tiles of a building in the map's OSM download."""
    info, _proj = network.projector(map_dir)
    feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or []
    m = np.zeros((H, W), bool)
    for f in feats:
        if f.kind != "way" or not f.tags.get("building") or not f.geometry:
            continue
        lat = sum(p[0] for p in f.geometry) / len(f.geometry)
        lon = sum(p[1] for p in f.geometry) / len(f.geometry)
        x, y = proj.to_px(lat, lon)
        if 0 <= x < W and 0 <= y < H:
            m[int(y), int(x)] = True
    return ndimage.distance_transform_edt(~m) <= TOWN_REACH


GRID_REACH = 12         # tiles from a building's outline: a road there runs on the grid (snap.GRID)


def grid_mask(map_dir: str, proj, H: int, W: int, reach: float = GRID_REACH) -> np.ndarray:
    """Within `reach` tiles of a building's outline in the map's OSM download (snap.GRID)."""
    info, _proj = network.projector(map_dir)
    feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or []
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    for f in feats:
        if f.kind != "way" or not f.tags.get("building") or not f.geometry or len(f.geometry) < 3:
            continue
        d.polygon([proj.to_px(lat, lon) for lat, lon in f.geometry], fill=1)
    m = np.array(img, bool)
    return ndimage.distance_transform_edt(~m) <= reach


def main() -> int:
    map_dir, area, out = sys.argv[1], sys.argv[2], sys.argv[3]
    box = tuple(int(v) for v in sys.argv[sys.argv.index("--box") + 1].split(",")) if "--box" in sys.argv \
        else (100, 0, 1100, 1000)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    nodes, links, proj, pins = network.build(map_dir)
    H, W = proj.height, proj.width
    x0, y0, x1, y1 = box
    keep = [i for i, l in enumerate(links) if any(x0 <= x < x1 and y0 <= y < y1 for x, y in l.pts)]
    sub = [links[i] for i in keep]
    used = {l.u for l in sub} | {l.v for l in sub}
    sub_nodes = {k: v for k, v in nodes.items() if k in used}
    print(f"{len(sub)} links, {len(sub_nodes)} nodes in the square {box}")
    os_roads = []
    if "--no-os" not in sys.argv and os.path.exists(os.path.join(area, "os_roads.geojson")):
        os_roads = osroads.load(area)
        os_report = osroads.conflate(sub, proj, os_roads, box)
        json.dump(os_report, open(os.path.join(out, "os_roads.json"), "w"), indent=1)
    if "--no-lidar" not in sys.argv:
        lidar = widths.Lidar(widths.box_for(proj, sub), os.path.join(area, "lidar"))
        measured = widths.measure(sub, proj, lidar)
        before = {i: l.width for i, l in enumerate(sub)}
        n = widths.apply(sub, measured)
        print(f"LiDAR: {len(measured)} of {len(sub)} links measured, {n} widths changed")
        json.dump({str(i): dict(m, name=sub[i].tags.get("name"), cls=sub[i].cls, was=before[i], now=sub[i].width)
                   for i, m in measured.items()}, open(os.path.join(out, "widths.json"), "w"), indent=1)
    town = town_mask(map_dir, proj, H, W)
    from roadgraph import modes
    mode = sys.argv[sys.argv.index("--roads") + 1] if "--roads" in sys.argv else "grid" if "--grid" in sys.argv \
        else "graph"
    try:
        print("road mode:", modes.prepare(mode, map_dir, area, H, W))
        paths, corner, merged, failed = snap.snap(sub_nodes, sub, town,
                                                  pins={n: w for n, w in pins.items() if n in sub_nodes})
    finally:
        modes.reset()
    t1 = time.time()
    padded = {i: [(x + PAD, y + PAD) for x, y in p] for i, p in paths.items()}
    owner, half = raster.rasterise(padded, sub, H + 2 * PAD, W + 2 * PAD)
    report = raster.checks(paths, sub, sub_nodes, corner, merged, owner, half)
    report["links"] = len(sub)
    report["routed"] = len(paths)
    report["failed"] = [sub[i].ways for i in failed]
    report["seconds"] = {"snap": round(t1 - t0, 1), "rasterise and check": round(time.time() - t1, 1)}
    json.dump(report, open(os.path.join(out, "checks.json"), "w"), indent=1)
    json.dump([{"link": i, "cls": sub[i].cls, "width": sub[i].width, "path": [list(c) for c in raster.vertices(p)]}
               for i, p in paths.items()], open(os.path.join(out, "paths.json"), "w"))
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Pictures, 2 px a tile.
    S = 2
    img = Image.new("RGB", (x1 - x0, y1 - y0), (46, 58, 40))
    a = np.array(img)
    full = owner[y0 + PAD:y1 + PAD, x0 + PAD:x1 + PAD] != 0
    a[full] = (150, 150, 150)
    for (x, y) in half:
        x, y = x - PAD, y - PAD
        if x0 <= x < x1 and y0 <= y < y1:
            a[y - y0, x - x0] = (205, 205, 205)
    img = Image.fromarray(a).resize(((x1 - x0) * S, (y1 - y0) * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for l in sub:
        d.line([((x - x0) * S, (y - y0) * S) for x, y in l.pts], fill=(230, 40, 40), width=1)
    # OS roads the map lacks, in blue.
    if os_roads:
        missing = {r["id"] for r in os_report["os only"]}
        for ln, props in os_roads:
            if props["id"] in missing:
                px = [proj.to_px(*widths.TO_LATLON.transform(x, y)[::-1]) for x, y in ln.coords]
                d.line([((x - x0) * S, (y - y0) * S) for x, y in px], fill=(40, 120, 255), width=3)
    for nid in sub_nodes:
        cx, cy = corner[nid]
        d.ellipse([(cx - x0) * S - 2, (cy - y0) * S - 2, (cx - x0) * S + 2, (cy - y0) * S + 2], fill=(255, 220, 0))
    img.save(os.path.join(out, "snapped.png"))
    name = os.path.basename(os.path.normpath(map_dir))
    now = Image.open(os.path.join(map_dir, f"{name}.bmp")).convert("RGB").crop(box)
    now.resize(((x1 - x0) * S, (y1 - y0) * S), Image.NEAREST).save(os.path.join(out, "now.png"))
    print("pictures in", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
