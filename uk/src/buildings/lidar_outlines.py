"""Landmark outlines from the LiDAR, on a KnoxMap map's tiles, before Build.

    python lidar_outlines.py <knoxmap map dir> <area data dir> [--ids <osm_id> ...]

Run after tidy_outlines.py and before export_regional.py / roof_plan.py.

For landmarks - Grade I and II* listed buildings and places of worship, or
the OSM ids given - the outline the map builds is replaced by what the LiDAR
shows standing there: tiles more than BUILT_M above the ground, within REACH
tiles of the OSM outline (so trees and neighbours stay out), cleaned of
specks and slivers narrower than two tiles, the biggest piece kept with its
holes filled, and its edges squared to the tile grid. A church drawn in OSM
a few metres wide of its walls then comes out the width it really is, its
tower exactly where the tower is.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
from regional import roof_plan as RP  # noqa: E402
from buildings.tidy_outlines import snap_rectilinear  # noqa: E402

BUILT_M = 2.5
REACH = 4
MIN_TILES = 20
MAX_CHANGE = 0.6          # refuse an outline differing from OSM's by more than this share
MIN_SIDE = 3              # parts of the outline narrower than this are left off


def regions(m):
    lab, info = RP.regions(m.astype(int), m)
    return lab, {k: a for k, (c, a) in info.items() if c == 1}


def clean(built: np.ndarray) -> np.ndarray:
    """Open with a 2x2 square (drops specks and one-tile slivers), keep the
    biggest piece, fill its holes."""
    h, w = built.shape
    er = built.copy()
    er[:-1, :-1] = built[:-1, :-1] & built[1:, :-1] & built[:-1, 1:] & built[1:, 1:]
    er[-1, :] = False
    er[:, -1] = False
    op = er.copy()
    op[1:, :] |= er[:-1, :]
    op[:, 1:] |= er[:, :-1]
    op[1:, 1:] |= er[:-1, :-1]
    op &= built
    lab, info = regions(op)
    if not info:
        return op
    keep = lab == max(info, key=info.get)
    # Holes: empty tiles not reachable from the border.
    outside = np.zeros_like(keep)
    stack = [(y, x) for y in range(h) for x in (0, w - 1)] + [(y, x) for x in range(w) for y in (0, h - 1)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < h and 0 <= x < w and not keep[y, x] and not outside[y, x]:
            outside[y, x] = True
            stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    return ~outside


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    ids = set(sys.argv[sys.argv.index("--ids") + 1:]) if "--ids" in sys.argv else None
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json")))
    b = info["bbox"]
    proj = RP.Projector.build(b["south"], b["west"], b["north"], b["east"],
                              info["meters_per_tile"], info.get("rotation") or 0.0)
    gj_path = os.path.join(map_dir, f"{name}_buildings.geojson")
    gj = json.load(open(gj_path, encoding="utf-8"))
    real = {str(f["properties"]["osm_id"]): f for f in
            json.load(open(os.path.join(area, "buildings.geojson"), encoding="utf-8"))["features"]}
    corners = [proj.to_latlon(x, y) for x, y in ((0, 0), (proj.width, 0), (0, proj.height),
                                                (proj.width, proj.height))]
    es, ns = zip(*[RP.TO_BNG.transform(lon, lat) for lat, lon in corners])
    lidar = RP.Lidar(area, (int(min(es)) - 5, int(min(ns)) - 5, int(max(es)) + 5, int(max(ns)) + 5))

    done = 0
    for ft in gj["features"]:
        osm_id = str(ft["properties"].get("osm_id"))
        r = real.get(osm_id)
        if r is None or ft["geometry"]["type"] != "Polygon":
            continue
        rp = r["properties"]
        # Detached only: an attached building's LiDAR runs on into its
        # neighbours' roofs, and its outline would swallow them.
        landmark = ((rp.get("tags") or {}).get("amenity") == "place_of_worship" or
                    (rp.get("tags") or {}).get("building") in RP.CHURCHES or
                    (rp.get("listed_grade") in ("I", "II*") and rp.get("form") == "detached"))
        if ids is not None:
            landmark = osm_id in ids
        if not landmark:
            continue
        tile_poly = Polygon([proj.to_px(lat, lon) for lon, lat in ft["geometry"]["coordinates"][0]])
        real_poly = Polygon([proj.to_px(lat, lon) for lon, lat in r["geometry"]["coordinates"][0]])
        if ft["properties"].get("lidar_frame"):
            # Laid on the tiles by blocks.py, which knows exactly where the
            # real building is: read the LiDAR there, not by matching again.
            turn, dx, dy, fcx, fcy = ft["properties"]["lidar_frame"]
        else:
            turn, dx, dy = RP.best_match(real_poly, tile_poly)
            fcx, fcy = tile_poly.centroid.x, tile_poly.centroid.y
        x0 = int(math.floor(tile_poly.bounds[0])) - REACH
        y0 = int(math.floor(tile_poly.bounds[1])) - REACH
        w = int(math.ceil(tile_poly.bounds[2])) + REACH - x0
        h = int(math.ceil(tile_poly.bounds[3])) + REACH - y0
        xs, ys = np.meshgrid(np.arange(x0, x0 + w) + 0.5, np.arange(y0, y0 + h) + 0.5)
        a = math.radians(-turn)
        ux, uy = xs - fcx, ys - fcy
        rx = fcx + ux * math.cos(a) - uy * math.sin(a) - dx
        ry = fcy + ux * math.sin(a) + uy * math.cos(a) - dy
        ll = [proj.to_latlon(px, py) for px, py in zip(rx.ravel(), ry.ravel())]
        e, n = np.array([RP.TO_BNG.transform(lon, lat) for lat, lon in ll]).T
        dsm, dtm = lidar.sample(e, n)
        ndsm = (dsm - dtm).reshape(h, w)
        near = RP.raster(tile_poly.buffer(REACH, join_style=2), x0, y0, w, h)
        built = clean((ndsm > BUILT_M) & near)
        if built.sum() < MIN_TILES:
            print(f"{osm_id}: too little in the LiDAR, outline kept")
            continue
        # The outline as a few solid rectangles at least MIN_SIDE across: the
        # body, the tower, a real wing. Eaves, buttresses and a porch's edge
        # are left off; edges come out straight.
        rects = [r for r in RP.roof_rects(built.astype(int).tolist())
                 if min(r[2], r[3]) >= MIN_SIDE and r[2] * r[3] >= MIN_SIDE * MIN_SIDE]
        if not rects:
            print(f"{osm_id}: no solid part in the LiDAR, outline kept")
            continue
        poly = unary_union([box(x0 + rx, y0 + ry, x0 + rx + rw, y0 + ry + rh)
                            for rx, ry, rw, rh, _c in rects])
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        poly = Polygon(poly.exterior).simplify(0.01)
        change = poly.symmetric_difference(tile_poly).area / tile_poly.area
        if change > MAX_CHANGE:
            print(f"{osm_id}: LiDAR outline differs by {change:.0%}, outline kept")
            continue
        ring = [proj.to_latlon(x, y) for x, y in poly.exterior.coords]
        ft["geometry"] = mapping(Polygon([(lon, lat) for lat, lon in ring]))
        ft["properties"]["outline_from"] = "lidar"
        # The frame the LiDAR was read in. The outline now sits where the
        # LiDAR building is, so matching the (offset) OSM outline onto it
        # again would read the LiDAR a metre or two out: roof_plan.py reuses
        # this instead.
        ft["properties"]["lidar_frame"] = [turn, dx, dy, fcx, fcy]
        done += 1
        tb, pb = tile_poly.bounds, poly.bounds
        print(f"{osm_id} {rp.get('list_name') or ''}: OSM {tb[2] - tb[0]:.0f}x{tb[3] - tb[1]:.0f} "
              f"-> LiDAR {pb[2] - pb[0]:.0f}x{pb[3] - pb[1]:.0f} tiles, {len(poly.exterior.coords) - 1} corners, "
              f"area {tile_poly.area:.0f} -> {poly.area:.0f}")
    with open(gj_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)
    print(f"landmark outlines from LiDAR: {done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
