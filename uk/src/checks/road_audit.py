"""How many building tiles stand on roads in a built map.

    python road_audit.py <knoxmap map dir> [--all]

Run after Build (reads <map>_footprints.npz). For every road the renderer
drew (the straightened roads, at the width it drew them), the tiles of its
carriageway that a building covers. Named streets and other through roads
count; service lanes are listed apart (a drive into a yard often is under a
building's corner in OSM). --all lists every road, not just the totals.
Also, from the drawn ground itself (<map>.bmp): carriageway-coloured tiles
along a road's line (within a tile of its width; a tarmac yard is not a
road) under buildings, and touching a building's wall - what shows as a road
running into or under a house.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator import osm  # noqa: E402
from generator.octilinear import ROAD_CATEGORIES, straighten_roads  # noqa: E402
from generator.renderer import Projector, _is_polygon, _way_width_m, classify  # noqa: E402
from knoxbuild.population import load_footprints  # noqa: E402


def main() -> int:
    map_dir = sys.argv[1]
    every = "--all" in sys.argv
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json")))
    b = info["bbox"]
    proj = Projector.build(b["south"], b["west"], b["north"], b["east"], info["meters_per_tile"],
                           info.get("rotation") or 0.0)
    W, H = proj.width, proj.height
    built = np.zeros((H, W), np.int32)
    for k, (x0, y0, m, _lv, _kd) in enumerate(load_footprints(os.path.join(map_dir, f"{name}_footprints.npz")), 1):
        built[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m] = k
    feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"]))
    if info.get("straight_roads"):
        straighten_roads(feats, proj, classify, _is_polygon, map_dir=map_dir)
    totals = {"street": 0, "service": 0}
    from PIL import Image as _I, ImageDraw as _D
    lines_img = _I.new("1", (W, H), 0)
    lines_draw = _D.Draw(lines_img)
    hit_buildings = {"street": set(), "service": set()}
    for f in feats:
        if f.kind != "way" or _is_polygon(f) or len(f.geometry) < 2:
            continue
        cat = classify(f.tags)
        if cat not in ROAD_CATEGORIES:
            continue
        line = shapely.LineString([proj.to_px(la, lo) for la, lo in f.geometry])
        band = line.buffer(_way_width_m(f, cat) / 2 / info["meters_per_tile"], cap_style=2)
        near = line.buffer(_way_width_m(f, cat) / 2 / info["meters_per_tile"] + 1.0)
        for part in getattr(near, "geoms", [near]):
            if part.geom_type == "Polygon":
                lines_draw.polygon(list(part.exterior.coords), fill=1)
        if band.is_empty:          # a link the straightening shrank to its junction
            continue
        x0, y0, x1, y1 = (int(v) for v in band.bounds)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1 + 1), min(H, y1 + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        under = built[y0:y1, x0:x1][shapely.contains_xy(band, xs, ys)]
        n = int((under > 0).sum())
        if not n:
            continue
        kind = "service" if cat == "road_service" else "street"
        totals[kind] += n
        hit_buildings[kind] |= set(under[under > 0].tolist())
        if every:
            print(f"  {f.tags.get('name') or '(' + f.tags.get('highway', '') + ')'}: {n} tiles")
    from PIL import Image
    from generator import pz_colors as C
    g = np.array(Image.open(os.path.join(map_dir, f"{name}.bmp")).convert("RGB"))[:H, :W]
    road = np.zeros((H, W), bool)
    for c in (C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE):
        road |= np.all(g == c, axis=2)
    on = built > 0
    wall = np.zeros_like(on)
    wall[1:, :] |= on[:-1, :]
    wall[:-1, :] |= on[1:, :]
    wall[:, 1:] |= on[:, :-1]
    wall[:, :-1] |= on[:, 1:]
    # Only road surface along a road's line: a tarmac yard or car park drawn
    # in the same colour is not a road through a house.
    road &= np.array(lines_img, dtype=bool)
    print(f"{name}: drawn road surface under buildings {int((road & on).sum())} tiles, "
          f"against a wall {int((road & wall & ~on).sum())} tiles")
    print(f"{name}: building tiles on carriageway - streets {totals['street']} "
          f"({len(hit_buildings['street'])} buildings), service lanes {totals['service']} "
          f"({len(hit_buildings['service'])} buildings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
