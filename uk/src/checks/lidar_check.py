"""Check one building's roof plan against its LiDAR, in the map's tile frame.

    python lidar_check.py <knoxmap map dir> <area data dir> <osm_id> <out.png>

Three panels, north (game) up, one cell per tile:
  1. height above ground (LiDAR DSM - DTM), shaded 0..max, with the planned
     parts' outlines
  2. which way each tile's roof slopes: red east/west (ridge runs N-S),
     orange north/south (ridge runs E-W), blue flat
  3. the planned parts, labelled form and storeys
Also prints the height profiles across the building, to read ridges and
valleys off directly.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
from regional import roof_plan as RP  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

S = 16


def main() -> int:
    map_dir, area, osm_id, out = sys.argv[1:5]
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json")))
    b = info["bbox"]
    proj = RP.Projector.build(b["south"], b["west"], b["north"], b["east"],
                              info["meters_per_tile"], info.get("rotation") or 0.0)
    built = {str(f["properties"].get("osm_id")): f for f in
             json.load(open(os.path.join(map_dir, f"{name}_buildings.geojson"), encoding="utf-8"))["features"]}
    real = {str(f["properties"]["osm_id"]): f for f in
            json.load(open(os.path.join(area, "buildings.geojson"), encoding="utf-8"))["features"]}
    ft = built[osm_id]
    tile_poly = Polygon([proj.to_px(lat, lon) for lon, lat in ft["geometry"]["coordinates"][0]])
    real_poly = Polygon([proj.to_px(lat, lon) for lon, lat in real[osm_id]["geometry"]["coordinates"][0]])
    frame = ft["properties"].get("lidar_frame")
    if frame:
        turn, dx, dy, cx, cy = frame          # the outline was drawn from the LiDAR in this frame
    else:
        turn, dx, dy = RP.best_match(real_poly, tile_poly)
        cx, cy = tile_poly.centroid.x, tile_poly.centroid.y
    x0, y0 = int(math.floor(tile_poly.bounds[0])) - 2, int(math.floor(tile_poly.bounds[1])) - 2
    w = int(math.ceil(tile_poly.bounds[2])) - x0 + 2
    h = int(math.ceil(tile_poly.bounds[3])) - y0 + 2
    mask = RP.raster(tile_poly, x0, y0, w, h)
    xs, ys = np.meshgrid(np.arange(x0, x0 + w) + 0.5, np.arange(y0, y0 + h) + 0.5)
    a = math.radians(-turn)
    ux, uy = xs - cx, ys - cy
    rx = cx + ux * math.cos(a) - uy * math.sin(a) - dx
    ry = cy + ux * math.sin(a) + uy * math.cos(a) - dy
    ll = [proj.to_latlon(px, py) for px, py in zip(rx.ravel(), ry.ravel())]
    e, n = np.array([RP.TO_BNG.transform(lon, lat) for lat, lon in ll]).T
    lid = RP.Lidar(area, (int(e.min()) - 3, int(n.min()) - 3, int(e.max()) + 3, int(n.max()) + 3))
    dsm, dtm = lid.sample(e, n)
    dsm, ndsm = dsm.reshape(h, w), (dsm - dtm).reshape(h, w)
    gy, gx = np.gradient(dsm)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    cls = np.where(np.abs(gx) >= np.abs(gy), 1, 2)
    cls[slope <= RP.FLAT_SLOPE_DEG] = 0

    regional = json.load(open(os.path.join(map_dir, "regional.json"), encoding="utf-8"))
    parts = regional["buildings"].get(osm_id, {}).get("roof", [])
    print(f"turn {turn:+.0f} deg; {w}x{h} tiles; ndsm max {np.nanmax(ndsm[mask]):.1f} m; "
          f"p50 {np.nanpercentile(ndsm[mask], 50):.1f}")
    print("height across the building (mean along N-S, west to east), m:")
    with np.errstate(all="ignore"):
        print("  ", " ".join(f"{v:4.1f}" for v in np.nanmean(np.where(mask, ndsm, np.nan), axis=0)))
        print("height along the building (mean along E-W, north to south), m:")
        print("  ", " ".join(f"{v:4.1f}" for v in np.nanmean(np.where(mask, ndsm, np.nan), axis=1)))

    top = float(np.nanmax(ndsm[mask]))
    img = Image.new("RGB", (3 * w * S + 40, h * S + 30), (30, 30, 30))
    d = ImageDraw.Draw(img)
    for yy in range(h):
        for xx in range(w):
            X, Y = xx * S, 30 + yy * S
            v = ndsm[yy, xx]
            g = int(np.clip((0 if np.isnan(v) else v) / max(1, top) * 255, 0, 255))
            d.rectangle((X, Y, X + S - 1, Y + S - 1), fill=(g, g, g))
            col = {0: (70, 160, 255), 1: (235, 70, 70), 2: (255, 180, 40)}[int(cls[yy, xx])]
            if not mask[yy, xx]:
                col = tuple(c // 4 for c in col)
            X2 = X + w * S + 20
            d.rectangle((X2, Y, X2 + S - 1, Y + S - 1), fill=col)
            if mask[yy, xx]:
                d.rectangle((X + 2 * (w * S + 20), Y, X + 2 * (w * S + 20) + S - 1, Y + S - 1), fill=(55, 55, 55))
    for px, py, pw, ph, form, st in parts:
        bx, by = px - x0, py - y0
        for k in (0, 2):
            ox = k * (w * S + 20)
            d.rectangle((ox + bx * S, 30 + by * S, ox + (bx + pw) * S - 1, 30 + (by + ph) * S - 1),
                        outline=(0, 255, 0), width=2)
        d.text((2 * (w * S + 20) + bx * S + 3, 30 + by * S + 3), f"{form} {st}", fill=(255, 255, 255))
    d.text((4, 6), f"1. height above ground, 0-{top:.0f} m", fill=(255, 255, 255))
    d.text((w * S + 24, 6), "2. slopes: red E/W, orange N/S, blue flat", fill=(255, 255, 255))
    d.text((2 * (w * S + 20) + 4, 6), "3. planned parts (form, storeys)", fill=(255, 255, 255))
    img.save(out)
    print(out, img.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
