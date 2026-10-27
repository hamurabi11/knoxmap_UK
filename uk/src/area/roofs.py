"""Roof height and shape per building from Environment Agency LiDAR.

    python roofs.py <area data dir> [--fetch] [--box E0 N0 E1 N1]

Fetches (with --fetch) the 1 m composite last-return DSM and the DTM for the
area's buildings over the EA's WCS, then for every building in
buildings.geojson measures:
  height_m      ridge height above ground (95th percentile of DSM - DTM)
  eaves_m       eaves height (20th percentile, inside the footprint)
  storeys_lidar storeys estimated from the eaves height
  roof_shape    flat / gable / hipped / monopitch / complex
  ridge_deg     ridge direction, 0-180 degrees from east (pitched roofs)
  pitch_deg     typical roof pitch
Writes roofs.csv and a check image out/lidar/<area>_roofs.png.
Source: Environment Agency LiDAR Composite DSM/DTM 1 m, OGL v3.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

WCS = "https://environment.data.gov.uk/spatialdata/{ds}/wcs"
LAYERS = {
    "dsm": ("lidar-composite-digital-surface-model-last-return-dsm-1m",
            "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m"),
    "dtm": ("lidar-composite-digital-terrain-model-dtm-1m",
            "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"),
}
TILE_M = 1000            # fetch in 1 km tiles; the service limits request size
TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True)

# A roof pixel steeper than this is part of a slope. roof_plan.py reads the
# same DSM gradient (np.gradient on the same 1 m mosaic, same arctan/hypot
# formula) to plan the roof pieces actually built, and used to define this
# independently at 14 - the two had quietly drifted apart with no reason
# found in the log for them to differ. Settled on 14: roof_plan.py's value is
# the one that was tuned against real photos (LOG 2026-10-06, High Street and
# All Saints). roof_plan.py now imports this instead of redefining it.
FLAT_SLOPE_DEG = 14
PITCHED_SHARE = 0.45     # this share of sloped pixels makes the roof pitched
FACE_SHARE = 0.12        # a facing direction needs this share to count
HIP_END_SHARE = 0.20     # ...and each end of a hip needs this much
# Storeys from ridge height, calibrated on Odiham against EPC property type:
# bungalows' ridges median 4.8 m (90% under 5.3), houses' 7.4 m (90% over 6.3).
# Eaves heights were pulled down by extensions and conservatories.
STOREY_RIDGE_M = [(5.8, 1), (10.3, 2), (13.0, 3)]
MIN_PIXELS = 12


def fetch(kind: str, e0: int, n0: int, e1: int, n1: int, path: str) -> None:
    ds, cov = LAYERS[kind]
    url = (f"{WCS.format(ds=ds)}?service=WCS&version=2.0.1&request=GetCoverage"
           f"&coverageId={cov}&format=image/tiff&subset=E({e0},{e1})&subset=N({n0},{n1})")
    with urllib.request.urlopen(url, timeout=180) as r, open(path, "wb") as f:
        f.write(r.read())


def mosaic(kind: str, box, cache: str) -> np.ndarray:
    """Height grid for box (E0, N0, E1, N1), row 0 at the north edge."""
    e0, n0, e1, n1 = box
    out = np.full((n1 - n0, e1 - e0), np.nan, dtype="float32")
    for te in range(e0, e1, TILE_M):
        for tn in range(n0, n1, TILE_M):
            te1, tn1 = min(te + TILE_M, e1), min(tn + TILE_M, n1)
            path = os.path.join(cache, f"{kind}_{te}_{tn}_{te1}_{tn1}.tif")
            if not os.path.exists(path):
                fetch(kind, te, tn, te1, tn1, path)
            a = np.array(Image.open(path), dtype="float32")
            a[a < -1000] = np.nan
            r0 = n1 - tn1
            out[r0:r0 + a.shape[0], te - e0:te - e0 + a.shape[1]] = a
    return out


def footprint_mask(poly, box) -> tuple[np.ndarray, tuple[int, int]]:
    """Pixel mask of a polygon, shrunk 1 m so walls and eaves edges drop out."""
    e0, n0, e1, n1 = box
    inner = poly.buffer(-1.0)
    if inner.is_empty or inner.area < MIN_PIXELS:
        inner = poly
    minx, miny, maxx, maxy = inner.bounds
    c0, c1 = int(math.floor(minx - e0)), int(math.ceil(maxx - e0))
    r0, r1 = int(math.floor(n1 - maxy)), int(math.ceil(n1 - miny))
    w, h = max(1, c1 - c0), max(1, r1 - r0)
    img = Image.new("1", (w, h), 0)
    ring = [(x - e0 - c0, n1 - y - r0) for x, y in inner.exterior.coords] \
        if inner.geom_type == "Polygon" else \
        [(x - e0 - c0, n1 - y - r0) for x, y in max(inner.geoms, key=lambda g: g.area).exterior.coords]
    ImageDraw.Draw(img).polygon(ring, fill=1)
    return np.array(img, dtype=bool), (r0, c0)


def measure(poly, ndsm, slope, aspect, box) -> dict:
    mask, (r0, c0) = footprint_mask(poly, box)
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > ndsm.shape[0] or c0 + w > ndsm.shape[1]:
        return {}
    z = ndsm[r0:r0 + h, c0:c0 + w][mask]
    s = slope[r0:r0 + h, c0:c0 + w][mask]
    a = aspect[r0:r0 + h, c0:c0 + w][mask]
    ok = ~np.isnan(z)
    z, s, a = z[ok], s[ok], a[ok]
    if len(z) < MIN_PIXELS:
        return {"roof_shape": "too_small"}
    height = float(np.percentile(z, 95))
    eaves = float(np.percentile(z, 20))
    storeys = next((n for top, n in STOREY_RIDGE_M if height < top), 4)
    out = {"height_m": round(height, 1), "eaves_m": round(eaves, 1),
           "storeys_lidar": storeys}
    if height < 2.0:
        out["roof_shape"] = "no_building"   # demolished, or not built yet
        return out
    sloped = s > FLAT_SLOPE_DEG
    if sloped.mean() < PITCHED_SHARE:
        out["roof_shape"] = "flat"
        return out
    out["pitch_deg"] = round(float(np.median(s[sloped])), 0)
    # Which way the slopes face, in four quadrants (E, N, W, S).
    quad = ((a[sloped] + 45) % 360 // 90).astype(int)
    share = np.bincount(quad, minlength=4) / max(1, sloped.sum())
    faces = [q for q in range(4) if share[q] >= FACE_SHARE]
    if len(faces) == 1:
        out["roof_shape"] = "monopitch"
    elif len(faces) == 2 and abs(faces[0] - faces[1]) == 2:
        out["roof_shape"] = "gable"
    elif len(faces) >= 3:
        # A hip has all four faces, its two ends each a real share of the
        # roof; one weak extra face is a chimney, dormer or rear wing on a gable.
        main = 0 if share[0] + share[2] >= share[1] + share[3] else 1
        ends = (share[(main + 1) % 4], share[(main + 3) % 4])
        if len(faces) == 4 and min(ends) >= HIP_END_SHARE:
            out["roof_shape"] = "hipped"
        elif min(share[main], share[main + 2]) >= FACE_SHARE and max(ends) < HIP_END_SHARE:
            out["roof_shape"] = "gable"
        else:
            out["roof_shape"] = "complex"
    else:
        out["roof_shape"] = "complex"
    if out["roof_shape"] in ("gable", "hipped"):
        # The ridge runs across the strongest pair of opposite faces.
        pair = 0 if share[0] + share[2] >= share[1] + share[3] else 1
        out["ridge_deg"] = 90.0 if pair == 0 else 0.0   # E/W faces -> N-S ridge
    return out


def main() -> int:
    area = sys.argv[1]
    with open(os.path.join(area, "buildings.geojson"), encoding="utf-8") as f:
        feats = json.load(f)["features"]
    polys = [(ft["properties"]["osm_id"],
              transform(lambda x, y, z=None: TO_BNG.transform(x, y), shape(ft["geometry"])),
              ft["properties"]) for ft in feats]
    if "--box" in sys.argv:
        i = sys.argv.index("--box")
        box = tuple(int(v) for v in sys.argv[i + 1:i + 5])
        polys = [p for p in polys if box[0] <= p[1].centroid.x <= box[2]
                 and box[1] <= p[1].centroid.y <= box[3]]
    else:
        xs = [c for _, p, _ in polys for c in (p.bounds[0], p.bounds[2])]
        ys = [c for _, p, _ in polys for c in (p.bounds[1], p.bounds[3])]
        box = (int(min(xs)) - 5, int(min(ys)) - 5, int(max(xs)) + 5, int(max(ys)) + 5)
    cache = os.path.join(area, "lidar")
    os.makedirs(cache, exist_ok=True)
    dsm = mosaic("dsm", box, cache)
    dtm = mosaic("dtm", box, cache)
    ndsm = dsm - dtm
    gy, gx = np.gradient(dsm)            # rows run south, so north = -gy
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # Direction the slope faces (downhill), degrees anticlockwise from east.
    aspect = (np.degrees(np.arctan2(gy, -gx)) + 360) % 360

    rows = []
    for osm_id, poly, props in polys:
        r = {"osm_id": osm_id, "form": props.get("form"), "type": props.get("type")}
        r.update(measure(poly, ndsm, slope, aspect, box))
        rows.append(r)
    cols = ["osm_id", "form", "type", "height_m", "eaves_m", "storeys_lidar",
            "roof_shape", "pitch_deg", "ridge_deg"]
    with open(os.path.join(area, "roofs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Check image: shaded heights, footprints coloured by roof shape.
    shade = np.clip((np.nan_to_num(ndsm) / 12.0) * 255, 0, 255).astype("uint8")
    hill = np.clip(128 + 4 * (gx - gy), 0, 255).astype("uint8")
    img = Image.merge("RGB", [Image.fromarray(np.maximum(shade, hill // 2))] * 3).convert("RGB")
    colours = {"flat": (0, 160, 255), "gable": (255, 140, 0), "hipped": (220, 0, 220),
               "monopitch": (0, 200, 0), "complex": (255, 230, 0)}
    d = ImageDraw.Draw(img)
    e0, n0, e1, n1 = box
    for (osm_id, poly, _), r in zip(polys, rows):
        col = colours.get(r.get("roof_shape"))
        if not col:
            continue
        geom = poly if poly.geom_type == "Polygon" else max(poly.geoms, key=lambda g: g.area)
        pts = [(x - e0, n1 - y) for x, y in geom.exterior.coords]
        d.line(pts, fill=col, width=2)
        if r.get("ridge_deg") is not None:
            c = poly.centroid
            ang = math.radians(r["ridge_deg"])
            dx, dy = 4 * math.cos(ang), 4 * math.sin(ang)
            d.line([(c.x - e0 - dx, n1 - c.y + dy), (c.x - e0 + dx, n1 - c.y - dy)], fill=col, width=2)
    for i, (k, col) in enumerate(colours.items()):
        d.rectangle((6, 6 + 16 * i, 18, 18 + 16 * i), fill=col)
        d.text((24, 6 + 16 * i), k, fill=(255, 255, 255))
    out_dir = os.path.join(os.path.dirname(os.path.dirname(area)), "out", "lidar")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"{os.path.basename(area)}_roofs.png")
    img.save(img_path)

    from collections import Counter
    print(f"buildings measured: {len(rows)}  box {box}")
    print("roof shapes:", dict(Counter(r.get("roof_shape") for r in rows)))
    print("wrote", img_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
