"""Demo: roofs per footprint rectangle, KnoxMap today vs LiDAR.

    python demo_roof_parts.py <area data dir> E0 N0 E1 N1 <out.png>

Each outline is rasterised at 1 m (a tile, at 1 m a tile and no rotation),
split into rectangles with KnoxMap's own roof_rects, and roofed twice:
  left  - KnoxMap's _roof_pieces: ridge along the long side, hips at random
  right - from the LiDAR inside each rectangle: flat or pitched, which way
          the ridge runs, gable or hip
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
KNOXMAP = ukpaths.KNOX
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
sys.path.insert(0, KNOXMAP)
from knoxbuild.footprint import place  # noqa: E402
from knoxbuild.layout import roof_rects  # noqa: E402
from knoxbuild.tbx import _roof_pieces  # noqa: E402
from area.roofs import mosaic  # noqa: E402

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True)
SCALE = int(os.environ.get("DEMO_SCALE", "5"))   # pixels per metre in the picture
FLAT_SLOPE_DEG = 12
PITCHED_SHARE = 0.45
HIP_END_SHARE = 0.20

COL = {"flat": (70, 160, 255), "WE": (255, 150, 0), "NS": (255, 70, 70),
       "hip": (220, 0, 220)}


def lidar_roof(slope, aspect, x0, y0, w, h):
    """Roof for one rectangle (grid coords, y down) from the pixels inside it,
    one metre in from its edges. Returns ('flat'|'WE'|'NS', hip?) or None."""
    pad = 1 if min(w, h) > 4 else 0
    s = slope[y0 + pad:y0 + h - pad, x0 + pad:x0 + w - pad].ravel()
    a = aspect[y0 + pad:y0 + h - pad, x0 + pad:x0 + w - pad].ravel()
    ok = ~np.isnan(s)
    s, a = s[ok], a[ok]
    if len(s) < 6:
        return None
    sloped = s > FLAT_SLOPE_DEG
    if sloped.mean() < PITCHED_SHARE:
        return "flat", False
    quad = ((a[sloped] + 45) % 360 // 90).astype(int)          # 0 E, 1 N, 2 W, 3 S
    share = np.bincount(quad, minlength=4) / max(1, sloped.sum())
    ew, ns = share[0] + share[2], share[1] + share[3]
    ridge = "NS" if ew >= ns else "WE"      # east/west-facing slopes meet on a N-S ridge
    ends = (share[1], share[3]) if ridge == "NS" else (share[0], share[2])
    return ridge, min(ends) >= HIP_END_SHARE


def draw_roof(d, kind, hip, rx, ry, rw, rh, ox, oy):
    """One roof rectangle: outline, ridge line, hip lines."""
    x0, y0 = ox + rx * SCALE, oy + ry * SCALE
    x1, y1 = x0 + rw * SCALE, y0 + rh * SCALE
    col = COL["hip"] if hip else COL[kind]
    if kind == "flat":
        d.rectangle((x0, y0, x1, y1), fill=(40, 70, 110), outline=col, width=2)
        return
    d.rectangle((x0, y0, x1, y1), outline=col, width=2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if kind == "WE":
        inset = (y1 - y0) / 2 if hip else 0
        d.line((x0 + inset, cy, x1 - inset, cy), fill=col, width=3)
        if hip:
            for (ax, ay), (bx, by) in (((x0, y0), (x0 + inset, cy)), ((x0, y1), (x0 + inset, cy)),
                                       ((x1, y0), (x1 - inset, cy)), ((x1, y1), (x1 - inset, cy))):
                d.line((ax, ay, bx, by), fill=col, width=2)
    else:
        inset = (x1 - x0) / 2 if hip else 0
        d.line((cx, y0 + inset, cx, y1 - inset), fill=col, width=3)
        if hip:
            for (ax, ay), (bx, by) in (((x0, y0), (cx, y0 + inset)), ((x1, y0), (cx, y0 + inset)),
                                       ((x0, y1), (cx, y1 - inset)), ((x1, y1), (cx, y1 - inset))):
                d.line((ax, ay, bx, by), fill=col, width=2)


def main() -> int:
    area = sys.argv[1]
    E0, N0, E1, N1 = map(int, sys.argv[2:6])
    out = sys.argv[6]
    cache = os.path.join(area, "lidar")
    # Whole 1 km tiles around the window, cut down to it.
    tb = (E0 // 1000 * 1000, N0 // 1000 * 1000, -(-E1 // 1000) * 1000, -(-N1 // 1000) * 1000)
    dsm_all = mosaic("dsm", tb, cache)
    r0, c0 = tb[3] - N1, E0 - tb[0]
    dsm = dsm_all[r0:r0 + (N1 - N0), c0:c0 + (E1 - E0)]
    gy, gx = np.gradient(dsm)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(gy, -gx)) + 360) % 360

    with open(os.path.join(area, "buildings.geojson"), encoding="utf-8") as f:
        feats = json.load(f)["features"]

    W, H = (E1 - E0) * SCALE, (N1 - N0) * SCALE
    gap, head = 20, 30
    # Background: hillshade of the surface, the same under both panels.
    hs = np.clip(140 + 6 * (gx - gy), 0, 255)
    hs = np.nan_to_num(hs, nan=0).astype("uint8")
    bg = Image.fromarray(hs).resize((W, H), Image.NEAREST).convert("RGB")
    img = Image.new("RGB", (2 * W + gap, H + head + 70), (25, 25, 25))
    img.paste(bg, (0, head))
    img.paste(bg, (W + gap, head))
    d = ImageDraw.Draw(img)
    d.text((6, 8), "KnoxMap today: ridge along each rectangle's long side, hips at random",
           fill=(255, 255, 255))
    d.text((W + gap + 6, 8), "Per roof part from LiDAR: flat/pitched, ridge direction, gable/hip",
           fill=(255, 255, 255))

    # Fit every outline to the tile grid exactly as a build does (footprint.place,
    # biggest first, neighbours not overlapping), then roof what it fitted.
    # Tile space here: x = east of E0, y = south of N1, 1 m a tile, no rotation.
    occupied = np.zeros((N1 - N0, E1 - E0), dtype=bool)
    lots = np.zeros_like(occupied)
    todo = []
    for ft in feats:
        props = ft["properties"]
        if props.get("form") == "outbuilding":
            continue
        poly = transform(lambda x, y, z=None: TO_BNG.transform(x, y), shape(ft["geometry"]))
        if not (E0 <= poly.centroid.x <= E1 and N0 <= poly.centroid.y <= N1):
            continue
        ring = poly.exterior.coords if poly.geom_type == "Polygon" else \
            max(poly.geoms, key=lambda g: g.area).exterior.coords
        todo.append((-poly.area, [(x - E0, N1 - y) for x, y in ring]))
    todo.sort(key=lambda t: t[0])

    counts = {"parts": 0, "agree": 0, "buildings": 0}
    for _neg_area, px in todo:
        fp, _why = place(px, occupied, min_side=3, max_side=200, snap_degrees=15, lots=lots)
        if fp is None:
            continue
        rects = roof_rects(fp.mask.astype(int).tolist())
        if not rects:
            continue
        counts["buildings"] += 1
        gx0, gy0 = fp.x0 + E0, N1 - fp.y0          # back to metres for the LiDAR window
        # Pixel offset of this building's grid in each panel.
        ox, oy = fp.x0 * SCALE, head + fp.y0 * SCALE
        today = _roof_pieces(rects, True, True)
        for roof_type, _depth, _cap, (rx, ry, rw, rh) in today:
            kind = "flat" if roof_type == "FlatTop" else ("WE" if roof_type.endswith("WE") else "NS")
            if roof_type == "Peak30Quad":
                kind = "WE" if rw >= rh else "NS"
            draw_roof(d, kind, roof_type == "Peak30Quad", rx, ry, rw, rh, ox, oy)
        for rx, ry, rw, rh, _caps in rects:
            # This rectangle's place in the LiDAR window.
            wx, wy = gx0 - E0 + rx, N1 - gy0 + ry
            if wx < 0 or wy < 0 or wx + rw > dsm.shape[1] or wy + rh > dsm.shape[0]:
                continue
            got = lidar_roof(slope, aspect, wx, wy, rw, rh)
            if not got:
                continue
            kind, hip = got
            draw_roof(d, kind, hip, rx, ry, rw, rh, ox + W + gap, oy)
            counts["parts"] += 1
            long_kind = "WE" if rw >= rh else "NS"
            counts["agree"] += kind == long_kind

    y = H + head + 10
    for i, (k, label) in enumerate((("WE", "ridge east-west"), ("NS", "ridge north-south"),
                                    ("hip", "hipped"), ("flat", "flat"))):
        d.rectangle((10 + 170 * i, y, 24 + 170 * i, y + 14), fill=COL[k])
        d.text((30 + 170 * i, y + 1), label, fill=(255, 255, 255))
    d.text((10, y + 26), f"{counts['buildings']} buildings, {counts['parts']} roof parts. "
           f"LiDAR ridge runs along the long side in {counts['agree']}/{counts['parts']} parts; "
           f"the rest are pitched across, or flat.", fill=(220, 220, 220))
    d.text((10, y + 42), "North is up. Background: LiDAR surface (EA Composite DSM 1 m, OGL).",
           fill=(160, 160, 160))
    img.save(out)
    print(out, img.size, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
