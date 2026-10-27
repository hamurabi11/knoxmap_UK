"""Demo: roof parts decided by the LiDAR, for a few buildings.

    python demo_roof_examples.py <area data dir> <out.png> <osm_id> [<osm_id> ...]

One row per building, three panels:
  1. LiDAR roof faces, coloured by the way each face slopes
  2. KnoxMap today: the footprint covered by the largest rectangles
     (roof_rects), each roofed by _roof_pieces (ridge along the long side,
     hips at random on simple houses)
  3. Parts from the LiDAR: the footprint split where the ridge direction
     changes, each part roofed the way its faces slope; a simple block whose
     four faces all show is one hipped roof
Footprints are fitted with KnoxMap's own footprint.place at 1 m a tile.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from knoxbuild.footprint import place  # noqa: E402
from knoxbuild.layout import roof_rects  # noqa: E402
from knoxbuild.tbx import _roof_pieces  # noqa: E402
from area.roofs import mosaic  # noqa: E402

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True)
S = 12                    # pixels per metre
PAD = 4                   # metres of surroundings around each building
FLAT_SLOPE_DEG = 12
HIP_SHARE = 0.18

# Faces by the way they slope (0 E, 1 N, 2 W, 3 S). East/west faces meet on a
# north-south ridge (reds), north/south faces on an east-west ridge (oranges).
FACE_COL = {0: (235, 70, 70), 2: (150, 35, 35), 1: (255, 180, 40), 3: (190, 115, 0)}
FLAT_COL = (70, 160, 255)
ROOF_COL = {"WE": (255, 150, 0), "NS": (255, 70, 70), "flat": (70, 160, 255),
            "hip": (220, 0, 220)}


def faces(slope, aspect, mask):
    """Per tile: -1 flat, else facing 0-3. The outermost ring of the footprint
    reads the wall drop, not the roof, so it is filled from inside."""
    q = (((aspect + 45) % 360) // 90).astype(int)
    q[slope <= FLAT_SLOPE_DEG] = -1
    q[np.isnan(slope)] = -1
    inner = mask.copy()
    inner[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] \
        & mask[1:-1, :-2] & mask[1:-1, 2:]
    inner[0, :] = inner[-1, :] = False
    inner[:, 0] = inner[:, -1] = False
    out = np.full(mask.shape, -9, dtype=int)          # -9 = not yet known
    out[inner] = q[inner]
    return fill_unknown(out, mask)


def fill_unknown(cls, mask):
    cls = cls.copy()
    for _ in range(20):
        todo = mask & (cls == -9)
        if not todo.any():
            break
        for y, x in zip(*np.nonzero(todo)):
            nb = cls[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            nb = nb[nb != -9]
            if len(nb):
                vals, counts = np.unique(nb, return_counts=True)
                cls[y, x] = vals[np.argmax(counts)]
    cls[mask & (cls == -9)] = -1
    return cls


def ridge_class(face):
    """-1 flat, 1 north-south ridge (E/W faces), 2 east-west ridge (N/S faces)."""
    out = np.full(face.shape, -1, dtype=int)
    out[(face == 0) | (face == 2)] = 1
    out[(face == 1) | (face == 3)] = 2
    return out


def smooth(cls, mask, size=5, passes=2):
    """Majority of each tile's neighbourhood, inside the footprint only:
    removes the ridge line itself, chimneys and dormers as separate parts."""
    r = size // 2
    for _ in range(passes):
        new = cls.copy()
        for y, x in zip(*np.nonzero(mask)):
            nb = cls[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            m = mask[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            vals, counts = np.unique(nb[m], return_counts=True)
            new[y, x] = vals[np.argmax(counts)]
        cls = new
    return cls


MIN_PART_TILES = 16       # a roof part smaller than this joins its neighbour
END_SHARE = 0.35          # an opposite-facing piece this small at an end is a hip


def components(cls, mask):
    """Connected regions of equal class: (label grid, {label: (class, area)})."""
    lab = np.full(cls.shape, -1, dtype=int)
    info = {}
    n = 0
    for sy, sx in zip(*np.nonzero(mask)):
        if lab[sy, sx] >= 0:
            continue
        c, stack, area = cls[sy, sx], [(sy, sx)], 0
        lab[sy, sx] = n
        while stack:
            y, x = stack.pop()
            area += 1
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < cls.shape[0] and 0 <= nx < cls.shape[1] and mask[ny, nx] \
                        and lab[ny, nx] < 0 and cls[ny, nx] == c:
                    lab[ny, nx] = n
                    stack.append((ny, nx))
        info[n] = [c, area]
        n += 1
    return lab, info


def neighbours(lab, k):
    """Labels touching region k, with the length of border shared."""
    m = lab == k
    out = {}
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(np.roll(m, dy, 0), dx, 1)
        for v in lab[shifted & ~m]:
            if v >= 0:
                out[v] = out.get(v, 0) + 1
    return out


def lidar_parts(face, mask):
    """[(x, y, w, h, kind, hip)] covering the footprint."""
    h, w = mask.shape
    sloped = face[mask] >= 0
    # A simple block with all four faces showing is one hipped roof.
    if mask.mean() >= 0.9 and sloped.mean() >= 0.45:
        share = np.bincount(face[mask][sloped], minlength=4) / max(1, sloped.sum())
        ew, ns = share[0] + share[2], share[1] + share[3]
        if min(share) >= 0.12:
            return [(0, 0, w, h, "NS" if ew >= ns else "WE", True)]
    cls = smooth(ridge_class(face), mask, size=5, passes=3)
    lab, info = components(cls, mask)
    hipped = set()
    # Fold small regions into the neighbour they share most border with. A
    # small pitched piece facing the other way at the end of a big pitched
    # region is that region's hip end, not a roof of its own.
    while True:
        small = [k for k, (c, a) in info.items() if a < MIN_PART_TILES]
        ends = []
        for k, (c, a) in info.items():
            if c < 0 or k in small:
                continue
            nb = neighbours(lab, k)
            if not nb:
                continue
            big = max(nb, key=nb.get)
            bc, ba = info[big]
            if bc >= 0 and bc != c and a < END_SHARE * ba:
                ends.append(k)
        todo = small or ends
        if not todo:
            break
        k = min(todo, key=lambda j: info[j][1])
        nb = neighbours(lab, k)
        if not nb:
            break
        target = max(nb, key=nb.get)
        if k in ends:
            hipped.add(target)
        lab[lab == k] = target
        info[target][1] += info.pop(k)[1]
    parts = []
    kinds = {2: "WE", 1: "NS", -1: "flat"}
    for k, (c, _a) in info.items():
        rects = roof_rects((lab == k).astype(int).tolist())
        rects.sort(key=lambda r: -r[2] * r[3])
        for i, (rx, ry, rw, rh, _caps) in enumerate(rects):
            # Slivers left at a region's edge roof flat, as KnoxMap would.
            kind = kinds[c] if min(rw, rh) >= 3 else "flat"
            parts.append((rx, ry, rw, rh, kind, k in hipped and i == 0 and kind != "flat"))
    return parts


def draw_roof(d, kind, hip, rx, ry, rw, rh, ox, oy):
    x0, y0 = ox + rx * S, oy + ry * S
    x1, y1 = x0 + rw * S, y0 + rh * S
    col = ROOF_COL["hip"] if hip else ROOF_COL[kind]
    if kind == "flat":
        d.rectangle((x0, y0, x1, y1), fill=(40, 70, 110), outline=col, width=2)
        return
    d.rectangle((x0, y0, x1, y1), outline=col, width=2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if kind == "WE":
        inset = min((y1 - y0) / 2, (x1 - x0) / 2) if hip else 0
        d.line((x0 + inset, cy, x1 - inset, cy), fill=col, width=4)
        if hip:
            for ax, ay, bx in ((x0, y0, x0 + inset), (x0, y1, x0 + inset),
                               (x1, y0, x1 - inset), (x1, y1, x1 - inset)):
                d.line((ax, ay, bx, cy), fill=col, width=2)
    else:
        inset = min((x1 - x0) / 2, (y1 - y0) / 2) if hip else 0
        d.line((cx, y0 + inset, cx, y1 - inset), fill=col, width=4)
        if hip:
            for ax, ay, by in ((x0, y0, y0 + inset), (x1, y0, y0 + inset),
                               (x0, y1, y1 - inset), (x1, y1, y1 - inset)):
                d.line((ax, ay, cx, by), fill=col, width=2)


def main() -> int:
    area, out = sys.argv[1], sys.argv[2]
    ids = sys.argv[3:]
    with open(os.path.join(area, "buildings.geojson"), encoding="utf-8") as f:
        feats = {str(ft["properties"]["osm_id"]): ft for ft in json.load(f)["features"]}
    cache = os.path.join(area, "lidar")
    rows = []
    for osm_id in ids:
        ft = feats[osm_id]
        p = ft["properties"]
        poly = transform(lambda x, y, z=None: TO_BNG.transform(x, y), shape(ft["geometry"]))
        minx, miny, maxx, maxy = poly.bounds
        E0, N0 = int(minx) - PAD, int(miny) - PAD
        E1, N1 = int(maxx) + PAD + 1, int(maxy) + PAD + 1
        tb = (E0 // 1000 * 1000, N0 // 1000 * 1000, -(-E1 // 1000) * 1000, -(-N1 // 1000) * 1000)
        dsm = mosaic("dsm", tb, cache)
        dtm = mosaic("dtm", tb, cache)
        r0, c0 = tb[3] - N1, E0 - tb[0]
        dsm = dsm[r0:r0 + N1 - N0, c0:c0 + E1 - E0]
        dtm = dtm[r0:r0 + N1 - N0, c0:c0 + E1 - E0]
        gy, gx = np.gradient(dsm)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        aspect = (np.degrees(np.arctan2(gy, -gx)) + 360) % 360
        ring = poly.exterior.coords if poly.geom_type == "Polygon" else \
            max(poly.geoms, key=lambda g: g.area).exterior.coords
        px = [(x - E0, N1 - y) for x, y in ring]
        occupied = np.zeros(dsm.shape, dtype=bool)
        fp, why = place(px, occupied, min_side=3, max_side=200, snap_degrees=15)
        if fp is None:
            print(osm_id, "not placed:", why)
            continue
        mask = fp.mask.astype(bool)
        win = (slice(fp.y0, fp.y0 + fp.height), slice(fp.x0, fp.x0 + fp.width))
        face = faces(slope[win], aspect[win], mask)
        today = _roof_pieces(roof_rects(mask.astype(int).tolist()), True, True)
        mine = lidar_parts(face, mask)
        height = float(np.nanpercentile((dsm - dtm)[win][mask], 95))
        rows.append((osm_id, p, dsm, gx, gy, fp, mask, face, today, mine, height))

    head, label_h, gap = 26, 44, 16
    pw = max(r[2].shape[1] for r in rows) * S
    rh = [r[2].shape[0] * S + label_h for r in rows]
    img = Image.new("RGB", (3 * pw + 4 * gap, head + sum(rh) + gap * len(rows) + 60), (25, 25, 25))
    d = ImageDraw.Draw(img)
    for i, t in enumerate(("1. LiDAR roof faces (colour = way the face slopes)",
                           "2. KnoxMap today", "3. Roof parts from the LiDAR")):
        d.text((gap + i * (pw + gap), 8), t, fill=(255, 255, 255))
    y = head
    for osm_id, p, dsm, gx, gy, fp, mask, face, today, mine, height in rows:
        hgt, wid = dsm.shape
        hs = np.nan_to_num(np.clip(130 + 6 * (gx - gy), 0, 255)).astype("uint8")
        bg = Image.fromarray(hs).resize((wid * S, hgt * S), Image.NEAREST).convert("RGB")
        # Panel 1: faces coloured over the building, hillshade around it.
        p1 = bg.copy()
        d1 = ImageDraw.Draw(p1)
        for (ty, tx) in zip(*np.nonzero(mask)):
            f = face[ty, tx]
            col = FLAT_COL if f < 0 else FACE_COL[f]
            X, Y = (fp.x0 + tx) * S, (fp.y0 + ty) * S
            d1.rectangle((X, Y, X + S - 1, Y + S - 1), fill=col)
        for k, panel in enumerate((p1, bg.copy(), bg.copy())):
            img.paste(panel, (gap + k * (pw + gap), y))
        for k, parts in ((1, [(rx, ry, rw, rh_, t) for t, _d, _c, (rx, ry, rw, rh_) in today]),
                         (2, mine)):
            ox, oy = gap + k * (pw + gap) + fp.x0 * S, y + fp.y0 * S
            for part in parts:
                if k == 1:
                    rx, ry, rw, rh_, rt = part
                    kind = "flat" if rt == "FlatTop" else ("WE" if rt.endswith("WE") else "NS")
                    hip = rt == "Peak30Quad"
                    if hip:
                        kind = "WE" if rw >= rh_ else "NS"
                else:
                    rx, ry, rw, rh_, kind, hip = part
                draw_roof(d, kind, hip, rx, ry, rw, rh_, ox, oy)
        n_flat_today = sum(1 for t in today if t[0] == "FlatTop")
        name = p.get("list_name") or (p.get("tags") or {}).get("name") or ""
        d.text((gap, y + hgt * S + 6),
               f"OSM {osm_id}  {p.get('type')}  ({p.get('form')})  {name}  "
               f"ridge {height:.1f} m", fill=(255, 255, 255))
        d.text((gap, y + hgt * S + 22),
               f"today: {len(today)} roof pieces, {n_flat_today} flat   |   "
               f"LiDAR: {len(mine)} parts: " + ", ".join(
                   f"{'hip ' if hp else ''}{k}" for *_r, k, hp in mine[:8]),
               fill=(200, 200, 200))
        y += hgt * S + label_h + gap
    legend = [("ridge east-west", ROOF_COL["WE"]), ("ridge north-south", ROOF_COL["NS"]),
              ("hipped", ROOF_COL["hip"]), ("flat", ROOF_COL["flat"]),
              ("faces E / W", FACE_COL[0]), ("faces N / S", FACE_COL[1])]
    for i, (t, col) in enumerate(legend):
        d.rectangle((gap + 150 * i, y + 4, gap + 14 + 150 * i, y + 18), fill=col)
        d.text((gap + 20 + 150 * i, y + 5), t, fill=(255, 255, 255))
    d.text((gap, y + 28), "North is up, 1 m a tile. LiDAR: EA Composite Last Return DSM / DTM 1 m (OGL).",
           fill=(160, 160, 160))
    img.save(out)
    print(out, img.size)
    for osm_id, p, *_rest, today, mine, height in rows:
        print(osm_id, p.get("type"), "today", len(today), "pieces;", "lidar", [(k, hp) for *_r, k, hp in mine])
    return 0


if __name__ == "__main__":
    sys.exit(main())
