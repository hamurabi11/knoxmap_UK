"""Plan each building's roof as ranges, from LiDAR, on a KnoxMap map's tiles.

    python roof_plan.py <knoxmap map dir> <area data dir> [--dormers] [--report <osm_id> ...]

Run after tidy_outlines.py and export_regional.py; adds a "roof" list to each
building in the map's regional.json, which knoxbuild/regional_uk.py turns into
storeys per part and roof pieces at Build.

For every building whose real outline (area buildings.geojson, by OSM id) is
in the map:
  1. Match the real outline to the outline the map will build (turned by the
     map, straightened with the roads, stood upright by tidy_outlines.py):
     the turn and shift that make the two overlap best.
  2. Sample the LiDAR surface and ground at the centre of each of the
     building's tiles, through that match, so the heights line up with the
     tiles the game will build.
  3. Read the roof: a tile is flat, or slopes east/west (a ridge running
     north-south on the tiles) or north/south (a ridge running east-west).
     Smoothed, split into ranges where the ridge turns, small pieces folded
     into their neighbours.
  4. Each range gets its own storeys from its ridge height, and its form:
     a gable along its ridge, a lean-to when its slopes all face one way, or
     flat.
Parts are written in map tile coordinates: [x, y, w, h, form, storeys], with
form one of "WE", "NS" (gable ridges), "flat", "SlopeN/E/S/W" (lean-tos,
named for the way the slope faces, downhill).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import Polygon, shape
from shapely.ops import transform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator.renderer import Projector  # noqa: E402
from knoxbuild.layout import roof_rects  # noqa: E402
from knoxbuild.regional_uk import ATTIC_EAVES  # noqa: E402
from area.roofs import FLAT_SLOPE_DEG, STOREY_RIDGE_M, mosaic  # noqa: E402

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True)

PITCHED_SHARE = 0.4
MIN_PART_TILES = 12
SMOOTH = 5
# STOREY_RIDGE_M (storeys from a part's ridge height) also comes from
# roofs.py, where it was calibrated on EPC property type.
LEAN_TO_SHARE = 0.75      # this share of a part's slopes facing one way: a lean-to
VALLEY_FROM = 8           # ranges at least this wide are looked at for valleys
VALLEY_DEPTH_M = 0.8      # a dip this deep between two ridges is a valley
TOWER_ABOVE_M = 4.0       # a tower stands this far above the building's usual height
TOWER_TILES = 9           # and covers at least this many tiles
TOWER_MAX_SHARE = 0.35    # but no more than this share of the building
TOWER_SOLID = 0.5         # a row or column is tower if tall along this share of it
PLAIN_STOREY_M = 2.7      # a storey's height where there is no roof in it (towers, eaves);
                          # a game storey reads about this tall
SLIVER_TILES = 2          # a part this narrow beside a tower belongs to the tower
FLAT_STEP_M = 3.0         # flat roof at two heights this far apart: two parts (a hall
                          # behind a taller range)
# An attic storey (rooms in the roof, as vanilla B42 builds them): given to
# the top pitched range of a building whose homes (shops.py: knox_homes)
# would otherwise get less than HOME_M2 each on the storeys the ridge height
# gives - a converted loft, often with dormers, that the ridge misses.
HOME_M2 = 50              # gross floor per home, about (NDSS 1-bed 2-person + walls, stairs)
# ATTIC_EAVES (the roof slopes run this far in from each eave, a storey high)
# comes from knoxbuild.regional_uk, which draws the attic: one source of
# truth, so the two files cannot drift (LOG 2026-10-10).
ATTIC_MIN_ACROSS = 2 * ATTIC_EAVES + 5  # a range this wide across its ridge, so the attic
                          # is 5+ wide (vanilla: 4-5) and its inner gable has room to peak
ATTIC_HEADROOM_M = 2.0    # and its ridge this far above its walls' storeys
CHURCHES = {"church", "chapel", "cathedral"}
# OSM building tags a house may carry (front gables: front_gables)
HOUSES = {"house", "detached", "semidetached_house", "terrace", "bungalow", "residential", "yes"}
# A church tower's own wall (West tile of a full set, first shaped piece):
# Odiham's is brick on a flint church; soft red brick is the nearest vanilla.
TOWER_WALL = ["walls_exterior_house_01_004", "walls_exterior_house_05_032"]
MAX_PARTS = 8
SHIFT = 3                 # tiles the LiDAR may be slid to sit on the outline
GROUND_M = 2.0            # an edge strip of the outline lower than this is not building
GROUND_TILES = 2          # and comes off the range only this deep or deeper
# Tall single rooms: their ridge height is not storeys. Churches are planned
# all the same (their tower, nave and aisles are the point of them), at the
# cost of floors in the nave until halls get a tall room of their own.
HALLS = {"school", "hall", "sports_hall", "hangar", "warehouse", "barn",
         "riding_hall", "gymnasium"}
HALL_USES = {"school", "community_centre", "theatre", "cinema"}


def storeys_of(ridge_m: float) -> int:
    return next((n for top, n in STOREY_RIDGE_M if ridge_m < top), 4)


def raster(poly: Polygon, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    import shapely
    xs, ys = np.meshgrid(np.arange(x0, x0 + w) + 0.5, np.arange(y0, y0 + h) + 0.5)
    return shapely.contains_xy(poly, xs, ys)


def best_match(real: Polygon, built: Polygon) -> tuple[float, float, float]:
    """(turn, dx, dy) taking the real outline onto the built one."""
    c0, c1 = real.centroid, built.centroid
    moved = affinity.translate(real, c1.x - c0.x, c1.y - c0.y)
    best = (-1.0, 0.0)
    for turn in np.arange(-45, 45.5, 1.0):
        cand = affinity.rotate(moved, turn, origin=c1)
        inter = cand.intersection(built).area
        iou = inter / max(1e-6, cand.union(built).area)
        if iou > best[0]:
            best = (iou, float(turn))
    return best[1], c1.x - c0.x, c1.y - c0.y


class Lidar:
    def __init__(self, area_dir: str, box):
        cache = os.path.join(area_dir, "lidar")
        self.box = box
        self.dsm = mosaic("dsm", box, cache)
        self.dtm = mosaic("dtm", box, cache)

    def sample(self, e: np.ndarray, n: np.ndarray):
        e0, n0, e1, n1 = self.box
        c = np.clip((e - e0).astype(int), 0, self.dsm.shape[1] - 1)
        r = np.clip((n1 - n).astype(int), 0, self.dsm.shape[0] - 1)
        return self.dsm[r, c], self.dtm[r, c]


def smooth(cls, mask, size=SMOOTH, passes=2):
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


def regions(cls, mask):
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


def border(lab, k):
    m = lab == k
    out = {}
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.roll(np.roll(m, dy, 0), dx, 1)
        for v in lab[sh & ~m]:
            if v >= 0:
                out[v] = out.get(v, 0) + 1
    return out


def plan_building(dsm, ndsm, mask, church=False, front=None):
    """Parts in building-grid coords: [x, y, w, h, form, storeys].

    `church`: storeys from the eaves (a church's height is mostly steep roof
    over lower walls), and its tower counted at a plain storey height.
    `front`: a house's street side (shops.py knox_front), for its front gable
    (front_gables)."""
    gy, gx = np.gradient(dsm)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # 0 flat, 1 slopes east/west (ridge N-S), 2 slopes north/south (ridge W-E)
    cls = np.where(np.abs(gx) >= np.abs(gy), 1, 2)
    cls[(slope <= FLAT_SLOPE_DEG) | np.isnan(slope)] = 0
    # The outermost ring reads the wall drop, not the roof: take it from inside.
    inner = mask.copy()
    inner[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] \
        & mask[1:-1, :-2] & mask[1:-1, 2:]
    inner[[0, -1], :] = False
    inner[:, [0, -1]] = False
    cls = np.where(inner, cls, -9)
    for _ in range(4):
        for y, x in zip(*np.nonzero(mask & (cls == -9))):
            nb = cls[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            nb = nb[nb != -9]
            if len(nb):
                vals, counts = np.unique(nb, return_counts=True)
                cls[y, x] = vals[np.argmax(counts)]
    cls[mask & (cls == -9)] = 0
    if (cls[mask] > 0).mean() < PITCHED_SHARE * 0.5:
        cls[mask] = 0
    # Flat roof at two heights (a supermarket's single-storey hall behind a
    # taller range with a flat top): the high flat is a class of its own (3),
    # so the ranges split at the step and each gets its own storeys.
    flat = mask & (cls == 0) & ~np.isnan(ndsm)
    if flat.sum() >= 2 * MIN_PART_TILES:
        lo_m, hi_m = np.percentile(ndsm[flat], [25, 90])
        if hi_m - lo_m >= FLAT_STEP_M:
            cls[flat & (ndsm > (lo_m + hi_m) / 2)] = 3
    raw = cls.copy()            # before smoothing, which erases dormers
    cls = smooth(cls, mask)

    def leaf(x, y, w, h):
        """Form, storeys and ridge height of one rectangle, from its tiles."""
        sub = np.zeros_like(mask)
        sub[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
        if not sub.any():
            return None
        z = ndsm[sub]
        z = z[~np.isnan(z)]
        ridge = float(np.percentile(z, 90)) if len(z) else 0.0
        if ridge < 2.0:
            return None
        counts = np.bincount(cls[sub], minlength=4)
        counts = np.array([counts[0] + counts[3], counts[1], counts[2]])
        c = int(np.argmax(counts))
        if c == 0 and counts[1:].sum() >= PITCHED_SHARE * counts.sum():
            c = 1 if counts[1] >= counts[2] else 2
        form = {0: "flat", 1: "NS", 2: "WE"}[c]
        if c > 0:
            # All slopes one way: a lean-to, named for the way it faces. Read
            # inside the rectangle's edge: along a taller neighbour (a tower)
            # the drop off its wall reads as slope all facing one way.
            inner = np.zeros_like(sub)
            if w >= 4 and h >= 4:
                inner[y + 1:y + h - 1, x + 1:x + w - 1] = sub[y + 1:y + h - 1, x + 1:x + w - 1]
            else:
                inner = sub
            g = (gx if c == 1 else gy)[inner & (cls == c)]
            g = g[~np.isnan(g)]
            if len(g) >= 4:
                down = (g < 0).mean()     # height falling with x: faces east
                if down >= LEAN_TO_SHARE:
                    form = "SlopeE" if c == 1 else "SlopeS"
                elif down <= 1 - LEAN_TO_SHARE:
                    form = "SlopeW" if c == 1 else "SlopeN"
        if church:
            # Eaves read off the LiDAR where they are: along the range's long
            # sides that are the building's outside wall (an aisle's outer
            # wall). Inferring them from the ridge (half the width below it)
            # made a narrow aisle as tall as the nave beside it. A range with
            # no outside long side (the nave between its aisles) keeps that.
            edges = []
            if form == "NS":
                for col, beyond in ((x, x - 1), (x + w - 1, x + w)):
                    if beyond < 0 or beyond >= mask.shape[1] or not mask[y:y + h, beyond].any():
                        z = ndsm[y:y + h, col][mask[y:y + h, col]]
                        edges.append(float(np.nanmedian(z)) if len(z) else np.nan)
            elif form == "WE":
                for row, beyond in ((y, y - 1), (y + h - 1, y + h)):
                    if beyond < 0 or beyond >= mask.shape[0] or not mask[beyond, x:x + w].any():
                        z = ndsm[row, x:x + w][mask[row, x:x + w]]
                        edges.append(float(np.nanmedian(z)) if len(z) else np.nan)
            edges = [v for v in edges if not np.isnan(v)]
            eaves = min(edges) if edges else ridge - (min(w, h) / 2 if form in ("WE", "NS") else 0)
            # Whole storeys, with a little allowance for LiDAR noise at the cut.
            storeys = max(1, int((eaves + 0.6) // PLAIN_STOREY_M))
        else:
            storeys = storeys_of(ridge)
        return [x, y, w, h, form, storeys, round(ridge, 1)]

    def purity(x, y, w, h):
        sub = cls[y:y + h, x:x + w][mask[y:y + h, x:x + w]]
        return int(np.bincount(sub, minlength=3).max()) if len(sub) else 0

    def split(x, y, w, h, depth=0):
        """Cut a rectangle in two along one straight line where the roof
        clearly changes across it (a front range and a back range), else keep it."""
        total = int(mask[y:y + h, x:x + w].sum())
        here = purity(x, y, w, h)
        best = None
        if depth < 3 and total >= 2 * MIN_PART_TILES and here < 0.8 * total:
            for c in range(3, w - 2):                     # upright cuts
                gain = purity(x, y, c, h) + purity(x + c, y, w - c, h) - here
                if best is None or gain > best[0]:
                    best = (gain, (x, y, c, h), (x + c, y, w - c, h))
            for c in range(3, h - 2):                     # level cuts
                gain = purity(x, y, w, c) + purity(x, y + c, w, h - c) - here
                if best is None or gain > best[0]:
                    best = (gain, (x, y, w, c), (x, y + c, w, h - c))
        if best and best[0] >= 0.15 * total:
            return split(*best[1], depth + 1) + split(*best[2], depth + 1)
        return [(x, y, w, h)]

    def valleys(x, y, w, h, form):
        """A wide range split at the valleys between its ridges: the height
        profile across it, averaged along the ridge, has a ridge (peak) per
        range and a valley between each pair. Nave and aisles; a double pile."""
        across = h if form == "WE" else w
        if across < VALLEY_FROM:
            return [(x, y, w, h)]
        block = np.where(mask[y:y + h, x:x + w], dsm[y:y + h, x:x + w], np.nan)
        with np.errstate(all="ignore"):
            prof = np.nanmean(block, axis=1 if form == "WE" else 0)
        cuts = [i for i in range(2, across - 2)
                if not np.isnan(prof[i - 1:i + 2]).any()
                and prof[i] < prof[i - 1] and prof[i] <= prof[i + 1]
                and min(np.nanmax(prof[:i]), np.nanmax(prof[i + 1:])) - prof[i] >= VALLEY_DEPTH_M]
        # Keep valleys at least three tiles apart, deepest first.
        kept = []
        for c in sorted(cuts, key=lambda i: prof[i]):
            if all(abs(c - k) >= 3 for k in kept):
                kept.append(c)
        edges = [0] + sorted(kept) + [across]
        if form == "WE":
            return [(x, y + a, w, b - a) for a, b in zip(edges, edges[1:])]
        return [(x + a, y, b - a, h) for a, b in zip(edges, edges[1:])]

    # A tower: a compact patch standing well above the rest (a church tower,
    # a turret), carved out as a part of its own before the roof is read.
    rest = mask.copy()
    parts = []
    base = float(np.nanpercentile(ndsm[mask], 60))
    tall = mask & (ndsm > base + TOWER_ABOVE_M)
    # Only churches: on a house or a shop the tall part is the front range
    # over lower rear wings, which the ranges below already handle.
    if church and tall.sum() >= TOWER_TILES:
        lab, info = regions(tall.astype(int), tall)
        for k, (c, area) in info.items():
            if c != 1 or area < TOWER_TILES:
                continue
            ys, xs = np.nonzero(lab == k)
            # The tower's extent by the rows and columns that are tall for at
            # least half its run: one noisy tile at the edge made All Saints'
            # tower a tile wider than its nave and off its centre line.
            blob = lab == k
            span_y = slice(ys.min(), ys.max() + 1)
            span_x = slice(xs.min(), xs.max() + 1)
            cols = [c for c in range(xs.min(), xs.max() + 1) if blob[span_y, c].mean() >= TOWER_SOLID]
            rows = [r for r in range(ys.min(), ys.max() + 1) if blob[r, span_x].mean() >= TOWER_SOLID]
            if not cols or not rows:
                continue
            tx0, ty0, tx1, ty1 = min(cols), min(rows), max(cols) + 1, max(rows) + 1
            if (tx1 - tx0) * (ty1 - ty0) > TOWER_MAX_SHARE * mask.sum():
                continue
            z = ndsm[ty0:ty1, tx0:tx1][mask[ty0:ty1, tx0:tx1]]
            height = float(np.nanpercentile(z, 90))
            parts.append([int(tx0), int(ty0), int(tx1 - tx0), int(ty1 - ty0), "flat",
                          max(storeys_of(height), round(height / PLAIN_STOREY_M)), round(height, 1)])
            rest[ty0:ty1, tx0:tx1] = False

    def trim(x, y, w, h):
        """Edge rows and columns the LiDAR shows as ground (the outline
        overshooting the building) come off the range as low strips."""
        def low(sl):
            z = ndsm[sl][mask[sl]]
            return len(z) > 0 and float(np.nanmedian(z)) < GROUND_M
        def depth(side):
            """Ground rows/columns in from one side, before the building starts."""
            n = 0
            while True:
                if side == "W" and w - n > 3:
                    sl = (slice(y, y + h), slice(x + n, x + n + 1))
                elif side == "E" and w - n > 3:
                    sl = (slice(y, y + h), slice(x + w - 1 - n, x + w - n))
                elif side == "N" and h - n > 3:
                    sl = (slice(y + n, y + n + 1), slice(x, x + w))
                elif side == "S" and h - n > 3:
                    sl = (slice(y + h - 1 - n, y + h - n), slice(x, x + w))
                else:
                    return n
                if not low(sl):
                    return n
                n += 1

        # One low row is the eaves or a metre of outline error: only a strip
        # of ground at least GROUND_TILES deep comes off.
        strips = []
        n = depth("W")
        if n >= GROUND_TILES:
            strips.append([x, y, n, h]); x += n; w -= n
        n = depth("E")
        if n >= GROUND_TILES:
            strips.append([x + w - n, y, n, h]); w -= n
        n = depth("N")
        if n >= GROUND_TILES:
            strips.append([x, y, w, n]); y += n; h -= n
        n = depth("S")
        if n >= GROUND_TILES:
            strips.append([x, y + h - n, w, n]); h -= n
        return (x, y, w, h), strips

    for bx, by, bw, bh, _caps in roof_rects(rest.astype(int).tolist()):
        for rect0 in split(bx, by, bw, bh):
            rect, strips = trim(*rect0)
            for sx_, sy_, sw_, sh_ in strips:
                parts.append([sx_, sy_, sw_, sh_, "flat", 1, 0.0])
            p = leaf(*rect)
            if not p:
                continue
            if p[4] in ("WE", "NS"):
                for sub in valleys(*rect, p[4]):
                    q = leaf(*sub)
                    if q:
                        q[4] = p[4] if q[4] not in ("WE", "NS") else q[4]
                        parts.append(q)
            else:
                parts.append(p)
    # Strips a tile or two wide along a tower are the tower's own edge (the
    # LiDAR's fall-off around it): the tower takes them.
    towers = [p for p in parts if p[4] == "flat" and p[5] == max(q[5] for q in parts)
              and p[6] >= base + TOWER_ABOVE_M]
    for t in towers:
        for p in [p for p in parts if p is not t and min(p[2], p[3]) <= SLIVER_TILES]:
            px, py, pw, ph = p[:4]
            tx, ty, tw, th = t[:4]
            along_x = pw >= ph and px >= tx and px + pw <= tx + tw and (py + ph == ty or py == ty + th)
            along_y = ph > pw and py >= ty and py + ph <= ty + th and (px + pw == tx or px == tx + tw)
            if along_x:
                t[1], t[3] = min(ty, py), th + ph
                parts.remove(p)
            elif along_y:
                t[0], t[2] = min(tx, px), tw + pw
                parts.remove(p)
    # A lean-to is an outshut against something taller. Slopes all one way on
    # a part as tall as the building are half of a roof whose other half the
    # LiDAR gave to the neighbour, or one side of a hip: built alone at full
    # height they stand up like a board. Those become gables on the same ridge.
    top = max((p[5] for p in parts), default=0)
    for p in parts:
        if p[4].startswith("Slope") and p[5] >= top:
            p[4] = "NS" if p[4] in ("SlopeE", "SlopeW") else "WE"
    if front and not church:
        parts = front_gables(parts, raw, mask, gx, gy, front)
    return parts, dormers_of(parts, raw, mask, gx, gy)


# Front gables (to-do K09.23, 2026-10-19): a gabled cross-wing at one end of a
# house's main range, its gable to the street (the Edwardian and 1930s semi's
# gable over the bay, the 1980s estate house's small gable). Too small for a
# part of its own (MIN_PART_TILES), so the ranges above fold it into the main
# range; read again here from the LiDAR before smoothing: an end bay of the
# range whose slopes run across the range - both ways, about a ridge of its
# own; a hipped end's fall one way only - becomes a part with its ridge
# across, the range shortened. One a house; the data decides, no rule adds one.
FRONT_GABLE = (3, 4)           # the cross-wing this many squares along the range
FRONT_GABLE_SHARE = 0.45       # this share of its squares sloping across the range...
FRONT_GABLE_BOTH = 0.3         # ...each way at least this share of those
FRONT_GABLE_RANGE = (3, (5, 10))   # the range left at least this long (principle 8), and this deep
MADE = {"front_gables": 0}


def front_gables(parts, raw, mask, gx, gy, front):
    if front not in ("N", "S", "E", "W"):
        return parts
    out, done = [], False
    for p in parts:
        x, y, w, h, form, st = p[:6]
        along_x = form == "WE"
        if done or form not in ("WE", "NS") or st < 2 or (front in ("N", "S")) != along_x:
            out.append(p)
            continue
        length, depth = (w, h) if along_x else (h, w)
        if length < FRONT_GABLE_RANGE[0] + FRONT_GABLE[0] or not FRONT_GABLE_RANGE[1][0] <= depth <= FRONT_GABLE_RANGE[1][1]:
            out.append(p)
            continue
        across = 1 if along_x else 2                   # raw: 1 slopes E/W (ridge N-S), 2 slopes N/S
        grad = gx if along_x else gy
        best = None
        for k in FRONT_GABLE:
            for end in (0, 1):
                if along_x:
                    a = x if end == 0 else x + w - k
                    sl = (slice(y, y + h), slice(a, a + k))
                else:
                    a = y if end == 0 else y + h - k
                    sl = (slice(a, a + k), slice(x, x + w))
                m = mask[sl]
                cells, g = raw[sl][m], grad[sl][m]
                if m.sum() < 0.8 * k * depth:
                    continue
                turned = cells == across
                share = turned.mean()
                if share < FRONT_GABLE_SHARE or share <= (cells == 3 - across).mean():
                    continue
                gt = g[turned]
                gt = gt[~np.isnan(gt)]
                if len(gt) < 4 or min((gt < 0).mean(), (gt > 0).mean()) < FRONT_GABLE_BOTH:
                    continue                               # one way only: a hipped end
                if best is None or share > best[0]:
                    best = (share, k, end)
        if best is None:
            out.append(p)
            continue
        _share, k, end = best
        rest, wing = list(p), list(p)
        if along_x:
            wing[0], wing[2] = (x if end == 0 else x + w - k), k
            rest[0], rest[2] = (x + k if end == 0 else x), w - k
            wing[4] = "NS"
        else:
            wing[1], wing[3] = (y if end == 0 else y + h - k), k
            rest[1], rest[3] = (y + k if end == 0 else y), h - k
            wing[4] = "WE"
        out += [rest, wing]
        done = True
        MADE["front_gables"] += 1
    return out


DORMER_TILES = (2, 9)     # a dormer's sideways faces cover this many tiles
DORMER_WIDTH = (2, 3)     # and it is built this wide along the ridge
DORMER_EVERY = 4          # one dormer per this many tiles of ridge, at most


def dormers_of(parts, raw, mask, gx, gy):
    """Dormers on the top ranges' slopes: small clusters of tiles facing
    sideways to the roof they sit in, on one side of the ridge and clear of
    the gable ends. A dormer's own little roof slopes both ways sideways, so
    the cluster must face both ways; a hip end, a valley or a chimney faces
    one. At most one dormer per DORMER_EVERY tiles of ridge. [x, y, w, h, RoofType]."""
    found = []
    top = max((p[5] for p in parts), default=0)
    for x, y, w, h, form, st, _ridge in parts:
        if form not in ("WE", "NS") or st < top or min(w, h) < 5:
            continue
        side = 1 if form == "WE" else 2          # faces across the main slopes
        sub = np.zeros_like(mask)
        sub[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
        # Clear of the gable ends, where the end slopes read the same way.
        if form == "WE":
            sub[:, :x + 2] = False
            sub[:, x + w - 2:] = False
        else:
            sub[:y + 2, :] = False
            sub[y + h - 2:, :] = False
        lab, info = regions(np.where(sub, raw, -1), sub)
        here = []
        for k, (c, area) in info.items():
            if c != side or not DORMER_TILES[0] <= area <= DORMER_TILES[1]:
                continue
            g = (gx if side == 1 else gy)[lab == k]
            g = g[~np.isnan(g)]
            both = min((g > 0).sum(), (g < 0).sum())
            if both < 1 or both < 0.25 * len(g):
                continue
            ys, xs = np.nonzero(lab == k)
            cx, cy = int(round(xs.mean())), int(round(ys.mean()))
            dw = max(DORMER_WIDTH[0], min(DORMER_WIDTH[1], xs.max() - xs.min() + 1))
            if form == "WE":
                ridge_y = y + h / 2
                kind = "DormerN" if cy < ridge_y else "DormerS"
                dx0 = max(x + 2, min(x + w - 2 - dw, cx - dw // 2))
                dy0 = y if kind == "DormerN" else y + h - 2
                here.append([dx0, dy0, dw, 2, kind])
            else:
                dh = max(DORMER_WIDTH[0], min(DORMER_WIDTH[1], ys.max() - ys.min() + 1))
                ridge_x = x + w / 2
                kind = "DormerW" if cx < ridge_x else "DormerE"
                dy0 = max(y + 2, min(y + h - 2 - dh, cy - dh // 2))
                dx0 = x if kind == "DormerW" else x + w - 2
                here.append([dx0, dy0, 2, dh, kind])
        found += here[:max(1, (w if form == "WE" else h) // DORMER_EVERY)]
    # One dormer per spot: drop any that overlap one already kept.
    kept = []
    for d in found:
        if all(d[0] + d[2] <= k[0] or k[0] + k[2] <= d[0] or d[1] + d[3] <= k[1]
               or k[1] + k[3] <= d[1] for k in kept):
            kept.append(d)
    return kept


def largest_rect(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    """The largest rectangle of True in a grid, (x, y, w, h)."""
    h, w = grid.shape
    run = np.zeros(w, dtype=int)
    best, out = 0, None
    for y in range(h):
        run = np.where(grid[y], run + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = run[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                start, top = stack.pop()
                if top * (x - start) > best:
                    best, out = top * (x - start), (start, y - top + 1, x - start, top)
            stack.append((start, cur))
    return out


def add_attics(parts, homes: int, dwelling: str | None, m2_per_tile: float,
               front: str | None = None) -> list:
    """An attic storey where a building's homes do not fit the storeys its
    ridge gives (HOME_M2 each). The attic goes in the largest rectangle of
    its top storeys' roof - one range, which the LiDAR may have read as a
    slope, a flat top and a slope (a roof with rooms in it) - if that is wide
    enough across its ridge and high enough. The range is added as a part of
    its own, after the others (it takes their tiles), one storey higher,
    its ridge the way most of its pitched roof runs. Returns its rect and the
    eave its dormers look out of: the street side (`front`, shops.py) when
    that is an eave, else none."""
    if not homes or dwelling not in ("flats", "shop_flats") or not parts:
        return []
    lowest = 1 if dwelling == "shop_flats" else 0     # the shop has the ground floor
    floor = sum(p[2] * p[3] * max(0, p[5] - lowest) for p in parts) * m2_per_tile
    if floor >= HOME_M2 * homes:
        return []
    top = max(p[5] for p in parts)
    tops = [p for p in parts if p[5] == top and p[4] in ("WE", "NS", "flat")]
    if not tops:
        return []
    gx0 = min(p[0] for p in tops)
    gy0 = min(p[1] for p in tops)
    grid = np.zeros((max(p[1] + p[3] for p in tops) - gy0, max(p[0] + p[2] for p in tops) - gx0), bool)
    for p in tops:
        grid[p[1] - gy0:p[1] - gy0 + p[3], p[0] - gx0:p[0] - gx0 + p[2]] = True
    got = largest_rect(grid)
    if not got:
        return []
    x, y, w, h = int(got[0] + gx0), int(got[1] + gy0), int(got[2]), int(got[3])
    inside = [p for p in tops if p[0] < x + w and x < p[0] + p[2] and p[1] < y + h and y < p[1] + p[3]]
    ns = sum(p[2] * p[3] for p in inside if p[4] == "NS")
    we = sum(p[2] * p[3] for p in inside if p[4] == "WE")
    form = "NS" if ns > we else "WE" if we > ns else ("WE" if w >= h else "NS")
    across = h if form == "WE" else w
    ridge = max(p[6] for p in inside)
    if across < ATTIC_MIN_ACROSS or ridge < top * PLAIN_STOREY_M + ATTIC_HEADROOM_M:
        return []
    parts.append([x, y, w, h, form, top + 1, ridge])
    eaves = ("E", "W") if form == "NS" else ("N", "S")
    return [[x, y, w, h, front if front in eaves else ""]]


def read_lidar(proj, lidar, tile_poly, real_poly, frame=None) -> dict | None:
    """The LiDAR on a building's tiles in the finished map.

    Without a `frame`, the real outline is matched to the built one (turn and
    shift) and the LiDAR then slid up to SHIFT tiles to sit on the outline
    (OSM outlines are traced off aerial photos and sit a metre or two out).
    With one - a frame stored by an earlier step - it is read exactly there.
    Returns x0, y0, w, h, mask, dsm, ndsm, the frame actually used (slide
    folded in, so a later step reading in it lands on the same tiles) and
    the slide.
    """
    x0, y0 = int(math.floor(tile_poly.bounds[0])), int(math.floor(tile_poly.bounds[1]))
    w = int(math.ceil(tile_poly.bounds[2])) - x0
    h = int(math.ceil(tile_poly.bounds[3])) - y0
    mask = raster(tile_poly, x0, y0, w, h)
    if mask.sum() < 12:
        return None
    if frame:
        turn, dx, dy, cx, cy = frame
    else:
        turn, dx, dy = best_match(real_poly, tile_poly)
        cx, cy = tile_poly.centroid.x, tile_poly.centroid.y
    P = 0 if frame else SHIFT
    xs, ys = np.meshgrid(np.arange(x0 - P, x0 + w + P) + 0.5, np.arange(y0 - P, y0 + h + P) + 0.5)
    a = math.radians(-turn)
    ux, uy = xs - cx, ys - cy
    rx = cx + ux * math.cos(a) - uy * math.sin(a) - dx
    ry = cy + ux * math.sin(a) + uy * math.cos(a) - dy
    lat_lon = [proj.to_latlon(px, py) for px, py in zip(rx.ravel(), ry.ravel())]
    e, n = np.array([TO_BNG.transform(lon, lat) for lat, lon in lat_lon]).T
    dsm_all, dtm_all = lidar.sample(e, n)
    dsm_all = dsm_all.reshape(h + 2 * P, w + 2 * P)
    ndsm_all = dsm_all - dtm_all.reshape(h + 2 * P, w + 2 * P)
    best = (-1.0, 0, 0)
    for sy in range(-P, P + 1):
        for sx in range(-P, P + 1):
            on = ndsm_all[P + sy:P + sy + h, P + sx:P + sx + w] > 2.5
            iou = (on & mask).sum() / max(1, (on | mask).sum())
            if iou > best[0] + 0.01 or (abs(iou - best[0]) <= 0.01 and abs(sx) + abs(sy) < abs(best[1]) + abs(best[2])):
                best = (iou, sx, sy)
    _iou, sx, sy = best
    # Reading tile (x, y) at (x + sx, y + sy) is the same as moving the frame's
    # offset by that slide turned through the frame's turn.
    sdx = sx * math.cos(a) - sy * math.sin(a)
    sdy = sx * math.sin(a) + sy * math.cos(a)
    return {"x0": x0, "y0": y0, "w": w, "h": h, "mask": mask,
            "dsm": dsm_all[P + sy:P + sy + h, P + sx:P + sx + w],
            "ndsm": ndsm_all[P + sy:P + sy + h, P + sx:P + sx + w],
            "frame": [turn, dx - sdx, dy - sdy, cx, cy], "shift": (sx, sy)}


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    report = set(sys.argv[sys.argv.index("--report") + 1:]) if "--report" in sys.argv else set()
    with_dormers = "--dormers" in sys.argv
    name = os.path.basename(os.path.normpath(map_dir))
    with open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    b = info["bbox"]
    proj = Projector.build(b["south"], b["west"], b["north"], b["east"],
                           info["meters_per_tile"], info.get("rotation") or 0.0)
    with open(os.path.join(map_dir, f"{name}_buildings.geojson"), encoding="utf-8") as f:
        built = {str(ft["properties"].get("osm_id")): ft for ft in json.load(f)["features"]}
    with open(os.path.join(area, "buildings.geojson"), encoding="utf-8") as f:
        real = {str(ft["properties"]["osm_id"]): ft for ft in json.load(f)["features"]}
    reg_path = os.path.join(map_dir, "regional.json")
    with open(reg_path, encoding="utf-8") as f:
        regional = json.load(f)

    # LiDAR for the whole map, once.
    corners = [proj.to_latlon(x, y) for x, y in ((0, 0), (proj.width, 0), (0, proj.height),
                                                (proj.width, proj.height))]
    es, ns = zip(*[TO_BNG.transform(lon, lat) for lat, lon in corners])
    lidar = Lidar(area, (int(min(es)) - 5, int(min(ns)) - 5, int(max(es)) + 5, int(max(ns)) + 5))

    planned = skipped = halls = 0
    forms = {}
    shifts = {}
    for osm_id, ft in built.items():
        rid = str(ft["properties"].get("parent_osm_id") or osm_id)   # a unit's own building
        # A building OSM lacks, drawn from the LiDAR by blocks.py, has no
        # real outline here - only the frame it was placed in, which is all
        # the reading needs.
        if (rid not in real and not ft["properties"].get("lidar_frame")) or ft["geometry"]["type"] != "Polygon":
            continue
        tags = ft["properties"]
        if (tags.get("building") in HALLS or tags.get("amenity") in HALL_USES):
            # One tall room reads as storeys in the LiDAR: KnoxMap's own height.
            entry = regional["buildings"].get(osm_id)
            if entry:
                entry.pop("levels", None)
                entry.pop("roof", None)
            halls += 1
            continue
        rid = str(ft["properties"].get("parent_osm_id") or osm_id)
        tile_poly = Polygon([proj.to_px(lat, lon) for lon, lat in ft["geometry"]["coordinates"][0]])
        real_poly = Polygon([proj.to_px(lat, lon) for lon, lat in real[rid]["geometry"]["coordinates"][0]])             if rid in real else tile_poly
        if not tile_poly.is_valid or not real_poly.is_valid or tile_poly.area < 12:
            continue
        got = read_lidar(proj, lidar, tile_poly, real_poly, ft["properties"].get("lidar_frame"))
        if got is None:
            continue
        x0, y0, w, h, mask, dsm, ndsm = (got[k] for k in ("x0", "y0", "w", "h", "mask", "dsm", "ndsm"))
        turn = got["frame"][0]
        shifts[got["shift"]] = shifts.get(got["shift"], 0) + 1
        if np.nanpercentile(ndsm[mask], 90) < 2.0:
            skipped += 1                 # not in the LiDAR (built since)
            continue
        church = tags.get("building") in CHURCHES or tags.get("amenity") == "place_of_worship"
        house = tags.get("knox_dwelling") in ("house", None) and tags.get("building") in HOUSES
        parts, dormers = plan_building(dsm, ndsm, mask, church=church,
                                       front=tags.get("knox_front") if house else None)
        if not parts:
            skipped += 1
            continue
        entry = regional["buildings"].setdefault(osm_id, {})
        if entry.get("flat_roof"):
            # A superstore (export_regional.uk_shopfronts): one flat roof at its full
            # height, not the LiDAR's ranges (stray gables and lean-tos on a 1990s store).
            top = max(q[5] for q in parts)
            parts = [list(q[:4]) + ["flat", top] + list(q[6:]) for q in parts]
            dormers = []
        elif entry.get("pitched_roof"):
            # A barn or farm shed (export_regional.uk_farm): pitched, its ridge along its
            # length, wherever the roof plan made it flat (the LiDAR's flat or complex readings);
            # a strip under 4 squares stays flat.
            parts = [list(q[:4]) + [("WE" if q[2] >= q[3] else "NS") if q[4] == "flat" and min(q[2], q[3]) >= 4
                                    else q[4]] + list(q[5:]) for q in parts]
        attics = add_attics(parts, int(tags.get("knox_homes") or 0), tags.get("knox_dwelling"),
                            info["meters_per_tile"] ** 2, tags.get("knox_front"))
        entry["attic"] = [[x0 + a[0], y0 + a[1]] + a[2:] for a in attics]
        forms["attic"] = forms.get("attic", 0) + len(attics)
        entry["roof"] = [[x0 + p[0], y0 + p[1]] + p[2:6] for p in parts]
        # Dormers are off until they are placed and drawn right (LOG 2026-10-06).
        entry["dormers"] = [[x0 + d[0], y0 + d[1]] + d[2:] for d in dormers] if with_dormers else []
        forms["dormer"] = forms.get("dormer", 0) + len(entry["dormers"])
        entry["levels"] = max(p[5] for p in parts)
        if church:
            entry["hall"] = True         # nave and aisles open to the roof
            entry["tower_wall"] = TOWER_WALL
        planned += 1
        for p in parts:
            forms[p[4]] = forms.get(p[4], 0) + 1
        if osm_id in report:
            print(f"--- {osm_id} turn {turn:+.0f} deg, {w}x{h} tiles")
            for p in parts:
                print("   ", p)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(regional, f, indent=1)
    print(f"roof plans: {planned} buildings; {skipped} without usable LiDAR, "
          f"{halls} halls/churches (KnoxMap roofs)")
    print("  parts by form:", forms, "; front gables:", MADE["front_gables"])
    moved = sum(n for (sx, sy), n in shifts.items() if sx or sy)
    print(f"  LiDAR slid onto the outline for {moved} of {sum(shifts.values())} buildings;"
          f" most common shifts: {sorted(shifts.items(), key=lambda kv: -kv[1])[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
