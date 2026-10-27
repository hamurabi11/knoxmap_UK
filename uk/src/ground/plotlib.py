"""What the garden plots step (plots2.py) needs from the old raster plots.py and
boundaries.py, which it replaced (2026-10-13 review: about 1,300 lines of their
mains were no longer run): the plot rules' constants, the LiDAR heights along a
side, the grid cuts through a block, a drive for a plot with no frontage, the
fields mask, and the tiles under a shape."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import shapely
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from shapely.geometry import LineString, Polygon  # noqa: E402

from area.roofs import mosaic  # noqa: E402

RECT_TILES = 3             # a plot is rectangles at least this many tiles each way (PLAN.md
                           # principle 13: square plots; blocks.py's own RECT_TILES, for
                           # buildings, is intentionally smaller - principle 12 allows a
                           # building rectangle as narrow as 2)
PLOT_RECTS = 3             # ...and at most this many
OPEN_M = 30.0              # outside residential landuse a plot reaches at most this far past its row
FIELD_TAGS = {
    "landuse": {"farmland", "meadow", "orchard", "vineyard", "allotments", "recreation_ground",
                "village_green", "greenfield", "plant_nursery", "forest", "cemetery"},
    "leisure": {"park", "pitch", "playground", "recreation_ground", "golf_course", "nature_reserve",
                "common", "sports_centre", "track"},
    "natural": {"wood", "scrub", "grassland", "heath", "wetland"},
}
# Building sites: no gardens cut on them, the outline a site fence (Heras
# panels; tall chain-link stands in). Alton's centre, "Rivermead Gardens",
# came out as dirt cut into empty garden plots.
SITE_TAGS = {"landuse": {"construction", "brownfield"}}
SITE_STYLE = "tall_chainlink"
SITE_MIN_TILES = 400                  # smaller is a works compound or a gap, not a site
RESIDENTIAL_AREAS = {"residential", "commercial"}
# How built-up the ground round a row is - the share of it under buildings
# within DENSITY_M. Measured on the High Street map: estate houses 0.06-0.14
# (10th-90th percentile), the High Street 0.14-0.44, median 0.27. At the
# middle of DENSE_LO-DENSE_HI or more a row is in town: a front with no
# hedge in the LiDAR is railings, and any other side a tall hedge.
DENSITY_M = 20.0
DENSE_LO, DENSE_HI = 0.15, 0.30
OUTBUILDING_M2 = 30.0      # a building this small is a shed: no plot of its own
SHED_VALUES = {"shed", "garage", "garages", "hut", "service", "toilets", "boathouse",
               "allotment_house", "greenhouse", "carport"}
NOT_BUILDINGS = {"roof", "carport", "canopy", "ruins", "collapsed", "demolished", "no",
                 "bridge", "storage_tank", "tank", "silo", "transformer_tower", "chimney",
                 "tower", "grandstand", "construction"}
NOT_HOME_TAGS = {"church", "chapel", "cathedral", "mosque", "synagogue", "temple", "school",
                 "college", "university", "kindergarten", "hospital", "civic", "public",
                 "government", "industrial", "warehouse", "train_station", "transportation",
                 "sports_hall", "fire_station"}
NOT_HOME_AMENITY = {"place_of_worship", "school", "college", "kindergarten", "hospital",
                    "townhall", "library", "community_centre", "police", "fire_station",
                    "doctors", "clinic", "village_hall"}
NOT_HOME_AREAS = {"schoolyard", "cemetery", "hospital_grounds", "worship_grounds"}
PAVEMENT_REACH = 4         # pale concrete this near a carriageway is its pavement
DRIVE_MAX = 60             # a drive made for a plot with no frontage is at most this many tiles long
TOWN_FRONT = 8             # a town building this near a street opens onto it across its forecourt
ROW_FORMS = {"terrace_row", "semi_pair"}   # built forms (features.py) whose homes each get a garden
DRIVE_SPLIT = 4            # ...each tile of it with one plot either side costs this many tiles more
LANE_END = 6               # a lane running this far straight away from a plot's side...
LANE_WIDE = 6              # ...and narrower than this along it runs end-on into the plot: its drive
SIDE_CORRIDOR_M = 1.5      # a side is read this far either side: it lands within a metre or two
MIN_GARDEN = 4             # a plot with fewer tiles than this beyond its buildings has no garden
GAP_TILES = 4              # land left between two plots, at most this far from one, goes to one of them
# A side is hedge where this share of it shows hedge in the point cloud
# (boundaries.py asked 0.5; a side lands within a metre or two of the real
# line, so it reads the hedge only in part).
PLOT_HEDGE_SHARE = 0.35

COLOURS = {"hedge_tall": (20, 90, 20), "hedge_low": (110, 200, 80), "brick_tall": (120, 20, 20),
           "brick_low": (230, 70, 50), "tall_wooden": (120, 70, 30), "short_wooden": (230, 180, 110),
           "black_metal": (10, 10, 10), "uk_brick_tall": (120, 20, 20), "uk_brick_low": (230, 70, 50)}

FRONT_STYLES = {"hedge_low", "uk_brick_low", "short_wooden"}   # plots2's style() gives these to fronts only

# The LiDAR reading of a side (from boundaries.py).
CORRIDOR_M = 0.75
RES = 0.5
HEDGE_M = 1.2
FLANK_M = 2.5
WALL_LOW_M = 0.7
WALL_HIGH_M = 3.0
WALL_SHARE = 0.4
OLD_TYPES = ("victorian", "edwardian", "georgian", "listed", "pre1900", "timber", "rural_cottage")


class Heights:
    """Tallest vegetation and tallest solid thing above the ground, on a
    RES-metre grid in British National Grid, from the point cloud."""

    def __init__(self, area, box_):
        e0, n0, e1, n1 = box_
        self.box = box_
        dtm = mosaic("dtm", (int(e0), int(n0), int(e1), int(n1)), os.path.join(area, "lidar"))
        d = np.load(os.path.join(area, "nlp", "core.npz"))
        x, y, z, cls = d["x"], d["y"], d["z"], d["cls"]
        m = (x >= e0) & (x < e1) & (y >= n0) & (y < n1)
        x, y, z, cls = x[m], y[m], z[m], cls[m]
        gr = np.clip((n1 - y).astype(int), 0, dtm.shape[0] - 1)
        gc = np.clip((x - e0).astype(int), 0, dtm.shape[1] - 1)
        above = z - np.nan_to_num(dtm[gr, gc], nan=-999.0)
        w, h = int((e1 - e0) / RES), int((n1 - n0) / RES)
        r = np.clip(((n1 - y) / RES).astype(int), 0, h - 1)
        c = np.clip(((x - e0) / RES).astype(int), 0, w - 1)
        self.veg = np.zeros((h, w), np.float32)
        self.solid = np.zeros((h, w), np.float32)
        v = (cls >= 3) & (cls <= 5)
        s = (cls == 1) | (cls == 6)
        np.maximum.at(self.veg, (r[v], c[v]), above[v])
        np.maximum.at(self.solid, (r[s], c[s]), above[s])

    def along(self, line, corridor=CORRIDOR_M):
        """(share hedge, share wall-like, median wall-like height) along a line,
        looking `corridor` metres either side of it."""
        e0, n0, e1, n1 = self.box
        h, w = self.veg.shape
        k = int(math.ceil(corridor / RES))
        (ax, ay), (bx, by) = line.coords[0], line.coords[-1]
        ln_ = max(1e-6, math.hypot(bx - ax, by - ay))
        nx, ny = -(by - ay) / ln_, (bx - ax) / ln_

        def window(px, py):
            rr, cc = int((n1 - py) / RES), int((px - e0) / RES)
            return max(0, rr - k), min(h, rr + k + 1), max(0, cc - k), min(w, cc + k + 1)

        def tall(px, py):
            r0, r1, c0, c1 = window(px, py)
            return r0 < r1 and c0 < c1 and float(self.veg[r0:r1, c0:c1].max()) >= HEDGE_M
        hedge = wall = 0
        tops = []
        steps = max(2, int(line.length / RES))
        for i in range(steps + 1):
            p = line.interpolate(i / steps, normalized=True)
            r0, r1, c0, c1 = window(p.x, p.y)
            if r0 >= r1 or c0 >= c1:
                continue
            vmax = float(self.veg[r0:r1, c0:c1].max())
            smax = float(self.solid[r0:r1, c0:c1].max())
            if vmax >= HEDGE_M and not (tall(p.x + nx * FLANK_M, p.y + ny * FLANK_M)
                                        and tall(p.x - nx * FLANK_M, p.y - ny * FLANK_M)):
                hedge += 1
            elif WALL_LOW_M <= smax <= WALL_HIGH_M:
                wall += 1
                tops.append(smax)
        n = steps + 1
        return hedge / n, wall / n, (float(np.median(tops)) if tops else 0.0)


def raster(poly, W, H):
    """Tiles whose middle is inside poly: (rows, cols)."""
    if poly is None or poly.is_empty:            # a plot given away whole: no tiles
        return np.zeros(0, int), np.zeros(0, int)
    x0, y0, x1, y1 = poly.bounds
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros(0, int), np.zeros(0, int)
    xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    inside = shapely.contains_xy(poly, xs, ys)
    rr, cc = np.nonzero(inside)
    return rr + y0, cc + x0


def frame_to_real(x, y, fr):
    """A point in the game's tiles back to the real map's tiles (blocks.py's frame)."""
    t, dx, dy, gx, gy = fr
    a = math.radians(-t)
    ux, uy = x - gx, y - gy
    return gx + ux * math.cos(a) - uy * math.sin(a) - dx, gy + ux * math.sin(a) + uy * math.cos(a) - dy


def cuts_on(members, bbox, axis):
    """Straight cuts across one axis that pass through no member: (gap,
    position, members before, members after), one per gap between them."""
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    spans = sorted((bbox[m][lo], bbox[m][hi], m) for m in members)
    gaps = []
    end = spans[0][1]
    for a, b, _m in spans[1:]:
        if a >= end:                       # touching counts as a gap of 0: a party wall
            gaps.append((a - end, (end + a) // 2))
        end = max(end, b)
    out = []
    for gap, c in gaps:
        left = [m for m in members if bbox[m][hi] <= c]
        right = [m for m in members if bbox[m][lo] >= c]
        if len(left) + len(right) == len(members):
            out.append((gap, c, left, right))
    return out


def drive_to_street(m, plot, street, passable, k):
    """The shortest straight drive, RECT_TILES wide, from plot k (mask m) up
    the array (row 0 is up) to a street: (score, drive tiles, mouth tiles),
    or None. Tiles may be another plot's (they are taken) but not a
    building's; the score is the length, plus DRIVE_SPLIT for each row where
    one plot lies on both sides of it (it would cut that garden in two)."""
    h, w = m.shape
    has = m.any(axis=0)
    cols = np.nonzero(has)[0]
    if not len(cols):
        return None
    top = np.where(has, m.argmax(axis=0), -1)
    best = None
    for c in range(int(cols.min()), int(cols.max()) - RECT_TILES + 2):
        span = range(c, c + RECT_TILES)
        if (top[c:c + RECT_TILES] < 0).any():
            continue
        start = int(top[c:c + RECT_TILES].max()) - 1
        end = None
        for y in range(start, max(-1, start - DRIVE_MAX), -1):
            if any(street[y, x] for x in span):
                end = y + 1
                break
            if not all(passable[y, x] or plot[y, x] == k for x in span):
                break
        if end is None or end > start + 1:
            continue
        split = 0
        for y in range(end, start + 1):
            a = plot[y, c - 1] if c > 0 else 0
            b = plot[y, c + RECT_TILES] if c + RECT_TILES < w else 0
            split += bool(a and a == b and a != k)
        score = (start - end + 1) + DRIVE_SPLIT * split
        if best is None or score < best[0]:
            corr = np.zeros(m.shape, bool)
            for x in span:
                corr[end:top[x], x] = True
            corr &= plot != k
            mouth = np.zeros(m.shape, bool)
            for x in span:
                if end > 0 and street[end - 1, x]:
                    mouth[end, x] = True
            best = (score, corr, mouth)
    return best


_FEATURES: dict = {}


def osm_features(map_dir, info, proj) -> list:
    """The map's OSM features where Generate put them: moved with the roads
    as the road graph (or the straightening) moved them, as Build reads them
    (knoxbuild/build.py _points_of_use). Read once a run."""
    if map_dir not in _FEATURES:
        from generator import osm
        from generator.octilinear import straighten_roads
        from generator.renderer import _is_polygon, classify
        feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or []
        if info.get("straight_roads"):
            straighten_roads(feats, proj, classify, _is_polygon, map_dir=map_dir)
        _FEATURES[map_dir] = feats
    return _FEATURES[map_dir]


# Footpaths that part gardens: OSM's footways, paths, cycle and bridleways,
# not pavements (footway=sidewalk / crossing: the street's own) or steps.
PATH_HALF = 1.5            # tiles either side of a footpath's line: three wide, so the grass blending in from both sides leaves dirt in the middle (two wide showed a thin line)
PATH_CARRY = 12.0          # tiles: a footpath point moves with the buildings this near it (blocks.py's moves)
NOT_GARDEN_PATH = {"sidewalk", "crossing", "traffic_island"}


def building_moves(map_dir, name, proj):
    """How blocks.py moved each building after Generate: [(Generate's centre,
    its move)] in tiles, matched by osm_id between <map>_buildings.orig.geojson
    (Generate's) and <map>_buildings.geojson (placed)."""
    import json
    paths_ = [os.path.join(map_dir, f"{name}_buildings{s_}.geojson") for s_ in (".orig", "")]
    if not all(os.path.exists(p_) for p_ in paths_):
        return []
    def centres(path):
        out = {}
        for ft in json.load(open(path, encoding="utf-8"))["features"]:
            g = ft.get("geometry") or {}
            if g.get("type") != "Polygon":
                continue
            poly = Polygon([proj.to_px(la, lo) for lo, la in g["coordinates"][0]])
            if poly.is_valid and poly.area > 0:
                out[str(ft["properties"].get("osm_id"))] = poly.centroid
        return out
    before, after = centres(paths_[0]), centres(paths_[1])
    out = [((c.x, c.y), (after[k].x - c.x, after[k].y - c.y)) for k, c in before.items() if k in after]
    # A move over PATH_CARRY is an outline split or matched afresh, not a move.
    return [(c, m) for c, m in out if math.hypot(*m) <= PATH_CARRY]


def carried(line, moves, tree):
    """A footpath moved with the buildings near it (PATH_CARRY): each point by
    the distance-weighted mean of their moves, fading to none where none is
    near - a path between two houses stays between them (user, 2026-10-13)."""
    from shapely.geometry import Point
    out = []
    for x, y in line.coords:
        sx = sy = wsum = 0.0
        for i in tree.query(Point(x, y).buffer(PATH_CARRY)):
            (cx, cy), (mx, my) = moves[int(i)]
            d = math.hypot(cx - x, cy - y)
            if d < PATH_CARRY:
                w = (1 - d / PATH_CARRY) ** 2
                sx, sy, wsum = sx + w * mx, sy + w * my, wsum + w
        k = 1 / max(wsum, 1.0)                 # fewer or further buildings: less of their move
        out.append((x + sx * k, y + sy * k))
    return LineString(out)


def footpaths(map_dir, info, proj) -> list:
    """Each footpath as (line in tiles, paved?), where Generate put it and
    then moved with the buildings blocks.py moved round it."""
    from generator.renderer import _is_polygon, classify
    out = []
    for f in osm_features(map_dir, info, proj):
        tags = f.tags or {}
        if f.kind != "way" or _is_polygon(f) or len(f.geometry) < 2:
            continue
        if classify(tags) not in ("dirt_path", "paved_path") or tags.get("footway") in NOT_GARDEN_PATH                 or tags.get("highway") == "steps" or tags.get("area") == "yes":
            continue
        out.append((LineString([proj.to_px(la, lo) for la, lo in f.geometry]), classify(tags) == "paved_path"))
    name = os.path.basename(os.path.normpath(map_dir))
    moves = building_moves(map_dir, name, proj)
    if moves:
        from shapely.geometry import Point
        tree = shapely.STRtree([Point(c) for c, _m in moves])
        out = [(carried(ln, moves, tree), pv) for ln, pv in out]
    return out


# Railways: the track bed Generate paints (renderer "railway", 4 m), from
# OSM's lines where Generate put them. No garden crosses one (Alton: plots
# were cut over the line and trees.py grassed 11,000 of its 15,000 tiles).
RAIL_HALF = 2.5


def railway_shape(map_dir, info, proj):
    """The track beds, as one shape in tiles (empty if none)."""
    from generator.renderer import _is_polygon, classify
    lines = [LineString([proj.to_px(la, lo) for la, lo in f.geometry])
             for f in osm_features(map_dir, info, proj)
             if f.kind == "way" and len(f.geometry) >= 2 and not _is_polygon(f)
             and classify(f.tags or {}) == "railway"]
    return shapely.union_all([ln.buffer(RAIL_HALF, cap_style=2) for ln in lines]) if lines else Polygon()


def tagged_polygons(map_dir, info, proj, tags: dict) -> list:
    """OSM's areas with any of these tags, where the straightened map puts them."""
    from generator.renderer import _is_polygon
    out = []
    for f in osm_features(map_dir, info, proj):
        t = f.tags or {}
        if not any(t.get(k) in vals for k, vals in tags.items()):
            continue
        rings = []
        if f.kind == "way" and _is_polygon(f):
            rings = [f.geometry]
        elif f.kind == "relation":
            rings = [ring for role, ring in (getattr(f, "role_geoms", None) or []) if role in ("outer", "")]
        for ring in rings:
            if len(ring) < 4:
                continue
            poly = Polygon([proj.to_px(la, lo) for la, lo in ring])
            out.append(poly if poly.is_valid else poly.buffer(0))
    return out


def field_mask(map_dir, info, proj, W, H, tags: dict = FIELD_TAGS):
    """Tiles OSM maps as fields, parks, woods and the like (or `tags`), where
    the straightened map puts them."""
    out = np.zeros((H, W), bool)
    for poly in tagged_polygons(map_dir, info, proj, tags):
        rr, cc = raster(poly, W, H)
        out[rr, cc] = True
    return out
