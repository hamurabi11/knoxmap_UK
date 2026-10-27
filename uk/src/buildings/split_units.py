"""Split long outlines into the separate shops and houses they really are.

    python split_units.py <knoxmap map dir> <area data dir> [--report <osm_id> ...]

Run after tidy_outlines.py / lidar_outlines.py, before export_regional.py.

OSM often draws a row of shops or a terrace as one polygon. For every long
outline (LONG_MIN tiles or more, and long for its width) this gathers, along
its length:
  LiDAR   the ridge height of each slice across it, and which way its roof
          slopes; a step of STEP_M or more, or a change of direction (a gable
          to the street beside a range along it), is a boundary between two
          buildings
  listing separate Historic England list entries standing inside it
  shops   shops, pubs, offices and the like mapped inside it
  homes   property points (UPRNs), grouped along its length (flats over a
          shop stack in one place, so they count once)
Addresses decide first (PLAN A6): the property points (UPRN, and OSM shop
points; addresses.py), carried into the game through the building's frame,
are grouped along its frontage - points within HOME_GAP of each other are one
property (flats over a shop stack in one place) - and the building is cut
midway between neighbouring properties, each cut moved to a LiDAR roof step
within SNAP tiles, no property narrower than MIN_HOME. The frontage is the
long side with more road and pavement beyond it; a deep block with addresses
more than FRONT_DEPTH tiles back as well has its back cut off between them
and split by its own. Only where fewer than two properties are found on the
frontage does it fall back to the roof, listings and shops: LiDAR boundaries
first, then midway between two or more listing entries or shops, units at
least MIN_UNIT tiles long. Units cut on an earlier run are rejoined first, so
each run starts again from the whole building. Each unit becomes a building of its own in
the map's outline file: osm_id "<id>u<k>", parent_osm_id, unit k, and the
LiDAR frame it was read in, so the roof planner reads each unit exactly
where this did. Everything decided is written to <map>/units_report.csv.
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import os
import sys

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box, mapping

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
from regional import roof_plan as RP  # noqa: E402

LONG_MIN = 12          # tiles: outlines at least this long are looked at
LONG_ASPECT = 1.8      # ...and this much longer than wide (or LONG_ANY long)
LONG_ANY = 18
MIN_UNIT = 3           # tiles of frontage a unit needs
STEP_M = 1.0           # ridge height step marking a boundary
STEP_RUN = 3           # slices each side of a step that must agree
LEVEL_M = 0.6          # ...sitting level to within this
DIRECTION_STEP = 0.6   # change in the share of slopes facing across
LEVEL_FACING = 0.25    # ...each side consistent to within this
END_ZONE = 4           # no cut this close to either end
HOME_GAP = 3           # tiles between property points for separate homes
SNAP = 2               # tiles an address cut may move to meet a LiDAR roof step
MIN_HOME = 4           # tiles of frontage a property cut by its addresses needs (rooms and a stair)
ADDRESS_REACH = 1.0    # tiles outside an outline an address point may still be its
FRONT_LOOK = 8         # tiles beyond a long side looked at for the street
OWN_LOOK = 3           # tiles beyond a back part's free sides looked at for a way of its own
FRONT_DEPTH = 12       # tiles from the street side an address counts as on the frontage
BACK_MIN = 5           # a back part cut off at least this deep...
FRONT_MIN = 6          # ...leaving the front at least this deep
HALL_MAX_M = 5.0       # a slice no higher than this (LiDAR, above ground)...
HALL_FLAT = 0.7        # ...and mostly flat is hall: it goes to the nearest business
POI_KEYS = ("shop", "amenity", "office", "craft", "tourism")
SKIP = {"church", "chapel", "cathedral", "school", "hall", "barn", "garage", "garages"}


def load_pois(map_dir: str) -> list[tuple[float, float, dict]]:
    path = glob.glob(os.path.join(map_dir, "*_osm.json.gz"))[0]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        feats = json.load(f)["features"]
    out = []
    for ft in feats:
        if ft["kind"] != "node" or not any(k in ft["tags"] for k in POI_KEYS):
            continue
        g = ft["geometry"]
        g = json.loads(g) if isinstance(g, str) else g
        lat, lon = g[0] if isinstance(g[0], list) else g
        out.append((lat, lon, ft["tags"]))
    return out


def steps(profile: np.ndarray, threshold: float, spread: float) -> list[tuple[int, float]]:
    """(index, size) of each step in a profile: a cut c where the STEP_RUN
    slices before and after each sit level (within `spread`) and differ by
    at least `threshold`, kept where it is the biggest step within STEP_RUN.

    Level on both sides is what separates two buildings from one roof: a
    hip or a gable end falls away gradually, a neighbour's roof does not.
    Nothing within END_ZONE of either end, where every roof drops away."""
    n = len(profile)
    size = np.zeros(n)
    for c in range(END_ZONE, n - END_ZONE + 1):
        a = profile[max(0, c - STEP_RUN):c]
        b = profile[c:c + STEP_RUN]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) >= 2 and len(b) >= 2 and np.ptp(a) <= spread and np.ptp(b) <= spread:
            size[c] = abs(np.median(a) - np.median(b))
    out = []
    for c in range(END_ZONE, n - END_ZONE + 1):
        lo, hi = max(0, c - STEP_RUN), min(n, c + STEP_RUN + 1)
        if size[c] >= threshold and size[c] == size[lo:hi].max():
            out.append((c, float(size[c])))
    return out


def groups(positions: list[float], gap: float) -> list[float]:
    """Centres of runs of positions closer than `gap` to each other."""
    out, run = [], []
    for p in sorted(positions):
        if run and p - run[-1] > gap:
            out.append(sum(run) / len(run))
            run = []
        run.append(p)
    if run:
        out.append(sum(run) / len(run))
    return out


def keep_apart(cuts: list[tuple[int, float]], length: int) -> list[int]:
    """Cuts at least MIN_UNIT apart and from the ends, strongest first."""
    kept = []
    for c, _s in sorted(cuts, key=lambda cs: -cs[1]):
        if max(MIN_UNIT, END_ZONE) <= c <= length - max(MIN_UNIT, END_ZONE) \
                and all(abs(c - k) >= MIN_UNIT for k in kept):
            kept.append(c)
    return sorted(kept)


def address_groups(along: list[float], gap: float) -> list[tuple[float, float]]:
    """(first, last) position of each run of address positions no further
    than `gap` apart along a frontage: one property each (flats over a shop
    stack in one place, and a shop's EPC and OSM points sit together)."""
    runs, run = [], []
    for p in sorted(along):
        if run and p - run[-1] > gap:
            runs.append((run[0], run[-1]))
            run = []
        run.append(p)
    if run:
        runs.append((run[0], run[-1]))
    return runs


def address_runs(points: list[tuple[float, bool, bool]], gap: float) -> list[dict]:
    """Properties along a frontage from (position, business, big format)
    points: runs no further than `gap` apart, each with its extent and
    whether a business, and one needing a large floor, is in it."""
    runs, run = [], []
    for pt in sorted(points):
        if run and pt[0] - run[-1][0] > gap:
            runs.append(run)
            run = []
        run.append(pt)
    if run:
        runs.append(run)
    return [{"first": r[0][0], "last": r[-1][0], "business": any(b for _p, b, _g in r),
             "big": any(g for _p, _b, g in r)} for r in runs]


def assign_slices(runs: list[dict], hall: list[bool], n: int, lidar: list[int]) -> list[tuple[int, int, int]]:
    """Each slice along a building to the property it belongs to: the nearest
    one; a slice of hall (a low flat single storey, where nobody lives) to the
    nearest business, one needing a large floor first. Returns segments
    (start, end, run) with no segment narrower than MIN_HOME (merged into the
    one before), each cut moved to a LiDAR roof step within SNAP tiles."""
    business = [k for k, r in enumerate(runs) if r["business"]]
    big = [k for k, r in enumerate(runs) if r["big"]]

    def dist(i, r):
        return 0.0 if r["first"] <= i + 0.5 <= r["last"] + 1 else min(abs(i + 0.5 - r["first"]), abs(i + 0.5 - r["last"]))
    owner = []
    for i in range(n):
        pool = (big or business) if hall[i] and business else range(len(runs))
        owner.append(min(pool, key=lambda k: dist(i, runs[k])))
    segs = []
    for i, o in enumerate(owner):
        if segs and segs[-1][2] == o:
            segs[-1][1] = i + 1
        else:
            segs.append([i, i + 1, o])
    # Cuts to a roof step nearby, then no property too narrow.
    for a, b in zip(segs, segs[1:]):
        near = [l for l in lidar if abs(l - b[0]) <= SNAP and a[0] + MIN_HOME <= l <= b[1] - MIN_HOME]
        if near:
            c = min(near, key=lambda l: abs(l - b[0]))
            a[1] = b[0] = c
    merged = []
    for sg in segs:
        if merged and (sg[1] - sg[0] < MIN_HOME or merged[-1][1] - merged[-1][0] < MIN_HOME):
            keep = merged[-1] if merged[-1][1] - merged[-1][0] >= sg[1] - sg[0] else sg
            merged[-1] = [merged[-1][0], sg[1], keep[2]]
        else:
            merged.append(list(sg))
    return [tuple(sg) for sg in merged]


def address_cuts(runs: list[tuple[float, float]], lidar: list[int], n: int) -> list[int]:
    """Cuts midway between neighbouring properties, each moved to a LiDAR roof
    step within SNAP tiles, no unit narrower than MIN_HOME."""
    cuts = []
    for (_a, last), (first, _b) in zip(runs, runs[1:]):
        c = (last + first) / 2
        near = [l for l in lidar if abs(l - c) <= SNAP and last < l < first]
        c = min(near, key=lambda l: abs(l - c)) if near else int(round(c))
        cuts.append(int(c))
    kept = []
    for c in sorted(set(cuts)):
        if c >= MIN_HOME and n - c >= MIN_HOME and (not kept or c - kept[-1] >= MIN_HOME):
            kept.append(c)
    return kept


def rejoin_units(features: list[dict], proj) -> list[dict]:
    """A building this tool cut before, put back together (its units are strips
    of it, so their union is the outline as it was), so each run starts
    again from the whole building."""
    from shapely.ops import unary_union
    units, out = {}, []
    for ft in features:
        p = ft["properties"]
        if p.get("parent_osm_id") and p.get("unit") is not None and "unit_why" in p:
            units.setdefault(str(p["parent_osm_id"]), []).append(ft)
        else:
            out.append(ft)
    for parent, fts in units.items():
        polys = [Polygon([proj.to_px(la, lo) for lo, la in f["geometry"]["coordinates"][0]]) for f in fts]
        whole = unary_union([q if q.is_valid else q.buffer(0) for q in polys])
        whole = whole.buffer(0.01, join_style=2).buffer(-0.01, join_style=2)
        if whole.geom_type != "Polygon":
            whole = max(whole.geoms, key=lambda g: g.area)
        props = dict(min(fts, key=lambda f: f["properties"]["unit"])["properties"])
        for k in ("parent_osm_id", "unit", "units", "unit_why", "unit_band", "unit_hall_of"):
            props.pop(k, None)
        props["osm_id"] = int(parent) if parent.isdigit() else parent
        ring = [proj.to_latlon(x, y) for x, y in whole.exterior.coords]
        out.append({"type": "Feature", "properties": props,
                    "geometry": mapping(Polygon([(lon, lat) for lat, lon in ring]))})
    return out


def street_mask(map_dir: str, name: str):
    """Roads and their pavements on the ground as Generate drew it."""
    from PIL import Image
    from scipy import ndimage
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
    import ukpaths  # noqa: E402
    sys.path.insert(0, ukpaths.KNOX)
    from generator import pz_colors as C
    path = os.path.join(map_dir, f"{name}_ground_base.prekerbs.bmp")
    if not os.path.exists(path):
        path = os.path.join(map_dir, f"{name}_ground_base.bmp")
    g = np.array(Image.open(path).convert("RGB"))

    def colour(c):
        return (g[:, :, 0] == c[0]) & (g[:, :, 1] == c[1]) & (g[:, :, 2] == c[2])
    road = np.zeros(g.shape[:2], bool)
    for c in (C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE):
        road |= colour(c)
    return road | (colour(C.PALE_CONCRETE) & ndimage.binary_dilation(road, iterations=4))


def ways_mask(map_dir: str, info, proj, street):
    """The street mask (roads, lanes, their pavements) with OSM's footpaths."""
    from PIL import Image, ImageDraw
    from ground import plotlib as P
    H, W = street.shape
    img = Image.new("1", (W, H), 0)
    dr = ImageDraw.Draw(img)
    for line, _paved in P.footpaths(map_dir, info, proj):
        dr.line([tuple(c) for c in line.coords], fill=1, width=2)
    return street | np.array(img, bool)


def own_way(ways, x0, y0, w, h, along_x: bool, band, mask=None) -> bool:
    """Whether a back part - `band`, (lo, hi) across the depth from the
    building's own corner - fronts a road, lane or footpath of its own: any
    within OWN_LOOK of the part's own squares, outside the building (so a lane
    between two wings counts as well as one past the far side or an end). A
    house behind the street with its door on a side lane is a home of its own
    (user, 2026-10-19, 52 High Street, Odiham: a brick house behind the corner
    one, its door on the lane); a back part with none is the street house's
    rear wing. `mask` is the building's squares in its box (h x w), else the
    whole box."""
    from scipy import ndimage
    H, W = ways.shape
    k = OWN_LOOK
    x0, y0 = int(round(x0)), int(round(y0))
    m = np.ones((h, w), bool) if mask is None else np.asarray(mask, bool)[:h, :w]
    lo, hi = max(0, band[0]), min(h if along_x else w, band[1])
    part = np.zeros_like(m)
    if along_x:              # depth along y: the band is rows lo..hi
        part[lo:hi, :] = m[lo:hi, :]
    else:                    # depth along x: the band is columns lo..hi
        part[:, lo:hi] = m[:, lo:hi]
    if not part.any():
        return False
    big = np.pad(part, k)
    near = ndimage.binary_dilation(big, np.ones((2 * k + 1, 2 * k + 1), bool)) & ~np.pad(m, k)
    ys, xs = np.nonzero(near)
    ys, xs = ys + y0 - k, xs + x0 - k
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return bool(ways[ys[inside], xs[inside]].any())


def street_at_end(street, x0, y0, x1, y1, along_x: bool) -> bool:
    """The street is at a short end, not along a long side: front and back
    cannot be told by depth (32 High Street, Odiham: its long sides see no
    street, its east end 144 squares of it)."""
    H, W = street.shape
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    if along_x:
        sides = (street[max(0, y0 - FRONT_LOOK):max(0, y0), max(0, x0):min(W, x1)].sum(),
                 street[min(H, y1):min(H, y1 + FRONT_LOOK), max(0, x0):min(W, x1)].sum())
        ends = (street[max(0, y0):min(H, y1), max(0, x0 - FRONT_LOOK):max(0, x0)].sum(),
                street[max(0, y0):min(H, y1), min(W, x1):min(W, x1 + FRONT_LOOK)].sum())
    else:
        sides = (street[max(0, y0):min(H, y1), max(0, x0 - FRONT_LOOK):max(0, x0)].sum(),
                 street[max(0, y0):min(H, y1), min(W, x1):min(W, x1 + FRONT_LOOK)].sum())
        ends = (street[max(0, y0 - FRONT_LOOK):max(0, y0), max(0, x0):min(W, x1)].sum(),
                street[min(H, y1):min(H, y1 + FRONT_LOOK), max(0, x0):min(W, x1)].sum())
    return max(ends) > max(sides)


def frontage(street, x0, y0, x1, y1, along_x: bool) -> str:
    """Which long side of a box faces the street: "lo" (N or W) or "hi"."""
    H, W = street.shape
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    if along_x:
        lo = street[max(0, y0 - FRONT_LOOK):max(0, y0), max(0, x0):min(W, x1)].sum()
        hi = street[min(H, y1):min(H, y1 + FRONT_LOOK), max(0, x0):min(W, x1)].sum()
    else:
        lo = street[max(0, y0):min(H, y1), max(0, x0 - FRONT_LOOK):max(0, x0)].sum()
        hi = street[max(0, y0):min(H, y1), min(W, x1):min(W, x1 + FRONT_LOOK)].sum()
    return "lo" if lo >= hi else "hi"


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    report = set(sys.argv[sys.argv.index("--report") + 1:]) if "--report" in sys.argv else set()
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json")))
    b = info["bbox"]
    proj = RP.Projector.build(b["south"], b["west"], b["north"], b["east"],
                              info["meters_per_tile"], info.get("rotation") or 0.0)
    gj_path = os.path.join(map_dir, f"{name}_buildings.geojson")
    gj = json.load(open(gj_path, encoding="utf-8"))
    gj["features"] = rejoin_units(gj["features"], proj)
    real = {str(f["properties"]["osm_id"]): f for f in
            json.load(open(os.path.join(area, "buildings.geojson"), encoding="utf-8"))["features"]}
    # Listings, shops and addresses where they really are; each building
    # carries them into the game through its own frame (trees.to_game).
    from ground.trees import to_game
    from buildings import addresses as A
    from area import area_conf
    nhle = area_conf.extra(area, "nhle_points.geojson")      # none: no listings to split by
    listings = [proj.to_px(f["geometry"]["coordinates"][0][1], f["geometry"]["coordinates"][0][0])
                for f in json.load(open(nhle))["features"]] if nhle else []
    addrs = A.load(map_dir, area, proj, info)
    shop_pts = [(a.x, a.y) for a in addrs if a.source == "osm"]
    street = street_mask(map_dir, name)
    ways = ways_mask(map_dir, info, proj, street)
    corners = [proj.to_latlon(x, y) for x, y in ((0, 0), (proj.width, 0), (0, proj.height),
                                                (proj.width, proj.height))]
    es, ns = zip(*[RP.TO_BNG.transform(lon, lat) for lat, lon in corners])
    lidar = RP.Lidar(area, (int(min(es)) - 5, int(min(ns)) - 5, int(max(es)) + 5, int(max(ns)) + 5))
    from scipy.spatial import cKDTree
    # reshape(-1, 2): an empty area (no uprn.csv rows, no OSM shop points) is
    # a real case downstream tolerates (query_ball_point just finds nothing),
    # but np.array([]) on an empty list is 1-D and cKDTree rejects it outright.
    addr_tree = cKDTree(np.array([(a.x, a.y) for a in addrs], dtype=float).reshape(-1, 2))

    rows, new_features = [], []
    split_n = looked = 0
    for ft in gj["features"]:
        props = ft["properties"]
        osm_id = str(props.get("osm_id"))
        if ft["geometry"]["type"] != "Polygon" or osm_id not in real or props.get("parent_osm_id") \
                or props.get("building") in SKIP or props.get("amenity") == "place_of_worship":
            new_features.append(ft)
            continue
        poly = Polygon([proj.to_px(lat, lon) for lon, lat in ft["geometry"]["coordinates"][0]])
        minx, miny, maxx, maxy = poly.bounds
        along_x = (maxx - minx) >= (maxy - miny)
        length, width = (maxx - minx, maxy - miny) if along_x else (maxy - miny, maxx - minx)
        if length < LONG_MIN or (length < LONG_ANY and length < LONG_ASPECT * width):
            new_features.append(ft)
            continue
        looked += 1
        real_poly = Polygon([proj.to_px(lat, lon) for lon, lat in real[osm_id]["geometry"]["coordinates"][0]])
        if not poly.is_valid or not real_poly.is_valid:
            new_features.append(ft)
            continue
        try:
            got = RP.read_lidar(proj, lidar, poly, real_poly, props.get("lidar_frame"))
        except shapely.errors.GEOSException:
            got = None          # an outline the matching cannot read: left whole
        if got is None:
            new_features.append(ft)
            continue
        x0, y0, w, h, mask, dsm, ndsm = (got[k] for k in ("x0", "y0", "w", "h", "mask", "dsm", "ndsm"))
        n = w if along_x else h
        depth_n = h if along_x else w
        # Ridge height and roof direction, slice by slice along the length.
        gy, gx = np.gradient(dsm)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        across = (np.abs(gx) < np.abs(gy)) if along_x else (np.abs(gx) >= np.abs(gy))
        ridge, facing = np.full(n, np.nan), np.full(n, np.nan)
        flat_share = np.zeros(n)
        for i in range(n):
            sl = (slice(None), i) if along_x else (i, slice(None))
            z = ndsm[sl][mask[sl]]
            z = z[~np.isnan(z)]
            if len(z) >= 2:
                ridge[i] = np.percentile(z, 80)
                pitched = mask[sl] & (slope[sl] > RP.FLAT_SLOPE_DEG)
                flat_share[i] = 1 - pitched.sum() / max(1, mask[sl].sum())
                if pitched.sum() >= 2:
                    facing[i] = across[sl][pitched].mean()
        # Hall: a low, flat single storey (a supermarket's floor, a works).
        hall = [bool(not np.isnan(ridge[i]) and ridge[i] <= HALL_MAX_M and flat_share[i] >= HALL_FLAT)
                for i in range(n)]
        lidar_cuts = steps(ridge, STEP_M, LEVEL_M) + steps(facing, DIRECTION_STEP, LEVEL_FACING)
        lidar_at = [c for c, _s in lidar_cuts]
        # Addresses carried into the game through this building's frame, as a
        # position along its length and a depth from its street side.
        fr = props.get("lidar_frame") or got["frame"]
        near = addr_tree.query_ball_point((real_poly.centroid.x, real_poly.centroid.y), length + width + 10)
        side = frontage(street, x0, y0, x0 + w, y0 + h, along_x)
        reach = poly.buffer(ADDRESS_REACH)
        front_pos, back_pos, depths, front_pts = [], [], [], []
        for j in near:
            gxp, gyp = to_game(addrs[j].x, addrs[j].y, fr)
            if not reach.contains(Point(gxp, gyp)):
                continue
            pos = (gxp - x0) if along_x else (gyp - y0)
            d = (gyp - y0) if along_x else (gxp - x0)
            depth = d if side == "lo" else depth_n - d
            depths.append(depth)
            (front_pos if depth <= FRONT_DEPTH else back_pos).append(pos)
            if depth <= FRONT_DEPTH:
                front_pts.append((pos, addrs[j].business, addrs[j].big))

        def along(pts):
            out = []
            for p in pts:
                q = to_game(p[0], p[1], fr)
                if poly.contains(Point(q)):
                    out.append((q[0] - x0) if along_x else (q[1] - y0))
            return out
        list_pos = along(listings)
        shop_pos = along(shop_pts)
        runs = address_groups(front_pos, HOME_GAP)
        back_runs = address_groups(back_pos, HOME_GAP)
        props_runs = address_runs(front_pts, HOME_GAP)
        reason = ""
        segments = None
        # A shop that needs a large floor (a supermarket) is the whole ground
        # floor of its building, the homes flats over it (user, 2026-10-09):
        # the building is kept whole.
        # ...and so is one whose own OSM tags say it is a supermarket or a department
        # store (Alton's Waitrose: one address, so its roof steps cut it in six
        # "works", 2026-10-20; asset_creator/commercial_kit PLAN step 2).
        big_tags = ((props.get("building") or "") in ("supermarket", "department_store")
                    or (props.get("shop") or "") in ("supermarket", "department_store"))
        if big_tags or any(r["big"] for r in props_runs):
            rows.append({"osm_id": osm_id, "length": int(length), "width": int(width), "lidar_steps": "",
                         "listings": 0, "shops": 0, "addresses_front": len(front_pos),
                         "addresses_back": len(back_pos), "properties_front": len(props_runs),
                         "properties_back": 0, "units": 1, "why": "large shop: kept whole"})
            new_features.append(ft)
            continue
        if len(props_runs) >= 2:
            segments = assign_slices(props_runs, hall, n, lidar_at)
            cuts = [sg[0] for sg in segments[1:]]
            reason = "addresses" if cuts else ""
        elif len(props_runs) == 1 and not props_runs[0]["business"] and not list_pos and not shop_pos:
            # One home: a step in its roof is its own lower wing, not another
            # building (a semi's rear range came out a one-storey house of its
            # own: Normandy Street, Alton, 2026-10-19)
            cuts = []
            reason = "one home: kept whole"
        else:
            # Too few addresses to say: the roof, listings and shops (as before).
            cuts = keep_apart(lidar_cuts, n)
            reason = "lidar" if cuts else ""
            evidence = max(len(list_pos), len(shop_pos))
            if evidence >= 2:
                pos = sorted(list_pos if len(list_pos) >= len(shop_pos) else shop_pos)
                mids = [(int(round((a + c) / 2)), 0.5) for a, c in zip(pos, pos[1:])]
                more = keep_apart([(c, 99.0) for c in cuts] + mids, n)
                if len(more) > len(cuts):
                    reason = (reason + "+" if reason else "") + ("listings" if len(list_pos) >= len(shop_pos) else "shops")
                    cuts = more
        # A deep block with properties at the back as well as on the street:
        # the back cut off at a line between them, and split by its own.
        depth_cut = None
        if runs and back_runs and depth_n >= FRONT_DEPTH + BACK_MIN \
                and street_at_end(street, x0, y0, x0 + w, y0 + h, along_x):
            reason = (reason + "+" if reason else "") + "street at an end"
        elif runs and back_runs and depth_n >= FRONT_DEPTH + BACK_MIN:
            front_d = max(d for d in depths if d <= FRONT_DEPTH)
            back_d = min(d for d in depths if d > FRONT_DEPTH)
            dc = int(round((front_d + back_d) / 2))
            dc = max(FRONT_MIN, min(depth_n - BACK_MIN, dc))
            depth_cut = dc if side == "lo" else depth_n - dc
            back_band = (depth_cut, depth_n + 1) if side == "lo" else (-1, depth_cut)
            if own_way(ways, x0, y0, w, h, along_x, back_band, mask):
                reason = (reason + "+" if reason else "") + "back"
            else:
                # No way of its own: the street house's rear wing, its
                # addresses homes under the street door.
                depth_cut = None
                reason = (reason + "+" if reason else "") + "rear wing"
        back_cuts = address_cuts(back_runs, lidar_at, n) if depth_cut is not None and len(back_runs) >= 2 else []
        rows.append({"osm_id": osm_id, "length": int(length), "width": int(width),
                     "lidar_steps": ";".join(f"{c}:{s:.1f}" for c, s in sorted(lidar_cuts)),
                     "listings": len(list_pos), "shops": len(shop_pos),
                     "addresses_front": len(front_pos), "addresses_back": len(back_pos),
                     "properties_front": len(runs), "properties_back": len(back_runs),
                     "units": (len(cuts) + 1) + ((len(back_cuts) + 1) if depth_cut is not None else 0),
                     "why": reason or "kept whole"})
        if osm_id in report:
            print(f"--- {osm_id} {props.get('name') or ''} {int(length)}x{int(width)} along "
                  f"{'x' if along_x else 'y'}, street side {side}")
            print(f"    lidar cuts {sorted(lidar_cuts)}; front addresses at {sorted(round(p, 1) for p in front_pos)}"
                  f" -> {len(runs)} properties; back {sorted(round(p, 1) for p in back_pos)}")
            print(f"    -> cuts {cuts}, depth cut {depth_cut}, back cuts {back_cuts} ({reason or 'kept whole'})")
        if not cuts and depth_cut is None:
            new_features.append(ft)
            continue
        split_n += 1
        # Bands across the depth (front, back), each cut along its length.
        if depth_cut is None:
            bands = [("front", (-1, depth_n + 1), cuts)]
        else:
            lo_band, hi_band = (-1, depth_cut), (depth_cut, depth_n + 1)
            bands = [("front", lo_band if side == "lo" else hi_band, cuts),
                     ("back", hi_band if side == "lo" else lo_band, back_cuts)]
        made = []
        # Which property each front unit belongs to, and the unit holding its
        # addresses: a unit of another property's (its hall) says whose.
        owner_of = [sg[2] for sg in segments] if segments and len(segments) == len(cuts) + 1 else None
        home_unit = {}
        if owner_of:
            for k_, sg in enumerate(segments):
                r = props_runs[sg[2]]
                if sg[0] <= r["first"] + 0.5 and r["last"] <= sg[1]:
                    home_unit.setdefault(sg[2], k_)
        for band, (d0, d1), band_cuts in bands:
            edges = [-1] + band_cuts + [n + 1]
            for seg_i, (a, c) in enumerate(zip(edges, edges[1:])):
                strip = box(x0 + a, y0 + d0, x0 + c, y0 + d1) if along_x else box(x0 + d0, y0 + a, x0 + d1, y0 + c)
                part = poly.intersection(strip)
                if part.is_empty:
                    continue
                if part.geom_type != "Polygon":
                    part = max(part.geoms, key=lambda g: g.area)
                if part.area < MIN_UNIT * MIN_UNIT:
                    continue
                ring = [proj.to_latlon(x, y) for x, y in part.exterior.coords]
                unit_props = dict(props)
                unit_props.update({"osm_id": f"{osm_id}u{len(made)}", "parent_osm_id": osm_id,
                                   "unit": len(made), "lidar_frame": got["frame"], "unit_why": reason,
                                   "unit_band": band})
                unit_props.pop("unit_hall_of", None)
                if band == "front" and owner_of:
                    hk = home_unit.get(owner_of[seg_i])
                    if hk is not None and hk != seg_i and props_runs[owner_of[seg_i]]["business"]:
                        unit_props["unit_hall_of"] = f"{osm_id}u{hk}"
                made.append({"type": "Feature", "properties": unit_props,
                             "geometry": mapping(Polygon([(lon, lat) for lat, lon in ring]))})
        for f in made:
            f["properties"]["units"] = len(made)
        new_features += made
    gj["features"] = new_features
    with open(gj_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(gj, f)
    os.replace(gj_path + ".tmp", gj_path)
    with open(os.path.join(map_dir, "units_report.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["osm_id"])
        wr.writeheader()
        wr.writerows(rows)
    from collections import Counter
    print(f"long outlines looked at: {looked}; split: {split_n} into "
          f"{sum(r['units'] for r in rows if r['units'] > 1)} units")
    print("  why:", dict(Counter(r["why"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
