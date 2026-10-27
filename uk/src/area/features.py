"""Per-building evidence for assigning building types.

    python features.py <knoxmap map dir> <data dir> [south west north east]

Reads the map's cached OSM download (every building keeps its OSM id there),
plus whatever evidence files exist in <data dir> at its data level (area_conf):
    nhle_points.geojson, nhle_polygons.geojson   (fetch_nhle.py)
    uprn.csv                                     (filter_uprn.py)
Without UPRNs a single outline's homes come from OSM's tag alone
(refine_with_tags: building=terrace a terrace row, semidetached_house a pair, apartments flats).
Writes <data dir>/buildings.geojson and buildings.csv, one row per building.

Everything is measured in British National Grid metres (EPSG:27700).
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import os
import sys
from collections import defaultdict

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import transform
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: our packages
from area import area_conf  # noqa: E402

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True)

# Not buildings for our purposes: KnoxMap skips these too.
NOT_BUILDINGS = {"roof", "carport", "ruins", "construction", "tank", "silo",
                 "storage_tank", "bunker", "no"}
# Outbuildings: they never make a house a semi or a terrace.
OUTBUILDING_TAGS = {"garage", "garages", "shed", "hut", "service", "barn",
                    "farm_auxiliary", "greenhouse", "stable", "cowshed", "carport"}
OUTBUILDING_M2 = 30.0
PARTY_WALL_M = 2.5          # shared edge longer than this is a party wall
TOUCH_M = 0.4               # gap still counted as touching (OSM drawing slop)
STREET_SEARCH_M = 80.0

HOUSE_TAGS = {"house", "detached", "semidetached_house", "terrace", "bungalow",
              "residential", "yes", "cottage", "farmhouse"}


def bng(lat_lon_ring):
    return [TO_BNG.transform(lon, lat) for lat, lon in lat_lon_ring]


def load_osm(map_dir: str):
    path = glob.glob(os.path.join(map_dir, "*_osm.json.gz"))[0]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"]
    for f in feats:
        if isinstance(f["geometry"], str):
            f["geometry"] = json.loads(f["geometry"])
    return feats


def buildings_from(feats, box):
    out = []
    for f in feats:
        tags = f["tags"]
        btag = (tags.get("building") or "").lower()
        if not btag or btag in NOT_BUILDINGS or f["kind"] != "way":
            continue
        ring = f["geometry"]
        if len(ring) < 4:
            continue
        poly = Polygon(bng(ring))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 4:
            continue
        lat = sum(p[0] for p in ring) / len(ring)
        lon = sum(p[1] for p in ring) / len(ring)
        if box and not (box[0] <= lat <= box[2] and box[1] <= lon <= box[3]):
            continue
        out.append({"osm_id": f["osm_id"], "tags": tags, "btag": btag,
                    "poly": poly, "lat": lat, "lon": lon})
    return out


def streets_from(feats):
    lines, names = [], []
    for f in feats:
        t = f["tags"]
        if f["kind"] != "way" or "highway" not in t or "name" not in t:
            continue
        if t["highway"] in ("footway", "path", "cycleway", "bridleway", "steps", "track"):
            continue
        pts = bng(f["geometry"])
        if len(pts) >= 2:
            lines.append(LineString(pts))
            names.append(t["name"])
    return lines, names


def shape_stats(poly: Polygon) -> dict:
    mrr = poly.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    sides = sorted({round(math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])), 2)
                    for i in range(2)})
    long_side = max(sides)
    short_side = min(sides) if len(sides) > 1 else long_side
    # Orientation of the long side, 0-180 degrees from east.
    i = 0 if math.dist((xs[0], ys[0]), (xs[1], ys[1])) >= math.dist((xs[1], ys[1]), (xs[2], ys[2])) else 1
    ang = math.degrees(math.atan2(ys[i + 1] - ys[i], xs[i + 1] - xs[i])) % 180
    return {
        "area_m2": round(poly.area, 1),
        "perimeter_m": round(poly.length, 1),
        "rectangularity": round(poly.area / mrr.area, 3) if mrr.area else 0,
        "long_m": round(long_side, 1),
        "short_m": round(short_side, 1),
        "aspect": round(long_side / short_side, 2) if short_side else 0,
        "orient_deg": round(ang, 1),
        "vertices": len(poly.exterior.coords) - 1,
    }


def is_outbuilding(b) -> bool:
    return b["btag"] in OUTBUILDING_TAGS or (
        b["btag"] in ("yes",) and b["poly"].area < OUTBUILDING_M2)


def party_walls(blds):
    """Pairs of main buildings sharing a wall, with the shared length."""
    polys = [b["poly"] for b in blds]
    tree = STRtree(polys)
    pairs = {}
    for i, b in enumerate(blds):
        if is_outbuilding(b):
            continue
        grown = b["poly"].buffer(TOUCH_M)
        for j in tree.query(grown):
            j = int(j)
            if j <= i or is_outbuilding(blds[j]):
                continue
            shared = grown.intersection(blds[j]["poly"].exterior).length
            if shared >= PARTY_WALL_M:
                pairs[(i, j)] = shared
    return pairs


def built_form(blds, pairs):
    """detached / semi / end_terrace / mid_terrace / outbuilding, by the
    groups of buildings joined by party walls."""
    adj = defaultdict(set)
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    group_of, groups = {}, []
    for i in range(len(blds)):
        if i in group_of:
            continue
        stack, members = [i], []
        group_of[i] = len(groups)
        while stack:
            k = stack.pop()
            members.append(k)
            for n in adj[k]:
                if n not in group_of:
                    group_of[n] = len(groups)
                    stack.append(n)
        groups.append(members)
    for i, b in enumerate(blds):
        size = len(groups[group_of[i]])
        if is_outbuilding(b):
            form = "outbuilding"
        elif size == 1:
            form = "detached"
        elif size == 2:
            form = "semi"
        else:
            form = "end_terrace" if len(adj[i]) <= 1 else "mid_terrace"
        b["form"] = form
        b["group"] = group_of[i]
        b["group_size"] = size
        b["party_walls"] = len(adj[i])


def add_streets(blds, lines, names):
    tree = STRtree(lines)
    for b in blds:
        c = b["poly"].centroid
        near = tree.query(c.buffer(STREET_SEARCH_M))
        best = None
        for k in near:
            d = lines[int(k)].distance(b["poly"])
            if best is None or d < best[0]:
                best = (d, names[int(k)], lines[int(k)])
        if best:
            b["street"] = best[1]
            b["setback_m"] = round(best[0], 1)
            # Does the long side run along the street (frontage) or end-on?
            line = best[2]
            p = line.interpolate(line.project(c))
            q = line.interpolate(min(line.length, line.project(c) + 2))
            if p.distance(q) > 0:
                street_ang = math.degrees(math.atan2(q.y - p.y, q.x - p.x)) % 180
                diff = abs(street_ang - b["orient_deg"]) % 180
                b["to_street_deg"] = round(min(diff, 180 - diff), 1)
        else:
            b["street"], b["setback_m"] = "", None


def add_listing(blds, data_dir):
    pol_path = area_conf.extra(data_dir, "nhle_polygons.geojson")
    pt_path = area_conf.extra(data_dir, "nhle_points.geojson")
    if not pol_path or not pt_path:
        return
    entries = []
    for path in (pol_path, pt_path):
        with open(path, encoding="utf-8") as f:
            for feat in json.load(f)["features"]:
                geom = transform(lambda x, y, z=None: TO_BNG.transform(x, y), shape(feat["geometry"]))
                if not geom.is_valid:          # Westminster's has self-touching rings
                    geom = geom.buffer(0)
                entries.append((geom, feat["properties"]))
    # Candidates near each building only (thousands of entries in a city), in list order.
    tree = STRtree([g.buffer(3) if g.geom_type == "Point" else g for g, _ in entries])
    for b in blds:
        hit = None
        for k in sorted(int(i) for i in tree.query(b["poly"])):
            geom, props = entries[k]
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                inter = geom.intersection(b["poly"]).area
                if inter >= 0.3 * b["poly"].area or inter >= 0.3 * geom.area:
                    hit = props
                    break
            elif b["poly"].buffer(3).intersects(geom):
                hit = props
                break
        if hit:
            b["listed_grade"] = hit.get("Grade")
            b["list_entry"] = hit.get("ListEntry")
            b["list_name"] = hit.get("Name")


def add_uprns(blds, data_dir):
    path = area_conf.extra(data_dir, "uprn.csv")
    if not path:
        return
    pts, ids = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append(Point(float(row["X_COORDINATE"]), float(row["Y_COORDINATE"])))
            ids.append(row["UPRN"])
    tree = STRtree(pts)
    for b in blds:
        inside = [ids[int(k)] for k in tree.query(b["poly"]) if b["poly"].contains(pts[int(k)])]
        b["uprns"] = ";".join(inside)
        b["uprn_count"] = len(inside)


def refine_with_uprns(blds):
    """One outline holding several homes. Mappers often draw a pair of semis
    or a whole terrace row as a single polygon, so it has no party wall to
    find; the number of addresses inside it gives it away."""
    for b in blds:
        n = b.get("uprn_count")
        if n is None or b["form"] == "outbuilding" or b["group_size"] > 1:
            continue
        per_home = b["area_m2"] / n if n else 0
        if n == 2:
            b["form"] = "semi_pair"
        elif 3 <= n <= 12 and per_home <= 110 and (
                b["aspect"] >= 1.8 or b["rectangularity"] < 0.85):
            # Long, or stepped/L-shaped from rear extensions, at a house's
            # footprint per home. Flats vs terrace is settled by EPC later.
            b["form"] = "terrace_row"
        elif n >= 3:
            b["form"] = "multi_unit"      # flats, or shops with flats over


# OSM's tag on a single outline (no party wall) when there are no UPRNs to count.
TAG_FORMS = {"terrace": "terrace_row", "semidetached_house": "semi_pair", "apartments": "multi_unit",
             "flats": "multi_unit"}


def refine_with_tags(blds):
    """No property points: a single outline that OSM calls a terrace or flats."""
    for b in blds:
        if "uprn_count" not in b and b["form"] == "detached" and b["btag"] in TAG_FORMS:
            b["form"] = TAG_FORMS[b["btag"]]


NON_RESIDENTIAL = {"school", "church", "chapel", "hospital", "commercial", "retail",
                   "industrial", "public", "civic", "office", "warehouse", "pavilion",
                   "fire_station", "college", "kindergarten", "hotel", "supermarket",
                   "sports_hall", "government", "train_station", "transportation"}


def mark_non_residential(blds):
    """Form describes houses; a school's wings sharing walls are not semis."""
    for b in blds:
        t = b["tags"]
        if b["btag"] in NON_RESIDENTIAL or any(k in t for k in ("amenity", "shop", "office")):
            if b["form"] != "outbuilding":
                b["form"] = "non_residential"


def check_against_osm(blds):
    """How often the geometric form agrees with the form OSM mappers tagged."""
    expect = {"semidetached_house": {"semi", "semi_pair"},
              "terrace": {"mid_terrace", "end_terrace", "terrace_row"},
              "detached": {"detached"}, "bungalow": {"detached", "semi"}}
    rows = defaultdict(lambda: defaultdict(int))
    for b in blds:
        if b["btag"] in expect:
            rows[b["btag"]][b["form"]] += 1
    print("\nform from geometry vs OSM tag:")
    for tag, got in rows.items():
        total = sum(got.values())
        ok = sum(n for f, n in got.items() if f in expect[tag])
        print(f"  {tag:20s} {ok}/{total} agree   {dict(got)}")


def main() -> int:
    map_dir, data_dir = sys.argv[1], sys.argv[2]
    box = tuple(map(float, sys.argv[3:7])) if len(sys.argv) >= 7 else None
    feats = load_osm(map_dir)
    blds = buildings_from(feats, box)
    for b in blds:
        b.update(shape_stats(b["poly"]))
    built_form(blds, party_walls(blds))
    add_streets(blds, *streets_from(feats))
    add_listing(blds, data_dir)
    add_uprns(blds, data_dir)
    refine_with_uprns(blds)
    refine_with_tags(blds)
    mark_non_residential(blds)

    cols = ["osm_id", "btag", "form", "group", "group_size", "party_walls", "area_m2",
            "rectangularity", "long_m", "short_m", "aspect", "orient_deg", "vertices",
            "street", "setback_m", "to_street_deg", "listed_grade", "list_entry",
            "list_name", "uprn_count", "uprns", "lat", "lon"]
    with open(os.path.join(data_dir, "buildings.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(blds)
    to_wgs = Transformer.from_crs(27700, 4326, always_xy=True)
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "geometry": json.loads(json.dumps(transform(to_wgs.transform, b["poly"]).__geo_interface__)),
        "properties": {k: b.get(k) for k in cols if k not in ("lat", "lon")} | {"tags": b["tags"]},
    } for b in blds]}
    with open(os.path.join(data_dir, "buildings.geojson"), "w", encoding="utf-8") as f:
        json.dump(fc, f)

    from collections import Counter
    print(f"buildings: {len(blds)}")
    print("forms:", dict(Counter(b["form"] for b in blds)))
    print("listed:", sum(1 for b in blds if b.get("listed_grade")))
    print("with street:", sum(1 for b in blds if b.get("street")))
    if any("uprn_count" in b for b in blds):
        print("UPRNs per building:", dict(sorted(Counter(min(b["uprn_count"], 5) for b in blds).items())))
    check_against_osm(blds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
