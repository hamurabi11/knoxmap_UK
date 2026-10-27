"""OS Open Roads against the OSM graph (new_road_approach.md, build order 3):
each OSM link matched to the OS road links along it, for the class, the name
and road number, and a check that neither has roads the other lacks.

    python osroads.py <knoxmap map dir> <area data dir> [<oproad_gb.gpkg>]

extracts the OS road links round the map from the national GeoPackage (data/
raw/oproad/Data/oproad_gb.gpkg, unzipped from oproad_gpkg_gb.zip) into
<area>/os_roads.geojson (British National Grid, metres), once.

Matching: every SAMPLE_M along an OSM link, the nearest OS link within
MATCH_M whose bearing there is within MATCH_DEG (either way); a link's match
is the OS road covering most of its samples, and how much of it is covered.
OS's road function sets the class where the two agree on the road and
either calls it a minor road or more (OS is surveyed; OSM's tags are
whoever drew it). Below that OSM's own residential / service split is kept:
OS calls many estate closes "restricted local access" (unadopted), which
are streets all the same; a track OS calls a road becomes a service road.
An OSM link with less than
COVERED of it on an OS road is OSM's alone (a private drive, a car park
aisle, a track), and an OS road with less than COVERED of it under OSM's is
missing from the map.

Contains OS data (c) Crown copyright and database rights 2026 (OS OpenData
licence, http://os.uk/opendata/licence).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from roadgraph.network import CLASSES, RANK, WIDTH_RANGE  # noqa: E402
from roadgraph.widths import TO_LATLON, to_bng  # noqa: E402
from area import area_conf  # noqa: E402

GPKG = os.path.join(HERE, "..", "..", "data", "raw", "oproad", "Data", "oproad_gb.gpkg")
MARGIN_M = 300
SAMPLE_M = 5.0
MATCH_M = 10.0            # OSM's line and OS's this far apart are one road
MATCH_DEG = 30.0
COVERED = 0.5
INSIDE = 0.8
ATTRIBUTION = "Contains OS data (c) Crown copyright and database rights 2026"

# OS road function -> our class (form of way Track -> track).
FUNCTION = {"Motorway": "motorway", "A Road": "a_road", "B Road": "b_road", "Minor Road": "minor",
            "Local Road": "local", "Local Access Road": "local",
            "Restricted Local Access Road": "service", "Secondary Access Road": "service"}
KEEP = ("id", "road_classification", "road_function", "form_of_way", "road_classification_number",
        "name_1", "road_structure", "primary_route", "trunk_road")


def _wkb(blob: bytes) -> bytes:
    """The WKB in a GeoPackage geometry blob (after its header and envelope)."""
    flags = blob[3]
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    return blob[8 + envelope:]


def extract(gpkg: str, box, path: str) -> int:
    """OS road links meeting box (E0, N0, E1, N1) into a GeoJSON at path."""
    e0, n0, e1, n1 = box
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    rows = con.execute(
        f"SELECT geometry, {', '.join(KEEP)} FROM road_link WHERE fid IN "
        "(SELECT id FROM rtree_road_link_geometry WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?)",
        (e0, e1, n0, n1)).fetchall()
    feats = []
    for geom, *vals in rows:
        line = shapely.from_wkb(_wkb(geom))
        feats.append({"type": "Feature", "properties": dict(zip(KEEP, vals)),
                      "geometry": {"type": "LineString", "coordinates": [[round(x, 2), round(y, 2)]
                                                                        for x, y in line.coords]}})
    json.dump({"type": "FeatureCollection", "name": "OS Open Roads, road links", "crs": "EPSG:27700",
               "attribution": ATTRIBUTION, "box": list(box), "features": feats}, open(path, "w"))
    return len(feats)


def load(area: str) -> list:
    """[(LineString in BNG metres, properties)] from the area's extract; none at
    the osm_lidar data level or with no extract (area_conf): conflate keeps OSM's
    classes and names, and every link is OSM's alone."""
    path = area_conf.extra(area, "os_roads.geojson")
    if path is None:
        return []
    gj = json.load(open(path, encoding="utf-8"))
    return [(shapely.LineString(f["geometry"]["coordinates"]), f["properties"]) for f in gj["features"]]


def os_class(props: dict) -> str | None:
    if props.get("form_of_way") == "Track":
        return "track"
    return FUNCTION.get(props.get("road_function"))


def _bearing(line, d: float) -> float:
    """Bearing (degrees, 0-180: a road has no direction) at d along line."""
    a = line.interpolate(max(0.0, d - 2.0))
    b = line.interpolate(min(line.length, d + 2.0))
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x)) % 180


def _off(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def match(osm_lines: list, os_roads: list) -> list:
    """Per OSM line (BNG): (index of the OS road covering most of it or None,
    share of it covered by any OS road, share by that one)."""
    lines = [ln for ln, _p in os_roads]
    tree = shapely.STRtree(lines)
    out = []
    for line in osm_lines:
        ds = np.arange(SAMPLE_M / 2, max(line.length, SAMPLE_M), SAMPLE_M)
        votes: dict = {}
        hit = 0
        for d in ds:
            p = line.interpolate(d)
            bearing = _bearing(line, d)
            best = None
            for k in tree.query(p, predicate="dwithin", distance=MATCH_M):
                ok = lines[k]
                along = ok.project(p)
                if _off(bearing, _bearing(ok, along)) > MATCH_DEG:
                    continue
                dist = ok.distance(p)
                if best is None or dist < best[0]:
                    best = (dist, int(k))
            if best:
                hit += 1
                votes[best[1]] = votes.get(best[1], 0) + 1
        top = max(votes, key=votes.get) if votes else None
        out.append((top, hit / len(ds), (votes[top] / len(ds)) if top is not None else 0.0))
    return out


def conflate(links, proj, os_roads: list, box=None, log=print) -> dict:
    """Each OSM link's OS match; the class set from OS where matched (the
    width moved into the new class's range, or to its default if it was the
    old class's default). A report: matches, class changes, names that
    differ, OSM's roads OS lacks, OS's roads the map lacks (those with
    INSIDE of their length in box, x0, y0, x1, y1 in tiles: the links are
    all those reaching into it)."""
    if not os_roads:
        log(f"OS Open Roads: none for this area, OSM's classes kept ({len(links)} links)")
        return {"links": len(links), "matched": 0, "class changed": [], "names differ": [], "os only": [],
                "osm only": [{"link": i, "class": l.cls, "name": l.tags.get("name"), "covered": 0.0,
                              "ways": l.ways[:3]} for i, l in enumerate(links)], "no os": True}
    bng = [shapely.LineString(to_bng(proj, l.pts)) for l in links]
    found = match(bng, os_roads)
    report = {"links": len(links), "matched": 0, "class changed": [], "names differ": [], "osm only": [],
              "os only": []}
    for i, (l, (k, covered, _share)) in enumerate(zip(links, found)):
        if k is None or covered < COVERED:
            report["osm only"].append({"link": i, "class": l.cls, "name": l.tags.get("name"),
                                       "covered": round(covered, 2), "ways": l.ways[:3]})
            continue
        report["matched"] += 1
        props = os_roads[k][1]
        l.tags["os"] = {key: props.get(key) for key in ("road_function", "road_classification_number",
                                                       "name_1", "form_of_way")}
        name = l.tags.get("name")
        if props.get("name_1") and name and props["name_1"].lower() != name.lower():
            report["names differ"].append({"link": i, "osm": name, "os": props["name_1"]})
        cls = _class_from(os_class(props), l.cls)
        if cls != l.cls:
            report["class changed"].append({"link": i, "name": name or props.get("name_1"), "osm": l.cls, "os": cls,
                                            "os function": props.get("road_function"),
                                            "number": props.get("road_classification_number")})
            highway = (l.tags.get("highway") or "").lower()
            was_default = CLASSES.get(highway, (None, None))[1] == l.width
            lo, hi = WIDTH_RANGE[cls]
            l.width = _default_width(cls) if was_default else min(hi, max(lo, l.width))
            l.cls = cls
    # OS roads under no OSM link, of those inside the box.
    osm_tree = shapely.STRtree(bng)
    for ln, props in os_roads:
        ds = np.arange(SAMPLE_M / 2, max(ln.length, SAMPLE_M), SAMPLE_M)
        pts = [ln.interpolate(d) for d in ds]
        if box is not None:
            px = [proj.to_px(*TO_LATLON.transform(p.x, p.y)[::-1]) for p in pts]
            if np.mean([box[0] <= x < box[2] and box[1] <= y < box[3] for x, y in px]) < INSIDE:
                continue
        hit = sum(1 for p in pts if len(osm_tree.query(p, predicate="dwithin", distance=MATCH_M)))
        if hit / len(ds) < COVERED:
            report["os only"].append({"id": props["id"], "function": props.get("road_function"),
                                      "form": props.get("form_of_way"), "name": props.get("name_1"),
                                      "length_m": round(ln.length), "covered": round(hit / len(ds), 2)})
    log(f"OS Open Roads: {report['matched']} of {len(links)} links matched, "
        f"{len(report['class changed'])} classes changed, {len(report['osm only'])} OSM's alone, "
        f"{len(report['os only'])} OS roads missing from OSM")
    return report


def _class_from(os_cls: str | None, osm_cls: str) -> str:
    """The class to use, given OS's and OSM's."""
    minor = RANK["minor"]
    if os_cls is None:
        return osm_cls
    if RANK[os_cls] <= minor or RANK[osm_cls] <= minor:
        return os_cls
    if osm_cls == "track" and os_cls != "track":
        return "service"
    return osm_cls


def _default_width(cls: str) -> int:
    return next(w for c, w in CLASSES.values() if c == cls)


def main() -> int:
    from roadgraph import network
    map_dir, area = sys.argv[1], sys.argv[2]
    gpkg = sys.argv[3] if len(sys.argv) > 3 else GPKG
    _info, proj = network.projector(map_dir)
    corners = to_bng(proj, [(0, 0), (proj.width, 0), (0, proj.height), (proj.width, proj.height)])
    es, ns = zip(*corners)
    box = (int(min(es)) - MARGIN_M, int(min(ns)) - MARGIN_M, int(max(es)) + MARGIN_M, int(max(ns)) + MARGIN_M)
    n = extract(gpkg, box, os.path.join(area, "os_roads.geojson"))
    print(f"{n} OS road links in {box} -> {os.path.join(area, 'os_roads.geojson')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
