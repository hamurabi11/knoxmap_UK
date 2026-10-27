"""The road graph from OpenStreetMap, in a map's tiles: junction-to-junction
links, each with a class and a width in whole tiles (new_road_approach.md,
"Prepare the inputs" 2).

The map's frame is KnoxMap's (its Projector: the map turned so the High
Street runs up-right), so what comes out lines up with the map as built.
Footways and paths are left for their own pass; tunnels are left out (the
ground is flat). A way's ends and every node it shares with another way are
graph nodes; a node where just two ways of one class meet end to end is
merged away, so a link runs junction to junction. Then generalised: a small
turning loop is a dead end, a dual carriageway one wide road, a lane hugging
a bigger road part of it.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.ops import substring

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator import osm  # noqa: E402
from generator.renderer import Projector  # noqa: E402

# OSM highway -> (class, default width in tiles; whole tiles, odd ones too as
# vanilla's streets.xml has them - 5 and 7 wide are common - raster.py draws
# an odd-width road half a tile over so its edges are tile edges). The proposal's
# table: A road 8, B / minor / residential 6, single-track lane 4, track 2.
CLASSES = {
    "motorway": ("motorway", 14), "motorway_link": ("motorway", 8),
    "trunk": ("a_road", 8), "trunk_link": ("a_road", 6),
    "primary": ("a_road", 8), "primary_link": ("a_road", 6),
    "secondary": ("b_road", 6), "secondary_link": ("b_road", 6),
    "tertiary": ("minor", 6), "tertiary_link": ("minor", 6),
    "unclassified": ("minor", 6), "residential": ("local", 6), "living_street": ("local", 6),
    "road": ("local", 6), "service": ("service", 4), "track": ("track", 2),
}
# Routed highest first (snap.py): the main roads get the straightest runs.
RANK = {"motorway": 0, "a_road": 1, "b_road": 2, "minor": 3, "local": 4, "service": 5, "track": 6}
LOOP_MAX = 80            # tiles round: a loop shorter than this is a turning loop
DUAL_MAX = 16            # tiles apart at most, the two ways of a dual carriageway
ABSORB_GAP = 2           # tiles: a lane no further than this from a bigger road's edge is part of it
ATTACH_MAX = 10          # tiles a junction of an absorbed lane may move to a bigger road's
WIDTH_RANGE = {"motorway": (10, 16), "a_road": (6, 10), "b_road": (6, 8), "minor": (4, 10),
               "local": (4, 8), "service": (2, 6), "track": (2, 4)}


@dataclass
class Link:
    u: int                       # node ids at its ends
    v: int
    pts: list                    # polyline in tiles, u to v
    cls: str
    width: int                   # tiles, whole
    tags: dict = field(default_factory=dict)
    ways: list = field(default_factory=list)


def _key(lat, lon):
    return round(lat * 1e7), round(lon * 1e7)


def projector(map_dir: str):
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    b = info["bbox"]
    return info, Projector.build(b["south"], b["west"], b["north"], b["east"],
                                 info["meters_per_tile"], info.get("rotation") or 0.0)


def whole(w: float) -> int:
    return max(2, int(round(w)))


def build(map_dir: str) -> tuple[dict, list[Link], object, dict]:
    """(nodes {id: (x, y) in tiles}, links, projector, pins) for a map's OSM
    download, the roads as they really are (not straightened), cut at the
    map's edge. Pins {node: the way into the map, (dx, dy)} are the nodes
    where a road leaves the map: worked out from the real road alone, so a
    neighbouring map built in the same frame finds the same point and the
    roads meet."""
    info, proj = projector(map_dir)
    ways = road_ways(osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or [])
    if info.get("shape"):
        ways = _in_shape(ways, info["shape"], proj)
    # Graph nodes: way ends, and nodes shared by more than one way.
    uses: dict = {}
    for f in ways:
        for n, (lat, lon) in enumerate(f.geometry):
            k = _key(lat, lon)
            uses[k] = uses.get(k, 0) + (2 if n in (0, len(f.geometry) - 1) else 1)
    node_of: dict = {}
    nodes: dict = {}

    def node(k, lat, lon):
        if k not in node_of:
            node_of[k] = len(node_of)
            nodes[node_of[k]] = proj.to_px(lat, lon)
        return node_of[k]
    links: list[Link] = []
    for f in ways:
        cls, w = CLASSES[f.tags["highway"].lower()]
        tagged = _width_tag(f.tags)
        lo, hi = WIDTH_RANGE[cls]
        width = min(hi, max(lo, whole(tagged))) if tagged else w
        start = 0
        geo = f.geometry
        for n in range(1, len(geo)):
            k = _key(*geo[n])
            if n == len(geo) - 1 or uses[k] >= 3:
                a, b = geo[start], geo[n]
                links.append(Link(node(_key(*a), *a), node(k, *b), [proj.to_px(la, lo) for la, lo in geo[start:n + 1]],
                                  cls, width, dict(f.tags), [f.osm_id]))
                start = n
    links = _merge_through(links)
    # A loop drawn as one way back to its own junction: a turning loop if
    # short (dropped: the road ends where it starts), else two links split at
    # its middle by a new node.
    links = [l for l in links if l.u != l.v or _length(l.pts) >= LOOP_MAX]
    for k, l in enumerate(links):
        if l.u == l.v and len(l.pts) > 2:
            half = _length(l.pts) / 2
            m = next(n for n in range(1, len(l.pts)) if _length(l.pts[:n + 1]) >= half)
            mid = len(nodes) + 1_000_000
            nodes[mid] = l.pts[m]
            links[k] = Link(l.u, mid, l.pts[:m + 1], l.cls, l.width, l.tags, l.ways)
            links.append(Link(mid, l.v, l.pts[m:], l.cls, l.width, l.tags, l.ways))
    links = _merge_through(_absorb_lanes(_dual_carriageways(links)))
    links, pins = _clip(links, nodes, proj.width, proj.height)
    # Nodes no link uses any more (merged away) go.
    used = {l.u for l in links} | {l.v for l in links}
    nodes = {k: v for k, v in nodes.items() if k in used}
    return nodes, links, proj, pins


# Sites whose service ways are their own car parks and drives, not streets
# (a school's one-way aisles beside West Street, Robert May's).
SITE_AMENITIES = {"parking", "school", "college", "university", "hospital", "clinic", "kindergarten",
                  "community_centre", "place_of_worship", "fuel", "supermarket"}
SITE_LANDUSE = {"retail", "commercial", "industrial"}
SITE_SHARE = 0.6             # of a service way's length inside a site for it to be the site's
CAR_PARK_SHARE = 0.5         # ...and of any minor road's inside a car park for it to be the car park's
MINOR_IN_CAR_PARK = ("unclassified", "residential", "living_street")


def car_park_ways(feats) -> list:
    """The ways that are a car park or a site's own drives, not streets:
    service ways tagged a parking aisle, or mostly (SITE_SHARE) inside a car
    park, or private or one-way and mostly inside a site (a school, a
    hospital, a retail park); and any minor road (MINOR_IN_CAR_PARK) mostly
    (CAR_PARK_SHARE) inside a car park, whatever its highway tag - King
    Street's customer car park is a loop mapped highway=unclassified, which
    was routed as a 6-wide street with pavements over the car park (user,
    2026-10-11). Drawn as a car park (handover.py), not routed."""
    sites = []
    for f in feats:
        t = f.tags
        if (t.get("amenity") in SITE_AMENITIES or t.get("landuse") in SITE_LANDUSE) and f.kind == "way" \
                and len(f.geometry) > 3 and f.geometry[0] == f.geometry[-1]:
            try:
                poly = shapely.Polygon([(lo, la) for la, lo in f.geometry])
            except Exception:
                continue
            if poly.is_valid and poly.area > 0:
                sites.append((poly, t.get("amenity") == "parking"))
    tree = shapely.STRtree([p for p, _c in sites]) if sites else None
    out = []
    for f in feats:
        t = f.tags
        hw = (t.get("highway") or "").lower()
        if f.kind != "way" or len(f.geometry) < 2 or hw not in ("service",) + MINOR_IN_CAR_PARK:
            continue
        if hw != "service":
            if tree is not None:
                line = shapely.LineString([(lo, la) for la, lo in f.geometry])
                if any(sites[k][1] and line.intersection(sites[k][0]).length >= CAR_PARK_SHARE * line.length
                       for k in tree.query(line)):
                    out.append(f)
            continue
        if t.get("service") == "parking_aisle":
            out.append(f)
            continue
        if tree is None or t.get("service") in ("driveway", "alley"):
            continue
        line = shapely.LineString([(lo, la) for la, lo in f.geometry])
        private = t.get("access") in ("private", "customers", "permissive", "destination", "no") or t.get("oneway") == "yes"
        for k in tree.query(line):
            poly, is_car_park = sites[k]
            if (is_car_park or private) and line.intersection(poly).length >= SITE_SHARE * line.length:
                out.append(f)
                break
    return out


def road_ways(feats) -> list:
    """The OSM ways the graph is made of: roads of a class it knows, not areas,
    tunnels, or car parks and sites' own drives (car_park_ways)."""
    parks = {id(f) for f in car_park_ways(feats)}
    out = []
    for f in feats:
        if id(f) in parks:
            continue
        h = (f.tags.get("highway") or "").lower()
        if f.kind != "way" or h not in CLASSES or len(f.geometry) < 2:
            continue
        if f.tags.get("area") == "yes" or f.tags.get("tunnel") in ("yes", "culvert"):
            continue
        out.append(f)
    return out


def _in_shape(ways: list, shape: dict, proj) -> list:
    """A map cut to a drawn outline (new_map.py --shape; Westminster,
    2026-10-21) keeps its main roads whole, as Generate does
    (rendering/shape.THROUGH_ROADS: the town is not an island), and of every
    other road only the part inside the outline; outside, Generate's ground
    is grass."""
    from dataclasses import replace

    from generator.renderer import classify
    from generator.rendering.projection import shape_px
    from generator.rendering.shape import THROUGH_ROADS
    clip = shape_px(shape, proj)
    out = []
    for f in ways:
        if classify(f.tags, False) in THROUGH_ROADS:
            out.append(f)
            continue
        px = [proj.to_px(la, lo) for la, lo in f.geometry]
        line = shapely.LineString(px)
        if clip.contains(line):
            out.append(f)
            continue
        # The way's own nodes keep their exact lat/lon (junctions are matched
        # on them); only the new end at the outline is worked out.
        real = {(round(x, 6), round(y, 6)): ll for (x, y), ll in zip(px, f.geometry)}
        part = line.intersection(clip)
        for piece in getattr(part, "geoms", [part]):
            if piece.geom_type == "LineString" and piece.length >= 1:
                geo = [real.get((round(x, 6), round(y, 6))) or tuple(proj.to_latlon(x, y)) for x, y in piece.coords]
                out.append(replace(f, geometry=geo))
    return out


def _clip(links: list[Link], nodes: dict, W: int, H: int) -> tuple[list[Link], dict]:
    """Links cut to the map (0..W, 0..H); a new node, a pin, where one
    crosses the edge. A road leaving and coming back is two pieces."""
    frame = shapely.box(0, 0, W, H)
    out, pins = [], {}

    def end_node(p, own, own_pos):
        if math.dist(p, own_pos) < 1e-6:
            return own
        nid = len(nodes) + 2_000_000
        nodes[nid] = p
        x, y = p
        pins[nid] = (0, 1) if y < 1e-6 else (0, -1) if y > H - 1e-6 else (1, 0) if x < 1e-6 else (-1, 0)
        return nid
    for l in links:
        line = shapely.LineString(l.pts)
        if frame.contains(line):
            out.append(l)
            continue
        part = line.intersection(frame)
        for piece in getattr(part, "geoms", [part]):
            if piece.is_empty or piece.geom_type != "LineString" or piece.length < 1:
                continue
            pts = [tuple(c) for c in piece.coords]
            if line.project(shapely.Point(pts[0])) > line.project(shapely.Point(pts[-1])):
                pts = pts[::-1]                     # the piece the link's way round
            u = end_node(pts[0], l.u, l.pts[0])
            v = end_node(pts[-1], l.v, l.pts[-1])
            out.append(Link(u, v, pts, l.cls, l.width, l.tags, l.ways))
    return out, pins


def _samples(pts, every=2.0):
    line = shapely.LineString(pts)
    n = max(2, int(line.length / every) + 1)
    return shapely.line_interpolate_point(line, np.linspace(0, line.length, n))


def _dual_carriageways(links: list[Link]) -> list[Link]:
    """A dual carriageway (two one-way ways of a class side by side, running
    opposite ways) made one road: the centreline between them, as wide as
    both and the reservation. The second way's end junctions become the
    first's."""
    alias: dict = {}
    gone = set()
    for i, a in enumerate(links):
        if i in gone or a.tags.get("oneway") != "yes":
            continue
        la = shapely.LineString(a.pts)
        for j, b in enumerate(links):
            if j <= i or j in gone or b.tags.get("oneway") != "yes" or b.cls != a.cls:
                continue
            lb = shapely.LineString(b.pts)
            da, db = shapely.distance(_samples(a.pts), lb), shapely.distance(_samples(b.pts), la)
            heading_a = np.subtract(a.pts[-1], a.pts[0])
            heading_b = np.subtract(b.pts[-1], b.pts[0])
            if np.quantile(da, 0.8) > DUAL_MAX or np.quantile(db, 0.8) > DUAL_MAX or heading_a @ heading_b >= 0:
                continue
            sep = float(np.median(np.concatenate([da, db])))
            mid = [((x + p.x) / 2, (y + p.y) / 2) for (x, y), p in
                   zip(a.pts, shapely.get_point(shapely.shortest_line(shapely.points(a.pts), lb), 1))]
            links[i] = a = Link(a.u, a.v, mid, a.cls, min(16, whole(a.width + sep)),
                                dict(a.tags, dual_carriageway="yes"), a.ways + b.ways)
            la = shapely.LineString(a.pts)
            for old, new in ((b.v, a.u), (b.u, a.v)):
                if old != new:
                    alias[old] = new
            gone.add(j)
    out = []
    for k, l in enumerate(links):
        if k not in gone:
            l.u, l.v = alias.get(l.u, l.u), alias.get(l.v, l.v)
            if l.u != l.v:
                out.append(l)
    return out


def _absorb_lanes(links: list[Link]) -> list[Link]:
    """A service lane or track that never leaves the reach of bigger roads (a
    parking strip along the High Street) is part of them: dropped, one at a
    time. A junction of it other roads still use moves to the nearest
    junction of a bigger road (within ATTACH_MAX), or it stays.

    Central London (2026-10-21): each sample is measured only against roads
    within reach (an index a pass), and a lane that failed is not tried
    again until something that could change its answer has: dropping a road
    only takes reach away, so only a lane beside the one dropped (one arm
    fewer at a junction), a lane whose line moved, or one ranked below a
    road that moved, is tried again - in the same order, so the same lanes
    go as when every lane was tried after every drop."""
    failed: set = set()                    # id() of lanes that cannot go yet
    line_of: dict = {}                     # id(link) -> (its pts, its line)

    def line(m):
        got = line_of.get(id(m))
        if got is None or got[0] is not m.pts:
            got = line_of[id(m)] = (m.pts, shapely.LineString(m.pts))
        return got[1]
    while True:
        arms: dict = {}
        pos: dict = {}
        for l in links:
            for n, p in ((l.u, l.pts[0]), (l.v, l.pts[-1])):
                arms.setdefault(n, []).append(l)
                pos[n] = p
        all_lines = [line(m) for m in links]
        tree = shapely.STRtree(all_lines)
        widest = max((m.width for m in links), default=0.0)
        for k, l in enumerate(links):
            if RANK[l.cls] < RANK["service"] or id(l) in failed:
                continue
            big = [m for m in links if RANK[m.cls] < RANK[l.cls]]
            reach = l.width / 2 + ABSORB_GAP

            def covered(p):
                near = tree.query(p, predicate="dwithin", distance=reach + widest / 2)
                return any(RANK[links[j].cls] < RANK[l.cls]
                           and shapely.distance(p, all_lines[j]) - links[j].width / 2 <= reach for j in near)
            if not big or not all(covered(p) for p in _samples(l.pts)):
                failed.add(id(l))
                continue
            alias = {}
            for n in (l.u, l.v):
                if len(arms[n]) == 1 or any(RANK[m.cls] < RANK[l.cls] for m in arms[n]):
                    continue
                d, to = min((math.dist(pos[n], pos[b]), b) for m in big for b in (m.u, m.v))
                if d > ATTACH_MAX:
                    break
                alias[n] = to
            else:
                kept, moved = [], []
                for j, m in enumerate(links):
                    if j == k:
                        continue
                    if m.u in alias or m.v in alias:
                        moved.append(m)
                    if m.u in alias:
                        m.u, m.pts = alias[m.u], [pos[alias[m.u]]] + m.pts[1:]
                    if m.v in alias:
                        m.v, m.pts = alias[m.v], m.pts[:-1] + [pos[alias[m.v]]]
                    if m.u != m.v:
                        kept.append(m)
                links = kept
                failed -= {id(m) for n in (l.u, l.v) for m in arms[n]}
                failed -= {id(m) for m in moved}
                if moved:
                    low = min(RANK[m.cls] for m in moved)
                    failed -= {id(m) for m in links if RANK[m.cls] > low}
                line_of.pop(id(l), None)
                break
            failed.add(id(l))
        else:
            return links


def _width_tag(tags: dict) -> float | None:
    raw = tags.get("width") or tags.get("est_width")
    if not raw:
        return None
    try:
        return float(str(raw).split()[0].replace(",", ".").rstrip("m"))
    except ValueError:
        return None


def _length(pts) -> float:
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def _merge_through(links: list[Link]) -> list[Link]:
    """Links joined end to end at a node only they meet at, of one class and
    width, made one (a road drawn as several ways is one link)."""
    while True:
        at: dict = {}
        for i, l in enumerate(links):
            at.setdefault(l.u, []).append(i)
            at.setdefault(l.v, []).append(i)
        merged = False
        for nid, ids in at.items():
            if len(ids) != 2 or ids[0] == ids[1]:
                continue
            a, b = links[ids[0]], links[ids[1]]
            if (a.cls, a.width) != (b.cls, b.width):
                continue
            pa = a.pts if a.v == nid else a.pts[::-1]
            ua = a.u if a.v == nid else a.v
            pb = b.pts if b.u == nid else b.pts[::-1]
            vb = b.v if b.u == nid else b.u
            if ua == vb:
                # A loop back to where it started: a turning loop at a road's
                # end is a dead end (half of it kept); a bigger one is a real
                # loop road, left two links.
                if _length(pa + pb[1:]) < LOOP_MAX:
                    links = [l for i, l in enumerate(links) if i != ids[1]]
                    merged = True
                    break
                continue
            new = Link(ua, vb, pa + pb[1:], a.cls, a.width, a.tags, a.ways + b.ways)
            links = [l for i, l in enumerate(links) if i not in ids] + [new]
            merged = True
            break
        if not merged:
            return links


def way_lines(links, paths, real: dict) -> dict:
    """{OSM way id: (its snapped line in tiles, the widest link it is part of)}:
    each way of a routed link its stretch of the link's path, found by where
    the way's points lie along the link's real line (within the link's width,
    3 tiles at least), carried over in proportion along the path; a way over
    several links their stretches joined in the way's own order and direction.
    real: {way id: its real points in tiles}. (handover.way_lines; mode_test.py.)"""
    pieces: dict = {}
    for i, path in paths.items():
        l = links[i]
        line_r = shapely.LineString(l.pts)
        line_p = shapely.LineString(path)
        k = line_p.length / max(line_r.length, 1e-9)
        tol = max(3.0, float(l.width))
        for w in dict.fromkeys(l.ways):
            pts = real.get(w)
            if not pts:
                continue
            near = [(n, line_r.project(shapely.Point(p))) for n, p in enumerate(pts)
                    if line_r.distance(shapely.Point(p)) <= tol]
            if not near:
                continue
            s = [d for _n, d in near]
            sub = substring(line_p, min(s) * k, max(s) * k)
            coords = list(sub.coords) if sub.geom_type == "LineString" else [sub.coords[0]]
            if near[0][1] > near[-1][1]:
                coords = coords[::-1]            # the way's own direction
            pieces.setdefault(w, []).append((near[0][0], coords, l))
    out = {}
    for w, ps in pieces.items():
        ps.sort(key=lambda t: t[0])
        geo = []
        for _n, coords, _l in ps:
            for c in coords:
                if not geo or (abs(geo[-1][0] - c[0]) > 1e-6 or abs(geo[-1][1] - c[1]) > 1e-6):
                    geo.append(c)
        if len(geo) < 2:
            continue
        out[w] = (geo, max((t[2] for t in ps), key=lambda l: l.width))
    return out
