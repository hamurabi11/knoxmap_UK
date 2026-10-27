"""HM Land Registry's INSPIRE titles as evidence for the garden plots (PLAN K,
"Plots from INSPIRE as evidence"; LOG 2026-10-16 #4). Used by plots2.py
--inspire-attrs: plots2 makes every plot itself; a building's title says
which garage or shed is its home's, how far its garden reaches, whether it
reaches a road, and where its drive is. (The titles as the plots' own
geometry was tried and not taken forward: LOG 2026-10-16 #2-#3, git 027322b.)

clean() makes INSPIRE a coverage (its titles overlap and leave hairline
gaps); attributes() reads each building's title in real tiles (BNG through
the map's Projector, 1 m a tile).
"""
from __future__ import annotations

import json
import math
from collections import Counter

import shapely
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union

GAP_M = 0.6           # INSPIRE's gaps narrower than twice this go to a neighbour
ROAD_HIGHWAYS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
                 "living_street", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link",
                 "tertiary_link", "road"}
# In real tiles (metres on a map of 1 m a tile, as every map is so far).
ROAD_TOUCH = 8.0      # a title this near a road's centreline reaches the road
REACH_MAX = 80.0      # a ray from a house this long at most
DRIVE_IN = 3.0        # a road running at least this far into a title and ending there (at no junction) is its drive
DRIVE_MAX = 40.0      # ...as is a service road running through it no further than this


def frame_to_game(x, y, fr):
    """The inverse of plotlib.frame_to_real: a real-tile point into the game."""
    t, dx, dy, gx, gy = fr
    a = math.radians(t)
    ux, uy = x + dx - gx, y + dy - gy
    return gx + ux * math.cos(a) - uy * math.sin(a), gy + ux * math.sin(a) + uy * math.cos(a)


def biggest(g):
    if g.is_empty or g.geom_type == "Polygon":
        return g
    parts = [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
    return max(parts, key=lambda p: p.area) if parts else Polygon()


def areal(g):
    """The polygon parts of g (a cut can leave lines and points)."""
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and not p.is_empty]
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)


def poly_map(poly, f):
    poly = biggest(poly)
    if poly.is_empty:
        return poly
    out = Polygon([f(x, y) for x, y in poly.exterior.coords], [[f(x, y) for x, y in h.coords] for h in poly.interiors])
    return out if out.is_valid else biggest(out.buffer(0))


def _safe_area(a, b):
    try:
        return a.intersection(b).area
    except shapely.errors.GEOSException:
        return a.buffer(0).intersection(b.buffer(0)).area


def faces_to(faces, polys):
    """Each face to the poly covering most of it (over half): {poly index: [faces]}, and the faces left."""
    tree = shapely.STRtree([g if not g.is_empty else Point(-1e9, -1e9) for g in polys])
    got, left = {}, []
    for f in faces:
        hit = [(_safe_area(polys[int(i)], f), int(i)) for i in tree.query(f) if not polys[int(i)].is_empty]
        hit = [h for h in hit if h[0] > 0.5 * f.area]
        if hit:
            got.setdefault(max(hit)[1], []).append(f)
        else:
            left.append(f)
    return got, left


def to_neighbour(pieces, polys, min_edge=0.0, exclude=None):
    """Each piece to the poly sharing most of its edge (at least min_edge;
    not exclude[n] for piece n, if given): the area given."""
    tree = shapely.STRtree([g if not g.is_empty else Point(-1e9, -1e9) for g in polys])
    given = 0.0
    for n_, q in enumerate(pieces):
        if q.is_empty:
            continue
        rim = q.buffer(0.05)
        best, bi = 0.0, None
        for j in tree.query(rim):
            j = int(j)
            if polys[j].is_empty or (exclude is not None and exclude[n_] == j):
                continue
            ln = polys[j].boundary.intersection(rim).length
            if ln > best:
                best, bi = ln, j
        if bi is not None and best > min_edge:
            polys[bi] = areal(polys[bi].union(q))
            given += q.area
    return given


def load_titles(path, to_real, view):
    """The titles touching the map, in real tiles: [(inspire id, polygon)]."""
    out = []
    for ft in json.load(open(path, encoding="utf-8"))["features"]:
        g = shape(ft["geometry"])
        for part in getattr(g, "geoms", [g]):
            q = biggest(poly_map(part, to_real).intersection(view))
            if not q.is_empty and q.area >= 1.0:
                out.append((ft["properties"]["inspire_id"], q))
    return out


def clean(polys):
    """INSPIRE as a coverage, in place: overlaps to the smaller title, gaps
    narrower than 2 GAP_M and holes under 4 m2 to the neighbour along most of
    them, then the titles rebuilt from one noded network on a 5 cm grid (so
    near-coincident neighbours' edges become one line). Returns stats."""
    before = sum(g.area for g in polys) - shapely.union_all(polys).area
    tree = shapely.STRtree(polys)
    done = {}
    for i in sorted(range(len(polys)), key=lambda i: polys[i].area):
        g = polys[i]
        prior = [done[int(j)] for j in tree.query(g) if int(j) in done]
        if prior:
            g = areal(g.difference(shapely.union_all(prior)))       # every part: a strip across a title cuts it in two
        done[i] = shapely.set_precision(g, 1e-3) if not g.is_empty else g
    polys[:] = [done[i] for i in range(len(polys))]
    union = shapely.union_all([g for g in polys if not g.is_empty])
    gaps = union.buffer(GAP_M, join_style=2).buffer(-GAP_M, join_style=2).difference(union)
    holes = [Polygon(h) for g in getattr(union, "geoms", [union]) for h in g.interiors
             if Polygon(h).area < 4.0 and not Polygon(h).intersects(gaps)]          # not counted twice
    pieces = [g for g in getattr(gaps, "geoms", [gaps]) if not g.is_empty and g.area > 1e-4] + holes
    filled = to_neighbour(pieces, polys)
    net = unary_union(shapely.set_precision(shapely.union_all([g.boundary for g in polys if not g.is_empty]), 0.05))
    faces = [shapely.make_valid(f) for f in shapely.polygonize([net]).geoms]
    got, left = faces_to(faces, polys)
    hair = [f for f in left if f.area < 1.0]
    rebuilt = [areal(shapely.union_all(got[i])) if i in got else Polygon() for i in range(len(polys))]
    to_neighbour(hair, rebuilt)
    polys[:] = rebuilt
    return {"overlap_m2": round(before, 0), "gaps_filled_m2": round(filled, 0)}


def real_ways(map_dir, info, proj):
    """OSM's roads where they really are (not straightened), in real tiles:
    [(tags, line)]."""
    import os
    from generator import osm
    feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or []
    return [(f.tags, LineString([proj.to_px(lat, lon) for lat, lon in f.geometry])) for f in feats
            if f.kind == "way" and f.tags.get("highway") in ROAD_HIGHWAYS and len(f.geometry) >= 2]


def _key(p):
    return (round(p[0], 2), round(p[1], 2))


def drives_in(title, ways, nodes):
    """The roads that run into a title and stop there - a dead end inside
    it, not the end of a way split at a junction (`nodes`: how many times
    each point is a vertex of any road) - or a service road running only a
    short way through it: its drives, as the lines inside it (real tiles)."""
    out = []
    for tags, ln in ways:
        if not ln.intersects(title):
            continue
        part = ln.intersection(title)
        if part.length < DRIVE_IN:
            continue
        ends_in = any(title.contains(Point(e)) and nodes[_key(e)] == 1 for e in (ln.coords[0], ln.coords[-1]))
        if not ends_in and (tags.get("highway") != "service" or part.length > DRIVE_MAX):
            continue
        for seg in getattr(part, "geoms", [part]):
            if seg.geom_type == "LineString" and seg.length >= DRIVE_IN:
                out.append(seg)
    return out


def _ray(poly, c, d):
    """How far from c along unit d before leaving poly (0 if c is outside)."""
    ln = LineString([c, (c[0] + d[0] * REACH_MAX, c[1] + d[1] * REACH_MAX)])
    hit = poly.intersection(ln)
    for part in getattr(hit, "geoms", [hit]):
        if part.geom_type == "LineString" and Point(c).distance(part) < 1e-6:
            return part.length
    return 0.0


def attributes(path, proj, blds, ways, to_real):
    """What each building's title says, from INSPIRE in real tiles. blds:
    [{"real": polygon (real tiles), "home": bool, "prop": property key,
    "centre": game point, "frame": lidar_frame}]; ways: real_ways();
    to_real(x, y, frame): a game point back to the real map. Returns
    {building index: {"title", "area", "homes" (properties with a home in
    the title), "road" (the title reaches a road, or has a drive), "reach"
    {"N","E","S","W"} (tiles from the house's middle to the title's edge,
    along the game's directions), "owner" (for a building with no home: the
    home sharing its title, if it is the title's only one), "drives" (roads
    running into the title and stopping there, real tiles; on the buildings
    of the title's one home property only)}}."""
    view = shapely.box(-50, -50, proj.width + 50, proj.height + 50)

    def bng_to_real(x, y):
        from buildings.blocks import TO_WGS
        lon, lat = TO_WGS(x, y)
        return proj.to_px(lat, lon)
    titles = load_titles(path, bng_to_real, view)
    polys = [t[1] for t in titles]
    clean(polys)
    tree = shapely.STRtree([g if not g.is_empty else Point(-1e9, -1e9) for g in polys])
    roads = shapely.union_all([ln for _t, ln in ways]) if ways else LineString()
    shapely.prepare(roads)
    nodes = Counter(_key(c) for _t, ln in ways for c in ln.coords)
    members = {}
    for j, b in enumerate(blds):
        if b.get("real") is None or b["real"].is_empty or not b.get("frame"):
            continue
        c = b["real"].representative_point()
        hit = [int(i) for i in tree.query(c) if polys[int(i)].contains(c)]
        if hit:
            members.setdefault(hit[0], []).append(j)
    out = {}
    game_dirs = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
    for t, js in members.items():
        g = polys[t]
        home_props = {blds[j]["prop"] for j in js if blds[j]["home"]}
        touches = bool(g.distance(roads) <= ROAD_TOUCH) if not roads.is_empty else True
        owner, drives = None, []
        if len(home_props) == 1:
            prop = next(iter(home_props))
            owner = max((j for j in js if blds[j]["home"] and blds[j]["prop"] == prop),
                        key=lambda j: blds[j]["real"].area)
            drives = drives_in(g, ways, nodes)
        for j in js:
            b = blds[j]
            gx, gy = b["centre"]
            c0 = to_real(gx, gy, b["frame"])
            c = c0
            if not g.contains(Point(c)):                # an L-shaped house: its middle outside its title
                p_ = b["real"].representative_point()
                c = (p_.x, p_.y)
            reach = {}
            for name, (dx, dy) in game_dirs.items():
                e = to_real(gx + dx, gy + dy, b["frame"])
                d = (e[0] - c0[0], e[1] - c0[1])        # the game's direction, in real tiles
                L = math.hypot(*d) or 1.0
                reach[name] = round(_ray(g, c, (d[0] / L, d[1] / L)), 1)
            mine = owner is not None and b["home"] and b["prop"] == blds[owner]["prop"]
            out[j] = {"title": titles[t][0], "area": round(g.area, 0), "homes": len(home_props),
                      "road": touches or bool(drives), "reach": reach,
                      "owner": owner if (owner is not None and not b["home"]) else None,
                      "drives": drives if mine else []}
    return out
