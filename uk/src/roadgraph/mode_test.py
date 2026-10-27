"""Score the road modes (modes.py) on a square of a map, every one on the same numbers
(LOG 2026-10-25 #2).

    python mode_test.py <source map dir> <area data dir> [--box x0,y0,x1,y1]
                        [--runs graph,grid,cost-mid ...] [--out DIR]
        (a run "<preset>@front": its buildings squared about their street-facing edge,
        blocks.py's FRONT_ANCHOR; the preset routed once for both)
    python mode_test.py <source map dir> <area data dir> --score-built <map dir> [<map dir> ...]
        (no routing: built maps handed over from the source map, each scored on its own
        routing and Build's footprints - the same numbers, the square only: built.md,
        built.json, built_<map>.png, built_all.png)

Routes the square's roads once per run (RUNS: a mode and its settings), as the
hand-over would, and scores each:

  frontage at line   share of each street side with a building on its frontage line
                     (frontage_measure.py's measure: within TOL tiles of the side's median
                     distance), length-weighted over the square's named streets
  flush / stepped    ...the same samples by how the front runs (LOG 2026-10-25 #3: a 45 or
                     2:1 front of houses squared one by one is a sawtooth, two-thirds of it
                     still "at line"): flush - at the line and locally smooth (the distances
                     within STEP_WIN tiles either way span no more than STEP_RANGE: a staircase
                     of single tiles is 0.5); stepped - within STEP_REACH of the line but
                     jagged; off - further; gap - no building within REACH
  flush ceiling      the share of the built frontage (samples with a building) on stretches of
                     the routed road along the grid - a building squared to the grid can be
                     flush only there - and flush as a share of it (LOG 2026-10-25 #4)
  frontage spread    the sides' interquartile range of that distance, tiles (median)
  built-up 2:1       the same, only the samples on built-up 2:1 stretches of the routed road
                     (and 45, grid), so a slope's own fronts are seen apart
  on street          building squares on the street as drawn (carriageway, bays, verge,
                     pavement; road_overlap.py's count)
                     - split into slivers along a street and wedges at a junction (within
                     the widest arm's half width + WEDGE_PAD of a junction's corner)
  centreline         mean, 95th percentile and max distance of the routed road from the
                     real centreline
  turns              per link, and mid-frontage: not at a junction, a frontage gap or a
                     plot boundary (snap.turn_ok's places, wherever there are buildings)
  length             by direction class (grid / 45 / 2:1), all and where it is built up

Buildings are not built here (no Build), but placed as blocks.py places them (its own
code, imported): each real outline moved corner by corner by the displacement Generate
would give it (the routed roads' shift samples, octilinear.displacement), then
blocks.py's steps on the square - registration, blocks carried along their street (the
routed street, blocks.load_streets' game_lines) or moved rigidly, squared to the tiles,
nudged and trimmed off the roads, long buildings laid straight, off the painted street
(checked against Build on the built Alton_town_graph: LOG 2026-10-25 #3). Writes
<out>/table.md, <out>/scores.json and pictures (<run>.png, all.png).
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import time

import numpy as np
import shapely
import shapely.affinity
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from roadgraph import modes, network, osroads, raster, roadside, snap, widths  # noqa: E402
from roadgraph.run import town_mask  # noqa: E402
from buildings import blocks as B  # noqa: E402

BOX = (560, 640, 1120, 1200)          # Alton's centre: the High Street, Market, Amery and Turk Streets
STEP, REACH, TOL = 2.0, 30.0, 1.5     # frontage_measure.py's
STEP_WIN, STEP_RANGE, STEP_REACH = 4.0, 1.5, 4.0     # flush / stepped (tiles)
FRONT_MIN = 40.0                      # tiles: streets shorter than this are not scored for frontage
FRONT_CLASSES = ("a_road", "b_road", "minor", "local")
WEDGE_PAD = 4.0                       # tiles past a junction's widest half width: its wedge
# The runs: a name -> (mode, snap constants for it) - modes.PRESETS.
RUNS = modes.PRESETS


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


class Square:
    """The square's road graph, measured once (OS classes, LiDAR widths, roadside)."""

    def __init__(self, map_dir: str, area: str, box):
        self.map_dir, self.area, self.box = map_dir, area, box
        nodes, links, proj, pins = network.build(map_dir)
        self.proj, self.H, self.W = proj, proj.height, proj.width
        x0, y0, x1, y1 = box
        keep = [i for i, l in enumerate(links) if any(x0 <= x < x1 and y0 <= y < y1 for x, y in l.pts)]
        self.links = [links[i] for i in keep]
        self.keep = keep                              # each link's index in the whole map's graph
        used = {l.u for l in self.links} | {l.v for l in self.links}
        self.nodes = {k: v for k, v in nodes.items() if k in used}
        self.pins = {n: w for n, w in pins.items() if n in self.nodes}
        # as handover.py: with no OS extract (area_conf) every link is OSM's alone
        self.os_report = osroads.conflate(self.links, proj, osroads.load(area), box, log=lambda *a: None)
        lidar = widths.Lidar(widths.box_for(proj, self.links), os.path.join(area, "lidar"))
        sides: dict = {}
        measured = widths.measure(self.links, proj, lidar, sides)
        widths.apply(self.links, measured)
        from generator import osm
        info, _p = network.projector(map_dir)
        feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or []
        ways = {f.osm_id: ([proj.to_px(la, lo) for la, lo in f.geometry], f.tags) for f in network.road_ways(feats)}
        self.chosen, _tally = roadside.decide(self.links, sides, ways, measured)
        self.hand = roadside.handedness(proj)
        self.town = town_mask(map_dir, proj, self.H, self.W)
        self.fields = modes.fields(map_dir, area, self.H, self.W)
        self.info = info
        self.real_ways = {w: pts for w, (pts, _t) in ways.items()}
        self.way_tags = {w: t for w, (_pts, t) in ways.items()}
        # The real buildings near the square, as polygons in tiles.
        m = 60
        self.buildings = []
        for f in feats:
            if f.kind != "way" or not f.tags.get("building") or not f.geometry or len(f.geometry) < 3:
                continue
            p = shapely.Polygon([proj.to_px(la, lo) for la, lo in f.geometry])
            c = p.centroid
            if p.is_valid and x0 - m <= c.x < x1 + m and y0 - m <= c.y < y1 + m:
                self.buildings.append(p)
        self.real_mask = _fill(self.buildings, self.H, self.W)
        # For blocks.py: the area's outlines near the square, its LiDAR ground, the railway,
        # the streets it carries blocks along.
        near = shapely.box(x0 - m, y0 - m, x1 + m, y1 + m)
        self.area_feats = {}
        for ft in json.load(open(os.path.join(area, "buildings.geojson"), encoding="utf-8"))["features"]:
            if ft["geometry"]["type"] != "Polygon":
                continue
            px = shapely.Polygon([proj.to_px(la, lo) for lo, la in ft["geometry"]["coordinates"][0]])
            if px.is_valid and near.contains(px.centroid):
                self.area_feats[str(ft["properties"]["osm_id"])] = ft
        corners = [proj.to_latlon(x, y) for x, y in ((x0 - m, y0 - m), (x1 + m, y0 - m), (x0 - m, y1 + m),
                                                     (x1 + m, y1 + m))]
        es, ns = zip(*[B.TO_BNG(lon, lat) for lat, lon in corners])
        with contextlib.redirect_stdout(io.StringIO()):
            self.ground = B.Ground(area, (int(min(es)) - 5, int(min(ns)) - 5, int(max(es)) + 5, int(max(ns)) + 5))
        from buildings import rail_kit
        from generator.renderer import _is_polygon, classify
        self.rail = rail_kit.RailLand(feats, area, classify, _is_polygon)
        self.names = B.streets_measured(map_dir, info)
        self.box_mask = np.zeros((self.H, self.W), bool)
        self.box_mask[y0:y1, x0:x1] = True

    def route(self, mode: str, params: dict, fields: dict | None = None) -> dict:
        log: list = []
        t0 = time.time()
        try:
            info = modes.prepare(mode, self.map_dir, self.area, self.H, self.W, fields or self.fields, params)
            paths, corner, merged, failed = snap.snap(self.nodes, self.links, self.town, log=log.append,
                                                      pins=self.pins)
            let_go = list(snap.JOG_LET_GO)
        finally:
            modes.reset()
        # Junctions: corners where three or more routed links meet, and their widest arm.
        arms: dict = {}
        for i in paths:
            for n in (self.links[i].u, self.links[i].v):
                arms.setdefault(merged[n], []).append(self.links[i].width)
        junctions = [(corner[r], max(ws)) for r, ws in arms.items() if len(ws) >= 3]
        return {"info": info, "paths": paths, "failed": failed, "let_go": let_go,
                "junctions": junctions, "seconds": time.time() - t0, "log": log}


def _fill(polys, H, W) -> np.ndarray:
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        d.polygon(list(p.exterior.coords), fill=1)
    return np.array(img, bool)


def _move(sq: Square, paths: dict):
    """Generate's displacement for this routing: move(lat, lon) -> (lat, lon)."""
    from generator import octilinear
    samples = []
    for i, path in paths.items():
        real, snapped = shapely.LineString(sq.links[i].pts), shapely.LineString(path)
        if real.length == 0 or snapped.length == 0:
            continue
        for t in np.linspace(0.0, 1.0, max(1, int(real.length // 4)) + 1):
            p, q = real.interpolate(t, normalized=True), snapped.interpolate(t, normalized=True)
            samples.append((np.array([p.x, p.y]), np.array([q.x - p.x, q.y - p.y])))
    return octilinear.displacement(samples, sq.proj)


def moved_rigid(sq: Square, move) -> list:
    """The real buildings near the square, each moved rigidly by `move` (_move's) at its middle."""
    out = []
    for p in sq.buildings:
        c = p.centroid
        nx, ny = sq.proj.to_px(*move(*sq.proj.to_latlon(c.x, c.y)))
        out.append(shapely.affinity.translate(p, nx - c.x, ny - c.y))
    return out


def placed(sq: Square, paths: dict, street: np.ndarray, passage_ways=None) -> np.ndarray:
    """Where blocks.py would put the square's buildings on this routing: its own steps,
    as blocks.main runs them (`street`: the street as painted, for street_pass;
    `passage_ways`: OSM ways run under a building, as if tagged a passage). Each
    square's lot (0 none)."""
    move = _move(sq, paths)
    mpt = sq.info["meters_per_tile"]
    index = {id(l): i for i, l in enumerate(sq.links)}
    game_lines = {}
    for w, (geo, l) in network.way_lines(sq.links, paths, sq.real_ways).items():
        side = max((sum(s_) for s_ in sq.chosen.get(index[id(l)], ())), default=0)
        game_lines[w] = (geo, {"width": str(l.width), "sidewalk:both:width": str(side * mpt)})
    # Generate's buildings: each corner moved by the displacement.
    gj = {"features": []}
    for oid, ft in sq.area_feats.items():
        ring = [list(move(la, lo))[::-1] for lo, la in ft["geometry"]["coordinates"][0]]
        gj["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                               "properties": {**(ft["properties"].get("tags") or {}), "osm_id": oid}})
    H, W, proj = sq.H, sq.W, sq.proj
    with contextlib.redirect_stdout(io.StringIO()):
        keep, mine = B.outlines_to_place(gj, sq.area_feats, proj, W, H)
        for m in mine:
            m["rail"] = sq.rail.kind(m["ft"]["properties"], m["bng"])
        mine = [m for m in mine if m["rail"] != "footbridge"]
        mine, _rows, _homes = B.join_rows(mine, proj)
        for m in mine:
            m.setdefault("rail", None)

        def size(i):
            return mine[i]["bng"].area
        ground = sq.ground
        B.register_blocks(ground, mine, B.groups_of(len(mine), list(B.neighbours_touching(mine)), size))
        new, _only = B.lidar_buildings(ground, mine, keep, proj, W, H, sq.rail if sq.rail else None)
        mine += new
        pairs, slivers, _open = B.neighbours(mine, ground, lambda i: mine[i]["reg"])
        blocks = B.groups_of(len(mine), pairs, size)
        shifts, _gains = B.register_blocks(ground, mine, blocks)
        where = {i: bi for bi, members in enumerate(blocks) for i in members}
        block_slivers: dict = {}
        for i, _j, gap in slivers:
            block_slivers.setdefault(where[i], []).append(gap)
        for m in mine:
            m["cover"] = B.outline_cover(ground, shapely.affinity.translate(m["bng"], *m["reg"]))
        B.square_big(ground, mine, proj)
        streets, all_roads, road_list = B.load_streets(sq.map_dir, sq.info, proj, sq.names,
                                                       [m["px"] for m in mine], game_lines, passage_ways)
        at = B.Placement(H, W)
        for bi, members in enumerate(blocks):
            B.place_block(bi, members, mine, block_slivers.get(bi, []), shifts[bi], streets, all_roads,
                          road_list, proj, at, B.RECT_TILES, B.RECT_COUNT, False)
        B.straighten_pass(at, mine, False)
        B.street_pass(at, mine, street, B.carriageways(road_list, W, H), False)
        _pieces, final, _lost, _count = B.cut_into_lots(at.owner, len(mine))
    return final


def slope_at(line, d) -> str:
    """The slope class of the line's segment at distance d along it."""
    a, b = line.interpolate(max(0.0, d - 0.5)), line.interpolate(min(line.length, d + 0.5))
    dx, dy = round(b.x - a.x, 3), round(b.y - a.y, 3)
    if abs(dx) < 0.05 or abs(dy) < 0.05:
        return "grid"
    return "45" if abs(abs(dx) - abs(dy)) < 0.1 else "2:1"


def distances(line, built: np.ndarray, sign: int) -> list:
    """frontage_measure.distances: every STEP tiles along, square to the line on one side,
    the distance to the first building (None within REACH)."""
    out = []
    H, W = built.shape
    L = line.length
    for d in np.arange(0.0, L, STEP):
        foot = line.interpolate(d)
        a, b = line.interpolate(max(0.0, d - 1.0)), line.interpolate(min(L, d + 1.0))
        t = np.array([b.x - a.x, b.y - a.y])
        t /= max(np.hypot(*t), 1e-9)
        n = (-t[1], t[0])
        hit = None
        for r in np.arange(0.5, REACH, 0.5):
            x, y = int(foot.x + sign * n[0] * r), int(foot.y + sign * n[1] * r)
            if 0 <= x < W and 0 <= y < H and built[y, x]:
                hit = r
                break
        out.append(hit)
    return out


def categories(ds: list) -> list:
    """Each sample flush, stepped, off or gap (see the module's flush / stepped)."""
    got = [v for v in ds if v is not None]
    if not got:
        return ["gap"] * len(ds)
    med = float(np.median(got))
    k = int(round(STEP_WIN / STEP))
    out = []
    for i, v in enumerate(ds):
        if v is None:
            out.append("gap")
            continue
        near = [u for u in ds[max(0, i - k):i + k + 1] if u is not None]
        rough = max(near) - min(near)
        if abs(v - med) <= TOL and rough <= STEP_RANGE:
            out.append("flush")
        elif abs(v - med) <= STEP_REACH:
            out.append("stepped")
        else:
            out.append("off")
    return out


def side_score(ds: list, cats: list | None = None) -> dict | None:
    got = np.array([v for v in ds if v is not None])
    if len(got) < 3:
        return None
    med = float(np.median(got))
    q1, q3 = np.percentile(got, [25, 75])
    cats = cats or categories(ds)
    return {"at": float(np.mean([v is not None and abs(v - med) <= TOL for v in ds])), "spread": float(q3 - q1),
            "n": len(ds), **{c: cats.count(c) / len(ds) for c in ("flush", "stepped", "off", "gap")}}


def frontage(sq: Square, lines: dict, built: np.ndarray, names=None, by_slope=False) -> dict:
    """Length-weighted share at the line, flush, stepped, off and gap, and the median spread,
    over the scored streets' sides; by_slope: the categories also for the samples on
    built-up stretches of each slope class ("built-up grid", "built-up 45", "built-up 2:1")."""
    at, wt, spreads = 0.0, 0.0, []
    cat = {c: 0.0 for c in ("flush", "stepped", "off", "gap")}
    built_n = built_grid = flush_n = 0
    slope_cats: dict = {}
    dens = sq.fields["density"]
    for i, line in lines.items():
        l = sq.links[i]
        if l.cls not in FRONT_CLASSES or not l.tags.get("name") or (names and l.tags.get("name") not in names):
            continue
        if shapely.LineString(l.pts).length < FRONT_MIN:
            continue
        for sign in (1, -1):
            ds = distances(line, built, sign)
            cats = categories(ds)
            s = side_score(ds, cats)
            if not s:
                continue
            at += s["at"] * s["n"]
            wt += s["n"]
            spreads.append(s["spread"])
            for c in cat:
                cat[c] += s[c] * s["n"]
            for d, c in zip(np.arange(0.0, line.length, STEP), cats):
                k = slope_at(line, d)
                if c != "gap":
                    built_n += 1
                    built_grid += k == "grid"
                    flush_n += c == "flush"
                if by_slope:
                    foot = line.interpolate(d)
                    if dens[min(sq.H - 1, int(foot.y)), min(sq.W - 1, int(foot.x))] >= snap.JOG_DENSITY:
                        slope_cats.setdefault(k, {c_: 0 for c_ in cat})[c] += 1
    out = {"at line": at / wt if wt else float("nan"),
           "spread": float(np.median(spreads)) if spreads else float("nan"), "sides": len(spreads),
           **{c: v / wt if wt else float("nan") for c, v in cat.items()},
           "ceiling": built_grid / built_n if built_n else float("nan"),
           "flush of ceiling": flush_n / built_grid if built_grid else float("nan")}
    for k, cs in slope_cats.items():
        n = sum(cs.values())
        out[f"built-up {k}"] = {**{c: v / n for c, v in cs.items()}, "samples": n}
    return out


def slope_class(dx, dy) -> str:
    return "grid" if dx == 0 or dy == 0 else "45" if abs(dx) == abs(dy) else "2:1"


def mid_frontage(sq: Square, v, hd_vec, width) -> bool:
    """A turn at v (heading hd_vec before it) where there is frontage: not a junction,
    gap or plot boundary (snap.turn_ok's rule, with the density test left out)."""
    hd = min(range(snap.N), key=lambda h: snap._adiff(snap.ANGLE[h], math.degrees(math.atan2(hd_vec[1], hd_vec[0])) % 360))
    saved = snap.COST, snap.JOG, snap.JOG_DENSITY
    snap.COST = np.ones((1, 1))
    snap.JOG = {"dist": sq.fields["building_dist"], "plots": sq.fields["plots"]}
    snap.JOG_DENSITY = 0.0
    try:
        return not snap.turn_ok(v[0], v[1], hd, width, (-1e9, -1e9), (-1e9, -1e9))
    finally:
        snap.COST, snap.JOG, snap.JOG_DENSITY = saved


def score(sq: Square, paths: dict, built: np.ndarray | None = None, junctions=None,
          passage_ways=None, passage: np.ndarray | None = None) -> tuple[dict, dict]:
    """The numbers for one routing (and the masks for its picture)."""
    owner, half = raster.rasterise(paths, sq.links, sq.H, sq.W)
    verge, pave, bays = roadside.draw(paths, sq.links, sq.chosen, owner, half, sq.hand)
    road = owner != 0
    for (x, y) in half:
        if 0 <= y < sq.H and 0 <= x < sq.W:
            road[y, x] = True
    street = road | verge | pave | bays
    if passage is not None:
        street = street & ~passage                    # a passage runs under its building: not overlap
    lots = placed(sq, paths, street, passage_ways) if built is None else None
    built = lots > 0 if built is None else built
    x0, y0, x1, y1 = sq.box
    win = (slice(y0, y1), slice(x0, x1))
    out = {"links routed": len(paths)}
    lines = {i: shapely.LineString(p) for i, p in paths.items() if len(p) >= 2}
    out["frontage"] = frontage(sq, lines, built, by_slope=True)
    out["high street"] = frontage(sq, lines, built, {"High Street"})
    on = built & street
    out["on street"] = int(on[win].sum())
    # ...split: wedges at junctions, slivers along streets.
    wedge = np.zeros_like(on)
    for (cx, cy), w in junctions or []:
        r = w / 2 + WEDGE_PAD
        a0, a1 = max(0, int(cy - r) - 1), min(sq.H, int(cy + r) + 2)
        b0, b1 = max(0, int(cx - r) - 1), min(sq.W, int(cx + r) + 2)
        yy, xx = np.mgrid[a0:a1, b0:b1]
        wedge[a0:a1, b0:b1] |= (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r
    out["on street: wedges"] = int((on & wedge)[win].sum())
    out["on street: slivers"] = out["on street"] - out["on street: wedges"]
    # Distance from the real centreline, a sample a tile.
    ds = []
    for i, line in lines.items():
        real = shapely.LineString(sq.links[i].pts)
        ds += [real.distance(line.interpolate(d)) for d in np.arange(0.0, line.length + 1e-9, 1.0)]
    out["centreline mean"] = float(np.mean(ds)) if ds else float("nan")
    out["centreline p95"] = float(np.percentile(ds, 95)) if ds else float("nan")
    out["centreline max"] = float(np.max(ds)) if ds else float("nan")
    turns = mid = 0
    length = {"grid": 0.0, "45": 0.0, "2:1": 0.0}
    built_up = {"grid": 0.0, "45": 0.0, "2:1": 0.0}
    dens = sq.fields["density"]
    for i, p in paths.items():
        vs = raster.vertices(p)
        for a, b in zip(vs, vs[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            k = slope_class(dx, dy)
            seg = math.hypot(dx, dy)
            length[k] += seg
            if dens[int((a[1] + b[1]) / 2), int((a[0] + b[0]) / 2)] >= snap.JOG_DENSITY:
                built_up[k] += seg
        for a, v, b in zip(vs, vs[1:], vs[2:]):
            turns += 1
            if mid_frontage(sq, v, (v[0] - a[0], v[1] - a[1]), sq.links[i].width) and \
                    min(math.dist(v, vs[0]), math.dist(v, vs[-1])) > max(snap.JOG_END, sq.links[i].width):
                mid += 1
    out["turns per link"] = turns / max(len(paths), 1)
    out["mid-frontage turns"] = mid
    tot, tb = sum(length.values()) or 1.0, sum(built_up.values()) or 1.0
    out["length share"] = {k: v / tot for k, v in length.items()}
    out["built-up length share"] = {k: v / tb for k, v in built_up.items()}
    return out, {"road": road, "verge": verge, "pave": pave | bays, "built": built, "on": on, "owner": owner,
                 "wedge": wedge, "lots": lots}


def picture(sq: Square, masks: dict, paths: dict, title: str, S: int = 2) -> Image.Image:
    x0, y0, x1, y1 = sq.box
    a = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    a[:] = (46, 58, 40)
    crop = lambda m: m[y0:y1, x0:x1]
    a[crop(masks["verge"])] = (84, 120, 64)
    a[crop(masks["pave"])] = (190, 186, 172)
    a[crop(masks["road"])] = (120, 120, 120)
    a[crop(masks["built"])] = (150, 96, 64)
    a[crop(masks["on"])] = (230, 60, 200)                   # on the street, as the score counts it
    img = Image.fromarray(a).resize(((x1 - x0) * S, (y1 - y0) * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for i in paths:
        d.line([((x - x0) * S, (y - y0) * S) for x, y in sq.links[i].pts], fill=(230, 40, 40), width=1)
    d.rectangle([0, 0, len(title) * 7 + 8, 16], fill=(20, 20, 20))
    d.text((4, 2), title, fill=(255, 255, 255), font=ImageFont.load_default())
    return img


def overlay(img: Image.Image, sq: Square, mask: np.ndarray, colour, S: int = 2) -> Image.Image:
    """A mask (map tiles) painted over a picture of the square (picture's S)."""
    x0, y0, x1, y1 = sq.box
    a = np.array(img)
    a[np.repeat(np.repeat(mask[y0:y1, x0:x1], S, 0), S, 1)] = colour
    return Image.fromarray(a)


def sheet(pics: list, cols: int, gap: int = 0) -> Image.Image:
    """Pictures of one size side by side, `cols` to a row."""
    w, h = pics[0].size
    cols = max(1, min(cols, len(pics)))
    out = Image.new("RGB", ((w + gap) * cols - gap, (h + gap) * -(-len(pics) // cols) - gap), (20, 20, 20))
    for k, p in enumerate(pics):
        out.paste(p, ((k % cols) * (w + gap), (k // cols) * (h + gap)))
    return out


def run_preset(sq: Square, name: str, fields=None, passage_ways=None, r=None) -> tuple[dict, dict, dict]:
    """One run: a preset (RUNS; "<preset>@front" squares about the fronts too, as a
    front-anchored preset does) routed - or `r`, a routing already made - and its
    buildings placed and scored. `passage_ways`: lanes run under their buildings, their
    squares there a passage, not overlap (masks["passage"]). (scores, masks, routing)"""
    preset, _at, anchor = name.partition("@")
    if r is None:
        mode, params = RUNS[preset]
        r = sq.route(mode, params, fields)
    passage = None
    if passage_ways:
        img = Image.new("1", (sq.W, sq.H), 0)
        d = ImageDraw.Draw(img)
        for w, (geo, l) in network.way_lines(sq.links, r["paths"], sq.real_ways).items():
            if w in passage_ways:
                d.line([tuple(p) for p in geo], fill=1, width=int(l.width) + 1)
        passage = np.array(img, bool)
    B.ANCHOR_FRONT = anchor == "front" or modes.front_anchored(preset)
    try:
        s, masks = score(sq, r["paths"], junctions=r["junctions"], passage_ways=passage_ways, passage=passage)
    finally:
        B.ANCHOR_FRONT = False
    masks["passage"] = passage
    s.update({"seconds": r["seconds"], "failed": len(r["failed"]), "jog let go": len(r["let_go"]),
              "mode": r["info"]})
    return s, masks, r


def _slope_cell(f: dict, k: str) -> str:
    c = f.get(f"built-up {k}")
    return f"{c['flush']:.0%} / {c['stepped']:.0%} ({c['samples']})" if c else "-"


def table(rows: dict) -> str:
    head = ["run", "front at line", "flush / stepped / off / gap", "built-up 2:1 flush / stepped (samples)",
            "ceiling / flush of it", "spread", "High St at line", "High St flush / stepped",
            "High St ceiling / flush of it", "High St spread", "on street",
            "slivers", "wedges", "stray mean / p95 / max", "turns/link", "mid-frontage turns",
            "grid / 45 / 2:1 (all)", "grid / 45 / 2:1 (built-up)", "jog let go", "s"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in rows.items():
        ls, lb = r["length share"], r["built-up length share"]
        out.append("| " + " | ".join([
            name, f"{r['frontage']['at line']:.0%}",
            " / ".join(f"{r['frontage'][c]:.0%}" for c in ("flush", "stepped", "off", "gap")),
            _slope_cell(r["frontage"], "2:1"),
            f"{r['frontage']['ceiling']:.0%} / {r['frontage']['flush of ceiling']:.0%}",
            f"{r['frontage']['spread']:.1f}", f"{r['high street']['at line']:.0%}",
            f"{r['high street']['flush']:.0%} / {r['high street']['stepped']:.0%}",
            f"{r['high street']['ceiling']:.0%} / {r['high street']['flush of ceiling']:.0%}",
            f"{r['high street']['spread']:.1f}", f"{r['on street']:,}",
            f"{r['on street: slivers']:,}", f"{r['on street: wedges']:,}",
            f"{r['centreline mean']:.2f} / {r['centreline p95']:.1f} / {r['centreline max']:.1f}",
            f"{r['turns per link']:.2f}", str(r["mid-frontage turns"]),
            " / ".join(f"{ls[k]:.0%}" for k in ("grid", "45", "2:1")),
            " / ".join(f"{lb[k]:.0%}" for k in ("grid", "45", "2:1")),
            f"{r.get('jog let go', '')}", f"{r.get('seconds', 0):.0f}"]) + " |")
    return "\n".join(out)


def built_map(sq: Square, built_dir: str) -> tuple[dict, dict, np.ndarray]:
    """A built map's routing (the square's links of it, by their place in the whole
    graph), its junctions (corners where three or more of its paths end) and Build's
    footprints."""
    import pickle
    from collections import Counter
    from knoxbuild.population import load_footprints
    name = os.path.basename(os.path.normpath(built_dir))
    links_b, paths_b, *_rest = pickle.load(open(os.path.join(built_dir, f"{name}_roadgraph_state.pkl"), "rb"))
    paths = {k: paths_b[i] for k, i in enumerate(sq.keep) if i in paths_b}
    ends = Counter(tuple(p[e]) for p in paths.values() for e in (0, -1))
    width_at: dict = {}
    for k, p in paths.items():
        for e in (0, -1):
            width_at[tuple(p[e])] = max(width_at.get(tuple(p[e]), 0), sq.links[k].width)
    junctions = [(c, width_at[c]) for c, n in ends.items() if n >= 3]
    game = np.zeros((sq.H, sq.W), bool)
    for x0, y0, m, _lv, _k in load_footprints(os.path.join(built_dir, f"{name}_footprints.npz")):
        hh, ww = m.shape
        game[y0:y0 + hh, x0:x0 + ww] |= m
    return paths, junctions, game


def score_built(sq: Square, dirs: list, out: str) -> int:
    rows = {}
    pics = []
    for d in dirs:
        paths, junctions, game = built_map(sq, d)
        label = os.path.basename(os.path.normpath(d))
        s_, masks = score(sq, paths, game, junctions)
        info = json.load(open(os.path.join(d, f"{label}_info.json"), encoding="utf-8"))
        s_["mode"] = info.get("road_mode", {"mode": "graph"})
        rows[label] = s_
        pics.append(picture(sq, masks, paths, f"{label} (built)"))
        pics[-1].save(os.path.join(out, f"built_{label}.png"))
    md = ["Built maps, Build's footprints, the square only:", "", table(rows)]
    open(os.path.join(out, "built.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(rows, open(os.path.join(out, "built.json"), "w"), indent=1, default=float)
    sheet(pics, len(pics)).save(os.path.join(out, "built_all.png"))
    print("\n".join(md))
    return 0


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    box = tuple(int(v) for v in _arg("--box", ",".join(map(str, BOX))).split(","))
    runs = _arg("--runs", ",".join(RUNS)).split(",")
    out = _arg("--out", os.path.join(HERE, "..", "..", "out", "roadgraph", "mode_test"))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sq = Square(map_dir, area, box)
    print(f"{len(sq.links)} links in {box}; set up in {time.time() - t0:.0f} s")
    if "--score-built" in sys.argv:
        dirs = [a for a in sys.argv[sys.argv.index("--score-built") + 1:] if not a.startswith("--")]
        return score_built(sq, dirs, out)
    rows, pics = {}, []
    # The real streets, as a row: the real lines, the real buildings.
    real_paths = {i: [tuple(map(float, p)) for p in l.pts] for i, l in enumerate(sq.links)}
    lines = {i: shapely.LineString(p) for i, p in real_paths.items()}
    rows_real = {"frontage": frontage(sq, lines, sq.real_mask), "high street": frontage(sq, lines, sq.real_mask,
                                                                                       {"High Street"})}
    routed: dict = {}                         # a preset routed once, placed both ways
    for name in runs:
        preset = name.partition("@")[0]
        s, masks, routed[preset] = run_preset(sq, name, r=routed.get(preset))
        rows[name] = s
        pics.append(picture(sq, masks, routed[preset]["paths"], name))
        pics[-1].save(os.path.join(out, f"{name}.png"))
        print(f"{name}: {json.dumps({k: v for k, v in s.items() if k != 'mode'}, default=float)}")
    md = [f"Square {box}: {len(sq.links)} links. Real streets: frontage at line "
          f"{rows_real['frontage']['at line']:.0%}, spread {rows_real['frontage']['spread']:.1f}; High Street "
          f"{rows_real['high street']['at line']:.0%}, spread {rows_real['high street']['spread']:.1f}.", "",
          table(rows)]
    open(os.path.join(out, "table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump({"box": box, "real": rows_real, "runs": rows}, open(os.path.join(out, "scores.json"), "w"), indent=1,
              default=float)
    sheet(pics, 3).save(os.path.join(out, "all.png"))         # all runs side by side, three to a row
    print("\n".join(md))
    print("written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
