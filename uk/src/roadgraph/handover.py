"""Hand the road graph over to KnoxMap (new_road_approach.md, build order 5):
a new map, its roads the road graph's, for the rest of the pipeline.

    python handover.py <source map dir> <area data dir> <new map name> [--force] [--roads graph|grid|cost] [--lanes]
    python handover.py <source map dir> <area data dir> <new map name> --check   (after the later steps)
    python handover.py <source map dir> <area data dir> <new map name> --repaint
        (the routing the last full run kept; the dropped ways, Generate and the
        painting again - after a change to those, without routing for an hour)
    python handover.py <source map dir> <area data dir> <new map name> --markings
        (only the markings again, from the routing the last full run kept:
        then kerbs.py, Build and compile)

1. The road graph over the whole source map: OSM, OS Open Roads, LiDAR
   widths, snapped and rasterised (as run.py does on a square).
2. <new>/<new>_roadgraph.json: every OSM way of the graph its snapped line
   (its stretch of each link it is part of, found by where it lies along the
   link's real line, carried over in proportion along the snapped path), its
   width and, where OS changed it, its class; the ways the graph dropped
   (folded into a bigger road, a turning loop, a merged junction); and
   "shift", points along the roads before and after (shift_samples), by
   which Generate moves everything else with the roads near it (build
   order 4: buildings, plots, hedges, paths, land use).
3. Generate with that file in KNOXMAP_ROAD_GRAPH, so straightening is
   replaced by the road graph's lines and nothing else moves: the source
   map's box, turn, settings and OSM download.
4. The ground's roads painted from the road graph's tiles, on the ground
   as Generate draws it with none of the graph's roads (a second Generate,
   every graph way dropped): each side of each road its parking bays,
   verge and pavement (roadside.py: from OSM and the LiDAR; in town only),
   then the carriageways (half tiles as road), worn as Generate wears them.
   <new>_roadside.npz keeps the verge, pavement and parking masks for the
   later steps (a verge is grass, so they cannot tell it from a lawn by its
   colour: plots2.py keeps gardens off it, kerbs.py kerbs it). The
   vegetation map's street dressing - edge lines, lamps, drains, hydrants -
   laid again for this ground (redress), and a white line along the
   carriageway where bays run beside it.

Skipped when nothing it depends on has changed since it last ran (FINGERPRINT:
the routing's code - ROUTING_CODE, not markings.py or this file - the source
map's OSM download and settings, the area's OS roads and LiDAR tiles,
vanilla's shapes) - the later steps re-run on the map as it is. --markings
redoes the markings and street dressing from the routing the last run kept;
--force runs it all (after a change to the painting here).

--roads: how built-up streets are routed (modes.py: graph, the default, as
before; grid; cost; or one of modes.PRESETS, e.g. cost-low-occ1), recorded in
the map's info.json as "road_mode" and part of the fingerprint. cost-front (the
UK option chosen after Alton, LOG 2026-10-25 #4) also writes front_anchor.on
into the new map, so blocks.py squares its buildings about their fronts; a
later hand-over with another option takes that file away again (only the one it
wrote: one put there by hand stays).

--lanes (lanes.py, LOG 2026-10-25 #7, #8; off by default): driveways leave the road
network and are painted as plot paving (dark tarmac, plots2.py's drive land) from
their street end to the first building; OSM-only service lanes are routed with the
buildings as hard obstacles - stuck, through a gap the LiDAR shows, else a passage
under their building (capped: lanes.PASSAGE_CAP and the building's depth), else
dropped and listed. The passages go into the roadgraph file ("passages"), which
blocks.py (the building kept over them) and regional/passages.py (the ground floor
left out) read; the driveways too ("drives", for --repaint).

The map's info.json carries "road_graph": every later step that straightens
roads (blocks.py ... Build) passes its map folder to straighten_roads, which
then reads this map's graph (octilinear.road_graph_of) and refuses a
KNOXMAP_ROAD_GRAPH left set for another map.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
KNOX = ukpaths.KNOX
SRC, AREA, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = os.path.join(KNOX, "output", NAME)
GRAPH = os.path.abspath(os.path.join(OUT, f"{NAME}_roadgraph.json"))
MODE = sys.argv[sys.argv.index("--roads") + 1] if "--roads" in sys.argv else "graph"
LANES = "--lanes" in sys.argv
ROAD_MODE = {"mode": MODE}   # what the routing used (modes.prepare), into the map's info.json
LANE_RESULT: dict = {}       # --lanes: the driveways, passages and dropped lanes (road_graph)
FRONT_FLAG = "written by handover.py --roads"     # front_anchor.on's text when this wrote it

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, KNOX)
from roadgraph import markings, modes, network, osroads, raster, roadside, snap, widths  # noqa: E402
from roadgraph.run import town_mask  # noqa: E402
from area import area_conf  # noqa: E402
from generator import osm, renderer  # noqa: E402
from generator import pz_colors as C  # noqa: E402

PAINTED_MIN = 0.9            # --check: this share of the routed road must be road on the finished ground
BAY_TILES = 3                # car-park asphalt each side of a car park's aisle (network.car_park_ways)
VERGE = C.MEDIUM_GRASS       # the ground Generate gives residential land
HIGHWAY_OF = {"motorway": "motorway", "a_road": "primary", "b_road": "secondary", "minor": "unclassified",
              "local": "residential", "service": "service", "track": "track"}
# Carriageways 150 150 150, which streets/uk_road_rules.py draws in vanilla's
# street2; service lanes the worn tarmac (street4).
COLOUR_OF = {"motorway": C.MEDIUM_ASPHALT, "a_road": C.MEDIUM_ASPHALT, "b_road": C.MEDIUM_ASPHALT,
             "minor": C.MEDIUM_ASPHALT, "local": C.MEDIUM_ASPHALT, "service": C.DARKEST_ASPHALT, "track": C.DIRT}


def road_feats():
    """The source map's OSM ways the road graph is made of."""
    info, _p = network.projector(SRC)
    return network.road_ways(osm.load_cache(os.path.join(SRC, info["osm_cache"]), tuple(info["osm_bbox"])) or [])


# A street's own pavement, crossings and islands mapped as ways of their own
# (central London: 7,032 sidewalks, 3,812 crossings). The graph draws the
# pavement itself (roadside.py; sidewalk=separate counts as one), so these
# drawn too, where the road really was, ran beside it doubled and adrift
# (Westminster Bridge, 2026-10-21).
STREET_OWN = {"sidewalk", "crossing", "traffic_island"}


ALONG_SHARE = 0.9           # of an untagged cycleway's length beside a road for it to be the road's cycle track
ALONG_GAP_M = 6             # metres beyond the road's half width that still count as beside it
AREA_SHARE = 0.75           # of an area:highway's area (it runs on round the corner at the bridge ends: 0.77)
LANE_M = 3.3                # a lane's width, for a road tagged lanes= (Westminster Bridge: 4)


def street_own_ways() -> list:
    """Those ways' ids: tagged sidewalk / crossing / traffic island, or a
    cycleway (London's segregated tracks, Westminster Bridge's) or an
    area:highway lying ALONG_SHARE (of its length, or area) along a road of
    the graph."""
    info, proj = network.projector(SRC)
    feats = osm.load_cache(os.path.join(SRC, info["osm_cache"]), tuple(info["osm_bbox"])) or []
    out, tracks = [], []
    for f in feats:
        h = (f.tags.get("highway") or "").lower()
        if f.kind != "way" or h not in ("footway", "cycleway", "path") or len(f.geometry) < 2:
            continue
        if (f.tags.get("footway") or f.tags.get("cycleway") or f.tags.get("path") or "").lower() in STREET_OWN:
            out.append(f.osm_id)
        elif h == "cycleway":
            tracks.append(f)
    # A street's surface drawn as areas (area:highway: its pavements, islands,
    # carriageway; Westminster Bridge's pavements), also the street's own
    # where it lies along a road; a square (pedestrian) stays.
    tracks += [f for f in feats if f.kind == "way" and len(f.geometry) >= 4 and f.tags.get("area:highway")
               and f.tags["area:highway"].lower() != "pedestrian"]
    if tracks:
        mpt = info["meters_per_tile"]

        def half(r):                       # metres: the carriageway's half width, from its tags if it has them
            w = network.CLASSES[r.tags["highway"].lower()][1] * mpt
            try:
                w = max(w, float(r.tags.get("lanes", 0)) * LANE_M, network._width_tag(r.tags) or 0)
            except ValueError:
                pass
            return w / 2
        roads = [shapely.LineString([proj.to_px(la, lo) for la, lo in r.geometry]).buffer(
                 (half(r) + ALONG_GAP_M) / mpt) for r in road_feats()]
        tree = shapely.STRtree(roads)
        for f in tracks:
            pts = [proj.to_px(la, lo) for la, lo in f.geometry]
            geom = shapely.Polygon(pts).buffer(0) if f.tags.get("area:highway") else shapely.LineString(pts)
            size = (lambda g: g.area) if f.tags.get("area:highway") else (lambda g: g.length)
            near = tree.query(geom)
            if len(near) and size(geom) > 0 and \
                    size(geom.intersection(shapely.union_all([roads[k] for k in near]))) >=                     (AREA_SHARE if f.tags.get("area:highway") else ALONG_SHARE) * size(geom):
                out.append(f.osm_id)
    return out


def road_graph(log=print):
    nodes, links, proj, pins = network.build(SRC)
    os_report = osroads.conflate(links, proj, osroads.load(AREA), log=log)
    lanes = []
    if LANES:
        # Driveways out of the network (painted as paving later); the OSM-only lanes, renumbered.
        from roadgraph import lanes as LN
        way_tags = {f.osm_id: f.tags for f in road_feats()}
        drive, lanes_all = LN.plan(links, os_report, way_tags)
        out_ = set(drive)
        others = [l for i, l in enumerate(links) if i not in out_]
        LANE_RESULT["drives"] = [{"geometry": [list(proj.to_latlon(x, y)) for x, y in links[i].pts],
                                  "end": LN.street_end(links[i], others)} for i in drive]
        # a street a driveway met is one link again (no false junction to turn at), so the
        # lanes are found again by their ways
        lane_ways = {w for i in lanes_all for w in links[i].ways}
        links = network._merge_through(others)
        lanes = [k for k, l in enumerate(links) if l.cls == "service" and lane_ways.intersection(l.ways)]
        log(f"lanes: {len(drive)} driveways out of the network, {len(lanes)} OSM-only service lanes")
    lidar = widths.Lidar(widths.box_for(proj, links), os.path.join(AREA, "lidar"))
    sides: dict = {}
    measured = widths.measure(links, proj, lidar, sides)
    log(f"LiDAR: {len(measured)} of {len(links)} links measured, {widths.apply(links, measured)} widths changed")
    ways = {f.osm_id: ([proj.to_px(la, lo) for la, lo in f.geometry], f.tags) for f in road_feats()}
    chosen, tally = roadside.decide(links, sides, ways, measured)
    log(f"roadside (link sides): {tally}")
    town = town_mask(SRC, proj, proj.height, proj.width)
    fields = modes.fields(SRC, AREA, proj.height, proj.width) if LANES else None
    got: dict = {}

    def route(hard):
        # (reads `links` as it is now: the final routing below has the dropped lanes out)
        snap.HARD = hard
        got["r"] = snap.snap(nodes, links, town, log=lambda *a: None, pins=pins)
        return got["r"][0]
    ROAD_MODE.update(modes.prepare(MODE, SRC, AREA, proj.height, proj.width, fields))
    try:
        links, chosen = _route_lanes(route, links, chosen, lanes, fields, proj, log) if LANES else (links, chosen)
        if not LANES:
            route({})
    finally:
        modes.reset()
    paths, corner, merged, failed = got["r"]
    log(f"snapped: {len(paths)} of {len(links)} links routed, {len(failed)} failed")
    for i in failed:                                  # by name: what is missing from the map
        l_ = links[i]
        log(f"  not routed: link {i} {l_.tags.get('name') or '(no name)'} ({l_.cls}, ways {l_.ways[:2]}, "
            f"{shapely.LineString(l_.pts).length:.0f} tiles)")
    owner, half = raster.rasterise(paths, links, proj.height, proj.width)
    report = raster.checks(paths, links, nodes, corner, merged, owner, half)
    for k in ("topology", "geometry", "clearance"):
        log(f"  {k}: {report[k]}")
    return links, paths, proj, owner, half, town, chosen, lidar


def _route_lanes(route, links, chosen, lanes, fields, proj, log):
    """--lanes: the OSM-only lanes through lanes.resolve (route(hard) routes the network),
    then, with any lane dropped or made a passage, the final routing - dropped lanes out
    (links and their roadside renumbered), passages let under their buildings. The links
    and roadside as routed last; what became of the lanes into LANE_RESULT."""
    from roadgraph import lanes as LN
    from buildings import blocks as BK
    buildings_px = [p for p in map(shapely.Polygon, modes.building_rings(SRC)) if p.is_valid]

    def lidar_built():
        corners = [proj.to_latlon(x, y) for x, y in ((0, 0), (proj.width, 0), (0, proj.height),
                                                     (proj.width, proj.height))]
        es, ns = zip(*[BK.TO_BNG(lon, lat) for lat, lon in corners])
        ground = BK.Ground(AREA, (int(min(es)) - 5, int(min(ns)) - 5, int(max(es)) + 5, int(max(ns)) + 5))
        return LN.lidar_on_tiles(ground, proj, proj.height, proj.width)
    res = LN.resolve(route, lanes, fields["footprints"], lidar_built, buildings_px,
                     {k: links[k].pts for k in lanes}, log=log)
    LANE_RESULT["passages"] = sorted({w for k in res["passages"] for w in links[k].ways})
    LANE_RESULT["dropped"] = [{"ways": links[k].ways, "name": links[k].tags.get("name"), **res["checks"][k]}
                              for k in res["dropped"]]
    LANE_RESULT["gap"] = [links[k].ways for k in res["gap"]]
    log(f"lanes: through a LiDAR gap {len(res['gap'])}, passages {len(res['passages'])}, "
        f"dropped {len(res['dropped'])}")
    if res["passages"] or res["dropped"]:
        keep, at, hard = LN.finish(res, len(links))
        f2 = dict(fields, footprints=LN.clear_footprints(fields["footprints"],
                                                         [links[k].pts for k in res["passages"]]))
        ROAD_MODE.update(modes.prepare(MODE, SRC, AREA, proj.height, proj.width, f2))
        chosen = {at[i]: v for i, v in chosen.items() if i in at}
        links[:] = [links[k] for k in keep]                 # (route reads this list)
        route(hard)
    ROAD_MODE["lanes"] = {k: len(v) for k, v in LANE_RESULT.items()}
    return links, chosen


def way_lines(links, paths, proj, chosen: dict) -> dict:
    """{"ways": {id: {"geometry", "tags"}}, "drop": [...], "shift": [...]} for
    the roadgraph file. A way's tags: its width, its class where OS changed
    it, and its roadside as drawn here (roadside.decide; the wider side's
    bays, verge and pavement) as sidewalk:both:width, which every step reads
    for how far a building stands off the road (blocks.py)."""
    feats = road_feats()
    real = {f.osm_id: [proj.to_px(la, lo) for la, lo in f.geometry] for f in feats}
    highway = {f.osm_id: (f.tags.get("highway") or "").lower() for f in feats}
    out = {"ways": {}, "drop": []}
    for w, (geo, l) in network.way_lines(links, paths, real).items():
        tags = {"width": str(l.width)}
        side = max((sum(s) for s in chosen.get(links.index(l), ())), default=0)
        tags["sidewalk:both:width"] = str(side * proj.meters_per_tile)
        if network.CLASSES.get(highway.get(w, ""), (None,))[0] != l.cls:
            tags["highway"] = HIGHWAY_OF[l.cls]
        out["ways"][str(w)] = {"geometry": [list(proj.to_latlon(x, y)) for x, y in geo], "tags": tags}
    out["drop"] = [f.osm_id for f in feats if str(f.osm_id) not in out["ways"]] + street_own_ways()
    out["shift"] = shift_samples(links, paths, proj)
    for k in ("passages", "drives", "dropped", "gap"):      # --lanes (road_graph)
        if LANE_RESULT.get(k):
            out[k] = LANE_RESULT[k]
    out["road_mode"] = ROAD_MODE                    # what the routing used (back on --repaint)
    return out


SHIFT_EVERY = 4              # tiles between the points along a road that say how far it moved


def shift_samples(links, paths, proj) -> list:
    """[[lat, lon, lat, lon]]: points every SHIFT_EVERY tiles along each
    routed link, where they really are and where the snapped road put them
    (the same share of the way along it). Generate moves everything else
    with them (octilinear._from_road_graph): buildings, plots, hedges, paths
    and land use follow the roads (new_road_approach.md, build order 4)."""
    out = []
    for i, path in paths.items():
        real, snapped = shapely.LineString(links[i].pts), shapely.LineString(path)
        if real.length == 0 or snapped.length == 0:
            continue
        for t in np.linspace(0.0, 1.0, max(1, int(real.length // SHIFT_EVERY)) + 1):
            p, q = real.interpolate(t, normalized=True), snapped.interpolate(t, normalized=True)
            out.append([*proj.to_latlon(p.x, p.y), *proj.to_latlon(q.x, q.y)])
    return out


def generate(graph: str, out: str, name: str) -> None:
    """Generate `name` in `out` from the source map's box, turn, settings and
    OSM download, the roads from the road graph file `graph`."""
    os.environ["KNOXMAP_ROAD_GRAPH"] = graph
    info = json.load(open(os.path.join(SRC, f"{os.path.basename(os.path.normpath(SRC))}_info.json"), encoding="utf-8"))
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(SRC, "settings.json"), os.path.join(out, "settings.json"))
    shutil.copy(os.path.join(SRC, info["osm_cache"]), os.path.join(out, info["osm_cache"]))
    b = info["bbox"]
    features = osm.load_cache(os.path.join(out, info["osm_cache"]), tuple(info["osm_bbox"]))
    cwd = os.getcwd()
    os.chdir(KNOX)
    try:
        renderer.render(features, b["south"], b["west"], b["north"], b["east"],
                        meters_per_tile=info["meters_per_tile"], output_dir=os.path.relpath(out, KNOX), map_name=name,
                        spawn_density=10, tree_density=1.0, rotation=info["rotation"],
                        osm_cache=info["osm_cache"], osm_bbox=tuple(info["osm_bbox"]),
                        shape=info.get("shape"), straight_roads=True)
    finally:
        os.chdir(cwd)
    # The later steps start again from copies of the map as it was before
    # them (trees.py the ground and vegetation before trees, kerbs.py before
    # kerbs, blocks.py and others the buildings, fences and structures as
    # Generate made them) - kept from an earlier Generate they would put the
    # old roads back. Gone, each step takes them afresh from this map.
    for suffix in ("_ground_base.pretrees.bmp", "_veg_base.pretrees.bmp", "_ground_base.prekerbs.bmp",
                   "_veg_base.prekerbs.bmp", "_kerbs_state.json", "_buildings.orig.geojson",
                   "_fences.orig.geojson", "_structures.orig.json"):
        stale = os.path.join(out, name + suffix)
        if os.path.exists(stale):
            os.remove(stale)
    gi_path = os.path.join(out, f"{name}_info.json")
    gi = json.load(open(gi_path, encoding="utf-8"))
    gi["road_graph"] = os.path.basename(graph)       # every later step needs KNOXMAP_ROAD_GRAPH set to it
    gi["road_mode"] = ROAD_MODE                       # how its built-up streets were routed (modes.py)
    json.dump(gi, open(gi_path, "w", encoding="utf-8"), indent=1)
    # The road option's buildings: cost-front squares them about their fronts (blocks.py).
    flag = os.path.join(out, "front_anchor.on")
    ours = os.path.exists(flag) and open(flag, encoding="utf-8").read().startswith(FRONT_FLAG)
    if modes.front_anchored(ROAD_MODE.get("preset") or MODE):
        open(flag, "w", encoding="utf-8").write(f"{FRONT_FLAG} {ROAD_MODE.get('preset') or MODE}\n")
    elif ours:
        os.remove(flag)



def car_parks(proj, H: int, W: int, shift: list) -> np.ndarray:
    """Car-park asphalt round the car parks' and sites' aisles (not routed:
    drawn where they are, moved with the roads as Generate moves them)."""
    from PIL import ImageDraw
    from generator import octilinear
    move = octilinear.displacement([(np.array(proj.to_px(a, b)), np.subtract(proj.to_px(c, d), proj.to_px(a, b)))
                                    for a, b, c, d in shift], proj)
    info, _p = network.projector(SRC)
    feats = osm.load_cache(os.path.join(SRC, info["osm_cache"]), tuple(info["osm_bbox"])) or []
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    for f in network.car_park_ways(feats):
        pts = [proj.to_px(*move(la, lo)) for la, lo in f.geometry]
        d.line(pts, fill=1, width=2 * BAY_TILES + 3, joint="curve")
    return np.array(img, bool)


def repaint(bare: np.ndarray, proj, links, paths, owner, half, town, chosen, hand, parks=None, lidar=None,
            shift=()) -> dict:
    """The ground's roads from the road graph's tiles, on `bare` (the ground
    with none of the graph's roads), each side's verge and pavement beside
    them; the verge and pavement masks into <new>_roadside.npz."""
    path = os.path.join(OUT, f"{NAME}_ground_base.bmp")
    H, W = bare.shape[:2]
    owner = owner[:H, :W]
    road = owner != 0
    for (x, y) in half:
        if 0 <= y < H and 0 <= x < W:
            road[y, x] = True
    # Each road tile's link: its own, or the nearest full tile's.
    lab = np.where(owner > 0, owner - 1, -1)
    _d, (iy, ix) = ndimage.distance_transform_edt(lab < 0, return_indices=True)
    link_at = lab[iy, ix]
    cls_at = np.array([l.cls for l in links])[link_at]
    out = bare.copy()
    water = np.all(bare == C.WATER, axis=-1)
    parks = parks[:H, :W] & ~water if parks is not None else np.zeros((H, W), bool)
    out[parks] = C.DARK_ASPHALT                                # car parks first; streets drawn over them
    verge, pave, bays = roadside.draw(paths, links, chosen, owner, half, hand)
    if LANE_RESULT.get("drives"):
        # --lanes: the driveways as plot paving (plots2.py's drive land), from the street into
        # the plot as far as the first building (Generate's, moved with the town)
        from roadgraph import lanes as LN
        from generator import octilinear
        move = octilinear.displacement([(np.array(proj.to_px(a, b)), np.subtract(proj.to_px(c, d), proj.to_px(a, b)))
                                        for a, b, c, d in shift], proj)
        gj = json.load(open(os.path.join(OUT, f"{NAME}_buildings.geojson"), encoding="utf-8"))["features"]
        blds = []
        for ft in gj:
            g = ft["geometry"]
            for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]] if g["type"] == "Polygon" else []):
                p = shapely.Polygon([proj.to_px(la, lo) for lo, la in poly[0]])
                if p.is_valid:
                    blds.append(p)
        drives = [([proj.to_px(*move(la, lo)) for la, lo in dv["geometry"]], dv["end"]) for dv in LANE_RESULT["drives"]]
        paving = LN.drive_paving(drives, blds, H, W)
        out[paving & ~road & ~water & ~verge & ~pave & ~bays] = C.DARK_ASPHALT
    keep = ~road & ~water & town[:H, :W]
    verge &= keep & ~parks                                     # a car park's edge is paved
    pave &= keep
    bays &= keep
    out[verge] = VERGE
    out[pave] = C.PALE_CONCRETE
    out[bays] = C.MEDIUM_ASPHALT                               # the road's tarmac, kerbed from the pavement
    for cls, colour in COLOUR_OF.items():
        out[road & (cls_at == cls)] = colour
    # Generate's worn patches, on the roads as drawn here.
    img = Image.fromarray(out)
    renderer._weather_roads(img, proj)
    out = np.array(img)
    # Both ground maps: the steps before trees.py read the drawn one (plots2.py
    # did, and saw Generate's own roads instead of the graph's).
    Image.fromarray(out).save(path)
    Image.fromarray(out).save(os.path.join(OUT, f"{NAME}.bmp"), format="BMP")
    np.savez_compressed(os.path.join(OUT, f"{NAME}_roadside.npz"), verge=verge, pavement=pave, parking=bays)
    # UK markings and signals (markings.py), laid by kerbs.py with its kerbs.
    bayed = {i for i, both in chosen.items() if any(side[2] for side in both)}
    rows, counts, heads = markings.lay(paths, links, owner, half, town[:H, :W], bayed, bays, pave | verge, pave | verge,
                                       hand, proj, AREA, lidar)
    json.dump(rows, open(os.path.join(OUT, f"{NAME}_markings.json"), "w", encoding="utf-8"))
    edges = redress(img, road, markings.edge_keep(owner, links, town), heads)
    return {"road tiles": int(road.sum()), "verge tiles": int(verge.sum()), "pavement tiles": int(pave.sum()),
            "parking bay tiles": int(bays.sum()), "edge lines kept": edges, "markings": counts}


STREET_PIECES = (34, 36)     # vegetation's red 12: green 34 kerbs, 36 street furniture (pz_colors)
EDGE_LINES = ((C.EDGE_LINE_W, (-1, 0)), (C.EDGE_LINE_E, (1, 0)), (C.EDGE_LINE_N, (0, -1)), (C.EDGE_LINE_S, (0, 1)))


def redress(ground: Image.Image, road: np.ndarray, edge_keep: np.ndarray, heads=()) -> int:
    """The vegetation map dressed for the ground as painted here, not as
    Generate drew it: its kerbs, edge lines and street furniture (lamps,
    hydrants, drains, grime) taken off and laid again by Generate's own pass
    (renderer._paint_street_furniture) - a lamp on the pavement or verge
    where it really is, not in a parking bay - its edge lines kept on A and
    B roads out of town only (edge_keep) and there on straight runs only
    (markings.straight_runs: on a stepped stretch they came out staggered),
    its lane lines (KnoxMap's solid yellow centre line) all taken off for
    markings.py's, nothing growing on the road, no street furniture where a
    signal head stands (heads). The bays' lines are markings.py's. Safe to
    run again (--markings): it takes off what it lays. Returns the edge-line
    squares kept."""
    vpath = os.path.join(OUT, f"{NAME}_veg_base.bmp")
    veg = np.array(Image.open(vpath).convert("RGB"))
    H, W = road.shape
    street_piece = (veg[..., 0] == 12) & np.isin(veg[..., 1], STREET_PIECES)
    edge = np.zeros(veg.shape[:2], bool)
    for colour, _d in EDGE_LINES:
        edge |= np.all(veg == colour, axis=-1)
    lane = (veg[..., 0] == 12) & (veg[..., 1] == 35) & ~edge
    veg[street_piece | edge | lane] = 0
    img = Image.fromarray(veg)
    renderer._clear_road_vegetation(img, ground)
    hier = os.path.join(OUT, f"{NAME}_road_hierarchy.bmp")
    renderer._paint_street_furniture(img, ground, Image.open(hier) if os.path.exists(hier) else None)
    veg = np.array(img)
    kept = 0
    for colour, side in zip((C.EDGE_LINE_W, C.EDGE_LINE_E, C.EDGE_LINE_N, C.EDGE_LINE_S), "WENS"):
        mine = np.all(veg[:H, :W] == colour, axis=-1)
        keep = markings.straight_runs(mine & edge_keep[:H, :W], side, markings.BAY_RUN)
        veg[:H, :W][mine & ~keep] = 0
        kept += int(keep.sum())
    # No grime on the carriageway (user, 2026-10-11: the roads looked like
    # tiles, not smooth like vanilla's): Generate lays it in clumps (11.8% of
    # road squares, up to 146 together), and each square's grime is its own
    # hard-edged sprite, so a clump shows every square. Cracks stay.
    grime = np.all(veg[:H, :W] == C.ASPHALT_GRIME, axis=-1) & road
    veg[:H, :W][grime] = 0
    for x, y in heads:
        if (veg[y, x, 0], veg[y, x, 1]) == (12, 36):
            veg[y, x] = 0
    for p in (vpath, os.path.join(OUT, f"{NAME}_veg.bmp")):
        Image.fromarray(veg).save(p, format="BMP")
    return kept


def fingerprint() -> str:
    """A hash of everything the hand-over's result depends on."""
    import glob
    import hashlib
    h = hashlib.sha256()
    info = json.load(open(os.path.join(SRC, f"{os.path.basename(os.path.normpath(SRC))}_info.json"), encoding="utf-8"))
    # Only the code the routing depends on: markings.py, handover.py itself and
    # the like changed often, and each change forced an 11-minute re-route
    # the result did not need (--markings redoes the markings from STATE).
    files = [os.path.join(HERE, f + ".py") for f in ROUTING_CODE]
    files += [os.path.join(SRC, info["osm_cache"]), os.path.join(SRC, "settings.json"),
              os.path.join(AREA, "os_roads.geojson"),
              os.path.join(HERE, "..", "..", "data", "vanilla_roads", "stretch_shapes.json")]
    # The data level (area_conf): the OS extract and the titles read or not (full's hash as before).
    if area_conf.level(AREA) != "full":
        h.update(f"data={area_conf.level(AREA)}".encode())
    # A road mode or --lanes adds what only they read (graph's hash as before: no map
    # re-routed for a change it does not use).
    if MODE != "graph":
        h.update(f"roads={MODE}".encode())
        files += [os.path.join(HERE, "modes.py"), os.path.join(AREA, "inspire.geojson")]
    if LANES:
        h.update(b"lanes")
        files += [os.path.join(HERE, "lanes.py"), os.path.join(HERE, "..", "buildings", "blocks.py")]
    for f in files:
        if os.path.exists(f):
            h.update(os.path.basename(f).encode())
            h.update(open(f, "rb").read())
    # The LiDAR tiles by name and size (hashing gigabytes of rasters would cost more than it saves).
    for f in sorted(glob.glob(os.path.join(AREA, "lidar", "*.tif")) +
                    glob.glob(os.path.join(widths.INTENSITY_DIRS, "*.tif"))):
        h.update(f"{os.path.basename(f)}:{os.path.getsize(f)}".encode())
    return h.hexdigest()


FINGERPRINT = os.path.join(OUT, f"{NAME}_roadgraph.fingerprint")
ROUTING_CODE = ("network", "osroads", "widths", "roadside", "snap", "templates", "raster")   # what the fingerprint hashes


def check() -> int:
    """After the later steps: the share of the routed road tiles that are road
    (or pavement, a car park, a kerbed edge) on the finished ground - under
    PAINTED_MIN, something painted another layout over them."""
    z = np.load(os.path.join(OUT, f"{NAME}_roadgraph_tiles.npz"))
    road = z["owner"] != 0
    g = np.array(Image.open(os.path.join(OUT, f"{NAME}_ground_base.bmp")).convert("RGB"))
    H, W = min(g.shape[0], road.shape[0]), min(g.shape[1], road.shape[1])
    road, g = road[:H, :W], g[:H, :W]
    paved = np.zeros((H, W), bool)
    for c in (C.MEDIUM_ASPHALT, C.DARK_ASPHALT, C.DARKEST_ASPHALT, C.PALE_CONCRETE, C.DIRT, C.LIGHT_ASPHALT,
              C.DARK_POTHOLE, C.LIGHT_POTHOLE):
        paved |= np.all(g == c, axis=-1)
    share = float((paved & road).sum() / max(road.sum(), 1))
    print(f"routed road tiles painted as road on the finished ground: {share:.1%}")
    return 0 if share >= PAINTED_MIN else 1


STATE = os.path.join(OUT, f"{NAME}_roadgraph_state.pkl")


def markings_only() -> int:
    """The markings again (markings.py), from the routing the last full run
    kept (STATE) and the roadside it painted, without routing again."""
    import pickle
    if not os.path.exists(STATE):
        print(f"no {STATE}: run the hand-over in full first")
        return 1
    links, paths, proj, owner, half, town, chosen = pickle.load(open(STATE, "rb"))
    rs = np.load(os.path.join(OUT, f"{NAME}_roadside.npz"))
    H, W = rs["verge"].shape
    side = rs["pavement"] | rs["verge"]
    lidar = widths.Lidar(widths.box_for(proj, links), os.path.join(AREA, "lidar"))
    bayed = {i for i, both in chosen.items() if any(s[2] for s in both)}
    owner = owner[:H, :W]
    rows, counts, heads = markings.lay(paths, links, owner, half, town[:H, :W], bayed, rs["parking"], side, side,
                                       roadside.handedness(proj), proj, AREA, lidar)
    json.dump(rows, open(os.path.join(OUT, f"{NAME}_markings.json"), "w", encoding="utf-8"))
    # The street dressing again, on the ground as it now is (the hand-over's road).
    road = owner != 0
    for (x, y) in half:
        if 0 <= y < H and 0 <= x < W:
            road[y, x] = True
    ground = Image.open(os.path.join(OUT, f"{NAME}_ground_base.bmp")).convert("RGB")
    edges = redress(ground, road, markings.edge_keep(owner, links, town), heads)
    print("markings:", counts, "edge lines kept:", edges)
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    if "--markings" in sys.argv:
        return markings_only()
    os.makedirs(OUT, exist_ok=True)
    stamp = fingerprint()
    done = [GRAPH, os.path.join(OUT, f"{NAME}_ground_base.bmp"), os.path.join(OUT, f"{NAME}_info.json")]
    if "--force" not in sys.argv and "--repaint" not in sys.argv and all(os.path.exists(f) for f in done) and os.path.exists(FINGERPRINT)             and open(FINGERPRINT).read().strip() == stamp:
        print(f"road graph unchanged since it last ran: {OUT} reused (--force to route it again)")
        return 0
    import pickle
    if "--repaint" in sys.argv:          # the routing the last run kept; the drops, Generate and painting again
        links, paths, proj, owner, half, town, chosen = pickle.load(open(STATE, "rb"))
        if os.path.exists(GRAPH):            # the road mode routed with, --lanes' driveways and passages
            with open(GRAPH, encoding="utf-8") as f:
                prev = json.load(f)
            LANE_RESULT.update({k: prev[k] for k in ("passages", "drives", "dropped", "gap") if prev.get(k)})
            ROAD_MODE.clear()
            ROAD_MODE.update(prev.get("road_mode") or {"mode": MODE})
        lidar = widths.Lidar(widths.box_for(proj, links), os.path.join(AREA, "lidar"))
    else:
        links, paths, proj, owner, half, town, chosen, lidar = road_graph()
        pickle.dump((links, paths, proj, owner, half, town, chosen), open(STATE, "wb"))   # for --markings
    lines = way_lines(links, paths, proj, chosen)
    json.dump(lines, open(GRAPH, "w", encoding="utf-8"))
    print(f"roadgraph: {len(lines['ways'])} ways redrawn, {len(lines['drop'])} dropped -> {GRAPH}")
    # The ground with none of the graph's roads, to paint them onto.
    bare_dir = OUT + "_bare"
    bare_graph = os.path.join(bare_dir, "bare_roadgraph.json")
    os.makedirs(bare_dir, exist_ok=True)
    # (Everything else moved as on the map itself: the same shift.)
    json.dump({"ways": {}, "drop": [int(w) for w in lines["ways"]] + lines["drop"], "shift": lines["shift"]},
              open(bare_graph, "w", encoding="utf-8"))
    generate(bare_graph, bare_dir, NAME + "_bare")
    bare = np.array(Image.open(os.path.join(bare_dir, f"{NAME}_bare_ground_base.bmp")).convert("RGB"))
    shutil.rmtree(bare_dir)
    generate(GRAPH, OUT, NAME)
    print("generated", OUT)
    print("repainted:", repaint(bare, proj, links, paths, owner, half, town, chosen, roadside.handedness(proj),
                                car_parks(proj, proj.height, proj.width, lines["shift"]), lidar, lines["shift"]))
    np.savez_compressed(os.path.join(OUT, f"{NAME}_roadgraph_tiles.npz"), owner=owner,
                        half=np.array([(x, y) for x, y in half], dtype=np.int32).reshape(-1, 2))
    open(FINGERPRINT, "w").write(stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
