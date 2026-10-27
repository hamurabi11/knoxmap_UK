"""Road bridges over water: the deck as a street, nothing else on the river round it
(LOG 2026-10-24 #W15).

On a road-graph map the hand-over paints the graph's carriageway across a
river but keeps the road's verge and pavement off the water
(roadgraph/handover.repaint: `keep = ~road & ~water`), and what OpenStreetMap
maps of a bridge besides its road stays where the bridge really is while the
road is routed square: the piers (holes in the river's multipolygon) were
painted as land, so grass with trees and bushes on the river; a footway area
or cycle track on the real line a paved strip on the water, and a plot with
hedges on it; the parapet walls stood on the water off the deck, and where
one crossed the routed road kerbs.py dropped it (principle 15), so a wall
along one side only. Westminster Bridge had all of it.

kerbs.py calls lay() on its ground before kerbing:

1. The bridges: ways of a road class (roadgraph/network.CLASSES) tagged
   bridge (not "no"), drawn where the road graph routed them
   (<map>_roadgraph.json), with water beside at least WET_MIN squares of
   that line.
2. Round each, its corridor: the routed line and the real one (as Generate
   moved it, octilinear.displacement) REACH_TILES to each side, ends square.
   Land there too thin to hold a disc THIN_R across (with the carriageway
   taken out) is in the river - a pier, a strip on the real line, a plot on
   it - and is water again: a thin bit of at least JUNK_MIN squares touching
   the wide bank along at most 1 / CONTACT_SHARE of its squares (a bank's
   corner beside the abutment is smaller, or meets the bank along its width).
   A pier or footbridge mapped in its own right (man_made=pier, a path tagged
   bridge the hand-over kept) stays.
3. The deck: the carriageway as routed, and on each side with a pavement
   (as the hand-over decided that road's sides, roadside.decide) a pavement
   over the water as wide as that side's pavement, verge and bays together
   (no grass on a deck), out from the carriageway; a side with no pavement
   gets none (its verge stays off the water). In town as on land (the town
   mask judged at the bridge's ends). Kerbed by kerbs.py like any pavement.
4. Parapets: generator/structures.py's black metal railing, as on every
   other bridge over water and every lifted deck, along each edge of the deck
   that meets water (rails()); Generate's railings in the corridor, laid for
   the deck as Generate drew it, come off (drop_old_rails()).
5. Boundaries (fences, walls, hedges, any source) standing on the water or
   the deck there are set aside (set_aside()).

blocks.py also refuses a LiDAR-only shape standing on a road bridge's deck
(Decks: the LiDAR sees the deck and its parapets as a building - a "works"
shed on Lambeth Bridge).
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import LineString
from shapely.ops import linemerge

REACH_TILES = 16      # each side of a bridge's lines (Westminster Bridge's piers run 15 out from its middle)
THIN_R = 4            # land not holding a disc of this radius (9 squares across) is thin (piers 6-7 wide)
JUNK_MIN = 10         # squares: a thin bit smaller than this is a bank's corner, left as it is
CONTACT_SHARE = 6     # a thin bit is in the river if it meets the wide bank along at most 1/6 of its squares
WATER_SHARE = 4       # ... and the water along at least 1/4 of them (not an island between two roads)
PAIR_TILES = 5        # past each of two side-by-side runs' carriageways: the water between them is deck
GAP_R = 2             # water in the deck's band narrower than a disc this radius (5 across) is deck
WET_MIN = 4           # squares of the routed line with water beside, for a bridge over water
WET_SIDE = 3          # ... beside: this many squares past the carriageway's edge
PROTECT_TILES = 2     # round a pier or footbridge mapped in its own right
DECK_M = 3.0          # metres each side of a road bridge's carriageway that are its deck (blocks.py, LiDAR)


@dataclass
class Result:
    corridor: np.ndarray            # (H, W) bool: round the bridges over water
    deck: np.ndarray                # their carriageway and pavements
    pave: np.ndarray                # pavement laid over the water here
    wet: np.ndarray                 # land made water again here
    water: np.ndarray               # water after
    rails: list = field(default_factory=list)   # [x, y, 0, "Furniture", tile]
    bridges: int = 0
    ways: list = field(default_factory=list)


def _px(proj, geometry, move=None):
    return [proj.to_px(*(move(la, lo) if move else (la, lo))) for la, lo in geometry]


def _fill(mask, geom, H, W):
    """Squares whose middle lies in `geom`, set in `mask`."""
    if geom.is_empty:
        return
    x0, y0, x1, y1 = geom.bounds
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(W - 1, int(np.ceil(x1))), min(H - 1, int(np.ceil(y1)))
    if x1 < x0 or y1 < y0:
        return
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    mask[y0:y1 + 1, x0:x1 + 1] |= shapely.contains_xy(geom, xs + 0.5, ys + 0.5)


def _is_colour(img, colours):
    out = np.zeros(img.shape[:2], bool)
    for c in colours:
        out |= np.all(img == c, axis=-1)
    return out


def _wet(line: LineString, half: float, water: np.ndarray) -> int:
    """Squares along `line` with water within its half width + WET_SIDE."""
    H, W = water.shape
    r = int(half + WET_SIDE)
    n = 0
    for t in np.arange(0.0, line.length + 0.5, 1.0):
        p = line.interpolate(min(t, line.length))
        x, y = int(p.x), int(p.y)
        if 0 <= x < W and 0 <= y < H:
            win = water[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            n += bool(win.mean() > 0.3)
    return n


def road_bridges(feats, classes) -> list:
    """OSM ways of a road class tagged bridge (and not a tunnel)."""
    return [f for f in feats if f.kind == "way" and len(f.geometry) >= 2
            and (f.tags.get("highway") or "").lower() in classes
            and f.tags.get("bridge") not in (None, "no")]


def lay(map_dir: str, ground: np.ndarray, buildings, road_colours, pave_colour, water_colour,
        log=print) -> Result | None:
    """The decks of the road bridges over water laid on `ground` (changed in
    place), the river round them cleared; None on a map without a road graph."""
    from generator import octilinear, osm
    from roadgraph import network, roadside
    name = os.path.basename(os.path.normpath(map_dir))
    gpath = os.path.join(map_dir, f"{name}_roadgraph.json")
    spath = os.path.join(map_dir, f"{name}_roadgraph_state.pkl")
    if not (os.path.exists(gpath) and os.path.exists(spath)):
        return None
    graph = json.load(open(gpath, encoding="utf-8"))
    info, proj = network.projector(map_dir)
    feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or []
    H, W = ground.shape[:2]
    water = _is_colour(ground, (water_colour,))
    street_c = tuple(road_colours) + (pave_colour,)
    road = _is_colour(ground, road_colours)
    built = buildings[:H, :W] if buildings is not None else np.zeros((H, W), bool)
    move = octilinear.displacement([(np.array(proj.to_px(a, b)), np.subtract(proj.to_px(c, d), proj.to_px(a, b)))
                                    for a, b, c, d in graph.get("shift", [])], proj)

    # A bridge is often several ways end to end (Westminster Bridge: four, one
    # of them 7 squares): joined into one run as routed, each run with its
    # ways' real lines, so the corridor has no gap at a join.
    ways = []
    for f in road_bridges(feats, network.CLASSES):
        g = graph["ways"].get(str(f.osm_id))
        if not g or len(g["geometry"]) < 2:
            continue
        line = LineString(_px(proj, g["geometry"]))
        if line.length >= 1:
            ways.append((f.osm_id, line, LineString(_px(proj, f.geometry, move)),
                         float(g["tags"].get("width", 6)) / 2))
    merged = linemerge([w[1] for w in ways]) if ways else None
    runs = list(getattr(merged, "geoms", [merged])) if merged is not None and not merged.is_empty else []
    over = []                    # (way ids, routed run, real lines as moved, half width)
    for run in runs:
        mine = [w for w in ways if run.distance(w[1].interpolate(0.5, normalized=True)) < 0.5]
        if not mine:
            continue
        half = max(w[3] for w in mine)
        if run.length < 2 or _wet(run, half, water) < WET_MIN:
            continue
        over.append(({w[0] for w in mine}, run, shapely.union_all([w[2] for w in mine]), half))
    res = Result(*(np.zeros((H, W), bool) for _ in range(4)), water=water)
    if not over:
        return res
    res.ways = sorted(w for ids, *_ in over for w in ids)

    # 2. The corridor, the carriageway, and the squares a pier or footbridge of its own keeps.
    corridor = np.zeros((H, W), bool)
    carriage = np.zeros((H, W), bool)
    for _w, line, real, half in over:
        hull = shapely.union_all([line, real]).convex_hull
        _fill(corridor, shapely.union_all([line.buffer(half + REACH_TILES, cap_style="flat"),
                                           real.buffer(half + REACH_TILES, cap_style="flat"), hull]), H, W)
        _fill(carriage, line.buffer(half + 1.0, cap_style="flat"), H, W)
    carriage &= road
    drop = set(graph.get("drop", []))
    protect = np.zeros((H, W), bool)
    for f in feats:
        if f.kind != "way" or len(f.geometry) < 2 or f.osm_id in drop:
            continue
        own = f.tags.get("man_made") == "pier" or (
            f.tags.get("bridge") not in (None, "no")
            and (f.tags.get("highway") or "").lower() in ("footway", "path", "cycleway", "steps", "bridleway", "pedestrian"))
        if not own:
            continue
        pts = _px(proj, f.geometry, move)
        if not any(corridor[int(y), int(x)] for x, y in pts if 0 <= x < W and 0 <= y < H):
            continue
        geom = shapely.Polygon(pts) if len(pts) >= 4 and pts[0] == pts[-1] else LineString(pts)
        _fill(protect, geom.buffer(PROTECT_TILES + 0.5), H, W)

    # The river's thin land, worked in the corridor's box.
    ys, xs = np.nonzero(corridor)
    pad = 2 * THIN_R + 2
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    win = (slice(y0, y1), slice(x0, x1))
    land = ~water[win] & ~carriage[win]
    disc = np.hypot(*np.mgrid[-THIN_R:THIN_R + 1, -THIN_R:THIN_R + 1]) <= THIN_R
    wide = ndimage.binary_opening(land, structure=disc)
    thin = (land & ~wide & corridor[win] & ~_is_colour(ground[win], street_c) & ~built[win] & ~protect[win])
    lab, n = ndimage.label(thin)
    touch = ndimage.binary_dilation(wide) & thin           # thin squares beside the wide bank
    sizes = ndimage.sum(np.ones_like(thin), lab, range(1, n + 1))
    contact = ndimage.sum(touch, lab, range(1, n + 1))
    shore = ndimage.binary_dilation(water[win]) & thin       # thin squares beside the water
    wetness = ndimage.sum(shore, lab, range(1, n + 1))
    junk_ids = [k + 1 for k in range(n) if sizes[k] >= JUNK_MIN and contact[k] * CONTACT_SHARE <= sizes[k]
                and wetness[k] * WATER_SHARE >= sizes[k]]
    wet = np.zeros((H, W), bool)
    wet[win] = np.isin(lab, junk_ids) & (lab > 0)
    ground[wet] = water_colour
    water = water | wet

    # 3. The deck's pavements over the water, each side as wide as the
    # hand-over made that side's pavement (and verge and bays: no grass on a
    # deck) out from the carriageway - none on a side with no pavement - in
    # town (the hand-over's town mask) judged at the bridge's ends, the river
    # itself being outside it.
    links, paths, _p, owner, half_t, town, chosen = pickle.load(open(spath, "rb"))
    town = town[:H, :W]
    hand = roadside.handedness(proj)
    pave = np.zeros((H, W), bool)
    for i, l in enumerate(links):
        if i not in paths or i not in chosen:
            continue
        mine = [o for o in over if o[0] & set(l.ways)]
        sides = [sum(s) if s[1] else 0 for s in chosen[i]]     # (left, right) tiles of deck past the kerb
        if not mine or not any(sides):
            continue
        band = np.zeros((H, W), bool)
        for _ids, line, _real, half in mine:
            _fill(band, line.buffer(half + max(sides) + 1.0, cap_style="flat"), H, W)
        if not (ndimage.binary_dilation(band, iterations=2) & town).any():
            continue
        carriage_i = band & (owner[:H, :W] == i + 1)
        for (x, y) in half_t:
            if 0 <= y < H and 0 <= x < W and band[y, x]:
                carriage_i[y, x] = True
        dist = ndimage.distance_transform_edt(~carriage_i)
        cand = band & water & ~road & ~built & (dist > 0) & (dist <= max(sides) + 0.5)
        ys, xs = np.nonzero(cand)
        if not len(ys):
            continue
        # Which side of the road each square is, as roadside.draw tells it.
        path = LineString(paths[i])
        tx, ty = xs + 0.5, ys + 0.5
        s = shapely.line_locate_point(path, shapely.points(tx, ty))
        a = shapely.line_interpolate_point(path, np.clip(s - 0.5, 0, path.length))
        b = shapely.line_interpolate_point(path, np.clip(s + 0.5, 0, path.length))
        ax, ay, bx, by = shapely.get_x(a), shapely.get_y(a), shapely.get_x(b), shapely.get_y(b)
        cross = (bx - ax) * (ty - (ay + by) / 2) - (by - ay) * (tx - (ax + bx) / 2)
        reach = np.where(cross * hand > 0, sides[0], sides[1])
        ok = dist[ys, xs] <= reach + 0.5
        pave[ys[ok], xs[ok]] = True
    ground[pave] = pave_colour
    water &= ~pave

    # 4. The deck, for its railings: the street within each bridge's band.
    # Water left in a gap narrower than GAP_R across within it - between the
    # two carriageways of one bridge (Waterloo Bridge), in a step of the
    # deck's edge - is deck too, paved (a central reservation).
    band = np.zeros((H, W), bool)
    for _w, line, _real, half in over:
        _fill(band, line.buffer(half + 4.0, cap_style="flat"), H, W)
    disc_g = np.hypot(*np.mgrid[-GAP_R:GAP_R + 1, -GAP_R:GAP_R + 1]) <= GAP_R
    gap = band & water & ~ndimage.binary_opening(water, structure=disc_g)
    gap &= ndimage.binary_dilation(band & _is_colour(ground, street_c))
    # And the water between two runs of bridge side by side (a dual
    # carriageway routed apart: Waterloo Bridge's 6 squares): where their
    # bands, PAIR_TILES past each carriageway, overlap.
    seen = np.zeros((H, W), bool)
    for _w, line, _real, half in over:
        one = np.zeros((H, W), bool)
        _fill(one, line.buffer(half + PAIR_TILES, cap_style="flat"), H, W)
        gap |= one & seen & water
        seen |= one
    ground[gap] = pave_colour
    pave |= gap
    water &= ~gap
    deck = band & _is_colour(ground, street_c)
    res.corridor, res.deck, res.pave, res.wet, res.water = corridor, deck, pave, wet, water
    res.rails = rails(deck, water)
    res.bridges = len(over)
    return res


def rails(deck: np.ndarray, water: np.ndarray) -> list:
    """generator/structures.py's railing along every edge of `deck` that
    meets water, as its rail_water lays one along a bridge over water."""
    from generator.structures import RAIL_N, RAIL_NW, RAIL_W
    H, W = deck.shape
    sides: dict = {}
    ys, xs = np.nonzero(deck)
    for x, y in zip(xs.tolist(), ys.tolist()):
        for nx, ny, ex, ey, side in ((x - 1, y, x, y, "W"), (x + 1, y, x + 1, y, "W"),
                                     (x, y - 1, x, y, "N"), (x, y + 1, x, y + 1, "N")):
            if 0 <= nx < W and 0 <= ny < H and water[ny, nx] and not deck[ny, nx]:
                sides.setdefault((ex, ey), set()).add(side)
    return [[x, y, 0, "Furniture", RAIL_NW if len(s) == 2 else RAIL_W if "W" in s else RAIL_N]
            for (x, y), s in sorted(sides.items())]


def drop_old_rails(rows: list, corridor: np.ndarray) -> list:
    """Generate's railings in the corridor taken out (laid along its own deck,
    on ground the hand-over painted again)."""
    from generator.structures import RAIL_N, RAIL_NW, RAIL_W
    H, W = corridor.shape
    old = {RAIL_N, RAIL_NW, RAIL_W}
    return [r for r in rows if not (r[2] == 0 and r[3] == "Furniture" and r[4] in old
                                    and 0 <= r[1] < H and 0 <= r[0] < W and corridor[r[1], r[0]])]


def set_aside(squares: list, res: Result) -> bool:
    """A boundary line (its sampled squares) standing on the water or a deck
    in a bridge's corridor: at least 3 of its squares in the corridor, and
    half of those on water or deck."""
    H, W = res.corridor.shape
    inside = [(x, y) for x, y in squares if 0 <= y < H and 0 <= x < W and res.corridor[y, x]]
    if len(inside) < 3:
        return False
    on = sum(1 for x, y in inside if res.water[y, x] or res.deck[y, x])
    return 2 * on >= len(inside)


class Decks:
    """Road bridges' decks in BNG metres (blocks.py): a LiDAR-only shape on
    one is the deck and its parapets, not a building."""

    def __init__(self, feats):
        from roadgraph import network
        from buildings.rail_kit import TO_BNG
        decks = []
        for f in road_bridges(feats, network.CLASSES):
            line = LineString([TO_BNG(lo, la) for la, lo in f.geometry])
            half = network.CLASSES[f.tags["highway"].lower()][1] / 2
            try:
                half = max(half, float(f.tags.get("lanes", 0)) * 3.3 / 2)
            except ValueError:
                pass
            decks.append(line.buffer(half + DECK_M, cap_style="flat"))
        self.decks = shapely.union_all(decks) if decks else None

    def __bool__(self):
        return self.decks is not None

    def lidar_on(self, bng) -> bool:
        if self.decks is None:
            return False
        return bng.representative_point().within(self.decks) or \
            bng.intersection(self.decks).area >= 0.5 * bng.area
