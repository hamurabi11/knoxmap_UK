"""Kerbs that follow the road's real edge, round corners included, in B42's own pieces' shapes
(drawn as UK concrete kerbs: uk_kerb).

    python kerbs.py <knoxmap map dir> [--sigma 1.5] [--picture out.png] [--box x,y,w,h]

Run after sports.py, before Build. The two files it changes are kept as they
were beside them (*.prekerbs.bmp), taken afresh whenever another step has
written them since, so it can be rerun at any point.

KnoxMap lays 8 kerb tiles by rule, one per way a pavement square can touch
road (worlded/patch_rules_roads.py), so a road can only turn in 45-degree
steps with sharp corners. Vanilla B42 curves its roads with kerbs in 16
directions and the bends between them (street_curbs_01, _diag, _diag_2), and
paints road over the part of a square the kerb cuts off (street_curbs_01_
blend_street, the same 88 shapes as the grass ones). Here:

0. The 45-degree roads are redrawn first (diagonals.py): one width all along,
   a pavement of 2 tiles a row hugging them; they get no kerbs (user,
   2026-10-09: the zig-zag kerbs on their steps looked wrong). A road-graph
   map's angled edges (45 degrees and 2:1) are drawn exactly already, on
   lines through tile corners, so a square one crosses is cut in one of 20
   ways: each laid as vanilla lays its own - tarmac, vanilla's kerb, overlay
   and stubs for that cut (ANGLED_TABLE), the overlay in the sheet of the
   ground beside (SIDE_SHEETS) (angled_squares). The edge kerbed is the
   tarmac's - the carriageway and its parking bays - and the back of an
   angled pavement is laid the same way, unkerbed: paving, with the garden's
   ground over the part beyond. Nothing on such a map is rounded (step 1).

1. Road (tarmac) and pavement squares are read from the ground. The road is
   drawn at SS points a tile and smoothed (a Gaussian of SIGMA tiles,
   thresholded at a half): corners of junctions, the 45-degree dog-legs of
   the straightened roads and the ends of cul-de-sacs come out round, and a
   straight edge stays exactly on its tile boundary. The road only moves
   within today's road and pavement, so nothing reaches a plot or building,
   and only where it meets pavement (a rural lane's verge is left alone)
   and is wide enough to round (THIN_TILES: drives and lanes keep their squares).
2. Each square the new edge crosses, on its pavement side, takes the kerb
   whose measured line (measure_kerbs.py) lies closest to the edge there -
   near-ties to the piece vanilla uses most (kerb_use.txt) - and the road
   overlay that best covers its road part. Squares wholly road are road,
   wholly pavement pavement. A road-graph map's grass verges
   (<map>_roadside.npz) are kerbed the same way, the kerb on the verge's
   square, with straight and corner pieces only.
3. Ground: <map>_ground_base.bmp takes the new road and pavement.
   Vegetation: the rule kerbs (KERB_* colours) are cleared from
   <map>_veg_base.bmp, and lane lines left on a square that is now pavement.
   (Build makes <map>.bmp and <map>_veg.bmp from these two.) Tiles: into <map>_structures.json after sports.py's
   (kerbs_added: how many, replaced on each run).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "kerbs")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
sys.path.insert(0, os.path.join(HERE, ".."))   # src: our packages
from generator import pz_colors as C  # noqa: E402

SS = 8                     # sub-points a tile
SIGMA = 1.5                # tiles: how round the corners come out
BLOCK = 64                 # tiles a block is worked in, plus MARGIN all round
MARGIN = 8
ROAD = (C.MEDIUM_ASPHALT, C.DARK_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE)
PAVE = C.PALE_CONCRETE
RULE_KERBS = (C.KERB_W, C.KERB_N, C.KERB_S, C.KERB_E, C.KERB_NW, C.KERB_SW, C.KERB_NE, C.KERB_SE)
LANE_LINES = (C.LINE_YELLOW_N, C.LINE_YELLOW_W, C.LINE_WHITE_N, C.LINE_WHITE_W)
CAPS = {f"street_curbs_01_{i}" for i in range(32, 40)}     # a kerb that stops: not mid-run
KERB_LAYER, OVERLAY_LAYER = "FloorGrime2", "FloorGrime"
# Road over part of a pavement square: our tarmac in vanilla's 88 shapes (the
# UK tile pack, asset_creator/Tile_creator/src/overlays.py; vanilla's
# street_curbs_01_blend_street is street3 grey, paler than the carriageway).
OVERLAY_SHEET = "street_curbs_uk_01_blend_street"
# The kerbs themselves: ours, UK concrete kerbs in exactly vanilla's shapes
# (asset_creator streetscape.py, kerb_shapes.py; 2026-10-14), laid by the
# same name - the matching is vanilla's pieces' measured points either way.
UK_KERBS = ("street_curbs_01_diag_2_", "street_curbs_01_diag_", "street_curbs_01_")


def uk_kerb(name: str) -> str:
    """Vanilla's kerb piece -> ours of the same shape (anything else as it is)."""
    for sheet in UK_KERBS:
        if name.startswith(sheet) and not name.startswith("street_curbs_01_blend"):
            return "street_curbs_uk_01" + name[len("street_curbs_01"):]
    return name


MIN_ROUND_TILES = 3.0      # a patch of road gained or lost by rounding must come to this, or the edge stays square
THIN_TILES = 5             # tarmac narrower than this is held as it is (opening by a disc of radius 2)
HEDGE_REACH = 4            # tiles a hedge drawn on paving may move to the unpaved ground beside it
MIN_OVERLAY = 0.08         # share of a square that is road before it is painted
WORST_FIT = 0.45           # a square whose best kerb fits worse than this gets none
SQUARE_PIECES = None        # which pieces are straight or corner ones (set by load_pieces)


def load_pieces():
    kerbs = {k: np.array(v) for k, v in json.load(open(os.path.join(DATA, "kerbs_topdown.json"))).items()
             if len(v) >= 12 and k not in CAPS}
    use = {}
    for row in open(os.path.join(DATA, "kerb_use.txt")):
        n, name = row.split()
        use[name] = int(n)
    top = max(use.values())
    penalty = {k: 0.10 * (1 - math.log(use.get(k, 0) + 1) / math.log(top + 1)) for k in kerbs}
    verges = {int(k): np.array(v, bool) for k, v in json.load(open(os.path.join(DATA, "verge_cover.json"))).items()}
    names = list(kerbs)
    # One array of every piece's points, padded, for matching all at once.
    n = max(len(kerbs[k]) for k in names)
    pts = np.full((len(names), n, 2), np.nan)
    for i, k in enumerate(names):
        pts[i, :len(kerbs[k])] = kerbs[k]
    pen = np.array([penalty[k] for k in names])
    global SQUARE_PIECES
    SQUARE_PIECES = np.array(["_diag" not in k for k in names])
    return names, pts, pen, verges


def best_kerb(p, names, pts, pen, memo, square_only=False):
    """The piece whose line lies closest to the edge points p (on the square).
    square_only: only straight and corner pieces (an edge left on the grid)."""
    key = (square_only,) + tuple(np.round(p * 16).astype(int).ravel())
    if key in memo:
        return memo[key]
    d = np.sqrt(((p[None, :, None, :] - pts[:, None, :, :]) ** 2).sum(-1))   # piece, p, q
    d = np.where(np.isnan(d), np.inf, d)
    a = d.min(2).mean(1)                       # edge to piece
    b = np.nanmean(np.where(np.isinf(d.min(1)), np.nan, d.min(1)), 1)         # piece to edge
    score = a + b + pen
    if square_only:
        score = np.where(SQUARE_PIECES[:len(score)], score, np.inf)
    i = int(np.argmin(score))
    memo[key] = (float(score[i]), names[i])
    return memo[key]


# A road-graph map's angled edges run on lines through tile corners, so a
# square one crosses is cut in one of 20 ways: a 2:1 line corner to mid-edge
# (8) or a 45-degree one corner to corner (2), the road on either side.
# Each: (the line from, to; a point on its road side) -> vanilla's kerb, the
# overlay it lays with it on tarmac (stopping short of the line by the kerb's
# face: 2:1 overlay 15 where the exact cut is 7) and the stubs it lays beside
# it (dx, dy, piece). Read from vanilla's lot files (LOG 2026-10-11): the
# rows marked "roads" are what vanilla lays along its own straight 2:1 roads
# and 45-degree edges; the 2:1 rows marked "curves" vanilla never lays on a
# straight road (its 2:1 roads all run one way) - the pieces it uses for
# that line in its curves, to check in game.
ANGLED_TABLE = [
    ((1, 0), (.5, 1), (1, 1), "diag_2_40", 15, ((-1, 1, "diag_2_42"), (0, -1, "diag_2_42")), "roads"),
    ((1, 0), (.5, 1), (0, 0), "diag_2_43", 13, ((-1, 1, "diag_2_45"), (0, -1, "diag_2_45")), "roads"),
    ((.5, 0), (0, 1), (1, 1), "diag_2_41", 14, ((-1, 0, "diag_2_42"),), "roads"),
    ((.5, 0), (0, 1), (0, 0), "diag_2_44", 12, ((-1, 0, "diag_2_45"),), "roads"),
    ((0, 0), (.5, 1), (1, 0), "diag_1", 27, (), "curves"),
    ((0, 0), (.5, 1), (0, 1), "diag_4", 23, ((-1, 0, "diag_5"),), "curves"),
    ((.5, 0), (1, 1), (1, 0), "diag_0", 26, (), "curves"),
    ((.5, 0), (1, 1), (0, 0), "diag_3", 30, ((-1, -1, "diag_5"), (0, 1, "diag_5")), "curves"),
    ((0, 0), (1, .5), (0, 1), "diag_41", 25, (), "curves"),
    ((0, 0), (1, .5), (1, 0), "diag_44", 21, ((0, -1, "diag_45"),), "curves"),
    ((0, .5), (1, 1), (0, 1), "diag_40", 24, (), "curves"),
    ((0, .5), (1, 1), (1, 0), "diag_43", 20, ((-1, -1, "diag_45"), (1, 0, "diag_45")), "curves"),
    ((0, 1), (1, .5), (1, 1), "diag_2_0", 8, ((1, -1, "diag_2_2"), (-1, 0, "diag_2_2")), "curves"),
    ((0, 1), (1, .5), (0, 0), "diag_2_3", 2, ((1, -1, "diag_2_5"), (-1, 0, "diag_2_5")), "curves"),
    ((0, .5), (1, 0), (1, 1), "diag_2_1", 9, ((0, -1, "diag_2_2"),), "curves"),
    ((0, .5), (1, 0), (0, 0), "diag_2_4", 3, ((0, -1, "diag_2_5"),), "curves"),
    ((0, 1), (1, 0), (1, 1), "56", 32, ((0, 1, "59"), (1, 0, "59")), "roads"),
    ((0, 1), (1, 0), (0, 0), "57", 33, ((0, -1, "62"), (-1, 0, "62")), "roads"),
    ((0, 0), (1, 1), (0, 1), "50", 37, (), "roads"),
    ((0, 0), (1, 1), (1, 0), "49", 46, (), "roads"),
]
ANGLED_MATCH = 0.9          # an edge square must match a row's cut this well (else not on the lattice: no kerb)


def _cut(a, b, road, on_line=False) -> np.ndarray:
    """The road part (SS x SS) of a square cut by the line a-b, on road's
    side (on_line: the sample points on the line itself instead)."""
    j, i = (np.mgrid[0:SS, 0:SS] + 0.5) / SS
    side = lambda x, y: (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])   # noqa: E731
    if on_line:
        return np.abs(side(i, j)) < 1e-9
    return np.sign(side(i, j)) == np.sign(side(*road))


# Each row's cut, and the points it is compared on: not those on the line
# (a 45-degree one's diagonal), which the outline's rounding puts either side.
ANGLED_CUTS = [(_cut(a, b, road), ~_cut(a, b, road, on_line=True)) for a, b, road, *_ in ANGLED_TABLE]


def angled_piece(part):
    """(kerb, overlay, stubs) for an edge square cut as `part`, from
    ANGLED_TABLE; None if it matches no row (an edge off the lattice)."""
    iou = [(c & part & on).sum() / max(1, ((c | part) & on).sum()) for c, on in ANGLED_CUTS]
    k = int(np.argmax(iou))
    if iou[k] < ANGLED_MATCH:
        return None
    _a, _b, _road, kerb, overlay, stubs, _seen = ANGLED_TABLE[k]
    return f"street_curbs_01_{kerb}", overlay, [(dx, dy, f"street_curbs_01_{p}") for dx, dy, p in stubs]


def best_overlay(cov, verges):
    best = None
    for k, c in verges.items():
        union = (c | cov).sum()
        iou = (c & cov).sum() / union if union else 0.0
        if best is None or iou > best[0]:
            best = (iou, k)
    return best[1]


def _digest(path: str) -> str:
    import hashlib
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def is_colour(img, colours):
    m = np.zeros(img.shape[:2], bool)
    for c in colours:
        m |= (img[:, :, 0] == c[0]) & (img[:, :, 1] == c[1]) & (img[:, :, 2] == c[2])
    return m


# A road-graph map's angled edges - 45 degrees and 2:1 - run on lines through
# tile corners (roadgraph/raster.py). Kerbed square by square they were a
# staircase (user, 2026-10-11: zig-zag kerbs, and paler road teeth - the road
# overlay, street_curbs_01_blend_street, is street3 grey on street2 tarmac).
# Laid as vanilla lays its own (read from Muldraugh's lot files, LOG
# 2026-10-11): every square the edge crosses is tarmac, with the kerb whose
# line fits it (the 45-degree ones, street_curbs_01_diag_2 on 2:1) and the
# ground beside the road painted over the part beyond the kerb by an overlay
# of the same shapes in that ground's own sheet. Out of town, no kerb.
SIDES = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
# The overlay sheet for each ground beside the road.
SIDE_SHEETS = {
    # Our pavement flags in vanilla's 88 shapes (the UK tile pack,
    # asset_creator/Tile_creator/src/overlays.py, 2026-10-12). Before it,
    # vanilla's pale concrete (street_curbs_01_blend_gravel) stood in: a
    # paler triangle beside each angled kerb.
    C.PALE_CONCRETE: "street_curbs_uk_01_blend_pavement",
    # Our grasses (UK tile pack, 2026-10-12), as the map tools now draw them.
    C.MEDIUM_GRASS: "street_curbs_uk_01_blend_medium_grass",
    C.DARK_GRASS: "street_curbs_uk_01_blend_dark_grass",
    C.LIGHT_GRASS: "street_curbs_uk_01_blend_light_grass",
    C.DIRT: "street_curbs_01_blend_dirt",
    C.SAND: "street_curbs_01_blend_sand",
}
DEFAULT_SIDE = C.MEDIUM_GRASS            # any other ground (a garden's, a field's) beside an angled edge


HALVES = {"NE": ((0, 0), (1, 1), (1, 0)), "SW": ((0, 0), (1, 1), (0, 1)),
          "NW": ((0, 1), (1, 0), (0, 0)), "SE": ((0, 1), (1, 0), (1, 1))}     # a square's road half by its corner


def _squares(pts) -> list:
    """The squares along a polyline in tiles, a sample a tile."""
    out = []
    for (fx0, fy0), (fx1, fy1) in zip(pts, pts[1:]):
        k = max(1, int(max(abs(fx1 - fx0), abs(fy1 - fy0))))
        out += [(int(fx0 + (fx1 - fx0) * t / k), int(fy0 + (fy1 - fy0) * t / k)) for t in range(k + 1)]
    return out


def _crossed(shape, H, W, blocked) -> dict:
    """{(x, y): its part inside `shape`, SS x SS} for each square (not
    `blocked`) that an angled edge of the shape - neither along the grid -
    passes through."""
    import shapely
    xs, ys = [], []
    for line in shapely.get_parts(shape.boundary):
        c = np.asarray(line.coords)
        for (x0, y0), (x1, y1) in zip(c, c[1:]):
            dx, dy = x1 - x0, y1 - y0
            if abs(dx) < 1e-6 or abs(dy) < 1e-6:
                continue
            n = int(max(abs(dx), abs(dy)) * SS * 2) + 1
            t = (np.arange(n) + 0.5) / n
            xs.append(x0 + dx * t)
            ys.append(y0 + dy * t)
    cand = set()
    if xs:
        pts = np.unique(np.stack([np.floor(np.concatenate(xs)), np.floor(np.concatenate(ys))], 1).astype(int), axis=0)
        cand = {(int(x), int(y)) for x, y in pts if 0 <= x < W and 0 <= y < H and not blocked[y, x]}
    out: dict = {}
    if cand:
        sq = np.array(sorted(cand))
        j, i = (np.mgrid[0:SS, 0:SS] + 0.5) / SS
        px = sq[:, 0, None, None] + i[None]
        py = sq[:, 1, None, None] + j[None]
        parts = shapely.contains_xy(shape, px, py)
        for (x, y), part in zip(map(tuple, sq), parts):
            if part.all() or not part.any():
                continue
            out[(x, y)] = part
    return out


def repaint_paths(ground, plots_npz: str) -> int:
    """Footpaths painted where plots2.py laid them (its `path` and
    `path_paved` masks: OSM's paths on vanilla's 16 directions) instead of
    Generate's raster of the raw line (user, 2026-10-13: too zig-zagged, and
    the garden boundaries along them followed the plots, not the paint).
    The old path - dirt or paving narrower than four squares, off the
    street - goes back to the ground beside it; the new one is paving or
    dirt as OSM says. In place; returns the squares painted."""
    if not os.path.exists(plots_npz):
        return 0
    from scipy import ndimage
    z = np.load(plots_npz)
    if "path" not in z:
        return 0
    H, W = ground.shape[:2]
    new, paved = z["path"][:H, :W], z["path_paved"][:H, :W]
    keep_off = z["street"][:H, :W] | z["building"][:H, :W] | is_colour(ground, (C.WATER,) + tuple(ROAD))
    way = is_colour(ground, (C.DIRT, PAVE)) & ~z["street"][:H, :W]
    old = way & ~ndimage.binary_opening(way, structure=np.ones((4, 4), bool)) & ~new
    if old.any():
        _d, (iy, ix) = ndimage.distance_transform_edt(old | new | keep_off, return_indices=True)
        ground[old] = ground[iy[old], ix[old]]
    paint = new & ~keep_off
    ground[paint & paved] = PAVE
    ground[paint & ~paved] = C.DIRT
    return int(paint.sum())


def angled_squares(state_path, ground, verge, buildings):
    """({(x, y): (road part, SS x SS bool; the ground beside, a colour;
    kerbed)}, {(x, y): (pavement part; the ground beyond, a colour)}).
    The first, each square a road-graph map's angled edge of tarmac crosses:
    the carriageway and its parking bays as the hand-over drew them
    (roadside.outlines, raster.junction_cuts), their edges neither along the
    grid nor within other road; and the staircases the graph did not draw (a
    car park's edge, painted as a thick line: pavement or verge squares with
    road on two neighbouring sides, in a run along the diagonal), their road
    half. Kerbed where the ground beside is pavement or verge. The second,
    each square the back of an angled pavement crosses where a garden's or a
    field's ground lies beyond (SIDE_SHEETS, not paving)."""
    import pickle
    import shapely
    from roadgraph import raster, roadside
    links, paths, proj, _owner, _half, _town, chosen = pickle.load(open(state_path, "rb"))
    bands = roadside.outlines(paths, links, chosen, roadside.handedness(proj))
    cuts = [t for _, t in raster.junction_cuts(paths, links)]
    # Unsurfaced links (a track: dirt) are not tarmac: their angled edges were
    # made road squares and their corners overlaid with vanilla's orange dirt
    # edge (QA in game, 2026-10-19: a saw-tooth along every UK dirt track).
    tarmac = shapely.union_all([s for s in [bands[i][0] if i in bands else raster.outline(p, links[i].width)
                                            for i, p in paths.items() if links[i].cls != "track"] + cuts
                                if not s.is_empty])
    shapely.prepare(tarmac)
    H, W = ground.shape[:2]
    road = is_colour(ground, ROAD)
    pave = is_colour(ground, (PAVE,))
    blocked = buildings if buildings is not None else np.zeros((H, W), bool)
    out = _crossed(tarmac, H, W, blocked)
    # The ground beside each: the commonest of its neighbours that is neither
    # road nor another edge square nor a building (none: left as it is).
    res: dict = {}
    for (x, y), part in out.items():
        seen: dict = {}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if (dx or dy) and 0 <= nx < W and 0 <= ny < H and (nx, ny) not in out \
                        and not road[ny, nx] and not blocked[ny, nx]:
                    key = "verge" if verge[ny, nx] else "pave" if pave[ny, nx] else tuple(int(v) for v in ground[ny, nx])
                    seen[key] = seen.get(key, 0) + 1
        if not seen:
            continue
        key = max(seen, key=seen.get)
        if key == "pave":
            res[(x, y)] = (part, PAVE, True)
        elif key == "verge":
            res[(x, y)] = (part, C.MEDIUM_GRASS, True)
        elif key != tuple(C.WATER):
            res[(x, y)] = (part, key if key in SIDE_SHEETS else DEFAULT_SIDE, False)
    # Staircases the graph did not draw: their road half. (The edge squares
    # found above count as the roadside here, not road: the pavement beyond
    # a 45-degree edge is no staircase.)
    side, solid = pave | verge, road.copy()
    for x, y in out:
        side[y, x], solid[y, x] = True, False
    for name, (a, b) in (("NE", ("N", "E")), ("NW", ("N", "W")), ("SE", ("S", "E")), ("SW", ("S", "W"))):
        m = side & ~blocked
        for k, (dx, dy) in SIDES.items():
            nb = np.zeros((H, W), bool)
            nb[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = \
                solid[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
            m &= nb if k in (a, b) else ~nb
        sx, sy = (1, 1) if name in ("NE", "SW") else (1, -1)
        run = np.zeros((H, W), bool)                   # a corner of the same kind on the next step either way
        for ex, ey in ((sx, sy), (-sx, -sy)):
            nb = np.zeros((H, W), bool)
            nb[max(0, -ey):H - max(0, ey), max(0, -ex):W - max(0, ex)] = \
                m[max(0, ey):H - max(0, -ey), max(0, ex):W - max(0, -ex)]
            run |= nb
        for y, x in zip(*np.nonzero(m & run)):
            if (int(x), int(y)) not in out:
                res[(int(x), int(y))] = (_cut(*HALVES[name]), C.MEDIUM_GRASS if verge[y, x] else PAVE, True)
    # The back of the pavement: the outer edge of every link's roadside,
    # where pavement lies inside and natural ground beyond.
    outer = shapely.union_all([o[2] for o in bands.values() if not o[2].is_empty] + cuts)
    shapely.prepare(outer)
    back: dict = {}
    for (x, y), part in _crossed(outer, H, W, blocked).items():
        if (x, y) in out or (x, y) in res:
            continue
        inside, beyond = {}, {}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if not (dx or dy) or not (0 <= nx < W and 0 <= ny < H) or road[ny, nx] or blocked[ny, nx]:
                    continue
                into = inside if shapely.contains_xy(outer, nx + 0.5, ny + 0.5) else beyond
                key = "pave" if pave[ny, nx] else "verge" if verge[ny, nx] else tuple(int(v) for v in ground[ny, nx])
                into[key] = into.get(key, 0) + 1
        if not inside or not beyond or max(inside, key=inside.get) != "pave":
            continue
        key = max(beyond, key=beyond.get)
        if key in SIDE_SHEETS and key != tuple(PAVE):
            back[(x, y)] = (part, key)
    return res, back


def main() -> int:
    map_dir = sys.argv[1]
    sigma = float(sys.argv[sys.argv.index("--sigma") + 1]) if "--sigma" in sys.argv else SIGMA
    picture = sys.argv[sys.argv.index("--picture") + 1] if "--picture" in sys.argv else None
    view = [int(v) for v in sys.argv[sys.argv.index("--box") + 1].split(",")] if "--box" in sys.argv else None
    name = os.path.basename(os.path.normpath(map_dir))
    gpath = os.path.join(map_dir, f"{name}_ground_base.bmp")
    # Only what Build reads: it makes <map>.bmp and <map>_veg.bmp from these
    # (knoxbuild/yards.py) every time.
    files = [gpath, os.path.join(map_dir, f"{name}_veg_base.bmp")]
    state_path = os.path.join(map_dir, f"{name}_kerbs_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    # Each file as it was before kerbs.py: kept beside it, and taken afresh
    # whenever another step (Generate, trees.py, sports.py) has written it
    # since this one last did.
    for f in files:
        keep = f.replace(".bmp", ".prekerbs.bmp")
        if not os.path.exists(keep) or state.get(os.path.basename(f)) != _digest(f):
            shutil.copy(f, keep)
    pristine = {f: np.array(Image.open(f.replace(".bmp", ".prekerbs.bmp")).convert("RGB")) for f in files}
    # 45-degree roads redrawn first, one width all along with a tight
    # pavement (diagonals.py); the rest works on that ground.
    from streets import diagonals
    from buildings import blocks as B
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    bb = info["bbox"]
    proj = B.Projector.build(bb["south"], bb["west"], bb["north"], bb["east"], info["meters_per_tile"],
                             info.get("rotation") or 0.0)
    plots_npz = os.path.join(map_dir, f"{name}_plots.npz")
    buildings = np.load(plots_npz)["building"] if os.path.exists(plots_npz) else None
    street_before = is_colour(pristine[gpath], ROAD + (PAVE,))
    ground, diag_zone, stretches, stretch_lines = diagonals.clean(pristine[gpath], map_dir, info, proj, ROAD, PAVE,
                                                   is_colour, buildings)
    # Tarmac touching only at a corner joined edge to edge, and lane ends
    # left leading nowhere taken off (tidy.py).
    from streets import tidy
    tiles_npz = os.path.join(map_dir, f"{name}_roadgraph_tiles.npz")
    drawn = np.load(tiles_npz)["owner"] != 0 if os.path.exists(tiles_npz) else None
    ground, tidied = tidy.roads(ground, ROAD, C.DARKEST_ASPHALT, PAVE, is_colour, buildings, C.WATER, drawn)
    H, W = ground.shape[:2]
    repainted_paths = repaint_paths(ground, os.path.join(map_dir, f"{name}_plots.npz"))
    print(f"footpaths painted where the plots laid them: {repainted_paths} squares")
    # Road bridges over water: the deck's pavements over the river, the
    # river round it cleared of land (piers, strips), railings along its
    # edges (bridges.py) - before kerbing, so the new pavement is kerbed.
    from streets import bridges
    over = bridges.lay(map_dir, ground, buildings, ROAD, PAVE, C.WATER)
    if over is not None:
        print(f"road bridges over water: {over.bridges} ways; pavement laid over the water {int(over.pave.sum())} "
              f"squares, land in the river made water {int(over.wet.sum())}, railing tiles {len(over.rails)}")
    # A road-graph map's grass verges (roadgraph/roadside.py): kerbed like a
    # pavement, the kerb on the verge's square, its edge kept square.
    rs_path = os.path.join(map_dir, f"{name}_roadside.npz")
    verge = np.zeros((H, W), bool)
    if os.path.exists(rs_path):
        v = np.load(rs_path)["verge"]
        verge[:min(H, v.shape[0]), :min(W, v.shape[1])] = v[:H, :W]
    # A road-graph map draws its roads exactly (corners cut at junctions,
    # angled edges through tile corners): not rounded here, and its angled
    # edges laid as vanilla's (angled_squares).
    graph_map = bool(info.get("road_graph"))
    angled: dict = {}
    back: dict = {}
    state_pkl = os.path.join(map_dir, f"{name}_roadgraph_state.pkl")
    if graph_map and os.path.exists(state_pkl):
        angled, back = angled_squares(state_pkl, ground, verge, buildings)
    road0 = is_colour(ground, ROAD)
    pave0 = is_colour(ground, (PAVE,))
    street = road0 | pave0
    verge &= ~street
    edge_sq = np.zeros((H, W), bool)                 # angled edge squares: laid from ANGLED_TABLE, not fitted
    for (x, y) in angled:
        edge_sq[y, x] = True
    near_pave = ndimage.binary_dilation(pave0, iterations=2)
    # Narrow tarmac (a drive, a lane: under THIN_TILES across) keeps its
    # squares: smoothed, it would melt into the pavement beside it.
    disc = np.hypot(*np.mgrid[-2:3, -2:3]) <= 2.0
    thin = road0 & ~ndimage.binary_opening(road0, structure=disc)

    no_kerb = ndimage.binary_dilation(diag_zone, iterations=1)
    names, pts, pen, verges = load_pieces()
    memo: dict = {}
    road_new = road0.copy()
    host = {}                                        # (x, y) -> (kerb, overlay or None, overlay sheet or None)
    fits = []
    for by in range(0, H, BLOCK):
        for bx in range(0, W, BLOCK):
            x0, y0 = max(0, bx - MARGIN), max(0, by - MARGIN)
            x1, y1 = min(W, bx + BLOCK + MARGIN), min(H, by + BLOCK + MARGIN)
            r = road0[y0:y1, x0:x1]
            if not r.any() or not (pave0[y0:y1, x0:x1] | verge[y0:y1, x0:x1]).any():
                continue
            if graph_map:
                # The road as drawn, square by square; its angled edge
                # squares left out (neither road nor kerbed side), so no
                # kerb is fitted along them - they are laid below.
                out = edge_sq[y0:y1, x0:x1]
                fine = np.kron(r & ~out, np.ones((SS, SS), bool))
                hold = np.zeros_like(fine)
                rounded = np.zeros(r.shape, bool)
                side_f = np.kron((street[y0:y1, x0:x1] | verge[y0:y1, x0:x1]) & ~out, np.ones((SS, SS), bool))
            else:
                # The road, smoothed at SS points a tile, kept within the street.
                fine = np.kron(r.astype(np.float32), np.ones((SS, SS), np.float32))
                fine = ndimage.gaussian_filter(fine, sigma * SS) > 0.5
                fine &= np.kron(street[y0:y1, x0:x1], np.ones((SS, SS), bool))
                # Away from pavement the road keeps its squares (a lane's verge).
                hold = np.kron(r & (~near_pave[y0:y1, x0:x1] | thin[y0:y1, x0:x1]), np.ones((SS, SS), bool))
                fine |= hold
                # Curves sparingly (user, 2026-10-08): only a change worth a
                # real corner is kept - each connected patch of road gained or
                # lost must come to MIN_ROUND_TILES - and everywhere else the
                # edge stays on the grid. A one-tile step rounded came out as
                # saw-teeth in game.
                orig = np.kron(r, np.ones((SS, SS), bool))
                diff = fine ^ orig
                lab, n = ndimage.label(diff)
                if n:
                    sizes = ndimage.sum(diff, lab, index=np.arange(1, n + 1)) / (SS * SS)
                    small = np.zeros(n + 1, bool)
                    small[1:] = sizes < MIN_ROUND_TILES
                    fine = np.where(small[lab], orig, fine)
                # Where the edge was rounded (or is within a square of it), curved
                # and 2:1 pieces may be used; elsewhere straight and corner ones.
                rounded = ndimage.binary_dilation((fine ^ orig).reshape(y1 - y0, SS, x1 - x0, SS).any(axis=(1, 3)),
                                                  iterations=1)
                side_f = np.kron(street[y0:y1, x0:x1] | verge[y0:y1, x0:x1], np.ones((SS, SS), bool))
            cover = fine.reshape(y1 - y0, SS, x1 - x0, SS).mean(axis=(1, 3))
            # Edge points: between a road sub-point and the pavement (or verge) one beside it.
            pave_f = side_f & ~fine
            per = {}
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb_p = np.zeros_like(fine)
                sl_to = (slice(max(0, -dy), fine.shape[0] - max(0, dy)), slice(max(0, -dx), fine.shape[1] - max(0, dx)))
                sl_fr = (slice(max(0, dy), fine.shape[0] - max(0, -dy)), slice(max(0, dx), fine.shape[1] - max(0, -dx)))
                nb_p[sl_to] = pave_f[sl_fr]
                ys, xs = np.nonzero(fine & nb_p)
                # the boundary point, and a hair past it into the pavement: that square carries the kerb
                py, px = ys + 0.5 + dy * 0.5, xs + 0.5 + dx * 0.5
                hy, hx = (py + dy * 0.05 * SS) // SS, (px + dx * 0.05 * SS) // SS
                for a, b, yy, xx in zip(hy.astype(int), hx.astype(int), py, px):
                    per.setdefault((b, a), []).append((xx / SS - b, yy / SS - a))
            for (lx, ly), p in per.items():
                gx, gy = x0 + lx, y0 + ly
                if not (bx <= gx < bx + BLOCK and by <= gy < by + BLOCK) or len(p) < 3:
                    continue
                if no_kerb[gy, gx]:
                    continue                         # a diagonal road's edge: no kerbs (user, 2026-10-09)
                # An edge held on the grid (narrow tarmac, a rural lane) is a
                # staircase: straight and corner pieces only, as before.
                held = bool(hold[ly * SS:(ly + 1) * SS, lx * SS:(lx + 1) * SS].any()
                            or thin[max(0, gy - 1):gy + 2, max(0, gx - 1):gx + 2].any()
                            or not rounded[ly, lx])
                score, kerb = best_kerb(np.array(p), names, pts, pen, memo, square_only=held)
                if score > WORST_FIT:
                    continue
                part = fine[ly * SS:(ly + 1) * SS, lx * SS:(lx + 1) * SS]
                overlay = best_overlay(part, verges) if part.mean() >= MIN_OVERLAY else None
                host[(gx, gy)] = (kerb, overlay, None)
                fits.append(score)
            # Squares: road where mostly road and not carrying a kerb.
            inner = cover[by - y0:by - y0 + min(BLOCK, H - by), bx - x0:bx - x0 + min(BLOCK, W - bx)]
            road_new[by:by + inner.shape[0], bx:bx + inner.shape[1]] = (
                inner > 0.5) & street[by:by + inner.shape[0], bx:bx + inner.shape[1]]
    # A road-graph map's angled edge squares, as vanilla lays them: tarmac,
    # the kerb, overlay and stubs from ANGLED_TABLE where the roadside is
    # kerbed, else the ground's overlay cut exactly at the edge; a sliver of
    # road (off the lattice) left to the roadside.
    on_road = set()
    side_ground = {}
    stub_tiles = {}                                  # (x, y) -> vanilla's stub beside an angled kerb
    unmatched = 0
    for (x, y), (part, colour, kerbed) in angled.items():
        if part.mean() < MIN_OVERLAY:
            side_ground[(x, y)] = colour
            continue
        row = angled_piece(part) if kerbed else None
        unmatched += kerbed and row is None
        if row:
            kerb, overlay, stubs = row
            for dx, dy, piece in stubs:
                sx, sy = x + dx, y + dy
                if 0 <= sx < W and 0 <= sy < H and (sx, sy) not in angled and (sx, sy) not in host \
                        and not (buildings is not None and buildings[sy, sx]):
                    stub_tiles.setdefault((sx, sy), piece)
        else:
            kerb, rest = None, ~part
            overlay = best_overlay(rest, verges) if rest.mean() >= MIN_OVERLAY else None
        host[(x, y)] = (kerb, overlay, SIDE_SHEETS[colour])
        on_road.add((x, y))
    # The back of an angled pavement, as vanilla lays it: paving, the ground
    # beyond painted over the part past the edge (a sliver of pavement: left
    # to the ground).
    back_paved = np.zeros((H, W), bool)
    for (x, y), (part, colour) in back.items():
        if (x, y) in host or part.mean() < MIN_OVERLAY:
            continue
        rest = ~part
        host[(x, y)] = (None, best_overlay(rest, verges) if rest.mean() >= MIN_OVERLAY else None, SIDE_SHEETS[colour])
        back_paved[y, x] = True
    for (x, y) in host:
        if (x, y) not in on_road:
            road_new[y, x] = False                   # a kerb stands on pavement, as KnoxMap's do
    for (x, y) in on_road:
        road_new[y, x] = True
    for (x, y) in side_ground:
        road_new[y, x] = False
    became_pave = road0 & ~road_new
    became_road = road_new & ~road0

    # --- ground ---------------------------------------------------------------------------
    # A square that becomes road takes the tarmac of the road beside it.
    idx = ndimage.distance_transform_edt(~road0, return_distances=False, return_indices=True)
    outputs = {}                                     # path -> new image, written at the end
    for path in files[:1]:
        img = ground.copy()
        img[:H, :W][became_pave] = PAVE
        for (x, y), colour in side_ground.items():
            img[y, x] = colour
        img[:H, :W][back_paved] = PAVE
        ys, xs = np.nonzero(became_road)
        img[ys, xs] = ground[idx[0][ys, xs], idx[1][ys, xs]]
        outputs[path] = img

    # --- vegetation: the rule kerbs off, and lane lines off pavement ------------------------
    cleared = 0
    hedges_moved = hedges_dropped = 0
    for vp in files[1:]:
        veg = pristine[vp].copy()
        off = is_colour(veg[:H, :W], RULE_KERBS) | (is_colour(veg[:H, :W], LANE_LINES) & ~road_new)
        # Nothing grows on a square the redrawn diagonals made street.
        off |= (road_new | (street & ~street_before)) & ~street_before
        # A hedge mapped in OSM is drawn as bushes (Generate); where the
        # straightening moved the road and the hedge apart it stood on the
        # pavement or a paved yard. It goes to the nearest unpaved square
        # beside it (HEDGE_REACH), else it is dropped.
        paved = road_new | street | back_paved
        hedge = is_colour(veg[:H, :W], (C.BUSHES,)) & (paved | verge)
        if hedge.any():
            # Past the verge too: it is part of the street (principle 15).
            dist, (hy, hx) = ndimage.distance_transform_edt(paved | verge, return_indices=True)
            blocked_sq = buildings if buildings is not None else np.zeros((H, W), bool)
            ys, xs = np.nonzero(hedge & (dist <= HEDGE_REACH))
            ty, tx = hy[ys, xs], hx[ys, xs]
            free = (veg[ty, tx].max(axis=1) == 0) & ~blocked_sq[ty, tx]
            veg[ty[free], tx[free]] = C.BUSHES
            hedges_moved = int(free.sum())
            hedges_dropped = int(hedge.sum()) - hedges_moved
            off |= hedge
        # Nor anything else that grows (PLAN principle 15): road and paving
        # keep only the street's own pieces, which all have red 12 on the
        # vegetation map (kerbs, lines, grime, lamps, hydrants, bins...).
        growing = (veg[:H, :W].max(axis=2) > 0) & (veg[:H, :W, 0] != 12)
        off |= growing & paved
        # A verge is mown grass: its street trees stay, bushes and long grass go.
        off |= verge & is_colour(veg[:H, :W], (C.BUSHES, C.BUSHES_TREES_DARK_GRASS, C.DENSE_BUSHES_GRASS,
                                               C.GRASS_ON_DARK, C.LOT_OF_GRASS_AND_TREES))
        # Nothing grows on the river round a bridge, nor on its deck (bridges.py).
        if over is not None:
            off |= over.corridor & (over.water | over.deck) & (veg[:H, :W].max(axis=2) > 0) \
                & ~(over.deck & (veg[:H, :W, 0] == 12))
        cleared += int(off.sum())
        veg[:H, :W][off] = 0
        outputs[vp] = veg

    # --- boundaries on the redrawn diagonals' new pavement --------------------------------
    # A garden fence or hedge drawn along the old verge would stand on the
    # pavement now: left out (kept aside, and put back first on a rerun).
    fp = os.path.join(map_dir, f"{name}_fences.geojson")
    aside_path = os.path.join(map_dir, f"{name}_fences.kerbs_aside.json")
    fences_out = None
    fronts_report = (0, 0, 0)
    if os.path.exists(fp):
        fc = json.load(open(fp, encoding="utf-8"))
        # The lines set aside last time go back in first - but only when the
        # fences file is still the one this step wrote: plots2.py rewrites it
        # whole, and old lines from before that would come back as duplicates.
        same_file = state.get(os.path.basename(fp)) == _digest(fp)
        aside = json.load(open(aside_path, encoding="utf-8")) if os.path.exists(aside_path) and same_file else []
        from streets import fronts
        feats = [ft for ft in fc.get("features", []) + aside
                 if (ft.get("properties") or {}).get("source") != fronts.SOURCE]
        new_street = (road_new | street) & ~street_before
        kept, aside = [], []
        bridge_aside = 0
        for ft in feats:
            g = ft.get("geometry") or {}
            src = (ft.get("properties") or {}).get("source")
            # On the river or a deck round a bridge, whatever drew it (bridges.py).
            if over is not None and over.bridges and g.get("type") == "LineString":
                if bridges.set_aside(_squares([proj.to_px(la, lo) for lo, la in g["coordinates"]]), over):
                    aside.append(ft)
                    bridge_aside += 1
                    continue
            # A plot's own boundary is drawn on the street's edge on purpose
            # (plots2.py): sampled there it reads as street - not set aside.
            if g.get("type") == "LineString" and src not in ("sports", "plots"):
                pts = [proj.to_px(la, lo) for lo, la in g["coordinates"]]
                on = 0
                for (fx0, fy0), (fx1, fy1) in zip(pts, pts[1:]):
                    k = max(1, int(max(abs(fx1 - fx0), abs(fy1 - fy0))))
                    for t in range(k + 1):
                        qx, qy = int(fx0 + (fx1 - fx0) * t / k), int(fy0 + (fy1 - fy0) * t / k)
                        on += bool(0 <= qy < H and 0 <= qx < W and new_street[qy, qx])
                if on >= 3:
                    aside.append(ft)
                    continue
            kept.append(ft)
        # Garden fronts along the redrawn diagonals: one line along the back
        # of the pavement instead of each garden's steps (fronts.py).
        kept, front_aside, front_new, front_bushes = fronts.redraw(
            kept, stretch_lines, road_new | street,
            buildings if buildings is not None else np.zeros((H, W), bool), proj)
        aside += front_aside
        fc["features"] = kept + front_new
        fences_out = (fc, aside)
        vp = files[1]
        for bx_, by_ in front_bushes:
            if outputs[vp][by_, bx_].max() == 0:
                outputs[vp][by_, bx_] = C.BUSHES
        fronts_report = (len(front_aside), len(front_new), len(front_bushes))

    # --- tiles ----------------------------------------------------------------------------
    tiles = []
    for (x, y), (kerb, overlay, sheet) in sorted(host.items()):
        if overlay is not None:
            tiles.append([int(x), int(y), 0, OVERLAY_LAYER, f"{sheet or OVERLAY_SHEET}_{overlay}"])
        if kerb is not None:
            tiles.append([int(x), int(y), 0, KERB_LAYER, uk_kerb(kerb)])
    for (x, y), piece in sorted(stub_tiles.items()):
        tiles.append([int(x), int(y), 0, KERB_LAYER, uk_kerb(piece)])
    # The bridges' railings along their decks over the water (bridges.py).
    if over is not None:
        tiles += over.rails
    # A road-graph map's UK markings and signals (roadgraph/markings.py, from
    # the hand-over): paint only on what is still road, signal posts, signs and
    # the street furniture (any Furniture layer: furniture.py) only off it.
    marks_path = os.path.join(map_dir, f"{name}_markings.json")
    marks_laid = 0
    if graph_map and os.path.exists(marks_path):
        from roadgraph.furniture import partner, whole_poles
        kept = [[x, y, z, layer, tile] for x, y, z, layer, tile in json.load(open(marks_path, encoding="utf-8"))
                if 0 <= y < H and 0 <= x < W and (z > 0 or (road_new[y, x] != layer.startswith("Furniture")
                                                         and (layer.startswith("Furniture") or (x, y) not in host)))]
        # (A storey up - a telegraph pole's top, its wires - stands over whatever is below.)
        # Nothing of the street furniture on a building's square: the building's
        # lot replaces it, and half a bench left outside is an object the game
        # cannot remove (LOG 2026-10-18: a session hung on it).
        if buildings is not None:
            kept = [r for r in kept if r[2] > 0 or not r[3].startswith("Furniture") or not buildings[r[1], r[0]]]
        # A two-tile thing (a town sign, a bench) goes whole or not at all;
        # so does a telegraph pole, with its top and wires.
        kept = whole_poles(kept)
        here = {(x, y, tile) for x, y, _z, _layer, tile in kept}
        for row in kept:
            other = partner(row[0], row[1], row[4])
            if other is None or other in here:
                tiles.append(row)
                marks_laid += 1
    sp = os.path.join(map_dir, f"{name}_structures.json")
    data = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {"bridges": 0, "monuments": 0, "tiles": []}
    rows = data.get("tiles") or []
    added = data.get("kerbs_added") or 0
    if added:
        # The kerb tiles an earlier run of this script appended come off the
        # end before today's are added, so reruns don't pile up (measure_kerbs
        # tags kerbs_added with how many). If something else has rewritten
        # structures.json shorter since (a fresh Generate, a step that trims
        # its own tiles the same way) that count no longer fits the list, and
        # blindly slicing would wrap round and drop or duplicate someone
        # else's tiles instead of just this script's own.
        if added > len(rows):
            raise ValueError(f"{sp}: kerbs_added={added} but only {len(rows)} tiles are recorded - "
                             "structures.json looks like it was rewritten since the last kerbs.py run; "
                             "rerun the step that trims its own tiles, or clear kerbs_added, before this one")
        rows = rows[:len(rows) - added]
    if over is not None and over.bridges:
        rows = bridges.drop_old_rails(rows, over.corridor)    # Generate's, along its own deck
    data["tiles"] = rows + tiles
    data["kerbs_added"] = len(tiles)

    # Everything worked out: now written, each file whole or not at all, and
    # the record of what was written last of all.
    for path, img in outputs.items():
        Image.fromarray(img).save(path + ".tmp", format="BMP")
        os.replace(path + ".tmp", path)
    with open(sp + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(sp + ".tmp", sp)
    if fences_out is not None:
        for path, obj in ((fp, fences_out[0]), (aside_path, fences_out[1])):
            with open(path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(obj, fh)
            os.replace(path + ".tmp", path)
    with open(state_path + ".tmp", "w") as fh:
        record = {os.path.basename(f): _digest(f) for f in files}
        if fences_out is not None:
            record[os.path.basename(fp)] = _digest(fp)
        json.dump(record, fh)
    os.replace(state_path + ".tmp", state_path)

    diag = sum(1 for k, _, _ in host.values() if k and "_diag" in k)
    print("roads tidied:", tidied)
    if graph_map:
        print(f"road markings and signals laid: {marks_laid}")
    if angled:
        kerbed = sum(1 for p in on_road if host[p][0] is not None)
        print(f"angled road edges: {len(angled)} squares, {len(on_road)} laid as tarmac with the roadside over "
              f"its part ({kerbed} kerbed, {len(stub_tiles)} stubs beside, {unmatched} kerbed side off the "
              f"lattice left unkerbed), {len(side_ground)} left to the roadside; back of the pavement: "
              f"{int(back_paved.sum())} squares paved with the ground beyond over their part; "
              f"sheets {sorted({SIDE_SHEETS[c].replace('street_curbs_01_blend_', '') for _, c, _ in angled.values()})}")
    print(f"diagonal stretches redrawn: {stretches}; boundaries left off their new pavement: "
          f"{len(fences_out[1]) if fences_out else 0} ({bridge_aside if fences_out else 0} on a bridge's river or deck)")
    print(f"kerbed squares: {sum(k is not None for k, _, _ in host.values())} ({diag} on curves or 2:1 runs); "
          f"overlays: {sum(o is not None for _, o, _ in host.values())}")
    print(f"fit median {np.median(fits):.3f}, worst {max(fits):.3f}; pieces tried: {len(memo)} edge shapes")
    print(f"garden fronts along the diagonals: {fronts_report[0]} stepped pieces set aside, "
          f"{fronts_report[1]} lines and {fronts_report[2]} hedge bushes drawn along the pavement")
    print(f"hedges off the paving: {hedges_moved} squares moved beside it, {hedges_dropped} dropped")
    print(f"squares road -> pavement {int(became_pave.sum())}, pavement -> road {int(became_road.sum())}; "
          f"rule kerbs and stray lines cleared: {cleared // 2}")

    if picture:
        top = Image.new("RGB", (W, H), (70, 100, 60))
        tp = np.array(top)
        tp[street & ~road_new] = (175, 175, 170)
        tp[road_new] = (70, 70, 75)
        for (x, y) in host:
            k = host[(x, y)][0]
            tp[y, x] = (120, 160, 220) if k is None else (240, 240, 240) if "_diag" not in k else (230, 60, 60)
        img = Image.fromarray(tp)
        if view:
            vx, vy, vw, vh = view
            img = img.crop((vx, vy, vx + vw, vy + vh))
        img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(picture)
        print("wrote", picture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
