"""A big landmark's description written from a few lists: plan_kit.build(spec).

A building too big to describe by hand (the Palace of Westminster: 144 x 292
squares, ten courts, thousands of rooms) gets a small Python file of its own,
landmarks/<entry>_<name>_plan.py, holding only what is that building's: which
OSM pieces it is, its state rooms and corridors read off a plan, its doors to
the outside, its furniture, its towers. This kit does the rest, the same way
for any such building:
  - the footprint: every piece the block step laid inside the hull of the
    CORE pieces, as squares; slivers between pieces closed, the courts kept
    open, a piece standing apart left out (members, mask_of);
  - the heights and roofs from the LiDAR (DSM - DTM): each square's eaves
    storeys and roof form (lidar_storeys), evened out range by range
    (range_heights: the spec's RANGES first, read off a plan, then the
    largest rectangle again and again for what they leave), blocks standing
    above their range (raised_parts), the spec's PAVILIONS and RANGE_FIXES;
  - the rooms on every storey: THEMED rooms, the towers, CORRIDORS, then
    offices on a grid, with doors until every room can be reached (plan_level,
    doors_for); linenfold in the PANELLED rooms;
  - the outside: windows in bays on every outside wall, buttresses and
    parapets on the faces the camera sees, pinnacles on the skyline, turrets
    on the tall ranges', the pavilions' and the big raised parts' corners, a
    pinnacled parapet round the pavilions (exterior);
and writes landmarks/<OUT>, the description Build reads.

    python landmarks/<entry>_<name>_plan.py [<map dir>]     (KnoxMap's Python, from listed_building_maker/)

The spec is the landmark's module: its UPPER_CASE constants (the tuning ones
below are defaults it may override) and two optional hooks for what is its
own: `tower_exterior(tower_box, st) -> (windows, pieces)` (a clock face, a
spire, a crown) and `extra_stairs(tower_box) -> [[level, x, y, dir]]`.
The worked example is landmarks/1226284_parliament_plan.py.
"""
from __future__ import annotations

import json
import os
import sys
import types
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # asset_creator: acpaths
import acpaths  # noqa: E402
KNOX = acpaths.KNOX
sys.path.insert(0, KNOX)

# Defaults a spec may override. The values were tuned on the Palace of Westminster (1226284).
GAP = 2                                # squares: a gap this wide between two pieces is closed
SLIVER = 30                            # holes smaller than this are gaps between pieces, not courts
OFFICE_CELL = 8                        # squares: the grid the plain rooms are cut on
SMALL_ROOM = 9                         # squares: a scrap smaller than this joins its neighbour
# The ranges' storeys from the LiDAR (EA DSM - DTM, the area's lidar/ tiles): each square
# its eaves height (a low percentile of the roof round it: a pitched range's eaves, not its
# ridge) over the game's storey.
STOREY_M = 3.0                         # the game's storey, as the towers are measured
EAVES_PCT, EAVES_WIN = 25, 7           # percentile, window (squares) for a square's eaves
SMOOTH_WIN = 5                         # squares: storeys evened out by the commonest round each
FLAT, RIDGE_NS, RIDGE_WE = 1, 2, 3     # a roof square's form, from the LiDAR's slope (roofs.FLAT_SLOPE_DEG)
FORM_WIN = 7                           # squares: a square's form the commonest round it
RANGE_MIN = 20                         # squares: a rectangle smaller than this is a scrap of the range beside it
PITCHED_SHARE = 0.45                   # of a range's squares steeper than roofs.FLAT_SLOPE_DEG: pitched (roofs.py)
RANGE_MAX = 8                          # storeys a range may have (the towers apart)
SHIFT_MAX = 12                         # m: the footprint is tried this far each way on the LiDAR
SPLIT_BY = 2                           # storeys: a range whose two ends differ this much is cut at the step
SPLIT_MIN = 6                          # squares: neither side of the cut shorter than this
TURN_MAX = 4                           # degrees: and turned this far
TALL_M = 6                             # m: LiDAR ground this high is a building, for the shift
RAISE_BY = 2                           # storeys a block must stand above its range to be a part of its own
RAISE_MIN = 16                         # squares: a raised block at least this big (a central tower: about 18 x 18)
TOWER_CLEAR = 4                        # squares round the towers left to them
RAISE_MAX = 28                         # storeys: the raised parts' ceiling (a fleche can stand on top)
RAISE_WIDTH = 5                        # squares: a raised block at least this wide (a ridge strip is 3-4: Parliament's spine)
RAISE_TURRET = 8                       # squares: a raised part this wide on its shorter side has corner turrets
PAVILION_RISE = 3                      # storeys a PAVILIONS block rises over its range
TURRET_RANGE = 20                      # squares: a range this long, TURRET_FROM storeys or more, has corner turrets
TURRET_FROM = 5
# How the parts are found. "ranges": the spec's RANGES, then the largest rectangles, then blocks
# standing above them (raised_parts). "splits": cut from the LiDAR - the footprint split again and
# again where a straight cut best lowers the parts' miss, until each part is within SPLIT_TOL (the
# better match at every part count when tried on Parliament, LOG #W10); light wells from the ground.
PARTS = "ranges"
SPLIT_TOL = 2.5                        # m: a part is cut again while it misses the LiDAR by more, on average
SPLIT_SIDE = 4                         # squares: no part narrower
SPLIT_ONE_RANGE = 16                   # squares: a pitched part this narrow is one range, not cut along its ridge
GROUND_M = 2.5                         # m: LiDAR lower than this is ground (a court, a light well)
GAME_RISE_M = 3.0                      # m: the game's pitched roof rises a storey over 3 squares (45 degrees)
TALL_SLOPES = (0.6, 1.0, 1.7)          # m a square: real roofs tried beyond the game's (30, 45, 60 degrees, uncapped)
TALL_ROOF_M = 5.5                      # m: a part whose real roof rises more wants a roof piece of its own
# Widths (squares) the hall roof pieces come in (gothic_hall_uk_01: hall_roof_w<W>, its _dormer and
# hall_gable_w<W>; ridge north-south, standing on the hall's flat roof floor): a tall roof of one
# of these widths gets them, the game's own roof flat under them.
HALL_ROOF_WIDTHS = ()
HALL_RIDGE_M = {16: 10.25, 18: 9.15}   # m: their ridges over the roof floor (the lanterns stand there)
HALL_DORMER_EVERY = 4                  # segments: a lucarne every so many
PITCH_GIVE = 1.15                      # a pitched roof's miss at most this share over the flat one's
TALL_GAIN = 0.75                       # a tall roof's miss at most this share of the game roof's
TALL_MAX_SIDE = 20                     # squares: a tall roof spans one range (wider: two ranges side by side)
# The windows in the bays (gothic_revival_windows_uk_01; a spec may name others: two squares wide,
# the principal floor's in two halves, low and high) and a panel at the foot of a storey between
# one window and the one below (an edge piece, or None).
WIN, WIN_LOW, WIN_HIGH, WIN_PANEL = "perp_sq2", "perp_sq2_low", "perp_sq2_high", None
# The real floors as bands of game storeys: [(name, first storey, storeys)]. A real floor is far
# taller than the game's 3 m storey (Parliament's principal floor ~9 m), so the outside's height
# (the LiDAR's) is kept and each floor spans a band: its rooms laid on the band's first storey, the
# storeys above it open (the same rooms, no floor: tall rooms), but for the stair halls and runs,
# which keep a floor on every storey; one tall window a floor. Storeys past the last band are a
# floor each. Empty: every storey a floor (the old way).
FLOOR_BANDS = ()
# Where a floor is ({band name: [[x0, y0, x1, y1], ...]}, inclusive): a band named here is laid
# only inside its areas (the building's own plan: Parliament's second floor is a strip on the river
# front); elsewhere its storeys go on as the floor below (open: the tall floor and its roof space).
FLOOR_AREAS = {}


def floor_areas(s, shape):
    """{band's first storey: mask of where that floor is} for the bands FLOOR_AREAS names."""
    out = {}
    for name, first, _n in s.FLOOR_BANDS:
        if name in s.FLOOR_AREAS:
            a = np.zeros(shape, bool)
            for x0, y0, x1, y1 in s.FLOOR_AREAS[name]:
                a[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True
            out[first] = a
    return out


def governing(lvl, s, areas, shape):
    """Per square, the first storey of the floor storey `lvl` belongs to: its band's, or where
    that floor is not (FLOOR_AREAS), the floor below's, and so on down."""
    firsts = [first for _n, first, _k in s.FLOOR_BANDS]
    g = np.full(shape, band_of(lvl, s)[0], int)
    for first in sorted(areas, reverse=True):
        if first > lvl:
            continue
        below = max([f for f in firsts if f < first], default=0)
        g = np.where((g == first) & ~areas[first], below, g)
    return g


def band_of(lvl, s):
    """(first storey, storeys) of the floor band storey `lvl` is in."""
    for _name, first, n in s.FLOOR_BANDS:
        if first <= lvl < first + n:
            return first, n
    return lvl, 1


def floor_windows(lv, s, base=0):
    """[(storey, window)] and the storeys with a panel at their foot, for a wall `lv` storeys
    high from storey `base`: one tall window a floor (low and high halves over its first two
    storeys, a window a storey above them), panels at the foot of each floor over another and of
    each storey above a floor's tall window."""
    if not s.FLOOR_BANDS:
        wins = [(0, s.WIN)] if base == 0 else []
        if lv >= 3 and base == 0:
            wins += [(1, s.WIN_LOW), (2, s.WIN_HIGH)]
        wins += [(lvl, s.WIN) for lvl in range(max(base, 3 if lv >= 3 and base == 0 else 1), lv)]
        if base > 0:
            wins = [(lvl, s.WIN) for lvl in range(base, lv)]
        panels = [lvl for lvl in range(max(1, base + 1 if base else 1), lv) if not (lv >= 3 and lvl == 2 and base == 0)]
        return wins, panels
    wins, panels, lvl = [], [], base
    while lvl < lv:
        first, n = band_of(lvl, s)
        here = [k for k in range(max(first, base), min(first + n, lv))]
        if len(here) >= 2 and here[0] == first:
            wins += [(here[0], s.WIN_LOW), (here[1], s.WIN_HIGH)] + [(k, s.WIN) for k in here[2:]]
            panels += ([here[0]] if here[0] > 0 else []) + here[2:]
        else:
            wins += [(k, s.WIN) for k in here]
            panels += [k for k in here if k > 0]
        lvl = first + n
    return wins, panels
BAY = 4                                # Barry's rhythm: buttress pier, a 2-square window, a panelled square
TOWER_WIN_FROM = 9                     # the towers' own windows start above the ranges

# What a spec may leave out.
OPTIONAL = {"THEMED": [], "CORRIDORS": [], "OUTSIDE_DOORS": [], "STAIRS": [], "STAIRWELLS": [], "FLOORS": {},
            "FLOOR_DEFAULT": None, "PANELLED": set(), "FURNITURE": [], "EDGES": [], "PAVILIONS": [],
            "RANGE_FIXES": [], "RANGES": [], "TURRETS": [], "ROOF_DETAIL": False, "ROOF_PIECES": [], "PAVILION_PINNACLE": "pavilion_pinnacle", "TOWERS": {}, "TOWER_STOREYS": {}, "CLEAR_ROOF": None,
            "FULL_HEIGHT": True, "BLANK": True}


def settings(spec):
    """The kit's defaults, OPTIONAL's empties, then the spec's own names over them."""
    s = {k: v for k, v in globals().items() if k.isupper() and k not in ("HERE", "ROOT", "KNOX", "OPTIONAL")}
    s.update(OPTIONAL)
    s.update({k: v for k, v in vars(spec).items() if not k.startswith("_")})
    return types.SimpleNamespace(**s)


def projector(map_dir):
    from generator.rendering.projection import Projector
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    b = info["bbox"]
    return name, Projector.build(b["south"], b["west"], b["north"], b["east"], 1.0, info["rotation"])


def members(map_dir, s):
    """The pieces the block step laid that are the building: every piece of a CORE
    OSM id, and any other whose middle stands in their convex hull. Returns their
    ids, their union (map tiles) and each tower's own outline."""
    from shapely.geometry import shape
    from shapely.ops import transform, unary_union
    name, proj = projector(map_dir)
    feats = json.load(open(os.path.join(map_dir, f"{name}_buildings.geojson"), encoding="utf-8"))["features"]
    geo = {str(f["properties"]["osm_id"]): transform(lambda x, y, z=None: proj.to_px(y, x), shape(f["geometry"]))
           for f in feats}
    root = lambda k: k.split("p")[0].split("u")[0]  # noqa: E731
    hull = unary_union([g for k, g in geo.items() if root(k) in s.CORE]).convex_hull
    keep = sorted(k for k, g in geo.items() if root(k) in s.CORE or hull.contains(g.representative_point()))
    towers = {oid: unary_union([g for k, g in geo.items() if root(k) == oid]) for oid in s.TOWERS.values()}
    return keep, unary_union([geo[k] for k in keep]), towers


def mask_of(outline, s):
    """The outline as squares (True: the building's), with its origin in map tiles."""
    from scipy import ndimage as ndi
    from shapely import contains_xy
    x0, y0, x1, y1 = (int(round(v)) for v in outline.bounds)
    xs, ys = np.meshgrid(np.arange(x1 - x0) + x0 + .5, np.arange(y1 - y0) + y0 + .5)
    m = contains_xy(outline, xs, ys)
    # the block step's pieces stand a square or two apart (lots may not touch):
    # those gaps closed, so a porch or a tower joins the main building
    m |= ndi.binary_closing(m, structure=np.ones((1, s.GAP + 1), bool)) | \
        ndi.binary_closing(m, structure=np.ones((s.GAP + 1, 1), bool))
    lab, n = ndi.label(~m)
    edge = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    for k in range(1, n + 1):
        if k not in edge and (lab == k).sum() < s.SLIVER:
            m[lab == k] = True
    # one building: a piece standing apart (a strip by a river) is left out
    lab, n = ndi.label(m)
    if n > 1:
        big = 1 + int(np.argmax([(lab == k).sum() for k in range(1, n + 1)]))
        print(f"left out, apart from the building: {int((m & (lab != big)).sum())} squares in {n - 1} pieces")
        m &= lab == big
    return m, (x0, y0)


def lidar_storeys(map_dir, m, ox, oy, s):
    """Each square's storeys from the LiDAR, the footprint shifted onto it first
    (the block step and the road graph move a building a few metres: the shift
    whose tall ground best matches the footprint). Returns (eaves storeys evened
    out, roof form, the roof's own storeys uncapped)."""
    from scipy import ndimage as ndi
    from pyproj import Transformer
    sys.path.insert(0, os.path.join(acpaths.UK, "src", "area"))
    import roofs
    _name, proj = projector(map_dir)
    H, W = m.shape
    xs, ys = np.meshgrid(np.arange(W) + ox + .5, np.arange(H) + oy + .5)
    ll = [proj.to_latlon(x, y) for x, y in zip(xs.ravel(), ys.ravel())]
    E, N = Transformer.from_crs(4326, 27700, always_xy=True).transform([p[1] for p in ll], [p[0] for p in ll])
    E, N = np.array(E).reshape(H, W), np.array(N).reshape(H, W)
    box = (int(E.min()) - 60, int(N.min()) - 60, int(E.max()) + 60, int(N.max()) + 60)
    area = os.path.join(acpaths.UK, "data", s.LIDAR_AREA, "lidar")
    nd = roofs.mosaic("dsm", box, area) - roofs.mosaic("dtm", box, area)

    cE, cN = E.mean(), N.mean()

    def sample(dx, dy, rot=0.0, grid=None):
        """The LiDAR under each square, the footprint shifted (m) and turned (degrees) onto it."""
        grid = nd if grid is None else grid
        a = np.radians(rot)
        e = cE + (E - cE) * np.cos(a) - (N - cN) * np.sin(a) + dx
        n = cN + (E - cE) * np.sin(a) + (N - cN) * np.cos(a) + dy
        c = np.clip((e - box[0]).astype(int), 0, grid.shape[1] - 1)
        r = np.clip((box[3] - n).astype(int), 0, grid.shape[0] - 1)
        return grid[r, c]

    def fit(dx, dy, rot):           # the footprint's overlap with the LiDAR's tall ground
        tall = np.nan_to_num(sample(dx, dy, rot)) > s.TALL_M
        return (tall & m).sum() / max(1, (tall | m).sum())
    # the footprint lies a few metres off and turned a little on the real building (the block step
    # squared its pieces to the grid; Parliament: 1.5 degrees, 7 m over its length): shift and turn,
    # coarse then fine
    shifts = range(-s.SHIFT_MAX, s.SHIFT_MAX + 1, 2)
    turns = np.arange(-s.TURN_MAX, s.TURN_MAX + 0.01, 1.0)
    _f, dx, dy, rot = max((fit(a, b, t), a, b, t) for a in shifts for b in shifts for t in turns)
    score, dx, dy, rot = max((fit(a, b, t), a, b, t) for a in range(dx - 2, dx + 3) for b in range(dy - 2, dy + 3)
                             for t in np.arange(rot - 0.75, rot + 0.76, 0.25))
    h = np.where(m, sample(dx, dy, rot), np.nan)
    # the roof's slope, as roofs.py reads it (np.gradient of the 1 m DSM): flat leads or pitched slate
    gy, gx = np.gradient(np.nan_to_num(nd, nan=0.0))
    steep = m & (sample(dx, dy, rot, np.degrees(np.arctan(np.hypot(gx, gy)))) > roofs.FLAT_SLOPE_DEG)
    # a pitched square's ridge: across its slope (slopes facing east-west: a ridge running north-south)
    ns = np.abs(sample(dx, dy, rot, gx)) >= np.abs(sample(dx, dy, rot, gy))
    form = np.where(~m, 0, np.where(~steep, s.FLAT, np.where(ns, s.RIDGE_NS, s.RIDGE_WE)))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eaves = ndi.generic_filter(h, lambda v: np.nanpercentile(v, s.EAVES_PCT) if np.isfinite(v).any() else np.nan,
                                   size=s.EAVES_WIN, mode="constant", cval=np.nan)
    # the roof itself, lightly evened (a 3-square median), for blocks standing above their range:
    # the eaves reading (a low percentile over 7) wears a narrow tower away to nothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        top = ndi.generic_filter(h, lambda v: np.nanmedian(v) if np.isfinite(v).any() else np.nan,
                                 size=3, mode="constant", cval=np.nan)
    raw = np.where(m, np.rint(np.nan_to_num(top, nan=0) / s.STOREY_M), 0).astype(int)   # uncapped: towers too
    st = np.where(m, np.clip(np.rint(np.nan_to_num(eaves, nan=s.STOREY_M) / s.STOREY_M), 1, s.RANGE_MAX), 0).astype(int)
    # evened out: each square the commonest storeys round it (in the footprint)
    commonest = lambda v: np.bincount(v[v > 0].astype(int)).argmax() if (v > 0).any() else 0  # noqa: E731
    sm = ndi.generic_filter(st, commonest, size=s.SMOOTH_WIN, mode="constant", cval=0)
    st = np.where(m, np.maximum(sm, 1), 0)
    form = np.where(m, ndi.generic_filter(form, commonest, size=s.FORM_WIN, mode="constant", cval=0), 0)
    form = np.where(m & (form == 0), s.FLAT, form)
    s.lidar_fit = (dx, dy, rot)
    print(f"LiDAR: shifted {dx},{dy} m, turned {rot:.2f} deg (overlap {score:.3f}); storeys {dict(zip(*np.unique(st[m], return_counts=True)))}; "
          f"roof squares flat {int((form == s.FLAT).sum())}, ridge N-S {int((form == s.RIDGE_NS).sum())}, "
          f"ridge W-E {int((form == s.RIDGE_WE).sum())}")
    return st, form, raw, h


def rects_of(m):
    """The mask as rectangles [x, y, w, h]: runs of rows merged down."""
    out, open_ = [], {}
    for y in range(m.shape[0] + 1):
        runs = set()
        if y < m.shape[0]:
            x = 0
            while x < m.shape[1]:
                if m[y, x]:
                    s = x
                    while x < m.shape[1] and m[y, x]:
                        x += 1
                    runs.add((s, x))
                else:
                    x += 1
        for r in list(open_):
            if r not in runs:
                y0 = open_.pop(r)
                out.append([r[0], y0, r[1] - r[0], y - y0])
        for r in runs:
            open_.setdefault(r, y)
    return out


def rects_cover(cells):
    """A room's squares as inclusive rectangles [x0, y0, x1, y1]."""
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    sub = np.zeros((max(ys) - y0 + 1, max(xs) - x0 + 1), bool)
    for x, y in cells:
        sub[y - y0, x - x0] = True
    return [[x0 + x, y0 + y, x0 + x + w - 1, y0 + y + h - 1] for x, y, w, h in rects_of(sub)]


def largest_rect(m):
    """The biggest rectangle of True in m: (x, y, w, h), or None (histogram method)."""
    H, W = m.shape
    hist = np.zeros(W, int)
    best = (0, None)
    for y in range(H):
        hist = np.where(m[y], hist + 1, 0)
        stack = []
        for x in range(W + 1):
            hh = hist[x] if x < W else 0
            start = x
            while stack and stack[-1][1] >= hh:
                sx, sh = stack.pop()
                if sh * (x - sx) > best[0]:
                    best = (sh * (x - sx), (sx, y - sh + 1, x - sx, sh))
                start = sx
            stack.append((start, hh))
    return best[1]


def ranges_of(m, s, owner=None):
    """The footprint as its ranges: the largest rectangle taken again and again
    (a long wing is the biggest rectangle there is), until only scraps are left;
    each scrap then joins the range it touches most. `owner` holds ranges
    already claimed (1..n, from the spec's RANGES): those squares are not cut
    again, and scraps may join them too. Returns the new rectangles and the
    owner of every square (the new ranges numbered on from the claimed)."""
    from scipy import ndimage as ndi
    owner = np.zeros(m.shape, int) if owner is None else owner.copy()
    base = int(owner.max())
    todo = m & (owner == 0)
    out = []
    while True:
        r = largest_rect(todo)
        if r is None or r[2] * r[3] < s.RANGE_MIN:
            break
        x, y, w, h = r
        out.append([x, y, w, h])
        todo[y:y + h, x:x + w] = False
    for i, (x, y, w, h) in enumerate(out):
        owner[y:y + h, x:x + w] = base + i + 1
    left = todo.copy()
    for _ in range(50):                      # scraps: the squares left go to a range beside them
        if not left.any():
            break
        grow = ndi.grey_dilation(owner, size=3) * left
        owner = np.where(left & (grow > 0), grow, owner)
        left = m & (owner == 0)
    return out, owner


def _roof_form(form, sq, w, h, s):
    """A range's roof by its squares' majority: pitched along its length (h >= w:
    ridge north-south) where PITCHED_SHARE of them slope, else flat."""
    pitched = float(np.isin(form[sq], (s.RIDGE_NS, s.RIDGE_WE)).mean()) >= s.PITCHED_SHARE and min(w, h) >= 3
    return ("NS" if h >= w else "WE") if pitched else "flat"


def split_steps(st, sq, x, y, w, h, s, depth=2):
    """A range cut along its length where the LiDAR shows it stepping down (the
    ranges round Parliament's courts: one height a range overstated the lower end):
    at the cut whose two sides' median storeys differ most, if by SPLIT_BY or more
    and both sides SPLIT_MIN squares or longer. Returns [(x, y, w, h)]."""
    along = h >= w
    n = h if along else w
    best = None
    for c in range(s.SPLIT_MIN, n - s.SPLIT_MIN + 1):
        a = np.zeros_like(sq)
        if along:
            a[y:y + c, x:x + w] = True
        else:
            a[y:y + h, x:x + c] = True
        one, two = sq & a, sq & ~a
        if one.sum() < s.SPLIT_MIN or two.sum() < s.SPLIT_MIN:
            continue
        d = abs(float(np.median(st[one])) - float(np.median(st[two])))
        if d >= s.SPLIT_BY and (best is None or d > best[0]):
            best = (d, c)
    if best is None or depth == 0:
        return [(x, y, w, h)]
    c = best[1]
    halves = [(x, y, w, c), (x, y + c, w, h - c)] if along else [(x, y, c, h), (x + c, y, w - c, h)]
    out = []
    for hx, hy, hw, hh in halves:
        sub = np.zeros_like(sq)
        sub[hy:hy + hh, hx:hx + hw] = True
        out += split_steps(st, sq & sub, hx, hy, hw, hh, s, depth - 1)
    return out


def range_heights(st, form, m, s):
    """Each range one height (its squares' median storeys) and one roof (flat or
    pitched by its squares' majority, the ridge along its length): the per-square
    LiDAR reading evened out range by range. The spec's RANGES are claimed first,
    in order (the first claim wins; a range's form or storeys, if given, stand
    over the LiDAR's); what they leave is cut by ranges_of. Returns (storeys per
    square, roof parts (x, y, w, h, form, storeys), the squares of ranges whose
    form or storeys the spec set: no raised part is found in them, and each
    square's range, numbered from 1)."""
    owner = np.zeros(m.shape, int)
    listed = []                                     # (name, x, y, w, h, form or None, storeys or None)
    for name, (x0, y0, x1, y1), *given in s.RANGES:
        f, k = (list(given) + [None, None])[:2]
        sq = np.zeros_like(m)
        sq[y0:y1 + 1, x0:x1 + 1] = True
        sq &= m & (owner == 0)
        if not sq.any():
            print(f"range {name!r}: no squares left to claim")
            continue
        listed.append((name, x0, y0, x1 - x0 + 1, y1 - y0 + 1, f, k))
        owner[sq] = len(listed)
    rest, owner = ranges_of(m, s, owner)
    st2 = np.zeros_like(st)
    parts, set_by_spec, report = [], np.zeros_like(m), []
    for i, (name, x, y, w, h, f, k) in enumerate(listed):
        whole = owner == i + 1
        if f or k:
            set_by_spec |= whole
        # a range the spec does not fix is cut where it steps down (split_steps)
        pieces = [(x, y, w, h)] if (f or k) else split_steps(st, whole, x, y, w, h, s)
        for px, py, pw, ph in pieces:
            sq = np.zeros_like(m)
            sq[py:py + ph, px:px + pw] = True
            sq &= whole
            if not sq.any():
                continue
            kk = int(k or np.median(st[sq]))
            ff = f or _roof_form(form, sq, pw, ph, s)
            st2[sq] = kk
            # the roof part: the piece's box less the footprint's squares other ranges hold
            # (Build clips a part to the footprint, and a later part overwrites an earlier one)
            box = np.zeros_like(m)
            box[py:py + ph, px:px + pw] = True
            box &= ~m | sq
            for rx, ry, rw, rh in rects_of(box):
                parts.append((rx, ry, rw, rh, ff, kk))
            report.append((name + (f" ({len(pieces)} steps)" if len(pieces) > 1 else ""), int(sq.sum()), kk, ff,
                           int(np.median(st[sq])), _roof_form(form, sq, pw, ph, s)))
    for i, (x, y, w, h) in enumerate(rest):
        sq = owner == len(listed) + i + 1
        k = int(np.median(st[sq]))
        st2[sq] = k
        parts.append((x, y, w, h, _roof_form(form, sq, w, h, s), k))
    if listed:
        print("ranges (name, squares, storeys, form; the LiDAR's storeys, form):")
        for r in report:
            print("  ", r)
        print(f"   and {len(rest)} more cut from the {int((owner > len(listed)).sum())} squares left")
    # the scraps' squares are roofed with the range they joined (roof_plan gives an
    # uncovered square its nearest part)
    return np.where(m, np.maximum(st2, 1), 0), parts, set_by_spec, owner


def raised_parts(raw, st, ranges, s, form=None, tower_box=None, keep=None, owner=None):
    """Blocks the LiDAR shows standing RAISE_BY storeys or more above the range round
    them - a central tower, the smaller towers by the courts, a front's pavilions -
    each its largest rectangle at its own median height. `keep`: squares whose
    height or roof the spec set, where no block is looked for. `owner`: each
    square's range; a block is looked for range by range, so a low range's ridge
    does not run on into the tower beside it (at Westminster the spine's ridge
    over the low Members' Lobby range ran into the Central Tower, and the
    largest rectangle then cut the tower in strips)."""
    from scipy import ndimage as ndi
    # the DSM has no returns over the tallest, thinnest tops: such a gap
    # inside the footprint takes the highest reading round it
    raw = raw.copy()
    for _ in range(3):
        gap = ranges & (raw == 0)
        if not gap.any():
            break
        raw[gap] = ndi.grey_dilation(raw, size=5)[gap]
    near_tower = np.zeros_like(ranges)
    for tx, ty, tw, th in (tower_box or {}).values():         # the towers' own LiDAR spills past their boxes
        near_tower[max(0, ty - s.TOWER_CLEAR):ty + th + s.TOWER_CLEAR,
                   max(0, tx - s.TOWER_CLEAR):tx + tw + s.TOWER_CLEAR] = True
    high = ranges & ~near_tower & (raw >= st + s.RAISE_BY)
    if keep is not None:
        high &= ~keep
    high = ndi.binary_opening(high, structure=np.ones((3, 3), bool))     # not a chimney or a pinnacle
    if owner is None:
        owner = ranges.astype(int)
    lab, n = np.zeros(high.shape, int), 0
    for r in np.unique(owner[high]):
        lr, k = ndi.label(high & (owner == r))
        lab[lr > 0] = lr[lr > 0] + n
        n += k
    out = []
    rects = []
    for k in range(1, n + 1):
        part = lab == k                     # a block may hold several (a hall meeting a tower):
        while part.sum() >= s.RAISE_MIN:    # rectangles taken until scraps are left
            r = largest_rect(part)
            if r is None or r[2] * r[3] < s.RAISE_MIN:
                break
            if min(r[2], r[3]) >= s.RAISE_WIDTH:   # narrower is a pitched range's ridge, not a block
                rects.append(r)
            part[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = False
    for x, y, w, h in rects:
        kk = int(min(s.RAISE_MAX, np.median(raw[y:y + h, x:x + w])))
        st[y:y + h, x:x + w] = kk
        f = "flat"
        if form is not None and min(w, h) >= 3 and \
                float(np.isin(form[y:y + h, x:x + w], (s.RIDGE_NS, s.RIDGE_WE)).mean()) >= s.PITCHED_SHARE:
            f = "NS" if h >= w else "WE"                      # a tall hall's roof: pitched
        out.append((int(x), int(y), int(w), int(h), f, kk))
    print(f"raised parts from the LiDAR: {[(x, y, w, h, k) for x, y, w, h, _f, k in out]}")
    return out


def lidar_footprint(m, h, keep, s):
    """The footprint less what the LiDAR shows is ground (light wells, courts OSM drew as
    building); specks under 6 squares dropped, holes under 4 filled; `keep` (towers, pavilions)
    always building. A square the LiDAR has no reading for (a spire's top) stays."""
    from scipy import ndimage as ndi
    b = m & ~(np.nan_to_num(h, nan=99.0) < s.GROUND_M)
    lab, n = ndi.label(b)
    for k in range(1, n + 1):
        if (lab == k).sum() < 6:
            b[lab == k] = False
    lab, n = ndi.label(~b)
    for k in range(1, n + 1):
        sq = lab == k
        if sq.sum() < 4 and not sq[[0, -1], :].any() and not sq[:, [0, -1]].any():
            b[sq] = True
    b |= keep & m
    print(f"footprint from the LiDAR's ground: {int(m.sum())} -> {int(b.sum())} squares")
    return b


def _rise(w, h, form, slope=None):
    """A pitched roof's height over its eaves on a w x h rectangle (m): the game's (45 degrees,
    3 squares a side at most) or, given a slope, a real roof's (uncapped)."""
    yy, xx = np.mgrid[0:h, 0:w]
    if form == "NS":
        d = np.minimum(xx, w - 1 - xx) + 1
    elif form == "WE":
        d = np.minimum(yy, h - 1 - yy) + 1
    else:
        return np.zeros((h, w))
    return d * slope if slope else np.minimum(d, 3) * GAME_RISE_M / 3


def _fit(z, sq, x, y, w, h, s):
    """The roof and storeys that best match the LiDAR `z` on squares `sq` of the box: flat, the
    game's pitched roof, and real tall roofs (TALL_SLOPES) - a tall hall's slope is its roof,
    not storeys stepping; a pitched roof's ridge runs along the part's length. Returns (miss
    summed, form, storeys, real rise m or 0)."""
    zz, ss = z[y:y + h, x:x + w], sq[y:y + h, x:x + w]
    best, game = None, None
    for form in ("flat", "NS" if h >= w else "WE"):
        if form != "flat" and min(w, h) < 3:
            continue
        for slope in (None,) + (tuple(s.TALL_SLOPES) if form != "flat" and min(w, h) <= s.TALL_MAX_SIDE else ()):
            r = _rise(w, h, form, slope)
            res = zz[ss] - r[ss]
            k = int(max(1, np.rint(np.median(res) / s.STOREY_M)))
            err = float(np.abs(res - k * s.STOREY_M).sum())
            fit = (err, form, k, float(r[ss].max()) if slope else 0.0)
            if best is None or err < best[0]:
                best = fit
            if slope is None and (game is None or err < game[0]):
                game = fit
    # a tall roof only where it is clearly the better match: the storeys come in 3 m steps, and a
    # steeper roof can stand in for a storey's rounding
    if best[3] > s.TALL_ROOF_M and best[0] > s.TALL_GAIN * game[0]:
        return game
    return best


def lidar_parts(h, b, s, form=None):
    """The footprint `b` cut into parts from the LiDAR heights `h`: a box is cut where one
    straight cut best lowers the summed miss, until each part misses by SPLIT_TOL m or less on
    average, or is too small to cut (SPLIT_SIDE). A pitched part SPLIT_ONE_RANGE or narrower is
    not cut along its ridge (that would step its roof). Returns (storeys per square, roof parts
    (x, y, w, h, form, storeys), tall roofs (x, y, w, h, form, storeys, rise m), owner)."""
    top = float(np.nanmax(np.where(b, h, np.nan)))
    z = np.where(np.isfinite(h), h, top)                   # no return: a spire's top
    H, W = b.shape
    parts = []

    def cost(bx, by, bw, bh):
        return _fit(z, b, bx, by, bw, bh, s)[0] if b[by:by + bh, bx:bx + bw].any() else 0.0

    def go(x, y, w, h_):
        sub = b[y:y + h_, x:x + w]
        if not sub.any():
            return
        ys_, xs_ = np.nonzero(sub)
        x, y = x + int(xs_.min()), y + int(ys_.min())
        w, h_ = int(xs_.max() - xs_.min() + 1), int(ys_.max() - ys_.min() + 1)
        e0, f0, k0, r0 = _fit(z, b, x, y, w, h_, s)
        n = int(b[y:y + h_, x:x + w].sum())
        twin = False
        if r0 > s.TALL_ROOF_M:
            # a tall roof or two ranges side by side, each its own roof (a valley down the middle)?
            mid = (w if f0 == "NS" else h_) // 2
            for c in range(max(s.SPLIT_SIDE, mid - 2), mid + 3):
                if f0 == "NS" and c <= w - s.SPLIT_SIDE:
                    e = cost(x, y, c, h_) + cost(x + c, y, w - c, h_)
                elif f0 == "WE" and c <= h_ - s.SPLIT_SIDE:
                    e = cost(x, y, w, c) + cost(x, y + c, w, h_ - c)
                else:
                    continue
                twin |= e < 0.8 * e0
        if twin:
            pass                                           # cut below, along the ridge
        elif e0 / n <= s.SPLIT_TOL or max(w, h_) < 2 * s.SPLIT_SIDE:
            parts.append((x, y, w, h_, f0, k0, r0))
            return
        one = f0 != "flat" and min(w, h_) <= s.SPLIT_ONE_RANGE and not twin
        best = None
        step = 1 if max(w, h_) < 60 else 2
        if not (one and f0 == "WE"):                       # a cut across y runs along a W-E ridge
            for c in range(s.SPLIT_SIDE, h_ - s.SPLIT_SIDE + 1, step):
                e = cost(x, y, w, c) + cost(x, y + c, w, h_ - c)
                if best is None or e < best[0]:
                    best = (e, "y", c)
        if not (one and f0 == "NS"):
            for c in range(s.SPLIT_SIDE, w - s.SPLIT_SIDE + 1, step):
                e = cost(x, y, c, h_) + cost(x + c, y, w - c, h_)
                if best is None or e < best[0]:
                    best = (e, "x", c)
        if best is None:
            parts.append((x, y, w, h_, f0, k0, r0))
            return
        _e, ax, c = best
        if ax == "y":
            go(x, y, w, c)
            go(x, y + c, w, h_ - c)
        else:
            go(x, y, c, h_)
            go(x + c, y, w - c, h_)
    go(0, 0, W, H)
    # a part fitted flat where the LiDAR's slopes say pitched (PITCHED_SHARE of its squares) takes
    # the game's pitched roof if that fits nearly as well (PITCH_GIVE): its roof is far shallower
    # than the real one either way, but a ridge reads as the real roofline, a flat top does not
    if form is not None:
        pitched_now = 0
        for i, (x, y, w, h_, f, k, r) in enumerate(parts):
            if f != "flat" or min(w, h_) < 3:
                continue
            sq = np.zeros_like(b)
            sq[y:y + h_, x:x + w] = True
            sq &= b
            if not sq.any() or float(np.isin(form[sq], (s.RIDGE_NS, s.RIDGE_WE)).mean()) < s.PITCHED_SHARE:
                continue
            pf = "NS" if h_ >= w else "WE"
            rr = np.zeros_like(z)
            rr[y:y + h_, x:x + w] = _rise(w, h_, pf)
            res = z[sq] - rr[sq]
            kp = int(max(1, np.rint(np.median(res) / s.STOREY_M)))
            ep = float(np.abs(res - kp * s.STOREY_M).sum())
            ef = float(np.abs(z[sq] - k * s.STOREY_M).sum())
            if ep <= s.PITCH_GIVE * ef:
                parts[i] = (x, y, w, h_, pf, kp, r)
                pitched_now += 1
        print(f"flat parts the LiDAR's slopes call pitched, given the game's pitched roof: {pitched_now}")
    st = np.zeros((H, W), int)
    owner = np.zeros((H, W), int)
    out, tall = [], []
    for i, (x, y, w, h_, f, k, r) in enumerate(parts):
        k = int(min(k, s.RAISE_MAX))
        sq = np.zeros_like(b)
        sq[y:y + h_, x:x + w] = True
        sq &= b
        st[sq] = k
        owner[sq] = i + 1
        out.append((x, y, w, h_, f, k))
        if r > s.TALL_ROOF_M:
            # the piece's rise as the LiDAR has it: its top (95th percentile) over the eaves
            rise = float(np.percentile(z[sq], 95)) - k * s.STOREY_M
            tall.append((x, y, w, h_, f, k, round(rise, 1)))
    print(f"parts from the LiDAR: {len(out)} (miss {s.SPLIT_TOL} m or less, or too small to cut); "
          f"storeys {dict(zip(*np.unique(st[b], return_counts=True)))}")
    print(f"tall roofs (rise over {s.TALL_ROOF_M} m: a roof piece wanted): {tall}")
    return np.where(b, np.maximum(st, 1), 0), out, tall, owner


STAIR_RUN = 5                          # squares a staircase runs (knoxbuild.layout.STAIR_RUN)


RIDGE_MIN = 7                          # squares across: a pitched part this deep has Build's flat ridge band
LANTERN_RANGE = 24                     # squares long: a pitched range this long has ventilation lanterns
LANTERN_EVERY = 16                     # squares between them along its ridge


def roof_details(parts, m, in_tower, s):
    """On the pitched roofs (gothic_roof_uk_01): iron cresting along each ridge and
    ventilation lanterns spaced along the long ranges, on Build's ridge band - the
    level above the roof floor (k + 1 for a part of k storeys), its middle row (W-E)
    or column (N-S). The cresting stops a square short of a lantern."""
    if not s.ROOF_DETAIL:
        return []
    out = []
    for x, y, w, h, f, k in parts:
        if f not in ("NS", "WE") or min(w, h) < s.RIDGE_MIN:
            continue
        along = f == "NS"
        n = h if along else w
        ridge = (x + w // 2) if along else (y + h // 2)
        lanterns = set()
        if n >= s.LANTERN_RANGE:
            for j, at in enumerate(range(s.LANTERN_EVERY // 2, n - 2, s.LANTERN_EVERY)):
                lx, ly = (ridge - 1, y + at) if along else (x + at, ridge - 1)
                if m[ly:ly + 2, lx:lx + 2].all() and not in_tower(lx, ly):
                    out.append([int(k + 1), "roof_lantern_oct" if j % 2 else "roof_lantern_sq", int(lx), int(ly), "S"])
                    lanterns |= set(range(at - 1, at + 3))
        for t in range(n):
            if t in lanterns:
                continue
            cx, cy = (ridge, y + t) if along else (x + t, ridge)
            if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx] and not in_tower(cx, cy):
                out.append([int(k + 1), "ridge_cresting", int(cx), int(cy), "E" if along else "S"])
    return out


def stairwells(s, st):
    """The landmark's own staircases ([level, x, y, dir]) and their halls ({level:
    [(name, rect)]}): each of the spec's STAIRWELLS (name, x, y, dir, from level)
    climbs storey by storey while the squares it runs over are there on both the
    storey it rises from and the one it lands on - as many as the building needs,
    where it has them, not Build's one a storey."""
    stairs, halls = [], {}
    for name, x, y, sdir, first in s.STAIRWELLS:
        run = [(x, y + k) if sdir == "N" else (x + k, y) for k in range(s.STAIR_RUN)]
        hall = [x - 1, y - 1, x + 1, y + s.STAIR_RUN] if sdir == "N" else [x - 1, y - 1, x + s.STAIR_RUN, y + 1]
        top = first
        while all(0 <= yy < st.shape[0] and 0 <= xx < st.shape[1] and st[yy, xx] > top + 1 for xx, yy in run):
            stairs.append([top, x, y, sdir])
            top += 1
        for lvl in range(first, top + 1):
            halls.setdefault(lvl, []).append((name, hall))
        print(f"stairwell {name}: storeys {first}-{top}" if top > first else f"stairwell {name}: no storey to climb")
    return stairs, halls


def plan_level(m, lvl, s, tower_box=None, halls=()):
    """[(name, kind, set of squares)] for one storey (`m`: its squares): themed,
    the stair halls, each tower a room of its own, corridors, then offices."""
    from scipy import ndimage as ndi
    H, W = m.shape
    owner = np.zeros((H, W), np.int32)
    rooms = []

    def claim(name, kind, rects, void=False):
        cells = set()
        for x0, y0, x1, y1 in rects:
            for y in range(max(0, y0), min(H - 1, y1) + 1):
                for x in range(max(0, x0), min(W - 1, x1) + 1):
                    if m[y, x] and not owner[y, x]:
                        cells.add((x, y))
        if cells:
            rooms.append({"name": name, "kind": kind, "cells": cells, "void": void})
            for x, y in cells:
                owner[y, x] = len(rooms)
    for name, kind, levels, rects, voids in s.THEMED:
        if lvl in levels:
            claim(name, kind, rects, lvl in voids)
    for name, rect in halls:                                     # the stairs' halls, in the towers too
        claim(name, "hall", [rect])
    for name, (tx, ty, tw, th) in (tower_box or {}).items():     # before the corridors that reach them
        claim(name, "hall", [[tx, ty, tx + tw - 1, ty + th - 1]])
    for name, levels, rects in s.CORRIDORS:
        if levels is None or lvl in levels:
            claim(name, "hall", rects)
    # the rest: offices on a grid, one per connected piece of each cell
    cell = s.OFFICE_CELL
    for cy in range(0, H, cell):
        for cx in range(0, W, cell):
            free = (owner[cy:cy + cell, cx:cx + cell] == 0) & m[cy:cy + cell, cx:cx + cell]
            lab, n = ndi.label(free)
            for k in range(1, n + 1):
                ys, xs = np.nonzero(lab == k)
                cells = {(cx + int(x), cy + int(y)) for x, y in zip(xs, ys)}
                rooms.append({"name": "Office", "kind": "office", "cells": cells, "void": False})
                for x, y in cells:
                    owner[y, x] = len(rooms)
    # scraps join the neighbour they share most edge with
    for i, r in enumerate(rooms):
        if r["kind"] != "office" or len(r["cells"]) >= s.SMALL_ROOM or not r["cells"]:
            continue
        near = {}
        for x, y in r["cells"]:
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < W and 0 <= ny < H and owner[ny, nx] and owner[ny, nx] != i + 1:
                    j = owner[ny, nx] - 1
                    if rooms[j]["kind"] == "office" and not rooms[j]["void"]:
                        near[j] = near.get(j, 0) + 1
        if near:
            j = max(near, key=near.get)
            rooms[j]["cells"] |= r["cells"]
            for x, y in r["cells"]:
                owner[y, x] = j + 1
            r["cells"] = set()
    rooms = [r for r in rooms if r["cells"]]
    owner[:] = 0
    for i, r in enumerate(rooms):
        for x, y in r["cells"]:
            owner[y, x] = i + 1
    return rooms, owner


def doors_for(rooms, owner, lvl, entrances, landings=()):
    """Doors so that every room can be reached: each room one onto a corridor
    or hall if it has one, then rooms still cut off one onto a reached neighbour.
    `landings`: rooms a staircase of its own lands in (reached from below).
    Returns the doors and the rooms still cut off (indices)."""
    H, W = owner.shape
    edges = {}                                     # (i, j) -> [door squares]
    for y in range(H):
        for x in range(W):
            a = owner[y, x]
            if not a:
                continue
            for nx, ny, door in ((x + 1, y, (x + 1, y, "W")), (x, y + 1, (x, y + 1, "N"))):
                if nx < W and ny < H and owner[ny, nx] and owner[ny, nx] != a:
                    key = tuple(sorted((a - 1, owner[ny, nx] - 1)))
                    edges.setdefault(key, []).append(door)
    adj = {}
    for (i, j), ds in edges.items():
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    open_kind = lambda r: r["kind"] != "office" and not r["void"]  # noqa: E731
    chosen = set()

    def link(i, j):
        key = tuple(sorted((i, j)))
        if key not in chosen and len(edges[key]) >= 1:
            chosen.add(key)
    for i, r in enumerate(rooms):
        if r["void"]:
            continue
        if r["kind"] == "office":
            halls = [j for j in adj.get(i, []) if open_kind(rooms[j])]
            if halls:
                link(i, max(halls, key=lambda j: len(edges[tuple(sorted((i, j)))])))
        else:                                       # corridors and state rooms open onto each other
            for j in adj.get(i, []):
                if open_kind(rooms[j]):
                    link(i, j)
    # every room reached from the entrances (or, upstairs, from any hall)
    start = ({owner[y, x] - 1 for _l, x, y, _s in entrances if owner[y, x]} or
             {i for i, r in enumerate(rooms) if open_kind(r)}) | set(landings)
    cut = []
    while True:
        seen, q = set(start), deque(start)
        while q:
            i = q.popleft()
            for j in adj.get(i, []):
                if tuple(sorted((i, j))) in chosen and j not in seen:
                    seen.add(j)
                    q.append(j)
        cut = [i for i, r in enumerate(rooms) if i not in seen and not r["void"]]
        if not cut:
            break
        grew = False
        for i in cut:
            near = [j for j in adj.get(i, []) if j in seen and not rooms[j]["void"]]
            if near:
                link(i, max(near, key=lambda j: len(edges[tuple(sorted((i, j)))])))
                grew = True
        if not grew:
            break
    doors = []
    for key in sorted(chosen):
        ds = edges[key]
        x, y, side = ds[len(ds) // 2]               # the middle of the shared wall
        doors.append([lvl, x, y, side])
    return doors, cut


def landing_stairs(rooms, owner, cut, below, s):
    """A staircase up into each cluster of rooms cut off on this storey (an upper storey of
    offices with no hall of its own): a straight run of STAIR_RUN squares inside one of its
    rooms, over one reached room of the storey below (`below`: its owner grid and the rooms
    reached there). Returns ([run start x, y, dir], landing room index) pairs."""
    b_owner, b_rooms, b_reached = below
    H, W = owner.shape
    out, done = [], set()
    for i in cut:
        if i in done:
            continue
        found = None
        cells = sorted(rooms[i]["cells"], key=lambda c: (c[1], c[0]))
        for x, y in cells:
            for sdir in ("N", "W"):                  # a run along y (N) or along x (W), as tbx has them
                run = [(x, y + k) if sdir == "N" else (x + k, y) for k in range(s.STAIR_RUN)]
                if not all(0 <= yy < H and 0 <= xx < W and owner[yy, xx] == i + 1 for xx, yy in run):
                    continue
                low = {int(b_owner[yy, xx]) for xx, yy in run}
                if len(low) == 1 and 0 not in low:
                    j = low.pop() - 1
                    if j in b_reached and not b_rooms[j]["void"]:
                        found = (x, y, sdir)
                        break
            if found:
                break
        if found:
            out.append((found, i))
            done.add(i)
    return out


def panelling(rooms, owner, lvl, doors, s):
    """Linenfold on the panelled rooms' west and north walls (the faces seen
    from inside, LANDMARKS.md), a Gothic surround on each of their doors."""
    H, W = owner.shape
    door_at = {(x, y, side) for _l, x, y, side in doors}
    out = []
    for i, r in enumerate(rooms):
        if r["name"] not in s.PANELLED or r["void"]:
            continue
        for x, y in sorted(r["cells"]):
            for side, (nx, ny) in (("W", (x - 1, y)), ("N", (x, y - 1))):
                inside = 0 <= nx < W and 0 <= ny < H and owner[ny, nx] == i + 1
                if inside:
                    continue
                piece = "door_surround_gothic" if (x, y, side) in door_at else "panelling_linenfold"
                out.append([lvl, piece, x, y, side])
    return out


def _runs(edges):
    """Outside wall edges grouped into straight runs, each in order along the wall."""
    by = {}
    for x, y, side in edges:
        by.setdefault((side, x if side == "W" else y), []).append(y if side == "W" else x)
    out = []
    for (side, line), ks in by.items():
        ks.sort()
        run = [ks[0]]
        for k in ks[1:]:
            if k == run[-1] + 1:
                run.append(k)
            else:
                out.append((side, line, run))
                run = [k]
        out.append((side, line, run))
    return out


def exterior(m, st, tower_box, doors, s, range_parts=(), raised=(), pavilions=(), dress=()):
    """Windows on every outside wall, in bays; buttresses and parapets on the
    faces the camera sees (east: the W edge of the square beyond; south: the N
    edge of the square below); pinnacles on the skyline; turrets on the tall
    ranges' and the pavilions' corners. The towers are left to the spec's
    tower_exterior."""
    H, W = m.shape
    door_at = {(x, y, sd) for _l, x, y, sd in doors}
    edges = []                                           # (x, y, side, faces the camera)
    for y in range(H):
        for x in range(W):
            if not m[y, x]:
                continue
            if x == 0 or not m[y, x - 1]:
                edges.append((x, y, "W", False))                       # a west face
            if x == W - 1 or not m[y, x + 1]:
                edges.append((x + 1, y, "W", True))                    # an east face
            if y == 0 or not m[y - 1, x]:
                edges.append((x, y, "N", False))                       # a north face
            if y == H - 1 or not m[y + 1, x]:
                edges.append((x, y + 1, "N", True))                    # a south face
    seen = {(x, y, sd) for x, y, sd, c in edges if c}
    # step walls: where a part stands above the part beside it, its wall above the lower roof is
    # an outside wall too (windows from the lower part's roof up; over a pitched roof, a storey
    # higher, clear of its slope)
    pitched = np.zeros_like(m)
    for x, y, w, h, f, _k in range_parts:
        if f in ("NS", "WE"):
            pitched[max(0, y):y + h, max(0, x):x + w] = True
    step_edges, own_of, base_of = [], {}, {}
    for y in range(H):
        for x in range(W):
            if not m[y, x]:
                continue
            for nx, ny, e, cam in ((x + 1, y, (x + 1, y, "W"), True), (x - 1, y, (x, y, "W"), False),
                                   (x, y + 1, (x, y + 1, "N"), True), (x, y - 1, (x, y, "N"), False)):
                if 0 <= nx < W and 0 <= ny < H and m[ny, nx] and st[ny, nx] < st[y, x]:
                    step_edges.append((*e, cam))
                    own_of[e] = (x, y)
                    base_of[e] = int(st[ny, nx]) + (1 if pitched[ny, nx] else 0)
    from scipy import ndimage as ndi
    lab, _n = ndi.label(np.pad(~m, 1, constant_values=True))
    outer = lab[1:-1, 1:-1] == lab[0, 0]                 # ground joined to the map round it: not a court

    def outward(x, y, side):     # the edge's outside square is open ground round the building (not a court)
        for sx, sy in (((x - 1, y) if side == "W" else (x, y - 1)), (x, y)):
            if not (0 <= sx < W and 0 <= sy < H):
                return True
            if not m[sy, sx]:
                return bool(outer[sy, sx])
        return False

    def in_tower(x, y):
        return any(tx <= x < tx + tw and ty <= y < ty + th for tx, ty, tw, th in tower_box.values())

    def at(side, line, k):
        return (line, k) if side == "W" else (k, line)

    def owner_sq(x, y, side):      # the building's square an edge belongs to (an east/south face: the one inside)
        if (x, y, side) in own_of:                       # a step wall: the taller part's square
            return own_of[(x, y, side)]
        if 0 <= x < W and 0 <= y < H and m[y, x]:
            return x, y
        return (x - 1, y) if side == "W" else (x, y - 1)
    windows, pieces = [], []
    for side, line, run in _runs([(x, y, sd) for x, y, sd, _c in edges] + [(x, y, sd) for x, y, sd, _c in step_edges]):
        for b in range(0, len(run) - s.BAY + 1, s.BAY):
            pier, w0 = run[b], run[b + 1]
            wx, wy = at(side, line, w0)
            if any(in_tower(*owner_sq(*at(side, line, k), side)) for k in (pier, w0, w0 + 1)):
                continue
            if any((*at(side, line, k), side) in door_at for k in (w0, w0 + 1)):
                continue
            ox_, oy_ = owner_sq(wx, wy, side)
            lv = int(st[oy_, ox_])
            pair = [(*at(side, line, k), side) for k in (w0, w0 + 1)]
            if any(e in own_of for e in pair):
                # a step wall: both squares of the window on it, from the higher lower roof up to
                # the lower top
                if not all(e in own_of for e in pair):
                    continue
                base = max(base_of[e] for e in pair)
                top = min(int(st[own_of[e][1], own_of[e][0]]) for e in pair)
                wins, panels = floor_windows(top, s, base)
                windows += [[lvl, name, wx, wy, side] for lvl, name in wins]
                if s.WIN_PANEL:
                    pieces += [[lvl, s.WIN_PANEL, wx, wy, side] for lvl in panels if lvl > base]
                continue
            # one tall window a floor (or, without floor bands, the principal floor's two storeys
            # high and a window a storey above it); a carved panel between one and the one below
            wins, panels = floor_windows(lv, s)
            windows += [[lvl, name, wx, wy, side] for lvl, name in wins]
            if s.WIN_PANEL:
                pieces += [[lvl, s.WIN_PANEL, wx, wy, side] for lvl in panels]
            px, py = at(side, line, pier)
            qx, qy = owner_sq(px, py, side)
            if outward(px, py, side) and not in_tower(qx, qy):
                pieces.append([int(st[qy, qx]), "gothic_pinnacle", qx, qy, "S"])      # the skyline's pinnacles
            if (px, py, side) in seen:
                plv = int(st[owner_sq(px, py, side)[1], owner_sq(px, py, side)[0]])
                pieces += [[lvl, "buttress_shaft", px, py, side] for lvl in range(plv - 1)]
                pieces.append([plv - 1, "buttress_top", px, py, side])
    # parapets on the ranges' roof floor, on the faces the camera sees
    for x, y, side, cam in edges:
        if cam and not in_tower(*owner_sq(x, y, side)) and outward(x, y, side):
            sx, sy = owner_sq(x, y, side)
            pieces.append([int(st[sy, sx]), "parapet_pierced", x, y, side])
    for x, y, side, cam in step_edges:                  # and along the step walls the camera sees
        sx, sy = own_of[(x, y, side)]
        if cam and not in_tower(sx, sy):
            pieces.append([int(st[sy, sx]), "parapet_pierced", x, y, side])
    taken = {(p[2], p[3]) for p in pieces if p[1] == "gothic_pinnacle"}
    # the dressed pavilions: their roof piece on the flat top (level k), a tall pinnacle on each
    # corner square, ordinary pinnacles between along the parapet
    dressed = {(x, y, w, h) for x, y, w, h, _k, _r in dress}
    for x, y, w, h, k, roof in dress:
        corners = [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
        for cx, cy in corners:
            if 0 <= cy < H and 0 <= cx < W and m[cy, cx]:
                pieces.append([int(k), s.PAVILION_PINNACLE, int(cx), int(cy), "S"])
                taken.add((cx, cy))
        pieces.append([int(k), roof, int(x), int(y), "S"])
        for qx in range(x + 2, x + w - 2, 2):
            for qy in (y, y + h - 1):
                if m[qy, qx] and (qx, qy) not in taken:
                    pieces.append([int(k), "gothic_pinnacle", int(qx), int(qy), "S"])
        for qy in range(y + 2, y + h - 2, 2):
            for qx in (x, x + w - 1):
                if m[qy, qx] and (qx, qy) not in taken:
                    pieces.append([int(k), "gothic_pinnacle", int(qx), int(qy), "S"])
    parts = [("range", p) for p in range_parts] + [("raised", p) for p in raised] + \
        [("pavilion", p) for p in pavilions]
    # turrets where the landmark has them (spec TURRETS, e.g. measured off a model), on the roof
    # there; given them, ranges and raised parts get none by rule
    def on_dressed(cx, cy):        # a dressed pavilion has its own corner pinnacles, alike on each
        return any(x - 1 <= cx < x + w and y - 1 <= cy < y + h for x, y, w, h in dressed)
    for cx, cy in s.TURRETS:
        if on_dressed(cx, cy):
            continue
        if 0 <= cy < H and 0 <= cx < W and m[cy:cy + 2, cx:cx + 2].all() and not in_tower(cx, cy):
            pieces.append([int(st[cy:cy + 2, cx:cx + 2].max()), "turret_top", int(cx), int(cy), "S"])
            taken |= {(cx + i, cy + j) for i in (0, 1) for j in (0, 1)}
    for kind, (x, y, w, h, f, k) in parts:
        pav = kind == "pavilion"
        if pav and (x, y, w, h) in dressed:
            continue
        if s.TURRETS and not pav:
            continue
        if kind == "range" and (k < s.TURRET_FROM or max(w, h) < s.TURRET_RANGE):
            continue
        if kind == "raised" and min(w, h) < s.RAISE_TURRET:
            continue
        for cx, cy in ((x, y), (x + w - 2, y), (x, y + h - 2), (x + w - 2, y + h - 2)):
            if m[cy:cy + 2, cx:cx + 2].all() and not in_tower(cx, cy):
                pieces.append([int(k), "turret_top", int(cx), int(cy), "S"])
                taken |= {(cx + i, cy + j) for i in (0, 1) for j in (0, 1)}
        if pav:                                   # a pavilion's parapet: a pinnacle every other square
            for qx in range(x, x + w, 2):
                for qy in (y, y + h - 1):
                    if 0 <= qy < H and 0 <= qx < W and m[qy, qx] and (qx, qy) not in taken:
                        pieces.append([int(k), "gothic_pinnacle", int(qx), int(qy), "S"])
            for qy in range(y, y + h, 2):
                for qx in (x, x + w - 1):
                    if 0 <= qy < H and 0 <= qx < W and m[qy, qx] and (qx, qy) not in taken:
                        pieces.append([int(k), "gothic_pinnacle", int(qx), int(qy), "S"])
    pieces += roof_details(list(range_parts) + list(raised), m, in_tower, s)
    # single roof pieces where the landmark has them (a hall's louvre lantern): on the ridge
    # band of what stands there (its storeys + 1)
    for item, rx, ry in s.ROOF_PIECES:
        if 0 <= ry < H and 0 <= rx < W and m[ry, rx]:
            lvl = int(st[ry, rx]) + (1 if pitched[ry, rx] else 0)    # a pitched roof's ridge band, else its flat top
            for x, y, w, h, _f, k, _r in getattr(s, "_halls", []):
                if x <= rx < x + w and y <= ry < y + h:          # on a hall roof's ridge
                    lvl = int(k) + int(round(s.HALL_RIDGE_M.get(w, 9.0) / s.STOREY_M))
            pieces.append([lvl, item, int(rx), int(ry), "S"])
    if hasattr(s, "tower_exterior"):
        tw, tp = s.tower_exterior(tower_box, st)
        windows += tw
        pieces += tp
    return windows, pieces


def build(spec, map_dir=None) -> int:
    """The whole description from the spec, written to landmarks/<spec.OUT>."""
    s = settings(spec)
    map_dir = map_dir or os.path.join(KNOX, "output", s.MAP)
    keep, outline, tower_geo = members(map_dir, s)
    m, (ox, oy) = mask_of(outline, s)
    H, W = m.shape
    # a pavilion the landmark describes is building wherever it stands: Parliament's river
    # pavilions stand forward of the line OSM's outline and the block step drew (half of each
    # end pavilion fell outside the footprint)
    for pav in s.PAVILIONS:
        if len(pav) >= 6:
            px_, py_, pw, ph = pav[:4]
            m[max(0, py_):py_ + ph, max(0, px_):px_ + pw] = True
    print(f"footprint: {len(keep)} pieces, origin {ox},{oy}, {W} x {H}, {int(m.sum())} squares")
    # the towers' boxes first: their storeys stand over the LiDAR's
    tower_box = {}
    for name, oid in s.TOWERS.items():
        tx0, ty0, tx1, ty1 = (int(round(v)) for v in tower_geo[oid].bounds)
        tower_box[name] = (tx0 - ox, ty0 - oy, tx1 - tx0, ty1 - ty0)
        # to the footprint's squares inside it: OSM's outline rounded out can stand a square past
        # the walls (Victoria Tower's south and west faces: windows put in the air beyond them)
        bx, by, bw, bh = tower_box[name]
        sub = m[max(0, by):by + bh, max(0, bx):bx + bw]
        if sub.any():
            ys_, xs_ = np.nonzero(sub)
            tower_box[name] = (max(0, bx) + int(xs_.min()), max(0, by) + int(ys_.min()),
                               int(xs_.max() - xs_.min() + 1), int(ys_.max() - ys_.min() + 1))
        # and all of it building: a tower is one rectangle (its corner turrets' notches in OSM's
        # outline left its faces ragged, the windows on them in the air)
        bx, by, bw, bh = tower_box[name]
        m[max(0, by):by + bh, max(0, bx):bx + bw] = True
    st, form, raw, hgt = lidar_storeys(map_dir, m, ox, oy, s)
    tall_roofs = []
    if s.PARTS == "splits":
        keep_sq = np.zeros_like(m)
        for _name, (tx, ty, tw, th) in tower_box.items():
            keep_sq[ty:ty + th, tx:tx + tw] = True
        for pav in s.PAVILIONS:
            keep_sq[pav[1]:pav[1] + pav[3], pav[0]:pav[0] + pav[2]] = True
        m = lidar_footprint(m, hgt, keep_sq, s)
    no_towers = m.copy()
    for _name, (tx, ty, tw, th) in tower_box.items():
        no_towers[ty:ty + th, tx:tx + tw] = False
    if s.PARTS == "splits":
        # the towers' LiDAR spills past their walls (the 1 m grid's smear, the registration's
        # last metre): within TOWER_CLEAR of a tower, no higher than the roofs just beyond (a part
        # as tall as Elizabeth Tower stood beside it, its windows cutting across the clock)
        from scipy import ndimage as ndi
        hgt = hgt.copy()
        for _name, (tx, ty, tw, th) in tower_box.items():
            tb = np.zeros_like(m)
            tb[ty:ty + th, tx:tx + tw] = True
            near = ndi.binary_dilation(tb, iterations=s.TOWER_CLEAR) & ~tb & m
            ring = ndi.binary_dilation(tb, iterations=s.TOWER_CLEAR + 5) & ~ndi.binary_dilation(tb, iterations=s.TOWER_CLEAR) & m
            vals = hgt[ring & np.isfinite(hgt)]
            if near.any() and vals.size:
                cap = float(np.percentile(vals, 75))
                hgt[near] = np.minimum(np.where(np.isfinite(hgt[near]), hgt[near], cap), cap)
        st, range_parts, tall_roofs, range_of = lidar_parts(hgt, no_towers, s, form)
        set_by_spec = np.ones_like(m)                  # no raised parts looked for: the cuts found them
        # the tall roofs the hall pieces cover: the game's roof flat under them
        s._halls = [t for t in tall_roofs if t[4] == "NS" and t[2] in s.HALL_ROOF_WIDTHS]
        halls_at = {tuple(t[:4]) for t in s._halls}
        range_parts = [(x, y, w, h, "flat" if (x, y, w, h) in halls_at else f, k) for x, y, w, h, f, k in range_parts]
    else:
        st, range_parts, set_by_spec, range_of = range_heights(st, form, no_towers, s)
    for fx, fy, fw, fh, k, f in (s.RANGE_FIXES if s.PARTS != "splits" else []):
        sub = np.zeros_like(m)
        sub[fy:fy + fh, fx:fx + fw] = True
        sub &= no_towers
        if not sub.any():
            continue
        if k:
            st[sub] = k
        kk = k or int(np.median(st[sub]))
        range_parts.append((fx, fy, fw, fh, f or ("NS" if fh >= fw else "WE"), kk))
    for name, (tx, ty, tw, th) in tower_box.items():
        st[ty:ty + th, tx:tx + tw] = np.where(m[ty:ty + th, tx:tx + tw], s.TOWER_STOREYS[name], 0)
    st = np.where(m & (st == 0), 1, st)
    raised = raised_parts(raw, st, no_towers, s, form, tower_box, set_by_spec, range_of)
    pavilion_parts = []
    pavilion_dress = []                       # (x, y, w, h, storeys, roof piece) of the dressed ones
    for pav in s.PAVILIONS:
        px_, py_, pw, ph = pav[:4]
        sub = np.zeros_like(m)
        sub[py_:py_ + ph, px_:px_ + pw] = True
        sub &= m
        if sub.any() and len(pav) >= 6:
            # a pavilion the landmark describes: its walls' storeys, its own roof piece on its flat
            # top and a tall pinnacle on each corner (exterior)
            k = int(pav[4])
            st[sub] = k
            pavilion_parts.append((px_, py_, pw, ph, "flat", k))
            pavilion_dress.append((px_, py_, pw, ph, k, pav[5]))
        elif sub.any():
            # its own measured roof (the LiDAR's top inside it), at least a storey over the range
            # round it: a fixed rise overshot (Parliament's corner pavilions up to 15 m)
            k = int(max(np.median(st[sub]) + 1, np.rint(np.median(raw[sub])))) if (raw[sub] > 0).any()                 else int(max(st[sub].max(), np.median(st[sub]) + s.PAVILION_RISE))
            st[sub] = k
            pavilion_parts.append((px_, py_, pw, ph, "flat", k))
    rooms_out, doors, pieces = [], [], list(s.EDGES)
    unreached = {}
    if hasattr(s, "adjust_parts"):                # the landmark's own (a tower's drum standing on the roofs)
        st, range_parts = s.adjust_parts(st, range_parts, m)
    well_stairs, halls = stairwells(s, st)
    below = None
    every = list(range(int(st.max())))
    bases = [lvl for lvl in every if band_of(lvl, s)[0] == lvl]   # a floor's first storey: its rooms
    base_rooms, prev_base = {}, None
    areas = floor_areas(s, m.shape)
    for lvl in bases:
        # a floor laid only where it is (FLOOR_AREAS); elsewhere the floor below goes on
        rooms, owner = plan_level((st > lvl) & (areas[lvl] if lvl in areas else True), lvl, s, tower_box,
                                  halls.get(lvl, ()))
        ents = [d for d in s.OUTSIDE_DOORS if d[0] == lvl]
        here, cut = doors_for(rooms, owner, lvl, ents)
        # rooms cut off upstairs: a staircase of their own up from a reached room below, then again
        if cut and below is not None:
            ups = landing_stairs(rooms, owner, cut, below, s)
            if ups:                                   # up the whole floor below, storey by storey
                well_stairs += [[lv, x, y, sdir] for (x, y, sdir), _i in ups for lv in range(prev_base, lvl)]
                here, cut = doors_for(rooms, owner, lvl, ents, landings=[i for _r, i in ups])
        if cut:
            unreached[lvl] = len(cut)
            where = [(min(c[0] for c in rooms[i]["cells"]), min(c[1] for c in rooms[i]["cells"]),
                      len(rooms[i]["cells"])) for i in cut]
            print(f"storey {lvl}: {len(cut)} rooms cannot be reached (no neighbour reached): {where[:12]}")
        below = (owner, rooms, {i for i in range(len(rooms)) if i not in set(cut)})
        base_rooms[lvl], prev_base = rooms, lvl
        doors += here
        pieces += panelling(rooms, owner, lvl, here, s)
        for r in rooms:
            floor = s.FLOORS.get(r["name"]) or (s.FLOOR_DEFAULT if r["kind"] != "office" else None)
            rooms_out.append({"name": r["name"], "kind": r["kind"], "levels": [lvl],
                              "rects": rects_cover(r["cells"]), "furnish": r["kind"] == "office" and not r["void"],
                              **({"floor": floor} if floor and not r["void"] else {}),
                              **({"void": True} if r["void"] else {})})
        print(f"storey {lvl}: {len(rooms)} rooms ({sum(r['kind'] == 'office' for r in rooms)} offices)")
    # a floor's upper storeys: its first storey's rooms, open (no floor: tall rooms), but for the
    # rooms a staircase stands in or lands in and the stair halls, which keep their floors
    all_stairs = s.STAIRS + (s.extra_stairs(tower_box) if hasattr(s, "extra_stairs") else []) + well_stairs
    hall_names = {name for hs in halls.values() for name, _r in hs} | set(tower_box)
    theme = {name: (set(levels), set(voids)) for name, _k, levels, _r, voids in s.THEMED}
    for lvl in every:
        gov = governing(lvl, s, areas, m.shape)
        runs = set()
        for slvl, x, y, sdir in all_stairs:
            if slvl in (lvl, lvl - 1):
                for k in range(-1, s.STAIR_RUN + 1):
                    for d in (-1, 0, 1):
                        runs.add((x + d, y + k) if sdir == "N" else (x + k, y + d))
        n_open, n_any = 0, 0
        for g in sorted(set(np.unique(gov[st > lvl]).tolist())):
            if g == lvl:                               # laid on this storey (above)
                continue
            here_m = (st > lvl) & (gov == g)
            for r in base_rooms.get(g, []):
                cells = {(x, y) for x, y in r["cells"] if here_m[y, x]}
                if not cells:
                    continue
                n_any += 1
                floored = r["name"] in hall_names or bool(cells & runs)
                void = r["void"] or not floored
                name, kind = r["name"], r["kind"]
                top = g + band_of(g, s)[1]                 # the storey over that floor's band
                if name in theme:                          # a room of its own height (a chamber)
                    levels, voids = theme[name]
                    top = max(levels) + 1
                    if lvl in levels:
                        void = lvl in voids
                if lvl >= top and name not in hall_names:
                    # over the room's own top: its ceiling (the storey just over it, floored),
                    # then open roof space - but for a staircase's room
                    name, kind = "Roof space", "hall"
                    void = lvl > top and not (cells & runs)
                n_open += void
                floor = s.FLOORS.get(name) or (s.FLOOR_DEFAULT if kind != "office" else None)
                rooms_out.append({"name": name, "kind": kind, "levels": [lvl],
                                  "rects": rects_cover(cells), "furnish": kind == "office" and not void,
                                  **({"floor": floor} if floor and not void else {}),
                                  **({"void": True} if void else {})})
        if n_any:
            print(f"storey {lvl}: {n_any} rooms of the floor below it, {n_open} of them open")
    print(f"rooms that cannot be reached: {sum(unreached.values())} {unreached}")
    foot = [[ox + x, oy + y, w, h] for x, y, w, h in rects_of(m)]
    # roof: the ranges and the parts above them, then the towers, their own squares
    # as one rectangle each, flat and full height
    tower_rects = []
    for name, (tx, ty, tw, th) in tower_box.items():
        tower_rects.append([ox + tx, oy + ty, tw, th, "flat", s.TOWER_STOREYS[name]])
        print(f"{name}: {tw} x {th} at {ox + tx},{oy + ty}, {s.TOWER_STOREYS[name]} storeys")
    roof = [[int(ox + x), int(oy + y), int(w), int(h), f, int(k)] for x, y, w, h, f, k in range_parts + raised + pavilion_parts] \
        + tower_rects
    print("roofs:", {f: sum(1 for r in roof if r[4] == f) for f in ("NS", "WE", "flat")})
    windows, outside = exterior(m, st, tower_box, doors + s.OUTSIDE_DOORS, s, range_parts, raised,
                               pavilion_parts, pavilion_dress)
    pieces += outside
    # the hall roofs: gables on the end rows, segments between (a lucarne every few); the parapet's
    # ordinary pinnacles on the gable rows would stand in the gable wall
    for x, y, w, h, _f, k, _r in getattr(s, "_halls", []):
        y1 = y + h - 1
        pieces = [p for p in pieces if not (p[1] == "gothic_pinnacle" and p[3] in (y, y1) and x < p[2] < x + w - 1)]
        pieces += [[int(k), f"hall_gable_w{w}", int(x), int(y), "N"], [int(k), f"hall_gable_w{w}", int(x), int(y1), "S"]]
        for yy in range(y + 1, y1):
            dormer = (yy - y) % s.HALL_DORMER_EVERY == s.HALL_DORMER_EVERY // 2
            pieces.append([int(k), f"hall_roof_w{w}" + ("_dormer" if dormer else ""), int(x), int(yy), "S"])
    # a door where a window was drawn (a tower's window where the part beside it stands as tall):
    # the door's
    door_set = {(lv, x, y, sd) for lv, x, y, sd in doors + s.OUTSIDE_DOORS}

    def spans(w):                  # the squares a window covers: its width is in its name (perp_sq2: 2)
        lv, name, x, y, sd = w
        n = int(next((c for c in name if c.isdigit()), "1"))
        return [(lv, x, y + k, sd) if sd == "W" else (lv, x + k, y, sd) for k in range(n)]
    windows = [w for w in windows if not any(q in door_set for q in spans(w))]
    desc = {
        "entry": s.ENTRY, "name": s.NAME, "grade": s.GRADE,
        "osm_id": s.MAIN, "origin": [ox, oy], "size": [W, H],
        "about": s.ABOUT,
        "evidence": [e.format(pieces=len(keep)) for e in s.EVIDENCE],
        "style": s.STYLE,
        "walls": s.WALLS,
        "windows": windows,
        **({"clear_roof": s.CLEAR_ROOF} if s.CLEAR_ROOF else {}),
        "footprint": foot,
        "join": [k for k in keep if k != s.MAIN],
        "roof": roof,
        "full_height": s.FULL_HEIGHT,
        "blank": s.BLANK,                 # no windows of Build's own: every one is described
        "levels": int(st.max()),
        "rooms": rooms_out,
        "doors": doors + s.OUTSIDE_DOORS,
        "pieces": pieces,
        "furniture": s.FURNITURE,
        "stairs": all_stairs,
        **({"towers": s.TOWERS} if s.TOWERS else {}),
        **({"tall_roofs": [[int(ox + x), int(oy + y), w, h, f, k, r] for x, y, w, h, f, k, r in tall_roofs]}
           if tall_roofs else {}),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(spec.__file__)), s.OUT)
    json.dump(desc, open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out}: {len(rooms_out)} rooms, {len(desc['doors'])} doors, {len(foot)} footprint rectangles")
    return 0
