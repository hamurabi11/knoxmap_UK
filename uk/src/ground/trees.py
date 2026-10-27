"""Trees where they really stand, British species, on boundaries where they are.

    python trees.py <knoxmap map dir> <area data dir> [--garden N] [--picture out.png] [--box x,y,w,h]

Run after plots2.py (it needs the plots, <map>_plots.npz) and before Build.
Needs the UK tree rules in the map tools (uk_tree_rules.py, once).

The game map had trees only where OSM maps a wood; gardens had none. Odiham
has some 2,300 trees of 4 m and more in the town, and they stand on property
lines: 38% of tree tops within 2 m of one, where 10% of random points in the
same plots are (LOG 2026-10-07).

1. Tree tops from the National LiDAR Programme point cloud: the highest
   vegetation point (classes 3-5) above the ground on a 1 m grid, smoothed,
   and its local maxima at least TOP_M high and TOP_APART apart; canopy
   further than FILL_M from any top (a belt, crowns grown together) gets a
   top at its highest points too. Each top's crown is the canopy (CANOPY_M
   up) nearer it than any other top.
2. Into the game through the plot it stands in: tried with the frames of
   the plots whose houses are nearest (blocks.py's turn and shift for that
   house), the first that puts it inside that plot keeps it; else the
   nearest house's frame places it on open ground. Trees further than
   FAR_TILES from any house are left to the map's own woods.
   A plot with GARDEN_GREEN trees and shrubs or more in it is a garden:
   KnoxMap paves built-up ground, and the High Street's back land came out as
   paved yards where the point cloud shows lawns and trees. Its paving and
   gravel become grass, but for a patio a tile wide along the house; and
   pavement painted inside a home's garden (round a junction's corner; not
   its drive, a path or by a wall) is grass, its lamps and hydrants gone; written
   into <map>_ground_base.bmp, which Build reads (the ground as it was is
   kept as <map>_ground_base.pretrees.bmp).
3. A tree within SNAP_M of a real property line (HM Land Registry INSPIRE,
   where there is the file) goes onto its game plot's boundary: the tile of
   the plot nearest it along the plot's edge.
4. Never on a building or next to its wall, on a street (a grass verge is
   fine), drive, path or paving, in water, or within a tile of another tree; a tile that is any of
   those is swapped for the nearest free one within NUDGE tiles.
5. Species (uk_tree_rules.py's colours): a big tree (TALL_M, or crown
   BIG_CROWN_M) the same mix but only species with a big sprite (lime,
   maple, birch, hawthorn); on a boundary mostly hawthorn and holly, the
   hedgerow trees; in a garden mostly birch and maple, some lime, hawthorn,
   holly and a few conifers; on open ground hawthorn, maple, lime, birch.
   Each tree as big as its crown (SIZE_BY_CROWN): the jumbo, XL or XXL
   sprite of its species, 3, 5 or 7 tiles across where a normal tree is
   one; a size that has no room steps down to the next.
6. OSM woods are repainted in the British mix (255 0 0 -> 255 7 0, and the
   trees-and-grass colour 127 0 0 -> 127 7 0).
Written into <map>_veg_base.bmp, which Build reads, and <map>_veg.bmp; the
vegetation as it was before is kept as <map>_veg_base.pretrees.bmp and each
run starts again from it (delete it after a new Generate).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from buildings import blocks as B  # noqa: E402
from generator import pz_colors as C  # noqa: E402
from area import area_conf  # noqa: E402
from area.roofs import mosaic  # noqa: E402
from ground.uk_tree_rules import BIG, SIZES, SPECIES, WOOD_GRASS  # noqa: E402

TOP_M = 3.5            # a tree top this high or more
TOP_APART = 5          # ...the highest within this many metres (a 5 m window; at 7 a belt of
                       # trees came out as a few)
CANOPY_M = 3.0         # vegetation this high is crown
FAR_TILES = 150        # a tree further than this from any house is placed where it really is
                       # (the straightening fades out away from the roads); 1,620 were left out
SNAP_M = 3.0           # a tree this near a real property line goes on the game boundary
SNAP_REACH = 8         # ...the boundary tile nearest it within this many tiles
NUDGE = 2              # a blocked tile is swapped for a free one this near
TALL_M = 9.0           # a tree this tall, or with a crown...
# ...this wide (radius, m) is drawn this big: (size, tiles across, tiles kept
# clear of walls, of other trunks). Measured crowns: radius p25 3.1 m, median
# 4.0, p75 5.0 - a normal sprite is one tile across, and the map 1 m a tile.
# A sprite as wide as the crown: 7 tiles from a 6.5 m crown, 5 from 4.5,
# 3 from 2.5 (at 5 / 3.5 / 2.5 m radius the drawn crowns covered a third of
# the real canopy). Trunks may stand close to a wall: real crowns overhang
# roofs.
SIZE_BY_CROWN = [(3.25, "JUMBOXXL", 7, 2, 3), (2.25, "JUMBOXL", 5, 1, 2), (1.25, "JUMBO", 3, 1, 2)]
FILL_M = 4.0           # canopy further than this from any top gets a tree of its own (belts, merged crowns)

MIX = {   # species shares by where a tree stands
    "big": {"lime": 0.5, "maple": 0.5},
    "boundary": {"hawthorn": 0.45, "holly": 0.1, "maple": 0.25, "birch": 0.2},
    "garden": {"birch": 0.3, "maple": 0.27, "lime": 0.15, "hawthorn": 0.15, "holly": 0.05, "conifer": 0.08},
    "open": {"hawthorn": 0.3, "maple": 0.3, "lime": 0.2, "birch": 0.2},
}
GARDEN_GREEN = 2       # a plot with this many trees and shrubs or more is a garden, not paving (--garden)
SHRUB_M = 1.0          # a shrub: vegetation this high...
SHRUB_APART = 3        # ...the highest within this many metres, below TOP_M
PAVED_YARD = (C.PALE_CONCRETE, C.PAVING, C.LIGHT_ASPHALT)   # paving a garden turns to grass
HARD = (C.MEDIUM_ASPHALT, C.DARK_ASPHALT, C.DARKEST_ASPHALT, C.LIGHT_ASPHALT, C.PALE_CONCRETE,
        C.PAVING, C.PAVING_STONE, C.DIRT, C.WATER, C.DARK_POTHOLE, C.LIGHT_POTHOLE, C.SAND)


def to_game(x, y, fr):
    """A point in the real map's tiles into the game's (blocks.py's frame)."""
    t, dx, dy, gx, gy = fr
    a = math.radians(t)
    ux, uy = x + dx - gx, y + dy - gy
    return gx + ux * math.cos(a) - uy * math.sin(a), gy + ux * math.sin(a) + uy * math.cos(a)


def pick(mix, key):
    """A species from a mix, the same every run for the same tile."""
    r = (hash(key) & 0xFFFFFF) / 0x1000000
    acc = 0.0
    for name, share in mix.items():
        acc += share
        if r < acc:
            return name
    return name


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    garden_green = int(sys.argv[sys.argv.index("--garden") + 1]) if "--garden" in sys.argv else GARDEN_GREEN
    picture = sys.argv[sys.argv.index("--picture") + 1] if "--picture" in sys.argv else None
    view = [int(v) for v in sys.argv[sys.argv.index("--box") + 1].split(",")] if "--box" in sys.argv else None
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    b = info["bbox"]
    proj = B.Projector.build(b["south"], b["west"], b["north"], b["east"], info["meters_per_tile"],
                             info.get("rotation") or 0.0)
    W, H = proj.width, proj.height
    P = np.load(os.path.join(map_dir, f"{name}_plots.npz"))
    plot, frames, building, street = P["plot"], P["frames"], P["building"], P["street"]

    # --- 1: tree tops ----------------------------------------------------------------
    corners = [proj.to_latlon(x, y) for x, y in ((0, 0), (W, 0), (0, H), (W, H))]
    es, ns = zip(*[B.TO_BNG(lon, lat) for lat, lon in corners])
    E0, N0, E1, N1 = int(min(es)), int(min(ns)), int(max(es)) + 1, int(max(ns)) + 1
    d = np.load(os.path.join(area, "nlp", "core.npz"))
    x, y, z, cls = d["x"], d["y"], d["z"], d["cls"]
    m = (x >= E0) & (x < E1) & (y >= N0) & (y < N1) & (cls >= 3) & (cls <= 5)
    x, y, z = x[m], y[m], z[m]
    dtm = mosaic("dtm", (E0, N0, E1, N1), os.path.join(area, "lidar"))
    r = np.clip((N1 - y).astype(int), 0, dtm.shape[0] - 1)
    c = np.clip((x - E0).astype(int), 0, dtm.shape[1] - 1)
    chm = np.zeros(dtm.shape, np.float32)
    np.maximum.at(chm, (r, c), (z - np.nan_to_num(dtm[r, c], nan=1e9)).astype(np.float32))
    sm = ndimage.gaussian_filter(chm, 1.0)
    peak = (sm == ndimage.maximum_filter(sm, size=TOP_APART)) & (sm >= TOP_M)
    # Canopy no top accounts for - a belt of trees, crowns grown together -
    # gets trees of its own, at its highest points.
    gap = (chm >= CANOPY_M) & (ndimage.distance_transform_edt(~peak) > FILL_M)
    gap_sm = np.where(gap, sm, 0.0)
    fill = gap & (gap_sm == ndimage.maximum_filter(gap_sm, size=int(2 * FILL_M) + 1))
    tops = np.argwhere(peak | fill)
    lab = np.zeros(chm.shape, np.int32)
    lab[tops[:, 0], tops[:, 1]] = np.arange(1, len(tops) + 1)
    dist, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    own = np.where((chm >= CANOPY_M) & (dist <= 10), lab[iy, ix], 0)
    crown = np.sqrt(np.bincount(own.ravel(), minlength=len(tops) + 1)[1:] / np.pi)
    height = sm[tops[:, 0], tops[:, 1]]
    te, tn = E0 + tops[:, 1] + 0.5, N1 - tops[:, 0] - 0.5

    # Real property lines, for the trees that stand on them.
    lines = None
    ipath = area_conf.extra(area, "inspire.geojson")
    if ipath is not None:
        box_ = shapely.box(E0, N0, E1, N1)
        polys = [shape(f["geometry"]) for f in json.load(open(ipath, encoding="utf-8"))["features"]]
        lines = unary_union([p.boundary for p in polys if p.intersects(box_)])
    on_line = (shapely.distance(lines, shapely.points(te, tn)) <= SNAP_M) if lines is not None \
        else np.zeros(len(tops), bool)

    # --- 2: into the game -----------------------------------------------------------------
    have = np.nonzero(~np.isnan(frames[:, 0]))[0]               # plots with a frame (0-based)
    if not len(have):
        # No plot carries a building's LiDAR frame (KnoxMap fitted the outlines itself:
        # uk_pipeline.py's buildings "knoxmap"): nothing says where a real tree lands in
        # the game, so the map keeps KnoxMap's own trees.
        print("no plot has a LiDAR frame (no blocks.py): KnoxMap's own trees kept")
        return 0
    centres = np.c_[frames[have, 3] - frames[have, 1], frames[have, 4] - frames[have, 2]]
    tree_ = cKDTree(centres)

    def locate(e, n):
        """A real point (BNG) into the game: (plot or 0, x, y), or None off the map.
        Through the plot it stands in - the frames of the nearest houses, the
        first that lands it inside that house's plot - else the nearest
        house's frame; far from any house, where it really is."""
        lon_, lat_ = B.TO_WGS(e, n)
        rx, ry = proj.to_px(lat_, lon_)
        dd, idx = tree_.query((rx, ry), k=min(6, len(have)), distance_upper_bound=FAR_TILES)
        cands = [int(have[j]) for dist_, j in zip(np.atleast_1d(dd), np.atleast_1d(idx)) if np.isfinite(dist_)]
        for kk in cands:
            px, py = to_game(rx, ry, frames[kk])
            tx, ty = int(math.floor(px)), int(math.floor(py))
            if 0 <= tx < W and 0 <= ty < H and plot[ty, tx] == kk + 1:
                return kk + 1, tx, ty
        px, py = to_game(rx, ry, frames[cands[0]]) if cands else (rx, ry)
        gx, gy = int(math.floor(px)), int(math.floor(py))
        if not (0 <= gx < W and 0 <= gy < H):
            return None
        return int(plot[gy, gx]), gx, gy

    spots = [locate(te[i], tn[i]) for i in range(len(tops))]

    # --- 3: gardens, not paving -------------------------------------------------------------
    # KnoxMap paves built-up ground, so the back land of the High Street came
    # out as paved yards, where the point cloud shows lawns and trees. A plot
    # with GARDEN_GREEN trees or shrubs (a shrub: a high point of vegetation
    # SHRUB_M to TOP_M up) is a garden: its paving is grass, but for a patio
    # a tile wide along the house.
    shrub = (sm == ndimage.maximum_filter(sm, size=SHRUB_APART)) & (sm >= SHRUB_M) & (sm < TOP_M)
    green = np.zeros(int(plot.max()) + 1, np.int32)
    for sp_ in spots:
        if sp_ and sp_[0]:
            green[sp_[0]] += 1
    for rr_, cc_ in np.argwhere(shrub):
        sp_ = locate(E0 + cc_ + 0.5, N1 - rr_ - 0.5)
        if sp_ and sp_[0]:
            green[sp_[0]] += 1
    garden_plots = np.nonzero(green >= garden_green)[0]
    garden_plots = garden_plots[garden_plots > 0]
    gb_path = os.path.join(map_dir, f"{name}_ground_base.bmp")
    bmp_path = os.path.join(map_dir, f"{name}.bmp")
    ground_before = os.path.join(map_dir, f"{name}_ground_base.pretrees.bmp")
    if not os.path.exists(ground_before):
        Image.open(gb_path if os.path.exists(gb_path) else bmp_path).save(ground_before, format="BMP")
    ground = np.array(Image.open(ground_before).convert("RGB"))[:H, :W]
    paved = np.zeros((H, W), bool)
    for col in PAVED_YARD:
        paved |= np.all(ground == col, axis=2)
    # A shop's forecourt (plots2.py) is paved, whatever grows there.
    forecourt = P["forecourt"] if "forecourt" in P.files else np.zeros((H, W), bool)
    to_grass = np.isin(plot, garden_plots) & paved & ~street & ~ndimage.binary_dilation(building) & ~forecourt
    ground[to_grass] = C.MEDIUM_GRASS
    # Pavement the hand-over painted inside a garden - round a junction's
    # corner, past where the links' outlines stop (roadside.draw paints there
    # by distance) - is garden: the front stands on the pavement's edge, not
    # across it (user, 2026-10-14: fences at junctions; Odiham 1,587 squares,
    # three in four within 12 tiles of a link's end).
    rs_path = os.path.join(map_dir, f"{name}_roadside.npz")
    pave_in_garden = np.zeros((H, W), bool)
    if os.path.exists(rs_path):
        with np.load(rs_path) as rs:
            homes = P["home"] if "home" in P.files else np.ones(int(plot.max()), bool)
            home_plot = np.zeros((H, W), bool)
            home_plot[plot > 0] = homes[plot[plot > 0] - 1]
            pave_in_garden = (rs["pavement"][:H, :W] & home_plot & ~street & ~forecourt & ~P["drive"] & ~P["path"]
                              & np.all(ground == C.PALE_CONCRETE, axis=2) & ~ndimage.binary_dilation(building))
            # (a home's garden only, still painted pavement, off its drive and
            # paths, not against a wall - a shop's way to its door: the review)
        ground[pave_in_garden] = C.MEDIUM_GRASS
    ground[forecourt] = C.PAVING
    for gp in (gb_path, bmp_path):
        full = np.array(Image.open(ground_before).convert("RGB"))
        full[:H, :W] = ground
        Image.fromarray(full).save(gp, format="BMP")
    hard = np.zeros((H, W), bool)
    for col in HARD:
        hard |= np.all(ground == col, axis=2)
    hard &= ~forecourt                    # trees stand in a forecourt, in a planter or the paving

    veg_paths = [os.path.join(map_dir, f"{name}_veg_base.bmp"), os.path.join(map_dir, f"{name}_veg.bmp")]
    # Each run starts again from the vegetation as it was before any trees
    # (delete it after a new Generate or blocks.py run).
    before = os.path.join(map_dir, f"{name}_veg_base.pretrees.bmp")
    if not os.path.exists(before):
        Image.open(veg_paths[0]).save(before, format="BMP")
    veg = np.array(Image.open(before).convert("RGB"))[:H, :W]
    veg[pave_in_garden] = C.VEG_NOTHING           # the street's dressing (lamps, hydrants) went with its pavement
    wooded = np.all(veg == C.TREES, axis=2) | np.all(veg == C.TREES_DARK_GRASS, axis=2)
    # A road-graph map's grass verges (roadgraph/roadside.py) are street to
    # plots2.py, but a real tree on one is a street tree: it stays.
    rs_path = os.path.join(map_dir, f"{name}_roadside.npz")
    verge = np.zeros((H, W), bool)
    if os.path.exists(rs_path):
        v = np.load(rs_path)["verge"][:H, :W]
        verge[:v.shape[0], :v.shape[1]] = v
    blocked = (ndimage.binary_dilation(building) | (street & ~verge) | hard | wooded)
    clear_of = {1: blocked}
    for _c, _s, _w, wall, _t in SIZE_BY_CROWN:
        clear_of.setdefault(wall, ndimage.binary_dilation(building, iterations=wall) | blocked)
    taken = np.zeros((H, W), np.int8)            # each trunk: how far it keeps others off
    padp = np.pad(plot, 1)
    edge = (plot > 0) & ((padp[:-2, 1:-1] != plot) | (padp[2:, 1:-1] != plot)
                         | (padp[1:-1, :-2] != plot) | (padp[1:-1, 2:] != plot))

    def free(tx, ty, wall=1, apart=1):
        """Room for a trunk here: wall tiles clear of buildings, and no other
        trunk nearer than the larger of the two trees' spacing."""
        if not (0 <= tx < W and 0 <= ty < H) or clear_of[wall][ty, tx]:
            return False
        win = taken[max(0, ty - 4):ty + 5, max(0, tx - 4):tx + 5]
        if not win.any():
            return True
        oy_, ox_ = np.nonzero(win)
        dd = np.maximum(np.abs(oy_ + max(0, ty - 4) - ty), np.abs(ox_ + max(0, tx - 4) - tx))
        return bool((dd >= np.maximum(win[oy_, ox_], apart)).all())

    def nearest_free(tx, ty, k, wall=1, apart=1):
        """This tile, or the nearest free one within NUDGE, in plot k if k."""
        best = None
        for ddy in range(-NUDGE, NUDGE + 1):
            for ddx in range(-NUDGE, NUDGE + 1):
                qx, qy = tx + ddx, ty + ddy
                if free(qx, qy, wall, apart) and (not k or plot[qy, qx] == k):
                    dd = ddx * ddx + ddy * ddy
                    if best is None or dd < best[0]:
                        best = (dd, qx, qy)
        return best and best[1:]

    # --- 4-5: the trees ---------------------------------------------------------------------
    placed = []                                   # (x, y, species, where, size)
    count = {"outside the map": 0, "no room": 0, "in a wood already": 0}
    for i in range(len(tops)):
        if spots[i] is None:
            count["outside the map"] += 1
            continue
        k, gx, gy = spots[i]
        if wooded[gy, gx]:
            count["in a wood already"] += 1
            continue
        where = "garden" if k else "open"
        if k and on_line[i]:
            # Onto the plot's boundary: its edge tile nearest the tree.
            y0, y1, x0, x1 = max(0, gy - SNAP_REACH), gy + SNAP_REACH + 1, max(0, gx - SNAP_REACH), gx + SNAP_REACH + 1
            ey, ex = np.nonzero(edge[y0:y1, x0:x1] & (plot[y0:y1, x0:x1] == k) & ~blocked[y0:y1, x0:x1])
            if len(ey):
                j = int(np.argmin((ey + y0 - gy) ** 2 + (ex + x0 - gx) ** 2))
                gx, gy, where = int(ex[j] + x0), int(ey[j] + y0), "boundary"
        # As big as its crown, or the next size down that has room.
        sizes = [(s_, wall, apart) for c_, s_, _w, wall, apart in SIZE_BY_CROWN
                 if crown[i] >= c_ or (s_ == "JUMBO" and height[i] >= TALL_M)]
        size, spot = None, None
        for s_, wall, apart in sizes:
            spot = nearest_free(gx, gy, k, wall, apart)
            if spot is not None:
                size = (s_, apart)
                break
        if spot is None:
            spot = nearest_free(gx, gy, k)
        if spot is None:
            count["no room"] += 1
            continue
        gx, gy = spot
        # A big tree: the species of where it stands that have a big sprite.
        mix = {n: v for n, v in MIX[where].items() if n in BIG} if size else MIX[where]
        sp = pick(mix if sum(mix.values()) else MIX["big"], (gx, gy))
        taken[gy, gx] = size[1] if size else 1
        placed.append((gx, gy, sp, where, size[0] if size else None))

    # --- 6: write ---------------------------------------------------------------------------
    for vp in veg_paths:
        v = np.array(Image.open(before).convert("RGB"))
        v[np.all(v == C.TREES, axis=2)] = SPECIES["wood"][0]
        v[np.all(v == C.TREES_DARK_GRASS, axis=2)] = WOOD_GRASS[0]
        for gx, gy, sp, _w, size in placed:
            v[gy, gx] = (SIZES[size] if size else SPECIES)[sp][0]
        Image.fromarray(v).save(vp, format="BMP")

    by_where, by_sp = {}, {}
    across = {s_: w_ for _c, s_, w_, _a, _b in SIZE_BY_CROWN}
    by_size, cover = {}, 0.0
    for _x, _y, sp, wh, size in placed:
        by_where[wh] = by_where.get(wh, 0) + 1
        by_sp[sp] = by_sp.get(sp, 0) + 1
        by_size[size or "normal"] = by_size.get(size or "normal", 0) + 1
        cover += math.pi * (across.get(size, 1) / 2) ** 2
    # The real canopy inside the map (the survey box is wider than the turned map).
    cr, cc_ = np.nonzero(chm >= CANOPY_M)
    lon_c, lat_c = B.TO_WGS(E0 + cc_ + 0.5, N1 - cr - 0.5)
    real_cover = float(sum(1 for la, lo in zip(np.atleast_1d(lat_c), np.atleast_1d(lon_c))
                           if 0 <= proj.to_px(la, lo)[0] < W and 0 <= proj.to_px(la, lo)[1] < H))
    print(f"tree tops {len(tops)} (>= {TOP_M:.1f} m, and canopy gaps filled); crown radius median {np.median(crown):.1f} m; "
          f"on a real property line {int(on_line.sum())}")
    print(f"  trees placed {len(placed)}: {by_where}; not placed: {count}")
    print(f"  species: {dict(sorted(by_sp.items(), key=lambda kv: -kv[1]))}")
    print(f"  sizes: {by_size}; crowns drawn cover about {cover:.0f} tiles, "
          f"the real canopy (>= {CANOPY_M:.0f} m) inside the map {real_cover:.0f} m2 ({cover / max(1, real_cover):.0%})")
    print(f"  woods repainted in the British mix: {int(wooded.sum())} tiles")
    print(f"  gardens (>= {garden_green} trees and shrubs, of {int(shrub.sum())} shrubs found): "
          f"{len(garden_plots)} of {int(plot.max())} plots; paving turned to grass: {int(to_grass.sum())} tiles; "
          f"pavement inside gardens turned to grass: {int(pave_in_garden.sum())} tiles")

    if picture:
        x0, y0, w, h = view or (0, 0, W, H)
        K = 4
        img = Image.fromarray(ground[y0:y0 + h, x0:x0 + w]).resize((w * K, h * K), Image.NEAREST)
        dr = ImageDraw.Draw(img)
        col = {"lime": (40, 160, 40), "maple": (200, 120, 20), "birch": (230, 230, 150),
               "hawthorn": (255, 255, 255), "holly": (10, 70, 20), "conifer": (20, 110, 110)}
        for yy, xx in zip(*np.nonzero(building[y0:y0 + h, x0:x0 + w])):
            dr.rectangle([xx * K, yy * K, xx * K + K - 1, yy * K + K - 1], fill=(40, 40, 40))
        for gx, gy, sp, _w, _big in placed:
            if x0 <= gx < x0 + w and y0 <= gy < y0 + h:
                cx, cy = (gx - x0) * K + K // 2, (gy - y0) * K + K // 2
                dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=col[sp], outline=(0, 0, 0))
        img.save(picture)
        print("wrote", picture)
    return 0


if __name__ == "__main__":
    sys.exit(main())
