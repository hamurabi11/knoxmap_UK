"""45-degree roads redrawn cleanly: one width all along, a tight pavement, no blobs.

Called by kerbs.py on the ground as Generate drew it, before anything else.

Generate draws a road as a thick line with a round blob at every bend point
(renderer._draw_line). Along the grid that is exact; on the diagonal the
tarmac's width jumps by a tile wherever two segments join or a blob pokes
out, and the pavement comes out ragged - one tile a row in places, squares
touching only at their corners. Here every 45-degree stretch of road (the
straightened lines Generate drew from) is drawn again:

- the tarmac the same number of tiles in every row: its width x sqrt 2,
  rounded, on a line snapped to whole tiles, so two segments of one road
  agree where they meet;
- the pavement, where the road has one, a band of PAVE_ROW tiles a row each
  side, edge to edge with the tarmac: the tightest a pavement can follow a
  diagonal while its squares still share sides;
- what is left of the old tarmac and pavement in the stretch back to the
  ground beside it.

Only the middle of a stretch is redrawn (END_TILES clear of each end), so
junctions and bends keep Generate's drawing; a building's tiles are never
touched. Returns the new ground, a mask of the redrawn bands (kerbs.py lays
no kerbs on them), a count, and each stretch's line (down: along x = y;
u0, n: its tarmac's first value of u and tiles a row; side: pavement tiles a
row; lo, hi: its extent in v) for the garden fronts kerbs.py redraws.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)

CATEGORIES = ("road_major", "road_medium", "road_minor", "road_service")
DIAGONAL_TOL = 0.08          # |dx| and |dy| within this share of the length: a 45-degree segment
MIN_TILES = 8                # shorter stretches are left as drawn
END_TILES = 3                # kept clear of each end, beyond the road's own width
PAVE_ROW = 2                 # pavement tiles a row on the diagonal
SEARCH = 1.5                 # tiles past the old drawing's width where old road and pavement are cleared


def _segments(map_dir: str, info: dict, proj):
    """(category, feature, (ax, ay), (bx, by), diagonal) for every road segment."""
    from generator import osm
    from generator.octilinear import straighten_roads
    from generator.renderer import _is_polygon, classify
    feats = osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or []
    if info.get("straight_roads"):
        straighten_roads(feats, proj, classify, _is_polygon, map_dir=map_dir)
    out = []
    for f in feats:
        if f.kind != "way" or _is_polygon(f):
            continue
        cat = classify(f.tags, False)
        if cat not in CATEGORIES:
            continue
        pts = [proj.to_px(la, lo) for la, lo in f.geometry]
        for a, b in zip(pts, pts[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            diagonal = (length >= MIN_TILES * math.sqrt(2)
                        and abs(abs(dx) - abs(dy)) <= DIAGONAL_TOL * length)
            out.append((cat, f, a, b, diagonal))
    return out


def clean(ground: np.ndarray, map_dir: str, info: dict, proj, road_colours, pave_colour,
          is_colour, buildings: np.ndarray | None):
    from generator.renderer import _sidewalk_width_m, _way_width_m
    H, W = ground.shape[:2]
    if info.get("road_graph"):
        # The road graph drew its diagonals one width all along already
        # (roadgraph/handover.py): nothing to redraw.
        return ground.copy(), np.zeros((H, W), bool), 0, []
    mpt = info["meters_per_tile"]
    road0 = is_colour(ground, road_colours)
    pave0 = is_colour(ground, (pave_colour,))
    street0 = road0 | pave0
    blocked = buildings if buildings is not None else np.zeros((H, W), bool)
    # The ground beside the street, for squares the street gives up.
    _, (iy, ix) = ndimage.distance_transform_edt(street0, return_indices=True)
    out = ground.copy()
    zone = np.zeros((H, W), bool)
    info_out = []                                   # each stretch redrawn, for its gardens' fronts
    ys, xs = np.mgrid[0:H, 0:W]
    stretches = 0
    from PIL import Image, ImageDraw
    from generator.renderer import _pixel_width
    segs = _segments(map_dir, info, proj)

    def drawn_tarmac(items):
        img = Image.new("1", (W, H), 0)
        dr = ImageDraw.Draw(img)
        for cat, f, a, b, _diagonal in items:
            dr.line([a, b], fill=1, width=_pixel_width(_way_width_m(f, cat), mpt))
        return np.array(img)

    def line_of(a, b):
        """(runs along x = y, its across value u) of a diagonal segment."""
        down = ((b[0] - a[0]) > 0) == ((b[1] - a[1]) > 0)
        return down, (a[0] - a[1]) if down else (a[0] + a[1])
    # Every other road's tarmac, as Generate drew it: a side road joining the
    # diagonal keeps its mouth (no pavement across it, nothing cleared). The
    # roads along the grid are kept for every stretch; a diagonal one only
    # for a stretch it is not in line with - a diagonal side road meeting a
    # diagonal road square-on lost its mouth to the new pavement and was cut
    # off (LOG 2026-10-10).
    keep_grid = drawn_tarmac([s for s in segs if not s[4]])
    diagonals = [s for s in segs if s[4]]
    laid = np.zeros((H, W), bool)                    # tarmac the stretches before laid
    for i, (cat, f, a, b, diagonal) in enumerate(diagonals):
        down, ua_ = line_of(a, b)
        across = [s for j, s in enumerate(diagonals)
                  if j != i and (line_of(s[2], s[3])[0] != down
                                 or abs(line_of(s[2], s[3])[1] - ua_) > 2)]
        keep = keep_grid | drawn_tarmac(across)
        dx, dy = b[0] - a[0], b[1] - a[1]
        down = (dx > 0) == (dy > 0)                  # running along x = y, else x = -y
        # Across the road: u (whole tiles, constant along the diagonal);
        # along it: v.
        u_of = (xs - ys) if down else (xs + ys)
        v_of = (xs + ys) if down else (xs - ys)
        ua = (a[0] - a[1]) if down else (a[0] + a[1])
        va, vb = sorted(((a[0] + a[1]) if down else (a[0] - a[1]),
                         (b[0] + b[1]) if down else (b[0] - b[1])))
        n_guess = max(2, int(round(_pixel_width(_way_width_m(f, cat), mpt) * math.sqrt(2))))
        side = PAVE_ROW if _sidewalk_width_m(f, cat) > 0 else 0
        trim = (n_guess / 2 + END_TILES) * 2             # in v units (x + y moves 2 a diagonal step)
        lo, hi = va + trim, vb - trim
        if hi - lo < 2 * MIN_TILES:
            continue
        along = (v_of >= lo) & (v_of <= hi)
        # The tarmac as wide as Generate drew it here (its median tiles a row),
        # so the redrawn middle meets the ends it drew without a step.
        drawn = along & road0 & (np.abs(u_of - ua) <= n_guess) & ~keep
        rows = np.bincount(ys[drawn])
        rows = rows[rows > 0]
        n = int(np.median(rows)) if len(rows) else n_guess
        u0 = int(math.floor(ua - (n - 1) / 2 + 0.5))     # whole tiles: segments of one road agree
        road_band = along & (u_of >= u0) & (u_of < u0 + n)
        pave_band = along & (((u_of >= u0 - side) & (u_of < u0)) | ((u_of >= u0 + n) & (u_of < u0 + n + side)))
        # The old drawing reached about its own width and pavement either side
        # of the line, a tile or two more on the diagonal: cleared within that.
        old_side = _sidewalk_width_m(f, cat) / mpt * math.sqrt(2)
        reach = n / 2 + old_side + SEARCH
        near = along & (np.abs(u_of - ua) <= reach)
        # The old street in reach goes back to the ground beside it, then the
        # new bands are laid; neither over a building.
        # Old pavement that runs on into wider paving (a yard, a forecourt)
        # is part of that, not of the road: left.
        joined = ndimage.binary_dilation(pave0 & ~near, iterations=2) & pave0
        give_up = near & street0 & ~road_band & ~pave_band & ~blocked & ~keep & ~joined & ~laid
        out[give_up] = ground[iy[give_up], ix[give_up]]
        tarmac = ground[road0 & road_band]
        colour = tarmac[0] if len(tarmac) else ground[road0][0]
        out[road_band & ~blocked] = colour
        # Never pavement over tarmac a stretch before laid: where one road's
        # stretches overlap at a bend, a stripe of pavement ran down the road.
        out[pave_band & ~blocked & ~keep & ~laid] = pave_colour
        out[pave_band & ~blocked & keep] = colour        # a side road's mouth
        laid |= road_band & ~blocked
        zone |= (road_band | pave_band) & ~blocked & ~keep
        # A one-tile sliver left between the new pavement and paving beyond it
        # (a yard) is paved too.
        pv = is_colour(out, (pave_colour,))
        sliver = near & ~pv & ~is_colour(out, road_colours) & ~blocked
        sliver[:, 1:-1] &= pv[:, :-2] & pv[:, 2:]
        sliver[:, [0, -1]] = False
        out[sliver] = pave_colour
        info_out.append(dict(down=down, u0=u0, n=n, side=side, lo=lo, hi=hi))
        stretches += 1
    return out, zone, stretches, info_out
