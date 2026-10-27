"""Turn OSM fence and wall lines into Project Zomboid fence tiles.

A fence in the game is a tile on a tile's west or north edge, the way a wall
is. The styles below are copied from the mapping tools' Fences.txt, which is
what WorldEd's own fence tool draws with, and placed by the same rule that
tool uses (FenceTool::getWestEdgeTiles and its north twin in TileZed): along a
west edge, west1 and west2 alternate; along a north edge, north1 and north2;
a tile carrying both takes the corner piece; a loose end gets a post.

The fences are written as furniture in a building with no rooms, one per map
cell, so they compile into the lots like anything else in a .tbx.
"""
from __future__ import annotations

import collections
import math
import os
from xml.sax.saxutils import quoteattr

import knoxfiles

from . import catalog as C

CELL = 300

# From PZ_Mapping_Tools/config/Fences.txt. Names are zero-padded to three
# digits, the form BuildingFurniture.txt uses; both parse to the same tile.
STYLES = {
    "short_wooden": dict(west1=35, west2=34, north1=32, north2=33, nw=36, post=37),
    "tall_wooden": dict(west1=11, west2=10, north1=8, north2=9, nw=12, post=13),
    "short_chainlink": dict(west1=27, west2=26, north1=24, north2=25, nw=28, post=29),
    "tall_chainlink": dict(west1=59, west2=58, north1=56, north2=57, nw=60, post=61),
    "white_picket": dict(west1=4, west2=4, north1=5, north2=5, nw=6, post=7),
    "cattle": dict(west1=20, west2=20, north1=21, north2=21, nw=18, post=19),
    "black_metal": dict(west1=2, west2=2, north1=1, north2=1, nw=3, post=0),
    # The full-height bars of the Rosewood base wall. The game has both and
    # OSM says which: metal bars two metres tall came out half that, because
    # a fence's height was never read.
    "tall_metal": dict(west1=67, west2=66, north1=64, north2=65, nw=68, post=69),
    "short_picket": dict(west1=121, west2=121, north1=120, north2=120,
                         nw=122, post=123),
    "tall_concrete": dict(west1=40, west2=40, north1=41, north2=41, nw=42, post=43),
    # Red brick garden walls, from the house wall sets rather than Fences.txt
    # (an English town's boundaries are brick): a third of a storey, the
    # front garden wall; and a full storey, the high wall round a yard. The
    # pieces follow a wall set's order: west face, north face, corner, post.
    "brick_low": dict(west1="walls_exterior_house_low_01_012", west2="walls_exterior_house_low_01_012",
                      north1="walls_exterior_house_low_01_013", north2="walls_exterior_house_low_01_013",
                      nw="walls_exterior_house_low_01_014", post="walls_exterior_house_low_01_015"),
    "brick_tall": dict(west1="walls_exterior_house_01_004", west2="walls_exterior_house_01_004",
                       north1="walls_exterior_house_01_005", north2="walls_exterior_house_01_005",
                       nw="walls_exterior_house_01_006", post="walls_exterior_house_01_007"),
    # English garden walls from the KnoxMap UK tile pack (KnoxMapUK_Tiles,
    # asset_creator/Tile_creator/src/walls.py): red brick with a brick-on-edge
    # coping, low (front, about 0.95 m) and tall (side and back, 1.8 m).
    "uk_brick_low": dict(west1="walls_garden_uk_01_0", west2="walls_garden_uk_01_1",
                         north1="walls_garden_uk_01_2", north2="walls_garden_uk_01_3",
                         nw="walls_garden_uk_01_4", post="walls_garden_uk_01_5"),
    "uk_brick_tall": dict(west1="walls_garden_uk_01_8", west2="walls_garden_uk_01_9",
                          north1="walls_garden_uk_01_10", north2="walls_garden_uk_01_11",
                          nw="walls_garden_uk_01_12", post="walls_garden_uk_01_13"),
    # Hedges, as boundary pieces rather than bushes on the vegetation map:
    # vanilla's clipped hedge (vegetation_ornamental_01), tall and low, in
    # the order north, west, corner, end.
    "hedge_tall": dict(west1="vegetation_ornamental_01_001", west2="vegetation_ornamental_01_001",
                       north1="vegetation_ornamental_01_000", north2="vegetation_ornamental_01_000",
                       nw="vegetation_ornamental_01_002", post="vegetation_ornamental_01_003"),
    "hedge_low": dict(west1="vegetation_ornamental_01_005", west2="vegetation_ornamental_01_005",
                      north1="vegetation_ornamental_01_004", north2="vegetation_ornamental_01_004",
                      nw="vegetation_ornamental_01_006", post="vegetation_ornamental_01_007"),
}
# A brick wall this tall (metres, its height= tag) or more is the high one.
BRICK_TALL_M = 1.6

# Which of a pair is the taller, and the height in metres at which a fence
# becomes it. OSM fence heights are mostly 0.9 to 2.4; the game's short
# fences are waist high and its tall ones are over your head.
TALLER = {
    "short_wooden": ("tall_wooden", 1.4),
    "short_chainlink": ("tall_chainlink", 1.7),
    "black_metal": ("tall_metal", 1.7),
    "short_picket": ("white_picket", 1.0),
}
SHORTER = {tall: short for short, (tall, _m) in TALLER.items()}


def height_m(tags: dict) -> float | None:
    """A fence's height in metres, from however the mapper wrote it.

    "2", "2 m", "2.5m" and "6'" all occur; anything else is no answer rather
    than a guess.
    """
    raw = str(tags.get("height") or tags.get("est_height") or "").strip().lower()
    if not raw:
        return None
    if "'" in raw or "ft" in raw:
        feet = raw.replace("ft", "").split("'")[0].strip()
        try:
            return float(feet) * 0.3048
        except ValueError:
            return None
    for suffix in (" m", "m", " metres", " meters"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)].strip()
            break
    try:
        metres = float(raw.replace(",", "."))
    except ValueError:
        return None
    return metres if 0.2 <= metres <= 12.0 else None


def by_height(style: str, metres: float | None) -> str:
    """The same fence at the height the mapper gave it."""
    if metres is None:
        return style
    short = SHORTER.get(style, style)
    taller, at = TALLER.get(short, (None, 0.0))
    if taller is None:
        return style
    return taller if metres >= at else short


# Gates, for the fences drawn round back yards: (west-edge tile, north-edge tile).
GATES = {
    "short_wooden": ("fixtures_doors_fences_01_004", "fixtures_doors_fences_01_005"),
    "tall_wooden": ("fixtures_doors_fences_01_012", "fixtures_doors_fences_01_013"),
    "white_picket": ("fixtures_doors_fences_01_008", "fixtures_doors_fences_01_009"),
    "short_chainlink": ("fixtures_doors_fences_01_016", "fixtures_doors_fences_01_017"),
    "black_metal": ("fixtures_doors_fences_01_000", "fixtures_doors_fences_01_001"),
}
# The UK boundaries' gates: vanilla has none for brick or hedges - an iron
# gate in a low front wall, a wooden one in a hedge or a tall wall.
UK_GATES = {"uk_brick_low": "black_metal", "hedge_low": "short_wooden",
            "uk_brick_tall": "tall_wooden", "hedge_tall": "tall_wooden"}
# On a KnoxMap UK map, the way through a boundary OSM maps as a point
# (barrier=...), in a hedge or rural fence (UK_WAY_FOR): ours (asset_creator
# streetscape.py, countryside_uk_01), each (west-edge tile, north-edge tile).
# One with no boundary near (a gate across an unfenced track) stands nowhere:
# alone in a field it is a gate to nothing. A field gate
# opens (vanilla's fence gate's door, four sprites as its) and is climbed; a
# stile is climbed; a kissing gate is walked through. A cattle grid, a lift
# gate or an entrance stays a gap.
UK_WAYS = {"gate": ("countryside_uk_01_0", "countryside_uk_01_1"),
           "swing_gate": ("countryside_uk_01_0", "countryside_uk_01_1"),
           "metal_gate": ("countryside_uk_01_4", "countryside_uk_01_5"),
           "stile": ("countryside_uk_01_8", "countryside_uk_01_9"),
           "kissing_gate": ("countryside_uk_01_10", "countryside_uk_01_11")}
METAL_SHARE = 3          # one field gate in this many galvanised (by its place), the rest timber
# What goes through a boundary by its style (user, 2026-10-14: a gate must
# suit what it stands in): a hedge or a rural timber fence takes ours, by the
# OSM kind ("field"); a garden or town boundary its own kind of gate (a GATES
# style), whatever OSM calls it - an iron gate in a low brick wall, a timber
# door in a tall wall or fence; a stile only where ours goes. A style with no
# gate here keeps its line whole: no gap, no misfit gate.
UK_WAY_FOR = {"hedge_low": "field", "hedge_tall": "field", "short_wooden": "field",
              "uk_brick_low": "black_metal", "uk_brick_tall": "tall_wooden", "tall_wooden": "tall_wooden",
              "white_picket": "white_picket", "short_chainlink": "short_chainlink", "black_metal": "black_metal"}


def _tile(n) -> str:
    """A Fences.txt style's piece number, or a whole tile name as it is."""
    return n if isinstance(n, str) else f"fencing_01_{n:03d}"


# Land uses that are fenced off in real life, and their smallest size in
# tiles - a hundred-tile "base" is a war memorial, not a barracks.
FENCED_SITES = {"military": 400}


def _fenced_sites(areas) -> list:
    """The land-use polygons that get a fence of their own."""
    out = []
    for shape, props in getattr(areas, "_items", []) or ():
        least = FENCED_SITES.get((props.get("category") or ""))
        if least is None or shape.is_empty or shape.area < least:
            continue
        if getattr(shape, "exterior", None) is not None:
            out.append(shape)
    return out


def _mapped_lines(runs: list[list]):
    """The fences somebody put there, as one geometry to ask about."""
    if not runs:
        return None
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    kept = [LineString(pts) for pts in runs if len(pts) >= 2]
    return unary_union(kept) if kept else None


def _crosses(pts: list, drawn) -> bool:
    """Whether this run of fence meets one that was mapped or drawn."""
    if drawn is None or len(pts) < 2:
        return False
    from shapely.geometry import LineString

    try:
        return LineString(pts).intersects(drawn)
    except Exception:          # noqa: BLE001 - a line that is not a line
        return False


def style_for(tags: dict, area: str | None) -> str:
    """Which fence a line gets: from its own tags first, its surroundings next.

    Where the mapper gave a height, that decides between the short and the
    tall version of the fence they asked for - the game has both of most of
    them, and two-metre metal bars used to come out as the waist-high railing.
    """
    if tags.get("knox_style") in STYLES:
        # Chosen already, by whoever drew the line (a regional data step).
        return tags["knox_style"]
    barrier = tags.get("barrier")
    material = (tags.get("material") or tags.get("wall") or "").lower()
    if barrier in {"wall", "retaining_wall", "city_wall"} and material in {"brick", "masonry", "stone", "flint"}:
        try:
            tall = float(str(tags.get("height") or 0).split()[0]) >= BRICK_TALL_M
        except ValueError:
            tall = False
        # The brick pieces are house walls - a storey high, or a third of one
        # - not garden walls; hedges stand in until there are proper ones.
        return "hedge_tall" if tall else "hedge_low"
    if barrier in {"wall", "retaining_wall", "city_wall"}:
        return "tall_concrete"
    metres = height_m(tags)
    kind = (tags.get("fence_type") or tags.get("material") or "").lower()
    if kind in {"chain_link", "chain", "mesh"}:
        if metres is not None:
            return by_height("short_chainlink", metres)
        return "tall_chainlink" if area in {"industrial", "military"} else "short_chainlink"
    if kind in {"wood", "wooden", "split_rail", "board", "panel"}:
        return by_height("tall_wooden", metres)
    if kind in {"picket", "pales"}:
        return by_height("white_picket", metres)
    if kind in {"metal", "railing", "metal_bars", "bars", "iron", "steel"}:
        return by_height("black_metal", metres)
    if kind in {"barbed_wire", "electric", "wire"}:
        return "cattle"
    if kind in {"concrete", "brick", "stone", "masonry"}:
        return "tall_concrete"
    return by_height({
        "industrial": "tall_chainlink", "military": "tall_chainlink",
        "schoolyard": "short_chainlink", "sports": "short_chainlink",
        "residential": "tall_wooden", "cemetery": "black_metal",
        "worship_grounds": "black_metal", "hospital_grounds": "black_metal",
    }.get(area or "", "short_wooden"), metres)


# Boundaries along angled roads (KnoxMap UK, user 2026-10-12): the game has
# only west and north edges, so a garden front along a 45-degree or 2:1 street
# is a staircase. Where a run of edges steps regularly (DIAG_MIN_MOVES or
# more), the staircase stays as invisible pieces with the style's collision,
# and pictures of the boundary are laid along the straight line through its
# down the middle of the staircase (user, 2026-10-12: through its corners
# the picture stood up to 0.7 of a square off the collision; down the middle
# at most a third), from the UK tile pack's sheet (asset_creator/
# Tile_creator/src/diag_walls.py): per style 24 slots, 0-13 a picture per way
# such a line crosses a square (DIAG_CUTS), 16-18 invisible west, north, both. Only on maps with a regional.json (our
# UK maps, which ship the pack).
# KnoxMap UK's own walls -> vanilla's, for a map with its UK assets off (assets_uk.py)
VANILLA_FOR = {"uk_brick_low": "brick_low", "uk_brick_tall": "brick_tall"}

DIAG_SHEET = "boundaries_diag_uk_01"
DIAG_STYLES = ["uk_brick_low", "uk_brick_tall", "hedge_low", "hedge_tall",
               "short_wooden", "tall_wooden", "black_metal"]
DIAG_CUTS = [((0, 1), (1, 0)), ((0, 0), (1, 1)),
             ((0, 1), (1, .5)), ((0, .5), (1, 0)), ((0, 0), (1, .5)), ((0, .5), (1, 1)),
             ((0, 1), (.5, 0)), ((.5, 1), (1, 0)), ((0, 0), (.5, 1)), ((.5, 0), (1, 1)),
             ((.5, 0), (0, .5)), ((1, .5), (.5, 1)), ((.5, 0), (1, .5)), ((0, .5), (.5, 1))]
DIAG_SLOTS, DIAG_INVISIBLE = 24, 16       # slots a style; its invisible W, N, NW from here
DIAG_MIN_MOVES = 4
DIAG_GAP = 3                               # steps of a path or gate a front is joined across


def _diag_tile(style: str, slot: int) -> str:
    return f"{DIAG_SHEET}_{DIAG_STYLES.index(style) * DIAG_SLOTS + slot}"


def _cut_slot(p0, p1) -> int | None:
    """Which picture a line from p0 to p1 (square-local) is, either way round."""
    a = (round(p0[0] * 2) / 2, round(p0[1] * 2) / 2)
    b = (round(p1[0] * 2) / 2, round(p1[1] * 2) / 2)
    for i, (c, d) in enumerate(DIAG_CUTS):
        if (a, b) in ((c, d), (d, c)):
            return i
    return None


def _staircase_runs(moves: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """(start, end) move ranges stepping as a 45-degree (A B A B) or 2:1
    (A A B A A B) staircase, the longest from each start, at least DIAG_MIN_MOVES."""
    runs, i = [], 0
    while i < len(moves):
        best = i
        for period in (2, 3):
            pat = moves[i:i + period]
            if len(pat) < period or len({abs(m[0]) for m in pat}) < 2:
                continue                                   # needs both axes
            if period == 3 and len(set(pat)) != 2:
                continue
            j = i
            while j < len(moves) and moves[j] == pat[(j - i) % period]:
                j += 1
            # Whole repeats only: a run ending part-way through its pattern
            # gave a line off the 45-degree and 2:1 angles (4 across to 10
            # down), drawn askew to the road (user, 2026-10-12).
            j = i + (j - i) // period * period
            if j - i >= DIAG_MIN_MOVES and j > best:
                best = j
        if best > i:
            runs.append((i, best))
            i = best
        else:
            i += 1
    return runs


def diagonal_lines(edges: dict, styles=None) -> list:
    """The straight lines angled boundaries are drawn along: for each regular
    staircase of edges of one style (A B A B or A A B, DIAG_MIN_MOVES moves or
    more), (style, a, b, edge keys, gap step midpoints, wall step midpoints) -
    a and b the ends of the line down the staircase's middle, in tile
    coordinates."""
    styles = DIAG_STYLES if styles is None else styles
    seg = {}                                           # (style, vertex) -> [(other vertex, (square, side))]
    for (x, y), sides in edges.items():
        for side, style in sides.items():
            if style not in styles:
                continue
            a, b = ((x, y), (x, y + 1)) if side == "W" else ((x, y), (x + 1, y))
            seg.setdefault((style, a), []).append((b, ((x, y), side)))
            seg.setdefault((style, b), []).append((a, ((x, y), side)))
    # Pieces of one front broken by garden paths and gates (a gap of up to
    # DIAG_GAP steps) are bridged by stand-in edges, so the front is found as
    # one staircase (user, 2026-10-12: two- and three-step pieces between
    # paths were left zig-zagging); the stand-ins are no wall - the gaps stay.
    ends, piece_of = {}, {}
    for key in seg:                                    # each vertex's piece (its connected run of edges)
        if key in piece_of:
            continue
        stack, piece_of[key] = [key], key
        while stack:
            st, v = stack.pop()
            for w, _e in seg[(st, v)]:
                if (st, w) not in piece_of:
                    piece_of[(st, w)] = key
                    stack.append((st, w))
    for (style, v), links in seg.items():
        if len(links) == 1:
            ends.setdefault(style, []).append(v)
    for style, vs in ends.items():
        taken = set()
        for p in vs:
            if p in taken:
                continue
            cands = [q for q in vs if q != p and q not in taken and 1 <= abs(q[0] - p[0]) + abs(q[1] - p[1]) <= DIAG_GAP
                     and piece_of[(style, q)] != piece_of[(style, p)]
                     and q[0] != p[0] and q[1] != p[1]]           # a gap in a straight run is no staircase
            if not cands:
                continue
            q = min(cands, key=lambda q: abs(q[0] - p[0]) + abs(q[1] - p[1]))
            o = seg[(style, p)][0][0]
            came_x = o[1] == p[1]                       # p's own last step was along x
            sx = 1 if q[0] > p[0] else -1
            sy = 1 if q[1] > p[1] else -1
            path, v = [p], p
            order = ("y", "x") if came_x else ("x", "y")
            while v != q:                               # alternate axes, starting on the other one
                for ax in order + order:
                    if ax == "x" and v[0] != q[0]:
                        v = (v[0] + sx, v[1])
                        break
                    if ax == "y" and v[1] != q[1]:
                        v = (v[0], v[1] + sy)
                        break
                order = order[::-1]
                path.append(v)
            if any((style, w) in seg for w in path[1:-1]):
                continue                                # through another boundary
            for u, w in zip(path, path[1:]):
                sq, side = ((min(u[0], w[0]), u[1]), "N") if u[1] == w[1] else ((u[0], min(u[1], w[1])), "W")
                key = (sq, side, "gap")
                seg.setdefault((style, u), []).append((w, key))
                seg.setdefault((style, w), []).append((u, key))
            taken.update((p, q))
    lines, used = [], set()
    # Chains are walked from their ends (a vertex without exactly two links);
    # then any ring left (a plot fenced all round in one style: every vertex
    # has two) from any of its vertices.
    starts = [(k, links) for k, links in seg.items() if len(links) != 2] +         [(k, links) for k, links in seg.items() if len(links) == 2]
    for (style, start), links in starts:
        if (len(links) == 2 and any(e in used for _o, e in links)) or all(e in used for _o, e in links):
            continue
        for first in links:
            if first[1] in used:
                continue
            chain, keys, v, nxt = [start], [], start, first
            while nxt and nxt[1] not in used:
                used.add(nxt[1])
                keys.append(nxt[1])
                v = nxt[0]
                chain.append(v)
                ahead = [l for l in seg.get((style, v), []) if l[1] not in used]
                nxt = ahead[0] if len(seg.get((style, v), [])) == 2 and ahead else None
            moves = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(chain, chain[1:])]
            for i, j in _staircase_runs(moves):
                pts = chain[i:j + 1]
                dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
                level = [dx * p[1] - dy * p[0] for p in pts]          # which parallel line each corner is on
                size = math.hypot(dx, dy)
                n = (dy / size, -dx / size)                         # level falls by `size` a unit along n
                mid = (min(level) + max(level)) / 2                 # the line down the staircase's middle
                a, b = [(p[0] + n[0] * (lv - mid) / size, p[1] + n[1] * (lv - mid) / size)
                        for p, lv in ((pts[0], level[0]), (pts[-1], level[-1]))]
                run = keys[i:j]
                gaps = [((chain[i + k][0] + chain[i + k + 1][0]) / 2, (chain[i + k][1] + chain[i + k + 1][1]) / 2)
                        for k, key in enumerate(run) if len(key) == 3]
                real = [((chain[i + k][0] + chain[i + k + 1][0]) / 2, (chain[i + k][1] + chain[i + k + 1][1]) / 2)
                        for k, key in enumerate(run) if len(key) == 2]
                if real:
                    lines.append((style, a, b, [k for k in run if len(k) == 2], gaps, real))
    return lines


def line_squares(a, b):
    """Each square the line a-b crosses and the part of the line in it,
    square-local: [((x, y), p0, p1)]."""
    lo_x, hi_x = math.floor(min(a[0], b[0])), math.ceil(max(a[0], b[0]))
    lo_y, hi_y = math.floor(min(a[1], b[1])), math.ceil(max(a[1], b[1]))
    cuts = sorted({0.0, 1.0}
                  | {(k - a[0]) / (b[0] - a[0]) for k in range(lo_x, hi_x + 1)
                     if b[0] != a[0] and 0 < (k - a[0]) / (b[0] - a[0]) < 1}
                  | {(k - a[1]) / (b[1] - a[1]) for k in range(lo_y, hi_y + 1)
                     if b[1] != a[1] and 0 < (k - a[1]) / (b[1] - a[1]) < 1})
    out = []
    for t0, t1 in zip(cuts, cuts[1:]):
        if t1 - t0 < 1e-6:
            continue
        p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
        p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
        sx, sy = math.floor((p0[0] + p1[0]) / 2), math.floor((p0[1] + p1[1]) / 2)
        out.append(((sx, sy), (p0[0] - sx, p0[1] - sy), (p1[0] - sx, p1[1] - sy)))
    return out


def _diagonals(edges: dict, map_w: int, map_h: int, given: list | None = None) -> tuple[dict, set]:
    """Pictures along the regular staircases of edges in DIAG_STYLES:
    ({(x, y): tile} for the second layer, {((x, y), side)} edges they stand
    for), each on its own line - a plot line is on the street's edge, and the
    wall stands there (user, 2026-10-14). (Until then a wall with the street
    on the viewer's side was moved a square towards it, picture and collision
    together, as asked on 2026-10-12 when fronts were found staircases off
    the street's line; on plots2's exact lines that put 28-38% of Alton's
    angled fronts 0.7-0.9 of a square onto the pavement.)"""
    pictures, covered = {}, set()
    # Lines given exactly (plot lines), then the staircases found among the
    # rest of the edges.
    taken = {k for line in (given or []) for k in line[3]}
    rest: dict = {}
    for sq, sides in edges.items():
        for side, style in sides.items():
            if (sq, side) not in taken:
                rest.setdefault(sq, {})[side] = style
    for style, a, b, keys, gaps, real in list(given or []) + diagonal_lines(rest):
        # Onto the lattice the pictures are drawn for (lines through tile
        # corners and edge middles): the staircase's middle can sit a quarter
        # square off it, and then every other square's cut has no picture
        # (seen as a dashed wall, 2026-10-12). Both ends moved alike.
        off = (round(a[0] * 2) / 2 - a[0], round(a[1] * 2) / 2 - a[1])
        a2 = (a[0] + off[0], a[1] + off[1])
        b2 = (b[0] + off[0], b[1] + off[1])
        placed = 0
        for (sx, sy), p0, p1 in line_squares(a2, b2):
            slot = _cut_slot(p0, p1)
            if slot is None or not (0 <= sx < map_w and 0 <= sy < map_h):
                continue
            if gaps:                                      # nearest step a gap: no picture (the path's way through)
                mx, my = sx + (p0[0] + p1[0]) / 2, sy + (p0[1] + p1[1]) / 2
                near = lambda pts: min((px - mx) ** 2 + (py - my) ** 2 for px, py in pts)  # noqa: E731
                if near(gaps) < near(real):
                    continue
            pictures[(sx, sy)] = _diag_tile(style, slot)
            placed += 1
        if placed:
            covered.update(keys)
    return pictures, covered


def _grid_path(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """A polyline as a walk along tile corners, one grid step at a time.

    Diagonal runs become staircases along tile edges - a fence can only stand
    on an edge - choosing at each corner the step that keeps closer to the
    real line.
    """
    path: list[tuple[int, int]] = []
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        steps = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) * 4)))
        for i in range(steps + 1):
            t = i / steps
            v = (int(round(ax + (bx - ax) * t)), int(round(ay + (by - ay) * t)))
            if not path:
                path.append(v)
                continue
            px, py = path[-1]
            if v == (px, py):
                continue
            if v[0] != px and v[1] != py:
                # Two corners apart diagonally: go through whichever of the two
                # in-between corners sits nearer the true line.
                opt1, opt2 = (v[0], py), (px, v[1])
                mx, my = ax + (bx - ax) * t, ay + (by - ay) * t
                mid = min((opt1, opt2), key=lambda o: math.hypot(o[0] - mx, o[1] - my))
                path.append(mid)
            path.append(v)
    return path


def _tarmac_mask(bmp_path: str, width: int, height: int):
    """True where the ground is a road, service lane or car park."""
    import numpy as np
    from generator import pz_colors as C

    from .bitmaps import read_rgb

    if not os.path.exists(bmp_path):
        return None
    ground = read_rgb(bmp_path)
    if ground.shape[:2] != (height, width):
        return None
    mask = np.zeros((height, width), dtype=bool)
    for colour in (C.LIGHT_ASPHALT, C.MEDIUM_ASPHALT, C.DARK_ASPHALT,
                   C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE):
        same = ground[:, :, 0] == colour[0]
        same &= ground[:, :, 1] == colour[1]
        same &= ground[:, :, 2] == colour[2]
        mask |= same
    return mask


PAVEMENT_REACH = 4      # tiles from a road's tarmac that pale paving counts as its pavement


def _street_mask(bmp_path: str, width: int, height: int, tarmac):
    """True on road and pavement: the tarmac, and pale concrete within
    PAVEMENT_REACH of a road's (not a car park's) tarmac. Paving further off
    is a yard or a forecourt, where a fence may stand."""
    import numpy as np
    from scipy import ndimage
    from generator import pz_colors as C

    from .bitmaps import read_rgb

    if tarmac is None or not os.path.exists(bmp_path):
        return None
    ground = read_rgb(bmp_path)
    if ground.shape[:2] != (height, width):
        return None

    def colour(c):
        return (ground[:, :, 0] == c[0]) & (ground[:, :, 1] == c[1]) & (ground[:, :, 2] == c[2])

    road = np.zeros((height, width), dtype=bool)
    for c in (C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE):
        road |= colour(c)
    near = ndimage.binary_dilation(road, iterations=PAVEMENT_REACH) if road.any() else road
    return tarmac | (colour(C.PALE_CONCRETE) & near)


def _path_masks(bmp_path: str, width: int, height: int):
    """(footpath, anything a footpath carries on into, drive) masks of the ground."""
    import numpy as np
    from PIL import Image

    from generator import pz_colors as C

    if not os.path.exists(bmp_path):
        return None, None, None
    ground = np.asarray(Image.open(bmp_path).convert("RGB"))
    if ground.shape[:2] != (height, width):
        return None, None, None

    def match(colours):
        m = np.zeros((height, width), dtype=bool)
        for c in colours:
            m |= np.all(ground == c, axis=2)
        return m
    path = match((C.PAVING_STONE,))
    walk = path | match((C.PALE_CONCRETE, C.PAVING, C.LIGHT_ASPHALT, C.MEDIUM_ASPHALT,
                         C.DARK_ASPHALT, C.DARKEST_ASPHALT))
    return path, walk, match((C.DARK_ASPHALT,))


# How far either side of a gate or a crossing path the fence is looked at, in
# tiles: the fence line and the node or path it meets rarely land on the same
# tile once both are on the grid.
OPENING_REACH = 1


def _openings(out_dir: str, map_name: str, proj):
    """Where people get through a fence: gate nodes, and the paths that cross
    one. Points in tiles, with the paths as lines to be crossed with each
    fence where it is built, and each point's OSM barrier (gate, stile...)."""
    path = os.path.join(out_dir, f"{map_name}_fence_openings.geojson")
    feats = (knoxfiles.read(path, "the gates and paths through the fences",
                            default={}) or {}).get("features", [])
    if not feats:
        return [], [], []
    gates, paths, kinds = [], [], []
    for feat in feats:
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") == "Point" and len(coords) == 2:
            gates.append(proj.to_px(coords[1], coords[0]))
            kinds.append((feat.get("properties") or {}).get("barrier") or "gate")
        elif geom.get("type") == "LineString" and len(coords) >= 2:
            paths.append([proj.to_px(lat, lon) for lon, lat in coords])
    return gates, paths, kinds


def _path_crossings(fence: list, paths) -> list:
    """Where a footpath or a track crosses this fence, in tiles."""
    if paths is None or len(fence) < 2:
        return []
    from shapely.geometry import LineString

    try:
        line = LineString(fence)
        hit = line.intersection(paths)
    except Exception:          # a fence that is not a line: nothing crosses it
        return []
    if hit.is_empty:
        return []
    out = []
    for part in getattr(hit, "geoms", [hit]):
        if part.geom_type == "Point":
            out.append((part.x, part.y))
        elif not part.is_empty:
            middle = part.centroid
            out.append((middle.x, middle.y))
    return out


WAYS_LAID: collections.Counter = collections.Counter()   # what each UK way through came to (Build reports it)


def _uk_way(kind: str, style: str, x: int, y: int, side: str, squares: dict) -> str | None:
    """The piece for OSM's way through (`kind`) in a boundary of `style` on a
    square's W or N edge (UK_WAY_FOR), or None to keep the boundary whole.
    Ours (a field gate - a third galvanised, by the square - stile or
    kissing gate) only with the same boundary running on beside it along
    the line on both sides, not at a corner or a loose end; a garden or town
    gate with its wall or fence on one side at least (by the house, the
    other side is often the house's wall), never standing alone (review,
    2026-10-14: a gate on a one-square stub of wall)."""
    use = UK_WAY_FOR.get(style)
    if use is None:
        return None
    along = ((x, y - 1), (x, y + 1)) if side == "W" else ((x - 1, y), (x + 1, y))
    runs = [side in squares.get(sq, {}) and squares[sq][side][0] == style for sq in along]
    if use != "field":
        return None if kind == "stile" or not any(runs) else GATES[use][0 if side == "W" else 1]
    if not all(runs):
        return None
    if kind in ("gate", "swing_gate") and (x * 7 + y * 13) % METAL_SHARE == 0:
        kind = "metal_gate"
    return UK_WAYS[kind][0 if side == "W" else 1]


def _way_through(openings: dict, gated: set, gates: dict,
                 at: tuple[int, int], style: str, uk_kinds: dict | None = None) -> bool:
    """Whether the fence is built on this tile at all.

    True builds it as usual, False leaves a hole. A gate node hangs one gate
    on the first tile of its opening the fence actually reaches and the fence
    carries on either side of it, so a gate is a gate and not a gateway; a
    fence the game has no gate for gets the hole instead.
    """
    mark = openings.get(at)
    if mark is None:
        return True
    if mark and uk_kinds is not None:         # a UK map: our way through, by its kind
        kind = uk_kinds.get(mark)
        if kind not in UK_WAYS:
            return False                      # a cattle grid, a lift gate: a gap
        if mark not in gated:
            gated.add(mark)
            gates[at] = ("uk", kind, mark)    # swapped for ours below
        return True
    if mark:                                  # a gate the mapper put there
        if style not in GATES:
            return False                      # no gate for this fence: a gap
        if mark not in gated:
            gated.add(mark)
            gates[at] = style                 # swapped for the gate below
        return True
    return False                              # a path through: a gap


def build_fences(out_dir: str, map_name: str, proj, occupied, areas,
                 bdir: str, extra: list | None = None,
                 keep_clear: set | None = None, diagonals: bool | None = None) -> tuple[list, int]:
    """Write one fence .tbx per map cell that has fences. Returns placements.

    `keep_clear` is a set of (x, y) tiles - the ground in front of each door - that no fence
    edge may touch: a fence across a doorway is a gap in the fence instead."""
    clear = keep_clear or ()
    from .world import Placement

    path = os.path.join(out_dir, f"{map_name}_fences.geojson")
    lines = (knoxfiles.read(path, "the fences", default={}) or {}).get("features", [])
    sites = _fenced_sites(areas)
    # Property boundaries drawn by a regional step (marked source=inspire or
    # source=plots) already fence every garden: the back-yard fences
    # knoxbuild/yards.py makes up would be a second set inside them.
    if any((ft.get("properties") or {}).get("source") in ("inspire", "plots") for ft in lines):
        extra = []
    if not lines and not extra and not sites:
        return [], 0

    map_h, map_w = occupied.shape

    def building_at(x: int, y: int) -> bool:
        return 0 <= x < map_w and 0 <= y < map_h and bool(occupied[y, x])

    # Mappers draw a fence as one line and put the gate in as a separate point,
    # so a schoolyard fence crosses its own driveway and a factory fence the
    # road into the works. Built as drawn, the fence walls the road off. An
    # edge with tarmac on both sides is a road crossing, and is left open; a
    # fence along the side of a car park has tarmac on one side only and stays.
    tarmac = _tarmac_mask(os.path.join(out_dir, f"{map_name}.bmp"), map_w, map_h)
    # And the footpaths knoxbuild/yards.py lays from front doors to the
    # pavement: a garden wall across one is a gate's worth left open, or the
    # front door is walled in.
    path_mask, walkway, drive = _path_masks(os.path.join(out_dir, f"{map_name}.bmp"), map_w, map_h)
    # Drives as plots2.py found and made them (a feature of their own, not a
    # colour): their mouths stay open.
    plots_npz = os.path.join(out_dir, f"{map_name}_plots.npz")
    if drive is not None and os.path.exists(plots_npz):
        import numpy as np
        with np.load(plots_npz) as z:
            if "drive" in z:
                d_ = z["drive"]
                drive[:min(map_h, d_.shape[0]), :min(map_w, d_.shape[1])] |= d_[:map_h, :map_w]

    # Nothing stands on the street (KnoxMap UK, user rule 2026-10-09): an
    # edge with road or pavement on both sides gets no fence, wall or hedge,
    # whichever step drew the line. Along the back of the pavement (street
    # one side only) a boundary stays.
    street = _street_mask(os.path.join(out_dir, f"{map_name}.bmp"), map_w, map_h, tarmac)

    def crossing(ax: int, ay: int, bx: int, by: int, ours: bool = False) -> bool:
        inside = (0 <= ax < map_w and 0 <= ay < map_h and
                  0 <= bx < map_w and 0 <= by < map_h)
        if not inside:
            return False
        # A plot's own boundary (plots2.py) is drawn on the street's exact
        # edge: an angled front's staircase runs through road and pavement
        # squares. Its edges with tarmac or street on both sides are only
        # held back (on_street), and dropped later unless they are part of an
        # angled front (2026-10-13 review: exempting plot lines from every
        # rule walled up 780 path and drive mouths).
        both_tarmac = tarmac is not None and bool(tarmac[ay, ax]) and bool(tarmac[by, bx])
        both_street = street is not None and bool(street[ay, ax]) and bool(street[by, bx])
        if (both_tarmac or both_street) and not ours:
            return True
        # Road tarmac both sides (neither a drive): a plot line along an
        # angled lane's edge squares, not a mouth. Taken for one by the rule
        # below, every other step of the staircase went and the front was
        # laid as separate panels (PLAN K "dashed angled fronts", 2026-10-13).
        if both_tarmac and not (drive is not None and (drive[ay, ax] or drive[by, bx])):
            return "held"
        # A path or a drive meeting the pavement or the road: the way in. A
        # garden path's (no tarmac either side) may get a gate (below).
        way_a = (path_mask is not None and path_mask[ay, ax]) or (tarmac is not None and tarmac[ay, ax])
        way_b = (path_mask is not None and path_mask[by, bx]) or (tarmac is not None and tarmac[by, bx])
        if walkway is not None and ((way_a and walkway[by, bx]) or (way_b and walkway[ay, ax])):
            on_foot = not (tarmac is not None and (tarmac[ay, ax] or tarmac[by, bx]))
            return "mouth" if on_foot else True
        return "held" if (both_tarmac or both_street) else False

    # (x, y) -> {"W": style, "N": style}
    edges: dict[tuple[int, int], dict[str, str]] = {}
    ends: dict[tuple[int, int], str] = {}
    held: set = set()                    # plot edges kept only if part of an angled front
    # Mapped fences, then the ones drawn round back yards (knoxbuild/yards.py),
    # which come as tile points with their style. Each carries whether it is a
    # fence somebody put there - mapped in OpenStreetMap or drawn in the
    # editor - or one made up here, because where the two meet the real one
    # wins.
    # KnoxMap UK: each also carries whether it is one of our plot lines (plots2.py,
    # source=plots: exact, laid as given) - a real one too, from the data, not a guess.
    todo: list[tuple[list, str, bool, bool]] = []
    for feat in lines:
        coords = feat.get("geometry", {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        props_ = feat.get("properties") or {}
        # A plot line carries its exact line in tiles (plots2.py): lon/lat back
        # came 0.01 tile off the lattice.
        pts = [tuple(p_) for p_ in props_["tiles"]] if props_.get("tiles") else \
            [proj.to_px(lat, lon) for lon, lat in coords]
        mid = pts[len(pts) // 2]
        todo.append((pts, style_for(props_, areas.category_at(*mid) if areas else None),
                     True, props_.get("source") == "plots"))
    # Sites that are fenced in the world whether or not anyone drew the fence.
    # OpenStreetMap maps an army base as landuse=military far more often than
    # it maps the wire round it, so a base was open ground you could walk
    # across; the outline of the site is the fence. Road crossings are left
    # open by the rule above, which is what makes the gates.
    for shape in sites:
        ring = list(shape.exterior.coords)
        if len(ring) >= 4:
            todo.append(([(float(x), float(y)) for x, y in ring],
                         "tall_chainlink", True, False))
    gates: dict[tuple[int, int], str] = {}
    # A yard fence is this program's guess at what is round a house. Where
    # somebody has mapped or drawn a fence or a hedge that crosses it, theirs
    # is what is there, and the guess is dropped rather than laid over it.
    drawn = _mapped_lines([pts for pts, _style, real, _ours in todo if real])
    for item in extra or ():
        pts, style = item[0], item[1]
        if _crosses(pts, drawn):
            continue
        todo.append((pts, style, False, False))
        for gx, gy in (item[2] if len(item) > 2 else ()):
            gates[(gx, gy)] = style
    # KnoxMap UK, its UK assets off (assets_uk.py): the garden walls vanilla's brick walls,
    # and no angled pieces (their sheet is ours): the fences step as vanilla's do
    from .assets_uk import off as _uk_off
    if _uk_off(out_dir):
        todo = [(pts, VANILLA_FOR.get(st, st), real, plot) for pts, st, real, plot in todo]
        gates = {k: VANILLA_FOR.get(v, v) for k, v in gates.items()}
        diagonals = False

    # A gate on the fence, or a footpath through it: the mapper drew the way
    # in, and a fence built from the line alone stood across it. Each opening
    # takes its tiles out of the fence; where the game has a gate for that
    # fence, one of those tiles becomes the gate rather than a hole.
    gate_points, path_lines, gate_kinds = _openings(out_dir, map_name, proj)
    from . import regional_uk
    uk_kinds = {} if regional_uk.load(out_dir) is not None else None   # mark -> OSM barrier, on a UK map
    paths = None
    if path_lines:
        from shapely.geometry import LineString
        from shapely.ops import unary_union
        paths = unary_union([LineString(line) for line in path_lines
                             if len(line) >= 2])
    openings: dict[tuple[int, int], int] = {}

    def open_up(points, kind_gate: bool, kinds=()) -> None:
        for n, (px, py) in enumerate(points):
            mark = len(openings) + 1 if kind_gate else 0
            if mark and uk_kinds is not None:
                uk_kinds[mark] = kinds[n] if n < len(kinds) else "gate"
            for dx in range(-OPENING_REACH, OPENING_REACH + 1):
                for dy in range(-OPENING_REACH, OPENING_REACH + 1):
                    key = (int(round(px)) + dx, int(round(py)) + dy)
                    openings.setdefault(key, mark)

    open_up(gate_points, True, gate_kinds)
    gated: set[int] = set()
    given: list = []                     # plot lines' exact 45-degree and 2:1 segments, as diagonal_lines gives them
    mouths: dict = {}                    # (square, side) -> style: a path through the line, left open for now

    def lay(walk, style, ours, real) -> list:
        """The edges of a walk along tile corners, by the crossing rules: their keys. A
        made-up fence (not real) takes no edge a real one already has."""
        keys = []
        for (x1, y1), (x2, y2) in zip(walk, walk[1:]):
            if y1 == y2:                      # along a north edge
                x = min(x1, x2)
                # A fence never runs through a building: that edge is a wall.
                if building_at(x, y1) or building_at(x, y1 - 1):
                    continue
                if (x, y1) in clear or (x, y1 - 1) in clear:
                    continue
                c = crossing(x, y1, x, y1 - 1, ours)
                if c is True:
                    continue
                if c == "mouth":
                    mouths[((x, y1), "N")] = style
                    continue
                if not _way_through(openings, gated, gates, (x, y1), style, uk_kinds):
                    continue
                if c == "held":
                    held.add(((x, y1), "N"))
                side = edges.setdefault((x, y1), {})
                if real or "N" not in side:
                    side["N"] = style
                    keys.append(((x, y1), "N"))
            else:                             # along a west edge
                y = min(y1, y2)
                if building_at(x1, y) or building_at(x1 - 1, y):
                    continue
                if (x1, y) in clear or (x1 - 1, y) in clear:
                    continue
                c = crossing(x1, y, x1 - 1, y, ours)
                if c is True:
                    continue
                if c == "mouth":
                    mouths[((x1, y), "W")] = style
                    continue
                if not _way_through(openings, gated, gates, (x1, y), style, uk_kinds):
                    continue
                if c == "held":
                    held.add(((x1, y), "W"))
                side = edges.setdefault((x1, y), {})
                if real or "W" not in side:
                    side["W"] = style
                    keys.append(((x1, y), "W"))
        return keys

    for pts, style, real, ours in todo:
        if paths is not None:
            open_up(_path_crossings(pts, paths), False)
        walk = _grid_path(pts)
        if ours and style in DIAG_STYLES:
            # A plot line (plots2.py) is already exact: each 45-degree or 2:1
            # segment is an angled line as it stands - its staircase the
            # collision, the picture on the segment - not found again from
            # tile edges (review 2026-10-13: staircases were reconstructed,
            # bridged and held back only to be drawn where they already were).
            for a, b in zip(pts, pts[1:]):
                dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
                seg_walk = _grid_path([a, b])
                keys = lay(seg_walk, style, ours, real)
                lo_, hi_ = sorted((dx, dy))
                corners = all(abs(v - round(v)) < 1e-6 for v in (*a, *b))
                # Given only whole: ends on tile corners (else the pictures'
                # cuts miss every other square) and every edge laid (else a
                # picture stands where a door front or a crossing took its
                # collision); otherwise found again from its edges as before.
                if corners and len(keys) == len(seg_walk) - 1 and lo_ > 1e-6 \
                        and min(abs(lo_ / hi_ - r) for r in (0.5, 1.0)) < 1e-6 and hi_ >= 2:
                    real_pts = [(sq[0] + (0.5 if sd == "N" else 0.0), sq[1] + (0.5 if sd == "W" else 0.0))
                                for sq, sd in keys]
                    given.append((style, tuple(a), tuple(b), keys, [], real_pts))
        else:
            lay(walk, style, ours, real)
        if walk:
            ends[walk[0]] = style
            ends[walk[-1]] = style

    # Along angled streets: pictures on the line, the staircase invisible
    # (found first: held plot edges stay only if part of one).
    if diagonals is None:
        diagonals = os.path.exists(os.path.join(out_dir, "regional.json"))
    overlay: dict[tuple[int, int], str] = {}
    covered: set = set()
    if diagonals:
        overlay, covered = _diagonals(edges, map_w, map_h, given)
        overlay = {k: v for k, v in overlay.items() if not building_at(*k)}
    for (sq, side) in held - covered:
        sides = edges.get(sq)
        if sides and side in sides:
            del sides[side]
            if not sides:
                del edges[sq]
    # A garden path through a front one square wide: a gate in the line (one
    # in a low brick wall is iron, UK_GATES), not a gap. Wider - a footpath,
    # paving to the door all along - stays open.
    for (sq, side), style in mouths.items():
        step = (1, 0) if side == "N" else (0, 1)
        if ((sq[0] + step[0], sq[1] + step[1]), side) in mouths or                 ((sq[0] - step[0], sq[1] - step[1]), side) in mouths:
            continue
        # After the angled fronts are found: an angled line's staircase is not
        # gated (its picture would stand across the path) - left open.
        if any((k, sd) in covered for k, sd in ((sq, "N"), (sq, "W"))):
            continue
        gate_as = UK_GATES.get(style, style)
        if gate_as in GATES:
            edges.setdefault(sq, {})[side] = style
            gates[sq] = gate_as

    # Each square's sides: (style, invisible).
    squares: dict = {}
    for (x, y), sides in edges.items():
        for side, style in sides.items():
            # Hidden only behind a picture its style has (the UK sets).
            squares.setdefault((x, y), {})[side] = (style, ((x, y), side) in covered and style in DIAG_STYLES)
    invisible = lambda style, sides: _diag_tile(style, DIAG_INVISIBLE + (2 if len(sides) == 2 else  # noqa: E731
                                                                  (0 if "W" in sides else 1)))

    # A garden path's gate goes only where the gate piece can stand (a square
    # with that one side, seen); anywhere else the mouth is left open - each
    # side of it: a path's end square meeting the pavement on two sides is a
    # mouth twice, and once the first side's gate went the second stayed as a
    # fence across the path (Alton, 2026-10-14: 24 walled mouths). A mouth's
    # side is laid only by the gate rule above.
    for (sq, side) in mouths:
        sides = squares.get(sq)
        if not sides or side not in sides:
            continue
        if sq not in gates or len(sides) != 1 or sides[side][1]:
            del sides[side]
            gates.pop(sq, None)
            if not sides:
                del squares[sq]

    # Tile pieces, by the same rules as WorldEd's fence tool.
    pieces: dict[tuple[int, int], str] = {}
    for (x, y), sides in squares.items():
        if not (0 <= x < map_w and 0 <= y < map_h):
            continue
        straight = {sd: st for sd, (st, inv) in sides.items() if not inv}
        if not straight:                                    # only an angled front's collision
            style = (sides.get("N") or sides["W"])[0]
            pieces[(x, y)] = invisible(style, sides)
            continue
        # A straight piece; where an angled front's collision shares the
        # square it takes the corner, so both sides block (one side lost was
        # a walk-through gap: 102 of them, 2026-10-13).
        if len(sides) == 2:
            st = straight.get("N") or straight["W"]
            pieces[(x, y)] = _tile(STYLES[st]["nw"])
        elif "W" in straight:
            s_ = STYLES[straight["W"]]
            pieces[(x, y)] = _tile(s_["west1"] if y % 2 == 0 else s_["west2"])
        else:
            s_ = STYLES[straight["N"]]
            pieces[(x, y)] = _tile(s_["north1"] if x % 2 == 0 else s_["north2"])
    for (x, y), style in ends.items():
        if (x, y) in pieces or not (0 <= x < map_w and 0 <= y < map_h):
            continue
        if building_at(x, y) or (x, y) in clear:
            continue
        touching = [(key, side) for key, side in (((x, y), "W"), ((x, y), "N"),
                                                  ((x, y - 1), "W"), ((x - 1, y), "N"))
                    if side in squares.get(key, {})]
        # A post only at a straight boundary's loose end (none at an angled
        # front's: its picture is the end).
        if len(touching) == 1 and not squares[touching[0][0]][touching[0][1]][1]:
            pieces[(x, y)] = _tile(STYLES[style]["post"])

    # A gate replaces the fence piece on its square, facing the same way.
    for (x, y), style in gates.items():
        # On the map, like every other piece: a back yard's gate on the map's
        # far edge went into a cell past it, and WorldEd would not open the
        # project at all.
        if not (0 <= x < map_w and 0 <= y < map_h):
            continue
        sides = squares.get((x, y), {})
        # Not on an angled line's hidden staircase: the picture is the fence there.
        if any(inv for _st, inv in sides.values()) or len(sides) != 1:
            if isinstance(style, tuple):
                WAYS_LAID["at a corner or an angled line"] += 1
            continue
        if isinstance(style, tuple):              # a UK map: by what it stands in
            side = next(iter(sides))
            piece = _uk_way(style[1], sides[side][0], x, y, side, squares)
            WAYS_LAID[f"{style[1]} in {sides[side][0]}: " + ("none (the line kept whole)" if piece is None else
                                                              "ours" if piece.startswith("countryside") else "its gate")] += 1
            if piece is not None:
                pieces[(x, y)] = piece
            continue
        if style in GATES:
            pieces[(x, y)] = GATES[style][0 if "W" in sides else 1]

    by_cell: dict[tuple[int, int], dict[tuple[int, int], str]] = {}
    over_cell: dict[tuple[int, int], dict[tuple[int, int], str]] = {}
    for (x, y), tile in pieces.items():
        by_cell.setdefault((x // CELL, y // CELL), {})[(x, y)] = tile
    for (x, y), tile in overlay.items():
        by_cell.setdefault((x // CELL, y // CELL), {})
        over_cell.setdefault((x // CELL, y // CELL), {})[(x, y)] = tile

    placements = []
    for (cx, cy), cell_pieces in sorted(by_cell.items()):
        cell_over = over_cell.get((cx, cy), {})
        xs = [x for x, _ in list(cell_pieces) + list(cell_over)]
        ys = [y for _, y in list(cell_pieces) + list(cell_over)]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
        fname = f"{map_name}_fences_{cx}_{cy}.tbx"
        with open(os.path.join(bdir, fname), "w", encoding="utf-8") as f:
            f.write(render_fence_tbx(w, h, {(x - x0, y - y0): t for (x, y), t in cell_pieces.items()},
                                     {(x - x0, y - y0): t for (x, y), t in cell_over.items()}))
        placements.append(Placement(f"buildings/{fname}", x0, y0, w, h))
    print(f"  garden path mouths: {len(mouths)} edges, gated: {sum(1 for sq, _ in mouths if sq in gates and sq in squares)}")
    return placements, len(pieces) + len(overlay)


def render_fence_tbx(width: int, height: int,
                     pieces: dict[tuple[int, int], str],
                     overlay: dict[tuple[int, int], str] | None = None) -> str:
    """A building with no rooms whose only contents are fence tiles.

    Each distinct fence tile is its own single-tile furniture piece, defined for
    all four facings so the reader accepts it, and placed facing west.
    `overlay` tiles share a square with an invisible collision piece (an angled boundary's
    picture over the staircase that carries its collision).
    """
    overlay = overlay or {}
    tiles = sorted(set(pieces.values())) + sorted(set(overlay.values()))
    index = {t: i for i, t in enumerate(tiles)}
    # Pictures on the furniture layer, as every boundary piece; the invisible
    # collision under them on the walls layer (on WallFurniture, WorldEd moved
    # the pictures a square or two off their line).
    second = {t for t in pieces.values() if t.startswith(DIAG_SHEET + "_")
              and int(t.rsplit("_", 1)[1]) % DIAG_SLOTS >= DIAG_INVISIBLE}
    out = ['<?xml version="1.0" encoding="UTF-8"?>']
    attrs = [("version", 4), ("width", width), ("height", height),
             ("ExteriorWall", C.EXTERIOR_WALL), ("ExteriorWallTrim", 0),
             ("Door", C.DOOR), ("DoorFrame", C.DOOR_FRAME), ("Window", C.WINDOW),
             ("Curtains", C.CURTAINS), ("Shutters", 0), ("Stairs", C.STAIRS),
             ("RoofCap", C.ROOF_CAP), ("RoofSlope", C.ROOF_SLOPE),
             ("RoofTop", C.ROOF_TOP), ("GrimeWall", 0)]
    out.append("<building" + "".join(f" {k}={quoteattr(str(v))}" for k, v in attrs) + ">")
    for entry in C.TILE_ENTRIES:
        out.append(f' <tile_entry category={quoteattr(entry["category"])}>')
        for enum_name, tile in entry["tiles"].items():
            out.append(f'  <tile enum={quoteattr(enum_name)} tile={quoteattr(tile)}/>')
        out.append(" </tile_entry>")
    for tile in tiles:
        out.append(' <furniture layer="Walls">' if tile in second else " <furniture>")
        for orient in ("W", "N", "E", "S"):
            out.append(f'  <entry orient="{orient}">')
            out.append(f'   <tile x="0" y="0" name={quoteattr(tile)}/>')
            out.append("  </entry>")
        out.append(" </furniture>")
    out.append(" <used_tiles>" + " ".join(str(i) for i in range(1, len(C.TILE_ENTRIES) + 1))
               + "</used_tiles>")
    out.append(" <used_furniture>" + " ".join(str(i) for i in range(len(tiles)))
               + "</used_furniture>")
    out.append(" <floor>")
    for (x, y), tile in sorted(pieces.items()) + sorted(overlay.items()):
        out.append(f'  <object type="furniture" FurnitureTiles="{index[tile]}" '
                   f'orient="W" x="{x}" y="{y}"/>')
    grid = "\n" + "\n".join(",".join("0" for _ in range(width)) + ("," if y < height - 1 else "")
                            for y in range(height)) + "\n"
    out.append("  <rooms>" + grid + "</rooms>")
    out.append(" </floor>")
    out.append("</building>")
    return "\n".join(out) + "\n"
