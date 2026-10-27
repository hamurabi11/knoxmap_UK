"""Roads in long straight runs at 45-degree steps, the way Knox County's are.

The game's own roads run along the tiles or on the diagonal, nothing between.
A real street at 20 degrees drawn on tiles is a ragged staircase, its kerbs
and markings chopped at every step, and a curving suburban crescent is a
smear. With the "Knox County roads" setting the town is gently redrawn before
anything is painted:

1. The road network becomes a graph: its junctions and the corners left once
   small wiggles are smoothed away.
2. Each stretch between two of those points picks the nearest of the four
   directions the game draws well - along either axis or either diagonal -
   and all the points are moved together, as little as they can be, until
   every stretch runs that way (a least-squares fit; junctions are shared, so
   every road still meets the ones it met). A crescent becomes a few straight
   sides with 45-degree corners; a long road at 10 degrees becomes straight.
3. Everything else - buildings, parks, car parks, water, paths - moves with
   the roads around it, by a smooth blend of how far the nearby roads moved,
   so houses still front their street instead of standing in it.

Two things keep the network joined up where the mapping is loose. A road end
that stops just short of another road (WELD_TILES; a service lane that stops
at the kerb, WELD_LANE_TILES) is joined to it first, so the two are
straightened as one network instead of drifting apart. And a path or track
that meets a road - the same node, or an end within PATH_REACH_TILES - has
that end put on the straightened road (at its real distance from it), the
correction fading along the path; moved by the blend alone it was left up to
forty tiles away from the road it led to.

What is left is a town that reads like the real one, laid out like the game's.

"Grid where buildings are" (the setting's third choice, straight_roads 2): a
stretch with buildings beside it (BUILT_TILES of an outline) is held along one
of the grid's two directions only - no diagonal, the corners square - since
the buildings are laid on the grid and a street of them on the diagonal ran
across their fronts. A built stretch more than BUILT_ANGLE off the grid is
drawn as a stair of grid runs (STAIR_RUN tiles a run) along its real line. The
roads between the built-up stretches keep the 45-degree runs and corners that
join them up.
"""
from __future__ import annotations

import math
import os

import numpy as np

ROAD_CATEGORIES = {"road_major", "road_medium", "road_minor", "road_service",
                   "pedestrian"}
# Wiggles smaller than this, in tiles, are smoothed away first.
SIMPLIFY_TILES = 3.0
# How strongly a point holds to where it really is, against how strongly each
# stretch turns to its direction. Low: roads straighten, the town bends to fit.
HOLD = 0.02
# How far, in tiles, the rest of the town feels a road's move.
BLEND_TILES = 30
# How far straightening may drag a point from where the survey put it. The fit
# on its own has no such limit, and on a big town it used it: a main road at a
# slight angle straightened by moving its far end a couple of blocks, which
# left the road beyond it cut off, and streets on the other side of town ended
# up across the road they used to run beside. Anything that wants to move
# further than this is pinned where it really is and the fit is run again, so
# the straightening gives way rather than the map.
MAX_MOVE_TILES = 24.0
# The hold on a point that has been pinned: large enough that the stretches
# pulling on it cannot drag it anywhere.
PINNED_HOLD = 50.0
# A closed loop smaller than this across is a roundabout, whatever it is
# tagged. Fitting one to the eight directions turned it into a few crossed
# triangles; it is left exactly as it is instead, with its junctions pinned so
# every road still meets it.
ROUNDABOUT_TILES = 60.0
# How far apart the points of one straight run may be across it and still be
# rounded onto the same row, column or diagonal.
ROUND_SPREAD = 2.0
FIELD_CELL = 4                  # tiles per cell of the displacement field
WELD_TILES = 1.5               # a road end this near another road joins it
WELD_LANE_TILES = 6.0          # ...a service lane's end (it stops at the kerb)
PATH_REACH_TILES = 6.0         # a path end this near a road stays at that distance
PATH_CATEGORIES = {"dirt_path", "paved_path"}
BUILT_TILES = 12.0             # a stretch this near a building's outline is on the grid (straight_roads 2)
BUILT_ANGLE = 22.5             # ...and one further than this off the grid a stair of grid runs
STAIR_RUN = 40.0               # tiles a run of such a stair (long, so it reads as a street, not a saw)
BUILT_SIMPLIFY_TILES = 8.0     # ...its wiggles smoothed this far first: a gently curving built street
                               # in short stretches, each on its own axis, was a stair of little jogs
AXES = [(1.0, 0.0), (0.0, 1.0)]
# KnoxMap UK experiment: roads along the grid only, no diagonals. Set
# KNOXMAP_GRID_ROADS=1 for every step that straightens the roads (Generate,
# Build, and Building_types' blocks, kerbs and shops), or they disagree.
GRID_ONLY = os.environ.get("KNOXMAP_GRID_ROADS") == "1"
DIRECTIONS = [(1.0, 0.0), (0.0, 1.0)] if GRID_ONLY else     [(1.0, 0.0), (math.sqrt(.5), math.sqrt(.5)), (0.0, 1.0), (-math.sqrt(.5), math.sqrt(.5))]


def _from_road_graph(features: list, path: str, proj, classify, is_polygon) -> int:
    """KnoxMap UK experiment (Building_types/src/roadgraph): the roads as the
    road graph snapped them, read from `path` - {"ways": {osm id: {"geometry":
    [[lat, lon], ...], "tags": {...}}}, "drop": [osm id, ...], "shift":
    [[lat, lon, lat, lon], ...]} - in place of straightening. "shift" (points
    along the roads, where they were and where the graph put them) moves
    everything else with the roads near it, as straightening does; without
    it nothing else moves. Set KNOXMAP_ROAD_GRAPH to the file for every step
    that straightens the roads, or they disagree."""
    import json
    with open(path, encoding="utf-8") as fh:
        graph = json.load(fh)
    ways = {int(k): v for k, v in graph["ways"].items()}
    drop = set(graph.get("drop", []))
    kept, roads, real_lines, road_nodes = [], [], [], set()
    for f in features:
        if f.kind == "way" and f.osm_id in drop:
            continue
        g = ways.get(f.osm_id) if f.kind == "way" else None
        if g:
            before = list(f.geometry)
            road_nodes |= {_key(lat, lon) for lat, lon in before}
            f.geometry = [tuple(p) for p in g["geometry"]]
            f.tags = {**f.tags, **g.get("tags", {})}
            roads.append(f)
            real_lines.append(([proj.to_px(lat, lon) for lat, lon in before],
                               [proj.to_px(lat, lon) for lat, lon in f.geometry]))
        kept.append(f)
    features[:] = kept
    if graph.get("shift") and proj is not None:
        samples = []
        for la0, lo0, la1, lo1 in graph["shift"]:
            p0, p1 = np.array(proj.to_px(la0, lo0)), np.array(proj.to_px(la1, lo1))
            samples.append((p0, p1 - p0))
        _carry(features, roads, proj, classify, is_polygon, displacement(samples, proj), real_lines, road_nodes)
    return len(roads)


def _key(lat: float, lon: float) -> tuple[int, int]:
    return round(lat * 1e7), round(lon * 1e7)


def _simplify_idx(pts: list[tuple[float, float]], tol: float) -> list[int]:
    """Douglas-Peucker; the indices kept, both ends included."""
    if len(pts) < 3:
        return list(range(len(pts)))
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        best, best_d = -1, tol
        for i in range(a + 1, b):
            d = _dist_to_line(pts[i], pts[a], pts[b])
            if d > best_d:
                best, best_d = i, d
        if best >= 0:
            keep[best] = True
            stack += [(a, best), (best, b)]
    return [i for i, k in enumerate(keep) if k]


def _dist_to_line(p, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length == 0:
        return math.hypot(px - ax, py - ay)
    return abs(dx * (ay - py) - (ax - px) * dy) / length


def _legs(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """From a to b along the grid and the diagonal: half the straight part,
    the diagonal, the other half."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    if GRID_ONLY:
        # Along the longer axis half way, across, then the rest: a dog-leg.
        if abs(dx) >= abs(dy):
            return [(ax + dx // 2, ay), (ax + dx // 2, by), b]
        return [(ax, ay + dy // 2), (bx, ay + dy // 2), b]
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    diag = min(abs(dx), abs(dy))
    first = (max(abs(dx), abs(dy)) - diag) // 2
    p1 = (ax + sx * first, ay) if abs(dx) >= abs(dy) else (ax, ay + sy * first)
    return [p1, (p1[0] + sx * diag, p1[1] + sy * diag), b]


def _grid_legs(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """From a to b along the grid only (a built stretch): along the longer axis half
    way, across, then the rest - a square jog where the fit left the two ends a
    little out of line, never a diagonal."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 or dy == 0:
        return [b]
    if abs(dx) >= abs(dy):
        return [(ax + dx // 2, ay), (ax + dx // 2, by), b]
    return [(ax, ay + dy // 2), (bx, ay + dy // 2), b]


def _tidy(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Without repeated points or points in the middle of a straight run."""
    out: list[tuple[int, int]] = []
    for p in path:
        if out and p == out[-1]:
            continue
        if len(out) >= 2:
            (x0, y0), (x1, y1) = out[-2], out[-1]
            if (x1 - x0) * (p[1] - y1) == (y1 - y0) * (p[0] - x1) and \
                    (x1 - x0) * (p[0] - x1) + (y1 - y0) * (p[1] - y1) >= 0:
                out[-1] = p
                continue
        out.append(p)
    return out


def _rigid(f, px: list) -> bool:
    """Whether this road is left exactly as it was drawn: a roundabout, or any
    small closed loop, which has no straight sides to fit."""
    if (f.tags.get("junction") or "").lower() in {"roundabout", "circular"}:
        return True
    if len(px) < 4 or math.dist(px[0], px[-1]) > 1e-6:
        return False
    xs = [x for x, _ in px]
    ys = [y for _, y in px]
    return max(max(xs) - min(xs), max(ys) - min(ys)) <= ROUNDABOUT_TILES


def _fit(p0: np.ndarray, segs: np.ndarray, normals: np.ndarray,
         weights: np.ndarray, pinned: np.ndarray | None = None) -> np.ndarray:
    """The points, straightened: once held firmly where they are, then again
    held only loosely to that, which is what takes the last of the slant out."""
    def holds(value: float) -> np.ndarray:
        out = np.full((len(p0), 1), value)
        if pinned is not None:
            out[pinned] = PINNED_HOLD
        return out

    moved = _solve(p0, segs, normals, weights, hold=holds(HOLD))
    return _solve(moved, segs, normals, weights, iterations=1500,
                  hold=holds(HOLD / 50))


def _solve(p0: np.ndarray, segs: np.ndarray, normals: np.ndarray,
           weights: np.ndarray, iterations: int = 400, hold=None) -> np.ndarray:
    """Points as near p0 as they can be with every segment turned along its
    direction: minimise HOLD*|P - p0|^2 + sum w*(n . (P_j - P_i))^2, by
    conjugate gradients on the normal equations."""
    hold = HOLD if hold is None else hold
    i, j = segs[:, 0], segs[:, 1]

    def apply(x):
        out = hold * x
        v = weights * np.einsum("ij,ij->i", normals, x[j] - x[i])
        np.add.at(out, j, v[:, None] * normals)
        np.add.at(out, i, -v[:, None] * normals)
        return out

    b = hold * p0
    x = p0.copy()
    r = b - apply(x)
    p = r.copy()
    rs = float((r * r).sum())
    for _ in range(iterations):
        if rs < 1e-6:
            break
        ap = apply(p)
        alpha = rs / float((p * ap).sum())
        x += alpha * p
        r -= alpha * ap
        rs_new = float((r * r).sum())
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def _blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Three box blurs, near enough a Gaussian, along both axes."""
    for axis in (0, 1):
        for _ in range(3):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (radius + 1, radius)
            c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
            hi = np.take(c, range(2 * radius + 1, c.shape[axis]), axis=axis)
            lo = np.take(c, range(0, c.shape[axis] - 2 * radius - 1), axis=axis)
            a = (hi - lo) / (2 * radius + 1)
    return a


def _round_together(moved: np.ndarray, segs: np.ndarray, normals: np.ndarray) -> list:
    """Whole tiles for the points, the same row for every point along one
    east-west run, the same column along a north-south one and the same
    diagonal along a diagonal one - rounded one at a time, the ends of a long
    straight road can land either side of a tile edge, and it jogs."""
    n = len(moved)
    parent = {kind: list(range(n)) for kind in ("row", "col", "down", "up")}

    def find(kind, a):
        p = parent[kind]
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    for (a, b), (nx, ny) in zip(segs, normals):
        if abs(nx) < 1e-9:
            kind = "row"                  # runs along x: normal is (0, ±1)
        elif abs(ny) < 1e-9:
            kind = "col"
        elif nx * ny < 0:
            kind = "down"                 # along (1, 1): x - y is constant
        else:
            kind = "up"                   # along (-1, 1): x + y is constant
        ra, rb = find(kind, int(a)), find(kind, int(b))
        if ra != rb:
            parent[kind][ra] = rb

    def shared(kind, values):
        groups: dict[int, list[float]] = {}
        for k in range(n):
            groups.setdefault(find(kind, k), []).append(values[k])
        mean = {g: round(sum(v) / len(v)) for g, v in groups.items()}
        # Only where the run really did come out straight. A road the fit
        # could not straighten - because straightening it would have dragged
        # it off the map it is a model of - was still averaged onto one row
        # here, which moved it bodily sideways and left the roads it met
        # behind. That run keeps its own shape and is rounded point by point.
        together = {g: len(v) > 1 and max(abs(x - mean[g]) for x in v) <= ROUND_SPREAD
                    for g, v in groups.items()}
        return ([mean[find(kind, k)] for k in range(n)],
                [together[find(kind, k)] for k in range(n)])

    xs, ys = moved[:, 0], moved[:, 1]
    row, in_row = shared("row", ys)
    col, in_col = shared("col", xs)
    down, in_down = shared("down", xs - ys)
    up, in_up = shared("up", xs + ys)
    out = []
    for k in range(n):
        y = row[k] if in_row[k] else round(ys[k])
        if in_col[k]:
            x = col[k]
        elif in_down[k]:
            x = down[k] + y
        elif in_up[k]:
            x = up[k] - y
        else:
            x = round(xs[k])
        if not in_row[k] and not in_col[k] and in_down[k] and in_up[k]:
            # Where two diagonals cross, both hold.
            y = (up[k] - down[k]) // 2
            x = down[k] + y
        out.append((int(x), int(y)))
    return out


def _weld(roads, proj, classify) -> int:
    """Join road ends that stop just short of another road to it, in place:
    the end takes that road's nearest node, or a node put into it there."""
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree
    lines = [LineString([proj.to_px(la, lo) for la, lo in f.geometry]) for f in roads]
    tree = STRtree(lines)
    uses: dict = {}
    for f in roads:
        for lat, lon in f.geometry:
            k = _key(lat, lon)
            uses[k] = uses.get(k, 0) + 1
    welded = 0
    for i, f in enumerate(roads):
        reach = WELD_LANE_TILES if classify(f.tags) == "road_service" else WELD_TILES
        for end in (0, -1):
            lat, lon = f.geometry[end]
            if uses[_key(lat, lon)] > 1:
                continue                      # already meets another road
            p = Point(proj.to_px(lat, lon))
            best = None
            for j in tree.query(p.buffer(reach)):
                j = int(j)
                if j == i:
                    continue
                d = lines[j].distance(p)
                if d <= reach and (best is None or d < best[0]):
                    best = (d, j)
            if best is None:
                continue
            j = best[1]
            other = roads[j]
            # Its nearest node if that is close; else a node put in there.
            near = min(range(len(other.geometry)),
                       key=lambda n: Point(proj.to_px(*other.geometry[n])).distance(p))
            if Point(proj.to_px(*other.geometry[near])).distance(p) <= WELD_TILES:
                node = other.geometry[near]
            else:
                q = lines[j].interpolate(lines[j].project(p))
                node = proj.to_latlon(q.x, q.y)
                coords = list(lines[j].coords)
                at = lines[j].project(p)
                run = 0.0
                for n in range(len(coords) - 1):
                    seg = Point(coords[n]).distance(Point(coords[n + 1]))
                    if run + seg >= at:
                        other.geometry = list(other.geometry[:n + 1]) + [node] + list(other.geometry[n + 1:])
                        lines[j] = LineString([proj.to_px(la, lo) for la, lo in other.geometry])
                        break
                    run += seg
            geom = list(f.geometry)
            geom[end] = tuple(node)
            f.geometry = geom
            lines[i] = LineString([proj.to_px(la, lo) for la, lo in f.geometry])
            uses[_key(*node)] = uses.get(_key(*node), 0) + 1
            welded += 1
    return welded


def _move_path(geometry, proj, move, real_lines, junction_keys, index=None):
    """A path moved with the town, but with each end that meets a road (a
    shared node, or within PATH_REACH_TILES) put on the straightened road at
    its real distance from it; the correction fades along the path.
    `index`: (lines, STRtree) over real_lines' roads before, built once by
    _carry (KnoxMap UK, 2026-10-21: measuring every road for every path end
    took over an hour on central London)."""
    from shapely.geometry import LineString, Point
    moved = [proj.to_px(*move(lat, lon)) for lat, lon in geometry]
    n = len(geometry)
    fix = {}
    for end in (0, n - 1):
        lat, lon = geometry[end]
        p = Point(proj.to_px(lat, lon))
        best = None
        if index is not None:
            lines, tree = index
            near = sorted(int(k) for k in tree.query(p, predicate="dwithin", distance=PATH_REACH_TILES))
        else:
            lines, near = None, range(len(real_lines))
        for k in near:
            before, after = real_lines[k]
            line = lines[k] if lines is not None else LineString(before)
            d = line.distance(p)
            if d <= PATH_REACH_TILES and (best is None or d < best[0]):
                best = (d, before, after)
        if best is None:
            continue
        d, before, after = best
        if _key(lat, lon) in junction_keys:
            d = 0.0
        # Where along the road it met, as a share of the road; there on the
        # straightened road, then its real distance off it, on the side the
        # blend put it.
        bl, al = LineString(before), LineString(after)
        t = bl.project(p) / max(bl.length, 1e-6)
        q = al.interpolate(t * al.length)
        mx, my = moved[end]
        vx, vy = mx - q.x, my - q.y
        vlen = math.hypot(vx, vy)
        if vlen > 1e-6 and d > 0:
            tx, ty = q.x + vx / vlen * d, q.y + vy / vlen * d
        else:
            tx, ty = q.x, q.y
        fix[end] = (tx - mx, ty - my)
    if not fix:
        return [move(lat, lon) for lat, lon in geometry]
    # Corrections at the ends, faded linearly along the path's length.
    pts = [proj.to_px(lat, lon) for lat, lon in geometry]
    along = [0.0]
    for a, b in zip(pts, pts[1:]):
        along.append(along[-1] + math.dist(a, b))
    total = max(along[-1], 1e-6)
    out = []
    for k, (mx, my) in enumerate(moved):
        t = along[k] / total
        dx = dy = 0.0
        if 0 in fix:
            dx += fix[0][0] * (1 - t)
            dy += fix[0][1] * (1 - t)
        if n - 1 in fix:
            dx += fix[n - 1][0] * t
            dy += fix[n - 1][1] * t
        out.append(proj.to_latlon(mx + dx, my + dy))
    return out


def road_graph_of(map_dir: str) -> None:
    """KnoxMap UK: KNOXMAP_ROAD_GRAPH set to the map's own road graph (its
    info.json's "road_graph", written by Building_types' handover.py), so a
    step run on a road-graph map straightens as its Generate did; and an
    error if the environment names a road graph the map was not made with
    (another map's, or one left set for a map that has none)."""
    import json
    name = os.path.basename(os.path.normpath(map_dir))
    path = os.path.join(map_dir, f"{name}_info.json")
    info = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
    want = os.path.abspath(os.path.join(map_dir, info["road_graph"])) if info.get("road_graph") else None
    have = os.environ.get("KNOXMAP_ROAD_GRAPH")
    same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    if have and not (want and same(have, want)):
        raise RuntimeError(f"KNOXMAP_ROAD_GRAPH is {have}, but {name} was made "
                           + (f"with {want}" if want else "without a road graph") + ": unset it")
    if want:
        os.environ["KNOXMAP_ROAD_GRAPH"] = want


def _built_mask(features, proj, is_polygon) -> np.ndarray:
    """Tiles within BUILT_TILES of a building's outline."""
    from PIL import Image, ImageDraw
    from scipy import ndimage
    img = Image.new("1", (proj.width, proj.height), 0)
    d = ImageDraw.Draw(img)
    for f in features:
        if f.kind == "way" and f.tags.get("building") and len(f.geometry) >= 3:
            d.polygon([proj.to_px(lat, lon) for lat, lon in f.geometry], fill=1)
    return ndimage.distance_transform_edt(~np.array(img, bool)) <= BUILT_TILES


def _in_built(built: np.ndarray, pa, pb) -> bool:
    """Most of the stretch from pa to pb on the built mask."""
    H, W = built.shape
    on = 0
    for t in np.linspace(0.0, 1.0, 7):
        x, y = pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t
        on += bool(built[min(max(int(y), 0), H - 1), min(max(int(x), 0), W - 1)])
    return on >= 4


def _grid_mode(map_dir: str | None) -> bool:
    """Whether the map was drawn with "grid where buildings are" (info.json straight_roads 2)."""
    if not map_dir:
        return False
    import json
    name = os.path.basename(os.path.normpath(map_dir))
    try:
        with open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8") as f:
            return int(json.load(f).get("straight_roads") or 0) == 2
    except (OSError, ValueError, TypeError):
        return False


def straighten_roads(features: list, proj, classify, is_polygon, map_dir: str | None = None,
                     grid_built: bool | None = None) -> int:
    """Redraw the roads in `features` and move the rest of the town with them,
    in place. Returns how many roads were redrawn. map_dir: the map they are
    for - its road graph, if it was made with one, is used (road_graph_of).
    grid_built: roads where buildings are on the grid (the setting's third
    choice); None reads it from the map's info.json (every later step that
    straightens again must draw the roads where the renderer did)."""
    if grid_built is None:
        grid_built = _grid_mode(map_dir)
    if map_dir is not None:
        road_graph_of(map_dir)
    if os.environ.get("KNOXMAP_ROAD_GRAPH"):
        return _from_road_graph(features, os.environ["KNOXMAP_ROAD_GRAPH"], proj, classify, is_polygon)
    roads = [f for f in features
             if f.kind == "way" and len(f.geometry) >= 2 and not is_polygon(f)
             and classify(f.tags) in ROAD_CATEGORIES]
    if not roads:
        return 0
    _weld(roads, proj, classify)

    # --- the graph: junctions shared, corners per road ---------------------
    uses: dict[tuple[int, int], int] = {}
    for f in roads:
        for n, (lat, lon) in enumerate(f.geometry):
            k = _key(lat, lon)
            uses[k] = uses.get(k, 0) + (2 if n in (0, len(f.geometry) - 1) else 1)
    points: list[tuple[float, float]] = []
    junction: dict[tuple[int, int], int] = {}
    road_vertices: list[list[int]] = []      # per road, its graph points in order
    road_px: list[list[tuple[float, float]]] = []

    def vertex(xy, key=None):
        if key is not None and key in junction:
            return junction[key]
        points.append(xy)
        if key is not None:
            junction[key] = len(points) - 1
        return len(points) - 1

    built = _built_mask(features, proj, is_polygon) if grid_built else None
    rigid: list[bool] = []            # roads left exactly as they were drawn
    for f in roads:
        px = [proj.to_px(lat, lon) for lat, lon in f.geometry]
        road_px.append(px)
        rigid.append(_rigid(f, px))
        keys = [_key(lat, lon) for lat, lon in f.geometry]
        cuts = [n for n, k in enumerate(keys) if n in (0, len(keys) - 1) or uses[k] >= 2]
        verts = [vertex(px[0], keys[0])]
        for a, b in zip(cuts, cuts[1:]):
            tol = SIMPLIFY_TILES
            if built is not None and _in_built(built, px[a], px[b]):
                tol = BUILT_SIMPLIFY_TILES
            kept = _simplify_idx(px[a:b + 1], tol)
            for n in kept[1:-1]:
                verts.append(vertex(px[a + n]))
            verts.append(vertex(px[b], keys[b]))
        road_vertices.append(verts)

    # --- grid where buildings are: built stretches on the axes, steep ones stairs ---
    grid_pairs: set = set()
    if built is not None:
        for r, (held, verts) in enumerate(zip(rigid, road_vertices)):
            if held:
                continue
            new = [verts[0]]
            for a, b in zip(verts, verts[1:]):
                pa, pb = points[a], points[b]
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                length = math.hypot(dx, dy)
                if length < 1e-6 or not _in_built(built, pa, pb):
                    new.append(b)
                    continue
                off = math.degrees(math.atan2(min(abs(dx), abs(dy)), max(abs(dx), abs(dy))))
                if off <= BUILT_ANGLE or length < STAIR_RUN:
                    new.append(b)
                    grid_pairs.add((a, b))
                    continue
                # A stair: n steps, each a run along x then one along y.
                n = max(1, round(length / STAIR_RUN))
                prev = a
                for k in range(n):
                    sx, sy = pa[0] + dx * k / n, pa[1] + dy * k / n
                    corner = vertex((sx + dx / n, sy))
                    end = b if k == n - 1 else vertex((sx + dx / n, sy + dy / n))
                    grid_pairs |= {(prev, corner), (corner, end)}
                    new += [corner, end]
                    prev = end
            road_vertices[r] = new

    p0 = np.array(points, dtype=float)
    pinned = np.zeros(len(points), dtype=bool)
    segs, normals, weights = [], [], []
    for held, verts in zip(rigid, road_vertices):
        if held:
            # A roundabout holds its own shape and its junctions, so the roads
            # into it still meet it; nothing turns it to the eight directions.
            pinned[verts] = True
            continue
        for a, b in zip(verts, verts[1:]):
            if a == b:
                continue
            d = p0[b] - p0[a]
            length = float(np.hypot(*d))
            if length < 1e-6:
                continue
            u = d / length
            best = max(AXES if (a, b) in grid_pairs else DIRECTIONS, key=lambda t: abs(u[0] * t[0] + u[1] * t[1]))
            segs.append((a, b))
            normals.append((-best[1], best[0]))
            # Long stretches turn firmly; a short one between two junctions
            # gives way to the roads around it.
            weights.append(min(length, 60.0) / 10.0)
    if not segs:
        return 0
    segs, normals, weights = np.array(segs), np.array(normals), np.array(weights)
    # Two things undo a fit and send it round again (up to four times):
    # - nothing may be dragged a block from where the survey put it
    #   (upstream 1.7.2): whatever wanted to be is pinned, the roads around it
    #   take up the slack, and the ones that cannot stay as they were drawn;
    # - a stretch is only held parallel to its direction, not pointing along
    #   it, so a short link can come out reversed: London Road leaves the end
    #   of Odiham High Street westwards, but the long road beyond it, made one
    #   diagonal, moved its end forty tiles east, and the link ran 45 tiles
    #   east across the High Street and doubled back. A stretch that reverses
    #   is let go (no direction of its own; a dog-leg between its ends).
    free = np.zeros(len(segs), bool)
    moved = _fit(p0, segs, normals, weights, pinned)
    for _ in range(4):
        far = (np.hypot(*(moved - p0).T) > MAX_MOVE_TILES) & ~pinned
        d0 = p0[segs[:, 1]] - p0[segs[:, 0]]
        d1 = moved[segs[:, 1]] - moved[segs[:, 0]]
        flipped = ((d0 * d1).sum(1) < 0) & ~free
        if not far.any() and not flipped.any():
            break
        pinned |= far
        free |= flipped
        moved = _fit(p0, segs, normals, np.where(free, 0.0, weights), pinned)

    # --- the roads, from the moved points ---------------------------------
    snapped = _round_together(moved, segs[~free], normals[~free])
    real_lines = []                # each road before and after, in tiles
    for f, held, verts, px in zip(roads, rigid, road_vertices, road_px):
        if held:
            real_lines.append((px, px))   # a roundabout stays as drawn; paths still meet it
            continue
        path = [snapped[verts[0]]]
        for u_, v in zip(verts, verts[1:]):
            path += _grid_legs(path[-1], snapped[v]) if (u_, v) in grid_pairs else _legs(path[-1], snapped[v])
        path = _tidy(path)
        if len(path) == 1:
            # A short link between two junctions the straightening brought
            # together: it is the junction now. Left as it was, it stood up to
            # a dozen tiles from both roads it joined.
            path = [path[0], path[0]]
        if len(path) >= 2:
            f.geometry = [proj.to_latlon(x, y) for x, y in path]
            real_lines.append((px, path))

    # --- everything else moves with the roads near it ---------------------
    disp = moved - p0
    samples = []
    for held, verts, px in zip(rigid, road_vertices, road_px):
        if held:
            continue
        # Sample along each stretch, the move blended between its ends.
        for a, b in zip(verts, verts[1:]):
            pa, pb = p0[a], p0[b]
            steps = max(1, int(np.hypot(*(pb - pa)) / FIELD_CELL))
            for t in np.linspace(0.0, 1.0, steps + 1):
                samples.append((pa + (pb - pa) * t, disp[a] * (1 - t) + disp[b] * t))
    road_nodes = {_key(lat, lon) for f in roads for lat, lon in f.geometry}
    road_nodes |= set(junction)
    _carry(features, roads, proj, classify, is_polygon, displacement(samples, proj), real_lines, road_nodes)
    return len(roads)


def displacement(samples, proj):
    """move(lat, lon) -> (lat, lon): the town's move at a point, from
    `samples` [(point before, its move)] in tiles along the roads - their
    moves blended over BLEND_TILES, fading out far from any road."""
    gw, gh = proj.width // FIELD_CELL + 2, proj.height // FIELD_CELL + 2
    weight = np.zeros((gh, gw))
    shift = np.zeros((gh, gw, 2))
    for q, d in samples:
        cx, cy = int(q[0] // FIELD_CELL), int(q[1] // FIELD_CELL)
        if 0 <= cx < gw and 0 <= cy < gh:
            weight[cy, cx] += 1.0
            shift[cy, cx] += d
    radius = max(1, BLEND_TILES // FIELD_CELL)
    w_blur = _blur(weight, radius)
    field_x = _blur(shift[..., 0], radius) / (w_blur + 1e-3)
    field_y = _blur(shift[..., 1], radius) / (w_blur + 1e-3)
    # Far from any road the move fades out rather than jumping to nothing.
    fade = np.clip(w_blur / (w_blur.max() * 0.02 + 1e-9), 0.0, 1.0)
    field_x *= fade
    field_y *= fade

    def move(lat, lon):
        x, y = proj.to_px(lat, lon)
        cx = min(gw - 1, max(0, int(x // FIELD_CELL)))
        cy = min(gh - 1, max(0, int(y // FIELD_CELL)))
        return proj.to_latlon(x + field_x[cy, cx], y + field_y[cy, cx])
    return move


def _carry(features, roads, proj, classify, is_polygon, move, real_lines, road_nodes) -> None:
    """Everything in `features` but the roads moved by `move`, in place; a
    path's ends that meet a road put on it as it is now (_move_path, from
    real_lines [(road before, after)] in tiles; road_nodes: the roads' node
    keys before), and paths that shared a node made to share it still."""
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    road_ids = {id(f) for f in roads}
    lines = [LineString(before) for before, _after in real_lines]
    index = (lines, STRtree(lines)) if lines else None
    paths: list = []               # (feature, its nodes before) for the paths
    for f in features:
        if id(f) in road_ids:
            continue
        if f.kind == "way" and len(f.geometry) >= 2 and not is_polygon(f) \
                and classify(f.tags) in PATH_CATEGORIES:
            before = list(f.geometry)
            f.geometry = _move_path(f.geometry, proj, move, real_lines, junction_keys=road_nodes,
                                    index=index)
            paths.append((f, before))
            continue
        if f.geometry and f.kind in ("way", "node"):
            f.geometry = [move(lat, lon) for lat, lon in f.geometry]
        if f.kind == "relation" and getattr(f, "role_geoms", None):
            f.role_geoms = [(role, [move(lat, lon) for lat, lon in ring])
                            for role, ring in f.role_geoms]
            if f.geometry and isinstance(f.geometry[0][0], (list, tuple)):
                f.geometry = [ring for _role, ring in f.role_geoms]
    # Paths that shared a node share it still: each end was pinned on its own,
    # so where one met a road and the next did not they came apart.
    shared: dict = {}
    for f, before in paths:
        for n, (lat, lon) in enumerate(before):
            shared.setdefault(_key(lat, lon), []).append((f, n))
    for users in shared.values():
        if len(users) < 2:
            continue
        lat = sum(f.geometry[n][0] for f, n in users) / len(users)
        lon = sum(f.geometry[n][1] for f, n in users) / len(users)
        for f, n in users:
            geom = list(f.geometry)
            geom[n] = (lat, lon)
            f.geometry = geom
