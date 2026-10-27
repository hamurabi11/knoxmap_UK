"""The roadside (new_road_approach.md, build order 2's second half): a grass
verge and a pavement each side of a road, drawn along the snapped road.

English streets mostly have a pavement, residential closes included, often
behind a kerbed grass verge (user, 2026-10-11: Archery Fields - kerb, verge,
tarmac pavement, hedge, both sides). Vanilla's rule (pavements on main
streets only) is Kentucky's, not ours. Each side of each link:

1. OSM says there is no pavement on it (sidewalk=no, or one on the other
   side only): nothing.
2. The LiDAR shows a verge with a pavement behind it (widths._roadside) on
   VERGE_SHARE of the side's transects, VERGE_HITS of them at least, or on
   SIDE_SHARE where its street as a whole shows one on STREET_SHARE (a short
   link of a verged street has too few transects of its own): that verge
   and pavement, their median widths in whole tiles (1 to MAX_TILES). About
   one transect in seven shows the pattern on streets without a verge (a
   lawn, then a drive), so a single link side needs well over that.
3. OSM says there is a pavement, or says nothing and the road is a town
   street (PAVED: not a service lane or track): a pavement of PAVEMENT_TILES,
   or less where the LiDAR shows less room (widths.measure: half the real
   carriageway and the open ground beside it, less half the drawn road and
   its bays; at least 1 tile) - old streets' pavements are narrow, and a
   2-tile one ran under the buildings at the back of it.
4. Parking bays along the kerb where OSM tags them (parking:left/right/both,
   or the older parking:lane:*): PARKING_TILES by how the cars stand, between
   the carriageway and the verge or pavement (Odiham's High Street: pavement,
   parallel bays, then the road; user, 2026-10-11).
Only in town (the hand-over's town mask). The sides are the real ones, left
facing along the link from u to v; draw() turns them into the map's.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from scipy import ndimage

VERGE_SHARE, VERGE_HITS = 0.4, 4
STREET_SHARE, SIDE_SHARE = 0.3, 0.25
PAVEMENT_TILES = 2
MAX_TILES = 3
PAVED = ("a_road", "b_road", "minor", "local")
NO = ("no", "none")
# A bay a car stands in, across: vanilla's stalls are 3 x 5 tiles.
PARKING_TILES = {"parallel": 3, "diagonal": 4, "perpendicular": 5}
PARKS = ("lane", "street_side", "yes")               # parking:<side>=... that is a bay beside the kerb


def _tiles(m: float) -> int:
    return min(MAX_TILES, max(1, int(math.floor(m + 0.5))))


def osm_sides(tags: dict):
    """(left, right) of an OSM way, facing along it: True a pavement, False
    none, None not said."""
    left = right = None
    sw = (tags.get("sidewalk") or "").lower()
    if sw in ("both", "yes", "separate"):
        left = right = True
    elif sw == "left":
        left, right = True, False
    elif sw == "right":
        left, right = False, True
    elif sw in NO:
        left = right = False
    if tags.get("sidewalk:both"):
        left = right = tags["sidewalk:both"].lower() not in NO
    for key, k in (("sidewalk:left", 0), ("sidewalk:right", 1)):
        if tags.get(key):
            v = tags[key].lower() not in NO
            left, right = (v, right) if k == 0 else (left, v)
    return left, right


def osm_parking(tags: dict):
    """(left, right) of an OSM way, facing along it: a bay's depth in tiles
    (PARKING_TILES), 0 none, None not said."""
    out = [None, None]
    for k, side in ((0, "left"), (1, "right")):
        for key in (f"parking:{side}", "parking:both"):          # the current scheme
            v = (tags.get(key) or "").lower()
            if v:
                how = (tags.get(f"{key}:orientation") or "parallel").lower()
                out[k] = PARKING_TILES.get(how, PARKING_TILES["parallel"]) if v in PARKS else 0
                break
        else:
            for key in (f"parking:lane:{side}", "parking:lane:both"):   # the older one
                v = (tags.get(key) or "").lower()
                if v:
                    out[k] = PARKING_TILES.get(v, 0)
                    break
    return tuple(out)


def _osm_for_link(link, ways: dict, read=osm_sides):
    """(left, right) facing u to v by `read` from the first of the link's
    ways that says anything, turned round if the way runs v to u."""
    line = shapely.LineString(link.pts)
    tol = max(3.0, float(link.width))
    for w in link.ways:
        pts, tags = ways.get(w, (None, None))
        if not pts:
            continue
        lr = read(tags)
        if lr == (None, None):
            continue
        near = [line.project(shapely.Point(p)) for p in pts if line.distance(shapely.Point(p)) <= tol]
        if len(near) >= 2 and near[0] > near[-1]:
            lr = lr[::-1]
        return lr
    return None, None


def decide(links, sides: dict, ways: dict, measured: dict | None = None) -> tuple[dict, dict]:
    """({link index: ((verge, pavement, parking) left, (...) right)} in
    tiles, a tally). sides: widths.measure's per-link roadside; ways: {OSM
    way id: (its points in tiles, its tags)}; measured: widths.measure's
    result (in metres, at a tile a metre), for the room beside the road."""
    street: dict = {}
    for i, rows in sides.items():
        name = links[i].tags.get("name")
        if name:
            s = street.setdefault(name, [0, 0])
            s[0] += sum(r is not None for t in rows for r in t)
            s[1] += 2 * len(rows)
    # Which link sides have a verge, and the verge and pavement widths: one
    # pair a named street (pooled over its verged sides), else a link's own.
    verged: dict = {}
    pool: dict = {}
    for i, l in enumerate(links):
        rows = sides.get(i, [])
        s = street.get(l.tags.get("name"), [0, 0])
        verged_street = s[1] and s[0] / s[1] >= STREET_SHARE
        for k in (0, 1):
            hits = [t[k] for t in rows if t[k] is not None]
            share = len(hits) / len(rows) if rows else 0.0
            if (len(hits) >= VERGE_HITS and share >= VERGE_SHARE) or (verged_street and share >= SIDE_SHARE):
                verged[(i, k)] = True
                pool.setdefault(l.tags.get("name") or ("link", i), []).extend(hits)
    size = {key: (_tiles(float(np.median([h[0] for h in hits]))), _tiles(float(np.median([h[1] for h in hits]))))
            for key, hits in pool.items()}
    out, tally = {}, {"verge": 0, "pavement": 0, "none": 0, "osm no": 0, "parking": 0}
    for i, l in enumerate(links):
        osm = _osm_for_link(l, ways)
        park = _osm_for_link(l, ways, osm_parking)
        both = []
        for k in (0, 1):
            if osm[k] is False:
                side = (0, 0)
                tally["osm no"] += 1
            elif verged.get((i, k)):
                side = size[l.tags.get("name") or ("link", i)]
                tally["verge"] += 1
            elif osm[k] or (osm[k] is None and l.cls in PAVED):
                pave = PAVEMENT_TILES
                m = (measured or {}).get(i)
                if m:
                    # The two sides' smaller: a street's links are pooled, and run either way.
                    room = m["carriageway"] / 2 + min(m["left"], m["right"]) - l.width / 2 - (park[k] or 0)
                    pave = min(PAVEMENT_TILES, max(1, int(math.floor(room + 0.25))))
                    tally["narrow"] = tally.get("narrow", 0) + (pave < PAVEMENT_TILES)
                side = (0, pave)
                tally["pavement"] += 1
            else:
                side = (0, 0)
                tally["none"] += 1
            tally["parking"] += bool(park[k])
            both.append(side + (park[k] or 0,))
        out[i] = tuple(both)
    return out, tally


def handedness(proj) -> float:
    """+1 if the map's tiles keep the real world's turn (left of a heading
    is left on the map, x right and y down), -1 if they mirror it."""
    cx, cy = proj.width / 2, proj.height / 2
    la, lo = proj.to_latlon(cx, cy)
    e = np.subtract(proj.to_px(la, lo + 1e-4), (cx, cy))
    n = np.subtract(proj.to_px(la + 1e-4, lo), (cx, cy))
    # Facing east, north is on the left: the sign of east x north on the map.
    return float(np.sign(e[0] * n[1] - e[1] * n[0]))


END_TILES = 3      # within this of a link's end (a junction's corner), the roadside by distance alone


def outlines(paths: dict, links, chosen: dict, hand: float) -> dict:
    """{link: (to the bays' outer edge, the verge's, the pavement's)}: each
    an outline round the snapped path (raster.outline with each side's
    widths), its edges on the lattice through tile corners like the
    carriageway's, so an angled roadside is straight too (user, 2026-10-11:
    the High Street's bays and the back of the pavement stepped). Sides as
    draw() takes them: chosen's first, `left`, is the side where the path's
    cross product times `hand` is positive."""
    from .raster import outline
    out = {}
    for i, both in chosen.items():
        if i not in paths:
            continue
        sides = both if hand > 0 else both[::-1]       # outline's own left (cross > 0) first
        b = tuple(s[2] for s in sides)
        bv = tuple(s[2] + s[0] for s in sides)
        bvp = tuple(s[2] + s[0] + s[1] for s in sides)
        out[i] = tuple(outline(paths[i], links[i].width, e) for e in (b, bv, bvp))
    return out


def draw(paths: dict, links, chosen: dict, owner: np.ndarray, half: dict, hand: float):
    """(verge, pavement, parking) masks round the road tiles: each square off
    the road is its nearest road square's link's; out from the carriageway,
    its parking bays, verge and pavement - the square whose middle lies in
    that band's outline (outlines). Near a link's end, where its outline stops
    short of a junction's corner, by distance from the road on the side of the
    link the square lies."""
    H, W = owner.shape
    road = owner != 0
    for (x, y) in half:
        if 0 <= y < H and 0 <= x < W:
            road[y, x] = True
    reach = max(sum(side) for both in chosen.values() for side in both) + 0.5 if chosen else 0
    lab = np.where(owner > 0, owner - 1, -1)
    _d, (iy, ix) = ndimage.distance_transform_edt(lab < 0, return_indices=True)
    link_at = lab[iy, ix]                                # each square's nearest full road square's link
    dist, (ry, rx) = ndimage.distance_transform_edt(~road, return_indices=True)
    near = (dist > 0) & (dist <= reach)
    ys, xs = np.nonzero(near)
    owner_link = link_at[ry[ys, xs], rx[ys, xs]]
    verge = np.zeros((H, W), bool)
    pave = np.zeros((H, W), bool)
    bays = np.zeros((H, W), bool)
    bands = outlines(paths, links, chosen, hand)
    for i in np.unique(owner_link):
        if i < 0 or i not in bands:
            continue
        sel = owner_link == i
        tx, ty = xs[sel] + 0.5, ys[sel] + 0.5
        line = shapely.LineString(paths[i])
        if line.length == 0:
            continue
        pts = shapely.points(tx, ty)
        s = shapely.line_locate_point(line, pts)
        # Along the link: the band whose outline holds the square's middle.
        in_b, in_v, in_p = (shapely.contains_xy(o, tx, ty) for o in bands[i])
        bb, vv, pp = in_b, in_v & ~in_b, in_p & ~in_v
        # Near its ends: by distance, on the side of the link it lies.
        a = shapely.line_interpolate_point(line, np.clip(s - 0.5, 0, line.length))
        b = shapely.line_interpolate_point(line, np.clip(s + 0.5, 0, line.length))
        ax, ay = shapely.get_x(a), shapely.get_y(a)
        bx, by = shapely.get_x(b), shapely.get_y(b)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        cross = (bx - ax) * (ty - my) - (by - ay) * (tx - mx)
        left = cross * hand > 0
        d = dist[ys[sel], xs[sel]]
        end = ~in_p & ((s < END_TILES) | (s > line.length - END_TILES))
        for k, on in ((0, left & end), (1, ~left & end)):
            v, p, w = chosen[i][k]
            bb |= on & (d < w + 0.5)
            vv |= on & (d >= w + 0.5) & (d < w + v + 0.5)
            pp |= on & (d >= w + v + 0.5) & (d < w + v + p + 0.5)
        bays[ys[sel][bb], xs[sel][bb]] = True
        verge[ys[sel][vv], xs[sel][vv]] = True
        pave[ys[sel][pp], xs[sel][pp]] = True
    return verge, pave, bays
