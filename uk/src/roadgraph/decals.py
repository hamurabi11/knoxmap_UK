"""The UK road markings of one and two tiles on a road-graph map (user,
2026-10-14; the art: asset_creator streetscape.py road_markings_uk_02,
drawn by marking_faces.py), laid by markings.lay with the lines: loose tiles
by name on FloorGrime (the zebra's bands on FloorOverlay, where vanilla's
bars were), each marking whole or not at all, only on its own road's squares
and only where the road runs along the grid (no pieces at 45 degrees).

Where each goes, from OSM and OS Open Roads, else the road graph's own shape:
- Give-way triangles: on each arm that gives way and got its line
  (markings.give_ways), TRI_BACK tiles back from the line, in the middle of
  the approach lane. At a roundabout (OSM junction=roundabout, or OS
  form_of_way Roundabout) the circulating arms are the road through and
  every other arm gives way (markings.give_ways).
- Zebra crossings: white bands across the road, half a tile each with the
  road between (user, 2026-10-14: white only), where the crossing is one
  clean rectangle kerb to kerb (moved up to ZEBRA_SHIFT along the road to
  find one, else not drawn), the broken give-way line a tile beyond each side, and
  ZIGZAG tiles of zig-zags along both kerbs of both approaches (no double
  yellows anywhere along it: the zig-zags are the restriction).
- Speed roundels: 30 or 40 in the lane, ROUNDEL_IN tiles past a limit's
  signs (furniture.limit_signs) - repeated on the road, as the 1990s began to.
- Bends (out of town, A, B and minor roads, BEND_CLASSES): a bend where the
  real road (OSM's line: the snapped one is drawn in vanilla's 16
  directions, its turns spread out) turns BEND_ANGLE or more within
  BEND_SPAN tiles either side, away from junctions; SLOW in each approach lane BEND_SLOW tiles before it (on a
  one-way road only the way its traffic comes), and on
  roads with a centre line a deflection arrow before double white lines
  through it (markings.centre_lines lays the double whites, solid within
  BEND_REACH).
- Lane arrows: on one-way streets (OSM oneway=yes or -1, a single way so its
  direction is the traffic's - flow(); not a merged dual carriageway), an ahead arrow
  ARROW_IN tiles in from the start, and before the junction at the end one
  a lane (OSM lanes): by OSM turn:lanes where mapped, else by the exits
  there - the left lane ahead-or-left (or left, with nothing ahead), the
  right ahead-or-right. A two-way road's turn:lanes:forward / :backward
  (its half) likewise before the junction it leads to.
- Mini roundabouts (OSM highway=mini_roundabout, at its junction): the white
  disc, 2 x 2, on the junction's middle.
- Yellow boxes: across the road outside fire and ambulance stations (OSM,
  a point or a building's middle, its nearest road within BOX_REACH), and at
  any point tagged as a box junction: BOX_DEEP tiles along the road, kerb to
  kerb, a whole rectangle (moved along the road off a junction's mouth, or on
  the next road near).
"""
from __future__ import annotations

import math

import shapely

from .raster import vertices

SHEET = "road_markings_uk_02_"
LAYER = "FloorGrime"
TRI_BACK = 2                 # tiles behind the give-way line, the triangle's front
ZIGZAG = 8                   # tiles of zig-zags, each approach
ROUNDEL_IN = 4               # tiles past a limit's signs, its roundel's front
BEND_CLASSES = ("a_road", "b_road", "minor")
BEND_ANGLE = 25              # degrees the real road turns, at least, for a bend...
BEND_SPAN = 20               # ...within this many tiles either side (a curve of 90 m radius or tighter)
BEND_APART = 30              # tiles between bends, at least (one bend, not its samples)
BEND_JUNCTION = 20           # tiles from a junction: a turn there is the junction's
BEND_SLOW = 24               # tiles before the bend, SLOW's front (then the deflection arrow, then the double whites)
BEND_REACH = 10              # tiles each side of the bend, double white lines
DEFLECT_BACK = 4             # tiles before the double whites, the deflection arrow's front
ARROW_IN = 8                 # tiles from a one-way street's start, its ahead arrow's front
ARROW_BEFORE = 10            # tiles before the junction at the end, its lane arrows' front
ARROW_MIN = 30               # tiles long, at least, for arrows
BOX_REACH = 40               # tiles from a station to the road its box is on
BOX_DEEP = 4


# --- road_markings_uk_02's slots (as streetscape.markings_sheet lays them out) ------------------
READ = [("give_way_triangle", (1, 2)), ("limit_30", (1, 2)), ("limit_40", (1, 2))] + \
    [(f"arrow_{k}", (1, 2)) for k in ("ahead", "left", "right", "ahead_left", "ahead_right", "deflect")] + \
    [("slow", (2, 2))]


def _pieces(heading, size):
    w, l = size
    mw, mh = (w, l) if heading in (None, "N", "S") else (l, w)
    return [(gx, gy) for gy in range(mh) for gx in range(mw)]


def _slots() -> dict:
    names = []
    for name, size in READ:
        for h in "NESW":
            names += [f"{name}_{h}_{gx}{gy}" for gx, gy in _pieces(h, size)]
    names += ["zebra_NS", "zebra_EW"] + [f"zigzag_{e}" for e in "WNES"]
    names += [f"box_{k}" for k in ("fill", "W", "N", "E", "S", "NW", "NE", "SE", "SW")]
    names += ["double_white_W", "double_white_N"] + [f"mini_disc_{gx}{gy}" for gy in (0, 1) for gx in (0, 1)]
    return {n: k for k, n in enumerate(names)}


SLOT = _slots()
SIZE = dict(READ)
COMPASS = {(0, -1): "N", (1, 0): "E", (0, 1): "S", (-1, 0): "W"}


def tile(name: str) -> str:
    return SHEET + str(SLOT[name])


def _grid(h, k=3.0):
    """h as a grid step, if it runs along the grid (one component k times the other)."""
    a, b = abs(h[0]), abs(h[1])
    if a >= k * b:
        return (1 if h[0] > 0 else -1, 0)
    if b >= k * a:
        return (0, 1 if h[1] > 0 else -1)
    return None


def _count(m, kind):
    m.count[kind] = m.count.get(kind, 0) + 1


def _crossing(m, x, y) -> bool:
    """Whether a crossing's paint is on the square (a zebra's bands, a stop line,
    studs): a marking laid there would lie over it (a give-way triangle in a
    zebra beside a junction, 2026-10-14)."""
    from .markings import BAR_EW, BAR_NS, LINES
    t = m.tiles.get((x, y, "FloorOverlay"), "")
    return t in (tile("zebra_NS"), tile("zebra_EW"), LINES + str(BAR_NS), LINES + str(BAR_EW))


def _put_all(m, cells, kind, layer=LAYER):
    """cells: [(x, y, tile)] - all laid, or none (one already holds something,
    or a crossing's paint is there)."""
    if not cells or any((x, y, layer) in m.tiles or _crossing(m, x, y) for x, y, _t in cells):
        _count(m, f"{kind} (no room)")
        return False
    for x, y, t in cells:
        m.tiles[(x, y, layer)] = t
    _count(m, kind)
    return True


def lane(owner, i, pt, h, hand):
    """The squares of link i across the road at pt that lie left of its
    centre for a driver heading h (their lane), leftmost first."""
    from .markings import _across, _left
    row = [t for t in _across(owner, i, pt, h) if _left(h, t, pt, hand)]
    return sorted(row, key=lambda t: -((h[0] * (t[1] + 0.5 - pt[1]) - h[1] * (t[0] + 0.5 - pt[0])) * hand))


def across(owner, i, pt, h, hand):
    """All the squares of link i across the road at pt, leftmost first for heading h."""
    from .markings import _across
    row = _across(owner, i, pt, h)
    return sorted(row, key=lambda t: -((h[0] * (t[1] + 0.5 - pt[1]) - h[1] * (t[0] + 0.5 - pt[0])) * hand))


def read(m, owner, i, name, h, front, kind):
    """Marking `name` for a driver heading h (a grid step), its front row the
    squares `front` (as many as it is wide), running back from them."""
    w, l = SIZE[name]
    if len(front) != w:
        _count(m, f"{kind} (lane too narrow)")
        return False
    sq = [(x - h[0] * k, y - h[1] * k) for x, y in front for k in range(l)]
    H, W = owner.shape
    if not all(0 <= y < H and 0 <= x < W and owner[y, x] == i + 1 for x, y in sq):
        _count(m, f"{kind} (off its road)")
        return False
    ox, oy = min(x for x, _y in sq), min(y for _x, y in sq)
    head = COMPASS[h]
    names = [f"{name}_{head}_{x - ox}{y - oy}" for x, y in sq]
    if not all(n in SLOT for n in names):                  # its front squares not side by side (a gap in the lane)
        _count(m, f"{kind} (lane broken)")
        return False
    return _put_all(m, [(x, y, tile(n)) for (x, y), n in zip(sq, names)], kind)


def flow(link, path):
    """Which way traffic runs along path: 1 its way, -1 against it, 0 both ways,
    None one-way but its way not known (a link of several OSM ways may have
    been joined against one of them; a merged dual carriageway is two-way)."""
    ow = link.tags.get("oneway")
    if ow not in ("yes", "1", "-1") or link.tags.get("dual_carriageway"):
        return 0
    if len(link.ways) != 1:
        return None
    vs = vertices(path)
    along = math.dist(vs[0], link.pts[0]) <= math.dist(vs[0], link.pts[-1])
    f = 1 if along else -1
    return -f if ow == "-1" else f


def _middle(squares, n=1):
    k = max(0, (len(squares) - n) // 2)
    return squares[k:k + n] if len(squares) >= n else []


# --- give-way triangles ----------------------------------------------------------------------------
def triangles(m, owner, drawn, hand):
    """drawn: (link, the line's row point, the arm's way out from the junction) of each give-way line."""
    for i, pt, ax in drawn:
        towards = (-ax[0], -ax[1])
        front_pt = (pt[0] + ax[0] * TRI_BACK, pt[1] + ax[1] * TRI_BACK)
        read(m, owner, i, "give_way_triangle", towards, _middle(lane(owner, i, front_pt, towards, hand)),
             "give-way triangle")


# --- zebras --------------------------------------------------------------------------------------
ZEBRA_SHIFT = 3              # tiles along the road a zebra may move to find a clean place


def _zebra_place(owner, i, q, ax, depth):
    """The crossing's middle (q moved up to ZEBRA_SHIFT along the road) where
    its bands and both give-way rows are one and the same span kerb to kerb,
    the road's edge a kerb on both sides; None if there is none."""
    from .markings import _across
    H, W = owner.shape
    c = (1, 0) if ax[0] == 0 else (0, 1)
    lo, hi = -(depth // 2), depth - depth // 2
    for shift in (0, 1, -1, 2, -2, 3, -3)[:2 * ZEBRA_SHIFT + 1]:
        qq = (q[0] + ax[0] * shift, q[1] + ax[1] * shift)
        spans = set()
        for k in range(lo - 1, hi + 1):
            row = sorted(_across(owner, i, (qq[0] + ax[0] * k, qq[1] + ax[1] * k), ax),
                         key=lambda t: t[0] * c[0] + t[1] * c[1])
            if len(row) < 2:
                spans.add(None)
                break
            a, b = row[0][0] * c[0] + row[0][1] * c[1], row[-1][0] * c[0] + row[-1][1] * c[1]
            kerbs = all(not (0 <= t[1] + c[1] * g < H and 0 <= t[0] + c[0] * g < W)
                        or owner[t[1] + c[1] * g, t[0] + c[0] * g] == 0 for t, g in ((row[0], -1), (row[-1], 1)))
            spans.add((a, b) if len(row) == b - a + 1 and kerbs else None)
        if len(spans) == 1 and None not in spans:
            return qq
    return None


def zebra(m, owner, i, q, ax, depth):
    """The bands across the crossing, its give-way lines, its zig-zags - where
    the crossing is one clean rectangle (_zebra_place: user, 2026-10-14, the
    zebra by a junction's mouth came out ragged). Returns its middle, or None."""
    from .markings import DOUBLE_YELLOW, EDGE, FRESH_WHITE, LINES, _across
    H, W = owner.shape
    q = _zebra_place(owner, i, q, ax, depth)
    if q is None:
        _count(m, "zebra at a ragged place (not drawn)")
        return None
    band = tile("zebra_NS" if ax[0] == 0 else "zebra_EW")
    c = (1, 0) if ax[0] == 0 else (0, 1)                      # across the road
    lo, hi = -(depth // 2), depth - depth // 2

    def row_at(k):
        return sorted(_across(owner, i, (q[0] + ax[0] * k, q[1] + ax[1] * k), ax),
                      key=lambda t: t[0] * c[0] + t[1] * c[1])

    for k in range(lo, hi):
        for x, y in row_at(k):
            m.tiles.pop((x, y, "FloorOverlay"), None)          # (a centre line's dash under it)
            m.put(x, y, "FloorOverlay", band, "zebra band")
    for k in range(lo - 1 - ZIGZAG, hi + ZIGZAG + 1):          # the zig-zags are the restriction: no yellows in the zone
        for x, y in row_at(k):
            if m.tiles.get((x, y, "FloorGrime2"), "").startswith(DOUBLE_YELLOW):
                del m.tiles[(x, y, "FloorGrime2")]
    for s in (1, -1):
        k = (hi if s > 0 else lo - 1)
        side = COMPASS[(-ax[0] * s, -ax[1] * s)]               # the edge towards the crossing
        for x, y in row_at(k):
            if (x * c[0] + y * c[1]) % 2 == 0:                 # the same dashes both sides of the crossing
                m.tiles.pop((x, y, "FloorOverlay"), None)
                m.put(x, y, "FloorOverlay", LINES + str(FRESH_WHITE + EDGE[side]), "zebra give-way line")
        for end in (0, -1):                                    # each kerb's run, from the crossing out
            for n in range(1, ZIGZAG + 1):
                row = row_at(k + s * n)
                if len(row) < 2:
                    break
                x, y = row[end]
                sgn = -1 if end == 0 else 1
                bx, by = x + c[0] * sgn, y + c[1] * sgn
                if (0 <= by < H and 0 <= bx < W) and owner[by, bx] != 0:
                    break                                       # the kerb turns away (a mouth): the run ends
                if (x, y, LAYER) in m.tiles:
                    break
                m.tiles[(x, y, LAYER)] = tile(f"zigzag_{COMPASS[(c[0] * sgn, c[1] * sgn)]}")
                _count(m, "zig-zag")
    return q


# --- speed roundels ------------------------------------------------------------------------------
def roundels(m, owner, paths, limits, hand):
    """limits: (link entered, the junction or bend it is entered at, 'speed_30' / 'speed_40' / ...)."""
    from .furniture import SIGN_IN, _along
    for i, corner, sign in limits:
        if sign not in ("speed_30", "speed_40"):
            continue
        pt, h = _along(paths[i], corner, SIGN_IN + ROUNDEL_IN)
        g = _grid(h)
        if g is None:
            _count(m, "speed roundel on a slant (not drawn)")
            continue
        read(m, owner, i, "limit_" + sign[-2:], g, _middle(lane(owner, i, pt, g, hand)), "speed roundel")


# --- bends ---------------------------------------------------------------------------------------
def _heading(line, s):
    a = line.interpolate(max(0.0, s - 0.5))
    b = line.interpolate(min(line.length, s + 0.5))
    n = math.hypot(b.x - a.x, b.y - a.y) or 1.0
    return ((b.x - a.x) / n, (b.y - a.y) / n)


def bends(paths, links, arms, town) -> dict:
    """{link: [distances along its path of each sharp bend]}: out of town,
    BEND_CLASSES, the real road turning BEND_ANGLE within BEND_SPAN either
    side, clear of junctions; the sharpest point of each, carried to the
    snapped path by its share along the link."""
    H, W = town.shape
    out = {}
    for i, path in paths.items():
        if links[i].cls not in BEND_CLASSES:
            continue
        line = shapely.LineString(links[i].pts)
        L = line.length
        drawn = shapely.LineString(path).length
        vs = vertices(path)
        ends = [0.0 if len(arms.get(tuple(vs[0]), ())) >= 3 else None,
                L if len(arms.get(tuple(vs[-1]), ())) >= 3 else None]
        found = []
        s = BEND_SPAN
        while s <= L - BEND_SPAN:
            p = line.interpolate(s)
            x, y = int(p.x), int(p.y)
            near_junction = any(e is not None and abs(s - e) < BEND_JUNCTION for e in ends)
            if 0 <= y < H and 0 <= x < W and not town[y, x] and not near_junction:
                a, b = _heading(line, s - BEND_SPAN), _heading(line, s + BEND_SPAN)
                turn = math.degrees(math.acos(max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))))
                if turn >= BEND_ANGLE:
                    if found and s - found[-1][0] < BEND_APART:
                        if turn > found[-1][1]:
                            found[-1] = (s, turn)
                    else:
                        found.append((s, turn))
            s += 1.0
        if found:
            out[i] = [s / L * drawn for s, _t in found]
    return out


def bend_marks(m, owner, paths, links, bent, hand):
    """SLOW before each bend in both approach lanes; a deflection arrow where the road has a centre line."""
    from .markings import has_centre
    for i, at in bent.items():
        line = shapely.LineString(paths[i])
        fl = flow(links[i], paths[i])
        if fl is None:
            _count(m, "bend on a one-way road of unknown way (not marked)")
            continue
        for s_b in at:
            for sgn in (1, -1):                                # coming from the path's start, then its end
                if fl and sgn != fl:
                    continue                                   # (a one-way road: no traffic from that end)
                for name, back in (("slow", BEND_SLOW), ("arrow_deflect", BEND_REACH + DEFLECT_BACK)):
                    if name == "arrow_deflect" and not has_centre(links[i]):
                        continue
                    s = s_b - sgn * back
                    if not 0 < s < line.length:
                        continue
                    h = _heading(line, s)
                    g = _grid((h[0] * sgn, h[1] * sgn))
                    if g is None:
                        _count(m, f"{name} on a slant (not drawn)")
                        continue
                    p = line.interpolate(s)
                    ln = lane(owner, i, (p.x, p.y), g, hand)
                    front = _middle(ln, 2) if name == "slow" else ln[-1:]
                    read(m, owner, i, name, g, front, "SLOW before a bend" if name == "slow" else "deflection arrow")
        _count(m, "bends")


# --- lane arrows ---------------------------------------------------------------------------------
TURN_ARROW = {frozenset({"through"}): "ahead", frozenset({"left"}): "left", frozenset({"right"}): "right",
              frozenset({"through", "left"}): "ahead_left", frozenset({"through", "right"}): "ahead_right",
              frozenset({"slight_left"}): "left", frozenset({"slight_right"}): "right"}


def _is_roundabout(link) -> bool:
    return link.tags.get("junction") in ("roundabout", "circular") or \
        (link.tags.get("os") or {}).get("form_of_way") == "Roundabout"


def _exits(arms, corner, i, h, hand):
    """Which ways the roads out of the junction at corner go, for a driver arriving heading h."""
    left = (-hand * h[1], hand * h[0])
    ways = set()
    for j, d, _hw in arms.get(corner, ()):
        if j == i:
            continue
        if d[0] * h[0] + d[1] * h[1] > 0.6:
            ways.add("ahead")
        elif d[0] * left[0] + d[1] * left[1] > 0.4:
            ways.add("left")
        elif d[0] * left[0] + d[1] * left[1] < -0.4:
            ways.add("right")
    return ways


def _by_exits(n, ways):
    if n == 1:                                       # one lane: an arrow only where it has one way to go
        if "ahead" in ways or not ways:
            return ["ahead"]
        return [next(iter(ways))] if len(ways) == 1 else [None]
    first = "ahead_left" if {"ahead", "left"} <= ways else "left" if "left" in ways else "ahead"
    last = "ahead_right" if {"ahead", "right"} <= ways else "right" if "right" in ways else "ahead"
    return [first] + ["ahead"] * (n - 2) + [last]


def _lanes_split(squares, n):
    """squares (leftmost first) split into n lanes; each lane's middle square."""
    if n <= 0 or len(squares) < n:
        return []
    size = len(squares) / n
    return [squares[int(size * k + size / 2)] for k in range(n)]


def arrows(m, owner, paths, links, arms, hand):
    for i, path in paths.items():
        l = links[i]
        t = l.tags
        if l.cls in ("service", "track", "path", "footway") or _is_roundabout(l) or len(l.ways) != 1 \
                or t.get("dual_carriageway"):
            continue
        fl = flow(l, path)
        if fl is None:
            continue
        oneway = fl != 0
        jobs = []                                      # (forward?, lanes, turn:lanes, whole road?)
        if oneway:
            jobs.append((True, t.get("lanes"), t.get("turn:lanes") or t.get("turn:lanes:forward"), True))
        else:
            for fwd, key in ((True, "forward"), (False, "backward")):
                if t.get(f"turn:lanes:{key}"):
                    jobs.append((fwd, t.get(f"lanes:{key}"), t.get(f"turn:lanes:{key}"), False))
        if not jobs:
            continue
        vs = vertices(path)
        line = shapely.LineString(path)
        if line.length < ARROW_MIN:
            continue
        along = math.dist(vs[0], l.pts[0]) <= math.dist(vs[0], l.pts[-1])   # the path runs the way's way
        for fwd, lanes_tag, turns, whole in jobs:
            forward = fl == 1 if oneway else along == fwd   # traffic runs from the path's start to its end
            end = tuple(vs[-1] if forward else vs[0])
            turn_list = [frozenset(p.split(";")) for p in turns.split("|")] if turns else None
            try:
                n = int(lanes_tag) if lanes_tag else (len(turn_list) if turn_list else 1)
            except ValueError:
                n = 1
            n = max(1, min(3, n))
            spots = [("ahead_start", ARROW_IN)] if oneway else []
            spots.append(("end", line.length - ARROW_BEFORE))
            for what, s_ in spots:
                s = s_ if forward else line.length - s_
                hh = _heading(line, s)
                g = _grid(hh if forward else (-hh[0], -hh[1]))
                if g is None:
                    _count(m, "lane arrow on a slant (not drawn)")
                    continue
                p = line.interpolate(s)
                row = across(owner, i, (p.x, p.y), g, hand) if whole else lane(owner, i, (p.x, p.y), g, hand)
                if what == "ahead_start":
                    kinds = ["ahead"] * n
                elif turn_list and len(turn_list) == n:
                    kinds = [TURN_ARROW.get(frozenset(x for x in tl if x != "none"), "ahead") for tl in turn_list]
                else:
                    kinds = _by_exits(n, _exits(arms, end, i, g, hand))
                for sq, kind in zip(_lanes_split(row, n), kinds):
                    if kind is not None:
                        read(m, owner, i, f"arrow_{kind}", g, [sq], "lane arrow")


# --- mini roundabouts, yellow boxes --------------------------------------------------------------
def mini_discs(m, owner, minis):
    H, W = owner.shape
    for cx, cy in minis:
        ox, oy = int(round(cx)) - 1, int(round(cy)) - 1
        cells = [(ox + gx, oy + gy, tile(f"mini_disc_{gx}{gy}")) for gy in (0, 1) for gx in (0, 1)]
        if not all(0 <= y < H and 0 <= x < W and owner[y, x] > 0 for x, y, _t in cells):
            _count(m, "mini roundabout disc (off the road)")
            continue
        _put_all(m, cells, "mini roundabout disc")


BOXES = ("N", "S", "W", "E", "NW", "NE", "SW", "SE")      # box_<edges> pieces besides the fill
BOX_AT = (("amenity", "fire_station"), ("emergency", "ambulance_station"), ("markings", "box_junction"),
          ("box_junction", "yes"))


def _box_at(owner, i, line, s, g):
    """The squares of a box across link i at s along its path - a whole
    rectangle, BOX_DEEP rows each the same span kerb to kerb - or None."""
    from .markings import _across
    q = line.interpolate(s)
    rows = [sorted(_across(owner, i, (q.x + g[0] * k, q.y + g[1] * k), g))
            for k in range(-(BOX_DEEP // 2), BOX_DEEP - BOX_DEEP // 2)]
    c = 0 if g[0] == 0 else 1                                   # the coordinate across the road
    spans = {(r[0][c], r[-1][c], len(r)) if r else None for r in rows}
    if len(spans) != 1 or None in spans:
        return None
    a, b, n = next(iter(spans))
    if n != b - a + 1 or n < 2:
        return None
    return {t for r in rows for t in r}


def boxes(m, owner, nodes, proj, paths):
    for lonlat, tags in nodes:
        if not any(tags.get(k) == v for k, v in BOX_AT):
            continue
        p = shapely.Point(proj.to_px(lonlat[1], lonlat[0]))
        near = sorted((shapely.LineString(path).distance(p), i) for i, path in paths.items())
        sq = None
        for d, i in near:
            if d > BOX_REACH:
                break
            line = shapely.LineString(paths[i])
            s0 = line.project(p)
            for shift in (0, 2, -2, 4, -4, 6, -6):              # along the road, off a junction's mouth
                s = s0 + shift
                if not BOX_DEEP <= s <= line.length - BOX_DEEP:
                    continue
                g = _grid(_heading(line, s))
                if g is not None:
                    sq = _box_at(owner, i, line, s, g)
                if sq:
                    break
            if sq:
                break
        if not sq:
            _count(m, "yellow box (no straight road whole across near)")
            continue
        cells = []
        for x, y in sorted(sq):
            edges = "".join(e for e, (dx, dy) in (("N", (0, -1)), ("S", (0, 1)), ("W", (-1, 0)), ("E", (1, 0)))
                            if (x + dx, y + dy) not in sq)
            key = "fill" if not edges else edges if edges in BOXES else edges[:2] if edges[:2] in BOXES else edges[0]
            cells.append((x, y, tile(f"box_{key}")))
        _put_all(m, cells, "yellow box")


def lay(m, owner, paths, links, arms, minis, drawn, limits, bent, nodes, proj, hand):
    """Every marking of road_markings_uk_02 but the zebras' (markings.crossings) and the double whites (centre_lines)."""
    triangles(m, owner, drawn, hand)
    roundels(m, owner, paths, limits, hand)
    bend_marks(m, owner, paths, links, bent, hand)
    arrows(m, owner, paths, links, arms, hand)
    mini_discs(m, owner, minis)
    boxes(m, owner, nodes, proj, paths)
