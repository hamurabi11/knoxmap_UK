"""UK street furniture on a road-graph map (PLAN E; the art is
asset_creator's, 1990s: street_furniture_uk_01/02, road_signs_uk_01), laid
with the markings (markings.lay) as loose tiles by name on the Furniture
layers: kerbs.py stands them only off the road, Build lays them.

Where each goes, from OSM where it maps it, else from the road graph's own
data, never by guesswork about a zone:
- Signal heads at signal crossings (markings.crossings: by its stop lines,
  facing the traffic) and at junction signals (OSM highway=traffic_signals
  not on a crossing): one on the left kerb of each arm, the through road's
  arms on green, the side roads' on red.
- Belisha beacons (lit) on the pavement at both ends of each zebra.
- Give-way and stop signs at OSM highway=give_way / stop points, on the left
  kerb of the approach to the junction the point stands by, facing it.
- Speed limit signs where OSM's maxspeed changes from one link to the next
  (a road's two links at a bend, or a junction's through pair): on both
  kerbs of the road entered, facing the traffic entering it. 60 and 70 mph
  are the national limit; 20 mph (zones were all but unknown in the 1990s)
  counts as 30.
- The town's name sign where a road enters the 30 limit from a higher one,
  on the left verge a little past the 30 signs (TOWNS: the area's town, area_conf.town). A
  two-tile thing (a town sign, a bench) one of whose squares kerbs.py turns
  to road goes whole (partner()): half a sign is worse than none.
- No through road signs at the mouth of a cul-de-sac (a local or minor link
  from a junction to a dead end, away from the map's edge).
- Post boxes, benches and bollards at OSM amenity=post_box / bench and
  barrier=bollard points within POINT_REACH of a
  road: carried with the road (its snapped line), as far out from it as the
  point is from the real road, on the nearest pavement or verge square
  there (a bench in a churchyard comes out to the pavement by it); a post
  box with its front away from the road, a bench facing the
  street (two squares along it), a bollard cast iron in town and concrete
  out of it; a bollard on the carriageway itself (a road closed to cars) is
  left out. A pillar box or kiosk faces S or E, the faces the game shows
  (seen()): away from the road, or along the pavement where that is N or W.
- Telephone kiosks at OSM's telephones and where a 1990s town had them: by
  post offices, at stations, at places' middles, by pubs (kiosk_sites,
  KIOSK_APART apart), on the pavement nearest the site (KIOSK_REACH), its
  door away from the road.
- Telegraph poles along the country roads (POLE_CLASSES, out of town):
  one every POLE_EVERY tiles on the road's left, a storey and a half tall
  (its foot on the ground, its top with the crossarm a storey up), the two
  wires a storey up between neighbouring poles in line along the grid.
- A road works pit on the pavement: ROADWORKS a map, at the middle of the
  longest straight local streets with a pavement two wide on their south or
  east side - two squares fenced round with red and white barriers (lamps
  at the corners), a road works sign before it facing the near lane.
- The town centre's bus stops and shelters, bins, cabinets, cycle stands,
  pub signs and guard railing: town_centre.py.
Signs stand on the pavement or verge where the roadside has one there, else
(out of town, where roadside.py draws none) on the first square off the road.
Each square takes one thing; what could not be stood is counted.
No entry signs wait: a one-way link's direction is not kept reliably.
"""
from __future__ import annotations

import math
import os

import shapely

from . import town_centre
from .raster import vertices

F1, F2, RS = "street_furniture_uk_01_", "street_furniture_uk_02_", "road_signs_uk_01_"
F3 = "street_furniture_uk_03_"               # the town centre's (town_centre.py): a shelter's pieces as a bench's
ORDER = "SENW"                               # each sheet's facings, in order
SIGNAL_STATES = ("red", "red_amber", "green", "amber")
ROAD_SIGNS = ("speed_30", "speed_40", "national", "no_entry", "give_way", "stop", "road_works", "no_through_road")
TOWNS = ("odiham", "alton")                  # road_signs_uk_01's place signs, in order
# A bench's pieces as vanilla's park bench numbers them: (facing, its square in the bench) -> slot.
BENCH = {("S", (0, 0)): 4, ("S", (1, 0)): 5, ("E", (0, 1)): 6, ("E", (0, 0)): 7,
         ("N", (0, 0)): 8, ("N", (1, 0)): 9, ("W", (0, 1)): 10, ("W", (0, 0)): 11}
LAYER = "Furniture"
LAYER2 = "Furniture2"                       # a second thing on a square: a barrier's other edge
SIGN_IN = 4                                  # tiles into the road entered, a limit sign (past a rounded corner)
PLACE_IN = 8                                 # ...and its town sign (clear of the corner kerbs.py rounds off)
EDGE_CLEAR = 10                              # tiles from the map's edge: no signs (a road's limit there is not its own)
CUL_DE_SAC_MIN = 20                          # tiles long at least, for a no through road sign
JUNCTION_REACH = 18                          # tiles from a junction, its signals' OSM point
CROSSING_SIGNALS = 10                        # tiles from a signal crossing: a traffic_signals point is its own (its stop line)
ROADWORKS = 1
CS = "countryside_uk_01_"                    # asset_creator streetscape.py: the pole's foot 12, top 13 (E-W) / 14 (N-S), wires 15 / 16
POLE_EVERY = 40                              # tiles between telegraph poles (the 40 m of a BT line)
POLE_CLASSES = ("b_road", "minor", "local")
BOLLARD_ON_ROAD = 1.0                        # tiles from the real road's line: a bollard on the carriageway
POINT_REACH = 15                             # tiles from a road, an OSM point of furniture, at most
POINT_SEARCH = 4                             # tiles round where it falls, the square it may take
KIOSK_APART = 200                            # tiles (metres) between telephone kiosks, at least
KIOSK_REACH = 40                             # tiles from its site, the pavement a kiosk may stand on
KIOSK_AT = (("amenity", ("post_office",)), ("railway", ("station",)),
            ("place", ("town", "village", "hamlet", "suburb", "neighbourhood", "quarter")),
            ("amenity", ("pub",)))           # where a kiosk stood, first first (kiosk_sites)


def pillar_box(f):
    return F1 + str(ORDER.index(f))


def kiosk(k6, f):
    return F1 + str((4 if k6 else 8) + ORDER.index(f))


BEACON_LIT = F1 + "13"


def signal(state, f):
    return F1 + str(16 + 4 * SIGNAL_STATES.index(state) + ORDER.index(f))


def road_sign(name, f):
    return RS + str(4 * ROAD_SIGNS.index(name) + ORDER.index(f))


def place_sign(town, f, piece):
    pieces = ((0, 0), (1, 0)) if f in "SN" else ((0, 0), (0, 1))
    return RS + str(32 + 8 * TOWNS.index(town) + 2 * ORDER.index(f) + pieces.index(piece))


def partner(x, y, tile):
    """The other piece of a two-tile thing at (x, y): (x, y, tile) of it, or None for a one-tile thing."""
    sheet, idx = tile.rsplit("_", 1)
    idx = int(idx)
    if sheet + "_" == RS and idx >= 32:
        k = idx - 32
        f, i = ORDER[(k % 8) // 2], k % 2
        pieces = ((0, 0), (1, 0)) if f in "SN" else ((0, 0), (0, 1))
        p, q = pieces[i], pieces[1 - i]
        return x + q[0] - p[0], y + q[1] - p[1], RS + str(idx - i + (1 - i))
    if sheet + "_" in (F2, F3) and 4 <= idx < 12:            # a bench, or a bus shelter: the same pieces
        (f, p), = [key for key, v in BENCH.items() if v == idx]
        (q, other), = [(key[1], v) for key, v in BENCH.items() if key[0] == f and v != idx]
        return x + q[0] - p[0], y + q[1] - p[1], sheet + "_" + str(other)
    return None


def compass(v) -> str:
    """The grid direction nearest v (x east, y south)."""
    if abs(v[0]) >= abs(v[1]):
        return "E" if v[0] > 0 else "W"
    return "S" if v[1] > 0 else "N"


STEP = {"E": (1, 0), "W": (-1, 0), "S": (0, 1), "N": (0, -1)}


def squarely(v, k=2.0) -> bool:
    """Whether v runs along the grid (one component at least k times the other)."""
    a, b = abs(v[0]), abs(v[1])
    return max(a, b) >= k * min(a, b)


def speed_sign(maxspeed) -> str | None:
    try:
        mph = int(str(maxspeed).split()[0])
    except (ValueError, IndexError):
        return None
    if mph >= 60:
        return "national"
    if mph == 40:
        return "speed_40"
    if mph in (20, 30):
        return "speed_30"
    return None


class Stand:
    """Squares furniture may stand on: off the road, on the pavement or verge, not yet taken."""

    def __init__(self, m, owner, ok, hand):
        self.m, self.owner, self.ok, self.hand = m, owner, ok, hand
        self.H, self.W = owner.shape
        self.taken: set = set()

    def inside(self, x, y, margin=0):
        return margin <= x < self.W - margin and margin <= y < self.H - margin

    def free(self, x, y, layer=LAYER, loose=False):
        return (self.inside(x, y) and self.owner[y, x] == 0 and (loose or bool(self.ok[y, x]))
                and (x, y, layer) not in self.m.tiles)

    def put(self, x, y, tile, kind, layer=LAYER):
        self.m.put(x, y, layer, tile, kind)
        self.taken.add((x, y))

    def left(self, h):
        """The unit vector to the left of heading h (on the map, as markings._left)."""
        return (-self.hand * h[1], self.hand * h[0])

    def kerb(self, pt, h, side, reach=12, sign=False):
        """The first square off the road stepping out from pt (on the road)
        square to heading h, to its left (side 1) or right (-1); None unless it
        is pavement or verge and free (a sign: any free square, where the
        roadside has no pavement or verge just there)."""
        lx, ly = self.left(h)
        last = None
        for k in range(1, 4 * reach):
            t = 0.25 * k * side
            x, y = int(math.floor(pt[0] + lx * t)), int(math.floor(pt[1] + ly * t))
            if (x, y) == last:
                continue
            last = (x, y)
            if not self.inside(x, y):
                return None
            if self.owner[y, x] == 0:
                if self.free(x, y):
                    return (x, y)
                bare = sign and not any(self.ok[yy, xx] for xx in range(x - 2, x + 3) for yy in range(y - 2, y + 3)
                                        if self.inside(xx, yy))
                return (x, y) if bare and self.free(x, y, loose=True) else None
        return None

    def near(self, target, reach=None):
        """The free pavement or verge square nearest the point target, within reach."""
        reach = reach or POINT_SEARCH
        tx, ty = target
        best = None
        for y in range(int(ty) - reach, int(ty) + reach + 1):
            for x in range(int(tx) - reach, int(tx) + reach + 1):
                if self.free(x, y):
                    d = math.dist((x + 0.5, y + 0.5), target)
                    if d <= reach and (best is None or d < best[0]):
                        best = (d, (x, y))
        return best[1] if best else None

    def count(self, kind):
        self.m.count[kind] = self.m.count.get(kind, 0) + 1


def _along(path, corner, s):
    """The point s tiles along a path from its end at corner, and the unit heading away from the corner there."""
    line = shapely.LineString(path)
    if tuple(vertices(path)[0]) != tuple(corner):
        line = shapely.LineString(list(line.coords)[::-1])
    s = min(s, line.length * 0.5)
    p = line.interpolate(s)
    a, b = line.interpolate(max(0.0, s - 0.5)), line.interpolate(min(line.length, s + 0.5))
    n = math.hypot(b.x - a.x, b.y - a.y) or 1.0
    return (p.x, p.y), ((b.x - a.x) / n, (b.y - a.y) / n)


def signal_heads_at_junctions(st: Stand, nodes, proj, arms, links, paths, through_pair):
    # A signal crossing's own lights: OSM puts a traffic_signals point at each
    # of its stop lines beside the crossing point (markings.crossings lays
    # their heads); read as a junction's, they stood heads on the arms of the
    # side road next to it (Odiham, which has no junction signals; user, 2026-10-14).
    crossing_at = [proj.to_px(ll[1], ll[0]) for ll, t in nodes
                   if (t.get("crossing") or "") == "traffic_signals"
                   or (t.get("crossing_ref") or "").lower() in ("pelican", "puffin", "toucan", "pegasus")]
    done = set()
    for lonlat, tags in nodes:
        if tags.get("highway") != "traffic_signals" or (tags.get("crossing") or "") == "traffic_signals":
            continue
        x, y = proj.to_px(lonlat[1], lonlat[0])
        if any(math.dist((x, y), c) <= CROSSING_SIGNALS for c in crossing_at):
            continue
        near = min((c for c, a in arms.items() if len(a) >= 3), key=lambda c: math.dist(c, (x, y)), default=None)
        if near is None or math.dist(near, (x, y)) > JUNCTION_REACH or near in done:
            continue
        done.add(near)
        arm = arms[near]
        through = through_pair(arm, links)
        major = max(h for _i, _d, h in arm)
        for k, (i, _d, _h) in enumerate(arm):
            pt, d = _along(paths[i], near, math.ceil(major) + 2)
            heading = (-d[0], -d[1])                     # coming up to the junction
            sq = st.kerb(pt, heading, 1, sign=True)
            if sq is None:
                st.count("junction signal head without a kerb")
                continue
            st.put(*sq, signal("green" if k in through else "red", compass(d)), "junction signal head")
        st.count("junction signals")


def beacons(st: Stand, zebras):
    for q, ax in zebras:
        for side in (1, -1):
            sq = st.kerb(q, ax, side)
            if sq is None:
                st.count("beacon without a pavement")
                continue
            st.put(*sq, BEACON_LIT, "belisha beacon")


def junction_signs(st: Stand, nodes, proj, links, paths, arms, on_graph):
    for lonlat, tags in nodes:
        kind = tags.get("highway")
        if kind not in ("give_way", "stop"):
            continue
        got = on_graph(lonlat, proj, links, paths)
        if got is None:
            st.count(f"{kind} sign off the roads")
            continue
        i, q, d = got
        vs = vertices(paths[i])
        ends = [tuple(vs[0]), tuple(vs[-1])]
        ends = [e for e in ends if len(arms.get(e, ())) >= 3] or ends
        end = min(ends, key=lambda e: math.dist(e, q))
        heading = d if math.dist(end, (q[0] + d[0], q[1] + d[1])) < math.dist(end, q) else (-d[0], -d[1])
        sq = st.kerb(q, heading, 1, sign=True)
        if sq is None:
            st.count(f"{kind} sign without a kerb")
            continue
        st.put(*sq, road_sign(kind, compass((-heading[0], -heading[1]))), f"{kind} sign")


def limit_signs(st: Stand, links, paths, arms, through_pair, town_name, limits=None):
    for corner, arm in arms.items():
        if not st.inside(int(corner[0]), int(corner[1]), EDGE_CLEAR):
            continue
        pairs = [(0, 1)] if len(arm) == 2 else ([through_pair(arm, links)] if len(arm) >= 3 else [])
        for pair in pairs:
            if len(pair) != 2:
                continue
            for a, b in (pair, pair[::-1]):              # driving from arm a's road into arm b's
                ia, ib = arm[a][0], arm[b][0]
                sa, sb = speed_sign(links[ia].tags.get("maxspeed")), speed_sign(links[ib].tags.get("maxspeed"))
                if sa is None or sb is None or sa == sb:
                    continue
                pt, h = _along(paths[ib], corner, SIGN_IN)
                face = compass((-h[0], -h[1]))
                if limits is not None:
                    limits.append((ib, corner, sb))          # its roundel on the road (decals.roundels)
                for side in (1, -1):
                    sq = st.kerb(pt, h, side, sign=True)
                    if sq is None:
                        st.count("limit sign without a kerb")
                        continue
                    st.put(*sq, road_sign(sb, face), "speed limit sign")
                if sb == "speed_30" and town_name in TOWNS:
                    place(st, paths[ib], corner, town_name)


def place(st: Stand, path, corner, town):
    pt, h = _along(path, corner, PLACE_IN)
    if not squarely(h):
        st.count("town sign on a slant (not stood)")
        return
    face = compass((-h[0], -h[1]))
    s1 = st.kerb(pt, h, 1, sign=True)
    if s1 is None:
        st.count("town sign without a verge")
        return
    out = STEP[compass(st.left(h))]
    s2 = (s1[0] + out[0], s1[1] + out[1])
    if not (st.free(*s2) or (not st.ok[s1[1], s1[0]] and st.free(*s2, loose=True))):
        st.count("town sign without room for both posts")
        return
    for sq in (s1, s2):
        other = s2 if sq == s1 else s1
        piece = (int(sq[0] > other[0]), int(sq[1] > other[1]))
        st.put(*sq, place_sign(town, face, piece), "town sign piece")


def cul_de_sacs(st: Stand, links, paths, arms):
    for i, path in paths.items():
        if links[i].cls not in ("local", "minor"):
            continue
        vs = vertices(path)
        if len(vs) < 2:
            continue
        a, b = tuple(vs[0]), tuple(vs[-1])
        for dead, mouth in ((a, b), (b, a)):
            if len(arms.get(dead, ())) == 1 and len(arms.get(mouth, ())) >= 3 \
                    and st.inside(int(dead[0]), int(dead[1]), EDGE_CLEAR) \
                    and shapely.LineString(path).length >= CUL_DE_SAC_MIN:
                major = max(hh for _i, _d, hh in arms[mouth])
                pt, h = _along(path, mouth, math.ceil(major) + 2)
                sq = st.kerb(pt, h, 1, sign=True)
                if sq is None:
                    st.count("no through road sign without a kerb")
                    continue
                st.put(*sq, road_sign("no_through_road", compass((-h[0], -h[1]))), "no through road sign")


def _side_of(link, p) -> int:
    """1 if point p is left of link's real line (markings._left's sense, with hand), else -1; and its distance."""
    line = shapely.LineString(link.pts)
    s = line.project(shapely.Point(p))
    a, b = line.interpolate(max(0.0, s - 0.5)), line.interpolate(min(line.length, s + 0.5))
    near = line.interpolate(s)
    cross = (b.x - a.x) * (p[1] - near.y) - (b.y - a.y) * (p[0] - near.x)
    return cross, math.dist(p, (near.x, near.y))


def _snap(p, links, paths):
    """(link, its snapped point, unit direction there, cross, distance) for
    a map point p within POINT_REACH of a road: where it lies along the real
    road, carried to the snapped one (as markings.on_graph); cross > 0 on
    the real road's left with hand (_side_of)."""
    best = None
    for i in paths:
        d = shapely.LineString(links[i].pts).distance(shapely.Point(p))
        if d <= POINT_REACH and (best is None or d < best[0]):
            best = (d, i)
    if best is None:
        return None
    i = best[1]
    line = shapely.LineString(links[i].pts)
    t = line.project(shapely.Point(p)) / max(line.length, 1e-9)
    snapped = shapely.LineString(paths[i])
    q = snapped.interpolate(t, normalized=True)
    a = snapped.interpolate(max(0.0, t - 0.02), normalized=True)
    b = snapped.interpolate(min(1.0, t + 0.02), normalized=True)
    n = math.hypot(b.x - a.x, b.y - a.y) or 1.0
    cross, off = _side_of(links[i], p)
    return i, (q.x, q.y), ((b.x - a.x) / n, (b.y - a.y) / n), cross, off


def _stand(st: Stand, p, links, paths, what, bollard=False):
    """(square, road direction there, the way away from the road) for a map
    point p: carried with its road, as far out from the snapped road as it is
    from the real one (past the kerb at least), the nearest pavement or verge
    square there; None (counted) where there is none."""
    got = _snap(p, links, paths)
    if got is None:
        st.count(f"{what} away from the roads")
        return None
    i, q, d, cross, off = got
    if bollard and off <= BOLLARD_ON_ROAD:
        st.count("bollard on the carriageway (left out)")
        return None
    side = 1 if cross * st.hand > 0 else -1          # (the map's left: markings._left)
    lx, ly = st.left(d)
    out_by = max(off, links[i].width / 2 + 0.5) * side
    sq = st.near((q[0] + lx * out_by, q[1] + ly * out_by))
    if sq is None:
        st.count(f"{what} without a pavement")
        return None
    return sq, d, compass((lx * side, ly * side))


def seen(f) -> str:
    """A front the game shows: S or E (a north or west face is drawn from behind);
    a box whose front would face north or west turns along the pavement instead."""
    return {"N": "E", "W": "S"}.get(f, f)


def points(st: Stand, nodes, proj, links, paths, on_graph, town):
    for lonlat, tags in nodes:
        what = tags.get("amenity") if tags.get("amenity") in ("post_box", "bench") else \
            ("bollard" if tags.get("barrier") == "bollard" else None)
        if what is None:
            continue
        p = proj.to_px(lonlat[1], lonlat[0])
        if not st.inside(int(p[0]), int(p[1])):
            continue                                     # (off the map)
        got = _stand(st, p, links, paths, what, bollard=what == "bollard")
        if got is None:
            continue
        sq, d, out = got
        if what == "post_box":
            st.put(*sq, pillar_box(seen(out)), "post box")
        elif what == "bollard":
            st.put(*sq, F2 + ("12" if town[sq[1], sq[0]] else "13"), "bollard")
        else:
            bench(st, sq, d, out)


def kiosk_sites(nodes, proj, inside=None) -> list:
    """Where the telephone kiosks stand: (map point, K6 or not, why). OSM's
    own telephones first; then, as a 1990s town had them (OSM keeps few: most
    went after 2000), by each post office, at the railway station, at the
    middle of each place (village, hamlet, suburb...) and by pubs - in that
    order, none within KIOSK_APART of one already chosen. A lookup, not a
    model: the kinds of place that had a kiosk, from the user's brief."""
    chosen = []

    def add(p, k6, why):
        if all(math.dist(p, c[0]) >= KIOSK_APART for c in chosen):
            chosen.append((p, k6, why))

    ranked = [[] for _ in KIOSK_AT]
    for lonlat, tags in nodes:
        p = proj.to_px(lonlat[1], lonlat[0])
        if inside is not None and not inside(int(p[0]), int(p[1])):
            continue                                     # (off the map: it would keep an on-map site away)
        if tags.get("amenity") == "telephone":
            text = " ".join(str(v).lower() for v in tags.values())
            add(p, "k6" in text or (tags.get("colour") or "").lower() == "red", "telephone")
            continue
        for rank, (key, values) in enumerate(KIOSK_AT):
            if tags.get(key) in values:
                ranked[rank].append((p, tags))
                break
    for rank, (key, _values) in enumerate(KIOSK_AT):
        for p, tags in ranked[rank]:
            add(p, None, tags.get(key))
    return chosen


def _pavement_by(st: Stand, p, reach=KIOSK_REACH):
    """The free pavement or verge square nearest p within reach (a building's
    middle lies well back from the road), and the way away from the road
    beside it; None if there is none."""
    sq = st.near(p, reach)
    if sq is None:
        return None
    for k in range(1, 4):                            # the road nearest beside it
        for f, (dx, dy) in STEP.items():
            x, y = sq[0] + dx * k, sq[1] + dy * k
            if st.inside(x, y) and st.owner[y, x] != 0:
                return sq, {"N": "S", "S": "N", "E": "W", "W": "E"}[f]
    return sq, "S"


def kiosks(st: Stand, nodes, proj, links, paths, town):
    """A kiosk at each site (kiosk_sites), on the pavement nearest it: a K6
    where OSM says, or out of town and in villages (where BT kept most of
    them), else BT's KX100."""
    for p, k6, why in kiosk_sites(nodes, proj, st.inside):
        if not st.inside(int(p[0]), int(p[1])):
            continue
        got = _pavement_by(st, p)
        if got is None:
            st.count("telephone kiosk without a pavement near")
            continue
        sq, out = got
        if k6 is None:
            k6 = why in ("village", "hamlet") or not town[sq[1], sq[0]]
        st.put(*sq, kiosk(k6, seen(out)), f"telephone kiosk ({why})")


def bench(st: Stand, sq, d, out, sheet=F2, kind="bench piece"):
    """A bench (or a bus shelter: sheet F3) at the back of the pavement from
    sq, facing the road, two squares along it."""
    what = kind.replace(" piece", "")
    if not squarely(d):
        st.count(f"{what} on a slant (not stood)")
        return
    step = STEP[out]
    for _ in range(3):                                   # to the back of the pavement
        nxt = (sq[0] + step[0], sq[1] + step[1])
        if not st.free(*nxt):
            break
        sq = nxt
    face = {"N": "S", "S": "N", "E": "W", "W": "E"}[out]
    along = STEP[compass(d)]
    for k in (1, -1):
        s2 = (sq[0] + along[0] * k, sq[1] + along[1] * k)
        if st.free(*s2):
            for a, b in ((sq, s2), (s2, sq)):
                piece = (int(a[0] > b[0]), int(a[1] > b[1]))
                st.put(*a, sheet + str(BENCH[(face, piece)]), kind)
            return
    st.count(f"{what} without room")


def roadworks(st: Stand, links, paths, n=ROADWORKS):
    """n pits on the pavement, fenced, each with its sign before it."""
    runs = []
    for i, path in paths.items():
        if links[i].cls != "local":
            continue
        vs = vertices(path)
        for a, b in zip(vs, vs[1:]):
            if (a[0] == b[0]) != (a[1] == b[1]):         # along the grid
                runs.append((math.dist(a, b), i, tuple(a), tuple(b)))
    made = 0
    for _len, i, a, b in sorted(runs, reverse=True):
        if made >= n:
            break
        h = ((b[0] - a[0]) / _len, (b[1] - a[1]) / _len)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        # The pavement south of an east-west street, east of a north-south one:
        # every edge of the pit then belongs to a square of the pit or beyond it.
        want = "S" if h[1] == 0 else "E"
        side = 1 if compass(st.left(h)) == want else -1
        kerb = st.kerb(mid, h, side)
        if kerb is None:
            continue
        along = STEP["E"] if h[1] == 0 else STEP["S"]
        out = STEP[want]
        pit = [kerb, (kerb[0] + along[0], kerb[1] + along[1])]
        beyond = [(p[0] + out[0], p[1] + out[1]) for p in pit] + [(pit[1][0] + along[0], pit[1][1] + along[1])]
        if not all(st.free(*p) for p in pit + beyond) or not all(st.free(*p, LAYER2) for p in pit):
            continue
        # Edges: the road side and the near end on the pit's own squares, the far side and end on those beyond.
        near_edge = "N" if want == "S" else "W"
        end_edge = "W" if want == "S" else "N"
        pieces = [(p, near_edge) for p in pit] + [(p, near_edge) for p in beyond[:2]] \
            + [(pit[0], end_edge), (beyond[2], end_edge)]
        used = {}
        for (x, y), e in pieces:
            layer = LAYER if (x, y) not in used else LAYER2
            used[(x, y)] = True
            lamp = (x, y) in (pit[0], beyond[2]) and e == end_edge
            st.put(x, y, F2 + str({("W", False): 18, ("N", False): 19, ("W", True): 20, ("N", True): 21}[(e, lamp)]),
                   "road works barrier", layer)
        # The near lane comes towards the works with the pavement on its left.
        lane = h if side == 1 else (-h[0], -h[1])
        pt = (mid[0] - lane[0] * 10, mid[1] - lane[1] * 10)
        sq = st.kerb(pt, lane, 1, sign=True)
        if sq is not None:
            st.put(*sq, road_sign("road_works", compass((-lane[0], -lane[1]))), "road works sign")
        made += 1
        st.count("road works")


def telegraph_poles(st: Stand, links, paths, town):
    """Poles along the country roads, wired a storey up where neighbours stand in line."""
    for i, path in sorted(paths.items()):
        if links[i].cls not in POLE_CLASSES:
            continue
        line = shapely.LineString(path)
        prev = None
        s = POLE_EVERY / 2
        while s < line.length - 5:
            p = line.interpolate(s)
            a, b = line.interpolate(max(0.0, s - 0.5)), line.interpolate(min(line.length, s + 0.5))
            s += POLE_EVERY
            n = math.hypot(b.x - a.x, b.y - a.y) or 1.0
            h = ((b.x - a.x) / n, (b.y - a.y) / n)
            if not st.inside(int(p.x), int(p.y)) or town[int(p.y), int(p.x)]:
                prev = None
                continue
            sq = st.kerb((p.x, p.y), h, 1, sign=True)
            if sq is None:
                prev = None
                continue
            run = "EW" if compass(h) in "EW" else "NS"
            st.put(*sq, CS + "12", "telegraph pole")
            st.m.put_above(sq[0], sq[1], 1, LAYER, CS + ("13" if run == "EW" else "14"), "telegraph pole top")
            if prev is not None:
                (px, py), prun = prev
                if run == prun == "EW" and py == sq[1]:
                    span = [(x, py) for x in range(min(px, sq[0]) + 1, max(px, sq[0]))]
                elif run == prun == "NS" and px == sq[0]:
                    span = [(px, y) for y in range(min(py, sq[1]) + 1, max(py, sq[1]))]
                else:
                    span = []
                for x, y in span:
                    st.m.put_above(x, y, 1, LAYER, CS + ("15" if run == "EW" else "16"), "telegraph wire")
            prev = (sq, run)


def whole_poles(rows: list) -> list:
    """The rows (x, y, z, layer, tile) with every telegraph pole whole: a top
    only over a foot still there, a run of wires only with a pole at each end
    (kerbs.py takes off a foot on a square it makes road; its top and wires
    would hang in the air)."""
    feet = {(r[0], r[1]) for r in rows if r[4] == CS + "12"}
    runs = {CS + "15": (1, 0), CS + "16": (0, 1)}
    wired = {(r[0], r[1], r[4]) for r in rows if r[4] in runs}
    ok = set()
    for x, y, t in wired:
        if (x, y, t) in ok:
            continue
        dx, dy = runs[t]
        run = [(x, y)]
        for k in (1, -1):                                  # the whole run this wire is in
            a, b = x + dx * k, y + dy * k
            while (a, b, t) in wired:
                run.append((a, b))
                a, b = a + dx * k, b + dy * k
        run.sort()
        (x0, y0), (x1, y1) = run[0], run[-1]
        if (x0 - dx, y0 - dy) in feet and (x1 + dx, y1 + dy) in feet:
            ok.update((a, b, t) for a, b in run)
    return [r for r in rows
            if not (r[4] in (CS + "13", CS + "14") and (r[0], r[1]) not in feet)
            and not (r[4] in runs and (r[0], r[1], r[4]) not in ok)]


def lay(m, nodes, proj, links, paths, owner, ok, hand, arms, zebras, area, town, through_pair, on_graph,
        limits=None, signals=()) -> set:
    """Every piece of furniture into m; returns the squares taken (no street dressing there)."""
    st = Stand(m, owner, ok, hand)
    from area import area_conf
    name = area_conf.town(area)            # area.json's town, else the folder's name to its first _
    beacons(st, zebras)
    signal_heads_at_junctions(st, nodes, proj, arms, links, paths, through_pair)
    junction_signs(st, nodes, proj, links, paths, arms, on_graph)
    limit_signs(st, links, paths, arms, through_pair, name, limits)
    cul_de_sacs(st, links, paths, arms)
    points(st, nodes, proj, links, paths, on_graph, town)
    town_centre.lay(st, nodes, proj, links, paths, signals)
    kiosks(st, nodes, proj, links, paths, town)
    roadworks(st, links, paths)
    telegraph_poles(st, links, paths, town)
    return st.taken
