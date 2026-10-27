"""A 1990s English town centre's street furniture on a road-graph map (user,
2026-10-14; the art: asset_creator townscape.py, street_furniture_uk_03 and
pub_signs_uk_01), laid with the rest of the furniture (furniture.lay) on
the Furniture layer, off the road.

Where each goes - OSM's points (area/fetch_road_nodes.py), carried with
their road as furniture.points carries a post box (furniture._stand: as far
out from the snapped road as the point is from the real one, the nearest
pavement or verge square there):
- Bus stops (highway=bus_stop): the flag on the pavement, reaching out
  towards the road; where OSM says shelter=yes, a shelter at the back of the
  pavement facing the road, two squares along it (as a bench), the flag
  where the stop is.
- Litter bins (amenity=waste_basket), grit bins (amenity=grit_bin, its front
  to the road), street cabinets (man_made=street_cabinet, BT's green, its
  doors to the road), cycle stands (amenity=bicycle_parking, two side by side
  along the kerb, each across it as bikes stand).
- Pub signs (amenity=pub, a point or its building's middle; its sign's
  sprites by its name: PUBS, asset_creator pubs.py): on the pavement nearest
  the pub within PUB_REACH, its board facing along the street.
- Guard railing at the 1990s signal crossings (markings.crossings): along
  the kerb either side of the crossing, RAIL_FROM to RAIL_TO squares from
  it, on both pavements - only where the kerb runs straight beside it.
"""
from __future__ import annotations

from . import furniture as fu

F3, PS = "street_furniture_uk_03_", "pub_signs_uk_01_"
# asset_creator pubs.PUBS's keys, in order (a pub's sign is its place in it,
# two sprites each: its board facing N-S, then E-W); a new pub at the end.
PUBS = ("anchor_inn", "waterwitch", "red_lion", "bell_inn", "swan", "greyfriar", "ivy_house", "white_horse",
        "wheatsheaf", "market_hotel", "bakers_arms", "george", "eight_bells", "crown_hotel", "railway_arms",
        "french_horn", "white_hart_inn")
PUB_REACH = 25               # tiles from a pub's point, the pavement its sign may stand on
RAIL_FROM, RAIL_TO = 2, 4    # squares along the kerb from a signal crossing's middle, its guard railing
BACK = {"N": "S", "S": "N", "E": "W", "W": "E"}


def slug(name: str) -> str:
    """An OSM pub name -> its key in PUBS (as asset_creator pubs.slug)."""
    s = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    s = s[4:] if s.startswith("the_") else s
    return "_".join(p for p in s.split("_") if p)


def bus_stop(face):
    return F3 + str(fu.ORDER.index(face))


def grit_bin(face):
    return F3 + str(13 + fu.ORDER.index(face))


def cabinet(face):
    return F3 + str(17 + fu.ORDER.index(face))


def pub_sign(key, board):
    return PS + str(2 * PUBS.index(key) + (0 if board == "S" else 1))


GUARD = {"W": 23, "N": 24, "E": 25, "S": 26}


def points(st, nodes, proj, links, paths):
    for lonlat, tags in nodes:
        kind = ("bus_stop" if tags.get("highway") == "bus_stop" else
                tags.get("amenity") if tags.get("amenity") in ("waste_basket", "grit_bin", "bicycle_parking") else
                "cabinet" if tags.get("man_made") == "street_cabinet" else None)
        if kind is None:
            continue
        p = proj.to_px(lonlat[1], lonlat[0])
        if not st.inside(int(p[0]), int(p[1])):
            continue
        got = fu._stand(st, p, links, paths, kind.replace("_", " "))
        if got is None:
            continue
        sq, d, out = got
        road = BACK[out]                                   # the way to the road
        if kind == "bus_stop":
            bus_stop_and_shelter(st, sq, d, out, tags.get("shelter") == "yes")
        elif kind == "waste_basket":
            st.put(*sq, F3 + "12", "litter bin")
        elif kind == "grit_bin":
            st.put(*sq, grit_bin(fu.seen(road)), "grit bin")
        elif kind == "cabinet":
            st.put(*sq, cabinet(fu.seen(road)), "street cabinet")
        else:
            stand = F3 + ("22" if fu.compass(d) in "EW" else "21")    # across the kerb
            st.put(*sq, stand, "cycle stand")
            along = fu.STEP[fu.compass(d)]
            nxt = (sq[0] + along[0], sq[1] + along[1])
            if fu.squarely(d) and st.free(*nxt):
                st.put(*nxt, stand, "cycle stand")


def bus_stop_and_shelter(st, sq, d, out, shelter):
    """The flag on the stop's square, reaching to the road; a shelter behind it
    at the back of the pavement where there is room, else on the stop's square
    with the flag beside it along the kerb (review, 2026-10-14: the shelter
    had taken the flag's square, 8 stops of 10 in Alton lost their flag)."""
    flag = bus_stop(BACK[out])
    if not shelter:
        st.put(*sq, flag, "bus stop")
        return
    step = fu.STEP[out]
    back = (sq[0] + step[0], sq[1] + step[1])
    before = len(st.taken)
    if st.free(*back):
        st.put(*sq, flag, "bus stop")
        fu.bench(st, back, d, out, sheet=F3, kind="bus shelter piece")
        if len(st.taken) > before + 1:
            return
    if (sq[0], sq[1]) not in st.taken:
        fu.bench(st, sq, d, out, sheet=F3, kind="bus shelter piece")
        along = fu.STEP[fu.compass(d)]
        for k in (1, -1, 2, -2):                           # the flag beside the shelter, along the kerb
            x, y = sq[0] + along[0] * k, sq[1] + along[1] * k
            if st.free(x, y):
                st.put(x, y, flag, "bus stop")
                return
        st.count("bus stop flag without room beside its shelter")


def pub_signs(st, nodes, proj):
    for lonlat, tags in nodes:
        if tags.get("amenity") != "pub" or not tags.get("name"):
            continue
        key = slug(tags["name"])
        if key not in PUBS:
            st.count("pub sign without its board (not in PUBS)")
            continue
        p = proj.to_px(lonlat[1], lonlat[0])
        if not st.inside(int(p[0]), int(p[1])):
            continue
        got = fu._pavement_by(st, p, PUB_REACH)
        if got is None:
            st.count("pub sign without a pavement near")
            continue
        sq, out = got
        st.put(*sq, pub_sign(key, "E" if out in "NS" else "S"), "pub sign")      # the board along the street


def guard_rails(st, signals):
    """Railing along both kerbs either side of each signal crossing."""
    for q, ax in signals:
        for side in (1, -1):
            kerb = st.kerb(q, ax, side)
            if kerb is None:
                continue
            road = None
            for f, (dx, dy) in fu.STEP.items():               # the edge the road lies beyond
                x, y = kerb[0] + dx, kerb[1] + dy
                if st.inside(x, y) and st.owner[y, x] != 0 and (dx, dy) not in (ax, (-ax[0], -ax[1])):
                    road = f
            if road is None:
                continue
            rx, ry = fu.STEP[road]
            for sgn in (1, -1):
                for k in range(RAIL_FROM, RAIL_TO + 1):
                    x, y = kerb[0] + ax[0] * k * sgn, kerb[1] + ax[1] * k * sgn
                    if not (st.inside(x + rx, y + ry) and st.owner[y + ry, x + rx] != 0) or st.owner[y, x] != 0:
                        break                                 # the kerb turns away: no railing past it
                    if st.free(x, y):                         # (past the signal head standing in the run)
                        st.put(x, y, F3 + str(GUARD[road]), "guard railing")


def lay(st, nodes, proj, links, paths, signals):
    pub_signs(st, nodes, proj)
    guard_rails(st, signals)
    points(st, nodes, proj, links, paths)
