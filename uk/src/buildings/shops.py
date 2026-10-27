"""What each building is used for - shops, pubs, offices - moved with the building.

    python shops.py <knoxmap map dir> <area data dir> [--epc <epc_nondomestic.csv>]

Run after split_units.py (the units are the shops) and before Build.

Build gives a building the uses of the OSM shop and amenity points that fall
inside it, where the road straightening put them. blocks.py moves the
buildings by their own real positions, so the points missed: of the 21 on
the High Street map, 8 landed in a building (LOG 2026-10-07), and the rest
were built as houses. And OSM maps only some of a High Street's shops.

So the uses are gathered here, in real coordinates, from:
  OSM     shop, amenity, office, craft, healthcare, leisure and tourism points
  EPC     non-domestic energy certificates: the commercial use of a property
          (retail, restaurant/pub, office, hotel, surgery), at its UPRN
  FSA     food hygiene ratings: every pub, restaurant, cafe, takeaway and
          food shop, at its geocode
and each is carried into the game through a building's frame (blocks.py's
turn and shift for it) - the nearest buildings tried, the one it lands in
(within REACH tiles: a mapper puts a shop's point at its door, on the
pavement) takes it. A building gets them as `knox_uses`, which Build reads in
place of points for the buildings blocks.py placed.

A building with a use and homes in it too (property points that are not a
business's) is flats over a shop: `knox_flats_over`, which Build makes a
block of flats with the shop on the ground floor, shop front and all, from
two storeys up (it did so only from three).
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # src: our packages
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from buildings import blocks as B  # noqa: E402
from knoxbuild.uses import USE_KEYS, use_of  # noqa: E402
from ground.trees import to_game  # noqa: E402

REACH = {"osm": 2.0, "epc": 1.5, "fsa": 3.0}   # tiles outside a building a use may still be its
NEAR_M = 60                                     # buildings whose real centre is this near are tried

# Non-domestic EPC property types -> the OSM tags Build reads (a retail
# "shop=retail" is a general store; "shop=yes" would be no use at all).
EPC_TAGS = [
    (("retail",), {"shop": "retail"}),
    (("restaurant", "a3/a4/a5"), {"amenity": "restaurant"}),
    (("office",), {"office": "yes"}),
    (("hotel",), {"tourism": "hotel"}),
    (("primary health care",), {"amenity": "doctors"}),
]
FSA_TAGS = {
    "Pub/bar/nightclub": {"amenity": "pub"},
    "Restaurant/Cafe/Canteen": {"amenity": "restaurant"},
    "Takeaway/sandwich shop": {"amenity": "fast_food"},
    "Retailers - other": {"shop": "convenience"},
    "Retailers - supermarkets/hypermarkets": {"shop": "supermarket"},
    "Hotel/bed & breakfast/guest house": {"tourism": "guest_house"},
}


FRONT_REACH = 15        # tiles a face looks out for a street
OWN_STREET_REACH = 20   # tiles a building's own street may be from it to be its front


def _street(map_dir: str, name: str) -> np.ndarray:
    """Roads and the pavements beside them (not yards, not car parks)."""
    from PIL import Image
    from scipy import ndimage
    from generator import pz_colors as C
    path = os.path.join(map_dir, f"{name}_ground_base.prekerbs.bmp")
    if not os.path.exists(path):
        path = os.path.join(map_dir, f"{name}_ground_base.bmp")
    g = np.array(Image.open(path).convert("RGB"))

    def colour(c):
        return (g[:, :, 0] == c[0]) & (g[:, :, 1] == c[1]) & (g[:, :, 2] == c[2])
    road = np.zeros(g.shape[:2], bool)
    for c in (C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE):
        road |= colour(c)
    return road | (colour(C.PALE_CONCRETE) & ndimage.binary_dilation(road, iterations=4))


def _front_side(poly, street: np.ndarray, occupied: np.ndarray, me: int) -> str | None:
    """"N", "S", "W" or "E": the face of a building nearest a street, looking
    out from a quarter, half and three quarters along each face; a look that
    meets another building first counts for nothing (a party wall)."""
    H, W = street.shape
    x0, y0, x1, y1 = (int(round(v)) for v in poly.bounds)
    best, best_d = None, FRONT_REACH + 1
    for name, (dx, dy) in (("N", (0, -1)), ("S", (0, 1)), ("W", (-1, 0)), ("E", (1, 0))):
        if dx == 0:
            starts = [(x0 + (x1 - x0) * f // 4, y0 if dy < 0 else y1 - 1) for f in (1, 2, 3)]
        else:
            starts = [(x0 if dx < 0 else x1 - 1, y0 + (y1 - y0) * f // 4) for f in (1, 2, 3)]
        ds = []
        for sx, sy in starts:
            for d in range(1, FRONT_REACH + 1):
                x, y = sx + dx * d, sy + dy * d
                if not (0 <= x < W and 0 <= y < H):
                    break
                o = occupied[y, x]
                if o and o != me:
                    break
                if street[y, x]:
                    ds.append(d)
                    break
        if ds:
            d = min(ds) - 0.2 * len(ds)           # a face the street runs all along wins a tie
            if d < best_d:
                best, best_d = name, d
    return best


def _named_roads(map_dir: str, info: dict, proj) -> dict:
    """Named roads as the game draws them (the straightened lines): name -> lines."""
    from shapely.geometry import LineString
    from streets import diagonals as D
    out = {}
    for _cat, f, a, b, _d in D._segments(map_dir, info, proj):
        if f.tags.get("name"):
            out.setdefault(f.tags["name"], []).append(LineString([a, b]))
    return out


def _side_towards(poly, lines, street: np.ndarray, occupied: np.ndarray, me: int) -> str | None:
    """The face of a building towards the nearest point of its own street, if
    that street is within OWN_STREET_REACH and the look out from that face
    meets the street before any other building."""
    near = min(lines, key=lambda l: l.distance(poly))
    if near.distance(poly) > OWN_STREET_REACH:
        return None
    c = poly.centroid
    q = near.interpolate(near.project(c))
    dx, dy = q.x - c.x, q.y - c.y
    side = ("E" if dx > 0 else "W") if abs(dx) > abs(dy) else ("S" if dy > 0 else "N")
    H, W = street.shape
    x0, y0, x1, y1 = (int(round(v)) for v in poly.bounds)
    sx, sy = {"N": ((x0 + x1) // 2, y0), "S": ((x0 + x1) // 2, y1 - 1),
              "W": (x0, (y0 + y1) // 2), "E": (x1 - 1, (y0 + y1) // 2)}[side]
    ddx, ddy = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}[side]
    for d in range(1, OWN_STREET_REACH + 1):
        x, y = sx + ddx * d, sy + ddy * d
        if not (0 <= x < W and 0 <= y < H):
            return None
        o = occupied[y, x]
        if o and o != me:
            return None
        if street[y, x]:
            return side
    return None


def same_name(name: str) -> str:
    """A business name as two sources would both write it: "The Red Lion"
    and "Red Lion", "Bel & the Dragon" and "Bel And The Dragon"."""
    words = name.lower().replace("&", " and ").replace("'", "").split()
    words = ["".join(c for c in w if c.isalnum()) for w in words]
    return " ".join(w for w in words if w and w != "the")


# OSM building tags for work, not homes; and an untagged building this big
# (square metres) with no home or business found in it is a works or depot
# (Alton's TNT depot was eight blocks of flats).
WORK_BUILDINGS = {"industrial", "commercial", "office", "retail", "warehouse", "factory",
                  "manufacture", "service", "depot", "hangar", "storage_tank", "supermarket"}
WORKS_M2 = 400
WORK_MAN_MADE = {"storage_tank", "silo", "works", "water_tower", "wastewater_plant"}
LIDAR_WORKS_M2 = 60                   # Build makes an annex this small a shed (ANNEX_SHED_M2)


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    from area import area_conf
    epc_path = sys.argv[sys.argv.index("--epc") + 1] if "--epc" in sys.argv else \
        os.path.join(area_conf.epc_dir(area) or area, "epc_nondomestic.csv")   # no council: no business EPCs
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    b = info["bbox"]
    proj = B.Projector.build(b["south"], b["west"], b["north"], b["east"], info["meters_per_tile"],
                             info.get("rotation") or 0.0)
    W, H = proj.width, proj.height

    # --- the buildings, in the game and where they really are ---------------------------
    gj_path = os.path.join(map_dir, f"{name}_buildings.geojson")
    gj = json.load(open(gj_path, encoding="utf-8"))
    blds = []
    for ft in gj["features"]:
        p = ft["properties"]
        for k in ("knox_uses", "knox_flats_over", "knox_dwelling", "knox_homes", "knox_front"):
            p.pop(k, None)
        if not p.get("placed") or not p.get("lidar_frame") or ft["geometry"]["type"] != "Polygon":
            continue
        fr = p["lidar_frame"]
        poly = Polygon([proj.to_px(la, lo) for lo, la in ft["geometry"]["coordinates"][0]])
        if not poly.is_valid:
            poly = poly.buffer(0)
        blds.append({"ft": ft, "frame": fr, "poly": poly, "centre": (fr[3] - fr[1], fr[4] - fr[2])})
    if not blds:
        raise ValueError(f"{gj_path}: no placed building has a lidar_frame - run blocks.py "
                         "(and split_units.py, if this map has units) before shops.py")
    tree = cKDTree(np.array([bd["centre"] for bd in blds]))

    def home_of(lat, lon, source):
        """The building a real point belongs to, or None."""
        rx, ry = proj.to_px(lat, lon)
        best = None
        for j in tree.query_ball_point((rx, ry), NEAR_M / info["meters_per_tile"]):
            gx, gy = to_game(rx, ry, blds[j]["frame"])
            d = blds[j]["poly"].distance(Point(gx, gy))
            if d <= REACH[source] and (best is None or d < best[0]):
                best = (d, j)
        return best and best[1]

    # --- uses, from the three sources ---------------------------------------------------------
    uses = {}                                     # building -> [tags]
    count = {}

    def give(j, tags, source):
        if j is None or not use_of(tags):
            count[source + " unplaced"] = count.get(source + " unplaced", 0) + (j is None)
            return
        uses.setdefault(j, []).append(dict(tags, knox_source=source))
        count[source] = count.get(source, 0) + 1

    osm_names = set()                             # businesses OSM has placed
    with gzip.open(os.path.join(map_dir, info["osm_cache"]), "rt", encoding="utf-8") as f:
        for ft in json.load(f)["features"]:
            t = ft.get("tags") or {}
            if ft["kind"] == "node" and any(k in t for k in USE_KEYS):
                lat, lon = ft["geometry"][0]
                j = home_of(lat, lon, "osm")
                give(j, t, "osm")
                if j is not None and t.get("name"):
                    osm_names.add(same_name(t["name"]))

    uprn_at = {}
    uprn_path = area_conf.extra(area, "uprn.csv")     # none: uses from OSM (and FSA) alone
    if uprn_path:
        with open(uprn_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uprn_at[row["UPRN"]] = (float(row["LATITUDE"]), float(row["LONGITUDE"]))
    business_uprns = set()
    if os.path.exists(epc_path):
        latest = {}
        with open(epc_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u = (row.get("uprn") or "").split(".")[0]
                if u in uprn_at and (u not in latest or row["lodgement_date"] > latest[u]["lodgement_date"]):
                    latest[u] = row
        for u, row in latest.items():
            business_uprns.add(u)
            kind = (row.get("property_type") or "").lower()
            tags = next((t for keys, t in EPC_TAGS if any(k in kind for k in keys)), None)
            if tags:
                give(home_of(*uprn_at[u], "epc"), tags, "epc")
    fsa_path = area_conf.extra(area, "fsa.json")
    if fsa_path:
        for e in json.load(open(fsa_path, encoding="utf-8")):
            tags = FSA_TAGS.get(e.get("type"))
            if not tags:
                continue
            tags = dict(tags, name=e.get("name") or "")
            if tags.get("amenity") == "restaurant" and any(w in tags["name"].lower() for w in ("cafe", "café", "coffee", "tea")):
                tags["amenity"] = "cafe"
            # FSA points are geocoded from the postcode, OSM's are at the door:
            # one OSM has placed already would land a second time on the
            # nearest building to the postcode (The Red Lion, 45 m behind
            # the High Street).
            if same_name(tags["name"]) in osm_names:
                count["fsa already in osm"] = count.get("fsa already in osm", 0) + 1
                continue
            give(home_of(e["lat"], e["lon"], "fsa"), tags, "fsa")

    # --- one business counted once -------------------------------------------------------------
    # A shop mapped in OSM (or rated by the FSA) and its business rates
    # certificate are the same business, not two: in a building with named
    # uses, as many EPC uses as there are named ones are dropped, those of the
    # same kind first. Extra certificates are other businesses and stay.
    deduped = 0
    for j, tl in list(uses.items()):
        # A food hygiene record of the same kind as an OSM shop in the same
        # building is that shop under its company name ("The Southern
        # Co-Operative Limited" for "The Co-operative Food").
        osm_kinds = {use_of(t)[0] for t in tl if t.get("knox_source") == "osm" and use_of(t)}
        same = [t for t in tl if t.get("knox_source") == "fsa" and use_of(t) and use_of(t)[0] in osm_kinds]
        if same:
            tl = [t for t in tl if not any(t is d for d in same)]
            uses[j] = tl
            deduped += len(same)
        named = [t for t in tl if t.get("knox_source") in ("osm", "fsa")]
        epc = [t for t in tl if t.get("knox_source") == "epc"]
        if not named or not epc:
            continue
        kinds = {use_of(t)[0] for t in named if use_of(t)}
        epc.sort(key=lambda t: (use_of(t)[0] if use_of(t) else "") not in kinds)
        drop = epc[:len(named)]
        uses[j] = [t for t in tl if not any(t is d for d in drop)]
        deduped += len(drop)
    count["epc same as a named one"] = deduped

    # --- a large-format shop takes its building's hall ------------------------------------------
    # split_units.py gives a low flat single storey of a building (its hall)
    # to the nearest business and marks it `unit_hall_of`: a supermarket,
    # department store, DIY or the like moves its use there, so it has a sales
    # floor and not a frontage the width of a house.
    from buildings import addresses as A
    index_of = {str(bd["ft"]["properties"]["osm_id"]): j for j, bd in enumerate(blds)}
    halls = 0
    for j, bd in enumerate(blds):
        owner = bd["ft"]["properties"].get("unit_hall_of")
        k = index_of.get(str(owner)) if owner else None
        if k is None or not uses.get(k):
            continue
        big = [t for t in uses[k] if t.get("shop") in A.BIG_FORMAT]
        if big:
            uses[j] = uses.get(j, []) + big
            uses[k] = [t for t in uses[k] if not any(t is b for b in big)]
            if not uses[k]:
                del uses[k]
            halls += 1
    count["large-format shops moved into their hall"] = halls

    # --- a supermarket mapped as its outline (its own tags, not a point in it) ------------------
    # Alton's Waitrose and M&S Foodhall: shop=supermarket on the building, no
    # point, no rates record of their own - built as works and a house
    # (asset_creator/commercial_kit PLAN step 2, 2026-10-20). Not a piece or a
    # unit's back part: the outline itself.
    own = 0
    for j, bd in enumerate(blds):
        p = bd["ft"]["properties"]
        if p.get("piece") or "p" in str(p.get("osm_id")):
            continue
        shop = p.get("shop") or ("supermarket" if p.get("building") == "supermarket" else None)
        if shop in ("supermarket",) or shop in A.BIG_FORMAT:
            if not any(t.get("shop") == shop for t in uses.get(j, [])):
                uses.setdefault(j, []).insert(0, {"shop": shop, "name": p.get("name"), "knox_source": "osm"})
                own += 1
    count["supermarkets mapped as their outline"] = own

    # --- homes in each building: property points that are not a business's ---------------------
    homes = {}
    home_types = {}                               # building -> domestic EPC property types
    # The district's domestic EPCs, beside its non-domestic ones (Alton read
    # Hart's: none matched, so every address in an office was a home).
    dom = A._latest(os.path.join(os.path.dirname(epc_path), "epc_domestic.csv"))
    for u, (lat, lon) in uprn_at.items():
        if u in business_uprns:
            continue
        j = home_of(lat, lon, "epc")
        if j is None:
            continue
        # In a building OSM says is for work, an address is a home only if
        # it has a domestic EPC (else a business with no rates record).
        if blds[j]["ft"]["properties"].get("building") in WORK_BUILDINGS and u not in dom:
            continue
        homes[j] = homes.get(j, 0) + 1
        home_types.setdefault(j, []).append((dom.get(u) or {}).get("property_type") or "")
    # No UPRNs (area_conf: osm_lidar, or none fetched): OSM's house numbers are the
    # homes (addresses.load). They are a few of the homes, so a building without one
    # is not known to have none: the works / annex rules below that read "no address"
    # as "no home" are left out, and Build decides from the tags.
    every_address = bool(uprn_at)
    if not every_address:
        for a in A.load(map_dir, area, proj, info):
            if a.source != "osm_addr":
                continue
            j = home_of(*proj.to_latlon(a.x, a.y), "epc")
            if j is not None and blds[j]["ft"]["properties"].get("building") not in WORK_BUILDINGS:
                homes[j] = homes.get(j, 0) + 1
                home_types.setdefault(j, []).append("")

    # --- what each building is (knox_dwelling) -----------------------------------------------
    # From its own addresses only (PLAN principle 14), for Build to take as it
    # stands: a shop with flats over it, a shop, a house, flats, or - a part of
    # a building whose addresses are all in its other parts - its back range.
    with_address = {str(blds[j]["ft"]["properties"].get("parent_osm_id") or blds[j]["ft"]["properties"]["osm_id"])
                    for j in set(homes) | set(uses)}
    dwelling_count = {}
    for j, bd in enumerate(blds):
        p = bd["ft"]["properties"]
        n_homes = homes.get(j, 0)
        types = [t for t in home_types.get(j, []) if t]
        if uses.get(j):
            kind = "shop_flats" if n_homes else "shop"
        elif n_homes:
            housey = [t for t in types if t in ("House", "Bungalow")]
            flatty = [t for t in types if t in ("Flat", "Maisonette")]
            if n_homes == 1:
                kind = "flats" if flatty else "house"
            else:
                kind = "house" if housey and not flatty else "flats"
        elif p.get("building") in WORK_BUILDINGS or p.get("man_made") in WORK_MAN_MADE:
            kind = "works"                        # no home, no shop: a works, a depot, offices (any part of one)
        elif p.get("source") == "lidar" and (every_address or bd["poly"].area < LIDAR_WORKS_M2):
            # Not in OSM and no address (new homes have UPRNs): an outbuilding,
            # never a home - a works when big, else a shed.
            kind = "works" if bd["poly"].area >= LIDAR_WORKS_M2 else "annex"
        elif every_address and p.get("parent_osm_id") and str(p["parent_osm_id"]) in with_address:
            kind = "annex"
        elif every_address and p.get("building") in ("yes", None) and bd["poly"].area >= WORKS_M2:
            kind = "works"
        else:
            kind = None                           # nothing known: Build decides
        if kind:
            p["knox_dwelling"] = kind
            p["knox_homes"] = n_homes
            dwelling_count[kind] = dwelling_count.get(kind, 0) + 1

    # --- the street side (knox_front): the face towards the nearest road or its
    # pavement, never a yard or a car park; looked for from three points on each
    # face, stopping at another building. Build puts the shop front and the
    # front door there.
    street = _street(map_dir, name)
    occupied = np.zeros((H, W), np.int32)
    from PIL import Image as _Im, ImageDraw as _Dr
    for j, bd in enumerate(blds):
        im = _Im.new("1", (W, H), 0)
        for part in getattr(bd["poly"], "geoms", [bd["poly"]]):
            _Dr.Draw(im).polygon(list(part.exterior.coords), fill=1)
        occupied[np.array(im)] = j + 1
    # Each building's own street (features.py: the named street it is
    # addressed to and fronts), as drawn in the game: its face towards that
    # street is the front when nothing stands between them; else the face
    # nearest any street (a building behind the frontage, on a yard or lane).
    named = _named_roads(map_dir, info, proj)
    own_street = {}
    feats_csv = os.path.join(area, "buildings.csv")
    if os.path.exists(feats_csv):
        with open(feats_csv, encoding="utf-8") as fh:
            own_street = {r["osm_id"]: r.get("street") or "" for r in csv.DictReader(fh)}
    fronts = {}
    for j, bd in enumerate(blds):
        p = bd["ft"]["properties"]
        st = own_street.get(str(p.get("parent_osm_id") or p.get("osm_id")))
        side = None
        if st in named:
            side = _side_towards(bd["poly"], named[st], street, occupied, j + 1)
        if side is None:
            side = _front_side(bd["poly"], street, occupied, j + 1)
        if side:
            bd["ft"]["properties"]["knox_front"] = side
            fronts[side] = fronts.get(side, 0) + 1

    flats_over = 0
    for j, bd in enumerate(blds):
        # Every building gets the list, empty or not: Build then takes no
        # stray point for it (a point the straightening put inside a house).
        bd["ft"]["properties"]["knox_uses"] = uses.get(j, [])
    for j, tl in uses.items():
        p = blds[j]["ft"]["properties"]
        if homes.get(j) and tl:
            p["knox_flats_over"] = True
            flats_over += 1
    with open(gj_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)

    kinds = {}
    for tl in uses.values():
        for t in tl:
            u = use_of(t)[0]
            kinds[u] = kinds.get(u, 0) + 1
    print(f"buildings {len(blds)}: {len(uses)} with a use ({flats_over} with homes too: flats over a shop)")
    print(f"  uses placed by source: {dict(sorted(count.items()))}")
    print(f"  front rooms: {dict(sorted(kinds.items(), key=lambda kv: -kv[1]))}")
    print(f"  what each building is: {dict(sorted(dwelling_count.items(), key=lambda kv: -kv[1]))}")
    print(f"  street side found for {sum(fronts.values())} of {len(blds)}: {fronts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
