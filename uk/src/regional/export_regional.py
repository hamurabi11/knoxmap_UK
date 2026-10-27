"""Write KnoxMap's regional.json for a map from the building-type data.

    python export_regional.py <area data dir> <palette.json> <knoxmap map dir>

Per building (by OSM id):
  style   from its look (subtypes.py: its sub-type as in 1993) via the palette's
          subtype_styles, else its type via type_styles; listed buildings and
          unlisted High Street buildings (shops included) from their own lists.
          Where a list has several styles, the OSM id picks one, so neighbours
          differ and a rebuild picks the same.
  levels  LiDAR storeys, where the LiDAR saw the building; else the median of
          the buildings round it, at most GUESS_MAX_LEVELS (not churches,
          schools, or a building whose OSM tags give its storeys).
Buildings blocks.py drew from the LiDAR (source=lidar) take the style most of
their typed neighbours have.
  upper_wall, door_surround, bay  a house's parts from its style (house_parts):
          its upper floors' wall, its front door's porch, the bays in front of
          its street windows.
"""
import csv
import json
from collections import Counter
import os
import sys

NO_LIDAR = {"", "no_building", "too_small"}
NEIGHBOURS = 6        # a LiDAR-drawn building takes the style of this many typed neighbours
NEIGHBOUR_M = 80      # ...within this far
GUESS_NEIGHBOURS = 8  # a building with no storeys of its own takes the median of this many near it
GUESS_MAX_LEVELS = 3  # ...at most this many (Odiham's tallest homes)
GUESS_DEFAULT = 2     # ...or this many with none near
BAY_SETBACK_M = 2.0   # a bay stands in front of the wall: only where the house is this far back from the road


def pick(options, osm_id):
    return options[int(osm_id) % len(options)] if options else None


def house_parts(out, pal, area, feats) -> dict:
    """Each house's parts from its palette style (user, 2026-10-19; LOG
    2026-10-19 #9, #11): `wall_trim` {"plinth": the style's `plinth`}
    (unless the listed step chose trims), `upper_wall` (a wall set for the storeys above the
    ground: [its West tile, its first window-shaped tile], on houses of two
    storeys or more), `door_surround` (the style's porch, unless the listed
    step chose one) and `bay` (where the house stands BAY_SETBACK_M or more
    back from the road: buildings.geojson `setback_m`, a unit by its parent's)
    and `garage` (to-do K09.24: the style's garage door kit; Build finds the
    flat-roofed wing beside the house that is the garage, or, where the
    style has them `integral` and the house is detached, builds one in)."""
    styles = {s["name"]: s for s in pal["palette"].get("add", [])}
    setback, form = {}, {}
    path = os.path.join(area, "buildings.geojson")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ft in json.load(f)["features"]:
                p = ft["properties"]
                if p.get("setback_m") is not None:
                    setback[str(p.get("osm_id"))] = float(p["setback_m"])
                if p.get("form"):
                    form[str(p.get("osm_id"))] = p["form"]
    for ft in feats:
        p = ft["properties"]
        parent = str(p.get("parent_osm_id") or "")
        if parent in setback:
            setback.setdefault(str(p["osm_id"]), setback[parent])
        if parent in form:
            form.setdefault(str(p["osm_id"]), form[parent])
    n = Counter()
    for oid, e in out.items():
        # Downpipes on every building, not only the listed ones (user,
        # 2026-10-19; Build's drains, the listed maker's drain kit).
        if e.get("style") and not e.get("drains"):
            e["drains"] = True
            n["drains"] += 1
        s = styles.get(e.get("style"))
        if not s:
            continue
        if s.get("upper_wall") and (e.get("levels") or 1) >= 2:
            e["upper_wall"] = [f"{s['upper_wall']}_000", f"{s['upper_wall']}_016"]
            if s.get("upper_sides"):
                e["upper_sides"] = s["upper_sides"]
            n["upper_wall"] += 1
        if s.get("porch") and not e.get("door_surround"):
            e["door_surround"] = s["porch"]
            n["porch"] += 1
        if s.get("plinth") and not e.get("wall_trim"):
            e["wall_trim"] = {"plinth": s["plinth"]}
            n["plinth"] += 1
        if s.get("bay") and setback.get(oid, 0) >= BAY_SETBACK_M:
            e["bay"] = s["bay"]
            n["bay"] += 1
        g = s.get("garage")
        if g and (e.get("levels") or 1) <= 3:
            e["garage"] = {"kit": g["kit"], "integral": bool(g.get("integral")) and form.get(oid) == "detached"}
            n["garage"] += 1
    return dict(n)


def house_rooms(out, area, feats) -> int:
    """What Build's room solver (knoxbuild/rooms_uk.py, ROOMS.md) needs of each
    building: `rooms` {"look": its sub-type as in 1993, "form": detached /
    semi / terrace..., "epc_rooms": its EPCs' habitable rooms}. A unit takes
    its parent's."""
    path = os.path.join(area, "types.csv")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        info = {r["osm_id"]: r for r in csv.DictReader(f)}
    parent = {str(ft["properties"]["osm_id"]): str(ft["properties"].get("parent_osm_id") or "")
              for ft in feats}
    props = {str(ft["properties"]["osm_id"]): ft["properties"] for ft in feats}
    n = 0
    for oid, e in out.items():
        r = info.get(oid) or info.get(parent.get(oid, ""))
        p = props.get(oid) or props.get(parent.get(oid, "")) or {}
        # shops.py's verdict: a house, flats, homes over a shop...; and its homes (UPRNs)
        dwelling, homes = p.get("knox_dwelling"), p.get("knox_homes")
        # (a shop with no homes too: the solver lays out its ground floor and store rooms)
        if dwelling not in ("shop_flats", "flats", "shop") and (not r or r.get("form") in ("outbuilding", "non_residential")):
            continue
        r = r or {}
        rooms = {"look": r.get("look") or r.get("subtype") or "", "form": r.get("form") or ""}
        if dwelling:
            rooms["dwelling"] = dwelling
        if isinstance(homes, (int, float)) and homes:
            rooms["homes"] = int(homes)
        # the EPCs' rooms are each dwelling's: a building of several not cut
        # into units (flats in a house) takes none
        own = oid in info
        if str(r.get("epc_rooms") or "").isdigit() and not (own and int(r.get("uprn_count") or 1) > 1) \
                and dwelling not in ("shop_flats", "flats", "shop"):
            rooms["epc_rooms"] = int(r["epc_rooms"])
        e["rooms"] = rooms
        n += 1
    return n


def shop_trades(out) -> int:
    """Each signed shop's trade (`trade`: butcher, bank, estate_agent...), the trade its
    fascia was chosen for (names.py SIGNS), so Build lays out and stocks the shop the sign
    names (regional_uk.uses_with; ROOMS.md section 17). Runs on a finished regional.json
    too (after uk_shopfronts and uk_pubs)."""
    if not os.path.isdir(COMMERCIAL_KIT):
        return 0
    sys.path.insert(0, COMMERCIAL_KIT)
    import names as NM
    n = 0
    for e in out.values():
        sign = (e.get("fascia") or {}).get("sign")
        if sign in NM.SIGNS:
            e["trade"] = NM.SIGNS[sign][1]
            n += 1
        else:
            e.pop("trade", None)
    return n


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths
import ukpaths  # noqa: E402
COMMERCIAL_KIT = ukpaths.assets("commercial_kit") or ""       # none without asset_creator: no shop signs
OLD_TYPES = {"vernacular", "georgian", "victorian", "edwardian"}
OLD_ERAS = {"pre1900", "1900_1929"}
SPIFFOS_UNIT_M2 = 120      # a Spiffo's: the size of an ordinary high street unit
SPIFFOS_MAX_M2 = 180       # ...and no bigger than this (a parade of 238 m2 read as one long Spiffo's)
SUPERSTORE_M2 = 600        # a supermarket this big with no homes over it: a superstore (commercial_kit step 2)
SUPERSTORE_STYLES = ("uk_superstore_brick", "uk_superstore_buff")


def outline_m2(ft) -> float:
    """A building's real footprint in m2, from its lon/lat ring."""
    import math
    ring = ft["geometry"]["coordinates"][0] if ft["geometry"]["type"] == "Polygon" else []
    if not ring:
        return 0.0
    k = 111320 * math.cos(math.radians(ring[0][1]))
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:]))) / 2 * k * 111320


UNIT_M2 = (150, 5000)       # a works this big (no home, no shop): an industrial unit (commercial_kit step 3)
UNIT_STYLES = ("uk_unit_juniper", "uk_unit_grey")
UNIT_FLAT_TAGS = {"industrial", "warehouse", "manufacture", "factory", "storage"}
UNIT_SKIP_TAGS = {"farm_auxiliary", "barn", "farm", "cowshed", "stable", "greenhouse", "church", "school",
                  "garage", "garages", "shed", "hut", "roof", "carport", "construction", "ruins"}


def _building(p) -> str:
    """The id a look is chosen by: a piece's parent's (its pieces are one building, cut so
    the lots do not overlap: one look, one door), else the outline's own."""
    return str(p["parent_osm_id"]) if p.get("piece") and p.get("parent_osm_id") else str(p.get("osm_id"))


def _biggest(feats, want) -> dict:
    """building -> the id of its biggest outline among feats that `want` (a door goes on it)."""
    best = {}
    for ft in feats:
        p = ft["properties"]
        if want(p):
            b, m2 = _building(p), outline_m2(ft)
            if m2 > best.get(b, (0, None))[0]:
                best[b] = (m2, str(p.get("osm_id")))
    return {b: oid for b, (_m, oid) in best.items()}


def _listed(rows, p) -> bool:
    """Listed by its row (types.csv: listed_grade), its own or its parent outline's: the
    listed maker designs it (`listed` is only written by its apply.py, after this step)."""
    oid = str(p.get("osm_id"))
    return bool((rows.get(str(p.get("parent_osm_id") or oid)) or rows.get(oid) or {}).get("listed_grade"))


def uk_units(out, rows, feats) -> dict:
    """Industrial units (asset_creator/commercial_kit/industrial.py, PLAN step 3): a works
    (shops.py: no home, no shop) of UNIT_M2, not listed, not pre-1930, not a farm
    building, takes a unit style (UNIT_STYLES: profiled cladding over a brick dado), no
    windows above its ground floor (`blank_upper`) and a roller shutter on its front
    (`loading`: the style's kit; Build regional_uk.loading_for) - all only where Build's
    kind for it is industrial (regional_uk.style_for: a school's buildings are works
    here too) - and, where OSM tags it industrial or a warehouse, one flat roof."""
    import hashlib
    n = Counter()
    for ft in feats:
        p = ft["properties"]
        if p.get("knox_dwelling") != "works" or (p.get("building") or "") in UNIT_SKIP_TAGS:
            continue
        oid = str(p.get("osm_id"))
        r = rows.get(str(p.get("parent_osm_id") or oid)) or rows.get(oid) or {}
        if r.get("listed_grade") or r.get("era") in OLD_ERAS or r.get("type") in OLD_TYPES:
            continue
        m2 = outline_m2(ft)
        if not UNIT_M2[0] <= m2 <= UNIT_M2[1]:
            continue
        e = out.setdefault(oid, {})
        h = int(hashlib.md5(_building(p).encode()).hexdigest()[:8], 16)
        style = UNIT_STYLES[h % len(UNIT_STYLES)]
        e.update(style=style, blank_upper=True, loading={"kit": style[3:]})
        if p.get("building") in UNIT_FLAT_TAGS:
            e["flat_roof"] = True       # only where OSM says it is industrial (Build decides the rest)
        n[style] += 1
    return dict(n)


FARM_TAGS = {"barn", "farm_auxiliary", "cowshed", "stable"}
MODERN_ERAS = {"1950_1966", "1967_1982", "1983_1995", "1996_plus"}
FARM_SHED_M2 = 300          # a farm building of no known age this big: the modern shed
FARM_MIN_M2 = 20


def uk_farm(out, area, feats) -> dict:
    """Barns and farm buildings (asset_creator/commercial_kit/farm.py, PLAN step 4), a
    lookup: a farm-tagged building (FARM_TAGS) that is no one's home (a barn conversion
    stays a home) and not listed (the listed maker's) is the old timber barn when its
    age (types.csv era / type, outbuildings too) is pre-1930, the 1970s-90s shed when
    post-1950, and with no age the shed from FARM_SHED_M2 up, the barn below it. The
    barn's roof corrugated iron or still clay tile (by id), pitched (`pitched_roof`, roof_plan); no windows (`blank`); the
    barn's cart doors or the shed's sliding door on its front (`loading`, laid where
    Build builds a barn)."""
    import hashlib
    ages = {}
    with open(os.path.join(area, "types.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ages[r["osm_id"]] = r
    def farm(p):
        return (p.get("building") or "") in FARM_TAGS and p.get("knox_dwelling") not in ("house", "flats")
    # a building's size over all its pieces (a 400 m2 barn cut in two is still the shed)
    size = Counter()
    for ft in feats:
        if farm(ft["properties"]):
            size[_building(ft["properties"])] += outline_m2(ft)
    doors = _biggest(feats, farm)
    n = Counter()
    for ft in feats:
        p = ft["properties"]
        if not farm(p):
            continue
        oid, b = str(p.get("osm_id")), _building(p)
        r = ages.get(str(p.get("parent_osm_id") or oid)) or ages.get(oid) or {}
        if r.get("listed_grade") or size[b] < FARM_MIN_M2:
            continue
        e = out.setdefault(oid, {})
        if r.get("era") in OLD_ERAS or r.get("type") in OLD_TYPES:
            shed = False
        elif r.get("era") in MODERN_ERAS:
            shed = True
        else:
            shed = size[b] >= FARM_SHED_M2
        h = int(hashlib.md5(b.encode()).hexdigest()[:8], 16)
        style = "uk_shed_yorkshire" if shed else ("uk_barn_black_tile" if h % 5 < 2 else "uk_barn_black_iron")
        e.update(style=style, blank=True, pitched_roof=True)
        if doors.get(b) == oid:
            e["loading"] = {"kit": "shed_yorkshire" if shed else "barn_black", "kind": "barn"}
        n[style] += 1
    return dict(n)


def uk_pubs(out, feats) -> dict:
    """Pubs (asset_creator/commercial_kit, PLAN step 5): a building with a pub's point
    whose pub has its hanging sign (asset_creator/src/pubs.py PUBS, by its name's slug)
    gets that pub's fascia in its brewery's livery (names.py pub_<key>, fascias.py) and
    a traditional front in the brewery's paint on its shop front's units, and its biggest unit a beer garden (`beer_garden`: Build's
    beer_gardens.py stands picnic benches on the open ground behind it)."""
    built = os.path.join(COMMERCIAL_KIT, "built", "fascias.json")
    if not os.path.exists(built):
        return {}
    tiles = json.load(open(built, encoding="utf-8"))["signs"]
    sys.path.insert(0, os.path.normpath(os.path.join(COMMERCIAL_KIT, "..", "src")))
    sys.path.insert(0, COMMERCIAL_KIT)
    from pubs import slug
    import names as NM
    import shopfronts as SF
    n, biggest = Counter(), {}
    for ft in feats:
        p = ft["properties"]
        names_ = [u.get("name") for u in [p] + list(p.get("knox_uses") or []) if u.get("amenity") == "pub" and u.get("name")]
        keys = [f"pub_{slug(nm)}" for nm in names_ if f"pub_{slug(nm)}" in tiles]
        if not keys:
            continue
        oid = str(p.get("osm_id"))
        e = out.get(oid) or {}
        if e.get("uk_front"):
            e["fascia"] = {"sign": keys[0], **tiles[keys[0]]}
            # a traditional front in the brewery's paint (an aluminium shop front read as a shop)
            e["uk_front"] = SF.entry(NM.brewery_of(keys[0][4:])[5])
            n["fascias"] += 1
        parent = str(p.get("parent_osm_id") or oid)
        if p.get("knox_dwelling") not in ("house",) and outline_m2(ft) > biggest.get(parent, (0, None))[0]:
            biggest[parent] = (outline_m2(ft), oid)
    for _m2, oid in biggest.values():
        out.setdefault(oid, {})["beer_garden"] = True
        n["beer_gardens"] += 1
    return dict(n)


def uk_exchanges(out, rows, feats) -> dict:
    """Telephone exchanges (asset_creator/commercial_kit, PLAN step 5): a building named
    one, or tagged telecom=exchange / building=telephone_exchange: the 1960s GPO box -
    buff brick to a flat roof (uk_exchange_buff, `flat_roof`), on its biggest unit a
    galvanised van door on a wall the camera sees and TELEPHONE EXCHANGE in cast letters
    beside it (`loading` with `signs`, as a repair garage's)."""
    built = os.path.join(COMMERCIAL_KIT, "built", "signs.json")
    if not os.path.exists(built):
        return {}
    letters = json.load(open(built, encoding="utf-8"))["signs"].get("exchange")
    ids = set()
    for ft in feats:
        p = ft["properties"]
        tagged = [p] + list(p.get("knox_uses") or [])
        if any("telephone exchange" in (u.get("name") or "").lower() or u.get("telecom") == "exchange"
               or u.get("building") == "telephone_exchange" for u in tagged):
            ids.add(str(p.get("parent_osm_id") or p.get("osm_id")))

    def exchange(p):
        return (str(p.get("parent_osm_id") or p.get("osm_id")) in ids and not _listed(rows, p)
                and p.get("knox_dwelling") not in ("house", "flats"))   # "The Old Telephone Exchange" stays a home
    doors = {str(p_id) for p_id in _biggest(feats, exchange).values()}
    n = Counter()
    for ft in feats:
        p = ft["properties"]
        oid = str(p.get("osm_id"))
        if not exchange(p):
            continue
        e = out.setdefault(oid, {})
        for k in ("loading", "blank_upper", "upper_wall", "upper_sides", "shopfront", "uk_front", "fascia"):
            e.pop(k, None)
        e.update(style="uk_exchange_buff", flat_roof=True)
        if oid in doors and letters:
            e["loading"] = {"kit": "exchange_buff", "kind": "*", "seen": True,
                            "signs": [{"tiles": letters, "where": "after"}]}
        n["exchanges"] += 1
    return dict(n)


GARAGE_USES = (("shop", "car_repair"), ("craft", "car_repair"), ("shop", "car"), ("shop", "tyres"))


def uk_garages(out, rows, feats) -> dict:
    """Repair garages (asset_creator/commercial_kit, PLAN step 5): a building with a car
    repair, tyre or motor dealer's point in it (its own tags, its knox_uses, or its
    parent outline's: Kwik Fit's was cut into three) that is no one's home and not listed:
    plain red brick (uk_garage_brick), a cream roller shutter on a wall the camera sees
    (`loading`, any kind: Build's own for these is garage, shop or works), and beside it
    the MOT board and the chain's fascia (`signs`: signs.py, fascias.py's tiles). Its shop
    front and shop sign are dropped (a workshop's wall, not a glass front)."""
    built = os.path.join(COMMERCIAL_KIT, "built", "signs.json")
    if not os.path.exists(built):
        return {}
    sys.path.insert(0, COMMERCIAL_KIT)
    import names as NM
    mot = json.load(open(built, encoding="utf-8"))["signs"].get("mot")
    if not mot:
        return {}
    garage_ids, signed = set(), {}
    # (each unit of a cut outline carries the point: the shop unit's, then the biggest)
    for ft in sorted(feats, key=lambda f: (f["properties"].get("knox_dwelling") == "shop", outline_m2(f)), reverse=True):
        p = ft["properties"]
        parent = str(p.get("parent_osm_id") or p.get("osm_id"))
        if any(u.get(k) == v for u in [p] + list(p.get("knox_uses") or []) for k, v in GARAGE_USES):
            garage_ids.add(parent)
            if p.get("knox_dwelling") not in ("house", "flats"):
                signed.setdefault(parent, str(p.get("osm_id")))     # its signs on the unit with the point
    for ft in sorted(feats, key=outline_m2, reverse=True):      # ...else on its biggest unit
        p = ft["properties"]
        parent = str(p.get("parent_osm_id") or p.get("osm_id"))
        if parent in garage_ids and p.get("knox_dwelling") not in ("house", "flats"):
            signed.setdefault(parent, str(p.get("osm_id")))
    n = Counter()
    for ft in feats:
        p = ft["properties"]
        oid = str(p.get("osm_id"))
        if (str(p.get("parent_osm_id") or oid) not in garage_ids or p.get("knox_dwelling") in ("house", "flats")
                or _listed(rows, p)):
            continue
        e = out.setdefault(oid, {})
        fascia = e.get("fascia") if isinstance(e.get("fascia"), dict) else None
        signs = [{"tiles": mot, "where": "after"}] if signed.get(str(p.get("parent_osm_id") or oid)) == oid else []
        board = next((v for v in (fascia or {}).get("versions", []) if v["length"] <= 3), None)
        if signs and board and fascia.get("sign") in NM.CHAINS:
            signs.append({"tiles": {"W": board["W"], "N": board["N"]}, "where": "before"})
        for k in ("shopfront", "uk_front", "fascia", "flat_roof"):
            e.pop(k, None)
        e.update(style="uk_garage_brick", blank_upper=True,
                 loading={"kit": "garage_brick", "kind": "*", "seen": True, "signs": signs})
        n["garages"] += 1
        n["mot"] += bool(signs)
        n["chain"] += len(signs) > 1
    return dict(n)


CAR_PARK_REACH_M = 80      # a superstore faces the car parks within this of it


def car_park_side(ft, parks, proj) -> str | None:
    """The side of a building (game N/E/S/W) the car parks near it lie on: each park
    within CAR_PARK_REACH_M, weighted by its size over its distance, summed as a
    direction from the building's middle (in the game's turned frame)."""
    import math
    ring = ft["geometry"]["coordinates"][0] if ft["geometry"]["type"] == "Polygon" else []
    if not ring:
        return None
    lat = sum(c[1] for c in ring) / len(ring)
    lon = sum(c[0] for c in ring) / len(ring)
    bx, by = proj.to_px(lat, lon)
    k = 111320 * math.cos(math.radians(lat))
    vx = vy = 0.0
    for plat, plon, m2 in parks:
        d = math.hypot((plat - lat) * 111320, (plon - lon) * k)
        if d > CAR_PARK_REACH_M or d < 1:
            continue
        px, py = proj.to_px(plat, plon)
        n = math.hypot(px - bx, py - by) or 1
        vx += (px - bx) / n * m2 / d
        vy += (py - by) / n * m2 / d
    if not vx and not vy:
        return None
    return ("E" if vx > 0 else "W") if abs(vx) >= abs(vy) else ("S" if vy > 0 else "N")


def _parks(map_dir):
    """Car parks of the map's areas (amenity=parking): (lat, lon, m2) of each."""
    path = os.path.join(map_dir, f"{os.path.basename(os.path.normpath(map_dir))}_areas.geojson")
    if not os.path.exists(path):
        return []
    out = []
    for f in json.load(open(path, encoding="utf-8"))["features"]:
        if f["properties"].get("amenity") != "parking":
            continue
        polys = f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]]
        for poly in polys:
            ring = poly[0]
            g = {"geometry": {"type": "Polygon", "coordinates": [ring]}}
            out.append((sum(c[1] for c in ring) / len(ring), sum(c[0] for c in ring) / len(ring), outline_m2(g)))
    return out


def _projector(map_dir):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from buildings import blocks as B
    with open(os.path.join(map_dir, f"{os.path.basename(os.path.normpath(map_dir))}_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    bb = info["bbox"]
    return B.Projector.build(bb["south"], bb["west"], bb["north"], bb["east"], info["meters_per_tile"],
                             info.get("rotation") or 0.0)


def uk_shopfronts(out, rows, feats, map_dir=None) -> dict:
    """Each shop's front and sign from asset_creator's commercial kit (PLAN step 1):
    `uk_front` (a finish of shopfronts_uk_01) and `fascia` (a names.SIGNS key, or
    none). The sign is names.sign_for's lookup (a chain from the shop's own name,
    else its trade's, at most one of each chain a town). The front: a traditional
    painted one on a listed or pre-1930 building unless a 1993 chain had refitted it
    (a chain's sign and not listed: aluminium), else aluminium; the paint by OSM id.
    Both are written with their tile names (shopfronts.entry, built/fascias.json).
    A supermarket of SUPERSTORE_M2 or more with no homes over it takes a superstore
    style (SUPERSTORE_STYLES: brick to the eaves - the user dropped the cladding, 2026-10-20),
    no windows above its ground floor (`blank_upper`), and its front on the side its car parks are
    (`front`, car_park_side; Build's regional_uk.front_of).
    A town whose OSM has no McDonald's gets one Spiffo's (`spiffos`: Build gives it
    the game's Spiffo's rooms, regional_uk.uses_with): its Subway, else its biggest
    takeaway, else a cafe - its front on a face the camera sees, one business, a unit of an
    ordinary size first - in the game's
    own Spiffo's front and neon (SF.SPIFFOS_FRONT)."""
    if not os.path.isdir(COMMERCIAL_KIT):
        return {}
    sys.path.insert(0, COMMERCIAL_KIT)
    import names as NM
    import shopfronts as SF
    trad = [f[0] for f in SF.FINISHES if f[1] == "trad"]
    modern = [f[0] for f in SF.FINISHES if f[1] == "modern"]
    built = os.path.join(COMMERCIAL_KIT, "built", "fascias.json")      # fascias.py's tiles for each sign
    fascia_tiles = json.load(open(built, encoding="utf-8"))["signs"] if os.path.exists(built) else {}
    big = os.path.join(COMMERCIAL_KIT, "built", "fascias_big.json")    # ...and the superstores' big signs
    big_tiles = json.load(open(big, encoding="utf-8"))["signs"] if os.path.exists(big) else {}
    taken: set = set()
    n = Counter()
    parks = proj = None
    spiffo_pool = []          # (front unseen, several businesses, too big, listed, rank, size off a unit's, osm id)
    for ft in sorted(feats, key=lambda f: str(f["properties"].get("osm_id"))):
        p = ft["properties"]
        if p.get("knox_dwelling") not in ("shop", "shop_flats"):
            continue
        oid = str(p.get("osm_id"))
        parent = str(p.get("parent_osm_id") or oid)
        r = rows.get(parent) or rows.get(oid) or {}
        sign = NM.sign_for(oid, NM.shop_tags(p), p.get("name") or "", taken)
        listed = bool(r.get("listed_grade"))
        old = listed or r.get("type") in OLD_TYPES or r.get("era") in OLD_ERAS
        chain = bool(sign) and not sign.startswith("ind_")
        h = NM._hash(oid, "front")
        finish = trad[h % len(trad)] if old and (listed or not chain) else modern[h % len(modern)]
        e = out.setdefault(oid, {})
        # the tiles themselves, inlined (as apply.py does a landmark's): Build reads nothing else
        e["uk_front"] = SF.entry(finish)
        if sign and sign in fascia_tiles:
            e["fascia"] = {"sign": sign, **fascia_tiles[sign]}
        uses = p.get("knox_uses") or []
        # A superstore: brick to its eaves (supermarket.py's kits; user, 2026-10-20: no cladding)
        if (p.get("knox_dwelling") == "shop" and any(u.get("shop") == "supermarket" for u in uses)
                and outline_m2(ft) >= SUPERSTORE_M2 and not listed):
            e["style"] = SUPERSTORE_STYLES[h % len(SUPERSTORE_STYLES)]
            e["blank_upper"] = True                     # no windows above its ground floor
            e["superstore"] = True                      # glass only round its entrance (tbx.py)
            e["flat_roof"] = True                       # one flat roof (roof_plan.py)
            # its lorry dock round the back (industrial.py's shutter in its brick; Build
            # regional_uk.loading_for: the side away from its front, never the car park's)
            e["loading"] = {"kit": e["style"][3:], "kind": "shop", "back": True}
            if sign in big_tiles:
                # its name big on the cladding, over the ordinary fascia's versions
                e["fascia"] = {"sign": sign, "plain": None,
                               "versions": big_tiles[sign] + (fascia_tiles.get(sign) or {}).get("versions", [])}
            if map_dir:
                if parks is None:
                    parks, proj = _parks(map_dir), _projector(map_dir)
                side = car_park_side(ft, parks, proj)
                if side:
                    e["front"] = side                   # a superstore faces its car park, not the road
            n["superstores"] += 1
        names = " ".join([p.get("name") or ""] + [u.get("name") or "" for u in uses]).lower()
        amen = {(u.get("amenity") or "") for u in uses}
        rank = 0 if "subway" in names else 1 if "fast_food" in amen else 2 if "cafe" in amen else None
        # not a hotel, hall or bar (no fascia), not a chain under its own name (the
        # Subway's sign is a 1993 stand-in chain's, which it may still give up)
        own_chain = any(NM.chain_of(x) for x in [p.get("name")] + [u.get("name") for u in uses])
        if rank is not None and sign and not own_chain:
            # One business in one ordinary unit, its size nearest a high street unit's
            # (Odiham's first pick was a 34-square parade of two businesses, 2026-10-20).
            m2 = outline_m2(ft)
            businesses = sum(1 for u in uses if u.get("name"))
            # its front where the game's camera sees it (south or east): on a north or
            # west wall the neon is seen from inside only (Odiham's first unit faced west)
            hidden = p.get("knox_front") not in ("S", "E")
            spiffo_pool.append((hidden, businesses > 1, m2 > SPIFFOS_MAX_M2, listed, rank,
                                abs(m2 - SPIFFOS_UNIT_M2), oid, sign))
        n["trad" if finish in trad else "modern"] += 1
        n["signed"] += bool(sign)
    # A Spiffo's, the game's own chain, in a town with no McDonald's (user,
    # 2026-10-20): its Subway (none in 1993), else its biggest takeaway or cafe.
    if "spiffos" not in taken and spiffo_pool:
        *_rank, oid, _sign = min(spiffo_pool)
        e = out[oid]
        e["spiffos"] = True
        e["uk_front"] = SF.entry("spiffos")                 # the game's own Spiffo's walls and neon
        e["fascia"] = dict(SF.SPIFFOS_FASCIA)
        n["spiffos"] = oid
    return dict(n)


def main() -> int:
    area, palette_path, map_dir = sys.argv[1:4]
    with open(palette_path, encoding="utf-8") as f:
        pal = json.load(f)
    roofs = {}
    roofs_path = os.path.join(area, "roofs.csv")
    if os.path.exists(roofs_path):
        with open(roofs_path, encoding="utf-8") as f:
            roofs = {r["osm_id"]: r for r in csv.DictReader(f)}
    def options_for(r):
        """The styles a building of this kind may take, and why."""
        named = next((s for word, s in pal.get("name_styles", {}).items()
                      if word in (r["list_name"] or "").upper()), None)
        if named:
            return [named], "name"
        if r["listed_grade"]:
            return pal["listed_styles"], "listed"
        if r["street"] == "High Street":
            return pal["high_street_unlisted"], "high_street"
        # The sub-type as it stood in 1993 (area/subtypes.py), else the older type.
        look = r.get("look") or r.get("subtype")
        if look and look in pal.get("subtype_styles", {}):
            return pal["subtype_styles"][look], look
        if r["type"] in pal["type_styles"]:
            return pal["type_styles"][r["type"]], r["type"]
        return [], "knoxmap"

    out, why, rows = {}, {}, {}
    with open(os.path.join(area, "types.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["form"] == "outbuilding":
                continue
            rows[r["osm_id"]] = r
            entry = {}
            options, reason = options_for(r)
            if options:
                # an era-unknown look: one style for its whole development (a street of them alike)
                entry["style"] = pick(options, r["dev"] if reason.startswith("unknown_") and r.get("dev") else r["osm_id"])
            roof = roofs.get(r["osm_id"])
            if roof and roof.get("roof_shape") not in NO_LIDAR and roof.get("storeys_lidar"):
                entry["levels"] = int(roof["storeys_lidar"])
            if entry:
                out[r["osm_id"]] = entry
                why[reason] = why.get(reason, 0) + 1

    # Units split out of a long outline (split_units.py): each takes its
    # parent's kind, but neighbours are styled apart so the row reads as
    # separate buildings. Pieces (blocks.py) keep the parent's look. Where the parent's kind allows only one style the
    # palette's unit_styles give the variety.
    name = os.path.basename(os.path.normpath(map_dir))
    gj_path = os.path.join(map_dir, f"{name}_buildings.geojson")
    units = 0
    if os.path.exists(gj_path):
        with open(gj_path, encoding="utf-8") as f:
            feats = json.load(f)["features"]
        for ft in feats:
            p = ft["properties"]
            parent = str(p.get("parent_osm_id") or "")
            if not parent or parent not in rows:
                continue
            if p.get("piece"):
                # A piece of one building, cut so the lots do not overlap
                # (blocks.py): it looks like the rest of that building.
                out[str(p["osm_id"])] = dict(out.get(parent, {}))
                units += 1
                continue
            options, reason = options_for(rows[parent])
            # A housing kit's style (by_id_only) stays: the generic unit_styles
            # took the bays, porches and plinths off every split terrace (QA, 2026-10-19).
            kit = {s["name"] for s in pal["palette"].get("add", []) if s.get("by_id_only")}
            if len(options) < 2 and not set(options) & kit:
                options = pal.get("unit_styles") or options
            entry = {k: v for k, v in out.get(parent, {}).items() if k != "style"}
            if options:
                entry["style"] = options[(int(parent) + int(p.get("unit", 0))) % len(options)]
            out[str(p["osm_id"])] = entry
            units += 1
        if units:
            why["unit"] = units
    # Buildings OSM lacks, drawn from the LiDAR (blocks.py): no type of their
    # own, so the style most of the typed buildings around them have - an
    # estate is built at one time, in one look.
    lidar_built = 0
    if os.path.exists(gj_path):
        import math
        with open(os.path.join(area, "buildings.csv"), encoding="utf-8") as f:
            where = {r["osm_id"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)
                     if r.get("lat") and r.get("lon")}
        typed = [(where[k], v["style"]) for k, v in out.items() if k in where and v.get("style")]
        for ft in feats:
            p = ft["properties"]
            if p.get("source") != "lidar":
                continue
            ring = ft["geometry"]["coordinates"][0]
            lat = sum(c[1] for c in ring) / len(ring)
            lon = sum(c[0] for c in ring) / len(ring)

            def dist(ll):
                return math.hypot((ll[0] - lat) * 111320, (ll[1] - lon) * 111320 * math.cos(math.radians(lat)))
            near = sorted(typed, key=lambda t: dist(t[0]))[:NEIGHBOURS]
            near = [st for ll, st in near if dist(ll) <= NEIGHBOUR_M]
            style = Counter(near).most_common(1)[0][0] if near else (pal.get("unit_styles") or [None])[0]
            if style:
                out[str(p["osm_id"])] = {"style": style}
                lidar_built += 1
        if lidar_built:
            why["lidar"] = lidar_built
    # A shop's front (shops.py: knox_dwelling): a painted storefront in one of
    # the palette's heritage colours, picked by OSM id like the wall styles,
    # and a unit by its parent and place in the row, so the shops of a row
    # differ. Build uses it where it draws Erika's storefronts.
    shopfronts = 0
    if os.path.exists(gj_path) and pal.get("shopfronts"):
        for ft in feats:
            p = ft["properties"]
            if p.get("knox_dwelling") not in ("shop", "shop_flats"):
                continue
            parent = str(p.get("parent_osm_id") or p.get("osm_id"))
            if not parent.isdigit():
                continue
            key = int(parent) + int(p.get("unit") or 0)
            out.setdefault(str(p["osm_id"]), {})["shopfront"] = pal["shopfronts"][key % len(pal["shopfronts"])]
            shopfronts += 1
    uk_fronts = uk_shopfronts(out, rows, feats, map_dir) if os.path.exists(gj_path) else {}
    units_made = uk_units(out, rows, feats) if os.path.exists(gj_path) else {}
    farm_made = uk_farm(out, area, feats) if os.path.exists(gj_path) else {}
    garages_made = uk_garages(out, rows, feats) if os.path.exists(gj_path) else {}
    pubs_made = uk_pubs(out, feats) if os.path.exists(gj_path) else {}
    exchanges_made = uk_exchanges(out, rows, feats) if os.path.exists(gj_path) else {}
    trades_made = shop_trades(out)
    # A building the LiDAR did not see (built since 2019, or outside the
    # area's data) had no storeys of its own, and Build guessed: an untagged
    # 190 m2 building became five storeys of flats in a two-storey town. It
    # takes the storeys of the buildings round it instead (the median of the
    # nearest GUESS_NEIGHBOURS within NEIGHBOUR_M, at most GUESS_MAX_LEVELS;
    # GUESS_DEFAULT with none near). Churches and schools are left to the
    # roof planner and Build.
    guessed = 0
    if os.path.exists(gj_path):
        import math
        import numpy as np

        def centre(ft):
            ring = ft["geometry"]["coordinates"][0]
            return (sum(c[1] for c in ring) / len(ring), sum(c[0] for c in ring) / len(ring))
        known = [(centre(ft), out[str(ft["properties"].get("osm_id"))]["levels"]) for ft in feats
                 if ft["geometry"]["type"] == "Polygon" and "levels" in out.get(str(ft["properties"].get("osm_id")), {})]
        if known:
            pts = np.array([[ll[0] * 111320, ll[1] * 111320 * math.cos(math.radians(ll[0]))] for ll, _l in known])
            lv = np.array([lvl for _ll, lvl in known])
        for ft in feats:
            p = ft["properties"]
            oid = str(p.get("osm_id"))
            if ft["geometry"]["type"] != "Polygon" or "levels" in out.get(oid, {}) or p.get("building:levels"):
                continue
            if (p.get("amenity") or "") in ("place_of_worship", "school") or \
                    (p.get("building") or "") in ("church", "chapel", "cathedral", "school"):
                continue
            levels = GUESS_DEFAULT
            if known:
                lat, lon = centre(ft)
                d = np.hypot(pts[:, 0] - lat * 111320, pts[:, 1] - lon * 111320 * math.cos(math.radians(lat)))
                near = np.argsort(d)[:GUESS_NEIGHBOURS]
                near = near[d[near] <= NEIGHBOUR_M]
                if len(near):
                    levels = int(round(float(np.median(lv[near]))))
            out.setdefault(oid, {})["levels"] = max(1, min(GUESS_MAX_LEVELS, levels))
            guessed += 1
    # Arches: the squares where a road or path passes through a building
    # (passages.py), left out of its ground floor by Build.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import passages as P
    from buildings import blocks as B
    with open(os.path.join(map_dir, f"{os.path.basename(os.path.normpath(map_dir))}_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    bb = info["bbox"]
    proj = B.Projector.build(bb["south"], bb["west"], bb["north"], bb["east"], info["meters_per_tile"],
                             info.get("rotation") or 0.0)
    arches = P.passages(map_dir, info, proj)
    for oid, cells in arches.items():
        out.setdefault(oid, {})["passage"] = cells
    parts = house_parts(out, pal, area, feats if os.path.exists(gj_path) else [])
    parts["rooms"] = house_rooms(out, area, feats if os.path.exists(gj_path) else [])
    regional = {"palette": pal["palette"], "buildings": out}
    # a forecourt's tiles (commercial_kit/petrol.py), inlined for Build's pumps.py
    for key, built_name in (("fuel", "fuel.json"), ("pub_props", "pub_props.json")):
        path = os.path.join(COMMERCIAL_KIT, "built", built_name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                regional[key] = json.load(f)    # a forecourt's, a beer garden's tiles (pumps.py, beer_gardens.py)
    path = os.path.join(map_dir, "regional.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(regional, f, indent=1)
    print(f"wrote {path}: {len(out)} buildings")
    print(f"  passages through buildings (arches): {len(arches)} buildings, "
          f"{sum(len(c) for c in arches.values())} squares")
    print("  style chosen by:", why)
    print("  house parts (upper floor walls, porches, bays):", parts)
    print("  styles:", dict(Counter(e.get("style") for e in out.values())))
    print(f"  shop fronts: {shopfronts}", dict(Counter(e["shopfront"] for e in out.values() if "shopfront" in e)))
    print("  UK shop fronts (commercial_kit):", uk_fronts)
    print("  industrial units (commercial_kit):", units_made)
    print("  farm buildings (commercial_kit):", farm_made)
    print("  repair garages (commercial_kit):", garages_made)
    print("  pubs (commercial_kit):", pubs_made)
    print("  telephone exchanges (commercial_kit):", exchanges_made)
    print("  shop trades for Build's rooms (names.py):", trades_made)
    print("  with storeys:", sum(1 for e in out.values() if "levels" in e),
          f"({guessed} of them from the buildings round them: the LiDAR did not see them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
