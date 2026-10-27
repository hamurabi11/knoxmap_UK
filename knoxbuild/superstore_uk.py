"""Large supermarkets (ROOMS.md section 18): a superstore's inside, from the game's
own Gigamarts (B42: seven in Knox County; their rooms, furniture and sprites read
from the map files, 2026-10-26).

A template, not the room solver: a superstore is a closed pattern. In the store's
own frame (its front the car park or street, `a` along the front, `d` in from it):

    d   0..   the entrance: trolley bays and baskets by the doors, the customer
              service desk, the café and the customer toilets at the far end
        +2    the checkouts across the store (counter and till, two squares of belt)
        +6    the sales floor in strips along the front, from the entrance end:
              produce (crates and stands), drinks, the grocery aisles (gondolas),
              frozen (ice chests, freezers), household, pharmacy, clothing
              (rails, changing cubicles) - the extras by size (TIERS)
        back  the counters: butcher, bakery, deli, each its own room with its
              display counters on the sales floor in front; chillers along the rest
        back  back of house (BOH): the stock room's racking, the cold store, the
              janitor, the staff stairs; the lorry dock in its back wall (Build's
              loading_for / lay_garage)

One storey (user, 2026-10-26), with a staff floor over the back of house: a
corridor along the back wall with the stairs, offices, the cash office, a canteen,
lockers and toilets. `roof` gives Build the matching roof: the whole store flat at
one storey, the back-of-house strip at two.

The sales floor's rooms have no walls between them (Build's open edges, as the
mall's units open onto its concourse); each is its own game room, so each takes
its own loot allowance (the game caps loot per room; its own Gigamart floors are
433-1,130 m2 rooms, ROOM_MAX here).
"""
from __future__ import annotations

import collections
import json
import os
import random

# ---- tables -------------------------------------------------------------------

# By floor area (m2): what a store of that size has. A Gigamart is 600-1,100 m2.
# (floor area, tier, its counters along the back, in order from the entrance's end)
TIERS = [(0, "supermarket", ["bakery", "butcher"]),
         (2500, "superstore", ["bakery", "butcher", "deli"])]
# The sales floor as department blocks (user, 2026-10-26: "more variety ... a cafe should be
# one of them"): (block, its game room, from, to - shares of the floor's length from the
# entrance's end, by floor area (an L-shaped store's narrow arm is a small share) - and
# which half of the floor's depth: front, back or all).
BLOCKS = {
    "supermarket": [("produce", "grocery", 0, .22, "front"), ("household", "housewarestore", 0, .22, "back"),
                    ("aisles", "gigamart", .22, .66, "all"), ("drinks", "liquorstore", .66, .84, "all"),
                    ("frozen", "gigamart", .84, 1, "all")],
    # (beers, wines and spirits - "pretty big in every UK supermarket" (user) - a block of its own)
    "superstore": [("produce", "grocery", 0, .14, "front"), ("pharmacy", "pharmacy", 0, .14, "back"),
                   ("aisles", "gigamart", .14, .49, "all"), ("drinks", "liquorstore", .49, .62, "all"),
                   ("frozen", "gigamart", .62, .76, "all"),
                   ("household", "housewarestore", .76, .89, "front"),
                   ("cafe", "cafe", .89, 1, "front"), ("clothing", "clothingstore", .76, 1, "back")],
}
# Each block's fittings: the runs standing in it (in turn, run by run), and what lines its walls.
AISLES = {"aisles": ["shop_shelf", "shop_shelf", "shop_shelf_white", "shop_shelf", "shop_freezer"],
          "drinks": ["shop_shelf", "shop_fridge_white", "shop_shelf", "shop_shelf_white"],     # wine, beer coolers
          "household": ["shop_shelf_white", "shop_shelf"],
          "pharmacy": ["shop_shelf_white"], "frozen": ["shop_freezer"]}
WALLS = {"aisles": ["shop_fridge_open", "shop_shelf"],
         "drinks": ["shop_fridge_double", "shop_fridge_white", "shop_shelf", "shop_fridge_double"],
         "frozen": ["shop_fridge_double", "shop_fridge_white"], "household": ["shop_shelf"],
         "pharmacy": ["shop_shelf_white"], "produce": ["shop_fridge_open"], "clothing": ["shop_shelf"]}
ENDS = ["uk_sale_sign", "shop_bin", "uk_price_sign", "crate"]   # at the end of each run: the promotions
# Each counter: its game room, its width, its display, and its kit along its back wall
# (the game's Gigamart bakery: deck ovens, steel worktops and tables, a sink, a steel fridge).
COUNTERS = {
    "bakery": ("bakery", 9, "uk_bakery_case",
               ["oven_commercial", "oven_commercial", "counter_2", "counter_2", "kitchen_sink_dark",
                "uk_steel_fridge", "counter_2"]),
    "butcher": ("butcher", 9, "shop_display",
                ["counter_2", "counter_2", "kitchen_sink_dark", "uk_steel_fridge", "counter_2", "uk_steel_fridge"]),
    "deli": ("gigamartkitchen", 8, "shop_display",
             ["counter_2", "oven_commercial", "counter_2", "kitchen_sink_dark", "uk_steel_fridge"]),
}
ENTRANCE = 2         # squares deep: the doors, trolleys, baskets, the service desk
CHECKOUT = 6         # ...then the checkouts (CHECKOUTS: up to 5 squares deep) and a row to queue in
COUNTER_DEPTH = 5    # the counter rooms: the display, the staff's aisle, the kit
BAY = 8              # squares from one brick pier to the next along the glazed front
SHOPFRONT_HALF = 6   # squares either side of the doors left to the entrance's shop front (regional_uk)
ROOF_BAND = 4        # squares deep: the tiled band along the front (user, from Sainsbury's Alton; 0: none)
BOH = (8, 16)        # the back of house, squares deep (a fifth of the store's depth)
CAFE_KITCHEN = 3     # squares deep: the cafe's kitchen, the block's whole width along its back
# ...then its servery: the back counter against the kitchen wall, the staff's aisle, the front
# counter facing the tables (a flap at its wall end), and a row to queue in
CAFE_BACK = ["counter_2", "uk_coffee", "counter_2", "kitchen_sink_dark", "uk_steel_fridge", "counter_2"]
CAFE_FRONT = ["counter_2", "uk_bakery_case", "counter_2", "counter_2"]
# a tiled floor in the cafe (user: "a nice tile instead of wooden floor"): warm beige tiles
CAFE_FLOOR = "floors_interior_tilesandwood_01_23"
DOOR_TO = {"cafekitchen": "cafe"}
DROPPABLE = {"crate", "shelf", "shelf_1", "shelf_2", "metal_rack", "bookshelf", "shop_shelf", "shop_shelf_white",
             "shop_freezer", "shop_fridge", "shop_fridge_white", "shop_fridge_double", "shop_fridge_open", "uk_bin",
             "shop_bin", "uk_sale_sign", "uk_price_sign", "counter", "counter_6", "uk_produce_stand", "clothes_rack",
             "clothes_rack_small", "mannequin"}
CAFE_WAY = 2         # squares kept clear along the café's open sides     # a room's door into the one it serves
TOILETS = 6          # squares of the entrance band at its far end
ROOM_MAX = 1100      # m2: a sales room bigger than this is cut in two (a Gigamart's floor is up to 1,130)
STAFF_ROOMS = ["breakroom", "office", "office", "lockerroom", "bathroom", "bathroom", "storage", "office"]
STAFF_WIDE = 6
TROLLEY_BAYS = 4     # rails of the trolley bay inside the doors
POSTER_EVERY = 7     # squares of wall between posters
OPEN = {"gigamart", "grocery", "liquorstore", "housewarestore", "pharmacy", "clothingstore", "cafe"}
COUNTER_ROOMS = {"bakery", "butcher", "gigamartkitchen"}   # walled, open only along their counter

# Rooms the game has that Build did not (the Gigamart's own: its loot tables)
NEW_ROOMS = {
    "gigamart": ("FLOOR_TILE_PALE", "Gigamart", ["shop_shelf"], 28, 150, 176),
    "gigamartkitchen": ("FLOOR_TILE_PALE", "Gigamart Kitchen", ["counter", "kitchen_sink", "shop_fridge_double"], 200, 160, 120),
    "changeroom": ("FLOOR_WOOD", "Changing Room", ["mirror", "chair"], 200, 170, 140),
}
# Gigamart pieces Build's catalog lacked (sprites, how they run, from the game's maps)
NEW_PIECES = {
    "uk_belt": {"N": {"0,0": "industry_02_41"}, "W": {"0,0": "industry_02_40"}},          # checkout belt (41 runs along x)
    "uk_trolley_rail": {"W": {"0,0": "fixtures_railings_01_48"}, "N": {"0,0": "fixtures_railings_01_49"}},
    "uk_baskets": {"N": {"0,0": "location_shop_greenes_01_37"}, "W": {"0,0": "location_shop_greenes_01_38"}},
    "uk_produce_stand": {"N": {"0,0": "location_shop_generic_01_94", "1,0": "location_shop_generic_01_93"},
                         "W": {"0,0": "location_shop_generic_01_95", "0,1": "location_shop_generic_01_92"}},
    "uk_bakery_case": {"N": {"0,0": "location_shop_generic_01_34", "1,0": "location_shop_generic_01_35"}},
    "uk_ice_chest": {"W": {"0,0": "appliances_refrigeration_01_21", "0,1": "appliances_refrigeration_01_20"}},
    "uk_sale_sign": {"N": {"0,0": "location_shop_generic_01_70"}},
    "uk_price_sign": {"N": {"0,0": "location_shop_greenes_01_45"}, "W": {"0,0": "location_shop_greenes_01_46"}},
    "uk_scale": {"N": {"0,0": "location_shop_greenes_01_6"}},                               # the produce scale
    # (the game's sprites in the order N, W, S, E: read off them, 2026-10-26 - the coils were facing out)
    "uk_steel_fridge": {"N": {"0,0": "appliances_refrigeration_01_8"}, "W": {"0,0": "appliances_refrigeration_01_9"},
                        "S": {"0,0": "appliances_refrigeration_01_10"}, "E": {"0,0": "appliances_refrigeration_01_11"}},
    "uk_poster": {"N": {"0,0": "location_shop_accessories_01_27"}, "W": {"0,0": "location_shop_accessories_01_26"}},
    "uk_atm": {"N": {"0,0": "location_business_bank_01_64"}, "W": {"0,0": "location_business_bank_01_65"}},
    "uk_bin": {"N": {"0,0": "trashcontainers_01_20"}, "W": {"0,0": "trashcontainers_01_20"}},
    "uk_litter_bin": {"N": {"0,0": "trashcontainers_01_21"}, "W": {"0,0": "trashcontainers_01_21"}},
    "uk_coffee": {"N": {"0,0": "appliances_cooking_01_56"}, "W": {"0,0": "appliances_cooking_01_57"},
                  "E": {"0,0": "appliances_cooking_01_58"}, "S": {"0,0": "appliances_cooking_01_59"}},
}
# The game's own checkout unit, square by square, by the store's front (the side its
# doors are on): (along the front, in from the front, [sprites]), repeated every 4
# squares along the front. North: the Gigamart at cell 46,26 (a counter and till at the
# lane's end by the doors, two squares of belt, the cashier's stool, a rail down the
# lane); south: the same turned round; east: the north Rosewood Giga Mart (cell 31,43:
# grey lane barriers, a steel desk and till, three squares of belt). A west front takes
# the east unit's pieces mirrored.
CHECKOUTS = {
    "N": [(0, 0, ["fixtures_railings_01_114", "location_shop_generic_01_20"]),
          (1, 0, ["fixtures_railings_01_113", "location_shop_generic_01_21", "location_shop_accessories_01_1"]),
          (2, 0, ["fixtures_railings_01_112"]), (0, 1, ["industry_02_41"]),
          (1, 1, ["furniture_seating_indoor_01_51"]), (2, 1, ["fixtures_railings_01_112"]),
          (0, 2, ["industry_02_41"]), (2, 2, ["fixtures_railings_01_112"]), (2, 3, ["fixtures_railings_01_115"])],
    "S": [(0, 0, ["location_shop_generic_01_17"]),
          (1, 0, ["location_shop_generic_01_17", "location_shop_accessories_01_2"]),
          (2, 0, ["fixtures_railings_01_112"]), (0, 1, ["industry_02_41"]),
          (1, 1, ["furniture_seating_indoor_01_51"]), (2, 1, ["fixtures_railings_01_112"]),
          (0, 2, ["industry_02_41"]), (2, 2, ["fixtures_railings_01_112"])],
    "E": [(0, 0, ["construction_01_45", "fixtures_counters_01_39", "location_shop_accessories_01_3"]),
          (0, 1, ["construction_01_44", "furniture_seating_indoor_01_48"]), (0, 3, ["construction_01_45"]),
          (0, 4, ["construction_01_44", "construction_01_34", "fixtures_counters_01_35"]),
          (1, 0, ["industry_02_40"]), (1, 1, ["industry_02_45"]), (1, 2, ["industry_02_45"]),
          (1, 3, ["fixtures_counters_01_33"]), (1, 4, ["construction_01_33", "fixtures_counters_01_34"])],
    "W": [(0, 0, ["construction_01_45", "fixtures_counters_01_35", "location_shop_accessories_01_0"]),
          (0, 1, ["construction_01_44", "furniture_seating_indoor_01_48"]), (0, 3, ["construction_01_45"]),
          (0, 4, ["construction_01_44", "construction_01_34", "fixtures_counters_01_39"]),
          (1, 0, ["industry_02_40"]), (1, 1, ["industry_02_45"]), (1, 2, ["industry_02_45"]),
          (1, 3, ["fixtures_counters_01_33"]), (1, 4, ["construction_01_33", "fixtures_counters_01_34"])],
}
CHECKOUT_EVERY = 4
# An outcrop of the footprint off the store's main body (outcrops: not the entrance's lobby,
# not at the back) is a concession of the kind a 1990s superstore let by its doors (user: "this
# odd bit of geometry ... could probably be a cobbler"): (its trade, the game's room whose loot
# suits it), chosen by the building's seed.
CONCESSIONS = [("cobbler and key cutting", "toolstore"), ("dry cleaner", "laundry"),
               ("photo processing", "camerastore"), ("florist", "gardenstore"),
               ("newsagent and tobacco", "conveniencestore"), ("pharmacy", "pharmacy")]
CONCESSION_M2 = (6, 150)
DOCK_GAP = 4         # squares of wall between a store's two lorry docks
ATMS = 3             # standing ATMs on a wall between the checkouts and the doors
# hung on a wall, not standing on the floor
WALL_PIECES = {"uk_poster": "WallFurniture"}

# Department signs hung on a wall over each block (asset_creator/commercial_kit/dept_signs.py:
# built/dept_signs.json, by chain - the store's fascia sign - and department, at 2 to 6
# squares long; "generic" for any other store): the block's sign, by its zone
from ukdirs import assets as _assets          # asset_creator, if there is one (KnoxMap UK)
SIGNS_JSON = _assets("commercial_kit", "built", "dept_signs.json") or ""
SIGN_OF = {"produce": "produce", "drinks": "drinks", "aisles": "grocery", "frozen": "frozen",
           "household": "household", "pharmacy": "pharmacy", "cafe": "cafe", "clothing": "clothing",
           "bakery": "bakery", "butcher": "butcher", "deli": "deli", "entrance": "services"}
_SIGNS = []

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
STEP = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}


def register() -> None:
    """The Gigamart's rooms and pieces into Build's tables (in each worker too)."""
    from . import catalog as C
    from . import layout as L
    for kind, (floor, label, wish, r, g, b) in NEW_ROOMS.items():
        L.ROOM_STYLE.setdefault(kind, (getattr(C, floor), label, wish))
        C.ROOM_COLORS.setdefault(kind, f"{r} {g} {b}")
    for role, tiles in NEW_PIECES.items():
        C.FURNITURE.setdefault(role, tiles)
    for role, layer in WALL_PIECES.items():
        C.FURNITURE_LAYERS.setdefault(role, layer)


# ---- the store's frame --------------------------------------------------------

class Frame:
    """(a along the front, d in from it) <-> the building's (x, y)."""

    def __init__(self, w: int, h: int, front: str):
        self.w, self.h, self.front = w, h, front
        self.length = w if front in ("N", "S") else h
        self.depth = h if front in ("N", "S") else w

    def xy(self, a: int, d: int):
        f = self.front
        if f == "S":
            return a, self.h - 1 - d
        if f == "N":
            return a, d
        if f == "E":
            return self.w - 1 - d, a
        return d, a

    def ad(self, x: int, y: int):
        f = self.front
        if f == "S":
            return x, self.h - 1 - y
        if f == "N":
            return x, y
        if f == "E":
            return y, self.w - 1 - x
        return y, x


def _cells(mask, w, h):
    return {(x, y) for y in range(h) for x in range(w) if mask is None or mask[y][x]}


def lines(frame: Frame, cells: set):
    """The main front line and back line (in d): where most of the store's front
    and back stand (a lobby standing proud, or a wing behind, is off them)."""
    first, last = {}, {}
    for x, y in cells:
        a, d = frame.ad(x, y)
        first[a] = min(first.get(a, d), d)
        last[a] = max(last.get(a, d), d)
    front = collections.Counter(first.values()).most_common(1)[0][0]
    back = collections.Counter(last.values()).most_common(1)[0][0]
    return front, back


def boh_from(frame: Frame, cells: set) -> int:
    """The d from which the back of house runs (to the back wall and past it)."""
    front, back = lines(frame, cells)
    deep = back - front + 1
    return back - max(BOH[0], min(BOH[1], round(deep / 5))) + 1


def tier(cells: set):
    """(floor area, tier name, counters) for a store of this many squares."""
    area = len(cells)
    return [t for t in TIERS if area >= t[0]][-1]


def roof(x0: int, y0: int, mask, w: int, h: int, front: str, planned: dict | None) -> dict | None:
    """Build's roof plan for a superstore: flat at one storey over the store, two over
    the back-of-house strip (its staff floor), and a tiled band along its main front line
    (ROOF_BAND: a hipped range on the flat roof, as the 1980s-90s stores' pitched fronts),
    through regional_uk.roof_plan."""
    from . import regional_uk as U
    frame = Frame(w, h, front)
    cells = _cells(mask, w, h)
    if not cells:
        return planned
    d0 = boh_from(frame, cells)
    strip = [frame.xy(a, d) for a in range(frame.length) for d in range(d0, frame.depth)]
    xs, ys = [x for x, _ in strip], [y for _, y in strip]
    parts = [[x0, y0, w, h, "flat", 1],
             [x0 + min(xs), y0 + min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, "flat", 2]]
    if ROOF_BAND:
        f_line, _b = lines(frame, cells)
        band = [frame.xy(a, d) for a in range(frame.length) for d in range(f_line, f_line + ROOF_BAND)]
        bx, by = [x for x, _ in band], [y for _, y in band]
        parts.append([x0 + min(bx), y0 + min(by), max(bx) - min(bx) + 1, max(by) - min(by) + 1,
                      "WE" if front in ("N", "S") else "NS", 1])
    full = mask if mask is not None else [[1] * w for _ in range(h)]
    return U.roof_plan({"buildings": {"store": {"roof": parts, "hip": True}}}, "store", x0, y0, full) or planned


# ---- the layout ---------------------------------------------------------------

def plan_for_job(job: tuple, roof_plan: dict | None):
    """A superstore's Building (Build's _make_one), or None to leave it to Build."""
    (w, h, levels, commercial, seed, kind, mask, settings, style, label, path, street, retail,
        uses, hotel, entrances, profile, mapped, garage_door, preset_file, party, *planned) = job
    if not (style or {}).get("superstore") or not street:
        return None
    if not roof_plan or len(roof_plan.get("level_masks") or []) != 2:
        return None
    return layout(w, h, mask, street, roof_plan["level_masks"], uses or [], seed, profile, party or {},
                  chain=((style or {}).get("uk_fascia") or {}).get("sign"))


def layout(w, h, mask, front, level_masks, uses, seed=0, profile=None, party=None, chain=None):
    from . import layout as L
    register()
    rng = random.Random(seed)
    frame = Frame(w, h, front)
    cells = _cells(mask, w, h)
    upper = {(x, y) for y in range(h) for x in range(w) if level_masks[1][y][x]}
    f_line, b_line = lines(frame, cells)
    d_boh = boh_from(frame, cells)
    d_counter = d_boh - COUNTER_DEPTH
    d_check = f_line + ENTRANCE
    d_sales = d_check + CHECKOUT
    _area, name, counters = tier(cells)
    d_mid = (d_sales + d_boh) // 2

    # the entrance: the middle of the longest stretch of front wall
    ad = {c: frame.ad(*c) for c in cells}
    front_cells = sorted(a for (x, y), (a, d) in ad.items() if _outside(cells, x, y, front))
    door_a = _middle_of_longest(front_cells)
    length = frame.length
    toward_low = door_a < length / 2

    # each `a` by the share of the sales floor's area before it, from the entrance's end
    per_a = collections.Counter(a for (a, d) in ad.values() if d_sales <= d < d_boh)
    order = sorted(per_a) if toward_low else sorted(per_a, reverse=True)
    total, seen, share = sum(per_a.values()) or 1, 0, {}
    for a in order:
        share[a] = seen / total
        seen += per_a[a]

    def span(lo, hi):
        """The `a` whose share of the floor from the entrance's end is lo..hi (lo inclusive)."""
        inside = [a for a in order if lo <= share[a] < hi] or [order[min(len(order) - 1, int(lo * len(order)))]]
        return min(inside), max(inside) + 1

    def far(a, lo, hi):
        """How far `a` is in from the far end of lo..hi (0 at the far end)."""
        return hi - 1 - a if toward_low else a - lo
    blocks = [(b, k, *span(lo, hi), half) for b, k, lo, hi, half in BLOCKS[name]]
    whole = {b for b, _k, _lo, _hi, half in blocks if half == "all"}
    # the counters along the back of the sales floor, from the aisles' entrance end
    aisles = next(((lo, hi) for b, _k, lo, hi, _h in blocks if b == "aisles"), (0, length))
    cbounds, at = {}, (aisles[0] if toward_low else aisles[1])
    for c in counters:
        wide = COUNTERS[c][1]
        cbounds[c] = (at, at + wide) if toward_low else (at - wide, at)
        at = at + wide if toward_low else at - wide         # side by side (a gap between was a pocket)
    band = [a for (a, d) in ad.values() if f_line <= d < d_sales]
    band_lo, band_hi = (min(band), max(band) + 1) if band else (0, length)

    def block_of(a, d):
        half = "front" if d < d_mid else "back"
        for b, k, lo, hi, h_ in blocks:
            if lo <= a < hi and h_ in (half, "all"):
                return b, k, lo, hi
        return blocks[-1][:4]

    zone, kind_of = {}, {"boh": "grocerystorage", "entrance": "gigamart", "toilets": "bathroom",
                         "changing": "changeroom", "cafekitchen": "cafekitchen"}
    lobby = frame.xy(door_a, ad_first(ad, door_a))
    for n, part in enumerate(outcrops(frame, cells, ad, lobby, d_boh)):
        trade, kind = CONCESSIONS[(seed + n) % len(CONCESSIONS)]
        z = f"concession{n}"
        kind_of[z] = kind
        for c in part:
            zone[c] = z
    for c, (a, d) in ad.items():
        if c in zone:
            continue
        if d >= d_boh:
            zone[c] = "boh"
        elif d < d_sales:
            zone[c] = "toilets" if far(a, band_lo, band_hi) < TOILETS else "entrance"
        elif d >= d_counter and any(lo <= a < hi for lo, hi in cbounds.values()):
            z = next(k for k, (lo, hi) in cbounds.items() if lo <= a < hi)
            zone[c] = z
            kind_of[z] = COUNTERS[z][0]
        else:
            b, k, lo, hi = block_of(a, d)
            if b == "cafe" and d >= d_mid - CAFE_KITCHEN:
                zone[c] = "cafekitchen"
            elif b == "clothing" and far(a, lo, hi) < 6 and d >= d_boh - 2:
                zone[c] = "changing"
            else:
                # a block over the floor's whole depth is two rooms, front and back (the cross aisle)
                z = b + ("" if b not in whole else "_f" if d < d_mid else "_b")
                zone[c] = z
                kind_of[z] = k

    # rooms: each zone joined up and cut where it must be (ROOM_MAX, toilets, cubicles)
    groups = collections.defaultdict(set)
    for c, z in zone.items():
        groups[z].add(c)
    pieces = []                                 # (kind, cells, zone)
    for z, cs in groups.items():
        if z == "boh":
            pieces += _back_of_house(frame, cs, ad, front, counters, cbounds, d_boh)
            continue
        if z in ("toilets", "changing"):
            pieces += [(kind_of[z], p, z) for p in _cut_along(cs, ad, 3 if z == "toilets" else 2)]
            continue
        for part in _connected(cs):
            for p in _cut_to(part, ad, ROOM_MAX):
                pieces.append((kind_of[z], p, z))
    ground = _plan(L, w, h, pieces, cells)
    for r in ground.rooms:
        if r.kind == "cafe":
            r.landmark_floor = CAFE_FLOOR       # (tbx.py: a room's own floor)

    # upper: the staff floor over the back of house
    upieces = _staff_floor(frame, upper, ad, front)
    first = _plan(L, w, h, upieces, upper)

    # the stairs: along the back wall, at the far end from the dock's middle
    stairs = _stairs(frame, ground, first, cells, upper, front, toward_low)

    # ways through: open edges on the sales floor, doors elsewhere
    open_ids = {i for i, r in enumerate(ground.rooms, 1) if r.kind in OPEN}
    _open(ground, open_ids)
    # a counter room opens onto the floor along its counter only (user: "walls around them
    # except the side which has counters"), as the Rosewood Giga Mart's bakery and deli
    counter_ids = {i for i, r in enumerate(ground.rooms, 1) if r.kind in COUNTER_ROOMS}
    for (x, y), (a, d) in ad.items():
        if ground.grid[y][x] in counter_ids and d == d_counter:
            fx, fy = frame.xy(a, d - 1)
            if 0 <= fy < h and 0 <= fx < w and ground.grid[fy][fx] in open_ids:
                ground.open_edge.add(_edge(x, y, front))
    open_ids |= counter_ids
    entrance_room = ground.grid[frame.xy(door_a, f_line)[1]][frame.xy(door_a, f_line)[0]] or \
        next(iter(open_ids))
    _doors(ground, {entrance_room} | open_ids, stair_cells=set(_flight(stairs)) if stairs else set(),
           stairs=stairs, level=0, levels=2)
    _doors(first, {first.grid[_flight(stairs)[0][1]][_flight(stairs)[0][0]]} if stairs else {1},
           stair_cells=set(_flight(stairs)) if stairs else set(), stairs=stairs, level=1, levels=2)
    # the shop front: the front band's outside wall on the street side, its door
    ground.shop_front = {_edge(x, y, front) for (x, y) in cells if _outside(cells, x, y, front)
                         and ad[(x, y)][1] < d_check}
    dx, dy = frame.xy(door_a, ad_first(ad, door_a))
    door = _edge(dx, dy, front)
    if door in ground.shop_front:
        ground.doors.append(door)
    for plan in (ground, first):
        plan.kind = "shop"
        plan.windows = []
        plan.drawn = True
        plan.profile = profile
        plan.party = {e for e, n in (party or {}).items() if n > (0 if plan is ground else 1)}
    # the frontage glazed (front_glass), its bays' doors where they open onto the entrance band
    glass, bay_doors = front_glass(cells, ad, front, door_a, d_check)
    ground.windows = [_edge(*c, front) for c in glass]
    entrance = {i for i, z in ground.zones.items() if z == "entrance"}
    ground.doors += [_edge(*c, front) for c in bay_doors if ground.grid[c[1]][c[0]] in entrance]
    ground.mask = [[(x, y) in cells for x in range(w)] for y in range(h)]
    first.mask = [[(x, y) in upper for x in range(w)] for y in range(h)]

    # pillars on the open floor, as the game's Gigamarts have (pillars_uk), before the fittings
    from . import pillars_uk
    pillars_uk.add(ground)
    building = L.Building(width=w, height=h, storeys=[ground, first], profile=profile)
    for lvl, plan in enumerate(building.storeys):
        L._furnish(plan, random.Random(seed + 977 * lvl), stairs, front)
    pillars = {i for i, r in enumerate(ground.rooms, 1) if r.kind == "empty"}
    ground.furniture = [f for f in ground.furniture       # (a pillar is a room of its own: no switch in it)
                        if not (f[0] == L.SWITCH and ground.grid[f[2]][f[1]] in pillars)]
    _fit(L, ground, frame, ad, front, door_a, d_check, d_sales, d_counter, d_boh, f_line, rng, chain, d_mid)
    # whatever still can't be reached (a stock room's crate behind its racks...) is taken out:
    # a shelf no one can get to is no use
    _free_the_way(L, ground)
    if stairs:
        building.stairs.append(stairs)
        L._clear_for_stairs(ground, *stairs)
        L._clear_for_stairs(first, *stairs)
    L._clear_the_way(building)
    return building


def outcrops(frame: Frame, cells: set, ad: dict, lobby, d_boh: int) -> list:
    """Small parts of the footprint off the store's main body, each joined up: squares in
    front of its main front line or beyond its main side lines (where most of its front
    and sides stand), not behind the back of house's line, not the part holding the
    entrance (the lobby), CONCESSION_M2 in area. A store's shape alone says it: no list."""
    f_line, _b = lines(frame, cells)
    first, last = {}, {}
    for (a, d) in ad.values():
        first[d] = min(first.get(d, a), a)
        last[d] = max(last.get(d, a), a)
    a_lo = collections.Counter(first.values()).most_common(1)[0][0]
    a_hi = collections.Counter(last.values()).most_common(1)[0][0]
    off = {c for c, (a, d) in ad.items() if d < d_boh and (d < f_line or a < a_lo or a > a_hi)}
    out = []
    for part in _connected(off):
        if lobby in part or not CONCESSION_M2[0] <= len(part) <= CONCESSION_M2[1]:
            continue
        if min(max(x for x, _ in part) - min(x for x, _ in part), max(y for _, y in part) - min(y for _, y in part)) < 2:
            continue                        # a strip a square or two wide is the wall's line, not a shop
        out.append(part)
    return out


def second_dock(building, garage: dict) -> bool:
    """A superstore's second lorry dock (user, 2026-10-26): the first's shutter again,
    DOCK_GAP squares along the same stretch of wall (its `run`), in one room; laid as the
    first is (regional_uk.lay_garage). False where the wall has no room for it."""
    from . import regional_uk as U
    run = [tuple(e) for e in garage.get("run") or []]
    first = [tuple(e) for e in garage.get("door") or []]
    if len(run) < 6 + DOCK_GAP or not first or first[0] not in run:
        return False
    i0 = run.index(first[0])
    starts = sorted((i for i in range(len(run) - 2) if abs(i - i0) >= 3 + DOCK_GAP),
                    key=lambda i: abs(abs(i - i0) - (3 + DOCK_GAP)))
    st = building.storeys[0]
    for i in starts:
        g2 = dict(garage, door=run[i:i + 3], anchor=run[i], at=i)
        g2.pop("signs", None)
        probe = dict(g2)
        if not U._loading_in_one_room(st, probe):
            continue
        # (sliding into one room may bring it back beside the first)
        if min(abs(e[0] - f[0]) + abs(e[1] - f[1]) for e in probe["door"] for f in first) <= DOCK_GAP:
            continue
        U.lay_garage(building, probe)
        return True
    return False


def ad_first(ad, a):
    """The front-most d of column a (the lobby's front where it stands proud)."""
    return min(d for (aa, d) in ad.values() if aa == a)


def _outside(cells, x, y, side):
    dx, dy = STEP[side]
    return (x, y) in cells and (x + dx, y + dy) not in cells


def _edge(x, y, side):
    return {"N": (x, y, "N"), "W": (x, y, "W"), "S": (x, y + 1, "N"), "E": (x + 1, y, "W")}[side]


def _middle_of_longest(values):
    best, run = [], []
    for v in sorted(set(values)) + [None]:
        if run and v is not None and v == run[-1] + 1:
            run.append(v)
            continue
        if len(run) > len(best):
            best = run
        run = [v] if v is not None else []
    return best[len(best) // 2] if best else 0


def front_glass(cells: set, ad: dict, front: str, door_a: int, d_check: int):
    """The frontage glazed either side of the entrance's shop front (user, from Sainsbury's
    Alton and the game's grocery stores: glass walls, double doors, posters in the glass):
    the front band's outside squares in bays (tbx.py frames each run, two squares or more)
    between brick piers every BAY squares; and in every other bay of four or more, the
    square of a pair of glass doors (its pair: the next pane, tbx.py). Returns (the glazed
    squares, the doors' squares), inside the store."""
    glass = [(x, y) for (x, y) in sorted(cells, key=lambda c: ad[c])
             if _outside(cells, x, y, front) and ad[(x, y)][1] < d_check
             and abs(ad[(x, y)][0] - door_a) > SHOPFRONT_HALF and ad[(x, y)][0] % BAY != 0]
    runs = [r for r in _runs_of(glass, ad) if len(r) >= 2]
    doors = [r[len(r) // 2 - 1] for k, r in enumerate(runs) if k % 2 == 0 and len(r) >= 4]
    return [c for r in runs for c in r], doors


def store_front(mask, w: int, h: int, front: str) -> dict:
    """A store's front as layout reads it: its frame, squares, (a, d) of each, main front
    line, the entrance's `a` and the checkouts' `d`."""
    frame = Frame(w, h, front)
    cells = _cells(mask, w, h)
    ad = {c: frame.ad(*c) for c in cells}
    f_line, _b = lines(frame, cells)
    door_a = _middle_of_longest(sorted(a for (x, y), (a, d) in ad.items() if _outside(cells, x, y, front)))
    return {"frame": frame, "cells": cells, "ad": ad, "f_line": f_line, "door_a": door_a,
            "d_check": f_line + ENTRANCE}


def _runs_of(cells: list, ad: dict) -> list:
    """Cells along the front in unbroken runs (consecutive `a` on the same `d`)."""
    runs = []
    for c in sorted(cells, key=lambda c: (ad[c][1], ad[c][0])):
        if runs and ad[runs[-1][-1]][1] == ad[c][1] and ad[runs[-1][-1]][0] == ad[c][0] - 1:
            runs[-1].append(c)
        else:
            runs.append([c])
    return runs


def _connected(cs: set):
    left, out = set(cs), []
    while left:
        todo, part = [next(iter(left))], set()
        while todo:
            c = todo.pop()
            if c in part or c not in left:
                continue
            part.add(c)
            x, y = c
            todo += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        left -= part
        out.append(part)
    return out


def _cut_to(cs: set, ad, most: int):
    """A room no bigger than `most`: cut along `a` into equal pieces, each joined up."""
    n = max(1, -(-len(cs) // most))
    if n == 1:
        return [cs]
    by_a = sorted(cs, key=lambda c: ad[c][0])
    size = -(-len(by_a) // n)
    out = []
    for k in range(n):
        out += _connected(set(by_a[k * size:(k + 1) * size]))
    return [p for p in out if p]


def _cut_along(cs: set, ad, wide: int):
    """Cut along `a` into pieces `wide` squares across (toilets, cubicles)."""
    cols = collections.defaultdict(set)
    for c in cs:
        cols[ad[c][0]].add(c)
    keys = sorted(cols)
    out = []
    for k in range(0, len(keys), wide):
        part = set().union(*(cols[a] for a in keys[k:k + wide]))
        out += _connected(part)
    return out


def _back_of_house(frame, cs, ad, front, counters, cbounds, d_boh):
    """The stock room (racking), cut to 300 m2 rooms; the cold store behind the
    butcher (freezers round its walls: _fit); a janitor's cupboard; the stairs'
    hall is laid by _stairs. (kind, cells, zone) each."""
    out = []
    cold = set()
    if "butcher" in cbounds:
        lo, hi = cbounds["butcher"]
        cold = {c for c in cs if lo <= ad[c][0] < lo + 4 and ad[c][1] < d_boh + 4}
    jan = set()
    if cbounds:
        lo, hi = list(cbounds.values())[-1]
        jan = {c for c in cs if hi <= ad[c][0] < hi + 3 and ad[c][1] < d_boh + 3}
    rest = cs - cold - jan
    for part in _connected(rest):
        out += [("grocerystorage", p, "boh") for p in _cut_to(part, ad, 300)]
    # the cold store: the game's groceryfreezer has no loot table; grocerystorage's fills freezers
    out += [("grocerystorage", p, "cold") for p in _connected(cold)]
    out += [("janitor", p, "boh") for p in _connected(jan)]
    return out


def _staff_floor(frame, upper, ad, front):
    """The staff floor: a corridor two squares deep along the back wall (the stairs
    come up in it), the rooms in front of it, STAFF_WIDE across."""
    if not upper:
        return []
    back = max(ad[c][1] for c in upper)
    corridor = {c for c in upper if ad[c][1] >= back - 1}
    rest = upper - corridor
    out = [("hall", p, "staff") for p in _connected(corridor)]
    for n, p in enumerate(_cut_along(rest, ad, STAFF_WIDE)):
        out.append((STAFF_ROOMS[n % len(STAFF_ROOMS)], p, "staff"))
    return out


def _plan(L, w, h, pieces, cells):
    grid = [[0] * w for _ in range(h)]
    rooms = []
    zones = {}
    for kind, cs, z in pieces:
        if not cs:
            continue
        zones[len(rooms) + 1] = z
        rooms.append(L.Room(min(x for x, _ in cs), min(y for _, y in cs), max(x for x, _ in cs),
                            max(y for _, y in cs), kind=kind, fixed=True,
                            is_core=kind == "hall"))
        for x, y in cs:
            grid[y][x] = len(rooms)
    plan = L.Plan(width=w, height=h, rooms=rooms, grid=grid, drawn=True)
    plan.zones = zones              # each room's strip (produce, frozen, aisles...): _fit
    return plan


def _flight(stairs):
    x, y, d = stairs
    dx, dy = (0, 1) if d == "N" else (1, 0)
    return [(x + dx * i, y + dy * i) for i in range(5)]


def _stairs(frame, ground, first, cells, upper, front, toward_low):
    """A flight of 5 along the back wall at the end away from the entrance; the
    ground floor's squares under it and the row in front made its hall. The game's
    stairs climb N or W only: along a N/S back wall they run W, else N."""
    back = OPPOSITE[front]
    run = "W" if back in ("N", "S") else "N"
    dx, dy = (1, 0) if run == "W" else (0, 1)
    ins = (0, -1) if back == "S" else (0, 1) if back == "N" else (-1, 0) if back == "E" else (1, 0)
    cands = []
    for (x, y) in upper:
        flight = [(x + dx * i, y + dy * i) for i in range(5)]
        if not all(c in upper and c in cells and _outside(cells, *c, back) for c in flight):
            continue
        front_row = [(cx + ins[0], cy + ins[1]) for cx, cy in flight]
        if not all(c in upper for c in front_row):
            continue
        a = frame.ad(x, y)[0]
        cands.append((-a if toward_low else a, (x, y)))
    if not cands:
        return None
    _k, (x, y) = min(cands)
    flight = [(x + dx * i, y + dy * i) for i in range(5)]
    hall = set(flight) | {(cx + ins[0], cy + ins[1]) for cx, cy in flight}
    # the ground floor: the hall taken out of whatever room held those squares
    from . import layout as L
    ground.rooms.append(L.Room(min(c[0] for c in hall), min(c[1] for c in hall), max(c[0] for c in hall),
                               max(c[1] for c in hall), kind="hall", fixed=True, is_core=True))
    for cx, cy in hall:
        ground.grid[cy][cx] = len(ground.rooms)
    # upstairs the flight is in the corridor (the hall along the back wall)
    hall_up = first.grid[y][x]
    for cx, cy in flight:
        first.grid[cy][cx] = hall_up
    return (x, y, run)


def _open(plan, open_ids):
    """No wall between two sales-floor rooms."""
    g = plan.grid
    h, w = len(g), len(g[0])
    for y in range(h):
        for x in range(w):
            a = g[y][x]
            if a not in open_ids:
                continue
            if x > 0 and g[y][x - 1] in open_ids and g[y][x - 1] != a:
                plan.open_edge.add((x, y, "W"))
            if y > 0 and g[y - 1][x] in open_ids and g[y - 1][x] != a:
                plan.open_edge.add((x, y, "N"))


def _doors(plan, reached: set, stair_cells: set, stairs, level: int, levels: int):
    """A door into every room from one already reached, until all are: the middle of
    their shared wall, off the stairs' flight (its walkable end aside)."""
    g = plan.grid
    h, w = len(g), len(g[0])
    flight = _flight(stairs) if stairs else []
    blocked = set(flight)
    if flight:
        if level < levels - 1:
            blocked.discard(flight[-1])     # the foot, on the floor it rises from
        if level > 0:
            blocked.discard(flight[0])      # the head, on the floor it reaches
    reached = set(reached) - {0}
    rooms = set(range(1, len(plan.rooms) + 1))
    for _ in range(len(rooms)):
        todo = rooms - reached
        if not todo:
            break
        grew = False
        for r in sorted(todo):
            edges = collections.defaultdict(list)
            for y in range(h):
                for x in range(w):
                    if g[y][x] != r or (x, y) in blocked:
                        continue
                    for nx, ny, e in ((x - 1, y, (x, y, "W")), (x + 1, y, (x + 1, y, "W")),
                                      (x, y - 1, (x, y, "N")), (x, y + 1, (x, y + 1, "N"))):
                        if 0 <= nx < w and 0 <= ny < h and g[ny][nx] in reached and (nx, ny) not in blocked:
                            edges[g[ny][nx]].append(e)
            if edges:
                # a room's door into the room it serves where it has one (DOOR_TO: the cafe's
                # kitchen into its servery), else its longest shared wall
                want = DOOR_TO.get(plan.rooms[r - 1].kind)
                served = [es for n, es in edges.items() if plan.rooms[n - 1].kind == want]
                best = max(served or edges.values(), key=len)
                best.sort()
                plan.doors.append(best[len(best) // 2])
                reached.add(r)
                grew = True
        if not grew:
            break


# ---- the fittings -------------------------------------------------------------

def _fit(L, plan, frame, ad, front, door_a, d_check, d_sales, d_counter, d_boh, f_line, rng, chain=None,
         d_mid=None):
    """The store's fittings, in place of Build's (its switches stay): the checkouts,
    trolleys, baskets and service desk; each block by what it sells (AISLES, WALLS,
    ENDS); the counters' displays and kit; the café; the cold store's freezers; posters."""
    from . import catalog as C
    g = plan.grid
    zones = {i: z[:-2] if z.endswith(("_f", "_b")) else z for i, z in plan.zones.items()}
    fitted = {i for i, z in zones.items() if z in set(AISLES) | set(WALLS) | set(COUNTERS)
              | {"entrance", "produce", "cafe", "clothing", "cold"}}
    fit_cells = {c for c in ad if g[c[1]][c[0]] in fitted}
    plan.furniture = [f for f in plan.furniture
                      if (f[1], f[2]) not in fit_cells or C.FURNITURE_LAYERS.get(f[0], "Furniture") != "Furniture"]
    taken = set()
    doors = set(plan.doors)
    for x, y, d in plan.doors:
        for c in (((x - 1, y), (x, y), (x - 2, y), (x + 1, y)) if d == "W" else ((x, y - 1), (x, y), (x, y - 2), (x, y + 1))):
            taken.add(c)

    # a pillar (pillars_uk: a room of one square) is no wall: nothing lines it, no sign or
    # poster hangs on it (user: the pillars had fittings and light switches round them)
    pillar_ids = {i for i, r in enumerate(plan.rooms, 1) if r.kind == "empty"}
    round_pillar = {}
    for y, row in enumerate(g):
        for x, v in enumerate(row):
            if v in pillar_ids:
                near = [g[ny][nx] for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                        if 0 <= ny < len(g) and 0 <= nx < len(row) and g[ny][nx] not in pillar_ids]
                round_pillar[(x, y)] = collections.Counter(near).most_common(1)[0][0] if near else v

    def room(c):
        x, y = c
        if c in round_pillar:
            return round_pillar[c]
        return g[y][x] if 0 <= y < len(g) and 0 <= x < len(g[0]) else 0

    def free(c):
        return c in fit_cells and c not in taken

    # a light switch's square stays clear (user: "can't find lightswitches ... covered by shelves")
    taken.update((f[1], f[2]) for f in plan.furniture if f[0] == L.SWITCH)

    def walled(c, side):
        """A wall on this side of c (not an open edge, a door or a pillar)."""
        dx, dy = STEP[side]
        e = _edge(*c, side)
        return room((c[0] + dx, c[1] + dy)) != room(c) and e not in plan.open_edge and e not in doors

    def put(role, x, y, orient):
        cs = L._cells_for(role, x, y, orient) if orient in C.FURNITURE[role] else None
        if not cs or not all(free(c) for c in cs):
            return False
        if len({room(c) for c in cs}) > 1:
            return False            # a piece stands in one room (the game's rule: validate_tbx)
        plan.furniture.append((role, x, y, orient))
        taken.update(cs)
        return True

    along_depth = "W" if front in ("N", "S") else "N"      # pieces running in from the front
    across = "N" if front in ("N", "S") else "W"            # pieces running along it
    back_side = OPPOSITE[front]

    # the checkouts: the game's own unit (CHECKOUTS), every CHECKOUT_EVERY squares along
    # the front, clear of the doors; a bin at the end of each lane
    unit = CHECKOUTS[front]
    for a in range(2, frame.length - 6, CHECKOUT_EVERY):
        if abs(a - door_a) <= 5:
            continue
        spots = [(frame.xy(a + u, d_check + v), tiles) for u, v, tiles in unit]
        lane = [frame.xy(a + 3, d_check + v) for v in range(5)]
        if not all(free(c) for c, _t in spots) or len({room(c) for c, _t in spots}) > 1:
            continue
        for (x, y), tiles in spots:
            for t in tiles:
                plan.furniture.append((_raw(C, t), x, y, "N"))
            taken.add((x, y))
        taken.update(lane)
        if free(frame.xy(a + 3, d_check - 1)):
            put("uk_bin", *frame.xy(a + 3, d_check - 1), "N")
    # the trolley bay inside the doors, the baskets beside it, the service desk across
    for k in range(TROLLEY_BAYS):
        a = door_a + 2 + 2 * k
        for d in range(ad_first(ad, a) if any(aa == a for aa, _ in ad.values()) else f_line, d_check):
            put("uk_trolley_rail", *frame.xy(a, d), along_depth)
    for a in range(door_a + 3 + 2 * TROLLEY_BAYS, door_a + 6 + 2 * TROLLEY_BAYS):
        put("uk_baskets", *frame.xy(a, d_check - 1), across if across in C.FURNITURE["uk_baskets"] else "N")
    desk = [frame.xy(door_a - k, f_line + 1) for k in range(3, 6)]
    if all(free(c) for c in desk):
        for c in desk:
            put("shop_counter", *c, L._facing("shop_counter", front))
        plan.furniture.append(("register", *desk[1], L._facing("register", front)))
    # ATMs on a north or west wall of the entrance band (the sprites the game has), the
    # nearest the doors; litter bins either side of the doors
    # (not in a lobby that stands out of the front: its sides are glass, an ATM stood free there)
    band = sorted((c for c in fit_cells if f_line <= ad[c][1] < d_check and zones.get(room(c)) == "entrance"),
                  key=lambda c: abs(ad[c][0] - door_a))
    atms = 0
    for c in band:
        if atms >= ATMS:
            break
        for side in ("N", "W"):
            dx, dy = STEP[side]
            e = _edge(*c, side)
            if side != front and room((c[0] + dx, c[1] + dy)) != room(c) and e not in plan.open_edge \
                    and e not in doors and put("uk_atm", *c, side):
                atms += 1
                break
    for k in (-2, 2):
        put("uk_litter_bin", *frame.xy(door_a + k, ad_first(ad, door_a)), "N")

    # the counters (open to the floor, as the game's): the display along the front, the
    # staff's aisle behind it with the till, the trade's kit along the back wall
    for i, z in zones.items():
        if z not in COUNTERS:
            continue
        cs = {c for c in fit_cells if room(c) == i}
        _kind, _w, display, kit = COUNTERS[z]
        row = sorted((c for c in cs if ad[c][1] == d_counter), key=lambda c: ad[c][0])
        if row:
            taken.add(row[-1])              # the flap: the staff's way in from the floor, kept clear
            fa = ad[row[-1]][0]             # ...and the floor in front of it
            taken.update(frame.xy(fa, d_counter - k) for k in (1, 2))
        for c in row[:-1]:
            if True:
                o = "N" if display == "uk_bakery_case" else L._facing(display, back_side)
                if not put(display, *c, o):
                    put("counter_2", *c, L._facing("counter_2", back_side))
        aisle = sorted((c for c in cs if ad[c][1] == d_counter + 1), key=lambda c: ad[c][0])
        if row[:-1]:
            # the till on the first counter (the far end from the flap), the bin by it
            plan.furniture.append(("register", *row[0], L._facing("register", back_side)))
        if aisle:
            put("uk_bin", *aisle[0], "N")
        taken.update(aisle)                                     # the staff's aisle stays clear
        n = 0
        for c in sorted((c for c in cs if ad[c][1] == d_boh - 1), key=lambda c: ad[c][0]):
            role = kit[n % len(kit)]
            if put(role, *c, L._facing(role, back_side)):
                n += 1

    # a walkway into the café along each of its open sides (user: "is there access to the
    # cafe?" - a run of fittings stood against its whole side): CAFE_WAY squares of the
    # neighbouring blocks kept clear
    cafe = {c for c in ad if zones.get(room(c)) == "cafe"}
    for (x, y) in cafe:
        for dx, dy in STEP.values():
            for k in range(1, CAFE_WAY + 1):
                n = (x + dx * k, y + dy * k)
                if n in cafe or room(n) == 0 or zones.get(room(n)) in ("cafekitchen", None):
                    break
                taken.add(n)
    # each block: its walls lined with its fittings, backs to the wall (WALLS); then its floor
    for i, z in zones.items():
        cs = {c for c in fit_cells if room(c) == i}
        if not cs or z in COUNTERS or z in ("entrance", "cafe", "cold"):
            continue
        roles = WALLS.get(z, [])
        n = 0
        for c in sorted(cs, key=lambda c: ad[c]):
            for side, (dx, dy) in STEP.items():
                e = _edge(*c, side)
                if room((c[0] + dx, c[1] + dy)) == i or e in plan.open_edge or e in doors:
                    continue
                # (a fitting with no sprite for this wall - the open chiller has N and W
                # only - would face it: the next in the list that has one, else none)
                fits = [r for r in roles[n % len(roles):] + roles[:n % len(roles)] if side in C.FURNITURE[r]]
                if fits and put(fits[0], *c, side):
                    n += 1
                break
        if z == "clothing":
            _grid_of(put, cs, [("clothes_rack", across), ("clothes_rack", across), ("mannequin", "N"),
                               ("clothes_rack_small", across)], 3)
        elif z == "produce":
            _grid_of(put, cs, [("shop_bin", "N"), ("shop_bin", "N"), ("uk_produce_stand", across), ("shop_bin", "N"),
                               ("uk_produce_stand", across), ("shop_bin", "N"), ("shop_bin", "N"),
                               ("uk_produce_stand", across), ("shop_bin", "N"), ("shop_bin", "N"),
                               ("uk_produce_stand", across), ("uk_scale", "N")], 2)
        elif z in AISLES:
            _aisles(C, put, cs, front, AISLES[z], ENDS, walled)
    # the café: a servery along its kitchen (the back counter, the staff's aisle, the front
    # counter with the till and the cakes), a row to queue in, small tables over the rest
    for i, z in zones.items():
        if z != "cafe":
            continue
        cs = {c for c in fit_cells if room(c) == i}
        if not cs:
            continue
        d_top = max(ad[c][1] for c in cs)                      # the row against the kitchen

        def row(d):
            return sorted((c for c in cs if ad[c][1] == d), key=lambda c: ad[c][0])
        n = 0
        for c in row(d_top):                                    # the back counter
            role = CAFE_BACK[n % len(CAFE_BACK)]
            if put(role, *c, L._facing(role, back_side)):
                n += 1
        taken.update(row(d_top - 1))                            # the staff's aisle
        front_row = row(d_top - 2)
        if front_row:
            # the flap: the end against the store's wall (no floor beyond it), else the far end
            lo, hi = front_row[0], front_row[-1]
            a_lo = ad[lo][0]
            gap = lo if frame.xy(a_lo - 1, ad[lo][1]) not in ad else hi
            n = 0
            for c in front_row:
                if c == gap:
                    taken.add(c)
                    continue
                role = CAFE_FRONT[n % len(CAFE_FRONT)]
                o = "N" if role == "uk_bakery_case" else L._facing(role, back_side)
                if not put(role, *c, o):
                    role = "counter_2"
                    if not put(role, *c, L._facing(role, back_side)):
                        continue
                if n == 0:
                    plan.furniture.append(("register", *c, L._facing("register", back_side)))
                n += 1
        taken.update(row(d_top - 3))                            # the queue
        dining = {c for c in cs if ad[c][1] < d_top - 3}
        if dining:
            ends = sorted(dining, key=lambda c: ad[c])
            for c in ends[:1] + ends[-1:]:
                put("uk_bin", *c, "N")
            xs = sorted({x for x, _ in dining})
            ys = sorted({y for _, y in dining})
            for y in range(min(ys) + 1, max(ys), 3):
                for x in range(min(xs) + 2, max(xs) - 1, 3):
                    if put("round_table", x, y, "N"):
                        put("chair", x - 1, y, "W")
                        put("chair", x + 1, y, "E")
    # the cold store: chest freezers round its walls, in place of Build's racking
    for i, z in zones.items():
        if z != "cold":
            continue
        for (x, y) in sorted(c for c in fit_cells if room(c) == i):
            for side, (dx, dy) in STEP.items():
                if room((x + dx, y + dy)) != i and put("shop_freezer", x, y, L._facing("shop_freezer", side)):
                    break
    # the department signs, then posters, on the north and west walls (the game's own hang only there)
    hung = _signs(C, plan, zones, ad, room, doors, chain)
    sales = {i for i, z in zones.items() if z not in COUNTERS and z != "cold"}
    n = 0
    for c in sorted((c for c in fit_cells if room(c) in sales), key=lambda c: ad[c]):
        for side in ("N", "W"):
            dx, dy = STEP[side]
            e = _edge(*c, side)
            if room((c[0] + dx, c[1] + dy)) == room(c) or e in plan.open_edge or e in doors or e in hung:
                continue
            n += 1
            if n % POSTER_EVERY == 0:
                plan.furniture.append(("uk_poster", c[0], c[1], side))


def signs() -> dict:
    """built/dept_signs.json's signs ({} where the kit is not built: no signs)."""
    if not _SIGNS:
        try:
            with open(SIGNS_JSON, encoding="utf-8") as f:
                _SIGNS.append(json.load(f).get("signs") or {})
        except (OSError, ValueError):
            _SIGNS.append({})
    return _SIGNS[0]


def _signs(C, plan, zones, ad, room, doors, chain) -> set:
    """Each block's department sign (SIGN_OF) on the longest straight stretch of its own
    north or west wall (not glazing, a door or an open side), the longest version that fits,
    one per block; the edges hung. A sign is a furniture piece on the WallFurniture layer."""
    table = signs()
    kit = table.get(chain) or table.get("generic") or {}
    hung, done = set(), set()
    front = set(plan.shop_front or ())
    for i, z in sorted(zones.items()):
        dept = SIGN_OF.get(z)
        if dept is None or dept in done or dept not in kit:
            continue
        cs = {c for c in ad if room(c) == i}
        best = None                             # (length, side, first cell)
        for side in ("N", "W"):
            dx, dy = STEP[side]
            along = (1, 0) if side == "N" else (0, 1)

            def wall(c):
                e = _edge(*c, side)
                return (c in cs and room((c[0] + dx, c[1] + dy)) != i and e not in plan.open_edge
                        and e not in doors and e not in front and e not in hung)
            for c in cs:
                if not wall(c) or wall((c[0] - along[0], c[1] - along[1])):
                    continue                    # the start of a stretch only
                n = 0
                while wall((c[0] + along[0] * n, c[1] + along[1] * n)):
                    n += 1
                if best is None or n > best[0]:
                    best = (n, side, c)
        if best is None:
            continue
        n, side, (x, y) = best
        fits = [v for v in kit[dept]["versions"] if v["length"] <= n]
        if not fits:
            continue
        v = max(fits, key=lambda v: v["length"])
        k0 = (n - v["length"]) // 2                         # in the middle of its stretch
        along = (1, 0) if side == "N" else (0, 1)
        x, y = x + along[0] * k0, y + along[1] * k0
        role = f"uk_dept:{chain or 'generic'}:{dept}:{v['length']}"
        if role not in C.FURNITURE:
            C.FURNITURE[role] = {
                "N": {f"{k},0": t for k, t in enumerate(v["N"])},
                "W": {f"0,{k}": t for k, t in enumerate(v["W"])}}
            C.FURNITURE_LAYERS[role] = "WallFurniture"
        plan.furniture.append((role, x, y, side))
        # (and a square clear each end: two signs never hang end to end)
        hung |= {_edge(x + along[0] * k, y + along[1] * k, side) for k in range(-1, v["length"] + 1)}
        done.add(dept)
    return hung


def _raw(C, sprite: str) -> str:
    """A sprite as a furniture role of its own (a piece of the game's checkout unit)."""
    role = f"uk_t:{sprite}"
    C.FURNITURE.setdefault(role, {"N": {"0,0": sprite}})
    return role


def _free_the_way(L, plan) -> None:
    """Take out what can't be reached, and what boxes in what must stay (an oven in a corner
    behind a rack) or stands on a light switch's square: DROPPABLE pieces only, a few rounds."""
    from . import catalog as C

    def cells(f):
        role, x, y, o = f
        return L._cells_for(role, x, y, o) if role in C.FURNITURE and o in C.FURNITURE[role] else [(x, y)]
    switches = {(f[1], f[2]) for f in plan.furniture if f[0] == L.SWITCH}
    plan.furniture = [f for f in plan.furniture
                      if not (f[0] in DROPPABLE and C.FURNITURE_LAYERS.get(f[0], "Furniture") == "Furniture"
                              and set(cells(f)) & switches)]
    for _round in range(4):
        bad = unreachable(plan)
        if not bad:
            return
        drop = {p for p in bad if p[0] in DROPPABLE}
        keep = [p for p in bad if p[0] not in DROPPABLE]
        near = set()
        for role, x, y in keep:
            for f in plan.furniture:
                if (f[0], f[1], f[2]) == (role, x, y):
                    near |= {(cx + dx, cy + dy) for cx, cy in cells(f) for dx, dy in STEP.values()}
        plan.furniture = [f for f in plan.furniture if (f[0], f[1], f[2]) not in drop
                          and not (f[0] in DROPPABLE and set(cells(f)) & near)]


def unreachable(plan) -> list:
    """Fittings a shopper cannot get to (user, 2026-10-26: "you can't get to the units along
    the wall"): a walk from the outside doors over the floor - through doors and open edges
    only between rooms, round every piece on the Furniture layer and the pillars - and each
    piece with no square beside it (across no wall) that the walk reached. (role, x, y)."""
    from . import catalog as C
    from . import layout as L
    g = plan.grid
    h, w = len(g), len(g[0])
    solid = {}
    for role, x, y, o in plan.furniture:
        if C.FURNITURE_LAYERS.get(role, "Furniture") != "Furniture" or role not in C.FURNITURE                 or o not in C.FURNITURE[role] or role in ("register",) or role.startswith("uk_t:"):
            continue
        for c in L._cells_for(role, x, y, o):
            solid[c] = (role, x, y)
    for i, r in enumerate(plan.rooms, 1):
        if r.kind == "empty":                       # a pillar
            for y in range(h):
                for x in range(w):
                    if g[y][x] == i:
                        solid[(x, y)] = ("pillar", x, y)
    passes = set(plan.doors) | set(plan.open_edge)

    def room(c):
        x, y = c
        return g[y][x] if 0 <= x < w and 0 <= y < h else 0

    def link(a, b):
        """No wall between neighbouring squares a and b."""
        if room(a) == room(b):
            return True
        (ax, ay), (bx, by) = a, b
        e = (max(ax, bx), ay, "W") if ay == by else (ax, max(ay, by), "N")
        return e in passes

    starts = []
    for x, y, d in plan.doors:
        for c in ((x, y), (x - 1, y) if d == "W" else (x, y - 1)):
            if room(c) and c not in solid:
                starts.append(c)
    seen, todo = set(), list(starts)
    while todo:
        c = todo.pop()
        if c in seen or not room(c) or c in solid:
            continue
        seen.add(c)
        x, y = c
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n not in seen and link(c, n):
                todo.append(n)
    pieces = {}
    for c, piece in solid.items():
        if piece[0] != "pillar":
            pieces.setdefault(piece, []).append(c)
    out = []
    for piece, cells in pieces.items():
        if not any(n in seen and link(c, n) for c in cells
                   for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1))):
            out.append(piece)
    return out


def _grid_of(put, cs, pieces, pitch):
    """Islands on a grid `pitch` squares apart (an aisle round each), the pieces in turn."""
    xs = sorted({x for x, _ in cs})
    ys = sorted({y for _, y in cs})
    n = 0
    for y in range(min(ys) + 2, max(ys) - 1, pitch):
        for x in range(min(xs) + 2, max(xs) - 1, pitch):
            role, orient = pieces[n % len(pieces)]
            if put(role, x, y, orient):
                n += 1


def _aisles(C, put, cs, front, mix, ends, walled=None):
    """Long runs of back-to-back fittings running in from the front, the game's way: two
    squares of fitting and an aisle of two (a run every 4 squares across), each run one
    fitting from `mix` in turn, a promotion (`ends`) at each end. A run along y is the
    fitting facing E beside the one facing W; along x, facing S over facing N."""
    xs = [x for x, _ in cs]
    ys = [y for _, y in cs]
    along_y = front in ("N", "S")
    lo_a, hi_a = (min(xs), max(xs)) if along_y else (min(ys), max(ys))
    lo_r, hi_r = (min(ys), max(ys)) if along_y else (min(xs), max(xs))
    n = 0
    # (user: "you can't get to the units along the wall"): from a lined wall, its fittings and
    # an aisle of two before a run (3); from an open edge one, the neighbour's aisle adding
    # to it; and at a run's ends a promotion as well (4 from a wall, 2 from an open edge)
    lo_side, hi_side = ("W", "E") if along_y else ("N", "S")
    lo_t_side, hi_t_side = ("N", "S") if along_y else ("W", "E")

    def edge_walled(side, k, axis_a):
        return walled is not None and any(walled(c, side) for c in cs
                                          if (c[0] if axis_a == along_y else c[1]) == k)
    m_lo = 3 if edge_walled(lo_side, lo_a, True) else 1
    m_hi = 3 if edge_walled(hi_side, hi_a, True) else 1
    t_lo = 4 if edge_walled(lo_t_side, lo_r, False) else 2
    t_hi = 4 if edge_walled(hi_t_side, hi_r, False) else 2
    room_for = (hi_a - m_hi) - (lo_a + m_lo) + 1
    runs = max(0, (room_for + 2) // 4)
    first = lo_a + m_lo + (room_for - (4 * runs - 2)) // 2
    for k, s in enumerate(range(first, first + 4 * runs, 4)):
        role = mix[k % len(mix)]
        step = 1 if len(C.FURNITURE[role].get("E" if along_y else "S", {})) == 1 else 2
        laid = []
        for t in range(lo_r + t_lo, hi_r - t_hi - step + 2, step):
            if along_y:
                ok = put(role, s, t, "E") and put(role, s + 1, t, "W")
            else:
                ok = put(role, t, s, "S") and put(role, t, s + 1, "N")
            if ok:
                laid.append(t)
        if laid:
            for t in (min(laid) - 1, max(laid) + step):
                piece = ends[n % len(ends)]
                x, y = (s, t) if along_y else (t, s)
                if put(piece, x, y, "N" if "N" in C.FURNITURE[piece] else next(iter(C.FURNITURE[piece]))):
                    n += 1
