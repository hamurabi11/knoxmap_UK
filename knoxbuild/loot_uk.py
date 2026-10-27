"""Fittings that get loot (KnoxMap UK, user 2026-10-26: "please do the loot fixes").

The game fills a container from the loot table for its room's name and the
container's type (tools/loot_check.py). Build's shop tables stood a butcher's
display case (shop_display: container displaycasebutcher) in clothes, jewellery,
camera, tool and sports shops and behind cafe counters, and a glass display case
(shop_case: displaycase) in pharmacies, bakeries and music shops - containers
those rooms' tables never fill, so they were always empty. Each is swapped here
for a fitting its room's own table does fill, as the game's own shops have:

    clothes, camera, electronics, department, gun, jewellery shops: displaycase
    bakery, cafe (and its eateries' counters): displaycasebakery (the bakery case)
    pharmacy, optometrist: shelves
    music, sewing, shoe, sports, garden, tool shops: shelves or a counter

A lookup, applied to Build's tables (interiors.STORES, interiors.BACK_COUNTERS,
layout.ROOM_STYLE wishlists) once per process on a UK map (build._make_one).
tests/test_loot.py checks every fitting against the game's tables. Also a bin inside every
shop's door (FRONT_BIN, user 2026-10-26).
"""
from __future__ import annotations

# The bakery case (the game's Gigamart bakery: location_shop_generic_01_34/35, displaycasebakery),
# and a shop's bin (the game's round bin, trashcontainers_01_20: container bin)
BAKERY_CASE = {"uk_bakery_case": {"N": {"0,0": "location_shop_generic_01_34", "1,0": "location_shop_generic_01_35"}},
               "uk_bin": {"N": {"0,0": "trashcontainers_01_20"}, "W": {"0,0": "trashcontainers_01_20"}}}
# ...one inside every shop's door (user, 2026-10-26: "bins to the furniture for commercial
# buildings"): first among the pieces furnish_store stands inside the shop front
FRONT_BIN = "uk_bin"

CASE = "shop_case"              # displaycase
SHELF = "shop_shelf"            # shelves
WOOD = "shop_shelf_wood"        # shelves
WHITE = "shop_shelf_white"      # shelves
BAKERY = "uk_bakery_case"       # displaycasebakery
# room kind: {fitting: its replacement}
SWAPS = {
    "camerastore": {"shop_display": CASE}, "electronicsstore": {"shop_display": CASE},
    "clothesstore": {"shop_display": CASE}, "clothingstore": {"shop_display": CASE},
    "departmentstore": {"shop_display": CASE}, "gunstore": {"shop_display": CASE},
    "jewelrystore": {"shop_display": CASE},
    "bakery": {"shop_display": BAKERY, "shop_case": "shop_counter"},
    "pharmacy": {"shop_display": WHITE, "shop_case": WHITE},
    "optometrist": {"shop_display": WHITE, "shop_case": WHITE},
    "musicstore": {"shop_case": SHELF},
    "sewingstore": {"shop_display": WOOD}, "shoestore": {"shop_display": WOOD},
    "sportstore": {"shop_display": WOOD}, "gardenstore": {"shop_display": "shop_counter"},
    "toolstore": {"shop_display": "shop_counter"},
}
# what stands along a stock room's walls (layout.SHELVING; one not listed took the default,
# which has a bookcase: user 2026-10-26 "not sure why book cases are in the back stores")
STOCKROOM_SHELVING = ("metal_rack", "crate", "shelf", "shelf_1")
STOCKROOMS = ("grocerystorage", "liquorstorage", "toolstorage", "clothingstorage", "pharmacystorage",
              "electronicsstorage", "furniturestorage", "gardenstorage", "sportstorage", "storage")
# the counter behind a cafe's dining room (interiors.BACK_COUNTERS by style)
COUNTER_SWAPS = {"cafe": {"shop_display": BAKERY}}

_DONE = [False]


def apply() -> None:
    """Swap the unfilled fittings in Build's tables (once per process)."""
    if _DONE[0]:
        return
    from . import catalog as C
    from . import interiors as I
    from . import layout as L
    for role, tiles in BAKERY_CASE.items():
        C.FURNITURE.setdefault(role, tiles)
    for kind, swap in SWAPS.items():
        spec = I.STORES.get(kind)
        if spec:
            for key, value in list(spec.items()):
                if isinstance(value, list):
                    spec[key] = [swap.get(v, v) for v in value]
                elif isinstance(value, str):
                    spec[key] = swap.get(value, value)
        if kind in L.ROOM_STYLE:
            floor, label, wish = L.ROOM_STYLE[kind]
            L.ROOM_STYLE[kind] = (floor, label, [swap.get(v, v) for v in wish])
    for spec in I.STORES.values():
        if FRONT_BIN not in spec.get("front", []):
            spec["front"] = [FRONT_BIN] + list(spec.get("front", []))
    for kind in STOCKROOMS:
        L.SHELVING.setdefault(kind, STOCKROOM_SHELVING)
    L.SHELVING.setdefault("janitor", ("metal_rack", "shelf"))
    L.SHELVING.setdefault("lockerroom", ("shelf", "shelf_1"))
    for style, swap in COUNTER_SWAPS.items():
        if style in I.BACK_COUNTERS:
            I.BACK_COUNTERS[style] = [swap.get(v, v) for v in I.BACK_COUNTERS[style]]
    _DONE[0] = True
