"""A superstore's front, outside (KnoxMap UK, user 2026-10-26, from Sainsbury's Alton's
Street View and the game's grocery stores): a canopy over the walk along its main front,
paving under it, bollards along its edge, posters and the café's sign in the glass, phone
boxes and litter bins.

Loose map tiles in tile-only lots (structures.pack_loose), as the petrol forecourts'
(pumps.py), laid after every building has its tiles. Its parts:

    canopy, posters, café sign   asset_creator/commercial_kit/storefront.py's
                                 (built/storefront.json; none where the kit isn't built)
    bollards, phone boxes        the street kit's (street_furniture_uk_01/02)
    litter bins                  the game's

The store's front - its main front line, its entrance, its glazed bays - is read as
superstore_uk lays them out (store_front, front_glass), so a poster always hangs in glass.
The car park is not this module's (the user's car park tool).
"""
from __future__ import annotations

import json
import os

from generator import pz_colors as C

from .bitmaps import read_rgb, same_colour

from ukdirs import assets as _assets          # asset_creator, if there is one (KnoxMap UK)
KIT_JSON = _assets("commercial_kit", "built", "storefront.json") or ""
CANOPY_DEEP = 3          # squares out from the front wall
COLUMN_EVERY = 6         # squares between the canopy's columns along its outer edge
BOLLARD_EVERY = 2
DOOR_CLEAR = 2           # squares either side of a door's pair with no bollard
POSTER_EVERY = 2         # glazed squares between posters
BOLLARD = "street_furniture_uk_02_12"                   # bollard_iron (asset_creator/src/street_objects.py)
KIOSK = {"S": "street_furniture_uk_01_8", "E": "street_furniture_uk_01_9",    # BT's KX100, by the way it faces
         "N": "street_furniture_uk_01_10", "W": "street_furniture_uk_01_11"}
KIOSKS = 2
KIOSK_FROM_DOOR = 9      # squares along the front from the main entrance
LITTER_BIN = "trashcontainers_01_21"
# Carriageways: nothing of the store's is laid on one
ROAD = {C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _kit() -> dict:
    try:
        with open(KIT_JSON, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def place_storefronts(out_dir: str, map_name: str, bdir: str, occupied, decided: list) -> list:
    """Every superstore's front outside (`decided`: Build's buildings, a superstore's style
    carrying `superstore_front`); its lots' placements."""
    from .structures import pack_loose
    stores = [d for d in decided if (d[11] or {}).get("superstore") and (d[11] or {}).get("superstore_front")]
    bmp = os.path.join(out_dir, f"{map_name}.bmp")
    if not stores or not os.path.exists(bmp):
        return []
    from PIL import Image
    ground = read_rgb(bmp)
    H, W = ground.shape[:2]
    road = None
    for colour in ROAD:
        m = same_colour(ground, colour)
        road = m if road is None else road | m
    kit = _kit()
    tiles: list = []
    for d in stores:
        tiles += _one(d, kit, ground, road, occupied, W, H)
    tiles = [t for t in tiles if isinstance(t[4], str) and t[4]]   # (a kit entry not a sprite: left out)
    if not tiles:
        return []
    Image.fromarray(ground).save(bmp, format="BMP")
    print(f"superstore fronts     : {len(stores)} ({len(tiles)} tiles)")
    return pack_loose(bdir, map_name, "storefronts", tiles)


def _one(d, kit, ground, road, occupied, W, H) -> list:
    from . import superstore_uk as S
    _f, _l, x0, y0, w, h = d[:6]
    style, mask = d[11], d[12]
    front = style["superstore_front"]
    chain = ((style.get("uk_fascia") or {}).get("sign")) or "generic"
    sf = S.store_front(mask, w, h, front)
    frame, cells, ad, f_line = sf["frame"], sf["cells"], sf["ad"], sf["f_line"]
    glass, bay_doors = S.front_glass(cells, ad, front, sf["door_a"], sf["d_check"])
    first = {}
    for (a, dd) in ad.values():
        first[a] = min(first.get(a, dd), dd)
    main = sorted(a for a, dd in first.items() if dd == f_line)
    if not main:
        return []
    out: list = []
    taken: set = set()

    def world(a, dd):
        x, y = frame.xy(a, dd)
        return x0 + x, y0 + y

    def free(x, y):
        return 0 <= x < W and 0 <= y < H and not occupied[y, x] and not road[y, x] and (x, y) not in taken

    def put(x, y, z, layer, tile, solid=True):
        out.append((x, y, z, layer, tile))
        if solid:
            taken.add((x, y))
            occupied[y, x] = True

    # the doors along the front: the main entrance's pair and each bay's
    doors_a = [sf["door_a"], sf["door_a"] + 1] + [a for c in bay_doors for a in (ad[c][0], ad[c][0] + 1)]

    def near_door(a, k):
        return any(abs(a - da) <= k for da in doors_a)

    # the canopy over the walk along the main front line, paving under it
    canopy = (kit.get("canopy") or {}).get(chain) or (kit.get("canopy") or {}).get("generic")
    lo, hi = main[0], main[-1]
    outer = f_line - CANOPY_DEEP
    under = []
    for a in range(lo, hi + 1):
        for dd in range(f_line - 1, outer - 1, -1):
            x, y = world(a, dd)
            if not free(x, y) or (x - x0, y - y0) in cells:
                break
            under.append((a, dd, x, y))
            ground[y, x] = C.PAVING
    if canopy and under:
        # (storefront.py: a run along x - a N or S front - takes the x pieces, along y the y
        # ones; the band on the outer edge where the camera sees it, S or E; each end's edge
        # across the canopy's depth)
        along_x = front in ("N", "S")
        tops = canopy.get("top_x" if along_x else "top_y") or canopy.get("top") or []
        ends = canopy.get("end_x" if along_x else "end_y") or {}
        end_layer = "Furniture3" if along_x else "Furniture2"
        top_end = canopy.get("top_end") or {}
        columns = {a for a in range(lo, hi + 1) if ((a - lo) % COLUMN_EVERY == 0 or a == hi) and not near_door(a, 0)}

        def end_key(a, dd):
            """W/E (a run along x) or N/S (along y): which end of the canopy, on the map."""
            ex, ey = world(a, dd)
            lx, ly = world(lo, dd)
            hx, hy = world(hi, dd)
            low_end = (ex == min(lx, hx)) if along_x else (ey == min(ly, hy))
            return ("W" if low_end else "E") if along_x else ("N" if low_end else "S")
        for a, dd, x, y in under:
            # the top: a solid bay at each end; the steel beam in line with each column;
            # clean and weathered glass mixed between (storefront.py)
            if a in (lo, hi) and top_end.get(end_key(a, dd)):
                out.append((x, y, 1, "RoofTop", top_end[end_key(a, dd)]))
            elif tops:
                beam = a in columns and len(tops) > 2
                out.append((x, y, 1, "RoofTop", tops[2] if beam else tops[(a * 7 + dd) % min(2, len(tops))]))
            if dd == outer:
                if front == "S" and canopy.get("band_x"):
                    out.append((x, y, 0, "Furniture2", canopy["band_x"]))
                if front == "E" and canopy.get("band_y"):
                    out.append((x, y, 0, "Furniture3", canopy["band_y"]))
                if canopy.get("column") and a in columns:
                    put(x, y, 0, "Furniture", canopy["column"])
            if a in (lo, hi) and isinstance(ends, dict) and ends.get(end_key(a, dd)):
                out.append((x, y, 0, end_layer, ends[end_key(a, dd)]))     # (the near end's only: the band round its corner)
    # bollards along the walk's edge, clear of the doors
    for a in range(lo, hi + 1, BOLLARD_EVERY):
        if near_door(a, DOOR_CLEAR):
            continue
        x, y = world(a, outer - 1)
        if free(x, y):
            put(x, y, 0, "Furniture", BOLLARD)
    # phone boxes on the walk, past the entrance on its longer side, facing the store
    step = 1 if (hi - sf["door_a"]) >= (sf["door_a"] - lo) else -1
    for k in range(KIOSKS):
        a = sf["door_a"] + step * (KIOSK_FROM_DOOR + k)
        x, y = world(a, outer)
        if lo <= a <= hi and free(x, y) and not near_door(a, 1):
            put(x, y, 0, "Furniture", KIOSK[OPPOSITE[front]])
    # a litter bin either side of every door, against the wall
    for da in doors_a[::2]:
        for a in (da - 2, da + 3):
            if a in first:
                x, y = world(a, first[a] - 1)
                if free(x, y):
                    put(x, y, 0, "Furniture", LITTER_BIN)
    # posters in the glass, seen from outside (a south or east front: the camera's sides),
    # the café's sign in the bay at its end of the store
    posters = (kit.get("posters") or {}).get(chain) or (kit.get("posters") or {}).get("generic") or {}
    cafe = (kit.get("cafe") or {}).get(chain) or (kit.get("cafe") or {}).get("generic") or {}
    face = {"S": "N", "E": "W"}.get(front)
    if face and (posters.get(face) or cafe.get(face)):
        door_cells = {c for c in bay_doors} | {frame.xy(ad[c][0] + 1, ad[c][1]) for c in bay_doors}
        panes = [c for c in glass if c not in door_cells]
        toward_high = sf["door_a"] < frame.length / 2         # the café: the far end from the entrance
        panes.sort(key=lambda c: ad[c][0], reverse=toward_high)
        cafe_panes = panes[:2] if cafe.get(face) else []
        n = 0
        for c in panes:
            a, dd = ad[c]
            x, y = world(a, dd - 1)                             # the square outside the glass
            if c in cafe_panes:
                out.append((x, y, 0, "WallFurniture", cafe[face]))
            elif posters.get(face) and n % POSTER_EVERY == 0:
                out.append((x, y, 0, "WallFurniture", posters[face][(n // POSTER_EVERY) % len(posters[face])]))
            n += 1
    for _a, _d, x, y in under:
        occupied[y, x] = True                  # the walk: no tree or prop on it after
    return out
