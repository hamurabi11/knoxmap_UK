"""The things that make a piece of land what it is, once the ground is painted.

Some land uses are not buildings and not terrain, so they fell through both:
a cemetery came out as a lawn with flowers on it, and an army base as a patch
of bare dirt. What is missing in each case is the furniture of the place -
the headstones in rows, the supply crates and drums behind the wire - and
that is what this puts there.

Everything here is driven by the land-use polygons the renderer wrote
(<map>_areas.geojson, read through knoxbuild/areas.py), which come from
OpenStreetMap tags used the world over: landuse=cemetery and amenity=grave_yard
anywhere there are graves, landuse=military and the military=* tags anywhere
there is a base. Spacing is worked out in metres, so a map drawn at two metres
a tile gets the same graveyard as one drawn at half a metre.

The props go into tile-only lots, one per map cell, the way the fuel pumps and
the bridges do (knoxbuild/structures.py render_tiles_tbx).
"""
from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image

from generator import pz_colors as C

from . import catalog
from .bitmaps import read_rgb, same_colour
from .structures import CELL, render_tiles_tbx

# location_community_cemetary_01: headstones, wooden crosses and the angel on
# her plinth. Several shapes, because a churchyard where every stone is the
# same reads as a car park. Indices measured on the sheet (2026-10-13): 0-5,
# 8, 9 headstones; 11-14 angels; 16-21, 24-29, 32-35, 40-43 halves of open
# graves (two-square objects); 22, 23, 36, 37, 39 crosses; 44 on are empty
# (drawn as red question marks). The old lists had angels and open graves
# as headstones and empty slots as crosses.
HEADSTONES = ["location_community_cemetary_01_0", "location_community_cemetary_01_1",
              "location_community_cemetary_01_2", "location_community_cemetary_01_3",
              "location_community_cemetary_01_4", "location_community_cemetary_01_5",
              "location_community_cemetary_01_8", "location_community_cemetary_01_9"]
CROSSES = ["location_community_cemetary_01_22", "location_community_cemetary_01_23",
           "location_community_cemetary_01_36", "location_community_cemetary_01_37",
           "location_community_cemetary_01_39"]
ANGELS = ["location_community_cemetary_01_11", "location_community_cemetary_01_12",
          "location_community_cemetary_01_13", "location_community_cemetary_01_14"]
CROSS_SHARE = 0.12         # of the graves, a wooden cross instead of a stone
ANGEL_SHARE = 0.006        # and once in a while a monument
GRAVE_SPACING_M = 2.0      # along a row
GRAVE_ROW_M = 3.0          # between rows
PATH_EVERY_ROWS = 8        # a way through, so it is not one solid block
MAX_GRAVES = 40000         # a whole county of churchyards, and no more

# location_military_generic_01: ammunition crates, fuel drums and lockers.
# Nothing with English writing on it - the same base has to read as a base in
# any country the map is of.
# (KnoxMap UK: not _16 and _17 - the two halves of an "Army Property" sign,
# SpriteGridPos 0,0 and 1,0: laid one at a time as crates, each was half an
# object the game cannot remove; LOG 2026-10-19.)
CRATES = ["location_military_generic_01_8", "location_military_generic_01_9",
          "location_military_generic_01_0", "location_military_generic_01_1"]
DRUMS = ["location_military_generic_01_14", "location_military_generic_01_15",
         "location_military_generic_01_22", "location_military_generic_01_23"]
DUMP_EVERY_M = 26.0        # a stack of stores about this often across the site
DUMP_MIN_M2 = 400          # sites smaller than this get nothing
MAX_DUMPS = 600
PICNIC_SPACING_M = 38.0
PICNIC_MIN_M2 = 500
MAX_PICNIC_TABLES = 300

LAYER = "Furniture"
# Ground a prop never stands on: the carriageway, its pavement, and water.
ROAD = {C.DARK_ASPHALT, C.MEDIUM_ASPHALT, C.LIGHT_ASPHALT, C.DARKEST_ASPHALT,
    C.PALE_CONCRETE, C.DARK_POTHOLE, C.LIGHT_POTHOLE}
BLOCKED = ROAD | {C.WATER}


def lattice_on_map(first: int, stop: int, step: int, limit: int) -> range:
    """The values first, first + step, ... below `stop` that are on the map, 0..limit.

    A polygon's whole bounding box used to be walked and what fell off the map
    thrown away afterwards. A military range mapped as one polygon is a hundred
    kilometres across, and a few thousand headstone-sized steps become billions
    of Shapely calls. The values are the same ones on the same lattice; the ones
    below 0 or at `limit` and past are simply never visited."""
    if first < 0:
        first += -(-(-first) // step) * step         # the first value that is 0 or more
    return range(first, min(stop, limit), step)


def _blocked_mask(ground: np.ndarray) -> np.ndarray:
    out = np.zeros(ground.shape[:2], dtype=bool)
    for colour in BLOCKED:
        out |= same_colour(ground, colour)
    return out


def _structure_squares(out_dir: str, map_name: str, height: int, width: int) -> np.ndarray:
    """Ground squares the structures file lays something on (<map>_structures.json:
    bridges, monuments, the playgrounds and pitches of sports.py, listed
    structures). Its lot and the props' lot are separate, so a picnic table put
    on a square of a playground roundabout replaced half of it in the compile:
    a half multi-tile object, which the game hangs on (LOG 2026-10-18 #1,
    Westminster_graph 2026-10-22)."""
    import json
    out = np.zeros((height, width), dtype=bool)
    try:
        with open(os.path.join(out_dir, f"{map_name}_structures.json"), encoding="utf-8") as f:
            rows = json.load(f).get("tiles") or []
    except (OSError, ValueError, AttributeError):
        return out
    for x, y, z, *_ in rows:
        if z == 0 and 0 <= x < width and 0 <= y < height:
            out[y, x] = True
    return out


def _polygons(areas, category: str) -> list:
    """The land-use polygons of one category, in tiles."""
    return [shape for shape, props in getattr(areas, "_items", [])
            if (props.get("category") or "") == category and not shape.is_empty]


def place_props(out_dir: str, map_name: str, bdir: str, occupied: np.ndarray,
                areas, metres_per_tile: float,
                seed: int = 0x6FADE) -> tuple[list, dict]:
    """Headstones in the cemeteries and stores on the army bases.

    Returns (placements, counts). Tiles already taken by a building, a road or
    water are left alone, and anything put down is cleared of trees so a
    headstone does not grow out of a pine.
    """
    from shapely.geometry import Point

    from .world import Placement

    bmp = os.path.join(out_dir, f"{map_name}.bmp")
    counts = {"graves": 0, "dumps": 0, "picnic_tables": 0}
    if not os.path.exists(bmp) or areas is None:
        return [], counts
    cemeteries = _polygons(areas, "cemetery")
    bases = _polygons(areas, "military")
    parks = _polygons(areas, "park") + _polygons(areas, "playground")
    if not cemeteries and not bases and not parks:
        return [], counts

    ground = read_rgb(bmp)
    height, width = ground.shape[:2]
    blocked = _blocked_mask(ground)
    blocked |= occupied[:height, :width]
    blocked |= _structure_squares(out_dir, map_name, height, width)
    veg_path = os.path.join(out_dir, f"{map_name}_veg.bmp")
    veg = read_rgb(veg_path) if os.path.exists(veg_path) else None
    rng = random.Random(seed)
    tiles: list[tuple[int, int, int, str, str]] = []

    def put(x: int, y: int, tile: str) -> bool:
        if not (0 <= x < width and 0 <= y < height) or blocked[y, x]:
            return False
        blocked[y, x] = True
        occupied[y, x] = True
        if veg is not None:
            veg[y, x] = C.VEG_NOTHING
        tiles.append((x, y, 0, LAYER, tile))
        return True

    def step(metres: float) -> int:
        return max(1, int(round(metres / max(0.05, metres_per_tile))))

    across, down = step(GRAVE_SPACING_M), step(GRAVE_ROW_M)
    for poly in cemeteries:
        if counts["graves"] >= MAX_GRAVES:
            break
        minx, miny, maxx, maxy = (int(v) for v in poly.bounds)
        for y in lattice_on_map(miny + down, maxy, down, height):
            row = (y - (miny + down)) // down + 1     # counted from the polygon's own edge
            if row % PATH_EVERY_ROWS == 0:
                continue                       # a way through the plots
            for x in lattice_on_map(minx + across, maxx, across, width):
                if counts["graves"] >= MAX_GRAVES:
                    break
                if not poly.contains(Point(x + 0.5, y + 0.5)):
                    continue
                roll = rng.random()
                if roll < ANGEL_SHARE:
                    tile = rng.choice(ANGELS)
                elif roll < ANGEL_SHARE + CROSS_SHARE:
                    tile = rng.choice(CROSSES)
                else:
                    tile = rng.choice(HEADSTONES)
                if put(x, y, tile):
                    counts["graves"] += 1

    gap = step(DUMP_EVERY_M)
    for poly in bases:
        if counts["dumps"] >= MAX_DUMPS:
            break
        if poly.area * metres_per_tile * metres_per_tile < DUMP_MIN_M2:
            continue
        minx, miny, maxx, maxy = (int(v) for v in poly.bounds)
        for y in lattice_on_map(miny + gap, maxy, gap, height):
            for x in lattice_on_map(minx + gap, maxx, gap, width):
                if counts["dumps"] >= MAX_DUMPS:
                    break
                if not poly.contains(Point(x + 0.5, y + 0.5)):
                    continue
                # A stack of stores: a short row of crates with a drum or two
                # at the end of it, as they are stacked on a real apron.
                run = rng.randint(2, 5)
                horizontal = rng.random() < 0.5
                placed = 0
                for k in range(run):
                    tile = rng.choice(DRUMS if k == run - 1 and rng.random() < 0.4
                                      else CRATES)
                    px = x + (k if horizontal else 0)
                    py = y + (0 if horizontal else k)
                    placed += bool(put(px, py, tile))
                if placed:
                    counts["dumps"] += 1

    pitch = step(PICNIC_SPACING_M)
    for poly in parks:
        if counts["picnic_tables"] >= MAX_PICNIC_TABLES:
            break
        if poly.area * metres_per_tile * metres_per_tile < PICNIC_MIN_M2:
            continue
        minx, miny, maxx, maxy = poly.bounds
        x_start = int(minx) + pitch // 2
        y_start = int(miny) + pitch // 2
        for gy, y in enumerate(range(y_start, int(maxy), pitch)):
            for gx, x in enumerate(range(x_start, int(maxx), pitch)):
                if counts["picnic_tables"] >= MAX_PICNIC_TABLES:
                    break
                if rng.random() >= 0.72:
                    continue
                cx = x + rng.randint(-max(1, pitch // 5), max(1, pitch // 5))
                cy = y + rng.randint(-max(1, pitch // 5), max(1, pitch // 5))
                orient = ("W", "N")[(gx + gy + rng.randrange(2)) % 2]
                definition = catalog.FURNITURE["picnic_table"][orient]
                footprint = []
                for offset, tile in definition.items():
                    dx, dy = (int(value) for value in offset.split(","))
                    px, py = cx + dx, cy + dy
                    if (not (0 <= px < width and 0 <= py < height)
                            or blocked[py, px]
                            or not poly.contains(Point(px + 0.5, py + 0.5))):
                        footprint = []
                        break
                    footprint.append((px, py, tile))
                if len(footprint) != len(definition):
                    continue
                for px, py, tile in footprint:
                    put(px, py, tile)
                counts["picnic_tables"] += 1

    if not tiles:
        return [], counts
    if veg is not None:
        Image.fromarray(veg).save(veg_path, format="BMP")

    by_cell: dict[tuple[int, int], list] = {}
    for t in tiles:
        by_cell.setdefault((t[0] // CELL, t[1] // CELL), []).append(t)
    placements = []
    for (cx, cy), group in sorted(by_cell.items()):
        gx0 = min(t[0] for t in group)
        gy0 = min(t[1] for t in group)
        gw = max(t[0] for t in group) - gx0 + 1
        gh = max(t[1] for t in group) - gy0 + 1
        fname = f"{map_name}_props_{cx}_{cy}.tbx"
        with open(os.path.join(bdir, fname), "w", encoding="utf-8") as f:
            f.write(render_tiles_tbx(gw, gh, [(x - gx0, y - gy0, z, layer, tile)
                                              for x, y, z, layer, tile in group]))
        placements.append(Placement(f"buildings/{fname}", gx0, gy0, gw, gh))
    return placements, counts
