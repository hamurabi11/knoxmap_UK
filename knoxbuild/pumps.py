"""Fuel pumps on the forecourt of every petrol station.

A petrol station was only its shop: amenity=fuel made the building a gas
station store inside, and outside it stood in the grass with nowhere to fill a
car. Here each one gets a forecourt of tarmac on the side facing the street
and a row of the game's own pumps down the middle of it - the Fossoil and
Gas 2 Go pumps Knox County's stations have, which hold fuel (fuelAmount) and
work with a hose and a jerry can.

A station mapped only as a point, with no building around it, gets the
forecourt and pumps where the point is.

The pumps go into tile-only lots, one per map cell, like the fences and the
bridges (see structures.py).

KnoxMap UK (a map whose regional.json carries `fuel`: asset_creator/commercial_kit/
petrol.py's tiles): a 1993 British forecourt in place of Knox County's - its own pumps
(each with vanilla's pump's properties), a canopy over them on four columns with its
brand's band round its edge (lettered on the south and east, the camera's sides) and
a price sign on its pole at the forecourt's street corner.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from generator import pz_colors as C

from .bitmaps import read_rgb, same_colour
from .structures import CELL, render_tiles_tbx

# One tile each, holding 20000 units of fuel. The pump facing south stands in
# a row that runs west to east, the one facing east in a row north to south.
PUMPS = {
    "fossoil": {"x": "location_shop_fossoil_01_14", "y": "location_shop_fossoil_01_12"},
    "gas2go": {"x": "location_shop_gas2go_01_14", "y": "location_shop_gas2go_01_12"},
}
FORECOURT_DEPTH = 8         # tiles out from the shop front
FORECOURT_MIN_DEPTH = 6
FORECOURT_MIN_LENGTH = 9
PUMP_SPACING = 4
FORECOURT_REACH = 6         # how much further it may run to meet the street
MAX_PUMPS = 4
# Carriageways: a forecourt stops short of the street rather than covering it.
ROAD = {C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE}
NOT_GROUND = {C.WATER}

SIDES = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}


CANOPY_REACH = 2            # squares the canopy runs beyond its pumps, along the row and either side


def _uk_fuel(out_dir: str) -> dict | None:
    """The commercial kit's forecourt tiles, inlined in the map's regional.json (none
    with the map's UK assets off: the game's own forecourt)."""
    from .assets_uk import off
    if off(out_dir):
        return None
    path = os.path.join(out_dir, "regional.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("fuel")
    except (OSError, ValueError):
        return None


def _uk_canopy(tiles, fuel, rect, pumps_at, along_x, street):
    """A canopy over the pumps: its top a storey up (RoofTop), a column at each corner,
    its band on each edge (Furniture2 north and south, Furniture3 east and west: a
    corner holds two), the brand's name across the south and east bands' middles; and
    the price sign at the forecourt's corner on the street (`rect`), facing S or E."""
    parts = fuel["parts"]
    x, y, w, h = rect
    pts = [(t[0], t[1]) for t in pumps_at]
    if along_x:
        xa, xb = min(p[0] for p in pts) - CANOPY_REACH, max(p[0] for p in pts) + CANOPY_REACH
        ya, yb = pts[0][1] - CANOPY_REACH, pts[0][1] + CANOPY_REACH
    else:
        ya, yb = min(p[1] for p in pts) - CANOPY_REACH, max(p[1] for p in pts) + CANOPY_REACH
        xa, xb = pts[0][0] - CANOPY_REACH, pts[0][0] + CANOPY_REACH
    xa, xb, ya, yb = max(xa, x), min(xb, x + w - 1), max(ya, y), min(yb, y + h - 1)
    if xb - xa < 2 or yb - ya < 2:
        return
    taken = set(pts)
    n = len(parts["band_S_letters"])
    for cx in range(xa, xb + 1):
        for cy in range(ya, yb + 1):
            tiles.append((cx, cy, 1, "RoofTop", parts["roof"]))
    for cx, cy in ((xa, ya), (xb, ya), (xa, yb), (xb, yb)):
        if (cx, cy) not in taken:
            tiles.append((cx, cy, 0, "Furniture", parts["column"]))
            taken.add((cx, cy))
    s0, e0 = (xa + xb + 1 - n) // 2, (ya + yb + 1 - n) // 2
    for cx in range(xa, xb + 1):
        k = cx - s0
        tiles.append((cx, yb, 0, "Furniture2", parts["band_S_letters"][k] if 0 <= k < n else parts["band_S_plain"]))
        tiles.append((cx, ya, 0, "Furniture2", parts["band_N_plain"]))
    ne = len(parts["band_E_letters"])
    e0 = (ya + yb + 1 - ne) // 2
    for cy in range(ya, yb + 1):
        k = cy - e0
        tiles.append((xb, cy, 0, "Furniture3", parts["band_E_letters"][k] if 0 <= k < ne else parts["band_E_plain"]))
        tiles.append((xa, cy, 0, "Furniture3", parts["band_W_plain"]))
    # the price sign: the forecourt's street corner, clear of the canopy's squares
    corner = {"S": (x, y + h - 1), "N": (x, y), "E": (x + w - 1, y), "W": (x, y)}.get(street, (x, y + h - 1))
    if corner not in taken and not (xa <= corner[0] <= xb and ya <= corner[1] <= yb):
        face = "price_e" if street in ("E", "W") else "price"   # (a W street's faces east too: only S and E are seen)
        tiles.append((corner[0], corner[1], 0, "Furniture", parts[f"{face}_lo"]))
        tiles.append((corner[0], corner[1], 1, "Furniture", parts[f"{face}_hi"]))


def _forecourt(x0, y0, w, h, side, depth):
    """The rectangle (x, y, width, height) in front of a footprint's side."""
    length_x = max(w, FORECOURT_MIN_LENGTH)
    length_y = max(h, FORECOURT_MIN_LENGTH)
    fx = x0 + (w - length_x) // 2
    fy = y0 + (h - length_y) // 2
    if side == "S":
        return fx, y0 + h + 1, length_x, depth
    if side == "N":
        return fx, y0 - 1 - depth, length_x, depth
    if side == "E":
        return x0 + w + 1, fy, depth, length_y
    return x0 - 1 - depth, fy, depth, length_y


def place_pumps(out_dir: str, map_name: str, bdir: str, occupied: np.ndarray,
                stations: list, points: list,
                forecourts: list | None = None,
                canopies: list | None = None) -> tuple[list, int]:
    """Forecourts and pumps for `stations` - (x0, y0, width, height, street
    side or None) of each station's building - and `points`, (x, y) of the
    stations mapped with no building, and `canopies`, the outlines of the
    roofs mapped over forecourts. Returns (placements, pumps placed), and
    adds each forecourt (x, y, width, height) to `forecourts` when given."""
    from .world import Placement

    bmp = os.path.join(out_dir, f"{map_name}.bmp")
    canopies = canopies or []
    if not (stations or points or canopies) or not os.path.exists(bmp):
        return [], 0
    ground = read_rgb(bmp)
    veg_path = os.path.join(out_dir, f"{map_name}_veg.bmp")
    veg = read_rgb(veg_path) if os.path.exists(veg_path) else None
    H, W = ground.shape[:2]

    blocked = occupied[:H, :W].copy()
    for colour in ROAD | NOT_GROUND:
        blocked |= same_colour(ground, colour)

    def clear(x, y, w, h):
        return (x >= 0 and y >= 0 and x + w <= W and y + h <= H
                and not blocked[y:y + h, x:x + w].any())

    tiles: list[tuple[int, int, int, str, str]] = []
    pumps = 0
    uk = _uk_fuel(out_dir)

    paved = np.zeros((H, W), dtype=bool)
    for colour in ROAD | {C.PALE_CONCRETE, C.DARK_ASPHALT}:
        paved |= same_colour(ground, colour)

    def reach(x, y, w, h, side):
        """The forecourt run on out to the pavement or road when that is a
        few tiles further, so cars can drive onto it."""
        dx, dy = SIDES[side]
        for _ in range(FORECOURT_REACH):
            if dx:
                nx = x + w if dx > 0 else x - 1
                row = (nx, y, 1, h)
            else:
                ny = y + h if dy > 0 else y - 1
                row = (x, ny, w, 1)
            rx, ry, rw, rh = row
            if not (rx >= 0 and ry >= 0 and rx + rw <= W and ry + rh <= H):
                break
            if paved[ry:ry + rh, rx:rx + rw].any():
                return x, y, w, h
            if blocked[ry:ry + rh, rx:rx + rw].any():
                break
            if dx:
                x, w = (x, w + 1) if dx > 0 else (x - 1, w + 1)
            else:
                y, h = (y, h + 1) if dy > 0 else (y - 1, h + 1)
        return None

    def lay(x, y, w, h, brand, pumps_at=None, side=None):
        nonlocal pumps
        if forecourts is not None:
            forecourts.append((x, y, w, h))
        ground[y:y + h, x:x + w] = C.DARK_ASPHALT
        if veg is not None:
            veg[y:y + h, x:x + w] = 0
        blocked[y:y + h, x:x + w] = True
        occupied[y:y + h, x:x + w] = True
        # The pumps in a row along the shop front, halfway across the part of
        # the forecourt nearest the shop.
        px0, py0, pw, ph = pumps_at or (x, y, w, h)
        along_x = pw >= ph
        length = pw if along_x else ph
        count = max(1, min(MAX_PUMPS, (length - 2) // PUMP_SPACING))
        start = (length - (count - 1) * PUMP_SPACING) // 2
        row = []
        for k in range(count):
            t = start + k * PUMP_SPACING
            px, py = (px0 + t, py0 + ph // 2) if along_x else (px0 + pw // 2, py0 + t)
            pump = uk["parts"]["pump_x" if along_x else "pump_y"] if uk else PUMPS[brand]["x" if along_x else "y"]
            tiles.append((px, py, 0, "Furniture", pump))
            row.append((px, py))
            pumps += 1
        if uk:
            _uk_canopy(tiles, uk, (x, y, w, h), row, along_x, side)

    def brand_for(x, y):
        return "fossoil" if (x * 7 + y * 13) % 2 else "gas2go"

    # Canopies first: they are where the pumps really are. The ground under
    # one becomes forecourt, and its pumps stand in one row down the middle,
    # or two when it is deep enough for cars either side of each.
    from shapely.geometry import Polygon, box
    done_near: list[tuple[float, float]] = []
    for outline in canopies:
        shape = Polygon(outline)
        if not shape.is_valid:
            shape = shape.buffer(0)
        if shape.is_empty:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in shape.bounds)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        cw, ch = x1 - x0, y1 - y0
        if cw < 3 or ch < 3:
            continue
        under = np.zeros((ch, cw), dtype=bool)
        for yy in range(ch):
            for xx in range(cw):
                under[yy, xx] = shape.contains(box(x0 + xx + .25, y0 + yy + .25,
                                                   x0 + xx + .75, y0 + yy + .75))
        under &= ~blocked[y0:y1, x0:x1]
        if under.sum() < 12:
            continue
        ground[y0:y1, x0:x1][under] = C.DARK_ASPHALT
        if veg is not None:
            veg[y0:y1, x0:x1][under] = 0
        along_x = cw >= ch
        depth = ch if along_x else cw
        rows = [depth // 3, depth - 1 - depth // 3] if depth >= 9 else [depth // 2]
        brand = brand_for(x0, y0)
        placed = 0
        length = cw if along_x else ch
        count = max(1, min(MAX_PUMPS, (length - 2) // PUMP_SPACING))
        step = length / count
        for r in rows:
            for k in range(count):
                t = int(step * (k + 0.5))
                xx, yy = (t, r) if along_x else (r, t)
                if under[yy, xx]:
                    tiles.append((x0 + xx, y0 + yy, 0, "Furniture",
                                  uk["parts"]["pump_x" if along_x else "pump_y"] if uk else PUMPS[brand]["x" if along_x else "y"]))
                    placed += 1
        pumps += placed
        blocked[y0:y1, x0:x1] |= under
        occupied[y0:y1, x0:x1] |= under
        if forecourts is not None:
            forecourts.append((x0, y0, cw, ch))
        if placed:
            done_near.append((x0 + cw / 2, y0 + ch / 2))

    def near_canopy(x, y, reach=40):
        return any(abs(x - cx) < reach and abs(y - cy) < reach for cx, cy in done_near)

    for x0, y0, w, h, side in stations:
        if near_canopy(x0 + w / 2, y0 + h / 2):
            continue          # its pumps are under its canopy already
        order = ([side] if side in SIDES else []) + [s for s in SIDES if s != side]
        done = False
        for depth in (FORECOURT_DEPTH, FORECOURT_MIN_DEPTH):
            for s in order:
                rect = _forecourt(x0, y0, w, h, s, depth)
                if clear(*rect):
                    lay(*(reach(*rect, s) or rect), brand_for(x0, y0), pumps_at=rect, side=s)
                    done = True
                    break
            if done:
                break

    for px, py in points:
        px, py = int(px), int(py)
        if near_canopy(px, py):
            continue
        for fw, fh in ((12, FORECOURT_DEPTH), (FORECOURT_DEPTH, 12),
                       (FORECOURT_MIN_LENGTH, FORECOURT_MIN_DEPTH),
                       (FORECOURT_MIN_DEPTH, FORECOURT_MIN_LENGTH)):
            rect = (px - fw // 2, py - fh // 2, fw, fh)
            if clear(*rect):
                lay(*rect, brand_for(px, py))
                break

    if not tiles:
        return [], 0
    Image.fromarray(ground).save(bmp, format="BMP")
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
        fname = f"{map_name}_pumps_{cx}_{cy}.tbx"
        with open(os.path.join(bdir, fname), "w", encoding="utf-8") as f:
            f.write(render_tiles_tbx(gw, gh, [(x - gx0, y - gy0, z, layer, tile)
                                              for x, y, z, layer, tile in group]))
        placements.append(Placement(f"buildings/{fname}", gx0, gy0, gw, gh))
    return placements, pumps
