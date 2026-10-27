"""KnoxMap UK: picnic benches in a pub's beer garden.

A pub the regional step marks `beer_garden` (Building_types export_regional.uk_pubs)
gets up to BENCHES picnic benches (asset_creator/commercial_kit/pub_props.py, its
tiles inlined in regional.json `pub_props`) on the open ground behind it - the side
away from its street, else its two ends, never its street front (a pub whose street
is not known gets none): each bench two squares along the pub's wall, in rows ROWS
squares out from it, a square clear round it, on ground no building, lot, road, water,
pavement, forecourt or car park holds. Laid as pumps.py lays the pumps: in
tile-only lots, one per map cell.
"""
from __future__ import annotations

import os

import numpy as np

from generator import pz_colors as C

from .bitmaps import read_rgb, same_colour
from .pumps import NOT_GROUND, ROAD, SIDES
from .structures import CELL, render_tiles_tbx

BENCHES = 3
ROWS = (2, 5)               # squares out from the wall, a row of benches
SPACING = 3                 # squares between benches' starts along a row (a bench is 2)
BACK = {"N": "S", "S": "N", "E": "W", "W": "E"}


def place_benches(out_dir: str, map_name: str, bdir: str, occupied: np.ndarray,
                  pubs: list, overlay: dict | None) -> tuple[list, int]:
    """Benches behind each pub of `pubs` - (x0, y0, width, height, street side or None).
    Returns (placements, benches placed)."""
    from .world import Placement

    parts = ((overlay or {}).get("pub_props") or {}).get("parts")
    bmp = os.path.join(out_dir, f"{map_name}.bmp")
    if not pubs or not parts or not os.path.exists(bmp):
        return [], 0
    ground = read_rgb(bmp)
    H, W = ground.shape[:2]
    blocked = occupied[:H, :W].copy()
    # nor on pavement, a forecourt or a car park (review: a corner pub's end on a side
    # street had its benches on the pavement)
    for colour in ROAD | NOT_GROUND | {C.PALE_CONCRETE, C.DARK_ASPHALT}:
        blocked |= same_colour(ground, colour)

    def clear(x, y, w, h):
        return (x >= 0 and y >= 0 and x + w <= W and y + h <= H
                and not blocked[y:y + h, x:x + w].any())

    tiles, n = [], 0
    for x0, y0, w, h, street in pubs:
        if street not in BACK:
            continue                                # no street known: its front is not known either
        back = BACK[street]
        sides = [back] + [s for s in SIDES if s not in (back, street)]
        placed = 0
        for side in sides:
            dx, dy = SIDES[side]
            along_x = dx == 0                       # a north or south wall: benches along x
            length = w if along_x else h
            for out in ROWS:
                for t in range(0, length - 1, SPACING):
                    if placed >= BENCHES:
                        break
                    if along_x:
                        bx = x0 + t
                        by = y0 + h - 1 + out if dy > 0 else y0 - out
                        cells = [(bx, by), (bx + 1, by)]
                        names = (parts["bench_x_0"], parts["bench_x_1"])
                    else:
                        by = y0 + t
                        bx = x0 + w - 1 + out if dx > 0 else x0 - out
                        cells = [(bx, by), (bx, by + 1)]
                        names = (parts["bench_y_0"], parts["bench_y_1"])
                    # its own squares and one round them clear (room to sit, to walk past)
                    cx0, cy0 = min(c[0] for c in cells) - 1, min(c[1] for c in cells) - 1
                    cw, ch = (4, 3) if along_x else (3, 4)
                    if not clear(cx0, cy0, cw, ch):
                        continue
                    for (cx, cy), name in zip(cells, names):
                        tiles.append((cx, cy, 0, "Furniture", name))
                        blocked[cy, cx] = True
                        occupied[cy, cx] = True
                    placed += 1
            if placed:
                break                               # one side's garden
        n += placed
    if not tiles:
        return [], 0
    by_cell: dict[tuple[int, int], list] = {}
    for t in tiles:
        by_cell.setdefault((t[0] // CELL, t[1] // CELL), []).append(t)
    placements = []
    for (cx, cy), group in sorted(by_cell.items()):
        gx0, gy0 = min(t[0] for t in group), min(t[1] for t in group)
        gw = max(t[0] for t in group) - gx0 + 1
        gh = max(t[1] for t in group) - gy0 + 1
        fname = f"{map_name}_benches_{cx}_{cy}.tbx"
        with open(os.path.join(bdir, fname), "w", encoding="utf-8") as f:
            f.write(render_tiles_tbx(gw, gh, [(x - gx0, y - gy0, z, layer, tile) for x, y, z, layer, tile in group]))
        placements.append(Placement(f"buildings/{fname}", gx0, gy0, gw, gh))
    return placements, n
