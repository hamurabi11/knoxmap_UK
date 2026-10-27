"""An easter egg on a KnoxMap UK map (user, 2026-10-14): one blue police
box, the Metropolitan Police's of 1929 (asset_creator: street_furniture_uk_02
22-23, facing S or E), standing alone in a field.

The field: open grass (the ground grass, nothing growing on it, no building)
well away from any road or paving (FIELD_ROAD), any building (FIELD_HOUSE)
and the map's edge, with open grass all round it (FIELD_OPEN): out in the
farmland, not in a garden or a park. Which of those squares, and which way
it faces, is chosen at random from a seed of the map's name: the same box
on every build of a map, a different field on each map.
"""
from __future__ import annotations

import os
import random

import numpy as np

from generator import pz_colors as C

from .bitmaps import read_rgb, same_colour

POLICE_BOX = {"S": "street_furniture_uk_02_22", "E": "street_furniture_uk_02_23"}
FIELD_ROAD = 30          # tiles from any road or paving, at least
FIELD_HOUSE = 25         # ...from any building
FIELD_OPEN = 3           # open grass this far all round
EDGE = 40                # ...and from the map's edge
GRASS = (C.DARK_GRASS, C.MEDIUM_GRASS, C.LIGHT_GRASS)
PAVED = (C.PALE_CONCRETE, C.MEDIUM_ASPHALT, C.DARK_ASPHALT, C.DARKEST_ASPHALT, C.LIGHT_ASPHALT,
         C.PAVING_STONE)


def police_box(out_dir: str, map_name: str, occupied) -> list:
    """The box's loose tile, (x, y, level, layer, tile), or none where no field will do."""
    from scipy import ndimage
    bmp = os.path.join(out_dir, f"{map_name}.bmp")
    veg_path = os.path.join(out_dir, f"{map_name}_veg.bmp")
    if not os.path.exists(bmp):
        return []
    ground = read_rgb(bmp)
    h, w = ground.shape[:2]
    grass = np.zeros((h, w), bool)
    for c in GRASS:
        grass |= same_colour(ground, c)
    paved = np.zeros((h, w), bool)
    for c in PAVED:
        paved |= same_colour(ground, c)
    open_ = grass & ~occupied[:h, :w]
    if os.path.exists(veg_path):
        open_ &= same_colour(read_rgb(veg_path), C.VEG_NOTHING)
    far_road = ndimage.distance_transform_edt(~paved) >= FIELD_ROAD if paved.any() else np.ones((h, w), bool)
    far_house = ndimage.distance_transform_edt(~occupied[:h, :w]) >= FIELD_HOUSE
    clear = ndimage.binary_erosion(open_, iterations=FIELD_OPEN)
    ok = clear & far_road & far_house
    ok[:EDGE, :] = ok[-EDGE:, :] = False
    ok[:, :EDGE] = ok[:, -EDGE:] = False
    ys, xs = np.nonzero(ok)
    if not len(xs):
        return []
    rng = random.Random(f"police box:{map_name}")
    k = rng.randrange(len(xs))
    facing = rng.choice(("S", "E"))
    return [(int(xs[k]), int(ys[k]), 0, "Furniture", POLICE_BOX[facing])]
