"""UK foliage and grasses, rendered at the game's camera (street_objects.py's).

    python foliage_textures.py          (first: the cards' pictures, PIL)
    "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b -P foliage.py -- [slot ...]

Writes asset_creator/renders/vegetation_uk_01/ as street_objects.py writes
its sheets (sprite, map positions, manifest); then street_sheet.py makes
the sheet and its depth picture and Tile_creator/src/pack.py packs it.

Low-poly the way game foliage is made: a plant is a few clumps, each two
crossed cards (8 triangles) carrying a picture of the plant
(foliage_textures.py), cut out by its alpha; a shrub is two or three
lobes, each a small dark core with leaf-spray cards over it (seventy in all), which gives the broken, leafy edge of
vanilla's bushes. Clumps and cards are placed from a seed per slot, so a
render is the same every time. Lit as everything else (walls.py's flat
light): cards facing every way shade a plant from 0.7 to 1.0 by itself.

Sizes as vanilla's plants (measured from d_plants_1, blends_grassoverlays_01,
vegetation_foliage_01): grass to about 0.7 m, nettles and cow parsley to
1.1, foxgloves to 1.3, shrubs 1.1-1.8 m, all within their square.
"""
from __future__ import annotations

import math
import os
import random
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import street_objects as so  # noqa: E402

TEX = os.path.join(so.RENDERS, "foliage")
PX = 400                                    # foliage_textures.py's px a metre
SHEET = "vegetation_uk_01"


def tex(name):
    if name not in so._FACES:
        so._FACES[name] = bpy.data.images.load(os.path.join(TEX, name + ".png"))
    return so._FACES[name]


def card_material(name):
    img = tex(name)
    return so.material(f"fol_{name}", image=img, interp="Linear", alpha=img, grime=0.0), img


def cross(cx, cy, species, k, scale, turn):
    """Two crossed cards of species' picture k, standing at (cx, cy)."""
    mat, img = card_material(f"{species}_{k}")
    w, h = img.size[0] / PX * scale, img.size[1] / PX * scale
    for a in (turn, turn + 90):
        ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
        so.quad((cx - ux * w / 2, cy - uy * w / 2, 0.0), (ux * w, uy * w, 0.0), (0.0, 0.0, h), mat, (-uy, ux, 0.0))


def plants(species, clumps, seed, scale=(0.8, 1.15), spread=0.34):
    """`clumps` clumps of a species over the square."""
    rng = random.Random(seed)
    for n in range(clumps):
        cx, cy = 0.5 + rng.uniform(-spread, spread), 0.5 + rng.uniform(-spread, spread)
        cross(cx, cy, species, rng.randrange(4), rng.uniform(*scale), rng.uniform(0, 90))
    return {"geometry": []}


def meadow(flowers, seed):
    """Long grass over the whole square, with its flowers (or seed heads) among it."""
    rng = random.Random(seed)
    for n in range(7):
        species = flowers if (flowers and n % 2) else "grass"
        cx, cy = 0.5 + rng.uniform(-0.34, 0.34), 0.5 + rng.uniform(-0.34, 0.34)
        cross(cx, cy, species, rng.randrange(4), rng.uniform(0.8, 1.1), rng.uniform(0, 90))
    return {"geometry": []}


SHRUBS = {   # species: (height m, half-width m, its core's colour)
    "hawthorn": (1.6, 0.44, (38, 62, 30)),
    "elder": (1.8, 0.45, (50, 78, 36)),
    "holly": (1.7, 0.40, (22, 46, 26)),
    "gorse": (1.1, 0.46, (46, 64, 30)),
}


def shrub(species, seed):
    """A shrub: two or three lobes (an irregular bush, not an egg), each a
    small dark core and leaf sprays over an ellipsoid round it."""
    rng = random.Random(seed)
    height, half, core = SHRUBS[species]
    sprays = [card_material(f"{species}_{k}")[0] for k in range(4)]
    core_mat = so.material(f"fol_core_{species}", core, grime=0.15)
    size = 0.36
    lobes = []
    for n in range(rng.choice((2, 3))):
        r = half * rng.uniform(0.62, 0.8)
        hz = height * rng.uniform(0.32, 0.45)
        cx = 0.5 + rng.uniform(-1, 1) * (half - r)
        cy = 0.5 + rng.uniform(-1, 1) * (half - r)
        cz = height - hz if n == 0 else hz * rng.uniform(0.9, 1.3)
        lobes.append((np.array((cx, cy, cz)), np.array((r, r, hz))))
        so.sphere(cx, cy, cz, r * 0.6, core_mat, segments=8, rings=6, squash_z=hz / r)
    for _ in range(70):
        c, radii = rng.choice(lobes)
        d = np.array([rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0.3, 1)])
        d /= np.linalg.norm(d) or 1.0
        p = c + d * radii * rng.uniform(0.75, 1.0)
        p[0], p[1] = np.clip(p[:2], 0.05 + size / 2, 0.95 - size / 2)
        p[2] = max(p[2], size * 0.45)
        n = d
        u = np.cross(n, (0, 0, 1.0))
        if np.linalg.norm(u) < 1e-3:
            u = np.array((1.0, 0, 0))
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        a = rng.uniform(0, 2 * math.pi)
        u, v = u * math.cos(a) + v * math.sin(a), -u * math.sin(a) + v * math.cos(a)
        so.quad(tuple(p - (u + v) * size / 2), tuple(u * size), tuple(v * size), rng.choice(sprays), tuple(n))
    return {"geometry": []}


slot = so.slot
SLOTS = [
    slot("meadow_grass", meadow, (None, 1)), slot("meadow_seeding", meadow, ("grass_seeding", 2)),
    slot("meadow_buttercups", meadow, ("grass_buttercups", 3)), slot("meadow_daisies", meadow, ("grass_daisies", 4)),
    slot("nettles_a", plants, ("nettles", 3, 11)), slot("nettles_b", plants, ("nettles", 4, 12)),
    slot("cow_parsley_a", plants, ("cow_parsley", 3, 13)), slot("cow_parsley_b", plants, ("cow_parsley", 2, 14)),
    slot("bracken_a", plants, ("bracken", 2, 15, (0.8, 1.0), 0.2)), slot("bracken_b", plants, ("bracken", 3, 16, (0.7, 0.9), 0.25)),
    slot("brambles_flowering", plants, ("brambles_flowering", 2, 17, (0.8, 1.0), 0.2)),
    slot("brambles_fruiting", plants, ("brambles_fruiting", 2, 18, (0.8, 1.0), 0.2)),
    slot("foxgloves", plants, ("foxglove", 2, 19, (0.8, 1.05), 0.25)),
    slot("bluebells", plants, ("bluebells", 5, 20)), slot("daffodils", plants, ("daffodils", 4, 21)),
    slot("rushes", plants, ("rushes", 3, 22, (0.8, 1.1), 0.25)),
    slot("hawthorn", shrub, ("hawthorn", 23)), slot("elder", shrub, ("elder", 24)),
    slot("holly", shrub, ("holly", 25)), slot("gorse", shrub, ("gorse", 26)),
    slot("empty"), slot("empty"), slot("empty"), slot("empty"),
]
so.SHEETS[SHEET] = SLOTS

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    so.render_sheet(SHEET, set(args))
