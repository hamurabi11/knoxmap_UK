"""The UK tile packs as Build 42 mods: .pack, .tiles and mod.info.

    python pack.py [--install]

Reads the sheets in build/sheets/<group>/ (sheet_paths.py; SHEETS below), writes
build/mod/<pack>/ (42/mod.info, common/mod.info, common/media/...) for each of
PACKS - KnoxMapUK_Tiles (every map) and KnoxMapUK_Westminster (Westminster's own
sheets) - and with --install copies them into the game's mods folder (~/Zomboid/mods).

Formats, as Erika's Tiles ships them (decoded 2026-10-12):
- .pack ("PZPK", version 1): one page per sheet, mask 1; each sprite cropped
  to its visible box, with its offset in the 128 x 256 frame (a floor:
  126 x 64 at 0, 192). The page's PNG is the sheet itself.
- .tiles ("tdef", version 1): per tileset its name, image, columns, rows, id
  and per tile its properties; strings end in a newline, ints little-endian.
  The game numbers a sprite tiledef * 100000 + tileset id * 1000 + index, so
  TILEDEF must be unique among the mods a player runs (Erika's is 1991).
  A readable copy goes beside it as .tiles.txt.

Properties are copied from the vanilla tile each slot stands in for (read
from the game's own newtiledefinitions.tiles): data, not art.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import struct
import sys

import numpy as np
from PIL import Image
from sheet_paths import sheet_file

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "..", "build")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # asset_creator: acpaths
import acpaths  # noqa: E402
KNOXMAP = acpaths.KNOX

PACK = "KnoxMapUK_Tiles"
NAME = "KnoxMap UK Tiles"
TILEDEF = 7342               # working number; to settle before the first release (APPROACH.md 5)


def floor_set(base: int) -> list:
    """A floor set's 16 slots -> the vanilla set starting at base (floor_tiles.py's layout)."""
    return [base + i for i in range(12)] + [None] * 4


# sheet: (columns, rows, vanilla sheet its slots copy properties from, slot -> vanilla index).
# A sheet's tileset id is its place in this list (write_tdef): add new sheets
# at the end, or every sprite id after it changes and saved worlds draw the
# wrong tiles.
SHEETS = {
    "blends_street_uk_01": (8, 2, "blends_street_01", floor_set(96)),    # pavement flags (street)
    "blends_street_uk_02": (8, 2, "blends_street_01", floor_set(80)),    # carriageway (street2)
    "blends_street_uk_03": (8, 2, "blends_street_01", floor_set(32)),    # worn tarmac (street4)
    "blends_natural_uk_01": (8, 6, "blends_natural_01", [16 + i for i in range(16)] + [32 + i for i in range(16)]
                             + [48 + i for i in range(16)]),
    "street_curbs_uk_01_blend_pavement": (8, 11, "street_curbs_01_blend_gravel", list(range(88))),
    "street_curbs_uk_01_blend_dark_grass": (8, 11, "street_curbs_01_blend_dark_grass", list(range(88))),
    "street_curbs_uk_01_blend_medium_grass": (8, 11, "street_curbs_01_blend_medium_grass", list(range(88))),
    "street_curbs_uk_01_blend_light_grass": (8, 11, "street_curbs_01_blend_light_grass", list(range(88))),
    "street_curbs_uk_01_blend_street": (8, 11, "street_curbs_01_blend_street", list(range(88))),
    # Garden walls (walls.py): low ones as vanilla's low brick wall (hoppable),
    # tall ones as its tall concrete fence (climbable); slots name their vanilla tile.
    "walls_garden_uk_01": (8, 2, None, [f"walls_exterior_house_low_01_{i}" for i in (12, 12, 13, 13, 14, 15)] + [None] * 2
                           + [f"fencing_01_{i}" for i in (40, 40, 41, 41, 42, 43)] + [None] * 2),
}
# Angled boundaries (diag_walls.py): per style 14 pictures (no properties:
# seen, not solid) and three invisible pieces with the style's own straight
# pieces' properties (west, north, both), which Build lays on the staircase.
DIAG_STYLE_TILES = {   # style: its straight west, north and corner pieces (vanilla names)
    "uk_brick_low": ("walls_exterior_house_low_01_12", "walls_exterior_house_low_01_13", "walls_exterior_house_low_01_14"),
    "uk_brick_tall": ("fencing_01_40", "fencing_01_41", "fencing_01_42"),
    "hedge_low": ("vegetation_ornamental_01_5", "vegetation_ornamental_01_4", "vegetation_ornamental_01_6"),
    "hedge_tall": ("vegetation_ornamental_01_1", "vegetation_ornamental_01_0", "vegetation_ornamental_01_2"),
    "short_wooden": ("fencing_01_35", "fencing_01_32", "fencing_01_36"),
    "tall_wooden": ("fencing_01_11", "fencing_01_8", "fencing_01_12"),
    "black_metal": ("fencing_01_2", "fencing_01_1", "fencing_01_3"),
}
DIAG_ORDER = ["uk_brick_low", "uk_brick_tall", "hedge_low", "hedge_tall", "short_wooden", "tall_wooden", "black_metal"]
SHEETS["boundaries_diag_uk_01"] = (8, 21, None, [v for st in DIAG_ORDER
                                                 for v in [None] * 16 + list(DIAG_STYLE_TILES[st]) + [None] * 5])
# Dirt from the user's photo (dirt.py), after every other sheet (a sheet's id is its place).
SHEETS["blends_natural_uk_02"] = (8, 2, "blends_natural_01", floor_set(64))
# 1990s street furniture (asset_creator/src/street_objects.py, Blender; street_sheet.py):
# pillar boxes and K6 / KX100 kiosks facing S, E, N, W, Belisha beacons (unlit,
# lit), a cone, then traffic signals (red, red + amber, green, amber; each
# facing S, E, N, W). Each stands in for vanilla's mail box, pay phone, post,
# cone and signal head.
_SD, _LO = "street_decoration_01_", "lighting_outdoor_01_"
SHEETS["street_furniture_uk_01"] = (8, 4, None, [_SD + str(i) for i in (8, 9, 10, 11)]
                                    + [_SD + str(i) for i in (39, 38, 47, 46)] * 2
                                    + [_SD + "56", _SD + "56", _SD + "26", None]
                                    + [_LO + str(i) for i in (12, 13, 14, 15)] * 4)
# More of it (street_objects.py): TV aerials (vanilla's old antenna), benches
# (its two-tile park bench, piece for piece), bollards and keep-left bollards
# (the barrier post), road works barriers (the short wooden fence: a W or N
# edge, hoppable), road signs and town signs (the signal head, stripped as
# above), front doors (vanilla's door, four sprites a door as its are).
_FS, _AN = "furniture_seating_outdoor_01_", "appliances_com_01_80"
SHEETS["street_furniture_uk_02"] = (8, 3, None, [_AN] * 4 + [_FS + str(i) for i in range(8)]
                                    + [_SD + "56"] * 6
                                    + ["fencing_01_35", "fencing_01_32", "fencing_01_35", "fencing_01_32"]
                                    + [_SD + "39", _SD + "38"])      # the police box (an easter egg), S and E
SHEETS["road_signs_uk_01"] = (8, 6, None, [_LO + "12"] * 48)
SHEETS["doors_uk_01"] = (8, 4, None, [f"fixtures_doors_01_{i}" for i in range(4)] * 8)
# UK foliage (asset_creator/src/foliage.py): meadow grass, nettles, cow
# parsley, bracken, brambles, foxgloves, bluebells, daffodils, rushes as
# vanilla's ground cover (low) or its low bush (tall); hawthorn, elder, holly
# and gorse as its bush; the fruiting brambles keep the bush's berries to forage.
_GC, _FO = "vegetation_groundcover_01_0", "vegetation_foliage_01_"
SHEETS["vegetation_uk_01"] = (8, 3, None, [_GC] * 4 + [_FO + "11"] * 7 + [_FO + "8", _FO + "11"] + [_GC] * 3
                              + [_FO + "8"] * 4 + [None] * 4)
# The street's last American pieces and the countryside's (asset_creator
# streetscape.py, 2026-10-14): the street lamp (vanilla's lamp, its arm N E S W,
# sodium-orange light); double yellows (vanilla's lines); field gates (its
# fence gate), stiles (its short wooden fence), kissing gates (nothing: walked
# through), telegraph pole (solid), its top and wires; and the kerbs, slot for
# slot vanilla's three kerb sheets (kerb_shapes.py: the same shapes).
SHEETS["street_lighting_uk_01"] = (8, 1, None, [f"lighting_outdoor_01_{i}" for i in (8, 9, 10, 11)] + [None] * 4)
SHEETS["road_markings_uk_01"] = (8, 1, None, ["street_trafficlines_01_24"] * 8)
_FG = "fixtures_doors_fences_01_"
SHEETS["countryside_uk_01"] = (8, 3, None, [_FG + str(i) for i in (4, 5, 6, 7)] * 2
                               + ["fencing_01_35", "fencing_01_32"] + [None] * 14)
SHEETS["street_curbs_uk_01"] = (8, 8, "street_curbs_01", list(range(64)))
SHEETS["street_curbs_uk_01_diag"] = (8, 10, "street_curbs_01_diag", list(range(80)))
SHEETS["street_curbs_uk_01_diag_2"] = (8, 10, "street_curbs_01_diag_2", list(range(80)))
# The road markings of one and two tiles (streetscape.py, marking_faces.py;
# 2026-10-14) as vanilla's lines; the town centre (townscape.py): bus stops
# (the signal post, as the signs), shelters (nothing: walked into; their
# glass is not a wall - PLAN E), street cabinets (solid), the litter bin (vanilla's public bin: a bin to
# search), grit bins (its grey bin), guard railing (W
# and N: the short wooden fence, hoppable; E and S only seen), cycle stands solidtrans; pubs' signs
# (the signal post).
_TC = "trashcontainers_01_"
SHEETS["road_markings_uk_02"] = (8, 14, None, ["street_trafficlines_01_24"] * 109 + [None] * 3)
SHEETS["street_furniture_uk_03"] = (8, 4, None, [_LO + "12"] * 4 + [None] * 8 + [_TC + "21"] + [_TC + "18"] * 4
                                    + [None] * 6 + ["fencing_01_35", "fencing_01_32", None, None] + [None] * 5)
SHEETS["pub_signs_uk_01"] = (8, 5, None, [_LO + "12"] * 34 + [None] * 6)
# Road gullies (streetscape.py, user 2026-10-14): square British grates, W N E S,
# as vanilla's drains (its two: W and N), for which register.py swaps them.
SHEETS["street_drains_uk_01"] = (8, 1, None, [_SD + "13", _SD + "14", _SD + "13", _SD + "14"] + [None] * 4)
DROP_PROPS = {"CanScrap", "Material"}
# An invisible collision piece must not turn visible or be carried off: no snow
# or burnt overlay, not movable (the picture beside it would be left behind).
INVISIBLE_DROP = {"SnowTile", "BurntTile", "IsMoveAble", "PickUpLevel", "PickUpTool", "PickUpWeight",
                  "PlaceTool", "MoveType", "CustomName", "GroupName", "Facing"}        # the concrete fence's scrap metal: not a brick wall's
# Street furniture: not carried off (a kiosk is no wall phone; turning one would
# swap in vanilla's sprites), no vanilla snow overlay (another shape), and the
# signal head's street light dropped (one amber light, and an on-sprite we have not).
STREET_DROP = {"SnowTile", "IsMoveAble", "PickUpLevel", "PickUpTool", "PickUpWeight", "PlaceTool", "MoveType",
               "HasLightOnSprite", "lightswitch", "streetlight", "lightR", "lightG", "lightB",
               "BurntTile", "ForceSingleItem",        # (a barrier burnt: vanilla's burnt wooden fence)
               # A car's knock swapped in vanilla's dented US mail box or cone (review, 2026-10-14);
               # the donor's scrap materials and a bin's trash flag are its, not ours.
               "DamagedSprite", "Material2", "Material3", "MaterialType", "IsTrashCan"}
# Properties a slot has that its vanilla tile lacks: a kiosk fills its square
# (vanilla's phone hangs on a wall), a signal stands on its own pole.
ADD_PROPS = {"street_furniture_uk_01": {**{k: [("solid", "")] for k in range(4, 12)},
                                        12: [("CustomName", "Belisha Beacon")], 13: [("CustomName", "Belisha Beacon")],
                                        **{k: [("solidtrans", "")] for k in range(16, 32)}},
             "street_furniture_uk_02": {12: [("StopCar", ""), ("CustomName", "Bollard")],      # iron, concrete: cars stop
                                        13: [("StopCar", ""), ("CustomName", "Bollard")],
                                        **{k: [("CustomName", "Keep Left Bollard")] for k in range(14, 18)},
                                        **{k: [("solid", ""), ("CustomName", "Police Box")] for k in (22, 23)}},
             "street_lighting_uk_01": {k: [("lightR", "255"), ("lightG", "150"), ("lightB", "50"),
                                           ("CustomName", "Street Lamp")] for k in range(4)},
             "countryside_uk_01": {**{k: [("CustomName", "Field Gate")] for k in range(8)},
                                   8: [("CustomName", "Stile")], 9: [("CustomName", "Stile")],
                                   10: [("CustomName", "Kissing Gate")], 11: [("CustomName", "Kissing Gate")],
                                   12: [("CustomName", "Telegraph Pole"), ("solidtrans", ""), ("BlocksPlacement", "")]},
             "vegetation_uk_01": {k: [("CustomName", n)] for k, n in enumerate(
                 ["Long Grass"] * 4 + ["Nettles"] * 2 + ["Cow Parsley"] * 2 + ["Bracken"] * 2 + ["Brambles"] * 2
                 + ["Foxgloves", "Bluebells", "Daffodils", "Rushes", "Hawthorn", "Elder", "Holly", "Gorse"])},
             "road_signs_uk_01": {k: [("solidtrans", "")] for k in range(48)},
             "street_furniture_uk_03": {**{k: [("solidtrans", ""), ("CustomName", "Bus Stop")] for k in range(4)},
                                        **{k: [("CustomName", "Bus Shelter")] for k in range(4, 12)},
                                        12: [("CustomName", "Litter Bin")],
                                        **{k: [("CustomName", "Grit Bin")] for k in range(13, 17)},
                                        **{k: [("solid", ""), ("CustomName", "Street Cabinet")] for k in range(17, 21)},
                                        21: [("solidtrans", ""), ("CustomName", "Cycle Stand")],
                                        22: [("solidtrans", ""), ("CustomName", "Cycle Stand")],
                                        **{k: [("CustomName", "Guard Railing")] for k in range(23, 27)}},
             "pub_signs_uk_01": {k: [("solidtrans", ""), ("CustomName", "Pub Sign")] for k in range(34)}}
# Properties of the vanilla tile a slot must not take: the barrier post's and
# old antenna's place in their own multi-tile objects (ours stand alone).
SLOT_DROP = {"street_furniture_uk_01": {k: {"SpriteGridPos", "GroupName", "CustomName", "Facing", "IsLow"}
                                        for k in (12, 13)},     # the beacons: not a piece of the barrier post's barrier
             "street_lighting_uk_01": {k: {"HasLightOnSprite", "SnowTile", "lightR", "lightG", "lightB"} for k in range(4)},
             "countryside_uk_01": {**{k: {"CustomName"} for k in range(8)},
                                   8: {"BurntTile", "SnowTile"}, 9: {"BurntTile", "SnowTile"}},
             "vegetation_uk_01": {k: {"CustomName"} | ({"container"} if k != 11 else set()) for k in range(20)},
             "street_furniture_uk_02": {**{k: {"SpriteGridPos", "Facing"} for k in range(4)},
                                        **{k: {"CustomName", "GroupName", "Facing", "MoveType"} for k in (22, 23)},
                                        **{k: {"SpriteGridPos", "Facing", "GroupName", "CustomName"} for k in range(12, 18)}},
             "street_furniture_uk_03": {k: {"Facing", "GroupName", "CustomName", "BurntTile", "SnowTile"} for k in range(12, 27)}}
STREET_SHEETS = {"street_furniture_uk_01", "street_furniture_uk_02", "road_signs_uk_01", "vegetation_uk_01",
                 "street_furniture_uk_03", "pub_signs_uk_01", "street_drains_uk_01"}

# Build 42: the shape behind each wall sprite (tileGeometry.txt; a square is
# -5000..5000 across, 10000 a metre up) and the vanilla sprite whose depth
# picture it borrows (tileDepthTextureAssignments.txt), as Erika's Tiles does.
_T, _LOW, _TALL = 2190, 9690, 18280          # walls.py: T 14 px, 11 and 22 courses with the coping
_BOX = {"west": [(-5000, 0, -5000, -5000 + _T, 5000)], "north": [(-5000, 0, -5000, 5000, -5000 + _T)],
        "post": [(-5000, 0, -5000, -5000 + _T, -5000 + _T)]}
_BOX["corner"] = _BOX["west"] + _BOX["north"]
GEOMETRY = {"walls_garden_uk_01": {base + i: [(x0, y0, z0, x1, h, z1) for x0, y0, z0, x1, z1 in _BOX[kind]]
                                   for base, h in ((0, _LOW), (8, _TALL))
                                   for i, kind in enumerate(("west", "west", "north", "north", "corner", "post"))}}
_DIAG_CUTS = [((0, 1), (1, 0)), ((0, 0), (1, 1)), ((0, 1), (1, .5)), ((0, .5), (1, 0)), ((0, 0), (1, .5)),
              ((0, .5), (1, 1)), ((0, 1), (.5, 0)), ((.5, 1), (1, 0)), ((0, 0), (.5, 1)), ((.5, 0), (1, 1)),
              ((.5, 0), (0, .5)), ((1, .5), (.5, 1)), ((.5, 0), (1, .5)), ((0, .5), (.5, 1))]
_DIAG_H = {"uk_brick_low": 62, "uk_brick_tall": 117, "hedge_low": 60, "hedge_tall": 108,
           "short_wooden": 66, "tall_wooden": 146, "black_metal": 70}      # px, as diag_walls.py
# The angled pictures: DIAG_BOXES upright boxes, each round its own stretch of
# the line plus the wall's thickness, so together they cover the whole picture
# (three small boxes left most of each picture without depth - gaps in game,
# 2026-10-12; a turned box would be one, but its sign is not checked yet).
DIAG_BOXES, _HALF_T = 8, 700


def _diag_boxes(a, b, h):
    out = []
    for k in range(DIAG_BOXES):
        f0, f1 = k / DIAG_BOXES, (k + 1) / DIAG_BOXES
        xs = [-5000 + 10000 * (a[0] + (b[0] - a[0]) * f) for f in (f0, f1)]
        zs = [-5000 + 10000 * (a[1] + (b[1] - a[1]) * f) for f in (f0, f1)]
        out.append((round(min(xs)) - _HALF_T, 0, round(min(zs)) - _HALF_T,
                    round(max(xs)) + _HALF_T, h, round(max(zs)) + _HALF_T))
    return out


GEOMETRY["boundaries_diag_uk_01"] = {k * 24 + c: _diag_boxes(a, b, _DIAG_H[st] * 156)
                                     for k, st in enumerate(DIAG_ORDER) for c, (a, b) in enumerate(_DIAG_CUTS)}
# (The garden walls borrowed vanilla walls' and fences' depth pictures until
# 2026-10-14: they covered 76-82% of a wall piece and 20-42% of a pier, and the
# game cut the rest - gaps. walls.py now writes their own DEPTH_ sheet.)
DEPTH: dict = {}


def _street_geometry(sheet) -> dict:
    """street_sheet.py's collision shapes (cylinders standing on the floor, boxes)."""
    path = sheet_file(sheet + ".geometry.json")
    if not os.path.exists(path):
        return {}
    shapes = json.load(open(path, encoding="utf-8"))["geometry"]
    return {int(k): [("cylinder", *g["cylinder"]) if "cylinder" in g else tuple(g["box"]) for g in v]
            for k, v in shapes.items() if v}


for _sheet in ("street_furniture_uk_01", "street_furniture_uk_02", "road_signs_uk_01", "doors_uk_01",
               "vegetation_uk_01", "street_lighting_uk_01", "countryside_uk_01", "street_furniture_uk_03",
               "pub_signs_uk_01"):
    GEOMETRY[_sheet] = _street_geometry(_sheet)
# The listed buildings' kits (asset_creator/listed_building_maker/makers:
# wall_kit.py, roof_kit.py, window_kit.py; 2026-10-18), in this order for
# good (a sheet's tileset id is its place). Each sheet's <sheet>.slots.json
# names, per slot, the vanilla piece it stands in for (its outline was cut to
# that piece's): the slot takes that piece's properties and its depth picture
# (vanilla's own tileDepthTextureAssignments). A window slot is a dict: its
# vanilla window and its own WindowShape.
WALL_KITS = ["brick_flemish", "brick_painted", "brick_chequer", "render_white", "render_cream", "timber_white",
             "timber_brick", "flint", "tile_hung", "weatherboard_black", "weatherboard_white"]
ROOF_KITS = ["tile", "slate", "thatch"]
KIT_SHEETS = ([f"walls_house_uk_01_{m}" for m in WALL_KITS] + [f"roofs_uk_01_{c}" for c in ROOF_KITS]
              + [f"roofs_30_uk_01_{c}" for c in ROOF_KITS] + ["fixtures_windows_uk_01", "doorways_uk_01",
                                                                    "listed_structures_uk_01", "wall_trim_uk_01",
                                                                    "drains_uk_01"]
              # landmarks (listed_building_maker/recipes, recipe_kit.py; 2026-10-19): the church's
              # own wall kit, then the recipe sheets in recipes/sheets.order.json's order
              + ["walls_house_uk_01_flint_church"] + json.load(open(os.path.join(
                  os.path.dirname(os.path.abspath(__file__)), "..", "..", "listed_building_maker", "recipes",
                  "sheets.order.json"), encoding="utf-8"))
              # housing style sets by sub-type (asset_creator/housing_kit; 2026-10-19), in
              # housing_kit/sheets.order.json's order (after the recipes: their ids move when a
              # recipe sheet is added, so the pack and the maps are rebuilt together)
              + json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                            "housing_kit", "sheets.order.json"), encoding="utf-8"))
              # the commercial kit (asset_creator/commercial_kit; 2026-10-20): shop fronts and
              # fascias, in commercial_kit/sheets.order.json's order, after the housing sets
              + json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                            "commercial_kit", "sheets.order.json"), encoding="utf-8")))
# Westminster's own sheets (Parliament, Westminster Bridge: listed_building_maker/recipes/
# westminster.order.json; 2026-10-25) are a pack of their own, KnoxMapUK_Westminster, numbered
# in that file's order: a sheet added for Westminster moves no id of Odiham's or Alton's.
WESTMINSTER_SHEETS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                 "listed_building_maker", "recipes", "westminster.order.json"),
                                    encoding="utf-8"))
# Not carried over: wall paint (it swaps in vanilla's painted sprites), and a
# window's pick-up (the item would place vanilla's window) and snow overlay
# (vanilla's window's shape).
KIT_DROP = {"IsPaintable", "IsMoveAble", "PickUpLevel", "PickUpTool", "PickUpWeight", "PlaceTool", "MoveType"}
WINDOW_DROP = KIT_DROP | {"SnowTile", "WindowShape", "GroupName", "CustomName"}


def _vanilla_depth() -> dict:
    path = os.path.join(game_dir(), "media", "tileDepthTextureAssignments.txt")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        k, sep, v = line.strip().rstrip(",").partition(" = ")
        if sep:
            out[k] = v
    return out


def _add_kits() -> None:
    # No built sheets at all (the KnoxMap UK fork carries asset_creator's code, not its
    # art): nothing is packed here, and register.py's rules need only the names above.
    import glob
    if not glob.glob(os.path.join(BUILD, "sheets", "*", "*.slots.json")):     # (sheet_paths makes the folders)
        return
    vdepth = _vanilla_depth()
    for sheet in KIT_SHEETS + WESTMINSTER_SHEETS:
        path = sheet_file(sheet + ".slots.json")
        if not os.path.exists(path):
            # A sheet's tileset id is its place in this order: one left out would
            # shift every sheet after it (and the maps that name them). Make it first.
            raise SystemExit(f"pack.py: {sheet} has not been drawn yet ({path} is missing): run its maker "
                             f"(listed_building_maker/makers, recipe_kit.py for a recipe sheet), then pack again")
        slots = json.load(open(path, encoding="utf-8"))
        names, drop, add = [], {}, {}
        for k, s in enumerate(slots):
            if isinstance(s, dict) and s.get("like") and "WindowShape" not in s:
                # a recipe piece (recipe_kit.py): a vanilla tile's properties, ours added, our own depth
                names.append(s["like"])
                # its own properties replace the like-tile's of the same name (a chest's CustomName);
                # never the like-tile's place in a multi-tile object: our piece is its own sprite on
                # each square, and a one-square statue liking half a gallery statue, or a bench turned
                # east liking a pew laid west-east, was half an object to the game (multitile_check:
                # 392 on Parliament, 2026-10-22; the game hangs on those, LOG 2026-10-18)
                drop[k] = KIT_DROP | {"SpriteGridPos"} | set(s.get("drop", ())) | {pv[0] for pv in s.get("props", ())}
                add[k] = [tuple(pv) for pv in s.get("props", ())]
                continue
            if isinstance(s, dict) and "props" in s:      # a piece of its own: only these properties
                names.append(None)
                add[k] = [tuple(pv) for pv in s["props"]]
                continue
            if isinstance(s, dict):
                names.append(s["like"])
                drop[k] = WINDOW_DROP
                add[k] = [("WindowShape", str(s["WindowShape"])), ("GroupName", "UK Window"),
                          ("CustomName", "Window")]
            else:
                names.append(s)
                drop[k] = KIT_DROP
            if names[-1]:
                DEPTH[f"{sheet}_{k}"] = vdepth.get(names[-1], names[-1])
        SHEETS[sheet] = (8, -(-len(slots) // 8), None, names)
        SLOT_DROP[sheet] = drop
        ADD_PROPS[sheet] = add


# Rules.txt and Blends.txt: the vanilla floor sets ours replace (register.py):
# (our sheet, our first slot, vanilla sheet, its first slot, how many).
FLOOR_REPLACES = [
    ("blends_street_uk_01", 0, "blends_street_01", 96, 12),      # pavement (street)
    ("blends_street_uk_02", 0, "blends_street_01", 80, 12),      # carriageway (street2)
    ("blends_street_uk_03", 0, "blends_street_01", 32, 12),      # worn tarmac (street4)
    ("blends_natural_uk_01", 0, "blends_natural_01", 16, 16),    # dark grass
    ("blends_natural_uk_01", 16, "blends_natural_01", 32, 16),   # medium grass
    ("blends_natural_uk_01", 32, "blends_natural_01", 48, 16),   # light grass
    ("blends_natural_uk_02", 0, "blends_natural_01", 64, 12),    # dirt (footpaths, tracks: dirt.py)
]


def game_dir() -> str:
    cfg = json.load(open(os.path.join(KNOXMAP, "knoxmap_config.json"), encoding="utf-8"))
    return cfg["pz_install"]


_add_kits()

# The packs: id -> (name, tiledef, description, its sheets in tileset-id order). A sheet's
# tileset id is its place in its own pack's list.
PACKS = {
    PACK: (NAME, TILEDEF, "English street surfaces, garden walls, front doors, foliage and 1990s street furniture "
                          "and signs for KnoxMap UK maps (Odiham). Own photos, CC BY 4.0.",
           [s for s in SHEETS if s not in WESTMINSTER_SHEETS]),
    "KnoxMapUK_Westminster": ("KnoxMap UK Westminster Tiles", 7345,
                              "The Palace of Westminster and Westminster Bridge for the KnoxMap UK Westminster map: "
                              "Gothic Revival walls, windows, roofs, towers and the palace's interiors.",
                              WESTMINSTER_SHEETS),
}
SHEET_PACK = {s: p for p, (_n, _t, _d, sheets) in PACKS.items() for s in sheets}


def read_tdef(path: str) -> dict[str, list[list[tuple[str, str]]]]:
    """Every tileset's per-tile properties in a .tiles file."""
    b, o = open(path, "rb").read(), 4
    def i32():
        nonlocal o
        o += 4
        return struct.unpack_from("<i", b, o - 4)[0]
    def s():
        nonlocal o
        e = b.index(b"\n", o)
        v, o = b[o:e].decode("latin-1"), e + 1
        return v
    i32()
    out = {}
    for _ in range(i32()):
        name, _img, _c, _r, _id, n = s(), s(), i32(), i32(), i32(), i32()
        out[name] = [[(s(), s()) for _ in range(i32())] for _ in range(n)]
    return out


def write_tdef(path: str, tilesets: list[tuple[str, int, int, int, list]]) -> None:
    out = io.BytesIO()
    put = lambda v: out.write(struct.pack("<i", v))
    say = lambda t: out.write(t.encode("latin-1") + b"\n")
    out.write(b"tdef")
    put(1)
    put(len(tilesets))
    txt = ["version = 1"]
    for name, cols, rows, tid, props in tilesets:
        say(name); say(name + ".png"); put(cols); put(rows); put(tid); put(len(props))
        txt += ["tileset", "{", f"    file = {name}", f"    size = {cols},{rows}", f"    id = {tid}"]
        for i, p in enumerate(props):
            put(len(p))
            for k, v in p:
                say(k); say(v)
            txt += [f"    // {name}_{i}", "    tile", "    {", f"        xy = {i % cols},{i // cols}"]
            txt += [f"        {k} = {v}" for k, v in p] + ["    }"]
        txt.append("}")
    open(path, "wb").write(out.getvalue())
    open(path + ".txt", "w", encoding="utf-8", newline="\n").write("\n".join(txt) + "\n")


def write_pack(path: str, sheets: dict[str, str], pack: str = PACK) -> int:
    """One page per sheet; returns the number of sprites."""
    out = io.BytesIO()
    put = lambda v: out.write(struct.pack("<i", v))
    def string(t):
        put(len(t)); out.write(t.encode("latin-1"))
    out.write(b"PZPK")
    put(1)
    put(len(sheets))
    sprites = 0
    for k, (name, png) in enumerate(sheets.items()):
        im = Image.open(png).convert("RGBA")
        a = np.asarray(im)
        cols = im.width // 128
        entries = []
        for r in range(im.height // 256):
            for c in range(cols):
                alpha = a[r * 256:(r + 1) * 256, c * 128:(c + 1) * 128, 3]
                ys, xs = np.nonzero(alpha)
                if len(xs) == 0:
                    continue
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                entries.append((f"{name}_{r * cols + c}", c * 128 + x0, r * 256 + y0,
                                x1 - x0, y1 - y0, x0, y0, 128, 256))
        string(f"{pack}{k}")
        put(len(entries))
        put(1)                                   # mask, as Erika's
        for e in entries:
            string(e[0])
            out.write(struct.pack("<8i", *map(int, e[1:])))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        put(len(buf.getvalue()))
        out.write(buf.getvalue())
        sprites += len(entries)
    open(path, "wb").write(out.getvalue())
    return sprites


def write_geometry(media: str, sheets) -> None:
    """tileGeometry.txt and tileDepthTextureAssignments.txt, in Erika's layout (these sheets')."""
    t = "\t"
    g = ["tileGeometry", "{", t + "VERSION = 2,", ""]
    for sheet, tiles in GEOMETRY.items():
        if sheet not in sheets:
            continue
        cols = SHEETS[sheet][0]
        g += [t + "tileset", t + "{", 2 * t + f"name = {sheet},", ""]
        for i, boxes in tiles.items():
            g += [2 * t + f"/* {sheet}_{i} */", 2 * t + "tile", 2 * t + "{", 3 * t + f"xy = {i % cols}x{i // cols},", ""]
            for shape in boxes:
                if shape[0] == "cylinder":             # (x, bottom, z, radius, height): vanilla's, stood up
                    _, x, y0, z, r, h = shape
                    g += [3 * t + "cylinder", 3 * t + "{", 4 * t + f"translate = {x}x{y0 + h // 2}x{z},",
                          4 * t + "rotate = 2700000x0x0,", 4 * t + f"radius1 = {r},", 4 * t + f"radius2 = {r},",
                          4 * t + f"height = {h},", 3 * t + "}"]
                    continue
                x0, y0, z0, x1, y1, z1 = shape
                g += [3 * t + "box", 3 * t + "{", 4 * t + "translate = 0x0x0,", 4 * t + "rotate = 0x0x0,",
                      4 * t + f"min = {x0}x{y0}x{z0},", 4 * t + f"max = {x1}x{y1}x{z1},", 3 * t + "}"]
            g += [2 * t + "}", ""]
        g += [t + "}"]
    g += ["}"]
    d = ["tileDepthTextureAssignments", "{", t + "VERSION = 1,"] + [t + f"{k} = {v}," for k, v in DEPTH.items()
                                                                 if k.rsplit("_", 1)[0] in sheets] + ["}"]
    for name, lines in (("tileGeometry.txt", g), ("tileDepthTextureAssignments.txt", d)):
        open(os.path.join(media, name), "w", encoding="utf-8", newline="\r\n").write("\n".join(lines) + "\n")


def build(pack: str = PACK) -> str:
    title, tiledef, desc, sheets = PACKS[pack]
    vanilla = read_tdef(os.path.join(game_dir(), "media", "newtiledefinitions.tiles"))
    root = os.path.join(BUILD, "mod", pack)
    shutil.rmtree(root, ignore_errors=True)
    media = os.path.join(root, "common", "media")
    os.makedirs(os.path.join(media, "texturepacks"))
    os.makedirs(os.path.join(root, "42"))
    tilesets, pngs = [], {}
    for tid, name in enumerate(sheets, 1):
        cols, rows, like, slots = SHEETS[name]
        props, ours = [], {}
        for k, i in enumerate(slots):
            if i is None:                         # no vanilla tile: only what we give it
                props.append(sorted(ADD_PROPS.get(name, {}).get(k, [])))
                continue
            sheet, idx = (like, i) if isinstance(i, int) else (i.rsplit("_", 1)[0], int(i.rsplit("_", 1)[1]))
            ours.setdefault(f"{sheet}_{idx}", f"{name}_{k}")
            drop = (DROP_PROPS | INVISIBLE_DROP if name == "boundaries_diag_uk_01" else
                    DROP_PROPS | STREET_DROP if name in STREET_SHEETS else DROP_PROPS)
            drop = drop | SLOT_DROP.get(name, {}).get(k, set())
            props.append(sorted([(p, v) for p, v in vanilla[sheet][idx] if p not in drop or isinstance(i, int)]
                                + ADD_PROPS.get(name, {}).get(k, [])))
        # A property naming a vanilla tile this sheet stands in for names ours
        # (a corner's CornerNorthWall / CornerWestWall).
        props = [[(p, ours.get(v, v)) for p, v in pr] for pr in props]
        tilesets.append((name, cols, rows, tid, props))
        pngs[name] = sheet_file(name + ".png")
    write_tdef(os.path.join(media, pack + ".tiles"), tilesets)
    write_geometry(media, sheets)
    # Build 42 draws a sprite with the depth picture of its sheet (Erika's
    # layout: loose PNGs); a sheet without one is guessed and cut in game.
    os.makedirs(os.path.join(media, "depthmaps"))
    for name in sheets:
        src = sheet_file(f"DEPTH_{name}.png")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(media, "depthmaps", f"DEPTH_{name}.png"))
    n = write_pack(os.path.join(media, "texturepacks", pack + ".pack"), pngs, pack)
    info = (f"name={title}\nid={pack}\n"
            f"description={desc}\n"
            f"versionMin=42.0.0\npack={pack}\ntiledef={pack} {tiledef}\n")
    for where in ("", "common", "42"):
        open(os.path.join(root, where, "mod.info"), "w", encoding="utf-8", newline="\n").write(info)
    print(f"wrote {root}: {len(tilesets)} sheets, {n} sprites")
    return root


def install(root: str) -> None:
    cfg = json.load(open(os.path.join(KNOXMAP, "knoxmap_config.json"), encoding="utf-8"))
    dest = os.path.join(cfg["zomboid_dir"], "mods", os.path.basename(root))
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(root, dest)
    print("installed", dest)


if __name__ == "__main__":
    for p in PACKS:
        r = build(p)
        if "--install" in sys.argv:
            install(r)
