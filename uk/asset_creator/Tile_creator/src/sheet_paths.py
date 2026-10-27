"""Where a tile sheet's files live: Tile_creator/build/sheets/<group>/.

    python sheet_paths.py        sort a flat build/sheets/ into its groups (once), write its README

Every sheet's files sit together in the folder of its group: the sheet
(<sheet>.png), its depth picture (DEPTH_<sheet>.png), its slots' vanilla
pieces and properties (<sheet>.slots.json) and, for the sheets whose makers write
one, its sprites' geometry (<sheet>.geometry.json). The group is the sheet's name read against GROUPS,
first match: a new sheet named like its kind lands with them, and every
script that writes or reads a sheet asks `sheet_file` for the path.
"""
from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SHEETS = os.path.normpath(os.path.join(HERE, "..", "build", "sheets"))

# (folder, what is in it, the sheet-name prefixes that go there), first match wins.
GROUPS = [
    ("ground", "floors and what is painted on them: pavement, tarmac, grass, dirt, kerbs, road markings, drains",
     ("blends_", "street_curbs_", "road_markings_", "street_drains_")),
    ("street", "street furniture: signs, lamps, bins, benches, post boxes, pub signs",
     ("road_signs_", "street_furniture_", "street_lighting_", "pub_signs_")),
    ("nature", "plants: hedges, bushes, countryside", ("vegetation_", "countryside_")),
    ("walls_garden", "garden and boundary walls", ("walls_garden_", "boundaries_")),
    ("walls_house", "house wall kits, one sheet a material (listed_building_maker/makers/wall_kit.py)",
     ("walls_house_",)),
    ("roofs", "roof kits, 45 and 30 degrees (makers/roof_kit.py)", ("roofs_",)),
    ("building_parts", "doors, windows, doorways, trims, drains, listed structures (listed_building_maker/makers)",
     ("doors_", "fixtures_windows_", "doorways_", "wall_trim_", "drains_", "listed_structures_", "bays_")),
    ("commercial", "shop fronts, fascias and the commercial kit's pieces (asset_creator/commercial_kit)",
     ("shopfronts_", "fascias_", "commercial_", "walls_commercial_uk", "storefront_")),
    ("landmarks", "a landmark's own tiles, drawn from recipes (listed_building_maker/recipes, makers/recipe_kit.py)",
     ("church_", "inn_", "castle_", "manor_", "chapel_")),
]
OTHER = "other"
SUFFIXES = (".slots.json", ".geometry.json", ".json", ".png")


def sheet_name(fname: str) -> str:
    """The sheet a file belongs to: DEPTH_x.png, x.slots.json ... -> x."""
    name = os.path.basename(fname)
    if name.startswith("DEPTH_"):
        name = name[6:]
    for suf in SUFFIXES:
        if name.endswith(suf):
            return name[:-len(suf)]
    return name


def group_of(sheet: str) -> str:
    return next((folder for folder, _what, prefixes in GROUPS if sheet.startswith(prefixes)), OTHER)


def sheet_file(fname: str) -> str:
    """The path of one of a sheet's files ("church_windows_uk_01.png", "DEPTH_x.png",
    "x.slots.json"), its group's folder made if need be."""
    folder = os.path.join(SHEETS, group_of(sheet_name(fname)))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, os.path.basename(fname))


def readme() -> str:
    lines = ["# Tile sheets", "",
             "Written by the makers, read by `pack.py` and `register.py`. Each sheet's files are",
             "together: `<sheet>.png`, `DEPTH_<sheet>.png`, `<sheet>.slots.json` and, where its maker",
             "writes one, `<sheet>.geometry.json`.",
             "The folder is chosen by the sheet's name (`Tile_creator/src/sheet_paths.py`, GROUPS).", "",
             "| Folder | What | Sheets named |", "|---|---|---|"]
    for folder, what, prefixes in GROUPS:
        lines.append(f"| `{folder}/` | {what} | {', '.join(f'`{p}*`' for p in prefixes)} |")
    lines.append(f"| `{OTHER}/` | anything not named as above | |")
    return "\n".join(lines) + "\n"


def main() -> None:
    moved = 0
    for fname in sorted(os.listdir(SHEETS)):
        src = os.path.join(SHEETS, fname)
        if os.path.isfile(src) and fname != "README.md":
            shutil.move(src, sheet_file(fname))
            moved += 1
    with open(os.path.join(SHEETS, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme())
    counts = {g: len(os.listdir(os.path.join(SHEETS, g))) for g in sorted(os.listdir(SHEETS))
              if os.path.isdir(os.path.join(SHEETS, g))}
    print(f"moved {moved} files; files per folder: {counts}")


if __name__ == "__main__":
    main()
