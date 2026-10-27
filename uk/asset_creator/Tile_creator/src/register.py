"""The UK sheets into the map tools, in place of the vanilla surfaces they replace.

    python register.py [--undo]
    python register.py --vanilla-to <dir>
    python register.py --rules-only

--rules-only puts our tiles into Rules.txt and Blends.txt only (no sheet PNGs into
the tools, no Tilesets.txt entries): where the pack is installed as a mod but its
sheets are not here (the KnoxMap UK fork, after the app's "Download and install").
The Rust compiler needs only the names; WorldEd needs the sheets too.

--vanilla-to writes Rules.txt and Blends.txt as --undo would make them into <dir>
and leaves the map tools as they are: the rules a map with UK assets off is
compiled by (KnoxMap UK's uk_pipeline.py, Knoxmap/tools/terrain_source.py).

For each sheet in pack.SHEETS: its PNG into the map tools' Tiles/2x and a
tileset entry in Tilesets.txt (file = KnoxMapUK_Tiles.pack/<sheet>, as setup
does for Erika's Tiles); then every name of a vanilla floor tile ours replace
(SWAP: pavement, carriageway, worn tarmac) swapped for ours in Rules.txt and
Blends.txt. The overlay sheets are not in the rules: kerbs.py lays them.
The street lamps swapped for ours (LAMP_SWAP), and the storm drains (DRAIN_SWAP). And our foliage mixed into the rules' plant lists (ALIAS_ADDS: Rules.txt
aliases the vegetation map's grass, plants, ferns and bushes pick from at
random), summer's only: bluebells, daffodils and rushes are made but not
laid (the game opens in July; no rule marks wet ground). Every map compiled after
this takes the new surface (clear a map's lots/ and compile_state.json: the
incremental compile does not see a change of rules). Each edit swaps names
only (vanilla's for ours, or back with --undo), so it can run again and keeps
what other scripts changed in the same files (uk_road_rules.py,
uk_tree_rules.py); Rules.txt, Blends.txt and Tilesets.txt are backed up once
as *.pre-uk-tiles.
"""
from __future__ import annotations

import os
import re
import shutil
import sys

import pack
from sheet_paths import sheet_file

TOOLS = os.path.join(pack.KNOXMAP, "vendor", "PZMappingTools")
CONFIG = os.path.join(TOOLS, "config")
FILES = ("Rules.txt", "Blends.txt", "Tilesets.txt")
# Vanilla tile -> ours, wherever Rules.txt (its aliases: street, street2,
# street2blends, street4, street4blends, the grasses and their _n/_w/_e/_s
# blends) and Blends.txt name it: the pavement (colour 120), carriageways and
# car parks (150, 100), service lanes (80), dark, medium and light grass.
SWAP = {f"{vsheet}_{vbase + i}": f"{ours}_{start + i}"
        for ours, start, vsheet, vbase, n in pack.FLOOR_REPLACES for i in range(n)}
# Rules.txt alias -> our foliage (vegetation_uk_01 slots) added to its list:
# each alias picks at random among its tiles, so these are a share of it.
VEG = "vegetation_uk_01_"
ALIAS_ADDS = {
    "darklonggrass": [0, 1], "medlonggrass": [0, 2], "lightlonggrass": [1, 3],     # meadow grass, buttercups, daisies
    "plantsmedium": [4, 5, 6, 7, 10, 11, 12],                                     # nettles, cow parsley, brambles, foxgloves
    "genericfern": [8, 9, 8, 9],                                                  # bracken
    "bushes": [16, 17, 18, 19],                                                   # hawthorn, elder, holly, gorse
}
# ...and rules (by label) whose own lists take ours too: the plants and fern
# lists above are in no rule our maps' vegetation uses (2026-10-14: none of
# their colours on Odiham or Alton), so the hedgerow and woodland-floor
# plants go under the woods' edge (uk_tree_rules.py's wood with grass).
RULE_ADDS = {
    "UK wood + dark grass": [8, 11, 4, 12],                                       # bracken, brambles in fruit, nettles, foxgloves
}


def alias_adds(text: str, back: bool) -> tuple[str, int]:
    """Our foliage into (or, back, out of) each ALIAS_ADDS alias's tile list; run again, the same."""
    n = 0
    for alias, slots in ALIAS_ADDS.items():
        m = re.search(r"(name = " + re.escape(alias) + r"\s*\n\s*tiles = \[\n)(.*?)(\n\s*\])", text, re.S)
        if not m:
            print(f"Rules.txt: no alias {alias}")
            continue
        lines = [l for l in m.group(2).split("\n") if VEG not in l]
        if not back:
            lines += [f"\t{VEG}{k}" for k in slots]
            n += len(slots)
        text = text[:m.start(2)] + "\n".join(lines) + text[m.end(2):]
    for label, slots in RULE_ADDS.items():
        head = re.search(r"label = " + re.escape(label) + r"\s*\n.*?tiles = \[\n", text, re.S)
        tail = re.compile(r"\n\s*\]").search(text, head.end()) if head else None
        if not tail:
            print(f"Rules.txt: no rule {label}")
            continue
        lines = [l for l in text[head.end():tail.start()].split("\n") if VEG not in l]
        if not back:
            lines += [f"        {VEG}{k}" for k in slots]
            n += len(slots)
        text = text[:head.end()] + "\n".join(lines) + text[tail.start():]
    return text, n


# The street lamps the rules lay (KnoxMap's patch_rules_roads.py: the arm
# reaching N, E, S, W) -> ours, the 1990s concrete column and sodium lantern.
LAMP_SWAP = {f"lighting_outdoor_01_{8 + k}": f"street_lighting_uk_01_{k}" for k in range(4)}
LAMP_NAME = re.compile(r"\b(lighting_outdoor_01_(?:8|9|10|11)|street_lighting_uk_01_[0-3])\b")
# ...and the storm drains: vanilla's US kerb inlets for our square gully grates (W, N).
DRAIN_SWAP = {"street_decoration_01_13": "street_drains_uk_01_0", "street_decoration_01_14": "street_drains_uk_01_1"}
DRAIN_NAME = re.compile(r"\b(street_decoration_01_1[34]|street_drains_uk_01_[01])\b")


TILE_NAME = re.compile(r"\b(blends_(?:street|natural)_(?:uk_)?0\d_\d+)\b")


def _read(path: str) -> tuple[str, bool]:
    raw = open(path, "rb").read()
    return raw.decode("utf-8").replace("\r\n", "\n"), b"\r\n" in raw[:4096]


def _write(path: str, text: str, crlf: bool) -> None:
    with open(path, "w", encoding="utf-8", newline="\r\n" if crlf else "\n") as f:
        f.write(text)


def main() -> int:
    back = "--undo" in sys.argv
    if "--vanilla-to" in sys.argv:
        return vanilla_to(sys.argv[sys.argv.index("--vanilla-to") + 1])
    if "--rules-only" in sys.argv:
        for f in ("Rules.txt", "Blends.txt"):
            b = os.path.join(CONFIG, f + ".pre-uk-tiles")
            if not os.path.exists(b):
                shutil.copy(os.path.join(CONFIG, f), b)
        swap_rules(CONFIG, CONFIG, back)
        print("UK tiles in the rules (no sheets)" if not back else "rules back to the vanilla surfaces")
        return 0
    for f in FILES:
        b = os.path.join(CONFIG, f + ".pre-uk-tiles")
        if not os.path.exists(b):
            shutil.copy(os.path.join(CONFIG, f), b)

    path = os.path.join(CONFIG, "Tilesets.txt")
    tilesets, crlf = _read(path)
    for name, (cols, rows, _like, _slots) in pack.SHEETS.items():
        entry = f"tileset\n{{\n    file = {pack.SHEET_PACK[name]}.pack/{name}\n    size = {cols},{rows}\n}}\n"
        # Any entry for this sheet, whatever size it says or pack it names: a sheet that
        # grew kept a stale size, and the tools cut its tiles wrong (2026-10-12); Westminster's
        # sheets moved to their own pack (2026-10-25).
        old = re.compile(r"tileset\n\{\n    file = (?:" + "|".join(map(re.escape, pack.PACKS)) + r")\.pack/"
                         + re.escape(name) + r"\n    size = \d+,\d+\n\}\n")
        tilesets = old.sub("", tilesets)
        if back:
            continue
        shutil.copy(sheet_file(name + ".png"),
                    os.path.join(TOOLS, "Tiles", "2x", name + ".png"))
        tilesets = tilesets.rstrip("\n") + "\n" + entry
    _write(path, tilesets, crlf)

    swap_rules(CONFIG, CONFIG, back)
    print("map tools back to the vanilla surfaces" if back else "UK sheets registered")
    return 0


def vanilla_to(out_dir: str) -> int:
    """The map tools' Rules.txt and Blends.txt with the vanilla tiles back, into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    swap_rules(CONFIG, out_dir, True)
    print(f"vanilla rules -> {out_dir}")
    return 0


def swap_rules(src_dir: str, out_dir: str, back: bool) -> None:
    """Rules.txt and Blends.txt from src_dir, our tiles in (or, back, out), into out_dir."""
    swap = {v: k for k, v in SWAP.items()} if back else SWAP
    for f in ("Rules.txt", "Blends.txt"):
        path = os.path.join(src_dir, f)
        text, crlf = _read(path)
        n = 0
        def one(t):
            nonlocal n
            n += t.group(1) in swap
            return swap.get(t.group(1), t.group(1))
        text = TILE_NAME.sub(one, text)
        print(f"{f}: {n} tile names swapped")
        if f == "Rules.txt":
            lamps = {v: k for k, v in LAMP_SWAP.items()} if back else LAMP_SWAP
            text, k = LAMP_NAME.subn(lambda t: lamps.get(t.group(1), t.group(1)), text)
            print(f"Rules.txt: {k} street lamps {'back to vanilla' if back else 'ours'}")
            drains = {v: k for k, v in DRAIN_SWAP.items()} if back else DRAIN_SWAP
            text, k = DRAIN_NAME.subn(lambda t: drains.get(t.group(1), t.group(1)), text)
            print(f"Rules.txt: {k} storm drains {'back to vanilla' if back else 'ours'}")
            text, k = alias_adds(text, back)
            print(f"Rules.txt: {k} foliage tiles in the plant lists")
        _write(os.path.join(out_dir, f), text, crlf)


if __name__ == "__main__":
    sys.exit(main())
