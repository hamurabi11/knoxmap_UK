"""British trees for the map tools: one vegetation colour per species.

    python uk_tree_rules.py [<Rules.txt>]

The tools turn each pixel of a map's vegetation bitmap into tiles by the
rules in config/Rules.txt; the vanilla tree colour (255 0 0) picks at random
from four American species (red maple, river birch, American linden,
dogwood). Build 42 ships no British tree, so these rules use the nearest
look-alikes, one colour each, so trees.py can choose the species per tree:

    lime      American linden      same genus (Tilia)
    maple     red maple            sycamore, field maple
    birch     river birch          silver birch
    hawthorn  cockspur hawthorn    same genus (Crataegus), white May blossom
    holly     American holly       same genus (Ilex)
    conifer   Canadian hemlock     yew, Leyland cypress in gardens
    wood      a mix of the above   OSM woods, in place of the vanilla mix
    wood+grass  the mix with dark grass, in place of 127 0 0
    ... jumbo, jumboxl, jumboxxl   lime, maple, birch, hawthorn at 3, 5 and
              7 tiles across (SIZES), for trees as big as they really are

Tiles 10 and 11 of each broadleaf sheet: the grown trees of its green row (8-11 grow
from a sapling; 12-15 are the same in summer, or under snow for the
conifers; 0-7 bare, 16-23 autumn); 2 and 3 of holly and hemlock (one row of
growth, 0-3, then the same under snow, 4-7). Dogwood, redbud, silverbell, yellowwood
and Virginia pine are left out: nothing British looks like them. There is no
oak, ash or beech; lime and maple stand in for the big trees.

The rules are appended, their labels starting "UK " (the file takes no
comments: a marker line broke every compile), and replaced on a rerun; the
file is backed up first as Rules.txt.pre-uk-trees.
"""
from __future__ import annotations

import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
RULES = os.path.join(ukpaths.KNOX, "vendor", "PZMappingTools", "config", "Rules.txt")
OURS = re.compile(r"\n*rule\s*\{\s*label = UK [^\n]*\n.*?\n\}[ \t]*(?=\n|$)", re.S)
MARKER = re.compile(r"\n*# --- (end )?KnoxMap UK trees[^\n]*")

# Colours, as trees.py paints them on the vegetation bitmap.
SPECIES = {
    "lime": ((255, 1, 0), ["e_americanlinden_1_10", "e_americanlinden_1_11", "e_americanlinden_1_11"]),
    "maple": ((255, 2, 0), ["e_redmaple_1_10", "e_redmaple_1_11", "e_redmaple_1_11"]),
    "birch": ((255, 3, 0), ["e_riverbirch_1_10", "e_riverbirch_1_11"]),
    "hawthorn": ((255, 4, 0), ["e_cockspurhawthorn_1_9", "e_cockspurhawthorn_1_10", "e_cockspurhawthorn_1_11"]),
    "holly": ((255, 5, 0), ["e_americanholly_1_2", "e_americanholly_1_3"]),
    "conifer": ((255, 6, 0), ["e_canadianhemlock_1_2", "e_canadianhemlock_1_3"]),
    "wood": ((255, 7, 0), ["e_americanlinden_1_11", "e_redmaple_1_11", "e_riverbirch_1_11",
                           "e_redmaple_1_10", "e_cockspurhawthorn_1_10", "e_americanholly_1_3"]),
}
# Big trees: Build 42's jumbo sheets, one sprite on one tile but wider - a
# normal tree is one tile across, where a real crown is 6-12 m (the map is
# 1 m a tile). JUMBO is about 3 tiles across (8x2 sheet: 4-7 in leaf, the
# hawthorn's 4-5 in May blossom; 0-3 bare, 8-11 autumn), JUMBOXL about 5 and
# JUMBOXXL about 7 (8x3 sheets: 3 and 9 in leaf, 2 and 8 the spring - white
# on the hawthorn; 0-1, 6-7 bare, 4-5 and 10-11 autumn, 16-23 stumps). Only
# the green ones: May blossom read as white specks at town scale.
_LEAF = {"JUMBO": (5, 7), "JUMBOXL": (3, 9), "JUMBOXXL": (3, 9)}
_SHEET = {"lime": "americanlinden", "maple": "redmaple", "birch": "riverbirch", "hawthorn": "cockspurhawthorn"}
SIZES = {   # size -> species -> (colour, tiles); also the tiles across, as trees.py places them
    size: {sp: ((255, base + k, 0), [f"e_{sheet}{size}_1_{i}" for i in (_LEAF[size] if sp != "hawthorn" or size != "JUMBO" else (6, 7))])
           for k, (sp, sheet) in enumerate(_SHEET.items(), 1)}
    for size, base in (("JUMBO", 10), ("JUMBOXL", 20), ("JUMBOXXL", 30))
}
BIG = SIZES["JUMBO"]
WOOD_GRASS = ((127, 7, 0), ["e_americanlinden_1_11", "e_redmaple_1_11", "e_riverbirch_1_10",
                            "e_cockspurhawthorn_1_10", "darklonggrass", "darklonggrass",
                            "darkmedgrass", "darkshortgrass"])


def rule(label, colour, tiles):
    body = "\n".join(f"        {t}" for t in tiles)
    return (f"rule\n{{\n    label = {label}\n    bitmap = 1\n    color = {colour[0]} {colour[1]} {colour[2]}\n"
            f"    tiles = [\n{body}\n    ]\n    layer = 0_Vegetation\n}}\n")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else RULES
    crlf = b"\r\n" in open(path, "rb").read(4096)          # keep the file's line endings
    text = open(path, encoding="utf-8").read()
    backup = path + ".pre-uk-trees"
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    text = MARKER.sub("", OURS.sub("", text))
    block = [rule(f"UK {name}", colour, tiles) for name, (colour, tiles) in SPECIES.items()]
    for size, kinds in SIZES.items():
        block += [rule(f"UK {name} {size.lower()}", colour, tiles) for name, (colour, tiles) in kinds.items()]
    block.append(rule("UK wood + dark grass", *WOOD_GRASS))
    text = text.rstrip("\n") + "\n\n" + "\n".join(block)
    with open(path, "w", encoding="utf-8", newline="\r\n" if crlf else "\n") as f:
        f.write(text)
    print(f"{len(block)} UK tree rules written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
