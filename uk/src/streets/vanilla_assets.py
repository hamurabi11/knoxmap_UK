"""A map's loose tiles with no UK tile (UK assets off: Knoxmap/knoxbuild/assets_uk.py).

    python vanilla_assets.py <map dir>
    python vanilla_assets.py <map dir> --check     (after Build: exit 1 if any UK tile is left)

After kerbs.py, which writes the last of them (kerbs, overlays, markings,
signs and street furniture from roadgraph/, rails, sports), into
<map>_structures.json: our kerbs and road overlays back to vanilla's pieces of
the same name (street_curbs_uk_01 is drawn in exactly vanilla's shapes and
slots: kerbs.uk_kerb), our pavement overlay to vanilla's pale concrete as
before the UK pack (street_curbs_01_blend_gravel); every other UK tile (road
markings, signs, street furniture, pub signs: vanilla has none of them) left
out. A lookup, run by uk_pipeline.py when uk.json's uk_assets is false.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

UK = re.compile(r"_uk_\d\d")
# Our sheet -> vanilla's (the rest of the name is the same piece).
SWAP = [("street_curbs_uk_01_blend_pavement", "street_curbs_01_blend_gravel"),
        ("street_curbs_uk_01", "street_curbs_01")]


def vanilla_name(name: str) -> str | None:
    """A tile's vanilla piece, or None to leave it out; a vanilla tile as it is."""
    if not UK.search(name):
        return name
    for ours, theirs in SWAP:
        if name.startswith(ours):
            return theirs + name[len(ours):]
    return None


def check(map_dir: str) -> int:
    """Every UK tile left in the built map (its buildings and loose tiles): none, or exit 1."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
    import ukpaths  # noqa: E402
    sys.path.insert(0, ukpaths.KNOX)
    from knoxbuild import assets_uk
    left = assets_uk.uk_tiles(map_dir)
    if not left:
        print("no UK tile in the map: it needs no KnoxMapUK_Tiles")
        return 0
    sheets = Counter()
    for t, n in left.items():
        sheets[re.sub(r"_\d+$", "", t)] += n
    print(f"UK tiles still in the map ({sum(left.values())}):")
    for k, n in sheets.most_common(20):
        print(f"  {n:6d}  {k}")
    return 1


def main() -> int:
    map_dir = sys.argv[1]
    if "--check" in sys.argv:
        return check(map_dir)
    name = os.path.basename(os.path.normpath(map_dir))
    path = os.path.join(map_dir, f"{name}_structures.json")
    data = json.load(open(path, encoding="utf-8"))
    swapped, dropped, kept = Counter(), Counter(), []
    for t in data.get("tiles", []):
        v = vanilla_name(t[4])
        if v is None:
            dropped[re.sub(r"_\d+$", "", t[4])] += 1
            continue
        if v != t[4]:
            swapped[re.sub(r"_\d+$", "", t[4])] += 1
        kept.append([*t[:4], v, *t[5:]])
    data["tiles"] = kept
    data["uk_assets_off"] = {"swapped": sum(swapped.values()), "left out": sum(dropped.values())}
    json.dump(data, open(path, "w", encoding="utf-8"))
    print(f"{name}: UK tiles swapped for vanilla {sum(swapped.values())}, left out {sum(dropped.values())}")
    for k, n in dropped.most_common(8):
        print(f"  left out {n:6d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
