"""Multi-tile objects in a compiled map with a piece missing or out of place: the
game cannot remove such an object (it logs "Failed to find all parts of a
multi-tile object!"), and one session hung on tens of thousands of those
(LOG 2026-10-18). Run after compile, before install.

    python multitile_check.py <knoxmap map dir> [--rebuild-table]

What "in place" means is learnt from the game's own maps, not from
SpriteGridPos alone (vanilla's rugs, for one, are not laid along their grid
axes): for every multi-tile sprite (one with SpriteGridPos, in the game's
tile definitions or our pack), where its sibling pieces - sprites of the same
sheet, object name and facing - stand from it in vanilla's lots, as they stand
in most of its placements there. A part in our map is broken when a sibling
expected that way is not on the square expected. The table is built once
(vanilla's lots read, a few minutes) into data/vanilla_multitile.json. Our
pack's sprites, which vanilla never places, are checked by their grid
positions (pieces along x and y as their SpriteGridPos says).
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
KNOX = ukpaths.KNOX
TABLE = os.path.normpath(os.path.join(HERE, "..", "..", "data", "vanilla_multitile.json"))
PACK_TILES = [os.path.normpath(os.path.join(ukpaths.ASSETS or "", "Tile_creator", "build",
                                            "mod", p, "common", "media", p + ".tiles"))
              for p in ("KnoxMapUK_Tiles", "KnoxMapUK_Westminster")]
REACH = 3            # squares: siblings are looked for this far away
MOST = 0.8           # a sibling offset seen in this share of vanilla's placements is expected
CELL = 256           # squares a lot cell (32 chunks of 8)
sys.path.insert(0, os.path.join(KNOX, "tools"))
sys.path.insert(0, ukpaths.assets("Tile_creator", "src") or "")


def multi_sprites() -> dict:
    """{sprite: (object key, (gx, gy, gz))} for every multi-tile sprite known."""
    import pack
    out = {}
    paths = [os.path.join(pack.game_dir(), "media", "newtiledefinitions.tiles")]
    paths += [p for p in PACK_TILES if os.path.exists(p)]
    for path in paths:
        for sheet, tiles in pack.read_tdef(path).items():
            for i, props in enumerate(tiles):
                d = dict(props)
                if "SpriteGridPos" in d:
                    g = ([int(v) for v in d["SpriteGridPos"].split(",")] + [0, 0])[:3]
                    key = f"{sheet}|{d.get('CustomName', '')}|{d.get('GroupName', '')}|{d.get('Facing', '')}"
                    out[f"{sheet}_{i}"] = (key, tuple(g))
    return out


def cells_of(lots: str):
    for h in glob.glob(os.path.join(lots, "*.lotheader")):
        m = re.findall(r"(\d+)_(\d+)\.lotheader", os.path.basename(h))
        if m:
            yield h, int(m[0][0]), int(m[0][1])


def placements(lots: str, sprites: dict, quick=None, failed=None):
    """{(x, y, z): [multi-tile sprites there]} over a lots folder; cells that
    cannot be read are added to `failed`, if given."""
    from cell_check import read_cell
    out = collections.defaultdict(list)
    for h, cx, cy in cells_of(lots):
        if quick is not None:
            names = set(open(h, "rb").read().decode("latin1").split("\n"))
            if names.isdisjoint(quick):
                continue
        try:
            cell = read_cell(lots, cx, cy)
        except Exception:
            if failed is not None:
                failed.append((cx, cy))
            continue
        for (x, y, z), names in cell.items():
            for n in names:
                if n in sprites:
                    out[(cx * CELL + x, cy * CELL + y, z)].append(n)
    return out


def siblings_at(at: dict, sprites: dict, p, n):
    key = sprites[n][0]
    x, y, z = p
    found = set()
    for dx in range(-REACH, REACH + 1):
        for dy in range(-REACH, REACH + 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                for m in at.get((x + dx, y + dy, z + dz), ()):
                    if m != n and sprites[m][0] == key:
                        found.add((dx, dy, dz, m))
    return found


def build_table(sprites: dict) -> dict:
    """{sprite: [[dx, dy, dz, sibling], ...]} as vanilla places them."""
    import pack
    maps = os.path.join(pack.game_dir(), "media", "maps")
    seen = collections.defaultdict(collections.Counter)
    count = collections.Counter()
    for m in sorted(os.listdir(maps)):
        lots = os.path.join(maps, m)
        if not glob.glob(os.path.join(lots, "*.lotheader")):
            continue
        at = placements(lots, sprites, quick=set(sprites))
        for p, names in at.items():
            for n in names:
                count[n] += 1
                for s in siblings_at(at, sprites, p, n):
                    seen[n][s] += 1
    table = {n: [list(s) for s, c in seen[n].items() if c >= MOST * count[n]] for n in count}
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    json.dump(table, open(TABLE, "w"), indent=0)
    return table


def expected(n: str, sprites: dict, table: dict) -> list:
    """Where n's siblings should stand: vanilla's way where vanilla places it
    consistently enough to say, else by grid position (our pack's, and a
    vanilla sprite placed too rarely or too variously for the table)."""
    if table.get(n):
        return [tuple(s) for s in table[n]]
    key, (gx, gy, gz) = sprites[n]
    return [(ox - gx, oy - gy, oz - gz, m) for m, (k, (ox, oy, oz)) in sprites.items()
            if k == key and m != n]


def main() -> int:
    """Exit 0: nothing broken; 1: pieces missing or out of place; 2: nothing
    was checked (no lots, no multi-tile parts, or cells that could not be
    read) - not a pass."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    map_dir = args[0]
    sprites = multi_sprites()
    table = build_table(sprites) if ("--rebuild-table" in sys.argv or not os.path.exists(TABLE)) \
        else json.load(open(TABLE))
    lots = os.path.join(map_dir, "lots")
    failed: list = []
    at = placements(lots, sprites, failed=failed)
    # The map's world origin (world x of map x 0), for reporting map squares.
    ox = oy = 0
    pzw = glob.glob(os.path.join(map_dir, "*.pzw"))
    if pzw:
        m = re.search(r'<worldOrigin origin="(-?\d+),(-?\d+)"', open(pzw[0], encoding="utf-8").read())
        if m:
            ox, oy = int(m.group(1)) * 300, int(m.group(2)) * 300   # world cells are 300 squares
    broken = collections.Counter()
    where = collections.defaultdict(list)
    parts = 0
    for (x, y, z), names in at.items():
        for n in names:
            parts += 1
            want = expected(n, sprites, table)
            missing = [s for s in want if s[3] not in at.get((x + s[0], y + s[1], z + s[2]), ())]
            if missing:
                broken[sprites[n][0].split("|")[0]] += 1
                where[sprites[n][0].split("|")[0]].append((x - ox, y - oy, z, n))
    total = sum(broken.values())
    print(f"{os.path.basename(os.path.normpath(map_dir))}: multi-tile parts {parts}; "
          f"with a piece missing or out of place: {total}")
    for sheet, c in broken.most_common():
        print(f"  {sheet}: {c}  e.g. map {where[sheet][:3]}")
    if failed or not parts:
        print(f"  NOT CHECKED: {'cells unreadable ' + str(failed[:5]) if failed else 'no multi-tile parts in ' + lots}")
        return 2
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
