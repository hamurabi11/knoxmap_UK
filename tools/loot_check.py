"""Does the furniture in each room get loot? (KnoxMap UK, 2026-10-26)

The game fills a container from its loot table for the room's name and the
container's type (Distributions.lua: room -> container -> lists; the container
type is the sprite's `container` property in the tile definitions). A room with
no table of its own, or a container its room's table does not name, falls back
to the `all` table - which holds the game's ordinary household containers (a
home's fridge, a hall's dresser) on purpose. A container type in neither is
never filled: a butcher's display case in a clothes shop stays empty.

Some containers are filled only for the tiles their table names: every list in
the room's `freezer` (say) carries a `forceForTiles`, and a freezer whose sprite
is in none of them stays empty (the Gigamart's freezer knows the game's chest
freezers, not the shop freezers appliances_refrigeration_01_48-51). Those are
reported as MISS, by sprite. KnoxMap's own additions (knoxbuild/lua/freezers.lua,
shipped in every map mod) are counted in.

    python tools/loot_check.py <map>/lots        # every room name in a compiled map

Reads the game install (GAME below); `tables()` and `containers()` are used by
tests/test_loot.py on Build's own fittings.
"""
from __future__ import annotations

import collections
import functools
import glob
import os
import re
import sys

GAME = os.environ.get("PZ_GAME_MEDIA", r"C:\Program Files (x86)\Steam\steamapps\common\ProjectZomboid\media")


def available() -> bool:
    return os.path.exists(os.path.join(GAME, "newtiledefinitions.tiles.txt"))


@functools.lru_cache(maxsize=1)
def containers() -> dict:
    """{sprite name: container type} from the game's tile definitions. A fridge or
    freezer has no `container` property: the game makes it a "fridge" (IsFridge)
    or a "freezer" (Freezer alone; a fridge's own freezer compartment is not counted)."""
    out, name, cold = {}, None, {}
    with open(os.path.join(GAME, "newtiledefinitions.tiles.txt"), encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("// "):
                name = s[3:]
            elif s.startswith("container = ") and name:
                out[name] = s[12:]
            elif name and s.startswith("IsFridge ="):
                cold[name] = "fridge"
            elif name and s.startswith("Freezer =") and name not in cold:
                cold[name] = "freezer"
    for n, kind in cold.items():
        out.setdefault(n, kind)
    return out


@functools.lru_cache(maxsize=1)
def tables() -> dict:
    """{room name: set of container types its loot table fills}, with the game's
    aliases (SuburbsDistributions: garage = mechanic, diningroom = dining...)."""
    items = os.path.join(GAME, "lua", "server", "Items")
    text = _distribution_text()
    start = text.index("{", text.index("distributionTable"))
    rooms =collections.defaultdict(set)
    depth, room = 0, None
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{|\{|\}", text[start:]):
        if m.group(0).endswith("{"):
            depth += 1
            if m.group(1) and depth == 2:
                room = m.group(1)
                rooms[room]
            elif m.group(1) and depth == 3 and room:
                rooms[room].add(m.group(1))
        else:
            depth -= 1
            if depth == 0:
                break
    sub = open(os.path.join(items, "SuburbsDistributions.lua"), encoding="utf-8", errors="replace").read()
    for alias, real in re.findall(r"SuburbsDistributions\.(\w+)\s*=\s*SuburbsDistributions\.(\w+)", sub):
        if real in rooms:
            rooms[alias] = rooms[real]
    return dict(rooms)


def _distribution_text() -> str:
    items = os.path.join(GAME, "lua", "server", "Items")
    return re.sub(r"--[^\n]*", "", open(os.path.join(items, "Distributions.lua"), encoding="utf-8",
                                        errors="replace").read())


FREEZERS_LUA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "knoxbuild", "lua", "freezers.lua")


def knoxmap_force() -> tuple[set, list]:
    """The rooms and tiles knoxbuild/lua/freezers.lua adds to their freezers' forceForTiles."""
    try:
        text = open(FREEZERS_LUA, encoding="utf-8").read()
    except OSError:
        return set(), []
    def strings(name):
        m = re.search(name + r"\s*=\s*\{([^}]*)\}", text)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else []
    return set(strings("ROOMS")), strings("TILES")


FILLS_LUA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knoxbuild", "lua", "fills.lua")


def knoxmap_fills() -> list:
    """(room, container) given an ordinary loot list by knoxbuild/lua/fills.lua."""
    try:
        text = open(FILLS_LUA, encoding="utf-8").read()
    except OSError:
        return []
    return re.findall(r'\{\s*"(\w+)",\s*"(\w+)",\s*"\w+"\s*\}', text)


@functools.lru_cache(maxsize=1)
def forced() -> dict:
    """{(room, container): set of tiles} for every container whose lists are all
    forceForTiles ones: filled for those tiles only. With KnoxMap's additions."""
    text = _distribution_text()
    start = text.index("{", text.index("distributionTable"))
    out = {}
    depth, room, cont, c0 = 0, None, None, 0
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{|\{|\}", text[start:]):
        if m.group(0).endswith("{"):
            depth += 1
            if m.group(1) and depth == 2:
                room = m.group(1)
            elif m.group(1) and depth == 3 and room:
                cont, c0 = m.group(1), m.end()
        else:
            if depth == 3 and cont:
                body = text[start + c0:start + m.start()]
                entries = re.findall(r"\{\s*name\s*=[^{}]*\}", body)
                if entries and all("forceForTiles" in e for e in entries):
                    tiles = set()
                    for e in entries:
                        f = re.search(r'forceForTiles\s*=\s*"([^"]*)"', e)
                        tiles |= {t for t in (f.group(1) if f else "").split(";") if t}
                    out[(room, cont)] = tiles
                cont = None
            depth -= 1
            if depth == 0:
                break
    rooms, tiles = knoxmap_force()
    for (room, cont), have in out.items():
        if room in rooms and cont == "freezer":
            have |= set(tiles)
    for room, cont in knoxmap_fills():          # an ordinary list added (knoxbuild/lua/fills.lua): not tile-only
        out.pop((room, cont), None)
    sub = open(os.path.join(GAME, "lua", "server", "Items", "SuburbsDistributions.lua"),
               encoding="utf-8", errors="replace").read()
    for alias, real in re.findall(r"SuburbsDistributions\.(\w+)\s*=\s*SuburbsDistributions\.(\w+)", sub):
        for (room, cont), have in list(out.items()):
            if room == real:
                out[(alias, cont)] = have
    return out


def _game_name(sprite: str) -> str:
    sheet, _, n = sprite.rpartition("_")
    return f"{sheet}_{int(n)}" if n.isdigit() else sprite


def container_of(sprite: str) -> str | None:
    """A sprite's container type (Build writes its numbers padded: _080 is the game's _80)."""
    return containers().get(_game_name(sprite))


def filled(room: str, container: str) -> str:
    """'own' (the room's table fills it), 'all' (the fallback does), or 'none'."""
    t = tables()
    if container in t.get(room, ()):
        return "own"
    return "all" if container in t.get("all", ()) else "none"


def filled_tile(room: str, sprite: str) -> str:
    """As filled() for this sprite's container, or 'miss' when the table that fills
    it has only forceForTiles lists and none names this sprite."""
    container = container_of(sprite)
    if container is None:
        return "none"
    how = filled(room, container)
    if how == "none":
        return how
    tiles = forced().get((room if how == "own" else "all", container))
    return "miss" if tiles is not None and _game_name(sprite) not in tiles else how


def main(argv) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from building_stats import read_header, read_squares
    lots = argv[0]
    cont = containers()
    have = collections.defaultdict(collections.Counter)
    sprites = collections.defaultdict(collections.Counter)      # room -> sprite with a container
    area, count = collections.Counter(), collections.Counter()
    for hp in glob.glob(os.path.join(lots, "*.lotheader")):
        cx, cy = (int(v) for v in os.path.basename(hp)[:-10].split("_"))
        names, lo, hi, rms, _b = read_header(hp)
        pk = os.path.join(lots, f"world_{cx}_{cy}.lotpack")
        if not os.path.exists(pk):
            continue
        sq = read_squares(pk, names, lo, hi)
        for rname, level, rects in rms:
            count[rname] += 1
            for (x, y, w, h) in rects:
                area[rname] += w * h
                for i in range(w):
                    for j in range(h):
                        for t in sq.get((x + i, y + j, level), []):
                            if t in cont:
                                have[rname][cont[t]] += 1
                                sprites[rname][t] += 1
    t = tables()
    print(f"{'room':22} {'rooms':>6} {'m2':>7}  containers by who fills them")
    for rname in sorted(count, key=lambda r: -area[r]):
        by = collections.defaultdict(dict)
        for c, n in have[rname].items():
            by[filled(rname, c)][c] = n
        tag = "" if rname in t else "NO TABLE"
        print(f"{rname:22} {count[rname]:6} {area[rname]:7}  {tag:8} own={by['own']} all={by['all']} NONE={by['none']}")
    misses = [(rname, t, n) for rname, c in sprites.items() for t, n in c.items() if filled_tile(rname, t) == "miss"]
    print()
    if misses:
        print("MISS - the table fills only the tiles its forceForTiles names, not these:")
        for rname, t, n in sorted(misses):
            print(f"  {rname:22} {t:36} {cont[t]:14} {n}")
    else:
        print("forceForTiles: no container on a tile its table leaves out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
