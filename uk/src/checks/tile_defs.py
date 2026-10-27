"""Tiles a compiled map uses that the game will not know: no tile definition.

    python tile_defs.py <map dir> [--tiles <file.tiles> ...]

Reads every tile name the map's lots use (lots/*.lotheader) and the tilesets
the game defines (its media/*.tiles, the mods the installed map requires -
Erika's Tiles, the KnoxMapUK_* packs - and any --tiles given), and lists the names
no tileset covers: a sheet missing, or an index past its last tile. The game
logs each as "SEVERE: Missing tile definition" and places nothing there (the
stone style's 30-degree gable caps, walls_exterior_roofs_30_21: the mapping
tools have the sheet, game 42.21 does not; user's screenshots 2026-10-14).
"""
from __future__ import annotations

import glob
import os
import struct
import sys
from collections import Counter

GAME = os.path.join(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"),
                    "Steam", "steamapps", "common", "ProjectZomboid")
WORKSHOP = os.path.join(os.path.dirname(os.path.dirname(GAME)), "workshop", "content", "108600")
MODS = os.path.join(os.path.expanduser("~"), "Zomboid", "mods")


def tilesets(path: str) -> dict[str, int]:
    """{tileset name: tiles in it} of a .tiles file ("tdef")."""
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
        name, _img, _cols, _rows, _id, n = s(), s(), i32(), i32(), i32(), i32()
        for _ in range(n):
            for _ in range(i32()):
                s(), s()
        out[name] = n
    return out


def defined(extra: list[str]) -> dict[str, int]:
    files = glob.glob(os.path.join(GAME, "media", "*.tiles"))
    files += glob.glob(os.path.join(WORKSHOP, "*", "mods", "Erikas Tiles", "common", "media", "*.tiles"))
    files += glob.glob(os.path.join(MODS, "KnoxMapUK_*", "common", "media", "*.tiles"))     # every KnoxMap UK pack
    out: dict[str, int] = {}
    for f in files + extra:
        try:
            for name, n in tilesets(f).items():
                out[name] = max(out.get(name, 0), n)
        except (OSError, struct.error, ValueError):
            print("  could not read", f)
    return out


def tile_names(path: str) -> list[str]:
    """The tile names a .lotheader lists: "LOTH", version, count, then each
    name ending in a newline (as tools/building_stats.py reads it)."""
    data = open(path, "rb").read()
    pos = 8 if data[:4] == b"LOTH" else 4
    (count,) = struct.unpack_from("<i", data, pos)
    pos += 4
    out = []
    for _ in range(count):
        end = data.index(b"\n", pos)
        out.append(data[pos:end].decode("latin-1"))
        pos = end + 1
    return out


def used(map_dir: str) -> Counter:
    """{tile name: lot cells using it} from the lots' headers."""
    out: Counter = Counter()
    for f in glob.glob(os.path.join(map_dir, "lots", "*.lotheader")):
        out.update(set(tile_names(f)))
    return out


def main() -> int:
    map_dir = sys.argv[1]
    extra = sys.argv[sys.argv.index("--tiles") + 1:] if "--tiles" in sys.argv else []
    have = defined(extra)
    names = used(map_dir)
    missing: Counter = Counter()
    for name, cells in names.items():
        sheet, idx = name.rsplit("_", 1)
        if sheet not in have or int(idx) >= have[sheet]:
            missing[sheet] += cells
    print(f"{os.path.basename(os.path.normpath(map_dir))}: {len(names)} tile names in its lots, "
          f"{len(have)} tilesets defined; sheets with tiles the game does not define: {len(missing)}")
    for sheet, cells in missing.most_common():
        print(f"  {sheet}{'' if sheet in have else ' (no tileset)'}: {cells} uses (tile x lot cell)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
