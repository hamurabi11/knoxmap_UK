"""Checks on compiled lots that fail a build.

Every ground-level square with anything on it must have a floor. In B42 a ground square with tiles but no floor
makes the game drop the upper storeys of its whole chunk, silently (turned_buildings LOG #D49: a turned house's
upper storey vanished because two squares beside it had lost their only floor). compile_map runs this after every
compile: a floorless square is given the surrounding ground (a warning); the build fails only where no ground tile
can be worked out.

    python tools/lotcheck.py <project dir>       (the map's lots/)

Floors are told by the `solidfloor` tile property, read from the tile definitions the map can use: the game's,
and every installed mod's (~/Zomboid/mods and the Steam workshop folder), so KnoxMap UK's, Erika's and the turned
buildings' packs are all known.
"""
from __future__ import annotations

import glob
import os
import struct
import sys

STEAM = r"C:\Program Files (x86)\Steam\steamapps"
_FLOORS: set | None = None


def _read_tdef(path: str) -> dict:
    """{tile name: set of property keys} of one .tiles file (B42 'tdef' or older)."""
    out = {}
    d = open(path, "rb").read()
    o = 0

    def i32():
        nonlocal o
        v = struct.unpack_from("<i", d, o)[0]; o += 4
        return v

    def line():
        nonlocal o
        e = d.index(b"\n", o); s = d[o:e].decode("latin-1"); o = e + 1
        return s
    if d[:4] == b"tdef":
        o = 4
        i32()
    n = i32()
    for _ in range(n):
        name = line(); line()
        cols, rows, _tid, ntiles = i32(), i32(), i32(), i32()
        for t in range(ntiles):
            np_ = i32()
            keys = set()
            for _p in range(np_):
                keys.add(line()); line()
            out[f"{name}_{t}"] = keys
    return out


def floor_tiles() -> set:
    global _FLOORS
    if _FLOORS is None:
        paths = glob.glob(os.path.join(STEAM, "common", "ProjectZomboid", "media", "*.tiles"))
        paths += glob.glob(os.path.join(os.path.expanduser("~"), "Zomboid", "mods", "*", "**", "*.tiles"), recursive=True)
        paths += glob.glob(os.path.join(STEAM, "workshop", "content", "108600", "*", "mods", "*", "**", "*.tiles"),
                           recursive=True)
        floors = set()
        for p in paths:
            try:
                for name, keys in _read_tdef(p).items():
                    if "solidfloor" in keys:
                        floors.add(name)
            except Exception:
                continue
        _FLOORS = floors
    return _FLOORS


def _ground_squares(lots: str, cx: int, cy: int):
    """{(x, y): [tile names]} of a cell's ground level."""
    h = open(os.path.join(lots, f"{cx}_{cy}.lotheader"), "rb").read()
    o = 4
    _ver, n = struct.unpack_from("<2I", h, o); o += 8
    names = []
    for _ in range(n):
        e = h.index(b"\n", o); names.append(h[o:e].decode("latin-1")); o = e + 1
    _fa, _fb, minz, maxz = struct.unpack_from("<4i", h, o)
    p = open(os.path.join(lots, f"world_{cx}_{cy}.lotpack"), "rb").read()
    nch = struct.unpack_from("<I", p, 8)[0]
    offs = struct.unpack_from(f"<{nch}q", p, 12)
    out = {}
    for ci in range(nch):
        q = offs[ci]
        if q < 0:
            continue
        bx, by = (ci // 32) * 8, (ci % 32) * 8
        i, skip = 0, 0
        while i < 64 * (1 - minz) and i < 64 * (maxz - minz + 1):
            v = struct.unpack_from("<i", p, q)[0]; q += 4
            if v == -1:
                i += struct.unpack_from("<i", p, q)[0]; q += 4
                continue
            vals = struct.unpack_from(f"<{v}i", p, q); q += 4 * v
            zi, rem = divmod(i, 64)
            if minz + zi == 0 and len(vals) > 1:
                x, y = divmod(rem, 8)
                out[(bx + x, by + y)] = [names[t] for t in vals[1:]]
            i += 1
    return out


def ground_floor_failures(lots: str, cells=None) -> list:
    """[(world x, y)] of ground squares with tiles but no floor among them."""
    floors = floor_tiles()
    if cells is None:
        cells = [tuple(int(v) for v in os.path.basename(f)[:-10].split("_"))
                 for f in glob.glob(os.path.join(lots, "*.lotheader"))]
    bad = []
    for cx, cy in cells:
        if not os.path.exists(os.path.join(lots, f"{cx}_{cy}.lotheader")):
            continue
        for (x, y), ts in _ground_squares(lots, cx, cy).items():
            if not any(t in floors for t in ts):
                bad.append((cx * 256 + x, cy * 256 + y, ts[:3]))
    return bad


def _ground(name: str) -> bool:
    return name.startswith(("blends_natural", "blends_grass", "blends_street", "floors_exterior"))


def repair(lots: str, cells=None, log=print) -> tuple[int, list]:
    """Each floorless ground square given the outside ground nearest it (the commonest ground tile within 1, then 2,
    3 squares) under its tiles, the cell written back. Returns (squares repaired, [(x, y) none could be found])."""
    import collections
    from lotio import Cell
    bad = ground_floor_failures(lots, cells)
    by_cell = collections.defaultdict(list)
    for x, y, _ts in bad:
        by_cell[(x // 256, y // 256)].append((x, y))
    fixed, stuck = 0, []
    for (cx, cy), sqs in by_cell.items():
        c = Cell.read(lots, cx, cy)
        for (x, y) in sqs:
            lx, ly = x - cx * 256, y - cy * 256
            tile = None
            for r in (1, 2, 3):
                seen = collections.Counter()
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        v = c.squares.get((lx + dx, ly + dy, 0))
                        if v and _ground(v[0]):
                            seen[v[0]] += 1
                if seen:
                    tile = seen.most_common(1)[0][0]
                    break
            if tile is None:
                stuck.append((x, y))
                continue
            c.squares[(lx, ly, 0)] = [tile] + list(c.squares[(lx, ly, 0)])
            fixed += 1
        c.write(lots)
    if fixed:
        log(f"lotcheck: WARNING {fixed} ground squares had tiles but no floor; given the surrounding ground "
            f"(first {[(x, y) for x, y, _t in bad[:5]]})")
    return fixed, stuck


def check(project_dir: str, log=print) -> None:
    """Every ground square with tiles has a floor: the floorless ones given the surrounding ground (a warning);
    raise only where no ground tile can be worked out."""
    _fixed, stuck = repair(os.path.join(project_dir, "lots"), log=log)
    if stuck:
        raise RuntimeError(f"{len(stuck)} ground squares have tiles but no floor and no ground tile near them to "
                           f"put under them (the game drops the storeys above): first {stuck[:5]}")


if __name__ == "__main__":
    bad = ground_floor_failures(os.path.join(sys.argv[1], "lots"))
    print(f"{len(bad)} ground squares without a floor" + (f": {bad[:10]}" if bad else ""))
    sys.exit(1 if bad else 0)
