"""Find rooms open to the sky: top-floor tiles no roof piece covers.

    python roof_cover_check.py <knoxmap map dir> [--list N]

The other half of roof_float_check.py (roof with nothing under it). A tile
of floor f with a room on it and none above it is the top of the building
there; it must be under a FlatTop on floor f, or under a pitched, lean-to or
dormer piece on floor f + 1 (a pitched roof stands on the floor above the
rooms it covers). Reports the share of top tiles left open, per building, and
lists the worst.

Also lean-tos with an open high side. A lean-to (SlopeN/E/S/W, named for the
way it faces downhill) is open along its high edge: it must stand against a
taller part - rooms on the same floor as the roof just past its high edge, or
another roof piece there to run up against.
Where nothing is there, you look under the slope into the room below.
A row of shops or a terrace is built as one building per unit, wall to wall,
so the taller part may be next door: rooms on that floor in the building
beyond (by the map's placements) back a lean-to too.
"""
import csv
import glob
import os
import sys
import xml.etree.ElementTree as ET


def grids_of(root, w, h):
    out = []
    for f in root.findall("floor"):
        g = f.find("rooms")
        vals = [int(v) for v in ((g.text if g is not None else "") or "").replace("\n", ",").split(",")
                if v.strip()]
        out.append(vals if len(vals) == w * h else [0] * (w * h))
    return out


def open_tiles(path):
    """(open tiles, top tiles, [(floor, x, y)] of the open ones)."""
    root = ET.parse(path).getroot()
    if root.tag != "building":
        return 0, 0, []
    w, h = int(root.get("width")), int(root.get("height"))
    floors = root.findall("floor")
    grids = grids_of(root, w, h)
    covered = [set() for _ in range(len(floors) + 1)]
    for lvl, f in enumerate(floors):
        for o in f.iter("object"):
            if o.get("type") != "roof":
                continue
            x, y, rw, rh = (int(o.get(k)) for k in ("x", "y", "width", "height"))
            # A flat roof, and an attic's dormer (over its pocket of room),
            # roof the storey they are on; pitched ones the storey below.
            kind = o.get("RoofType") or ""
            under = lvl if kind == "FlatTop" or kind.startswith("Dormer") else lvl - 1
            if under < 0:
                continue
            covered[under] |= {(cx, cy) for cy in range(y, y + rh) for cx in range(x, x + rw)}
    tops, holes = 0, []
    for lvl in range(len(floors)):
        above = grids[lvl + 1] if lvl + 1 < len(grids) else [0] * (w * h)
        for cy in range(h):
            for cx in range(w):
                if grids[lvl][cy * w + cx] and not above[cy * w + cx]:
                    tops += 1
                    if (cx, cy) not in covered[lvl]:
                        holes.append((lvl, cx, cy))
    return len(holes), tops, holes


HIGH_SIDE = {"SlopeE": (-1, 0), "SlopeW": (1, 0), "SlopeS": (0, -1), "SlopeN": (0, 1)}


def rooms_by_floor(m):
    """{floor: {(map x, map y): building file}} for every building's rooms."""
    name = os.path.basename(os.path.normpath(m))
    origin = {}
    with open(os.path.join(m, f"{name}_placements.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            origin[r["file"]] = (int(r["tile_x"]), int(r["tile_y"]))
    out = {}
    for path in glob.glob(os.path.join(m, "buildings", "*.tbx")):
        base = os.path.basename(path)
        if base not in origin:
            continue
        root = ET.parse(path).getroot()
        if root.tag != "building":
            continue
        w, h = int(root.get("width")), int(root.get("height"))
        ox, oy = origin[base]
        for lvl, g in enumerate(grids_of(root, w, h)):
            cells = out.setdefault(lvl, {})
            for i, v in enumerate(g):
                if v:
                    cells[(ox + i % w, oy + i // w)] = base
    return origin, out


def open_lean_tos(path, origin=None, rooms=None):
    """[(floor, type, x, y, w, h, share of its high edge with nothing taller)].
    With the map's placements and rooms (rooms_by_floor), a building next
    door with rooms on that floor backs it too."""
    base = os.path.basename(path)
    ox, oy = (origin or {}).get(base, (None, None))

    def next_door(lvl, cx, cy):
        if ox is None:
            return False
        other = (rooms or {}).get(lvl, {}).get((ox + cx, oy + cy))
        return other is not None and other != base

    root = ET.parse(path).getroot()
    if root.tag != "building":
        return []
    w, h = int(root.get("width")), int(root.get("height"))
    grids = grids_of(root, w, h)
    out = []
    for lvl, f in enumerate(root.findall("floor")):
        # Tiles under some other roof piece on this floor (a slope the
        # lean-to can run up against).
        roofed = {}
        for o in f.iter("object"):
            if o.get("type") == "roof":
                x, y, rw, rh = (int(o.get(k)) for k in ("x", "y", "width", "height"))
                for cy in range(y, y + rh):
                    for cx in range(x, x + rw):
                        roofed.setdefault((cx, cy), []).append(id(o))
        for o in f.iter("object"):
            kind = o.get("RoofType") or ""
            if o.get("type") != "roof" or kind not in HIGH_SIDE:
                continue
            x, y, rw, rh = (int(o.get(k)) for k in ("x", "y", "width", "height"))
            dx, dy = HIGH_SIDE[kind]
            if dx:
                cx = x - 1 if dx < 0 else x + rw
                edge = [(cx, cy) for cy in range(y, y + rh)]
            else:
                cy = y - 1 if dy < 0 else y + rh
                edge = [(cx, cy) for cx in range(x, x + rw)]
            backed = sum(1 for cx, cy in edge if (0 <= cx < w and 0 <= cy < h and (
                grids[lvl][cy * w + cx] or any(r != id(o) for r in roofed.get((cx, cy), []))))
                or next_door(lvl, cx, cy))
            share = 1 - backed / max(1, len(edge))
            if share > 0:
                out.append((lvl, kind, x, y, rw, rh, share))
    return out


def main() -> int:
    m = sys.argv[1]
    n_list = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 10
    total_open = total_top = 0
    worst = []
    for path in sorted(glob.glob(os.path.join(m, "buildings", "*.tbx"))):
        n, tops, holes = open_tiles(path)
        total_open += n
        total_top += tops
        if n:
            floors = sorted({lvl for lvl, _x, _y in holes})
            worst.append((n, tops, os.path.basename(path), floors))
    origin, rooms = rooms_by_floor(m)
    lean = [(path, lt) for path in sorted(glob.glob(os.path.join(m, "buildings", "*.tbx")))
            for lt in open_lean_tos(path, origin, rooms)]
    fully = [x for x in lean if x[1][6] >= 0.99]
    print(f"lean-tos with an open high side: {len(lean)} ({len(fully)} wholly open), "
          f"in {len({p for p, _ in lean})} buildings")
    for path, (lvl, kind, x, y, rw, rh, share) in sorted(lean, key=lambda t: -t[1][6] * t[1][4] * t[1][5])[:n_list]:
        print(f"  {os.path.basename(path)}  floor {lvl} {kind} at {x},{y} {rw}x{rh}: {share:.0%} of its high edge open")
    print(f"top-floor tiles open to the sky: {total_open} of {total_top} "
          f"({100 * total_open / max(1, total_top):.1f}%), in {len(worst)} buildings")
    for n, tops, name, floors in sorted(worst, reverse=True)[:n_list]:
        print(f"  {n:4d} of {tops:4d}  {name}  (floors {floors})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
