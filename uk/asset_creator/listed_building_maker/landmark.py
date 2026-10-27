"""A landmark built to its own description: the tools an agent (or a person) uses.

    python landmark.py brief <entry> [<map dir>]    what to read before writing: the listing, the photos,
                                                    Build's shell storey by storey, the pieces there are
    python landmark.py check <entry> [<map dir>]    the description against the recipes and Build's shell
    python landmark.py run   <entry> [<map dir>] [--tag T] [--kit]
                                                    remake the tiles, apply, Build, compile the landmark's
                                                    cells, and draw it (outside, roofs off, close-ups)
                                                    into pictures/<entry>/

(KnoxMap's Python. <map dir> defaults to Knoxmap/output/Odiham_town_graph.)

The loop (LANDMARKS.md): brief -> write recipes/<sheet>.json for what the
listing names and the kit does not have -> recipe_kit.py --preview -> look ->
write landmarks/<entry>_<name>.json -> check -> run -> look at the pictures
beside the photos -> change -> run again.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # asset_creator: acpaths
import acpaths  # noqa: E402
KNOXMAP = acpaths.KNOX
PY = sys.executable
MAP = os.path.join(KNOXMAP, "output", "Odiham_town_graph")
DATA = os.path.join(acpaths.UK, "data")


def area_of(map_dir) -> str:
    """The area data folder a map was made from (Odiham_town_graph -> odiham_core)."""
    town = os.path.basename(os.path.normpath(map_dir)).split("_")[0].lower()
    return os.path.join(DATA, f"{town}_core")
PICTURES = os.path.join(HERE, "pictures")
sys.path.insert(0, HERE)


def description(entry) -> tuple[str, dict]:
    paths = sorted(glob.glob(os.path.join(HERE, "landmarks", f"{entry}_*.json")))
    if not paths:
        raise SystemExit(f"no landmarks/{entry}_*.json yet (brief first, then write one)")
    return paths[0], json.load(open(paths[0], encoding="utf-8"))


def built_items() -> dict:
    items = {}
    for path in sorted(glob.glob(os.path.join(HERE, "recipes", "built", "*.json"))):
        b = json.load(open(path, encoding="utf-8"))
        for name, rec in b["items"].items():
            items[name] = dict(rec, sheet=b["sheet"])
    return items


def placed(map_dir, osm_id):
    """(tbx path, x0, y0, w, h) of the building Build made for this OSM id, from the placements."""
    import csv
    name = os.path.basename(os.path.normpath(map_dir))
    rows = list(csv.reader(open(os.path.join(map_dir, f"{name}_placements.csv"), encoding="utf-8")))
    regional = json.load(open(os.path.join(map_dir, "regional.json"), encoding="utf-8"))
    roof = (regional["buildings"].get(str(osm_id)) or {}).get("roof") or []
    if not roof:
        return None
    rx0, ry0 = min(p[0] for p in roof), min(p[1] for p in roof)
    rx1, ry1 = max(p[0] + p[2] for p in roof), max(p[1] + p[3] for p in roof)
    best, cover = None, 0
    for r in rows:
        if len(r) < 6 or not r[2].lstrip("-").isdigit():
            continue
        x0, y0, w, h = (int(v) for v in r[2:6])
        ox = max(0, min(x0 + w, rx1) - max(x0, rx0)) * max(0, min(y0 + h, ry1) - max(y0, ry0))
        if ox > cover:
            best, cover = (os.path.join(map_dir, "buildings", r[0]), x0, y0, w, h), ox
    return best


def shell(tbx_path):
    """Build's storeys as text: each storey's rooms by letter, '.' outside."""
    t = ET.parse(tbx_path).getroot()
    w, h = int(t.get("width")), int(t.get("height"))
    rooms = [r.get("Name") for r in t.findall("room")]
    out = [f"{w} x {h} squares; x east 0-{w - 1}, y south 0-{h - 1}; outside squares x = {w} and y = {h}"]
    for lvl, f in enumerate(t.findall("floor")):
        r = f.find("rooms")
        if r is None:
            continue
        g = [[int(v) for v in row.strip().strip(",").split(",")] for row in r.text.strip().split("\n")]
        if not any(any(row) for row in g):
            continue
        ids = sorted({v for row in g for v in row if v})
        key = {v: chr(65 + k) for k, v in enumerate(ids)}
        out.append(f"\nstorey {lvl} (z {3 * lvl}-{3 * lvl + 3} m): " +
                   ", ".join(f"{key[v]} {rooms[v - 1]}" for v in ids))
        out.append("     " + "".join(str(x % 10) for x in range(w)))
        for y, row in enumerate(g):
            out.append(f"{y:4d} " + "".join(key[v] if v else "." for v in row))
        stairs = [o.attrib for o in f.findall("object") if o.get("type") == "stairs"]
        if stairs:
            out.append("     stairs: " + ", ".join(f"({s['x']},{s['y']},{s['dir']})" for s in stairs))
    return "\n".join(out)


def path_ends(map_dir, tbx_path, x0, y0):
    """Where the map's own paths (paving, flags, footpaths: the ground as Generate
    drew it, before Build's) meet the building's ground floor: the squares just
    outside its walls, as (x, y, side) in the building's squares - where its doors
    belong (user, 2026-10-19: paths line up with doors)."""
    sys.path.insert(0, KNOXMAP)
    import numpy as np
    from generator import pz_colors as C
    from knoxbuild.yards import PAVED
    from knoxbuild.bitmaps import read_rgb, same_colour
    name = os.path.basename(os.path.normpath(map_dir))
    base = os.path.join(map_dir, f"{name}_ground_base.bmp")
    if not os.path.exists(base):
        return []
    ground = read_rgb(base)
    path = np.zeros(ground.shape[:2], bool)
    for c in PAVED + (C.PAVING_STONE, C.DIRT):
        path |= same_colour(ground, c)
    t = ET.parse(tbx_path).getroot()
    g = [[int(v) for v in row.strip().strip(",").split(",")]
         for row in t.find("floor").find("rooms").text.strip().split("\n")]
    h, w = len(g), len(g[0])
    inside = lambda x, y: 0 <= x < w and 0 <= y < h and g[y][x]  # noqa: E731
    # the ground under the buildings next door is not a path (an annex built on a
    # paved yard): their ground floors, read from their own files
    import csv
    for r in csv.reader(open(os.path.join(map_dir, f"{name}_placements.csv"), encoding="utf-8")):
        if len(r) < 6 or not r[2].lstrip("-").isdigit() or os.path.basename(tbx_path) == r[0]:
            continue
        bx, by, bw, bh = (int(v) for v in r[2:6])
        if bx > x0 + w + 1 or by > y0 + h + 1 or bx + bw < x0 - 1 or by + bh < y0 - 1:
            continue
        try:
            other = ET.parse(os.path.join(map_dir, "buildings", r[0])).getroot()
            og = [[int(v) for v in row.strip().strip(",").split(",")]
                  for row in other.find("floor").find("rooms").text.strip().split("\n")]
        except (OSError, ET.ParseError, AttributeError, ValueError):
            continue
        for yy, row in enumerate(og):
            for xx, v in enumerate(row):
                if v and 0 <= by + yy < path.shape[0] and 0 <= bx + xx < path.shape[1]:
                    path[by + yy, bx + xx] = False
    out = []
    for y in range(-1, h + 1):
        for x in range(-1, w + 1):
            if inside(x, y) or not (0 <= y0 + y < path.shape[0] and 0 <= x0 + x < path.shape[1]):
                continue
            if not path[y0 + y, x0 + x]:
                continue
            # the wall it meets: a door object's square and side (outside south/east: its own square)
            if inside(x + 1, y):
                out.append((x + 1, y, "W"))
            if inside(x - 1, y):
                out.append((x, y, "W"))
            if inside(x, y + 1):
                out.append((x, y + 1, "N"))
            if inside(x, y - 1):
                out.append((x, y, "N"))
    return sorted(set(out))


def cmd_brief(a):
    import lexicon as L
    entry = a.entry
    text = json.load(open(os.path.join(HERE, "cache", "text", f"{entry}.json"), encoding="utf-8"))
    print(f"# {text.get('entry_name') or entry} (list entry {entry})\n")
    print(text["description"].strip(), "\n")
    r = L.read(text["description"], text.get("entry_name", ""))
    print(f"read: kind {r.kind}, date {r.era}, features {sorted(r.features())}\n")
    photos = json.load(open(os.path.join(HERE, "cache", "photos", f"{entry}.json"), encoding="utf-8"))
    print("## Photos (look and measure only: never copied or traced into a tile)")
    for p in photos:
        print(f"- {p.get('file', p['source'])} ({p.get('licence', '-')}, {p.get('author', '-')}; use: {p['use']}) {p['page']}")
    print("- also: commons.wikimedia.org categories named after the building (and its '(interior)')\n")
    listed = json.load(open(os.path.join(a.map_dir, "listed_buildings.json"), encoding="utf-8"))
    rows = [b for b in listed["buildings"] if str(b["entry"]) == str(entry) and b.get("main")]
    if rows:
        b = rows[0]
        print(f"## On the map: OSM {b['osm_id']}, style {b['style']}, storeys {b.get('levels', b['was']['levels'])} ({b.get('levels_from', 'as Build had it')})")
        regional = json.load(open(os.path.join(a.map_dir, "regional.json"), encoding="utf-8"))
        e = regional["buildings"][str(b["osm_id"])]
        print("roof parts (x, y, w, h in map tiles, form, storeys):", e.get("roof"))
        where = placed(a.map_dir, b["osm_id"])
        if where:
            path, x0, y0, w, h = where
            print(f"Build's box: origin {x0},{y0} size {w} x {h} ({os.path.basename(path)})\n")
            print("## Build's shell (before the description)" if not e.get("landmark") else
                  "## The building as built now")
            print(shell(path))
            ends = path_ends(a.map_dir, path, x0, y0)
            print("\n## Where the map's paths meet the walls (put the doors here, or say why not)")
            print("  " + (", ".join(f"({x},{y},{d})" for x, y, d in ends) or "none: the paths stop short of the walls"))
    items = built_items()
    print("\n## Pieces there are (recipes/built): name kind sheet")
    for name, rec in sorted(items.items(), key=lambda kv: (kv[1]["sheet"], kv[0])):
        extra = f" {len(rec['squares'])} squares" if rec["kind"] == "window" else ""
        print(f"  {name:22s} {rec['kind']:7s} {rec['sheet']}{extra}")
    print("\nMaterials and patterns: makers/materials.py (python makers/recipe_kit.py --materials out.png)")


def cmd_check(a) -> int:
    path, d = description(a.entry)
    items = built_items()
    if d.get("kind") == "structure":
        missing = sorted({p[1] for p in d.get("pieces", [])} - set(items))
        print(f"{os.path.basename(path)}: a structure, {len(d.get('pieces', []))} pieces" +
              (f"; no such items: {missing}" if missing else ""))
        print("ok" if not missing else "errors")
        return 1 if missing else 0
    w, h = d["size"]
    errors, notes = [], []
    # an outline extended past the box (`extend`, map tiles) widens what is allowed
    ox0 = oy0 = 0
    for ex, ey, ew, eh in d.get("extend", []):
        ox0, oy0 = min(ox0, ex - d["origin"][0]), min(oy0, ey - d["origin"][1])
        w, h = max(w, ex + ew - d["origin"][0]), max(h, ey + eh - d["origin"][1])
    for r in d.get("rooms", []):
        if r.get("floor") and r["floor"] not in items:
            errors.append(f"room {r['name']}: no floor {r['floor']!r}")
        for x0, y0, x1, y1 in r["rects"]:
            if not (ox0 <= x0 <= x1 < w and oy0 <= y0 <= y1 < h):
                errors.append(f"room {r['name']}: rect {[x0, y0, x1, y1]} outside 0-{w - 1} x 0-{h - 1}")
    for kind, rows in (("piece", d.get("pieces", [])), ("furniture", d.get("furniture", []))):
        for lvl, name, x, y, turn in rows:
            rec = items.get(name)
            if not rec:
                errors.append(f"{kind} {name} at {x},{y}: no such item (recipe_kit.py)")
                continue
            if rec["kind"] == "edge" and turn not in ("W", "N"):
                errors.append(f"{name} at {x},{y}: an edge piece goes on a W or N edge, not {turn}")
            if rec["kind"] == "object":
                orient = {"N": "S", "S": "N", "E": "W", "W": "E"}[turn]
                if not any(orient in o for o in rec["tiles"].values()):
                    errors.append(f"{name} at {x},{y}: not drawn facing {turn}")
            if not (ox0 - 1 <= x <= w and oy0 - 1 <= y <= h):
                errors.append(f"{name} at {x},{y}: off the building's squares (0-{w}, 0-{h})")
    sys.path.insert(0, KNOXMAP)
    from knoxbuild.landmark_uk import kit_window
    for lvl, name, x, y, side in d.get("windows", []):
        rec = items.get(name) or kit_window(name)
        if not rec or rec["kind"] != "window":
            errors.append(f"window {name} at {x},{y}: no such window")
            continue
        n = len(rec["squares"])
        on_edge = (side == "W" and x in (ox0, 0, w)) or (side == "N" and y in (oy0, 0, h))
        if not on_edge:
            notes.append(f"window {name} at {x},{y},{side} (storey {lvl}) is not on the box's edge: "
                         "fine where that storey's outside wall is there (a tower above the roofs)")
        if (side == "W" and y + n > h) or (side == "N" and x + n > w):
            errors.append(f"window {name} at {x},{y}: runs past the wall's end")
    seen = {}
    for lvl, name, x, y, side in d.get("windows", []):
        rec = items.get(name) or kit_window(name)
        for k in range(len(rec["squares"]) if rec else 1):
            at = (lvl, x, y + k, side) if side == "W" else (lvl, x + k, y, side)
            if at in seen:
                errors.append(f"windows {seen[at]} and {name} share {at}")
            seen[at] = name
    for lvl, x, y, side in d.get("doors", []):
        if (lvl, x, y, side) in seen:
            errors.append(f"door at {(x, y, side)} storey {lvl} is in a window")
    print(f"{os.path.basename(path)}: {len(d.get('rooms', []))} rooms, {len(d.get('windows', []))} windows, "
          f"{len(d.get('pieces', []))} pieces, {len(d.get('furniture', []))} furniture")
    for n in notes[:12]:
        print("  note:", n)
    for e in errors:
        print("  ERROR:", e)
    print("ok" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


def place_structure(d, map_dir, key) -> int:
    """A structure (a ruin: no rooms, no roof) into <map>_structures.json: each
    piece's tiles at its map squares and storeys.

    The tiles go just after the listed structures' block at the front (apply.py
    `listed_added`; kerbs.py and rails.py count theirs from the end), and are
    recorded under `landmarks[key]`, so a rerun takes out exactly its own and no
    other landmark's. Fence lines inside `clear_fences` [x0, y0, x1, y1] are taken
    out of <map>_fences.geojson: from the file as Generate wrote it (kept as
    .orig.geojson; a fresh Generate, which has no `cleared` record, refreshes it),
    every landmark's clearing applied together."""
    import shutil
    from collections import Counter
    name = os.path.basename(os.path.normpath(map_dir))
    items = built_items()
    path = os.path.join(map_dir, f"{name}_structures.json")
    data = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"tiles": []}
    mine_before = Counter(tuple(t) for t in (data.get("landmarks") or {}).get(key, []))
    tiles = []
    for t in data.get("tiles", []):
        if mine_before[tuple(t)]:
            mine_before[tuple(t)] -= 1
            continue
        tiles.append(t)
    used = {(t[0], t[1], t[2], t[3]) for t in tiles}
    layers = ("Furniture", "Furniture2", "Furniture3", "Furniture4", "WallOverlay2", "WallOverlay3")
    mine = []
    for lvl, item, x, y, turn in d.get("pieces", []):
        rec = items[item]
        orient = turn if rec["kind"] == "edge" else {"N": "S", "S": "N", "E": "W", "W": "E"}[turn]
        for k, orients in rec["tiles"].items():
            for tkey, tile in (orients.get(orient) or {}).items():
                ox, oy = map(int, tkey.split(","))
                at = (x + ox, y + oy, lvl + int(k))
                layer = next(L for L in layers if at + (L,) not in used)
                used.add(at + (layer,))
                mine.append([at[0], at[1], at[2], layer, tile])
    front = int(data.get("listed_added") or 0)
    data["tiles"] = tiles[:front] + mine + tiles[front:]
    data.setdefault("landmarks", {})[key] = mine
    data.pop("landmark_added", None)
    json.dump(data, open(path, "w", encoding="utf-8"))
    fpath = os.path.join(map_dir, f"{name}_fences.geojson")
    if (d.get("clear_fences") or os.path.exists(fpath.replace(".geojson", ".orig.geojson"))) and os.path.exists(fpath):
        sys.path.insert(0, KNOXMAP)
        from knoxbuild.build import Projector
        info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
        b = info["bbox"]
        proj = Projector.build(b["south"], b["west"], b["north"], b["east"], info.get("meters_per_tile", 1.0),
                               info.get("rotation", 0.0))
        orig = fpath.replace(".geojson", ".orig.geojson")
        current = json.load(open(fpath, encoding="utf-8"))
        cleared = dict(current.get("cleared") or {})
        if not os.path.exists(orig) or "cleared" not in current:
            shutil.copyfile(fpath, orig)                 # as Generate wrote it (fresh, or the first time)
        if d.get("clear_fences"):
            cleared[key] = d["clear_fences"]
        else:
            cleared.pop(key, None)
        fences = json.load(open(orig, encoding="utf-8"))

        def inside(f, box):
            x0, y0, x1, y1 = box
            pts = f["geometry"]["coordinates"]
            pts = pts[0] if isinstance(pts[0][0], list) else pts
            px = [proj.to_px(la, lo) for lo, la in pts]
            return all(x0 <= q[0] <= x1 and y0 <= q[1] <= y1 for q in px)
        kept = [f for f in fences["features"] if not any(inside(f, box) for box in cleared.values())]
        print(f"  fences: {len(fences['features']) - len(kept)} line(s) taken out ({sorted(cleared)})")
        json.dump(dict(fences, features=kept, cleared=cleared), open(fpath, "w", encoding="utf-8"))
    print(f"  {len(mine)} tiles placed in {os.path.basename(path)}")
    return len(mine)


def cmd_run(a) -> int:
    if cmd_check(a):
        return 1
    _path, d = description(a.entry)
    if d.get("kind") == "structure":
        return run_structure(a, d)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def run(args, cwd, tail=1):
        p = subprocess.run([PY] + args, cwd=cwd, env=env, capture_output=True, text=True)
        out = (p.stdout + p.stderr).strip().splitlines()
        print("  " + "\n  ".join(out[-tail:]) if out else "  (no output)")
        if p.returncode:
            print("  FAILED:", " ".join(args[:2]))
            print("\n".join(out[-25:]))
            raise SystemExit(1)
        return out

    makers = os.path.join(HERE, "makers")
    if a.kit:
        kit = (d.get("walls") or {}).get("exterior")
        if kit:
            run(["wall_kit.py", kit], makers)
    run(["recipe_kit.py"], makers, tail=4)
    tc = os.path.join(ROOT, "asset_creator", "Tile_creator", "src")
    run(["pack.py"], tc)
    run(["register.py"], tc)
    run(["apply.py", a.map_dir, area_of(a.map_dir), "--entries", str(a.entry)], HERE)
    run(["-m", "knoxbuild", a.map_dir], KNOXMAP)
    run(["tools/validate_tbx.py", os.path.join(a.map_dir, "buildings")], KNOXMAP)
    where = placed(a.map_dir, d["osm_id"])
    if not where:
        print("  the building is not on the map")
        return 1
    _p, x0, y0, w, h = where
    m = 12
    run(["-c", "import sys; sys.path.insert(0, 'tools'); from compile_map import compile_map; "
         f"print('cells', compile_map({a.map_dir!r}, fresh=False, only_cells=[[{x0 - m}, {y0 - m}, {x0 + w + m}, {y0 + h + m}]]))"],
        KNOXMAP)
    out = os.path.join(PICTURES, str(a.entry))
    os.makedirs(out, exist_ok=True)
    box = f"{x0 - 8},{y0 - 6},{w + 16},{h + 12}"
    shots = [("outside", ["--box", box, "--size", "1400x1100"]),
             ("inside", ["--box", box, "--size", "1400x1100", "--no-roofs"])]
    # close-ups: the camera's two faces (south and east) and the inside, a third at a time
    third = max(8, -(-h // 3))
    for k in range(0, h, third):
        shots.append((f"inside_{k}", ["--box", f"{x0},{y0 + k},{w},{min(third, h - k)}", "--scale", "1", "--no-roofs"]))
    # the whole building at the game's own scale (a box through the building would cut it open)
    span = w + h + 32
    shots.append(("outside_full", ["--box", box, "--scale", "1", "--size", f"{span * 64}x{span * 32 + 2400}"]))
    for name, args in shots:
        path = os.path.join(out, f"{name}_{a.tag}.png")
        run(["tools/make_pictures.py", a.map_dir, "--out", path] + args, KNOXMAP)
        _crop(path)
    print("pictures in", out)
    return 0


def run_structure(a, d) -> int:
    """The loop for a structure: tiles, pack, place, Build, compile its cells, pictures."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    map_dir = os.path.join(KNOXMAP, "output", d["map"]) if d.get("map") else a.map_dir

    def run(args, cwd, tail=1):
        p = subprocess.run([PY] + args, cwd=cwd, env=env, capture_output=True, text=True)
        out = (p.stdout + p.stderr).strip().splitlines()
        print("  " + "\n  ".join(out[-tail:]) if out else "  (no output)")
        if p.returncode:
            print("\n".join(out[-25:]))
            raise SystemExit(1)
    run(["recipe_kit.py"], os.path.join(HERE, "makers"), tail=2)
    tc = os.path.join(ROOT, "asset_creator", "Tile_creator", "src")
    run(["pack.py"], tc)
    run(["register.py"], tc)
    place_structure(d, map_dir, str(a.entry))
    run(["-m", "knoxbuild", map_dir], KNOXMAP)
    x0, y0, x1, y1 = d["box"]
    m = 12
    run(["-c", "import sys; sys.path.insert(0, 'tools'); from compile_map import compile_map; "
         f"print('cells', compile_map({map_dir!r}, fresh=False, only_cells=[[{x0 - m}, {y0 - m}, {x1 + m}, {y1 + m}]]))"],
        KNOXMAP)
    out = os.path.join(PICTURES, str(a.entry))
    os.makedirs(out, exist_ok=True)
    w, h = x1 - x0, y1 - y0
    span = w + h + 40
    for name, args in (("outside", ["--box", f"{x0 - 10},{y0 - 10},{w + 20},{h + 20}", "--size", "1400x1100"]),
                       ("outside_full", ["--box", f"{x0 - 4},{y0 - 4},{w + 8},{h + 8}", "--scale", "1",
                                         "--size", f"{span * 64}x{span * 32 + 2400}"])):
        path = os.path.join(out, f"{name}_{a.tag}.png")
        run(["tools/make_pictures.py", map_dir, "--out", path] + args, KNOXMAP)
        _crop(path)
    print("pictures in", out)
    return 0


def _crop(path):
    """Trim a picture's empty background (the full-scale one is mostly sky)."""
    import numpy as np
    from PIL import Image
    im = Image.open(path)
    a = np.asarray(im.convert("RGB")).astype(int)
    ys, xs = np.nonzero(np.abs(a - a[2, 2]).sum(-1) > 20)
    if len(ys):
        im.crop((max(0, xs.min() - 8), max(0, ys.min() - 8), xs.max() + 9, ys.max() + 9)).save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("brief", "check", "run"):
        p = sub.add_parser(name)
        p.add_argument("entry")
        p.add_argument("map_dir", nargs="?", default=MAP)
        if name == "run":
            p.add_argument("--tag", default="now")
            p.add_argument("--kit", action="store_true", help="remake the description's wall kit too")
    a = ap.parse_args(argv)
    a.map_dir = os.path.abspath(a.map_dir)                 # apply.py and Build run from their own folders
    return {"brief": cmd_brief, "check": cmd_check, "run": cmd_run}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
