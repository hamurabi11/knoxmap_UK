"""KnoxMap UK: a landmark built to its own description (a Grade I church, an inn).

Build lays a building out from its footprint, its storeys and the roof the
LiDAR gave it (regional_uk.roof_plan). For a landmark that is only the
shell: the listed building maker writes a description of what is inside
and on it (asset_creator/listed_building_maker/landmarks/<entry>_*.json,
copied into regional.json by apply.py as the entry's `landmark`, with the
tiles recipe_kit.py drew for it), and `dress` puts that into Build's plan
before tbx.py writes it:

  rooms        named, per storey, from rectangles (the first to claim a
               square has it); a square Build covers that none claims goes
               to its nearest room. Their floors are the recipe sheets'.
  open_levels  storeys whose rooms run into each other with no wall (a
               church's nave, aisles and chancel: the arcades are pieces).
  void         rooms that are the open space under the roof (no floor);
               `keep_floor` a gallery in it.
  doors, windows   at squares; a window is the recipe's, square by square, or
               the window kit's ("uk:sash_georgian").
  stairs       [level, x, y, dir]: where the stairs rise, if not where Build put them.
  clear_roof   Build's roof taken off a rectangle a piece stands in for (a spire).
  pieces, furniture, vanilla   tiles painted on the storeys: arcades,
               piers, boards, pews, the font, buttresses, the tower's
               parapet and pinnacles. A piece taller than a storey goes on
               the storeys above too, as its recipe cut it.
  furnish      Build furnishes a room so marked by its kind (an inn's bedrooms);
               every room gets its light switch whatever.
  walls        the outside walls' kit, the tower's kit from a storey up,
               one plaster for every room inside.

Coordinates are the building's own squares from its box's north-west
corner (`origin`, the box in map tiles when the description was written; if
Build has since moved the box, the description moves with the building).
Build keeps its storey count, and its stairs and roof unless the description
gives its own (`stairs`; `roof` through apply.py). Other keys: `openings`,
room `walled`/`furnish`, `walls.parts`, `keep_clear`; `join`, `extend` and
`footprint` act earlier, in Build (regional_uk.join_units, extend_outlines);
`door_surround` is apply.py's. See
asset_creator/listed_building_maker/LANDMARKS.md.
"""
from __future__ import annotations

import re
from collections import deque

from .layout import Plan, Room

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
# Painted tile layers a landmark may use, in order (WallFurniture is the trims', WallFurniture2 the drains').
LAYERS = ("Furniture", "Furniture2", "Furniture3", "Furniture4", "WallOverlay2", "WallOverlay3",
          "WallOverlay4", "WallFurniture3", "WallFurniture4")
# On the roof floor and above (a parapet, a spire, a gable's tile-hanging): roof
# layers, laid over the roof's own tiles (buildtiles.py).
ROOF_LAYERS = ("RoofTop", "Roof2") + LAYERS


def _cells(rects, w, h):
    for x0, y0, x1, y1 in rects:
        for y in range(max(0, y0), min(h - 1, y1) + 1):
            for x in range(max(0, x0), min(w - 1, x1) + 1):
                yield x, y


def dress(building, roof: dict | None, style: dict) -> tuple:
    """The landmark's rooms, openings and pieces into Build's plan; the
    style and roof plan changed to suit. Returns (building, roof, style)."""
    lm = style["landmark"]
    d, items = lm["desc"], lm["items"]
    # a description begun before Build's box is known (only its joins and
    # style): its coordinates are Build's own
    origin = d.get("origin") or lm["at"]
    dx = origin[0] - lm["at"][0]
    dy = origin[1] - lm["at"][1]
    w, h = building.width, building.height
    n = len(building.storeys)
    style = dict(style)
    roof = dict(roof or {})
    void: dict[int, set] = {}
    keep: dict[int, set] = {}

    # ---- rooms, storey by storey ----
    new = []
    # walls shared with the building next door, storey by storey (Build's): no door
    # or window goes in one - the neighbour's wall stands on that edge too
    party = [set(getattr(st, "party", ()) or ()) for st in building.storeys]
    if not d.get("rooms"):
        # No rooms of its own yet (a first look): Build's rooms and furniture stay.
        for old in building.storeys:
            old.window_entries = {}
            for r in old.rooms:
                r.landmark_floor, r.landmark_name, r.landmark_walled, r.landmark_furnish = None, r.kind, False, True
    from . import layout as L
    unknown = sorted({r.get("kind", "church") for r in d.get("rooms", ()) if r.get("kind", "church") not in L.ROOM_STYLE})
    if unknown:
        raise ValueError(f"landmark {d.get('entry')}: room kinds Build does not know: {unknown} "
                         f"(one of layout.ROOM_STYLE, e.g. church, hall, storage, bedroom, kitchen)")
    described = {lvl for r in d.get("rooms", ()) for lvl in r["levels"]}
    for lvl, old in enumerate(building.storeys if d.get("rooms") else ()):
        if lvl not in described:
            # a storey the description says nothing of keeps Build's own plan
            print(f"landmark {d.get('entry')}: storey {lvl} has no rooms in the description: Build's are kept")
            old.window_entries = {}
            for r in old.rooms:
                r.landmark_floor, r.landmark_name, r.landmark_walled, r.landmark_furnish = None, r.kind, False, True
            new.append(old)
            continue
        here = {(x, y) for y, row in enumerate(old.grid) for x, v in enumerate(row) if v}
        # the storey's own footprint too: Build's plan keeps one piece of a storey
        # in two (Parliament's two towers each upper storey: only the one with the
        # stairs), and the description lays the whole storey out itself
        if old.mask is not None:
            here |= {(x, y) for y, row in enumerate(old.mask) for x, v in enumerate(row) if v}
        grid = [[0] * w for _ in range(h)]
        rooms = []
        for r in d["rooms"]:
            if lvl not in r["levels"]:
                continue
            mine = [(x, y) for x, y in _cells([[a + dx, b + dy, c + dx, e + dy] for a, b, c, e in r["rects"]], w, h)
                    if (x, y) in here and not grid[y][x]]
            if not mine:
                continue
            xs, ys = [c[0] for c in mine], [c[1] for c in mine]
            room = Room(min(xs), min(ys), max(xs), max(ys), r.get("kind", "church"))
            room.landmark_floor = (items.get(r["floor"]) or {}).get("tile") if r.get("floor") else None
            room.landmark_name = r["name"]
            room.landmark_walled = bool(r.get("walled"))
            room.landmark_furnish = bool(r.get("furnish"))
            rooms.append(room)
            for x, y in mine:
                grid[y][x] = len(rooms)
            if r.get("void"):
                void.setdefault(lvl, set()).update(mine)
            if r.get("keep_floor"):
                keep.setdefault(lvl, set()).update(mine)
        _fill(grid, here)
        lost = sum(1 for x, y in here if not grid[y][x])
        if lost:
            print(f"landmark {d.get('entry')}: storey {lvl}: {lost} squares Build covers reach no room (left out)")
        plan = Plan(width=w, height=h, rooms=rooms, grid=grid, mask=old.mask, kind=old.kind,
                    profile=old.profile, drawn=True, party=set(old.party or ()))
        plan.window_entries = {}
        if lvl in d.get("open_levels", ()):
            walled = {k + 1 for k, rm in enumerate(rooms) if rm.landmark_walled}
            plan.open_edge = _open_edges(grid, walled)
        # openings named one by one (a Norman tower's arches): [level, x, y, side, length]
        for o_lvl, ox, oy, side, length in d.get("openings", ()):
            if o_lvl == lvl:
                plan.open_edge |= {(ox + dx, oy + dy + k, side) if side == "W" else (ox + dx + k, oy + dy, side)
                                   for k in range(length)}
        new.append(plan)
    if d.get("rooms"):
        building.storeys = new
    new = building.storeys
    if d.get("rooms"):
        roof["void"] = {lvl: cells - keep.get(lvl, set()) for lvl, cells in void.items()}
    # Build's roof taken off where a piece stands in its place (a spire on a tower):
    # `clear_roof` [x0, y0, x1, y1], the pieces lying wholly inside it.
    if d.get("clear_roof") and roof.get("pieces"):
        x0, y0, x1, y1 = d["clear_roof"]
        x0, y0, x1, y1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
        roof["pieces"] = {lvl: [p for p in ps if not (x0 <= p[3][0] and y0 <= p[3][1]
                                                      and p[3][0] + p[3][2] - 1 <= x1 and p[3][1] + p[3][3] - 1 <= y1)]
                          for lvl, ps in roof["pieces"].items()}

    # ---- doors and windows ----
    skipped = []
    for lvl, x, y, side in d.get("doors", ()):
        if lvl < n:
            if (x + dx, y + dy, side) in party[lvl]:
                skipped.append(("door", lvl, x, y, side))
                continue
            if (x + dx, y + dy, side) not in new[lvl].doors:
                new[lvl].doors.append((x + dx, y + dy, side))
    for lvl, name, x, y, side in d.get("windows", ()):
        rec = items.get(name) or kit_window(name)
        if not rec or lvl >= n:
            continue
        for k, sq in enumerate(rec["squares"]):
            at = (x + dx, y + dy + k, side) if side == "W" else (x + dx + k, y + dy, side)
            if at in party[lvl]:
                skipped.append(("window", lvl, x, y, side))
                continue
            if at in new[lvl].doors:
                skipped.append(("window in a door", lvl, x, y, side))
                continue
            if at not in new[lvl].windows:
                new[lvl].windows.append(at)
            new[lvl].window_entries[at] = {"category": "windows", "tiles": {"West": sq["W"], "North": sq["N"]}}

    if skipped:
        print(f"landmark {d.get('entry')}: left out (a wall shared with the building next door, or a door's): {skipped}")

    # ---- stairs: the description's, where it gives them ([level, x, y, dir]) ----
    # A landmark may have as many staircases as it needs (Parliament: one in each
    # range and tower): the first a storey names replaces Build's own, the rest
    # stand beside it (`extra_stairs`, written by tbx.py; the game cuts each one's
    # opening in the floor above itself).
    building.extra_stairs = []
    if d.get("stairs"):
        stairs = list(building.stairs) + [None] * max(0, n - 1 - len(building.stairs))
        named = set()
        for lvl, x, y, sdir in d["stairs"]:
            if lvl >= n - 1:
                continue
            if lvl in named:
                building.extra_stairs.append((lvl, x + dx, y + dy, sdir))
            else:
                stairs[lvl] = (x + dx, y + dy, sdir)
                named.add(lvl)
        building.stairs = [st for st in stairs[:n - 1] if st is not None]

    # ---- Build's own furnishing ----
    # Every room its light switch (the game hangs a room's light off it: a
    # room without one is dark), and a room the description marks `furnish`
    # (an inn's bedrooms, its kitchen) everything Build gives its kind.
    import random
    rng = random.Random(f"landmark:{d.get('entry')}")
    # squares kept clear of Build's furniture: a staircase's run and its landing
    # above (the description may have moved them), and `keep_clear`
    # [[level, x0, y0, x1, y1]] (a fireplace's hearth)
    clear: dict[int, set] = {}
    for lvl, st in list(enumerate(building.stairs)) + [(e[0], e[1:]) for e in building.extra_stairs]:
        if st is None:
            continue
        sx, sy, sdir = st
        run = {(sx + k, sy) if sdir == "W" else (sx, sy + k) for k in range(L.STAIR_RUN)}
        clear.setdefault(lvl, set()).update(run)
        clear.setdefault(lvl + 1, set()).update(run)
    for lvl, x0, y0, x1, y1 in d.get("keep_clear", ()):
        clear.setdefault(lvl, set()).update(_cells([[x0 + dx, y0 + dy, x1 + dx, y1 + dy]], w, h))
    for lvl, plan in enumerate(new):
        if not d.get("rooms") or lvl not in described:
            plan.furniture = [f for f in plan.furniture if (f[1], f[2]) not in clear.get(lvl, ())]
            continue
        stairs = building.stairs[lvl] if lvl < len(building.stairs) else None
        plan.furniture = []
        # the windows count as openings while it is furnished: no switch or picture hangs on one
        own_edges = plan.open_edge
        plan.open_edge = set(own_edges) | set(plan.windows)
        L._furnish(plan, rng, stairs, None)
        plan.open_edge = own_edges
        keep_all = {k + 1 for k, rm in enumerate(plan.rooms) if rm.landmark_furnish}
        hollow = void.get(lvl, set()) - keep.get(lvl, set())
        plan.furniture = [f for f in plan.furniture
                          if (plan.grid[f[2]][f[1]] in keep_all or f[0] == L.SWITCH)
                          and (f[1], f[2]) not in hollow and (f[1], f[2]) not in clear.get(lvl, ())
                          and (f[1], f[2], f[3]) not in set(plan.windows)]

    # ---- painted pieces ----
    painted: dict[int, dict[str, dict]] = {}

    def paint(lvl, x, y, tile):
        if not (0 <= x <= w and 0 <= y <= h):
            return
        layers = painted.setdefault(lvl, {})
        for name in (ROOF_LAYERS if lvl >= n else LAYERS):
            if (x, y) not in layers.setdefault(name, {}):
                layers[name][(x, y)] = tile
                return

    for lvl, name, x, y, turn in list(d.get("pieces", ())) + list(d.get("furniture", ())):
        rec = items.get(name)
        if not rec:
            continue
        orient = turn if rec["kind"] == "edge" else OPPOSITE[turn]
        for k, orients in rec["tiles"].items():
            for key, tile in (orients.get(orient) or {}).items():
                ox, oy = map(int, key.split(","))
                paint(lvl + int(k), x + dx + ox, y + dy + oy, tile)
    for lvl, tile, x, y in d.get("vanilla", ()):
        paint(lvl, x + dx, y + dy, tile)
    building.painted = painted

    # ---- walls ----
    walls = d.get("walls") or {}
    if walls.get("exterior"):
        kit = walls["exterior"]
        style = _rekit(style, kit)
        # a style whose walls are vanilla (uk_stone: walls_exterior_house_01) has no kit tiles for
        # _rekit to swap: the kit is its outside wall outright (Parliament's ashlar was grey stone)
        if "walls_house_uk_01_" not in str((style.get("exterior") or {}).get("tiles")):
            from .regional_uk import wall_entry
            k = f"walls_house_uk_01_{kit}"
            style["exterior"] = wall_entry(f"{k}_0", f"{k}_16")
    tower = walls.get("tower")
    if tower and roof.get("accent"):
        k = f"walls_house_uk_01_{tower['kit']}"
        from .regional_uk import wall_entry
        roof["accent"] = dict(roof["accent"], entry=wall_entry(f"{k}_0", f"{k}_16"),
                              from_level=tower.get("from_level", 0))
    # parts in walls of their own (a timber-framed rear wing): [{kit, rects, levels}]
    from .regional_uk import wall_entry
    parts = []
    for part in walls.get("parts", ()):
        k = f"walls_house_uk_01_{part['kit']}"
        cells = set(_cells([[a + dx, b + dy, c + dx, e + dy] for a, b, c, e in part["rects"]], w, h))
        parts.append({"cells": cells, "entry": wall_entry(f"{k}_0", f"{k}_16"),
                      **({"levels": part["levels"]} if part.get("levels") is not None else {})})
    if parts:
        roof["accents"] = parts
    if walls.get("interior"):
        from . import catalog as C
        inside = next((e for e in C.INTERIOR_WALLS if e["tiles"].get("West") == walls["interior"]), None)
        if inside:
            style["interior"] = inside
            style["one_interior"] = True
    # every door its own (the description's: an oak door on a church), not only a front one
    if d.get("door"):
        from .regional_uk import front_door
        style["front_door"] = front_door(d["entry"], d["door"])
        style["doors_all_front"] = True
    # its own parapet, pinnacles and bands: only the plinth of the listed trims
    if style.get("wall_trim"):
        style["wall_trim"] = {k: v for k, v in style["wall_trim"].items() if k == "plinth"}
    return building, roof, style


# The window kit's own windows (window_kit.py WINDOWS: a W and N pair each), for a
# landmark's ordinary windows: "uk:sash_georgian".
KIT_WINDOWS = ("sash_georgian", "sash_victorian", "casement", "casement_leaded", "mullion", "sash_arched")


def kit_window(name: str) -> dict | None:
    if not name.startswith("uk:") or name[3:] not in KIT_WINDOWS:
        return None
    k = KIT_WINDOWS.index(name[3:])
    return {"kind": "window", "squares": [{"W": f"fixtures_windows_uk_01_{2 * k:03d}",
                                           "N": f"fixtures_windows_uk_01_{2 * k + 1:03d}"}]}


def _rekit(style: dict, kit: str) -> dict:
    """The style's wall-kit tiles (walls_house_uk_01_<material>_N) from another kit."""
    import json
    text = json.dumps({k: v for k, v in style.items() if k != "landmark"})
    text = re.sub(r"walls_house_uk_01_[a-z_]+?_(\d+)\"", lambda m: f"walls_house_uk_01_{kit}_{m.group(1)}\"", text)
    out = json.loads(text)
    out["landmark"] = style["landmark"]
    return out


def _fill(grid, here):
    """Squares Build covers that no room claimed: to the nearest room."""
    q = deque((x, y) for x, y in here if grid[y][x])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in here and not grid[ny][nx]:
                grid[ny][nx] = grid[y][x]
                q.append((nx, ny))


def _open_edges(grid, walled=frozenset()) -> set:
    """Every edge between two rooms, except a walled room's: (x, y, "W") a square's
    west edge, (x, y, "N") its north."""
    out = set()
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if not v:
                continue
            if x > 0 and row[x - 1] and row[x - 1] != v and not {v, row[x - 1]} & walled:
                out.add((x, y, "W"))
            if y > 0 and grid[y - 1][x] and grid[y - 1][x] != v and not {v, grid[y - 1][x]} & walled:
                out.add((x, y, "N"))
    return out
