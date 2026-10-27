"""Serialise a Plan to BuildingEd's .tbx format.

Schema taken from BuildingWriter in timbaker/buildinged. Two index conventions
differ and are easy to get backwards:

    tile entries   1-based, and 0 means "none"   (entryIndex)
    furniture      0-based, no null form         (furnitureIndex)

Element order inside <building> follows the writer: tile entries, furniture,
user_tiles, used_tiles, used_furniture, the <room> list, then <floor>.
"""
from __future__ import annotations

import random
import zlib
from xml.sax.saxutils import escape, quoteattr

from . import catalog as C
from .layout import ROOM_STYLE, Building, Plan, _erika_ready, roof_rects


def _stable_seed(*parts) -> int:
    """A seed that is the same in every process. hash() of a string is not:
    Python randomises it per process, so two builds of one map picked
    different floors and trims, and the build could not be repeated."""
    return zlib.crc32(repr(parts).encode("utf-8"))


# Version 4 is the first that carries a per-room Ceiling tile. Writing 3 still
# loads - the reader accepts 1..7 - but then it silently back-fills ceilings
# itself, so we may as well state them.
VERSION = 4


def _attrs(pairs: list[tuple[str, object]]) -> str:
    # Nearly every value is a coordinate or an index. Quoting those through
    # quoteattr, nine million calls for a town, was 40% of the time spent
    # writing buildings; numbers never need escaping.
    return "".join(f' {k}="{v}"' if type(v) is int else f" {k}={quoteattr(str(v))}"
                   for k, v in pairs)


# A pitched roof needs room for two slopes; a narrower strip of a house (a
# porch, one step of a turned footprint) keeps a flat roof.
PEAK_MIN_TILES = 4
# A room on an upper storey with this share of its tiles over a hall's open
# space has no floor (knoxbuild/regional_uk.py, "void").
VOID_SHARE = 0.6
# The 30-degree roofs the game's own houses wear, which BuildingEd sizes to an
# odd number of tiles across, 3 to 11; a house up to two tiles wider takes an
# 11 and a flat strip. Wider than that, the steep 45-degree gable with a flat
# top in the middle is all the editor offers.
PEAK30_MAX_ACROSS = 11
HIP_MAX_ACROSS = 7
_PEAK_DEPTHS = {1: "Point5", 2: "One", 3: "OnePoint5", 4: "Two", 5: "TwoPoint5"}
_NO_CAPS = {"cappedW": False, "cappedN": False, "cappedE": False, "cappedS": False}


def _peak_depth(across: int) -> str:
    """BuildingEd's depth for a 45-degree peaked roof this many tiles across."""
    return _PEAK_DEPTHS.get(across, "Three")


# A one-storey part of a two-storey house is pitched too (it was always flat); only a strip
# narrower than NARROW_PEAK_MIN stays flat. Set PITCH_LOWER_ROOFS = False for the old roofs.
PITCH_LOWER_ROOFS = True
NARROW_PEAK_MIN = 2


def _roof_pieces(rects, peaked: bool, roof30: bool = True):
    """(RoofType, Depth, caps, (x, y, w, h)) for each roof over a footprint.

    Flat roofs are never capped: at depth three a cap is a storey-high wall of
    roof tiles laid over the top storey's own walls. A pitched roof is capped
    at its gable ends where they are the outside of the house, and left open
    where it runs into the next roof.

    30-degree roofs need an odd width across their slopes. On an even one the
    roof covers one tile less and a flat strip takes the last row, on the side
    away from the camera (north or west) where it shows least. A house that is
    a single rectangle gets a hip roof about half the time, as Knox County's
    do; an L-shape keeps gables, whose ends meet cleanly.
    """
    out = []
    single = len(rects) == 1
    for rx, ry, rw, rh, caps in rects:
        across = min(rw, rh)
        if not peaked or across < NARROW_PEAK_MIN:
            out.append(("FlatTop", "Three", dict(_NO_CAPS), (rx, ry, rw, rh)))
            continue
        along_x = rw >= rh
        # a narrow wing (2-3 across) takes a low 45-degree roof rather than a flat one
        if not roof30 or across > PEAK30_MAX_ACROSS + 2 or across < PEAK_MIN_TILES:
            cap = dict(_NO_CAPS)
            if along_x:
                cap["cappedW"], cap["cappedE"] = caps["cappedW"], caps["cappedE"]
                out.append(("PeakWE", _peak_depth(rh), cap, (rx, ry, rw, rh)))
            else:
                cap["cappedN"], cap["cappedS"] = caps["cappedN"], caps["cappedS"]
                out.append(("PeakNS", _peak_depth(rw), cap, (rx, ry, rw, rh)))
            continue
        odd = across if across % 2 else across - 1
        # Hips only on small houses: across a wide one the four slopes meet
        # in a tall point that looks like a hat.
        hip = single and odd <= HIP_MAX_ACROSS and (rx * 31 + ry * 17 + rw * 7 + rh) % 2 == 0
        # ...and only with an odd long side too: WorldEd lays an even one on
        # the next odd square past the roof, which the lot compiler's roof
        # builder (buildroofs.py) does not (KnoxMap UK, 2026-10-13: 22 hips,
        # 956 squares different between the two compilers).
        hip = hip and (rw if along_x else rh) % 2 == 1
        cap = dict(_NO_CAPS)
        if along_x:
            box = (rx, ry + (rh - odd), rw, odd)
            if rh != odd:
                out.append(("FlatTop", "Three", dict(_NO_CAPS), (rx, ry, rw, rh - odd)))
            if not hip:
                cap["cappedW"], cap["cappedE"] = caps["cappedW"], caps["cappedE"]
            out.append(("Peak30Quad" if hip else "Peak30WE", "Zero", cap, box))
        else:
            box = (rx + (rw - odd), ry, odd, rh)
            if rw != odd:
                out.append(("FlatTop", "Three", dict(_NO_CAPS), (rx, ry, rw - odd, rh)))
            if not hip:
                cap["cappedN"], cap["cappedS"] = caps["cappedN"], caps["cappedS"]
            out.append(("Peak30Quad" if hip else "Peak30NS", "Zero", cap, box))
    return out


# Flat roofs carry plant: bare tar from edge to edge read as unfinished. One
# air-conditioning unit per ROOF_AC_EVERY tiles of roof, a vent per
# ROOF_VENT_EVERY, and a hatch, kept off the edges and off each other.
ROOF_MIN_TILES = 60
ROOF_AC_EVERY = 90
ROOF_VENT_EVERY = 60


def _rooftop(grid: list[list[int]], width: int, height: int,
             seed: str | int = 0) -> list[tuple[str, int, int, str]]:
    import random

    tiles = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1)
             if all(grid[y + dy][x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))]
    area = sum(1 for row in grid for v in row if v)
    if area < ROOF_MIN_TILES or not tiles:
        return []
    rng = random.Random(f"{seed}:{width}:{height}:{area}")
    wanted = (["roof_hatch"] + ["roof_ac"] * max(1, area // ROOF_AC_EVERY)
              + ["roof_vent"] * (area // ROOF_VENT_EVERY))
    taken: set[tuple[int, int]] = set()
    out = []
    rng.shuffle(tiles)
    for role in wanted:
        for x, y in tiles:
            if any((x + dx, y + dy) in taken for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            taken.add((x, y))
            out.append((role, x, y, rng.choice(("W", "N"))))
            break
    return out


ENTRANCE = 8      # KnoxMap UK: a superstore's glazed entrance bay, squares


LOBBY_DEPTH = 2          # KnoxMap UK: a superstore's entrance lobby stands this many squares proud, at least
LOBBY_LONG = (4, 30)     # ...and is this many squares along its front


def _lobby(grid, side: str):
    """A superstore's entrance lobby, read from its footprint (user, 2026-10-20: the big
    glazed entrance that stands out of the front): on the front `side` (S or E, the faces
    the camera sees), the main wall line is where most of the front stands; a stretch of
    LOBBY_LONG squares standing LOBBY_DEPTH or more proud of it, all at one depth, is the
    lobby, where the main line runs on either side of it (not an L-shaped store's wing).
    Returns its glazed runs - its front, then the returns where they are outside walls -
    and the entrance door in the middle of its front (the door step lays its pair beside
    it); None if there is none."""
    from collections import Counter
    h, w = len(grid), len(grid[0]) if grid else 0

    def cell(x, y):
        return 0 <= x < w and 0 <= y < h and bool(grid[y][x])
    if side == "S":
        outer = {x: max(y for y in range(h) if grid[y][x]) for x in range(w) if any(grid[y][x] for y in range(h))}
    else:
        outer = {y: max(x for x in range(w) if grid[y][x]) for y in range(h) if any(grid[y])}
    if not outer:
        return None
    main = Counter(outer.values()).most_common(1)[0][0]
    best, run = None, []
    keys = sorted(outer)
    for k in keys + [None]:
        same = run and k is not None and k == run[-1] + 1 and outer[k] == outer[run[-1]]
        if run and not same:
            depth = outer[run[-1]] - main
            flanked = outer.get(run[0] - 1) == main and outer.get(run[-1] + 1) == main
            if depth >= LOBBY_DEPTH and LOBBY_LONG[0] <= len(run) <= LOBBY_LONG[1] and flanked:
                score = len(run) * depth
                if best is None or score > best[0]:
                    best = (score, run[0], run[-1], outer[run[-1]])
            run = []
        if k is not None:
            run.append(k)
    if best is None:
        return None
    _s, a, b, far = best
    length = b - a + 1
    mid = a + (length - 1) // 2
    if side == "S":
        front = ("N", far + 1, a, length)
        returns = [("W", a, main + 1, far - main), ("W", b + 1, main + 1, far - main)]
        returns = [r for r, outside in zip(returns, (a - 1, b + 1))
                   if all(not cell(outside, y) for y in range(main + 1, far + 1))]
        doors = [(mid, far + 1, "N")]                   # its pair: the door step's partner
    else:
        front = ("W", far + 1, a, length)
        returns = [("N", a, main + 1, far - main), ("N", b + 1, main + 1, far - main)]
        returns = [r for r, outside in zip(returns, (a - 1, b + 1))
                   if all(not cell(x, outside) for x in range(main + 1, far + 1))]
        doors = [(far + 1, mid, "W")]
    return [front] + returns, doors


def _run_edges(run) -> set:
    """A run's (d, fixed, start, length) edges as (x, y, d)."""
    d, fixed, start, length = run
    return {(fixed, t, d) if d == "W" else (t, fixed, d) for t in range(start, start + length)}


def _window_bays(windows, taken: set) -> list:
    """KnoxMap UK: a storey's windows as shop-window runs (d, fixed, start, length), each
    unbroken line of two or more along one wall, clear of the squares already glazed."""
    lines: dict = {}
    for x, y, d in windows:
        if (x, y, d) not in taken:
            lines.setdefault((d, x if d == "W" else y), []).append(y if d == "W" else x)
    bays = []
    for (d, fixed), along in sorted(lines.items()):
        along = sorted(set(along))
        start = prev = along[0]
        for t in along[1:] + [None]:
            if t is not None and t == prev + 1:
                prev = t
                continue
            if prev - start + 1 >= 2:
                bays.append((d, fixed, start, prev - start + 1))
            if t is not None:
                start = prev = t
    return bays


def _entrance(run, doors) -> tuple[str, int, int, int]:
    """A superstore's glazed bay: ENTRANCE squares of a shop front run, centred on its
    shop door where it has one (else on the run's middle), kept inside the run."""
    d, fixed, start, length = run
    if length <= ENTRANCE:
        return run
    at = [y if d == "W" else x for x, y, dd in doors
          if dd == d and (x if d == "W" else y) == fixed and start <= (y if d == "W" else x) < start + length]
    mid = at[len(at) // 2] if at else start + length // 2
    first = max(start, min(start + length - ENTRANCE, mid - ENTRANCE // 2))
    return (d, fixed, first, ENTRANCE)


def _storefront_runs(storey, doors) -> list[tuple[str, int, int, int]]:
    """The ground floor's shop glass as straight runs of wall: (edge dir,
    fixed coordinate, first tile along, length). A run spans from its first
    glazed tile to its last, taking in the doors and the bits of wall beside
    them, and stops wherever the line is no longer an outside wall."""
    grid = storey.grid
    h, w = len(grid), len(grid[0])

    def cell(x, y):
        return grid[y][x] if 0 <= x < w and 0 <= y < h else 0

    def outside(d, fixed, t):
        x, y = (fixed, t) if d == "W" else (t, fixed)
        a, b = (cell(x - 1, y), cell(x, y)) if d == "W" else (cell(x, y - 1), cell(x, y))
        return bool(a) != bool(b)

    lines: dict[tuple[str, int], list[int]] = {}
    for x, y, d in storey.shop_front:
        lines.setdefault((d, x if d == "W" else y), []).append(y if d == "W" else x)
    door_at = {(d, x if d == "W" else y, y if d == "W" else x) for x, y, d in doors}
    runs = []
    for (d, fixed), along in sorted(lines.items()):
        lo, hi = min(along), max(along)
        # A shop door just past the last pane belongs to the shop front too.
        for step in (-1, 1):
            end = lo if step < 0 else hi
            for k in (1, 2):
                if (d, fixed, end + k * step) in door_at and all(
                        outside(d, fixed, end + i * step) for i in range(1, k + 1)):
                    end += k * step
                    break
            lo, hi = (end, hi) if step < 0 else (lo, end)
        start = None
        for t in range(lo, hi + 2):
            if t <= hi and outside(d, fixed, t):
                start = t if start is None else start
            elif start is not None:
                runs.append((d, fixed, start, t - start))
                start = None
    return runs


def _outside(grid, x, y, back) -> bool:
    """Whether a door at (x, y) with `back` the square behind it has a room on one side only."""
    def room(sx, sy):
        return 0 <= sy < len(grid) and 0 <= sx < len(grid[0]) and bool(grid[sy][sx])
    return room(x, y) != room(*back)


def _faces(grid, x, y, d) -> str:
    """The side of the building an outside door opens on (N, S, W or E)."""
    def room(sx, sy):
        return 0 <= sy < len(grid) and 0 <= sx < len(grid[0]) and bool(grid[sy][sx])
    if d == "N":
        return "N" if room(x, y) else "S"
    return "W" if room(x, y) else "E"


def _add(entries: list[dict], entry: dict | None) -> int:
    """The 1-based index of `entry` in the tile-entry table, appending it if it
    is not there yet; 0, BuildingEd's "none", for no entry."""
    if not entry:
        return 0
    C._complete_window_cutouts(entry)
    if entry in entries:
        return entries.index(entry) + 1
    entries.append(entry)
    return len(entries)


# How many room kinds in a building carry wall trim. Knox County's schools
# How many different interior wall sets one building uses. Knox County's
# figure, measured over 77 buildings in Muldraugh, is 4.48.
#
# A mall is the exception: every unit in it is let to somebody else and
# painted to suit, and the game's own runs to 117 different wall tiles in the
# one building where an ordinary house has a dozen. It gets one per room kind
# instead, out of the whole list.
INTERIOR_WALLS_PER_BUILDING = 4
# One per storey for a mall concourse, so the ground floor seen through the
# atrium is plainly a different floor from the gallery you are standing on.
CONCOURSE_FLOORS = ("tile_cream", "tile_grey", "civic_pale", "tile_white")
WALLS_PER_KIND = {"mall": 99}
# run to 0.25 pieces of walls_interior_detailing per 10 m2 and ours to 2.43
# with every room trimmed, so about one kind in ten keeps it.
TRIM_SHARE = 0.12


def _roof_object(roof_type, depth, cap, rx, ry, rw, rh, cap_idx, slope_idx, top_idx) -> str:
    attrs = [
        ("type", "roof"), ("width", rw), ("height", rh),
        ("RoofType", roof_type), ("Depth", depth),
        ("cappedW", str(cap["cappedW"]).lower()), ("cappedN", str(cap["cappedN"]).lower()),
        ("cappedE", str(cap["cappedE"]).lower()), ("cappedS", str(cap["cappedS"]).lower()),
        ("CapTiles", cap_idx), ("SlopeTiles", slope_idx), ("TopTiles", top_idx),
        ("x", rx), ("y", ry),
    ]
    return f"  <object{_attrs(attrs)}/>"


def _settle_planned(planned: dict, storeys: list, width: int, height: int) -> dict:
    """KnoxMap UK: a planned roof piece over a storey the layout took out after
    the plan was made (a part walled off by a lift shaft: layout's
    _cut_off_by_the_shaft, upstream 1.7.6) comes down onto the storey that is
    there - a pitched one to the floor above the top storey with rooms under
    it, a flat one onto that storey. Never moved up; dormers stay."""
    def top_under(rx, ry, rw, rh):
        for lv in range(len(storeys) - 1, -1, -1):
            grid = storeys[lv].grid
            if any(grid[y][x] for y in range(max(0, ry), min(height, ry + rh))
                   for x in range(max(0, rx), min(width, rx + rw))):
                return lv
        return None
    out: dict = {}
    for lv, pieces in planned.items():
        for piece in pieces:
            roof_type, depth, _cap, rect = piece
            to = lv
            if not roof_type.startswith("Dormer"):
                top = top_under(*rect)
                if top is not None:
                    # A flat roof lies on its storey; a parapet (flat, Point5)
                    # and a pitched roof on the floor above.
                    want = top if roof_type == "FlatTop" and depth != "Point5" else top + 1
                    to = min(lv, want)
            out.setdefault(to, []).append(piece)
    return out


def render_tbx(plan: Plan | Building, name: str,
               style: dict | None = None, roof_plan: dict | None = None) -> str:
    """Return the complete .tbx document for a plan or a stack of them.

    `style` supplies this building's own exterior and interior wall materials,
    and optionally a floor that overrides the usual per-room palette. Each .tbx
    carries its own tile-entry table, so styles cost nothing globally: the
    style's entries are simply appended after the shared ones and referenced by
    their new indices.
    """
    building = plan if isinstance(plan, Building) else Building(
        width=plan.width, height=plan.height, storeys=[plan], profile=plan.profile)
    storeys = building.storeys
    # A roof planned part by part, used only if the storeys really step the
    # way it was planned for (the layout falls back to one height when the
    # tallest part has no room for stairs).
    planned = (roof_plan or {}).get("pieces") if getattr(building, "stepped", False) else None
    if planned is not None:
        planned = _settle_planned(planned, storeys, building.width, building.height)

    entries = list(C.TILE_ENTRIES)
    exterior_idx = C.EXTERIOR_WALL
    interior_idx = C.INTERIOR_WALL
    floor_override = None
    window_idx = C.WINDOW
    roof_cap_idx = C.ROOF_CAP
    curtains_idx = C.CURTAINS
    front_idx = front_curtains = None
    slope_idx, top_idx, peaked = C.ROOF_SLOPE, C.ROOF_TOP, False
    trim_idx = shutters_idx = grime_idx = 0
    roof30 = False

    if style:
        C._complete_window_cutouts(style["exterior"])
        C._complete_window_cutouts(style["interior"])
        entries.append(style["exterior"])
        exterior_idx = len(entries)
        entries.append(style["interior"])
        interior_idx = len(entries)
        if style.get("floor"):
            entries.append(style["floor"])
            floor_override = len(entries)
        if style.get("window"):
            entries.append(style["window"])
            window_idx = len(entries)
        curtains_idx = _add(entries, style.get("curtains")) if "curtains" in style else C.CURTAINS
        # Taller buildings of a kind take bigger windows: the last row whose
        # storey count this building reaches.
        for levels, entry, curtains in style.get("windows_by_levels") or ():
            if len(storeys) >= levels and levels > 1:
                window_idx = _add(entries, entry)
                curtains_idx = _add(entries, curtains)
        trim_idx = _add(entries, style.get("trim"))
        shutters_idx = _add(entries, style.get("shutters"))
        grime_idx = _add(entries, style.get("grime"))
        if style.get("roof"):
            roof = style["roof"]
            if roof.get("slopes"):
                slope_idx = _add(entries, roof["slopes"])
            top_idx = _add(entries, roof["tops"])
            peaked = bool(roof.get("peaked"))
            # 30-degree roofs only where the gable ends have 30-degree tiles.
            roof30 = "CapPeak30S1" in ((roof.get("caps") or {}).get("tiles") or {})                 and "Slope30S1" in ((roof.get("slopes") or {}).get("tiles") or {})
        if style.get("shop_front"):
            entry, curtains = style["shop_front"]
            front_idx = _add(entries, entry)
            front_curtains = _add(entries, curtains)
    # With Erika's Tiles, a shop front is a wall of glass in a painted frame,
    # not windows set in brick, and a sign hangs over it.
    runs: list[tuple[str, int, int, int]] = []
    glazed: set[tuple[int, int, str]] = set()
    user_tiles: dict[int, dict[tuple[int, int], str]] = {}
    store_ext = store_int = store_door = 0
    # KnoxMap UK: a home's own front door (regional_uk.front_door) in its one
    # front door (front_door_at, below); back doors and inside doors stay vanilla's.
    home_door = _add(entries, (style or {}).get("front_door"))
    # A part in a wall of its own (a church's brick tower): every outside edge
    # of it, on every storey, as a one-tile wall in that wall.
    accent = (roof_plan or {}).get("accent") if planned is not None else None
    # KnoxMap UK: more parts in walls of their own (a landmark's timber-framed rear
    # wings, landmark_uk.py `walls.parts`), each as the tower's accent.
    accents = ([accent] if accent else []) + list((roof_plan or {}).get("accents") or [])
    # KnoxMap UK: a house's upper floors in a wall of their own (regional_uk
    # `upper`; with no roof plan, from its style: the ridge along its longer side).
    up = (roof_plan or {}).get("upper")
    house_upper = (style or {}).get("house_upper")
    if not up and house_upper and len(storeys) > 1:
        from . import regional_uk
        cells = {(x, y) for y, row in enumerate(storeys[0].grid) for x, v in enumerate(row) if v}
        ridge = "WE" if building.width >= building.height else "NS"
        up = regional_uk.upper_accents(house_upper["entry"], cells,
                                       [(cells, ridge)] if house_upper.get("sides") == "long" else None)
    if up:
        accents.extend(up if isinstance(up, list) else [up])
    accent_ids = [_add(entries, a["entry"]) for a in accents]
    # KnoxMap UK: a British shop front (regional_uk `uk_storefront`, asset_creator's
    # commercial_kit): the run square by square - a pilaster at its first and last
    # square, panes between, the door frame where the shop door is - and its fascia
    # (regional_uk.fascia_tiles) on the wall furniture of every run the street sees.
    uk_front = (style or {}).get("uk_storefront")
    uk_ids: dict = {}
    if uk_front and storeys and storeys[0].shop_front:
        from . import regional_uk
        uk_ids = {k: _add(entries, uk_front[k]) for k in ("start", "pane", "end", "start_in", "pane_in", "end_in")}
        store_door = _add(entries, uk_front["door"])
        runs = [r for r in _storefront_runs(storeys[0], storeys[0].doors) if r[3] >= 2]
        grid = storeys[0].grid

        def seen(d, fixed, start):
            return ((d == "N" and (fixed >= len(grid) or not grid[fixed][start]))
                    or (d == "W" and (fixed >= len(grid[0]) or not grid[start][fixed])))
        fascia = (style or {}).get("uk_fascia")
        # The sign on the longest run the street sees; the others the board alone
        # (a front that steps back read as two shops of the same name).
        signed = max((r for r in runs if seen(*r[:3])), key=lambda r: r[3], default=None)
        signs_on = list(runs)
        if (style or {}).get("superstore"):
            # A superstore is brick with one glazed entrance bay (review 2026-10-20: 40
            # squares of glass read as a car showroom): the glass ENTRANCE squares round
            # the shop door, the sign still over the whole front.
            # One entrance: the lobby that stands out of the front where the footprint has
            # one (_lobby: glass round its three sides, the pair of doors in its front, the
            # sign over it); else the signed run's bay; the store's other front runs brick.
            side = None if not signed else ("S" if signed[0] == "N" else "E")
            lobby = _lobby(grid, side) if side else None
            if lobby:
                runs, lobby_doors = lobby
                signed = runs[0]
                signs_on = list(runs)
                for dd in lobby_doors:
                    if dd not in storeys[0].doors:
                        storeys[0].doors.append(dd)
            else:
                signs_on = [signed] if signed else []
                runs = [_entrance(signed, storeys[0].doors)] if signed else []
            # KnoxMap UK (2026-10-26): the shop windows superstore_uk set along the front, each
            # run of them a framed bay of its own between brick piers; no fascia over them
            bays = _window_bays(storeys[0].windows, glazed | {e for r in runs for e in _run_edges(r)})
            runs = runs + bays
            signs_on = signs_on + [None] * len(bays)
        for (d, fixed, start, length), full in zip(runs, signs_on):
            glazed.update((fixed, t, d) if d == "W" else (t, fixed, d)
                          for t in range(start, start + length))
            if not seen(d, fixed, start) or full is None:
                continue
            d, fixed, start, length = full                  # the sign along the whole front
            level = 0
            if full == signed:
                board = regional_uk.fascia_tiles(fascia, length)
                level = regional_uk.fascia_level(fascia, length)
            else:
                board = ([(fascia["plain"]["W"], fascia["plain"]["N"])] * length
                         if fascia and fascia.get("plain") else None)
            for j, (west, north) in enumerate(board or ()):
                if not (west if d == "W" else north):
                    continue
                t = start + j
                user_tiles.setdefault(level, {})[(fixed, t) if d == "W" else (t, fixed)] = west if d == "W" else north
    elif front_idx and C.ERIKA_STOREFRONTS and storeys[0].shop_front and _erika_ready():
        rng = random.Random(name)
        # KnoxMap UK: a colour the building's data chose (regional_uk.py
        # `storefront`, the colour's sheet); any of its frames.
        sheet = (style or {}).get("storefront")
        pool = [s for s in C.ERIKA_STOREFRONTS
                if sheet and s[0]["tiles"]["West"].startswith(sheet + "_")]
        ext, inte, door = rng.choice(pool or C.ERIKA_STOREFRONTS)
        store_ext, store_int, store_door = _add(entries, ext), _add(entries, inte), _add(entries, door)
        # A one-tile run is a step in a slanted wall, not a shop window.
        runs = [r for r in _storefront_runs(storeys[0], storeys[0].doors) if r[3] >= 2]
        for d, fixed, start, length in runs:
            glazed.update((fixed, t, d) if d == "W" else (t, fixed, d)
                          for t in range(start, start + length))
        # The sign goes on the longest run the street can see, one storey up
        # so it hangs above the glass. Only the south and east faces are
        # seen: a sign on a north or west wall is drawn on its inner side.
        grid = storeys[0].grid
        seen = [r for r in runs
                if (r[0] == "N" and (r[1] >= len(grid) or not grid[r[1]][r[2]]))
                or (r[0] == "W" and (r[1] >= len(grid[0]) or not grid[r[2]][r[1]]))]
        if seen and (style or {}).get("shop_signs", True):
            d, fixed, start, length = max(seen, key=lambda r: r[3])
            fits = [s for s in C.ERIKA_SIGNS.get(d, ()) if len(s) <= length]
            if fits:
                sign = rng.choice(fits)
                first = start + (length - len(sign)) // 2
                user_tiles[1] = {((fixed, first + j) if d == "W" else (first + j, fixed)): tile
                                 for j, tile in enumerate(sign)}
    # KnoxMap UK: a repair garage's MOT board and fascia on the wall beside its shutter
    # (regional_uk._lay_signs `sign_at`; a north wall's tile in its run's square, as a fascia's)
    for x, y, d, tile in (((style or {}).get("garage") or {}).get("sign_at") or ()):
        user_tiles.setdefault(0, {}).setdefault((x, y), tile)
    # The one front door: the ground floor's outside door on the street's side
    # (build.py's street_side), else its first outside door (the layout's front).
    front_door_at = None
    if home_door and storeys:
        outside = [(x, y, d) for x, y, d in storeys[0].doors if (x, y, d) not in glazed
                   and _outside(storeys[0].grid, x, y, (x - 1, y) if d == "W" else (x, y - 1))]
        street = (style or {}).get("street_side")
        facing = [door for door in outside if _faces(storeys[0].grid, *door) == street]
        front_door_at = (facing or outside or [None])[0]
    # KnoxMap UK: a doorcase, hood or porch round it (regional_uk `door_surround`,
    # a listed building's from its listing): on the door's own square, where
    # that square is outside the house - the wall's outside then faces east
    # or south, the faces the camera sees (on a west or north front the
    # outside is the hidden side of the wall).
    surround = (style or {}).get("door_surround")
    if surround and front_door_at:
        x, y, d = front_door_at
        grid0 = storeys[0].grid
        inside = 0 <= y < len(grid0) and 0 <= x < len(grid0[0]) and grid0[y][x]
        if not inside:
            user_tiles.setdefault(0, {})[(x, y)] = surround[0 if d == "W" else 1]
    # KnoxMap UK: bays in front of the street front's ground-floor windows
    # (regional_uk `bay`), laid as the door surround is, on a layer of their
    # own (Curtains2). The house window behind a bay stays, to open and climb
    # through (QA, 2026-10-19: taken out, it left the front a solid wall),
    # drawn invisible: the lot compiler stacks a window over every tile on its
    # square, and the vanilla one stood in front of the bay.
    bays: dict[int, dict[tuple[int, int], str]] = {}
    bay = (roof_plan or {}).get("bay") or (style or {}).get("house_bay")
    if bay and storeys:
        _lay_bays(storeys, bay, (style or {}).get("street_side"), front_door_at, user_tiles, bays)
    # KnoxMap UK: a plinth, a first-floor band and an eaves cornice along the
    # outside walls the camera sees (regional_uk `wall_trim`: a listed
    # building's from its listing): one wall piece serves every storey, so
    # what belongs to one storey is laid over it, square by square - the
    # plinth on the ground floor, the band at the first floor's foot, the
    # cornice on each square's top storey, a parapet on the roof's floor
    # beside it. A square already holding a doorway or a sign keeps it.
    # KnoxMap UK: drains (regional_uk `drains`; a listed building's): a
    # downpipe at each end of every outside wall run the camera sees, from the
    # gully grate on the ground to the hopper under the eaves - on a layer of
    # its own (WallFurniture2), beside the trims on WallFurniture.
    drains: dict[int, dict[tuple[int, int], str]] = {}
    pipes = (style or {}).get("drains")
    if pipes and storeys:
        _lay_drains(storeys, building.width, building.height, pipes, drains, bays)
    trims = (style or {}).get("wall_trim") or {}
    if trims and storeys:
        _lay_trims(storeys, building.width, building.height, trims, user_tiles, bays)
    # A depth-three flat roof walls in its storey with the cap entry's
    # CapGap tiles, which BuildingTemplates.txt sets to stucco - every top
    # floor came out stucco whatever the building was made of. Give each
    # building a cap entry whose gaps are its own exterior wall.
    ext_tiles = entries[exterior_idx - 1]["tiles"]
    if ext_tiles.get("West") and ext_tiles.get("North"):
        base = ((style or {}).get("roof") or {}).get("caps") or entries[C.ROOF_CAP - 1]
        cap = dict(base["tiles"])
        cap["CapGapE3"], cap["CapGapS3"] = ext_tiles["West"], ext_tiles["North"]
        entries.append({"category": "roof_caps", "tiles": cap})
        roof_cap_idx = len(entries)

    rooftop = [] if peaked or planned is not None else _rooftop(
        storeys[-1].grid, building.width, building.height, name)
    # Which furniture roles this building actually uses, in first-use order.
    roles: list[str] = []
    for storey in storeys:
        for role, _x, _y, _o in storey.furniture:
            if role not in roles:
                roles.append(role)
    for role, _x, _y, _o in rooftop:
        if role not in roles:
            roles.append(role)

    out: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>']

    building_attrs = [
        ("version", VERSION),
        ("width", building.width),
        ("height", building.height),
        ("ExteriorWall", exterior_idx),
        ("ExteriorWallTrim", trim_idx),
        ("Door", C.DOOR),
        ("DoorFrame", C.DOOR_FRAME),
        ("Window", window_idx),
        ("Curtains", curtains_idx),
        ("Shutters", shutters_idx),
        ("Stairs", C.STAIRS),
        ("RoofCap", roof_cap_idx),
        ("RoofSlope", slope_idx),
        ("RoofTop", top_idx),
        ("GrimeWall", grime_idx),
    ]
    # A floor per room kind, drawn once for this building from the shares the
    # game lays them in (layout.FLOOR_CHOICES). One house keeps one floor per
    # kind so it reads as one house; the next house down the street draws
    # again. A style's own floor still overrides the lot. This has to happen
    # before the tile entries are written out below, and before used_tiles is
    # counted, or the new entries never reach the file.
    from .layout import FLOOR_CHOICES, KIND_FLOOR_CHOICES

    # What sort of building this is, for the rooms it floors differently: a
    # station's corridor is not somebody's hallway.
    of_kind = KIND_FLOOR_CHOICES.get(
        next((s.kind for s in storeys if s.kind), None) or "", {})
    # The style's floor is the building's default, not its whole story. It
    # used to override every room, which meant a school, a shop, a civic
    # building and a block of flats each had one floor throughout - a shop
    # tile down a school corridor, one carpet over a flat's bathroom and its
    # kitchen alike - and none of the per-room floors below ever applied to
    # them. Knox County gives a bathroom 18 different floors and a kitchen 35.
    # A room with a floor of its own takes it; the rest fall back to the
    # style's.
    # Which room kinds this building trims (see InteriorWallTrim below).
    trim_rng = random.Random(
        _stable_seed(name, building.width, building.height, 'trim'))
    trimmed = {k for k in sorted({r.kind for r in building.rooms})
               if trim_rng.random() < TRIM_SHARE}
    room_floor: dict[str, int] = {}
    pick_rng = random.Random(
        _stable_seed(name, len(building.rooms), building.width,
                     building.height))
    for kind in sorted({r.kind for r in building.rooms}):
        options = [o for o in of_kind.get(kind, FLOOR_CHOICES.get(kind, ()))
                   if o in C.EXTRA_FLOORS]
        if not options:
            continue
        if building.profile:
            choice = pick_rng.choices(
                options, weights=[building.profile.floor_weight(option)
                                  for option in options], k=1)[0]
        else:
            choice = pick_rng.choice(options)
        entries.append(C.floor_entry(choice))
        room_floor[kind] = len(entries)

    # A mall's concourse runs the same floor on every storey in Knox County,
    # which reads as one surface when two of them are in view at once and the
    # opening is all that tells them apart. Each storey gets its own here, so
    # a glance down the atrium shows which floor is which.
    storey_floor: dict[int, int] = {}
    if any(r.kind == "concourse" for r in building.rooms):
        for level in range(len(storeys)):
            name_ = CONCOURSE_FLOORS[level % len(CONCOURSE_FLOORS)]
            entries.append(C.floor_entry(name_))
            storey_floor[level] = len(entries)
    level_of = {id(r): lvl for lvl, st in enumerate(storeys) for r in st.rooms}

    # Interior walls, room by room. InteriorWall has always been a per-room
    # attribute here and every room was given the same one, so a building was
    # one colour throughout however many its style could have used. Knox
    # County paints 4.48 different sets into a single building - a kitchen is
    # not the colour of the bedroom next door - over 9 in all. A few are dealt
    # out per room kind rather than per room, so two bedrooms match and the
    # kitchen does not, the way the floors above already work.
    room_wall: dict[str, int] = {}
    palette = [interior_idx]
    spare = [e for e in C.INTERIOR_WALLS
             if not style or e != style.get("interior")]
    # KnoxMap UK (in game, 2026-10-19): not a shop's glass walls inside a home.
    if getattr(storeys[0], "kind", None) in (None, "house", "apartment", "shed", "barn"):
        spare = [e for e in spare
                 if not str((e.get("tiles") or {}).get("West", "")).startswith("walls_commercial")]
    pick_rng.shuffle(spare)
    kinds_here = sorted({r.kind for r in building.rooms})
    want = min(WALLS_PER_KIND.get(getattr(storeys[0], "kind", None) or "",
                                  INTERIOR_WALLS_PER_BUILDING),
               len(kinds_here))
    if (style or {}).get("one_interior"):
        want = 1                                # KnoxMap UK: a landmark's one plaster (landmark_uk.py)
    for entry in spare[:max(0, want - 1)]:
        entries.append(entry)
        palette.append(len(entries))
    for i, kind in enumerate(kinds_here):
        # A mall gives each trade its own; everywhere else they share a few.
        room_wall[kind] = (palette[i % len(palette)] if want >= len(kinds_here)
                           else pick_rng.choice(palette))

    # KnoxMap UK: a landmark's own floors, room by room, and its windows,
    # window by window (landmark_uk.py).
    own_floor = {id(r): _add(entries, {"category": "floors", "tiles": {"Floor": r.landmark_floor}})
                 for r in building.rooms if getattr(r, "landmark_floor", None)}
    own_window = {(lvl, at): _add(entries, entry) for lvl, st in enumerate(storeys)
                  for at, entry in (getattr(st, "window_entries", None) or {}).items()}

    # Every tile entry has to be in the table before the table is written:
    # one added later indexes past the end of it.
    rails = any(st.railing for st in storeys)
    rail_idx = _add(entries, C.RAILING) if rails else 0
    rail_ext_idx = _add(entries, C.RAILING_EXT) if rails else 0

    out.append(f"<building{_attrs(building_attrs)}>")

    for entry in entries:
        out.append(f' <tile_entry category={quoteattr(entry["category"])}>')
        for enum_name, tile in entry["tiles"].items():
            out.append(f'  <tile enum={quoteattr(enum_name)} tile={quoteattr(tile)}/>')
        out.append(" </tile_entry>")

    for role in roles:
        # BuildingWriter only writes a layer when it is not the default. Leaving
        # it off a wall piece drops it onto the floor layer, which is how light
        # switches, paintings and mirrors ended up standing mid-room.
        layer = C.FURNITURE_LAYERS.get(role, "Furniture")
        out.append(" <furniture>" if layer == "Furniture"
                   else f" <furniture layer={quoteattr(layer)}>")
        # All four facings, in the order BuildingWriter emits them. Writing only
        # W and N leaves the E/S slots empty, so any object facing that way
        # renders as nothing at all.
        for orient in ("W", "N", "E", "S"):
            tiles = C.FURNITURE[role].get(orient)
            if not tiles:
                continue
            out.append(f'  <entry orient="{orient}">')
            for key, tile in tiles.items():
                dx, dy = key.split(",")
                out.append(f'   <tile x="{dx}" y="{dy}" name={quoteattr(tile)}/>')
            out.append("  </entry>")
        out.append(" </furniture>")

    painted = getattr(building, "painted", None) or {}         # KnoxMap UK: landmark_uk.py's pieces
    names = sorted({t for layer in (user_tiles, drains, bays) for tiles in layer.values() for t in tiles.values()}
                   | {t for layers in painted.values() for tiles in layers.values() for t in tiles.values()})
    if names:
        out.append(" <user_tiles>")
        out.extend(f"  <tile tile={quoteattr(t)}/>" for t in names)
        out.append(" </user_tiles>")

    def user_tile_layer(level: int, which=None, layer="WallFurniture") -> str | None:
        tiles = (user_tiles if which is None else which).get(level)
        if not tiles:
            return None
        cols, rows = building.width + 1, building.height + 1
        text = ["\n"]
        for y in range(rows):
            row = [str(names.index(tiles[(x, y)]) + 1) if (x, y) in tiles else "0"
                   for x in range(cols)]
            text.append(",".join(row) + ("," if y < rows - 1 else "") + "\n")
        return f'  <tiles layer="{layer}">' + escape("".join(text)) + "</tiles>"

    used = " ".join(str(i) for i in range(1, len(entries) + 1))
    out.append(f" <used_tiles>{used}</used_tiles>")
    used_f = " ".join(str(i) for i in range(len(roles)))
    out.append(f" <used_furniture>{used_f}</used_furniture>")

    # A hall open to its roof (knoxbuild/regional_uk.py): rooms on the upper
    # storeys that lie over it get no floor - their walls stand, the inside
    # is one tall space - and the rooms under them no ceiling.
    voids = (roof_plan or {}).get("void") if planned is not None else None
    void_ids, open_ids = set(), set()
    if voids:
        for lvl, st in enumerate(storeys):
            cells_of: dict[int, list] = {}
            for y, row in enumerate(st.grid):
                for x, v in enumerate(row):
                    if v:
                        cells_of.setdefault(v, []).append((x, y))
            here, above = voids.get(lvl, set()), voids.get(lvl + 1, set())
            for v, cells in cells_of.items():
                room = st.rooms[v - 1]
                if here and sum(c in here for c in cells) >= VOID_SHARE * len(cells):
                    void_ids.add(id(room))
                if above and sum(c in above for c in cells) >= VOID_SHARE * len(cells):
                    open_ids.add(id(room))

    def open_tile(level, x, y):
        """A tile of an upper storey that is part of the hall's open space."""
        if not void_ids or (x, y) not in voids.get(level, ()):
            return False
        v = storeys[level].grid[y][x] if 0 <= y < building.height and 0 <= x < building.width else 0
        return bool(v) and id(storeys[level].rooms[v - 1]) in void_ids

    for room in building.rooms:
        floor_idx, _label, _ = ROOM_STYLE[room.kind]
        floor_idx = room_floor.get(room.kind, floor_idx)
        # The shipped templates set Name identical to InternalName; the room
        # name is what reaches the game's room definitions, so don't get
        # creative with it.
        # An open-plan flat's one room is ours; the game knows it as a living
        # room, and the name is what its loot tables key off (layout.ROOM_NAME).
        from .layout import ROOM_NAME
        game_name = ROOM_NAME.get(room.kind, room.kind)
        room_attrs = [
            ("Name", game_name),
            ("InternalName", game_name),
            ("Color", C.ROOM_COLORS[room.kind]),
            ("InteriorWall", room_wall.get(room.kind, interior_idx)),
            # Skirting and picture rail. Set on every room it came to 2.43
            # pieces per 10 m2 against Knox County's 0.25 - a band of
            # detailing round every wall of every room, which at a distance
            # reads as fittings hung all the way along. Only some rooms get
            # it, drawn once per building per kind so one room is not
            # trimmed and the one next door bare.
            ("InteriorWallTrim",
             C.INTERIOR_WALL_TRIM if room.kind in trimmed else 0),
            # A room with a floor of its own first, then the style's, then
            # the room kind's own default.
            ("Floor", 0 if id(room) in void_ids else (
             (storey_floor.get(level_of.get(id(room)), 0)
              if room.kind == "concourse" else 0)
             or own_floor.get(id(room)) or room_floor.get(room.kind) or floor_override or floor_idx)),
            ("GrimeFloor", 0),
            ("GrimeWall", 0),
            # A concourse has no ceiling: it is open to the storey above, and
            # BuildingEd draws a room's ceiling on the floor above it - which
            # laid a lid straight across the hole and hid the ground floor
            # the hole is cut to show. Nor has a hall's, open to its roof.
            ("Ceiling", 0 if room.kind == "concourse" or id(room) in void_ids
             or id(room) in open_ids else C.CEILING),
        ]
        out.append(f" <room{_attrs(room_attrs)}/>")

    attic: list[str] = []
    # pitched roofs over a lower storey's exposed part stand on the floor above that storey,
    # as the top storey's stand on the roof floor
    above_roofs: dict[int, list[str]] = {}
    for level, storey in enumerate(storeys):
        out.append(" <floor>")
        out.extend(above_roofs.pop(level, []))

        for x, y, direction in storey.doors:
            # No door out into a hall's open space: it would open onto a drop.
            back = (x - 1, y) if direction == "W" else (x, y - 1)
            if open_tile(level, x, y) or open_tile(level, *back):
                continue
            # In the shop glass: a glass door, framed by the glass wall itself.
            shop_door = level == 0 and (x, y, direction) in glazed
            tile = store_door if shop_door else C.DOOR
            if level == 0 and (x, y, direction) == front_door_at:
                tile = home_door
            elif home_door and not shop_door and (style or {}).get("doors_all_front"):
                tile = home_door                # KnoxMap UK: a landmark's doors are all its own (landmark_uk.py)
            attrs = [("type", "door"), ("FrameTile", 0 if shop_door else C.DOOR_FRAME),
                     ("x", x), ("y", y), ("dir", direction),
                     ("Tile", tile)]
            out.append(f"  <object{_attrs(attrs)}/>")
            if shop_door and (style or {}).get("superstore"):
                # KnoxMap UK: a superstore's entrance is a pair of doors (review 2026-10-20):
                # the next square of the glass along the wall gets the second
                nx, ny = (x, y + 1) if direction == "W" else (x + 1, y)
                if (nx, ny, direction) in glazed and (nx, ny, direction) not in storey.doors:
                    out.append("  <object" + _attrs([("type", "door"), ("FrameTile", 0), ("x", nx), ("y", ny),
                                                     ("dir", direction), ("Tile", tile)]) + "/>")

        # The railing round the hole in this floor, one run per edge. A wall
        # object overrides what the room adjacency would have drawn, which is
        # how the shop glass below works too.
        # Nothing at all where an escalator arrives: a wall object with no
        # tile on it, which is what stops the exterior wall coming back.
        for x, y, d in sorted(storey.open_edge):
            out.append("  <object" + _attrs([
                ("type", "wall"), ("length", 1), ("InteriorTile", 0),
                ("ExteriorTrim", 0), ("InteriorTrim", 0), ("x", x), ("y", y),
                ("dir", "N" if d == "W" else "W"), ("Tile", 0)]) + "/>")
        # A party wall on this building's south or east side is drawn on the square
        # beyond it - the next house's - over that house's own north or west wall there,
        # which faces into its rooms and carries its wallpaper; ours, the outside brick,
        # won (user, 2026-10-20: in a terrace each party wall was the neighbour's
        # brick from inside). So none of ours there: the neighbour's wall is the
        # one wall on that edge, as far up as the neighbour stands (`party`).
        pgrid = storey.grid

        def ours(gx, gy):
            return 0 <= gy < len(pgrid) and 0 <= gx < len(pgrid[0]) and bool(pgrid[gy][gx])
        for x, y, d in sorted(getattr(storey, "party", ())):
            if (ours(x, y - 1) if d == "N" else ours(x - 1, y)) and not ours(x, y):
                out.append("  <object" + _attrs([
                    ("type", "wall"), ("length", 1), ("InteriorTile", 0),
                    ("ExteriorTrim", 0), ("InteriorTrim", 0), ("x", x), ("y", y),
                    ("dir", "N" if d == "W" else "W"), ("Tile", 0)]) + "/>")

        if storey.railing and rail_idx:
            for x, y, d in sorted(storey.railing):
                rail_attrs = [("type", "wall"), ("length", 1),
                              ("InteriorTile", rail_idx), ("ExteriorTrim", 0),
                              ("InteriorTrim", 0), ("x", x), ("y", y),
                              ("dir", "N" if d == "W" else "W"),
                              ("Tile", rail_ext_idx)]
                out.append(f"  <object{_attrs(rail_attrs)}/>")

        if level == 0 and uk_ids:
            for d, fixed, start, length in runs:
                for j in range(length):
                    t = start + j
                    x, y = (fixed, t) if d == "W" else (t, fixed)
                    piece = "start" if j == 0 else "end" if j == length - 1 else "pane"
                    tile = uk_ids[piece]
                    out.append("  <object" + _attrs([
                        ("type", "wall"), ("length", 1), ("InteriorTile", uk_ids[piece + "_in"]),
                        ("ExteriorTrim", 0), ("InteriorTrim", 0), ("x", x), ("y", y),
                        ("dir", "N" if d == "W" else "W"), ("Tile", tile)]) + "/>")
        elif level == 0:
            for d, fixed, start, length in runs:
                x, y = (fixed, start) if d == "W" else (start, fixed)
                # A wall object runs along y when its dir is N, placing west
                # walls, and along x when it is W.
                attrs = [("type", "wall"), ("length", length), ("InteriorTile", store_int),
                         ("ExteriorTrim", 0), ("InteriorTrim", 0), ("x", x), ("y", y),
                         ("dir", "N" if d == "W" else "W"), ("Tile", store_ext)]
                out.append(f"  <object{_attrs(attrs)}/>")

        def room_here(x, y):
            return 0 <= y < building.height and 0 <= x < building.width and storey.grid[y][x]
        for accent, accent_idx in zip(accents, accent_ids):
            if level < accent.get("from_level", 0) or level not in accent.get("levels", range(level + 1)):
                continue
            for x, y in sorted(accent["cells"]):
                if not room_here(x, y):
                    continue
                # (tile the wall stands on, dir): "N" puts a west wall, "W" a north one.
                # KnoxMap UK: an accent's `faces` keeps it to those sides of the
                # building (a house's upper-floor panel on its front and back only).
                faces = accent.get("faces")
                for face, (nx, ny, wx, wy, d) in zip("WENS", ((x - 1, y, x, y, "N"), (x + 1, y, x + 1, y, "N"),
                                                            (x, y - 1, x, y, "W"), (x, y + 1, x, y + 1, "W"))):
                    if faces and face not in faces:
                        continue
                    if not room_here(nx, ny):
                        out.append("  <object" + _attrs([
                            ("type", "wall"), ("length", 1), ("InteriorTile", interior_idx),
                            ("ExteriorTrim", 0), ("InteriorTrim", 0), ("x", wx), ("y", wy),
                            ("dir", d), ("Tile", accent_idx)]) + "/>")

        for x, y, direction in storey.windows:
            if level == 0 and (x, y, direction) in glazed:
                continue                        # the glass wall is the window
            if level >= 1 and (style or {}).get("blank_upper"):
                continue                        # KnoxMap UK: a superstore's cladding has no windows
            if (style or {}).get("superstore") and level >= 1:
                continue                        # ...and its brick none but its entrance glass and the
                                                # ground floor's windows superstore_uk set (2026-10-26:
                                                # windows in the brick, posters in them; not a glass wall)
            own = (level, (x, y, direction)) in own_window
            if (style or {}).get("blank") and not own:
                continue                        # ...nor a barn's or a farm shed's boarding (a landmark's
                                                # own windows stay: Parliament's are all described)
            tile, curtains = window_idx, curtains_idx
            if own:
                tile, curtains = own_window[(level, (x, y, direction))], 0
            if front_idx and (x, y, direction) in getattr(storey, "shop_front", ()):
                tile, curtains = front_idx, front_curtains
            attrs = [("type", "window"), ("CurtainsTile", curtains),
                     ("ShuttersTile", 0 if tile == front_idx or own else shutters_idx),
                     ("x", x), ("y", y),
                     ("dir", direction), ("Tile", tile)]
            out.append(f"  <object{_attrs(attrs)}/>")

        for role, x, y, orient in storey.furniture:
            if open_tile(level, x, y):
                continue                        # nothing to stand on
            attrs = [("type", "furniture"),
                     ("FurnitureTiles", roles.index(role)),
                     ("orient", orient), ("x", x), ("y", y)]
            out.append(f"  <object{_attrs(attrs)}/>")

        # A staircase belongs to the storey it rises from; the reader puts the
        # opening in the floor above by itself. Only N and W are valid
        # directions - Stairs::bounds returns an empty rect for anything else,
        # and an empty rect is treated as an invalid object.
        if level < len(building.stairs):
            sx, sy, sdir = building.stairs[level]
            attrs = [("type", "stairs"), ("x", sx), ("y", sy),
                     ("dir", sdir), ("Tile", C.STAIRS)]
            out.append(f"  <object{_attrs(attrs)}/>")
        # KnoxMap UK: a landmark's other staircases (landmark_uk.dress `extra_stairs`)
        for lvl, sx, sy, sdir in getattr(building, "extra_stairs", ()):
            if lvl == level:
                out.append("  <object" + _attrs([("type", "stairs"), ("x", sx), ("y", sy),
                                                  ("dir", sdir), ("Tile", C.STAIRS)]) + "/>")

        # Roofs on the top storey only - one on each would bury every floor
        # below under a ceiling of roof tiles - and shaped to the footprint
        # rather than its bounding box, so an L-shaped building does not carry
        # a roof over its own back yard.
        # A roof over whatever of this storey has no storey above it: the
        # whole top floor, and the ledge where a tower steps back.
        above = storeys[level + 1].grid if level + 1 < len(storeys) else None
        # A square left open in the storey above - the hole down the middle of
        # a mall's upper floor - has no room there, but it is still built
        # over: roofing it would put a ceiling across the concourse and hide
        # the floor the hole exists to show.
        open_above = (storeys[level + 1].void
                      if level + 1 < len(storeys) else set())
        exposed = [[v if above is None or not (above[y][x] or (x, y) in open_above)
                    else 0
                    for x, v in enumerate(row)] for y, row in enumerate(storey.grid)]
        if planned is not None:
            # A roof planned part by part (knoxbuild/regional_uk.py): each part's
            # pieces on their own floor. A pitched piece on the floor above
            # its part's top storey stands among the taller part's rooms on
            # that floor, as a lower wing's roof does.
            for roof_type, depth, cap, (rx, ry, rw, rh) in planned.get(level, []):
                out.append(_roof_object(roof_type, depth, cap, rx, ry, rw, rh,
                                        roof_cap_idx, slope_idx, top_idx))
        elif any(any(row) for row in exposed):
            rects = roof_rects(exposed)
            for roof_type, depth, cap, (rx, ry, rw, rh) in _roof_pieces(
                    rects, peaked and (above is None or PITCH_LOWER_ROOFS), roof30):
                roof_attrs = [
                    ("type", "roof"),
                    ("width", rw),
                    ("height", rh),
                    ("RoofType", roof_type),
                    ("Depth", depth),
                    ("cappedW", str(cap["cappedW"]).lower()),
                    ("cappedN", str(cap["cappedN"]).lower()),
                    ("cappedE", str(cap["cappedE"]).lower()),
                    ("cappedS", str(cap["cappedS"]).lower()),
                    ("CapTiles", roof_cap_idx),
                    ("SlopeTiles", slope_idx),
                    ("TopTiles", top_idx),
                    ("x", rx), ("y", ry),
                ]
                line = f"  <object{_attrs(roof_attrs)}/>"
                # BuildingEd measures a roof's height up from the floor it is
                # on. A flat roof at depth three on the top storey ends level
                # with its ceiling, which is right; a pitched roof there rose
                # through the top storey, its gable ends covering the upstairs
                # walls. Pitched roofs go on the roof floor above instead.
                if roof_type == "FlatTop":
                    out.append(line)
                elif above is None:
                    attic.append(line)
                else:
                    above_roofs.setdefault(level + 1, []).append(line)

        # Match BuildingWriter byte for byte: a comma follows every value
        # except the very last one, and each row ends with a newline.
        grid = building.grid_for(level)
        text = ["\n"]
        count, total = 0, building.width * building.height
        for y in range(building.height):
            for x in range(building.width):
                text.append(str(grid[y][x]))
                count += 1
                if count < total:
                    text.append(",")
            text.append("\n")
        out.append("  <rooms>" + escape("".join(text)) + "</rooms>")
        if (layer := user_tile_layer(level)):
            out.append(layer)
        if (layer := user_tile_layer(level, drains, "WallFurniture2")):
            out.append(layer)
        if (layer := user_tile_layer(level, bays, "Curtains2")):
            out.append(layer)
        for lname, tiles in (painted.get(level) or {}).items():
            if (layer := user_tile_layer(0, {0: tiles}, lname)):
                out.append(layer)

        out.append(" </floor>")

    if planned is not None:
        for roof_type, depth, cap, (rx, ry, rw, rh) in planned.get(len(storeys), []):
            attic.append(_roof_object(roof_type, depth, cap, rx, ry, rw, rh,
                                      roof_cap_idx, slope_idx, top_idx))
    # The roof floor: no rooms. It holds the pitched roofs, and the flat roof
    # tops BuildingEd places here from the depth-three roofs below.
    empty = ["\n"]
    count, total = 0, building.width * building.height
    for _y in range(building.height):
        for _x in range(building.width):
            empty.append("0")
            count += 1
            if count < total:
                empty.append(",")
        empty.append("\n")
    out.append(" <floor>")
    out.extend(attic)
    for role, x, y, orient in rooftop:
        attrs = [("type", "furniture"), ("FurnitureTiles", roles.index(role)),
                 ("orient", orient), ("x", x), ("y", y)]
        out.append(f"  <object{_attrs(attrs)}/>")
    out.append("  <rooms>" + escape("".join(empty)) + "</rooms>")
    if (layer := user_tile_layer(len(storeys))):
        out.append(layer)
    for lname, tiles in (painted.get(len(storeys)) or {}).items():
        if (layer := user_tile_layer(0, {0: tiles}, lname)):
            out.append(layer)
    out.append(" </floor>")
    # KnoxMap UK: a landmark's pieces above the roof floor (a spire, landmark_uk.py)
    # go on floors of their own, as many as they reach.
    top = max([lvl for lvl, layers in painted.items() if any(layers.values())] or [0])
    for lvl in range(len(storeys) + 1, max(top, len(storeys) + (1 if attic else 0)) + 1):
        # A pitched roof's top rises a full storey above the roof floor, and
        # BuildingEd lays what is up there on the floor above that.
        out.append(" <floor>")
        out.append("  <rooms>" + escape("".join(empty)) + "</rooms>")
        for lname, tiles in (painted.get(lvl) or {}).items():
            if (layer := user_tile_layer(0, {0: tiles}, lname)):
                out.append(layer)
        out.append(" </floor>")
    out.append("</building>")
    return "\n".join(out) + "\n"


def _lay_drains(storeys, width: int, height: int, pipes, drains: dict, bays: dict | None = None) -> None:
    """KnoxMap UK: a downpipe at each end of every outside wall run the camera
    sees (west walls run along y, north walls along x), from the ground floor
    to the top storey over its square: foot, middle, top, or a one-storey
    piece (regional_uk.DRAINS: [side][end][part]). Not along a party wall (the
    next house's side: QA, 2026-10-19), nor through a bay."""
    def inside(level, x, y):
        g = storeys[level].grid if 0 <= level < len(storeys) else None
        return bool(g) and 0 <= y < len(g) and 0 <= x < len(g[0]) and bool(g[y][x])
    party = set(getattr(storeys[0], "party", ())) if storeys else set()
    in_bay = set((bays or {}).get(0, {}))
    for side in (0, 1):
        edges = [(x, y) for x in range(width + 1) for y in range(height + 1)
                 if inside(0, *((x - 1, y) if side == 0 else (x, y - 1))) and not inside(0, x, y)
                 and (x, y, "WN"[side]) not in party and (x, y) not in in_bay]
        key = (lambda e: (e[0], e[1])) if side == 0 else (lambda e: (e[1], e[0]))
        edges.sort(key=key)
        wall_runs, run = [], []
        for e in edges:
            if run and key(e)[0] == key(run[-1])[0] and key(e)[1] == key(run[-1])[1] + 1:
                run.append(e)
            else:
                if run:
                    wall_runs.append(run)
                run = [e]
        if run:
            wall_runs.append(run)
        for run in wall_runs:
            for end, (x, y) in ((0, run[0]), (1, run[-1])):
                if end == 1 and len(run) == 1:
                    break
                inner = (x - 1, y) if side == 0 else (x, y - 1)
                top = 0
                while inside(top + 1, *inner):
                    top += 1
                for level in range(top + 1):
                    part = 3 if top == 0 else 0 if level == 0 else 2 if level == top else 1
                    drains.setdefault(level, {}).setdefault((x, y), pipes[side][end][part])


BAYS_MAX = 3          # squares of bay a house at most (a wide bay is two)
SINGLE_BAYS = False   # one-square bays, archived for now (user, 2026-10-25): only two-square bays are laid


def _lay_bays(storeys, bay: dict, street, door_at, user_tiles: dict, bays: dict) -> None:
    """Bays (regional_uk.bay_tiles: {part: (west, north)}) into `bays` on the squares
    outside the street front's ground-floor windows - where the window's own
    square is outside the house, its wall's outside facing east or south, the
    faces the camera sees - at most BAYS_MAX, no two side by side, none on the
    front door's square or one already holding a piece. A two-storey bay
    where the window above is also a window, else a one-storey one. The
    first bay is two squares wide (user, 2026-10-19) where the square next
    to its window along the wall is free outside wall too. The windows
    behind the bays stay, drawn with the bay's invisible window (`window`)."""
    grid0 = storeys[0].grid

    def inside(x, y):
        return 0 <= y < len(grid0) and 0 <= x < len(grid0[0]) and grid0[y][x]

    def free(x, y, d, two_storey):
        """A square a bay can stand on: outside, the house behind its wall."""
        behind = (x - 1, y) if d == "W" else (x, y - 1)
        return (not inside(x, y) and inside(*behind) and (x, y) not in user_tiles.get(0, {})
                and (x, y) not in bays.get(0, {}) and not (door_at and (x, y) == door_at[:2])
                and not (two_storey and (x, y) in user_tiles.get(1, {})))
    above = set(storeys[1].windows) if len(storeys) > 1 else set()
    taken: list[tuple[int, int, str]] = []
    for x, y, d in sorted(storeys[0].windows):
        if len(taken) >= BAYS_MAX or not free(x, y, d, False):
            continue
        if street and _faces(grid0, x, y, d) != street:
            continue
        if any(td == d and abs(tx - x) + abs(ty - y) == 1 for tx, ty, td in taken):
            continue
        side = 0 if d == "W" else 1
        two = "upper" in bay and (x, y, d) in above and (x, y) not in user_tiles.get(1, {})
        step = (0, 1) if d == "W" else (1, 0)                       # along the wall
        pair = None
        if (not taken or not SINGLE_BAYS) and any(p.startswith("wide_") for p in bay):
            for a in ((x, y), (x - step[0], y - step[1])):          # the window first, else second
                b = (a[0] + step[0], a[1] + step[1])
                other = b if a == (x, y) else a
                if free(*other, d, two):
                    pair = (a, b)
                    break
        if pair and ("wide_upper" if two else "wide_single") in bay:
            for k, sq in enumerate(pair):
                if two:
                    bays.setdefault(0, {})[sq] = bay["wide_lower"][side][k]
                    bays.setdefault(1, {})[sq] = bay["wide_upper"][side][k]
                else:
                    bays.setdefault(0, {})[sq] = bay["wide_single"][side][k]
                taken.append((sq[0], sq[1], d))
            continue
        if not SINGLE_BAYS:
            continue                    # one-square bays archived (user, 2026-10-25): two-square bays only
        if two:
            bays.setdefault(0, {})[(x, y)] = bay["lower"][side]
            bays.setdefault(1, {})[(x, y)] = bay["upper"][side]
        elif "single" in bay:
            bays.setdefault(0, {})[(x, y)] = bay["single"][side]
        else:
            continue
        taken.append((x, y, d))
    hidden = bay.get("window")
    for level, st in enumerate(storeys[:2]):
        for w in st.windows:
            if tuple(w) in taken and w[:2] in bays.get(level, {}) and hidden:
                entries = dict(getattr(st, "window_entries", None) or {})
                entries[tuple(w)] = hidden
                st.window_entries = entries


def _lay_trims(storeys, width: int, height: int, trims: dict, user_tiles: dict, bays: dict | None = None) -> None:
    """KnoxMap UK: the plinth on the ground floor, the band at the first
    floor's foot, the cornice on each square's top storey and a parapet on
    the roof's floor beside it, along the outside walls the camera sees; a
    square already holding a doorway, a sign or a bay keeps it, and a party
    wall (the next house's side: QA, 2026-10-19) has none."""
    def inside(level, x, y):
        g = storeys[level].grid if 0 <= level < len(storeys) else None
        return bool(g) and 0 <= y < len(g) and 0 <= x < len(g[0]) and bool(g[y][x])
    # (KnoxMap UK housing styles: no plinth across a doorway; a plinth with
    # corner pieces puts its post at each outside corner the camera sees, the
    # square diagonally off the building's south-east corner.)
    doors = {(x, y, d) for x, y, d in storeys[0].doors} if storeys else set()
    party0 = set(getattr(storeys[0], "party", ())) if storeys else set()
    plinth = trims.get("plinth")
    if plinth and len(plinth) > 3 and storeys:
        for x in range(width + 1):
            for y in range(height + 1):
                if (inside(0, x - 1, y - 1) and not inside(0, x - 1, y) and not inside(0, x, y - 1)
                        and not inside(0, x, y) and (x, y) not in user_tiles.get(0, {})
                        and (x, y) not in (bays or {}).get(0, {})
                        and (x, y - 1, "W") not in party0 and (x - 1, y, "N") not in party0):
                    user_tiles.setdefault(0, {})[(x, y)] = plinth[3]
    for level in range(len(storeys)):
        wanted = [k for k, ok in (("plinth", level == 0), ("band", level == 1)) if ok and k in trims]
        for x in range(width + 1):
            for y in range(height + 1):
                if (x, y) in user_tiles.get(level, {}) or (x, y) in (bays or {}).get(level, {}):
                    continue
                # a west wall seen from the east, else a north one seen from the south
                for side, inner in ((0, (x - 1, y)), (1, (x, y - 1))):
                    if not inside(level, *inner) or inside(level, x, y):
                        continue
                    if (x, y, "WN"[side]) in getattr(storeys[level], "party", ()):
                        continue
                    kinds = [k for k in wanted if not (k == "plinth" and (x, y, "WN"[side]) in doors)]
                    if "cornice" in trims and not inside(level + 1, *inner):
                        kinds.append("cornice")
                    if kinds:
                        # one sprite a square: the cornice over the band over the plinth
                        pick = "cornice" if "cornice" in kinds else kinds[-1]
                        piece = side
                        # (KnoxMap UK housing plinths: a square walled west and north,
                        # an inside corner, takes the piece with both.)
                        if (pick == "plinth" and side == 0 and len(trims[pick]) > 2 and inside(level, x, y - 1)
                                and (x, y, "N") not in doors):
                            piece = 2
                        user_tiles.setdefault(level, {})[(x, y)] = trims[pick][piece]
                    if "parapet" in trims and not inside(level + 1, *inner):
                        user_tiles.setdefault(level + 1, {}).setdefault((x, y), trims["parapet"][side])
                    break
