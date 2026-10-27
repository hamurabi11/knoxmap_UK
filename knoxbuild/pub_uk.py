"""English pubs for the room solver (rooms_uk.py, ROOMS.md section 17).

A town pub in 1990: a public bar and a lounge (or saloon) bar along the street,
each with its own door from the street and the counter across its back; the
cellar behind the counter, with its drop door to the yard; gents and ladies
off the bars (licensed premises have both); a snug or games room off a bar
and a kitchen for bar food in some. Upstairs the licensee's flat,
reached by a private stair from behind the bar, or where nobody lives over it
a function room.

The rooms are tables like every other programme's (rooms_uk.json shops:
grounds.pub, uppers.pub; kinds bar_room, cellar, function_room), solved by
rooms_uk as base@pub: shop_house@pub (the landlord's flat), shop_flats@pub,
shop_only@pub. This module says which building is a pub, fits its bars out
(fit_bars: the counter, the back bar, the tables) and says what a pub's plan
must have (check).
"""
from __future__ import annotations

BAR = "bar"     # uses.py: amenity=pub / bar / nightclub / biergarten -> ("bar", "barstorage")


def pub_use(uses) -> tuple | None:
    """The pub's use among a building's (front room, back room) uses, or None.
    A pub's food hygiene record or business rates are the same pub, so a pub
    with other uses is still one pub."""
    for use in uses or []:
        if use and use[0] == BAR:
            return tuple(use)
    return None


COUNTER = "counter_4"                    # Build's own bar counter (interiors.BACK_COUNTERS["bar"])
BACK_BAR = ["shop_fridge", "shelf"]     # the bottle fridges and shelves behind it, in turn
TABLE_SET = [("round_table", 1, 0, "W"), ("chair", 0, 0, "W"), ("chair", 2, 0, "E")]
TABLE_PITCH = 3                         # a small table and two stools every 3 squares each way (9 m2)
COUNTER_MAX = 8                         # squares of counter in one run
MIN_DEPTH = 5                           # back bar, the staff's aisle, the counter, two rows to stand or sit
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
STEP = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}


def fit_bars(plan, street: str | None) -> int:
    """Fit out each bar of a pub, after Build's furnishing (to_building): the
    counter in a run across the room one aisle in from a wall - the wall with
    the cellar behind it if it can - its flap left open at one end; bottle
    fridges and shelves on the wall behind; small tables and stools over the
    floor (TABLE_PITCH). Build's own fitting (interiors.furnish_dining) gave a big bar a
    table or two and a counter of a piece or none. A bar too small for a run
    keeps Build's. What hangs on the walls (the light switch, pictures) stays.
    Returns the bars fitted."""
    from . import catalog as C
    from . import interiors as I
    from . import layout as L
    fitted = 0
    for idx, room in enumerate(plan.rooms, 1):
        if room.kind != BAR:
            continue
        cells = I._cells(plan, idx, room)
        # the doorways, two squares in, stay clear
        clear = set()
        for x, y, d in plan.doors:
            for sq in ((x - 1, y), (x, y)) if d == "W" else ((x, y - 1), (x, y)):
                if sq in cells:
                    clear.add(sq)
                    dx, dy = (1 if sq[0] == x else -1, 0) if d == "W" else (0, 1 if sq[1] == y else -1)
                    clear.add((sq[0] + dx, sq[1] + dy))
        run = _counter_run(plan, idx, cells, clear, street)
        if run is None:
            continue
        back, squares = run
        frame = I.Frame(cells, OPPOSITE[back])
        # Build's standing pieces out of this bar; the wall-hung ones stay
        plan.furniture = [f for f in plan.furniture
                          if (f[1], f[2]) not in cells or C.FURNITURE_LAYERS.get(f[0], "Furniture") != "Furniture"]
        occupied = set(clear)
        for a, d in squares:
            x, y = frame.xy(a, d)
            plan.furniture.append((COUNTER, x, y, L._facing(COUNTER, back)))
            occupied |= {(x, y), frame.xy(a, d - 1)}          # and room to stand at it
        span = [a for a, _ in squares]
        for n, a in enumerate(range(min(span), max(span) + 2)):
            for d in (frame.depth - 2, frame.depth - 1):
                occupied.add(frame.xy(a, d))                  # the staff's side, the flap
            x, y = frame.xy(a, frame.depth - 1)
            if (x, y) in cells and (x, y) not in clear and a <= max(span):
                role = BACK_BAR[n % len(BACK_BAR)]
                plan.furniture.append((role, x, y, L._facing(role, back)))
        _tables(plan, cells, occupied)
        fitted += 1
    return fitted


def _tables(plan, cells: set, occupied: set) -> int:
    """A small table every third square each way (TABLE_PITCH), a stool either
    side of it where the floor is free; an aisle is left between rows. The grid
    is set where it seats the most (a narrow bar has one row it can use)."""
    def lay(ox, oy):
        taken, out = set(occupied), []
        for (x, y) in sorted(cells, key=lambda c: (c[1], c[0])):
            if (x - ox) % TABLE_PITCH or (y - oy) % TABLE_PITCH:
                continue
            sides = [(x - 1, y), (x + 1, y)]
            if (x, y) in taken or not all(s in cells and s not in taken for s in sides):
                continue
            out.append((x, y))
            taken |= {(x, y)} | set(sides)
        return out, taken
    best, taken = max((lay(ox, oy) for oy in range(TABLE_PITCH) for ox in range(TABLE_PITCH)),
                      key=lambda got: len(got[0]))
    for x, y in best:
        for role, dx, dy, o in TABLE_SET:
            plan.furniture.append((role, x - 1 + dx, y + dy, o))
    occupied |= taken
    return len(best)


def _counter_run(plan, idx, cells, clear, street):
    """(the wall behind the counter, [(along, depth) of each counter square]) for
    the first wall that takes a run of 3 or more and its flap: the one the cellar is behind,
    then the one facing the street door, then the others; None if none does."""
    from . import interiors as I
    cellar = {i for i, r in enumerate(plan.rooms, 1) if r.kind == "barstorage"}

    def behind(side):
        dx, dy = STEP[side]
        return sum(1 for x, y in cells if 0 <= y + dy < len(plan.grid) and 0 <= x + dx < len(plan.grid[0])
                   and plan.grid[y + dy][x + dx] in cellar)
    order = sorted("NSEW", key=lambda s: (-behind(s), s != OPPOSITE.get(street or "", ""), s))
    for back in order:
        frame = I.Frame(cells, OPPOSITE[back])
        if frame.depth < MIN_DEPTH:
            continue
        d = frame.depth - 3
        squares = []
        for a in range(frame.length):
            # the counter square, the staff's aisle and the back bar in the room, and two
            # rows in front of it to stand or sit (an L-shaped bar's short arm has none)
            ok = all(frame.xy(a, k) in cells and frame.xy(a, k) not in clear for k in (d, d + 1, d + 2))                 and all(frame.xy(a, k) in cells for k in (d - 2, d - 1))
            if ok:
                squares.append((a, d))
                if len(squares) >= COUNTER_MAX:
                    break
            elif squares:
                break
        if len(squares) >= 4:
            return back, squares[:-1]          # the last square is the flap: the staff's way in
    return None


def check(lay) -> list[str]:
    """What a pub's plan must have, beyond rooms_uk.check: each bar its own
    door from the street, the gents, and the cellar on the back wall."""
    from . import rooms_uk as R
    out = []
    st = lay.storeys[0]
    rooms = st["rooms"]
    bars = [r for r in rooms if r["kind"] == "bar_room"]
    if not bars:
        return ["no bar"]
    doors = set(map(tuple, st["doors"]))
    # (a snug or games room off a bar has no street door of its own)
    for i, bar in enumerate(b for b in bars if b["spec"].get("root")):
        mine = {R._edge(x, y, lay.house.street) for (x, y) in bar["squares"]
                if lay.house.outside_edge(0, x, y, lay.house.street)}
        if not mine & doors:
            out.append(f"bar {i + 1}: no door from the street")
    if not any(r["kind"] == "wc" for r in rooms):
        out.append("no toilets")
    back = R.OPPOSITE[lay.house.street]
    sides = (back,) + (("E", "W") if back in ("N", "S") else ("N", "S"))
    cellars = [r for r in rooms if r["kind"] == "cellar"]
    if not cellars:
        out.append("no cellar")
    elif any(R._segments(lay.house, 0, sd) for sd in sides) and not any(
            lay.house.outside_edge(0, x, y, sd) for r in cellars for (x, y) in r["squares"] for sd in sides):
        # (a pub with party walls all round has nowhere for one)
        out.append("the cellar has no outside wall for its drop door")
    return out
