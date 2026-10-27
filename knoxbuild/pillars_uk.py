"""Pillars in big open rooms (user, 2026-10-26: "some pillars for really big open
areas ... same for other buildings too", from the game's own Gigamarts).

The game's Gigamarts stand white pillars on a grid in their sales floors, about
8 squares apart (Knox County, cell 52,11: walls_commercial_02 every 8 squares down
the store; cell 25,20: walls_interior_house_03). A pillar is four interior walls
round one square: here a room of one square (the game's "empty" kind, no door),
so Build draws its walls as it draws any wall between two rooms, in the bigger
room's interior wall.

Only where the floor is open: a room PITCH squares or more each way, the pillar
on its grid, with its eight neighbours in the room, nothing standing on it or
beside it, and clear of doorways. Not in a sports hall, a church, a theatre or a
vehicle bay (CLEAR), which stand on clear spans.

Not on a shop's sales floor either (SALES, user 2026-10-27): a pillar is a one-square
room, and B42's wall cutaway keeps cutting its four walls in and out as the player
walks the open floor round it (they flickered in the Alton superstores).
"""
from __future__ import annotations

PITCH = 8                   # squares between pillars (the Gigamart's)
MIN_SIDE = 2 * PITCH - 2    # a room this wide each way at least (two pillars across)
MIN_AREA = 200              # m2
KIND = "empty"              # the game's room name for a space with nothing in it
CLEAR = {"gym", "church", "theatre", "garage", "firegarage", "mechanic", "barn", "shed", "hall", "lobby",
         "bathroom", "classroom", "spiffo_dining"}
# a shop's open sales floor: superstore_uk's OPEN rooms and its concessions' kinds
SALES = {"gigamart", "grocery", "liquorstore", "housewarestore", "pharmacy", "clothingstore", "cafe",
         "conveniencestore", "toolstore", "camerastore", "gardenstore"}
CLEAR |= SALES


def register() -> None:
    from . import catalog as C
    from . import layout as L
    L.ROOM_STYLE.setdefault(KIND, (C.FLOOR_LINO, "Empty", []))
    C.ROOM_COLORS.setdefault(KIND, "200 200 200")


def add(plan) -> int:
    """Pillars into each big open room of one storey; the number laid."""
    from . import layout as L
    register()
    g = plan.grid
    h, w = len(g), len(g[0]) if g else 0
    taken = set()
    for role, x, y, orient in plan.furniture:
        try:
            taken.update(L._cells_for(role, x, y, orient))
        except KeyError:
            taken.add((x, y))
    for x, y, d in list(plan.doors) + list(getattr(plan, "open_edge", ())):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                taken.add((x + dx, y + dy))
    taken |= set(getattr(plan, "void", ()) or ())
    laid = 0
    for idx, room in list(enumerate(plan.rooms, 1)):
        if room.kind in CLEAR or room.kind == KIND or room.is_core or room.is_shaft:
            continue
        cells = {(x, y) for y in range(room.y0, room.y1 + 1) for x in range(room.x0, room.x1 + 1)
                 if 0 <= y < h and 0 <= x < w and g[y][x] == idx}
        if len(cells) < MIN_AREA or room.x1 - room.x0 + 1 < MIN_SIDE or room.y1 - room.y0 + 1 < MIN_SIDE:
            continue
        # the grid centred in the room, so the walls have the same margin each side
        ox = room.x0 + ((room.x1 - room.x0) % PITCH) // 2 + PITCH // 2
        oy = room.y0 + ((room.y1 - room.y0) % PITCH) // 2 + PITCH // 2
        for y in range(oy, room.y1, PITCH):
            for x in range(ox, room.x1, PITCH):
                ring = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                if not all(c in cells for c in ring) or any(c in taken for c in ring):
                    continue
                plan.rooms.append(L.Room(x, y, x, y, kind=KIND, fixed=True))
                g[y][x] = len(plan.rooms)
                taken.update(ring)
                laid += 1
    return laid
