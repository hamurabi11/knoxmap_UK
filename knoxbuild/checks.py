"""Hard checks on what Build wrote, run at the end of every Build (build.py):
a failure stops the build with a non-zero exit, naming each building.

empty_storeys: a storey with no rooms. In the game every square of a storey
with no room is outdoors (no ceiling, no loot, nobody's home), so a building
whose upper floors came out roomless looks finished from the street and is a
shell inside. Also run by tools/validate_tbx.py (and so the smoke test) on
the .tbx alone.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

STOREY_OBJECTS = {"door", "window", "stairs", "wall"}      # only a storey has these (a roof floor may
                                                            # carry rooftop plant: furniture)
FLOOR_LAYERS = {"Floor", "FloorOverlay", "FloorGrime", "FloorGrime2"}


def empty_storeys(path: str, storeys: int | None = None) -> list[str]:
    """Storeys built with no rooms: in the game every square of one is
    outdoors (no room, no ceiling, no loot), so its floor is a roof to stand on.

    A floor with no room on it is an empty storey when
      - it is one of the building's `storeys` (Build's plan.storeys, when given:
        the floors above them are only roof),
      - a floor above it has rooms (a storey missing out of the middle), or
      - it holds what only a storey has: a door, window, stair or wall, or
        tiles on a floor layer (rooftop plant and roof tiles may stand on a roof).
    Without `storeys`, a file with no room on any floor is a pack of loose
    tiles (Build's fences, structures, chimneys), not a building: not checked.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [f"XML parse error: {exc}"]
    floors = root.findall("floor")

    def has_rooms(floor) -> bool:
        g = floor.find("rooms")
        return any(v.strip() not in ("", "0") for v in ((g.text if g is not None else "") or "")
                   .replace("\n", ",").split(","))

    rooms = [has_rooms(f) for f in floors]
    if storeys is None and not any(rooms):
        return []       # no room anywhere: a pack of loose tiles (fences, structures, chimneys), not a building
    out = []
    for level, floor in enumerate(floors):
        if rooms[level]:
            continue
        why = None
        if storeys is not None and level < storeys:
            why = f"one of its {storeys} storeys"
        elif any(rooms[level + 1:]):
            why = "rooms on a floor above it"
        else:
            things = sorted({o.get("type") for o in floor.findall("object")} & STOREY_OBJECTS)
            tiles = sorted(t.get("layer") for t in floor.findall("tiles")
                           if t.get("layer") in FLOOR_LAYERS
                           and any(v.strip() not in ("", "0") for v in (t.text or "").replace("\n", ",").split(",")))
            if things or tiles:
                why = "it holds " + ", ".join(things + [f"{t} tiles" for t in tiles])
        if why:
            out.append(f"storey {level} has no rooms ({why})")
    return out


def check_buildings(bdir: str, rows: list) -> list[str]:
    """empty_storeys for every building Build wrote (its placements rows:
    file, name, levels): one line each, "file (name): storey N has no rooms (why)"."""
    import os
    out = []
    for r in rows:
        for e in empty_storeys(os.path.join(bdir, r["file"]), int(r["levels"])):
            out.append(f"{r['file']} ({r['name']}): {e}")
    return out
