"""One listed building's design: its listing read against its LiDAR and outline.

`design(reading, entry, lidar, palette)` -> a spec dict:
  style      "uk_<wall>__<roof>__<window>": the wall the text names (WALL_KIT:
             a house wall kit of makers/wall_kit.py, or a palette stand-in),
             its roof covering (ROOF_KIT: makers/roof_kit.py) and its windows
             (makers/window_kit.py: from the text, else by date); `style_spec`
             turns the name into a palette style. Left out where the text names
             no wall and no roof (the building keeps export_regional's style).
  levels     storeys: the text's, unless the LiDAR's differ by more than one
             (then the LiDAR's: the text may describe a lower front range);
             the text's attic is the roof planner's, not an extra storey.
  door       doors_uk_01's door for the front (DOOR_FOR, else by era).
  needs      the assets this building wants that are still to make.
  notes      where text and LiDAR disagree, for the report.

Lookup tables only (feedback: closed problems as tables). The order of
WALL_ORDER says which material is the face seen when a text names several:
render and paint cover what is under them; brick refronts a timber frame.
"""
from __future__ import annotations

import re

import assets as AS
import lexicon as L

WALL_ORDER = ["wall.render", "wall.painted_brick", "wall.red_brick", "wall.flint", "wall.stone",
              "wall.timber_frame", "wall.weatherboard", "wall.tile_hung", "wall.blue_brick"]
BARN_ORDER = ["wall.weatherboard", "wall.red_brick", "wall.flint", "wall.timber_frame", "wall.stone",
              "wall.render", "wall.painted_brick", "wall.tile_hung", "wall.blue_brick"]
# Wall material -> the house wall kit (Tile_creator/build/sheets/<group>/walls_house_uk_01_<kit>),
# or a palette style where there is no kit (stone: vanilla's; brick after 1840: Victorian).
WALL_KIT = {"wall.render": "render_white", "wall.painted_brick": "brick_painted", "wall.red_brick": "brick_flemish",
            "wall.flint": "flint", "wall.stone": "stone", "wall.timber_frame": "timber_white",
            "wall.weatherboard": "weatherboard_white", "wall.tile_hung": "tile_hung", "wall.blue_brick": "brick_chequer"}
LIGHT = {"render_white", "render_cream", "brick_painted", "timber_white", "weatherboard_white"}
KITS = {"brick_flemish", "brick_painted", "brick_chequer", "render_white", "render_cream", "timber_white",
        "timber_brick", "flint", "tile_hung", "weatherboard_black", "weatherboard_white"}
CREAM = re.compile(r"\b(cream|ochre|buff|yellow)\b", re.I)
# A granary on staddle stones ("Timber-framed small granary on 9 staddles"): the
# commercial kit's boarding over the gap under its floor (commercial_kit/farm.py).
STADDLE = re.compile(r"\bstaddle", re.I)
NOGGING = re.compile(r"brick[- ]?nogg|brick infill|infilled? (with|in) brick|brick panels", re.I)
CHIMNEY = {"flint": "fixtures_fireplaces_01_068", "stone": "fixtures_fireplaces_01_068"}   # else red brick, 064
ROOF_KIT = {"roof.plain_tile": "tile", "roof.slate": "slate", "roof.stone_slate": "slate", "roof.thatch": "thatch"}
# makers/window_kit.py WINDOWS, in its order (a window's West sprite is slot 2 k).
WINDOW_KIT = ["sash_georgian", "sash_victorian", "casement", "casement_leaded", "mullion", "sash_arched"]
# doors_uk_01's doors (Knoxmap/knoxbuild/regional_uk.py DOOR_STYLES).
DOOR_FOR = [("door.boarded", "oak_cottage"), ("door.six_panel", "hardwood_fanlight"),
            ("door.fanlight", "hardwood_fanlight"), ("door.four_panel", "black_victorian"),
            ("door.panelled", "black_victorian")]
DOOR_BY_ERA = [(1700, "oak_cottage"), (1840, "hardwood_fanlight"), (1900, "black_victorian"), (9999, "blue_edwardian")]
WINDOW_BY_ERA = [(1700, "casement_leaded"), (1840, "sash_georgian"), (9999, "sash_victorian")]
MAX_LEVEL_GAP = 1


def _by_era(table, era):
    return next(v for until, v in table if (era or 1800) < until)


def _first(reading: L.Reading, feats: list[str]) -> str | None:
    """Of these features (old, front), the one the text names first."""
    found = [(reading.body.find(reading.hits[f][0].quote), f) for f in feats if reading.has(f)]
    return min(found)[1] if found else None


WALLING = re.compile(r"\bwall(ing|s)\b|\bbrickwork\b", re.I)


FACED = re.compile(r"\b(faced (in|with)|refaced (in|with)|refronted (in|with)?|the front is|front(age)? (is|of))\b", re.I)


def wall_of(reading: L.Reading) -> str | None:
    """The main wall material: what the front is faced with ("Timber-framed
    house, faced with brickwork"; "The front is painted brickwork"), else the
    first one named in the text's first sentence about its walling ("Walling is
    in flint, with ... some brickwork"), else by WALL_ORDER (BARN_ORDER)."""
    faced = None                                        # the last one says it best ("faced with
    for m in FACED.finditer(reading.body):              # brickwork ... The front is painted brickwork")
        rest = reading.body[m.end():m.end() + 50]
        named = [(L.FEATURES_RX[f].search(rest).start(), f) for f in WALL_ORDER if L.FEATURES_RX[f].search(rest)]
        if named:
            faced = min(named)[1]
    if faced:
        return faced
    found = []
    for f in WALL_ORDER:
        for h in reading.hits.get(f, ()):
            if h.scope == "front" and not h.modern and WALLING.search(h.quote):
                found.append((reading.body.find(h.quote), L.FEATURES_RX[f].search(h.quote).start(), f))
    if found:
        return min(found)[2]
    order = BARN_ORDER if reading.kind == "barn" else WALL_ORDER
    return next((f for f in order if reading.has(f)), None)


def style_spec(name: str, palette: dict) -> dict:
    """A palette `add` spec for a style name "uk_<wall>__<roof>__<window>"
    (roof and window may be empty: the wall style's own)."""
    wall, roof, window = name[3:].split("__")
    if wall in KITS:
        sheet = f"walls_house_uk_01_{wall}"
        spec = {"like": "brick", "wall": f"{sheet}_000", "shaped": f"{sheet}_016", "caps": f"{sheet}_048",
                "caps_shallow": f"{sheet}_066", "caps_30": f"{sheet}_074", "roof_from": "brick",   # wall_kit.FIRST
                "trim_from": "timber",                    # no trim: vanilla brick's dark storey trim shows on render
                "grime": wall not in LIGHT,               # nor its soot line at each storey's foot
                "shutters": False,                        # nor outside shutters (American)
                "chimney": CHIMNEY.get(wall, "fixtures_fireplaces_01_064")}
    else:
        added = {s["name"]: s for s in palette.get("add", [])}
        spec = dict(added.get(wall) or {"like": wall})
    spec["name"] = name
    if roof:
        spec["roof_from"] = "brick"                       # roofs_02's slots, the kit's sheets in their place
        spec["roof_sheet"], spec["roof_sheet_30"] = f"roofs_uk_01_{roof}", f"roofs_30_uk_01_{roof}"
    if window:
        spec["window"] = f"fixtures_windows_uk_01_{2 * WINDOW_KIT.index(window):03d}"
    spec["by_id_only"] = True
    return spec


def window_of(reading: L.Reading) -> str:
    """The windows the text names; "" for a church (traceried, Gothic: its
    style's own windows until there is a church kit)."""
    if reading.kind == "church":
        return ""
    era = reading.era
    if reading.has("window.mullion"):
        return "mullion"
    if reading.has("window.round_headed") and not reading.has("window.casement"):
        return "sash_arched"
    if reading.has("window.sash"):
        victorian = re.search(r"victorian sash|2[- ]over[- ]2|plate glass", reading.body, re.I)
        return "sash_victorian" if victorian or (era or 1800) >= 1840 else "sash_georgian"
    if reading.has("window.casement") or reading.has("window.leaded"):
        return "casement_leaded" if reading.has("window.leaded") or (era or 1800) < 1700 else "casement"
    return _by_era(WINDOW_BY_ERA, era)


TIMBER_PORCH = re.compile(r"(timber|oak|gabled|tiled|weather[- ]?boarded|rustic)\W+(\w+\W+){0,3}porch|porch\W+(\w+\W+){0,4}(timber|oak|gabled|tiled)", re.I)
CLASSICAL_PORCH = re.compile(r"\b(doric|tuscan|ionic|corinthian)\b|\bporch of \d+ columns|\bcolumned porch", re.I)


def surround_of(reading: L.Reading) -> str | None:
    """The front door's doorcase, hood or porch (makers/doorway_kit.py), from the listing."""
    if reading.has("door.porch"):
        # A porch the text gives columns or an order is the columned one, whatever
        # the building's date (The George, C15 with a Greek Doric porch).
        if CLASSICAL_PORCH.search(reading.body):
            return "porch"
        cottage = reading.kind in ("cottage", "farmhouse") or (reading.era or 1800) < 1700
        return "porch_timber" if TIMBER_PORCH.search(reading.body) or cottage else "porch"
    if reading.has("door.doorcase") or reading.has("door.pediment"):
        return "doorcase"
    if reading.has("door.hood"):
        return "hood"
    return None


def trims_of(reading: L.Reading, style: str | None) -> dict:
    """Plinth, first-floor band and eaves cornice (makers/trim_kit.py) the
    listing names: painted on render and painted brick, stone or brick on brick."""
    painted = bool(style) and any(k in style for k in ("render", "painted", "cream", "timber_white"))
    out = {}
    if reading.has("trim.plinth"):
        out["plinth"] = "plinth_black" if painted else "plinth_brick"
    if reading.has("trim.band"):
        out["band"] = "band_white" if painted else "band_stone"
    if reading.has("trim.cornice"):
        out["cornice"] = "cornice_white" if painted else "cornice_stone"
    if reading.has("roof.parapet"):
        out["parapet"] = "parapet_white" if painted else "parapet_brick"
    return out


def wall_kit(reading: L.Reading, wall: str | None) -> str | None:
    if wall is None:
        return "stone" if reading.kind == "church" else None
    kit = WALL_KIT[wall]
    if wall == "wall.red_brick" and (reading.era or 1800) >= 1840:
        return "victorian"                                # no Victorian kit: the palette's deep red brick
    if wall == "wall.render" and CREAM.search(reading.body):
        return "render_cream"
    if wall == "wall.timber_frame" and NOGGING.search(reading.body):
        return "timber_brick"
    if reading.kind == "barn" and STADDLE.search(reading.body):
        return "granary_staddle"            # on staddles, whatever its walling ("Timber-framed small granary on 9 staddles")
    if wall == "wall.weatherboard" and reading.kind == "barn":
        return "weatherboard_black"
    return kit


def design(reading: L.Reading, entry: dict, lidar: dict | None, palette: dict) -> dict:
    spec = {"entry": entry["entry"], "name": entry["name"], "grade": entry["grade"], "kind": reading.kind,
            "era": reading.era, "notes": []}
    # walls, roof and windows
    wall = wall_of(reading)
    if reading.kind == "barn" and wall == "wall.timber_frame" and reading.has("wall.weatherboard"):
        wall = "wall.weatherboard"      # a barn's frame is seen as its boarding ("timber-framed ... weather-boarded walls")
    base = wall_kit(reading, wall)
    roof = _first(reading, list(ROOF_KIT))
    if base or roof:
        base = base or (lidar or {}).get("style") or "red_brick"
        window = window_of(reading)
        spec["style"] = f"uk_{base}__{ROOF_KIT.get(roof, '')}__{window}"
        spec["wall"], spec["roof"], spec["window"] = wall, roof, window
    if reading.kind == "barn" and not any(f.startswith("window.") for f in reading.features()):
        spec["blank"] = True            # a barn or granary whose listing names no windows has none
    # storeys
    text, seen = reading.storeys, (lidar or {}).get("levels")
    if text and (seen is None or abs(text - seen) <= MAX_LEVEL_GAP or (reading.attic and seen == text + 1)):
        spec["levels"] = text
        spec["levels_from"] = "listing" if seen in (None, text) else f"listing (LiDAR {seen})"
    elif seen:
        spec["levels"] = seen
        spec["levels_from"] = "LiDAR"
        if text:
            spec["notes"].append(f"listing says {text} storeys, LiDAR {seen}: LiDAR kept")
    spec["attic"] = reading.attic
    # front door
    door = next((d for f, d in DOOR_FOR if reading.has(f)), None)
    # (a barn's door, unnamed, is boarded: a fanlit hardwood door went on a granary)
    spec["door"] = door or ("oak_cottage" if reading.kind == "barn" else _by_era(DOOR_BY_ERA, reading.era))
    spec["door_from"] = "listing" if door else "era"
    spec["door_surround"] = surround_of(reading)
    spec["wall_trim"] = trims_of(reading, spec.get("style"))
    # what is still missing for it
    spec["needs"] = [f for f in reading.features() if AS.BY_ID.get(f) and AS.BY_ID[f].state in ("missing", "uk_sheet")]
    return spec
