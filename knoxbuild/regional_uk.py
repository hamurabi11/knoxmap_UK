"""Regional overlay: house styles and per-building choices from outside KnoxMap.

A map folder may hold `regional.json`, written by a regional data pipeline
(for England: EPC, listing, LiDAR). Without it nothing here changes a build.

    {
      "palette": {
        "drop": ["clapboard", "logs"],
        "add": [{"name": "red_brick", "like": "brick",
                 "wall": "walls_exterior_house_01_004",
                 "shaped": "walls_exterior_house_05_032",
                 "window": "fixtures_windows_white_024",
                 "roof_from": "brick"}]
      },
      "buildings": {"624047339": {"style": "red_brick", "levels": 2}}
    }

A style is made from an existing one (`like`) with its outside wall swapped:
`wall` is the West tile of a full wall set on walls_exterior_house_01/02,
whose corner, window and door tiles follow it in the sheet's order (+1..+3,
+8..+11), and `shaped` is the first of the 32 window-shaped pieces that go
with it on sheets 05-14. A building with a special kind (shop, church...)
keeps its own rooms and inside walls and takes only the outside.

A borrowed roof (`roof_from`) brings the donor's gable caps, the wall-coloured
triangles at a gable end. `caps`, `caps_shallow` and `caps_30` give the first
tile of a cap set in the same layout (walls_exterior_roofs_NN, its shallow
pieces, walls_exterior_roofs_30_NN) to match the new wall; the gap pieces under
a cap always take the style's own wall. `chimney` is its chimney stack
(fixtures_fireplaces_01 64-70).
"""
from __future__ import annotations

import collections
import copy
import json
import math
import os
import re

FILE = "regional.json"


def load(out_dir: str) -> dict | None:
    path = os.path.join(out_dir, FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        regional = json.load(f)
    # UK assets off (uk_assets.off in the map's folder): no UK tile, vanilla stand-ins
    from . import assets_uk
    return assets_uk.vanilla(regional) if assets_uk.off(out_dir) else regional


def _tile(name: str, offset: int) -> str:
    sheet, idx = re.match(r"^(.*)_(\d+)$", name).groups()
    return f"{sheet}_{int(idx) + offset:03d}"


def wall_entry(wall: str, shaped: str | None = None) -> dict:
    """An exterior wall tile entry from a full wall set's West tile (and the
    first of its 32 window-shaped pieces), as make_style builds one."""
    tiles = {"West": wall, "North": _tile(wall, 1), "NorthWest": _tile(wall, 2),
             "SouthEast": _tile(wall, 3), "WestWindow": _tile(wall, 8),
             "NorthWindow": _tile(wall, 9), "WestDoor": _tile(wall, 10),
             "NorthDoor": _tile(wall, 11)}
    if shaped:
        for k in range(1, 17):
            tiles[f"WestWindow{k}"] = _tile(shaped, 2 * (k - 1))
            tiles[f"NorthWindow{k}"] = _tile(shaped, 2 * (k - 1) + 1)
    return {"category": "exterior_walls", "tiles": tiles}


def _recap(caps: dict, spec: dict, exterior: dict) -> None:
    """Move a style's gable caps onto the cap sets its spec names, in place:
    each tile keeps its place in its set, so only the first tile is needed."""
    groups = {"caps": lambda k: not k.startswith(("CapShallow", "CapGap")) and "30" not in k,
              "caps_shallow": lambda k: k.startswith("CapShallow"),
              "caps_30": lambda k: "30" in k}
    for field, member in groups.items():
        first = spec.get(field)
        keys = [k for k in caps if member(k)]
        if not first or not keys:
            continue
        index = {k: int(caps[k].rsplit("_", 1)[1]) for k in keys}
        base = min(index.values())
        for k in keys:
            caps[k] = _tile(first, index[k] - base)
    if "CapGapE3" in caps:
        caps["CapGapE3"] = exterior["West"]
    if "CapGapS3" in caps:
        caps["CapGapS3"] = exterior["North"]


def make_style(spec: dict, styles: dict[str, dict]) -> dict:
    style = copy.deepcopy(styles[spec.get("like", "brick")])
    style["name"] = spec["name"]
    if spec.get("wall"):
        w = spec["wall"]
        tiles = {"West": w, "North": _tile(w, 1), "NorthWest": _tile(w, 2),
                 "SouthEast": _tile(w, 3), "WestWindow": _tile(w, 8),
                 "NorthWindow": _tile(w, 9), "WestDoor": _tile(w, 10),
                 "NorthDoor": _tile(w, 11)}
        if spec.get("shaped"):
            for k in range(1, 17):
                tiles[f"WestWindow{k}"] = _tile(spec["shaped"], 2 * (k - 1))
                tiles[f"NorthWindow{k}"] = _tile(spec["shaped"], 2 * (k - 1) + 1)
        style["exterior"] = {"category": "exterior_walls", "tiles": tiles}
    if spec.get("window"):
        style["window"] = {"category": "windows",
                           "tiles": {"West": spec["window"], "North": _tile(spec["window"], 1)}}
    for part in ("roof", "interior", "trim", "shutters", "curtains", "grime"):
        donor = spec.get(f"{part}_from")
        if donor:
            style[part] = copy.deepcopy(styles[donor][part])
    caps = ((style.get("roof") or {}).get("caps") or {}).get("tiles")
    if caps:
        _recap(caps, spec, style["exterior"]["tiles"])
    # A roof covering of its own (listed_building_maker's roof_kit.py): the
    # donor's slopes and tops with these sheets in place of its roofs_NN and
    # roofs_30_NN (every vanilla roof set has its pieces in the same slots).
    for key, pattern in (("roof_sheet", r"^roofs_\d+_(\d+)$"), ("roof_sheet_30", r"^roofs_30_\d+_(\d+)$")):
        if spec.get(key) and style.get("roof"):
            for group in ("slopes", "tops"):
                tiles = (style["roof"].get(group) or {}).get("tiles") or {}
                for k, v in tiles.items():
                    m = re.match(pattern, v)
                    if m:
                        tiles[k] = f"{spec[key]}_{int(m.group(1)):03d}"
    if spec.get("flat_tops") and style.get("roof"):
        # One tile for every flat top: a superstore's flat roof is the tar vanilla's
        # commercial buildings have, not its pitched parts' covering laid flat.
        tops = (style["roof"].get("tops") or {}).get("tiles") or {}
        for k in tops:
            tops[k] = spec["flat_tops"]
    if spec.get("chimney"):
        style["chimney"] = spec["chimney"]
    elif spec.get("chimney") is False:
        style["no_chimney"] = True       # no stack (a superstore): build.py lays none
    if spec.get("grime") is False:
        # No soot at the foot of each storey: lost on brick, a black line on
        # white render or paint (listed_building_maker's light wall kits).
        style["grime"] = None
    if spec.get("shutters") is False:
        style["shutters"] = None         # English sashes have no outside shutters
    if spec.get("steep") and style.get("roof"):
        # Without its 30-degree tiles a style's roofs are built at 45 degrees
        # (tbx.py picks Peak30 only when they are there): steep, like an
        # English tiled roof, and no flat strip on an even width.
        for group in ("slopes", "caps", "tops"):
            tiles = (style["roof"].get(group) or {}).get("tiles")
            if tiles:
                for key in [k for k in tiles if "30" in k]:
                    del tiles[key]
    return style


def apply_palette(overlay: dict | None) -> None:
    """Add and drop house styles, in place, before any building is styled."""
    if not overlay or not overlay.get("palette"):
        return
    from . import catalog as C
    pal = overlay["palette"]
    by_name = {s["name"]: s for s in C.HOUSE_STYLES}
    added = [make_style(spec, by_name) for spec in pal.get("add", [])]
    by_name.update({s["name"]: s for s in added})
    overlay["_styles"] = by_name
    if pal.get("shutters") is False:
        # No outside shutters on any style, the stand-ins too (QA, 2026-10-19:
        # American shutters on every stand-in house were the worst of the look).
        for s in by_name.values():
            s["shutters"] = None
    keep = [s for s in C.HOUSE_STYLES if s["name"] not in set(pal.get("drop", []))]
    # A style only for the buildings that name it (a listed building's own
    # wall and roof, listed_building_maker) stays out of the pool Build picks
    # other buildings' styles from.
    C.HOUSE_STYLES[:] = keep + [s for s, spec in zip(added, pal.get("add", [])) if not spec.get("by_id_only")]


def _entry(overlay, osm_id):
    if not overlay or osm_id is None:
        return None
    return (overlay.get("buildings") or {}).get(str(osm_id))


# Build 42's storeys above ground are levels 0-31; the top one is left for the
# roof floor (a spire's or a parapet's pieces), so a building has at most 31.
B42_MAX_STOREYS = 31


def levels_for(overlay, osm_id, levels: int, max_levels: int) -> int:
    """The overlay's storeys, held to the map's max_levels - or, for a building
    marked full_height (a landmark tower: Parliament's, user 2026-10-21), to
    the game's own ceiling."""
    e = _entry(overlay, osm_id)
    if e and e.get("levels"):
        top = B42_MAX_STOREYS if e.get("full_height") else max_levels
        return max(1, min(int(e["levels"]), top))
    return levels


_SLOPE_DEPTHS = {1: "One", 2: "Two"}
FULL_PEAK = 6      # a 45-degree gable peaks fully up to this many tiles across
PEAK30_MAX = 13    # a 30-degree gable, up to this (tbx.PEAK30_MAX_ACROSS + 2)
SLIVER = 2         # a roof strip this narrow cannot pitch along its length
LEAN_BACKED = 0.9  # a lean-to's high edge must stand against something this much of its length
ATTIC_EAVES = 3    # an attic's roof slopes run this far in from each eave: a storey high at depth Three
DORMER_EVERY = 4   # an attic's dormers: one per this many tiles of its street eave,
DORMER_WIDE = 2    # each this wide along the eave, a room pocket ATTIC_EAVES - 1 deep
DORMER_DEPTH = "Two"


def _gable(pieces, kind, box, st, open_end, roof30, hip=False):
    """A gable over `box` whose ridge runs `kind` (WE or NS): 45 degrees while
    it peaks across (up to FULL_PEAK + 1 tiles), else 30 degrees on an odd
    width with a flat strip on the north or west side, as tbx.py does.
    `hip` (a listed building whose listing says hipped: listed_building_maker):
    a 30-degree hip (Peak30Quad) instead where the game can build one - both
    ends free, at most HIP_MAX_ACROSS across (odd), an odd length (tbx.py's
    rule for the lot compiler)."""
    from .tbx import _NO_CAPS, _peak_depth, HIP_MAX_ACROSS
    bx, by, bw, bh = box
    across = bh if kind == "WE" else bw

    def caps(x, y, w, h):
        cap = dict(_NO_CAPS)
        if kind == "WE":
            cap["cappedW"] = open_end([(x - 1, yy) for yy in range(y, y + h)])
            cap["cappedE"] = open_end([(x + w, yy) for yy in range(y, y + h)])
        else:
            cap["cappedN"] = open_end([(xx, y - 1) for xx in range(x, x + w)])
            cap["cappedS"] = open_end([(xx, y + h) for xx in range(x, x + w)])
        return cap

    odd = across if across % 2 else across - 1
    if hip and roof30 and 3 <= odd <= HIP_MAX_ACROSS and (bw if kind == "WE" else bh) % 2 == 1:
        free = caps(*box)
        if all(free.get(k) for k in (("cappedW", "cappedE") if kind == "WE" else ("cappedN", "cappedS"))):
            peak = (bx, by + across - odd, bw, odd) if kind == "WE" else (bx + across - odd, by, odd, bh)
            strip = (bx, by, bw, across - odd) if kind == "WE" else (bx, by, across - odd, bh)
            pieces.setdefault(st, []).append(("Peak30Quad", "Zero", dict(_NO_CAPS), peak))
            if strip[2] and strip[3]:
                pieces.setdefault(st - 1, []).append(("FlatTop", "Three", dict(_NO_CAPS), strip))
            return
    if across <= FULL_PEAK + 1 or not roof30:
        pieces.setdefault(st, []).append(
            ("PeakWE" if kind == "WE" else "PeakNS", _peak_depth(across), caps(*box), box))
        return
    if kind == "WE":
        peak = (bx, by + across - odd, bw, odd)
        strip = (bx, by, bw, across - odd)
    else:
        peak = (bx + across - odd, by, odd, bh)
        strip = (bx, by, across - odd, bh)
    pieces.setdefault(st, []).append(
        ("Peak30WE" if kind == "WE" else "Peak30NS", "Zero", caps(*peak), peak))
    if strip[2] and strip[3]:
        pieces.setdefault(st - 1, []).append(("FlatTop", "Three", dict(_NO_CAPS), strip))


# A lean-to is named for the way it faces, downhill; its high edge is the
# other side.
_FLIP = {"SlopeN": "SlopeS", "SlopeS": "SlopeN", "SlopeE": "SlopeW", "SlopeW": "SlopeE"}


def _high_edge(kind, x, y, w, h):
    """The tiles just past a lean-to's high edge."""
    if kind == "SlopeE":
        return [(x - 1, yy) for yy in range(y, y + h)]
    if kind == "SlopeW":
        return [(x + w, yy) for yy in range(y, y + h)]
    if kind == "SlopeS":
        return [(xx, y - 1) for xx in range(x, x + w)]
    return [(xx, y + h) for xx in range(x, x + w)]


def join_units(geo: dict, overlay) -> dict:
    """KnoxMap UK: units a landmark is made of joined into one building (an inn the
    address step split into its hotel and its restaurant). A regional entry's
    `join` lists the other units' OSM ids: their outlines are unioned into its
    own, their uses added to its, and they are not built on their own."""
    joins = {oid: e["join"] for oid, e in ((overlay or {}).get("buildings") or {}).items() if e.get("join")}
    if not joins:
        return geo
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    feats = geo["features"]
    by = {str(f.get("properties", {}).get("osm_id")): f for f in feats}
    dropped, new = set(), {}
    for oid, others in joins.items():
        f = by.get(oid)
        parts = [by[o] for o in others if o in by]
        if not f or not parts:
            continue
        eps = 1e-7                                       # a hair, in degrees: units that only touch
        g = unary_union([shape(x["geometry"]).buffer(eps) for x in [f] + parts]).buffer(-eps)
        if g.geom_type != "Polygon":
            continue                                    # not touching: left as they are (a landmark
                                                        # laid apart gives its `footprint` instead)
        props = dict(f["properties"])
        props["knox_uses"] = list(props.get("knox_uses") or []) + [
            u for x in parts for u in (x["properties"].get("knox_uses") or [])]
        new[oid] = dict(f, geometry=mapping(g), properties=props)
        dropped |= {o for o in others if o in by}
    return dict(geo, features=[new.get(str(x.get("properties", {}).get("osm_id")), x) for x in feats
                               if str(x.get("properties", {}).get("osm_id")) not in dropped])


def extend_outlines(geo: dict, overlay, proj) -> dict:
    """KnoxMap UK: parts of a landmark its outline misses (St Lawrence's north
    chapel, which neither OSM nor the LiDAR outline has): a regional entry's
    `extend`, rectangles [x, y, w, h] in map tiles, unioned into its outline.
    Or its `footprint`: rectangles that are its outline, in place of the one
    the block step laid; its `join`ed pieces are then not built."""
    entries = (overlay or {}).get("buildings") or {}
    ext = {oid: e["extend"] for oid, e in entries.items() if e.get("extend")}
    # `footprint`: the landmark's own squares, in place of the outline the block
    # step laid (the Quaker meeting house: its range and its wing)
    own = {oid: e["footprint"] for oid, e in entries.items() if e.get("footprint")}
    if not ext and not own:
        return geo
    from shapely.geometry import Polygon, mapping, shape
    from shapely.ops import unary_union

    def boxes(rects):
        out = []
        for x, y, w, h in rects:
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            out.append(Polygon([tuple(reversed(proj.to_latlon(px, py))) for px, py in corners]))
        return out
    applied = set()
    feats = []
    for f in geo["features"]:
        oid = str(f.get("properties", {}).get("osm_id"))
        if oid in own:
            g = unary_union([b.buffer(1e-7) for b in boxes(own[oid])]).buffer(-1e-7)
            if g.geom_type == "MultiPolygon":
                # a speck the buffering leaves where two rectangles meet at a corner
                # (no wider than the buffer); a real part is kept, and then refused
                g = unary_union([p for p in g.geoms if p.minimum_rotated_rectangle.length > 1e-5])
            if g.geom_type == "Polygon":
                f = dict(f, geometry=mapping(g))
                applied.add(oid)
                if ext.get(oid):
                    print(f"landmark {oid}: its footprint is its outline - its `extend` is not used")
            else:
                print(f"landmark {oid}: its footprint's rectangles do not touch - the outline is kept")
        elif ext.get(oid):
            g = unary_union([shape(f["geometry"]).buffer(1e-7)] + [b.buffer(1e-7) for b in boxes(ext[oid])]).buffer(-1e-7)
            if g.geom_type == "Polygon":
                f = dict(f, geometry=mapping(g))
        feats.append(f)
    joined = {o for oid in applied for o in entries[oid].get("join") or ()}
    return dict(geo, features=[f for f in feats if str(f.get("properties", {}).get("osm_id")) not in joined])


def landmark_of(overlay, osm_id, x0: int, y0: int) -> dict | None:
    """A landmark's description and its tiles (listed_building_maker apply.py,
    from landmarks/<entry>_*.json), with where Build put its box (landmark_uk.py)."""
    e = _entry(overlay, osm_id)
    if not e or not e.get("landmark"):
        return None
    return dict(e["landmark"], at=[x0, y0])


def roof_plan(overlay, osm_id, x0: int, y0: int, mask, roof30: bool = True) -> dict | None:
    """Storeys per part and the roof pieces over them, for one placed building.

    The overlay's "roof" list holds parts in map tiles, [x, y, w, h, form,
    storeys], form one of WE / NS (a gable whose ridge runs that way), flat,
    or SlopeN/E/S/W (a lean-to). Returned, in this building's own tiles:
      levels       the most storeys any part has
      level_masks  per storey, the tiles that storey covers
      pieces       {floor: [(RoofType, Depth, caps, (x, y, w, h))]}
    Pitched roofs sit on the floor above their part's top storey, flat ones on
    that storey itself, as tbx.py places KnoxMap's own.
    """
    import numpy as np
    from .layout import roof_rects
    from .tbx import _NO_CAPS, _peak_depth

    e = _entry(overlay, osm_id)
    if not e or not e.get("roof"):
        return None
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    owner = np.full((h, w), -1, dtype=int)
    parts = e["roof"]
    for i, (px, py, pw, ph, _form, _st) in enumerate(parts):
        xa, ya = max(0, px - x0), max(0, py - y0)
        xb, yb = min(w, px - x0 + pw), min(h, py - y0 + ph)
        if xa < xb and ya < yb:
            owner[ya:yb, xa:xb] = np.where(m[ya:yb, xa:xb], i, owner[ya:yb, xa:xb])
    if (owner[m] < 0).all():
        return None
    # Tiles no part covers go with their nearest part. (Shifted without
    # wrapping round: np.roll would hand the top row the bottom row's part.)
    def shifted(a, dy, dx):
        out = np.full_like(a, -1)
        ys = slice(max(dy, 0), h + min(dy, 0))
        xs = slice(max(dx, 0), w + min(dx, 0))
        yo = slice(max(-dy, 0), h + min(-dy, 0))
        xo = slice(max(-dx, 0), w + min(-dx, 0))
        out[ys, xs] = a[yo, xo]
        return out

    for _ in range(h + w):
        todo = m & (owner < 0)
        if not todo.any():
            break
        grown = owner.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = shifted(owner, dy, dx)
            grown = np.where(todo & (grown < 0) & (sh >= 0), sh, grown)
        if (grown == owner).all():
            break
        owner = grown
    # A gable or flat part no more than SLIVER tiles across, beside a part
    # of the same storeys, is the LiDAR's eaves and belongs to that part.
    # Left alone, two side by side (Odiham's High Street rows) became a
    # lean-to running up into a one-tile flat strip: open along its top.
    # Folded into the neighbour with the longest shared edge, repeatedly, so
    # a run of slivers ends up in the range they lie along.
    from collections import Counter
    folding = True
    while folding:
        folding = False
        for i, p in enumerate(parts):
            if p[4] not in ("WE", "NS", "flat"):
                continue
            reg = (owner == i) & m
            if not reg.any():
                continue
            ys, xs = np.nonzero(reg)
            if min(xs.max() - xs.min(), ys.max() - ys.min()) + 1 > SLIVER:
                continue
            edge = Counter()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                beside = shifted(owner, dy, dx)[reg]
                edge.update(int(j) for j in beside[(beside >= 0) & (beside != i)])
            near = [j for j, _n in edge.most_common()
                    if int(parts[j][5]) == int(p[5]) and parts[j][4] in ("WE", "NS", "flat")]
            if near:
                owner[reg] = near[0]
                folding = True
    level_of = np.zeros((h, w), dtype=int)
    for i, p in enumerate(parts):
        level_of[owner == i] = int(p[5])
    # Attics (Building_types roof_plan.py `attic`): rooms in the roof, the
    # way vanilla B42 builds them (a house in Muldraugh, cell 6,35). The
    # part's top storey is the attic: the roof slopes from both eaves lie on
    # the attic floor and are a storey high ATTIC_EAVES in, where the attic's
    # own walls stand, hidden behind them; the peak over the middle is on the
    # floor above. The eave strips' top storey is the one below.
    attics = {}
    pockets = {}
    for rect in e.get("attic") or []:
        for i, p in enumerate(parts):
            if list(p[:4]) == list(rect[:4]) and p[4] in ("WE", "NS"):
                reg = (owner == i) & m
                if not reg.any():
                    continue
                ys, xs = np.nonzero(reg)
                ax, ay, aw, ah = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
                if int(reg.sum()) != aw * ah:
                    # Not solid: the part's own box (roof_plan.py) reached
                    # past the real mask here (an L-shaped or notched
                    # building). The slopes and dormers below assume a solid
                    # rectangle, so this part skips the attic and is roofed
                    # the plain way, through roof_rects below as usual.
                    continue
                across = aw if p[4] == "NS" else ah
                # roof_plan.py's ATTIC_MIN_ACROSS (2*ATTIC_EAVES + 5) already
                # gates this before an attic is planned; this is only a floor
                # against the inner gable going empty or negative (>= 1 wide)
                # if a plan from an older run ever supplies a narrower one.
                if across < 2 * ATTIC_EAVES + 3:
                    continue
                eaves = np.zeros_like(m)
                if p[4] == "NS":
                    eaves[:, ax:ax + ATTIC_EAVES] = True
                    eaves[:, ax + aw - ATTIC_EAVES:ax + aw] = True
                else:
                    eaves[ay:ay + ATTIC_EAVES, :] = True
                    eaves[ay + ah - ATTIC_EAVES:ay + ah, :] = True
                level_of[reg & eaves] = int(p[5]) - 1
                attics[i] = (ax, ay, aw, ah)
                # Dormers on the street eave: pockets of the attic out into
                # the slope, the eave's last tile left as roof in front.
                side = rect[4] if len(rect) > 4 else ""
                along = ah if p[4] == "NS" else aw
                n = max(0, (along - 2) // DORMER_EVERY) if side else 0
                spots = [round((k + 0.5) * along / n - DORMER_WIDE / 2) for k in range(n)] if n else []
                cells = []
                for s0 in spots:
                    for t in range(s0, s0 + DORMER_WIDE):
                        for d in range(ATTIC_EAVES - 1):
                            if side == "E":
                                cells.append((ax + aw - ATTIC_EAVES + d, ay + t))
                            elif side == "W":
                                cells.append((ax + ATTIC_EAVES - 1 - d, ay + t))
                            elif side == "S":
                                cells.append((ax + t, ay + ah - ATTIC_EAVES + d))
                            else:
                                cells.append((ax + t, ay + ATTIC_EAVES - 1 - d))
                if cells and all(0 <= x < w and 0 <= y < h and m[y, x] for x, y in cells):
                    for x, y in cells:
                        level_of[y, x] = int(p[5])
                    pockets[i] = (side, spots)
    levels = int(level_of[m].max())
    level_masks = [(m & (level_of > lvl)).tolist() for lvl in range(levels)]
    # An arch (KnoxMap UK, Building_types/src/regional/passages.py): where a
    # road or path passes through the building, the ground floor is left out
    # and the floors above bridge it - only with a floor above to bridge it.
    passage = [(x - x0, y - y0) for x, y in e.get("passage") or ()]
    passage = [(x, y) for x, y in passage if 0 <= x < w and 0 <= y < h and m[y, x] and level_of[y, x] >= 2]
    if passage and levels >= 2:
        ground = m & (level_of > 0)
        for x, y in passage:
            ground[y, x] = False
        level_masks[0] = ground.tolist()

    def covered(x, y):
        return 0 <= x < w and 0 <= y < h and m[y, x]

    pieces: dict[int, list] = {}

    def outside(cells):
        return sum(1 for x, y in cells if not covered(x, y)) * 2 > len(cells)

    def backed(cells, st):
        """Most of these tiles hold something a lean-to roofed at storey
        `st` can run up against: a taller part, or an equal pitched one."""
        good = sum(1 for x, y in cells if covered(x, y) and owner[y, x] >= 0 and (
            level_of[y, x] > st or (level_of[y, x] == st and parts[owner[y, x]][4] in ("WE", "NS"))))
        return good >= LEAN_BACKED * len(cells)

    for i, (_px, _py, _pw, _ph, form, st) in enumerate(parts):
        region = (owner == i) & m
        if not region.any():
            continue
        st = int(st)

        def open_end(cells):
            # Capped where the roof ends at the outside, or over a lower part,
            # or over a flat or lean-to part as high (in game, 2026-10-19: left
            # open, the gable showed the ceiling through it).
            return sum(1 for x, y in cells
                       if not covered(x, y) or level_of[y, x] < st
                       or (level_of[y, x] == st and owner[y, x] >= 0 and parts[owner[y, x]][4] not in ("WE", "NS"))
                       ) * 2 > len(cells)

        if i in attics:
            ax, ay, aw, ah = attics[i]
            side, spots = pockets.get(i, ("", []))
            ends = (lambda cells: sum(1 for x, y in cells
                                      if not covered(x, y) or level_of[y, x] < st - 1) * 2 > len(cells))
            def slope(kind, box, along_ns):
                """The slope along one eave, in runs between its dormers:
                each dormer a Dormer piece over its pocket and a low slope
                over the eave tile in front of it."""
                bx, by, bw, bh = box
                length = bh if along_ns else bw
                cuts = [0]
                for s0 in (sorted(spots) if kind[-1] == side else []):
                    cuts += [s0, s0 + DORMER_WIDE]
                cuts.append(length)
                for k, (a, b) in enumerate(zip(cuts, cuts[1:])):
                    if b <= a:
                        continue
                    seg = (bx, by + a, bw, b - a) if along_ns else (bx + a, by, b - a, bh)
                    cap = dict(_NO_CAPS)
                    if k % 2 == 0:
                        if along_ns:
                            cap["cappedN"] = a == 0 and ends([(x, ay - 1) for x in range(bx, bx + bw)])
                            cap["cappedS"] = b == length and ends([(x, ay + ah) for x in range(bx, bx + bw)])
                        else:
                            cap["cappedW"] = a == 0 and ends([(ax - 1, y) for y in range(by, by + bh)])
                            cap["cappedE"] = b == length and ends([(ax + aw, y) for y in range(by, by + bh)])
                        pieces.setdefault(st - 1, []).append((kind, "Three", cap, seg))
                        continue
                    # A dormer: its roof over the pocket, the eave in front.
                    sx, sy, sw, sh = seg
                    eave = {"E": (sx + sw - 1, sy, 1, sh), "W": (sx, sy, 1, sh),
                            "S": (sx, sy + sh - 1, sw, 1), "N": (sx, sy, sw, 1)}[side]
                    pocket = {"E": (sx, sy, sw - 1, sh), "W": (sx + 1, sy, sw - 1, sh),
                              "S": (sx, sy, sw, sh - 1), "N": (sx, sy + 1, sw, sh - 1)}[side]
                    pieces.setdefault(st - 1, []).append((kind, "One", dict(_NO_CAPS), eave))
                    pieces.setdefault(st - 1, []).append(("Dormer" + side, DORMER_DEPTH, dict(_NO_CAPS), pocket))

            if form == "NS":
                slope("SlopeW", (ax, ay, ATTIC_EAVES, ah), True)
                slope("SlopeE", (ax + aw - ATTIC_EAVES, ay, ATTIC_EAVES, ah), True)
                inner = (ax + ATTIC_EAVES, ay, aw - 2 * ATTIC_EAVES, ah)
            else:
                slope("SlopeN", (ax, ay, aw, ATTIC_EAVES), False)
                slope("SlopeS", (ax, ay + ah - ATTIC_EAVES, aw, ATTIC_EAVES), False)
                inner = (ax, ay + ATTIC_EAVES, aw, ah - 2 * ATTIC_EAVES)
            _gable(pieces, form, inner, st, open_end, roof30)
            continue

        for rx, ry, rw, rh, _c in roof_rects(region.astype(int).tolist()):
            kind = form
            if kind in ("WE", "NS", "flat") and min(rw, rh) <= SLIVER:
                # A strip too narrow to pitch along its length: a lean-to
                # down to the outside if it is on the building's edge.
                if rh <= SLIVER and rw > rh:
                    kind = ("SlopeN" if outside([(x, ry - 1) for x in range(rx, rx + rw)]) else
                            "SlopeS" if outside([(x, ry + rh) for x in range(rx, rx + rw)]) else "flat")
                elif rw <= SLIVER:
                    kind = ("SlopeW" if outside([(rx - 1, y) for y in range(ry, ry + rh)]) else
                            "SlopeE" if outside([(rx + rw, y) for y in range(ry, ry + rh)]) else "flat")
            if kind in ("WE", "NS"):
                # One crest per range: the planner has already split a range
                # wherever the LiDAR shows a valley. Narrow ranges are 45-degree
                # gables; one too wide to peak at 45 is a 30-degree gable, which
                # peaks across up to 13 tiles. Only wider than that is it split
                # into gables side by side.
                across = rh if kind == "WE" else rw
                n = 1 if across <= PEAK30_MAX else math.ceil(across / PEAK30_MAX)
                cuts = [across * k // n for k in range(n + 1)]
                for a, b in zip(cuts, cuts[1:]):
                    if kind == "WE":
                        box = (rx, ry + a, rw, b - a)
                    else:
                        box = (rx + a, ry, b - a, rh)
                    _gable(pieces, kind, box, st, open_end, roof30, hip=bool(e.get("hip")))
                continue
            cap = dict(_NO_CAPS)
            if kind.startswith("Slope"):
                # A lean-to is open along its high edge: it must run up
                # against something taller - a taller part, or the slope of
                # an equal part's pitched roof. Otherwise it faces the other
                # way if that side is backed, else it becomes a gable (it
                # closes its own ends), or, too narrow to pitch, flat.
                if not backed(_high_edge(kind, rx, ry, rw, rh), st):
                    flipped = _FLIP[kind]
                    if backed(_high_edge(flipped, rx, ry, rw, rh), st):
                        kind = flipped
                    elif min(rw, rh) > SLIVER:
                        _gable(pieces, "NS" if kind in ("SlopeE", "SlopeW") else "WE",
                               (rx, ry, rw, rh), st, open_end, roof30)
                        continue
                    else:
                        pieces.setdefault(st - 1, []).append(("FlatTop", "Three", cap, (rx, ry, rw, rh)))
                        continue
                across = rw if kind in ("SlopeE", "SlopeW") else rh
                pieces.setdefault(st, []).append(
                    (kind, _SLOPE_DEPTHS.get(across, "Three"), cap, (rx, ry, rw, rh)))
            else:
                pieces.setdefault(st - 1, []).append(("FlatTop", "Three", cap, (rx, ry, rw, rh)))
    # A hall (a church's nave and aisles) is open from the ground floor to
    # its roof: its storeys above the ground are walls with no floor. The
    # tower - the tallest flat part - keeps its floors and stairs, and may be
    # built in a wall of its own (a brick tower on a flint church).
    void: dict[int, set] = {}
    accent = None
    if e.get("hall"):
        for i, p in enumerate(parts):
            if p[4] == "flat" and int(p[5]) >= levels:
                if e.get("tower_wall"):
                    ys, xs = np.nonzero((owner == i) & m)
                    accent = {"cells": set(zip(xs.tolist(), ys.tolist())),
                              "entry": wall_entry(*e["tower_wall"])}
                continue
            ys, xs = np.nonzero((owner == i) & m)
            cells = set(zip(xs.tolist(), ys.tolist()))
            for lvl in range(1, int(p[5])):
                void.setdefault(lvl, set()).update(cells)
    # Dormers sit on the top roof's slopes, on the same floor as that roof.
    for dx, dy, dw, dh, kind in e.get("dormers") or []:
        bx, by = dx - x0, dy - y0
        if 0 <= bx and 0 <= by and bx + dw <= w and by + dh <= h and m[by:by + dh, bx:bx + dw].all():
            pieces.setdefault(levels, []).append((kind, "Two", dict(_NO_CAPS), (bx, by, dw, dh)))
    # A house's upper floors in a wall of their own (export_regional
    # house_parts `upper_wall`: brick below, roughcast or tile-hanging above),
    # and its bays (`bay`).
    upper = None
    if e.get("upper_wall"):
        ys, xs = np.nonzero(m)
        cells = set(zip(xs.tolist(), ys.tolist()))
        # Part by part: an L-shaped house's wing runs the other way (one ridge
        # for the whole house put its panel on the wing's gable end).
        ridges = []
        for i, p in enumerate(parts):
            if p[4] in ("WE", "NS"):
                py_, px_ = np.nonzero((owner == i) & m)
                ridges.append((set(zip(px_.tolist(), py_.tolist())), p[4]))
        upper = upper_accents(wall_entry(*e["upper_wall"]), cells,
                              ridges if e.get("upper_sides") == "long" else None)
    # Each part's squares as they ended up, in this building's tiles: its box,
    # form, storeys and how many squares (garage_for finds the garage wing).
    shapes = []
    for i, p in enumerate(parts):
        py_, px_ = np.nonzero((owner == i) & m)
        if len(px_):
            shapes.append((int(px_.min()), int(py_.min()), int(px_.max()), int(py_.max()), p[4], int(p[5]),
                           int(len(px_))))
    return {"levels": levels, "level_masks": level_masks, "pieces": pieces, "void": void,
            "accent": accent, "upper": upper, "bay": bay_tiles(e.get("bay")), "parts": shapes}


# Chimney stacks: the game's own, as Riverside has them (B42): one tile, a
# stack a little over a storey tall standing on the roof's floor, with a pair
# of pots on top of it, above the fireplace. fixtures_fireplaces_01 64-70 are
# the stack in seven bricks and stones, 72 and 73 the pots. A palette style
# names its stack (`chimney`); others take CHIMNEY_FOR their name, else red.
CHIMNEY_STACK = "fixtures_fireplaces_01_64"     # pink-red brick
CHIMNEY_FOR = {"brick": "fixtures_fireplaces_01_67", "render": "fixtures_fireplaces_01_66",
               "panel": "fixtures_fireplaces_01_65", "stucco": "fixtures_fireplaces_01_69"}
CHIMNEY_POTS = ("fixtures_fireplaces_01_72", "fixtures_fireplaces_01_73")
STACK_LONG = 9                                  # a range this long gets a stack at each end
STACK_APART = 5                                 # tiles between stacks, at least
STACK_MAX = 3                                   # stacks on one building, at most
# How high a ridge stands, in storeys: a 45-degree gable by its BuildingEd
# depth (thirds of a storey), a 30-degree one by its width (a 45-degree roof
# rises a third of a storey a tile; a 30-degree one tan 30 of that).
_DEPTH_STOREYS = {"Point5": 1 / 6, "One": 1 / 3, "OnePoint5": 1 / 2, "Two": 2 / 3,
                  "TwoPoint5": 5 / 6, "Three": 1.0}
_RISE30 = math.tan(math.radians(30)) / 3
STACK_HIGH_RIDGE = 1.0   # over a ridge higher than this the stack stands a floor up


def chimney_for(style: dict | None) -> str:
    """The stack tile for a house style."""
    style = style or {}
    return style.get("chimney") or CHIMNEY_FOR.get(style.get("name") or "", CHIMNEY_STACK)


def chimney_tiles(roof: dict | None, x0: int, y0: int, stack: str = CHIMNEY_STACK) -> list:
    """Stacks on the ridges of a planned roof's top ranges, in map tiles, as
    (x, y, level, layer, tile) for structures.pack_loose."""
    if not roof:
        return []
    top = roof["levels"]
    spots = []
    # Longest ranges first, so the main roof gets its stacks before a wing.
    ranges = sorted(roof["pieces"].get(top, []), key=lambda p: -p[3][2] * p[3][3])
    for roof_type, depth, _cap, (rx, ry, rw, rh) in ranges:
        if roof_type in ("PeakWE", "Peak30WE") and rw >= 4 and rh >= 3:
            ridge = ry + (rh - 1) // 2
            here = [(rx + 1, ridge)] + ([(rx + rw - 2, ridge)] if rw >= STACK_LONG else [])
            across = rh
        elif roof_type in ("PeakNS", "Peak30NS") and rh >= 4 and rw >= 3:
            ridge = rx + (rw - 1) // 2
            here = [(ridge, ry + 1)] + ([(ridge, ry + rh - 2)] if rh >= STACK_LONG else [])
            across = rw
        else:
            continue
        high = (across / 2 * _RISE30) if roof_type.startswith("Peak30") else _DEPTH_STOREYS.get(depth, 1.0)
        for s in here:
            # Side-by-side gables of a double-pile roof share their stacks.
            if len(spots) < STACK_MAX and all(abs(s[0] - t[0]) + abs(s[1] - t[1]) > STACK_APART
                                              for t, _h in spots):
                spots.append((s, high))
    # The stack stands on the roof's floor, as in the game's own towns, and
    # clears a ridge up to a storey high; over a higher ridge (a wide
    # 30-degree roof) it stands a floor up, its foot hidden in the roof.
    # (It was a box of brick walls, the same red on every house.)
    out = []
    for n, ((sx, sy), high) in enumerate(spots):
        z = top if high <= STACK_HIGH_RIDGE else top + 1
        out.append((x0 + sx, y0 + sy, z, "Furniture", stack))
        out.append((x0 + sx, y0 + sy, z, "Furniture2", CHIMNEY_POTS[(sx + sy + n) % 2]))
    return out


# Plinths, bands and cornices (listed_building_maker/makers/trim_kit.py, sheet
# wall_trim_uk_01): each a west and a north piece, at these slots.
TRIM_SHEET = "wall_trim_uk_01"
TRIMS = {"plinth_black": 0, "plinth_brick": 2, "band_white": 4, "band_stone": 6, "cornice_white": 8,
         "cornice_stone": 10, "parapet_brick": 12, "parapet_white": 14}
# The housing styles' plinths (asset_creator/housing_kit/plinths.py; a style's
# `plinth`, export_regional house_parts), each in its own wall kit's brick.
HOUSE_TRIM_SHEET = "wall_trim_uk_02"
# Four pieces each: west wall, north wall, both (inside corner), outside corner's post.
HOUSE_TRIMS = {"dpc_buff70": 0, "dpc_orange80": 4, "dpc_red50": 8, "dpc_rededw": 12, "plinth_rededw": 16,
               "plinth_flemish": 20,
               # council housing, flats, bungalows, 1980s terraces (2026-10-19)
               "dpc_iwc": 24, "dpc_fletton": 28, "dpc_brown70": 32, "dpc_buff80": 36, "dpc_red80": 40}


# Downpipes (listed_building_maker/makers/drain_kit.py, sheet drains_uk_01):
# [side: west, north][end of the wall run: start, end][foot, middle, top, one storey].
DRAIN_SHEET = "drains_uk_01"
DRAINS = [[[f"{DRAIN_SHEET}_{4 * (2 * s + a) + p:03d}" for p in range(4)] for a in range(2)] for s in range(2)]


UNIT_KEYS = ("style", "blank_upper", "blank", "loading", "upper_wall", "upper_sides")


def style_for(overlay, osm_id, kind: str | None, style: dict) -> dict:
    e = _entry(overlay, osm_id)
    if e and e.get("loading") and (e["loading"].get("kind") or "industrial") not in ("*", kind):
        # an industrial unit's look only where Build builds a works (a school's or a
        # hospital's buildings are works to the shops step too: Alton College's were clad),
        # a farm building's where it builds a barn
        e = {k: v for k, v in e.items() if k not in UNIT_KEYS}
    style = _shop_front(overlay, e, _styled(overlay, e, kind, style))
    # A building's own wall trims (a listed building's: `wall_trim`
    # {"plinth": "plinth_black", ...}), as (west, north) sprites for tbx.py.
    trims = {}
    for k, v in ((e or {}).get("wall_trim") or {}).items():
        sheet, i, n = (TRIM_SHEET, TRIMS[v], 2) if v in TRIMS else (HOUSE_TRIM_SHEET, HOUSE_TRIMS.get(v), 4)
        if i is not None:
            trims[k] = tuple(f"{sheet}_{i + j:03d}" for j in range(n))
    if trims:
        style = dict(style)
        style["wall_trim"] = trims
    # A house's upper wall and bays (export_regional house_parts) for tbx.py
    # when the building has no roof plan of its own (a unit split from a
    # terrace: QA, 2026-10-19, 861 of Alton's had neither).
    if (e or {}).get("upper_wall") or (e or {}).get("bay"):
        style = dict(style)
        if e.get("upper_wall"):
            style["house_upper"] = {"entry": wall_entry(*e["upper_wall"]), "sides": e.get("upper_sides")}
        if e.get("bay"):
            style["house_bay"] = bay_tiles(e["bay"])
    if (e or {}).get("drains"):
        style = dict(style)
        style["drains"] = DRAINS
    if (e or {}).get("rooms"):
        # what the room solver needs (rooms_uk.py): its look, form, EPC rooms
        style = dict(style, uk_rooms=dict(e["rooms"], osm_id=str(osm_id)))
    if overlay is not None:
        style = dict(style, uk_map=True)      # a UK map's building (pillars_uk, build._make_one)
    return style


def _styled(overlay, e, kind: str | None, style: dict) -> dict:
    if not e or not e.get("style"):
        return style
    chosen = (overlay.get("_styles") or {}).get(e["style"])
    if chosen is None:
        return style
    if kind is None:
        return chosen
    # A shop or church keeps its rooms and inside; only the outside changes.
    mixed = dict(style)
    # (Grime is the outside too: a light wall kit has none - listed_building_maker.)
    for part in ("exterior", "window", "roof", "trim", "shutters", "grime", "chimney", "no_chimney"):
        if part in chosen:
            mixed[part] = chosen[part]
    mixed["name"] = f"{style.get('name', kind)}+{e['style']}"
    return mixed


def _shop_front(overlay, e, style: dict) -> dict:
    """A shop's storefront (with Erika's Tiles): the colour's sheet the data
    picked (`shopfront`, Building_types export_regional.py) and, when the
    palette says so, no signs (Erika's are American shop names)."""
    front = (e or {}).get("shopfront")
    signs = ((overlay or {}).get("palette") or {}).get("shop_signs", True)
    uk = (e or {}).get("uk_front")
    if (not front and signs and not uk and not (e or {}).get("blank_upper") and not (e or {}).get("superstore")
            and not (e or {}).get("blank")):
        return style
    style = dict(style)                  # never the shared catalogue style itself
    if front:
        style["storefront"] = front
    if not signs:
        style["shop_signs"] = False
    if (e or {}).get("blank_upper"):
        style["blank_upper"] = True      # a superstore's upper walls: cladding, no windows
    if (e or {}).get("blank"):
        style["blank"] = True            # a farm building: no windows at all (tbx.py)
    if (e or {}).get("superstore"):
        style["superstore"] = True       # ...and its glass only round its entrance (tbx.py)
    if uk:
        # KnoxMap UK: a British shop front and its fascia (asset_creator/commercial_kit:
        # shopfronts.py, fascias.py; chosen by export_regional.uk_shopfronts), laid by
        # tbx.py in place of Erika's whether or not her tiles are installed.
        style["uk_storefront"] = uk_storefront_entries(uk)
        style["uk_fascia"] = (e or {}).get("fascia")
    return style


def front_of(overlay, osm_id) -> str | None:
    """The side a regional step chose for a building's front, over its street side
    (a superstore's faces its car park: export_regional.uk_shopfronts `front`)."""
    return (_entry(overlay, osm_id) or {}).get("front")


# The trade on a shop's fascia (asset_creator/commercial_kit/names.py: from its name,
# its tags or a 1993 market town's mix) as the OSM tags uses.py makes its rooms from,
# so the sign and the loot agree. Trades with no entry (pub, spiffos, fuel) keep theirs.
TRADE_TAGS = {
    "baby": {"shop": "clothes"}, "bakery": {"shop": "bakery"}, "bank": {"amenity": "bank"},
    "bookmaker": {"office": "bookmaker"}, "butcher": {"shop": "butcher"}, "cafe": {"amenity": "cafe"},
    "car_parts": {"shop": "car_parts"}, "car_repair": {"shop": "car_repair"}, "charity": {"shop": "clothes"},
    "chemist": {"shop": "chemist"}, "chinese": {"amenity": "fast_food", "cuisine": "chinese"},
    "clothes": {"shop": "clothes"}, "convenience": {"shop": "convenience"}, "electrical": {"shop": "electronics"},
    "estate_agent": {"office": "estate_agent"}, "fast_food": {"amenity": "fast_food", "cuisine": "burger"},
    "fish_chips": {"amenity": "fast_food", "cuisine": "fish_and_chips"}, "florist": {"shop": "florist"},
    "frozen_food": {"shop": "grocery"}, "general": {"shop": "general"}, "gift": {"shop": "gift"},
    "greengrocer": {"shop": "greengrocer"}, "hairdresser": {"shop": "hairdresser"},
    "hardware": {"shop": "hardware"}, "health_food": {"shop": "deli"},
    "indian": {"amenity": "restaurant", "cuisine": "indian"}, "italian": {"amenity": "restaurant", "cuisine": "italian"},
    "jewelry": {"shop": "jewelry"}, "kebab": {"amenity": "fast_food", "cuisine": "kebab"},
    "laundry": {"shop": "laundry"}, "music": {"shop": "music"}, "newsagent": {"shop": "newsagent"},
    "off_licence": {"shop": "alcohol"}, "office": {"office": "yes"},
    "pizza": {"amenity": "fast_food", "cuisine": "pizza"}, "post_office": {"amenity": "post_office"},
    "restaurant": {"amenity": "restaurant"}, "shoes": {"shop": "shoes"}, "supermarket": {"shop": "supermarket"},
    "thai": {"amenity": "restaurant", "cuisine": "thai"}, "travel_agency": {"office": "travel_agent"},
    "variety": {"shop": "variety_store"}, "video": {"shop": "video"},
}
# Uses from business rates (EPC) and food hygiene (FSA) records say little ("retail",
# "office"); the fascia's trade was read from the same records' names.
VAGUE_SOURCES = ("epc", "fsa")


HEALTH = {"doctors", "dentist", "clinic", "veterinary", "hospital", "pharmacy"}


def _family(tags: dict) -> str:
    """What sort of business a use is, for telling one business's records from
    another's: health, a stay (hotel, guest house), else trade (shops, offices,
    eateries, banks: the business rates' A1/A2/A3 and B1)."""
    if tags.get("healthcare") or tags.get("amenity") in HEALTH:
        return "health"
    if tags.get("tourism"):
        return "stay"
    return "trade"


def _vague(tags: dict) -> bool:
    """A shop, office or eatery use from those records (not a surgery's, pub's or hotel's)."""
    return tags.get("knox_source") in VAGUE_SOURCES and bool(
        tags.get("shop") or tags.get("office") or tags.get("amenity") in ("restaurant", "fast_food", "cafe"))


def uses_with(overlay, osm_id, inside: list[dict]) -> list[dict]:
    """A building's use tags, with a Spiffo's first where the regional step made it
    the town's (export_regional.uk_shopfronts `spiffos`: a town with no McDonald's),
    and its vague uses (business rates, food hygiene) as its fascia's trade
    (export_regional `trade`) - or dropped where its own tags (inside[0], the
    building's) give its use."""
    from .uses import use_of
    entry = _entry(overlay, osm_id) or {}
    inside = list(inside)
    trade = TRADE_TAGS.get(entry.get("trade"))
    # A business named in OSM - its building's own tags (a building society mapped
    # on its outline) or a point - and its business rates or food hygiene record
    # are one business, not two (shops.py's rule, which did not see the building's
    # own tags): as many records go as there are named uses.
    # Only one of the same family (a shop and "retail", a vet and "doctors"): a
    # doctors' record in a florist's building is another business.
    named = collections.Counter(_family(t) for t in inside
                                if t.get("knox_source") not in VAGUE_SOURCES and use_of(t))
    gone = set()
    for i, t in enumerate(inside):
        fam = _family(t)
        if t.get("knox_source") in VAGUE_SOURCES and named[fam] > 0:
            named[fam] -= 1
            gone.add(i)
    inside = [t for i, t in enumerate(inside) if i not in gone]
    if trade:
        inside = [dict(trade, knox_source="fascia") if _vague(t) else t for t in inside]
    if entry.get("spiffos"):
        return [{"amenity": "fast_food", "brand": "Spiffo's"}] + inside
    return inside


def fascia_level(fascia: dict | None, run_length: int) -> int:
    """The storey a run's sign hangs on: the ground floor's, or the version's own
    `level` (the game's Spiffo's neon, drawn to hang a storey up)."""
    fits = [v for v in (fascia or {}).get("versions", []) if v["length"] <= run_length]
    return max(fits, key=lambda v: v["length"]).get("level", 0) if fits else 0


def uk_storefront_entries(uk: dict) -> dict:
    """A UK front's BuildingEd entries: the wall entries for a run's first, middle and
    last squares (each with the door frame for a door in it, and the corner and post),
    and its door."""
    def wall(piece, category="exterior_walls"):
        t = {"West": uk[piece]["West"], "North": uk[piece]["North"],
             "WestDoor": uk["door_frame"]["West"], "NorthDoor": uk["door_frame"]["North"],
             "NorthWest": uk["corner"], "SouthEast": uk["post"],
             "WestWindow": uk[piece]["West"], "NorthWindow": uk[piece]["North"]}
        return {"category": category, "tiles": t}
    # the glass's inside face is the same pieces, as an interior wall entry (as Erika's sets are)
    out = {p: wall(p) for p in ("start", "pane", "end")}
    out.update({f"{p}_in": wall(p, "interior_walls") for p in ("start", "pane", "end")})
    out["door"] = {"category": "doors", "tiles": dict(uk["door"])}
    return out


def fascia_tiles(fascia: dict | None, run_length: int) -> list[tuple[str, str]] | None:
    """A sign's (W, N) sprites for each square of a shop front run, in run order (low x or
    y first): its longest version that fits, centred, the plain board either side; None
    where no version fits. (The storey it hangs on: fascia_level.)"""
    if not fascia:
        return None
    fits = [v for v in fascia.get("versions", []) if v["length"] <= run_length]
    if not fits:
        return None
    v = max(fits, key=lambda v: v["length"])
    first = (run_length - v["length"]) // 2
    # (a sign with no board of its own - the game's Spiffo's neon - leaves the rest bare)
    plain = (fascia["plain"]["W"], fascia["plain"]["N"]) if fascia.get("plain") else (None, None)
    out = [plain] * run_length
    for j in range(v["length"]):
        out[first + j] = (v["W"][j], v["N"][j])
    return out


# --- 1990s English homes' front doors and TV aerials (asset_creator's art) ---------------------
# Front doors: doors_uk_01, four sprites a door as vanilla's doors are
# (closed W, closed N, open W, open N), a home's outside doors on its
# ground floor (tbx.py). How often each, by the user's brief of a 1990s
# town: most homes refitted with white uPVC or a stained hardwood door with
# a fanlight, the rest the painted panelled doors of older houses and a
# cottage's ledged oak door. A lookup by the building's id, not a guess at
# its age (no data says which door a house has).
DOOR_SHEET = "doors_uk_01"
DOOR_STYLES = ("black_victorian", "red_victorian", "blue_edwardian", "green_victorian", "cream_victorian",
               "white_upvc", "hardwood_fanlight", "oak_cottage")          # the sheet's order
DOOR_SHARE = {"white_upvc": 30, "hardwood_fanlight": 14, "black_victorian": 12, "blue_edwardian": 10,
              "red_victorian": 8, "green_victorian": 8, "oak_cottage": 12, "cream_victorian": 6}
HOMES = (None, "house", "apartment")


# Doorcases, hoods and porches (listed_building_maker/makers/doorway_kit.py,
# sheet doorways_uk_01): each a west and a north piece, at these slots.
SURROUND_SHEET = "doorways_uk_01"
SURROUNDS = {"doorcase": 0, "hood": 2, "porch": 4, "porch_timber": 6}


def surround_of(overlay, osm_id) -> tuple[str, str] | None:
    """The doorcase, hood or porch a regional step chose for this building's
    front door (a listed building's from its listing), as its (west, north) sprites."""
    name = (_entry(overlay, osm_id) or {}).get("door_surround")
    if name in HOUSE_PORCHES:
        k = HOUSE_PORCHES[name]
        return f"{HOUSE_PORCH_SHEET}_{k:03d}", f"{HOUSE_PORCH_SHEET}_{k + 1:03d}"
    k = SURROUNDS.get(name)
    return None if k is None else (f"{SURROUND_SHEET}_{k:03d}", f"{SURROUND_SHEET}_{k + 1:03d}")


# The housing styles' porches and bays (asset_creator/housing_kit/parts.py; a
# style's `porch` and `bay`, export_regional house_parts): each a west-wall
# sprite and the north-wall one after it. A bay is one storey (`single`, with
# its roof) or two (`lower` under `upper`, which has the roof); tbx.py
# `_lay_bays` lays it in front of the street front's windows.
HOUSE_PORCH_SHEET = "doorways_uk_02"
HOUSE_PORCHES = {"canopy_bracket": 0, "porch_30s": 2, "canopy_flat": 4, "canopy_tiled": 6}
BAY_SHEET = "bays_uk_01"
# A two-square bay ("wide_...") has four sprites: west first square, west
# second, north first, north second (the second a square on along the wall).
BAYS = {"vic": {"single": 0, "wide_single": 16},
        "edw": {"lower": 2, "upper": 4, "single": 6, "wide_lower": 20, "wide_upper": 24, "wide_single": 28},
        "iw": {"lower": 8, "upper": 10, "single": 12, "wide_lower": 32, "wide_upper": 36, "wide_single": 40},
        "e80": {"single": 14, "wide_single": 44}}


def upper_accents(entry: dict, cells: set, ridges: list | None = None) -> list[dict]:
    """A house's upper floors in a wall of their own (export_regional
    `upper_wall`), as tbx.py accents from the first floor up: over every cell,
    or with `ridges` [(cells, "WE" | "NS")] (a style's `upper_sides` "long")
    only the walls along each ridge, its gable ends left in the ground floor's
    brick, and a brick pier at each end of those walls (the end square of each
    row along the ridge) - the 1960s-70s estate house's panel between brick
    ends (user, 2026-10-19). Every cell when no ridge leaves any (QA: a house
    of flat or one-square parts lost its panel)."""
    out = []
    for part, ridge in ridges or []:
        along = 0 if ridge == "WE" else 1
        ends = {}
        for c in part:
            k = c[1 - along]
            lo, hi = ends.get(k, (c[along], c[along]))
            ends[k] = (min(lo, c[along]), max(hi, c[along]))
        inner = {c for c in part if c[along] not in ends[c[1 - along]]}
        if inner:
            out.append({"cells": inner, "entry": entry, "from_level": 1, "faces": "NS" if ridge == "WE" else "WE"})
    return out or [{"cells": set(cells), "entry": entry, "from_level": 1}]


# A bay's house window stays (to open, smash and climb through: QA, 2026-10-19),
# drawn with an invisible window (housing_kit `bay_hidden`) so the bay shows.
BAY_WINDOW = {"category": "windows", "tiles": {"West": "fixtures_windows_uk_02_012",
                                                "North": "fixtures_windows_uk_02_013"}}


# Integrated garages (to-do K09.24; asset_creator/housing_kit/garages.py, sheet
# garage_doors_uk_01): a block of 16 a wall kit, in vanilla's walls_garage_01
# layout - a west wall's right, middle, left pieces (0-2), a north wall's
# left, middle, right (3-5), the same open at +8. Laid as furniture on the
# Walls layer, three squares in place of the garage's street wall
# (buildtiles.py), as vanilla's BuildingEd furniture "Walls - Garage" is.
GARAGE_SHEET = "garage_doors_uk_01"
GARAGE_KITS = {"brick_buff_70s": 0, "brick_brown_70s": 1, "brick_orange_80s": 2, "brick_red_50s": 3,
               "brick_buff_80s": 4, "brick_red_80s_band": 5}
GARAGE_ACROSS = (3, 4)           # a garage wing this many squares along the street (2.5-4 m)
GARAGE_DEEP = (4, 8)             # and this deep
INTEGRAL = (3, 5)                # a garage built into a detached house: 3 squares wide, 5 deep
INTEGRAL_FRONT, INTEGRAL_DEPTH = 8, 7   # in a house at least this wide along the street and deep


# Industrial units' roller shutters (asset_creator/commercial_kit/industrial.py, sheet
# commercial_doors_uk_01) and the farm buildings' doors (farm.py, commercial_doors_uk_02):
# kit: (sheet, its block of 16 slots in vanilla's garage door layout).
LOADING_KITS = {"unit_juniper": ("commercial_doors_uk_01", 0), "unit_grey": ("commercial_doors_uk_01", 1),
                "barn_black": ("commercial_doors_uk_02", 0), "shed_yorkshire": ("commercial_doors_uk_02", 1),
                "superstore_brick": ("commercial_doors_uk_01", 2), "superstore_buff": ("commercial_doors_uk_01", 3),
                "garage_brick": ("commercial_doors_uk_01", 4), "exchange_buff": ("commercial_doors_uk_01", 5)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def loading_role(kit: str) -> str | None:
    """A unit's shutter as a furniture role, as garage_role makes a garage door's."""
    from . import catalog as C
    if kit not in LOADING_KITS:
        return None
    sheet, k = LOADING_KITS[kit]
    role = f"uk_loading_{kit}"
    if role not in C.FURNITURE:
        t = lambda i: f"{sheet}_{16 * k + i:03d}"                   # noqa: E731
        C.FURNITURE[role] = {"W": {"0,0": t(2), "0,1": t(1), "0,2": t(0)},
                             "N": {"0,0": t(3), "1,0": t(4), "2,0": t(5)}}
        C.FURNITURE_LAYERS[role] = "Walls"
    return role


def loading_for(overlay, osm_id, mask, street: str | None) -> dict | None:
    """An industrial unit's roller shutter (export_regional.uk_units `loading`): three
    squares in the middle of the longest straight stretch of its street wall, laid as a
    garage door is (lay_garage), in garage_for's record shape (`rect`: the squares inside)."""
    g = (_entry(overlay, osm_id) or {}).get("loading")
    if not g or mask is None:
        return None
    # its street side first, else the faces the camera sees (S, E), else the others: the
    # first with a straight stretch long enough (many units face a yard, not a road);
    # a superstore's lorry dock (`back`) its back first, its front (the car park) never
    order = [street] + [s for s in ("S", "E", "N", "W") if s != street]
    if g.get("back") and street in OPPOSITE:
        order = [OPPOSITE[street]] + [s for s in ("S", "E", "N", "W") if s not in (street, OPPOSITE[street])]
    if g.get("seen"):
        # a repair garage's door and signs where the camera sees them (S, E: a north or west
        # wall's outside is never seen), its street side first if that is one of them
        order = [s for s in [street] if s in ("S", "E")]
        order += [s for s in ("S", "E", "N", "W") if s not in order]
    for side in order:
        got = _loading_on(g, mask, side) if side in ("N", "S", "E", "W") else None
        if got:
            if g.get("signs"):
                got["signs"] = g["signs"]
            return got
    return None


def _loading_on(g, mask, street: str) -> dict | None:
    role = loading_role(g.get("kit"))
    if role is None:
        return None
    m = [list(map(bool, row)) for row in mask]
    h, w = len(m), len(m[0]) if m else 0

    def inside(x, y):
        return 0 <= x < w and 0 <= y < h and m[y][x]
    # every straight stretch of outside wall facing that side, anywhere along the
    # footprint (an L-shaped unit's outermost row was a stub), the longest kept
    best = []
    if street in ("N", "S"):
        step = 1 if street == "S" else -1
        for y in range(h):
            cur = []
            for x in range(w + 1):
                if x < w and inside(x, y) and not inside(x, y + step):
                    cur.append((x, y + 1 if street == "S" else y, "N"))
                    continue
                if len(cur) > len(best):
                    best = cur
                cur = []
    else:
        step = 1 if street == "E" else -1
        for x in range(w):
            cur = []
            for y in range(h + 1):
                if y < h and inside(x, y) and not inside(x + step, y):
                    cur.append((x + 1 if street == "E" else x, y, "W"))
                    continue
                if len(cur) > len(best):
                    best = cur
                cur = []
    if len(best) < 5:
        return None
    k = (len(best) - 3) // 2
    if g.get("signs"):
        # room for the signs after the door (the MOT board's 3, the exchange's 5) where the
        # stretch allows: the door nearer its start
        after = max((len(s["tiles"]["N"]) for s in g["signs"] if s.get("where") == "after"), default=0)
        k = min(k, max(0, len(best) - 3 - after))
    door = best[k:k + 3]
    if street in ("N", "S"):
        y = door[0][1] - (1 if street == "S" else 0)
        rect = (door[0][0], y, door[-1][0], y)
    else:
        x = door[0][0] - (1 if street == "E" else 0)
        rect = (x, door[0][1], x, door[-1][1])
    return {"rect": rect, "door": door, "anchor": door[0], "role": role, "run": best, "at": k}


def _lay_signs(st, garage: dict) -> None:
    """A garage's wall signs (`signs`: tiles W and N in run order, `where` after or before
    the door) on the squares of its door's wall run beside it, where they hold no door;
    the windows under them taken out. Kept on the record as `sign_at` [x, y, dir, tile]
    for tbx.py (its user tiles, the ground floor's)."""
    run, k = garage.get("run") or [], garage.get("at", 0)
    doors = {tuple(dd) for dd in st.doors} | {tuple(e) for e in garage["door"]}
    out = []
    for sign in garage.get("signs") or ():
        n = len(sign["tiles"]["N"])
        squares = run[k + 3:k + 3 + n] if sign.get("where") == "after" else run[max(0, k - n):k]
        if len(squares) < n or any(tuple(e) in doors for e in squares):
            continue
        for j, (x, y, d) in enumerate(squares):
            out.append([x, y, d, sign["tiles"][d][j]])
    taken = {(x, y, d) for x, y, d, _t in out}
    st.windows = [ww for ww in st.windows if tuple(ww) not in taken]
    garage["sign_at"] = out


def garage_role(kit: str) -> str | None:
    """The furniture role of a kit's garage door, put in the catalogue the
    first time it is asked for (in every worker process: tbx.py writes its
    tiles from catalog.FURNITURE)."""
    from . import catalog as C
    k = GARAGE_KITS.get(kit)
    if k is None:
        return None
    role = f"uk_garage_{kit}"
    if role not in C.FURNITURE:
        t = lambda i: f"{GARAGE_SHEET}_{16 * k + i:03d}"            # noqa: E731
        C.FURNITURE[role] = {"W": {"0,0": t(2), "0,1": t(1), "0,2": t(0)},
                             "N": {"0,0": t(3), "1,0": t(4), "2,0": t(5)}}
        C.FURNITURE_LAYERS[role] = "Walls"
    return role


def garage_for(overlay, osm_id, roof: dict | None, mask, street: str | None) -> dict | None:
    """Where a house's garage is, in its own tiles: {"rect": (x0, y0, x1, y1)
    inclusive, "door": [(x, y, dir) x 3], "anchor": (x, y, dir), "role"}.

    A garage wing: a part of the roof plan one storey high, flat or a
    lean-to, GARAGE_ACROSS squares along the street and GARAGE_DEEP deep, on
    the street side of a house of two storeys or more - the flat-roofed wing
    beside the front door that the LiDAR and the outline show (the user's
    1970s references, 2026-10-19). Else, where the style builds them in and
    the house is detached (export_regional `garage`), an integral garage at a
    front corner of a house wide and deep enough. None where neither fits."""
    g = (_entry(overlay, osm_id) or {}).get("garage")
    if not g or street not in ("N", "S", "E", "W") or mask is None:
        return None
    role = garage_role(g.get("kit"))
    if role is None:
        return None
    m = [list(map(bool, row)) for row in mask]
    h, w = len(m), len(m[0]) if m else 0

    def inside(x, y):
        return 0 <= x < w and 0 <= y < h and m[y][x]

    def front_of(x0, y0, x1, y1):
        """The rect's street wall as edges (x, y, dir), if every square beyond it is outside."""
        if street == "S":
            edges = [(x, y1 + 1, "N") for x in range(x0, x1 + 1)]
            out = all(not inside(x, y1 + 1) for x in range(x0, x1 + 1))
        elif street == "N":
            edges = [(x, y0, "N") for x in range(x0, x1 + 1)]
            out = all(not inside(x, y0 - 1) for x in range(x0, x1 + 1))
        elif street == "E":
            edges = [(x1 + 1, y, "W") for y in range(y0, y1 + 1)]
            out = all(not inside(x1 + 1, y) for y in range(y0, y1 + 1))
        else:
            edges = [(x0, y, "W") for y in range(y0, y1 + 1)]
            out = all(not inside(x0 - 1, y) for y in range(y0, y1 + 1))
        return edges if out else None

    def full(x0, y0, x1, y1):
        return all(inside(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))

    along_x = street in ("N", "S")
    found = None
    parts = (roof or {}).get("parts") or []
    if parts and max(p[5] for p in parts) >= 2:
        for x0, y0, x1, y1, form, storeys, n in parts:
            if storeys != 1 or not (form == "flat" or form.startswith("Slope")):
                continue
            across, deep = (x1 - x0 + 1, y1 - y0 + 1) if along_x else (y1 - y0 + 1, x1 - x0 + 1)
            if not (GARAGE_ACROSS[0] <= across <= GARAGE_ACROSS[1] and GARAGE_DEEP[0] <= deep <= GARAGE_DEEP[1]):
                continue
            if n < 0.85 * across * deep or not full(x0, y0, x1, y1):
                continue
            if front_of(x0, y0, x1, y1):
                found = (x0, y0, x1, y1)
                break
    if found is None and g.get("integral"):
        ys = [y for y in range(h) if any(m[y])]
        xs = [x for x in range(w) if any(m[y][x] for y in range(h))]
        if ys and xs:
            bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
            front, depth = (bx1 - bx0 + 1, by1 - by0 + 1) if along_x else (by1 - by0 + 1, bx1 - bx0 + 1)
            if front >= INTEGRAL_FRONT and depth >= INTEGRAL_DEPTH:
                a, d = INTEGRAL
                # the two front corners, the east or south end first (the side the camera sees)
                if street == "S":
                    tries = [(bx1 - a + 1, by1 - d + 1, bx1, by1), (bx0, by1 - d + 1, bx0 + a - 1, by1)]
                elif street == "N":
                    tries = [(bx1 - a + 1, by0, bx1, by0 + d - 1), (bx0, by0, bx0 + a - 1, by0 + d - 1)]
                elif street == "E":
                    tries = [(bx1 - d + 1, by1 - a + 1, bx1, by1), (bx1 - d + 1, by0, bx1, by0 + a - 1)]
                else:
                    tries = [(bx0, by1 - a + 1, bx0 + d - 1, by1), (bx0, by0, bx0 + d - 1, by0 + a - 1)]
                for r in tries:
                    if full(*r) and front_of(*r):
                        found = r
                        break
    if found is None:
        return None
    edges = front_of(*found)
    k = (len(edges) - 3) // 2                       # the door's three squares, centred on the front
    door = edges[k:k + 3]
    if len(door) < 3:
        return None
    return {"rect": found, "door": door, "anchor": door[0], "role": role}


def _loading_in_one_room(st, garage: dict) -> bool:
    """A unit's shutter is three squares of one room's wall (validate_tbx: a door object
    may not reach into another room): slid along its wall, in place, to the nearest
    three squares inside one room; False where there are none."""
    grid = st.grid
    h, w = len(grid), len(grid[0]) if grid else 0

    def room(x, y):
        return grid[y][x] if 0 <= x < w and 0 <= y < h else 0
    x0, y0, d = garage["door"][0]
    # the square inside the wall for an edge (x, y, d): the edge's own square, or the one
    # before it where the building lies on that side
    def inner(x, y):
        if d == "N":
            return (x, y) if room(x, y) else (x, y - 1)
        return (x, y) if room(x, y) else (x - 1, y)
    along = [(x0 + k, y0) if d == "N" else (x0, y0 + k) for k in range(-w - h, w + h)]
    starts = sorted(range(len(along) - 2), key=lambda i: abs(i - (w + h)))
    for i in starts:
        sq = [inner(*along[i + j]) for j in range(3)]
        rooms = {room(*s) for s in sq}
        outside = [along[i + j] for j in range(3)]
        # the same wall: each edge has the building on exactly one side
        def wall(e):
            ex, ey = e
            a, b = (room(ex, ey), room(ex, ey - 1)) if d == "N" else (room(ex, ey), room(ex - 1, ey))
            return bool(a) != bool(b)
        if len(rooms) == 1 and 0 not in rooms and all(wall(e) for e in outside):
            garage["door"] = [(ex, ey, d) for ex, ey in outside]
            garage["anchor"] = garage["door"][0]
            if garage.get("run"):
                run = [tuple(e) for e in garage["run"]]
                if garage["door"][0] in run:
                    garage["at"] = run.index(garage["door"][0])
                else:
                    garage.pop("signs", None)           # slid off its stretch: no room known for them
            garage["rect"] = (min(s[0] for s in sq), min(s[1] for s in sq), max(s[0] for s in sq), max(s[1] for s in sq))
            return True
    return False


def lay_garage(building, garage: dict) -> None:
    """The garage door on the garage's street wall (garage_for): no door or
    window left on its three edges, nothing furnished across it inside, its
    pieces as one furniture object on the Walls layer."""
    from .layout import _cells_for
    st = building.storeys[0]
    if garage["role"].startswith("uk_loading_") and not _loading_in_one_room(st, garage):
        return                           # no stretch of the wall in one room: no shutter
    door = {tuple(e) for e in garage["door"]}
    st.doors = [dd for dd in st.doors if tuple(dd) not in door]
    st.windows = [ww for ww in st.windows if tuple(ww) not in door]
    if garage.get("signs"):
        _lay_signs(st, garage)
    # the squares just inside the door, kept clear for the car
    x0, y0, x1, y1 = garage["rect"]
    clear = set()
    for x, y, d in door:
        clear |= ({(x, y), (x, y - 1)} if d == "N" else {(x, y), (x - 1, y)})
    clear = {(x, y) for x, y in clear if x0 <= x <= x1 and y0 <= y <= y1}
    keep = []
    for role, fx, fy, orient in st.furniture:
        try:
            cells = set(_cells_for(role, fx, fy, orient))
        except Exception:  # noqa: BLE001
            cells = {(fx, fy)}
        if not (cells & clear):
            keep.append((role, fx, fy, orient))
    ax, ay, ad = garage["anchor"]
    keep.append((garage["role"], ax, ay, ad))
    st.furniture = keep


def bay_tiles(name) -> dict | None:
    """A bay's pieces as {part: (west sprite, north sprite)}; a wide part's
    as {part: ((west first, west second), (north first, north second))}."""
    b = BAYS.get(name)
    if b is None:
        return None
    t = lambda k: f"{BAY_SHEET}_{k:03d}"                                   # noqa: E731
    tiles = {part: ((t(k), t(k + 1)), (t(k + 2), t(k + 3))) if part.startswith("wide_") else (t(k), t(k + 1))
             for part, k in b.items()}
    tiles["window"] = BAY_WINDOW
    return tiles


def door_of(overlay, osm_id) -> str | None:
    """The door a regional step chose for this building (a listed building's
    from its listing: listed_building_maker), else None."""
    door = (_entry(overlay, osm_id) or {}).get("door")
    return door if door in DOOR_STYLES else None


def front_door(seed, door: str | None = None) -> dict:
    """A tile entry (category doors) for a home's front door: `door` where one
    is chosen, else by seed."""
    import random
    names = sorted(DOOR_SHARE)
    style = door or random.Random(f"door:{seed}").choices(names, weights=[DOOR_SHARE[n] for n in names])[0]
    base = 4 * DOOR_STYLES.index(style)
    return {"category": "doors", "tiles": {k: f"{DOOR_SHEET}_{base + i:03d}" for i, k in
                                           enumerate(("West", "North", "WestOpen", "NorthOpen"))}}


def with_front_door(style: dict, special, seed, door: str | None = None, surround=None) -> dict:
    if special not in HOMES:
        return style
    style = dict(style)                  # never the shared catalogue style itself
    style["front_door"] = front_door(seed, door)
    if surround:
        style["door_surround"] = surround
    return style


# TV aerials: street_furniture_uk_02_0-3 (pointing S, E, N, W), on the first
# stack of most homes, a level over it (the mast reaches down to the stack's
# top). Every aerial in a town points at its transmitter: Hannington, north
# Hampshire's (51.307 N, 1.198 W), north-west of Odiham and Alton.
AERIAL_SHEET = "street_furniture_uk_02_"
AERIAL_SHARE = 0.85
TRANSMITTER = (51.307, -1.198)


def aerial_facing(proj) -> str:
    """The way to the transmitter on this map (its turn and mirroring in proj)."""
    cx, cy = proj.width / 2, proj.height / 2
    tx, ty = proj.to_px(*TRANSMITTER)
    dx, dy = tx - cx, ty - cy
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


def aerial_tiles(stacks: list, special, seed, facing: str) -> list:
    """An aerial over the first of a home's stacks (chimney_tiles' rows), or none."""
    import random
    rows = [r for r in stacks if r[3] == "Furniture"]
    if special not in HOMES or not rows or random.Random(f"aerial:{seed}").random() >= AERIAL_SHARE:
        return []
    x, y, z, _layer, _tile = rows[0]
    return [(x, y, z + 1, "Furniture", AERIAL_SHEET + str("SENW".index(facing)))]
