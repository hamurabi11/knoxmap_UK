"""Listed buildings into a map: each built to its listing, LiDAR and outline.

    python apply.py <map dir> <area data dir> [--entries N ...] [--dry-run]

Run after roof_plan.py (Building_types README step 8), before Build. For each
list entry of the area (listing.py's cache; fetched if missing):

1. **Which buildings it is.** The map's own outlines inside its box
   (`<map>_buildings.geojson`, real coordinates: OSM or LiDAR outlines that
   blocks.py registered on the LiDAR, a unit by its own outline) are the
   polygons built: the LiDAR is the truth for shape and place (PLAN principle
   1). Historic England's polygon picks them where it is a real outline
   (overlap of 30% of either); most are a marker at the grid reference, which
   picks the outline it falls in, else the nearest within POINT_M (`match`).
   An outline two entries reach goes to the nearer.
2. **What it is.** The text read (lexicon.py), the LiDAR's storeys for the
   building (`<area>/roofs.csv`, by OSM id or a unit's parent), the design
   (design.py): style (wall and roof), storeys, front door.
3. **Into regional.json** (Knoxmap/knoxbuild/regional_uk.py reads it): each
   matched building's `style`, `levels` (buildings of at least MAIN_SHARE of
   the entry's biggest only: not its sheds), `door`, and `listed` (entry,
   grade, name: a record, Build ignores it); the new styles into the palette's
   `add` with `by_id_only` (kept off other buildings). The roof ranges,
   attics and dormers roof_plan.py drew from the LiDAR are left as they are.
   The file as it was is kept as `regional.prelisted.json` the first time.

Structures (walls, bridges, a memorial...) are listed in the report, not built
here. Writes `<map>/listed_buildings.json` (matches, designs, what each needs).
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import shutil
import sys
from collections import Counter

from shapely.geometry import box, shape
from shapely.strtree import STRtree

import design as D
import lexicon as L
import listing as LI

# KnoxMap's generator (its projection, the road graph's shift): one path for the whole run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # asset_creator: acpaths
import acpaths  # noqa: E402
sys.path.insert(0, acpaths.KNOX)
# Building_types' packages: the area's data level (area/area_conf.py)
sys.path.insert(0, os.path.join(acpaths.UK, "src"))
from area import area_conf  # noqa: E402

POINT_M = 5.0
OUTLINE_M2 = 20.0
OVERLAP = 0.3
MAIN_SHARE = 0.5
M_PER_DEG = 111320.0


# The keys of a regional.json entry this script sets (a rerun puts them back first),
# and those a landmark's description may set besides.
OWNED = ("style", "levels", "door", "door_surround", "wall_trim", "drains", "hip", "listed", "landmark",
         "extend", "join", "footprint", "blank", "full_height")
LANDMARK_OWNED = ("roof", "hall", "bay", "upper_wall")   # a landmark drops the last two (below)
LANDMARKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landmarks")
BUILT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes", "built")


def landmark(entry, osm_id) -> dict | None:
    """A landmark's description (landmarks/<entry>_*.json) and the tile records
    of every recipe sheet (recipes/built/, recipe_kit.py), by item name."""
    import glob
    paths = sorted(glob.glob(os.path.join(LANDMARKS, f"{entry}_*.json")))
    if not paths:
        return None
    desc = load_json(paths[0])
    if str(desc.get("osm_id", osm_id)) != str(osm_id):
        return None
    items, sheet_of = {}, {}
    for path in sorted(glob.glob(os.path.join(BUILT, "*.json"))):
        for k, v in load_json(path)["items"].items():
            # an item name is the landmark's only handle on it: two sheets with one name and
            # the later (by file name) won, silently - All Saints, Odiham drew Westminster's
            # Gothic pinnacle, and Odiham needed the Westminster pack (2026-10-25)
            if k in sheet_of:
                raise SystemExit(f"apply.py: item {k!r} is in {sheet_of[k]} and {os.path.basename(path)}: "
                                 f"rename one in its recipe")
            sheet_of[k] = os.path.basename(path)
            items[k] = v
    used = {p[1] for p in desc.get("pieces", []) + desc.get("furniture", []) + desc.get("windows", [])}
    used |= {r["floor"] for r in desc.get("rooms", []) if r.get("floor")}
    missing = sorted(used - set(items))
    if missing:
        print(f"  landmark {entry}: no tiles for {missing} (run makers/recipe_kit.py)")
    return {"desc": desc, "items": {k: v for k, v in items.items() if k in used}}


def moved_with(lm: dict, geom, map_dir: str) -> dict:
    """The description moved with its building: a box of the described size placed
    elsewhere (blocks.py on another routing, 2026-10-25: St Lawrence, Alton, 2 east and 3
    south) has its map-tile parts - origin, extend, roof - moved as far; the rest is
    the building's own squares from its origin, and goes with it. One with its own
    `footprint` stands on those squares, and is left."""
    d = lm["desc"]
    if not d.get("origin") or not d.get("size") or d.get("footprint"):
        return lm
    proj = map_projector(map_dir)
    pts = [proj.to_px(lat, lon) for g in getattr(geom, "geoms", [geom]) for lon, lat in g.exterior.coords]
    x0, y0 = (int(round(min(v))) for v in zip(*pts))
    x1, y1 = (int(round(max(v))) for v in zip(*pts))
    dx, dy = x0 - d["origin"][0], y0 - d["origin"][1]
    if not dx and not dy:
        return lm
    if [x1 - x0, y1 - y0] != list(d["size"]):
        # placed elsewhere, but not the box described (a building joined to its units -
        # `join` - or trimmed by a tile): its map-tile parts are left where they were
        print(f"  landmark {d.get('entry')}: its building placed {dx:+d},{dy:+d} from where it was described, "
              f"{x1 - x0}x{y1 - y0} against {d['size'][0]}x{d['size'][1]}: not moved with it - its roof and "
              f"extend may stand off it (run its description again on this map)")
        return lm
    d = dict(d, origin=[x0, y0])
    if d.get("extend"):
        d["extend"] = [[r[0] + dx, r[1] + dy, *r[2:]] for r in d["extend"]]
    if d.get("roof"):
        d["roof"] = [[r[0] + dx, r[1] + dy, *r[2:]] for r in d["roof"]]
    print(f"  landmark {d.get('entry')}: its building placed {dx:+d},{dy:+d} from where it was described: moved with it")
    return dict(lm, desc=d)


@functools.lru_cache(maxsize=None)
def map_projector(map_dir: str):
    """The map's tile projection (its info.json)."""
    from generator.renderer import Projector
    name = os.path.basename(os.path.normpath(map_dir))
    info = load_json(os.path.join(map_dir, f"{name}_info.json"))
    b = info["bbox"]
    return Projector.build(b["south"], b["west"], b["north"], b["east"], info["meters_per_tile"],
                           info.get("rotation") or 0.0)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def map_box(map_dir: str):
    name = os.path.basename(os.path.normpath(map_dir))
    bb = load_json(os.path.join(map_dir, f"{name}_info.json"))["bbox"]
    return box(bb["west"], bb["south"], bb["east"], bb["north"])


def map_buildings(map_dir: str) -> list[tuple[str, object, dict]]:
    """The map's outlines Build will build: those inside the map's box (the
    buildings file also holds the ones round it, which nothing places)."""
    name = os.path.basename(os.path.normpath(map_dir))
    inside = map_box(map_dir)
    out = []
    for ft in load_json(os.path.join(map_dir, f"{name}_buildings.geojson"))["features"]:
        if ft["geometry"]["type"] not in ("Polygon", "MultiPolygon"):
            continue
        g = shape(ft["geometry"])
        if g.is_valid and not g.is_empty and inside.contains(g.centroid):
            out.append((str(ft["properties"].get("osm_id")), g, ft["properties"]))
    return out


def nhle_shapes(area: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for kind, fname in (("polygon", "nhle_polygons.geojson"), ("point", "nhle_points.geojson")):
        for ft in load_json(os.path.join(area, fname))["features"]:
            g = shape(ft["geometry"]).buffer(0) if kind == "polygon" else shape(ft["geometry"])
            out.setdefault(ft["properties"]["ListEntry"], {})[kind] = g
    return out


def area_m2(g) -> float:
    return g.area * M_PER_DEG ** 2 * math.cos(math.radians(g.centroid.y))


def match(entry_shapes: dict, blds, tree) -> list[tuple[int, float, float, float]]:
    """(index into blds, share of the building covered, IoU, metres from the
    entry's point) of the buildings an entry is.

    Historic England's polygon is the building's outline only where the entry
    has been mapped as one; most old entries' "polygon" is a 2.7 m2 marker at
    the grid reference (all 167 of Odiham's, 2026-10-18). A real outline
    (OUTLINE_M2 or more) is matched by overlap; a marker by the outline it
    falls in, else the nearest within POINT_M."""
    poly, pt = entry_shapes.get("polygon"), entry_shapes.get("point")
    if pt is None and poly is not None:
        pt = poly.centroid
    if pt is None:
        return []
    pt = pt.centroid
    deg = M_PER_DEG * math.cos(math.radians(pt.y))
    hits = []
    if poly is not None and not poly.is_empty and area_m2(poly) >= OUTLINE_M2:
        for i in tree.query(poly):
            g = blds[i][1]
            inter = g.intersection(poly).area
            if inter >= OVERLAP * g.area or inter >= OVERLAP * poly.area:
                hits.append((int(i), inter / g.area, inter / g.union(poly).area, g.distance(pt) * deg))
        return hits
    inside = [int(i) for i in tree.query(pt) if blds[i][1].contains(pt)]
    if inside:
        return [(i, 0.0, 0.0, 0.0) for i in inside]
    r = POINT_M / deg
    near = sorted((blds[i][1].distance(pt) * deg, int(i)) for i in tree.query(pt.buffer(r)))
    return [(near[0][1], 0.0, 0.0, near[0][0])] if near and near[0][0] <= POINT_M else []


def lidar_levels(area: str) -> dict[str, int]:
    path = os.path.join(area, "roofs.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["osm_id"]: int(r["storeys_lidar"]) for r in csv.DictReader(f) if r.get("storeys_lidar")}


# Structures (makers/structure_kit.py, sheet listed_structures_uk_01): kind ->
# [(z, tile)]. Listed walls are Build's boundaries (from OSM), not placed here.
STRUCTURE_TILES = {"memorial": [(0, "listed_structures_uk_01_0"), (1, "listed_structures_uk_01_1")],
                   "stocks": [(0, "listed_structures_uk_01_2")],
                   "milestone": [(0, "listed_structures_uk_01_3")]}
CARRIAGEWAY = [(150, 150, 150), (100, 100, 100), (80, 80, 80)]   # handover COLOUR_OF: asphalt
STAND_REACH = 6


def place_structures(map_dir: str, shapes: dict, structures: list[dict], dry: bool) -> list[dict]:
    """Each structure at its list entry's point, moved with the roads as the
    rest of the town was (the road graph's shift samples), on the nearest
    square within STAND_REACH that is neither a building nor carriageway.
    Written at the front of <map>_structures.json (Build lays them), counted
    as `listed_added` so a rerun replaces them; kerbs.py's own tiles are the
    ones at the end."""
    import numpy as np
    from PIL import Image
    from generator import octilinear
    name = os.path.basename(os.path.normpath(map_dir))
    proj = map_projector(map_dir)
    graph = os.path.join(map_dir, f"{name}_roadgraph.json")
    move = None
    if os.path.exists(graph):
        shift = load_json(graph).get("shift") or []
        move = octilinear.displacement([(np.array(proj.to_px(a, b_)), np.subtract(proj.to_px(c, d), proj.to_px(a, b_)))
                                        for a, b_, c, d in shift], proj)
    ground = np.asarray(Image.open(os.path.join(map_dir, f"{name}_ground_base.bmp")).convert("RGB"))
    H, Wd = ground.shape[:2]
    blocked = np.zeros((H, Wd), bool)
    for c in CARRIAGEWAY:
        blocked |= (ground == c).all(-1)
    placements = os.path.join(map_dir, f"{name}_placements.csv")
    if os.path.exists(placements):
        with open(placements, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                x, y, w, h = (int(r[k]) for k in ("tile_x", "tile_y", "width", "height"))
                blocked[max(0, y):y + h, max(0, x):x + w] = True
    else:
        # A map not built yet (a new hand-over, 2026-10-25): its buildings as blocks.py
        # placed them, which Build draws as they are.
        from PIL import ImageDraw
        img = Image.new("1", (Wd, H), 0)
        d = ImageDraw.Draw(img)
        for ft in load_json(os.path.join(map_dir, f"{name}_buildings.geojson"))["features"]:
            g = ft["geometry"]
            if not ft["properties"].get("placed") or g["type"] not in ("Polygon", "MultiPolygon"):
                continue                     # (what blocks.py left to KnoxMap: Build places it later)
            for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
                d.polygon([proj.to_px(lat, lon) for lon, lat in poly[0]], fill=1)
        blocked |= np.array(img, bool)
    placed = []
    for st in structures:
        tiles = STRUCTURE_TILES.get(st["kind"])
        pt = (shapes.get(st["entry"]) or {}).get("point")
        if not tiles or pt is None:
            st["placed"] = None
            continue
        lat, lon = pt.centroid.y, pt.centroid.x
        if move is not None:
            lat, lon = move(lat, lon)
        x0, y0 = (int(round(v)) for v in proj.to_px(lat, lon))
        spot = None
        for r in range(STAND_REACH + 1):
            ring = [(x0 + dx, y0 + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if max(abs(dx), abs(dy)) == r]
            ok = [(x, y) for x, y in ring if 0 <= x < Wd and 0 <= y < H and not blocked[y, x]]
            if ok:
                spot = min(ok, key=lambda q: (q[0] - x0) ** 2 + (q[1] - y0) ** 2)
                break
        st["placed"] = list(spot) if spot else None
        if spot:
            placed += [[spot[0], spot[1], z, "Furniture", t] for z, t in tiles]
    sp = os.path.join(map_dir, f"{name}_structures.json")
    if not dry and os.path.exists(sp):
        data = load_json(sp)
        rows = data.get("tiles") or []
        rows = rows[int(data.get("listed_added") or 0):]          # this script's last run's, at the front
        data["tiles"] = placed + rows
        data["listed_added"] = len(placed)
        with open(sp + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(sp + ".tmp", sp)
    return structures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("map_dir")
    ap.add_argument("area")
    ap.add_argument("--entries", type=int, nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # Always from the file as export_regional / roof_plan left it: a fresh
    # regional.json (no `listed` entries) becomes the new starting point, else
    # the one kept from before the last run - so a rerun starts clean.
    reg_path = os.path.join(a.map_dir, "regional.json")
    pre_path = os.path.join(a.map_dir, "regional.prelisted.json")
    regional = load_json(reg_path)
    before = json.loads(json.dumps(regional))   # as it is now (--entries keeps the other entries' work)
    fresh = not any("listed" in e for e in regional["buildings"].values())
    if not fresh:
        # Start from regional.json as it is now - a later roof_plan.py or
        # export_regional run stays - with what this script set taken back to
        # what it was before it ran (regional.prelisted.json): only its own keys.
        pre = load_json(pre_path)["buildings"]
        for oid, e in regional["buildings"].items():
            if "listed" not in e:
                continue
            was = pre.get(oid) or {}
            keys = OWNED + (LANDMARK_OWNED if "landmark" in e else ())
            for k in keys:
                if k in was:
                    e[k] = was[k]
                else:
                    e.pop(k, None)
    palette = regional["palette"]
    blds = map_buildings(a.map_dir)
    tree = STRtree([b[1] for b in blds])
    # No NHLE (the osm_lidar data level, or none fetched): no listed builds, the
    # Grade I / II* buildings built by the general rules - regional.json as
    # export_regional and roof_plan left it (a rerun's listed work taken back above).
    nhle = all(area_conf.extra(a.area, f) for f in ("nhle_polygons.geojson", "nhle_points.geojson"))
    if not nhle:
        print(f"no NHLE for {a.area} ({area_conf.level(a.area)}): no listed buildings")
    shapes = nhle_shapes(a.area) if nhle else {}
    seen_levels = lidar_levels(a.area)
    # Every entry is matched (who owns a building two entries reach, and every
    # structure, as a full run has them); only the --entries ones are designed.
    rows = LI.load(a.area) if nhle else []
    selected = {str(x) for x in a.entries} if a.entries else None

    report, styles_used, unmatched, structures = [], Counter(), [], []
    changed = {}
    # Match every entry first: a building two entries reach (two markers in
    # one terrace outline) goes to the entry whose point is nearest.
    readings, hits_of, owner = {}, {}, {}
    for row in rows:
        r = readings[row["entry"]] = L.read(row["text"], row["name"])
        if r.group != "building":
            structures.append({"entry": row["entry"], "name": row["name"], "kind": r.kind})
            continue
        hits_of[row["entry"]] = match(shapes.get(row["entry"], {}), blds, tree)
        for i, _c, _u, d in hits_of[row["entry"]]:
            if i not in owner or d < owner[i][0]:
                owner[i] = (d, row["entry"])
    shared = sum(1 for e, hs in hits_of.items() for h in hs if owner[h[0]][1] != e)
    for row in rows:
        if row["entry"] not in hits_of:
            continue
        if selected is not None and str(row["entry"]) not in selected:
            continue
        r = readings[row["entry"]]
        hits = [h for h in hits_of[row["entry"]] if owner[h[0]][1] == row["entry"]]
        if not hits:
            pt = shapes.get(row["entry"], {}).get("point")
            why = ("its building went to a nearer entry" if hits_of[row["entry"]] else
                   "outside the map" if pt is not None and not map_box(a.map_dir).contains(pt.centroid) else
                   f"no outline within {POINT_M:g} m")
            unmatched.append({"entry": row["entry"], "name": row["name"], "why": why})
            continue
        biggest = max(blds[i][1].area for i, _c, _u, _d in hits)
        for i, cover, iou, _d in hits:
            oid, geom, props = blds[i]
            main = geom.area >= MAIN_SHARE * biggest
            parent = str(props.get("parent_osm_id") or oid)
            prior = dict(regional["buildings"].get(oid) or {})
            if prior.get("loading"):
                # an industrial unit's look (export_regional.uk_units) is not a listed
                # building's: its style was read as the listing's wall (uk_uk_unit_grey__...)
                for k in ("style", "blank_upper", "blank", "loading", "flat_roof", "upper_wall", "upper_sides"):
                    prior.pop(k, None)
                regional["buildings"][oid] = prior
            lidar = {"levels": seen_levels.get(oid) or seen_levels.get(parent),
                     "style": prior.get("style")}
            spec = D.design(r, row, lidar, palette)
            e = dict(prior)
            # A landmark's description may set its style and redraw its roof plan
            # where the LiDAR's reading is wrong (lean-tos for two gabled ranges).
            lm = landmark(row["entry"], oid) if main else None
            if lm:
                lm = moved_with(lm, geom, a.map_dir)
            if lm and lm["desc"].get("style"):
                spec["style"] = lm["desc"]["style"]
            if lm and lm["desc"].get("roof"):
                e["roof"] = lm["desc"]["roof"]
                e["hall"] = lm["desc"].get("hall", True)
            if lm and lm["desc"].get("extend"):
                e["extend"] = lm["desc"]["extend"]      # parts its outline misses (map tiles)
            if lm and lm["desc"].get("footprint"):
                e["footprint"] = lm["desc"]["footprint"]  # its own squares, in place of its outline
            if lm and lm["desc"].get("join"):
                e["join"] = lm["desc"]["join"]          # its other units, joined into it by Build
            if lm and lm["desc"].get("full_height"):
                e["full_height"] = True                 # its towers past the map's max_levels (Parliament's)
            if lm and lm["desc"].get("blank"):
                e["blank"] = True                       # no windows of Build's own: the description gives them all
            if spec.get("style"):
                e["style"] = spec["style"]
                styles_used[spec["style"]] += 1
            if spec.get("blank") and props.get("knox_dwelling") not in ("house", "flats"):
                e["blank"] = True                        # no windows (design: a barn's listing names none; a conversion has them)
            # A landmark the roof planner drew from the LiDAR (a church's nave
            # and tower: `hall`) keeps its storeys: they are its tower's height.
            if main and spec.get("levels") and not e.get("hall"):
                e["levels"] = spec["levels"]
            elif e.get("hall"):
                if lm and lm["desc"].get("roof"):    # the description's roof: its tallest part's storeys
                    e["levels"] = max(int(p[5]) for p in lm["desc"]["roof"])
                spec["levels"], spec["levels_from"] = e.get("levels"), "roof plan (landmark)"
            # a wall style with its own upper storey (a granary's staddles only at its foot:
            # the palette's upper_wall, as export_regional.house_parts gives a house's), its
            # storeys now known
            base = {x["name"]: x for x in palette.get("add", [])}.get((spec.get("style") or "")[3:].split("__")[0])
            if base and base.get("upper_wall") and (e.get("levels") or 1) >= 2:
                e["upper_wall"] = [f"{base['upper_wall']}_000", f"{base['upper_wall']}_016"]
            if spec.get("attic") and not e.get("attic"):
                spec["notes"].append("the listing names an attic; the roof plan made none")
            if main:
                e["door"] = spec["door"]
                if spec.get("door_surround"):
                    e["door_surround"] = spec["door_surround"]
            if spec.get("wall_trim"):
                e["wall_trim"] = spec["wall_trim"]
            if main:
                e["drains"] = True                       # every one of them has downpipes (user, 2026-10-18)
            # Hipped where the listing says so (regional_uk._gable makes a
            # 30-degree hip where the game can build one; else the gable stays).
            if r.has("roof.hipped") and not r.has("roof.half_hipped"):
                e["hip"] = True
            e["listed"] = {"entry": row["entry"], "grade": row["grade"], "name": row["name"]}
            # A landmark with a description of its own (landmarks/<entry>_*.json): it,
            # and the tiles its recipes drew, for Build (knoxbuild/landmark_uk.py).
            if lm:
                e["landmark"] = lm
                spec["notes"].append(f"built to its description ({len(lm['desc'].get('pieces', []))} pieces)")
                # The regional step's parts for an ordinary house of its style (a bay,
                # an upper floor in a wall of its own, a porch: export_regional
                # house_parts) are not a landmark's: it has only what its listing or
                # its description names (a Victorian bay on a 1672 meeting house).
                for k in ("bay", "upper_wall"):
                    e.pop(k, None)
                surround = lm["desc"].get("door_surround", spec.get("door_surround"))
                spec["door_surround"] = surround          # the report says what Build gets
                if surround:
                    e["door_surround"] = surround
                else:
                    e.pop("door_surround", None)
            changed[oid] = e
            report.append({**spec, "osm_id": oid, "main": main, "nhle_cover": round(cover, 2),
                           "nhle_iou": round(iou, 2), "lidar_levels": lidar["levels"],
                           "was": {k: (regional["buildings"].get(oid) or {}).get(k) for k in ("style", "levels")}})

    # The palette's own styles stay; a rerun's listed-only styles are made afresh.
    # (Ours are "uk_<wall>__<roof>__<window>"; another by-id style - housing_kit's
    # uk_estate_70s, 2026-10-19 - is the palette's own.)
    own = [s for s in palette.get("add", []) if not (s.get("by_id_only") and "__" in s["name"])]
    own_names = {s["name"] for s in own}
    new_styles = [D.style_spec(n, {"add": own}) for n in sorted(styles_used) if n not in own_names]
    print(f"{len(rows)} entries: {len(report)} map buildings matched to "
          f"{len({x['entry'] for x in report})} entries; no building found for {len(unmatched)}; "
          f"{len(structures)} structures (not built here); {shared} buildings two entries reached, "
          f"given to the nearer")
    print("  not matched:", dict(Counter(u["why"] for u in unmatched)))
    print("  styles:", dict(styles_used.most_common()))
    print("  new styles in the palette:", [s["name"] for s in new_styles])
    print("  levels from:", dict(Counter(x.get("levels_from", "-").split(" (")[0] for x in report if x["main"])))
    print("  doors:", dict(Counter((x["door"], x["door_from"]) for x in report if x["main"]).most_common(8)))
    print("  storeys changed:", sum(1 for x in report if x["main"] and x.get("levels")
                                     and x["was"]["levels"] not in (None, x["levels"])),
          " style changed:", sum(1 for x in report if x.get("style") and x["was"]["style"] != x["style"]))
    report_path = os.path.join(a.map_dir, "listed_buildings.json")
    if a.entries and not fresh and os.path.exists(report_path):
        # only some entries redone: the rest of the report as the last run left it
        done = {str(x) for x in a.entries}
        last = load_json(report_path)
        report = [x for x in last.get("buildings", []) if str(x["entry"]) not in done] + report
        unmatched = [x for x in last.get("unmatched", []) if str(x.get("entry")) not in done] + unmatched
    out = {"map": os.path.basename(os.path.normpath(a.map_dir)), "buildings": report,
           "unmatched": unmatched, "structures": structures, "new_styles": new_styles}
    with open(os.path.join(a.map_dir, "listed_buildings.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    place_structures(a.map_dir, shapes, structures, a.dry_run)
    print("  structures:", [(x["kind"], x["name"][:30], x.get("placed")) for x in structures if x.get("placed") is not None],
          f"({sum(1 for x in structures if x.get('placed') is None)} not placed: walls, bridges, or off the map)")
    with open(os.path.join(a.map_dir, "listed_buildings.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    if a.dry_run:
        print("dry run: regional.json and structures.json not changed")
        return 0
    if fresh:
        shutil.copyfile(reg_path, pre_path)
    if a.entries and not fresh:
        # Only some entries were redone: every other listed building stays as the
        # last run left it, with the styles it uses.
        done = {str(x) for x in a.entries}
        kept = {oid: e for oid, e in before["buildings"].items()
                if "listed" in e and str(e["listed"]["entry"]) not in done and oid not in changed}
        changed = {**kept, **changed}
        used = {e.get("style") for e in kept.values()}
        names = {s["name"] for s in own + new_styles}
        new_styles += [s for s in before["palette"].get("add", [])
                       if s.get("by_id_only") and s["name"] in used and s["name"] not in names]
    regional["buildings"].update(changed)
    palette["add"] = own + new_styles
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(regional, f, indent=1)
    print(f"wrote {reg_path}: {len(changed)} buildings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
