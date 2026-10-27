"""Arches: the squares where a road or path passes through a building.

A way OSM tags as passing under a building - tunnel=building_passage,
covered=yes, or a negative layer - that really crosses a building's outline
is a passage through it; so is any way crossing a building on another layer
than its own (OSM's layers: the two do not meet - the shop on Odiham's High
Street tagged layer=-1 over a plain footway to the car park behind, user
2026-10-13) (a carriage arch, a covered way into a yard or car
park; Odiham has ten, user 2026-10-11). For each placed building (unit) its
passage squares: the way's line as the game draws it, PASSAGE_TILES wide
(VEHICLE or FOOT), inside the building's placed outline. export_regional.py
writes them into regional.json ("passage", map tiles); Build leaves them out
of the ground floor of a building two storeys or more (knoxbuild/regional_uk.py
roof_plan), so the floors above bridge the passage.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)

VEHICLE, FOOT = 3, 2          # tiles across a passage for traffic, for people on foot
ON_FOOT = ("footway", "path", "cycleway", "steps", "pedestrian", "bridleway")


def layer_of(tags: dict) -> int:
    layer = str(tags.get("layer", "0")).strip()
    return int(layer) if layer.lstrip("-").isdigit() else 0


def is_passage(tags: dict) -> bool:
    return bool(tags.get("highway")) and (
        tags.get("tunnel") == "building_passage" or tags.get("covered") == "yes" or layer_of(tags) < 0)


def made_passages(map_dir: str, info: dict) -> set:
    """The OSM ways the road graph's hand-over made passages (--lanes: its file's
    "passages", LOG 2026-10-25 #8); none on a map with no road graph (info.json's
    "road_graph": a stale file left beside one counts for nothing)."""
    if not info.get("road_graph"):
        return set()
    path = os.path.join(map_dir, info["road_graph"])
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f).get("passages") or [])


def base_id(osm_id) -> str:
    """The OSM way a placed piece or unit came from (123u1, 123p2 -> 123)."""
    return re.sub(r"(u\d+|p\d+)+$", "", str(osm_id))


def passages(map_dir: str, info: dict, proj) -> dict:
    """{building osm_id (units included): [[x, y], ...] map tiles}."""
    from generator import osm
    from generator.octilinear import straighten_roads
    from generator.renderer import _is_polygon, classify
    cache = os.path.join(map_dir, info["osm_cache"])
    real = osm.load_cache(cache, tuple(info["osm_bbox"])) or []
    # The ways that really cross a building (a tunnel under a bridge does not).
    built = [(shapely.Polygon([(lo, la) for la, lo in f.geometry]), layer_of(f.tags), str(f.osm_id)) for f in real
             if f.kind == "way" and f.tags.get("building") and len(f.geometry) > 3]
    built = [b for b in built if b[0].is_valid]
    outlines = [b[0] for b in built]
    tree = shapely.STRtree(outlines)
    ids = {}
    # ...and the lanes the hand-over made passages (--lanes): each only through the buildings
    # its real line crosses - the rest of the way (a link's other end) passes under nothing
    made = made_passages(map_dir, info)
    only: dict = {}                        # made passage way -> the buildings it may pass under
    for f in real:
        if f.kind != "way" or len(f.geometry) < 2 or not f.tags.get("highway") or f.tags.get("area") == "yes":
            continue
        line = shapely.LineString([(lo, la) for la, lo in f.geometry])
        mine = layer_of(f.tags)
        hits = [int(k) for k in tree.query(line) if line.crosses(outlines[int(k)]) or line.within(outlines[int(k)])]
        if hits and (is_passage(f.tags) or f.osm_id in made or any(built[k][1] != mine for k in hits)):
            ids[f.osm_id] = VEHICLE if f.tags.get("highway") not in ON_FOOT else FOOT
            if f.osm_id in made and not is_passage(f.tags):
                only[f.osm_id] = {built[k][2] for k in hits}
    if not ids:
        return {}
    game = osm.load_cache(cache, tuple(info["osm_bbox"])) or []
    if info.get("straight_roads"):
        straighten_roads(game, proj, classify, _is_polygon, map_dir=map_dir)
    bands = [(f.osm_id, shapely.LineString([proj.to_px(la, lo) for la, lo in f.geometry]).buffer(ids[f.osm_id] / 2,
                                                                                                 cap_style=2))
             for f in game if f.kind == "way" and f.osm_id in ids and len(f.geometry) >= 2]
    name = os.path.basename(os.path.normpath(map_dir))
    with open(os.path.join(map_dir, f"{name}_buildings.geojson"), encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    out = {}
    for ft in feats:
        if not (ft["properties"].get("placed") and ft["geometry"]["type"] == "Polygon"):
            continue
        poly = shapely.Polygon([proj.to_px(la, lo) for lo, la in ft["geometry"]["coordinates"][0]])
        bid = base_id(ft["properties"].get("parent_osm_id") or ft["properties"]["osm_id"])
        for way, band in bands:
            if way in only and bid not in only[way]:
                continue
            cut = poly.intersection(band)
            if cut.is_empty:
                continue
            x0, y0, x1, y1 = (int(np.floor(v)) for v in cut.bounds)
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            inside = shapely.contains_xy(cut, xs, ys)
            cells = [[int(x), int(y)] for x, y in zip(xs[inside] - 0.5, ys[inside] - 0.5)]
            if cells:
                out.setdefault(str(ft["properties"]["osm_id"]), []).extend(cells)
    return out
