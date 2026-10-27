"""An area's settings (data/<area>/area.json) and which of its extra data the
chain may read (PLAN U, stage 1).

    {"name": "Alton", "town": "alton", "council": "east_hampshire",
     "data": "full" | "osm_lidar"}

Two data levels:

- `osm_lidar`: the map's OSM download, its extra Overpass queries (rail land,
  road nodes), the EA DSM/DTM and the point cloud. Nothing else is read, even
  where its file is in the area folder.
- `full` (the default): adds OS Open Roads, OS Open UPRN, NHLE, EPC (from the
  council's folder, data/<council>), FSA and INSPIRE, each where its file is.

Every step asks `extra(area, file)` for an extra source's path and goes
without it on None (its fallback is in its own docstring). `council` names the
folder of the council's EPC extracts; an area without one reads no EPC (before
2026-10-27 such an area read Hart's). `town` names the place signs
(roadgraph/furniture.py); without it, the area folder's name up to its first _.
"""
from __future__ import annotations

import json
import os

LEVELS = ("full", "osm_lidar")
# The files only the full level reads.
EXTRA = ("os_roads.geojson", "uprn.csv", "nhle_points.geojson", "nhle_polygons.geojson",
         "nhle_monuments.geojson", "fsa.json", "inspire.geojson", "inspire_town.geojson")


def settings(area: str) -> dict:
    path = os.path.join(area, "area.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def level(area: str) -> str:
    lv = settings(area).get("data") or "full"
    if lv not in LEVELS:
        raise ValueError(f"{area}/area.json: data {lv!r}, not one of {LEVELS}")
    return lv


def extra(area: str, name: str) -> str | None:
    """The path of one of the area's extra files (EXTRA), or None: not at this
    data level, or not there."""
    assert name in EXTRA, name
    path = os.path.join(area, name)
    return path if level(area) == "full" and os.path.exists(path) else None


def is_extra(path: str) -> bool:
    """A path a step was given on its command line (plots2's --inspire-attrs):
    one of the extra files, of an area at the osm_lidar level?"""
    area = os.path.dirname(os.path.abspath(path))
    return os.path.basename(path) in EXTRA and level(area) != "full"


def epc_dir(area: str) -> str | None:
    """The folder of the area's council's EPC extracts, or None (osm_lidar, no
    council set, or no such folder)."""
    council = settings(area).get("council")
    if level(area) != "full" or not council:
        return None
    path = os.path.normpath(os.path.join(area, "..", council))
    return path if os.path.isdir(path) else None


def town(area: str) -> str:
    return (settings(area).get("town") or os.path.basename(os.path.normpath(area)).split("_")[0]).lower()
