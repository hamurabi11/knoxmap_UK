"""How built-up streets are routed: the road graph's modes (LOG 2026-10-25 #2).

    graph   the routing as it was: vanilla's sixteen directions everywhere
    grid    within GRID_REACH tiles of a building, the grid only (snap.GRID, LOG 2026-10-20 #26)
    cost    a frontage-density field (snap.COST): a step off the grid costs more the more
            building outline is near it - 45 degrees a little, 2:1 a lot - and where it is
            dense a road turns only at a junction, a gap in the frontage or a plot boundary
            (snap.JOG); with OCC_WEIGHT, each building square under the carriageway costs
            (snap.OCC)

prepare() sets snap up for a mode and returns what the map's info.json records
("road_mode"); reset() puts it back. The fields are worked out on the map the
roads are routed on (the hand-over's source map), from its OSM download's
buildings and, for the plot boundaries, the area's INSPIRE titles
(<area>/inspire.geojson; none, no plot rule).
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import network, snap
from area import area_conf

MODES = ("graph", "grid", "cost")
# Named settings (mode, snap constants), for the hand-over's --roads and mode_test.py's
# runs (LOG 2026-10-25 #2, #3); the three modes are presets of themselves. ("cost-mid" is
# "cost" by another name: the LOG's tables use it.)
LOW = {"grid": 0.0, "45": 0.25, "2:1": 0.75}
MID = {"grid": 0.0, "45": 0.5, "2:1": 1.5}
PRESETS = {
    "graph": ("graph", {}),
    "grid": ("grid", {}),
    "cost": ("cost", {}),
    "cost-low": ("cost", {"DIR_PENALTY": LOW}),
    "cost-mid": ("cost", {"DIR_PENALTY": MID}),
    "cost-high": ("cost", {"DIR_PENALTY": {"grid": 0.0, "45": 1.0, "2:1": 3.0}}),
    # a density threshold no square reaches: the jog rule off, and with it every use of the
    # threshold (straights squared where dense, vanilla shapes held to the grid, JOG_RUN)
    "cost-nojog": ("cost", {"DIR_PENALTY": MID, "JOG_DENSITY": 2.0}),
    "cost-low-occ0.25": ("cost", {"DIR_PENALTY": LOW, "OCC_WEIGHT": 0.25}),
    "cost-low-occ1": ("cost", {"DIR_PENALTY": LOW, "OCC_WEIGHT": 1.0}),
    "cost-low-occ4": ("cost", {"DIR_PENALTY": LOW, "OCC_WEIGHT": 4.0}),
    # The UK road option chosen after Alton (LOG 2026-10-25 #3, #4; user: "save it as another
    # road option"): cost-low-occ1's routing, and the map's buildings squared about their
    # street-facing edge (blocks.py FRONT_ANCHOR - handover.py writes front_anchor.on).
    "cost-front": ("cost", {"DIR_PENALTY": LOW, "OCC_WEIGHT": 1.0}),
}
FRONT_ANCHORED = {"cost-front"}         # presets that square the map's buildings about their fronts


def front_anchored(name: str) -> bool:
    """A road option whose map has its buildings squared about their street-facing edge."""
    return name in FRONT_ANCHORED

# The density field: building outline (tiles of it) near a square, smoothed over
# DENSITY_SIGMA tiles, as a share of DENSITY_FULL - about what a street lined both
# sides with buildings has (measured on Alton, 2026-10-25, along the real centrelines
# at sigma 6: the High Street 0.036 / 0.074 / 0.125 at the 10th / 50th / 90th
# percentile, Market, Turk and Amery Streets 0.006 / 0.097 / 0.136, every road in the
# centre 0 / 0.039 / 0.116). Capped at 1: the High Street reads about 0.7.
DENSITY_SIGMA = 6.0
DENSITY_FULL = 0.10


def _features(map_dir: str):
    from generator import osm
    info, proj = network.projector(map_dir)
    return osm.load_cache(os.path.join(map_dir, info["osm_cache"]), tuple(info["osm_bbox"])) or [], proj


def building_rings(map_dir: str) -> list:
    """The map's OSM buildings: each outline's points in tiles."""
    feats, proj = _features(map_dir)
    return [[proj.to_px(lat, lon) for lat, lon in f.geometry] for f in feats
            if f.kind == "way" and f.tags.get("building") and f.geometry and len(f.geometry) >= 3]


def buildings(map_dir: str, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """(footprints, outlines): the map's OSM buildings filled, and their outlines as
    lines a tile wide."""
    fill, edge = Image.new("1", (W, H), 0), Image.new("1", (W, H), 0)
    df, de = ImageDraw.Draw(fill), ImageDraw.Draw(edge)
    for pts in building_rings(map_dir):
        df.polygon(pts, fill=1)
        de.line(pts + [pts[0]], fill=1, width=1)
    return np.array(fill, bool), np.array(edge, bool)


def density(outlines: np.ndarray, sigma: float = DENSITY_SIGMA, full: float = DENSITY_FULL) -> np.ndarray:
    """0-1: building outline near each square (gaussian, sigma tiles), over `full`."""
    return np.clip(ndimage.gaussian_filter(outlines.astype(float), sigma) / full, 0.0, 1.0)


def plots(map_dir: str, area: str, H: int, W: int) -> np.ndarray | None:
    """Each square's INSPIRE title (1, 2, ...; 0 none), or None with no titles."""
    path = area_conf.extra(area, "inspire.geojson")
    if path is None:
        return None
    from . import widths
    _info, proj = network.projector(map_dir)
    img = Image.new("I", (W, H), 0)
    d = ImageDraw.Draw(img)
    for k, f in enumerate(json.load(open(path, encoding="utf-8"))["features"], 1):
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            d.polygon([proj.to_px(*widths.TO_LATLON.transform(e, n)[::-1]) for e, n in poly[0]], fill=k)
    return np.array(img, np.int32)


def fields(map_dir: str, area: str, H: int, W: int) -> dict:
    """Everything the modes need, worked out once: footprints, outlines, density,
    each square's distance in tiles to the nearest building, plot labels."""
    fill, edge = buildings(map_dir, H, W)
    return {"footprints": fill, "outlines": edge, "density": density(edge),
            "building_dist": ndimage.distance_transform_edt(~fill), "plots": plots(map_dir, area, H, W)}


_SAVED: dict = {}             # snap's constants as they were before a run's params


def reset() -> None:
    """snap as it was: no mode, no hard obstacles, its constants back."""
    snap.GRID = None
    snap.COST = None
    snap.JOG = None
    snap.OCC = None
    snap.HARD = {}
    for k, v in _SAVED.items():
        setattr(snap, k, v)
    _SAVED.clear()


def prepare(mode: str, map_dir: str, area: str, H: int, W: int, f: dict | None = None,
            params: dict | None = None) -> dict:
    """snap set up for `mode` (fields worked out unless given); `params` overrides
    snap's constants by name (DIR_PENALTY, COST_SCALE, JOG_DENSITY ...) for this run.
    Returns {"mode", "params"} for the map's info.json."""
    if mode not in PRESETS:
        raise SystemExit(f"road mode {mode!r}: one of {', '.join(PRESETS)}")
    name = mode
    mode, given = PRESETS[name]
    preset = name if name != mode else None
    params = {**given, **(params or {})}
    reset()
    used = {}
    for k, v in (params or {}).items():
        if not hasattr(snap, k):
            raise SystemExit(f"road mode: snap has no {k}")
        _SAVED.setdefault(k, getattr(snap, k))
        setattr(snap, k, v)
        used[k] = v
    if mode == "grid":
        from .run import grid_mask
        _info, proj = network.projector(map_dir)
        snap.GRID = grid_mask(map_dir, proj, H, W)
    elif mode == "cost":
        f = f or fields(map_dir, area, H, W)
        snap.COST = f["density"]
        snap.JOG = {"dist": f["building_dist"], "plots": f["plots"]}
        if snap.OCC_WEIGHT > 0:
            snap.OCC = f["footprints"]
        used = {k: getattr(snap, k) for k in ("DIR_PENALTY", "COST_SCALE", "ARM_COST", "JOG_DENSITY", "JOG_GAP",
                                              "JOG_SIDE", "JOG_END", "JOG_RUN", "OCC_WEIGHT")}
        used.update(DENSITY_SIGMA=DENSITY_SIGMA, DENSITY_FULL=DENSITY_FULL)
    out = {"mode": mode, "preset": preset, "params": used} if preset else {"mode": mode, "params": used}
    if preset and front_anchored(preset):
        out["front_anchor"] = True
    return out
