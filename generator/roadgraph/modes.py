"""How built-up streets are routed.

    graph   vanilla's sixteen directions everywhere
    cost    a frontage-density field (snap.COST): a step off the grid costs more the more
            building outline is near it - 45 degrees a little, 2:1 a lot - and where it is
            dense a road turns only at a junction or a gap in the frontage (snap.JOG); each
            building square under the carriageway costs too (snap.OCC), so a road keeps off
            the buildings it runs past

prepare() sets snap up for a mode; reset() puts it back. The fields are worked out
from the map's own OSM buildings.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import snap

LOW = {"grid": 0.0, "45": 0.25, "2:1": 0.75}
PRESETS = {
    "graph": ("graph", {}),
    "cost": ("cost", {"DIR_PENALTY": LOW, "OCC_WEIGHT": 1.0}),
}

# The density field: building outline (tiles of it) near a square, smoothed over
# DENSITY_SIGMA tiles, as a share of DENSITY_FULL - about what a street lined both
# sides with buildings has (a high street reads about 0.7; capped at 1).
DENSITY_SIGMA = 6.0
DENSITY_FULL = 0.10
DIST_CAP = 16                # tiles: building distances past this are all "far" (the jog rule
                             # asks up to half the widest road plus snap.JOG_GAP)


def building_rings(features, proj) -> list:
    """The map's OSM buildings: each outline's points in tiles."""
    return [[proj.to_px(lat, lon) for lat, lon in f.geometry] for f in features
            if f.kind == "way" and f.tags.get("building") and f.geometry and len(f.geometry) >= 3]


def buildings(rings, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """(footprints, outlines): the buildings filled, and their outlines as lines a tile wide."""
    fill, edge = Image.new("1", (W, H), 0), Image.new("1", (W, H), 0)
    df, de = ImageDraw.Draw(fill), ImageDraw.Draw(edge)
    for pts in rings:
        df.polygon(pts, fill=1)
        de.line(pts + [pts[0]], fill=1, width=1)
    return np.array(fill, bool), np.array(edge, bool)


def gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    """a blurred by a gaussian of `sigma` tiles (to 4 sigma; zeros beyond the edge)."""
    r = int(math.ceil(4 * sigma))
    x = np.arange(-r, r + 1)
    k1 = np.exp(-x * x / (2 * sigma * sigma))
    k1 /= k1.sum()
    return snap.convolve(a.astype(float), np.outer(k1, k1))


def density(outlines: np.ndarray, sigma: float = DENSITY_SIGMA, full: float = DENSITY_FULL) -> np.ndarray:
    """0-1: building outline near each square (gaussian, sigma tiles), over `full`."""
    return np.clip(gaussian(outlines, sigma) / full, 0.0, 1.0)


def distance(fill: np.ndarray, cap: int = DIST_CAP) -> np.ndarray:
    """Each square's distance in tiles to the nearest filled one, exact up to `cap`
    (beyond it, cap + 1): the nearest along each column, then across the row."""
    H, W = fill.shape
    far = cap + 1
    rows = np.arange(H)[:, None]
    above = np.maximum.accumulate(np.where(fill, rows, -10 ** 9), axis=0)
    below = np.minimum.accumulate(np.where(fill, rows, 10 ** 9)[::-1], axis=0)[::-1]
    g = np.minimum(np.minimum(rows - above, below - rows), far).astype(float)
    best = g * g
    for dx in range(1, far):
        for src, dst in ((slice(dx, None), slice(None, -dx)), (slice(None, -dx), slice(dx, None))):
            best[:, dst] = np.minimum(best[:, dst], g[:, src] ** 2 + dx * dx)
    return np.minimum(np.sqrt(best), far)


def fields(rings, H: int, W: int) -> dict:
    """Everything the cost mode needs, worked out once: footprints, density, and each
    square's distance in tiles to the nearest building."""
    fill, edge = buildings(rings, H, W)
    return {"footprints": fill, "density": density(edge), "building_dist": distance(fill)}


_SAVED: dict = {}             # snap's constants as they were before a run's params


def reset() -> None:
    """snap as it was: no mode, its constants back."""
    snap.COST = None
    snap.JOG = None
    snap.OCC = None
    for k, v in _SAVED.items():
        setattr(snap, k, v)
    _SAVED.clear()


def prepare(mode: str, rings, H: int, W: int) -> dict:
    """snap set up for `mode` ("graph" or "cost"). Returns {"mode", "params"} for the
    map's info.json."""
    if mode not in PRESETS:
        raise ValueError(f"road mode {mode!r}: one of {', '.join(PRESETS)}")
    kind, params = PRESETS[mode]
    reset()
    for k, v in params.items():
        _SAVED.setdefault(k, getattr(snap, k))
        setattr(snap, k, v)
    used = dict(params)
    if kind == "cost":
        f = fields(rings, H, W)
        snap.COST = f["density"]
        snap.JOG = {"dist": f["building_dist"], "plots": None}
        if snap.OCC_WEIGHT > 0:
            snap.OCC = f["footprints"]
        used = {k: getattr(snap, k) for k in ("DIR_PENALTY", "COST_SCALE", "ARM_COST", "JOG_DENSITY", "JOG_GAP",
                                              "JOG_END", "JOG_RUN", "OCC_WEIGHT")}
        used.update(DENSITY_SIGMA=DENSITY_SIGMA, DENSITY_FULL=DENSITY_FULL)
    return {"mode": mode, "params": used}
