"""Road widths from LiDAR (new_road_approach.md, build order 2): the
carriageway from intensity, the open corridor from the height above ground.

Across each link every STEP_M a transect TRANSECT_M each side of the OSM
centreline, sampled every SAMPLE_M:
- asphalt is dark to the scanner's laser (median 85-105 on Odiham's roads
  against 160-190 for pavement, verge and garden), so the carriageway is the
  dark run round the darkest point within SEEK_M of the line: grown out while
  darker than halfway between that point and the transect's far ends (8-15 m
  out), a gap of one sample allowed (a white line, a drain);
- the corridor is the open ground round it: grown out until something stands
  OBSTACLE_M above the ground for a metre (a wall, hedge, building or tree),
  a parked car's width allowed through.
- the roadside each side (roadside.py decides from it): from the
  carriageway's edge out, a run of bright ground (grass) then a dark one (a
  tarmac pavement), each VERGE_RUN_M long, and then neither more of the dark
  (a drive) nor nothing - a verge and the pavement behind it.
A transect is used if the asphalt stands out by CONTRAST at least; a link's
widths are the medians of its transects (a named street's pooled), given
MIN_TRANSECTS, AGREE_SHARE of them within AGREE_M of it. Pavements,
driveways and parking bays are often tarmac too and only ever widen the dark
run, so the width used is kept within the class's range
(network.WIDTH_RANGE). A dual carriageway is not measured (its centreline is
the reservation).

Only the tiles already downloaded are read: the DSM and DTM mosaics in the
area's lidar/ folder and the National LiDAR Programme intensity tiles
(Intensity_*.tif with their .tfw) in data/raw/nlp* (INTENSITY_DIRS). Each
survey reads on its own scale (2021 about 0.4 of 2019): every tile is
stretched so its 10th and 90th percentiles land on INTENSITY_REF, the 2019
Odiham tile's, which the thresholds here were set on.
"""
from __future__ import annotations

import glob
import math
import os
import re

import numpy as np
import shapely
from PIL import Image
from pyproj import Transformer

from .network import WIDTH_RANGE, whole

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True)
TO_LATLON = Transformer.from_crs(27700, 4326, always_xy=True)     # gives (lon, lat)
STEP_M, TRANSECT_M, SAMPLE_M = 4.0, 15.0, 0.5
END_SKIP_M = 8.0          # transects this near a link's end see the other road
SEEK_M = 3.0              # OSM's line may be this far off the asphalt's middle
FAR_M = 8.0               # beyond this, the transect is the road's surroundings
CONTRAST = 40.0           # intensity the asphalt must be below its surroundings
OBSTACLE_M = 1.0          # this high above the ground stops the corridor
CAR_M = 2.5               # ...unless it is no wider than a parked car
MIN_TRANSECTS = 5
AGREE_M, AGREE_SHARE = 1.5, 0.4   # this share of transects within this of the median
VERGE_RUN_M = (1.0, 3.5)  # a verge, and the pavement behind it, each this long
INTENSITY_DIRS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw", "nlp*")
INTENSITY_REF = (90.0, 385.0)   # 10th and 90th percentile of NLP 2019 SU75sw
ROADSIDE_M = 9.0          # how far out from the carriageway the roadside is read


class Lidar:
    """Height above ground and intensity on one 1 m grid over a BNG box
    (row 0 at the north edge)."""

    def __init__(self, box, lidar_dir: str, intensity_dir: str = INTENSITY_DIRS):
        self.box = e0, n0, e1, n1 = box
        dsm, dtm = self._cached(lidar_dir, "dsm"), self._cached(lidar_dir, "dtm")
        self.ndsm = dsm - dtm
        self.inten = np.full_like(dsm, np.nan)
        Image.MAX_IMAGE_PIXELS = None
        for tif in glob.glob(os.path.join(intensity_dir, "Intensity_*.tif")):
            a = np.array(Image.open(tif), np.float32)
            a[a <= 0] = np.nan
            p10, p90 = np.nanpercentile(a, (10, 90))
            a = INTENSITY_REF[0] + (a - p10) * (INTENSITY_REF[1] - INTENSITY_REF[0]) / (p90 - p10)
            with open(os.path.splitext(tif)[0] + ".tfw") as f:
                tfw = [float(v) for v in f.read().split()]
            te0, tn1 = round(tfw[4] - tfw[0] / 2), round(tfw[5] - tfw[3] / 2)   # pixel centres to edges
            self._paste(self.inten, a, te0, tn1)

    def _cached(self, folder: str, kind: str) -> np.ndarray:
        e0, n0, e1, n1 = self.box
        out = np.full((n1 - n0, e1 - e0), np.nan, np.float32)
        for tif in glob.glob(os.path.join(folder, f"{kind}_*.tif")):
            te0, _tn0, _te1, tn1 = map(int, re.findall(r"\d+", os.path.basename(tif))[:4])
            a = np.array(Image.open(tif), np.float32)
            a[a < -1000] = np.nan
            self._paste(out, a, te0, tn1)
        return out

    def _paste(self, out, a, te0: int, tn1: int) -> None:
        """a (its top-left corner at te0, tn1) into out where out has nothing yet."""
        e0, n0, e1, n1 = self.box
        ea, eb = max(te0, e0), min(te0 + a.shape[1], e1)
        na, nb = max(tn1 - a.shape[0], n0), min(tn1, n1)
        if ea >= eb or na >= nb:
            return
        dst = out[n1 - nb:n1 - na, ea - e0:eb - e0]
        src = a[tn1 - nb:tn1 - na, ea - te0:eb - te0]
        empty = np.isnan(dst)
        dst[empty] = src[empty]

    def sample(self, grid: np.ndarray, es: np.ndarray, ns: np.ndarray) -> np.ndarray:
        e0, n0, e1, n1 = self.box
        c = np.floor(es - e0).astype(int)
        r = np.floor(n1 - ns).astype(int)
        ok = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
        out = np.full(es.shape, np.nan, np.float32)
        out[ok] = grid[r[ok], c[ok]]
        return out


def to_bng(proj, pts):
    out = []
    for x, y in pts:
        lat, lon = proj.to_latlon(x, y)
        out.append(TO_BNG.transform(lon, lat))
    return out


def box_for(proj, links, margin: int = 30):
    es, ns = zip(*(p for l in links for p in to_bng(proj, [l.pts[0], l.pts[len(l.pts) // 2], l.pts[-1]])))
    return (int(min(es)) - margin, int(min(ns)) - margin, int(max(es)) + margin, int(max(ns)) + margin)


def _grow(ok: np.ndarray, i0: int, step: int, gap: int) -> int:
    """From i0, how far along (by step) ok holds, gaps of up to gap allowed."""
    i, last, miss = i0, i0, 0
    while 0 <= i + step < len(ok):
        i += step
        if ok[i]:
            last, miss = i, 0
        else:
            miss += 1
            if miss > gap:
                break
    return last


def transect(inten: np.ndarray, ndsm: np.ndarray, offs: np.ndarray) -> dict | None:
    """One transect's carriageway and corridor (metres), or None if the
    asphalt does not show."""
    near = np.abs(offs) <= SEEK_M
    if np.isnan(inten).mean() > 0.2 or not np.any(near & ~np.isnan(inten)):
        return None
    smooth = np.convolve(np.nan_to_num(inten, nan=np.nanmedian(inten)), np.ones(3) / 3, "same")
    idx = np.nonzero(near)[0]
    c0 = idx[np.argmin(smooth[idx])]
    dark = smooth[c0]
    far = np.abs(offs - offs[c0]) >= FAR_M
    if not np.any(far):
        return None
    surround = float(np.nanmedian(inten[far]))
    if surround - dark < CONTRAST:
        return None
    asphalt = (smooth < dark + 0.5 * (surround - dark)) & ~(ndsm > 2.5)
    lo, hi = _grow(asphalt, c0, -1, 1), _grow(asphalt, c0, 1, 1)
    clear = ~(ndsm > OBSTACLE_M)
    car = int(round(CAR_M / SAMPLE_M))
    clo, chi = _grow(clear, c0, -1, car), _grow(clear, c0, 1, car)
    sides = tuple(_roadside(asphalt, ~clear, offs, edge, step) for edge, step in ((hi, 1), (lo, -1)))
    return {"carriageway": (hi - lo + 1) * SAMPLE_M, "offset": float(offs[(lo + hi) // 2]),
            "right": (lo - clo) * SAMPLE_M, "left": (chi - hi) * SAMPLE_M,
            "corridor": (chi - clo + 1) * SAMPLE_M, "sides": sides}


def _roadside(asphalt: np.ndarray, blocked: np.ndarray, offs: np.ndarray, edge: int, step: int):
    """(verge, pavement) in metres on one side of a transect, from its
    carriageway's edge sample out (step +1 left, -1 right), or None: a
    bright run then a dark one, each VERGE_RUN_M long, then something else."""
    runs: list = []
    i = edge + step
    while 0 <= i < len(asphalt) and abs(offs[i] - offs[edge]) <= ROADSIDE_M:
        if blocked[i]:
            runs.append(["X", 1])
            break
        k = "D" if asphalt[i] else "B"
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
        i += step
    lo, hi = (round(m / SAMPLE_M) for m in VERGE_RUN_M)
    if len(runs) >= 3 and runs[0][0] == "B" and runs[1][0] == "D" and all(lo <= r[1] <= hi for r in runs[:2]):
        return runs[0][1] * SAMPLE_M, runs[1][1] * SAMPLE_M
    return None


def measure(links, proj, lidar: Lidar, sides: dict | None = None) -> dict:
    """{link index: medians of carriageway, corridor, the roadside each
    side, the asphalt's offset from OSM's line; and how many transects}. A
    named street's links share their transects (one street, one width).
    sides, if given, gets each link's own transects' roadside: {link index:
    [(left, right)]}, each (verge m, pavement m) or None, left facing u to v."""
    offs = np.arange(-TRANSECT_M, TRANSECT_M + SAMPLE_M / 2, SAMPLE_M)
    pooled: dict = {}
    for i, l in enumerate(links):
        key = (l.tags.get("name"), l.cls) if l.tags.get("name") else i
        rows = pooled.setdefault(key, ([], []))
        rows[0].append(i)
        if l.tags.get("dual_carriageway"):
            continue
        line = shapely.LineString(to_bng(proj, l.pts))
        for d in np.arange(END_SKIP_M, line.length - END_SKIP_M, STEP_M):
            p, q = line.interpolate(d), line.interpolate(d + 1.0)
            dx, dy = q.x - p.x, q.y - p.y
            n = math.hypot(dx, dy)
            if n == 0:
                continue
            nx, ny = -dy / n, dx / n          # to the left, facing along the link
            es, ns = p.x + nx * offs, p.y + ny * offs
            t = transect(lidar.sample(lidar.inten, es, ns), lidar.sample(lidar.ndsm, es, ns), offs)
            if t:
                rows[1].append(t)
                if sides is not None:
                    sides.setdefault(i, []).append(t["sides"])
    out = {}
    for members, rows in pooled.values():
        cw = np.array([r["carriageway"] for r in rows])
        if len(rows) < MIN_TRANSECTS or np.mean(np.abs(cw - np.median(cw)) <= AGREE_M) < AGREE_SHARE:
            continue
        med = {k: round(float(np.median([r[k] for r in rows])), 1)
               for k in ("carriageway", "corridor", "left", "right", "offset")}
        med["transects"] = len(rows)
        for i in members:
            out[i] = med
    return out


def apply(links, measured: dict, m_per_tile: float = 1.0) -> int:
    """Each measured link's width the whole number of tiles nearest its
    carriageway, within its class's range. The number changed."""
    changed = 0
    for i, m in measured.items():
        lo, hi = WIDTH_RANGE[links[i].cls]
        w = min(hi, max(lo, whole(m["carriageway"] / m_per_tile)))
        changed += w != links[i].width
        links[i].width = w
    return changed
