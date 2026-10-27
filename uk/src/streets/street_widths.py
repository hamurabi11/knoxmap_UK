"""Give streets their real width wall to wall in a KnoxMap OSM download, before Generate.

    python street_widths.py <osm cache .json.gz> <area data dir> [--streets "High Street" ...]

KnoxMap draws a street from its class and lanes: a two-lane tertiary road is
7.4 m of carriageway and 3.5 m of pavement each side (14.4 m), a residential
street 6 m and 3 m each side (12 m), with a strip of grass verge. Real streets
differ both ways:
  - a market street is wider: Odiham High Street is about 20 m wall to wall,
    and the difference became forecourts between the pavement and the shops;
  - an old lane is narrower: King Street is 8.5 m, so a 12 m street ran
    under the buildings on both sides.
Every 5 m along each named way, a line is cast square to it both sides to the
first real building; where both sides hit within REACH_M the two distances
make a gap. Then, per way:
  - narrowed, every named street whose gap (NARROW_PCT percentile, so pinch
    points count; or twice the distance to the nearer side's buildings, for a
    street built up on one side) is less than KnoxMap would draw: carriageway and pavements
    shrink together (carriageway no less than MIN_ROAD_M);
  - set to the gap (median), the --streets named (default the High Street),
    wider or narrower, carriageway no wider than MAX_WIDTH_M;
  - either way paved to the walls: verge=no.
Written as width= (carriageway), sidewalk:both:width= and verge=no, which
KnoxMap's renderer reads. A way with fewer than MIN_SAMPLES gaps is left
alone (or, for a named street, takes the street's median). The download is
kept as <cache>.orig and each run starts again from it.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator.renderer import SIDEWALK_M, _way_width_m, classify  # noqa: E402

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True).transform
STEP_M = 5.0
REACH_M = 30.0
MIN_SAMPLES = 5
# Two lanes and a parking lane each side. Where the frontages stand further
# apart than that plus pavements (a market place), the rest stays forecourt.
MAX_WIDTH_M = 13.5
MIN_ROAD_M = 4.5       # a carriageway two cars can pass on, just
MIN_PAVEMENT_M = 1.0
NARROW_PCT = 25


def main() -> int:
    cache, area = sys.argv[1], sys.argv[2]
    streets = sys.argv[sys.argv.index("--streets") + 1:] if "--streets" in sys.argv else ["High Street"]
    orig = cache + ".orig"
    if not os.path.exists(orig):
        shutil.copy(cache, orig)
    with gzip.open(orig, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    with open(os.path.join(area, "buildings.geojson"), encoding="utf-8") as f:
        blds = [transform(TO_BNG, shape(ft["geometry"])) for ft in json.load(f)["features"]]
    tree = STRtree(blds)

    def geom(ft):
        g = ft["geometry"]
        return json.loads(g) if isinstance(g, str) else g

    ways = [ft for ft in payload["features"]
            if ft["kind"] == "way" and ft["tags"].get("name")
            and classify(ft["tags"]) in SIDEWALK_M]
    gaps: dict[int, list[float]] = {}
    near: dict[int, list[float]] = {}      # each side's distance, where it has a building
    for k, ft in enumerate(ways):
        line = LineString([TO_BNG(lon, lat) for lat, lon in geom(ft)])
        found = []
        for d in np.arange(0.0, line.length, STEP_M):
            p, q = line.interpolate(d), line.interpolate(min(line.length, d + 1.0))
            if p.equals(q):
                p, q = line.interpolate(max(0.0, d - 1.0)), p
            ux, uy = q.x - p.x, q.y - p.y
            n = float(np.hypot(ux, uy))
            nx, ny = -uy / n, ux / n
            sides = []
            for s in (1, -1):
                ray = LineString([(p.x, p.y), (p.x + s * nx * REACH_M, p.y + s * ny * REACH_M)])
                hits = [ray.intersection(blds[i]) for i in tree.query(ray)]
                ds = [Point(p.x, p.y).distance(h) for h in hits if not h.is_empty]
                sides.append(min(ds) if ds else None)
            if None not in sides:
                found.append(sum(sides))
            near.setdefault(k, []).extend(v for v in sides if v is not None)
        gaps[k] = found
    every = [g for k, v in gaps.items() if ways[k]["tags"]["name"] in streets for g in v]
    street_gap = float(np.median(every)) if every else None
    changed = 0
    for k, ft in enumerate(ways):
        cat = classify(ft["tags"])
        named = ft["tags"]["name"] in streets

        class _F:
            tags = ft["tags"]
        default = _way_width_m(_F, cat) + 2 * SIDEWALK_M[cat]
        # A street can be no wider than twice the way to its nearer frontage
        # (a lane with cottages on one side and a field on the other).
        one_side = (2 * float(np.percentile(near[k], NARROW_PCT))
                    if len(near.get(k, [])) >= MIN_SAMPLES else None)
        if len(gaps[k]) >= MIN_SAMPLES:
            median, low = float(np.median(gaps[k])), float(np.percentile(gaps[k], NARROW_PCT))
            if one_side is not None:
                low = min(low, one_side)
        elif named and street_gap:
            median = low = street_gap
        elif one_side is not None:
            median = low = one_side
        else:
            continue
        if named:
            gap = median
        elif low < default:
            gap = low
        else:
            continue
        # Carriageway first, from what is left after the class's pavements;
        # then the pavements take the rest, both within sensible bounds.
        road = min(MAX_WIDTH_M, max(MIN_ROAD_M, gap - 2 * SIDEWALK_M[cat]))
        pave = max(MIN_PAVEMENT_M, min(SIDEWALK_M[cat] + 0.5, (gap - road) / 2))
        print(f"  way {ft.get('osm_id') or '?'} {ft['tags']['name']}: {len(gaps[k])} gaps, "
              f"{'median' if named else f'p{NARROW_PCT}'} {gap:.1f} m (KnoxMap {default:.1f}) -> "
              f"road {road:.1f} + 2 x {pave:.1f} m pavement")
        ft["tags"]["width"] = f"{road:.1f}"
        ft["tags"]["sidewalk:both:width"] = f"{pave:.1f}"
        ft["tags"]["verge"] = "no"
        ft["tags"]["width:source"] = "frontage gap, street_widths.py"
        changed += 1
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"{changed} of {len(ways)} named ways set; {', '.join(streets)} median gap "
          f"{street_gap if street_gap is None else round(street_gap, 1)} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
