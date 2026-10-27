"""Where buildings and the road graph's street overlap on a built map, and why.

    python road_overlap.py <map dir> <area data dir> [--top N]

For every placed building square on the street the hand-over painted
(carriageway, parking bays, verge, pavement: <map>_roadgraph_tiles.npz,
<map>_roadside.npz), the street's link (its nearest carriageway square's)
and the cause, from the building's real outline and the link's real line
(the hand-over's state, <map>_roadgraph_state.pkl, same tile frame):
- width: the real building stands closer to the real centreline than the
  street is drawn on that side (half the carriageway + bays + verge +
  pavement) - the drawn corridor is wider than the real gap;
- position: it really stands clear of the drawn width, but placed it does
  not - the road and building moved apart differently (two roads pulling
  apart, a bend, a junction corner);
- no real outline: a building added from the LiDAR.
Counts in squares, by layer and cause, and the links with most, so a road
fix (corridor fitted to the frontage gap, or buildings as a routing cost:
PLAN H) can be aimed at the larger share.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)

from buildings import blocks as B  # noqa: E402
from roadgraph import roadside  # noqa: E402

LAYERS = ("carriageway", "bays", "verge", "pavement")
SLACK = 0.5        # tiles: a building this little inside the drawn width is the drawing's rounding, not width


def base_id(props) -> str:
    """The OSM id a placed piece or unit came from."""
    return re.sub(r"(u\d+|p\d+)+$", "", str(props.get("parent_osm_id") or props.get("osm_id")))


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json"), encoding="utf-8"))
    links, paths, proj, owner, _half, _town, chosen = pickle.load(
        open(os.path.join(map_dir, f"{name}_roadgraph_state.pkl"), "rb"))
    hand = roadside.handedness(proj)
    side = np.load(os.path.join(map_dir, f"{name}_roadside.npz"))
    H, W = owner.shape
    layer = np.zeros((H, W), np.int8)                 # 0 none, else 1 + LAYERS index (carriageway wins)
    for k, mask in reversed(list(enumerate((owner != 0, side["parking"], side["verge"], side["pavement"])))):
        layer[mask] = k + 1
    _d, (iy, ix) = ndimage.distance_transform_edt(owner <= 0, return_indices=True)   # -1: road from two halves, no link
    link_at = owner[iy, ix] - 1                       # each square's nearest carriageway square's link

    real = {str(f["properties"]["osm_id"]): shape(f["geometry"])
            for f in json.load(open(os.path.join(area, "buildings.geojson"), encoding="utf-8"))["features"]}
    real.update(B.relation_outlines(map_dir, info))
    placed = json.load(open(os.path.join(map_dir, f"{name}_buildings.geojson"), encoding="utf-8"))["features"]

    tiles = Counter()                                 # (layer, cause) -> squares
    by_link = defaultdict(Counter)                    # link -> cause -> squares
    deficit = []                                      # drawn half minus real distance, per building and link (width)
    groups = 0
    for ft in placed:
        g = shape(ft["geometry"])
        for part in getattr(g, "geoms", [g]):
            px = shapely.Polygon([proj.to_px(lat, lon) for lon, lat in part.exterior.coords])
            x0, y0, x1, y1 = px.bounds
            x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
            x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
            if x1 <= x0 or y1 <= y0:
                continue
            mk = B.tile_mask([px], x0, y0, x1 - x0, y1 - y0) > 0
            lay = layer[y0:y1, x0:x1]
            on = mk & (lay > 0)
            if not on.any():
                continue
            rid = base_id(ft["properties"])
            outline = real.get(rid)
            ys, xs = np.nonzero(on)
            for i in np.unique(link_at[ys + y0, xs + x0]):
                sel = link_at[ys + y0, xs + x0] == i
                groups += 1
                if outline is None:
                    cause = "no real outline"
                else:
                    rp = B.lonlat_to_px(proj, outline)
                    line = shapely.LineString(links[i].pts)
                    d_real = line.distance(rp)
                    # Which side of the real line the building really stands.
                    c = rp.centroid
                    s = line.project(c)
                    a = line.interpolate(max(0.0, s - 0.5))
                    b = line.interpolate(min(line.length, s + 0.5))
                    cross = (b.x - a.x) * (c.y - (a.y + b.y) / 2) - (b.y - a.y) * (c.x - (a.x + b.x) / 2)
                    v, p, w = chosen[i][0 if cross * hand > 0 else 1]
                    drawn = int(links[i].width) / 2 + v + p + w
                    cause = "width" if d_real < drawn - SLACK else "position"
                    if cause == "width":
                        deficit.append(drawn - d_real)
                for k in range(len(LAYERS)):
                    n = int((lay[ys[sel], xs[sel]] == k + 1).sum())
                    if n:
                        tiles[(LAYERS[k], cause)] += n
                        by_link[i][cause] += n

    causes = ("width", "position", "no real outline")
    total = sum(tiles.values())
    print(f"{name}: {total} building squares on the street ({groups} building-link pairs)")
    print(f"  {'':12s}" + "".join(f"{c:>17s}" for c in causes))
    for lay_ in LAYERS:
        print(f"  {lay_:12s}" + "".join(f"{tiles[(lay_, c)]:17d}" for c in causes))
    print(f"  {'all':12s}" + "".join(f"{sum(tiles[(l_, c)] for l_ in LAYERS):17d}" for c in causes))
    if deficit:
        q = np.percentile(deficit, [50, 90])
        print(f"  width: drawn street wider than the real gap by {q[0]:.1f} tiles (median), {q[1]:.1f} (90th percentile)")
    print(f"  links with most:")
    for i, cnt in sorted(by_link.items(), key=lambda kv: -sum(kv[1].values()))[:top]:
        l_ = links[i]
        print(f"    {sum(cnt.values()):5d}  {l_.tags.get('name') or '(unnamed)':28s} {l_.cls:10s} width {l_.width}, "
              f"sides {chosen[i]}  " + ", ".join(f"{c} {n}" for c, n in cnt.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
