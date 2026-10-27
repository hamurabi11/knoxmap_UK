"""Which roads lose their connections when KnoxMap straightens them.

    python road_links.py <knoxmap map dir> [--list N]

Every end of every way (roads, service lanes, paths) that meets another way
in reality - the same node, or within LINK_M - should still meet it in the
game, after the Knox County straightening (generator/octilinear.py). Counts,
by kind of way, the ends that do and the ones left more than GAP tiles off,
and lists the worst. Also counts closed loops (a crescent drawn as a ring).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import shapely
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator import osm  # noqa: E402
from generator.octilinear import ROAD_CATEGORIES, straighten_roads  # noqa: E402
from generator.renderer import Projector, _is_polygon, classify  # noqa: E402

LINK_M = 1.0
GAP = 1.5
WAYS = ROAD_CATEGORIES | {"dirt_path", "paved_path"}


def main() -> int:
    map_dir = sys.argv[1]
    n_list = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 10
    name = os.path.basename(os.path.normpath(map_dir))
    info = json.load(open(os.path.join(map_dir, f"{name}_info.json")))
    b = info["bbox"]
    proj = Projector.build(b["south"], b["west"], b["north"], b["east"], info["meters_per_tile"],
                           info.get("rotation") or 0.0)
    cache = os.path.join(map_dir, info["osm_cache"])
    real = osm.load_cache(cache, tuple(info["osm_bbox"]))
    game = osm.load_cache(cache, tuple(info["osm_bbox"]))
    if info.get("straight_roads"):
        straighten_roads(game, proj, classify, _is_polygon, map_dir=map_dir)

    def ways(feats):
        return {f.osm_id: f for f in feats if f.kind == "way" and len(f.geometry) >= 2
                and not _is_polygon(f) and classify(f.tags) in WAYS}
    rw, gw = ways(real), ways(game)
    ids = [k for k in rw if k in gw]
    rl = [shapely.LineString([proj.to_px(la, lo) for la, lo in rw[k].geometry]) for k in ids]
    gl = [shapely.LineString([proj.to_px(la, lo) for la, lo in gw[k].geometry]) for k in ids]
    tree = STRtree(rl)
    W, H = proj.width, proj.height
    ok, broken, worst, loops = Counter(), Counter(), [], Counter()
    for i, k in enumerate(ids):
        cat = classify(rw[k].tags)
        if rl[i].is_closed:
            loops[cat] += 1
        for end in (0, -1):
            p = shapely.Point(rl[i].coords[end])
            if not (0 <= p.x < W and 0 <= p.y < H):
                continue
            for j in tree.query(p.buffer(LINK_M)):
                j = int(j)
                if j == i or rl[j].distance(p) > LINK_M:
                    continue
                q = shapely.Point(gl[i].coords[end])
                gap = gl[j].distance(q)
                key = f"{cat} -> {classify(rw[ids[j]].tags)}"
                if gap > GAP:
                    broken[key] += 1
                    worst.append((gap, k, rw[k].tags.get("name") or rw[k].tags.get("highway"),
                                  rw[ids[j]].tags.get("name") or rw[ids[j]].tags.get("highway"),
                                  tuple(round(v) for v in q.coords[0])))
                else:
                    ok[key] += 1
                break
    print(f"{name}: way ends that meet another way in reality")
    for key in sorted(set(ok) | set(broken), key=lambda k: -broken[k]):
        print(f"  {key:30s} still meet {ok[key]:4d}   left apart {broken[key]:4d}")
    print("closed loops by kind:", dict(loops))
    for gap, k, a, bname, at in sorted(worst, reverse=True)[:n_list]:
        print(f"  {gap:5.1f} tiles: way {k} ({a}) off {bname}, at game {at}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
