"""The railway's land from OpenStreetMap: what tells a building on it apart.

    python fetch_rail_land.py <area data dir> S W N E

One Overpass query for the box, written to <area>/rail_land.geojson - each
area with its tags (closed ways as polygons; relations' outer rings):
- landuse=railway (the railway's estate: lines, yards, sidings);
- railway=station / platform / yard drawn as areas, and public_transport=
  station / platform ones that are a railway's (train=yes; not a bus station's).
blocks.py reads it (buildings/rail_kit.py): a station's buildings are kept on
it, LiDAR-only shapes on it are not buildings (trains, canopies, kit). Kept
apart from KnoxMap's download, like fetch_road_nodes.py, so it needs no fresh
download. OpenStreetMap, ODbL (SOURCES.md).
"""
from __future__ import annotations

import collections
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator import osm  # noqa: E402

QUERY = """[out:json][timeout:120];
(
  wr["landuse"="railway"]({s},{w},{n},{e});
  wr["railway"~"^(station|platform|yard)$"]({s},{w},{n},{e});
  wr["public_transport"~"^(station|platform)$"]["train"="yes"]({s},{w},{n},{e});
);
out geom;"""


def rings(el):
    """The element's outer rings as [[lon, lat], ...] lists (closed)."""
    if el["type"] == "way":
        g = [[p["lon"], p["lat"]] for p in el.get("geometry", [])]
        return [g] if len(g) >= 4 and g[0] == g[-1] else []
    out = []
    for m in el.get("members", []):
        if m.get("role") in ("outer", "") and m.get("geometry"):
            g = [[p["lon"], p["lat"]] for p in m["geometry"]]
            if len(g) >= 4 and g[0] == g[-1]:
                out.append(g)
    return out


def main() -> int:
    area = sys.argv[1]
    s, w, n, e = (float(v) for v in sys.argv[2:6])
    query = QUERY.format(s=s, w=w, n=n, e=e)
    data, errors = None, []
    for endpoint in osm.ANSWERING_ENDPOINTS:
        try:
            r = requests.post(endpoint, data={"data": query}, headers=osm.HEADERS, timeout=(10, 150))
            if r.status_code == 200:
                data = r.json()
                break
            errors.append(f"{endpoint}: {r.status_code}")
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{endpoint}: {exc}")
    if data is None:
        print("no Overpass server answered:", *errors, sep="\n  ")
        return 1
    feats = []
    kinds = collections.Counter()
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        for ring in rings(el):
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"osm_id": el["id"], "osm_type": el["type"], **tags}})
            kind = ("landuse=railway" if tags.get("landuse") == "railway"
                    else f"railway={tags['railway']}" if tags.get("railway")
                    else f"public_transport={tags.get('public_transport')}")
            kinds[kind] += 1
    out = os.path.join(area, "rail_land.geojson")
    json.dump({"type": "FeatureCollection", "features": feats,
               "attribution": "© OpenStreetMap contributors, ODbL"}, open(out, "w", encoding="utf-8"))
    print(f"{len(feats)} areas -> {out}")
    for kind, k in kinds.most_common():
        print(f"  {k:4d}  {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
