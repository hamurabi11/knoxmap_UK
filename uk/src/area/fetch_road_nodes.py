"""Crossings, traffic signals, give-way and stop points from OpenStreetMap.

    python fetch_road_nodes.py <area data dir> S W N E

What the road markings need (roadgraph/markings.py) and KnoxMap's own
download leaves out (its filter keeps no highway nodes): one Overpass query
for the box, written to <area>/road_nodes.geojson - each point with its
tags:
- highway=crossing (crossing=zebra|traffic_signals|uncontrolled|marked...,
  crossing_ref=zebra|pelican|puffin|toucan...), highway=traffic_signals
  (traffic_signals:direction), highway=give_way|stop|mini_roundabout;
- amenity=post_box, amenity=telephone, amenity=bench and barrier=bollard:
  the street furniture markings.py stands on the pavement (PLAN E);
- where a 1990s town had its telephone kiosks (furniture.kiosk_sites): post
  offices, pubs, railway stations (points, or a building's middle) and the
  centres of places (place=village, hamlet, suburb...);
- the town centre's furniture (townscape.py, 2026-10-14): bus stops
  (highway=bus_stop, shelter=yes), litter bins (amenity=waste_basket), grit
  bins, cycle stands (amenity=bicycle_parking), street cabinets
  (man_made=street_cabinet: BT's green ones), and where a yellow box goes -
  fire and ambulance stations (points or a building's middle) and any point
  tagged as a box junction.
A run adds to what an earlier one fetched (by OSM type and id): a smaller
box keeps the points a bigger one found.
Kept apart from KnoxMap's download, so changing it needs no fresh download
and full rerun of the pipeline. OpenStreetMap, ODbL (SOURCES.md).
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
  node["highway"~"^(crossing|traffic_signals|give_way|stop|mini_roundabout)$"]({s},{w},{n},{e});
  node["crossing"]({s},{w},{n},{e});
  node["amenity"~"^(post_box|telephone|bench)$"]({s},{w},{n},{e});
  node["barrier"="bollard"]({s},{w},{n},{e});
  nwr["amenity"~"^(post_office|pub)$"]({s},{w},{n},{e});
  nwr["railway"="station"]({s},{w},{n},{e});
  node["highway"="bus_stop"]({s},{w},{n},{e});
  node["amenity"~"^(waste_basket|grit_bin|bicycle_parking)$"]({s},{w},{n},{e});
  node["man_made"="street_cabinet"]({s},{w},{n},{e});
  nwr["amenity"="fire_station"]({s},{w},{n},{e});
  nwr["emergency"="ambulance_station"]({s},{w},{n},{e});
  node["markings"="box_junction"]({s},{w},{n},{e});
  node["box_junction"="yes"]({s},{w},{n},{e});
  node["place"~"^(town|village|hamlet|suburb|neighbourhood|quarter)$"]({s},{w},{n},{e});
);
out center;"""


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
    for el in data.get("elements", []):
        at = el if el.get("type") == "node" else el.get("center")
        if not at:
            continue
        props = {"osm_id": el["id"], **el.get("tags", {})}
        if el.get("type") != "node":
            props["osm_type"] = el["type"]
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [at["lon"], at["lat"]]},
                      "properties": props})
    out = os.path.join(area, "road_nodes.geojson")
    if os.path.exists(out):
        key = lambda f: (f["properties"].get("osm_type", "node"), f["properties"]["osm_id"])
        new = {key(f) for f in feats}
        feats = [f for f in json.load(open(out, encoding="utf-8"))["features"] if key(f) not in new] + feats
    json.dump({"type": "FeatureCollection", "features": feats,
               "attribution": "© OpenStreetMap contributors, ODbL"}, open(out, "w", encoding="utf-8"))
    kinds = collections.Counter()
    for f in feats:
        p = f["properties"]
        kinds[(p.get("highway") or p.get("amenity") or p.get("man_made") or p.get("emergency") or "-",
               p.get("crossing") or p.get("crossing_ref") or "")] += 1
    print(f"{len(feats)} points -> {out}")
    for (kind, sub), k in kinds.most_common():
        print(f"  {k:4d}  {kind}{' / ' + sub if sub else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
