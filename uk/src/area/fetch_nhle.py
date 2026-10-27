"""Fetch Historic England listed buildings and scheduled monuments for a box.

    python fetch_nhle.py <south> <west> <north> <east> <out_dir>

Writes nhle_points.geojson, nhle_polygons.geojson and nhle_monuments.geojson.
Source: National Heritage List for England, Open Government Licence v3.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

SERVICE = ("https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/arcgis/rest/services/"
           "National_Heritage_List_for_England_NHLE_v02_VIEW/FeatureServer")
LAYERS = {"nhle_points": 0, "nhle_polygons": 3, "nhle_monuments": 6}


def query(layer: int, south: float, west: float, north: float, east: float) -> dict:
    features, offset = [], 0
    while True:
        params = {
            "where": "1=1", "outFields": "*", "f": "geojson", "outSR": 4326,
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "resultOffset": offset, "resultRecordCount": 1000,
        }
        url = f"{SERVICE}/{layer}/query?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=120) as r:
            page = json.load(r)
        got = page.get("features", [])
        features += got
        if len(got) < 1000:
            return {"type": "FeatureCollection", "features": features}
        offset += len(got)


def main() -> int:
    south, west, north, east = map(float, sys.argv[1:5])
    out = sys.argv[5]
    os.makedirs(out, exist_ok=True)
    for name, layer in LAYERS.items():
        fc = query(layer, south, west, north, east)
        with open(os.path.join(out, f"{name}.geojson"), "w", encoding="utf-8") as f:
            json.dump(fc, f)
        print(f"{name}: {len(fc['features'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
