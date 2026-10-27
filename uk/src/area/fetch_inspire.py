"""Cut an area's HM Land Registry INSPIRE Index Polygons out of a council's GML.

    python fetch_inspire.py <council .gml> <area data dir> <e0> <n0> <e1> <n1>

The GML comes from https://use-land-property-data.service.gov.uk/datasets/inspire
(one zip per council, free, OGL; see SOURCES.md for the attribution it needs).
Writes <area>/inspire.geojson: the polygons touching the box, in British
National Grid metres (EPSG:27700), each with its INSPIRE id.
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET

from shapely.geometry import Polygon, box, mapping

GML = "{http://www.opengis.net/gml/3.2}"
LR = "{www.landregistry.gov.uk}"


def rings(poly_el):
    ext = poly_el.find(f"{GML}exterior/{GML}LinearRing/{GML}posList")
    holes = poly_el.findall(f"{GML}interior/{GML}LinearRing/{GML}posList")

    def pts(el):
        v = [float(t) for t in el.text.split()]
        return list(zip(v[0::2], v[1::2]))
    return pts(ext), [pts(h) for h in holes]


def main() -> int:
    src, area = sys.argv[1], sys.argv[2]
    e0, n0, e1, n1 = map(float, sys.argv[3:7])
    clip = box(e0, n0, e1, n1)
    out = []
    for _ev, el in ET.iterparse(src, events=("end",)):
        if el.tag != f"{LR}PREDEFINED":
            continue
        pid = el.findtext(f"{LR}INSPIREID")
        for poly_el in el.iter(f"{GML}Polygon"):
            ext, holes = rings(poly_el)
            if len(ext) < 4:
                continue
            p = Polygon(ext, holes)
            if not p.is_valid:
                p = p.buffer(0)
            if p.intersects(clip):
                out.append({"type": "Feature", "properties": {"inspire_id": pid},
                            "geometry": mapping(p)})
        el.clear()
    path = os.path.join(area, "inspire.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "crs": "EPSG:27700", "features": out}, f)
    print(f"wrote {path}: {len(out)} polygons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
