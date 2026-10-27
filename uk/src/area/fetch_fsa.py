"""Food businesses round a point, from the Food Standards Agency's hygiene ratings.

    python fetch_fsa.py <area data dir> <latitude> <longitude> [miles]

The FSA's Food Hygiene Rating Scheme lists every place that serves or sells
food - pubs, restaurants, cafes, takeaways, shops - with its business type
and where it is (OGL v3; see SOURCES.md). Fetched from its open API
(api.ratings.food.gov.uk, version 2) and written to <area>/fsa.json:
one entry per business, with name, type, address and latitude/longitude.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

API = ("https://api.ratings.food.gov.uk/Establishments?latitude={lat}&longitude={lon}"
       "&maxDistanceLimit={miles}&pageSize=500&pageNumber={page}")


def main() -> int:
    area, lat, lon = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    miles = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    out, page = [], 1
    while True:
        req = urllib.request.Request(API.format(lat=lat, lon=lon, miles=miles, page=page),
                                     headers={"x-api-version": "2", "accept": "application/json",
                                              "User-Agent": "KnoxMap-UK (Odiham map; building uses)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            got = json.load(r)
        rows = got.get("establishments") or []
        for e in rows:
            geo = e.get("geocode") or {}
            if not geo.get("latitude") or not geo.get("longitude"):
                continue
            out.append({"id": e.get("FHRSID"), "name": e.get("BusinessName"),
                        "type": e.get("BusinessType"),
                        "address": ", ".join(a for a in (e.get("AddressLine1"), e.get("AddressLine2"),
                                                         e.get("AddressLine3")) if a),
                        "lat": float(geo["latitude"]), "lon": float(geo["longitude"])})
        meta = got.get("meta") or {}
        if not rows or page >= int(meta.get("totalPages") or 1):
            break
        page += 1
    with open(os.path.join(area, "fsa.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)
    types = {}
    for e in out:
        types[e["type"]] = types.get(e["type"], 0) + 1
    print(f"{len(out)} food businesses within {miles} miles -> {os.path.join(area, 'fsa.json')}")
    print("  ", dict(sorted(types.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
