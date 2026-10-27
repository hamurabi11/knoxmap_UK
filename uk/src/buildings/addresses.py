"""Address points for a map: every property (UPRN), what it is, where it is.

Used by split_units.py (how many properties a building holds, and where) and
shops.py (what each building is). A point is:

- a business when a non-domestic EPC was lodged for its UPRN, or an OSM
  shop/amenity/office point stands there (OSM points are added as
  businesses of their own: many shops have no certificate);
- a home otherwise, with the domestic EPC's property type where there is one
  ("House", "Flat", "Maisonette", "Bungalow", "Park home"; "" when none).

Without UPRNs (the osm_lidar data level, or no uprn.csv: area/area_conf.py) the
homes are OSM's addresses: each node with addr:housenumber that is no business,
and each building tagged with one, a point per number at its middle ("1;3",
"2-8" on one side of the street: 4). Far fewer than the UPRNs (Alton: a few
hundred to some thousands), so split_units leans on the roof steps.

Points are in the real map's tiles (proj.to_px); a building carries them into
the game through its LiDAR frame (trees.to_game), the same way shops.py does.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import dataclass

USE_POINT_KEYS = ("shop", "amenity", "office", "craft", "tourism", "healthcare")
NOT_BUSINESS = {"bench", "post_box", "telephone", "waste_basket", "recycling", "parking",
                "bicycle_parking", "toilets", "drinking_water", "atm", "vending_machine",
                "charging_station", "parking_entrance", "fountain", "clock", "shelter",
                "grit_bin", "bus_stop", "taxi", "motorcycle_parking", "waste_disposal"}


# Shops that need a large sales floor: given the hall of their building
# (split_units.py, shops.py) rather than a frontage the width of a house.
BIG_FORMAT = {"supermarket", "department_store", "doityourself", "garden_centre", "furniture",
              "trade", "wholesale", "car", "variety_store"}


@dataclass
class Address:
    x: float                 # real map tiles
    y: float
    business: bool
    home_type: str = ""      # domestic EPC property type, homes only
    source: str = "uprn"     # "uprn", "osm" (a business point) or "osm_addr" (a home's address, no UPRNs)
    uprn: str = ""
    big: bool = False        # a business that needs a large floor (BIG_FORMAT)


def _latest(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = (row.get("uprn") or "").split(".")[0]
            if u and (u not in out or row["lodgement_date"] > out[u]["lodgement_date"]):
                out[u] = row
    return out


MAX_NUMBERS = 30          # house numbers one OSM building may stand for ("1-199" is a mistake)


def house_numbers(value: str) -> int:
    """How many homes an addr:housenumber names: "12" 1, "1;3;5" 3, "2-8" 4
    (a range runs along one side of the street when both ends are odd or both
    even), "4a" 1."""
    n = 0
    for part in value.replace(",", ";").split(";"):
        part = part.strip()
        a, _, b = part.partition("-")
        if b and a.strip().isdigit() and b.strip().isdigit():
            lo, hi = sorted((int(a), int(b)))
            n += (hi - lo) // (2 if (hi - lo) % 2 == 0 else 1) + 1
        elif part:
            n += 1
    return min(n, MAX_NUMBERS)


def load(map_dir: str, area: str, proj, info: dict) -> list[Address]:
    from area import area_conf
    council = area_conf.epc_dir(area)
    nondom = _latest(os.path.join(council, "epc_nondomestic.csv")) if council else {}
    dom = _latest(os.path.join(council, "epc_domestic.csv")) if council else {}
    out = []
    uprn_path = area_conf.extra(area, "uprn.csv")
    if uprn_path:
        with open(uprn_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u = row["UPRN"]
                x, y = proj.to_px(float(row["LATITUDE"]), float(row["LONGITUDE"]))
                if u in nondom:
                    out.append(Address(x, y, True, uprn=u))
                else:
                    out.append(Address(x, y, False, (dom.get(u) or {}).get("property_type") or "", uprn=u))
    with gzip.open(os.path.join(map_dir, info["osm_cache"]), "rt", encoding="utf-8") as f:
        for ft in json.load(f)["features"]:
            t = ft.get("tags") or {}
            if not uprn_path and t.get("addr:housenumber") and not any(k in t for k in USE_POINT_KEYS):
                pts = ft["geometry"] if isinstance(ft["geometry"], list) else json.loads(ft["geometry"])
                if ft["kind"] == "node" or (ft["kind"] == "way" and t.get("building") and len(pts) >= 3):
                    lat = sum(p[0] for p in pts) / len(pts)
                    lon = sum(p[1] for p in pts) / len(pts)
                    x, y = proj.to_px(lat, lon)
                    out += [Address(x, y, False, source="osm_addr") for _ in range(house_numbers(t["addr:housenumber"]))]
                continue
            if ft["kind"] != "node" or not any(k in t for k in USE_POINT_KEYS):
                continue
            if t.get("amenity") in NOT_BUSINESS or t.get("tourism") in ("information", "artwork", "viewpoint"):
                continue
            lat, lon = ft["geometry"][0]
            x, y = proj.to_px(lat, lon)
            out.append(Address(x, y, True, source="osm", big=t.get("shop") in BIG_FORMAT))
    return out
