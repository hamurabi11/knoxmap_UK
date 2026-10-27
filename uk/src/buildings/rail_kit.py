"""Buildings on the railway: which are the railway's own, which are in its way.

Read by blocks.py. A lookup on what OSM, the addresses and the LiDAR say, not
a learner (user, 2026-10-18):

- The railway's own kit stays where it stands, kept off the line as nothing
  else is (the track gives way to it, streets/rails.py): a building tagged as
  one (STATION_BUILDINGS, a railway=* of RAILWAY_BUILDINGS, or a train
  station's public_transport=station), or standing inside a station or
  platform area (<area>/rail_land.geojson, area/fetch_rail_land.py). Tested
  before the footbridge rule: a station a storey up over its line stays.
- A footbridge (building=bridge, or any building a storey up, across a line)
  is left out: built on the ground it would block the track.
- A LiDAR-only shape on the line (LIDAR_NEAR_M of it) or on the railway's
  land is not a building: parked carriages (Alton's are 4 x 15 and 4 x 24 m,
  along the track), canopies, kit. Trains and wagons in their place are for
  later (PLAN).
- Everything else is kept RAIL_HALF_M off the line, as off a road's
  carriageway (blocks.py roads_to_clear).
"""
from __future__ import annotations

import json
import os

import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import transform

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True).transform
STATION_BUILDINGS = {"train_station", "transportation", "signal_box", "lift_tower", "station"}
STATION_AREAS = {"station", "platform"}
RAILWAY_BUILDINGS = {"station", "signal_box", "platform", "halt", "ventilation_shaft", "engine_shed",
                     "workshop", "goods_shed", "wash"}       # railway=* values a building of the railway's carries
RAIL_HALF_M = 2.5         # the track's bed each side of the line (rails.py: three squares and one each side)
LIDAR_NEAR_M = 4.0        # a LiDAR-only shape this near a line is on the railway
BRIDGE_TOUCH_M = 1.0      # a footbridge: across a line, to within this


def is_rail_line(feat, classify, is_polygon) -> bool:
    return (feat.kind == "way" and len(feat.geometry) >= 2 and not is_polygon(feat)
            and classify(feat.tags or {}) == "railway")


class RailLand:
    """The real lines (from the map's OSM cache) and the railway's areas, in BNG metres."""

    def __init__(self, real_feats, area, classify, is_polygon):
        lines = [LineString([TO_BNG(lo, la) for la, lo in f.geometry])
                 for f in real_feats if is_rail_line(f, classify, is_polygon)
                 and (f.tags or {}).get("tunnel", "no") == "no"]
        self.lines = shapely.union_all(lines) if lines else None
        self.stations, self.land = [], []
        path = os.path.join(area, "rail_land.geojson")
        if os.path.exists(path):
            for ft in json.load(open(path, encoding="utf-8"))["features"]:
                poly = transform(TO_BNG, shape(ft["geometry"])).buffer(0)
                p = ft["properties"]
                self.land.append(poly)
                if p.get("railway") in STATION_AREAS or (p.get("public_transport") in STATION_AREAS
                                                         and p.get("train") == "yes"):
                    self.stations.append(poly)
        self.land = shapely.union_all(self.land) if self.land else None
        self.stations = shapely.union_all(self.stations) if self.stations else None

    def __bool__(self):
        return self.lines is not None

    def kind(self, tags: dict, bng: Polygon) -> str | None:
        """'station' (the railway's own: kept, no clearance), 'footbridge' (left
        out), or None."""
        if self.lines is None:
            return None
        b = (tags.get("building") or "").lower()
        layer = str(tags.get("layer", "0"))
        up = layer.isdigit() and int(layer) >= 1
        # The station first: a station building a storey up over its line is
        # still the station.
        if (b in STATION_BUILDINGS or tags.get("railway") in RAILWAY_BUILDINGS
                or (tags.get("public_transport") == "station" and tags.get("train") == "yes")
                or (self.stations is not None and bng.representative_point().within(self.stations))):
            return "station"
        if (b == "bridge" or up) and bng.distance(self.lines) <= BRIDGE_TOUCH_M:
            return "footbridge"
        return None

    def lidar_on(self, bng: Polygon) -> bool:
        """A LiDAR-only shape on the line or the railway's land."""
        if self.lines is None:
            return False
        return (bng.distance(self.lines) <= LIDAR_NEAR_M
                or (self.land is not None and bng.representative_point().within(self.land)))
