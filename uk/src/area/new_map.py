"""A new KnoxMap map for an area, from a script (what the app's Generate does).

    python new_map.py <name> S W N E [--settings <settings.json>] [--rotate DEG] [--widths <area data dir>]
                      [--shape <outline.geojson>] [--download-only] [--straight-roads 0|1|2]
    python new_map.py <name> --from-map <source map> [--straight-roads 0|1|2]

Run from Knoxmap/ with its environment. Into Knoxmap/output/<name>: the OSM
download for the box at any turn (renderer.cover_bbox, fetched in Overpass
tiles, cached as <name>_turned_osm.json.gz), then - with --widths - the
streets given their real widths (streets/street_widths.py, before the render),
then the map rendered at 1 m a tile, turned by --rotate or, by default, by the
streets' main direction (renderer.dominant_road_angle, as `align_streets`).
Settings from --settings (default: the Odiham source map's, else KnoxMap's own), saved beside it.
With --shape, the map is cut to a drawn outline as the app's polygon does
(rendering/shape.py: outside goes to grass; water and main roads stay).
The pipeline's step 2 (Building_types/README.md); then blocks.py on.

--straight-roads: KnoxMap's own roads (Settings.straight_roads: 0 as mapped, 1
straight runs and 45-degree diagonals, 2 square where the buildings are) in place
of the settings file's. --from-map: the same map again under a new name (the
source map's box, turn, settings and OSM download, copied, nothing fetched) -
uk_pipeline.py's roads step when the roads are KnoxMap's rather than the road
graph's.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
KNOX = ukpaths.KNOX
sys.path.insert(0, KNOX)
os.chdir(KNOX)

from generator import osm, renderer  # noqa: E402
from knoxbuild.settings import Settings  # noqa: E402

OVERPASS_TILE_KM2 = 30.0                       # as app.py


def from_map(name: str, src: str) -> tuple[list[float], float, str]:
    """The source map's box and turn; its OSM download (and the one before the street
    widths) copied in under this map's name. Returns (box, rotation, its settings file)."""
    sdir = os.path.join("output", src)
    info = json.load(open(os.path.join(sdir, f"{src}_info.json"), encoding="utf-8"))
    b = info["bbox"]
    out = os.path.join("output", name)
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(sdir, info["osm_cache"])
    for suffix in ("", ".orig"):
        if os.path.exists(cache + suffix):
            shutil.copy(cache + suffix, osm.cache_path(out, f"{name}_turned") + suffix)
    box = [b["south"], b["west"], b["north"], b["east"]]
    return box, float(info.get("rotation") or 0.0), os.path.join(sdir, "settings.json")


def main() -> int:
    name = sys.argv[1]
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None  # noqa: E731
    if arg("--from-map"):
        (south, west, north, east), turned, src_settings = from_map(name, arg("--from-map"))
        if "--rotate" not in sys.argv:
            sys.argv += ["--rotate", str(turned)]
        if "--settings" not in sys.argv:
            sys.argv += ["--settings", src_settings]
    else:
        south, west, north, east = (float(v) for v in sys.argv[2:6])
    settings_path = arg("--settings") or os.path.join(KNOX, "output", "Odiham_town", "settings.json")
    out = os.path.join("output", name)
    os.makedirs(out, exist_ok=True)
    # KnoxMap's own defaults where there is no Odiham map to copy (a fresh checkout of the fork)
    raw = json.load(open(settings_path, encoding="utf-8")) if os.path.exists(settings_path) else Settings().to_dict()
    raw["align_streets"], raw["rotate_degrees"] = 0, 0.0
    if arg("--straight-roads") is not None:
        raw["straight_roads"] = int(arg("--straight-roads"))
    json.dump(raw, open(os.path.join(out, "settings.json"), "w", encoding="utf-8"), indent=2)
    settings = Settings.from_dict(raw)
    mpt = 1.0
    fetch_box = renderer.cover_bbox(south, west, north, east, mpt)
    cache = osm.cache_path(out, f"{name}_turned")
    features = osm.load_cache(cache, fetch_box)
    if features is None:
        print("downloading OSM for", fetch_box)
        features = osm.fetch_features_tiled(*fetch_box, max_tile_km2=OVERPASS_TILE_KM2)
        osm.save_cache(cache, fetch_box, features)
    if "--download-only" in sys.argv:          # setup_area.py: features.py reads the download before the render
        print(f"{name}: OSM download only, {len(features)} features")
        return 0
    if arg("--widths"):
        subprocess.run([sys.executable, os.path.join(HERE, "..", "streets", "street_widths.py"), cache, arg("--widths")],
                       check=True)
        features = osm.load_cache(cache, fetch_box)
    if arg("--rotate") is not None:
        rotation = float(arg("--rotate"))
    else:
        angle, strength = renderer.dominant_road_angle(features, south, west, north, east)
        rotation = -angle if strength >= renderer.ALIGN_MIN_STRENGTH and abs(angle) >= 0.5 else 0.0
    shape = None
    if arg("--shape"):                         # a drawn outline (GeoJSON Polygon, lon/lat), as the app's
        shape = json.load(open(arg("--shape"), encoding="utf-8"))
        shape = shape.get("geometry", shape)
    print(f"rendering {name}: {len(features)} features, rotation {rotation:.1f}")
    result = renderer.render(features, south, west, north, east, meters_per_tile=mpt, output_dir=out, map_name=name,
                             spawn_density=settings.spawn_density, tree_density=settings.tree_density,
                             rotation=rotation, osm_cache=os.path.basename(cache), osm_bbox=fetch_box,
                             shape=shape, straight_roads=int(settings.straight_roads))   # 2 as Generate: square by buildings
    raw["rotate_degrees"] = rotation
    json.dump(raw, open(os.path.join(out, "settings.json"), "w", encoding="utf-8"), indent=2)
    print(f"{name}: {result.width} x {result.height} tiles, {result.cells_x} x {result.cells_y} cells")
    for stale in (f"{name}_buildings.orig.geojson", f"{name}_fences.orig.geojson"):
        if os.path.exists(os.path.join(out, stale)):
            os.remove(os.path.join(out, stale))
    return 0


if __name__ == "__main__":
    sys.exit(main())
