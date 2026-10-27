"""A new area in one command (PLAN U stage 2): its source map and all its data.

    python setup_area.py <map name> S W N E [--area <folder>] [--data full|osm_lidar]
                         [--town <name>] [--council-code E07000085] [--rotate DEG] [--settings <file>] [--force]

Run from Knoxmap/ with its environment. Writes data/<area>/ (by default the map
name in lower case, `Alton_town` -> `alton_town`) and output/<map name>/, in order:

1. the council (postcodes.io, from the box's middle: its name and code), unless
   --council-code is given or the level is osm_lidar; a box over two councils
   is reported, and the middle's is used;
2. the OSM download (new_map.py --download-only);
3. full: NHLE (fetch_nhle.py) and UPRN (filter_uprn.py, from data/raw's OS
   Open UPRN zip) for the download's box - features.py reads both;
4. features.py, then the map rendered with the streets' real widths
   (new_map.py --widths), turned by --rotate or the streets' main direction,
   with KnoxMap's settings from --settings (the app's panel; default Odiham's);
5. LiDAR heights and roofs (roofs.py --fetch);
6. the point cloud: the box's EA survey tiles (the catalogue's search), the
   newest National LiDAR Programme point cloud and intensity downloaded into
   data/raw/nlp<year>/ (and nlp<year>i/), cut to nlp/core.npz (nlp_crop.py);
7. rail land and road nodes (Overpass);
8. full: FSA round the middle, EPC for the council (extract_epc.py from
   data/raw's national zips, once per council into data/<council>), OS Open
   Roads (osroads.py, data/raw/oproad), INSPIRE (the council's zip from
   INSPIRE_DIR, cut to inspire.geojson and inspire_town.geojson);
9. assign_types.py;
10. data/<area>/area.json: name, town, council, data level, box and what was
   found ({source: true / false / "failed: ..."}).

A step whose output is there is skipped (--force: all again). A step of the
extra data that fails is noted in area.json and the rest go on (each later
step goes without it); a failure of the core (OSM, render, LiDAR) stops it.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
DATA = os.path.normpath(os.path.join(SRC, "..", "data"))
RAW = os.path.join(DATA, "raw")
# HM Land Registry's zips, one a council (KNOXMAP_INSPIRE_DIR, else data/raw/inspire)
INSPIRE_DIR = os.environ.get("KNOXMAP_INSPIRE_DIR") or os.path.join(RAW, "inspire")
POSTCODES = "https://api.postcodes.io/postcodes"
EA_SEARCH = "https://environment.data.gov.uk/backend/catalog/api/tiles/collections/survey/search"
NLP_PRODUCTS = ("national_lidar_programme_point_cloud", "national_lidar_programme_intensity")
BOX_MARGIN_M = 100        # round the map's corners out to this, for the point cloud and INSPIRE
UA = {"User-Agent": "KnoxMap-UK setup_area"}

sys.path.insert(0, SRC)
sys.path.insert(0, os.getcwd())          # Knoxmap/: generator


# A step's console window stays hidden when the app runs this (pythonw, no console of its own).
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(*args) -> None:
    """A step as a child process, its output passed on line by line (a pipe, so it reaches
    whoever reads this one's: a terminal or the app)."""
    print("  $", " ".join(str(a) for a in args[1:]), flush=True)
    p = subprocess.Popen([sys.executable, *[str(a) for a in args]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
                         env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, args[0])


def need(path: str, what: str) -> str:
    """path, or the step fails: `what` is not there."""
    if not path or not os.path.exists(path):
        raise RuntimeError(f"no {what}")
    return path


def _json(url: str, data: dict | None = None) -> dict:
    body = json.dumps(data).encode() if data is not None else None
    hdr = dict(UA, **({"Content-Type": "application/json"} if body else {}))
    with urllib.request.urlopen(urllib.request.Request(url, body, hdr), timeout=60) as r:
        return json.load(r)


def council_at(points: list[tuple[float, float]]) -> list[tuple[str, str] | None]:
    """(name, code) of the council (admin district) at each (lat, lon), from its nearest postcode."""
    got = _json(POSTCODES, {"geolocations": [{"latitude": la, "longitude": lo, "radius": 2000, "limit": 1}
                                             for la, lo in points]})
    out = []
    for r in got["result"]:
        res = (r.get("result") or [None])[0]
        out.append((res["admin_district"], res["codes"]["admin_district"]) if res else None)
    return out


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def map_box_bng(map_dir: str) -> tuple[int, int, int, int]:
    """The map's corners in BNG metres, rounded out to BOX_MARGIN_M."""
    from roadgraph import network
    from roadgraph.widths import to_bng
    _info, proj = network.projector(map_dir)
    es, ns = zip(*to_bng(proj, [(0, 0), (proj.width, 0), (0, proj.height), (proj.width, proj.height)]))
    m = BOX_MARGIN_M
    return (int(math.floor(min(es) / m) * m), int(math.floor(min(ns) / m) * m),
            int(math.ceil(max(es) / m) * m), int(math.ceil(max(ns) / m) * m))


def nlp_tiles(box) -> dict[str, list[dict]]:
    """{product: [the newest year's tiles over the box]} from the EA catalogue."""
    from pyproj import Transformer
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    e0, n0, e1, n1 = box
    ring = [list(to_ll.transform(e, n)) for e, n in ((e0, n0), (e1, n0), (e1, n1), (e0, n1), (e0, n0))]
    req = urllib.request.Request(EA_SEARCH, json.dumps({"type": "Polygon", "coordinates": [ring]}).encode(),
                                 dict(UA, **{"Content-Type": "application/geo+json"}))
    with urllib.request.urlopen(req, timeout=120) as r:
        results = json.load(r)["results"]
    out = {}
    for product in NLP_PRODUCTS:
        mine = [r for r in results if r["product"]["id"] == product]
        if mine:
            year = max(r["year"]["id"] for r in mine)
            out[product] = [r for r in mine if r["year"]["id"] == year]
    return out


EA_FILES = "https://environment.data.gov.uk/"     # the only place a tile is downloaded from


def download(url: str, path: str) -> None:
    if not url.startswith(EA_FILES):
        raise RuntimeError(f"not an Environment Agency download: {url}")
    tmp = path + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, path)


def fetch_nlp(area: str, box) -> int:
    """The point cloud (and intensity) tiles over the box into data/raw, the cloud cut to nlp/core.npz."""
    tiles = nlp_tiles(box)
    if "national_lidar_programme_point_cloud" not in tiles:
        raise RuntimeError("no National LiDAR Programme point cloud over the box")
    lazs = []
    for product, rows in tiles.items():
        for r in rows:
            year, tile = r["year"]["id"], r["tile"]["id"]
            folder = os.path.join(RAW, f"nlp{year}" + ("i" if product.endswith("intensity") else ""))
            os.makedirs(folder, exist_ok=True)
            ext = ".laz" if product.endswith("cloud") else ".tif"
            # any earlier download of the tile (nlp2019/ holds both kinds, from before this script)
            have = glob.glob(os.path.join(RAW, f"nlp{year}*", f"*{tile}*{ext}"))
            if not have:
                z = os.path.join(RAW, f"{product}-{year}-{tile}.zip")
                if not os.path.exists(z):
                    print(f"  downloading {r['label']}", flush=True)
                    download(r["uri"], z)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(folder)
                have = glob.glob(os.path.join(folder, f"*{tile}*{ext}"))
            if product.endswith("cloud"):
                lazs += have
    os.makedirs(os.path.join(area, "nlp"), exist_ok=True)
    run(os.path.join(HERE, "nlp_crop.py"), ",".join(lazs), os.path.join(area, "nlp", "core.npz"), *box)
    return len(lazs)


def inspire_zip(council: str) -> str | None:
    """The council's INSPIRE zip in INSPIRE_DIR (its name in the file's: "East Hampshire" ->
    East_Hampshire_District_Council.zip, "Westminster" -> City_of_Westminster.zip)."""
    s = slug(council)
    hits = sorted((f for f in glob.glob(os.path.join(INSPIRE_DIR, "*.zip")) if s in slug(os.path.basename(f))),
                  key=len)
    return hits[0] if hits else None


def main() -> int:
    a = sys.argv
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    name = a[1]
    south, west, north, east = (float(v) for v in a[2:6])
    force = "--force" in a
    area = os.path.join(DATA, arg("--area", name.lower()))
    level = arg("--data", "full")
    os.makedirs(area, exist_ok=True)
    map_dir = os.path.join("output", name)
    conf_path = os.path.join(area, "area.json")
    conf = json.load(open(conf_path, encoding="utf-8")) if os.path.exists(conf_path) else {}
    conf.update({"name": conf.get("name") or name, "town": arg("--town") or conf.get("town") or
                 name.split("_")[0].lower(), "data": level, "box": [south, west, north, east], "map": name})
    found = conf.setdefault("found", {})

    def save():
        json.dump(conf, open(conf_path, "w", encoding="utf-8"), indent=1)

    def step(label: str, outputs: list[str], fn, core: bool = False) -> None:
        if not force and outputs and all(os.path.exists(o) for o in outputs):
            print(f"== {label}: there", flush=True)
            found.setdefault(label, True)
            return
        print(f"== {label}", flush=True)
        try:
            got = fn()
            found[label] = got if got not in (None, False) else True
        except Exception as e:                  # an extra source: noted, the rest go on
            if core:
                raise
            found[label] = f"failed: {e}"[:300]
            print(f"   {label} failed: {e}", flush=True)
        save()

    lat, lon = (south + north) / 2, (west + east) / 2
    full = level == "full"

    # 1. the council
    if full and not conf.get("council"):
        def council():
            pts = [(lat, lon), (south, west), (south, east), (north, west), (north, east)]
            got = council_at(pts)
            if got[0] is None:
                raise RuntimeError("no postcode near the box's middle")
            names = sorted({g[0] for g in got if g})
            conf["council"], conf["council_name"] = slug(got[0][0]), got[0][0]
            conf["council_code"] = arg("--council-code") or got[0][1]
            if len(names) > 1:
                print(f"   the box is in {names}: using {got[0][0]}'s EPC and titles", flush=True)
                conf["councils_in_box"] = names
            return got[0][0]
        step("council", [], council)
    save()

    # 2. the OSM download
    settings = ["--settings", os.path.abspath(arg("--settings"))] if arg("--settings") else []
    step("osm", glob.glob(os.path.join(map_dir, "*_osm.json.gz")),
         lambda: run(os.path.join(HERE, "new_map.py"), name, south, west, north, east, "--download-only", *settings),
         core=True)
    from generator import renderer
    cover = renderer.cover_bbox(south, west, north, east, 1.0)

    # 3. what features.py reads
    if full:
        step("nhle", [os.path.join(area, "nhle_points.geojson")],
             lambda: run(os.path.join(HERE, "fetch_nhle.py"), *cover, area))
        uprn_zip = (sorted(glob.glob(os.path.join(RAW, "osopenuprn_*.zip"))) or [""])[-1]
        step("uprn", [os.path.join(area, "uprn.csv")],
             lambda: run(os.path.join(HERE, "filter_uprn.py"), need(uprn_zip, "OS Open UPRN zip in data/raw"), *cover,
                         os.path.join(area, "uprn.csv")))

    # 4. features, then the map with the streets' real widths
    step("features", [os.path.join(area, "buildings.geojson")],
         lambda: run(os.path.join(HERE, "features.py"), map_dir, area), core=True)
    render = [os.path.join(HERE, "new_map.py"), name, south, west, north, east, "--widths", area, *settings]
    if arg("--rotate") is not None:
        render += ["--rotate", arg("--rotate")]
    step("map", [os.path.join(map_dir, f"{name}_info.json")], lambda: run(*render), core=True)
    box = map_box_bng(map_dir)
    conf["box_bng"] = list(box)

    # 5. LiDAR heights and roofs
    step("lidar", [os.path.join(area, "roofs.csv")], lambda: run(os.path.join(HERE, "roofs.py"), area, "--fetch"),
         core=True)
    # 6. the point cloud
    step("point_cloud", [os.path.join(area, "nlp", "core.npz")], lambda: fetch_nlp(area, box))
    # 7. Overpass extras
    step("rail_land", [os.path.join(area, "rail_land.geojson")],
         lambda: run(os.path.join(HERE, "fetch_rail_land.py"), area, *cover))
    step("road_nodes", [os.path.join(area, "road_nodes.geojson")],
         lambda: run(os.path.join(HERE, "fetch_road_nodes.py"), area, *cover))

    # 8. the rest of the extra data
    if full:
        miles = round(math.dist((south, west), (north, east)) * 69.0 / 2 + 0.5, 1)
        step("fsa", [os.path.join(area, "fsa.json")], lambda: run(os.path.join(HERE, "fetch_fsa.py"), area, lat, lon, miles))
        council_dir = os.path.join(DATA, conf["council"]) if conf.get("council") else None

        def epc():
            if not council_dir or not conf.get("council_code"):
                raise RuntimeError("no council")
            run(os.path.join(HERE, "extract_epc.py"), need(os.path.join(RAW, "domestic-csv.zip"), "EPC zip in data/raw"),
                need(os.path.join(RAW, "non-domestic-csv.zip"), "EPC zip in data/raw"), conf["council_code"], council_dir)
        step("epc", [os.path.join(council_dir, "epc_domestic.csv")] if council_dir else [], epc)
        gpkg = os.path.join(RAW, "oproad", "Data", "oproad_gb.gpkg")

        def os_roads():
            zips = glob.glob(os.path.join(RAW, "oproad_gpkg_gb*.zip"))
            if not os.path.exists(gpkg) and zips:       # the download as it comes: unzipped once
                with zipfile.ZipFile(zips[0]) as zf:
                    zf.extractall(os.path.join(RAW, "oproad"))
            run(os.path.join(SRC, "roadgraph", "osroads.py"), map_dir, area, need(gpkg, "OS Open Roads in data/raw"))
        step("os_roads", [os.path.join(area, "os_roads.geojson")], os_roads)

        def inspire():
            z = inspire_zip(conf.get("council_name") or "") if conf.get("council_name") else None
            if not z:
                raise RuntimeError(f"no INSPIRE zip for {conf.get('council_name')} in {INSPIRE_DIR}")
            folder = os.path.join(RAW, f"inspire_{conf['council']}")
            gml = glob.glob(os.path.join(folder, "*.gml"))
            if not gml:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(folder)
                gml = glob.glob(os.path.join(folder, "*.gml"))
            run(os.path.join(HERE, "fetch_inspire.py"), gml[0], area, *box)
            shutil.copy(os.path.join(area, "inspire.geojson"), os.path.join(area, "inspire_town.geojson"))
        step("inspire", [os.path.join(area, "inspire.geojson")], inspire)

    # 9. types
    step("types", [os.path.join(area, "types.csv")], lambda: run(os.path.join(HERE, "assign_types.py"), area),
         core=True)
    save()
    print(f"\n{area}/area.json:")
    for k, v in found.items():
        print(f"  {k:12s} {v}")
    print(f"\nnext: the map's chain from the hand-over "
          f"(roadgraph/handover.py {map_dir} {area} {name}_graph), Building_types/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
