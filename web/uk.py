"""KnoxMap UK: the app's UK mode (PLAN U stage 4). The page's "UK mode" button shows
its panel (and centres the map on London); `KnoxMap.bat --uk` (KNOXMAP_UK=1) starts
with it on.

The UK chain is ../Building_types: an area's data set up in one command
(src/area/setup_area.py) and a map's chain run from its settings
(src/uk_pipeline.py, output/<map>/uk.json). Both run here as child processes,
one job a map at a time; the page polls /api/uk/job for its step and the last
lines it printed, and Stop ends the job's whole process tree (a step is a child
of the runner: ending the runner alone leaves it running on Windows). A stopped
step runs again next time (the runner's stamps). After a run the map opens in
the usual cards: Build, Compile, Install.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

from flask import Blueprint, jsonify, request

from web.common import BASE_DIR, OUTPUT_DIR, _json_body, failed, log

bp = Blueprint("uk", __name__)

import ukdirs  # noqa: E402 - where the UK chain is, in either layout
BT = Path(ukdirs.UK)
SETUP = BT / "src" / "area" / "setup_area.py"
PIPELINE = BT / "src" / "uk_pipeline.py"
NAME = re.compile(r"^(?!(?:CON|PRN|AUX|NUL|COM\d|LPT\d)$)[A-Za-z0-9_\-]{1,64}$", re.I)   # no Windows device names
LEVELS = ("full", "osm_lidar")
ROADS = ("graph", "cost-front", "knoxmap-straight", "knoxmap-square", "knoxmap-as-mapped")
BUILDINGS = ("lidar", "knoxmap")
ALIGNMENTS = ("real", "smart", "rectilinear", "rectangle")
BACKENDS = (None, "rust", "worlded")
TAIL = 60                      # lines of a job's output the page is shown
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

RAW = BT / "data" / "raw"
INSPIRE_DIR = Path(os.environ.get("KNOXMAP_INSPIRE_DIR") or (RAW / "inspire"))   # as setup_area.py

# The extra data (area/area_conf.py's full level). National files are downloaded by
# hand into data/raw (the first three need a free account or a licence click);
# the rest setup_area.py fetches for each area itself.
DATASETS = [
    {"key": "os_roads", "name": "OS Open Roads (GeoPackage, GB)", "files": ["oproad/Data/oproad_gb.gpkg",
                                                                            "oproad_gpkg_gb*.zip"],
     "url": "https://osdatahub.os.uk/downloads/open/OpenRoads", "save": "data/raw (the zip as it comes)",
     "gives": "road classes, names and numbers checked against OSM's"},
    {"key": "uprn", "name": "OS Open UPRN (CSV, GB)", "files": ["osopenuprn_*.zip"],
     "url": "https://osdatahub.os.uk/downloads/open/OpenUPRN", "save": "data/raw (the zip as it comes)",
     "gives": "every address: homes per building, units, garden plots"},
    {"key": "epc_domestic", "name": "EPC domestic certificates (all of England and Wales)",
     "files": ["domestic-csv.zip"], "url": "https://epc.opendatacommunities.org/downloads/domestic",
     "save": "data/raw/domestic-csv.zip (free account)", "gives": "house ages, types, built form, rooms"},
    {"key": "epc_nondomestic", "name": "EPC non-domestic certificates", "files": ["non-domestic-csv.zip"],
     "url": "https://epc.opendatacommunities.org/downloads/non-domestic",
     "save": "data/raw/non-domestic-csv.zip (free account)", "gives": "shops, offices, pubs where OSM has none"},
    {"key": "inspire", "name": "HM Land Registry INSPIRE Index Polygons (a zip per council)", "dir": "inspire",
     "url": "https://use-land-property-data.service.gov.uk/datasets/inspire/download",
     "save": "the council's zip into the INSPIRE folder", "gives": "land titles: garden plots, garages, drives"},
]
# The UK tile pack: a mod in the game's mods folder, from the fork's releases
# (uk/src/area/install_pack.py downloads, checks and installs it).
PACK = "KnoxMapUK_Tiles"
INSTALL_PACK = BT / "src" / "area" / "install_pack.py"
RELEASES_PAGE = "https://github.com/hamurabi11/knoxmap_UK/releases"
PACK_JOB = "_tile_pack"          # the job's key (no map has a name starting _)


def _pack_dir() -> Path:
    import knoxpaths
    return Path(knoxpaths.zomboid_user_dir()) / "mods" / PACK


AUTOMATIC = [
    ("OpenStreetMap (Overpass)", "the map, rail land, road points"),
    ("Environment Agency LiDAR DSM / DTM", "heights, storeys, roof shapes"),
    ("Environment Agency National LiDAR point cloud", "buildings, trees, boundaries (about 230 MB a 5 km tile)"),
    ("Historic England listed buildings (NHLE)", "listed buildings and landmarks (full data)"),
    ("Food Standards Agency ratings", "pubs, cafes, restaurants, food shops (full data)"),
    ("postcodes.io", "the area's council (full data)"),
]

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def enabled() -> bool:
    """UK mode on from the start (KnoxMap.bat --uk); the page's button turns it on and off."""
    return os.environ.get("KNOXMAP_UK", "").strip().lower() in ("1", "on", "yes", "true")


def _python() -> str:
    """The console Python beside the app's (pythonw has no output to pipe)."""
    exe = sys.executable
    alt = os.path.join(os.path.dirname(exe), "python.exe")
    return alt if os.path.basename(exe).lower() == "pythonw.exe" and os.path.exists(alt) else exe


def _start(key: str, kind: str, args: list[str]) -> dict:
    """A child process for map `key`, read on a thread: its STEP / == lines are the
    progress, the rest the tail."""
    with _LOCK:
        if _JOBS.get(key, {}).get("state") in ("running", "stopping"):
            raise RuntimeError(f"{key} is already running")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        p = subprocess.Popen([_python(), *args], cwd=str(BASE_DIR), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                             creationflags=NO_WINDOW, start_new_session=os.name != "nt")
        job = {"kind": kind, "state": "running", "step": None, "k": 0, "n": 0, "started": time.time(),
               "lines": deque(maxlen=TAIL), "proc": p, "exit": None}
        _JOBS[key] = job
    log.info("uk %s %s: %s", kind, key, " ".join(args))

    def read():
        for line in p.stdout:
            line = line.rstrip("\n")
            m = re.match(r"STEP (\d+)/(\d+) (\w+)", line) or re.match(r"== (\w+)", line)
            with _LOCK:
                job["lines"].append(line)
                if m and m.lastindex == 3:
                    job.update(k=int(m.group(1)), n=int(m.group(2)), step=m.group(3))
                elif m:
                    job.update(step=m.group(1), k=job["k"] + 1)
        code = p.wait()
        with _LOCK:
            job["exit"] = code
            job["state"] = "stopped" if job["state"] == "stopping" else ("done" if code == 0 else "failed")
        log.info("uk %s %s: %s (exit %s)", kind, key, job["state"], code)
    threading.Thread(target=read, daemon=True).start()
    return job


def _view(job: dict | None) -> dict:
    if not job:
        return {"state": "idle"}
    return {k: (list(v) if k == "lines" else v) for k, v in job.items() if k != "proc"}


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, creationflags=NO_WINDOW)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            os.kill(pid, signal.SIGTERM)


def _areas() -> list[dict]:
    out = []
    for f in sorted((BT / "data").glob("*/area.json")):
        try:
            a = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"area": f.parent.name, "name": a.get("name"), "data": a.get("data", "full"),
                    "council": a.get("council_name") or a.get("council"), "map": a.get("map"),
                    "box": a.get("box"), "found": a.get("found", {})})
    return out


def _uk_maps() -> list[dict]:
    out = []
    for f in sorted(OUTPUT_DIR.glob("*/uk.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"map": f.parent.name, "settings": s})
    return out


def _clean_settings(s: dict) -> dict:
    """uk.json's keys, each checked (uk_pipeline.py's docstring has their meaning)."""
    out = {}
    for k in ("uk_assets", "lanes", "rows_any_angle", "rooms", "dormers", "inspire_attrs", "compile"):
        if k in s:
            out[k] = bool(s[k])
    if "front_anchor" in s:
        out["front_anchor"] = None if s["front_anchor"] in (None, "", "auto") else bool(s["front_anchor"])
    if s.get("roads") in ROADS:
        out["roads"] = s["roads"]
    if s.get("buildings") in BUILDINGS:
        out["buildings"] = s["buildings"]
    if "building_alignment" in s:
        out["building_alignment"] = s["building_alignment"] if s["building_alignment"] in ALIGNMENTS else None
    if "backend" in s:
        out["backend"] = s["backend"] if s["backend"] in BACKENDS else None
    if "streets" in s:
        st = [x.strip() for x in (s["streets"] if isinstance(s["streets"], list) else str(s["streets"]).split(","))]
        out["streets"] = [x for x in st if x] or None
    if "world_cell" in s:
        wc = s["world_cell"]
        out["world_cell"] = [int(wc[0]), int(wc[1])] if isinstance(wc, (list, tuple)) and len(wc) == 2 \
            and all(str(v).strip().lstrip("-").isdigit() for v in wc) else None
    for k in ("area", "source"):
        if k in s and (s[k] is None or NAME.match(str(s[k]))):
            out[k] = s[k]
    return out


def _data_check() -> dict:
    """Which of the extra datasets are here, with where to get the rest."""
    import glob
    rows = []
    for d in DATASETS:
        if d.get("dir") == "inspire":
            zips = sorted(INSPIRE_DIR.glob("*.zip")) if INSPIRE_DIR.is_dir() else []
            have = [str(INSPIRE_DIR)] if zips else []
            note = f"{len(zips)} councils in {INSPIRE_DIR}" if zips else f"none in {INSPIRE_DIR}"
        else:
            have = [f for pat in d["files"] for f in glob.glob(str(RAW / pat))]
            note = ", ".join(os.path.relpath(f, BT / "data") for f in have[:2]) if have else ""
        rows.append({k: d[k] for k in ("key", "name", "url", "save", "gives")} | {"have": bool(have), "note": note})
    pack = _pack_dir() / "common" / "media" / "texturepacks" / f"{PACK}.pack"
    rows.insert(0, {"key": "tile_pack", "name": "UK tile pack (the KnoxMapUK_Tiles mod)", "url": RELEASES_PAGE,
                    "save": "the game's mods folder (Download and install does it)",
                    "gives": "the UK walls, windows, doors, shop fronts, kerbs, markings, signs and street "
                             "furniture: needed in the game for maps made with UK assets on",
                    "have": pack.exists(), "note": str(pack.parent.parent.parent.parent) if pack.exists() else "",
                    "install": True})
    councils = sorted(p.parent.name for p in (BT / "data").glob("*/epc_domestic.csv"))
    return {"datasets": rows, "automatic": [{"name": n, "gives": g} for n, g in AUTOMATIC],
            "raw": str(RAW), "inspireDir": str(INSPIRE_DIR), "epcCouncils": councils}


@bp.route("/api/uk/data")
def api_data():
    return jsonify(_data_check())


@bp.route("/api/uk/open", methods=["POST"])
def api_open():
    """A dataset's download page in the browser (only the listed ones), or the data folder."""
    b = _json_body() or {}
    if b.get("folder") in ("raw", "inspire"):
        folder = RAW if b["folder"] == "raw" else INSPIRE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(folder))  # noqa: S606 - our own data folder
        else:
            subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", str(folder)])
        return jsonify({"opened": str(folder)})
    url = RELEASES_PAGE if b.get("key") == "tile_pack" else \
        next((d["url"] for d in DATASETS if d["key"] == b.get("key")), None)
    if not url:
        return failed("Which dataset?", 400)
    webbrowser.open(url)
    return jsonify({"opened": url})


@bp.route("/api/uk/pack/install", methods=["POST"])
def api_pack_install():
    """Download the UK tile pack from the fork's latest release, check it, install it into
    the game's mods folder and register its ground tiles (install_pack.py), as a job."""
    try:
        _start(PACK_JOB, "pack", [str(INSTALL_PACK)])
    except RuntimeError as e:
        return failed(str(e), 409)
    return jsonify({"started": True, "job": PACK_JOB})


@bp.route("/api/uk/state")
def api_state():
    name = request.args.get("map", "")
    settings = None
    if NAME.match(name or ""):
        path = OUTPUT_DIR / name / "uk.json"
        if path.exists():
            settings = json.loads(path.read_text(encoding="utf-8"))
    with _LOCK:
        job = _view(_JOBS.get(name)) if name else {"state": "idle"}
    return jsonify({"areas": _areas(), "maps": _uk_maps(), "settings": settings, "job": job,
                    "levels": LEVELS, "roads": ROADS, "startOn": enabled()})


@bp.route("/api/uk/job")
def api_job():
    with _LOCK:
        return jsonify(_view(_JOBS.get(request.args.get("map", ""))))


@bp.route("/api/uk/setup", methods=["POST"])
def api_setup():
    """setup_area.py for the drawn box: a source map and the area's data."""
    b = _json_body() or {}
    name, area, level = str(b.get("name", "")).strip(), str(b.get("area") or "").strip(), b.get("data", "full")
    if not NAME.match(name) or (area and not NAME.match(area)) or level not in LEVELS:
        return failed("A map name (letters, digits, _ and -), an area folder like it, and a data level", 400)
    try:
        box = [float(b[k]) for k in ("south", "west", "north", "east")]
    except (KeyError, TypeError, ValueError):
        return failed("Draw the area first", 400)
    args = [str(SETUP), name, *map(str, box), "--data", level, "--area", area or name.lower()]
    if isinstance(b.get("settings"), dict):
        # KnoxMap's own settings from the page (the preset and its knobs), for the source map
        from knoxbuild.settings import Settings
        d = OUTPUT_DIR / name
        d.mkdir(exist_ok=True)
        path = d / "uk_source_settings.json"
        path.write_text(json.dumps(Settings.from_dict(b["settings"]).to_dict(), indent=1), encoding="utf-8")
        args += ["--settings", str(path)]
    if b.get("rotate") not in (None, ""):
        args += ["--rotate", str(float(b["rotate"]))]
    try:
        _start(name, "setup", args)
    except RuntimeError as e:
        return failed(str(e), 409)
    return jsonify({"started": True, "map": name, "area": area or name.lower()})


@bp.route("/api/uk/settings", methods=["POST"])
def api_settings():
    """Write a map's uk.json (made with the defaults by the first run if not there)."""
    b = _json_body() or {}
    name = str(b.get("map", "")).strip()
    if not NAME.match(name):
        return failed("Which map?", 400)
    d = OUTPUT_DIR / name
    d.mkdir(exist_ok=True)
    path = d / "uk.json"
    s = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    s.update(_clean_settings(b.get("settings") or {}))
    path.write_text(json.dumps(s, indent=1), encoding="utf-8")
    return jsonify({"map": name, "settings": s})


@bp.route("/api/uk/run", methods=["POST"])
def api_run():
    """uk_pipeline.py for a map: its uk.json's chain, the steps that changed."""
    b = _json_body() or {}
    name = str(b.get("map", "")).strip()
    if not NAME.match(name):
        return failed("Which map?", 400)
    if not (OUTPUT_DIR / name / "uk.json").exists() and not (b.get("source") and b.get("area")):
        return failed("Set up the area and choose its source map first", 400)
    args = [str(PIPELINE), name]
    for k in ("source", "area"):
        if b.get(k):
            if not NAME.match(str(b[k])):
                return failed(f"Bad {k}", 400)
            args += [f"--{k}", str(b[k])]
    if b.get("from"):
        if not NAME.match(str(b["from"])):
            return failed("Bad step", 400)
        args += ["--from", str(b["from"])]
    if b.get("force"):
        args.append("--force")
    try:
        _start(name, "pipeline", args)
    except RuntimeError as e:
        return failed(str(e), 409)
    return jsonify({"started": True, "map": name})


@bp.route("/api/uk/stop", methods=["POST"])
def api_stop():
    name = str((_json_body() or {}).get("map", "")).strip()
    with _LOCK:
        job = _JOBS.get(name)
        if not job or job["state"] != "running":
            return jsonify({"state": job["state"] if job else "idle"})
        job["state"] = "stopping"
        pid = job["proc"].pid
    _kill_tree(pid)
    log.info("uk stop %s", name)
    return jsonify({"state": "stopping"})
