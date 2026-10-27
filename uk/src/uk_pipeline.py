"""The UK chain for one map, in order, from its settings (PLAN U stage 3).

    python uk_pipeline.py <map name> [--source <source map>] [--area <area folder>]
                          [--from STEP] [--to STEP] [--only STEP] [--force] [--list]

Run from anywhere with Knoxmap's environment. The map is output/<map name>
(the road graph's map, made by the hand-over from the source map). Its
settings are output/<map name>/uk.json, written with the defaults on the first
run (--source and --area then needed):

    area            data/<area> (its area.json holds the data level)
    source          the source map (setup_area.py / new_map.py) the hand-over reads
    roads           the road graph's: "graph" | "cost-front" (handover --roads); or KnoxMap's own
                    (Settings.straight_roads, the map rendered again from the source's download,
                    new_map.py --from-map): "knoxmap-as-mapped" (0), "knoxmap-straight" (1,
                    KnoxMap's default), "knoxmap-square" (2: square where the buildings are)
    buildings       "lidar": blocks.py and lidar_outlines.py, the real outlines from the LiDAR;
                    "knoxmap": KnoxMap's own fitting at Build, and its uses from OSM's points
                    (blocks, lidar_outlines, split_units and shops left out)
    building_alignment  KnoxMap's fitting of outlines at Build: "real" | "smart" | "rectilinear" |
                    "rectangle" (the map's settings.json); null: as the settings have it
    lanes           driveways out, OSM-only lanes routed round buildings (handover --lanes)
    front_anchor    buildings squared about their street side (front_anchor.on);
                    null: as the hand-over left it (cost-front writes it)
    rows_any_angle  rows joined at any angle (rows_any_angle.on)
    rooms           the room solver (rooms_uk.on / rooms_uk.off)
    dormers         roof_plan --dormers
    inspire_attrs   plots2 --inspire-attrs <area>/inspire_town.geojson (left out at osm_lidar)
    streets         blocks.py --streets (null: its default)
    uk_assets       the UK tile pack (KnoxMapUK_Tiles) on the map; false: none of it, for players
                    without the mod (knoxbuild/assets_uk.py: uk_assets.off in the map's folder;
                    vanilla stand-ins, streets/vanilla_assets.py on the loose tiles, the ground
                    compiled by vanilla rules, no listed builds; uk_check fails if one is left)
    world_cell      [x, y]: the map's world cell (knoxbuild.placement); null: packed automatically
    compile         compile after validating
    backend         "rust" | "worlded" (KNOXMAP_BACKEND for the compile; null: compile_map's default)

Steps (Building_types/README.md): roads (the hand-over, or the render), blocks, lidar_outlines, split_units,
shops, rails, plots2, trees, sports, rails_apply, kerbs, regional, roof_plan,
listed, build, validate, compile. A step is skipped when its stamp - its
command, its code (the chain's packages; Build's knoxbuild, the listed maker's
folder), its options and the area's data
files - is what it was when it last finished. Each step works on the map in
place, so a step starting takes away the stamps of all after it: they run, in
this run or the next (a run stopped, or ended with --to). The hand-over keeps
its own fingerprint too, and counts as run only when it routed again (else it
reused its routing and the steps after it may still skip). Stamps in
uk_pipeline.json; everything a step prints in uk_pipeline.log. Each step starts with a line "STEP k/N name" (the
app's progress). Stopped (Ctrl+C, or the process ended), the step under way
keeps no stamp and runs again next time.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
KNOX = ukpaths.KNOX
MAKER = ukpaths.assets("listed_building_maker") or ""
REGISTER = ukpaths.assets("Tile_creator", "src", "register.py") or ""
TOOLS = os.path.join(KNOX, "vendor", "PZMappingTools")
PALETTE = os.path.join(BT, "palettes", "english.json")

GRAPH_ROADS = ("graph", "cost-front")
KNOXMAP_ROADS = {"knoxmap-as-mapped": 0, "knoxmap-straight": 1, "knoxmap-square": 2}
ALIGNMENTS = ("real", "smart", "rectilinear", "rectangle")
DEFAULTS = {"area": None, "source": None, "roads": "graph", "buildings": "lidar", "building_alignment": None,
            "uk_assets": True,
            "lanes": False, "front_anchor": None,
            "rows_any_angle": False, "rooms": True, "dormers": False, "inspire_attrs": True, "streets": None,
            "world_cell": None, "compile": False, "backend": None}


def steps(m: str, a: str, s: dict) -> list[tuple[str, list, list, tuple]]:
    """(name, command, its code - files or folders, the uk.json keys it uses)."""
    py = lambda *p: os.path.join(HERE, *p)  # noqa: E731
    streets = ["--streets", *s["streets"]] if s.get("streets") else []
    if s["roads"] in KNOXMAP_ROADS:              # KnoxMap's own roads: the source map again, under this name
        roads = [py("area", "new_map.py"), os.path.basename(m), "--from-map", s["source"] or "",
                 "--straight-roads", str(KNOXMAP_ROADS[s["roads"]])]
    else:
        roads = [py("roadgraph", "handover.py"), os.path.join("output", s["source"] or ""), a, os.path.basename(m),
                 "--roads", s["roads"]] + (["--lanes"] if s["lanes"] else [])
    inspire = os.path.join(a, "inspire_town.geojson")
    # The chain's packages import one another (shops uses ground.trees, plots2 buildings.blocks,
    # kerbs roadgraph...): every step's stamp takes all of them, so an edit anywhere runs the
    # chain again from the first step (the hand-over then reuses its routing by its own fingerprint).
    chain = [py(d) for d in ("buildings", "ground", "streets", "regional", "roadgraph")]
    chain += [py("area", "area_conf.py"), py("area", "roofs.py")]
    return [
        ("roads", roads, chain + [py("area", "new_map.py")], ("source", "roads", "lanes")),
        ("blocks", [py("buildings", "blocks.py"), m, a, *streets], chain,
         ("streets", "front_anchor", "rows_any_angle", "roads")),
        ("lidar_outlines", [py("buildings", "lidar_outlines.py"), m, a], chain, ()),
        ("split_units", [py("buildings", "split_units.py"), m, a], chain, ()),
        ("shops", [py("buildings", "shops.py"), m, a], chain, ()),
        ("rails", [py("streets", "rails.py"), m], chain, ()),
        ("plots2", [py("ground", "plots2.py"), m, a] +
         (["--inspire-attrs", inspire] if s["inspire_attrs"] and os.path.exists(inspire) else []),
         chain, ("inspire_attrs",)),
        ("trees", [py("ground", "trees.py"), m, a], chain, ()),
        ("sports", [py("ground", "sports.py"), m], chain, ()),
        ("rails_apply", [py("streets", "rails.py"), m, "--apply"], chain, ()),
        ("kerbs", [py("streets", "kerbs.py"), m], chain, ("uk_assets",)),
        # UK assets off: the loose tiles' UK pieces swapped or left out
        ("vanilla_assets", [py("streets", "vanilla_assets.py"), m], chain, ("uk_assets",)),
        ("regional", [py("regional", "export_regional.py"), a, PALETTE, m], chain + [PALETTE], ()),
        ("roof_plan", [py("regional", "roof_plan.py"), m, a] + (["--dormers"] if s["dormers"] else []),
         chain, ("dormers",)),
        ("listed", [os.path.join(MAKER, "apply.py"), m, a], [MAKER], ()),
        ("build", ["-m", "knoxbuild", m], [os.path.join(KNOX, "knoxbuild")],
         ("rooms", "world_cell", "buildings", "building_alignment")),
        ("validate", [os.path.join(KNOX, "tools", "validate_tbx.py"), os.path.join(m, "buildings")],
         [os.path.join(KNOX, "tools", "validate_tbx.py")], ()),
        # UK assets off: no UK tile left anywhere in the map; the vanilla rules for its ground
        ("uk_check", [py("streets", "vanilla_assets.py"), m, "--check"],
         [os.path.join(KNOX, "knoxbuild", "assets_uk.py")], ("uk_assets",)),
        ("vanilla_rules", [REGISTER, "--vanilla-to", os.path.join(TOOLS, "config_vanilla")],
         [REGISTER, os.path.join(TOOLS, "config", "Rules.txt"), os.path.join(TOOLS, "config", "Blends.txt")],
         ("uk_assets",)),
        ("compile", [os.path.join(KNOX, "tools", "compile_map.py"), m] +
         (["--backend", s["backend"]] if s.get("backend") else []),
         [os.path.join(KNOX, "tools", "compile_map.py")], ("backend",)),
    ]


def _code_hash(h, paths: list[str]) -> None:
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "**", "*.py"), recursive=True) +
                       glob.glob(os.path.join(p, "**", "*.json"), recursive=True)) if os.path.isdir(p) else [p]
        for f in files:
            if "__pycache__" in f or not os.path.isfile(f):
                continue
            h.update(os.path.relpath(f, BT).encode())
            h.update(open(f, "rb").read())


def area_signature(area: str) -> str:
    """The area's data files by name, size and time (not their bytes: the LiDAR is gigabytes)."""
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(area, "*")) + glob.glob(os.path.join(area, "nlp", "*")) +
                    glob.glob(os.path.join(area, "lidar", "*"))):
        if os.path.isfile(f):
            st = os.stat(f)
            h.update(f"{os.path.relpath(f, area)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def stamp(cmd: list, code: list, keys: tuple, s: dict, area_sig: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(cmd).encode())
    h.update(json.dumps({k: s.get(k) for k in keys}, sort_keys=True).encode())
    h.update(area_sig.encode())
    _code_hash(h, code)
    return h.hexdigest()


def switches(m: str, s: dict) -> None:
    """The files and settings the scripts read today, from uk.json."""
    def flag(name: str, on: bool | None):
        path = os.path.join(m, name)
        if on is None:
            return
        if on:
            open(path, "a").close()
        elif os.path.exists(path):
            os.remove(path)
    flag("front_anchor.on", s.get("front_anchor"))
    flag("rows_any_angle.on", s.get("rows_any_angle"))
    flag("rooms_uk.on", bool(s.get("rooms")))
    flag("rooms_uk.off", not s.get("rooms"))
    flag("uk_assets.off", not s.get("uk_assets", True))
    if s.get("building_alignment") in ALIGNMENTS:
        sp = os.path.join(m, "settings.json")
        if os.path.exists(sp):
            d = json.load(open(sp, encoding="utf-8"))
            if d.get("building_alignment") != s["building_alignment"]:
                d["building_alignment"] = s["building_alignment"]
                json.dump(d, open(sp, "w", encoding="utf-8"), indent=2)
    if s.get("world_cell"):
        sys.path.insert(0, KNOX)
        from knoxbuild import placement
        placement.save(m, tuple(s["world_cell"]))


def run_step(cmd: list, log, env: dict) -> int:
    """The step as a child process, its output to the console and the log; Ctrl+C or the
    runner being ended stops the child too."""
    p = subprocess.Popen([sys.executable, *[str(c) for c in cmd]], cwd=KNOX, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))   # no window from the app
    try:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return p.wait()
    except BaseException:
        p.kill()
        p.wait()
        raise


def main() -> int:
    a = sys.argv
    arg = lambda k: a[a.index(k) + 1] if k in a else None  # noqa: E731
    name = a[1]
    m = os.path.join("output", name)                  # relative to Knoxmap/, as every script takes it
    md = os.path.join(KNOX, m)
    os.makedirs(md, exist_ok=True)
    conf_path = os.path.join(md, "uk.json")
    s = dict(DEFAULTS, **(json.load(open(conf_path, encoding="utf-8")) if os.path.exists(conf_path) else {}))
    if arg("--source"):
        s["source"] = arg("--source")
    if arg("--area"):
        s["area"] = arg("--area")
    json.dump(s, open(conf_path, "w", encoding="utf-8"), indent=1)
    if not s["area"] or not s["source"]:
        print(f"{conf_path}: set `area` and `source` (or pass --area and --source)")
        return 2
    area = os.path.join(BT, "data", s["area"])
    if not os.path.isdir(area):
        print(f"no area folder {area} (setup_area.py makes one)")
        return 2
    if s["roads"] not in GRAPH_ROADS and s["roads"] not in KNOXMAP_ROADS:
        print(f"uk.json roads {s['roads']!r}: one of {GRAPH_ROADS + tuple(KNOXMAP_ROADS)}")
        return 2
    skip = {"compile"} if not s.get("compile") else set()
    # UK assets on: nothing to swap or check; off: no listed builds (they are made of UK kits)
    skip |= {"vanilla_assets", "uk_check", "vanilla_rules"} if s.get("uk_assets", True) else {"listed"}
    if s.get("buildings") == "knoxmap":            # KnoxMap fits the outlines itself at Build
        # ...and finds their uses from OSM's points, as it always has: units and uses in real
        # coordinates (split_units, shops) carry through blocks.py's LiDAR frames
        skip |= {"blocks", "lidar_outlines", "split_units", "shops"}
    plan = [st for st in steps(m, area, s) if st[0] not in skip]
    names = [st[0] for st in plan]
    if "--list" in a:
        print(" ".join(names))
        return 0
    for k in ("--from", "--to", "--only"):
        if arg(k) and arg(k) not in names:
            print(f"{k} {arg(k)}: not one of {names}")
            return 2
    first = names.index(arg("--only") or arg("--from") or names[0])
    last = names.index(arg("--only") or arg("--to") or names[-1])

    stamps_path = os.path.join(md, "uk_pipeline.json")
    stamps = json.load(open(stamps_path, encoding="utf-8")) if os.path.exists(stamps_path) else {}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if s.get("backend"):
        env["KNOXMAP_BACKEND"] = s["backend"]
    area_sig = area_signature(area)
    dirty = "--force" in a
    t0 = time.time()
    with open(os.path.join(md, "uk_pipeline.log"), "a", encoding="utf-8") as log:
        log.write(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(a[1:])}\n")
        for k, (step, cmd, code, keys) in enumerate(plan):
            if k < first or k > last:
                continue
            st = stamp(cmd, code, keys, s, area_sig)
            # --from / --only: that step runs whatever its stamp
            if not dirty and stamps.get(step) == st and not (k == first and (arg("--from") or arg("--only"))):
                print(f"STEP {k + 1}/{len(plan)} {step}: unchanged, skipped", flush=True)
                continue
            line = f"STEP {k + 1}/{len(plan)} {step} ({time.strftime('%H:%M:%S')})"
            print(line, flush=True)
            log.write(line + "\n")
            graph = os.path.join(md, f"{name}_roadgraph.json")
            before = os.path.getmtime(graph) if os.path.exists(graph) else None

            def stale_after():
                """Every step after this one is stale, in this run and any later one (a run
                stopped here, or one that ended with --to)."""
                for later in names[k + 1:]:
                    stamps.pop(later, None)
                json.dump(stamps, open(stamps_path, "w", encoding="utf-8"), indent=1)
            if step != "roads":        # the hand-over: only if it routes again (below)
                stale_after()
            switches(md, s)            # before each step: a cost-front hand-over writes front_anchor.on itself
            code_ = run_step(cmd, log, env)
            # the hand-over's road graph file changed (it routed again); a KnoxMap render always counts
            rerouted = step == "roads" and (s["roads"] in KNOXMAP_ROADS or before is None
                                            or os.path.getmtime(graph) != before)
            if rerouted:
                stale_after()
            if code_ != 0:
                print(f"STEP {step} failed (exit {code_}); see {os.path.join(md, 'uk_pipeline.log')}", flush=True)
                return code_
            stamps[step] = st
            json.dump(stamps, open(stamps_path, "w", encoding="utf-8"), indent=1)
            # the hand-over reused its routing: nothing after it need run for its sake
            if step == "roads" and not rerouted:
                continue
            dirty = True
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("stopped", flush=True)
        sys.exit(130)
