"""Minor lanes: driveways out of the road network, OSM-only service lanes kept off the
buildings (LOG 2026-10-25 #7, #8). Behind a switch - `handover.py --lanes` for a whole
map (plan / resolve / drive_paving / passage_check here), this file's main on a square.

    python lanes.py <source map dir> <area data dir> [--preset cost-front] [--out DIR]

On a square of a map (mode_test.py's), against the preset (cost-low-occ1 with
front-anchored squaring, `cost-front`, by default):

1. Driveways (every way of the link service=driveway) leave the road network: not
   routed, not street; each is painted as plot paving (DRIVE_TILES wide) along its real
   line moved with the town (Generate's displacement), from its street end into the plot
   as far as the first building it meets (drive_paving). Lost access: a road (not a driveway) that the network no longer joins to the rest.
2. OSM-only service lanes (no OS Open Roads match, osroads.conflate) are routed with the
   buildings as hard obstacles (snap.HARD: no building square under the carriageway,
   but by the lane's own ends), within the band of the real line. A lane with no such
   path is tried again with the LiDAR's buildings (blocks.Ground.built: building-class
   points, trees dropped) as the obstacles - a ground-level gap the outlines close up:
   it is routed through the gap. With no path there either, the LiDAR along its real
   line: one run of roof with open ground at both ends, no longer than the building it
   crosses is deep there and at most PASSAGE_CAP tiles - a passage (it runs under its
   building, as overlap_floor.py's passages); anything else - the lane is dropped and
   listed (it found no way round). Each passage's length under its building and the
   LiDAR's height along the lane are reported (lanes.md, passage_<way>.png).
3. Scored against the preset alone: building squares on the street, the overlap's classes
   (overlap_floor.classify), links lost, passages made.
4. The slivers left (slivers): their depth into the building, and the street as drawn
   against the real gap between buildings there; and two cheap fixes, in the harness
   only (fixes): the pavement a tile narrower where a building stands on its outer tile,
   and buildings pushed back up to 2 tiles - each scored.
Writes <out>/lanes.md, lanes.json, lanes.png (the preset beside it with lanes),
lanes_after.png and passage_<way>.png (mode_test/lanes/ by default). The square's
driveways are taken out but the streets they met are not joined again here, as
handover.py --lanes does (network._merge_through): the square keeps mode_test's
roadside by link.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import shapely
import shapely.affinity
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from roadgraph import snap  # noqa: E402
# (mode_test and overlap_floor - the harness - imported where it is used: handover.py
# takes plan / resolve / finish / clear_footprints / drive_paving / street_end without it)

DRIVE_TILES = 3             # a driveway's paving, tiles across
PASSAGE_CAP = 12.0          # tiles: a passage no longer under its building than this
PASSAGE_CLEAR = 5           # tiles: the band along a passage's real line the router's occupancy leaves out
ROOF_STEP = 0.5             # tiles between LiDAR samples along a lane's real line
ROOF_GAP = 2                # samples of open ground a run of roof may have and stay one run
DRIVE_COLOUR = (196, 170, 120)


def is_driveway(link, way_tags) -> bool:
    return bool(link.ways) and all((way_tags.get(w) or {}).get("service") == "driveway" for w in link.ways)


def subset(sq, keep: list):
    """The square with only the links `keep` (indices into sq.links): links, their
    roadside, their place in the whole graph, nodes, pins and OS report renumbered."""
    new = copy.copy(sq)
    at = {old: k for k, old in enumerate(keep)}
    new.links = [sq.links[i] for i in keep]
    new.keep = [sq.keep[i] for i in keep]
    new.chosen = {at[i]: v for i, v in sq.chosen.items() if i in at}
    used = {l.u for l in new.links} | {l.v for l in new.links}
    new.nodes = {k: v for k, v in sq.nodes.items() if k in used}
    new.pins = {k: v for k, v in sq.pins.items() if k in used}
    rep = dict(sq.os_report or {})
    for key in ("osm only", "class changed"):
        rep[key] = [dict(e, link=at[e["link"]]) for e in rep.get(key, []) if e["link"] in at]
    new.os_report = rep
    return new


def lidar_on_tiles(ground, proj, H: int, W: int, box=None) -> np.ndarray:
    """The LiDAR's buildings (blocks.Ground.built, a 1 m BNG grid) on the map's tiles,
    over `box` (x0, y0, x1, y1) or the whole map."""
    from buildings.blocks import TO_BNG
    e0, _n0, _e1, n1 = ground.box
    x0, y0, x1, y1 = box or (0, 0, W, H)
    out = np.zeros((H, W), bool)
    xs = np.arange(max(0, x0), min(W, x1))
    for y in range(max(0, y0), min(H, y1)):
        lat, lon = proj.to_latlon(xs + 0.5, np.full(len(xs), y + 0.5))       # a row in one go
        es, ns = TO_BNG(lon, lat)
        c = np.floor(es - e0).astype(int)
        r = np.floor(n1 - ns).astype(int)
        ok = (r >= 0) & (r < ground.built.shape[0]) & (c >= 0) & (c < ground.built.shape[1])
        row = np.zeros(len(xs), bool)
        row[ok] = ground.built[r[ok], c[ok]]
        out[y, xs] = row
    return out


def roof_runs(line_pts, built: np.ndarray) -> tuple[list, bool]:
    """Along a lane's real line: the runs of roof (sample ranges) and whether both ends are open."""
    line = shapely.LineString(line_pts)
    cov = []
    for d in np.arange(0.0, line.length + 1e-9, ROOF_STEP):
        p = line.interpolate(d)
        cov.append(bool(built[min(built.shape[0] - 1, int(p.y)), min(built.shape[1] - 1, int(p.x))]))
    runs, start, miss = [], None, 0
    for k, c in enumerate(cov):
        if c:
            start = k if start is None else start
            miss = 0
        elif start is not None:
            miss += 1
            if miss > ROOF_GAP:
                runs.append((start, k - miss))
                start, miss = None, 0
    if start is not None:
        runs.append((start, len(cov) - 1 - miss))
    return runs, bool(cov) and not cov[0] and not cov[-1]


def components(links) -> list[set]:
    """The network's parts: sets of link indices joined through shared nodes."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for l in links:
        parent[find(l.u)] = find(l.v)
    parts: dict = {}
    for i, l in enumerate(links):
        parts.setdefault(find(l.u), set()).add(i)
    return list(parts.values())


def plan(links, os_report: dict, way_tags: dict) -> tuple[list, list]:
    """(driveway links, OSM-only service lanes): indices into links."""
    drive = [i for i, l in enumerate(links) if is_driveway(l, way_tags)]
    osm_only = {e["link"] for e in (os_report or {}).get("osm only", [])}
    lanes = [i for i in sorted(osm_only) if links[i].cls == "service" and i not in set(drive)]
    return drive, lanes


def street_end(link, others) -> int:
    """Which end of a driveway (0 its first point, -1 its last) meets the rest of the network
    (`others`, the links not driveways): its first end where neither does - a driveway
    off another driveway, or one on its own."""
    nodes = {n for l in others for n in (l.u, l.v)}
    return 0 if link.u in nodes or link.v not in nodes else -1


def passage_check(pts, built: np.ndarray, buildings: list, height=None, cap: float = PASSAGE_CAP) -> tuple[bool, dict]:
    """A stuck lane a passage? One run of roof along its real line, open ground at both
    ends, the deepest building outline under the run no less deep along the lane than
    the run (a tile of slack: the LiDAR's edge) and the run at most `cap` tiles. (ok, what was found: length under roof, the building's depth
    there, why not, the LiDAR's height every ROOF_STEP along the line from `height`,
    a function of points.)"""
    line = shapely.LineString(pts)
    runs, open_ends = roof_runs(pts, built)
    info = {"runs": len(runs), "open ends": open_ends}
    if height is not None:
        ds = np.arange(0.0, line.length + 1e-9, ROOF_STEP)
        info["profile"] = [round(float(v), 1) if np.isfinite(v) else None
                           for v in height([line.interpolate(d).coords[0] for d in ds])]
    if len(runs) != 1 or not open_ends:
        return False, dict(info, why="not one run of roof with open ends")
    a, b = runs[0]
    length = (b - a + 1) * ROOF_STEP
    from shapely.ops import substring
    seg = substring(line, a * ROOF_STEP, (b + 1) * ROOF_STEP)
    crossing = max(((line.intersection(p).length, p) for p in buildings if p.intersects(seg)),
                   default=(0.0, None), key=lambda t: t[0])
    info.update({"under roof": round(length, 1), "building depth": round(crossing[0], 1)})
    if crossing[1] is None:
        return False, dict(info, why="no building outline under the roof (the LiDAR's own)")
    if length > crossing[0] + 1.0:
        return False, dict(info, why="longer than the building it crosses is deep")
    if length > cap:
        return False, dict(info, why=f"over {cap:g} tiles")
    return True, info


def resolve(route, lanes: list, footprints: np.ndarray, lidar_built, buildings_px: list, real_pts: dict,
            height=None, log=print) -> dict:
    """The lanes' fallbacks. route(hard) -> {link: path} routes the network with snap.HARD
    = hard; lidar_built() gives the LiDAR's buildings (worked out only when a lane is
    stuck); real_pts {lane: its real points}. Returns {"paths", "hard", "gap", "passages",
    "dropped", "checks"} - the last routing's paths and its obstacles, and the lanes by
    what became of them (passages and dropped still to take out: finish)."""
    hard = {k: footprints for k in lanes}
    paths = route(hard)
    stuck = [k for k in lanes if k not in paths]
    out = {"paths": paths, "gap": [], "passages": [], "dropped": [], "checks": {}, "hard": hard}
    if not stuck:
        return out
    lidar = lidar_built()
    hard = {k: (lidar if k in stuck else footprints) for k in lanes}
    paths = route(hard)
    out.update(paths=paths, hard=hard, gap=[k for k in stuck if k in paths])
    for k in stuck:
        if k in paths:
            continue
        ok, info = passage_check(real_pts[k], lidar, buildings_px, height)
        out["checks"][k] = info
        (out["passages"] if ok else out["dropped"]).append(k)
        log(f"  lane {k}: {'a passage' if ok else 'dropped'} ({info.get('why') or ''} "
            f"under roof {info.get('under roof')}, building {info.get('building depth')})")
    return out


def clear_footprints(footprints: np.ndarray, lines: list, width: int = PASSAGE_CLEAR) -> np.ndarray:
    """The footprints with a band `width` wide cleared along each line (a passage's real
    line: the router does not count its building against it)."""
    img = Image.new("1", (footprints.shape[1], footprints.shape[0]), 0)
    d = ImageDraw.Draw(img)
    for pts in lines:
        d.line([tuple(p) for p in pts], fill=1, width=width)
    return footprints & ~np.array(img, bool)


def finish(res: dict, n_links: int) -> tuple[list, dict, dict]:
    """After resolve: (the links kept - not dropped -, old index -> new, the hard obstacles
    for the final routing, renumbered: none for a passage, which runs under its building)."""
    dropped, passages = set(res["dropped"]), set(res["passages"])
    keep = [k for k in range(n_links) if k not in dropped]
    at = {old: k for k, old in enumerate(keep)}
    return keep, at, {at[k]: v for k, v in res["hard"].items() if k in at and k not in passages}


def drive_paving(drives: list, buildings_px, H: int, W: int, move=None, width: int = DRIVE_TILES) -> np.ndarray:
    """The driveways as plot paving: [(points in tiles, street end 0 or -1)], each drawn
    from its street end along its line until the first building it meets; `move` moves a
    point (x, y) with the town first (Generate's displacement)."""
    tree = shapely.STRtree(buildings_px) if buildings_px else None
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    for pts, end in drives:
        pts = [move(*p) for p in pts] if move else list(pts)
        if end == -1:
            pts = pts[::-1]
        line = shapely.LineString(pts)
        if tree is not None:
            # where the drive first enters a building: the nearest start of its parts inside one
            starts = []
            for i in tree.query(line):
                inside = line.intersection(buildings_px[int(i)])
                for part in getattr(inside, "geoms", [inside]):
                    if not part.is_empty:
                        starts.append(min(line.project(shapely.Point(c)) for c in part.coords))
            if starts:
                from shapely.ops import substring
                line = substring(line, 0.0, min(starts))
        if line.length > 0:
            d.line(list(line.coords), fill=1, width=width)
    return np.array(img, bool)


def route_lanes(sq, preset: str, obstacles: dict, fields=None) -> dict:
    """The preset's routing with snap.HARD = obstacles ({lane index: obstacle mask})."""
    from roadgraph import mode_test as T
    mode, params = T.RUNS[preset]
    snap.HARD = obstacles
    try:
        return sq.route(mode, params, fields)
    finally:
        snap.HARD = {}


def ndsm_at(sq):
    """height(points in tiles) -> the LiDAR's height above ground there (blocks.Ground)."""
    from buildings.blocks import TO_BNG
    g = sq.ground
    e0, _n0, _e1, n1 = g.box

    def height(pts):
        xs, ys = np.array(pts, float).T
        lat, lon = sq.proj.to_latlon(xs, ys)
        es, ns = TO_BNG(lon, lat)
        c, r = np.floor(es - e0).astype(int), np.floor(n1 - ns).astype(int)
        ok = (r >= 0) & (r < g.ndsm.shape[0]) & (c >= 0) & (c < g.ndsm.shape[1])
        out = np.full(len(pts), np.nan)
        out[ok] = g.ndsm[r[ok], c[ok]]
        return out
    return height


def profile_png(path: str, prof: list, title: str) -> None:
    """The LiDAR's height along a lane, a bar a sample (ROOF_STEP): roof brown, ground green."""
    S, H_ = 6, 160
    w = max(1, len(prof)) * S + 20
    img = Image.new("RGB", (w, H_ + 30), (250, 250, 248))
    d = ImageDraw.Draw(img)
    top = max([v for v in prof if v is not None] + [10.0])
    for k, v in enumerate(prof):
        if v is None:
            continue
        h = int(max(0.0, v) / top * H_)
        d.rectangle([10 + k * S, H_ + 10 - h, 10 + k * S + S - 2, H_ + 10], fill=(150, 96, 64) if v > 2.5 else (120, 160, 110))
    y = H_ + 10 - int(2.5 / top * H_)
    d.line([(10, y), (w - 10, y)], fill=(200, 40, 40))
    d.text((10, H_ + 14), f"{title}  (height above ground, m; red: 2.5 m; max {top:.1f})", fill=(20, 20, 20))
    img.save(path)


def slivers(sq, masks, fin, paths) -> list:
    """Each sliver patch (on the street, not at a junction): its depth into the building
    (squares / length along its long axis), and the street as drawn against the real gap
    between buildings there: (depth, drawn width, real gap)."""
    from scipy import ndimage
    from roadgraph import mode_test as T
    sl = masks["on"] & ~masks["wedge"] & sq.box_mask
    owner = masks["owner"]
    _d, (iy, ix) = ndimage.distance_transform_edt(owner <= 0, return_indices=True)
    link_at = owner[iy, ix] - 1
    lab, n = ndimage.label(sl)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        pts = np.c_[xs + 0.5, ys + 0.5]
        if len(pts) >= 2:
            c = pts - pts.mean(0)
            ax = np.linalg.svd(c, full_matrices=False)[2][0]
            length = float(np.ptp(c @ ax)) + 1.0
        else:
            length = 1.0
        depth = len(pts) / length
        i = int(np.bincount(link_at[ys, xs][link_at[ys, xs] >= 0]).argmax()) if (link_at[ys, xs] >= 0).any() else -1
        drawn = gap = None
        if i in paths and len(paths[i]) >= 2:
            l = fin.links[i]
            sides = fin.chosen.get(i, ((0, 0, 0), (0, 0, 0)))
            drawn = float(l.width) + sum(sum(sd) for sd in sides)
            # the real gap: at the same share of the way along the real line, square to it,
            # to the first real building each side
            path, real = shapely.LineString(paths[i]), shapely.LineString(l.pts)
            s_ = path.project(shapely.Point(pts.mean(0))) / max(path.length, 1e-9) * real.length
            s_ = min(s_, real.length - 1.0)
            piece = shapely.LineString([real.interpolate(max(0.0, s_)), real.interpolate(s_ + 1.0)])
            hits = [T.distances(piece, sq.real_mask, sgn)[0] for sgn in (1, -1)]
            gap = None if None in hits else hits[0] + hits[1]
        out.append((round(depth, 1), drawn, gap))
    return out


def _shifted(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """A mask moved dy rows and dx columns, nothing wrapping round the edges."""
    out = np.zeros_like(m)
    H, W = m.shape
    out[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = m[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
    return out


def fixes(sq, masks, fin, paths) -> dict:
    """The cheap fixes for slivers, in the harness only: push - each building on the street
    moved back from it along its front's normal (the grid axis from its overlap to its
    middle) by 1 or 2 tiles, where that lands on nothing else and clears more street;
    trim - the pavement (and verge) a tile narrower where a building stands on its
    outer tile; both - trim, then push what is left. Each scored: building squares on
    the street, slivers, frontage at line and flush (fronts against the routed street).
    masks: a run's (mode_test.run_preset), with its lots."""
    from scipy import ndimage
    from roadgraph import mode_test as T
    street = masks["road"] | masks["pave"] | masks["verge"]
    side = masks["pave"] | masks["verge"]
    lines = {i: shapely.LineString(p) for i, p in paths.items() if len(p) >= 2}

    def trim(st, built):
        outer = side & ndimage.binary_dilation(~street, structure=np.ones((3, 3)))
        return st & ~(outer & built)

    def push(st, labels):
        labels = labels.copy()
        moved = 0
        on_by = np.bincount(labels[st & (labels > 0)], minlength=labels.max() + 1)
        for lab in np.argsort(-on_by):
            if lab == 0 or on_by[lab] == 0:
                continue
            m = labels == lab
            ys, xs = np.nonzero(m & st)
            ay, ax_ = np.nonzero(m)
            dy, dx = ay.mean() - ys.mean(), ax_.mean() - xs.mean()
            step = (0, int(np.sign(dx))) if abs(dx) >= abs(dy) else (int(np.sign(dy)), 0)
            if step == (0, 0):
                continue
            best = None
            for k in (1, 2):
                sh = _shifted(m, k * step[0], k * step[1])
                if (sh & (labels > 0) & ~m).any():
                    break                                # a neighbour in the way: no further
                on = int((sh & st).sum())
                if on < int((m & st).sum()) and (best is None or on < best[0]):
                    best = (on, sh)
            if best:
                labels[m] = 0
                labels[best[1]] = lab
                moved += 1
        return labels, moved

    def score_(st, built, moved=0):
        on = built & st & sq.box_mask
        f = T.frontage(fin, lines, built)
        return {"on street": int(on.sum()), "slivers": int((on & ~masks["wedge"]).sum()),
                "at line": f["at line"], "flush": f["flush"], "buildings moved": moved}
    labels, built = masks["lots"], masks["built"]
    out = {"as it is": score_(street, built)}
    st_t = trim(street, built)
    out["trim"] = score_(st_t, built)
    pl, n = push(street, labels)
    out["push"] = score_(street, pl > 0, n)
    pl2, n2 = push(st_t, labels)
    out["both"] = score_(st_t, pl2 > 0, n2)
    return out


def main() -> int:
    from roadgraph import mode_test as T, overlap_floor as F
    map_dir, area = sys.argv[1], sys.argv[2]
    preset = T._arg("--preset", "cost-front")
    out = T._arg("--out", os.path.join(HERE, "..", "..", "out", "roadgraph", "mode_test", "lanes"))
    os.makedirs(out, exist_ok=True)
    sq = T.Square(map_dir, area, T.BOX)
    # --- the preset alone ---
    s0, m0, r0 = T.run_preset(sq, preset)
    _c0, cls0 = F.classify(sq, m0["on"] & sq.box_mask, m0["owner"])
    print(f"{preset}: {s0['on street']} building squares on the street")
    # --- 1. driveways out ---
    drive, lanes_all = plan(sq.links, sq.os_report, sq.way_tags)
    out_ = set(drive)
    rest = [i for i in range(len(sq.links)) if i not in out_]
    sub = subset(sq, rest)
    main_before = max(components(sq.links), key=len)
    main_after = max(components(sub.links), key=len)
    lost_access = [sub.links[k] for k in range(len(sub.links)) if rest[k] in main_before and k not in main_after]
    # --- 2. OSM-only lanes kept off the buildings ---
    new_of = {old: k for k, old in enumerate(rest)}
    lanes = [new_of[i] for i in lanes_all]
    fp = sq.fields["footprints"]
    x0, y0, x1, y1 = sq.box
    res = resolve(lambda hard: route_lanes(sub, preset, hard)["paths"], lanes, fp,
                  lambda: lidar_on_tiles(sq.ground, sq.proj, sq.H, sq.W, (x0 - 60, y0 - 60, x1 + 60, y1 + 60)),
                  sq.buildings, {k: sub.links[k].pts for k in lanes}, ndsm_at(sq))
    gap, passages, dropped = res["gap"], res["passages"], res["dropped"]
    # --- the final routing: dropped lanes out, passages under their buildings ---
    keep2, _at2, hard_fin = finish(res, len(sub.links))
    fin = subset(sub, keep2)
    pass_ways = {w for k in passages for w in sub.links[k].ways}
    fields = dict(sq.fields, footprints=clear_footprints(fp, [sub.links[k].pts for k in passages]))
    snap.HARD = hard_fin
    try:
        mode, params = T.RUNS[preset]
        r1 = fin.route(mode, params, fields)            # (HARD stays set through prepare)
    finally:
        snap.HARD = {}
    s1, m1, r1 = T.run_preset(fin, preset, passage_ways=pass_ways or None, r=r1)
    p0, p1 = r0["paths"], r1["paths"]
    lost_final = [k for k in hard_fin if k not in p1]
    # Driveways as plot paving, moved with the town, stopping at the first building.
    move = T._move(fin, p1)
    mv = lambda x, y: sq.proj.to_px(*move(*sq.proj.to_latlon(x, y)))  # noqa: E731
    others = [sq.links[i] for i in rest]
    paving = drive_paving([(sq.links[i].pts, street_end(sq.links[i], others)) for i in drive],
                          T.moved_rigid(sq, move), sq.H, sq.W, mv)
    paving &= ~(m1["road"] | m1["pave"] | m1["verge"])
    paving_under = int((paving & m1["built"] & sq.box_mask).sum())
    _c1, cls1 = F.classify(fin, m1["on"] & sq.box_mask, m1["owner"])
    passage_sq = int((m1["built"] & m1["passage"] & sq.box_mask).sum()) if m1["passage"] is not None else 0
    # --- passages: their length under the building and the roof along them ---
    pmd = ["| lane (ways) | became | under roof (tiles) | building depth there | why |", "|---|---|---|---|---|"]
    for k, info in res["checks"].items():
        ways = sub.links[k].ways[:2]
        became = "passage" if k in passages else "dropped"
        pmd.append(f"| {ways} | {became} | {info.get('under roof', '-')} | {info.get('building depth', '-')} | "
                   f"{info.get('why', '')} |")
        if info.get("profile"):
            profile_png(os.path.join(out, f"passage_{ways[0]}.png"), info["profile"], f"lane {ways}: {became}")
    # --- 4. slivers on the preset + lanes ---
    sl = slivers(sq, m1, fin, p1)
    depth_hist = np.histogram([dp for dp, _w, _g in sl], bins=[0, 1, 2, 3, 4, 6, 100])[0].tolist()
    diff = [w - g for _dp, w, g in sl if w is not None and g is not None]
    diff_hist = np.histogram(diff, bins=[-100, -2, 0, 2, 4, 6, 100])[0].tolist() if diff else []
    fx = fixes(sq, m1, fin, p1)
    # --- pictures ---
    pa = T.picture(sq, m0, p0, f"{preset}")
    pb = T.overlay(T.picture(fin, m1, p1, f"{preset} + lanes: drives tan, passages blue, dropped red"),
                   sq, paving, DRIVE_COLOUR)
    if m1["passage"] is not None:
        pb = T.overlay(pb, sq, m1["passage"] & m1["built"], F.COLOURS["passage"])
    dd = ImageDraw.Draw(pb)
    for k in dropped:
        dd.line([((x - x0) * 2, (y - y0) * 2) for x, y in sub.links[k].pts], fill=(255, 30, 30), width=3)
    T.sheet([pa, pb], 2, 8).save(os.path.join(out, "lanes.png"))
    pb.save(os.path.join(out, "lanes_after.png"))
    names = lambda ks, L: [f"{L[k].tags.get('name') or '(no name)'} (ways {L[k].ways[:2]})" for k in ks]  # noqa: E731
    rows = {
        preset: {"on street": s0["on street"], "slivers": s0["on street: slivers"], "wedges": s0["on street: wedges"],
                 "classes": cls0, "links": len(sq.links)},
        f"{preset} + lanes": {"on street": s1["on street"], "slivers": s1["on street: slivers"],
                              "wedges": s1["on street: wedges"], "classes": cls1, "links": len(fin.links),
                              "passage squares": passage_sq, "paving under buildings": paving_under},
    }
    md = [f"Square {sq.box}, {preset} against {preset} + lanes (lanes.py).", "",
          "| | building squares on the street | slivers | wedges | OS | other | passage | links |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        c = r["classes"]
        md.append(f"| {k} | {r['on street']:,} | {r['slivers']:,} | {r['wedges']:,} | {c['OS']['squares']:,} | "
                  f"{c['other']['squares']:,} | {c['passage']['squares']:,} | {r['links']} |")
    md += ["",
           f"Driveways out of the network: {len(drive)} links; painted as paving to the first building, "
           f"{paving_under} squares of it under buildings as placed; roads left without access: {len(lost_access)}.",
           f"OSM-only service lanes: {len(lanes)}; routed clear of the buildings at once "
           f"{len(lanes) - len(res['checks']) - len(gap)}, through a gap the LiDAR shows {len(gap)} {names(gap, sub.links)}, "
           f"made passages {len(passages)} {names(passages, sub.links)} ({passage_sq} squares under buildings), "
           f"dropped {len(dropped)} {names(dropped, sub.links)}; not routed in the final run {len(lost_final)}.",
           f"Links: {len(sq.links)} -> {len(fin.links)}.", "",
           f"Passages (capped at {PASSAGE_CAP:g} tiles and the building's depth; profiles passage_<way>.png):", ""] + pmd
    md += ["", f"Slivers on {preset} + lanes: {len(sl)} patches. Depth into the building (tiles): "
           f"{dict(zip(['<1', '1-2', '2-3', '3-4', '4-6', '6+'], depth_hist))}. Street as drawn minus the real gap "
           f"between buildings (tiles, {len(diff)} measured): "
           f"{dict(zip(['< -2', '-2-0', '0-2', '2-4', '4-6', '6+'], diff_hist)) if diff else '-'}.", "",
           "| cheap fix (harness only) | building squares on the street | slivers | at line | flush | buildings moved |",
           "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['on street']:,} | {v['slivers']:,} | {v['at line']:.1%} | {v['flush']:.1%} | {v['buildings moved']} |"
           for k, v in fx.items()]
    open(os.path.join(out, "lanes.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump({"rows": rows, "driveways": len(drive), "lost access": len(lost_access), "lanes": len(lanes),
               "gap": names(gap, sub.links), "passages": names(passages, sub.links),
               "dropped": names(dropped, sub.links), "checks": {str(sub.links[k].ways[:2]): v for k, v in res["checks"].items()},
               "slivers": sl, "fixes": fx}, open(os.path.join(out, "lanes.json"), "w"), indent=1, default=str)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
