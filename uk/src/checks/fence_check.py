"""Garden boundary checks on a built map (the start of PLAN K's tidying script).

    python fence_check.py <knoxmap map dir>

Reads what Build wrote (buildings/<map>_fences_<cx>_<cy>.tbx, placed as the
WorldEd project <map>.pzw places them) and the ground
(<map>.bmp), and counts:
- edges walling a way in: a boundary edge where a garden path (paving stone)
  or a drive (tarmac) meets the pavement or road - the opening Build leaves;
- edges across tarmac: tarmac on both sides (a drive or lane fenced across);
- angled pictures with no collision: a square with an angled boundary's
  picture and no wall piece on it or beside it (a hole you can walk through);
- dashed angled fronts: an angled picture with no other next to it but one
  two squares on - every other square of a front drawn (PLAN K);
- along the street: each piece running within 30 degrees of the street's
  exact edge (plots2's <map>_street.wkb) and within 4 tiles of it - in the
  street by more than half a square, on the edge, or set back (user,
  2026-10-14: fences on the road or pavement's edge).
Each is a count and the first few squares, so a change that makes things
worse shows at once (the 2026-10-12 plot change doubled the runs unnoticed).
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from generator import pz_colors as C  # noqa: E402

CELL = 300
DIAG = "boundaries_diag_uk_01_"
SLOTS, INVISIBLE = 24, 16


def pieces(map_dir: str, name: str) -> dict:
    """{(x, y): [tile, ...]} from the fence lots, each where the project places it."""
    out: dict = {}
    bdir = os.path.join(map_dir, "buildings")
    pzw = open(os.path.join(map_dir, f"{name}.pzw"), encoding="utf-8").read()
    at = {}
    for cx, cy, body in re.findall(r'<cell x="(\d+)" y="(\d+)"(.*?)</cell>', pzw, re.S):
        for lx, ly, mp in re.findall(r'<lot x="(-?\d+)" y="(-?\d+)"[^>]*map="buildings/([^"]+)"', body):
            at[mp] = (int(cx) * CELL + int(lx), int(cy) * CELL + int(ly))
    for f in os.listdir(bdir):
        m = re.match(rf"{re.escape(name)}_fences_(\d+)_(\d+)\.tbx$", f)
        if not m:
            continue
        ox, oy = at[f]
        t = open(os.path.join(bdir, f), encoding="utf-8").read()
        furn = re.findall(r'<furniture(?: layer="\w+")?>\s*<entry orient="W">\s*<tile x="0" y="0" name="([^"]+)"', t)
        for i, x, y in re.findall(r'FurnitureTiles="(\d+)" orient="W" x="(\d+)" y="(\d+)"', t):
            out.setdefault((ox + int(x), oy + int(y)), []).append(furn[int(i)])
    return out


def sides_of(tile: str) -> set:
    """Which edges a boundary tile blocks: W, N or both (posts and pictures: none)."""
    if tile.startswith(DIAG):
        slot = int(tile[len(DIAG):]) % SLOTS
        return {INVISIBLE: {"W"}, INVISIBLE + 1: {"N"}, INVISIBLE + 2: {"W", "N"}}.get(slot, set())
    # Straight pieces: the style tables' order (west, north, corner) is not
    # in the name; read it from the fence styles.
    return STRAIGHT.get(tile, set())


def main() -> int:
    map_dir = sys.argv[1]
    name = os.path.basename(os.path.normpath(map_dir))
    from knoxbuild.fences import STYLES, _tile
    global STRAIGHT
    STRAIGHT = {}
    for st in STYLES.values():
        for key, sides in (("west1", {"W"}), ("west2", {"W"}), ("north1", {"N"}), ("north2", {"N"}), ("nw", {"W", "N"})):
            STRAIGHT[_tile(st[key])] = sides
    g = np.asarray(Image.open(os.path.join(map_dir, f"{name}.bmp")).convert("RGB"))
    H, W = g.shape[:2]

    def colour(*cols):
        m = np.zeros((H, W), bool)
        for c in cols:
            m |= np.all(g == c, axis=2)
        return m
    tarmac = colour(C.MEDIUM_ASPHALT, C.DARK_ASPHALT, C.DARKEST_ASPHALT, C.DARK_POTHOLE, C.LIGHT_POTHOLE)
    path = colour(C.PAVING_STONE)
    walk = colour(C.PALE_CONCRETE, C.PAVING) | tarmac
    pcs = pieces(map_dir, name)
    edges = []                                   # ((x, y), side, tile)
    for (x, y), tiles in pcs.items():
        for t in tiles:
            for sd in sides_of(t):
                edges.append(((x, y), sd, t))

    def across(x, y, sd):
        bx, by = (x, y - 1) if sd == "N" else (x - 1, y)
        return (x, y), (bx, by)
    ok = lambda p: 0 <= p[0] < W and 0 <= p[1] < H  # noqa: E731
    mouths, tar, tar_seen, ex_m, ex_t = 0, 0, 0, [], []
    for (x, y), sd, t_ in edges:
        a, b = across(x, y, sd)
        if not (ok(a) and ok(b)):
            continue
        if ((path[a[1], a[0]] or tarmac[a[1], a[0]]) and walk[b[1], b[0]] and not tarmac[b[1], b[0]] and not tarmac[a[1], a[0]]) \
                or ((path[b[1], b[0]] or tarmac[b[1], b[0]]) and walk[a[1], a[0]] and not tarmac[a[1], a[0]] and not tarmac[b[1], b[0]]):
            mouths += 1
            ex_m.append(a)
        if tarmac[a[1], a[0]] and tarmac[b[1], b[0]]:
            tar += 1
            if not t_.startswith(DIAG):                   # a straight piece you see, not an angled front's collision
                tar_seen += 1
                ex_t.append(a)
    blocking = {p for (p, _sd, _t) in edges}
    holes, ex_h = 0, []
    for (x, y), tiles in pcs.items():
        if not any(t.startswith(DIAG) and int(t[len(DIAG):]) % SLOTS < 14 for t in tiles):
            continue
        near = any((x + dx, y + dy) in blocking for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        if not near:
            holes += 1
            ex_h.append((x, y))
    pics = {p for p, tiles in pcs.items()
            if any(t.startswith(DIAG) and int(t[len(DIAG):]) % SLOTS < 14 for t in tiles)}
    dashed = sorted(p for p in pics
                    if not any((p[0] + dx, p[1] + dy) in pics for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
                    and any((p[0] + dx, p[1] + dy) in pics for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)
                            if max(abs(dx), abs(dy)) == 2))
    print(f"{name}: boundary edges {len(edges)}")
    print(f"  edges walling a path or drive mouth: {mouths} {ex_m[:6]}")
    print(f"  edges across tarmac (both sides): {tar}, of them straight pieces you see: {tar_seen} {ex_t[:6]}")
    print(f"  angled pictures with no wall piece within a square: {holes} {ex_h[:6]}")
    print(f"  dashed angled fronts (a picture alone, another two squares on): {len(dashed)} of {len(pics)} {dashed[:8]}")
    wkb = os.path.join(map_dir, f"{name}_street.wkb")
    if os.path.exists(wkb):
        for kind, (n, bands) in along_street(pcs, wkb).items():
            print(f"  {kind} pieces along the street: {n}; " + ", ".join(f"{b} {v} ({v / max(1, n):.0%})" for b, v in bands.items()))
    return 0


def along_street(pcs: dict, wkb_path: str) -> dict:
    """{kind: (pieces, {band: count})} for the pieces running along the
    street: angled pictures (the middle of their cut) and straight sides."""
    import shapely
    from knoxbuild.fences import DIAG_CUTS
    street = shapely.from_wkb(open(wkb_path, "rb").read())
    edge = street.boundary
    pts, kinds, dirs = [], [], []
    for (x, y), tiles in pcs.items():
        for t in tiles:
            if t.startswith(DIAG):
                slot = int(t[len(DIAG):]) % SLOTS
                if slot < 14:
                    a, b = DIAG_CUTS[slot]
                    pts.append((x + (a[0] + b[0]) / 2, y + (a[1] + b[1]) / 2))
                    kinds.append("angled")
                    dirs.append((b[0] - a[0], b[1] - a[1]))
                continue
            sd = sides_of(t)
            for side, p, d in (("W", (x, y + 0.5), (0, 1)), ("N", (x + 0.5, y), (1, 0))):
                if side in sd:
                    pts.append(p)
                    kinds.append("straight")
                    dirs.append(d)
    if not pts:
        return {}
    P = shapely.points(np.array(pts))
    d = shapely.distance(P, edge)
    signed = np.where(shapely.contains(street, P), -d, d)
    e = shapely.get_point(shapely.shortest_line(P, edge), 1)
    s0 = shapely.line_locate_point(edge, e)
    ea, eb = (shapely.line_interpolate_point(edge, v) for v in (np.maximum(s0 - 0.7, 0), s0 + 0.7))
    ex, ey = shapely.get_x(eb) - shapely.get_x(ea), shapely.get_y(eb) - shapely.get_y(ea)
    D = np.array(dirs, float)
    cos = np.abs(D[:, 0] * ex + D[:, 1] * ey) / (np.hypot(D[:, 0], D[:, 1]) * np.maximum(np.hypot(ex, ey), 1e-9))
    near = (signed <= 4) & (cos > 0.866)
    out = {}
    for kind in ("angled", "straight"):
        sel = near & (np.array(kinds) == kind)
        v = signed[sel]
        out[kind] = (int(sel.sum()), {"in the street": int((v < -0.5).sum()), "on the edge": int((abs(v) <= 0.5).sum()),
                                      "set back up to 1.5": int(((v > 0.5) & (v <= 1.5)).sum()),
                                      "set back further": int((v > 1.5).sum())})
    return out


STRAIGHT: dict = {}

if __name__ == "__main__":
    sys.exit(main())
