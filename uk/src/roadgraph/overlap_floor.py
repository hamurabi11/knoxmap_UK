"""The building squares on the street that no road mode moves, and passages for them
(LOG 2026-10-25 #6).

    python overlap_floor.py <source map dir> <area data dir> [--runs graph,cost-low,cost-low-occ1]
                            [--passages cost-front] [--passage-max 6] [--front-near 6] [--out DIR]
                            [--floor-only]

1. The residual floor: each run (a road option, mode_test.RUNS) routed and placed as
   mode_test.py does (blocks.py's own steps); the squares where a building stands on
   the street in every run are the floor. Each patch of them (4-connected) is put in
   one class, the first that holds:
     passage    within PASSAGE_NEAR tiles of an OSM way tagged to pass under a building
                (passages.is_passage: tunnel=building_passage, covered=yes, a layer below 0)
                - the arches the map already builds as passages, and lanes at a
                building's edge
     OS         its street (the nearest carriageway square's link) has no OS Open Roads
                match, or OS gives it another class (osroads.conflate): a road OSM draws
                that OS does not, or not as OSM says
     other      the rest
2. Passages, a prototype on the lanes east of the High Street only (EAST): each road
   there (not a footway) that an OSM tag already puts under a building, or whose real
   line crosses a single building that fronts a named street for under PASSAGE_MAX
   tiles, is a passage: the router does not count that building against it
   (occupancy), blocks.py keeps the building over it (load_streets' passage_ways,
   as if tagged), and its squares under the building are a passage through the ground
   floor (regional/passages.py, Build), not overlap. The run with and without, in EAST.
   Build's defaults are not changed.

Writes <out>/floor.md, floor.json, floor.png (the floor's squares by class, on the
first run's picture) and passages.png (EAST, without and with).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from roadgraph import lanes as LN, mode_test as T  # noqa: E402
from regional.passages import ON_FOOT, is_passage  # noqa: E402

_arg = T._arg

PASSAGE_NEAR = 3.0          # tiles from an OSM passage's line: its squares
PASSAGE_MAX = 6.0           # tiles: a road crossing one frontage building for less is a passage
FRONT_NEAR = 6.0            # tiles: a building this near a named street fronts it
EAST = (845, 700, 1000, 1280)   # the lanes east of the High Street (map tiles of Alton_town)
CLASSES = ("passage", "OS", "other")
COLOURS = {"passage": (60, 160, 255), "OS": (255, 200, 0), "other": (255, 30, 30)}   # (T.picture's own
                                                    # pink is the rest of that run's building squares on the street)


def classify(sq, floor: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, dict]:
    """Each patch of the floor its class (CLASSES index + 1, 0 none) and the counts."""
    img = Image.new("1", (sq.W, sq.H), 0)
    d = ImageDraw.Draw(img)
    for w, tags in sq.way_tags.items():
        if is_passage(tags) and w in sq.real_ways:
            d.line([tuple(p) for p in sq.real_ways[w]], fill=1, width=1)
    near_passage = ndimage.distance_transform_edt(~np.array(img, bool)) <= PASSAGE_NEAR
    rep = sq.os_report or {}
    mismatch = {e["link"] for e in rep.get("osm only", [])} | {e["link"] for e in rep.get("class changed", [])}
    _d, (iy, ix) = ndimage.distance_transform_edt(owner <= 0, return_indices=True)
    link_at = owner[iy, ix] - 1
    lab, n = ndimage.label(floor)
    cls = np.zeros(floor.shape, np.int8)
    counts = {c: {"squares": 0, "patches": 0} for c in CLASSES}
    for k in range(1, n + 1):
        here = lab == k
        if (here & near_passage).any():
            c = "passage"
        elif any(int(i) in mismatch for i in np.unique(link_at[here])):
            c = "OS"
        else:
            c = "other"
        cls[here] = CLASSES.index(c) + 1
        counts[c]["squares"] += int(here.sum())
        counts[c]["patches"] += 1
    return cls, counts


def candidates(sq, box, most=PASSAGE_MAX, front=FRONT_NEAR) -> dict:
    """{OSM way id: why} - the roads in `box` to be passages (see the module's 2): crossing
    one building for under `most` tiles, the building within `front` of a named street."""
    x0, y0, x1, y1 = box
    named = [shapely.LineString(l.pts) for l in sq.links if l.tags.get("name") and l.cls in T.FRONT_CLASSES]
    tree = shapely.STRtree(sq.buildings)
    out = {}
    for w, pts in sq.real_ways.items():
        tags = sq.way_tags.get(w, {})
        if (tags.get("highway") or "") in ON_FOOT or len(pts) < 2:
            continue
        if not any(x0 <= x < x1 and y0 <= y < y1 for x, y in pts):
            continue
        line = shapely.LineString(pts)
        hits = [sq.buildings[int(k)] for k in tree.query(line)
                if line.crosses(sq.buildings[int(k)]) or line.within(sq.buildings[int(k)])]
        if is_passage(tags) and hits:
            out[w] = "tagged"
            continue
        if len(hits) != 1:
            continue
        across = line.intersection(hits[0]).length
        fronts = any(hits[0].distance(n) <= front for n in named if not n.equals(line))
        if 0 < across < most and fronts:
            out[w] = f"crosses one frontage building for {across:.1f} tiles"
    return out


def crop(img: Image.Image, sq, box, S=2) -> Image.Image:
    bx0, by0 = sq.box[:2]
    x0, y0, x1, y1 = box
    return img.crop(((x0 - bx0) * S, (y0 - by0) * S, (x1 - bx0) * S, (y1 - by0) * S))


def main() -> int:
    map_dir, area = sys.argv[1], sys.argv[2]
    runs = _arg("--runs", "graph,cost-low,cost-low-occ1").split(",")
    proto = _arg("--passages", "cost-front")
    out = _arg("--out", os.path.join(HERE, "..", "..", "out", "roadgraph", "mode_test", "floor"))
    os.makedirs(out, exist_ok=True)
    sq = T.Square(map_dir, area, T.BOX)
    # --- 1. the floor ---
    floor, first = None, None
    per_run = {}
    for name in runs:
        s, masks, r = T.run_preset(sq, name)
        per_run[name] = s["on street"]
        floor = masks["on"] if floor is None else floor & masks["on"]
        first = first or (masks, r["paths"])
        print(f"{name}: {s['on street']} building squares on the street")
    floor &= sq.box_mask
    cls, counts = classify(sq, floor, first[0]["owner"])
    pic = T.picture(sq, first[0], first[1], f"the floor ({', '.join(runs)}): passage blue, OS yellow, other red")
    for k, c in enumerate(CLASSES, 1):
        pic = T.overlay(pic, sq, cls == k, COLOURS[c])
    pic.save(os.path.join(out, "floor.png"))
    if "--floor-only" in sys.argv:
        print({c: counts[c] for c in CLASSES})
        return 0
    # --- 2. passages east of the High Street ---
    most, front = float(_arg("--passage-max", PASSAGE_MAX)), float(_arg("--front-near", FRONT_NEAR))
    cand = candidates(sq, EAST, most, front)
    strict = cand if (most, front) == (PASSAGE_MAX, FRONT_NEAR) else candidates(sq, EAST)
    print(f"passage candidates east of the High Street: {len(strict)} at {PASSAGE_MAX:g} / {FRONT_NEAR:g} tiles, "
          f"{len(cand)} at {most:g} / {front:g}")
    # the router lets them through their building
    fields = dict(sq.fields, footprints=LN.clear_footprints(sq.fields["footprints"], [sq.real_ways[w] for w in cand]))
    ex0, ey0, ex1, ey1 = EAST
    east = (slice(ey0, ey1), slice(ex0, ex1))
    s0, m0, r0 = T.run_preset(sq, proto)
    s1, m1, r1 = T.run_preset(sq, proto, fields, set(cand))
    p0, p1 = r0["paths"], r1["paths"]
    passage_cells = int((m1["built"] & m1["passage"])[east].sum()) if m1["passage"] is not None else 0
    proto_rows = {
        "without": {"on street": int(m0["on"][east].sum()), "built squares": int(m0["built"][east].sum())},
        "with": {"on street": int(m1["on"][east].sum()), "built squares": int(m1["built"][east].sum()),
                 "passage squares": passage_cells},
    }
    both = [crop(T.picture(sq, m, p, f"{proto} {k} passages"), sq, EAST) for k, m, p in
            (("without", m0, p0), ("with", m1, p1))]
    T.sheet(both, 2, 8).save(os.path.join(out, "passages.png"))
    total = int(floor.sum())
    md = [f"Square {sq.box}. Building squares on the street per run: "
          + ", ".join(f"{k} {v:,}" for k, v in per_run.items()) + f"; in every run (the floor): {total:,}.", "",
          "| class | squares | patches |", "|---|---|---|"]
    md += [f"| {c} | {counts[c]['squares']:,} | {counts[c]['patches']} |" for c in CLASSES]
    md += ["", f"Passages east of the High Street ({proto}, box {EAST}): at {PASSAGE_MAX:g} tiles across and "
           f"{FRONT_NEAR:g} from a named street {len(strict)} roads; tried at {most:g} and {front:g}: {len(cand)} - "
           + "; ".join(f"{w}: {why}" for w, why in cand.items()), "",
           "| | building squares on the street | building squares | passage squares |", "|---|---|---|---|",
           f"| without | {proto_rows['without']['on street']:,} | {proto_rows['without']['built squares']:,} | - |",
           f"| with | {proto_rows['with']['on street']:,} | {proto_rows['with']['built squares']:,} | "
           f"{proto_rows['with']['passage squares']:,} |"]
    open(os.path.join(out, "floor.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump({"runs": per_run, "floor": total, "classes": counts, "strict candidates": strict,
               "candidates": cand, "thresholds": [most, front], "passages": proto_rows},
              open(os.path.join(out, "floor.json"), "w"), indent=1)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
