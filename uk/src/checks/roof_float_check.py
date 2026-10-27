"""Find roof pieces that are not over walls, across a built map.

    python roof_float_check.py <knoxmap map dir> [--list N]

A pitched roof on floor f must cover rooms on floor f-1; a flat roof on
floor f covers rooms on floor f itself. Reports, per kind of piece, how many
roof tiles have nothing under them, and lists the worst buildings.
"""
import glob
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def main() -> int:
    m = sys.argv[1]
    n_list = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 10
    total, floating, worst = Counter(), Counter(), []
    for path in glob.glob(os.path.join(m, "buildings", "*.tbx")):
        root = ET.parse(path).getroot()
        if root.tag != "building":
            continue
        w, h = int(root.get("width")), int(root.get("height"))
        floors = root.findall("floor")
        grids = []
        for f in floors:
            g = f.find("rooms")
            vals = [int(v) for v in ((g.text if g is not None else "") or "").replace("\n", ",").split(",") if v.strip()]
            grids.append(vals if len(vals) == w * h else [0] * (w * h))
        bad_here = 0
        for lvl, f in enumerate(floors):
            for o in f.iter("object"):
                if o.get("type") != "roof":
                    continue
                kind = o.get("RoofType")
                x, y, rw, rh = (int(o.get(k)) for k in ("x", "y", "width", "height"))
                under = lvl if kind == "FlatTop" else lvl - 1
                cells = [(cx, cy) for cy in range(y, y + rh) for cx in range(x, x + rw) if cx < w and cy < h]
                empty = sum(1 for cx, cy in cells if under < 0 or not grids[under][cy * w + cx])
                key = "flat" if kind == "FlatTop" else ("dormer" if kind.startswith("Dormer") else
                                                         "lean-to" if kind.startswith("Slope") else "gable")
                total[key] += len(cells)
                floating[key] += empty
                bad_here += empty
        if bad_here:
            worst.append((bad_here, os.path.basename(path)))
    print("roof tiles with nothing under them:")
    for k in total:
        print(f"  {k:8s} {floating[k]:6d} of {total[k]:6d} ({100 * floating[k] / max(1, total[k]):.1f}%)")
    print(f"buildings affected: {len(worst)}")
    for n, name in sorted(worst, reverse=True)[:n_list]:
        print(f"  {n:4d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
