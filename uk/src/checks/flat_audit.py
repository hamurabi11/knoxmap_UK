"""Where a built map's flat roof comes from.

    python flat_audit.py <knoxmap map dir>

Roof area, in tiles, split into:
  pitched       45/30-degree slopes that peak fully
  flat top      the flat band BuildingEd puts along the top of a 45-degree
                gable wider than 6 tiles across (Depth Three is its limit)
  flat piece    FlatTop pieces, by width (slivers <= 2 wide, or 3+)
and whether the building had a roof planned from LiDAR or KnoxMap's own.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

FULL_PEAK = 6        # 45-degree gable peaks fully up to this many tiles across


def main() -> int:
    m = sys.argv[1]
    name = os.path.basename(os.path.normpath(m))
    regional = json.load(open(os.path.join(m, "regional.json"), encoding="utf-8"))
    planned_ids = {k for k, v in regional["buildings"].items() if v.get("roof")}
    import csv
    rows = list(csv.DictReader(open(os.path.join(m, f"{name}_placements.csv"), encoding="utf-8")))
    geo = json.load(open(os.path.join(m, f"{name}_buildings.geojson"), encoding="utf-8"))["features"]
    # Building file names carry the index of the outline in the geojson.
    planned_files = set()
    for r in rows:
        idx = int(re.match(rf"{re.escape(name)}_(\d+)", r["file"]).group(1))
        if str(geo[idx]["properties"].get("osm_id")) in planned_ids:
            planned_files.add(r["file"])
    area = Counter()
    for path in glob.glob(os.path.join(m, "buildings", f"{name}_[0-9]*.tbx")):
        src = "planned" if os.path.basename(path) in planned_files else "knoxmap"
        text = open(path, encoding="utf-8").read()
        for w, h, t in re.findall(r'type="roof"[^>]*?width="(\d+)"[^>]*?height="(\d+)"[^>]*?RoofType="(\w+)"', text):
            w, h = int(w), int(h)
            if t == "FlatTop":
                area[(src, "flat piece <=2 wide" if min(w, h) <= 2 else "flat piece 3+ wide")] += w * h
            elif t in ("PeakWE", "PeakNS"):
                across = h if t == "PeakWE" else w
                along = w if t == "PeakWE" else h
                band = max(0, across - FULL_PEAK)
                area[(src, "flat top on wide gable")] += band * along
                area[(src, "pitched")] += (across - band) * along
            else:
                area[(src, "pitched")] += w * h
    for src in ("planned", "knoxmap"):
        tot = sum(v for (s, _k), v in area.items() if s == src)
        if not tot:
            continue
        print(f"{src}: roof area {tot}")
        for (s, k), v in sorted(area.items()):
            if s == src:
                print(f"   {k:24s} {v:6d}  {100 * v / tot:4.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
