"""How tidy a built map's buildings and roofs are.

    python roof_stats.py <knoxmap map dir> [<knoxmap map dir> ...]

Per map: how far buildings sit off the tile grid, and the roofs written into
the .tbx files - pieces per building and how many are flat strips (a pitched
roof's flat strip, or a sliver too narrow to pitch).
"""
import csv
import glob
import os
import re
import statistics
import sys


def stats(map_dir: str) -> None:
    name = os.path.basename(os.path.normpath(map_dir))
    rows = list(csv.DictReader(open(os.path.join(map_dir, f"{name}_placements.csv"), encoding="utf-8")))
    angles = [float(r["angle"]) for r in rows]
    pieces, flats, bad = [], 0, 0
    for path in glob.glob(os.path.join(map_dir, "buildings", "*.tbx")):
        text = open(path, encoding="utf-8").read()
        roofs = re.findall(r'type="roof"[^>]*?width="(\d+)"[^>]*?height="(\d+)"[^>]*?RoofType="(\w+)"', text)
        if not roofs:
            continue
        n_flat = sum(1 for w, h, t in roofs if t == "FlatTop")
        pieces.append(len(roofs))
        flats += n_flat
        bad += n_flat > len(roofs) / 2
    total = sum(pieces)
    print(f"{name}")
    print(f"  buildings {len(rows)}; off the grid: <=8 deg {sum(a <= 8 for a in angles)}, "
          f"8-15 {sum(8 < a <= 15 for a in angles)}, >15 {sum(a > 15 for a in angles)}")
    print(f"  roof pieces {total} over {len(pieces)} roofed buildings: median "
          f"{statistics.median(pieces)} per building, max {max(pieces)}; "
          f"flat {flats} ({100 * flats / total:.0f}%); buildings mostly flat {bad}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        stats(d)
