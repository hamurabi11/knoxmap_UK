"""Top-down picture of a vanilla B42 cell's ground floor: one pixel block per square.

    python topdown.py <map dir> <cell x> <cell y> <out.png> [--scale 4] [--box x,y,w,h]

Colours: road surface dark grey, kerbs white, diagonal (2:1) kerbs red,
road lines yellow, grass green, other ground tan, anything built light grey.
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src: ukpaths (where KnoxMap and asset_creator are)
import ukpaths  # noqa: E402
sys.path.insert(0, ukpaths.KNOX)
from tools.render_lots import read_header, read_squares  # noqa: E402

CELL = 256


def colour(tiles: list[str]) -> tuple[int, int, int]:
    names = " ".join(tiles)
    if "street_curbs_01_diag" in names:
        return (220, 40, 40)
    if "street_curbs" in names:
        return (250, 250, 250)
    if "street_trafficlines" in names:
        return (240, 210, 40)
    floor = tiles[0] if tiles else ""
    if floor.startswith(("blends_street", "floors_exterior_street")):
        return (70, 70, 75)
    if floor.startswith("blends_natural_01") and any(t.startswith(("blends_natural_01", "e_newgrass", "blends_grassoverlays")) for t in tiles):
        return (90, 130, 60)
    if floor.startswith("blends_natural"):
        return (170, 150, 110)
    if floor.startswith("floors_exterior"):
        return (150, 150, 150)
    return (200, 200, 205) if floor else (20, 20, 20)


def main() -> int:
    map_dir, cx, cy, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    scale = int(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 4
    box = [int(v) for v in sys.argv[sys.argv.index("--box") + 1].split(",")] if "--box" in sys.argv else [0, 0, CELL, CELL]
    names, lo, hi = read_header(os.path.join(map_dir, f"{cx}_{cy}.lotheader"))
    sq = read_squares(os.path.join(map_dir, f"world_{cx}_{cy}.lotpack"), names, lo, hi,
                      (box[0], box[1], box[0] + box[2], box[1] + box[3]))
    img = Image.new("RGB", (box[2], box[3]), (20, 20, 20))
    for (x, y, z), tiles in sq.items():
        if z == 0:
            img.putpixel((x - box[0], y - box[1]), colour(tiles))
    img.resize((box[2] * scale, box[3] * scale), Image.NEAREST).save(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
