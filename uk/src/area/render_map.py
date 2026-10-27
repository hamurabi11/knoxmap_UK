"""Draw buildings.geojson coloured by a property, with a legend.

    python render_map.py <data dir> <property> <out.png> [scale_px_per_m]

Listed buildings get a red outline.
"""
import json
import os
import sys
from collections import Counter

from PIL import Image, ImageDraw
from pyproj import Transformer

TO_BNG = Transformer.from_crs(4326, 27700, always_xy=True)
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207),
           (127, 127, 127), (174, 199, 232), (255, 187, 120), (152, 223, 138)]
FIXED = {"outbuilding": (190, 190, 190), "": (230, 230, 230), "None": (230, 230, 230)}


def main() -> int:
    data_dir, prop, out = sys.argv[1], sys.argv[2], sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    with open(os.path.join(data_dir, "buildings.geojson"), encoding="utf-8") as f:
        feats = json.load(f)["features"]
    rings = []
    for ft in feats:
        ring = [TO_BNG.transform(x, y) for x, y in ft["geometry"]["coordinates"][0]]
        rings.append((ring, str(ft["properties"].get(prop)), ft["properties"].get("listed_grade")))
    xs = [x for r, _, _ in rings for x, _ in r]
    ys = [y for r, _, _ in rings for _, y in r]
    x0, y1 = min(xs) - 20, max(ys) + 20
    w = int((max(xs) + 20 - x0) * scale)
    h = int((y1 - (min(ys) - 20)) * scale)
    counts = Counter(v for _, v, _ in rings)
    values = [v for v, _ in counts.most_common()]
    colours, k = {}, 0
    for v in values:
        if v in FIXED:
            colours[v] = FIXED[v]
        else:
            colours[v] = PALETTE[k % len(PALETTE)]
            k += 1
    legend_h = 18 * len(values) + 10
    img = Image.new("RGB", (w, h + legend_h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for ring, v, listed in rings:
        pts = [((x - x0) * scale, (y1 - y) * scale) for x, y in ring]
        d.polygon(pts, fill=colours[v], outline=(200, 0, 0) if listed else (80, 80, 80))
        if listed:
            d.line(pts + [pts[0]], fill=(220, 0, 0), width=2)
    for i, v in enumerate(values):
        y = h + 5 + 18 * i
        d.rectangle((8, y, 22, y + 12), fill=colours[v], outline=(0, 0, 0))
        d.text((28, y), f"{v}  ({counts[v]})", fill=(0, 0, 0))
    d.text((w - 220, h + 5), "red outline = listed building", fill=(200, 0, 0))
    img.save(out)
    print(out, img.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
