"""The sheets from street_objects.py's renders: sheet, depth picture, preview.

    python street_sheet.py [sheet ...]     (the Knoxmap venv's Python: numpy, scipy, PIL)

For every sheet street_objects.py has rendered (or those named), reads
asset_creator/renders/<sheet>/ (each slot's sprite and map positions,
manifest.json) and writes into Tile_creator/build/sheets/<group>/, where pack.py
reads its sheets:
- <sheet>.png, 8 columns of 128 x 256 sprites;
- DEPTH_<sheet>.png, each sprite pixel's depth by the
  game's encoding (walls.py: 254.2 - 62.8 (x + y) - 42.4 z, map metres from
  the square's north corner) - a pixel the depth pass missed (an
  antialiased edge) takes its nearest neighbour's, so none is left without
  (a sprite pixel with no depth is cut in game: LOG 2026-10-13);
- <sheet>.geometry.json, each slot's collision shapes in the
  game's units (a square -5000..5000 across, 10000 a metre up), for
  pack.py's tileGeometry.txt;
and asset_creator/renders/<sheet>_preview.png (the sheet on the pavement
flags) and street_furniture_scene.png (a 1990s street corner).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Tile_creator", "src"))
import floor_tiles as F  # noqa: E402
import walls as W  # noqa: E402
from sheet_paths import sheet_file  # noqa: E402

RENDERS = os.path.join(ROOT, "renders")
ALL = ["street_furniture_uk_01", "street_furniture_uk_02", "road_signs_uk_01", "doors_uk_01", "vegetation_uk_01",
       "street_lighting_uk_01", "road_markings_uk_01", "countryside_uk_01",
       "street_curbs_uk_01", "street_curbs_uk_01_diag", "street_curbs_uk_01_diag_2",
       "road_markings_uk_02", "street_furniture_uk_03", "pub_signs_uk_01", "street_drains_uk_01"]


def slot_files(sheet, entry):
    base = os.path.join(RENDERS, sheet, f"{entry['slot']:02d}_{entry['name']}")
    return base + ".png", base + ".npy"


FLAT_Z = 0.03       # metres: a pixel this low is paint on the ground...
FLAT_BIAS = 10      # ...drawn this much nearer than the ground (depth units; 62.8 a metre)


def depth_of(rgba, pos):
    """The sprite's depth picture (RGBA, grey = depth, alpha where the sprite has pixels)."""
    seen = rgba[..., 3] > 0
    have = pos[..., 3] > 0.5
    value = W.DEPTH_V0 - W.DEPTH_ALONG * (pos[..., 0] + pos[..., 1]) - W.DEPTH_UP * pos[..., 2]
    # Paint on the ground (a road marking, a drain's grating) a little nearer
    # than the ground under it, as vanilla's kerbs are (DEPTH_street_curbs_01:
    # 1-23 nearer, 20 on average): level with it, the game's depth test was a
    # tie that rounding decided pixel by pixel, and double yellows on the
    # south and east kerbs came out dashed (user, 2026-10-18).
    # On a flat floor depth follows the screen row alone, so paint takes the
    # floor's by its row (the position pass gives a few edge pixels the
    # wrong surface, down to a metre below ground).
    rows = np.arange(rgba.shape[0])[:, None] + 0.5 + np.zeros((1, rgba.shape[1]))
    ground = W.DEPTH_V0 - W.DEPTH_ALONG * (rows - 192) / 32
    value = np.where(pos[..., 2] < FLAT_Z, ground - FLAT_BIAS, value)
    if have.any():
        _d, (iy, ix) = ndimage.distance_transform_edt(~have, return_indices=True)
        value = value[iy, ix]                          # every pixel: its nearest covered pixel's depth
    out = np.zeros(rgba.shape, np.uint8)
    out[..., :3] = np.clip(np.round(value), 0, 255).astype(np.uint8)[..., None]
    out[..., 3] = np.where(seen, 255, 0)
    out[~seen] = 0
    return out


def geometry_units(shapes):
    """Map-metre shapes -> the game's: x and z across (square middle 0, +-5000), y up (10000 a metre)."""
    out = []
    for g in shapes:
        if g[0] == "cylinder":
            _, x, y, z0, r, h = g
            out.append({"cylinder": [round((x - 0.5) * 10000), round(z0 * 10000), round((y - 0.5) * 10000),
                                     round(r * 10000), round(h * 10000)]})
        else:
            _, x0, y0, z0, x1, y1, z1 = g
            out.append({"box": [round((x0 - 0.5) * 10000), round(z0 * 10000), round((y0 - 0.5) * 10000),
                                round((x1 - 0.5) * 10000), round(z1 * 10000), round((y1 - 0.5) * 10000)]})
    return out


def main() -> int:
    named = [a for a in sys.argv[1:] if a in ALL]
    by_name = {}
    for sheet in ALL:
        if not os.path.exists(os.path.join(RENDERS, sheet, "manifest.json")):
            continue
        tiles, slots = sheet_from_renders(sheet, write=not named or sheet in named)
        by_name.update({e["name"]: tiles[e["slot"]] for e in slots})
    scene(by_name)
    return 0


def sheet_from_renders(sheet, write=True):
    manifest = json.load(open(os.path.join(RENDERS, sheet, "manifest.json")))
    slots = manifest["slots"]
    n = max(e["slot"] for e in slots) + 1
    tiles, depths, geometry = [None] * n, [None] * n, {}
    for e in slots:
        png, npy = slot_files(sheet, e)
        if not os.path.exists(png):
            continue
        rgba = np.asarray(Image.open(png).convert("RGBA")).copy()
        pos = np.load(npy)
        tiles[e["slot"]] = rgba
        depths[e["slot"]] = depth_of(rgba, pos)
        geometry[e["slot"]] = geometry_units(e.get("geometry", []))
        if not write:
            continue
        missing = int(((rgba[..., 3] > 0) & (pos[..., 3] <= 0.5)).sum())
        print(f"  {e['slot']:2d} {e['name']:24s} {e.get('triangles', 0):4d} triangles; "
              f"{int((rgba[..., 3] > 0).sum())} px, {missing} edge px given a neighbour's depth")
    if write:
        F.write_sheet(sheet_file(sheet + ".png"), tiles)
        F.write_sheet(sheet_file("DEPTH_" + sheet + ".png"), depths)
        json.dump({"sheet": sheet, "names": {e["slot"]: e["name"] for e in slots}, "geometry": geometry},
                  open(sheet_file(sheet + ".geometry.json"), "w"), indent=1)
        preview(tiles, sheet)
        print("wrote", sheet, sum(t is not None for t in tiles), "sprites")
    return tiles, slots


def preview(tiles, sheet):
    """Every sprite on a pavement flag, 8 to a row, at the game's size and 3x."""
    flag = Image.fromarray(F.sheet_tile(sheet_file("blends_street_uk_01.png"), 0), "RGBA")
    rows = -(-len(tiles) // 8)
    img = Image.new("RGBA", (8 * 128, rows * 256), (46, 46, 46, 255))
    for i, t in enumerate(tiles):
        if t is None:
            continue
        cell = Image.new("RGBA", (128, 256), (0, 0, 0, 0))
        cell.alpha_composite(flag)
        cell.alpha_composite(Image.fromarray(t, "RGBA"))
        img.alpha_composite(cell, ((i % 8) * 128, (i // 8) * 256))
    img.save(os.path.join(ROOT, "renders", f"{sheet}_preview.png"))


def scene(tile):
    """A 1990s street corner: pavement flags, a zebra's beacons, a signal, a
    pillar box, both kiosks and a cone, drawn as the game draws a map (back
    to front)."""
    flag = F.sheet_tile(sheet_file("blends_street_uk_01.png"), 0)
    tar_path = sheet_file("blends_street_uk_02.png")
    tar = F.sheet_tile(tar_path, 0) if os.path.exists(tar_path) else flag
    n = 10
    floor = {(x, y): (tar if y >= 6 else flag) for x in range(n) for y in range(n)}
    things = {(1, 4): "pillar_box_S", (3, 4): "k6_kiosk_S", (4, 4): "kx100_kiosk_S", (6, 5): "belisha_beacon_lit",
              (8, 5): "belisha_beacon", (2, 5): "signal_red_S", (5, 7): "traffic_cone", (6, 7): "traffic_cone",
              (7, 5): "signal_green_E", (1, 2): "bench_S_00", (2, 2): "bench_S_10", (0, 5): "bollard_iron",
              (9, 5): "keep_left_S", (8, 2): "sign_speed_30_E", (5, 1): "place_odiham_S_00", (6, 1): "place_odiham_S_10",
              (3, 8): "barrier_lamp_N", (4, 8): "barrier_N", (9, 3): "sign_road_works_E", (0, 3): "bollard_concrete"}
    w, h = n * 128, n * 64 + 192
    img = Image.new("RGBA", (w, h), (40, 40, 40, 255))
    order = sorted(floor, key=lambda p: (p[0] + p[1], p[0]))
    ox = (n - 1) * 64
    for x, y in order:
        img.alpha_composite(Image.fromarray(floor[(x, y)], "RGBA"), ((x - y) * 64 + ox, (x + y) * 32))
    for x, y in order:
        if tile.get(things.get((x, y))) is not None:
            img.alpha_composite(Image.fromarray(tile[things[(x, y)]], "RGBA"), ((x - y) * 64 + ox, (x + y) * 32))
    img.save(os.path.join(ROOT, "renders", "street_furniture_scene.png"))


if __name__ == "__main__":
    sys.exit(main())
