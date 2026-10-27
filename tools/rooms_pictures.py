"""Pictures of rooms_uk's example houses in the game's own tiles (ROOMS.md).

    python tools/rooms_pictures.py [out_dir] [--map output/Odiham_town_graph] [--only name]

Each example is solved, turned into Build's own Building (rooms_uk.to_building:
Build's furnishing), written as a .tbx with a UK style from the map's palette,
read back into tiles (buildtiles, as the Rust compiler does) and drawn
(render_lots.draw_squares). No map is changed.

One sheet per house: the outside with its roof, then each floor as the game
shows it from inside (the storeys above hidden, the near south and east walls
cut away), and the plan.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from PIL import Image, ImageDraw  # noqa: E402

import render_lots  # noqa: E402
from knoxbuild import buildtiles, regional_uk  # noqa: E402
from knoxbuild import rooms_uk as R  # noqa: E402
from knoxbuild.tbx import render_tbx  # noqa: E402

# Each example's look (palette styles, as export_regional gives these sub-types)
STYLE = {"vic_mid_terrace": "uk_victorian_terrace", "iw_semi": "uk_interwar_30s",
         "council_end": "uk_council_70s", "starter_mid": "uk_terrace_80s",
         "detached_garage": "uk_estate_80s", "bungalow": "uk_bungalow_70s",
         "l_detached": "uk_estate_80s", "semi_north": "uk_interwar_tudor",
         "shop_home": "uk_victorian_terrace", "shop_flats_3": "uk_georgian_town",
         "shop_two_flats": "uk_estate_80s", "flats_four_block": "uk_council_pw",
         "flats_80s_block": "uk_flats_80s", "flats_conversion": "uk_victorian_terrace"}
PAD = 3          # squares of ground round the house
GRASS = "blends_natural_01_16"


def tiles_of(lay, style):
    """{(x, y, z): [tile names]} for the house, PAD squares in from the corner."""
    building = R.to_building(lay, seed=7)
    style = dict(style, street_side=lay.house.street)
    b = buildtiles.parse_tbx(render_tbx(building, "rooms_uk", style))
    squares = {}
    for z in range(len(b.floors) + 1):
        try:
            got = buildtiles.level_tiles(b, z)
        except (IndexError, KeyError):
            continue
        for (x, y), names in got.items():
            squares[(x + PAD, y + PAD, z)] = list(names)
    return squares


def view(squares, w, h, level=None):
    """A copy for one view: level=None is the outside; else that floor from
    inside, the floors above and its roof gone, the near outside walls cut."""
    out = {}
    W, H = w + 2 * PAD, h + 2 * PAD
    for x in range(W):
        for y in range(H):
            out[(x, y, 0)] = [GRASS]
    for (x, y, z), names in squares.items():
        if level is not None:
            if z > level:
                continue
            # the cutaway: the walls on the squares just past the house's
            # south and east sides are its near walls
            hx, hy = x - PAD, y - PAD
            if z == level and (hx >= w or hy >= h):
                continue
            if z == level:
                names = [n for n in names if not n.startswith("roofs_")]
        key = (x, y, z)
        out[key] = (out.get(key, []) if z == 0 else []) + names
    return out


def _crop(im, margin=8):
    """Trim the empty background round a view."""
    bg = im.getpixel((1, 1))
    mask = Image.new("RGBA", im.size, bg)
    from PIL import ImageChops
    box = ImageChops.difference(im, mask).convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    if not box:
        return im
    return im.crop((max(0, box[0] - margin), max(0, box[1] - margin),
                    min(im.width, box[2] + margin), min(im.height, box[3] + margin)))


def sheet(name, lay, style, out_dir, scale):
    house = lay.house
    sq = tiles_of(lay, style)
    W, H = house.width + 2 * PAD, house.height + 2 * PAD
    top = max(z for (_, _, z) in sq)
    shots = [("outside", render_lots.draw_squares(view(sq, house.width, house.height), W, H, top, scale))]
    for lvl in range(len(lay.storeys)):
        label = ["ground floor", "first floor", "second floor", "third floor"][lvl]
        shots.append((label, render_lots.draw_squares(view(sq, house.width, house.height, lvl),
                                                      W, H, lvl, scale)))
    shots = [(label, _crop(im)) for label, im in shots]
    plan_path = os.path.join(out_dir, f"{name}_plan.png")
    R.render(lay, plan_path, name)
    plan = Image.open(plan_path).convert("RGBA")
    pics = [im for _, im in shots]
    gap = 12
    wide = sum(p.width for p in pics) + gap * (len(pics) - 1)
    out = Image.new("RGBA", (max(wide, plan.width), max(p.height for p in pics) + 24 + plan.height + 10),
                    (24, 26, 28, 255))
    d = ImageDraw.Draw(out)
    x = 0
    for (label, p) in shots:
        out.alpha_composite(p, (x, 22))
        d.text((x + 6, 4), f"{name}: {label} ({style['name']})", fill=(230, 230, 230))
        x += p.width + gap
    out.alpha_composite(plan, (0, 24 + max(p.height for p in pics) + 10))
    path = os.path.join(out_dir, f"{name}_interiors.png")
    R._save(out.convert("RGB"), path)
    return path


def real(a, styles):
    """Real houses from a built map (rooms_uk.map_houses: the street taken as S)."""
    import csv
    import random
    mname = os.path.basename(os.path.abspath(a.map))
    with open(os.path.join(a.map, f"{mname}_placements.csv"), encoding="utf-8") as f:
        style_of = {r["name"]: r["style"] for r in csv.DictReader(f)}
    houses = [x for x in R.map_houses(a.map)
              if max(sum(map(sum, m)) for m in x[0].masks) >= a.min_m2
              and (not a.storeys or len(x[0].masks) == a.storeys)]
    random.Random(a.seed).shuffle(houses)
    for house, prog, name in houses[:a.real]:
        lay = R.solve(house, prog, workers=8)
        tag = "".join(c if c.isalnum() else "_" for c in name)[:40]
        if lay is None:
            print(name, prog, "no layout")
            continue
        style = styles.get(style_of.get(name), styles["uk_estate_80s"])
        print(name, prog, "->", sheet(f"real_{tag}", lay, style, a.out_dir, a.scale))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=os.path.join("output", "rooms_demo"))
    ap.add_argument("--map", default=os.path.join("output", "Odiham_town_graph"))
    ap.add_argument("--only")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--real", type=int, default=0,
                    help="draw this many real houses from --map instead (those with a floor of --min-m2 or more)")
    ap.add_argument("--min-m2", type=int, default=0)
    ap.add_argument("--storeys", type=int, default=0, help="only houses with this many storeys")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    overlay = regional_uk.load(a.map)
    regional_uk.apply_palette(overlay)
    styles = overlay["_styles"]
    if a.real:
        return real(a, styles)
    for name, house, prog in R.demo_houses():
        if a.only and name != a.only:
            continue
        lay = R.solve(house, prog)
        if lay is None:
            print(name, "no layout")
            continue
        print(name, "->", sheet(name, lay, styles[STYLE[name]], a.out_dir, a.scale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
