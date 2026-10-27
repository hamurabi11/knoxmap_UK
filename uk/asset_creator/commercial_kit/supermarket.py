"""Supermarkets as they stood in 1993 (PLAN.md step 2): their wall kits.

    python supermarket.py [kit ...] [--preview]   (KnoxMap's Python)

Each kit is a whole wall set the way the housing kits are (wall_kit.kit: the wall
set, the 16 window-shaped pieces and the gable caps, painted into vanilla's
outlines, each slot taking its vanilla piece's properties), on a sheet
walls_commercial_uk_01_<kit>:

- superstore_brick   the 1990s superstore (the Tisco, Lainburys sort): a
                     red-brown multi facing brick in stretcher bond, a band of
                     blue engineering brick at the plinth, a soldier course
                     over its openings;
- superstore_buff    the same in a buff multi (the 1980s stores), a red band;
- cladding_juniper,  profiled steel cladding (trapezoidal ribs a fifth of a metre
  cladding_grey      apart) in the 1980s-90s colours, juniper green and
                     goosewing grey: the industrial units' (PLAN step 3; the user
                     dropped it from the supermarkets, 2026-10-20);
- superstore_brick_upper, superstore_buff_upper   the same bricks with no band, a
                     superstore's upper storey.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "housing_kit")))
import housing_kit as HK  # noqa: E402

WK, F = HK.WK, HK.F
PREVIEW = os.path.join(HERE, "preview")

# A course is wall_kit's (8 px); the plinth band is the bottom four courses.
PLINTH = slice(WK.STOREY - 32, WK.STOREY)

_RED_STORE = ([(132, 64, 52), (122, 58, 50), (142, 72, 58), (110, 56, 50), (150, 82, 62)],
              [0.3, 0.25, 0.2, 0.12, 0.13], (118, 108, 100))
_BUFF_STORE = ([(198, 168, 118), (186, 156, 108), (206, 178, 128), (172, 144, 100), (190, 160, 114)],
               [0.3, 0.25, 0.2, 0.12, 0.13], (160, 148, 128))
# Blue engineering brick: dark, smooth, a blue-purple cast (the same draws: the same
# number of tones and shares as the walls it bands, for HK.banded).
_BLUE_ENG = [(62, 60, 72), (56, 54, 66), (68, 66, 78), (50, 50, 62), (64, 62, 70)]
_RED_BAND = [(150, 76, 58), (140, 70, 54), (158, 84, 64), (130, 66, 52), (146, 80, 62)]


def cladding(rgb, rib=13, depth=0.22):
    """Profiled steel sheet, its ribs upright: a trapezoid each `rib` px (a
    crown, a sloping web either side, the pan), lit across the face; a lap
    every three sheets, a faint weathering streak down from the top."""
    def make(rng):
        h, w = WK.STOREY, WK.FACE
        x = np.arange(w)
        ph = (x % rib) / rib
        # crown 0-0.2, web 0.2-0.35 (in shadow), pan 0.35-0.85, web 0.85-1 (lit)
        shade = np.select([ph < 0.2, ph < 0.35, ph < 0.85], [1.0 + depth * 0.4, 1.0 - depth, 1.0],
                          1.0 + depth * 0.15)
        tex = np.ones((h, w, 3)) * np.array(rgb, float) * shade[None, :, None]
        tex *= 1 + rng.normal(0, 0.012, (h, 1, 1))
        streak = (rng.random(w) < 0.06)[None, :, None] * np.linspace(0.96, 1.0, h)[:, None, None]
        tex = np.where(streak > 0, tex * streak, tex)
        # the top flashing, one flat colour (review 2026-10-20: the ribs ran up into the
        # wall's thin top and showed as a comb of ticks)
        tex[:4] = np.array(rgb, float) * 0.82
        return tex
    return make


KITS = {
    # name: (texture maker, trim as wall_kit.MATERIALS: arches (a soldier course) and cills)
    "superstore_brick": (HK.banded(HK.multi_brick(*_RED_STORE, dark={1, 3}),
                                   HK.multi_brick(_BLUE_ENG, _RED_STORE[1], _RED_STORE[2], dark={1, 3}), PLINTH),
                         {"soldier": ((96, 48, 42), 8), "cill": ((92, 90, 96), 2)}),
    "superstore_buff": (HK.banded(HK.multi_brick(*_BUFF_STORE, dark={3}),
                                  HK.multi_brick(_RED_BAND, _BUFF_STORE[1], _BUFF_STORE[2], dark={3}), PLINTH),
                        {"soldier": ((142, 70, 54), 8), "cill": ((140, 76, 58), 2)}),
    # the same bricks without the plinth band, for a superstore's upper storey (user,
    # 2026-10-20: brick, not cladding; with the banded kit the band repeated halfway up)
    "superstore_brick_upper": (HK.multi_brick(*_RED_STORE, dark={1, 3}), {"cill": ((92, 90, 96), 2)}),
    "superstore_buff_upper": (HK.multi_brick(*_BUFF_STORE, dark={3}), {"cill": ((140, 76, 58), 2)}),
    # review (2026-10-20): the grey's fine even ribs shimmered at 1x - wider ribs, a
    # flat pan between, darker towards vanilla's greys
    "cladding_juniper": (cladding((70, 104, 84), rib=16), {"cill": ((60, 88, 72), 2)}),
    "cladding_grey": (cladding((128, 132, 134), rib=18, depth=0.16), {"cill": ((112, 116, 118), 2)}),
}


def sheet_of(kit: str) -> str:
    return f"walls_commercial_uk_01_{kit}"


def write_kit(name: str) -> str:
    WK.MATERIALS["_commercial_" + name] = KITS[name]
    tiles, _depths = WK.kit("_commercial_" + name)
    del WK.MATERIALS["_commercial_" + name]
    sheet = sheet_of(name)
    blank = np.zeros((256, 128, 4), np.uint8)
    F.write_sheet(WK.sheet_file(sheet + ".png"), [t if t is not None else blank for t in tiles])
    slots = [None] * WK.SLOTS
    for slot, vsheet, idx in WK.LAYOUT:
        slots[slot] = f"{vsheet}_{idx}"
    with open(WK.sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f)
    return sheet


def preview(names) -> None:
    """Each kit as a wall 6 squares long on its north face with a window in it, and a
    corner, beside vanilla's brick, so a reviewer sees it at the game's scale."""
    os.makedirs(PREVIEW, exist_ok=True)
    cw = 6 * 64 + 200
    img = Image.new("RGBA", (cw * len(names), 520), (96, 100, 96, 255))
    for c, name in enumerate(names):
        sheet = Image.open(WK.sheet_file(sheet_of(name) + ".png")).convert("RGBA")
        cols = sheet.width // 128

        def tile(k):
            r, q = divmod(k, cols)
            return sheet.crop((q * 128, r * 256, q * 128 + 128, r * 256 + 256))
        ox, oy = c * cw + 40, 40
        run = [1, 1, 9, 1, 1, 1]                         # N wall pieces (9: a window in it)
        for i, k in enumerate(run):
            img.alpha_composite(tile(k), (ox + 64 * i, oy + 32 * i))
        for i in range(3):                               # and a W wall coming back to the corner
            img.alpha_composite(tile(0), (ox - 64 * (i + 1) + 64, oy + 32 * (i + 1) - 32 * 2 + 64))
    img.save(os.path.join(PREVIEW, "supermarket_walls.png"))


def main(argv) -> int:
    names = [a for a in argv if not a.startswith("--")] or list(KITS)
    for n in names:
        print(write_kit(n))
    if "--preview" in argv:
        preview(names)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
