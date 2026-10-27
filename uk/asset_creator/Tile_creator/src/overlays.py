"""Edge overlays in our surfaces: the 88 shapes vanilla's kerbs are laid with.

    python overlays.py   -> build/sheets/<group>/street_curbs_uk_01_blend_pavement.png
                            build/sheets/<group>/street_curbs_uk_01_blend_street.png

Where a kerb crosses a square at an angle, the square is one floor and the
part beyond the kerb is painted over by an overlay of the ground there
(Building_types/src/streets/kerbs.py). Vanilla's overlay sheets
(street_curbs_01_blend_*: 88 shapes, 8 x 11) come in its own grounds only;
beside our pavement kerbs.py laid vanilla's pale concrete (`blend_gravel`, a
stand-in) and over part of a pavement square vanilla's street3 grey
(`blend_street`). These are the same 88 shapes in our flags, our tarmac and our grasses.

Each shape is where vanilla's cut runs, measured from its sheet and smoothed
to its line (a blur of SMOOTH px, then a half), so it meets vanilla's kerb
pieces exactly; the picture in it is ours (the full tile of our surface, so a
flag overlay's joints run on from the squares beside it).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
from scipy import ndimage

import flags
import floor_tiles as F
from sheet_paths import sheet_file

SMOOTH = 1.0
# our sheet: (vanilla sheet whose shapes it takes, our floor sheet whose full tile fills it)
OVERLAYS = {
    "street_curbs_uk_01_blend_pavement": ("street_curbs_01_blend_gravel", "blends_street_uk_01"),
    "street_curbs_uk_01_blend_street": ("street_curbs_01_blend_street", "blends_street_uk_02"),
    "street_curbs_uk_01_blend_dark_grass": ("street_curbs_01_blend_dark_grass", ("blends_natural_uk_01", 0)),
    "street_curbs_uk_01_blend_medium_grass": ("street_curbs_01_blend_medium_grass", ("blends_natural_uk_01", 16)),
    "street_curbs_uk_01_blend_light_grass": ("street_curbs_01_blend_light_grass", ("blends_natural_uk_01", 32)),
}
N = 88


def shapes(vanilla_sheet: str) -> list[np.ndarray]:
    a = np.asarray(Image.open(os.path.join(flags.VANILLA, vanilla_sheet + ".png")).convert("RGBA"))[..., 3]
    out = []
    for i in range(N):
        r, c = divmod(i, 8)
        m = a[r * 256:(r + 1) * 256, c * 128:(c + 1) * 128] > 0
        out.append((ndimage.gaussian_filter(m.astype(float), SMOOTH) > 0.5) & F.DIAMOND)
    return out


def main() -> None:
    for ours, (vanilla, floor) in OVERLAYS.items():
        floor, slot = floor if isinstance(floor, tuple) else (floor, 0)
        full = F.sheet_tile(sheet_file(floor + ".png"), slot)
        tiles = []
        for m in shapes(vanilla):
            t = full.copy()
            t[..., 3] = np.where(m, 255, 0)
            tiles.append(t if m.any() else None)
        F.write_sheet(sheet_file(ours + ".png"), tiles)
        print("wrote", ours, sum(t is not None for t in tiles), "shapes")


if __name__ == "__main__":
    main()
