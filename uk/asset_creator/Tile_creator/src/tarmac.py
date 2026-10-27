"""UK tarmac from the user's photos: carriageway (street2) and worn service lanes (street4).

    python tarmac.py [--level L]   -> build/sheets/<group>/blends_street_uk_02.png (clean)
                                      build/sheets/<group>/blends_street_uk_03.png (worn)
                                      build/preview/tarmac_*.png

Each from one clean area of a photo (reference/catalogue.json `samples`:
`Walking_Path_Asphalt` "clean", `Leaffy_Asphalt_Path` "worn"):
- leaves masked (saturated pixels) and filled from the tarmac round them;
- cut to 64 px a metre (PX_PER_M in the photo: about 750, from the chippings
  and the leaves) by area-averaging;
- split into grain (what is left after a blur of GRAIN_SIGMA texels) and the
  slow tone under it. A continuous surface cannot carry tone differences from
  square to square (a checkerboard; flags can, their joints hide it), so the
  slow tone is dropped - except, on worn tarmac, its dark stains, kept whole
  inside a square (STAIN_MARGIN from its edges) so no square cuts one off;
- grain size and colour at LEVEL between vanilla's (street2 102 grey, std 8.4;
  street4 122, std 9.2; both colourless) and the photo's (flags.LEVEL: 2/3);
  brightness kept at vanilla's, so kerbs and road markings sit as before;
- four variants from four windows of the area, each turned or mirrored.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import flags
import floor_tiles as F
from sheet_paths import sheet_file

PX_PER_M = 750
GRAIN_SIGMA = 2.0            # texels: tone slower than this is not grain
STAIN_MARGIN = 6             # texels a stain keeps from a square's edge
STAIN_DARK = 6.0             # how much darker than the tarmac a stain must be
COLOURS = 256

SURFACES = {
    # sheet: (photo, sample, vanilla mean grey, vanilla grain std, stains)
    "blends_street_uk_02": ("Walking_Path_Asphalt.jpeg", "clean", 102.0, 8.4, False),
    "blends_street_uk_03": ("Leaffy_Asphalt_Path.jpeg", "worn", 122.0, 9.2, True),
}


def area(photo: str, sample: str) -> np.ndarray:
    """The sample's area at 64 px a metre, leaves filled in."""
    cat = json.load(open(os.path.join(flags.ROOT, "reference", "catalogue.json"), encoding="utf-8"))
    box = next(p["samples"][sample] for p in cat["photos"] if p["file"] == photo)
    a = np.asarray(Image.open(os.path.join(flags.PICTURES, photo)).convert("RGB").crop(tuple(box)), float)
    hi, lo = a.max(-1), a.min(-1)
    leaf = ndimage.binary_dilation((hi - lo) > 0.22 * np.maximum(hi, 1) + 8, iterations=4)
    keep = ~leaf
    fill = np.stack([ndimage.gaussian_filter(a[..., c] * keep, 12) for c in range(3)], -1)
    weight = ndimage.gaussian_filter(keep.astype(float), 12)[..., None]
    a = np.where(leaf[..., None], fill / np.maximum(weight, 1e-3), a)
    w, h = round(a.shape[1] / PX_PER_M * F.TEX), round(a.shape[0] / PX_PER_M * F.TEX)
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((w, h), Image.BOX), float)


def surface(sheet: str, level: float) -> list[np.ndarray]:
    photo, sample, grey, vstd, stains = SURFACES[sheet]
    a = area(photo, sample)
    slow = np.stack([ndimage.gaussian_filter(a[..., c], GRAIN_SIGMA) for c in range(3)], -1)
    grain = a - slow
    lum = grain.mean(-1, keepdims=True)
    cast = grain - lum                               # colour in the grain
    std = lum.std() or 1.0
    amp = (vstd + (std - vstd) * level) / std        # grain size: vanilla's -> the photo's
    tex = grey + lum * amp + cast * level
    if stains:
        dark = slow.mean(-1) - np.median(slow.mean(-1))
        tex = tex + (np.minimum(dark + STAIN_DARK, 0) * level)[..., None]
    h, w = tex.shape[:2]
    rng = np.random.default_rng(3)
    tiles = []
    for k in range(4):
        y, x = rng.integers(0, h - F.TEX + 1), rng.integers(0, w - F.TEX + 1)
        t = tex[y:y + F.TEX, x:x + F.TEX]
        if stains:                                   # a stain stays inside its square
            yy, xx = np.mgrid[0:F.TEX, 0:F.TEX]
            edge = np.minimum(np.minimum(yy, xx), np.minimum(F.TEX - 1 - yy, F.TEX - 1 - xx))
            fade = np.clip((edge - 1) / STAIN_MARGIN, 0, 1)[..., None]
            plain = grey + lum[y:y + F.TEX, x:x + F.TEX] * amp + cast[y:y + F.TEX, x:x + F.TEX] * level
            t = plain + (t - plain) * fade
        t = np.rot90(t, k)
        tiles.append(t[:, ::-1] if k % 2 else t)
    both = np.clip(np.concatenate(tiles, 1), 0, 255).astype(np.uint8)
    if level < 1:
        both = np.asarray(Image.fromarray(both).quantize(COLOURS, method=Image.Quantize.MEDIANCUT).convert("RGB"))
    return [F.project(t) for t in np.split(both, 4, 1)]


def main() -> None:
    level = float(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else flags.LEVEL
    os.makedirs(os.path.join(flags.BUILD, "preview"), exist_ok=True)
    van = os.path.join(flags.VANILLA, "blends_street_01.png")
    for sheet, base in (("blends_street_uk_02", 80), ("blends_street_uk_03", 32)):
        full = surface(sheet, level)
        F.write_sheet(sheet_file(sheet + ".png"), F.floor_set(full))
        vanilla = [F.sheet_tile(van, base + i) for i in (0, 5, 6, 7)]
        for i, t in enumerate(full[:1]):
            print(flags.stats(f"{sheet} {i}", t))
        print(flags.stats(f"vanilla {base}", vanilla[0]))
        pics = [F.patch(full), F.patch(vanilla)]
        w, h = pics[0].size
        side = Image.new("RGBA", (w * 2 + 20, h), (40, 40, 40, 255))
        for k, p in enumerate(pics):
            side.alpha_composite(p, (k * (w + 20), 0))
        side.save(os.path.join(flags.BUILD, "preview", f"tarmac_{sheet[-2:]}.png"))
        F.patch(full, 3).resize((3 * 128 * 3, 3 * 64 * 3), Image.NEAREST).save(
            os.path.join(flags.BUILD, "preview", f"tarmac_{sheet[-2:]}_zoom.png"))
        print("wrote", sheet, "at level", round(level, 3))


if __name__ == "__main__":
    main()
