"""UK grass from the user's photos, in place of vanilla's three grass floors.

    python grass.py [--level L]   -> build/sheets/<group>/blends_natural_uk_01.png (rows: dark, medium, light)
                                     build/preview/grass_*.png

Vanilla paints ground grass with three floor sets in blends_natural_01
(dark 16, medium 32, light 48: Rules.txt darkgrass / medgrass / lightgrass,
colours 90 100 35, 117 117 47, 145 135 60) - olive, Kentucky in late summer -
and stands B42's grass tufts (e_newgrass_1) on it. These are the floors only:
- dark from `Leafy_Grass_Thick` (lush park grass; its near half, where the
  photo looks most nearly down), medium and light from `Leafy_Grass` (a short
  lawn or verge, top-down; other windows, each at its vanilla set's
  brightness); `Muddy_Grass` (worn grass and bare earth) kept for a worn surface;
- leaves masked by hue (brown, orange, yellow: red above green) and filled
  from the grass round them;
- as tarmac.py: cut to 64 px a metre, the slow tone dropped (no checkerboard
  from square to square) but the grain kept a little coarser (GRAIN_SIGMA:
  clumps of blades), worn patches kept inside each square;
- colour: brightness vanilla's for each set; the hue LEVEL of the way from
  vanilla's olive to the photo's green (flags.LEVEL: 2/3).
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

PX_PER_M = 700
GRAIN_SIGMA = 4.0
PATCH_MARGIN = 8
COLOURS = 256
SETS = {   # vanilla base: (photo, sample, worn patches)
    16: ("Leafy_Grass_Thick.jpeg", "near", False),
    32: ("Leafy_Grass.jpeg", "lawn", False),
    48: ("Leafy_Grass.jpeg", "lawn", False),   # light: the lawn again (Muddy_Grass reads as bare earth: kept for a worn surface)
}
SHEET = "blends_natural_uk_01"


def area(photo: str, sample: str) -> np.ndarray:
    cat = json.load(open(os.path.join(flags.ROOT, "reference", "catalogue.json"), encoding="utf-8"))
    box = next(p for p in cat["photos"] if p["file"] == photo)["samples"][sample]
    a = np.asarray(Image.open(os.path.join(flags.PICTURES, photo)).convert("RGB").crop(tuple(box)), float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = a.mean(-1)
    leaf = (r > g * 0.98) & (r - b > 30)                       # brown, orange, yellow leaves
    leaf |= lum > np.percentile(lum, 99)                        # pale seeds and litter
    leaf = ndimage.binary_dilation(leaf, iterations=3)
    keep = ~leaf
    fill = np.stack([ndimage.gaussian_filter(a[..., c] * keep, 10) for c in range(3)], -1)
    weight = ndimage.gaussian_filter(keep.astype(float), 10)[..., None]
    a = np.where(leaf[..., None], fill / np.maximum(weight, 1e-3), a)
    w, h = round(a.shape[1] / PX_PER_M * F.TEX), round(a.shape[0] / PX_PER_M * F.TEX)
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((w, h), Image.BOX), float)


def vanilla_full(base: int) -> list[np.ndarray]:
    van = os.path.join(flags.VANILLA, "blends_natural_01.png")
    return [F.sheet_tile(van, base + i) for i in (0, 5, 6, 7)]


def surface(base: int, level: float) -> list[np.ndarray]:
    photo, sample, worn = SETS[base]
    a = area(photo, sample)
    vmean = np.mean([t[F.DIAMOND][:, :3].astype(float).mean(0) for t in vanilla_full(base)], 0)
    slow = np.stack([ndimage.gaussian_filter(a[..., c], GRAIN_SIGMA) for c in range(3)], -1)
    grain = a - slow
    pmean = a.reshape(-1, 3).mean(0)
    target = vmean + (pmean - vmean) * level                    # hue between vanilla's and the photo's
    target *= vmean.mean() / max(target.mean(), 1)              # brightness vanilla's
    gain = 1.0                                                  # grain as the photo has it
    tex = target + grain * gain
    if worn:
        patch = slow - slow.reshape(-1, 3).mean(0)
        tex = tex + patch * level
    h, w = tex.shape[:2]
    rng = np.random.default_rng(base + 1)
    tiles = []
    yy, xx = np.mgrid[0:F.TEX, 0:F.TEX]
    edge = np.minimum(np.minimum(yy, xx), np.minimum(F.TEX - 1 - yy, F.TEX - 1 - xx))
    fade = np.clip((edge - 1) / PATCH_MARGIN, 0, 1)[..., None]
    for k in range(4):
        y, x = rng.integers(0, h - F.TEX + 1), rng.integers(0, w - F.TEX + 1)
        t = tex[y:y + F.TEX, x:x + F.TEX]
        if worn:                                                # a patch stays inside its square
            plain = target + grain[y:y + F.TEX, x:x + F.TEX] * gain
            t = plain + (t - plain) * fade
        t = np.rot90(t, k)
        tiles.append(t[:, ::-1] if k % 2 else t)
    both = np.clip(np.concatenate(tiles, 1), 0, 255).astype(np.uint8)
    both = np.asarray(Image.fromarray(both).quantize(COLOURS, method=Image.Quantize.MEDIANCUT).convert("RGB"))
    return [F.project(t) for t in np.split(both, 4, 1)]


HI = 128                     # texels a square for blades: neighbouring screen pixels are diagonal texels
BLADES = 5200                # blades a square
BLADE_LEN = (2, 6)           # screen pixels tall
BLADE_SPREAD = 0.55          # of the photo's blade-to-blade colour spread (vanilla's pixel contrast)


def blades(base: int, level: float, seed: int) -> np.ndarray:
    """Grass as the game draws it (user, 2026-10-12: ours had no blades):
    blades 1 px wide standing up the screen, dark at the root, light at the
    tip, thick enough to hide the ground. Drawn on a HI x HI top-down
    texture, where one pixel up the screen is two texels along (-1, -1) and
    one across is (+1, -1), so a blade is one pixel wide; wrapped, so tiles
    meet without a seam. Colours: the photo's green pixels (no leaves, no
    earth), LEVEL of the way from vanilla's olive to the photo's hue, the
    whole set to vanilla's brightness for this set."""
    photo, sample, _worn = SETS[base]
    a = area(photo, sample).reshape(-1, 3)
    green = a[(a[:, 1] > a[:, 0]) & (a[:, 1] > a[:, 2])]
    if len(green) < 50:
        green = a
    vmean = np.mean([t[F.DIAMOND][:, :3].astype(float).mean(0) for t in vanilla_full(base)], 0)
    pmean = green.mean(0)
    target = vmean + (pmean - vmean) * level
    target *= vmean.mean() / max(target.mean(), 1)
    dev = (green - pmean) * (0.6 + 0.4 * level) * BLADE_SPREAD
    grey = dev.mean(-1, keepdims=True)
    dev = grey + (dev - grey) * 0.6                            # the photo's grey-green blades less grey
    rng = np.random.default_rng(seed)
    tex = np.tile(target * 0.72, (HI, HI, 1))                  # the shade between the blades
    for _ in range(BLADES):
        x, y = rng.integers(0, HI, 2)
        n = rng.integers(*BLADE_LEN)
        col = target + dev[rng.integers(len(dev))]
        for k in range(n):                                    # root (dark) to tip (light)
            shade = 0.80 + 0.36 * k / max(1, n - 1)
            for dd in (0, 1):                                 # a pixel up is two texels: fill both
                tex[(y - 2 * k - dd) % HI, (x - 2 * k - dd) % HI] = col * shade
    tex *= target.mean() / tex.reshape(-1, 3).mean(0).mean()
    return tex


def project_hi(tex: np.ndarray) -> np.ndarray:
    out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
    tx = np.clip((F.WX * HI).astype(int), 0, HI - 1)
    ty = np.clip((F.WY * HI).astype(int), 0, HI - 1)
    out[..., :3] = tex[ty, tx]
    out[..., 3] = np.where(F.DIAMOND, 255, 0)
    return out


def blade_surface(base: int, level: float) -> list[np.ndarray]:
    texs = [blades(base, level, base * 10 + k) for k in range(4)]
    both = np.clip(np.concatenate(texs, 1), 0, 255).astype(np.uint8)
    both = np.asarray(Image.fromarray(both).quantize(COLOURS, method=Image.Quantize.MEDIANCUT).convert("RGB"))
    return [project_hi(t) for t in np.split(both, 4, 1)]


def main() -> None:
    level = float(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else flags.LEVEL
    sheet_tiles, rows = [], []
    for base in SETS:
        full = blade_surface(base, level)
        sheet_tiles += F.floor_set(full, second_strips=True)
        van = vanilla_full(base)
        print(flags.stats(f"ours {base}", full[0]), "|", flags.stats("vanilla", van[0]))
        rows.append((F.patch(full, 5), F.patch(van, 5)))
    F.write_sheet(sheet_file(SHEET + ".png"), sheet_tiles)
    w, h = rows[0][0].size
    out = Image.new("RGBA", (w * 2 + 20, h * 3 + 40), (40, 40, 40, 255))
    for r, (ours, van) in enumerate(rows):
        out.alpha_composite(ours, (0, r * (h + 20)))
        out.alpha_composite(van, (w + 20, r * (h + 20)))
    out.save(os.path.join(flags.BUILD, "preview", "grass_vs_vanilla.png"))
    print("wrote", SHEET, "at level", round(level, 3))


if __name__ == "__main__":
    main()
