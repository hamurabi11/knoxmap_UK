"""UK dirt footpath from the user's photo: packed earth and grit (OSM footways and tracks).

    python dirt.py [--level L]   -> build/sheets/<group>/blends_natural_uk_02.png
                                    build/preview/dirt.png, dirt_zoom.png

From one clean area of `Muddy_Path.jpeg` (reference/catalogue.json sample "path"),
as tarmac.py does tarmac:
- leaves (saturated pixels) and twigs (thin dark or pale streaks against the
  earth) masked and filled from the earth round them;
- cut to 64 px a metre (PX_PER_M: pebbles of 10-20 mm are 11-22 px in the photo);
- split into grain (finer than GRAIN_SIGMA texels: the grit and pebbles) and the
  slow tone under it, which is dropped (no square-to-square checkerboard);
- colour and grain size at LEVEL between vanilla's dirt (blends_natural_01_64)
  and the photo's (flags.LEVEL, 2/3), its brightness a quarter of the way to the photo's (BRIGHTNESS);
- four variants from four windows, each turned or mirrored, and vanilla's 16
  slots from them (floor_tiles.floor_set: the halves and blend strips the
  Blends.txt rules lay where dirt meets grass).
It stands in for vanilla's dirt (pack.FLOOR_REPLACES -> register.py), so every
dirt square of a map compiled afterwards - footpaths (kerbs.repaint_paths),
tracks, bare ground - takes it.
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

SHEET = "blends_natural_uk_02"
PHOTO, SAMPLE = "Muddy_Path.jpeg", "path"
PX_PER_M = 1100
GRAIN_SIGMA = 2.5            # texels: tone slower than this is not grain
VANILLA_BASE = 64            # blends_natural_01_64: vanilla's dirt
BRIGHTNESS = 0.25            # brightness a quarter of the way from vanilla's dark dirt to the photo's pale grit
                             # (user, 2026-10-18: "a bit darker" than halfway, for the railway bed)
                             # (user's choice 2026-10-13, of 2/3, vanilla's and halfway: build/preview/dirt_choice.png)
COLOURS = 256


def area() -> np.ndarray:
    """The sample's area at 64 px a metre, leaves and twigs filled in."""
    cat = json.load(open(os.path.join(flags.ROOT, "reference", "catalogue.json"), encoding="utf-8"))
    box = next(p["samples"][SAMPLE] for p in cat["photos"] if p["file"] == PHOTO)
    a = np.asarray(Image.open(os.path.join(flags.PICTURES, PHOTO)).convert("RGB").crop(tuple(box)), float)
    hi, lo = a.max(-1), a.min(-1)
    lum = a.mean(-1)
    leaf = (hi - lo) > 0.25 * np.maximum(hi, 1) + 10
    # A twig: a streak much darker or paler than the earth round it, longer
    # than a pebble (opened along its length, a pebble goes, a twig stays).
    local = ndimage.uniform_filter(lum, 41)
    odd = np.abs(lum - local) > 28
    twig = np.zeros_like(odd)
    for k in range(4):
        line = np.zeros((25, 25), bool)
        if k == 0:
            line[12, :] = True
        elif k == 1:
            line[:, 12] = True
        elif k == 2:
            np.fill_diagonal(line, True)
        else:
            np.fill_diagonal(line[:, ::-1], True)
        twig |= ndimage.binary_opening(odd, structure=line)
    bad = ndimage.binary_dilation(leaf | twig, iterations=5)
    keep = ~bad
    fill = np.stack([ndimage.gaussian_filter(a[..., c] * keep, 14) for c in range(3)], -1)
    weight = ndimage.gaussian_filter(keep.astype(float), 14)[..., None]
    a = np.where(bad[..., None], fill / np.maximum(weight, 1e-3), a)
    w, h = round(a.shape[1] / PX_PER_M * F.TEX), round(a.shape[0] / PX_PER_M * F.TEX)
    a = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((w, h), Image.BOX), float)
    # A photo is about a metre square: mirrored round itself, room for four windows.
    a = np.concatenate([a, a[:, ::-1]], 1)
    return np.concatenate([a, a[::-1]], 0)


def vanilla_dirt() -> tuple[np.ndarray, float]:
    """Vanilla's dirt: its mean colour and its grain size (luminance std) on the diamond."""
    t = F.sheet_tile(os.path.join(flags.VANILLA, "blends_natural_01.png"), VANILLA_BASE).astype(float)
    m = t[..., 3] > 0
    rgb = t[..., :3][m]
    return rgb.mean(0), float(rgb.mean(-1).std())


def surface(level: float) -> list[np.ndarray]:
    a = area()
    slow = np.stack([ndimage.gaussian_filter(a[..., c], GRAIN_SIGMA) for c in range(3)], -1)
    grain = a - slow
    lum = grain.mean(-1, keepdims=True)
    cast = grain - lum                               # colour in the grain (the odd red or blue pebble)
    v_rgb, v_std = vanilla_dirt()
    p_rgb = slow.reshape(-1, 3).mean(0)
    base = v_rgb + (p_rgb - v_rgb) * level           # colour: vanilla's -> the photo's
    target = v_rgb.mean() + (p_rgb.mean() - v_rgb.mean()) * BRIGHTNESS
    base = base * target / base.mean()               # ...at its own brightness (BRIGHTNESS)
    std = lum.std() or 1.0
    amp = (v_std + (std - v_std) * level) / std      # grain: vanilla's -> the photo's
    tex = base + lum * amp + cast * level
    h, w = tex.shape[:2]
    rng = np.random.default_rng(5)
    tiles = []
    for k in range(4):
        y, x = rng.integers(0, h - F.TEX + 1), rng.integers(0, w - F.TEX + 1)
        t = np.rot90(tex[y:y + F.TEX, x:x + F.TEX], k)
        tiles.append(t[:, ::-1] if k % 2 else t)
    both = np.clip(np.concatenate(tiles, 1), 0, 255).astype(np.uint8)
    if level < 1:
        both = np.asarray(Image.fromarray(both).quantize(COLOURS, method=Image.Quantize.MEDIANCUT).convert("RGB"))
    return [F.project(t) for t in np.split(both, 4, 1)]


def main() -> None:
    level = float(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else flags.LEVEL
    os.makedirs(os.path.join(flags.BUILD, "preview"), exist_ok=True)
    full = surface(level)
    F.write_sheet(sheet_file(SHEET + ".png"), F.floor_set(full))
    van = os.path.join(flags.VANILLA, "blends_natural_01.png")
    vanilla = [F.sheet_tile(van, VANILLA_BASE + i) for i in (0, 5, 6, 7)]
    print(flags.stats("dirt 0", full[0]))
    print(flags.stats(f"vanilla {VANILLA_BASE}", vanilla[0]))
    pics = [F.patch(full), F.patch(vanilla)]
    w, h = pics[0].size
    side = Image.new("RGBA", (w * 2 + 20, h), (40, 40, 40, 255))
    for k, p in enumerate(pics):
        side.alpha_composite(p, (k * (w + 20), 0))
    side.save(os.path.join(flags.BUILD, "preview", "dirt.png"))
    F.patch(full, 3).resize((3 * 128 * 3, 3 * 64 * 3), Image.NEAREST).save(
        os.path.join(flags.BUILD, "preview", "dirt_zoom.png"))
    print("wrote", SHEET, "at level", round(level, 3))


if __name__ == "__main__":
    main()
