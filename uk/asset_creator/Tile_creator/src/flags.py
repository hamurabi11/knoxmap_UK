"""UK pavement flags from the user's photos, in vanilla's style: the first floor set.

    python flags.py            -> build/sheets/<group>/blends_street_uk_01.png
                                  build/preview/flags_*.png

Each flag's face comes from a photo sample (reference/catalogue.json:
`UK_Pavement_Urban` A-L but C, `Kerb_Top_Example` M-O; E is the cracked flag),
cut to vanilla's level (APPROACH.md section 6):
- 0.5 m flags, 2 x 2 to a square, every flag whole (UK 600 mm flags rounded
  so joints sit on the tile grid and every variant meets every other; a
  staggered bond put the same flag across every tile edge, a regular stripe);
- each face area-averaged down to 32 x 32 texels (64 px a metre: about
  1/15 of the photo), its own detail scaled to FACE_STD, its tone pulled
  towards TARGET (vanilla's `street` slab is 132 129 120) with the photos'
  flag-to-flag variety kept at TONE_KEEP;
- joints drawn, not taken from the photo: a pale mortar texel on each flag's
  north and west edge, a shadow texel on its south and east edge;
- colours quantised over the whole set to COLOURS.

LEVEL (--level, 0-1; 2/3 by default, the user's choice) moves every one of those from vanilla's style (0) to the
photos at the game's resolution (1): the flags' own tone differences, colour
cast and grain kept whole, more colours, joints thin and dark as in the
photos. The resolution is the same at every level (64 px a metre is all a
floor square holds), and so is the overall brightness (the photos' is the
camera's exposure, not the stone's). --compare draws levels 0, 1/3, 2/3, 1.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

import floor_tiles as F
from sheet_paths import sheet_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PICTURES = os.path.join(ROOT, "reference", "Pictures")
BUILD = os.path.join(ROOT, "build")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # asset_creator: acpaths
import acpaths  # noqa: E402
VANILLA = acpaths.TOOLS_TILES
SHEET = "blends_street_uk_01"

FLAG = 32                    # texels a flag side (0.5 m)
TARGET = np.array([128.0, 127.0, 121.0])   # set mean; a shade greyer than vanilla's slab
TONE_KEEP = 0.35             # share of each flag's own tone difference kept
CHROMA_KEEP = 0.5            # share of each flag's colour cast kept
FACE_STD = 4.0               # detail within a face, a channel (vanilla 2-13)
MORTAR = 1.18                # joint brightness against the mean
SHADOW = 0.80
COLOURS = 160
LEVEL = 2 / 3                # the user's choice (2026-10-12): between vanilla's style and the photos
INSET = 0.15                 # share of a sample box's shorter side left out
# The same at LEVEL 1 (the photos): their own tone, cast and grain, thin dark
# joints, colours left alone.
REAL = dict(TONE_KEEP=1.0, CHROMA_KEEP=1.0, MORTAR=0.80, SHADOW=0.72, COLOURS=4096)


def at(name: str, level: float) -> float:
    """A setting at a LEVEL between vanilla's style (0) and the photos (1)."""
    a, b = globals()[name], REAL[name]
    if name == "COLOURS":                       # geometric: 160, 469 (256), 1376 (256), all
        return a * (b / a) ** level
    return a + (b - a) * level


def samples() -> dict[str, Image.Image]:
    cat = json.load(open(os.path.join(ROOT, "reference", "catalogue.json"), encoding="utf-8"))
    out = {}
    for ph in cat["photos"]:
        if "samples" not in ph:
            continue
        im = Image.open(os.path.join(PICTURES, ph["file"])).convert("RGB")
        for k, (x0, y0, x1, y1) in ph["samples"].items():
            # A centred square INSET inside the box: no stretching a long crop
            # into a square (it smeared grain into streaks) and no joint edges.
            side = min(x1 - x0, y1 - y0) * (1 - INSET)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            out[k] = im.crop(tuple(round(v) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)))
    return out


def face(im: Image.Image, level: float = 0.0) -> np.ndarray:
    """A flag face cut to FLAG x FLAG texels, vanilla's level of detail."""
    small = np.asarray(im.resize((FLAG, FLAG), Image.BOX), float)
    mean = small.reshape(-1, 3).mean(0)
    blur = np.asarray(Image.fromarray(small.astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)), float)
    detail = small - blur                       # grain and spots
    mottle = blur - mean                        # soft stains across the face
    d = np.concatenate([detail, mottle]).std() or 1.0
    grey = mean.mean()
    tone = (mean - grey) * at("CHROMA_KEEP", level) + grey   # colour cast reduced
    scale = FACE_STD / d + (1 - FACE_STD / d) * level         # the photo's own grain at 1
    return tone + (detail + (0.6 + 0.4 * level) * mottle) * scale


def set_tone(faces: dict[str, np.ndarray], level: float = 0.0) -> dict[str, np.ndarray]:
    """Means pulled towards TARGET, TONE_KEEP of their differences kept."""
    overall = np.mean([f.reshape(-1, 3).mean(0) for f in faces.values()], 0)
    keep = at("TONE_KEEP", level)
    return {k: TARGET + keep * (f - overall) for k, f in faces.items()}


def compose(faces: dict[str, np.ndarray], picks: list[str], rng, level: float = 0.0) -> np.ndarray:
    """One square: four whole flags, each a face turned at random."""
    tex = np.zeros((F.TEX, F.TEX, 3))
    kind = np.zeros((F.TEX, F.TEX), int)        # 0 face, 1 mortar, 2 shadow
    for k, name in enumerate(picks):
        v0, u0 = divmod(k, 2)
        v0, u0 = v0 * FLAG, u0 * FLAG
        tex[v0:v0 + FLAG, u0:u0 + FLAG] = np.rot90(faces[name], rng.integers(4))
        kind[v0:v0 + FLAG, u0 + FLAG - 1] = 2
        kind[v0 + FLAG - 1, u0:u0 + FLAG] = 2
        kind[v0:v0 + FLAG, u0] = 1
        kind[v0, u0:u0 + FLAG] = 1
    mean = tex.reshape(-1, 3).mean(0)
    tex[kind == 1] = mean * at("MORTAR", level)
    tex[kind == 2] *= at("SHADOW", level)
    return tex


def quantise(texs: list[np.ndarray], level: float = 0.0) -> list[np.ndarray]:
    both = np.clip(np.concatenate(texs, 1), 0, 255).astype(np.uint8)
    if level >= 1:
        return np.split(both, len(texs), 1)
    n = min(int(round(at("COLOURS", level))), 256)     # PIL's limit; above it barely shows
    q = np.asarray(Image.fromarray(both).quantize(n, method=Image.Quantize.MEDIANCUT).convert("RGB"))
    return np.split(q, len(texs), 1)


def stats(name: str, tile: np.ndarray) -> str:
    a = tile[F.DIAMOND][:, :3].astype(float)
    g = tile[..., :3].astype(float).mean(-1)
    m = F.DIAMOND[:, 1:] & F.DIAMOND[:, :-1]
    nd = np.abs(np.diff(g, axis=1))[m].mean()
    return (f"{name}: mean {a.mean(0).round().astype(int).tolist()} std {a.std(0).mean():.1f} "
            f"neighbour diff {nd:.1f} colours {len(np.unique(a.astype(int), axis=0))}")


def build(level: float = LEVEL) -> list[np.ndarray]:
    """The four full tiles at a LEVEL (same picks at every level)."""
    rng = np.random.default_rng(12)
    raw = {k: face(im, level) for k, im in samples().items()}
    faces = set_tone(raw, level)
    plain = [k for k in faces if k != "E"]
    picks = [list(rng.choice(plain, 4, replace=False)) for _ in range(4)]
    picks[3][2] = "E"                           # one variant has the cracked flag
    texs = quantise([compose(faces, p, rng, level) for p in picks], level)
    return [F.project(np.clip(t, 0, 255).astype(np.uint8)) for t in texs]


def compare() -> None:
    """Levels 0, 1/3, 2/3, 1 side by side, at the game's scale and at 3x."""
    levels = [0, 1 / 3, 2 / 3, 1]
    pics = [F.patch(build(lv)) for lv in levels]
    zooms = [F.patch(build(lv), 2).resize((2 * 128 * 3, 2 * 64 * 3), Image.NEAREST) for lv in levels]
    for name, ims in (("flags_levels.png", pics), ("flags_levels_zoom.png", zooms)):
        w, h = ims[0].size
        out = Image.new("RGBA", (w * 4 + 60, h), (40, 40, 40, 255))
        for k, p in enumerate(ims):
            out.alpha_composite(p, (k * (w + 20), 0))
        out.save(os.path.join(BUILD, "preview", name))
    for lv in levels:
        print(stats(f"level {lv:.2f}", build(lv)[0]))


def main() -> None:
    import sys
    os.makedirs(os.path.join(BUILD, "preview"), exist_ok=True)
    if "--compare" in sys.argv:
        compare()
        return
    level = float(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else LEVEL
    full = build(level)
    sheet = sheet_file(SHEET + ".png")
    F.write_sheet(sheet, F.floor_set(full))
    print("wrote", sheet, "at level", level)

    van = os.path.join(VANILLA, "blends_street_01.png")
    vanilla = [F.sheet_tile(van, i) for i in (96, 101, 102, 103)]
    vflags = [F.sheet_tile(os.path.join(VANILLA, "floors_exterior_tilesandstone_01.png"), i) for i in (5,)]
    for i, t in enumerate(full):
        print(stats(f"ours {i}", t))
    print(stats("vanilla street 96", vanilla[0]))
    print(stats("vanilla flags 5", vflags[0]))
    # Side by side at the game's scale and at 3x: ours, vanilla's slab, vanilla's flags.
    pics = [F.patch(full), F.patch(vanilla), F.patch(vflags)]
    w, h = pics[0].size
    side = Image.new("RGBA", (w * 3 + 40, h), (40, 40, 40, 255))
    for k, p in enumerate(pics):
        side.alpha_composite(p, (k * (w + 20), 0))
    side.save(os.path.join(BUILD, "preview", "flags_side_by_side.png"))
    zoom = F.patch(full, 3).resize((3 * 128 * 3, (3 * 64) * 3), Image.NEAREST)
    zoom.save(os.path.join(BUILD, "preview", "flags_zoom.png"))
    contact = Image.new("RGBA", (128 * 8, 72 * 2), (255, 0, 255, 255))
    for i, t in enumerate(F.floor_set(full)):
        if t is not None:
            r, c = divmod(i, 8)
            contact.alpha_composite(Image.fromarray(t[186:258]), (c * 128, r * 72))
    contact.resize((128 * 8 * 2, 72 * 2 * 2), Image.NEAREST).save(os.path.join(BUILD, "preview", "flags_sheet.png"))


if __name__ == "__main__":
    main()
