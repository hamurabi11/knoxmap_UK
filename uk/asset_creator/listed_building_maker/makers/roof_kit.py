"""English roof coverings: clay plain tile, Welsh slate and thatch, as roof sets.

    python roof_kit.py [covering ...]   (KnoxMap's Python; all by default)

Writes Tile_creator/build/sheets/<group>/roofs_uk_01_<covering>.png (45-degree and
shallow pieces: every slot of vanilla's roofs_02) and roofs_30_uk_01_<covering>.png
(30-degree pieces: every slot of roofs_30_02), each with its .slots.json
(the vanilla piece per slot: pack.py copies its properties and its depth picture), and previews. A style takes them through regional_uk's
`roof_sheet` / `roof_sheet_30` (the donor style's roof with these sheets'
names in place of its own: the slots are the same in every vanilla roof set).

Each piece is cut to the vanilla piece's outline, as wall_kit.py does. What
is measured from the vanilla sprite (geometry and light, no pixel copied):
its planes (a hip or valley piece has two: split by brightness, 2-means), each
plane's course direction (on screen a roof's courses run at a slope of +1/2,
along world x, or -1/2, along world y: whichever the sprite's own texture
varies least along), and each plane's light (its mean brightness against the
south slope's, SlopeS1). Then our covering is drawn along those courses: each
tile or slate its own tone, a shadow line under each course, joints broken
course to course; thatch as straw streaks running down the slope. A dark line
marks a hip or valley where two planes meet.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wall_kit as WK  # noqa: E402

F, W = WK.F, WK.W
SHEETS = {"roofs_uk_01_{c}": "roofs_02", "roofs_30_uk_01_{c}": "roofs_30_02"}

# covering: course spacing (px across the courses on screen), unit length
# along a course (px), base colour, tone spread, kind
COVERINGS = {
    # Hampshire clay plain tile, old and weathered (user, 2026-10-18: "too red/vibrant
    # and not worn enough"): a brown clay, each tile its own weathered tone, lichen.
    "tile": dict(course=6, unit=10, rgb=(112, 70, 56), spread=0.10, kind="units", weather=True),
    "slate": dict(course=8, unit=16, rgb=(64, 68, 78), spread=0.06, kind="units"),    # Welsh slate
    "thatch": dict(course=0, unit=0, rgb=(116, 98, 68), spread=0.10, kind="thatch"), # long straw / reed
    # Barry's cast-iron roof plates (Palace of Westminster, 1840s-50s), painted and weathered to a
    # mid lead-grey: wide plates in long courses, little tone between them (user, 2026-10-24: the
    # palace's roofs read light grey from above, our slate too dark and blue)
    # course and unit divide the 64 x 32 tile offset (57.24 px across, 71.55 along): the courses run on
    # unbroken from tile to tile over a wide flat roof (review, 2026-10-24)
    "iron_plate": dict(course=9.54, unit=23.85, rgb=(112, 116, 120), spread=0.035, kind="units"),
}
# roofs_02 32-63: another roof group (RoofGroup 4) that no style uses: left empty.
SKIP_45 = set(range(32, 64))
DIRS = [np.array([2.0, 1.0]) / np.sqrt(5), np.array([-2.0, 1.0]) / np.sqrt(5)]
_ys, _xs = np.mgrid[0:256, 0:128] + 0.5


def planes(tile: np.ndarray):
    """The sprite's planes: a label per pixel (0, 1; -1 outside), each plane's
    course direction (index into DIRS) and light (0-1)."""
    alpha = tile[..., 3] > 0
    lab = np.full(alpha.shape, -1)
    if alpha.sum() < 8:
        return lab, [], []
    # Brightness averaged over a few tiles' width (the sprite's own tiles and
    # their highlights averaged out), inside the outline only.
    raw = tile[..., :3].astype(float).mean(-1)
    num = ndimage.gaussian_filter(np.where(alpha, raw, 0), 6)
    den = ndimage.gaussian_filter(alpha.astype(float), 6)
    lum = num / np.maximum(den, 1e-6)
    # Two planes meet on a straight line: a hip or valley of an equal-pitch
    # roof is a world diagonal, upright on screen; a ridge runs at +-1/2. The
    # best such line (most of the brightness explained), if it explains enough.
    vals = lum[alpha]
    total = ((vals - vals.mean()) ** 2).sum()
    best = (0.0, None)
    for coord in (_xs, _xs - 2 * _ys, _xs + 2 * _ys):
        c = coord[alpha]
        for cut in np.unique(np.round(c))[3:-3:1]:
            side = c >= cut
            n1 = side.sum()
            if n1 < 0.12 * len(c) or n1 > 0.88 * len(c):
                continue
            m0, m1 = vals[~side].mean(), vals[side].mean()
            gain = n1 * (len(c) - n1) / len(c) * (m1 - m0) ** 2
            if gain > best[0]:
                best = (gain, (coord, cut))
    if best[1] is None or best[0] < 0.6 * total or abs(np.diff([vals.min(), vals.max()]))[0] < 6:
        lab[alpha] = 0
    else:
        coord, cut = best[1]
        lab[alpha] = (coord[alpha] >= cut).astype(int)
    gy, gx = np.gradient(raw)
    dirs, light = [], []
    for k in range(lab.max() + 1):
        m = (lab == k) & ndimage.binary_erosion(alpha, iterations=2)
        if m.sum() < 4:
            m = lab == k
        e = [np.abs(gx[m] * d[0] + gy[m] * d[1]).mean() for d in DIRS]
        dirs.append(int(np.argmin(e)))
        light.append(raw[lab == k].mean())
    return lab, dirs, light


def covering_pixels(c: dict, d: np.ndarray, rng, seed: int) -> np.ndarray:
    """The covering over the whole tile, its courses along screen direction d."""
    n = np.array([-d[1], d[0]])
    along = _xs * d[0] + _ys * d[1]
    across = _xs * n[0] + _ys * n[1]
    base = np.array(c["rgb"], float)
    if c["kind"] == "thatch":
        # Straw down the slope: noise stretched along the fall (across the courses).
        r = np.random.default_rng(seed)
        fine = r.normal(0, 1, (64, 512))
        streak = ndimage.gaussian_filter(fine, (0.6, 8))
        i = (np.round(along).astype(int) % 64, np.round(across * 2).astype(int) % 512)
        tone = 1 + 0.16 * streak[i] / max(streak.std(), 1e-6) * c["spread"] * 6
        tone *= 1 + 0.05 * np.sin(across / 9)               # the courses of the coat, faintly
        return base * tone[..., None]
    course = np.floor(across / c["course"]).astype(int)
    # Optional (housing_kit's concrete tiles, 2026-10-19; defaults as before):
    # stagger=False lays the joints in line course to course (interlocking
    # tiles); shadow= the course shadow's px and darkening; joint= the joint's
    # darkening; roll= px per roll of a profiled tile (Redland 49), its
    # light and shade across each unit; weather= a dict of its own tones.
    half = (course % 2) * c["unit"] / 2 if c.get("stagger", True) else 0
    unit = np.floor((along + half) / c["unit"]).astype(int)
    r = np.random.default_rng(seed)
    table = 1 + r.normal(0, c["spread"], (64, 64))
    tone = table[course % 64, unit % 64]
    out = base * tone[..., None]
    if c.get("roll"):
        ph = ((along + half) / c["roll"]) % 1
        crest = c.get("roll_skew")                            # a pantile: a narrow lit crest, a wide trough
        wave = (np.sin(2 * np.pi * ph) if crest is None else
                np.where(ph < crest, np.sin(np.pi * ph / crest), -np.sin(np.pi * (ph - crest) / (1 - crest))))
        out *= (1 + c.get("roll_depth", 0.14) * wave)[..., None]
    frac = across / c["course"] - np.floor(across / c["course"])
    px, dark = c.get("shadow", (1, 0.62))
    out[frac > 1 - px / c["course"]] *= dark                  # the shadow under each course
    if c.get("roll"):
        out[(frac > 1 - (px + 1) / c["course"]) & (frac <= 1 - px / c["course"])] *= 1.12   # the tile's lit nose
    jfrac = (along + half) / c["unit"]
    out[(jfrac - np.floor(jfrac)) < 1 / c["unit"]] *= c.get("joint", 0.8)     # the joints between tiles
    if isinstance(c.get("weather"), dict):
        w = c["weather"]
        tones = np.array(w["tones"])
        pick = r.choice(len(tones), (64, 64), p=w["p"])
        out = out * tones[pick[course % 64, unit % 64]]
        grey = out.mean(-1, keepdims=True)
        out = grey + (out - grey) * w["sat"]
        spots = r.random(out.shape[:2])
        lichen = spots < w["lichen"]
        if w.get("clump"):                                    # 2 px clumps, not single pale pixels
            lichen = ndimage.binary_dilation(lichen, np.array([[0, 1], [1, 1]], bool))
        out[(spots >= w["lichen"]) & (spots < w["lichen"] + w["moss"])] = (62, 62, 46)
        out[lichen] = w.get("lichen_rgb", (150, 146, 118))
    elif c.get("weather"):
        # Each tile one of a few weathered tones (burnt dark, brown, purplish,
        # faded); per tile, not in patches across the roof (a roof is many
        # sprites: a patch would end at each one's edge).
        tones = np.array([(1.0, 1.0, 1.0), (0.80, 0.76, 0.76), (1.04, 0.96, 0.86), (0.86, 0.80, 0.86),
                          (1.06, 1.04, 1.0), (0.72, 0.70, 0.68)])
        pick = r.choice(len(tones), (64, 64), p=[0.34, 0.18, 0.18, 0.12, 0.10, 0.08])
        out = out * tones[pick[course % 64, unit % 64]]
        grey = out.mean(-1, keepdims=True)
        out = grey + (out - grey) * 0.7                      # dulled: the clay's fired red long gone
        spots = r.random(out.shape[:2])
        out[spots < 0.006] = (150, 146, 118)                 # lichen
        out[(spots >= 0.006) & (spots < 0.011)] = (62, 62, 46)   # moss, soot
    return out + r.normal(0, 2.5, out.shape)


def paint(tile: np.ndarray, c: dict, rng, seed: int, top_light: float) -> tuple[np.ndarray, np.ndarray]:
    lab, dirs, light = planes(tile)
    out = np.zeros((256, 128, 4), np.uint8)
    depth = np.full((256, 128), np.nan)
    if not dirs:
        return out, depth
    cover = [covering_pixels(c, DIRS[d], rng, seed + k) for k, d in enumerate(dirs)]
    for k in range(len(dirs)):
        m = lab == k
        shade = np.clip(light[k] / top_light, 0.45, 1.15)
        out[m, :3] = np.clip(cover[k][m] * shade, 0, 255).astype(np.uint8)
        out[m, 3] = 255
    if len(dirs) > 1:                                         # the hip or valley line
        edge = (lab == 0) & ndimage.binary_dilation(lab == 1)
        out[edge, :3] = (out[edge, :3] * 0.7).astype(np.uint8)
    return out, depth


def write(covering: str) -> list[str]:
    c = COVERINGS[covering]
    rng = np.random.default_rng(sum(map(ord, covering)))
    written = []
    for pattern, vsheet in SHEETS.items():
        name = pattern.format(c=covering)
        n = 128
        vt = [WK.vanilla_tile(vsheet, i) for i in range(n)]
        top = max(planes(vt[0])[2])                     # SlopeS1 / Slope30S1: a south slope at full light
        tiles, depths = [], []
        for i, t in enumerate(vt):
            if vsheet == "roofs_02" and i in SKIP_45:
                t = np.zeros_like(t)
            tl, dp = paint(t, c, rng, 1000 * i + 7, top)
            tiles.append(tl)
            depths.append(dp)
        F.write_sheet(WK.sheet_file(name + ".png"), tiles)
        with open(WK.sheet_file(name + ".slots.json"), "w", encoding="utf-8") as f:
            json.dump([f"{vsheet}_{i}" if t[..., 3].any() else None for i, t in enumerate(tiles)], f)
        img = Image.new("RGBA", (8 * 128, 16 * 256), (70, 90, 70, 255))
        for k, t in enumerate(tiles):
            img.alpha_composite(Image.fromarray(t), ((k % 8) * 128, (k // 8) * 256))
        img.resize((512, 2048), Image.LANCZOS).save(os.path.join(WK.PREVIEW, f"kit_{name}.png"))
        written.append(name)
    return written


def main() -> None:
    for covering in sys.argv[1:] or COVERINGS:
        print("wrote", write(covering))


if __name__ == "__main__":
    main()
