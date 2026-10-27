"""The pictures on the foliage's cards, drawn flat, for foliage.py.

    python foliage_textures.py          (the Knoxmap venv's Python: PIL)

Writes asset_creator/renders/foliage/<species>_<k>.png: RGBA, transparent
round the plant, at about 400 px a metre (Blender's sampling antialiases
them down to the sprite's 64). Each is drawn from a seed, so the same
every run; four of each, so neighbouring clumps differ. Darker at the foot
(the plant's own shade) and a touch lighter at the tips, as vanilla's are.

British plants of an English summer (the game opens in July): meadow grass
(plain, seeding, with buttercups, with daisies), nettles, cow parsley,
bracken, brambles (in flower, in fruit), foxgloves, bluebells and daffodils
(spring: drawn, not put on the maps by default), soft rushes; and leaf
sprays for the shrubs - hawthorn, elder in flower, holly, gorse in flower.
Colours from the user's brief: vanilla's level of detail, the hues of real
UK hedgerows (no photographs traced).
"""
from __future__ import annotations

import math
import os
import random
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "renders", "foliage")
PX = 400                                    # px a metre
VARIANTS = 4

GRASS = [(92, 122, 48), (110, 136, 56), (128, 146, 62), (146, 150, 70), (168, 162, 92)]
DARK_LEAF = [(52, 82, 36), (62, 94, 40), (72, 104, 46)]
MID_LEAF = [(82, 120, 48), (96, 132, 54), (108, 142, 60)]


def jitter(c, rng, k=12):
    return tuple(max(0, min(255, v + rng.randint(-k, k))) for v in c) + (255,)


def card(w, h):
    im = Image.new("RGBA", (int(w * PX), int(h * PX)), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def shade_foot(im, depth=0.45, top=1.08):
    """Darker at the foot, a little lighter at the tips."""
    px = im.load()
    w, h = im.size
    for y in range(h):
        f = top - (top - depth) * (y / max(h - 1, 1)) ** 1.6
        for x in range(w):
            r, g, b, a = px[x, y]
            if a:
                px[x, y] = (min(255, int(r * f)), min(255, int(g * f)), min(255, int(b * f)), a)
    return im


def blade(d, rng, x, base, height, lean, colour, width):
    """A grass blade: a curve from its foot, leaning, thinning to its tip."""
    pts = []
    for k in range(9):
        t = k / 8
        pts.append((x + lean * t * t * height, base - t * height))
    for (a, b), k in zip(zip(pts, pts[1:]), range(8)):
        d.line([a, b], fill=colour, width=max(1, int(width * (1 - k / 9))))
    return pts[-1]


def grass(seed, heads=False, flowers=None):
    rng = random.Random(seed)
    im, d = card(0.6, 0.7)
    w, h = im.size
    for _ in range(70):
        x = rng.uniform(0.1, 0.9) * w
        height = rng.uniform(0.35, 0.95) * h
        tip = blade(d, rng, x, h, height, rng.uniform(-0.5, 0.5), jitter(rng.choice(GRASS), rng), rng.randint(3, 6))
        if heads and rng.random() < 0.35:
            d.ellipse((tip[0] - 4, tip[1] - 12, tip[0] + 4, tip[1] + 6), fill=jitter((176, 160, 104), rng))
    if flowers:
        colour, centre = flowers
        for _ in range(14):
            x, y = rng.uniform(0.12, 0.88) * w, rng.uniform(0.25, 0.75) * h
            d.line([(x, y), (x + rng.uniform(-8, 8), h)], fill=jitter((96, 128, 50), rng), width=2)
            r = rng.randint(6, 9)
            d.ellipse((x - r, y - r, x + r, y + r), fill=colour)
            if centre:
                d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=centre)
    return shade_foot(im)


def heart_leaf(d, rng, cx, cy, size, angle, colour, serrate=True):
    pts = []
    for k in range(20):
        t = 2 * math.pi * k / 20
        r = size * (0.55 + 0.45 * math.cos(t / 2) ** 2) * (1.08 if serrate and k % 2 else 1.0)
        x, y = r * math.sin(t) * 0.62, -r * math.cos(t)
        a = math.radians(angle)
        pts.append((cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)))
    d.polygon(pts, fill=colour)


def nettles(seed):
    rng = random.Random(seed)
    im, d = card(0.6, 1.0)
    w, h = im.size
    for _ in range(7):
        x = rng.uniform(0.15, 0.85) * w
        top = rng.uniform(0.1, 0.4) * h
        d.line([(x, h), (x, top)], fill=jitter((74, 92, 44), rng), width=4)
        y, size = h - 50, 30
        while y > top:
            for side in (-1, 1):
                heart_leaf(d, rng, x + side * size * 0.5, y, size, side * 60 + 180, jitter(rng.choice(DARK_LEAF), rng))
            y -= rng.uniform(42, 56)
            size *= 0.9
        d.line([(x, top), (x, top - 30)], fill=jitter((110, 120, 80), rng), width=3)   # the flower tassel
    return shade_foot(im, 0.5)


def cow_parsley(seed):
    rng = random.Random(seed)
    im, d = card(0.7, 1.1)
    w, h = im.size
    for _ in range(26):                                      # the feathery leaves low down
        x, y = rng.uniform(0.1, 0.9) * w, rng.uniform(0.6, 0.95) * h
        for _ in range(6):
            dx, dy = rng.uniform(-22, 22), rng.uniform(-22, 10)
            d.polygon([(x, y), (x + dx, y + dy), (x + dx * 0.6 + 6, y + dy * 0.6)], fill=jitter(rng.choice(MID_LEAF), rng))
    for _ in range(6):
        x0, x1 = rng.uniform(0.25, 0.75) * w, rng.uniform(0.15, 0.85) * w
        top = rng.uniform(0.05, 0.3) * h
        d.line([(x0, h), (x1, top)], fill=jitter((98, 120, 60), rng), width=3)
        for _ in range(60):                                  # the umbel: a cloud of white
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 34) ** 0.9
            px, py = x1 + r * math.cos(a), top + r * math.sin(a) * 0.45
            d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=jitter((236, 236, 224), rng, 8))
    return shade_foot(im, 0.55)


def bracken(seed):
    rng = random.Random(seed)
    im, d = card(0.9, 0.9)
    w, h = im.size
    for _ in range(4):
        x = rng.uniform(0.3, 0.7) * w
        lean = rng.choice((-1, 1)) * rng.uniform(0.5, 1.0)
        pts = [(x + lean * (t ** 2) * w * 0.4, h - t * h * rng.uniform(0.75, 0.95)) for t in [k / 14 for k in range(15)]]
        d.line(pts, fill=jitter((110, 112, 54), rng), width=4)
        for k in range(2, 14):                               # pinnae, both sides, shorter to the tip
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            nx, ny = -(by - ay), bx - ax
            n = math.hypot(nx, ny) or 1
            length = (1 - k / 15) * 90
            for side in (-1, 1):
                ex, ey = ax + side * nx / n * length, ay + side * ny / n * length + 14
                d.polygon([(ax, ay), (ex, ey), (bx, by)], fill=jitter((104, 142, 50), rng))
    return shade_foot(im, 0.55)


def leaflet(d, cx, cy, size, angle, colour):
    a = math.radians(angle)
    pts = []
    for k in range(12):
        t = 2 * math.pi * k / 12
        x, y = math.sin(t) * size * 0.45, -math.cos(t) * size
        pts.append((cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)))
    d.polygon(pts, fill=colour)


def brambles(seed, fruit):
    rng = random.Random(seed)
    im, d = card(0.9, 0.8)
    w, h = im.size
    for _ in range(4):                                       # arching stems
        x0 = rng.uniform(0.1, 0.9) * w
        x1 = x0 + rng.choice((-1, 1)) * rng.uniform(0.3, 0.6) * w
        peak = rng.uniform(0.1, 0.4) * h
        pts = [(x0 + (x1 - x0) * t, h - (h - peak) * 4 * t * (1 - t) - (h * 0.2) * t) for t in [k / 16 for k in range(17)]]
        d.line(pts, fill=jitter((96, 52, 54), rng), width=6)
        for (x, y) in pts[2::2]:
            for k in range(3):                               # a leaf of three leaflets
                leaflet(d, x, y, 22, rng.uniform(-60, 60) + (k - 1) * 50 + 180, jitter(rng.choice(DARK_LEAF), rng))
            if rng.random() < 0.55:
                if fruit:
                    col = rng.choice([(36, 20, 40), (50, 24, 50), (160, 40, 46)])
                    d.ellipse((x - 8, y + 8, x + 8, y + 24), fill=col)
                else:
                    d.ellipse((x - 9, y + 6, x + 9, y + 24), fill=jitter((240, 214, 224), rng, 8))
    return shade_foot(im, 0.55)


def foxglove(seed):
    rng = random.Random(seed)
    im, d = card(0.5, 1.3)
    w, h = im.size
    for _ in range(7):                                       # basal rosette, low and loose
        leaflet(d, rng.uniform(0.2, 0.8) * w, h - rng.uniform(10, 40), 40, rng.uniform(-85, 85) + 180,
                jitter((88, 116, 64), rng, 18))
    for _ in range(2):
        x = rng.uniform(0.35, 0.65) * w
        top = rng.uniform(0.04, 0.18) * h
        d.line([(x, h - 40), (x, top)], fill=jitter((88, 110, 60), rng), width=5)
        y = h * 0.62
        while y > top + 30:
            k = (y - top) / (h * 0.62 - top)
            r = 10 + 8 * k
            side = rng.choice((-1, 1))
            d.ellipse((x + side * 4 - r * 0.7, y - r, x + side * 4 + r * 0.7 + side * 6, y + r * 1.2),
                      fill=jitter((186, 74, 152), rng, 10))
            y -= 2 * r * 0.75
        d.ellipse((x - 5, top - 10, x + 5, top + 10), fill=(96, 120, 60))       # buds
    return shade_foot(im, 0.6)


def bluebells(seed):
    rng = random.Random(seed)
    im, d = card(0.5, 0.4)
    w, h = im.size
    for _ in range(14):
        x = rng.uniform(0.15, 0.85) * w
        blade(d, rng, x, h, rng.uniform(0.3, 0.6) * h, rng.uniform(-0.6, 0.6), jitter((70, 112, 60), rng), 6)
    for _ in range(5):
        x = rng.uniform(0.2, 0.8) * w
        tip = blade(d, rng, x, h, rng.uniform(0.7, 0.95) * h, rng.uniform(0.3, 0.7), jitter((80, 110, 66), rng), 3)
        for k in range(5):
            bx, by = tip[0] + 4, tip[1] + 14 + k * 16
            d.ellipse((bx - 6, by - 7, bx + 6, by + 8), fill=jitter((84, 86, 196), rng, 10))
    return shade_foot(im, 0.6)


def daffodils(seed):
    rng = random.Random(seed)
    im, d = card(0.5, 0.5)
    w, h = im.size
    for _ in range(12):
        blade(d, rng, rng.uniform(0.2, 0.8) * w, h, rng.uniform(0.5, 0.8) * h, rng.uniform(-0.3, 0.3),
              jitter((90, 128, 96), rng), 7)
    for _ in range(5):
        x = rng.uniform(0.2, 0.8) * w
        top = rng.uniform(0.1, 0.3) * h
        d.line([(x, h), (x, top)], fill=(86, 124, 80), width=4)
        for k in range(6):
            a = math.radians(k * 60)
            d.ellipse((x + 9 * math.cos(a) - 6, top + 9 * math.sin(a) - 6, x + 9 * math.cos(a) + 6,
                       top + 9 * math.sin(a) + 6), fill=(248, 226, 80, 255))
        d.ellipse((x - 6, top - 6, x + 6, top + 6), fill=(240, 170, 30, 255))
    return shade_foot(im, 0.6)


def rushes(seed):
    rng = random.Random(seed)
    im, d = card(0.5, 0.9)
    w, h = im.size
    for _ in range(40):
        x = rng.uniform(0.3, 0.7) * w
        tip = blade(d, rng, x, h, rng.uniform(0.6, 1.0) * h, rng.uniform(-0.25, 0.25), jitter((58, 92, 50), rng), 4)
        if rng.random() < 0.25:
            d.ellipse((tip[0] - 5, tip[1] + 30, tip[0] + 5, tip[1] + 50), fill=jitter((132, 110, 70), rng))
    return shade_foot(im, 0.55)


def spray(seed, leaf, colours, flowers=None, berries=None):
    """A spray of a shrub's leaves for one of its cards: (leaf shape, colours, extras)."""
    rng = random.Random(seed)
    im, d = card(0.4, 0.4)
    w, h = im.size
    for _ in range(46):
        x, y = rng.uniform(0.12, 0.88) * w, rng.uniform(0.12, 0.88) * h
        if math.hypot(x - w / 2, y - h / 2) > w * 0.44:
            continue
        col = jitter(rng.choice(colours), rng)
        if leaf == "lobed":                                  # hawthorn: small lobed leaves
            for k in range(3):
                leaflet(d, x + (k - 1) * 6, y, 13, rng.uniform(0, 360), col)
        elif leaf == "holly":
            heart_leaf(d, rng, x, y, 18, rng.uniform(0, 360), col)
        elif leaf == "needle":                               # gorse: green spines
            for _ in range(5):
                a = rng.uniform(0, 2 * math.pi)
                d.line([(x, y), (x + 14 * math.cos(a), y + 14 * math.sin(a))], fill=col, width=3)
        else:                                                # elder: pinnate, larger
            leaflet(d, x, y, 20, rng.uniform(0, 360), col)
    for extra, colour, r, n in ((flowers, None, 0, 0), (berries, None, 0, 0)):
        if extra:
            colour, r, n = extra
            for _ in range(n):
                x, y = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
                if math.hypot(x - w / 2, y - h / 2) < w * 0.42:
                    d.ellipse((x - r, y - r, x + r, y + r), fill=jitter(colour, rng, 8))
    return im


SPECIES = {
    "grass": lambda k: grass(100 + k),
    "grass_seeding": lambda k: grass(200 + k, heads=True),
    "grass_buttercups": lambda k: grass(300 + k, flowers=((246, 214, 40, 255), None)),
    "grass_daisies": lambda k: grass(400 + k, flowers=((244, 244, 238, 255), (240, 200, 40, 255))),
    "nettles": lambda k: nettles(500 + k),
    "cow_parsley": lambda k: cow_parsley(600 + k),
    "bracken": lambda k: bracken(700 + k),
    "brambles_flowering": lambda k: brambles(800 + k, fruit=False),
    "brambles_fruiting": lambda k: brambles(900 + k, fruit=True),
    "foxglove": lambda k: foxglove(1000 + k),
    "bluebells": lambda k: bluebells(1100 + k),
    "daffodils": lambda k: daffodils(1200 + k),
    "rushes": lambda k: rushes(1300 + k),
    "hawthorn": lambda k: spray(1400 + k, "lobed", DARK_LEAF + MID_LEAF[:1], berries=((150, 160, 70), 3, 6)),
    "elder": lambda k: spray(1500 + k, "elder", MID_LEAF, flowers=((236, 230, 196), 9, 5)),
    "holly": lambda k: spray(1600 + k, "holly", [(30, 62, 32), (38, 72, 38), (46, 82, 42)]),
    "gorse": lambda k: spray(1700 + k, "needle", [(70, 98, 40), (84, 108, 44)], flowers=((250, 206, 30), 5, 26)),
}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, make in SPECIES.items():
        for k in range(VARIANTS):
            make(k).save(os.path.join(OUT, f"{name}_{k}.png"))
    print(len(SPECIES), "species,", VARIANTS, "each ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
