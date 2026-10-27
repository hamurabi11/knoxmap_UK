"""English house walls from free (CC0) textures found by find_textures.py: a test.

    python house_walls.py   -> build/sheets/<group>/walls_house_uk_<material>.png (16 tiles each)
                               build/preview/house_walls.png, house_walls_<material>.png

Each MATERIAL is one picture found by find_textures.py (all CC0; downloaded to
reference/found/downloads/, recorded in Mapping/SOURCES.md), for an EPC period:
pebbledash (1930-1949 semis), white render (Georgian and interwar), flint
(before 1900), tile-hanging (1967-1975 estates; clay plain tiles hung on a
wall, from a roof-tile scan).

- Cut to the game's 64 px a metre (`px_per_m`: the source's own scale, from
  Poly Haven's and ambientCG's stated sizes or estimated from the pebbles and
  flints), the large-scale shading divided out, made seamless (each face
  blended with itself shifted half a tile, so it wraps both ways and storeys
  stack), the detail at flags.LEVEL between the material's flat colour and the
  photo's, brightness set to `bright` (vanilla's own walls: the cream set 196,
  red brick 115), colour strength `sat` (photos are more saturated than the
  game's art: pebbledash's coloured pebbles read as a confetti wall).
- Laid out as vanilla's wall sets (walls_exterior_house_01, measured
  2026-10-12): two variants, A in slots 0-3 and 8-11, B in 4-7 and 12-15;
  west, north, corner, post, window west, window north, door west, door north.
  A storey is 3 squares high (192 px); the window opening 0.23-0.77 along and
  1.17-2.25 up, the door 0.10-0.87 along and up to 2.36, as vanilla's, so its
  windows and doors fit. A 1 px darker reveal round each opening.
- Not yet in the palette (Building_types/palettes/english.json): a style
  needs its window-shaped pieces and gable caps too.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage

import find_textures as FT
import flags
import floor_tiles as F
import walls as W
from sheet_paths import sheet_file

DOWNLOADS = os.path.join(flags.ROOT, "reference", "found", "downloads")
VERSION = 6                  # previews by version: an open picture cannot be overwritten on Windows
STOREY = 192
T = 2                        # px: a house wall's thickness as vanilla draws it
WINDOW = (0.23, 0.77, 1.17, 2.25)
DOOR = (0.10, 0.87, 0.0, 2.36)

MICKLEHAM = ("https://upload.wikimedia.org/wikipedia/commons/a/a7/St_Michael%27s_Church%2C_Old_London_Road%2C_"
             "Mickleham_%28NHLE_Code_1028835%29_%28Chequerboard_Wall_of_Side_Chapel%29.JPG")
MICKLEHAM_PAGE = ("https://commons.wikimedia.org/wiki/File:St_Michael%27s_Church,_Old_London_Road,_Mickleham_"
                  "(NHLE_Code_1028835)_(Chequerboard_Wall_of_Side_Chapel).JPG")
# The chequer's flint squares (user, 2026-10-12: "use the squares as tiles", not
# a chequer wall): each an evenly lit sample of knapped flint, found by their
# dark, busy texture, 15% trimmed off each side (the clunch joint).
FLINT_PANELS = [(263, 777, 418, 919), (0, 816, 165, 1015), (440, 980, 603, 1110), (637, 1136, 778, 1269),
                (1222, 1306, 1375, 1444), (825, 1316, 984, 1463), (1413, 1472, 1600, 1610),
                (593, 1473, 805, 1638), (1014, 1481, 1181, 1619)]
MATERIALS = {
    "pebbledash": dict(url="https://upload.wikimedia.org/wikipedia/commons/9/94/Pebbledash_Texture_Example_Close.jpg",
                       page="https://commons.wikimedia.org/wiki/File:Pebbledash_Texture_Example_Close.jpg",
                       licence="CC0", px_per_m=18000, bright=170, period="1930-1949", tint=(1.0, 0.98, 0.93), sat=0.35),
    "render_white": dict(url="https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/white_stucco/white_stucco_diff_1k.jpg",
                         page="https://polyhaven.com/a/white_stucco", licence="CC0 1.0",
                         px_per_m=512, bright=200, period="Georgian, interwar", tint=(1.0, 1.0, 0.97)),
    # St Michael's Church, Mickleham: a chequerboard of knapped flint and clunch
    # under a band of plain flint (the user's find, 2026-10-12; the Nepcote
    # wall it replaces was blotchy and repeated). Chequer squares about 0.3 m.
    "flint": dict(url=MICKLEHAM, page=MICKLEHAM_PAGE, licence="CC0", file="flint_mickleham",
                  panels=FLINT_PANELS, px_per_m=640, bright=100, period="before 1900",
                  tint=(1.0, 1.0, 1.0), sat=0.5),
}


def fetch(name: str) -> Image.Image:
    m = MATERIALS[name]
    os.makedirs(DOWNLOADS, exist_ok=True)
    path = os.path.join(DOWNLOADS, m.get("file", name) + ".jpg")
    if not os.path.exists(path):
        req = urllib.request.Request(m["url"], headers={"User-Agent": FT.UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            open(path, "wb").write(r.read())
        meta = os.path.join(DOWNLOADS, "sources.json")
        known = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
        known[name] = {k: m[k] for k in ("url", "page", "licence")}
        json.dump(known, open(meta, "w", encoding="utf-8"), indent=1)
    return Image.open(path).convert("RGB")


def seamless(a: np.ndarray) -> np.ndarray:
    """Wraps both ways: blended with itself shifted by half, weighted to the shifted copy at the edges."""
    h, w = a.shape[:2]
    wy = np.minimum(np.arange(h), h - 1 - np.arange(h)) / (h / 2)
    wx = np.minimum(np.arange(w), w - 1 - np.arange(w)) / (w / 2)
    wgt = np.clip(np.minimum.outer(wy, wx) * 2.2, 0, 1)[..., None]
    shifted = np.roll(np.roll(a, h // 2, 0), w // 2, 1)
    return a * wgt + shifted * (1 - wgt)


def from_panels(name: str, level: float, rng) -> np.ndarray:
    """A wall laid from sample squares of the photo: each panel cut to the
    game's scale, its own mean taken out (no square shows lighter or darker),
    placed in a grid of PANEL px with random flips, the grid's seams softened."""
    m = MATERIALS[name]
    im = fetch(name)
    s = F.TEX / m["px_per_m"]
    tiles = []
    for x0, y0, x1, y1 in m["panels"]:
        dx, dy = (x1 - x0) * .15, (y1 - y0) * .15
        p = im.crop((round(x0 + dx), round(y0 + dy), round(x1 - dx), round(y1 - dy)))
        p = np.asarray(p.resize((max(4, round(p.width * s)), max(4, round(p.height * s))), Image.BOX), float)
        tiles.append(p - p.reshape(-1, 3).mean(0))
    size = min(min(t.shape[:2]) for t in tiles)
    out = np.zeros((STOREY, 128, 3))
    for y in range(0, STOREY, size):
        for x in range(0, 128, size):
            t = tiles[rng.integers(len(tiles))][:size, :size]
            t = t[::-1] if rng.random() < .5 else t
            t = t[:, ::-1] if rng.random() < .5 else t
            h, w = min(size, STOREY - y), min(size, 128 - x)
            out[y:y + h, x:x + w] = t[:h, :w]
    soft = np.stack([ndimage.gaussian_filter(out[..., c], 1.0) for c in range(3)], -1)
    grid = np.zeros((STOREY, 128), bool)
    grid[::size] = grid[size - 1::size] = True
    grid[:, ::size] = grid[:, size - 1::size] = True
    out = np.where(grid[..., None], soft, out)
    trimmed = [(x0 + (x1 - x0) * .15, y0 + (y1 - y0) * .15, x1 - (x1 - x0) * .15, y1 - (y1 - y0) * .15)
               for x0, y0, x1, y1 in m["panels"]]                 # without the clunch joint, as the faces
    mean = np.mean([np.asarray(im.crop(tuple(round(v) for v in b_))).reshape(-1, 3).mean(0) for b_ in trimmed], 0)
    colour = np.array(m["tint"]) * m["bright"] + (mean / mean.mean() * m["bright"] - np.array(m["tint"]) * m["bright"]) * level
    res = colour + out * (0.35 + 0.65 * level) * (m["bright"] / max(mean.mean(), 1))
    grey = res.mean(-1, keepdims=True)
    return grey + (res - grey) * m.get("sat", 1.0)


def material(name: str, level: float, rng) -> np.ndarray:
    """A seamless texture 2 tiles wide and a storey high (128 x 192), at the game's scale and LEVEL."""
    m = MATERIALS[name]
    if "panels" in m:
        return from_panels(name, level, rng)
    im = fetch(name)
    if m.get("crop"):
        im = im.crop(m["crop"])
    if m.get("rotate"):
        im = im.rotate(m["rotate"], expand=True)
    s = F.TEX / m["px_per_m"]
    small = np.asarray(im.resize((max(4, round(im.width * s)), max(4, round(im.height * s))), Image.BOX), float)
    if small.shape[0] < STOREY + 8 or small.shape[1] < 136:
        # A close-up covers less than a wall (pebbledash's photo is 13 cm across):
        # the wall filled from it in randomly flipped blocks - at 64 px a metre
        # pebbledash is a speckle, so the blocks do not show.
        bh, bw = small.shape[:2]
        fill = np.zeros((STOREY + 8 + bh, 136 + bw, 3))
        for y0 in range(0, fill.shape[0] - bh + 1, bh):
            for x0 in range(0, fill.shape[1] - bw + 1, bw):
                blk = small[::-1] if rng.random() < .5 else small
                fill[y0:y0 + bh, x0:x0 + bw] = blk[:, ::-1] if rng.random() < .5 else blk
        small = fill[:STOREY + 8, :136]
    lum = small.mean(-1)
    shade = ndimage.gaussian_filter(lum, 24)
    small = small * (lum.mean() / np.maximum(shade, 1))[..., None]          # large-scale light out
    y, x = rng.integers(0, small.shape[0] - STOREY + 1), rng.integers(0, small.shape[1] - 128 + 1)
    a = seamless(small[y:y + STOREY, x:x + 128])
    mean = a.reshape(-1, 3).mean(0)
    detail = a - mean
    flat = np.array(m["tint"]) * m["bright"]
    colour = flat + (mean / mean.mean() * m["bright"] - flat) * level      # the photo's colour, LEVEL of the way
    out = colour + detail * (0.35 + 0.65 * level) * (m["bright"] / max(mean.mean(), 1))
    grey = out.mean(-1, keepdims=True)                         # a photo's colour is stronger than the game's
    return grey + (out - grey) * m.get("sat", 1.0)


def opening(tex: np.ndarray, rect) -> np.ndarray:
    """The face with a hole (alpha 0) and a 1 px darker reveal round it."""
    out = np.dstack([tex.copy(), np.full(tex.shape[:2], 255.0)])
    w = tex.shape[1]
    s0, s1, z0, z1 = rect
    x0, x1 = round(s0 * w), round(s1 * w)
    r0, r1 = STOREY - round(z1 * F.TEX), STOREY - round(z0 * F.TEX)
    out[max(0, r0 - 1):r1 + 1, max(0, x0 - 1):x1 + 1, :3] *= 0.72
    out[r0:r1, x0:x1, 3] = 0
    return out


def face(out, tex, side: str) -> None:
    """A wall face on the square's west or north edge (wx or wy = 0), its thin top."""
    H = STOREY / W.PX
    t = T / W.PX
    if tex.shape[2] == 3:
        tex = np.dstack([tex, np.full(tex.shape[:2], 255.0)])
    if side == "W":
        W._draw(out, (t, 0, H), (0, 1, 0), (0, 0, -H), tex, W.EAST_LIGHT)
        W._draw(out, (0, 0, H), (0, 1, 0), (t, 0, 0), tex[:2] * [1.1, 1.1, 1.1, 1], 1.0)
    else:
        W._draw(out, (0, t, H), (1, 0, 0), (0, 0, -H), tex, 1.0)
        W._draw(out, (0, 0, H), (1, 0, 0), (0, t, 0), tex[:2] * [1.1, 1.1, 1.1, 1], 1.0)


def wall_set(name: str, level: float = flags.LEVEL) -> list[np.ndarray]:
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    tiles = [None] * 16
    for variant, base in ((0, 0), (1, 4)):
        tex = material(name, level, rng)
        a, b = tex[:, :64], tex[:, 64:]                         # west and north faces from different parts
        def tile(*faces):
            out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
            for tx, side in faces:
                face(out, tx, side)
            return out
        tiles[base + 0] = tile((a, "W"))
        tiles[base + 1] = tile((b, "N"))
        tiles[base + 2] = tile((a, "W"), (b, "N"))
        post = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)       # the corner where two walls meet: a T x T pillar
        H, t = STOREY / W.PX, T / W.PX
        W._draw(post, (t, 0, H), (0, t, 0), (0, 0, -H), np.dstack([a[:, :T], np.full((STOREY, T), 255.0)]), W.EAST_LIGHT)
        W._draw(post, (0, t, H), (t, 0, 0), (0, 0, -H), np.dstack([b[:, :T], np.full((STOREY, T), 255.0)]), 1.0)
        tiles[base + 3] = post
        tiles[base + 8] = tile((opening(a, WINDOW), "W"))
        tiles[base + 9] = tile((opening(b, WINDOW), "N"))
        tiles[base + 10] = tile((opening(a, DOOR), "W"))
        tiles[base + 11] = tile((opening(b, DOOR), "N"))
    return tiles


# ---- Drains (prototype): a downpipe on a wall, from the gutter to a gully ----
PIPE = (28, 28, 30)


def downpipe(side: str, part: str, along: float = 0.08) -> np.ndarray:
    """A black cast-iron downpipe on a west or north wall, `along` the square
    (0.08 or 0.92: at a building's corner, clear of the window opening
    0.23-0.77 - user, 2026-10-12: it stood in front of a window): 'top' (hopper head under the eaves), 'mid', or 'foot'
    (shoe into a gully grate on the ground). One storey a piece."""
    out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
    H = STOREY / W.PX
    off = 0.06                                                 # stands off the wall face
    pipe = np.zeros((STOREY, 4, 4))
    pipe[..., :3] = PIPE
    pipe[..., 3] = 255
    pipe[:, 0, :3] *= 1.6                                       # its lit edge
    if part == "top":
        pipe[:14] = 0
        pipe[14:24, :, 3] = 255
    if side == "W":
        o, a = (off, along - 2 / W.PX, H), (0, 4 / W.PX, 0)
    else:
        o, a = (along - 2 / W.PX, off, H), (4 / W.PX, 0, 0)
    W._draw(out, o, a, (0, 0, -H), pipe, 1.0)
    if part == "top":                                           # the hopper: a wider box at the top
        hop = np.zeros((10, 10, 4))
        hop[..., :3] = PIPE
        hop[..., 3] = 255
        hop[:2, :, :3] *= 1.8
        ho = (off, along - 5 / W.PX, H - 14 / W.PX) if side == "W" else (along - 5 / W.PX, off, H - 14 / W.PX)
        ha = (0, 10 / W.PX, 0) if side == "W" else (10 / W.PX, 0, 0)
        W._draw(out, ho, ha, (0, 0, -10 / W.PX), hop, 1.0)
    if part == "foot":                                          # gully grate: a dark square, grid lines
        g = np.zeros((10, 10, 4))
        g[..., :3] = 40
        g[::3, :, :3] = 85
        g[..., 3] = 255
        go = (0.02, along - 0.08, 0.001) if side == "W" else (along - 0.08, 0.02, 0.001)
        W._draw(out, go, (0, 0.16, 0) if side == "W" else (0.16, 0, 0), (0.16, 0, 0) if side == "W" else (0, 0.16, 0), g, 1.0)
    return out


def house(tiles, storeys=2, w=5, d=4, pipes=True) -> Image.Image:
    """A box house of `tiles` (w along x, d along y), windows on the south and
    east faces, a door; downpipes at the front corners; on our flags."""
    n = 12
    img = Image.new("RGBA", (n * 128, n * 64 + 192 * (storeys + 1)), (40, 40, 40, 255))
    fl = F.sheet_tile(sheet_file("blends_street_uk_01.png"), 0)
    ox, oy = 2, 2
    oy_px = 192 * storeys
    order = sorted(((x, y) for x in range(n) for y in range(n)), key=lambda p: (p[0] + p[1], p[0]))
    sx = lambda x, y: (x - y) * 64 + (n - 1) * 64
    sy = lambda x, y, z: (x + y) * 32 + oy_px - z * 192
    for x, y in order:
        img.alpha_composite(Image.fromarray(fl), (sx(x, y), sy(x, y, 0)))
    pipe_w = [downpipe("W", p, 0.92) for p in ("foot", "mid", "top")]   # by the near corner
    pipe_n = [downpipe("N", p, 0.08) for p in ("foot", "mid", "top")]   # at the far corner
    for z in range(storeys):
        for x, y in order:
            things = []
            # west and north walls of the box are hidden from this view; draw its east (x = ox + w) and south (y = oy + d) walls
            if x == ox + w and oy <= y < oy + d:
                v = 0 if (y % 2) else 4
                things.append(tiles[v + (8 if (y - oy) % 2 == 1 else 0)])
                if pipes and y == oy + d - 1:
                    things.append(pipe_w[0 if z == 0 else (2 if z == storeys - 1 else 1)])
            if y == oy + d and ox <= x < ox + w:
                v = 1 if (x % 2) else 5
                kind = 10 if (z == 0 and x == ox + 2) else (8 if (x - ox) % 2 == 1 else 0)
                things.append(tiles[v + kind])
                if pipes and x == ox:
                    things.append(pipe_n[0 if z == 0 else (2 if z == storeys - 1 else 1)])
            if x == ox + w and y == oy + d:
                things.append(tiles[3])
            for t in things:
                img.alpha_composite(Image.fromarray(t), (sx(x, y), sy(x, y, z)))
    return img.crop((300, 0, n * 128 - 200, n * 64 + 192 * storeys - 100))


def main() -> None:
    shots = []
    for name in MATERIALS:
        tiles = wall_set(name)
        F.write_sheet(sheet_file(f"walls_house_uk_{name}.png"), tiles)
        h = house(tiles)
        h.save(os.path.join(flags.BUILD, "preview", f"house_walls_{name}_v{VERSION}.png"))
        shots.append((name, h))
        print("wrote", name)
    w, hh = shots[0][1].size
    out = Image.new("RGBA", (w * 2, hh * 2), (40, 40, 40, 255))
    for k, (_n, h) in enumerate(shots):
        out.alpha_composite(h, ((k % 2) * w, (k // 2) * hh))
    out.save(os.path.join(flags.BUILD, "preview", f"house_walls_v{VERSION}.png"))


if __name__ == "__main__":
    main()
