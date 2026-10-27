"""Floor tiles in Project Zomboid's 2x format, from a top-down texture.

A floor square is drawn as a 126 x 64 diamond at the bottom of a 128 x 256
tile (rows 192-255, columns 0-125). It holds 4096 pixels, so a top-down texture of 64 x 64
texels a square (64 px a metre on our 1 m squares) maps onto it about one
texel a pixel; `project` samples it nearest, so lines drawn on texel rows come
out as the game's own 2:1 pixel stairs.

World axes on screen: x (east) runs down-right, y (south) down-left; the
diamond's top corner is the square's north-west corner (wx = wy = 0).

A floor set fills vanilla's 16-slot layout (as `blends_street_01` 96-111),
so it can take a vanilla surface's place in Rules.txt and Blends.txt:

    0      full          5, 6, 7   full (vanilla picks among 0, 5, 6, 7)
    1, 2   half: north-west of wx + wy = 1 / south-east of it
    3, 4   half: south-west of wx = wy / north-east of it
    8-11   blend strips along the north, west, east, south edges
           (Blends.txt dir = n, w, e, s)
    12-15  empty

Sizes were measured on vanilla's sheets (2026-10-12): halves ragged over
about 0.2 of a square, strips 0.05 deep at the median and 0.16 at most. The
ragged edges here are our own noise, not vanilla's.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

TILE_W, TILE_H = 128, 256
TEX = 64                     # texels a square side (64 px a metre)
TOP = 192                    # the diamond's top row in the tile

# Pixel centres; vanilla's diamond is centred on column 63 (rows 223-224 span
# columns 0-125), one pixel left of the tile's middle - matched exactly.
_ys, _xs = np.mgrid[0:TILE_H, 0:TILE_W] + 0.5
_xs = _xs + 1
WX = ((_xs - 64) / 64 + (_ys - TOP) / 32) / 2
WY = ((_ys - TOP) / 32 - (_xs - 64) / 64) / 2
DIAMOND = (WX >= 0) & (WX < 1) & (WY >= 0) & (WY < 1)


def project(tex: np.ndarray) -> np.ndarray:
    """A TEX x TEX x 3 top-down texture (rows = wy, columns = wx) as an RGBA tile."""
    out = np.zeros((TILE_H, TILE_W, 4), np.uint8)
    tx = np.clip((WX * TEX).astype(int), 0, TEX - 1)
    ty = np.clip((WY * TEX).astype(int), 0, TEX - 1)
    out[..., :3] = tex[ty, tx]
    out[..., 3] = np.where(DIAMOND, 255, 0)
    return out


def _ragged(keep_dist: np.ndarray, base: float, spread: float, rng) -> np.ndarray:
    """Pixels kept where their distance into the cut is under a noisy threshold."""
    r = rng.random(keep_dist.shape) ** 1.5
    return DIAMOND & (keep_dist < base + spread * r)


def masks(seed: int = 7) -> dict[str, np.ndarray]:
    """The half and strip shapes, as boolean pixel masks."""
    rng = np.random.default_rng(seed)
    s, d = WX + WY, WX - WY
    return {
        "half_nw": _ragged(s, 0.85, 0.2, rng),
        "half_se": _ragged(2 - s, 0.85, 0.2, rng),
        "half_sw": _ragged(d + 1, 0.85, 0.2, rng),
        "half_ne": _ragged(1 - d, 0.85, 0.2, rng),
        "strip_n": _ragged(WY, 0.02, 0.14, rng),
        "strip_w": _ragged(WX, 0.02, 0.14, rng),
        "strip_e": _ragged(1 - WX, 0.02, 0.14, rng),
        "strip_s": _ragged(1 - WY, 0.02, 0.14, rng),
    }


def _cut(tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = tile.copy()
    out[..., 3] = np.where(mask, 255, 0)
    return out


def floor_set(full: list[np.ndarray], second_strips: bool = False) -> list[np.ndarray | None]:
    """Vanilla's 16 slots from four full tiles (RGBA, from `project`).
    second_strips: slots 12-15 a second set of blend strips (n, w, e, s), as
    vanilla's grass sets have them (blends_natural_01 40-47: medgrass_n is 40
    and 44); else empty, as its street sets."""
    m = masks()
    a = full[0]
    out = [a, _cut(a, m["half_nw"]), _cut(a, m["half_se"]), _cut(a, m["half_sw"]),
           _cut(a, m["half_ne"]), full[1], full[2], full[3],
           _cut(a, m["strip_n"]), _cut(a, m["strip_w"]), _cut(a, m["strip_e"]),
           _cut(a, m["strip_s"]), None, None, None, None]
    if second_strips:
        m2, b = masks(seed=11), full[1]
        out[12:] = [_cut(b, m2["strip_n"]), _cut(b, m2["strip_w"]), _cut(b, m2["strip_e"]), _cut(b, m2["strip_s"])]
    return out


PAD = 2          # px of each sprite's edge colour carried out into its transparent pixels


def pad_edges(t: np.ndarray, n: int = PAD) -> np.ndarray:
    """The sprite's edge colours spread `n` px into its transparent pixels,
    their alpha left 0 (K09.25, 2026-10-19). Scaled or filtered - the game at
    any zoom but 1x, render_lots - a sprite's edge blends with the pixels
    beside it: ours were black (vanilla's are mid grey), so every wall
    piece took a dark rim, a line at each square."""
    from scipy import ndimage
    solid = t[..., 3] > 0
    if solid.all() or not solid.any():
        return t
    rgb = t[..., :3].astype(float)
    filled = solid.copy()
    k = np.ones((3, 3))
    for _ in range(n):
        weight = ndimage.convolve(filled.astype(float), k, mode="constant")
        total = np.stack([ndimage.convolve(rgb[..., c] * filled, k, mode="constant") for c in range(3)], -1)
        new = ~filled & (weight > 0)
        rgb[new] = total[new] / weight[new][:, None]
        filled |= new
    out = t.copy()
    out[~solid, :3] = np.clip(rgb[~solid], 0, 255).astype(np.uint8)
    return out


def write_sheet(path: str, tiles: list[np.ndarray | None], cols: int = 8) -> None:
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * TILE_H, cols * TILE_W, 4), np.uint8)
    depth = os.path.basename(path).startswith("DEPTH_")      # a depth picture is data: left as drawn
    for i, t in enumerate(tiles):
        if t is not None:
            r, c = divmod(i, cols)
            sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = t if depth else pad_edges(t)
    Image.fromarray(sheet, "RGBA").save(path)


def sheet_tile(path: str, index: int, cols: int = 8) -> np.ndarray:
    """One tile of a sheet PNG, RGBA."""
    a = np.asarray(Image.open(path).convert("RGBA"))
    r, c = divmod(index, cols)
    return a[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W]


def patch(tiles: list[np.ndarray], n: int = 6, seed: int = 1) -> Image.Image:
    """An n x n square of floor, each square a random pick of `tiles`, as the
    game draws it (no lighting)."""
    rng = np.random.default_rng(seed)
    w, h = n * 128, n * 64 + TOP
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for wy in range(n):
        for wx in range(n):
            t = Image.fromarray(tiles[rng.integers(len(tiles))], "RGBA")
            x = (wx - wy) * 64 + (n - 1) * 64
            y = (wx + wy) * 32
            img.alpha_composite(t, (x, y))
    return img.crop((0, TOP, w, h))
