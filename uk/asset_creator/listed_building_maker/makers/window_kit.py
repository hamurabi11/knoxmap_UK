"""English windows for the listed buildings: sashes, casements, leaded lights.

    python window_kit.py   (KnoxMap's Python)

Writes Tile_creator/build/sheets/<group>/fixtures_windows_uk_01.png and its
.slots.json, and a preview. Each window is a pair (west, north) in vanilla's
window layout: closed in slots 0-9, open at +32, smashed at +56, its glass
gone at +88 (the offsets vanilla's own windows name: OpenTileOffset,
SmashedTileOffset, GlassRemovedOffset), so the game swaps them as it does
vanilla's. WINDOWS gives each its opening: WindowShape 9 (tall, vanilla's
tall sash) or 13 (square), whose size is measured from vanilla's window-shaped
wall piece of that shape (its hole, in wall coordinates: wall_kit.FACES).

Drawn flat on the wall (64 px a metre: frame, glazing bars, glass with a
reflection) and projected with wall_kit's camera, west faces at its light.
Open: a sash's lower half raised behind the upper, a casement's leaves
swung out (only their edges seen). Smashed: the glass broken away in shards.
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

W = WK.W
SHEET = "fixtures_windows_uk_01"
WHITE, OAK, STONE = (234, 234, 228), (84, 66, 50), (186, 180, 164)
GLASS = (78, 100, 110)
LEAD = (58, 60, 62)
# name: (WindowShape, frame colour, kind)
WINDOWS = [
    ("sash_georgian", 9, WHITE, "sash6"),       # 6-over-6 in reveals (C18, early C19)
    ("sash_victorian", 9, WHITE, "sash2"),      # 2-over-2 (C19)
    ("casement", 13, WHITE, "casement"),         # small-pane timber casements, two leaves
    ("casement_leaded", 13, OAK, "leaded"),      # leaded lights in oak casements (cottages, C16-C17)
    ("mullion", 13, STONE, "mullion"),          # stone mullion and transom, three lights
    ("sash_arched", 10, WHITE, "sash6"),        # a round-headed sash (C18-C19 stair and Venetian lights)
]
STATES = {"closed": 0, "open": 32, "smashed": 56, "noglass": 88}
VANILLA_SLOT = {"closed": 16, "open": 48, "smashed": 72, "noglass": 104}   # fixtures_windows_white's tall sash


def _hole(shape: int, side: str = "W") -> np.ndarray:
    """The tile pixels of a WindowShape's hole in the west (or north) face, from
    vanilla's window-shaped wall piece (walls_exterior_house_05, the shape's W or N
    piece: a half-window's hole is mirrored on the north wall)."""
    t = WK.vanilla_tile("walls_exterior_house_05", 32 + 2 * (shape - 1) + (side == "N"))
    u, v, panel, _ = WK.FACES[side]
    hole = panel & ((WK._xs < 64) if side == "W" else (WK._xs >= 64)) & (t[..., 3] == 0)
    hole = ndimage.binary_opening(hole, iterations=1)
    lab, n = ndimage.label(hole)
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (np.argmax(sizes) + 1)


def opening(shape: int) -> tuple[int, int, int, int]:
    """(u0, u1, v0, v1) of a WindowShape's hole on the west face."""
    u, v, _panel, _ = WK.FACES["W"]
    inner = _hole(shape)
    return int(u[inner].min()), int(u[inner].max()) + 1, int(v[inner].min()), int(v[inner].max()) + 1


def outline(shape: int, side: str = "W") -> np.ndarray:
    """The hole as a 64 x 192 wall-texture mask (an arched head is not a box), on the
    west face or the north (u along +x)."""
    u, v, _panel, _ = WK.FACES[side]
    inner = _hole(shape, side)
    m = np.zeros((192, 64), bool)
    m[np.clip(v[inner], 0, 191).astype(int), u[inner]] = True
    return ndimage.binary_closing(m, iterations=2)


def _rect(a, x0, y0, x1, y1, rgb, alpha=255):
    a[max(0, y0):y1, max(0, x0):x1, :3] = rgb
    a[max(0, y0):y1, max(0, x0):x1, 3] = alpha


def glass(a, x0, y0, x1, y1, tint=GLASS):
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return
    ys, xs = np.mgrid[0:h, 0:w]
    refl = np.clip(1 - np.abs((xs + ys * 0.6) / max(w, 1) - 0.45) * 3, 0, 1) * 34   # a sky reflection band
    a[y0:y1, x0:x1, :3] = np.clip(np.array(tint) + refl[..., None], 0, 255)
    a[y0:y1, x0:x1, 3] = 235


def bars(a, x0, y0, x1, y1, cols, rows, rgb, width=1):
    for k in range(1, cols):
        x = x0 + round(k * (x1 - x0) / cols)
        _rect(a, x - width // 2, y0, x - width // 2 + width, y1, rgb)
    for k in range(1, rows):
        y = y0 + round(k * (y1 - y0) / rows)
        _rect(a, x0, y - width // 2, x1, y - width // 2 + width, rgb)


def lattice(a, x0, y0, x1, y1, rgb, step=6):
    """Diamond quarries: lead cames on both diagonals."""
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x - x0 + y - y0) % step == 0 or (x - x0 - (y - y0)) % step == 0:
                a[y, x, :3] = rgb
                a[y, x, 3] = 255


def draw(kind: str, frame, state: str, box, rng) -> np.ndarray:
    """The window on a 64 x 192 west-face texture (RGBA; transparent off it)."""
    a = np.zeros((192, 64, 4), np.uint8)
    u0, u1, v0, v1 = box
    f = 2
    _rect(a, u0, v0, u1, v1, frame)                                  # the frame
    ix0, iy0, ix1, iy1 = u0 + f, v0 + f, u1 - f, v1 - f
    lights = []                                                       # glazed areas: (x0, y0, x1, y1, cols, rows, lead)
    if kind.startswith("sash"):
        mid = (iy0 + iy1) // 2
        cols, rows = (3, 2) if kind == "sash6" else (2, 1)
        if state == "open":                                           # lower sash raised behind the upper
            lights = [(ix0, iy0, ix1, mid - 1, cols, rows, False)]
            _rect(a, ix0, mid - 1, ix1, mid + 1, frame)
            _rect(a, ix0, mid + 1, ix1, iy1, (0, 0, 0), 0)          # open below
            _rect(a, ix0, iy0 + (mid - iy0) // 2, ix1, iy0 + (mid - iy0) // 2 + 1, frame)
        else:
            lights = [(ix0, iy0, ix1, mid - 1, cols, rows, False), (ix0, mid + 1, ix1, iy1, cols, rows, False)]
            _rect(a, ix0, mid - 1, ix1, mid + 1, frame)              # the meeting rails
    elif kind in ("casement", "leaded"):
        midx = (ix0 + ix1) // 2
        lead = kind == "leaded"
        if state == "open":                                           # leaves swung out: their edges only
            _rect(a, ix0, iy0, ix1, iy1, (0, 0, 0), 0)
            _rect(a, ix0, iy0, ix0 + 1, iy1, frame)
            _rect(a, ix1 - 1, iy0, ix1, iy1, frame)
        else:
            _rect(a, midx - 1, iy0, midx + 1, iy1, frame)
            lights = [(ix0, iy0, midx - 1, iy1, 2, 3, lead), (midx + 1, iy0, ix1, iy1, 2, 3, lead)]
    elif kind == "mullion":
        w = ix1 - ix0
        xs = [ix0, ix0 + w // 3, ix0 + 2 * w // 3, ix1]
        ty = iy0 + (iy1 - iy0) // 3
        _rect(a, ix0, ty - 1, ix1, ty + 1, frame)                    # the transom
        for x in xs[1:-1]:
            _rect(a, x - 1, iy0, x + 1, iy1, frame)                  # the mullions
        for k in range(3):
            lights += [(xs[k] + (1 if k else 0), iy0, xs[k + 1] - (1 if k < 2 else 0), ty - 1, 1, 1, True),
                       (xs[k] + (1 if k else 0), ty + 1, xs[k + 1] - (1 if k < 2 else 0), iy1, 1, 1, True)]
        if state == "open":                                           # the middle lower light open
            x0, y0, x1, y1 = lights[3][:4]
            lights = lights[:3] + lights[4:]
            _rect(a, x0, y0, x1, y1, (0, 0, 0), 0)
    for x0, y0, x1, y1, cols, rows, lead in lights:
        if state == "noglass":
            _rect(a, x0, y0, x1, y1, (0, 0, 0), 0)
        else:
            glass(a, x0, y0, x1, y1, (82, 106, 98) if lead else GLASS)
            if state == "smashed":                                    # shards gone, a few left round the edges
                h, w = y1 - y0, x1 - x0
                keep = np.zeros((h, w), bool)
                keep[:, :max(1, w // 6)] = keep[:max(1, h // 8), :] = True
                keep |= rng.random((h, w)) < 0.12
                keep = ndimage.binary_dilation(keep, iterations=1)
                a[y0:y1, x0:x1, 3] = np.where(keep, a[y0:y1, x0:x1, 3], 0)
        if lead:
            lattice(a, x0, y0, x1, y1, LEAD)
        else:
            bars(a, x0, y0, x1, y1, cols, rows, frame)
    return a


def piece_hole(sheet: str, idx: int, side: str = "W") -> np.ndarray:
    """Any vanilla wall piece's hole (its largest see-through area in the west or north face) as
    a 64 x 192 wall-texture mask, as outline() does for a WindowShape's piece: a recipe window
    that must also cover the plain window piece's hole (Build lays walls_exterior_house_01_12/13's
    outline under a window it has no window-shaped piece for) unions it in (recipe "cover")."""
    t = WK.vanilla_tile(sheet, idx)
    u, v, panel, _ = WK.FACES[side]
    hole = panel & ((WK._xs < 64) if side == "W" else (WK._xs >= 64)) & (t[..., 3] == 0)
    hole = ndimage.binary_opening(hole, iterations=1)
    lab, n = ndimage.label(hole)
    m = np.zeros((192, 64), bool)
    if not n:
        return m
    inner = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    m[np.clip(v[inner], 0, 191).astype(int), u[inner]] = True
    return ndimage.binary_dilation(ndimage.binary_closing(m, iterations=2), iterations=1)


def project(tex: np.ndarray, side: str) -> np.ndarray:
    """A face texture (64 x 192 RGBA) onto a tile at the west or north wall."""
    u, v, panel, _ = WK.FACES[side]
    out = np.zeros((256, 128, 4), np.uint8)
    on = panel.copy()
    vi = np.clip(v, 0, 191).astype(int)
    texel = tex[vi, u]
    on &= texel[..., 3] > 0
    light = W.EAST_LIGHT if side == "W" else 1.0
    out[on, :3] = np.clip(texel[on, :3] * light, 0, 255).astype(np.uint8)
    out[on, 3] = texel[on, 3]
    return out


def main() -> None:
    rng = np.random.default_rng(1)
    tiles = [None] * 104
    slots = [None] * 104
    shapes = {}
    for k, (name, shape, frame, kind) in enumerate(WINDOWS):
        box = shapes.setdefault(shape, opening(shape))
        for state, off in STATES.items():
            tex = draw(kind, frame, state, box, rng)
            if shape == 10:                              # cut to the arched head, a frame round its curve
                m = outline(shape)
                tex[..., 3] = np.where(m, tex[..., 3], 0)
                rim = m & ~ndimage.binary_erosion(m, iterations=2)
                tex[rim, :3], tex[rim, 3] = frame, 255
            for j, side in enumerate(("W", "N")):
                slot = off + 2 * k + j
                tiles[slot] = project(tex, side)
                slots[slot] = {"like": f"fixtures_windows_white_{VANILLA_SLOT[state] + j}", "WindowShape": shape,
                               "name": name}
    blank = np.zeros((256, 128, 4), np.uint8)
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), [t if t is not None else blank for t in tiles])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=0)
    img = Image.new("RGBA", (128 * 10, 256 * 4), (150, 110, 95, 255))
    for r, off in enumerate(STATES.values()):
        for c in range(10):
            img.alpha_composite(Image.fromarray(tiles[off + c]), (c * 128, r * 256))
    img.save(os.path.join(WK.PREVIEW, f"kit_{SHEET}.png"))
    print("wrote", SHEET, {s: opening(s) for s in shapes})


if __name__ == "__main__":
    main()
