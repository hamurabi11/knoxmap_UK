"""Brick garden walls from the user's photos: boundary pieces for Build's fence styles.

    python walls.py   -> build/sheets/<group>/walls_garden_uk_01.png, build/preview/walls_*.png

A garden wall is a box on a square's edge (a west or a north one, as every
boundary in the game): one brick thick (T, 0.22 m), LOW (front walls, about
0.95 m) or TALL (side and back walls, about 1.8 m). Drawn by projecting its
faces onto the 2x tile, as floor_tiles.py does the floor: 64 px a metre along
the ground and up (a storey is 192 px), east-facing faces at EAST_LIGHT of
south-facing ones and tops at TOP_LIGHT (measured on vanilla's brick walls,
2026-10-12). No 3D model in the game: Build 42 is told the box separately
(tileGeometry.txt, by pack.py).

The bricks are laid, not photographed: stretcher bond, 16 x 5 px with the
mortar (four bricks a metre, so every piece meets the next), a brick-on-edge
coping on top, each brick's colour and grain taken from a random patch of
the photo's wall (reference/catalogue.json `Boundary_wall_brick_mediumhigh`
"wall") at flags.LEVEL between a single brick colour and the photo's spread;
the mortar the photo's. Damp darkens the bottom courses.

Pieces, in Build's order (knoxbuild/fences.py STYLES): west, west (second
bricks), north, north, corner (west and north), end post (a pier T x T).
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

import flags
import floor_tiles as F
from sheet_paths import sheet_file

SHEET = "walls_garden_uk_01"
PX = 64                      # px a metre, along and up
T = 14                       # px thick (0.22 m: one brick)
BRICK, COURSE = 16, 5        # px with mortar (4 a metre; 75 mm courses)
COPING_W, COPING_H = 4, 7    # brick-on-edge units
LOW, TALL = 11, 22           # courses below the coping
EAST_LIGHT, TOP_LIGHT = 0.80, 1.12
# Build 42 depth pictures (media/depthmaps/DEPTH_<sheet>.png, grey, smaller is
# nearer): measured on vanilla's fences (DEPTH_fencing_01, fitted to +-1),
# value = DEPTH_V0 - DEPTH_ALONG (wx + wy) - DEPTH_UP z, world in metres from
# the square's north corner. A sprite without one is drawn with a guessed
# depth that cuts it (slits and dips in angled walls, in game 2026-10-13).
DEPTH_V0, DEPTH_ALONG, DEPTH_UP = 254.2, 62.8, 42.4
DAMP = (3, 0.86)             # bottom courses, darkened to
PHOTO_PX_PER_M = 760         # the photo's scale (75 mm courses about 57 px)
# Colour anchored to vanilla's brick, as the flags' brightness is (the photo,
# overcast, makes the bricks pink: 164 131 124 against vanilla's 145 107 99
# for a whole low wall piece); the photo gives the spread brick to brick.
BRICK_TARGET = np.array([152.0, 102.0, 90.0])
MORTAR = np.array([168.0, 160.0, 150.0])   # pale cement, as in the photo (sampling it caught the joints' shadow)


def brick_library(rng) -> tuple[list[np.ndarray], np.ndarray]:
    """Brick faces (COURSE - 1 x BRICK - 1 x 3) and the mortar colour from the photo."""
    cat = json.load(open(os.path.join(flags.ROOT, "reference", "catalogue.json"), encoding="utf-8"))
    ph = next(p for p in cat["photos"] if p["file"] == "Boundary_wall_brick_mediumhigh.jpeg")
    im = Image.open(os.path.join(flags.PICTURES, ph["file"])).convert("RGB").crop(tuple(ph["samples"]["wall"]))
    w, h = (round(v / PHOTO_PX_PER_M * PX) for v in im.size)
    a = np.asarray(im.resize((w, h), Image.BOX), float)
    sat = a.max(-1) - a.min(-1)
    mortar = a[sat < np.percentile(sat, 15)].mean(0)            # the greyest: mortar
    faces = []
    for _ in range(400):
        y, x = rng.integers(0, h - 4), rng.integers(0, w - 15)
        f = a[y:y + COURSE - 1, x:x + BRICK - 1].copy()
        fs = f.max(-1) - f.min(-1)
        brick = fs >= np.median(fs)                             # the window's brick pixels, not its mortar
        colour = f[brick].mean(0)
        if colour[1] >= colour[0] or fs.mean() < np.percentile(sat, 40):
            continue                                            # ivy, or mostly mortar
        f[~brick] = colour                                      # mortar in the window -> the brick's own colour
        faces.append(f)
        if len(faces) == 80:
            break
    return faces, mortar


def styled(faces: list[np.ndarray], level: float) -> list[np.ndarray]:
    """Faces at LEVEL: brick-to-brick spread and grain kept at `level`."""
    means = np.array([f.reshape(-1, 3).mean(0) for f in faces])
    overall = means.mean(0)
    return [BRICK_TARGET + (m - overall) * (0.25 + 0.75 * level) + (f - m) * (0.4 + 0.6 * level)
            for f, m in zip(faces, means)]


def face_texture(length: int, courses: int, faces, mortar, rng, shift: int = 0) -> np.ndarray:
    """A wall face, `length` px along, courses + coping high (row 0 at the top)."""
    hgt = courses * COURSE + COPING_H
    tex = np.tile(mortar, (hgt, length, 1)).astype(float)
    for c in range(courses):
        y0 = hgt - (c + 1) * COURSE + 1                         # brick rows from the ground up
        off = (BRICK // 2 if c % 2 else 0) + shift
        for b in range(-1, length // BRICK + 2):
            x0 = b * BRICK + off
            f = faces[rng.integers(len(faces))]
            for dx in range(BRICK - 1):
                x = x0 + dx
                if 0 <= x < length:
                    tex[y0:y0 + COURSE - 1, x] = f[:, dx]
        if c < DAMP[0]:
            tex[y0 - 1:y0 + COURSE - 1] *= DAMP[1] + (1 - DAMP[1]) * c / DAMP[0]
    for u in range(0, length, COPING_W):                        # brick-on-edge coping
        f = faces[rng.integers(len(faces))]
        tex[0:COPING_H - 1, u:u + COPING_W - 1] = np.resize(f.mean((0, 1)), (COPING_H - 1, min(COPING_W - 1, length - u), 3)) \
            + rng.normal(0, 3, (COPING_H - 1, min(COPING_W - 1, length - u), 1))
    return tex


def top_texture(length: int, faces, mortar, rng) -> np.ndarray:
    """The coping's top: bricks' ends across the wall, a joint every COPING_W."""
    tex = np.tile(mortar, (T, length, 1)).astype(float)
    for u in range(0, length, COPING_W):
        f = faces[rng.integers(len(faces))]
        tex[:, u:u + COPING_W - 1] = f.mean((0, 1)) + rng.normal(0, 3, (T, min(COPING_W - 1, length - u), 1))
    return tex


def _draw(out, origin, a, b, tex, light, soft=False, depth=None):
    """Paint the face origin + s a + t b (s, t in 0..1; world wx, wy, z in metres)
    onto `out` with `tex` (rows along t, columns along s), nearest, crisp; and
    each painted pixel's depth value into `depth` (a float array), if given."""
    o, a, b = (np.array(v, float) for v in (origin, a, b))
    proj = lambda p: np.array([63 + PX * (p[0] - p[1]), 192 + PX / 2 * (p[0] + p[1]) - PX * p[2]])
    so, sa, sb = proj(o), proj(o + a) - proj(o), proj(o + b) - proj(o)
    m = np.linalg.inv(np.stack([sa, sb], 1))
    ys, xs = np.mgrid[0:F.TILE_H, 0:F.TILE_W] + 0.5
    st = np.einsum("ij,jyx->iyx", m, np.stack([xs - so[0], ys - so[1]]))
    s, t = st
    inside = (s >= 0) & (s < 1) & (t >= 0) & (t < 1)
    h, w = tex.shape[:2]
    ti = np.clip((t * h).astype(int), 0, h - 1)
    si = np.clip((s * w).astype(int), 0, w - 1)
    texel = tex[ti, si]
    if depth is not None:
        w = o[:, None, None] + s * a[:, None, None] + t * b[:, None, None]
        value = DEPTH_V0 - DEPTH_ALONG * (w[0] + w[1]) - DEPTH_UP * w[2]
    if soft:                                                    # a shadow: its own alpha, under what is drawn already
        inside &= (texel[..., 3] > 0) & (out[..., 3] == 0)
        out[inside, :3] = np.clip(texel[..., :3] * light, 0, 255)[inside]
        out[inside, 3] = texel[..., 3][inside]
        if depth is not None:
            depth[inside] = value[inside]
        return
    if tex.shape[2] == 4:                                       # see-through parts (railings, a hedge's ragged top)
        inside &= texel[..., 3] > 127
        texel = texel[..., :3]
    col = np.clip(texel * light, 0, 255)
    out[inside, :3] = col[inside]
    out[inside, 3] = 255
    if depth is not None:
        depth[inside] = value[inside]


def depth_tile(tile: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """A sprite's depth picture: grey value where the sprite has pixels (one
    not painted with a depth, as the invisible pieces' pixel: the floor's)."""
    ys, xs = np.mgrid[0:F.TILE_H, 0:F.TILE_W] + 0.5
    floor = DEPTH_V0 - DEPTH_ALONG * (ys - 192) / (PX / 2)          # the floor under each pixel
    v = np.where(np.isnan(depth), floor, depth)
    out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
    out[..., :3] = np.clip(np.round(v), 0, 255)[..., None]
    out[..., 3] = np.where(tile[..., 3] > 0, 255, 0)
    out[out[..., 3] == 0] = 0
    return out


def piece(kind: str, courses: int, faces, mortar, rng, depth=None) -> np.ndarray:
    """One boundary piece as an RGBA tile; each painted pixel's depth into
    `depth` (a float array), if given."""
    out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
    hgt = courses * COURSE + COPING_H
    H, t = hgt / PX, T / PX
    if kind in ("west", "corner"):                              # along wy at wx 0..t, seen from the east
        tex = face_texture(PX, courses, faces, mortar, rng)
        _draw(out, (t, 0, H), (0, 1, 0), (0, 0, -H), tex, EAST_LIGHT, depth=depth)
        _draw(out, (0, 0, H), (0, 1, 0), (t, 0, 0), top_texture(PX, faces, mortar, rng), TOP_LIGHT, depth=depth)
    if kind in ("north", "corner"):                             # along wx at wy 0..t, seen from the south
        tex = face_texture(PX, courses, faces, mortar, rng, shift=BRICK // 4)
        x0 = t if kind == "corner" else 0
        _draw(out, (x0, t, H), (1 - x0, 0, 0), (0, 0, -H), tex[:, int(x0 * PX):], 1.0, depth=depth)
        _draw(out, (x0, 0, H), (1 - x0, 0, 0), (0, t, 0), top_texture(PX, faces, mortar, rng)[:, int(x0 * PX):], TOP_LIGHT, depth=depth)
    if kind == "post":                                          # a pier T x T
        tex = face_texture(T, courses, faces, mortar, rng)
        _draw(out, (t, 0, H), (0, t, 0), (0, 0, -H), tex, EAST_LIGHT, depth=depth)
        _draw(out, (0, t, H), (t, 0, 0), (0, 0, -H), tex, 1.0, depth=depth)
        _draw(out, (0, 0, H), (t, 0, 0), (0, t, 0), top_texture(T, faces, mortar, rng)[:, :T], TOP_LIGHT, depth=depth)
    return out


def build(level: float = flags.LEVEL) -> tuple[list, list]:
    """The sheet's pieces and their depth pictures (DEPTH_<sheet>.png): our
    own, not vanilla's borrowed - a low wall's left 24% of its pixels with no
    depth under them, a pier 80%, and the game cuts what has none (seen as
    gaps, 2026-10-14; the angled sheet's fix of 2026-10-13)."""
    rng = np.random.default_rng(5)
    faces, _photo_mortar = brick_library(rng)
    faces, mortar = styled(faces, level), MORTAR
    tiles, depths = [], []
    for courses in (LOW, TALL):
        for kind in ("west", "west", "north", "north", "corner", "post", None, None):
            if kind is None:
                tiles.append(None)
                depths.append(None)
                continue
            dep = np.full((F.TILE_H, F.TILE_W), np.nan)
            tiles.append(piece(kind, courses, faces, mortar, rng, dep))
            depths.append(depth_tile(tiles[-1], dep))
    return tiles, depths


def main() -> None:
    tiles, depths = build()
    os.makedirs(os.path.join(flags.BUILD, "preview"), exist_ok=True)
    F.write_sheet(sheet_file(SHEET + ".png"), tiles)
    F.write_sheet(sheet_file("DEPTH_" + SHEET + ".png"), depths)
    sheet = Image.new("RGBA", (128 * 8, 256 * 2), (60, 60, 60, 255))
    for i, t in enumerate(tiles):
        if t is not None:
            sheet.alpha_composite(Image.fromarray(t), ((i % 8) * 128, (i // 8) * 256))
    sheet.resize((128 * 8 * 2, 256 * 2 * 2), Image.NEAREST).save(os.path.join(flags.BUILD, "preview", "walls_sheet.png"))
    print("wrote", SHEET, sum(t is not None for t in tiles), "pieces")
    scene(tiles)


def scene(tiles) -> None:
    """A front garden: low wall along its north and west sides with a pier at
    the gate, a tall wall behind, on our flags; vanilla's low brick wall and
    railings beside it for comparison."""
    flags_tile = F.sheet_tile(sheet_file("blends_street_uk_01.png"), 0)
    uk_grass = sheet_file("blends_natural_uk_01.png")
    grass = F.sheet_tile(uk_grass, 0) if os.path.exists(uk_grass) else F.sheet_tile(os.path.join(flags.VANILLA, "blends_natural_01.png"), 16)
    van = lambda name: F.sheet_tile(os.path.join(flags.VANILLA, name.rsplit("_", 1)[0] + ".png"), int(name.rsplit("_", 1)[1]))
    n = 9
    floor = {(x, y): (grass if 1 <= x <= 3 and 1 <= y <= 4 else flags_tile) for x in range(n) for y in range(n)}
    things = {}
    lw, ln, lc, lp = tiles[0], tiles[2], tiles[4], tiles[5]
    tw, tn, tc = tiles[8], tiles[10], tiles[12]
    things[(1, 1)] = [lc]
    for x in (2, 3):
        things[(x, 1)] = [ln if x != 3 else tiles[3]]
    for y in (2, 3):
        things[(1, y)] = [lw if y == 2 else tiles[1]]
    things[(1, 4)] = [lp]                                       # the gate's pier
    for x in range(5, 8):                                       # a tall wall behind a yard
        things[(x, 1)] = [tn]
    things[(5, 1)] = [tc]
    for y in (2, 3, 4):
        things[(5, y)] = [tw]
    for x in (1, 2, 3):                                         # vanilla's low wall and railings, nearer
        things[(x, 7)] = [van("walls_exterior_house_low_01_13")]
        things[(x + 4, 7)] = [van("fencing_01_1")]
    w, h = n * 128, n * 64 + 192
    img = Image.new("RGBA", (w, h), (40, 40, 40, 255))
    order = sorted(floor, key=lambda p: (p[0] + p[1], p[0]))
    for (x, y) in order:
        img.alpha_composite(Image.fromarray(floor[(x, y)]), ((x - y) * 64 + (n - 1) * 64, (x + y) * 32))
    for (x, y) in order:
        for t in things.get((x, y), []):
            img.alpha_composite(Image.fromarray(t), ((x - y) * 64 + (n - 1) * 64, (x + y) * 32))
    img.save(os.path.join(flags.BUILD, "preview", "walls_scene.png"))
    img.crop((200, 60, 1000, 560)).resize((1600, 1000), Image.NEAREST).save(os.path.join(flags.BUILD, "preview", "walls_scene_zoom.png"))


if __name__ == "__main__":
    main()
