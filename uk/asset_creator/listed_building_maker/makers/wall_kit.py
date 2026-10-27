"""House wall kits for the listed buildings: a whole palette style per material.

    python wall_kit.py [material ...]   (KnoxMap's Python; all materials by default)

Writes Tile_creator/build/sheets/<group>/walls_house_uk_01_<material>.png (+ .slots.json, each
slot's vanilla piece, and a preview in Tile_creator/build/preview/kit_<material>.png): everything a
palette style needs (Building_types/palettes/english.json `add`), at fixed
places in one sheet (LAYOUT):

    0-15    the wall set: west, north, corner, post, window W/N, door W/N,
            in two variants (A 0-3 and 8-11, B 4-7 and 12-15)   -> "wall"
    16-47   the 16 window-shaped pieces, W and N                 -> "shaped"
    48-61   gable caps at 45 degrees                             -> "caps"
    64-71   shallow gable caps                                   -> "caps_shallow"
    72-117  gable caps at 30 degrees                             -> "caps_30"

Every piece has the outline of the vanilla piece it stands in for (sizes and
anchoring measured from the vanilla sprite's alpha, as kerb_shapes.py does for
the kerbs; no vanilla pixel is copied): the outline is filled by projecting
our own material onto the wall faces (walls.py's camera: x = 63 + 64 (wx - wy),
y = 192 + 32 (wx + wy) - 64 z), west faces at walls.EAST_LIGHT. A 1 px reveal
darkens the edge of every opening; brick materials get rubbed or gauged brick
arches over openings and stone cills under windows; each slot's depth picture is the one
vanilla gives the piece it stands in for (pack.py, tileDepthTextureAssignments.txt).

Materials (MATERIALS): Flemish-bond red brick (the user's brick photo, as the
garden walls), painted brick, white and cream lime render (Poly Haven CC0),
black-and-white and oak-and-brick timber framing, flint with brick dressings
(Commons CC0), clay tile-hanging, black and white weatherboarding, and
chequer brick. Drawn procedurally where no photo is used.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
TC = os.path.normpath(os.path.join(HERE, "..", "..", "Tile_creator", "src"))
sys.path.insert(0, TC)
import flags  # noqa: E402
import floor_tiles as F  # noqa: E402
import house_walls as HW  # noqa: E402
import walls as W  # noqa: E402
from sheet_paths import sheet_file  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # asset_creator: acpaths
import acpaths  # noqa: E402
VANILLA = acpaths.TOOLS_TILES
PREVIEW = os.path.join(flags.BUILD, "preview")
STOREY = 192
T = 2 / 64                     # a house wall's thickness as vanilla draws it (house_walls.T)
REVEAL = 0.72

# (our slot, vanilla sheet, vanilla index): the outline each slot takes.
_SET = [4, 5, 6, 7, None, None, None, None, 12, 13, 14, 15]
LAYOUT = ([(k, "walls_exterior_house_01", v) for k, v in enumerate(_SET) if v is not None]
          + [(k + 4, "walls_exterior_house_01", v) for k, v in enumerate(_SET[:4])]
          + [(k + 12, "walls_exterior_house_01", v) for k, v in zip(range(4), _SET[8:])]
          + [(16 + k, "walls_exterior_house_05", 32 + k) for k in range(32)]
          # gable caps: all 18 of vanilla's (in game, 2026-10-19: with 14, the
          # caps regional_uk._recap asks for at 62-65 - PeakOnePt5E, PeakPt5E,
          # PeakTwoPt5E/S - were empty or shallow pieces: '???' and gables
          # with no apex), then the shallow and the 30-degree sets
          + [(48 + k, "walls_exterior_roofs_02", k) for k in range(18)]
          + [(66 + k, "walls_exterior_roofs_02", 48 + k) for k in range(8)]
          + [(74 + k, "walls_exterior_roofs_30_02", k) for k in range(46)])
SLOTS = 120                    # 15 rows of 8
FIRST = {"wall": 0, "shaped": 16, "caps": 48, "caps_shallow": 66, "caps_30": 74}

_sheet_cache: dict[str, np.ndarray] = {}


def vanilla_tile(sheet: str, idx: int) -> np.ndarray:
    if sheet not in _sheet_cache:
        _sheet_cache[sheet] = np.asarray(Image.open(os.path.join(VANILLA, sheet + ".png")).convert("RGBA"))
    a = _sheet_cache[sheet]
    cols = a.shape[1] // 128
    r, c = divmod(idx, cols)
    return a[r * 256:(r + 1) * 256, c * 128:(c + 1) * 128]


# ---- the faces: where each pixel of a tile lies on the west or north wall ----
_ys, _xs = np.mgrid[0:256, 0:128] + 0.5


def _face_coords():
    """Per pixel, for the west face (wx = T) and the north face (wy = T): the
    texture column (0-63 along the wall) and row (0 at the top of a storey),
    whether the pixel is on the face's storey-high panel, and its depth."""
    out = {}
    for side in ("W", "N"):
        if side == "W":
            s = (63 + 64 * T - _xs) / 64
            z = (192 + 32 * (T + s) - _ys) / 64
            wx, wy = T, s
        else:
            s = (_xs - 63 + 64 * T) / 64
            z = (192 + 32 * (s + T) - _ys) / 64
            wx, wy = s, T
        # Wrapped, not clamped (to-do K09.25): a piece's outline runs a pixel
        # or two past its square, over the next piece along; clamped, those
        # pixels repeated column 0 where the next square has column 63, so
        # every join lost a column and doubled one - the seam on plain walls.
        u = np.floor(s * 64).astype(int) % 64
        v = (3 - z) * 64
        panel = (s >= 0) & (s < 1) & (z >= 0) & (z < 3)
        depth = W.DEPTH_V0 - W.DEPTH_ALONG * (wx + wy) - W.DEPTH_UP * np.clip(z, 0, None)
        out[side] = (u, v, panel, depth)
    return out


FACES = _face_coords()


def paint(mask: np.ndarray, tex: np.ndarray, trim: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fill a vanilla piece's outline from our material. `tex` is a storey
    (192 rows) by two faces (128 columns: 0-63 the west face, 64-127 the
    north); `trim` draws arches and cills round openings (see `_trim`)."""
    alpha = mask[..., 3] > 0
    out = np.zeros((256, 128, 4), np.uint8)
    depth = np.full((256, 128), np.nan)
    # Which face each pixel is on: a west wall's sprite runs x = 0-65, a north
    # wall's x = 60-127 (measured on vanilla's), so a piece within one of
    # those is that wall; a corner or post has both, split at the middle.
    xs = np.nonzero(alpha.any(0))[0]
    if len(xs) and xs.max() <= 67:
        split = 999
    elif len(xs) and xs.min() >= 58:
        split = -1
    else:
        split = 64
    # The outer corner's post (a few columns wide) stands where a face's texture begins: on a
    # panelled ashlar its columns were the panel's dark border moulding - a dark seam down every
    # SE corner (review, 2026-10-24). A post takes its columns from mid-face instead.
    post = len(xs) > 0 and xs.max() - xs.min() <= 8
    for side, cols in (("W", _xs < split), ("N", _xs >= split)):
        u, v, panel, dep = FACES[side]
        if post:
            u = (u + 24) % 64
        here = alpha & cols
        # Below the storey's foot (vanilla's outline runs a pixel or two
        # past it, over the storey below) the texture runs on into the
        # next storey's top rows; clamped, it repeated the bottom row - a
        # mortar joint - as a pale line at every floor (K09.25).
        vi = np.where(v >= STOREY, np.floor(v) - STOREY, np.clip(v, 0, STOREY - 1)).astype(int)
        col = tex[vi, u + (0 if side == "W" else 64)].astype(float)
        col *= W.EAST_LIGHT if side == "W" else 1.0
        top = v < 0                                    # the wall's thin top, above the storey's face
        col[top] *= W.TOP_LIGHT / (W.EAST_LIGHT if side == "W" else 1.0)
        # Openings: inside this face's panel (two pixels in from its edges:
        # vanilla's outline stops a pixel short of them, which is no opening),
        # not in the outline.
        hole = ndimage.binary_erosion(panel & cols, iterations=2) & ~alpha
        rim = here & ndimage.binary_dilation(hole, iterations=1) & ~top
        col[rim] *= REVEAL
        if trim:
            _trim(col, here, hole, v, trim, W.EAST_LIGHT if side == "W" else 1.0)
        out[here, :3] = np.clip(col[here], 0, 255).astype(np.uint8)
        out[here, 3] = 255
        depth[here] = dep[here]
    return out, depth


def _trim(col, here, hole, v, trim, side_light) -> None:
    """Arches over openings and cills under windows, column by column (a wall
    face is upright on screen). trim = {"arch": (rgb, rows), "cill": (rgb, rows)}.
    Only real openings: an arch where there is wall over the opening, a cill
    where there is wall over and under it (a window, not a door; not the sky
    above a gable's slope)."""
    light = lambda x: side_light
    if "soldier" in trim:
        _soldier(col, here, hole, trim["soldier"], side_light)
    for x in np.nonzero(hole.any(0))[0]:
        ys = np.nonzero(hole[:, x])[0]
        runs = np.split(ys, np.nonzero(np.diff(ys) > 1)[0] + 1)
        for run in runs:
            top, bot = run[0], run[-1]
            above = top > 0 and here[top - 1, x]
            below = bot < 255 and here[bot + 1, x]
            if "arch" in trim and above:
                rgb, rows = trim["arch"]
                for k, y in enumerate(range(top - rows, top)):
                    if 0 <= y and here[y, x]:
                        shade = 0.88 if (x + k // 2) % 3 == 0 else 1.0     # the arch's radiating joints
                        col[y, x] = np.array(rgb) * shade * light(x)
            if "cill" in trim and above and below:
                rgb, rows = trim["cill"]
                for y in range(bot + 1, bot + 1 + rows):
                    if y < 256 and here[y, x]:
                        col[y, x] = np.array(rgb) * light(x) * (1.08 if y == bot + 1 else 0.9)


def _soldier(col, here, hole, spec, side_light, past=4) -> None:
    """A soldier course over each opening with wall over it (bricks on end:
    a joint every 3 px), `past` px beyond it each side (housing kits, review
    2026-10-19: a plain darker band vanished at 1x). The opening's head
    slants with the wall on screen: the course follows it column by column,
    carried on past the opening along the same slant."""
    rgb, rows = spec
    cols = np.nonzero(hole.any(0))[0]
    if not len(cols):
        return
    for run in np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1):
        tops = np.array([np.nonzero(hole[:, x])[0][0] for x in run], float)
        slope, icept = np.polyfit(run, tops, 1) if len(run) > 1 else (0.0, tops[0])
        mid = run[len(run) // 2]
        if not here[max(0, int(round(slope * mid + icept)) - 2), mid]:
            continue                                   # no wall over it (a gable's sky)
        for k, x in enumerate(range(run[0] - past, run[-1] + past + 1)):
            if not 0 <= x < col.shape[1]:
                continue
            top = int(round(slope * x + icept))
            for y in range(max(0, top - rows), top):
                if here[y, x] and not hole[y, x]:
                    joint = 0.62 if k % 3 == 0 else (1.06 if y == top - rows else 1.0)
                    col[y, x] = np.array(rgb) * side_light * joint


# ---- the materials: a storey high, two faces wide (192 x 128 x 3) ----
COURSES = [5, 5, 5, 5, 4] * 8          # 40 courses of 4.8 px: a storey (75 mm courses at 64 px a metre)
FLEMISH = [(15, "S"), (7, "H"), (14, "S"), (7, "H"), (14, "S"), (7, "H")]   # 64 px: three pairs (with joints)
FACE = 64                              # a material is made a face (a metre) at a time, seamless across it:
                                       # wall pieces repeat every metre, so anything wider shows a seam


def bond_texture(brick_of, mortar, rng, kind="flemish", header_of=None) -> np.ndarray:
    """Bricks laid course by course; `brick_of(w, h)` gives one brick's face,
    `header_of` a header's (Flemish bond). 1 px joints. Seamless both ways."""
    tex = np.tile(np.array(mortar, float), (STOREY, FACE, 1))
    y = 0
    for c, h in enumerate(COURSES):
        if kind == "flemish":
            units, off = FLEMISH, (11 if c % 2 else 0)
        else:                                           # stretcher bond
            units, off = [(16, "S")] * 4, (8 if c % 2 else 0)
        x = -off
        for w, k in units * 2:
            if x >= FACE:
                break
            face = (header_of or brick_of)(w - 1, h - 1) if k == "H" else brick_of(w - 1, h - 1)
            for dx in range(w - 1):
                tex[y:y + h - 1, (x + dx) % FACE] = face[:, dx]
            x += w
        y += h
    return tex


def _photo_bricks(rng, level):
    faces, mortar = W.brick_library(rng)
    faces = W.styled(faces, level)
    def brick(w, h):
        f = faces[rng.integers(len(faces))]
        return np.resize(f, (h, w, 3)) if f.shape[1] < w else f[:h, :w]
    return brick, mortar


# Vitrified headers (user, 2026-10-18, a photo of Georgian Flemish bond): the
# header ends burnt to a blue-grey glaze in the kiln, so the face reads as red
# stretchers chequered with grey: blue-grey, slate, pale grey, and a few
# purple-brown ones only half glazed.
HEADER_TONES = [((92, 98, 108), 0.42), ((70, 73, 80), 0.20), ((122, 124, 128), 0.16), ((104, 80, 74), 0.22)]


def red_brick(rng, level=flags.LEVEL):
    brick, mortar = _photo_bricks(rng, level)
    tones = np.array([t for t, _p in HEADER_TONES], float)
    shares = np.array([p for _t, p in HEADER_TONES])

    def header(w, h):
        tone = tones[rng.choice(len(tones), p=shares)] * rng.uniform(0.9, 1.08)
        face = np.full((h, w, 3), tone) + rng.normal(0, 6, (h, w, 1))
        face[:, :1] = face[:, :1] * 0.6 + brick(1, h)[:, :1] * 0.4      # the red showing at the glaze's edge
        face[:1] *= 1.12                                                 # the glaze catching the light
        return face
    return bond_texture(brick, mortar * 0.95, rng, "flemish", header)


def chequer_brick(rng, level=flags.LEVEL):
    brick, mortar = _photo_bricks(rng, level)
    blue = lambda w, h: np.full((h, w, 3), (92, 92, 104), float) + rng.normal(0, 5, (h, w, 1))
    return bond_texture(brick, mortar * 0.95, rng, "flemish", blue)


def painted_brick(rng, paint=(226, 222, 208)):
    brick = lambda w, h: np.full((h, w, 3), paint, float) + rng.normal(0, 2.5, (h, w, 1)) + rng.normal(0, 3)
    # (No tarred plinth: every storey takes the same piece, so it would band each floor.)
    return bond_texture(brick, np.array(paint) * 0.86, rng, "flemish")


def _photo_face(name, rng, **over):
    """A face of a photo material (house_walls.py), wrapped to repeat every metre."""
    HW.MATERIALS["_face"] = dict(HW.MATERIALS[name], **over)
    tex = HW.material("_face", flags.LEVEL, rng)
    del HW.MATERIALS["_face"]
    return HW.seamless(tex[:, :FACE])


def render(rng, tint=(1.0, 1.0, 0.97), bright=200):
    return _photo_face("render_white", rng, tint=tint, bright=bright)


def flint(rng):
    return _photo_face("flint", rng)


def _line(tex, p0, p1, width, rgb, rng):
    """A timber member from p0 to p1 (x, y), `width` px, with grain."""
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) * 2) + 1
    for k in range(n):
        x = p0[0] + (p1[0] - p0[0]) * k / (n - 1)
        y = p0[1] + (p1[1] - p0[1]) * k / (n - 1)
        for d in range(width):
            if abs(p1[0] - p0[0]) > abs(p1[1] - p0[1]):
                xx, yy = int(x) % FACE, int(y) + d - width // 2
            else:
                xx, yy = int(x + d - width // 2) % FACE, int(y)
            if 0 <= yy < STOREY:
                tex[yy, xx] = np.array(rgb) * (0.92 + 0.12 * ((d + k // 3) % 3 == 0))


def timber_frame(rng, infill="render", oak=(46, 40, 36)):
    """Close-studded frame, a metre of it: a post, studs a third of a metre
    apart, a mid rail, wall plate and sole plate, a brace in half the bays;
    infill lime render or brick nogging."""
    if infill == "render":
        tex = np.full((STOREY, FACE, 3), (210, 204, 188), float) + rng.normal(0, 3, (STOREY, FACE, 1))
        tex = ndimage.uniform_filter(tex, (3, 3, 1), mode="wrap")
    else:
        brick, mortar = _photo_bricks(rng, flags.LEVEL)
        # Old soft red nogging, weathered dark (user, 2026-10-19: "timber on brick needs to be darker").
        tex = bond_texture(brick, mortar, rng, "stretcher") * np.array([0.78, 0.74, 0.74])
    for k, x in enumerate((0, 21, 42)):
        _line(tex, (x + 2, 0), (x + 2, STOREY - 1), 6 if k == 0 else 4, oak, rng)
    for y, w in ((3, 6), (96, 5), (188, 7)):            # wall plate, mid rail, sole plate
        _line(tex, (0, y), (FACE - 1, y), w, oak, rng)
    if rng.random() < 0.5:
        _line(tex, (6, 94), (20, 8), 4, oak, rng)       # a brace in the first bay
    return tex


TILE_W = [11, 11, 10, 11, 11, 10]                       # 64 px: six plain tiles a metre (0.165 m)


def tile_hung(rng):
    """Clay plain tiles hung on battens: a 6 px gauge (32 courses a storey),
    joints broken course to course, each tile's own tone; a shadow under
    each course."""
    tex = np.zeros((STOREY, FACE, 3))
    for c, y in enumerate(range(0, STOREY, 6)):
        x = -5 if c % 2 else 0
        for w in TILE_W * 2:
            if x >= FACE:
                break
            tone = np.array((156, 82, 60)) * rng.uniform(0.84, 1.1) + rng.normal(0, 4, 3)
            for dx in range(w - 1):
                tex[y:y + 6, (x + dx) % FACE] = tone + rng.normal(0, 3, (6, 1))
            tex[y:y + 6, (x + w - 1) % FACE] = tone * 0.6
            x += w
        tex[y + 5] *= 0.62
    return tex


def weatherboard(rng, rgb=(46, 46, 48)):
    """Feather-edged boards lapped 12 px a board (16 a storey): a shadow under each lap, lighter to its foot."""
    tex = np.zeros((STOREY, FACE, 3))
    for k, y in enumerate(range(0, STOREY, 12)):
        tone = np.array(rgb) * rng.uniform(0.94, 1.06)
        for d in range(12):
            tex[y + d] = tone * (0.72 if d == 0 else 0.90 + 0.012 * d) + rng.normal(0, 2, (FACE, 1))
    return tex


STONE = (196, 190, 172)
RUBBED = (172, 92, 66)


def flint_church(rng):
    """A medieval church's walling: flint with blocks of clunch (chalk stone)
    and a few lacing courses of brick (materials.flint; All Saints, Odiham,
    looked at in photos only). A storey by a metre, seamless across the metre."""
    import materials as MAT
    rows, cols = np.mgrid[0:STOREY, 0:FACE] + 0.5
    u, v = cols.ravel() / 64.0, rows.ravel() / 64.0
    shift = float(rng.integers(0, 50)) * 0.22          # each face its own stretch of wall
    rgb = MAT.flint(u, v + shift, np.array([1.0, 0, 0]), {"clunch": 0.09, "lacing": 0.9, "wrap": 1.0})
    return np.clip(rgb.reshape(STOREY, FACE, 3), 0, 255)


def _material_face(name, spec):
    """A storey by a metre of a materials.py material, seamless across the metre."""
    import materials as MAT
    rows, cols = np.mgrid[0:STOREY, 0:FACE] + 0.5
    fn, ms = MAT.material(dict(spec, name=name, wrap=1.0))
    rgb = fn(cols.ravel() / 64.0, rows.ravel() / 64.0, np.array([1.0, 0, 0]), ms)
    return np.clip(rgb.reshape(STOREY, FACE, 3), 0, 255)


ANSTON_TRIM = (206, 180, 146)                          # matches materials.ANSTON, a little paler (was 208,172,128)
MATERIALS = {
    # name: (texture maker, trim)
    "brick_flemish": (red_brick, {"arch": (RUBBED, 6), "cill": (STONE, 2)}),
    "brick_painted": (painted_brick, {"cill": ((60, 58, 56), 2)}),
    "brick_chequer": (chequer_brick, {"arch": (RUBBED, 6), "cill": (STONE, 2)}),
    "render_white": (lambda rng: render(rng), {"cill": ((206, 202, 192), 2)}),
    "render_cream": (lambda rng: render(rng, (1.0, 0.95, 0.80), 196), {"cill": ((206, 202, 192), 2)}),
    "timber_white": (lambda rng: timber_frame(rng), None),
    "timber_brick": (lambda rng: timber_frame(rng, "brick", (50, 41, 35)), None),   # darker oak (user, 2026-10-19)
    "flint": (flint, {"arch": ((150, 84, 64), 5), "cill": ((150, 84, 64), 2)}),
    "flint_church": (flint_church, {"arch": (STONE, 6), "cill": (STONE, 3)}),
    # Gothic Revival: Anston limestone, close blind Perpendicular panelling (the
    # Palace of Westminster's fronts), and plain coursed for courts and towers
    "ashlar_perp": (lambda rng: _material_face("perp_panel", {}), {"cill": (ANSTON_TRIM, 3)}),
    "ashlar_anston": (lambda rng: _material_face("anston", {}), {"arch": (ANSTON_TRIM, 4), "cill": (ANSTON_TRIM, 3)}),
    "tile_hung": (tile_hung, None),
    "weatherboard_black": (lambda rng: weatherboard(rng), None),
    "weatherboard_white": (lambda rng: weatherboard(rng, (222, 222, 216)), None),
}


def kit(name: str) -> tuple[list, list]:
    make, trim = MATERIALS[name]
    rng = np.random.default_rng(sum(map(ord, name)))
    # Variants A and B, each a west face and a north face made apart.
    textures = [np.concatenate([make(rng), make(rng)], 1) for _ in range(2)]
    tiles, depths = [None] * SLOTS, [None] * SLOTS
    for slot, sheet, idx in LAYOUT:
        tex = textures[1] if slot in (4, 5, 6, 7, 12, 13, 14, 15) else textures[0]
        tiles[slot], depths[slot] = paint(vanilla_tile(sheet, idx), tex, trim)
    return tiles, depths


def write(name: str) -> str:
    tiles, depths = kit(name)
    sheet = f"walls_house_uk_01_{name}"
    blank = np.zeros((256, 128, 4), np.uint8)
    F.write_sheet(sheet_file(sheet + ".png"), [t if t is not None else blank for t in tiles])
    # Each slot's vanilla piece, whose properties pack.py copies (wall, window, door, cap).
    slots = [None] * SLOTS
    for slot, vsheet, idx in LAYOUT:
        slots[slot] = f"{vsheet}_{idx}"
    with open(sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f)
    preview(name, tiles)
    return sheet


def preview(name: str, tiles) -> None:
    """The sheet on grass, at half size."""
    os.makedirs(PREVIEW, exist_ok=True)
    img = Image.new("RGBA", (8 * 128, 15 * 256), (70, 90, 70, 255))
    for k, t in enumerate(tiles):
        if t is not None:
            img.alpha_composite(Image.fromarray(t), ((k % 8) * 128, (k // 8) * 256))
    img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(os.path.join(PREVIEW, f"kit_{name}.png"))


def main() -> None:
    for name in sys.argv[1:] or MATERIALS:
        print("wrote", write(name))


if __name__ == "__main__":
    main()
