"""Housing style tile sets by sub-type (to-do K09.22; Mapping LOG 2026-10-19).

    python housing_kit.py [part ...] [--preview]   (KnoxMap's Python)

The sub-types (Mapping/Building_types/src/area/subtypes.py) each want a look of
their own. This draws the pieces with the listed building maker's machinery
(wall_kit's painting onto vanilla's wall shapes, roof_kit's coverings over
vanilla's roof pieces, window_kit's window states and projection) and writes
sheets of its own, so the two makers do not tread on each other:

- walls_house_uk_02_<material>   a wall kit (walls, window-shaped pieces, caps)
- roofs_uk_02_<covering>, roofs_30_uk_02_<covering>   a roof covering
- fixtures_windows_uk_02, _03     the sub-types' windows (closed, open, smashed, glass gone), 7 a sheet

Their order in the pack is housing_kit/sheets.order.json (Tile_creator's pack.py
appends it to KIT_SHEETS): a sheet's place fixes its tile ids, so new sheets go
at the end. Every part is reviewed by an agent against the game's look before
it is used (user, 2026-10-19).

The first set (K09.22, by count): the 1960s-70s estate house (e70_house) -
buff multi-brown stretcher-bond brick, brown concrete interlocking tiles at a
shallow pitch, dark-stained timber picture windows with top vents. Then the
commonest sub-type of each main type (LOG 2026-10-19 #8), and (K09.26,
2026-10-19) council housing (inter-war, post-war, 1960s-70s, 1980s housing
association), bungalows, 1980s starter terraces, flats, and mock-Tudor
gables (K09.23): the styles are palettes/english.json's uk_council_*,
uk_social_80s, uk_terrace_80s*, uk_bungalow_*, uk_flats_*, uk_*_tudor.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
MAKERS = os.path.normpath(os.path.join(HERE, "..", "listed_building_maker", "makers"))
sys.path.insert(0, MAKERS)
import wall_kit as WK  # noqa: E402
import roof_kit as RK  # noqa: E402
import window_kit as WIN  # noqa: E402

F, W = WK.F, WK.W
ORDER = os.path.join(HERE, "sheets.order.json")
PREVIEW = os.path.join(HERE, "preview")


# --- walls -------------------------------------------------------------------------------------

def buff_stretcher(rng):
    """1960s-70s facing brick: sand-faced buff and brown multis in stretcher
    bond, flush-pointed. Each brick one of a few fired tones (light buff,
    sand, toffee, a darker brown) in a narrow band, a fine sanded speckle.
    Review (2026-10-19): a wide spread and mortar lighter than the bricks read
    as stripes beside vanilla; the mortar is now just darker than the bricks,
    and no two dark bricks sit side by side."""
    tones = np.array([(186, 152, 112), (170, 138, 98), (158, 124, 88), (194, 162, 122), (140, 106, 76)], float)
    share = np.array([0.30, 0.28, 0.22, 0.12, 0.08])
    dark = {2, 4}
    last = [-1]

    def brick(w, h):
        k = rng.choice(len(tones), p=share)
        while k in dark and last[0] in dark:
            k = rng.choice(len(tones), p=share)
        last[0] = k
        face = np.full((h, w, 3), tones[k] * rng.uniform(0.97, 1.03)) + rng.normal(0, 3, (h, w, 1))
        speck = rng.random((h, w)) < 0.05
        face[speck] *= rng.uniform(0.86, 0.92)               # the sanded face's darker grains
        return face
    tex = WK.bond_texture(brick, (160, 150, 134), rng, "stretcher") * 0.96
    tex[:2] *= 0.8                                           # the wall's top edge, outlined as vanilla's
    return tex


KISS = (92, 58, 62)             # a Fletton's kiss mark (review round 2: darker)


def multi_brick(tones, share, mortar, kind="stretcher", sigma=3, dark=(), header=None, kiss=0.0):
    """A facing brick of a few fired tones in a narrow band, a fine speckle,
    the mortar just darker than the bricks, no two dark bricks side by side
    (the 1970s brick's review, 2026-10-19: wide tones and light mortar read
    as stripes beside vanilla). `kiss`: the share of bricks with a Fletton's
    "kiss mark", a darker slanting band where the next brick touched it in
    the kiln."""
    tones = np.array(tones, float)

    def make(rng):
        last = [-1]

        def brick(w, h):
            k = rng.choice(len(tones), p=share)
            while k in dark and last[0] in dark:
                k = rng.choice(len(tones), p=share)
            last[0] = k
            face = np.full((h, w, 3), tones[k] * rng.uniform(0.97, 1.03)) + rng.normal(0, sigma, (h, w, 1))
            speck = rng.random((h, w)) < 0.05
            face[speck] *= rng.uniform(0.86, 0.92)
            if kiss and rng.random() < kiss:
                # review (2026-10-19): a dark purple-brown band, not a pale smear
                ys, xs = np.mgrid[0:h, 0:w]
                c = rng.uniform(0.2 * w, 0.8 * w)
                band = np.exp(-((xs - c - 1.2 * (ys - h / 2)) / 1.3) ** 2)[..., None] * 0.85
                face = face * (1 - band) + np.array(KISS, float) * band
            return face
        head = (lambda w, h: brick(w, h) * 0.94) if header == "same" else None
        tex = WK.bond_texture(brick, mortar, rng, kind, head)
        tex[:2] *= 0.8                                       # the wall's top edge, outlined as vanilla's
        return tex
    return make


def pebbledash(rng, paint=(214, 204, 178), glint=(1.10, 1.16), cap=255):
    """1930s pebbledash, painted cream: stones thrown into the render in
    2 px clumps, lit on top, a 2 px shadow under each; a broad mottle.
    Review (2026-10-19): single-pixel stones 3-8% off the paint averaged
    away at 1x (it read as flat paint), and a dark top edge and dirt line
    on every storey read as panels."""
    tex = np.full((192, WK.FACE, 3), paint, float) + rng.normal(0, 5, (192, WK.FACE, 1))
    stones = ndimage.binary_dilation(rng.random((192, WK.FACE)) < 0.12, np.array([[0, 1], [1, 1]], bool))
    lit = stones & (rng.random((192, WK.FACE)) < 0.6)
    tex[lit] *= rng.uniform(*glint)
    under = (np.roll(stones, 1, 0) | np.roll(stones, 2, 0)) & ~stones
    tex[under] *= rng.uniform(0.76, 0.82)
    mottle = ndimage.gaussian_filter(rng.normal(0, 1, (192, WK.FACE)), 3, mode="wrap")
    tex *= (1 + 0.02 * mottle / max(mottle.std(), 1e-6))[..., None]   # round 2: smaller and fainter (blotches at ±4%)
    return np.minimum(tex, cap)


def boards_70s(rng, rgb=(90, 64, 46), board=5):
    """Dark-stained horizontal boarding, 5 px boards (near vanilla siding's):
    a lap shadow under each, its top edge lit; a stained timber string course
    at the floor line (review, 2026-10-19: the listed maker's 12 px boards at
    a darker stain read as a glazed curtain wall)."""
    tex = np.zeros((192, WK.FACE, 3))
    for y in range(0, 192, board):
        tone = np.array(rgb, float) * rng.uniform(0.95, 1.05)
        for d in range(min(board, 192 - y)):
            k = 1.12 if d == 0 else (0.65 if d == board - 1 else 1.0)
            tex[y + d] = tone * k + rng.normal(0, 2, (WK.FACE, 1))
    tex[-2:] = (60, 44, 34)
    return tex


def panel_70s(rng, rgb=(204, 202, 194)):
    """A plain white panel (painted board or render), warm off-white, a little
    darker under the eaves, a cover strip at each joint, a white string course
    at the floor line (review: a flat 217 sheet glared beside vanilla)."""
    tex = np.full((192, WK.FACE, 3), rgb, float) + rng.normal(0, 3, (192, WK.FACE, 1))
    tex *= (1 - 0.04 * np.exp(-np.arange(192) / 24))[:, None, None]
    tex[:, :1] *= 0.88                                       # the cover strip over each joint
    tex[:2] *= 0.85                                          # the eaves' edge
    tex[-2:] = (230, 228, 220)
    return tex


def mock_tudor(rng, black=(50, 40, 34), render=(225, 221, 208)):
    """Inter-war and Edwardian mock-Tudor: black-stained boards planted on
    white render - studs a third of a metre apart, a rail at mid storey and a
    plate at the top and foot (Tudorbethan, 1900-1939; not the listed
    maker's oak frame: thinner, blacker, the render whiter and finer)."""
    # Review (2026-10-19): 4 px studs 21 px apart read as a barcode at 1x;
    # now 8 px boards half a metre apart, a heavy sole rail at the gable's
    # foot, the infill a fine roughcast, the boards a dark brown with grain.
    tex = pebbledash(rng, render, (1.04, 1.07), 236)
    bk = np.array(black, float)

    def board(ys, xs, along_x):
        n = (xs.stop - xs.start) if along_x else (ys.stop - ys.start)
        grain = rng.normal(0, 3, (1, n, 1)) if along_x else rng.normal(0, 3, (n, 1, 1))
        tex[ys, xs] = bk * rng.uniform(0.94, 1.06) + grain
    for x in (0, 32):                                   # studs, 8 px
        board(slice(0, 192), slice(x, x + 8), False)
        tex[:, x] *= 1.12                               # the board's lit edge
        tex[:, (x + 8) % WK.FACE] *= 0.86               # the render's shade beside it
    for y, h in ((92, 6), (182, 10)):                   # mid rail, sole rail
        board(slice(y, y + h), slice(0, WK.FACE), True)
        tex[y] *= 1.12
        if y + h < 192:
            tex[y + h] *= 0.84
    return tex


def shiplap(rng, rgb=(230, 226, 214), board=16):
    tex = np.zeros((192, WK.FACE, 3))
    for y in range(0, 192, board):
        tone = np.array(rgb, float) * rng.uniform(0.97, 1.02)
        for d in range(min(board, 192 - y)):
            k = 0.72 if d < 2 else (1.04 if d == 2 else 1.0 - 0.004 * d)
            tex[y + d] = tone * k + rng.normal(0, 2, (WK.FACE, 1))
    return tex


def banded(base, band, rows):
    """`base`'s bricks with the rows `rows` (whole courses) laid in `band`'s:
    both drawn from the same random state, so the band's bricks and joints
    are the wall's own (the two makers must take the same draws: the same
    number of tones, shares and dark set)."""
    import copy

    def make(rng):
        twin = copy.deepcopy(rng)
        tex = base(rng)
        tex[rows] = band(twin)[rows]
        return tex
    return make


def floor_band(base, rgb=(156, 152, 144), rows=5):
    """A concrete floor band (the slab's edge, 1950s-60s blocks of flats) at
    the foot of each storey: a lit top edge, a shadow line under it."""
    def make(rng):
        tex = base(rng)
        tex[-rows:] = np.array(rgb, float) + rng.normal(0, 5, (rows, WK.FACE, 1))
        tex[-rows:, ::21] *= 0.85                       # the slab's joints
        tex[-rows] *= 1.06
        tex[-rows - 1] *= 0.8
        return tex
    return make


# The window hole's foot (window_kit.opening(9): rows 9-118) is the cill line;
# a band of two courses under it (wall_kit.COURSES: rows 120-129).
CILL_BAND = slice(120, 140)      # review (2026-10-19): 2 courses read as a pin-stripe; now 4

# Inter-war council cottages: plain, rough red-brown "commons", darker and
# browner than the Edwardian pressed brick; cavity walls, so stretcher bond.
# Review (2026-10-19): at (126, 80, 67) it could not be told from the 1950s
# and Edwardian reds; now a purple-brown multi, a wider spread, about one
# brick in eight burnt dark, a few orange.
_IW_COUNCIL = ([(122, 72, 70), (110, 66, 66), (134, 84, 76), (84, 54, 56), (150, 92, 66), (116, 78, 72)],
               [0.28, 0.24, 0.18, 0.12, 0.06, 0.12], (102, 94, 88))
# Post-war council: Fletton commons, salmon-pink with dark kiss marks.
_FLETTON = ([(170, 104, 86), (160, 96, 80), (178, 112, 92), (150, 90, 78), (166, 106, 90)],
            [0.3, 0.25, 0.2, 0.12, 0.13], (136, 128, 120))
# 1960s-70s council and bungalows: a dark brown multi.
_BROWN_70S = ([(112, 80, 64), (100, 72, 60), (124, 90, 70), (88, 64, 54), (134, 100, 78)],
              [0.3, 0.25, 0.2, 0.12, 0.13], (108, 100, 90))
# 1980s housing association and flats: a pale yellow buff, red brick arches.
_BUFF_80S = ([(200, 172, 120), (188, 160, 110), (208, 182, 130), (176, 148, 102), (192, 158, 112)],
             [0.3, 0.25, 0.2, 0.12, 0.13], (164, 150, 128))
# 1980s starter terraces: a red multi (redder, darker than the estate house's
# orange), a band of buff bricks under the cills.
_RED_80S = ([(160, 78, 58), (148, 70, 54), (168, 88, 64), (134, 66, 52), (154, 84, 66)],
            [0.3, 0.25, 0.2, 0.12, 0.13], (142, 128, 116))


WALLS = {
    # name: (texture maker, trim: window arches and cills as wall_kit.MATERIALS)
    "brick_buff_70s": (buff_stretcher, {"cill": ((150, 146, 138), 2)}),       # a concrete or tile cill
    # Edwardian: smooth, bright red pressed facing bricks, Flemish bond with
    # red headers, rubbed red arches and stone cills.
    # Review (2026-10-19): crimson, apart from the 1980s orange; a warm lime
    # mortar near the bricks' value (a pale one read as stripes); the arches
    # brighter and the cills deeper, to show at 1x.
    # User (2026-10-19): "a bit too crimson ... tone it down": a duller,
    # browner red, still redder and darker than the 1980s orange.
    "brick_red_edw": (multi_brick([(152, 74, 58), (142, 66, 52), (160, 82, 64), (128, 62, 50)],
                                  [0.4, 0.3, 0.2, 0.1], (124, 100, 88), "flemish", dark={3}, header="same"),
                      {"arch": ((198, 106, 76), 7), "cill": (WK.STONE, 3)}),
    # 1930s: cream-painted pebbledash, a painted cill.
    "pebbledash_cream": (pebbledash, {"cill": ((226, 222, 210), 2)}),
    # The 1960s-70s estate house's upper-floor panel between its brick ends
    # (user's references, 2026-10-19): dark-stained timber boarding, or a
    # plain white panel (painted render or board).
    "boards_stained_70s": (lambda rng: boards_70s(rng), {"cill": ((60, 44, 34), 2)}),
    "panel_white_70s": (lambda rng: panel_70s(rng), {"cill": ((206, 204, 198), 2)}),
    # Edwardian upper floors: white roughcast (the 1930s pebbledash, finer, white).
    # review (2026-10-19): highlights clipped at 255 and glared; warmer, capped, a quarter less speckle
    "roughcast_white": (lambda rng: pebbledash(rng, (220, 217, 206), (1.06, 1.10), 236), {"cill": ((226, 222, 210), 2)}),
    # 1950s: red-brown mottled facing brick in stretcher bond, grey cement mortar, tile cills.
    "brick_red_50s": (multi_brick([(142, 76, 58), (128, 68, 54), (154, 88, 66), (118, 72, 60), (148, 96, 74)],
                                  [0.3, 0.25, 0.2, 0.12, 0.13], (124, 116, 108), dark={1, 3}),
                      {"cill": ((74, 50, 42), 3)}),                   # review: darker and deeper, to show
    # 1980s: orange-red and brown multi facing brick, stretcher bond, a soldier course over windows.
    "brick_orange_80s": (multi_brick([(176, 98, 66), (164, 88, 60), (152, 86, 62), (184, 112, 76), (132, 76, 56)],
                                     [0.3, 0.26, 0.2, 0.14, 0.1], (150, 138, 124), dark={2, 4}),
                         {"arch": ((128, 70, 50), 6), "cill": ((136, 80, 60), 2)}),   # review: the soldier course darker, to show
    # --- K09.26 (2026-10-19): council housing, flats, bungalows, 1980s terraces ---
    # Inter-war council (Addison / Wheatley cottages): rough red-brown commons,
    # a plain tile cill; their upper floors or whole fronts often in
    # unpainted grey roughcast.
    "brick_red_iwc": (multi_brick(*_IW_COUNCIL, dark={3}, sigma=4), {"cill": ((96, 58, 50), 3)}),
    "roughcast_grey": (lambda rng: pebbledash(rng, (184, 180, 168)), {"cill": ((200, 196, 186), 2)}),
    # Post-war council: Flettons, a concrete cill.
    "brick_fletton": (multi_brick(*_FLETTON, dark={1, 3}, kiss=0.17), {"cill": ((168, 164, 156), 3)}),
    # 1950s-60s council flats: the same brick, the floor slab's edge at each floor.
    "brick_fletton_band": (floor_band(multi_brick(*_FLETTON, dark={1, 3}, kiss=0.17)), {"cill": ((168, 164, 156), 3)}),
    # 1960s-70s council and bungalows: dark brown multi, a concrete cill.
    "brick_brown_70s": (multi_brick(*_BROWN_70S, dark={1, 3}), {"cill": ((150, 146, 138), 2)}),
    # 1980s housing association and flats: pale buff, red soldier arches, a brick-on-edge cill.
    # round 2: the soldier course red on the buff, bricks on end, 4 px past the opening each side
    "brick_buff_80s": (multi_brick(*_BUFF_80S, dark={3}), {"soldier": ((150, 72, 56), 8), "cill": ((142, 84, 64), 2)}),
    # 1980s starter terrace: red multi, a buff band under the cills, a dark soldier course.
    "brick_red_80s_band": (banded(multi_brick(*_RED_80S, dark={1, 3}),
                                  multi_brick([(196, 170, 120), (186, 160, 112), (202, 178, 128), (178, 150, 106),
                                               (190, 162, 116)], _BUFF_80S[1], _RED_80S[2], dark={1, 3}), CILL_BAND),
                           {"soldier": ((104, 50, 42), 8), "cill": ((112, 58, 46), 2)}),
    # K09.23 mock-Tudor: black-stained boards on white render, for gables
    # (a style's caps) and upper floors.
    "mocktudor": (lambda rng: mock_tudor(rng), None),
    # The 1960s-70s council upper floor (review, 2026-10-19: the listed maker's
    # 12 px boards read too fine there): white shiplap, 16 px boards, a 2 px shadow.
    "weatherboard_white_70s": (lambda rng: shiplap(rng), {"cill": ((206, 204, 198), 2)}),
}


def write_wall(name: str) -> str:
    WK.MATERIALS["_housing_" + name] = WALLS[name]
    tiles, depths = WK.kit("_housing_" + name)
    del WK.MATERIALS["_housing_" + name]
    sheet = f"walls_house_uk_02_{name}"
    blank = np.zeros((256, 128, 4), np.uint8)
    F.write_sheet(WK.sheet_file(sheet + ".png"), [t if t is not None else blank for t in tiles])
    slots = [None] * WK.SLOTS
    for slot, vsheet, idx in WK.LAYOUT:
        slots[slot] = f"{vsheet}_{idx}"
    with open(WK.sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f)
    _sheet_preview(sheet, tiles, 8)
    return sheet


# --- roofs -------------------------------------------------------------------------------------

ROOFS = {
    # Marley Modern / Redland 49: big, near-flat interlocking concrete tiles in
    # a dull brown, a faint granule texture, no lichen yet (fifteen years old).
    # Review (2026-10-19): rolls, a deeper course shadow and joints in line make
    # it concrete, not shingle; weathered gently (concrete greys, it does not fade
    # purple as clay does).
    # Round 2: the profile, not dark lines, carries it (a lined-up grid read as plaid).
    "concrete_brown": dict(course=11, unit=16, rgb=(100, 79, 66), spread=0.05, kind="units",
                           stagger=False, roll=8, roll_depth=0.18, shadow=(2, 0.55), joint=0.82,
                           weather=dict(tones=[(1, 1, 1), (0.92, 0.92, 0.9), (1.05, 1.03, 0.98)],
                                        p=[0.6, 0.25, 0.15], sat=0.85, lichen=0.0025, moss=0.003,
                                        clump=True, lichen_rgb=(132, 128, 104))),
    # 1950s-60s concrete plain tile (Marley Plain): small, flat, a dull
    # brown-grey, joints broken course to course as clay plain tile.
    # Review: darker and greyer than the 1970s brown, lichen fainter.
    "concrete_plain": dict(course=6, unit=10, rgb=(84, 72, 66), spread=0.04, kind="units",
                           shadow=(1, 0.6), joint=0.82,
                           weather=dict(tones=[(1, 1, 1), (0.9, 0.9, 0.9), (1.04, 1.03, 1.0)],
                                        p=[0.6, 0.25, 0.15], sat=0.7, lichen=0.0012, moss=0.002,
                                        clump=True, lichen_rgb=(110, 108, 90))),
    # 1980s concrete pantile (Redland Grovebury, Marley Ludlow): a big S-roll
    # each tile, laid in line, a brown-red through-colour, barely weathered.
    # Review: a sine roll as wide as the tile, a strong course shadow and joints
    # read as plaid; a skewed roll (crest 35%), the joint in the trough, the
    # courses longer. Round 2: the course ends back (it read as corrugated sheet).
    "pantile_brown": dict(course=16, unit=20, rgb=(104, 72, 60), spread=0.04, kind="units",
                          stagger=False, roll=20, roll_skew=0.35, roll_depth=0.32, shadow=(2, 0.68), joint=1.0,
                          weather=dict(tones=[(1, 1, 1), (0.94, 0.93, 0.92), (1.04, 1.02, 1.0)],
                                       p=[0.65, 0.2, 0.15], sat=0.9, lichen=0.0015, moss=0.0015,
                                       clump=True, lichen_rgb=(132, 128, 104))),
    # 1960s-70s council and flats (K09.26): Marley Modern / Redland 49 in
    # grey, the same big interlocking tile as concrete_brown, weathered darker.
    # Review (2026-10-19): at the brown one's 11 px courses it read as felt;
    # 4-5 courses a face, two bold rolls a tile, a tile-to-tile spread.
    "concrete_grey": dict(course=17, unit=22, rgb=(92, 90, 88), spread=0.07, kind="units",
                          stagger=False, roll=11, roll_depth=0.22, shadow=(2, 0.5), joint=0.9,   # round 2: joints softer (plaid)
                          weather=dict(tones=[(1, 1, 1), (0.9, 0.9, 0.9), (1.05, 1.04, 1.02)],
                                       p=[0.6, 0.25, 0.15], sat=0.6, lichen=0.003, moss=0.003,
                                       clump=True, lichen_rgb=(128, 126, 106))),
}


def write_roof(covering: str) -> list[str]:
    c = ROOFS[covering]
    written = []
    # Both sets shaded against the 45-degree set's lit slope: each against its
    # own made the 30-degree set 10% darker, where vanilla's is lighter (review).
    top = max(RK.planes(WK.vanilla_tile("roofs_02", 0))[2])
    for pattern, vsheet in RK.SHEETS.items():
        name = pattern.replace("uk_01", "uk_02").format(c=covering)
        vt = [WK.vanilla_tile(vsheet, i) for i in range(128)]
        tiles = []
        for i, t in enumerate(vt):
            if vsheet == "roofs_02" and i in RK.SKIP_45:
                t = np.zeros_like(t)
            tiles.append(RK.paint(t, c, None, 1000 * i + 11, top)[0])
        F.write_sheet(WK.sheet_file(name + ".png"), tiles)
        with open(WK.sheet_file(name + ".slots.json"), "w", encoding="utf-8") as f:
            json.dump([f"{vsheet}_{i}" if t[..., 3].any() else None for i, t in enumerate(tiles)], f)
        _sheet_preview(name, tiles, 8)
        written.append(name)
    return written


# --- windows ------------------------------------------------------------------------------------

WINDOW_SHEET = "fixtures_windows_uk_02"
STAINED = (74, 52, 38)          # dark-stained softwood (Sadolin teak), the 1970s look
WHITE = (232, 230, 222)         # painted softwood
UPVC = (238, 238, 234)          # white uPVC, new in the 1980s
STEEL = (214, 206, 182)         # Crittall steel, painted cream (review: apart from the 1980s white uPVC)
WINDOWS = [
    # name, WindowShape, frame, kind. All shape 9: the hole the wall kit's
    # window pieces cut (vanilla's tall sash), filled to its head - a shorter
    # window left the top of the hole open (review, 2026-10-19).
    ("picture_70s", 9, STAINED, "picture"),      # 1967-82: a big fixed pane, a top-hung vent across its head
    ("sash_edwardian", 9, WHITE, "sash_edw"),    # 1901-14: a six-pane upper sash over a plain lower one
    ("casement_30s", 9, WHITE, "casement_30s"),  # 1919-39: two casements, leaded top lights over them
    ("crittall_50s", 9, STEEL, "crittall"),      # 1945-66: slim steel casements, horizontal glazing bars
    ("upvc_80s", 9, UPVC, "upvc"),               # 1983-95: white uPVC, a top opener over two fixed lights
    ("timber_80s", 9, STAINED, "upvc"),          # the same in dark-stained softwood: most 1980s estates kept theirs in 1993 (review)
    ("bay_hidden", 9, (0, 0, 0), "hidden"),      # the house window behind a bay: a working window, drawn by the bay (QA)
    # K09.26 (2026-10-19), appended: slots move otherwise
    ("casement_iwc", 9, WHITE, "casement_small"),  # inter-war council: white timber casements, small panes
    ("casement_pw", 9, WHITE, "casement_plain"),   # post-war council: plain white casements, a top vent each side
    ("picture_white_70s", 9, WHITE, "picture"),    # 1960s-70s council: the picture window, painted white
]
# kind: frame width (px on the face), transom (share down the opening; None:
# none), the lights below it split by a mullion, the glazing bars
# (cols, rows) above / below, leaded top lights, what opens.
LAYOUTS = {
    "picture": dict(f=4, transom=0.28, mullion=False, bars=((1, 1), (1, 1)), lead=False, opens="top"),
    "upvc": dict(f=5, transom=0.28, mullion=True, bars=((1, 1), (1, 1)), lead=False, opens="top"),
    "sash_edw": dict(f=3, transom=0.5, mullion=False, bars=((3, 2), (1, 1)), lead=False, opens="sash"),
    "casement_30s": dict(f=3, transom=0.3, mullion=True, bars=((1, 1), (1, 1)), lead=True, opens="side"),
    "crittall": dict(f=2, transom=None, mullion=True, bars=((1, 1), (1, 3)), lead=False, opens="side"),
    "casement_small": dict(f=3, transom=0.3, mullion=True, bars=((1, 1), (1, 3)), lead=False, opens="side"),
    "casement_plain": dict(f=4, transom=0.26, mullion=True, bars=((1, 1), (1, 1)), lead=False, opens="side"),
}
PANE = (96, 114, 114)           # paler and clearer than the listed maker's old glass


def _pane(a, x0, y0, x1, y1, lift=0):
    """Glass, a sky reflection band, the head rail's shade along its top."""
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return
    ys, xs = np.mgrid[0:h, 0:w]
    refl = np.clip(1 - np.abs((xs + ys * 0.6) / max(w, 1) - 0.45) * 3, 0, 1) * 24
    rgb = np.array(PANE, float) + lift + refl[..., None]
    rgb[:2] *= 0.82
    a[y0:y1, x0:x1, :3] = np.clip(rgb, 0, 255)
    a[y0:y1, x0:x1, 3] = 200


def _smash(a, x0, y0, x1, y1, rng, teeth=7):
    """Shards left round the frame: jagged teeth from each edge (value noise
    along it), a corner triangle or two, the broken edge catching the light,
    two cracks running into the frame."""
    h, w = y1 - y0, x1 - x0
    ys, xs = np.mgrid[0:h, 0:w]

    def edge_noise(n):
        pts = rng.random(rng.integers(6, 10))
        return np.interp(np.linspace(0, len(pts) - 1, n), np.arange(len(pts)), pts)
    reach = np.minimum.reduce([
        ys - 1 - teeth * edge_noise(w)[None, :] ** 2,
        h - 1 - ys - 1 - teeth * edge_noise(w)[None, :] ** 2,
        xs - 1 - teeth * edge_noise(h)[:, None] ** 2,
        w - 1 - xs - 1 - teeth * edge_noise(h)[:, None] ** 2])
    keep = reach < 0
    for _ in range(rng.integers(1, 3)):                       # corner triangles
        L = rng.integers(8, 13)
        cx, cy = rng.choice([0, w - 1]), rng.choice([0, h - 1])
        keep |= (np.abs(xs - cx) + np.abs(ys - cy) * 0.7) < L
    rim = keep & ~ndimage.binary_erosion(keep)
    sub = a[y0:y1, x0:x1]
    sub[rim, :3] = np.clip(sub[rim, :3].astype(int) + 35, 0, 255)
    for _ in range(2):                                        # cracks into the frame from the shards
        t = rng.uniform(0, 2 * np.pi)
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        for r in range(0, max(w, h)):
            x, y = int(cx + r * np.cos(t)), int(cy + r * np.sin(t))
            if not (0 <= x < w and 0 <= y < h):
                break
            if keep[y, x]:
                sub[y, x, :3] = np.clip(sub[y, x, :3].astype(int) + 30, 0, 255)
    sub[~keep, 3] = 0


def draw_window(kind: str, frame, state: str, box, rng) -> np.ndarray:
    """The window on a 64 x 192 west-face texture, as window_kit.draw: a
    bevelled frame and sill, then the lights of its LAYOUTS entry."""
    if kind == "hidden":                          # one pixel, all but transparent: the pack keeps a sprite
        a = np.zeros((192, 64, 4), np.uint8)
        a[box[2] + 4, (box[0] + box[1]) // 2] = (0, 0, 0, 1)
        return a
    L = LAYOUTS[kind]
    a = np.zeros((192, 64, 4), np.uint8)
    u0, u1, v0, v1 = box
    f, fr = L["f"], np.array(frame, float)
    lit, shade = tuple(np.minimum(fr + 30, 255)), tuple(fr * 0.7)
    WIN._rect(a, u0, v0, u1, v1, frame)
    WIN._rect(a, u0, v0, u1, v0 + 1, lit)                       # bevel: lit top and left
    WIN._rect(a, u0, v0, u0 + 1, v1, lit)
    WIN._rect(a, u0, v1 - 1, u1, v1, shade)                     # shade: bottom and right
    WIN._rect(a, u1 - 1, v0, u1, v1, shade)
    WIN._rect(a, u0 - 1, v1, u1 + 1, v1 + 3, tuple(fr * 0.85))  # the sill
    WIN._rect(a, u0 - 1, v1, u1 + 1, v1 + 1, tuple(np.minimum(fr + 40, 255)))
    ix0, iy0, ix1, iy1 = u0 + f, v0 + f, u1 - f, v1 - f
    rail = max(2, f - 1)
    # The lights: (x0, y0, x1, y1, bars, upper?)
    lights = []
    if L["transom"] is None:
        top_y = iy0
    else:
        t = iy0 + round((iy1 - iy0) * L["transom"])
        WIN._rect(a, ix0, t - rail // 2, ix1, t - rail // 2 + rail, frame)
        top_y = t - rail // 2 + rail
    midx = (ix0 + ix1) // 2
    if L["transom"] is not None:
        y1 = t - rail // 2
        if L["lead"]:                                               # a top light over each casement
            WIN._rect(a, midx - rail // 2, iy0, midx - rail // 2 + rail, y1, frame)
            lights += [(ix0, iy0, midx - rail // 2, y1, L["bars"][0], True),
                       (midx - rail // 2 + rail, iy0, ix1, y1, L["bars"][0], True)]
        else:
            lights.append((ix0, iy0, ix1, y1, L["bars"][0], True))
    if L["mullion"]:
        WIN._rect(a, midx - rail // 2, top_y, midx - rail // 2 + rail, iy1, frame)
        lights += [(ix0, top_y, midx - rail // 2, iy1, L["bars"][1], False),
                   (midx - rail // 2 + rail, top_y, ix1, iy1, L["bars"][1], False)]
    else:
        lights.append((ix0, top_y, ix1, iy1, L["bars"][1], False))
    opener = {"top": 0, "sash": len(lights) - 1, "side": len(lights) - (2 if L["mullion"] else 1)}[L["opens"]]
    for k, (x0, y0, x1, y1, (cols, rows), upper) in enumerate(lights):
        if state == "open" and k == opener:
            if L["opens"] == "top":                                 # pushed out on its top hinge
                WIN._rect(a, x0, y0, x1, y1, (40, 40, 40))         # review: (30, 24, 20) read as a black bar
                g = y0 + 3 + round((y1 - y0 - 3) * 0.6)
                _pane(a, x0, y0 + 3, x1, g, lift=25)
                WIN._rect(a, x0 - 1, g, x1 + 1, g + 2, frame)
            elif L["opens"] == "sash":                              # the lower sash raised behind the upper
                WIN._rect(a, x0, y0, x1, y1, (0, 0, 0), 0)
                WIN._rect(a, x0, y0, x1, y0 + rail, frame)
            else:                                                   # swung out: its edge only, on the hinge
                WIN._rect(a, x0, y0, x1, y1, (0, 0, 0), 0)
                WIN._rect(a, x0, y0, x0 + 2, y1, tuple(fr * 0.8))
            continue
        leaded = L["lead"] and upper
        if state == "noglass":
            WIN._rect(a, x0, y0, x1, y1, (0, 0, 0), 0)
        else:
            _pane(a, x0, y0, x1, y1, lift=48 if leaded else 0)   # review: leaded lights paler than the panes, not a grille
            if state == "smashed":
                _smash(a, x0, y0, x1, y1, rng, 9 if k == 0 else 7)    # each light breaks its own way
        if leaded:
            WIN.lattice(a, x0, y0, x1, y1, (90, 90, 90), step=5)    # review: lighter, finer (read as blinds)
        elif cols > 1 or rows > 1:
            WIN.bars(a, x0, y0, x1, y1, cols, rows, frame, width=1 if f <= 2 else 2)
    return a


# A window sheet takes vanilla's layout (window_kit.STATES): its glass-gone
# row holds 8 windows, so the windows fill sheets of PER_SHEET in turn
# (fixtures_windows_uk_02, _03 ...); the first sheet's ids never move.
PER_SHEET = 7


def window_sheets() -> list[str]:
    n = (len(WINDOWS) + PER_SHEET - 1) // PER_SHEET
    return [f"fixtures_windows_uk_{2 + k:02d}" for k in range(n)]


def write_windows() -> list[str]:
    rng = np.random.default_rng(3)                 # one draw across the sheets, in order
    written = []
    for n, sheet in enumerate(window_sheets()):
        group = WINDOWS[n * PER_SHEET:(n + 1) * PER_SHEET]
        tiles, slots = [None] * 104, [None] * 104
        for k, (name, shape, frame, kind) in enumerate(group):
            box = WIN.opening(shape)
            for state, off in WIN.STATES.items():
                tex = draw_window(kind, frame, state, box, rng)
                for j, side in enumerate(("W", "N")):
                    slot = off + 2 * k + j
                    tiles[slot] = WIN.project(tex, side)
                    slots[slot] = {"like": f"fixtures_windows_white_{WIN.VANILLA_SLOT[state] + j}",
                                   "WindowShape": shape, "name": name}
        blank = np.zeros((256, 128, 4), np.uint8)
        F.write_sheet(WK.sheet_file(sheet + ".png"), [t if t is not None else blank for t in tiles])
        with open(WK.sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
            json.dump(slots, f, indent=0)
        img = Image.new("RGBA", (128 * 2 * len(group), 256 * 4), (176, 140, 100, 255))
        for r, off in enumerate(WIN.STATES.values()):
            for c in range(2 * len(group)):
                img.alpha_composite(Image.fromarray(tiles[off + c]), (c * 128, r * 256))
        os.makedirs(PREVIEW, exist_ok=True)
        img.save(os.path.join(PREVIEW, f"{sheet}.png"))
        written.append(sheet)
    return written


def window_tile(name: str) -> str:
    """A window's first tile (its west face, closed), as a sprite name."""
    k = [w[0] for w in WINDOWS].index(name)
    return f"{window_sheets()[k // PER_SHEET]}_{2 * (k % PER_SHEET):03d}"


# --- the order, previews -------------------------------------------------------------------------

def sheets() -> list[str]:
    """Every sheet this kit makes, in pack order (new ones only ever appended)."""
    out = [f"walls_house_uk_02_{m}" for m in WALLS] + window_sheets()
    for c in ROOFS:
        out += [f"roofs_uk_02_{c}", f"roofs_30_uk_02_{c}"]
    return out


def write_order() -> None:
    have = json.load(open(ORDER, encoding="utf-8")) if os.path.exists(ORDER) else []
    for s in sheets():
        if s not in have:
            have.append(s)
    json.dump(have, open(ORDER, "w", encoding="utf-8"), indent=1)


def _sheet_preview(name: str, tiles, cols: int) -> None:
    os.makedirs(PREVIEW, exist_ok=True)
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGBA", (cols * 128, rows * 256), (70, 90, 70, 255))
    for k, t in enumerate(tiles):
        if t is not None:
            img.alpha_composite(Image.fromarray(t), ((k % cols) * 128, (k // cols) * 256))
    img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(os.path.join(PREVIEW, f"{name}.png"))


def main() -> int:
    parts = [a for a in sys.argv[1:] if not a.startswith("--")] or ["walls", "roofs", "windows"]
    if "walls" in parts:
        for m in WALLS:
            print("wrote", write_wall(m))
    if "roofs" in parts:
        for c in ROOFS:
            print("wrote", write_roof(c))
    if "windows" in parts:
        print("wrote", write_windows())
    write_order()
    return 0


if __name__ == "__main__":
    sys.exit(main())
