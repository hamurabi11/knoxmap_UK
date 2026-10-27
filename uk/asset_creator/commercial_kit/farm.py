"""Barns and farm buildings as they stood in 1993 (PLAN.md step 4): walls, roofs, doors.

    python farm.py [--preview]   (KnoxMap's Python)

The Hampshire farmyard: the old timber barn (tarred weatherboarding on a brick
plinth, its roof re-covered in corrugated iron or still clay tile) and the
1970s-90s portal-frame shed (Yorkshire boarding over a concrete block dado, a
fibre-cement roof).

- walls_commercial_uk_01_barn_black      weatherboarding (wall_kit.weatherboard, the
  listed maker's barns', a shade lighter) on a plinth of soft red brick; _upper: the
  boarding alone, its upper storey and its gables
- walls_commercial_uk_01_shed_yorkshire  Yorkshire boarding (upright boards with gaps
  between) over a dado of concrete block; _upper: the boarding alone
- walls_commercial_uk_01_granary_staddle  the barn's boarding over the dark gap under a
  granary's floor, a staddle stone each metre (the listed maker's, for a listing's "staddles")
- roofs_uk_03_<covering>, roofs_30_uk_03_<covering>   corrugated_iron (galvanised,
  dull, weathered), fibre_cement (big-six sheet, lichened):
  roof_kit's coverings, as housing_kit's concrete tiles are
- commercial_doors_uk_02   16 slots a kit in vanilla's garage door layout, as
  industrial.py's shutters: barn_black the barn's double cart doors (boarded, strap
  hinged, near the eaves), shed_yorkshire a sliding steel door on its track
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import supermarket as SM  # noqa: E402
import industrial as IN  # noqa: E402

HK, WK, F, G = SM.HK, SM.WK, SM.F, IN.G
RK = HK.RK

PLINTH = 40                     # px of brick under a barn's boarding (0.6 m)
DADO = 76                       # px of block under a shed's boarding (1.2 m)


# review r1: the listed maker's (46, 46, 48) was a black shape at 1x; granary review: a cool
# blue-grey, its laps lost - a warm tarred black-brown, the laps' shadow deeper
TAR = (54, 49, 44)


def boarding(rng):
    """Tarred weatherboarding a shade lighter than the listed maker's, each lap's lit edge brighter."""
    tex = WK.weatherboard(rng, TAR)
    tex[11::12] *= 1.25
    tex[0::12] *= 0.85                                            # each lap's shadow, deeper under the lit edge
    return tex


def plinth(rng):
    """Tarred weatherboarding on soft red brick, the sole plate's shadow between."""
    tex = boarding(rng)
    brick = WK.red_brick(rng)
    tex[-PLINTH:] = brick[-PLINTH:]
    tex[-PLINTH - 3:-PLINTH] = np.array((34, 32, 30), float)      # the sole plate, tarred
    tex[-PLINTH:-PLINTH + 2] *= 0.6                               # its shadow on the brick
    return tex


STADDLE_H = 40                  # px of the gap under a granary's floor (0.6 m)


def staddle(rng):
    """A granary's boarding over the dark gap under its raised floor, a staddle stone in
    it each metre: a tapered shaft under a wider cap (the "mushroom"), weathered stone."""
    tex = boarding(rng)
    h = WK.STOREY
    top = h - STADDLE_H
    tex[top:] = np.array((26, 26, 24), float) + rng.normal(0, 2, (STADDLE_H, WK.FACE, 1))
    tex[top:top + 3] = (40, 38, 34)                               # the sill beam's underside
    stone = np.array((156, 150, 134), float)
    cx, cap = WK.FACE // 2, 8
    for y in range(top + 3, top + 3 + cap):                       # the cap, lit on top
        half = 15 - (y - top - 3) // 3                            # (review: wider over the shaft)
        tex[y, cx - half:cx + half] = stone * (1.12 if y < top + 5 else 0.95)
    for y in range(top + 3 + cap, h):                             # the shaft, widening to its foot
        k = (y - top - 3 - cap) / (STADDLE_H - 3 - cap)
        half = int(4 + 4 * k)
        tex[y, cx - half:cx + half] = stone * np.linspace(1.05, 0.8, 2 * half)[:, None]
    tex[top + 3 + cap - 1, cx - 12:cx + 12] *= 0.7                # the cap's underside
    tex[top + 3 + cap:top + 5 + cap, cx - 6:cx + 6] *= 0.6        # in the cap's shadow
    return tex


def yorkshire(rng, rgb=(112, 98, 74)):
    """Yorkshire boarding: upright sawn boards 14 px with 2 px gaps (the dark shed
    inside showing), each board its own weathered tone, tanalised green-brown."""
    h, w = WK.STOREY, WK.FACE
    tex = np.zeros((h, w, 3))
    for x0 in range(0, w, 16):
        tone = np.array(rgb, float) * rng.uniform(0.88, 1.1)
        tex[:, x0:x0 + 14] = tone * (1 + rng.normal(0, 0.03, (h, 1, 1)))
        tex[:, x0:x0 + 14] *= np.linspace(0.97, 1.03, 14)[None, :, None]   # sawn grain, lit to one side
        tex[:, x0 + 13] *= 0.85
        tex[:, x0 + 14:x0 + 16] = (26, 24, 22)                    # the gap
    tex += rng.normal(0, 2.5, (h, w, 1))
    tex[:4] *= 0.8                                                 # the eaves rail
    return tex


def block(rng):
    """Concrete block (440 x 215 mm, drawn 32 x 14 px so a metre holds two, laid half-bond), grey, the mortar darker."""
    h, w = WK.STOREY, WK.FACE
    tex = np.zeros((h, w, 3))
    base = np.array((138, 136, 128), float)        # review r1: 8% darker
    for r, y0 in enumerate(range(h, 0, -14)):
        off = 16 * (r % 2)
        for x0 in range(-off, w, 32):
            tex[max(y0 - 14, 0):y0, max(x0, 0):x0 + 32] = base * rng.uniform(0.93, 1.06)
        tex[max(y0 - 14, 0):max(y0 - 13, 0)] = (112, 110, 104)
        for x0 in range(-off, w, 32):
            if 0 <= x0 < w:
                tex[max(y0 - 14, 0):y0, x0] = (112, 110, 104)
    return tex + rng.normal(0, 3, (h, w, 1))


def shed(rng):
    tex = yorkshire(rng)
    tex[-DADO:] = block(rng)[-DADO:]
    tex[-10:] *= np.linspace(1.0, 0.82, 10)[:, None, None]       # dirt splashed up the foot
    tex[-DADO - 2:-DADO] = (70, 66, 60)                            # the boarding's bottom rail
    tex[-DADO:-DADO + 2] *= 0.6
    return tex


KITS = {
    "barn_black": (plinth, None),
    "barn_black_upper": (boarding, None),
    "granary_staddle": (staddle, None),
    "shed_yorkshire": (shed, {"cill": ((112, 110, 104), 2)}),
    "shed_yorkshire_upper": (yorkshire, None),
}

ROOF_PATTERN = "roofs_uk_03_{c}"
ROOFS = {
    # Galvanised corrugated iron (3 in corrugations, 5 px), sheets 0.8 m wide and 2.4 m
    # down the slope, lapped; thirty years on, each sheet its own way from dull grey to
    # rust (per sheet, as roof_kit's weather tones are per tile).
    # Self-review: orange rust sheet by sheet against bright grey read as plaid; a dull
    # grey, the rust brown and near it in light; in game (r1) still a checkerboard: the tones closer.
    # Review r1: still a diagonal checkerboard, and flat brown-grey: dull galvanised
    # (#8a8f8f), the sheets within 4%. No rusted sheets: a roof is one piece repeated, so a
    # rusted sheet stood at the same place in every square, a row of rust (r4).
    "corrugated_iron": dict(course=40, unit=52, rgb=(130, 134, 134), spread=0.02, kind="units",
                            stagger=False, roll=5, roll_depth=0.22, shadow=(2, 0.7), joint=0.92,
                            weather=dict(tones=[(1, 1, 1), (0.97, 0.97, 0.96), (1.02, 1.01, 1.0)],
                                         p=[0.4, 0.35, 0.25], sat=1.0, lichen=0.0, moss=0.0)),
    # Big-six fibre-cement sheet (146 mm corrugations, 9 px): pale grey, lichened,
    # dark where the moss holds at the laps.
    # Review r1: a checker on the tile grid, too clean and pale: #9a9c92 darkened, the
    # tones within 3%, more moss and lichen.
    "fibre_cement": dict(course=44, unit=64, rgb=(140, 142, 132), spread=0.015, kind="units",
                         stagger=False, roll=9, roll_depth=0.2, shadow=(2, 0.62), joint=0.85,
                         # (in game, r3: a tone a tile read as a checkerboard on the narrow spans)
                         weather=dict(tones=[(1, 1, 1), (0.97, 0.97, 0.96), (0.99, 0.99, 0.98)],
                                      p=[0.5, 0.3, 0.2], sat=0.8, lichen=0.009, moss=0.01,
                                      clump=True, lichen_rgb=(168, 164, 120))),
}

DOOR_SHEET = "commercial_doors_uk_02"
DOOR_KITS = ["barn_black", "shed_yorkshire"]        # appended only: a block's place fixes its ids
OPEN_W = 3 * 64 - 12
OPEN_H = {"barn_black": 172, "shed_yorkshire": 176}
HEAD = 8


def _boards(h, w, rng, rgb):
    """A leaf of upright tarred boards, 8 px each, the joints dark."""
    tex = np.zeros((h, w, 3))
    for x0 in range(0, w, 8):
        tex[:, x0:x0 + 8] = np.array(rgb, float) * rng.uniform(0.9, 1.1)
        tex[:, x0] *= 0.6
    return tex + rng.normal(0, 2, (h, w, 1))


def _sheet_door(h, w, rng):
    """A sliding door of galvanised corrugated steel: upright corrugations, a frame."""
    x = np.arange(w)
    wave = 1 + 0.16 * np.sin(2 * np.pi * x / 5)
    tex = np.ones((h, w, 3)) * np.array((120, 122, 118), float) * wave[None, :, None]   # review r1: 20% darker
    tex *= np.where(rng.random(w) < 0.12, 0.9, 1.0)[None, :, None]      # dull streaks down it
    tex += rng.normal(0, 2, (h, w, 1))
    tex[:, :3] = tex[:, -3:] = tex[:4] = tex[-4:] = (96, 98, 96)  # the frame
    tex[h // 2 - 2:h // 2 + 1] = (110, 112, 108)                  # the mid rail
    return tex


def face_texture(kit: str, state: str, rng) -> np.ndarray:
    """The door's wall, a metre of plain wall each side: STOREY x 5 m RGBA."""
    make = KITS[kit][0]
    wall = np.concatenate([make(rng) for _ in range(5)], 1)[:, :5 * 64]
    tex = np.concatenate([wall, np.full((WK.STOREY, 5 * 64, 1), 255.0)], 2)
    oh = OPEN_H[kit]
    u0 = 64 + (3 * 64 - OPEN_W) // 2
    u1 = u0 + OPEN_W
    v0 = WK.STOREY - oh
    if kit == "barn_black":
        # an oak head beam over the opening, the posts either side
        tex[v0 - HEAD:v0, u0 - 6:u1 + 6, :3] = np.array((82, 66, 50), float) * np.linspace(1.15, 0.8, HEAD)[:, None, None]
        for c0 in (u0 - 6, u1):
            tex[v0:, c0:c0 + 6, :3] = np.array((82, 66, 50), float) * np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7])[None, :, None]
        if state == "closed":
            half = OPEN_W // 2
            leaves = np.concatenate([_boards(oh, half, rng, (64, 58, 52)), _boards(oh, OPEN_W - half, rng, (64, 58, 52))], 1)
            leaves[:, half - 1:half + 1] = (20, 18, 16)            # the meeting stiles
            for y in (int(oh * 0.15), int(oh * 0.5), int(oh * 0.85)):   # strap hinges, rusted
                leaves[y:y + 3, 2:half * 2 // 3] = (90, 58, 40)
                leaves[y:y + 3, OPEN_W - half * 2 // 3:OPEN_W - 2] = (90, 58, 40)   # dark rust (review r1)
            leaves[:, :1] *= 0.6
            tex[v0:, u0:u1, :3] = leaves
        else:
            tex[v0:, u0:u1, 3] = 0                                  # the leaves swung back, out of sight
    else:
        # the sliding door's track over the opening, on the boarding (past the door's three
        # squares is never cut into a piece)
        tex[v0 - HEAD:v0 - 2, u0 - 40:u1 + 40, :3] = np.array((84, 86, 86), float) * np.linspace(1.2, 0.8, HEAD - 2)[:, None, None]
        tex[v0 - 2:v0, u0 - 40:u1 + 40, :3] *= 0.6
        if state == "closed":
            tex[v0:, u0:u1, :3] = _sheet_door(oh, OPEN_W, rng)
        else:
            tex[v0:, u0:u1, 3] = 0                                  # slid along, off the opening
    tex[v0:, u0 - 2:u0, :3] *= 0.6                                  # the reveal
    tex[v0:, u1:u1 + 2, :3] *= 0.75
    return tex


def _project(tex, side, piece):
    old = G.OPEN_U0, G.OPEN_W
    G.OPEN_U0, G.OPEN_W = (3 * 64 - OPEN_W) // 2, OPEN_W
    try:
        return G.project(tex, side, piece)
    finally:
        G.OPEN_U0, G.OPEN_W = old


def door_pieces(kit: str) -> list:
    rng = np.random.default_rng(sum(map(ord, kit + "door")))
    tiles = [None] * 16
    for state, off in (("closed", 0), ("open", 8)):
        for side in ("W", "N"):
            tex = face_texture(kit, state, rng)
            for piece in range(3):
                tiles[off + (2 - piece if side == "W" else 3 + piece)] = _project(tex, side, piece)[0]
    return tiles


def write_doors() -> str:
    tiles, slots = [], []
    blank = np.zeros((256, 128, 4), np.uint8)
    for kit in DOOR_KITS:
        block_ = door_pieces(kit)
        tiles += [t if t is not None else blank for t in block_]
        slots += [f"walls_garage_01_{k}" if block_[k] is not None else None for k in range(16)]
    F.write_sheet(WK.sheet_file(DOOR_SHEET + ".png"), tiles)
    with open(WK.sheet_file(DOOR_SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f)
    return DOOR_SHEET


def write_roofs() -> list[str]:
    return [n for c in ROOFS for n in _write_roof(c)]


def _write_roof(covering: str) -> list[str]:
    """housing_kit.write_roof's way, under this kit's sheet names (roofs_uk_03_*)."""
    c = ROOFS[covering]
    top = max(RK.planes(WK.vanilla_tile("roofs_02", 0))[2])
    written = []
    for pattern, vsheet in RK.SHEETS.items():
        name = pattern.replace("uk_01", "uk_03").format(c=covering)
        tiles = []
        for i in range(128):
            t = WK.vanilla_tile(vsheet, i)
            if vsheet == "roofs_02" and i in RK.SKIP_45:
                t = np.zeros_like(t)
            tiles.append(RK.paint(t, c, None, 1000 * i + 13, top)[0])
        F.write_sheet(WK.sheet_file(name + ".png"), tiles)
        with open(WK.sheet_file(name + ".slots.json"), "w", encoding="utf-8") as f:
            json.dump([f"{vsheet}_{i}" if t[..., 3].any() else None for i, t in enumerate(tiles)], f)
        written.append(name)
    return written


def write_kits() -> list[str]:
    out = []
    for name, spec in KITS.items():
        SM.KITS[name] = spec
        out.append(SM.write_kit(name))
    return out


def _tile(sheet, i):
    cols = sheet.width // 128
    r, c = divmod(i, cols)
    return sheet.crop((c * 128, r * 256, c * 128 + 128, r * 256 + 256))


def preview() -> None:
    """Each building: 8 squares of north wall, its door closed then open in the middle,
    under a strip of its roof (a south slope, vanilla roofs_02 0), at the game's scale."""
    os.makedirs(SM.PREVIEW, exist_ok=True)
    doors = Image.open(WK.sheet_file(DOOR_SHEET + ".png")).convert("RGBA")
    img = Image.new("RGBA", (2 * 760, 2 * 460), (96, 100, 96, 255))
    for k, kit in enumerate(DOOR_KITS):
        wall = Image.open(WK.sheet_file(SM.sheet_of(kit) + ".png")).convert("RGBA")
        for r, off in enumerate((0, 8)):
            ox, oy = k * 760 + 40, r * 460 + 60
            run = [("w", 1), ("w", 1), ("d", 16 * k + off + 3), ("d", 16 * k + off + 4),
                   ("d", 16 * k + off + 5), ("w", 1), ("w", 9), ("w", 1)]
            for i, (what, idx) in enumerate(run):
                img.alpha_composite(_tile(wall if what == "w" else doors, idx), (ox + 64 * i, oy + 32 * i))
    img.save(os.path.join(SM.PREVIEW, "farm_walls.png"))
    # the roofs: a 6 x 3 patch of each covering's south slope (roofs_02 0), beside clay tile
    names = [ROOF_PATTERN.format(c=c) for c in ROOFS] + ["roofs_uk_01_tile"]
    img = Image.new("RGBA", (len(names) * 560, 420), (96, 100, 96, 255))
    for n, name in enumerate(names):
        sheet = Image.open(WK.sheet_file(name + ".png")).convert("RGBA")
        t = _tile(sheet, 0)
        for a in range(6):
            for b in range(3):
                img.alpha_composite(t, (n * 560 + 120 + 64 * a - 64 * b, 40 + 32 * a + 32 * b))
    img.save(os.path.join(SM.PREVIEW, "farm_roofs.png"))


def main(argv) -> int:
    print(write_kits(), write_roofs(), write_doors())
    if "--preview" in argv:
        preview()
    order = os.path.join(HERE, "sheets.order.json")
    have = json.load(open(order, encoding="utf-8"))
    for s in [SM.sheet_of(k) for k in KITS] + _roof_sheets() + [DOOR_SHEET]:
        if s not in have:
            have.append(s)
    json.dump(have, open(order, "w", encoding="utf-8"), indent=1)
    return 0


def _roof_sheets() -> list[str]:
    return [p.replace("uk_01", "uk_03").format(c=c) for c in ROOFS for p in RK.SHEETS]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
