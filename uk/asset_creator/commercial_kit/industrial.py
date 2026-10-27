"""Industrial units as they stood in 1993 (PLAN.md step 3): their walls and loading doors.

    python industrial.py [--preview]   (KnoxMap's Python)

The 1980s-90s steel portal-frame unit on a trading estate: profiled steel cladding
(supermarket.py's juniper and goosewing grey) over a dado of brick a metre high,
a roller shutter loading door three squares wide in its front, a flat roof.

- walls_commercial_uk_01_unit_juniper, _unit_grey   a whole wall kit each (wall_kit.kit,
  as the housing and superstore kits): cladding with the brick dado; the unit's
  upper storey, where it has one, is plain cladding (supermarket.py's kits).
- commercial_doors_uk_01   roller shutters, 16 slots a kit in vanilla's garage door
  layout (walls_garage_01 0-5 closed, 8-13 open: GarageDoor 1-3 / 4-6, doorW / doorN),
  as housing_kit/garages.py makes the house garages' doors: the kit's own wall round
  an opening 2.6 m high under a steel head, the shutter's galvanised slats in it;
  open, the shutter is rolled into its box under the head. Build lays one as the
  garages are laid (knoxbuild/regional_uk.loading_for, lay_garage).
  Blocks 2-3, superstore_brick / _buff: a superstore's lorry dock in its own brick
  (supermarket.py's kits) - the shutter in a black dock shelter (its PVC curtains'
  frame round the opening) with a yellow and black bumper each side at its foot; at
  ground level, a dock leveller drawn as its steel lip (a raised dock would bar the way).
  Block 4, garage_brick: a repair garage's workshop door, painted cream, in plain red
  brick (supermarket.py's superstore_brick_upper). Block 5, exchange_buff: a telephone
  exchange's van door, galvanised, in buff brick.
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

HK, WK, F = SM.HK, SM.WK, SM.F
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "housing_kit")))
import garages as G  # noqa: E402

DADO = 64                       # px of brick at the foot of the wall (1 m)


def dado(cladding_make, brick_make):
    """Cladding with a metre of brick at its foot and a steel flashing between."""
    def make(rng):
        tex = cladding_make(rng)
        brick = brick_make(rng)
        tex[-DADO:] = brick[-DADO:]
        # the flashing over the brick: a lit top, its face, a shadow under it (review
        # 2026-10-20: it did not show at 1x)
        tex[-DADO - 5:-DADO - 3] = np.array((176, 180, 182), float)
        tex[-DADO - 3:-DADO] = np.array((112, 116, 118), float)
        tex[-DADO:-DADO + 1] *= 0.65
        return tex
    return make


_RED = HK.multi_brick(*SM._RED_STORE, dark={1, 3})
KITS = {
    "unit_juniper": (dado(SM.KITS["cladding_juniper"][0], _RED), {"cill": ((60, 88, 72), 2)}),
    "unit_grey": (dado(SM.KITS["cladding_grey"][0], _RED), {"cill": ((112, 116, 118), 2)}),
}
DOOR_SHEET = "commercial_doors_uk_01"
DOOR_KITS = ["unit_juniper", "unit_grey", "superstore_brick", "superstore_buff",
             "garage_brick", "exchange_buff"]      # appended only: a block's place fixes its ids
# a door kit drawn in another kit's wall: a repair garage's in the superstore's plain brick,
# a telephone exchange's in its buff
DOOR_WALL = {"garage_brick": "superstore_brick_upper", "exchange_buff": "superstore_buff_upper"}
OPEN_H = 166                                       # the shutter's opening, 2.6 m
OPEN_W = 3 * 64 - 16                               # nearly the three squares' width
HEAD = 14                                          # the shutter box over the opening
JAMB = 4                                           # px of steel guide rail each side
# the shutter's slats and the box per kit: on grey cladding a darker, warmer steel so the
# door reads (review 2026-10-20: the grey kit's shutter vanished into its wall)
SHUTTER = {"unit_juniper": ((170, 174, 176), (118, 122, 124)), "unit_grey": ((150, 146, 136), (82, 86, 90)),
           "superstore_brick": ((160, 162, 160), (70, 72, 76)), "superstore_buff": ((160, 162, 160), (70, 72, 76)),
           "garage_brick": ((186, 182, 170), (92, 92, 94)),         # a garage's: painted cream (review: toned down)
           "exchange_buff": ((132, 136, 136), (78, 80, 82))}        # an exchange's: galvanised, dark
SHELTER = 10                                       # px of the dock shelter's frame round the opening
BUMPER = (12, 20)                                  # a dock bumper, w x h px


def _shutter(h, w, rng, base=(168, 172, 174)):
    """Galvanised roller shutter: horizontal slats (bolder at 1x, review 2026-10-20), a bottom rail."""
    ys = np.arange(h)[:, None]
    slat = (ys % 6)
    base = np.array(base, float)
    tex = np.ones((h, w, 3)) * base * np.where(slat == 0, 0.66, np.where(slat == 1, 1.1, 1.0))[..., None]
    tex += rng.normal(0, 1.5, (h, w, 1))
    tex[-5:] *= 0.7                                # the bottom rail
    tex[:, :2] *= 0.75                             # the guides each side
    tex[:, -2:] *= 0.75
    return tex


def face_texture(kit: str, state: str, rng) -> np.ndarray:
    """The door's wall, a metre of plain wall each side: STOREY x 5 m RGBA."""
    make = (KITS.get(kit) or SM.KITS[DOOR_WALL.get(kit, kit)])[0]
    wall = np.concatenate([make(rng) for _ in range(5)], 1)[:, :5 * 64]
    tex = np.concatenate([wall, np.full((WK.STOREY, 5 * 64, 1), 255.0)], 2)
    u0 = 64 + (3 * 64 - OPEN_W) // 2
    u1 = u0 + OPEN_W
    v0 = WK.STOREY - OPEN_H
    slats, box = SHUTTER[kit]
    if kit.startswith("superstore"):
        # the dock shelter: a black frame round the opening, lit along its top
        s0, s1, t0 = u0 - JAMB - SHELTER, u1 + JAMB + SHELTER, v0 - HEAD - SHELTER
        tex[t0:, s0:s1, :3] = np.array((34, 34, 36), float)
        tex[t0:t0 + 2, s0:s1, :3] = (70, 70, 74)
        tex[t0:, s0:s0 + 2, :3] = (56, 56, 60)
    tex[v0 - HEAD:v0, u0 - JAMB:u1 + JAMB, :3] = np.array(box, float) * np.linspace(1.15, 0.8, HEAD)[:, None, None]
    # the guide rails either side, from the box to the ground, in both states
    for c0 in (u0 - JAMB, u1):
        tex[v0:, c0:c0 + JAMB, :3] = np.array(box, float) * np.array([1.2, 1.0, 0.85, 0.7])[None, :JAMB, None]
        tex[v0:, c0:c0 + JAMB, 3] = 255
    if state == "closed":
        tex[v0:, u0:u1, :3] = _shutter(OPEN_H, OPEN_W, rng, slats)
    else:
        tex[v0:, u0:u1, 3] = 0                      # rolled up into its box
        tex[v0:v0 + 4, u0:u1, :3] = 90
        tex[v0:v0 + 4, u0:u1, 3] = 255
    tex[v0:, u0 - 2:u0, :3] *= 0.6                  # the reveal
    tex[v0:, u1:u1 + 2, :3] *= 0.75
    if kit.startswith("superstore"):
        bw, bh = BUMPER
        # on the shelter's foot, inside the door's three squares (what lies past them
        # is never cut into a piece: the first bumpers were lost there)
        for c0 in (64, 4 * 64 - bw):
            bump = np.zeros((bh, bw, 3)) + (214, 172, 40)              # yellow, black hazard bands
            for k in range(0, bh, 8):
                bump[k:k + 4] = (36, 34, 30)
            bump[:, -2:] *= 0.7
            tex[WK.STOREY - bh - 6:WK.STOREY - 6, c0:c0 + bw, :3] = bump
            tex[WK.STOREY - bh - 6:WK.STOREY - 6, c0:c0 + bw, 3] = 255
        tex[-4:, u0:u1, :3] = (120, 122, 120)                          # the dock leveller's lip
        tex[-4:, u0:u1, 3] = 255
    return tex


def _project(tex, side, piece):
    """As garages.project, for this door's opening."""
    old = G.OPEN_U0, G.OPEN_W
    G.OPEN_U0, G.OPEN_W = (3 * 64 - OPEN_W) // 2, OPEN_W
    try:
        return G.project(tex, side, piece)
    finally:
        G.OPEN_U0, G.OPEN_W = old


def door_pieces(kit: str) -> list:
    rng = np.random.default_rng(sum(map(ord, kit + "shutter")))
    tiles = [None] * 16
    for state, off in (("closed", 0), ("open", 8)):
        for side in ("W", "N"):
            tex = face_texture(kit, state, rng)
            for piece in range(3):
                slot = off + (2 - piece if side == "W" else 3 + piece)
                tiles[slot] = _project(tex, side, piece)[0]
    return tiles


def write_kits() -> list[str]:
    out = []
    for name, spec in KITS.items():
        SM.KITS[name] = spec
        out.append(SM.write_kit(name))
    return out


def write_doors() -> str:
    tiles, slots = [], []
    blank = np.zeros((256, 128, 4), np.uint8)
    for kit in DOOR_KITS:
        block = door_pieces(kit)
        tiles += [t if t is not None else blank for t in block]
        slots += [f"walls_garage_01_{k}" if block[k] is not None else None for k in range(16)]
    F.write_sheet(WK.sheet_file(DOOR_SHEET + ".png"), tiles)
    with open(WK.sheet_file(DOOR_SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f)
    return DOOR_SHEET


def preview() -> None:
    """Each unit: 8 squares of its north wall with the shutter closed in the middle, and open."""
    os.makedirs(SM.PREVIEW, exist_ok=True)
    doors = Image.open(WK.sheet_file(DOOR_SHEET + ".png")).convert("RGBA")
    img = Image.new("RGBA", (len(DOOR_KITS) * 760, 2 * 420), (96, 100, 96, 255))
    for k, kit in enumerate(DOOR_KITS):
        wall = Image.open(WK.sheet_file(SM.sheet_of(DOOR_WALL.get(kit, kit)) + ".png")).convert("RGBA")

        def tile(sheet, i):
            cols = sheet.width // 128
            r, c = divmod(i, cols)
            return sheet.crop((c * 128, r * 256, c * 128 + 128, r * 256 + 256))
        for r, off in enumerate((0, 8)):
            ox, oy = k * 760 + 40, r * 420 + 20
            run = [("w", 1), ("w", 1), ("d", 16 * k + off + 3), ("d", 16 * k + off + 4),
                   ("d", 16 * k + off + 5), ("w", 1), ("w", 9), ("w", 1)]
            for i, (what, idx) in enumerate(run):
                img.alpha_composite(tile(wall if what == "w" else doors, idx), (ox + 64 * i, oy + 32 * i))
    img.save(os.path.join(SM.PREVIEW, "industrial_units.png"))


def main(argv) -> int:
    print(write_kits(), write_doors())
    if "--preview" in argv:
        preview()
    order = os.path.join(HERE, "sheets.order.json")
    have = json.load(open(order, encoding="utf-8"))
    for s in [SM.sheet_of(k) for k in KITS] + [DOOR_SHEET]:
        if s not in have:
            have.append(s)
    json.dump(have, open(order, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
