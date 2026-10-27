"""Up-and-over garage doors for the housing styles' integrated garages (to-do
K09.24; Mapping LOG 2026-10-19).

    python garages.py   (KnoxMap's Python)

Sheet garage_doors_uk_01: a block of 16 slots per wall kit (GARAGE_KITS,
appended only: a block's place fixes its ids), in vanilla's garage door
layout (walls_garage_01 0-5 and 8-13), so each slot takes its vanilla
piece's properties - GarageDoor 1-3 closed and 4-6 open, doorW / doorN, the
CutawayHint - and the game opens the three pieces together:

    +0..2   west wall: right, middle, left (closed)      +8..10  the same, open
    +3..5   north wall: left, middle, right (closed)     +11..13 the same, open

Build lays a door as furniture on the Walls layer (as vanilla's BuildingEd
furniture "Walls - Garage" does), three squares along the garage's street
wall, in place of the house's own wall there (knoxbuild/buildtiles.py), so
each piece is a whole storey of wall: the kit's own texture (the same random
draw wall_kit paints its walls with, so the courses run on from the wall
beside it), the opening (2.2 m by 2.1 m, centred on the three squares) under
a lintel, the door in it. Open, the door is up inside the head and the
opening below is clear.

Doors (DOORS): 1960s-70s dark-stained vertical boarding; 1980s white pressed
steel with vertical ribs; 1950s timber boards painted green.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import housing_kit as HK  # noqa: E402
import plinths as PL  # noqa: E402

WK, W, F = HK.WK, HK.W, HK.F
SHEET = "garage_doors_uk_01"
PX = 64                                     # px a metre
SPAN = 3 * PX                               # the door's three squares
OPEN_W, OPEN_H = 140, 134                   # the opening: 2.2 m x 2.1 m
OPEN_U0 = (SPAN - OPEN_W) // 2              # from the first square's start
OPEN_V0 = WK.STOREY - OPEN_H                # from the storey's top
LINTEL = 6                                  # px over the opening

# kit: (door, lintel rgb); appended only
GARAGE_KITS = [
    ("brick_buff_70s", "boards_stained", (164, 160, 152)),      # e70 estate house: a concrete lintel
    ("brick_brown_70s", "boards_stained", (150, 146, 138)),     # 70s bungalows, council
    ("brick_orange_80s", "steel_white", (128, 70, 50)),         # e80 estate house: a soldier course
    ("brick_red_50s", "boards_green", (160, 154, 146)),         # 1950s-60s private
    ("brick_buff_80s", "steel_white", (150, 82, 62)),           # 1980s housing association, flats
    ("brick_red_80s_band", "steel_white", (112, 58, 46)),       # 1980s starter terrace
]


def _door(kind: str, rng) -> np.ndarray:
    """The door's face, OPEN_H x OPEN_W RGB."""
    h, w = OPEN_H, OPEN_W
    ys, xs = np.mgrid[0:h, 0:w]
    if kind == "steel_white":
        base = np.array((222, 222, 216), float)
        rib = (xs % 7)
        tex = base * np.where(rib == 0, 0.84, np.where(rib == 1, 1.04, 1.0))[..., None]
        tex = tex + rng.normal(0, 1.5, (h, w, 1))
        tex[:, :3] *= 0.9                                   # the frame's edge
        tex[:, -3:] *= 0.9
    else:
        rgb = (86, 62, 46) if kind == "boards_stained" else (58, 92, 64)
        tex = np.zeros((h, w, 3))
        for x0 in range(0, w, 5):
            tone = np.array(rgb, float) * rng.uniform(0.94, 1.06)
            tex[:, x0:x0 + 5] = tone + rng.normal(0, 2, (h, min(5, w - x0), 1))
            tex[:, x0] *= 0.7                               # the groove between boards
        if kind == "boards_green":                          # painted: a little weathered at the foot
            tex[-10:] *= np.linspace(1.0, 0.86, 10)[:, None, None]
    tex[:3] *= 0.7                                          # the head's shadow on the door
    tex[3] *= 0.85
    hx, hy = w // 2, h // 2 + 8                             # the handle: a short dark bar (review)
    tex[hy:hy + 1, hx - 1:hx + 2] = (48, 48, 50)
    return tex


def face_texture(kit: str, door: str, lintel, state: str, north: bool, rng) -> np.ndarray:
    """The door's wall, a metre of plain wall each side: STOREY x (SPAN + 2 PX) RGBA."""
    wall = PL.wall_texture(kit)[:, (64 if north else 0):(128 if north else 64)]
    tex = np.concatenate([np.tile(wall, (1, 5, 1)), np.full((WK.STOREY, 5 * PX, 1), 255.0)], 2)
    u0, u1 = PX + OPEN_U0, PX + OPEN_U0 + OPEN_W
    v0 = OPEN_V0
    tex[v0 - LINTEL:v0, u0 - 4:u1 + 4, :3] = np.array(lintel, float) * np.linspace(1.08, 0.92, LINTEL)[:, None, None]
    if state == "closed":
        tex[v0:, u0:u1, :3] = _door(door, rng)
        tex[v0:, u0:u0 + 2, :3] *= 0.6                      # the reveal: the door is set in the wall
        tex[v0:, u1 - 2:u1, :3] *= 0.75
    else:                                                   # up and over: the door's foot under the head
        tex[v0:, u0:u1, 3] = 0
        lip = _door(door, rng)[-8:] * 0.62
        tex[v0:v0 + 8, u0:u1, :3] = lip
        tex[v0:v0 + 8, u0:u1, 3] = 255
        tex[v0:, u0 - 2:u0, :3] *= 0.6                      # the reveal's two sides
        tex[v0:, u1:u1 + 2, :3] *= 0.75
    return tex


def project(tex: np.ndarray, side: str, piece: int) -> tuple[np.ndarray, np.ndarray]:
    """Piece `piece` (0-2 along the wall: north to south on a west wall, west
    to east on a north wall) of the face texture onto vanilla's plain wall
    piece's outline, as wall_kit.paint does (the columns run on past the
    square's edge into the next piece's)."""
    vt = WK.vanilla_tile("walls_exterior_house_01", 4 if side == "W" else 5)
    alpha = vt[..., 3] > 0
    xs, ys = WK._xs, WK._ys
    s = (63 + 64 * WK.T - xs) / 64 if side == "W" else (xs - 63 + 64 * WK.T) / 64
    _u, v, _panel, dep = WK.FACES[side]
    col = np.clip(PX + PX * piece + np.floor(s * 64).astype(int), 0, tex.shape[1] - 1)
    # Below the storey's foot the wall runs on into the storey below's top
    # (wall_kit.paint); under the opening it is the opening's own foot.
    opening = (col >= PX + OPEN_U0) & (col < PX + OPEN_U0 + OPEN_W)
    row = np.where((v >= WK.STOREY) & ~opening, np.floor(v) - WK.STOREY,
                   np.clip(v, 0, WK.STOREY - 1)).astype(int)
    texel = tex[row, col]
    out = np.zeros((256, 128, 4), np.uint8)
    on = alpha & (texel[..., 3] > 0)
    light = np.full(alpha.shape, W.EAST_LIGHT if side == "W" else 1.0)
    light[v < 0] = W.TOP_LIGHT
    out[on, :3] = np.clip(texel[on, :3] * light[on, None], 0, 255).astype(np.uint8)
    out[on, 3] = 255
    depth = np.where(on, dep, np.nan)
    return out, depth


def kit_pieces(kit: str, door: str, lintel) -> list:
    rng = np.random.default_rng(sum(map(ord, kit + door)))
    tiles = [None] * 16
    for state, off in (("closed", 0), ("open", 8)):
        for side in ("W", "N"):
            tex = face_texture(kit, door, lintel, state, side == "N", rng)
            for piece in range(3):
                # vanilla's order: a west wall's pieces right, middle, left from its north end;
                # a north wall's left, middle, right from its west end
                slot = off + (2 - piece if side == "W" else 3 + piece)
                tiles[slot] = project(tex, side, piece)[0]
    return tiles


def main() -> int:
    tiles, slots = [], []
    blank = np.zeros((256, 128, 4), np.uint8)
    for kit, door, lintel in GARAGE_KITS:
        block = kit_pieces(kit, door, lintel)
        tiles += [t if t is not None else blank for t in block]
        slots += [f"walls_garage_01_{k}" if block[k] is not None else None for k in range(16)]
    F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f)
    # Preview: each kit's closed and open door on a west and a north wall, three squares each.
    img = Image.new("RGBA", (len(GARAGE_KITS) * 2 * 320, 2 * 360), (80, 100, 80, 255))
    for k in range(len(GARAGE_KITS)):
        for r, off in enumerate((0, 8)):
            base = 16 * k + off
            ox = k * 640
            for piece, slot in enumerate((2, 1, 0)):              # west wall, north end first
                img.alpha_composite(Image.fromarray(tiles[base + slot]), (ox + 128 - 64 * piece, r * 360 + 32 * piece))
            for piece, slot in enumerate((3, 4, 5)):              # north wall, west end first
                img.alpha_composite(Image.fromarray(tiles[base + slot]), (ox + 320 + 64 * piece, r * 360 + 32 * piece))
    os.makedirs(HK.PREVIEW, exist_ok=True)
    img.save(os.path.join(HK.PREVIEW, f"{SHEET}.png"))
    have = json.load(open(HK.ORDER, encoding="utf-8"))
    if SHEET not in have:
        have.append(SHEET)
        json.dump(have, open(HK.ORDER, "w", encoding="utf-8"), indent=1)
    print("wrote", SHEET, len(tiles), "slots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
