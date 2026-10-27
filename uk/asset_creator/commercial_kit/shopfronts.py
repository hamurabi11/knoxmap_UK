"""British shop fronts as they stood in 1993 (PLAN.md step 1).

    python shopfronts.py [--preview]   (KnoxMap's Python)

Sheet shopfronts_uk_01: a block of 12 slots per finish (FINISHES; blocks appended
only, so a block's place fixes its ids), in SET order:

    +0, +1    pane W, N            the run's middle squares
    +2, +3    start W, N           the run's first square: a pilaster and console at its start
    +4, +5    end W, N             the last square: the pilaster at its end
    +6, +7    door frame W, N      round the shop door (vanilla's glazed door goes in it)
    +8        north-west corner
    +9        south-east post
    +10, +11  (spare)

Each piece is drawn as an elevation, a storey high (192 rows, 64 px a metre), and
painted into the outline of the house wall piece it stands in for (wall_kit.paint:
walls_exterior_house_01, as the wall kits are), so it meets the kits' walls
exactly. Its glass keeps its own alpha (see-through, as vanilla's shop glass is);
everything else is opaque. Each slot takes the properties and depth picture of vanilla's
glass shop wall (walls_commercial_01: a wall with a locked window in it, the
door-frame wall, the corner and post), so the game treats the front as its own.

Two kinds of front:
- traditional (Victorian / Edwardian, the high street's older buildings): a
  cornice, the fascia board, a bead, transom lights, the shop window with a
  glazing bar, a cill and a panelled stall riser, all in one paint; pilasters with
  console brackets at the ends of the run;
- modern (1960s-80s aluminium): a deep box fascia, a thin frame, one sheet of
  glass, a low tiled stall riser.

The fascia board is left plain in the front's own paint: a named shop's sign
(fascias.py) is laid over it on the wall furniture layer.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MAKERS = os.path.normpath(os.path.join(HERE, "..", "listed_building_maker", "makers"))
sys.path.insert(0, MAKERS)
import wall_kit as WK  # noqa: E402

W = WK.W
SHEET = "shopfronts_uk_01"
PREVIEW = os.path.join(HERE, "preview")
STOREY = 192
M = 64                         # px a metre

# Rows (0 at the top of the storey) of the parts, in metres from the top.
def _r(m):
    return int(round(m * M))


# (name, kind, paint rgb). Traditional paints are the 1990s heritage colours; the
# modern ones the aluminium's finish.
FINISHES = [
    ("green", "trad", (34, 64, 48)),
    ("maroon", "trad", (92, 30, 38)),
    ("navy", "trad", (28, 40, 72)),
    ("black", "trad", (30, 30, 32)),
    ("cream", "trad", (222, 212, 184)),
    ("oak", "trad", (112, 74, 44)),
    ("bronze", "modern", (88, 70, 52)),
    ("silver", "modern", (172, 174, 176)),
    # a pub's (review 2026-10-20: the shop front read as a showroom), in its brewery's
    # paint (export_regional.uk_pubs): appended, a block's place fixes its ids
    ("pub_green", "pub", (34, 64, 48)),
    ("pub_navy", "pub", (28, 40, 72)),
    ("pub_maroon", "pub", (92, 30, 38)),
]
# Each slot of a block: (offset, piece, side, the house wall piece whose outline it
# takes, the vanilla commercial piece whose properties it takes).
SET = [
    (0, "pane", "W", 4, "walls_commercial_01_0"), (1, "pane", "N", 5, "walls_commercial_01_1"),
    (2, "start", "W", 4, "walls_commercial_01_0"), (3, "start", "N", 5, "walls_commercial_01_1"),
    (4, "end", "W", 4, "walls_commercial_01_0"), (5, "end", "N", 5, "walls_commercial_01_1"),
    (6, "door", "W", 12, "walls_commercial_01_8"), (7, "door", "N", 13, "walls_commercial_01_9"),
    (8, "corner", "NW", 6, "walls_commercial_01_18"), (9, "post", "SE", 7, "walls_commercial_01_2"),
]
BLOCK = 12
# The shop door laid in the door frame: vanilla's glazed doors (W, N, W open, N open).
DOORS = {"trad": ["fixtures_doors_01_36", "fixtures_doors_01_37", "fixtures_doors_01_38", "fixtures_doors_01_39"],
         # aluminium: the dark-framed two-pane door (review 2026-10-20: not the wooden one)
         "modern": ["fixtures_doors_01_48", "fixtures_doors_01_49", "fixtures_doors_01_50", "fixtures_doors_01_51"]}
DOORS["pub"] = DOORS["trad"]
GLASS = (62, 70, 74)           # a shop window's glass, seen from the street by day: a grey cast
GLASS_A = 118                  # its alpha: the shop shows through (review 2026-10-20: 150 hid it)
GLINT_SHARE = 0.35             # the share of panes with a glint (review: one on every pane read as stripes)


def _shade(rgb, k):
    return tuple(int(max(0, min(255, c * k))) for c in rgb)


def _lift(rgb, k):
    """A highlight that still shows on a dark paint (review: the black front's mouldings vanished)."""
    add = 30 if sum(rgb) < 200 else 0
    return tuple(int(max(0, min(255, c * k + add))) for c in rgb)


def _glass(tex, r0, r1, c0, c1, rng):
    """Glass: a grey cast, a soft sky reflection across its top, now and then one thin glint."""
    h, w = r1 - r0, c1 - c0
    if h <= 0 or w <= 0:
        return
    yy, xx = np.mgrid[0:h, 0:w]
    col = np.array(GLASS, float) * (1.0 + 0.30 * np.clip(1 - yy / max(h * 0.45, 1), 0, 1)[..., None])
    if rng.random() < GLINT_SHARE:
        off = rng.uniform(0, 40)
        glint = np.abs(xx + yy * 0.6 - off - 12) < 1.2
        col[glint] = col[glint] * 1.25 + 14
    col += rng.normal(0, 1.5, (h, w, 1))
    tex[r0:r1, c0:c1, :3] = np.clip(col, 0, 255)
    tex[r0:r1, c0:c1, 3] = GLASS_A


def _box(tex, r0, r1, c0, c1, rgb, rng=None, grain=2.0):
    col = np.full((r1 - r0, c1 - c0, 3), rgb, float)
    if rng is not None:
        col += rng.normal(0, grain, (r1 - r0, c1 - c0, 1))
    tex[r0:r1, c0:c1, :3] = np.clip(col, 0, 255)
    tex[r0:r1, c0:c1, 3] = 255


def _line(tex, r, c0, c1, rgb, rows=1):
    tex[r:r + rows, c0:c1, :3] = rgb
    tex[r:r + rows, c0:c1, 3] = 255


def trad(piece: str, paint, rng) -> np.ndarray:
    """A traditional front's elevation, one square (64 px) wide. Review (2026-10-20): a
    bar down every square's middle and transoms every third of a metre made a long run
    read as a greenhouse - the window is now one light a square, the mullion between
    squares; the pilasters wider and lighter, so each shop's ends show at 1x."""
    tex = np.zeros((STOREY, 64, 4), float)
    dark, light = _shade(paint, 0.70), _lift(paint, 1.22)
    # cornice and fascia board (0 - 0.62 m), bead under it
    _box(tex, 0, _r(0.12), 0, 64, light, rng)
    _line(tex, _r(0.12), 0, 64, dark, 2)
    _box(tex, _r(0.12) + 2, _r(0.58), 0, 64, paint, rng)
    _line(tex, _r(0.58), 0, 64, light, 2)
    _line(tex, _r(0.58) + 2, 0, 64, dark, 1)
    # transom lights (0.62 - 0.92 m): two over each square's window
    top = _r(0.64)
    _glass(tex, top, _r(0.90), 0, 64, rng)
    for c in (0, 31, 63):
        _box(tex, top, _r(0.90), max(c - 1, 0), min(c + 2, 64), paint)
    _box(tex, _r(0.90), _r(0.96), 0, 64, paint, rng)           # transom bar
    _line(tex, _r(0.96), 0, 64, dark, 1)
    # the window (0.96 - 2.35 m): one light, the mullion at the square's edges
    _glass(tex, _r(0.96) + 1, _r(2.35), 0, 64, rng)
    _box(tex, _r(0.96), _r(2.35), 0, 2, paint)
    _box(tex, _r(0.96), _r(2.35), 62, 64, paint)
    # cill and stall riser (2.35 - 3 m): a raised panel in a frame
    _box(tex, _r(2.35), _r(2.42), 0, 64, light, rng)
    _line(tex, _r(2.42), 0, 64, dark, 1)
    _box(tex, _r(2.42) + 1, STOREY, 0, 64, paint, rng)
    _box(tex, _r(2.52), _r(2.90), 6, 58, _lift(paint, 1.08), rng)
    _line(tex, _r(2.52), 6, 58, dark, 1)
    _line(tex, _r(2.90), 6, 58, light, 1)
    tex[_r(2.52):_r(2.90), 6, :3] = dark
    tex[_r(2.52):_r(2.90), 57, :3] = light
    _line(tex, STOREY - 3, 0, 64, _shade(paint, 0.5), 3)          # its foot, a plinth's shadow
    if piece in ("start", "end"):
        c0, c1 = (0, 14) if piece == "start" else (50, 64)
        _box(tex, _r(0.58), STOREY, c0, c1, _lift(paint, 1.18), rng)    # the pilaster
        inner = c1 - 1 if piece == "start" else c0
        outer = c0 if piece == "start" else c1 - 1
        tex[_r(0.58):STOREY, inner, :3] = dark
        tex[_r(0.58):STOREY, outer, :3] = _lift(paint, 1.4)
        _box(tex, _r(2.70), STOREY, c0, c1, _shade(paint, 0.9), rng)     # its base
        _box(tex, 0, _r(0.64), c0, c1, _lift(paint, 1.35), rng)          # the console bracket
        for k in range(0, _r(0.5), 5):
            _line(tex, _r(0.08) + k, c0 + 2, c1 - 2, dark, 1)
    return tex


ETCHED = (196, 200, 194)


def pub(piece: str, paint, rng) -> np.ndarray:
    """A Victorian pub's front: the traditional one with its lower lights etched (the
    frosted glass a passer-by cannot see through, a diaper of clear lines in it) under
    a glazing bar, and a heavy mullion between each square's window."""
    tex = trad(piece, paint, rng)
    dark, light = _shade(paint, 0.70), _lift(paint, 1.22)
    r0, r1 = _r(1.55), _r(2.35)
    c0, c1 = (14, 64) if piece == "start" else (0, 50) if piece == "end" else (0, 64)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    col = np.array(ETCHED, float) + rng.normal(0, 2, (r1 - r0, c1 - c0, 1))
    clear = ((xx + yy) % 9 == 0) | ((xx - yy) % 9 == 0)
    col[clear] = np.array(GLASS, float) * 1.2
    tex[r0:r1, c0:c1, :3] = col
    tex[r0:r1, c0:c1, 3] = 225
    _box(tex, r0 - 2, r0 + 1, c0, c1, paint)                          # the glazing bar over the etching
    _line(tex, r0 + 1, c0, c1, dark, 1)
    for c in ((0, 4), (60, 64)):                                      # heavy mullions
        if c[0] >= c0 - 1 and c[1] <= c1 + 1:
            _box(tex, _r(0.96), _r(2.35), c[0], c[1], light)
            tex[_r(0.96):_r(2.35), c[0] if c[0] else c[1] - 1, :3] = dark
    return tex


def modern(piece: str, paint, rng) -> np.ndarray:
    """An aluminium front's elevation."""
    tex = np.zeros((STOREY, 64, 4), float)
    dark, light = _shade(paint, 0.70), _shade(paint, 1.18)
    # box fascia (0 - 0.70 m): a deep box, its lit face plain
    _box(tex, 0, _r(0.06), 0, 64, light, rng, 1.0)
    _box(tex, _r(0.06), _r(0.66), 0, 64, _shade(paint, 1.05), rng, 1.0)
    _line(tex, _r(0.66), 0, 64, dark, 3)
    # frame and one sheet of glass (0.72 - 2.62 m)
    _box(tex, _r(0.70), _r(0.76), 0, 64, paint, rng, 1.0)
    _glass(tex, _r(0.76), _r(2.62), 0, 64, rng)
    _box(tex, _r(0.76), _r(2.62), 0, 2, paint)
    _box(tex, _r(0.76), _r(2.62), 62, 64, paint)
    # low stall riser in small dark tiles (2.62 - 3 m)
    _box(tex, _r(2.62), _r(2.66), 0, 64, light)
    riser = np.array((70, 66, 64), float)
    _box(tex, _r(2.66), STOREY, 0, 64, riser, rng, 3.0)
    for r in range(_r(2.66), STOREY, 6):
        _line(tex, r, 0, 64, _shade(riser, 0.75), 1)
    for c in range(0, 64, 8):
        tex[_r(2.66):STOREY, c, :3] = _shade(riser, 0.75)
    if piece in ("start", "end"):
        c0, c1 = (0, 6) if piece == "start" else (58, 64)
        _box(tex, _r(0.70), _r(2.62), c0, c1, _lift(paint, 1.1), rng, 1.0)   # a wider end stile
    return tex


def elevation(kind: str, piece: str, paint, rng) -> np.ndarray:
    return {"trad": trad, "pub": pub}.get(kind, modern)(piece, paint, rng)


def paint_piece(mask, face: np.ndarray) -> np.ndarray:
    """Our elevation into a wall piece's outline, keeping the glass's alpha."""
    tex = np.concatenate([face, face], axis=1)                        # the same front on the W and N faces
    rgb, _depth = WK.paint(mask, np.ascontiguousarray(tex[..., :3]))
    a = mask[..., 3] > 0
    alpha = np.zeros(a.shape, float)
    xs = np.nonzero(a.any(0))[0]
    split = 999 if len(xs) and xs.max() <= 67 else (-1 if len(xs) and xs.min() >= 58 else 64)
    cols = np.arange(128)[None, :]
    for side, here in (("W", cols < split), ("N", cols >= split)):
        u, v, _panel, _ = WK.FACES[side]
        vi = np.clip(v, 0, STOREY - 1).astype(int)
        al = tex[vi, u + (0 if side == "W" else 64), 3]
        al = np.where(v < 0, 255, al)                                  # the wall's thin top
        m = a & here
        alpha[m] = al[m]
    out = rgb.copy()
    out[..., 3] = np.where(a, alpha, 0).astype(np.uint8)
    return out


def build(preview: bool = False) -> dict:
    rng = np.random.default_rng(1993)
    tiles, slots = [], []
    for name, kind, paint in FINISHES:
        block = [None] * BLOCK
        sblock = [None] * BLOCK
        for off, piece, side, house_idx, like in SET:
            mask = WK.vanilla_tile("walls_exterior_house_01", house_idx)
            face = elevation(kind, "pane" if piece in ("corner", "post") else piece, paint, rng)
            block[off] = paint_piece(mask, face)
            sblock[off] = like             # a plain name: its properties and depth (pack.py)
        tiles.extend(t if t is not None else np.zeros((256, 128, 4), np.uint8) for t in block)
        slots.extend(sblock)
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=0)
    if preview:
        _preview(tiles)
    return {"sheet": SHEET, "tiles": len(tiles)}


# The game's own Spiffo's front, as it is (user, 2026-10-20: "the vanilla theming,
# made to fit UK commercial units"): its red-banded window walls, door walls,
# corner and post (location_restaurant_spiffos_01) along the unit's shop front run.
_SP = "location_restaurant_spiffos_01_"
SPIFFOS_FRONT = {"finish": "spiffos", "kind": "modern",
                 "pane": {"West": _SP + "005", "North": _SP + "017"},
                 "start": {"West": _SP + "004", "North": _SP + "016"},
                 "end": {"West": _SP + "006", "North": _SP + "018"},
                 "door_frame": {"West": _SP + "002", "North": _SP + "003"},
                 "corner": _SP + "007", "post": _SP + "022",
                 "door": dict(zip(("West", "North", "WestOpen", "NorthOpen"), DOORS["modern"]))}
# ...and its neon "Spiffo's" and burger over the front (location_restaurant_spiffos_03,
# six squares, in run order as fascias.py's), or on a shorter unit its small wall logo
# (spiffos_02 6, 7); no board between, the front's own red band shows.
_SN = "location_restaurant_spiffos_03_"
SPIFFOS_FASCIA = {"sign": "spiffos", "plain": None,
                  # the neon is drawn low in its tiles: the game hangs it a storey up
                  "versions": [{"length": 6, "level": 1, "W": [f"{_SN}{k:03d}" for k in (13, 12, 11, 10, 9, 8)],
                                "N": [f"{_SN}{k:03d}" for k in range(6)]},
                               {"length": 1, "W": ["location_restaurant_spiffos_02_007"],
                                "N": ["location_restaurant_spiffos_02_006"]}]}


def entry(finish: str) -> dict:
    """Build's storefront set for a finish: the wall entries for the run's pieces and its door."""
    if finish == "spiffos":
        return dict(SPIFFOS_FRONT)
    k = [f[0] for f in FINISHES].index(finish)
    kind = FINISHES[k][1]
    base = k * BLOCK

    def t(off):
        return f"{SHEET}_{base + off}"
    return {"finish": finish, "kind": kind,
            "pane": {"West": t(0), "North": t(1)}, "start": {"West": t(2), "North": t(3)},
            "end": {"West": t(4), "North": t(5)}, "door_frame": {"West": t(6), "North": t(7)},
            "corner": t(8), "post": t(9),
            "door": dict(zip(("West", "North", "WestOpen", "NorthOpen"), DOORS[kind]))}


def _vanilla(name: str) -> Image.Image:
    sheet, idx = name.rsplit("_", 1)
    return Image.fromarray(WK.vanilla_tile(sheet, int(idx)))


def _preview(tiles) -> None:
    """Each finish as a run of five squares (start, pane, door, pane, end) on a north
    wall - the face the camera sees from the south - with vanilla's door in its frame,
    a pavement under it and a brick wall beside it; four finishes a row."""
    os.makedirs(PREVIEW, exist_ok=True)
    floor = _vanilla("blends_street_01_96") if os.path.exists(os.path.join(WK.VANILLA, "blends_street_01.png")) else None
    brick = Image.fromarray(WK.vanilla_tile("walls_exterior_house_01", 5))
    cw, ch = 520, 380
    img = Image.new("RGBA", (cw * 4, ch * -(-len(FINISHES) // 4)), (96, 100, 96, 255))
    for r, (name, kind, _p) in enumerate(FINISHES):
        base = r * BLOCK
        ox, oy = (r % 4) * cw + 40, (r // 4) * ch + 20
        run = [base + 3, base + 1, base + 7, base + 1, base + 5]
        pieces = [brick] + [Image.fromarray(tiles[k]) for k in run] + [brick]
        for i, piece in enumerate(pieces):                                # a N wall runs down-right on screen
            x, y = ox + 64 * i, oy + 32 * i
            if floor is not None:
                img.alpha_composite(floor, (x, y + 32))
            img.alpha_composite(piece, (x, y))
            if i == 3:
                img.alpha_composite(_vanilla(DOORS[kind][1]), (x, y))
    img.save(os.path.join(PREVIEW, SHEET + ".png"))


if __name__ == "__main__":
    print(build("--preview" in sys.argv))
