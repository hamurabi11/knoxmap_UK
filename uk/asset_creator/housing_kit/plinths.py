"""Plinths for the housing styles' ground floors (user, 2026-10-19: "yeah let's
add them"; LOG 2026-10-19 #11).

    python plinths.py   (KnoxMap's Python)

Sheet wall_trim_uk_02, four pieces each (slot 4 k ...): a west wall, a north
wall, both (an inside corner) and an outside corner's post,
laid by Build along the ground floor's outside walls the camera sees, as the
listed maker's trims are (knoxbuild/tbx.py `_lay_trims`, regional_uk
HOUSE_TRIMS), never across a doorway. Each painted from its wall kit's own
texture (the same random draw wall_kit.kit makes), so its courses run on
from the wall's:

    0  dpc_buff70     1960s-70s buff brick: the damp-proof course, the courses under it darker
    4  dpc_orange80   1980s orange multi: the same
    8  dpc_red50      1950s red-brown: the same
   12  dpc_rededw     red pressed brick (the 1930s ground floor): the same
   16  plinth_rededw  Edwardian: a projecting plinth, four courses and a capping course over the d.p.c.
   20  plinth_flemish Georgian / Victorian / a timber frame's brick: the same in Flemish bond
   24  dpc_iwc        inter-war council brick (also under roughcast and pebbledash): a d.p.c.
   28  dpc_fletton    post-war Flettons
   32  dpc_brown70    1960s-70s dark brown multi
   36  dpc_buff80     1980s pale buff
   40  dpc_red80      1980s starter terrace red

Review (2026-10-19): each face runs a centimetre past its square (no pale
tick between squares); the corners have pieces of their own.

Nearly every English house since the bye-laws of the 1870s has a d.p.c. about
150 mm up (a thin dark line: slate, later bitumen) with harder, darker brick
under it, weathered by rain splash; Victorian and Edwardian houses often a
plinth stepped out under it.
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
import parts as P  # noqa: E402

WK, W, F = HK.WK, HK.W, HK.F
SHEET = "wall_trim_uk_02"
PX = 64
DPC_M = 0.15                       # the damp-proof course's height
DPC = (58, 56, 54)
DPC_BED = (128, 124, 116)          # on dark brick a dark line alone vanished (2026-10-19)


def wall_texture(kit: str) -> np.ndarray:
    """The texture wall_kit.kit paints a kit's plain walls with (its variant A:
    west face columns 0-63, north 64-127), drawn again from the same seed."""
    name = kit if kit in WK.MATERIALS else "_housing_" + kit
    if name not in WK.MATERIALS:
        WK.MATERIALS[name] = HK.WALLS[kit]
    make = WK.MATERIALS[name][0]
    rng = np.random.default_rng(sum(map(ord, name)))
    return np.concatenate([make(rng), make(rng)], 1).astype(float)


def _foot(tex, side, h_m):
    """The bottom h_m of a storey's face texture, as an RGBA texture."""
    rows = round(h_m * PX)
    face = tex[-rows:, (0 if side == 0 else 64):(64 if side == 0 else 128)]
    out = np.zeros((rows, 64, 4))
    out[..., :3] = face
    out[..., 3] = 255
    return out


E = 0.012                          # a face runs this far past its square at each end (no seam)


def _corner_pieces(face, o):
    """A trim's four pieces from `face(p, north, a0, a1)`, which draws it on
    one wall from a0 to a1 along it: the west wall, the north wall, both (an
    inside corner: the two meet at the square's corner) and an outside
    corner's post (the stub of each wall from the corner point to `o` out)."""
    def west():
        p = P.Bay(False)
        face(p, False, -E, 1 + E)
        return p.render(0)

    def north():
        p = P.Bay(True)
        face(p, True, -E, 1 + E)
        return p.render(0)

    def both():
        (a, da), (b, db) = west(), north()
        m = b[..., 3] > 0
        a[m], da[m] = b[m], db[m]
        return a, da

    def post():
        p = P.Bay(False)
        face(p, False, -E, o + E, post=True)                   # the east face of the stub
        q = P.Bay(True)
        face(q, True, -E, o + E, post=True)                    # the south face
        (a, da), (b, db) = p.render(0), q.render(0)
        m = b[..., 3] > 0
        a[m], da[m] = b[m], db[m]
        return a, da
    return [west, north, both, post]


def dpc(kit, under=0.74, bed=None, h_m=DPC_M):
    """The d.p.c. at DPC_M; `h_m` the plinth's height (under it the courses
    are harder and darker: four courses of engineering brick on the new
    kits - review, 2026-10-19: two courses read as a skirting line)."""
    tex = wall_texture(kit)
    t_ = {}

    def tex_for(north):
        if north not in t_:
            t = _foot(tex, int(north), h_m + 0.02)
            line = t.shape[0] - round(h_m * PX)                # the d.p.c. row, counted down from the top
            t[line:, :, :3] *= np.array([under, under - 0.02, under - 0.02])   # harder, darker brick under it
            t[-3:, :, :3] *= 0.85                              # rain splash at the foot
            t[line - 2:line, :, :3] = bed or DPC_BED           # the d.p.c.'s thick cement bed
            t[line:line + 1, :, :3] = DPC                      # ...and the felt's dark edge under it
            t_[north] = t
        return t_[north]

    def face(p, north, a0, a1, post=False):
        o = WK.T + 0.004
        t = tex_for(north)
        w = max(2, round((a1 - a0) * 64))
        t = t[:, :w] if post else t
        p.face((o, a0), (o, a1), 0, h_m + 0.02, t, P._light(north, 1, 0))
    return _corner_pieces(face, WK.T + 0.004)


def projecting(kit, h_m=0.45, step=0.025):
    tex = wall_texture(kit)
    ledge = tuple(np.clip(tex[-30:].reshape(-1, 3).mean(0) * 1.1, 0, 255))   # the cap's top: the kit's brick, lit
    t_ = {}

    def tex_for(north):
        if north not in t_:
            t = _foot(tex, int(north), h_m)
            t[:, :, :3] *= 0.92
            line = t.shape[0] - round(DPC_M * PX)
            t[line:, :, :3] *= 0.9
            t[line:line + 1, :, :3] = (72, 66, 62)             # review: one pixel, not a black crack
            t[:3, :, :3] = np.minimum(t[:3, :, :3] * 1.12, 255)    # the capping course, catching the light
            t_[north] = t
        return t_[north]

    def face(p, north, a0, a1, post=False):
        o = WK.T + step
        t = tex_for(north)
        if post:
            t = t[:, :max(2, round((a1 - a0) * 64))]
        p.plane((WK.T if not post else 0, a0, h_m), (0, a1 - a0, 0), (o - (WK.T if not post else 0), 0, 0),
                P.DK._flat(ledge, 4, 16), W.TOP_LIGHT)
        p.face((o, a0), (o, a1), 0, h_m, t, P._light(north, 1, 0))
    return _corner_pieces(face, WK.T + step)


PLINTHS = [("dpc_buff70", dpc("brick_buff_70s", under=0.80)), ("dpc_orange80", dpc("brick_orange_80s")),
           ("dpc_red50", dpc("brick_red_50s", under=0.68, bed=(160, 152, 140))), ("dpc_rededw", dpc("brick_red_edw")),
           ("plinth_rededw", projecting("brick_red_edw")), ("plinth_flemish", projecting("brick_flemish")),
           # K09.26 (2026-10-19), appended (slots fixed): council housing, flats,
           # bungalows, 1980s terraces; dpc_iwc also under roughcast and pebbledash
           ("dpc_iwc", dpc("brick_red_iwc", under=0.6, h_m=0.3)), ("dpc_fletton", dpc("brick_fletton", under=0.7, h_m=0.3)),
           ("dpc_brown70", dpc("brick_brown_70s", under=0.76, bed=(150, 144, 134), h_m=0.3)),
           ("dpc_buff80", dpc("brick_buff_80s", under=0.74, h_m=0.3)), ("dpc_red80", dpc("brick_red_80s_band", h_m=0.3))]
# The kit each plinth is painted from, for the preview.
PLINTH_KITS = ["brick_buff_70s", "brick_orange_80s", "brick_red_50s", "brick_red_edw", "brick_red_edw",
               "brick_flemish", "brick_red_iwc", "brick_fletton", "brick_brown_70s", "brick_buff_80s",
               "brick_red_80s_band"]


def main() -> int:
    tiles, depths = [], []
    for _name, pieces in PLINTHS:
        for make in pieces:
            t, d = make()
            tiles.append(t)
            depths.append(d)
    F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"), [W.depth_tile(t, d) for t, d in zip(tiles, depths)])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump([None] * len(tiles), f)                      # seen only, as the listed trims
    # A preview: each plinth's four pieces on its own wall kit's west, north,
    # inside-corner (NW) and outside-corner (SE post) pieces.
    walls = {"brick_flemish": "walls_house_uk_01_brick_flemish"}
    img = Image.new("RGBA", (128 * len(tiles), 256), (80, 100, 80, 255))
    for k, t in enumerate(tiles):
        kit = PLINTH_KITS[k // 4]
        sheet = walls.get(kit, f"walls_house_uk_02_{kit}")
        a = np.asarray(Image.open(WK.sheet_file(sheet + ".png")).convert("RGBA"))
        j = k % 4                                              # the kit's W, N, NW and SE pieces are its slots 0-3
        wall = a[0:256, j * 128:j * 128 + 128]
        img.alpha_composite(Image.fromarray(np.ascontiguousarray(wall)), (k * 128, 0))
        img.alpha_composite(Image.fromarray(t), (k * 128, 0))
    os.makedirs(HK.PREVIEW, exist_ok=True)
    img.save(os.path.join(HK.PREVIEW, f"{SHEET}.png"))
    have = json.load(open(HK.ORDER, encoding="utf-8"))
    if SHEET not in have:
        have.append(SHEET)
        json.dump(have, open(HK.ORDER, "w", encoding="utf-8"), indent=1)
    print("wrote", SHEET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
