"""Wall signs other than the shops' fascias (PLAN.md step 5), cut into a square's pieces.

    python signs.py [--preview]   (KnoxMap's Python)

Sheet commercial_signs_uk_01 (its DEPTH_ sheet and .slots.json) and built/signs.json
(each sign's W and N tiles in run order, for Build: regional_uk `wall_signs`):

- mot   a repair garage's "MOT TESTING STATION" board: white lettering on a blue
        ground with a white rule round it, 3 squares long and a metre high, at head
        height on the workshop's wall beside its shutter. Plain lettering (DejaVu), not
        the official sign's three triangles (PLAN.md: nothing traced from a real sign).
- exchange   "TELEPHONE EXCHANGE" in cast metal letters fixed to the brick (no board),
        6 squares: the 1960s exchange's only sign.
- sorting    "ROYAL MAIL  DELIVERY OFFICE" the same way, in red: a sorting office's
        (no building in either map is one yet; ready for one).

Drawn as fascias.py draws a board (2x the game's scale) and projected as its pieces
are (fascias.pieces: a west wall's from the far end, mirrored), at SIGN_ROWS of the
storey in place of the fascia's rows; each piece's depth its wall face's, a few cm
proud. Appended only: a sign's place fixes its ids.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fascias as FA  # noqa: E402

SF, WK, WIN = FA.SF, FA.WK, FA.WIN
SHEET = "commercial_signs_uk_01"
SIGN_ROWS = (44, 108)            # rows of the storey the board covers (0 at its top): 2.3 m to 1.3 m up
BLUE, WHITE = (22, 64, 150), (246, 246, 240)


def mot_board() -> Image.Image:
    w, h = 3 * 64, SIGN_ROWS[1] - SIGN_ROWS[0]
    img = Image.new("RGBA", (w, h), BLUE + (255,))
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, w - 4, h - 4], outline=WHITE + (255,), width=2)
    d.text((w // 2, 24), "MOT", font=FA._font("sans", 34), fill=WHITE + (255,), anchor="mm")
    d.text((w // 2, 50), "TESTING STATION", font=FA._font("sans", FA.LAST_CAP), fill=WHITE + (255,), anchor="mm")
    d.line([0, 0, w, 0], fill=tuple(int(c * 0.7) for c in BLUE) + (255,))
    d.line([0, h - 1, w, h - 1], fill=tuple(int(c * 0.5) for c in BLUE) + (255,))
    return img


def letters(text: str, ink, squares: int, size: int = 24, y: int = 14) -> Image.Image:
    """Cast letters on the wall itself: no board, raised - each lit along its top, a
    shadow below and to its right (review 2026-10-20: flat painted text was a smudge);
    high on the board's rows (`y`), over the doors' heads."""
    w, h = squares * 64, SIGN_ROWS[1] - SIGN_ROWS[0]
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = FA._font("sans", size)
    for dx, dy in ((2, 2), (1, 2), (2, 1)):
        d.text((w // 2 + dx, y + dy), text, font=f, fill=(24, 22, 20, 230), anchor="mm")   # the shadow
    d.text((w // 2, y - 1), text, font=f, fill=tuple(min(255, int(c * 1.45) + 30) for c in ink) + (255,), anchor="mm")
    d.text((w // 2, y), text, font=f, fill=tuple(ink) + (255,), anchor="mm")
    return img


def exchange_letters() -> Image.Image:
    return letters("TELEPHONE EXCHANGE", (120, 92, 54), 6)           # bronze (r1: pale ones were lost on the buff)


def sorting_letters() -> Image.Image:
    return letters("ROYAL MAIL  DELIVERY OFFICE", (200, 30, 34), 7, 20)


def pieces(img: Image.Image, bleed: bool = True) -> list[dict]:
    """fascias.pieces at SIGN_ROWS (no bleed for letters on the bare wall: it fattened each
    glyph, code review)."""
    a = np.asarray(img).astype(float)
    n = a.shape[1] // 64

    def face_of(i, mirror):
        face = np.zeros((SF.STOREY, 64, 4), float)
        cut = a[:, i * 64:(i + 1) * 64]
        face[SIGN_ROWS[0]:SIGN_ROWS[1]] = cut[:, ::-1] if mirror else cut
        return np.ascontiguousarray(face.astype(np.uint8))
    b = FA.bleed if bleed else (lambda t: t)
    return [{"W": b(WIN.project(face_of(n - 1 - i, True), "W")),
             "N": b(WIN.project(face_of(i, False), "N"))} for i in range(n)]


LETTERS = {"exchange", "sorting"}   # letters on the wall, not a board
SIGNS = {"mot": mot_board, "exchange": exchange_letters, "sorting": sorting_letters}       # appended only


def main(argv) -> int:
    tiles, built = [], {"sheet": SHEET, "signs": {}}
    for key, make in SIGNS.items():
        rec = {"W": [], "N": []}
        for p in pieces(make(), bleed=key not in LETTERS):
            for side in ("W", "N"):
                rec[side].append(f"{SHEET}_{len(tiles)}")
                tiles.append(p[side])
        built["signs"][key] = rec
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"),
                     [WK.W.depth_tile(t, FA._depth("WN"[k % 2], t)) for k, t in enumerate(tiles)])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump([None] * len(tiles), f)
    os.makedirs(FA.BUILT, exist_ok=True)
    with open(os.path.join(FA.BUILT, "signs.json"), "w", encoding="utf-8") as f:
        json.dump(built, f, indent=1)
    if "--preview" in argv:
        # the board on a north wall of plain brick, 5 squares, at the game's scale
        import supermarket as SM
        wall = Image.open(WK.sheet_file(SM.sheet_of("superstore_brick_upper") + ".png")).convert("RGBA")
        sheet = Image.open(WK.sheet_file(SHEET + ".png")).convert("RGBA")

        def tile(img, k):
            cols = img.width // 128
            r, c = divmod(k, cols)
            return img.crop((c * 128, r * 256, c * 128 + 128, r * 256 + 256))
        img = Image.new("RGBA", (600, 460), (96, 100, 96, 255))
        names = built["signs"]["mot"]["N"]
        for i in range(5):
            img.alpha_composite(tile(wall, 1), (40 + 64 * i, 40 + 32 * i))
            if 1 <= i <= 3:
                img.alpha_composite(tile(sheet, int(names[i - 1].rsplit("_", 1)[1])), (40 + 64 * i, 40 + 32 * i))
        img.save(os.path.join(SM.PREVIEW, "signs_mot.png"))
    order = os.path.join(HERE, "sheets.order.json")
    have = json.load(open(order, encoding="utf-8"))
    if SHEET not in have:
        have.append(SHEET)
    json.dump(have, open(order, "w", encoding="utf-8"), indent=1)
    print("wrote", SHEET, len(tiles))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
