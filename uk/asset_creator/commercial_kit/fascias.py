"""The shops' fascia signs (names.py SIGNS), cut into a square's pieces.

    python fascias.py [--preview]   (KnoxMap's Python)

Each sign is drawn as a board the length its name needs (2-8 squares, `length`)
and again at 3 and 2 squares for narrower shops, each with the first text that
fits legibly (text_for: the name, else its short form, names.short_of),
then cut a square a piece and projected onto the west and north wall faces at the
fascia's height (FASCIA_ROWS: over the shop front's own board,
shopfronts.py). A plain piece in the sign's ground fills the rest of a longer run.
Build lays them on the wall furniture layer of the shop front's squares
(knoxbuild/regional_uk.fascia_tiles).

Sheets fascias_uk_NN of 128 slots (8 x 16): signs fill them in names.SIGNS'
order, a sign never split over two sheets, so a sign's place fixes its ids (new
signs go in names.ADDED, at the end). What Build needs - each sign's versions and
their tiles - is written to built/fascias.json.

The lettering is DejaVu (fonts/LICENSE_DEJAVU); nothing is traced from any
company's sign. Styles (names.py): plain, stripe (an accent band under the name),
box (the name in an accent box), gilt (a traditional board: a bead round it,
shaded letters), pub (a pub's own board in its brewery's livery: shaded letters, ruled in
their colour, the brewery's name at its end; names.BREWERIES). An independent's "NAME  TRADE" shows its name only.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import names as N  # noqa: E402
import shopfronts as SF  # noqa: E402

WK = SF.WK
sys.path.insert(0, SF.MAKERS)
import window_kit as WIN  # noqa: E402

FONTS = os.path.normpath(os.path.join(HERE, "..", "fonts"))
FACE = {"sans": "DejaVuSans-Bold.ttf", "serif": "DejaVuSerif-Bold.ttf",
        "italic": "DejaVuSans-BoldOblique.ttf", "script": "DejaVuSerif-BoldItalic.ttf"}
FASCIA_ROWS = (4, 42)          # rows of the storey the board covers (0 at its top)
H = FASCIA_ROWS[1] - FASCIA_ROWS[0]
CAP = 27                       # the lettering's size in px; the sheets are 2x, so half that in game (review: 19 read as 6 px)
PAD = 10                       # px clear at each end of the board
MAXLEN = 8
SLOTS = 128
BUILT = os.path.join(HERE, "built")
MIN_CAP = 20                   # smaller lettering cannot be read at the game's scale
LAST_CAP = 16                  # ...except a narrow shop's short form, rather than no sign
LENGTHS = (3, 2)               # the shorter boards made beside a sign's own length (a run of 4 takes the 3, centred)


def _font(face: str, size: int):
    return ImageFont.truetype(os.path.join(FONTS, FACE[face]), size)


def _text_width(text: str, face: str, size: int) -> int:
    return int(math.ceil(_font(face, size).getlength(text)))


def name_of(key: str) -> str:
    """The board's text: the sign's name (an independent's trade line left off: at
    the game's scale it was 5 px, review 2026-10-20)."""
    return N.SIGNS[key][0].partition("  ")[0]


def length(text: str, face: str) -> int:
    """Squares a sign's board needs at the lettering's own size."""
    return max(2, min(MAXLEN, math.ceil((_text_width(text, face, CAP) + 2 * PAD) / 64)))


def text_for(key: str, squares: int) -> tuple[str, int, str]:
    """What a board of this many squares says, its size and face: the name, else the
    short form (names.short_of), at the largest size up to CAP that fits; a narrow
    board's short form may go down to LAST_CAP (review: a blank fascia reads as empty),
    in bold sans below MIN_CAP (review 2026-10-20: small serif and script blurred)."""
    face, style = N.SIGNS[key][4], N.SIGNS[key][6]
    room = squares * 64 - 2 * PAD - (12 if style == "box" else 0)
    small = "sans" if face in ("serif", "script") else face
    for text, least, f in ((name_of(key), MIN_CAP, face), (N.short_of(key), MIN_CAP, face),
                           (N.short_of(key), LAST_CAP, small)):
        size = CAP
        while size > least and _text_width(text, f, size) > room:
            size -= 1
        if _text_width(text, f, size) <= room:
            return text, size, f
    return N.short_of(key), LAST_CAP, small


def versions_of(key: str) -> list[int]:
    L = length(name_of(key), N.SIGNS[key][4])
    text, _t, _g, _i, face, accent, style = N.SIGNS[key]
    if style == "pub" and accent:                 # a pub's own length holds its brewery's name too
        L = max(2, min(MAXLEN, math.ceil((_text_width(name_of(key), face, CAP)
                                          + _text_width(accent, "sans", LAST_CAP) + 3 * PAD) / 64)))
    return [L] + [v for v in LENGTHS if v < L]


def board(key: str, squares: int) -> Image.Image:
    """The whole board, `squares` x 64 px by H, RGBA."""
    _text, _trade, ground, ink, _face, accent, style = N.SIGNS[key]
    text, size, face = text_for(key, squares)
    w = squares * 64
    img = Image.new("RGBA", (w, H), tuple(ground) + (255,))
    d = ImageDraw.Draw(img)
    main = text
    f_main = _font(face, size)
    tw = _text_width(text, face, size)
    x = (w - tw) // 2
    cy = H // 2 - (2 if style == "stripe" else 0)
    if style == "box" and accent:
        d.rectangle([max(2, x - 6), 4, min(w - 3, x + tw + 6), H - 5], fill=tuple(accent) + (255,))
    if style == "pub" and accent:
        # a pub's board: its brewery's name at the end where there is room, the pub's
        # name centred in the rest (names.BREWERIES)
        f_tag = _font("sans", LAST_CAP)
        tag_w = _text_width(accent, "sans", LAST_CAP)
        if tw + tag_w + 3 * PAD <= w:
            x = PAD + (w - 3 * PAD - tag_w - tw) // 2
            d.text((w - PAD - tag_w, H // 2), accent, font=f_tag, fill=tuple(ink) + (255,), anchor="lm")
            d.line([w - 2 * PAD - tag_w + PAD // 2, 6, w - 2 * PAD - tag_w + PAD // 2, H - 7],
                   fill=tuple(ink) + (255,), width=1)
    if style == "pub":
        # its own board, ruled in its lettering's colour (review 2026-10-20: on a front
        # in the same paint the bead alone left it reading as lettering on the frieze)
        d.rectangle([1, 1, w - 2, H - 2], outline=tuple(ink) + (255,), width=1)
    if style == "gilt":
        light = tuple(min(255, int(c * 1.5) + 30) for c in ground)
        d.rectangle([2, 2, w - 3, H - 3], outline=light + (255,), width=1)
        d.rectangle([4, 4, w - 5, H - 5], outline=tuple(int(c * 0.6) for c in ground) + (255,), width=1)
    if style == "stripe" and accent:
        d.rectangle([PAD // 2, H - 9, w - PAD // 2, H - 7], fill=tuple(accent) + (255,))
        d.rectangle([PAD // 2, H - 5, w - PAD // 2, H - 4], fill=tuple(accent) + (255,))

    def put(xy, s, f):
        if style in ("gilt", "pub"):                          # shaded letters, as a signwriter's
            d.text((xy[0] + 1, xy[1] + 1), s, font=f, fill=(20, 16, 12, 255), anchor="lm")
        d.text(xy, s, font=f, fill=tuple(ink) + (255,), anchor="lm")
    put((x, cy), main, f_main)
    # the board's top and bottom edges, as the game outlines a wall's
    d.line([0, 0, w, 0], fill=tuple(int(c * 0.7) for c in ground) + (255,))
    d.line([0, H - 1, w, H - 1], fill=tuple(int(c * 0.55) for c in ground) + (255,))
    return img


def bleed(tile: np.ndarray, depth: np.ndarray | None = None):
    """A sign piece run one pixel on into the transparent pixels either side of it. A
    wall piece's end overlaps its neighbour's by a column, and the next square's wall is
    drawn after this square's sign: the sign's last column was lost under it, a seam
    through the lettering at each join (review 2026-10-20). With `depth`, the pixels it
    adds take their neighbour's depth too, and (tile, depth) is returned."""
    out = tile.copy()
    dep = None if depth is None else depth.copy()
    empty = out[..., 3] == 0
    for shift in (1, -1):
        src = np.roll(tile, shift, axis=1)
        take = empty & (src[..., 3] > 0)
        if shift == 1:
            take[:, 0] = False
        else:
            take[:, -1] = False
        out[take] = src[take]
        if dep is not None:
            dep[take] = np.roll(depth, shift, axis=1)[take]
        empty &= ~take
    return out if depth is None else (out, dep)


def pieces(img: Image.Image) -> list[dict]:
    """A board cut a square a piece, each projected onto the W and N faces, in run
    order (the square at the run's low x or y first). A north wall's face runs left
    to right on screen as x grows, so its pieces are the board's columns in order;
    a west wall's runs right to left as y grows (walls.py's camera), so its pieces
    are the board's columns from the far end, each mirrored, and the lettering
    reads left to right on both."""
    a = np.asarray(img).astype(float)
    n = a.shape[1] // 64

    def face_of(i, mirror):
        face = np.zeros((SF.STOREY, 64, 4), float)
        cut = a[:, i * 64:(i + 1) * 64]
        face[FASCIA_ROWS[0]:FASCIA_ROWS[1]] = cut[:, ::-1] if mirror else cut
        return np.ascontiguousarray(face.astype(np.uint8))
    return [{"W": bleed(WIN.project(face_of(n - 1 - i, True), "W")), "N": bleed(WIN.project(face_of(i, False), "N"))}
            for i in range(n)]


PROUD = 0.03                   # m a board stands off its wall, so it draws over the shop front


def _depth(side: str, tile: np.ndarray) -> np.ndarray:
    _u, _v, _panel, depth = WK.FACES[side]
    return np.where(tile[..., 3] > 0, depth - WK.W.DEPTH_ALONG * PROUD, np.nan)


def _plain(key: str) -> Image.Image:
    _text, _trade, ground, *_r = N.SIGNS[key]
    img = Image.new("RGBA", (64, H), tuple(ground) + (255,))
    d = ImageDraw.Draw(img)
    d.line([0, 0, 64, 0], fill=tuple(int(c * 0.7) for c in ground) + (255,))
    d.line([0, H - 1, 64, H - 1], fill=tuple(int(c * 0.55) for c in ground) + (255,))
    return img


def build(preview: bool = False) -> dict:
    sheets: list[list] = [[]]
    built = {"sheets": [], "signs": {}}

    def name_of(sheet_no, k):
        return f"fascias_uk_{sheet_no + 1:02d}_{k}"

    for key, (text, _trade, _g, _i, face, _a, _s) in N.SIGNS.items():
        versions = versions_of(key)
        sprites = []                      # [(W, N)] in order: each version's pieces, then the plain piece
        for v in versions:
            sprites += [(p["W"], p["N"]) for p in pieces(board(key, v))]
        plain = pieces(_plain(key))[0]
        sprites.append((plain["W"], plain["N"]))
        need = 2 * len(sprites)
        if len(sheets[-1]) + need > SLOTS:
            sheets.append([])
        sheet_no, first = len(sheets) - 1, len(sheets[-1])
        for wv, nv in sprites:
            sheets[-1] += [wv, nv]
        rec, k = {"versions": []}, first
        for v in versions:
            rec["versions"].append({"length": v, "W": [name_of(sheet_no, k + 2 * j) for j in range(v)],
                                    "N": [name_of(sheet_no, k + 2 * j + 1) for j in range(v)]})
            k += 2 * v
        rec["plain"] = {"W": name_of(sheet_no, k), "N": name_of(sheet_no, k + 1)}
        built["signs"][key] = rec
    blank = np.zeros((256, 128, 4), np.uint8)
    for no, tiles in enumerate(sheets):
        sheet = f"fascias_uk_{no + 1:02d}"
        tiles = tiles + [blank] * (SLOTS - len(tiles))
        WK.F.write_sheet(WK.sheet_file(sheet + ".png"), tiles)
        # each piece's depth: its wall face's, PROUD m nearer (slots alternate W, N)
        WK.F.write_sheet(WK.sheet_file(f"DEPTH_{sheet}.png"),
                         [WK.W.depth_tile(t, _depth("WN"[k % 2], t)) for k, t in enumerate(tiles)])
        with open(WK.sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
            json.dump([None] * SLOTS, f)
        built["sheets"].append(sheet)
    os.makedirs(BUILT, exist_ok=True)
    with open(os.path.join(BUILT, "fascias.json"), "w", encoding="utf-8") as f:
        json.dump(built, f, indent=1)
    if preview:
        _preview()
    return {"sheets": built["sheets"], "signs": len(built["signs"])}


# --- the superstores' big signs ----------------------------------------------------------------
# A 1990s superstore's name stood on its cladding in letters a metre and more high
# (review 2026-10-20: a 3-square fascia on a 40-square front could not be found):
# a panel BIG_ROWS of the upper storey high, the name filling it, at its own length
# and at BIG_SHORT squares, on sheets of their own (fascias_big_uk_NN), hung a storey
# up (`level` 1) on the cladding.
BIG_ROWS = (40, 150)
BIG_CAP = 78
BIG_MAX = 12
BIG_SHORT = 6
BIG_TRADES = ("supermarket", "frozen_food")


def big_board(key: str, squares: int) -> Image.Image:
    text, _trade, ground, ink, face, accent, style = N.SIGNS[key]
    text = name_of(key)
    h, w = BIG_ROWS[1] - BIG_ROWS[0], squares * 64
    img = Image.new("RGBA", (w, h), tuple(ground) + (255,))
    d = ImageDraw.Draw(img)
    size = BIG_CAP
    while size > 30 and _text_width(text, face, size) > w - 4 * PAD:
        size -= 2
    tw = _text_width(text, face, size)
    cy = h // 2 - (6 if style == "stripe" else 0)
    if style == "box" and accent:
        d.rectangle([(w - tw) // 2 - 10, 10, (w + tw) // 2 + 10, h - 11], fill=tuple(accent) + (255,))
    if style == "stripe" and accent:
        d.rectangle([PAD, h - 22, w - PAD, h - 16], fill=tuple(accent) + (255,))
        d.rectangle([PAD, h - 12, w - PAD, h - 9], fill=tuple(accent) + (255,))
    d.text(((w - tw) // 2, cy), text, font=_font(face, size), fill=tuple(ink) + (255,), anchor="lm")
    # a frame and a shadow under it, so the panel sits on the cladding (review 2026-10-20:
    # flat and bright, it looked pasted on beside vanilla's signs)
    d.rectangle([0, 0, w - 1, h - 1], outline=tuple(int(c * 0.55) for c in ground) + (255,), width=3)
    d.rectangle([3, h - 7, w - 4, h - 4], fill=tuple(int(c * 0.78) for c in ground) + (255,))
    return img


def big_pieces(img: Image.Image) -> list[tuple]:
    a = np.asarray(img).astype(float)
    n = a.shape[1] // 64

    def face_of(i, mirror):
        face = np.zeros((SF.STOREY, 64, 4), float)
        cut = a[:, i * 64:(i + 1) * 64]
        face[BIG_ROWS[0]:BIG_ROWS[1]] = cut[:, ::-1] if mirror else cut
        return np.ascontiguousarray(face.astype(np.uint8))
    return [(bleed(WIN.project(face_of(n - 1 - i, True), "W")), bleed(WIN.project(face_of(i, False), "N")))
            for i in range(n)]


def build_big() -> dict:
    """The big signs' sheets and their records ({sign: [version, ...]}, each with level 1)."""
    sheets: list[list] = [[]]
    out: dict = {}
    for key, (text, trade, *_r) in N.SIGNS.items():
        if trade not in BIG_TRADES:
            continue
        L = max(BIG_SHORT, min(BIG_MAX, math.ceil((_text_width(name_of(key), N.SIGNS[key][4], BIG_CAP) + 4 * PAD) / 64)))
        versions = [L] + ([BIG_SHORT] if L > BIG_SHORT else [])
        sprites = []
        for v in versions:
            sprites += big_pieces(big_board(key, v))
        if len(sheets[-1]) + 2 * len(sprites) > SLOTS:
            sheets.append([])
        no, k = len(sheets) - 1, len(sheets[-1])
        for wv, nv in sprites:
            sheets[-1] += [wv, nv]
        recs = []
        for v in versions:
            recs.append({"length": v, "level": 1,
                         "W": [f"fascias_big_uk_{no + 1:02d}_{k + 2 * j}" for j in range(v)],
                         "N": [f"fascias_big_uk_{no + 1:02d}_{k + 2 * j + 1}" for j in range(v)]})
            k += 2 * v
        out[key] = recs
    blank = np.zeros((256, 128, 4), np.uint8)
    names = []
    for no, tiles in enumerate(sheets):
        sheet = f"fascias_big_uk_{no + 1:02d}"
        tiles = tiles + [blank] * (SLOTS - len(tiles))
        WK.F.write_sheet(WK.sheet_file(sheet + ".png"), tiles)
        WK.F.write_sheet(WK.sheet_file(f"DEPTH_{sheet}.png"),
                         [WK.W.depth_tile(t, _depth("WN"[k % 2], t)) for k, t in enumerate(tiles)])
        with open(WK.sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
            json.dump([None] * SLOTS, f)
        names.append(sheet)
    with open(os.path.join(BUILT, "fascias_big.json"), "w", encoding="utf-8") as f:
        json.dump({"sheets": names, "signs": out}, f, indent=1)
    img = Image.new("RGBA", (64 * BIG_MAX + 20, (BIG_ROWS[1] - BIG_ROWS[0] + 8) * len(out)), (110, 110, 106, 255))
    for i, key in enumerate(out):
        img.alpha_composite(big_board(key, out[key][0]["length"]), (10, i * (BIG_ROWS[1] - BIG_ROWS[0] + 8)))
    img.save(os.path.join(SF.PREVIEW, "fascias_big.png"))
    return {"sheets": names, "signs": len(out)}


def _preview() -> None:
    """Every sign's board, flat at 1x (as the game draws it), in sheet order."""
    os.makedirs(SF.PREVIEW, exist_ok=True)
    keys = list(N.SIGNS)
    cols, cw, rh = 3, 64 * MAXLEN + 20, H + 8
    # (and below the list: every sign's 2-square board, as the narrowest shops get it)
    rows = math.ceil(len(keys) / cols)
    img = Image.new("RGBA", (cols * cw, rows * rh), (110, 110, 106, 255))
    for i, key in enumerate(keys):
        b = board(key, versions_of(key)[0])
        img.alpha_composite(b, ((i % cols) * cw + 10, (i // cols) * rh + 4))
    img.save(os.path.join(SF.PREVIEW, "fascias_flat.png"))
    cols, cw = 8, 64 * 2 + 12
    img = Image.new("RGBA", (cols * cw, math.ceil(len(keys) / cols) * rh), (110, 110, 106, 255))
    for i, key in enumerate(keys):
        img.alpha_composite(board(key, 2), ((i % cols) * cw + 6, (i // cols) * rh + 4))
    img.save(os.path.join(SF.PREVIEW, "fascias_two_squares.png"))


if __name__ == "__main__":
    print(build("--preview" in sys.argv))
    print(build_big())
