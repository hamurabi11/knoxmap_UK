"""A superstore's department signs, hung high on its inside walls, cut into a square's pieces.

    python dept_signs.py [--preview]   (KnoxMap's Python)

A 1993 superstore named its departments on boards hung from the ceiling along its
walls, over the counters and aisle ends: BAKERY, FRESH PRODUCE, BEERS WINES &
SPIRITS... Each board here is drawn as fascias.py draws a fascia (2x the game's
scale, DejaVu lettering, the board's top and bottom edges darkened as a wall's) in
its store's livery (LIVERY: Lainburys white on orange, Waitrows white on green,
any other store white on dark blue), on two wires from the ceiling, at BOARD_ROWS
of the storey (about 2.2 to 2.8 m up: over head height, under the ceiling).

Each department (DEPTS: its text, then its short forms) is drawn at 4, 3 and 2
squares, and at 6 for a text that wants more than 4, each with the first text that
fits legibly (fit: fascias.text_for's rule - the text, else a short form, down to
MIN_CAP in the store's face; the last short form down to LAST_CAP in bold sans),
then cut a square a piece and projected onto the west and north wall faces
(fascias.pieces' way: a west wall's pieces from the far end, mirrored), each with
its depth picture, stood off the wall as a fascia is (fascias._depth).

The inside face of a building's west or north wall is the face the camera sees, so
the pieces are the same W and N faces as a fascia's: Build lays them on the wall
furniture layer of the squares inside the wall.

Sheets dept_signs_uk_NN of 128 slots (8 x 16): signs fill them in LIVERY's then
DEPTS' order, a sign never split over two sheets, slots alternating W, N; so a
sign's place fixes its ids (append new departments or liveries only). The sheets
are appended to sheets.order.json, which pack.py reads. built/dept_signs.json:
{"sheets": [...], "signs": {chain: {dept: {"versions": [{"length", "text", "W", "N"}]}}}}.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fascias as FA  # noqa: E402
import names as N  # noqa: E402

SF, WK, WIN = FA.SF, FA.WK, FA.WIN
PREFIX = "dept_signs_uk"
SLOTS = FA.SLOTS
WIRE_TOP = 0                   # the wires run up to the storey's top (the ceiling)
BOARD_ROWS = (20, 20 + FA.H)   # rows of the storey the board covers (0 at its top): 2.7 m to 2.1 m up
                               # (review r1: hung on a longer drop, not screwed under the ceiling)
SIZE = 24                      # a store's lettering, the same on all its boards (FA.CAP is 27)...
NARROW = (18, "sans")          # ...but a 2-square board's, the same on every 2-square board
H = FA.H
LENGTHS = (4, 3, 2)
LONG = 6                       # a text that wants more than 4 squares gets a board this long too

WHITE = (250, 248, 242)
# store: (ground, lettering, face). The chains' colours are names.CHAINS' (their
# fascia's lettering becomes the board's ground, its ground the lettering).
_L, _W = N.CHAINS["lainburys"], N.CHAINS["waitrows"]
LIVERY = {
    "lainburys": (_L[3], _L[2], _L[4]),          # white on Lainburys orange
    "waitrows": (_W[3], _W[2], _W[4]),           # white on Waitrows green
    "generic": ((26, 44, 96), WHITE, "sans"),    # white on dark blue, any other store
}

# department: its text, then its short forms (the last the narrowest board's)
DEPTS = {
    "bakery": ("BAKERY",),
    "butcher": ("BUTCHER",),
    "deli": ("DELICATESSEN", "DELI"),
    "produce": ("FRESH PRODUCE", "PRODUCE"),
    "frozen": ("FROZEN FOODS", "FROZEN"),
    "drinks": ("BEERS, WINES & SPIRITS", "BEERS & WINES", "WINES"),
    "household": ("HOUSEHOLD", "HOME"),
    "pharmacy": ("PHARMACY", "CHEMIST"),
    "clothing": ("CLOTHING", "CLOTHES"),
    "cafe": ("COFFEE SHOP", "CAFE"),
    "services": ("CUSTOMER SERVICES", "SERVICES"),
    "toilets": ("CUSTOMER TOILETS", "TOILETS"),
    "grocery": ("GROCERY",),
}


def _room(squares: int) -> int:
    return squares * 64 - 2 * FA.PAD - 8          # clear of the inset rule (review r1: cramped)


def fit(texts: tuple, face: str, squares: int) -> tuple[str, int, str]:
    """What a board of `squares` says, its size and face. One size a board width
    (review r1: each text fitted to its board jumped in size board to board): SIZE in
    the store's face, and on the narrowest board NARROW in bold sans (the chains'
    serif under MIN_CAP blurred, fascias review 2026-10-20); the first of `texts`
    (the text, then its short forms) that fits at it. Else fascias.text_for's rule:
    the first that fits at MIN_CAP or more, else the last at LAST_CAP in bold sans."""
    room = _room(squares)
    size, f0 = NARROW if squares <= 2 else (SIZE, face)
    for t in texts:
        if FA._text_width(t, f0, size) <= room:
            return t, size, f0
    for text, least, f in [(t, FA.MIN_CAP, face) for t in texts] + [(texts[-1], FA.LAST_CAP, "sans")]:
        size = SIZE
        while size > least and FA._text_width(text, f, size) > room:
            size -= 1
        if FA._text_width(text, f, size) <= room:
            return text, size, f
    raise ValueError(f"{texts[-1]!r} does not fit {squares} squares")


def versions_of(dept: str, face: str) -> list[int]:
    want = math.ceil((FA._text_width(DEPTS[dept][0], face, SIZE) + 2 * FA.PAD + 8) / 64)
    return ([LONG] if want > max(LENGTHS) else []) + list(LENGTHS)


def board(chain: str, dept: str, squares: int) -> Image.Image:
    """The board and its wires, `squares` x 64 px by BOARD_ROWS[1] rows (the storey's
    top down to the board's foot), RGBA: the store's ground, a rule inset round it in
    the lettering's colour, the text centred."""
    ground, ink, face = LIVERY[chain]
    text, size, f = fit(DEPTS[dept], face, squares)
    w, top = squares * 64, BOARD_ROWS[0]
    img = Image.new("RGBA", (w, BOARD_ROWS[1]), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for x in (FA.PAD + 4, w - FA.PAD - 6):                       # the wires it hangs from
        d.rectangle([x, WIRE_TOP, x + 1, top], fill=(58, 58, 62, 255))
    d.rectangle([0, top, w - 1, top + H - 1], fill=tuple(ground) + (255,))
    d.rectangle([3, top + 3, w - 4, top + H - 4], outline=tuple(ink) + (255,), width=2)
    tw = FA._text_width(text, f, size)
    d.text(((w - tw) // 2, top + H // 2), text, font=FA._font(f, size), fill=tuple(ink) + (255,), anchor="lm")
    d.line([0, top, w, top], fill=tuple(int(c * 0.7) for c in ground) + (255,))
    d.line([0, top + H - 1, w, top + H - 1], fill=tuple(int(c * 0.55) for c in ground) + (255,))
    return img


def pieces(img: Image.Image) -> list[dict]:
    """fascias.pieces with the board at rows 0..BOARD_ROWS[1] of the storey."""
    a = np.asarray(img).astype(float)
    n = a.shape[1] // 64

    def face_of(i, mirror):
        face = np.zeros((SF.STOREY, 64, 4), float)
        cut = a[:, i * 64:(i + 1) * 64]
        face[:a.shape[0]] = cut[:, ::-1] if mirror else cut
        return np.ascontiguousarray(face.astype(np.uint8))
    return [{"W": FA.bleed(WIN.project(face_of(n - 1 - i, True), "W")),
             "N": FA.bleed(WIN.project(face_of(i, False), "N"))} for i in range(n)]


def _tile_name(sheet_no: int, k: int) -> str:
    return f"{PREFIX}_{sheet_no + 1:02d}_{k}"            # as fascias.json's: no padding on the index


def build(preview: bool = False) -> dict:
    sheets: list[list] = [[]]
    built: dict = {"sheets": [], "signs": {}}
    for chain, (_g, _i, face) in LIVERY.items():
        built["signs"][chain] = {}
        for dept in DEPTS:
            versions = versions_of(dept, face)
            sprites = []
            for v in versions:
                sprites += [(p["W"], p["N"]) for p in pieces(board(chain, dept, v))]
            if len(sheets[-1]) + 2 * len(sprites) > SLOTS:
                sheets.append([])
            no, k = len(sheets) - 1, len(sheets[-1])
            for wv, nv in sprites:
                sheets[-1] += [wv, nv]
            rec = {"versions": []}
            for v in versions:
                rec["versions"].append({"length": v, "text": fit(DEPTS[dept], face, v)[0],
                                        "W": [_tile_name(no, k + 2 * j) for j in range(v)],
                                        "N": [_tile_name(no, k + 2 * j + 1) for j in range(v)]})
                k += 2 * v
            built["signs"][chain][dept] = rec
    blank = np.zeros((256, 128, 4), np.uint8)
    for no, tiles in enumerate(sheets):
        sheet = f"{PREFIX}_{no + 1:02d}"
        tiles = tiles + [blank] * (SLOTS - len(tiles))
        WK.F.write_sheet(WK.sheet_file(sheet + ".png"), tiles)
        WK.F.write_sheet(WK.sheet_file(f"DEPTH_{sheet}.png"),
                         [WK.W.depth_tile(t, FA._depth("WN"[k % 2], t)) for k, t in enumerate(tiles)])
        with open(WK.sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
            json.dump([None] * SLOTS, f)
        built["sheets"].append(sheet)
    os.makedirs(FA.BUILT, exist_ok=True)
    with open(os.path.join(FA.BUILT, "dept_signs.json"), "w", encoding="utf-8") as f:
        json.dump(built, f, indent=1)
    order = os.path.join(HERE, "sheets.order.json")
    with open(order, encoding="utf-8") as f:
        have = json.load(f)
    new = [s for s in built["sheets"] if s not in have]
    if new:                                               # appended only: the order fixes the pack's ids
        with open(order, "w", encoding="utf-8") as f:
            json.dump(have + new, f, indent=1)
    if preview:
        _preview(built)
    return {"sheets": built["sheets"], "signs": sum(len(v) for v in built["signs"].values())}


# --- previews ------------------------------------------------------------------------------------

def _contact() -> None:
    """Each store's boards flat, a department a row, its versions side by side, at the
    sheets' scale; and every store's at the game's scale (half), as a player sees them."""
    rh = BOARD_ROWS[1] + 8
    cw = 64 * (LONG + 4 + 3 + 2) + 5 * 12
    halves = []
    for chain in LIVERY:
        img = Image.new("RGBA", (cw, rh * len(DEPTS) + 8), (206, 202, 192, 255))
        for r, dept in enumerate(DEPTS):
            x = 12
            for v in versions_of(dept, LIVERY[chain][2]):
                img.alpha_composite(board(chain, dept, v), (x, 4 + r * rh))
                x += 64 * v + 12
        img.save(os.path.join(SF.PREVIEW, f"dept_signs_{chain}.png"))
        halves.append(img.resize((img.width // 2, img.height // 2), Image.LANCZOS))
    out = Image.new("RGBA", (sum(h.width for h in halves) + 8 * len(halves), halves[0].height), (96, 100, 96, 255))
    x = 0
    for h in halves:
        out.alpha_composite(h, (x, 0))
        x += h.width + 8
    out.save(os.path.join(SF.PREVIEW, "dept_signs_game_scale.png"))


def _sheet_tile(cache: dict, name: str) -> Image.Image:
    sheet, k = name.rsplit("_", 1)
    if sheet not in cache:
        cache[sheet] = Image.open(WK.sheet_file(sheet + ".png")).convert("RGBA")
    img, k = cache[sheet], int(k)
    cols = img.width // 128
    r, c = divmod(k, cols)
    return img.crop((c * 128, r * 256, c * 128 + 128, r * 256 + 256))


def _vanilla(sheet: str, k: int) -> Image.Image | None:
    try:
        return Image.fromarray(WK.vanilla_tile(sheet, k)).convert("RGBA")
    except Exception:  # noqa: BLE001 - the preview goes without
        return None


# a room's corner for each store: (west wall: [(dept, length)], north wall: [(dept, length)])
WALL_SCENES = {
    "lainburys": ([("drinks", 6)], [("bakery", 3), ("produce", 4)]),
    "waitrows": ([("services", 6)], [("deli", 4), ("frozen", 2)]),
    "generic": ([("household", 3), ("pharmacy", 3)], [("butcher", 3), ("cafe", 4)]),
}
ROOM = 10


def _corner(built: dict, chain: str, cache: dict) -> Image.Image:
    """A room's north-west corner ROOM squares each way, the inside of its west and
    north walls (vanilla's interior wall) and its floor, the store's signs hung on
    both walls, drawn back to front as the game draws them."""
    wall_w, wall_n = _vanilla("walls_interior_house_01", 0), _vanilla("walls_interior_house_01", 1)
    floor = _vanilla("floors_interior_tilesandwood_01", 0)
    west, north = WALL_SCENES[chain]

    def lay(runs):
        out, at = {}, 1
        for dept, length in runs:
            rec = next(v for v in built["signs"][chain][dept]["versions"] if v["length"] == length)
            for i in range(length):
                out[at + i] = (rec, i)
            at += length + 1
        return out
    on_w, on_n = lay(west), lay(north)
    ox, oy = 64 * ROOM + 40, 20
    img = Image.new("RGBA", (128 * ROOM + 120, 64 * ROOM + 300), (40, 40, 42, 255))
    for s in range(2 * ROOM - 1):                                  # back to front
        for x in range(ROOM):
            y = s - x
            if not 0 <= y < ROOM:
                continue
            sx, sy = ox + 64 * (x - y), oy + 32 * (x + y)
            if floor is not None:
                img.alpha_composite(floor, (sx, sy))
            if x == 0 and wall_w is not None:
                img.alpha_composite(wall_w, (sx, sy))
                if y in on_w:
                    rec, i = on_w[y]
                    img.alpha_composite(_sheet_tile(cache, rec["W"][i]), (sx, sy))
            if y == 0 and wall_n is not None:
                img.alpha_composite(wall_n, (sx, sy))
                if x in on_n:
                    rec, i = on_n[x]
                    img.alpha_composite(_sheet_tile(cache, rec["N"][i]), (sx, sy))
    return img


def _preview(built: dict) -> None:
    os.makedirs(SF.PREVIEW, exist_ok=True)
    _contact()
    cache: dict = {}
    corners = [_corner(built, chain, cache) for chain in LIVERY]
    img = Image.new("RGBA", (sum(c.width for c in corners), corners[0].height), (40, 40, 42, 255))
    x = 0
    for c in corners:
        img.alpha_composite(c, (x, 0))
        x += c.width
    img.save(os.path.join(SF.PREVIEW, "dept_signs_wall.png"))
    img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(
        os.path.join(SF.PREVIEW, "dept_signs_wall_game_scale.png"))


if __name__ == "__main__":
    print(build("--preview" in sys.argv))
