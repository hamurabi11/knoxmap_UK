"""A 1993 petrol station's forecourt (PLAN.md step 5): pumps, canopy, price sign.

    python petrol.py [--preview]   (KnoxMap's Python)

Sheet commercial_fuel_uk_01 (its DEPTH_ sheet, .slots.json) and built/fuel.json (the
tiles by part, for Build: knoxbuild/pumps.py lays them on a UK map's forecourts):

    pump_x, pump_y   a 1990s pump on its kerbed island, its display and nozzles facing
                     south (in a row west-east) / east (north-south); each slot takes
                     vanilla's pump's properties (location_shop_fossoil_01_14 / _12:
                     it holds fuel, works with a hose and a can)
    column           a canopy's white steel column
    roof             the canopy's flat top, a storey up (laid at z 1)
    band_S/E/N/W     the canopy's fascia band in the brand's colour on that edge of a
                     square; the S and E bands (the camera's) letter the brand's name
                     across BAND_LETTERS squares, the rest plain (band_*_plain)
    price_lo/hi      the price sign on its pole facing south (price_e_*: east), its two storeys: the brand, then
                     UNLEADED / 4 STAR / DIESEL in pence a litre (1993: 46.9, 50.9, 45.9)

Drawn with farm_props.Sprite (faces in metres through walls.py's camera, with depth).
One brand for now: TEXECO (names.py: Texaco and Valero, Odiham's), red, white letters.
Appended only: a part's place fixes its ids.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import farm_props as FP  # noqa: E402
import fascias as FA  # noqa: E402

WK, F, W = FP.WK, FP.F, FP.W
SHEET = "commercial_fuel_uk_01"
STOREY_M = 3.0
BRAND = {"name": "TEXECO", "ground": (200, 22, 32), "ink": (250, 250, 246)}
WHITE, STEEL = (236, 236, 232), (210, 212, 212)
BAND = (2.35, 3.0)                 # m: the canopy's band, under its roof at the storey's top
BAND_LETTERS = 4                   # squares the brand's name runs across on a band
PRICES = [("UNLEADED", "46.9"), ("4 STAR", "50.9"), ("DIESEL", "45.9")]


def _flat(rgb, rng, n=16, grain=2.0):
    return FP._tex(rgb, rng, n=n, grain=grain)


def _clip_box(s, x0, x1, y0, y1, z0, z1, side, top, level):
    """A box's part on storey `level`, drawn a storey down for its own sprite."""
    lo, hi = max(z0, level * STOREY_M), min(z1, (level + 1) * STOREY_M)
    if hi > lo:
        s.box(x0, x1, y0, y1, lo - level * STOREY_M, hi - level * STOREY_M, side, top)


def pump(rng, facing):
    """facing S: the display on the south face (a west-east row); E: on the east."""
    s = FP.Sprite()
    island = _flat((176, 174, 166), rng)
    island[:3, :, :3] = (214, 172, 40)                      # its kerb's yellow edge
    s.box(0.12, 0.88, 0.12, 0.88, 0.0, 0.14, island, _flat((168, 166, 160), rng))
    body = _flat(WHITE, rng, grain=1.5)
    head = _flat(BRAND["ground"], rng, grain=1.5)
    # the display and the nozzles' holsters on the facing side
    # (review r1: a plain modern cabinet, lost under the canopy) boxier and taller, a big
    # dark display panel, the brand's band across it, the hose holsters either side
    face = _flat(WHITE, rng, grain=1.5)
    face[1:7, 1:15, :3] = (28, 30, 32)                      # the display panel
    face[2:4, 2:14, :3] = (210, 120, 40)                    # its mechanical figures, amber
    face[5:6, 3:13, :3] = (210, 120, 40)
    face[8:10, :, :3] = BRAND["ground"]                     # the brand's band
    face[11:15, 0:3, :3] = (24, 24, 24)                     # the nozzles' holsters
    face[11:15, 13:16, :3] = (24, 24, 24)
    x0, x1, y0, y1 = (0.24, 0.76, 0.32, 0.68) if facing == "S" else (0.32, 0.68, 0.24, 0.76)
    s.box(x0, x1, y0, y1, 0.14, 1.75, body, body)
    s.box(x0 - 0.02, x1 + 0.02, y0 - 0.02, y1 + 0.02, 1.75, 2.1, head, _flat(np.array(BRAND["ground"]) * 0.9, rng))
    if facing == "S":
        s.face((x0, y1 + 0.005, 1.65), (x1 - x0, 0, 0), (0, 0, -1.3), face, (0, 1, 0))
    else:
        s.face((x1 + 0.005, y0, 1.65), (0, y1 - y0, 0), (0, 0, -1.3), face, (1, 0, 0))
    return s.draw()


def column(rng):
    s = FP.Sprite()
    s.box(0.42, 0.58, 0.42, 0.58, 0.0, BAND[0], _flat(WHITE, rng), _flat(WHITE, rng))
    s.box(0.38, 0.62, 0.38, 0.62, 0.0, 0.5, _flat((196, 196, 192), rng))   # its foot's guard
    return s.draw()


def roof(rng):
    """The canopy's top, a storey up: flat, white, a faint panel joint."""
    s = FP.Sprite()
    # (review r1: a joint each square read as floor tiles; white read as the ground's)
    t = _flat((192, 194, 194), rng, n=32, grain=2.0)
    s.face((0, 0, 0), (1, 0, 0), (0, 1, 0), t, (0, 0, 1))
    return s.draw()


def band_strip(letters: bool) -> np.ndarray:
    """The band's face along BAND_LETTERS squares (or one, plain): RGBA, 64 px a metre."""
    n = BAND_LETTERS if letters else 1
    h = int(round((BAND[1] - BAND[0]) * 64))
    img = Image.new("RGBA", (64 * n, h), BRAND["ground"] + (255,))
    d = ImageDraw.Draw(img)
    d.rectangle([0, h - 4, 64 * n, h], fill=(240, 240, 236, 255))          # the white trim under it
    if letters:
        d.text((32 * n, (h - 4) // 2), BRAND["name"], font=FA._font("sans", 30), fill=BRAND["ink"] + (255,), anchor="mm")
    a = np.asarray(img).astype(float)
    return a


def band(edge: str, piece: np.ndarray):
    """One square's band on its `edge` (S, E, N, W) from the strip's piece (64 x h)."""
    s = FP.Sprite()
    z0, z1 = BAND
    tex = piece
    if edge == "S":
        s.face((0, 1, z1), (1, 0, 0), (0, 0, z0 - z1), tex, (0, 1, 0))
    elif edge == "E":
        s.face((1, 0, z1), (0, 1, 0), (0, 0, z0 - z1), tex, (1, 0, 0))
    elif edge == "N":                                        # its inner face, the one the camera sees
        s.face((0, 0, z1), (1, 0, 0), (0, 0, z0 - z1), tex, (0, 1, 0))
    else:
        s.face((0, 0, z1), (0, 1, 0), (0, 0, z0 - z1), tex, (1, 0, 0))
    return FA.bleed(*s.draw())                              # (review r1: a seam at each join)


def price_panel() -> np.ndarray:
    """The price sign's panel: the brand on red, then the three prices on white."""
    w, h = 64, 160
    img = Image.new("RGBA", (w, h), (246, 246, 240, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 44], fill=BRAND["ground"] + (255,))
    d.text((w // 2, 22), BRAND["name"], font=FA._font("sans", 13), fill=BRAND["ink"] + (255,), anchor="mm")
    for k, (fuel, p) in enumerate(PRICES):
        y = 60 + 34 * k
        d.text((w // 2, y), fuel, font=FA._font("sans", 8), fill=(30, 30, 30, 255), anchor="mm")   # (review: margins)
        d.text((w // 2, y + 14), p, font=FA._font("sans", 14), fill=(20, 20, 20, 255), anchor="mm")
    d.rectangle([0, 0, w - 1, h - 1], outline=(150, 150, 150, 255))
    return np.asarray(img).astype(float)


def price_sign(rng, level, facing):
    """facing S or E: one panel square to the road (an east face's texture runs right to
    left on screen: mirrored, so it reads)."""
    s = FP.Sprite()
    pole = _flat((150, 152, 152), rng)
    _clip_box(s, 0.46, 0.54, 0.46, 0.54, 0.0, 2.2, pole, pole, level)
    # the panel: 0.9 m wide, 2.2 m to 4.7 m up, square to the road
    panel = price_panel()
    z0, z1 = 2.2, 4.7
    lo, hi = max(z0, level * STOREY_M), min(z1, (level + 1) * STOREY_M)
    if hi > lo:
        rows = panel.shape[0]
        r0, r1 = int((z1 - hi) / (z1 - z0) * rows), int((z1 - lo) / (z1 - z0) * rows)
        part = panel[r0:max(r1, r0 + 1)]
        zt, zb = hi - level * STOREY_M, lo - level * STOREY_M
        edge = _flat((120, 120, 120), rng)
        if facing == "S":
            s.box(0.05, 0.95, 0.5, 0.56, zb, zt, edge, edge)
            s.face((0.05, 0.565, zt), (0.9, 0, 0), (0, 0, zb - zt), part, (0, 1, 0))
        else:
            s.box(0.5, 0.56, 0.05, 0.95, zb, zt, edge, edge)
            s.face((0.565, 0.05, zt), (0, 0.9, 0), (0, 0, zb - zt), np.ascontiguousarray(part[:, ::-1]), (1, 0, 0))
    return s.draw()


PARTS = [  # (name, maker, the vanilla piece whose properties its slot takes); appended only
    ("pump_x", lambda r: pump(r, "S"), {"like": "location_shop_fossoil_01_14"}),
    ("pump_y", lambda r: pump(r, "E"), {"like": "location_shop_fossoil_01_12"}),
    ("column", column, None),
    ("roof", roof, None),
    ("price_lo", lambda r: price_sign(r, 0, "S"), None),
    ("price_hi", lambda r: price_sign(r, 1, "S"), None),
    ("price_e_lo", lambda r: price_sign(r, 0, "E"), None),
    ("price_e_hi", lambda r: price_sign(r, 1, "E"), None),
]


def main(argv) -> int:
    rng = np.random.default_rng(1993)
    tiles, depths, slots, built = [], [], [], {"sheet": SHEET, "brand": BRAND["name"], "parts": {}}

    def add(name, piece, slot=None):
        t, d = piece
        built["parts"][name] = f"{SHEET}_{len(tiles)}"
        tiles.append(t)
        depths.append(d)
        if slot is None and name in ("column", "price_lo", "price_e_lo"):
            slot = {"props": [["solidtrans", ""], ["CustomName", "Petrol Station"]]}
        slots.append(slot)
    for name, make, vanilla in PARTS:
        add(name, make(rng), vanilla)
    # the bands: the S and E edges lettered across BAND_LETTERS squares (in run order, low
    # x or y first; a west-to-east south band reads left to right, a north-to-south east
    # band too), the plain piece for the rest; the N and W bands plain
    strip, plain = band_strip(True), band_strip(False)
    for edge in ("S", "E"):
        names = []
        for i in range(BAND_LETTERS):
            names.append(None)
            if edge == "E":      # runs right to left on screen as y grows: the strip from its end, mirrored
                piece = np.ascontiguousarray(strip[:, 64 * (BAND_LETTERS - 1 - i):64 * (BAND_LETTERS - i)][:, ::-1])
            else:
                piece = strip[:, 64 * i:64 * (i + 1)]
            add(f"band_{edge}_{i}", band(edge, piece))
            names[-1] = built["parts"][f"band_{edge}_{i}"]
        built["parts"][f"band_{edge}_letters"] = names
    for edge in ("S", "E", "N", "W"):
        add(f"band_{edge}_plain", band(edge, plain))
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"), [W.depth_tile(t, d) for t, d in zip(tiles, depths)])
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f)
    os.makedirs(FA.BUILT, exist_ok=True)
    with open(os.path.join(FA.BUILT, "fuel.json"), "w", encoding="utf-8") as f:
        json.dump(built, f, indent=1)
    if "--preview" in argv:
        preview(tiles, built)
    order = os.path.join(HERE, "sheets.order.json")
    have = json.load(open(order, encoding="utf-8"))
    if SHEET not in have:
        have.append(SHEET)
    json.dump(have, open(order, "w", encoding="utf-8"), indent=1)
    print("wrote", SHEET, len(tiles))
    return 0


def preview(tiles, built) -> None:
    """A forecourt of 6 x 4 squares: the canopy on four columns, two pumps under it, the
    price sign at its corner; drawn square by square back to front at the game's scale."""
    idx = {n: int(v.rsplit("_", 1)[1]) for n, v in built["parts"].items() if isinstance(v, str)}
    img = Image.new("RGBA", (900, 640), (70, 72, 70, 255))
    ox, oy = 420, 120

    def put(t, x, y, z=0):
        img.alpha_composite(Image.fromarray(tiles[t]), (ox + 64 * (x - y), oy + 32 * (x + y) - 192 * z))
    cw, ch = 6, 4
    layer0, layer1 = [], []
    for y in range(ch):
        for x in range(cw):
            if (x, y) in ((0, 0), (cw - 1, 0), (0, ch - 1), (cw - 1, ch - 1)):
                layer0.append((idx["column"], x, y))
            if y == 1 and x in (1, 4):
                layer0.append((idx["pump_x"], x, y))
            if y == 0:
                layer0.append((idx["band_N_plain"], x, y))
            if x == 0:
                layer0.append((idx["band_W_plain"], x, y))
            if y == ch - 1:
                k = x - (cw - BAND_LETTERS) // 2
                layer0.append((idx[f"band_S_{k}"] if 0 <= k < BAND_LETTERS else idx["band_S_plain"], x, y))
            if x == cw - 1:
                k = y
                layer0.append((idx[f"band_E_{k}"] if 0 <= k < BAND_LETTERS else idx["band_E_plain"], x, y))
            layer1.append((idx["roof"], x, y))
    layer0.append((idx["price_lo"], 1, ch + 1))
    layer1.append((idx["price_hi"], 1, ch + 1))
    layer0.append((idx["price_e_lo"], cw + 1, 1))
    layer1.append((idx["price_e_hi"], cw + 1, 1))
    for t, x, y in sorted(layer0, key=lambda q: q[1] + q[2]):
        put(t, x, y)
    for t, x, y in sorted(layer1, key=lambda q: q[1] + q[2]):
        put(t, x, y, 1)
    img.save(os.path.join(FP.SM.PREVIEW, "petrol_forecourt.png"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
