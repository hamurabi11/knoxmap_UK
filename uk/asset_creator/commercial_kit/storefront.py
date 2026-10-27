"""A 1990s superstore's front seen from the street: its canopy, and posters in its windows.

    python storefront.py [--preview]   (KnoxMap's Python)

A Lainburys or Waitrows of about 1993 had a flat steel-and-glass canopy along its
front, two or three squares deep, at the top of the ground storey: patent glazing on
steel bars running out from the wall, slim steel columns along its outer edge, a band
in the chain's colour round it; and in the windows under it the week's posters (PRICE
CHECK, 2 FOR £3...) and the coffee shop's sign. Bollards along the pavement are the
street kit's (street_furniture_uk_02_12, asset_creator/src/street_objects.py: black
cast iron, a white band; vanilla has none, its "Post Barrier" is a car park's boom),
so none are drawn here. A car park and trolley shelters are another tool's.

Drawn as petrol.py draws its forecourt canopy (farm_props.Sprite: faces in metres
through walls.py's camera, with depth) and dept_signs.py its boards (each chain's
livery, dept_signs.LIVERY: Lainburys white on orange, Waitrows white on green, any
other store white on dark blue).

Sheet storefront_uk_01 (its DEPTH_ sheet, .slots.json), slots appended only (a part's
place fixes its id):

    0-2    top_x: the canopy's top for a run along x (a south-facing front): glass
           between steel bars running north-south, a bar at each square's edge and
           one down its middle. 0 clean, 1 weathered, 2 a steel beam down the
           middle in place of the bar (over a column's square)
    3-5    top_y: the same for a run along y (an east-facing front), the bars east-west
    6-     per chain (CHAINS' order), BLOCK slots each:
           +0 column, +1 band S, +2 band E, +3 top_end E (a run along x's east end),
           +4 top_end S (a run along y's south end), +5 top_end W (its west end),
           +6 top_end N (its north end), +7.. the posters (POSTERS' order), each W then N, then the cafe sign W, N

The glass is see-through (GLASS_A), as a shop window's is: under the canopy the
front and its posters show, as they do in the street. The column, the bands and the
posters are opaque.

Each slot's properties (.slots.json, pack.py): the column {"props": solidtrans,
CustomName "Canopy"}, as petrol's column; the rest none (a top, a band, a poster).

built/storefront.json, for Build:

    {"sheets": ["storefront_uk_01"],
     "layers": {part: [layer, z]},
     "bollard": "street_furniture_uk_02_12",
     "canopy": {chain: {"top_x": [3], "top_y": [3], "column": t,
                        "band_x": t, "band_y": t,
                        "end_x": {"E": t}, "end_y": {"S": t},
                        "top_end": {"W": t, "E": t, "N": t, "S": t}}},
     "posters": {chain: {"W": [6], "N": [6]}, ...},
     "poster_text": [6],
     "cafe": {chain: {"W": t, "N": t}}}

chain: lainburys, waitrows, generic. Tile names unpadded, as the kit's others
("storefront_uk_01_23"). Where each goes (also in "layers"):

    top_end         RoofTop, z 1, in place of top_x on a run along x's end squares
                    (W, E), of top_y on a run along y's (N, S): a solid bay 0.6 m wide
                    in the band's colour, each end alike
    top_x / top_y   RoofTop, z 1, on every other square of the canopy. top[0] and top[1]
                    mixed along the run; top[2] on each row of a column's line (the beam out to it)
    column          Furniture, z 0, on the canopy's outer row, every third or fourth square
    band_x          Furniture2, z 0: the S edge of the outer row of a run along x
    band_y          Furniture3, z 0: the E edge of the outer row of a run along y
    end_x           Furniture3, z 0: a run along x's E end, each square across its depth
                    (band_y's piece: the same band, turned the corner)
    end_y           Furniture2, z 0: a run along y's S end (band_x's piece)
                    The far ends (W, N) have no band: their return faces inwards, under
                    top_end's bay, and past the bay it stood over the front band's
                    corner (review r2: a darker stub, its stripe out of line)
    bollard         Furniture, z 0, one a square along the pavement's kerb
    posters, cafe   WallFurniture, z 0, on a glazed square of the front: W on a
                    west-edge wall (an east-facing front), N on a north-edge wall
                    (a south-facing front); a poster a square, not every square

A square's band pieces are on the layer of its edge as petrol's (Furniture2 north and
south, Furniture3 east and west: a corner square holds two).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dept_signs as DS  # noqa: E402
import petrol as P  # noqa: E402

FP, FA, WK, W = P.FP, P.FA, P.WK, P.W
SF, WIN = FA.SF, FA.WIN
PREFIX = "storefront_uk"
SHEET = f"{PREFIX}_01"
SLOTS = FA.SLOTS
CHAINS = ("lainburys", "waitrows", "generic")
BOLLARD = "street_furniture_uk_02_12"
STOREY_M = P.STOREY_M

BAND = (2.5, 3.0)              # m: the band, under the canopy's top at the storey's top
STEEL = (196, 198, 198)        # the glazing bars, painted light grey
BEAM = (150, 154, 156)
GLASS = (150, 168, 180)        # the glass from above: a sky's reflection over the pavement
GLASS_A = 96                   # its alpha (shopfronts' window glass: 118)
COLUMN = {"lainburys": (84, 86, 90), "waitrows": DS.LIVERY["waitrows"][0], "generic": (140, 144, 146)}

# the posters: (key, the words on it, for Build's records); drawn by poster()
POSTERS = [("price_check", "PRICE CHECK"), ("two_for", "2 FOR £3"), ("fresh", "FRESH FOR LESS"),
           ("open_hours", "OPEN 8AM-10PM"), ("lower_prices", "NEW LOWER PRICES"), ("half_price", "HALF PRICE")]
POSTER_ROWS = (50, 130)        # rows of the storey (0 at its top): 2.2 m down to 0.97 m up
POSTER_COLS = (4, 60)          # 0.875 m wide, inside the front's glass (shopfronts.modern: 0.76 m down)
BLOCK = 7 + 2 * len(POSTERS) + 2
WHITE = (250, 248, 242)
INK = (24, 24, 26)


def _ground(chain):
    return tuple(DS.LIVERY[chain][0])


def _face(chain):
    return DS.LIVERY[chain][2]


# --- the canopy -------------------------------------------------------------------------------------

def top_texture(variant: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """The top for a run along x, 64 x 64 (rows along y, columns along x): its colour and
    alpha. It changes along x only, so the squares of its depth meet without a joint;
    a bar at each edge is half a bar, two squares make one."""
    n = 64
    x = np.arange(n)
    bay = (x % 32) / 32                                            # across each half-metre bay
    rgb = np.ones((n, n, 3)) * np.array(GLASS, float)
    rgb *= (0.92 + 0.16 * np.clip(1 - np.abs(bay - 0.35) * 2.5, 0, 1))[None, :, None]   # a sheen across it
    rgb += rng.normal(0, 1.5, (n, n, 1))
    alpha = np.full((n, n), float(GLASS_A))
    bars = [(0, 2), (62, 64)] + ([] if variant == 2 else [(31, 33)])
    for c0, c1 in bars:
        rgb[:, c0:c1] = STEEL
        alpha[:, c0:c1] = 255
    rgb[:, 0] *= 0.82                                              # each bar's shadowed side
    rgb[:, 33 if variant != 2 else 0] *= 0.9
    if variant == 1:
        # weathered: grime washed against the bars (clear of the tile's top and bottom
        # rows: no seam with the next square; review r1: dead leaves read as stray pixels)
        for c0, c1 in ((2, 6), (27, 31), (33, 37), (58, 62)):
            rgb[4:60, c0:c1] = rgb[4:60, c0:c1] * 0.78 + np.array((92, 96, 80)) * 0.22
            alpha[4:60, c0:c1] = 150
    if variant == 2:
        rgb[:, 27:37] = BEAM                                       # the beam over the column
        rgb[:, 27] = np.array(BEAM) * 1.15
        rgb[:, 36] = np.array(BEAM) * 0.75
        alpha[:, 27:37] = 255
    return np.clip(rgb, 0, 255), alpha


END_BAY = 38                   # px of 64: the solid end bay over a W or N end's band (0.6 m)


def end_texture(chain: str, rng, near: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A run along x's west end square (`near`: its east end, mirrored): the glass, then
    the solid end bay over the band's return, in the band's colour, its inner edge lit.
    Review r2: both ends alike. Review r1: the glass a level up
    drew over the return's inner face, a see-through ghost box; the bay covers it (it
    covers just the face's reach: 0.5 m of band seen from 45 degrees)."""
    rgb, alpha = top_texture(0, rng)
    g = np.array(_ground(chain), float)
    rgb[:, :END_BAY] = g * 0.92 + rng.normal(0, 1.2, (64, END_BAY, 1))
    rgb[:, END_BAY - 2:END_BAY] = np.minimum(g * 1.3 + 25, 255)
    rgb[:, END_BAY] = STEEL
    alpha[:, :END_BAY + 1] = 255
    if near:
        rgb, alpha = rgb[:, ::-1], alpha[:, ::-1]
    return np.clip(rgb, 0, 255), alpha


def top(variant, along: str, rng, chain: str | None = None, near: bool = False):
    """The top's sprite, a storey up (laid at z 1): its colour drawn as petrol's roof,
    then its alpha drawn the same way (walls._draw paints a texel opaque or not at all)."""
    rgb, alpha = top_texture(variant, rng) if chain is None else end_texture(chain, rng, near)
    if along == "y":
        rgb, alpha = rgb.transpose(1, 0, 2), alpha.T
    out = np.zeros((256, 128, 4), np.uint8)
    depth = np.full((256, 128), np.nan)
    o, a, b = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    W._draw(out, o, a, b, rgb, FP._light(np.array((0.0, 0.0, 1.0))), depth=depth)
    mask = np.zeros_like(out)
    W._draw(mask, o, a, b, np.repeat(alpha[..., None], 3, axis=2), 1.0)
    out[..., 3] = np.where(out[..., 3] > 0, mask[..., 0], 0)
    return out, depth


def column(chain: str, rng):
    """A slim round steel column, its base plate and its cap under the top."""
    s = FP.Sprite()
    paint = COLUMN[chain]
    side = P._flat(paint, rng, grain=1.2)
    side[:, :2, :3] *= 1.18                                        # a lit edge down it
    s.cylinder((0.5, 0.5, 0.03), (0, 0, 1), 0.085, 2.85, side, P._flat(paint, rng))
    s.box(0.38, 0.62, 0.38, 0.62, 0.0, 0.03, P._flat((120, 122, 122), rng))          # base plate
    s.box(0.40, 0.60, 0.40, 0.60, 2.86, 2.96, P._flat(np.array(paint) * 0.85, rng))  # its cap
    return s.draw()


def band_texture(chain: str) -> np.ndarray:
    """The band's face: the chain's colour, a thin stripe in its lettering's colour near
    its foot, a light flashing at its top, a shadow line under it (64 x rows)."""
    ground, ink = _ground(chain), tuple(DS.LIVERY[chain][1])
    h = int(round((BAND[1] - BAND[0]) * 64))
    img = Image.new("RGBA", (64, h), ground + (255,))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 64, 1], fill=tuple(min(255, int(c * 1.25) + 20) for c in ground) + (255,))
    d.rectangle([0, h - 9, 64, h - 7], fill=ink + (255,))
    d.rectangle([0, h - 2, 64, h], fill=tuple(int(c * 0.55) for c in ground) + (255,))
    return np.asarray(img).astype(float)


def band(chain: str, edge: str, rng):
    """One square's band on its `edge` (S or E): its outer face. No top: the canopy's
    top, a level up, is drawn over it (review r1: its glass tinted the band's top edge
    to a coloured lip). No W or N piece: see end_x."""
    s = FP.Sprite()
    z0, z1 = BAND
    tex = band_texture(chain)
    if edge == "S":
        s.face((0, 1, z1), (1, 0, 0), (0, 0, z0 - z1), tex, (0, 1, 0))
    else:
        s.face((1, 0, z1), (0, 1, 0), (0, 0, z0 - z1), tex, (1, 0, 0))
    return FA.bleed(*s.draw())                                    # (petrol review r1: a seam at each join)


# --- the windows' posters ----------------------------------------------------------------------------

SQUEEZE = 0.72                 # a poster's words may be drawn this much narrower than the face's


def _say(img, xy, text, face, size, fill, room=50, outline=None, squeeze=SQUEEZE):
    """`text` centred on xy at `size`, squeezed across to fit `room` (a poster's condensed
    letters: review r1, at the face's own width the small lines were unreadable in game),
    smaller only past SQUEEZE; with a dark `outline` round it (vinyl on glass)."""
    while size > 8 and FA._text_width(text, face, size) * squeeze > room:
        size -= 1
    f = FA._font(face, size)
    tw = FA._text_width(text, face, size)
    pad = 2
    tmp = Image.new("RGBA", (tw + 2 * pad, 2 * size + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((tmp.width // 2, tmp.height // 2), text, font=f, fill=fill, anchor="mm",
                             stroke_width=1 if outline else 0, stroke_fill=outline)
    tmp = tmp.crop(tmp.getbbox())
    if tmp.width > room:
        tmp = tmp.resize((room, tmp.height), Image.LANCZOS)
    img.alpha_composite(tmp, (int(xy[0] - tmp.width / 2), int(xy[1] - tmp.height / 2)))


def poster(chain: str, key: str) -> Image.Image:
    """A window poster, POSTER_COLS wide by POSTER_ROWS high (an A0 sheet, 0.875 x 1.25 m),
    in the chain's livery: one big word or figure each, read at the game's scale as
    the week's offer; the small words for a closer look."""
    w, h = POSTER_COLS[1] - POSTER_COLS[0], POSTER_ROWS[1] - POSTER_ROWS[0]
    g, face = _ground(chain), _face(chain)
    G, Wt, K = g + (255,), WHITE + (255,), INK + (255,)
    img = Image.new("RGBA", (w, h), Wt)
    d = ImageDraw.Draw(img)
    cx = w // 2
    if key == "price_check":
        d.rectangle([0, 0, w, 50], fill=G)
        _say(img, (cx, 15), "PRICE", face, 19, Wt)
        _say(img, (cx, 36), "CHECK", face, 19, Wt)
        d.line([(15, 63), (24, 72), (42, 54)], fill=G, width=6)                  # a tick
    elif key == "two_for":
        _say(img, (cx, 13), "2 FOR", "sans", 17, G)
        _say(img, (cx, 46), "£3", "sans", 34, G)
        d.rectangle([4, 68, w - 5, 72], fill=G)
    elif key == "fresh":
        d.rectangle([0, 0, w, h], fill=(70, 128, 52, 255))
        _say(img, (cx, 12), "FRESH", face, 18, Wt)
        d.ellipse([10, 24, 34, 48], fill=(206, 40, 36, 255))                       # an apple
        d.ellipse([24, 28, 46, 50], fill=(232, 196, 40, 255))                      # and a lemon
        d.line([(22, 24), (25, 19)], fill=(70, 50, 30, 255), width=2)
        d.rectangle([0, 56, w, h], fill=Wt)
        _say(img, (cx, 67), "FOR LESS", "sans", 15, G)
    elif key == "open_hours":
        d.rectangle([0, 0, w, 24], fill=G)
        _say(img, (cx, 12), "OPEN", face, 19, Wt)
        _say(img, (cx, 39), "8am", "sans", 20, K)
        d.rectangle([cx - 6, 52, cx + 6, 54], fill=K)                             # "to"
        _say(img, (cx, 67), "10pm", "sans", 20, K)
    elif key == "lower_prices":
        # (review r2: in NOW OPEN SUNDAYS' slot - Sunday opening came with the 1994 Act)
        d.rectangle([0, 0, w, 44], fill=G)
        _say(img, (cx, 11), "NEW", "sans", 14, Wt)
        _say(img, (cx, 31), "LOWER", face, 19, Wt)
        _say(img, (cx, 54), "PRICES", "sans", 17, G, room=w - 8, squeeze=0.6)
        d.polygon([(cx - 9, 64), (cx + 9, 64), (cx, 75)], fill=(214, 30, 36, 255))   # down
    elif key == "half_price":
        d.rectangle([0, 0, w, h], fill=(214, 30, 36, 255))                         # a sale's red
        _say(img, (cx, 30), "½", "sans", 46, (252, 226, 60, 255))
        _say(img, (cx, 62), "PRICE", "sans", 18, Wt)
    d.rectangle([0, 0, w - 1, h - 1], outline=tuple(int(c * 0.6) for c in g) + (255,))
    return img


def cafe(chain: str) -> Image.Image:
    """The coffee shop's window sign: white vinyl on the glass, a cup and saucer with its
    steam, COFFEE SHOP under it, a rule in the chain's colour; all edged dark (review r1:
    seen through the canopy, against the lit shop, plain white washed out)."""
    w, h = POSTER_COLS[1] - POSTER_COLS[0], POSTER_ROWS[1] - POSTER_ROWS[0]
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    Wt, G = WHITE + (255,), _ground(chain) + (255,)
    cx = w // 2
    for k in (-7, 0, 7):                                           # the steam
        d.arc([cx + k - 3, 6, cx + k + 3, 14], 90, 270, fill=Wt, width=2)
        d.arc([cx + k - 3, 13, cx + k + 3, 21], 270, 90, fill=Wt, width=2)
    d.rectangle([cx - 13, 24, cx + 11, 27], fill=Wt)               # the cup: its rim,
    d.chord([cx - 13, 10, cx + 11, 44], 0, 180, fill=Wt)           # its bowl,
    d.ellipse([cx + 8, 28, cx + 18, 38], outline=Wt, width=3)      # its handle
    d.rectangle([cx - 20, 45, cx + 18, 48], fill=Wt)               # the saucer
    d.rectangle([4, 52, w - 5, 55], fill=G)
    _say(img, (cx, 63), "COFFEE", "sans", 15, Wt, room=w - 6)
    _say(img, (cx, 74), "SHOP", "sans", 13, Wt, room=w - 6)
    a = np.asarray(img).copy()
    grown = np.asarray(Image.fromarray(a[..., 3]).filter(ImageFilter.MaxFilter(3)))
    edge = (grown > 0) & (a[..., 3] == 0)
    a[edge] = (34, 36, 38, 255)
    return Image.fromarray(a)


def window_pieces(img: Image.Image) -> tuple:
    """A poster on a square of glazing: (W, N) sprites with their depths, stood off the
    wall as a fascia is (fascias._depth). A west wall's face runs right to left on
    screen as y grows (fascias.pieces): mirrored, so its words read."""
    face = np.zeros((SF.STOREY, 64, 4), np.uint8)
    face[POSTER_ROWS[0]:POSTER_ROWS[1], POSTER_COLS[0]:POSTER_COLS[1]] = np.asarray(img)
    wv = WIN.project(np.ascontiguousarray(face[:, ::-1]), "W")
    nv = WIN.project(face, "N")
    return (wv, FA._depth("W", wv)), (nv, FA._depth("N", nv))


# --- the sheet ---------------------------------------------------------------------------------------

LAYERS = {"top_x": ["RoofTop", 1], "top_y": ["RoofTop", 1], "top_end": ["RoofTop", 1], "column": ["Furniture", 0],
          "band_x": ["Furniture2", 0], "band_y": ["Furniture3", 0], "end_x": ["Furniture3", 0],
          "end_y": ["Furniture2", 0], "bollard": ["Furniture", 0], "posters": ["WallFurniture", 0],
          "cafe": ["WallFurniture", 0]}


def build(preview: bool = False) -> dict:
    rng = np.random.default_rng(1993)
    tiles, depths, slots = [], [], []

    def add(piece, slot=None) -> str:
        t, d = piece
        tiles.append(t)
        depths.append(d)
        slots.append(slot)
        return f"{SHEET}_{len(tiles) - 1}"                         # unpadded, as the kit's others
    tops = {along: [add(top(v, along, rng)) for v in range(3)] for along in ("x", "y")}
    built: dict = {"sheets": [SHEET], "layers": LAYERS, "bollard": BOLLARD, "canopy": {}, "posters": {},
                   "poster_text": [t for _k, t in POSTERS], "cafe": {}}
    for chain in CHAINS:
        col = add(column(chain, rng), {"props": [["solidtrans", ""], ["CustomName", "Canopy"]]})
        b = {e: add(band(chain, e, rng)) for e in "SE"}
        ends = {"E": add(top(None, "x", rng, chain, near=True)), "S": add(top(None, "y", rng, chain, near=True))}
        ends = {"W": add(top(None, "x", rng, chain)), "N": add(top(None, "y", rng, chain)), **ends}
        built["canopy"][chain] = {"top_x": tops["x"], "top_y": tops["y"], "column": col,
                                  "band_x": b["S"], "band_y": b["E"],
                                  "end_x": {"E": b["E"]}, "end_y": {"S": b["S"]},
                                  "top_end": ends}
        rec = {"W": [], "N": []}
        for key, _t in POSTERS:
            wv, nv = window_pieces(poster(chain, key))
            rec["W"].append(add(wv))
            rec["N"].append(add(nv))
        built["posters"][chain] = rec
        wv, nv = window_pieces(cafe(chain))
        built["cafe"][chain] = {"W": add(wv), "N": add(nv)}
    assert len(tiles) == 6 + BLOCK * len(CHAINS) <= SLOTS
    blank = np.zeros((256, 128, 4), np.uint8)
    pad = SLOTS - len(tiles)
    WK.F.write_sheet(WK.sheet_file(SHEET + ".png"), tiles + [blank] * pad)
    WK.F.write_sheet(WK.sheet_file(f"DEPTH_{SHEET}.png"),
                     [W.depth_tile(t, d) for t, d in zip(tiles, depths)] + [blank] * pad)
    with open(WK.sheet_file(SHEET + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots + [None] * pad, f)
    os.makedirs(FA.BUILT, exist_ok=True)
    with open(os.path.join(FA.BUILT, "storefront.json"), "w", encoding="utf-8") as f:
        json.dump(built, f, indent=1, ensure_ascii=False)
    order = os.path.join(HERE, "sheets.order.json")
    with open(order, encoding="utf-8") as f:
        have = json.load(f)
    new = [s for s in built["sheets"] if s not in have]
    if new:                                                       # appended only: the order fixes the pack's ids
        with open(order, "w", encoding="utf-8") as f:
            json.dump(have + new, f, indent=1)
    if preview:
        _preview(built)
    return {"sheets": built["sheets"], "tiles": len(tiles)}


# --- previews ----------------------------------------------------------------------------------------

PAVE = ("blends_street_01", 96)                                   # vanilla's pavement
FLOOR_IN = ("floors_interior_tilesandwood_01", 0)
FRONT = 7 * SF.BLOCK                                              # shopfronts' "silver" (modern) block
UPPER = "walls_commercial_uk_01_superstore_brick_upper"
RUN, DEEP = 10, 3


def _tile(cache: dict, name: str) -> Image.Image:
    sheet, k = name.rsplit("_", 1)
    if sheet not in cache:                                        # ours, else vanilla's
        path = WK.sheet_file(sheet + ".png")
        cache[sheet] = Image.open(path if os.path.exists(path) else
                                  os.path.join(WK.VANILLA, sheet + ".png")).convert("RGBA")
    img, k = cache[sheet], int(k)
    cols = img.width // 128
    r, c = divmod(k, cols)
    return img.crop((c * 128, r * 256, c * 128 + 128, r * 256 + 256))


def _scene(built: dict, chain: str, facing: str, cache: dict) -> Image.Image:
    """A store's front RUN squares long facing south (its wall on the N edge of row 0)
    or east (on the W edge of column 0), DEEP squares of canopy out from it, the
    pavement and a row of bollards beyond, the shop's floor behind its glass; its
    upper storey in brick with its big sign; drawn as the game draws, square by
    square back to front, a level at a time."""
    c = built["canopy"][chain]
    p = built["posters"][chain]
    big = json.load(open(os.path.join(FA.BUILT, "fascias_big.json"), encoding="utf-8"))["signs"].get(chain)
    lay: list = []                                                # (z, x, y, order, tile name)
    xs = range(-3, RUN + 1) if facing == "S" else range(-4, DEEP + 2)
    ys = range(-4, DEEP + 2) if facing == "S" else range(-3, RUN + 1)

    def along(x, y):                                             # (place along the run, out from the wall)
        return (x, y) if facing == "S" else (y, x)
    side = "N" if facing == "S" else "W"
    pos = {"pane": FRONT + (1 if side == "N" else 0), "door": FRONT + (7 if side == "N" else 6)}
    door = "fixtures_doors_01_49" if side == "N" else "fixtures_doors_01_48"
    for x in xs:
        for y in ys:
            a, o = along(x, y)
            inside = o < 0 and 0 <= a < RUN
            lay.append((0, x, y, 0, "_".join(map(str, FLOOR_IN)) if inside else "_".join(map(str, PAVE))))
            if not 0 <= a < RUN:
                continue
            if o == 0:                                            # the front: glazing, a door in it
                is_door = a == RUN // 2
                lay.append((0, x, y, 1, f"shopfronts_uk_01_{pos['door'] if is_door else pos['pane']}"))
                if is_door:
                    lay.append((0, x, y, 2, door))
                elif a in (1, 3, 6, 8):
                    k = [1, 3, 6, 8].index(a)
                    lay.append((0, x, y, 3, p[side][(k + (1 if chain == "waitrows" else 0)) % len(POSTERS)]))
                elif a == 2:
                    lay.append((0, x, y, 3, built["cafe"][chain][side]))
                lay.append((1, x, y, 1, f"{UPPER}_{1 if side == 'N' else 0}"))
                if big:
                    rec = next((v for v in big if v["length"] <= RUN), big[-1])
                    k = a - (RUN - rec["length"]) // 2
                    if 0 <= k < rec["length"]:
                        lay.append((1, x, y, 3, rec[side][k]))       # in run order (fascias.pieces)
            if 0 <= o < DEEP:                                     # the canopy
                v = 2 if a % 3 == 1 else (a * 7 + o * 3) % 2
                t = c[f"top_{'x' if facing == 'S' else 'y'}"][v]
                far, near = ("W", "E") if facing == "S" else ("N", "S")
                lay.append((1, x, y, 0, c["top_end"][far] if a == 0 else c["top_end"][near] if a == RUN - 1 else t))
                if o == DEEP - 1:
                    if a % 3 == 1:
                        lay.append((0, x, y, 4, c["column"]))
                    lay.append((0, x, y, 5, c["band_x"] if facing == "S" else c["band_y"]))
                if a == RUN - 1:
                    lay.append((0, x, y, 6, c["end_x"]["E"] if facing == "S" else c["end_y"]["S"]))
            if o == DEEP + 1:
                lay.append((0, x, y, 4, built["bollard"]))
    ox, oy = 64 * (RUN + 6), 440
    img = Image.new("RGBA", (128 * (RUN + 8), 32 * (RUN + DEEP) + 1000), (54, 56, 54, 255))
    for z, x, y, _o, name in sorted(lay, key=lambda q: (q[0], q[1] + q[2], q[1], q[3])):
        img.alpha_composite(_tile(cache, name), (ox + 64 * (x - y), oy + 32 * (x + y) - 192 * z))
    return img


def _crop(img: Image.Image, bg=(54, 56, 54)) -> Image.Image:
    a = np.asarray(img).astype(int)
    diff = np.abs(a[..., :3] - np.array(bg)).sum(-1) > 6
    ys, xs = np.nonzero(diff)
    return img.crop((max(0, xs.min() - 8), max(0, ys.min() - 8), xs.max() + 8, ys.max() + 8))


def _save(img: Image.Image, name: str) -> None:
    img.save(os.path.join(SF.PREVIEW, name + ".png"))
    img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(os.path.join(SF.PREVIEW, name + "_game_scale.png"))


def _stack(imgs, gap=12, horizontal=False):
    if horizontal:
        out = Image.new("RGBA", (sum(i.width for i in imgs) + gap * len(imgs), max(i.height for i in imgs)), (54, 56, 54, 255))
    else:
        out = Image.new("RGBA", (max(i.width for i in imgs), sum(i.height for i in imgs) + gap * len(imgs)), (54, 56, 54, 255))
    at = 0
    for i in imgs:
        out.alpha_composite(i, (at, 0) if horizontal else (0, at))
        at += (i.width if horizontal else i.height) + gap
    return out


def _preview(built: dict) -> None:
    os.makedirs(SF.PREVIEW, exist_ok=True)
    cache: dict = {}
    for facing, name in (("S", "storefront_south"), ("E", "storefront_east")):
        _save(_stack([_crop(_scene(built, ch, facing, cache)) for ch in CHAINS]), name)
    # the bollards in a row on the pavement, beside vanilla's props for scale
    img = Image.new("RGBA", (128 * 9, 64 * 9 + 200), (54, 56, 54, 255))
    lay = [(x, y, "_".join(map(str, PAVE))) for x in range(6) for y in range(3)]
    lay += [(x, 1, built["bollard"]) for x in range(6)]
    lay += [(1, 0, "street_decoration_01_8"), (4, 2, "street_decoration_01_56")]
    for x, y, n in sorted(lay, key=lambda q: (q[0] + q[1], 0 if q[2].startswith("blends") else 1)):
        img.alpha_composite(_tile(cache, n), (300 + 64 * (x - y), 60 + 32 * (x + y)))
    _save(_crop(img), "storefront_bollards")
    # the posters: flat, each chain a row; and on a glazed wall, N and W, no canopy over it
    rows = []
    for ch in CHAINS:
        imgs = [poster(ch, k) for k, _t in POSTERS] + [cafe(ch)]
        row = Image.new("RGBA", (len(imgs) * 64 + 8, 96), (62, 70, 74, 255))
        for i, im in enumerate(imgs):
            row.alpha_composite(im, (8 + 64 * i, 8))
        rows.append(row)
    _save(_stack(rows, 4), "storefront_posters_flat")
    walls = []
    for ch in CHAINS:
        p = built["posters"][ch]
        img = Image.new("RGBA", (128 * 10, 64 * 10 + 240), (54, 56, 54, 255))
        lay = []
        for k in range(8):                                        # a west wall (an east-facing front) y 0..7, x = 0
            lay.append((0, k, 0, "_".join(map(str, PAVE))))
            lay.append((0, k, 1, f"shopfronts_uk_01_{FRONT}"))
            lay.append((0, k, 2, p["W"][k] if k < len(POSTERS) else built["cafe"][ch]["W"] if k == 6 else None))
        for k in range(1, 8):                                     # and a north wall from its corner, x 1..7, y = 0
            lay.append((k, 0, 0, "_".join(map(str, PAVE))))
            lay.append((k, 0, 1, f"shopfronts_uk_01_{FRONT + 1}"))
            lay.append((k, 0, 2, p["N"][k - 1] if k - 1 < len(POSTERS) else built["cafe"][ch]["N"]))
        for x in range(1, 8):
            for y in range(1, 8):
                lay.append((x, y, 0, "_".join(map(str, PAVE))))
        for x, y, _o, n in sorted([q for q in lay if q[3]], key=lambda q: (q[0] + q[1], q[2])):
            img.alpha_composite(_tile(cache, n), (560 + 64 * (x - y), 40 + 32 * (x + y)))
        walls.append(_crop(img))
    _save(_stack(walls, 12, horizontal=True), "storefront_posters_wall")


if __name__ == "__main__":
    print(build("--preview" in sys.argv))
