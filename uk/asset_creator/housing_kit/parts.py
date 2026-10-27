"""Bay windows and porches for the housing styles (user, 2026-10-19: "have a go
at the bits that need geometry like bay windows and porches"; to-do K09.23).

    python parts.py   (KnoxMap's Python)

Two sheets, each piece for a window or door in a west wall (slot 2 k) and in a
north wall (2 k + 1), laid by Build on the window's or door's own square where
that square is outside the house (knoxbuild/tbx.py `_lay_bays`, the door
surround): a west wall's outside is then to its east, a north wall's to its
south - the faces the camera sees.

bays_uk_01 (BAYS): a canted bay - a front parallel to the wall and two sides
at 45 degrees - on a plinth, glazed, under a little hipped roof. Each style's
in its own wall material; a two-storey bay is a lower piece (no roof: the
upper sits on it) and an upper piece (with the roof), laid a storey apart:

    0  vic         Victorian: brick, white two-over-two sashes, slate roof (one storey)
    2  edw_lower   Edwardian: brick, white sashes (no roof)
    4  edw_upper   Edwardian upper: roughcast, white sashes, clay tile roof
    6  edw_single  Edwardian, one storey: brick, clay tile roof
    8  iw_lower    1930s: brick, white casements with leaded top lights (no roof)
   10  iw_upper    1930s upper: pebbledash, the same, clay tile roof
   12  iw_single   1930s, one storey: brick, clay tile roof
   14  e80         1980s: a shallow square bay, brick, stained casements, tiled lean-to

doorways_uk_02 (PORCHES): over a front door (house_walls.DOOR's opening):

    0  canopy_bracket  Victorian / Edwardian: a timber canopy, tiled lean-to, on two carved brackets
    2  porch_30s       1930s: an open porch, brick piers, a tiled gable
    4  canopy_flat     1950s-70s: a thin flat canopy, white fascia
    6  canopy_tiled    1980s: a tiled lean-to canopy on two stained timber brackets

Built from planes in metres on the square (wx east, wy south, z up) with
walls.py's camera and light, like the listed maker's doorway_kit; every
pixel's depth into the DEPTH_ sheet.
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

WK, W, F = HK.WK, HK.W, HK.F
DK = __import__("doorway_kit")
STOREY_M = 3.0
PX = 64                                    # texture px a metre (wall_kit's FACE)
WALL = DK.WALL
SLATE, TILE, FELT = (70, 74, 84), (120, 72, 58), (64, 62, 60)
WHITE, STAINED = HK.WHITE, HK.STAINED


def _mat(maker, seed=7):
    """A storey of a wall material (192 x 64 RGB), from housing_kit or wall_kit."""
    rng = np.random.default_rng(seed)
    return maker(rng)[:, :64].astype(float)


MATS = {
    "brick_vic": lambda: _mat(WK.red_brick),
    "brick_edw": lambda: _mat(HK.WALLS["brick_red_edw"][0]),
    "roughcast": lambda: _mat(lambda r: HK.pebbledash(r, (220, 217, 206), (1.06, 1.10), 236)),
    "pebbledash": lambda: _mat(HK.pebbledash),
    "brick_80s": lambda: _mat(HK.WALLS["brick_orange_80s"][0]),
}


class Bay:
    """One piece's planes in wall terms (out, along, z), in the order drawn
    (back to front); `render(k)` draws the part of them on the k-th square
    along the wall, in that square's own frame: a two-tile bay is one
    drawing cut into two sprites (user, 2026-10-19: "we might need 2 tile
    wide bays as well")."""

    def __init__(self, north: bool):
        self.north = north
        self.ops = []

    def v(self, out, along, z):
        return np.array((along, out, z) if self.north else (out, along, z), float)

    def face(self, p0, p1, z0, z1, tex, light):
        """A vertical face from wall point p0 = (out, along) to p1, z0..z1 (tex rows top down)."""
        self.plane((p0[0], p0[1], z1), (p1[0] - p0[0], p1[1] - p0[1], 0), (0, 0, -(z1 - z0)), tex, light)

    def plane(self, origin, a, b, tex, light):
        """origin + s a + t b; `b` must not run along the wall (clipping is along `a`)."""
        self.ops.append((np.array(origin, float), np.array(a, float), np.array(b, float), tex, light))

    def render(self, k: int = 0, last: bool = True):
        out = np.zeros((F.TILE_H, F.TILE_W, 4), np.uint8)
        depth = np.full((F.TILE_H, F.TILE_W), np.nan)
        for o, a, b, tex, light in self.ops:
            lo, hi = (-np.inf if k == 0 else k), (np.inf if last else k + 1)   # the end squares take the overhangs
            if abs(a[1]) < 1e-9:                                   # all at one place along the wall
                if not lo <= o[1] < hi:
                    continue
                s0, s1 = 0.0, 1.0
            else:
                s0, s1 = sorted(((lo - o[1]) / a[1], (hi - o[1]) / a[1]))
                s0, s1 = max(0.0, s0), min(1.0, s1)
                if s1 - s0 < 1e-6:
                    continue
            w = tex.shape[1]
            c0 = int(round(s0 * w))                                 # rounded: the two squares share no column (review)
            c1 = max(c0 + 1, int(round(s1 * w)))
            oo = o + s0 * a - np.array((0, k, 0))
            W._draw(out, self.v(*oo), self.v(*((s1 - s0) * a)), self.v(*b), tex[:, c0:c1], light, depth=depth)
        return out, depth


def _light(north: bool, normal_out: float, normal_along: float) -> float:
    """A face's light from its outward normal (out, along) in wall terms: east
    faces EAST_LIGHT, south faces 1.0, between them a mix."""
    n = np.array((normal_along, normal_out) if north else (normal_out, normal_along), float)
    n /= np.linalg.norm(n)
    east, south = max(n[0], 0), max(n[1], 0)
    tot = east + south or 1
    return (east * W.EAST_LIGHT + south * 1.0) / tot * (0.9 if n[1] < 0 or n[0] < 0 else 1.0)


def _window_face(mat, width_m, z0, z1, sill, head, frame, panes, lead=False, margin=3, fascia=None, string=None):
    """A face's texture (rows down from z1): the wall material, a window from
    `sill` to `head` (m above z0), `margin` px of wall (a corner post) each
    side; a fascia band (rgb) under the eaves, a gutter's shadow under it."""
    h, w = max(2, round((z1 - z0) * PX)), max(2, round(width_m * PX))
    tex = np.zeros((h, w, 4))
    rows = np.arange(h)
    tex[..., :3] = mat[(rows % mat.shape[0])][:, (np.arange(w) % mat.shape[1])]
    tex[..., 3] = 255
    y0, y1 = round((z1 - z0 - head) * PX), round((z1 - z0 - sill) * PX)
    x0, x1 = min(margin, w // 4), w - min(margin, w // 4)
    fr = np.array(frame, float)
    tex[y0:y1, x0:x1, :3] = fr
    f = 2
    ix0, iy0, ix1, iy1 = x0 + f, y0 + f, x1 - f, y1 - f
    if ix1 - ix0 >= 2 and iy1 - iy0 >= 4:
        a = np.zeros((iy1 - iy0, ix1 - ix0, 4), np.uint8)
        HK._pane(a, 0, 0, ix1 - ix0, iy1 - iy0)
        cols, rows_, transom = panes
        hh, ww = a.shape[:2]
        if transom:
            t = round(hh * transom)
            a[t - 1:t + 1, :, :3] = fr
            a[t - 1:t + 1, :, 3] = 255
            if lead:
                HK.WIN.lattice(a, 0, 0, ww, t - 1, (72, 76, 76), step=5)
        for k in range(1, cols):
            x = round(k * ww / cols)
            a[:, x - 1:x + 1, :3] = fr
            a[:, x - 1:x + 1, 3] = 255
        for k in range(1, rows_):
            y = round(k * hh / rows_)
            a[y - 1:y + 1, :, :3] = fr
            a[y - 1:y + 1, :, 3] = 255
        tex[iy0:iy1, ix0:ix1, :3] = a[..., :3]              # opaque: the bay's glass hides the wall behind it
        glassy = a[..., 3] < 255
        sub = tex[iy0:iy1, ix0:ix1]
        sub[glassy, :3] = sub[glassy, :3] * 0.78
    tex[y1:y1 + 2, :, :3] = (200, 196, 186)              # the cill
    if fascia is not None:
        f_px = round(0.12 * PX)
        tex[:f_px, :, :3] = fascia
        tex[f_px:f_px + 2, :, :3] *= 0.6                   # the gutter's shadow
    elif string is not None:
        tex[:3, :, :3] = string                            # a string course where the upper storey sits on it
    return tex


def _tiles(rgb, h=24, w=40, course=3, unit=5, tri=None):
    """Small roof tiles: a shadow under each course, joints broken course to
    course; `tri` "lo" keeps the triangle s + t <= 1 (a hip)."""
    rng = np.random.default_rng(h * 7 + w)
    tex = np.zeros((h, w, 4))
    tex[..., :3] = rgb
    tone = rng.normal(0, 0.035, (h // course + 1, w // unit + 2, 1)).repeat(course, 0).repeat(unit, 1)[:h, :w]
    tex[..., :3] *= 1 + tone
    tex[course - 1::course, :, :3] *= 0.7
    for r in range(h):
        off = (r // course) % 2 * (unit // 2)
        tex[r, (np.arange(w) + off) % unit == 0, :3] *= 0.85
    tex[..., 3] = 255
    if tri:
        t, s_ = np.mgrid[0:h, 0:w]
        tex[((s_ + 0.5) / w + (t + 0.5) / h) > 1, 3] = 0
    return tex


def _tri(rgb, n=16):
    """A right triangle (the lower-left half) of one colour: a cheek, a gable end."""
    tex = np.zeros((n, n, 4))
    tex[..., :3] = rgb
    t, s_ = np.mgrid[0:n, 0:n]
    tex[..., 3] = np.where((s_ + 0.5) / n <= (t + 0.5) / n, 255, 0)
    return tex


def bay(north, mat, frame, panes, roof=None, lead=False, depth_m=0.48, chamfer=0.22, z1=None, sill=0.75,
        head=2.3, a0=0.04, a1=0.96, margin=3, fascia=(226, 222, 212), string=(200, 196, 186), span=1):
    """A canted bay on one storey: sides from the wall at a0 / a1 out to the
    front, which is `chamfer` in from each; `roof` (rgb) a hipped roof over
    it. `span` 2: across two squares (a0 / a1 then up to 2), two sprites."""
    p = Bay(north)
    z1 = z1 or (2.62 if roof else STOREY_M)
    D, c = depth_m, chamfer
    side = np.hypot(D, c)
    pieces = [((0, a0), (D, a0 + c), side, (c, -D)),             # far side (towards a0)
              ((D, a0 + c), (D, a1 - c), a1 - a0 - 2 * c, (1, 0)),  # front
              ((D, a1 - c), (0, a1), side, (c, D))]                # near side
    for (p0, p1, width, nrm) in pieces:
        tex = _window_face(mat, width, 0, z1, sill, head, frame, panes if nrm == (1, 0) else (1, panes[1], panes[2]),
                           lead, margin, fascia if roof else None, None if roof else string)
        p.face(p0, p1, 0, z1, tex, _light(north, *nrm))
    if roof:
        # A hipped roof against the wall: a front slope over the bay's front,
        # a hip over each cant (review: one masked plane read as a pointed
        # gable), about 30 degrees.
        rh = 0.32 if c else 0.25
        top = z1 + rh
        e = 0.05                                                    # the eaves' overhang
        p.plane((WALL, a0 + c, top), (0, a1 - a0 - 2 * c, 0), (D + e - WALL, 0, -rh), _tiles(roof),
                W.TOP_LIGHT * 0.88)
        if c:
            p.plane((WALL, a0 + c, top), (0, -c - e, -rh), (D + e - WALL, 0, -rh), _tiles(roof, tri="lo"),
                    W.TOP_LIGHT * 0.78)
            p.plane((WALL, a1 - c, top), (0, c + e, -rh), (D + e - WALL, 0, -rh), _tiles(roof, tri="lo"),
                    W.TOP_LIGHT * 0.95)
        else:                                                       # a lean-to's closed cheeks
            for a, lt in ((a0, 0.8), (a1 - 1e-6, 0.95)):
                p.face((WALL, a), (D + e, a), z1, top, _tri(fascia), lt)
    if span == 1:
        return p.render(0)
    return [p.render(k, last=k == span - 1) for k in range(span)]


def box_bay(north, mat, frame, panes, roof, depth_m=0.44, a0=0.08, a1=0.92, z1=2.55, span=1):
    """A shallow square bay (1980s), a tiled lean-to over it, a stained fascia."""
    return bay(north, mat, frame, panes, roof, depth_m=depth_m, chamfer=0.0, z1=z1, a0=a0, a1=a1,
               fascia=STAINED, span=span)


def BAYS():
    m = {k: f() for k, f in MATS.items()}
    sash2 = (2, 2, None)            # two-over-two
    sash1 = (1, 2, None)            # one over one
    case = (2, 1, 0.3)              # casements under a transom
    return [
        ("vic", lambda n: bay(n, m["brick_vic"], WHITE, sash2, roof=SLATE, margin=5, fascia=(214, 208, 192))),
        ("edw_lower", lambda n: bay(n, m["brick_edw"], WHITE, sash1)),
        ("edw_upper", lambda n: bay(n, m["roughcast"], WHITE, sash1, roof=TILE, sill=0.85)),
        ("edw_single", lambda n: bay(n, m["brick_edw"], WHITE, sash1, roof=TILE)),
        ("iw_lower", lambda n: bay(n, m["brick_edw"], WHITE, case, lead=True)),
        ("iw_upper", lambda n: bay(n, m["pebbledash"], WHITE, case, roof=TILE, lead=True, sill=0.85)),
        ("iw_single", lambda n: bay(n, m["brick_edw"], WHITE, case, roof=TILE, lead=True)),
        ("e80", lambda n: box_bay(n, m["brick_80s"], STAINED, (2, 1, 0.28), roof=(104, 72, 60))),
        # Two squares wide (four slots each: west first square, west second,
        # north first, north second): a front of three lights between the cants.
        ("vic_wide", lambda n: bay(n, m["brick_vic"], WHITE, (3, 2, None), roof=SLATE, margin=5,
                                   fascia=(214, 208, 192), **WIDE)),
        ("edw_wide_lower", lambda n: bay(n, m["brick_edw"], WHITE, (3, 2, None), **WIDE)),
        ("edw_wide_upper", lambda n: bay(n, m["roughcast"], WHITE, (3, 2, None), roof=TILE, sill=0.85, **WIDE)),
        ("edw_wide_single", lambda n: bay(n, m["brick_edw"], WHITE, (3, 2, None), roof=TILE, **WIDE)),
        ("iw_wide_lower", lambda n: bay(n, m["brick_edw"], WHITE, (3, 1, 0.3), lead=True, **WIDE)),
        ("iw_wide_upper", lambda n: bay(n, m["pebbledash"], WHITE, (3, 1, 0.3), roof=TILE, lead=True, sill=0.85,
                                        **WIDE)),
        ("iw_wide_single", lambda n: bay(n, m["brick_edw"], WHITE, (3, 1, 0.3), roof=TILE, lead=True, **WIDE)),
        ("e80_wide", lambda n: box_bay(n, m["brick_80s"], STAINED, (3, 1, 0.28), roof=(104, 72, 60), a0=0.12,
                                       a1=1.88, span=2)),
    ]


WIDE = dict(span=2, a0=0.10, a1=1.90, depth_m=0.58, chamfer=0.40)   # a 1.8 m canted bay, 45-degree cants


# --- porches -------------------------------------------------------------------------------------

D0, D1, DH = DK.D0, DK.D1, DK.DH


def _lean_to(p, o0, o1, a0, a1, z_wall, z_eave, rgb, cheek=None):
    """A lean-to canopy roof from the wall down to its eaves, closed at its
    ends by triangular cheeks (rgb; review: open ends showed the wall)."""
    p.plane((o0, a0, z_wall), (0, a1 - a0, 0), (o1 - o0, 0, z_eave - z_wall), _tiles(rgb, 12, 36),
            W.TOP_LIGHT * 0.92)
    if cheek is not None:
        for a, lt in ((a0, 0.8), (a1, 0.95)):
            p.face((o0, a), (o1, a), z_eave, z_wall, _tri(cheek), lt)


def _bracket(p, a, depth, z_top, rgb, light=0.95):
    """A triangular bracket on the wall: as deep as `depth` at its top, to a point below."""
    tex = _tri(rgb)[::-1].copy()
    tex[:, :2, :3] *= 0.8
    p.face((WALL, a), (WALL + depth, a), z_top - depth, z_top, tex, light)


def canopy_bracket(north):
    p = Bay(north)
    for a in (D0 - 0.05, D1 + 0.01):                     # painted timber brackets, not white
        _bracket(p, a, 0.42, DH + 0.30, (196, 190, 176))
    p.face((WALL + 0.5, D0 - 0.14), (WALL + 0.5, D1 + 0.14), DH + 0.26, DH + 0.36, DK._flat(WHITE, 4, 16),
           _light(north, 1, 0))
    _lean_to(p, WALL, WALL + 0.52, D0 - 0.14, D1 + 0.14, DH + 0.66, DH + 0.36, TILE, cheek=WHITE)
    return p.render(0)


def _brick_box(p, o0, o1, a0, a1, z0, z1, mat):
    """A pier in brick: its front and side faces textured, its top flat."""
    h = max(2, round((z1 - z0) * PX))
    for q0, q1, nrm, w_m in (((o1, a0), (o1, a1), (1, 0), a1 - a0), ((o0, a1), (o1, a1), (0, 1), o1 - o0)):
        w = max(2, round(w_m * PX))
        tex = np.zeros((h, w, 4))
        tex[..., :3] = mat[np.arange(h) % mat.shape[0]][:, np.arange(w) % mat.shape[1]]
        tex[..., 3] = 255
        p.face(q0, q1, z0, z1, tex, _light(p.north, *nrm))
    p.plane((o0, a0, z1), (0, a1 - a0, 0), (o1 - o0, 0, 0), DK._flat((150, 80, 62)), W.TOP_LIGHT)


def porch_30s(north):
    p = Bay(north)
    brick = MATS["brick_edw"]()
    quarry = np.zeros((16, 16, 4))
    quarry[..., :3] = (150, 66, 50)
    quarry[::4, :, :3] *= 0.8
    quarry[:, ::4, :3] *= 0.8
    quarry[..., 3] = 255
    p.plane((WALL, D0 - 0.24, 0.06), (0, D1 - D0 + 0.48, 0), (0.86 - WALL, 0, 0), quarry, W.TOP_LIGHT)   # quarry tiles
    _brick_box(p, 0.58, 0.86, D0 - 0.26, D0 - 0.02, 0.06, DH + 0.05, brick)                               # stocky piers
    _brick_box(p, 0.58, 0.86, D1 + 0.02, D1 + 0.26, 0.06, DH + 0.05, brick)
    q = DK.Piece(north)
    q.box(0.74, 0.88, D0 - 0.26, D1 + 0.26, DH + 0.05, DH + 0.16, WHITE)                                  # the beam
    q.gable(WALL, 0.92, D0 - 0.30, D1 + 0.30, DH + 0.16, DH + 0.72, TILE, (226, 222, 210))
    out, dep = q.draw()
    base, base_d = p.render(0)
    m = out[..., 3] > 0                                   # the roof over the piers
    base[m], base_d[m] = out[m], dep[m]
    return base, base_d


def canopy_flat(north):
    q = DK.Piece(north)
    q.box(WALL, 0.72, D0 - 0.16, D1 + 0.16, DH + 0.10, DH + 0.20, (236, 234, 228))   # fascia and slab
    q.box(WALL + 0.02, 0.70, D0 - 0.14, D1 + 0.14, DH + 0.20, DH + 0.22, FELT)
    return q.draw()


def canopy_tiled(north):
    p = Bay(north)
    for a in (D0 - 0.05, D1):                            # stained brackets, lighter than the soffit (review)
        _bracket(p, a, 0.46, DH + 0.28, (150, 112, 80))
    p.face((WALL + 0.56, D0 - 0.16), (WALL + 0.56, D1 + 0.16), DH + 0.20, DH + 0.30, DK._flat(STAINED, 4, 16),
           _light(north, 1, 0))
    _lean_to(p, WALL, WALL + 0.58, D0 - 0.16, D1 + 0.16, DH + 0.60, DH + 0.30, (104, 72, 60), cheek=STAINED)
    return p.render(0)


PORCHES = [("canopy_bracket", canopy_bracket), ("porch_30s", porch_30s), ("canopy_flat", canopy_flat),
           ("canopy_tiled", canopy_tiled)]
BAY_SHEET, PORCH_SHEET = "bays_uk_01", "doorways_uk_02"
BAYS_ONE = ["vic", "edw_lower", "edw_upper", "edw_single", "iw_lower", "iw_upper", "iw_single", "e80"]


def index(sheet_items, name) -> int:
    return 2 * [n for n, _ in sheet_items].index(name)


def write(sheet, items) -> None:
    tiles, depths, names = [], [], []
    for name, make in items:
        for north in (False, True):
            got = make(north)
            for t, d in (got if isinstance(got, list) else [got]):
                tiles.append(t)
                depths.append(d)
                names.append(name)
    F.write_sheet(WK.sheet_file(sheet + ".png"), tiles)
    F.write_sheet(WK.sheet_file(f"DEPTH_{sheet}.png"), [W.depth_tile(t, d) for t, d in zip(tiles, depths)])
    # Seen only, no vanilla piece - but a bay's roofed piece is a roof to the
    # game, so it fades when the player is up on its floor (B42 fades a roof
    # on the player's level only by its WestRoof type: QA, 2026-10-19).
    roofed = sheet == BAY_SHEET
    slots = [{"props": [["WestRoofB", ""], ["RoofGroup", "3"], ["BlockRain", ""]]}
             if roofed and "lower" not in n else None for n in names]
    with open(WK.sheet_file(sheet + ".slots.json"), "w", encoding="utf-8") as f:
        json.dump(slots, f)
    # A preview: each piece on its wall, west then north, the vanilla door or window wall behind it.
    door = sheet == PORCH_SHEET
    wall = WK.vanilla_tile("walls_exterior_house_01", 14 if door else 4), \
        WK.vanilla_tile("walls_exterior_house_01", 15 if door else 5)
    # A wide piece's two squares side by side as they stand (the second a
    # square on along the wall: 64 px left and 32 down for a west wall, right
    # and down for a north one).
    wide_from = len(tiles) if door else 2 * len(BAYS_ONE)
    img = Image.new("RGBA", (128 * len(tiles), 256 + 64), (80, 100, 80, 255))
    k = 0
    while k < len(tiles):
        if k >= wide_from:
            for j, (dx, dy) in enumerate(((0, 0), (-64, 32), (0, 0), (64, 32))):
                x = k * 128 + (96 if j < 2 else 288)                # one origin a pair
                img.alpha_composite(Image.fromarray(np.ascontiguousarray(wall[j // 2])), (x + dx, dy))
                img.alpha_composite(Image.fromarray(tiles[k + j]), (x + dx, dy))
            k += 4
            continue
        img.alpha_composite(Image.fromarray(np.ascontiguousarray(wall[k % 2])), (k * 128, 0))
        img.alpha_composite(Image.fromarray(tiles[k]), (k * 128, 0))
        k += 1
    os.makedirs(HK.PREVIEW, exist_ok=True)
    img.save(os.path.join(HK.PREVIEW, f"{sheet}.png"))
    print("wrote", sheet)


def main() -> int:
    write(BAY_SHEET, BAYS())
    write(PORCH_SHEET, PORCHES)
    have = json.load(open(HK.ORDER, encoding="utf-8"))
    for s in (BAY_SHEET, PORCH_SHEET):
        if s not in have:
            have.append(s)
    json.dump(have, open(HK.ORDER, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
