"""The faces of the 1990s UK signs, drawn flat, for street_objects.py to put on its sign plates.

    python sign_faces.py          (the Knoxmap venv's Python: PIL)

Writes asset_creator/renders/sign_faces/<name>.png: RGBA, transparent round
the sign's shape (the plate is cut out by it, and its grey back takes the
same shape). Drawn at 4x a sign's size in the sprite, so Blender's
sampling antialiases them. The designs are TSRGD 1994's, simplified to what
reads at 30-40 px across: a red ring and black numerals, a red disc and white
bar, and so on. Lettering is DejaVu Sans (Bold), a free face (Bitstream Vera
licence: asset_creator/fonts/LICENSE_DEJAVU), standing in for Transport.

Front doors: DOORS below, each a face of the leaf (0.80 x 2.35 m, as
vanilla's door: a W door's sprite spans 49 of the edge's 63 px, 151 px
high), hinge side left, latch side right: panels, glazing, the letter plate
and knob drawn on (low-poly: the leaf is a slab; street_objects.py adds
the letter plate and knob proud of it). Both faces are the same picture,
mirrored, so the knob is on the latch side from either side.

Place signs: TOWNS below (name and the line under it); a new town is a line
there, then a re-render of its slots.

The town centre (townscape.py, 2026-10-14): the bus stop flag (TSRGD 1994
diagram 974: a bus in a red ring over BUS STOP), the grit bin's and litter
bin's lettering, and each pub's hanging board (pubs.py: its name, gold on
its colour, round an emblem painted simply - a glyph of DejaVu's, or a
few shapes).
"""
from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "renders", "sign_faces")
FONTS = os.path.join(os.path.dirname(HERE), "fonts")
FONT = os.path.join(FONTS, "DejaVuSans-Bold.ttf")
FONT_REG = os.path.join(FONTS, "DejaVuSans.ttf")

RED = (200, 16, 46, 255)          # BS 381C 537 signal red
BLUE = (0, 72, 150, 255)          # BS 381C 109
WHITE = (246, 246, 242, 255)
BLACK = (22, 22, 22, 255)
S = 512                           # a round sign's picture, px

TOWNS = {"odiham": ("Odiham", "Please drive carefully"),
         "alton": ("Alton", "Please drive carefully")}


def font(size, bold=True):
    return ImageFont.truetype(FONT if bold else FONT_REG, size)


def centred(d, xy, text, f, fill):
    d.text(xy, text, font=f, fill=fill, anchor="mm")


def disc(fill, ring=None, ring_w=0.12):
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if ring:
        d.ellipse((0, 0, S - 1, S - 1), fill=ring)
        k = S * ring_w
        d.ellipse((k, k, S - 1 - k, S - 1 - k), fill=fill)
    else:
        d.ellipse((0, 0, S - 1, S - 1), fill=fill)
    return im, d


def speed(n):
    im, d = disc(WHITE, RED)
    centred(d, (S / 2, S / 2 + 6), str(n), font(int(S * 0.38)), BLACK)
    return im


def national():
    im, d = disc(WHITE)
    d.ellipse((2, 2, S - 3, S - 3), outline=(60, 60, 60, 255), width=6)
    w = S * 0.13                                      # the black band, top right to bottom left
    c = S / 2
    u = (math.cos(math.radians(45)), -math.sin(math.radians(45)))   # along the band
    v = (-u[1], u[0])
    pts = [(c + u[0] * a + v[0] * b, c + u[1] * a + v[1] * b) for a, b in ((-S, -w), (S, -w), (S, w), (-S, w))]
    band = Image.new("L", (S, S), 0)
    ImageDraw.Draw(band).polygon(pts, fill=255)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse((10, 10, S - 11, S - 11), fill=255)
    from PIL import ImageChops
    im.paste(Image.new("RGBA", (S, S), BLACK), (0, 0), ImageChops.multiply(band, mask))
    return im


def no_entry():
    im, d = disc(RED)
    d.rectangle((S * 0.16, S * 0.40, S * 0.84, S * 0.60), fill=WHITE)
    return im


def keep_left():
    im, d = disc(BLUE)
    c = S / 2
    # A white arrow pointing down and to the left, at 45 degrees.
    pts = [(-0.30, 0.0), (0.0, -0.30), (0.0, -0.11), (0.30, -0.11), (0.30, 0.11), (0.0, 0.11), (0.0, 0.30)]
    a = math.radians(-45)
    rot = [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)) for x, y in pts]
    d.polygon([(c + x * S, c + y * S) for x, y in rot], fill=WHITE)
    return im


def triangle(up=True):
    """A red-bordered white triangle (warning, or with up=False give way); returns (im, d, inner box)."""
    w, h = S, int(S * 0.88)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if up:
        outer = [(w / 2, 0), (w - 1, h - 1), (0, h - 1)]
    else:
        outer = [(0, 0), (w - 1, 0), (w / 2, h - 1)]
    d.polygon(outer, fill=RED)
    cx, cy = sum(p[0] for p in outer) / 3, sum(p[1] for p in outer) / 3
    k = 0.66
    d.polygon([(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in outer], fill=WHITE)
    return im, d, (cx, cy)


def give_way():
    im, d, (cx, cy) = triangle(up=False)
    f = font(int(S * 0.085))
    centred(d, (cx, cy - S * 0.07), "GIVE", f, BLACK)
    centred(d, (cx, cy + S * 0.03), "WAY", f, BLACK)
    return im


def road_works():
    """Warning triangle, the man with the spade at his heap (simplified to read at sprite size)."""
    im, d, (cx, cy) = triangle(up=True)
    u = S / 1000
    ox, oy = cx - 30 * u, cy + 20 * u
    P = lambda x, y: (ox + x * u, oy + y * u)
    d.pieslice((P(20, 40)[0], P(20, 40)[1], P(200, 200)[0], P(200, 200)[1]), 180, 360, fill=BLACK)   # the heap
    d.ellipse((*P(-115, -175), *P(-55, -115)), fill=BLACK)                                            # head
    d.line([P(-80, -110), P(-40, 10)], fill=BLACK, width=int(42 * u))                                # body, leaning
    d.line([P(-40, 10), P(-100, 120)], fill=BLACK, width=int(34 * u))                                # back leg
    d.line([P(-40, 10), P(10, 120)], fill=BLACK, width=int(34 * u))                                  # front leg
    d.line([P(-70, -70), P(60, 40)], fill=BLACK, width=int(22 * u))                                  # arms and spade shaft
    d.polygon([P(45, 25), P(95, 60), P(80, 95), P(30, 60)], fill=BLACK)                               # spade blade
    return im


def stop():
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    oct_ = lambda r: [(S / 2 + r * math.cos(math.radians(22.5 + 45 * k)), S / 2 + r * math.sin(math.radians(22.5 + 45 * k)))
                      for k in range(8)]
    d.polygon(oct_(S / 2 / math.cos(math.radians(22.5)) - 1), fill=WHITE)
    d.polygon(oct_(S / 2 / math.cos(math.radians(22.5)) * 0.92), fill=RED)
    centred(d, (S / 2, S / 2 + 4), "STOP", font(int(S * 0.27)), WHITE)
    return im


def no_through_road():
    w, h = int(S * 0.8), S
    im = Image.new("RGBA", (w, h), BLUE)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w - 1, h - 1), outline=WHITE, width=10)
    d.rectangle((w * 0.40, h * 0.30, w * 0.60, h * 0.86), fill=WHITE)
    d.rectangle((w * 0.18, h * 0.16, w * 0.82, h * 0.30), fill=RED)
    return im


def place(name, line):
    """A town's entry sign: white board, black border, the name, the line under it."""
    w, h = 1280, 600
    im = Image.new("RGBA", (w, h), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w - 1, h - 1), outline=BLACK, width=22)
    d.rectangle((34, 34, w - 35, h - 35), outline=BLACK, width=6)
    size = 260
    while font(size).getlength(name) > w * 0.84:
        size -= 10
    centred(d, (w / 2, h * 0.40), name, font(size), BLACK)
    centred(d, (w / 2, h * 0.78), line, font(72, bold=False), BLACK)
    return im


DOOR_W, DOOR_H = 320, 940                  # 0.80 x 2.35 m at 400 px a metre
BRASS = (196, 160, 70, 255)
CHROME = (190, 194, 198, 255)
GLASS = (52, 64, 72, 255)

# style: (paint, layout, furniture). 1990s English front doors: Victorian
# four-panel doors in heritage colours, an Edwardian half-glazed door,
# the white uPVC door of every 1990s refit, a stained hardwood door with
# a fanlight, a ledged oak cottage door.
DOORS = {
    "black_victorian": ((26, 26, 28), "four", BRASS),
    "red_victorian": ((128, 22, 30), "four", BRASS),
    "blue_edwardian": ((28, 44, 92), "glazed_top", BRASS),
    "green_victorian": ((26, 66, 46), "four", BRASS),
    "cream_victorian": ((222, 212, 176), "four", BRASS),
    "white_upvc": ((236, 236, 232), "upvc", CHROME),
    "hardwood_fanlight": ((112, 52, 30), "fanlight", BRASS),
    "oak_cottage": ((150, 108, 62), "ledged", (30, 30, 30, 255)),
}


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c[:3]) + (255,)


def moulded(d, box, paint):
    """A recessed panel: a dark line top and left, a light one bottom and right (lit from the south-east)."""
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=shade(paint, 0.92))
    d.line([(x0, y1), (x0, y0), (x1, y0)], fill=shade(paint, 0.62), width=5)
    d.line([(x0, y1), (x1, y1), (x1, y0)], fill=shade(paint, 1.25), width=4)


def door(paint, layout, metal):
    w, h = DOOR_W, DOOR_H
    im = Image.new("RGBA", (w, h), shade(paint, 1.0))
    d = ImageDraw.Draw(im)
    m = 0.14 * w                                           # stiles
    if layout == "four":
        moulded(d, (m, 0.07 * h, w - m, 0.42 * h), paint)  # two tall panels above the lock rail, two below
        d.rectangle((w / 2 - 0.05 * w, 0.07 * h, w / 2 + 0.05 * w, 0.42 * h), fill=shade(paint, 1.0))
        moulded(d, (m, 0.56 * h, w - m, 0.92 * h), paint)
        d.rectangle((w / 2 - 0.05 * w, 0.56 * h, w / 2 + 0.05 * w, 0.92 * h), fill=shade(paint, 1.0))
    elif layout == "glazed_top":
        d.rectangle((m, 0.07 * h, w - m, 0.40 * h), fill=GLASS)
        d.rectangle((w / 2 - 0.03 * w, 0.07 * h, w / 2 + 0.03 * w, 0.40 * h), fill=shade(paint, 1.0))
        moulded(d, (m, 0.56 * h, w / 2 - 0.05 * w, 0.92 * h), paint)
        moulded(d, (w / 2 + 0.05 * w, 0.56 * h, w - m, 0.92 * h), paint)
    elif layout == "upvc":
        d.rectangle((0, 0, w - 1, h - 1), outline=shade(paint, 0.82), width=8)
        d.rectangle((0.16 * w, 0.08 * h, 0.84 * w, 0.44 * h), fill=GLASS, outline=shade(paint, 0.80), width=8)
        for k in (1, 2):                                   # "Georgian" bars in the glass
            d.line([(0.16 * w + k * 0.68 * w / 3, 0.08 * h), (0.16 * w + k * 0.68 * w / 3, 0.44 * h)],
                   fill=shade(paint, 1.0), width=6)
        d.line([(0.16 * w, 0.26 * h), (0.84 * w, 0.26 * h)], fill=shade(paint, 1.0), width=6)
        d.rectangle((0.16 * w, 0.60 * h, 0.84 * w, 0.92 * h), fill=shade(paint, 0.94), outline=shade(paint, 0.80), width=6)
    elif layout == "fanlight":
        d.pieslice((m, 0.06 * h, w - m, 0.06 * h + (w - 2 * m)), 180, 360, fill=GLASS)
        for a in (210, 240, 270, 300, 330):                # the sunburst glazing bars
            r = (w - 2 * m) / 2
            cx, cy = w / 2, 0.06 * h + r
            d.line([(cx, cy), (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))],
                   fill=shade(paint, 1.0), width=5)
        for k in range(3):
            y0 = 0.40 * h + k * 0.18 * h
            moulded(d, (m, y0, w - m, y0 + 0.14 * h), paint)
    elif layout == "ledged":
        for k in range(1, 6):                              # boards
            x = k * w / 6
            d.line([(x, 0), (x, h)], fill=shade(paint, 0.70), width=4)
    # Furniture: knob and letter plate (street_objects.py puts them proud too), knocker on the panelled ones.
    ly = 0.49 * h
    d.rectangle((w * 0.30, ly - 0.022 * h, w * 0.70, ly + 0.022 * h), fill=metal)
    d.rectangle((w * 0.33, ly - 0.010 * h, w * 0.67, ly + 0.010 * h), fill=shade(metal, 0.55))
    d.ellipse((w * 0.84 - 12, 0.52 * h - 12, w * 0.84 + 12, 0.52 * h + 12), fill=metal)
    if layout in ("four", "glazed_top", "fanlight"):
        d.rectangle((w / 2 - 16, 0.34 * h if layout != "fanlight" else 0.36 * h, w / 2 + 16,
                     (0.34 if layout != "fanlight" else 0.36) * h + 40), fill=metal)
    if layout == "ledged":
        d.ellipse((w * 0.80 - 18, 0.50 * h - 18, w * 0.80 + 18, 0.50 * h + 18), outline=metal, width=6)
    return im


# The police box (an easter egg: the Metropolitan Police's 1929 Mackenzie
# Trench box, as the real ones were - no other name on it): its signs and
# its sides, each side two leaves of three panels under a six-pane window.
POLICE_BLUE = (24, 44, 86, 255)


def police_header():
    w, h = 800, 130
    im = Image.new("RGBA", (w, h), BLACK)
    d = ImageDraw.Draw(im)
    big, small = font(72), font(28)
    centred(d, (w * 0.21, h / 2 + 2), "POLICE", big, WHITE)
    centred(d, (w * 0.53, h * 0.33), "PUBLIC", small, WHITE)
    centred(d, (w * 0.53, h * 0.70), "CALL", small, WHITE)
    centred(d, (w * 0.80, h / 2 + 2), "BOX", big, WHITE)
    return im


def police_side(notice: bool):
    w, h = 400, 840                                      # 0.86 x 1.80 m at about 465 px a metre
    im = Image.new("RGBA", (w, h), POLICE_BLUE)
    d = ImageDraw.Draw(im)
    dark, light = shade(POLICE_BLUE, 0.70), shade(POLICE_BLUE, 1.30)
    d.line([(w / 2, 0), (w / 2, h)], fill=dark, width=8)  # the two leaves
    for k in range(2):
        x0, x1 = k * w / 2 + 22, (k + 1) * w / 2 - 22
        win = (x0, 26, x1, 26 + (x1 - x0) * 1.15)
        d.rectangle(win, fill=(214, 220, 222, 255), outline=dark, width=6)
        for c in (1, 2):
            yy = win[1] + c * (win[3] - win[1]) / 3
            d.line([(x0, yy), (x1, yy)], fill=POLICE_BLUE, width=7)
        d.line([((x0 + x1) / 2, win[1]), ((x0 + x1) / 2, win[3])], fill=POLICE_BLUE, width=7)
        y = win[3] + 30
        for _ in range(3):
            moulded(d, (x0, y, x1, y + 150), POLICE_BLUE)
            y += 180
    if notice:                                           # the white notice on the left leaf
        x0, x1 = 34, w / 2 - 34
        y0 = 26 + (w / 2 - 44) * 1.15 + 44
        d.rectangle((x0, y0, x1, y0 + 120), fill=WHITE)
        f = font(17)
        for k, line in enumerate(("POLICE", "TELEPHONE", "FREE FOR USE", "OF PUBLIC")):
            centred(d, ((x0 + x1) / 2, y0 + 18 + k * 28), line, f, BLACK)
    d.ellipse((w / 2 + 14, h * 0.52 - 8, w / 2 + 30, h * 0.52 + 8), fill=(30, 30, 30, 255))   # the lock
    return im


GOLD = (214, 178, 84, 255)
CREAM = (238, 228, 196, 255)
BOARD = [(40, 62, 46, 255), (96, 28, 32, 255), (30, 40, 70, 255), (34, 34, 36, 255)]   # green, maroon, blue, black


def bus_flag():
    """Diagram 974: white, a bus in a red ring, BUS STOP under it."""
    w, h = 400, 480
    im = Image.new("RGBA", (w, h), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w - 1, h - 1), outline=BLACK, width=8)
    d.ellipse((60, 30, 340, 310), fill=RED)
    d.ellipse((96, 66, 304, 274), fill=WHITE)
    d.rounded_rectangle((122, 118, 278, 222), 14, fill=BLACK)            # the bus, side on
    for x in (134, 172, 210, 248):
        d.rectangle((x, 130, x + 26, 160), fill=WHITE)
    for x in (152, 248):
        d.ellipse((x - 16, 206, x + 16, 238), fill=BLACK)
    centred(d, (w / 2, 395), "BUS STOP", font(60), BLACK)
    return im


def lettering(text, ink, paper, w=360, h=120):
    im = Image.new("RGBA", (w, h), paper)
    centred(ImageDraw.Draw(im), (w / 2, h / 2 + 4), text, font(int(h * 0.62)), ink)
    return im


def _emblem(d, kind, cx, cy, r):
    """The pub's emblem in a circle r round (cx, cy)."""
    glyph = {"anchor": ("\u2693", GOLD), "lion": ("\u264c", (200, 40, 36, 255)), "crown": ("\u265b", GOLD),
             "white_horse": ("\u265e", WHITE), "hart": ("\u265e", WHITE), "horn": ("\u266b", GOLD),
             "ivy": ("\u2618", (90, 170, 80, 255))}
    if kind in glyph:
        ch, ink = glyph[kind]
        centred(d, (cx, cy), ch, font(int(r * 1.7), bold=False), ink)
        if kind == "hart":                                   # a hart, not a horse: its antlers
            for sx in (-1, 1):
                x0, y0 = cx + sx * 0.12 * r, cy - 0.55 * r
                d.line([(x0, y0), (x0 + sx * 0.25 * r, y0 - 0.35 * r), (x0 + sx * 0.45 * r, y0 - 0.45 * r)],
                       fill=WHITE, width=int(0.07 * r))
                d.line([(x0 + sx * 0.18 * r, y0 - 0.25 * r), (x0 + sx * 0.05 * r, y0 - 0.48 * r)],
                       fill=WHITE, width=int(0.06 * r))
    elif kind in ("bell", "bells"):
        for k, ox in enumerate((0,) if kind == "bell" else (-0.45, 0.45)):
            x = cx + ox * r
            d.polygon([(x - 0.22 * r, cy - 0.45 * r), (x + 0.22 * r, cy - 0.45 * r), (x + 0.45 * r, cy + 0.45 * r),
                       (x - 0.45 * r, cy + 0.45 * r)], fill=GOLD)
            d.ellipse((x - 0.22 * r, cy - 0.62 * r, x + 0.22 * r, cy - 0.28 * r), fill=GOLD)
            d.ellipse((x - 0.1 * r, cy + 0.42 * r, x + 0.1 * r, cy + 0.62 * r), fill=GOLD)
    elif kind == "swan":
        d.ellipse((cx - 0.6 * r, cy, cx + 0.5 * r, cy + 0.5 * r), fill=WHITE)
        d.line([(cx + 0.35 * r, cy + 0.1 * r), (cx + 0.45 * r, cy - 0.6 * r), (cx + 0.2 * r, cy - 0.7 * r)],
               fill=WHITE, width=int(0.14 * r), joint="curve")
        d.polygon([(cx + 0.2 * r, cy - 0.75 * r), (cx + 0.02 * r, cy - 0.66 * r), (cx + 0.2 * r, cy - 0.6 * r)],
                  fill=(230, 120, 30, 255))
    elif kind == "st_george":
        d.rectangle((cx - 0.6 * r, cy - 0.7 * r, cx + 0.6 * r, cy + 0.7 * r), fill=WHITE)
        d.rectangle((cx - 0.12 * r, cy - 0.7 * r, cx + 0.12 * r, cy + 0.7 * r), fill=RED)
        d.rectangle((cx - 0.6 * r, cy - 0.12 * r, cx + 0.6 * r, cy + 0.12 * r), fill=RED)
    elif kind == "sheaf":
        for a in range(-3, 4):
            d.line([(cx, cy + 0.7 * r), (cx + a * 0.14 * r, cy - 0.6 * r)], fill=GOLD, width=int(0.08 * r))
        d.ellipse((cx - 0.3 * r, cy + 0.05 * r, cx + 0.3 * r, cy + 0.25 * r), fill=(150, 110, 40, 255))
    elif kind == "loaf":
        d.ellipse((cx - 0.7 * r, cy - 0.35 * r, cx + 0.7 * r, cy + 0.45 * r), fill=(196, 140, 70, 255))
        for k in (-1, 0, 1):
            d.line([(cx + k * 0.3 * r - 0.1 * r, cy - 0.2 * r), (cx + k * 0.3 * r + 0.1 * r, cy + 0.2 * r)],
                   fill=(130, 84, 40, 255), width=int(0.07 * r))
    elif kind == "friar":
        d.polygon([(cx, cy - 0.55 * r), (cx + 0.5 * r, cy + 0.75 * r), (cx - 0.5 * r, cy + 0.75 * r)], fill=(120, 120, 116, 255))
        d.ellipse((cx - 0.25 * r, cy - 0.8 * r, cx + 0.25 * r, cy - 0.3 * r), fill=(96, 96, 92, 255))
        d.ellipse((cx - 0.13 * r, cy - 0.62 * r, cx + 0.13 * r, cy - 0.38 * r), fill=(220, 186, 150, 255))
    elif kind == "engine":
        d.rectangle((cx - 0.7 * r, cy - 0.15 * r, cx + 0.3 * r, cy + 0.35 * r), fill=(40, 90, 50, 255))
        d.rectangle((cx + 0.3 * r, cy - 0.5 * r, cx + 0.7 * r, cy + 0.35 * r), fill=(40, 90, 50, 255))
        d.rectangle((cx - 0.55 * r, cy - 0.5 * r, cx - 0.35 * r, cy - 0.15 * r), fill=BLACK)
        for x in (-0.45, 0.0, 0.45):
            d.ellipse((cx + (x - 0.17) * r, cy + 0.25 * r, cx + (x + 0.17) * r, cy + 0.6 * r), fill=GOLD)
    elif kind == "boat":
        d.polygon([(cx - 0.75 * r, cy + 0.1 * r), (cx + 0.75 * r, cy + 0.1 * r), (cx + 0.55 * r, cy + 0.45 * r),
                   (cx - 0.6 * r, cy + 0.45 * r)], fill=(130, 40, 40, 255))
        d.rectangle((cx - 0.4 * r, cy - 0.25 * r, cx + 0.3 * r, cy + 0.1 * r), fill=(40, 80, 50, 255))
        d.line([(cx - 0.9 * r, cy + 0.55 * r), (cx + 0.9 * r, cy + 0.55 * r)], fill=(120, 160, 190, 255), width=int(0.1 * r))
    else:                                                     # a monogram
        centred(d, (cx, cy), "M", font(int(r * 1.3)), GOLD)


def pub_board(key):
    """A pub's hanging board, 0.75 x 0.95 m: its colour, a gold frame, its
    name over and under the emblem."""
    import pubs
    name, emblem = pubs.PUBS[key]
    w, h = 450, 570
    im = Image.new("RGBA", (w, h), BOARD[list(pubs.PUBS).index(key) % len(BOARD)])
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w - 1, h - 1), outline=GOLD, width=14)
    d.rectangle((24, 24, w - 25, h - 25), outline=GOLD, width=3)
    words = name.upper().replace("THE ", "THE\n", 1).split("\n")
    top, bottom = (words[0], words[1]) if len(words) == 2 else ("", words[0])
    size = 70
    while font(size).getlength(bottom) > w * 0.84:
        size -= 4
    if top:
        centred(d, (w / 2, 72), top, font(54), CREAM)
    centred(d, (w / 2, h - 74), bottom, font(size), CREAM)
    _emblem(d, emblem, w / 2, h / 2, 140)
    return im


FACES = {
    "speed_30": lambda: speed(30), "speed_40": lambda: speed(40), "national": national,
    "no_entry": no_entry, "give_way": give_way, "stop": stop, "road_works": road_works,
    "no_through_road": no_through_road, "keep_left": keep_left,
    **{f"place_{k}": (lambda n=n, l=l: place(n, l)) for k, (n, l) in TOWNS.items()},
    **{f"door_{k}": (lambda v=v: door(*v)) for k, v in DOORS.items()},
    "police_header": police_header, "police_side": lambda: police_side(False),
    "police_front": lambda: police_side(True),
    "bus_flag": bus_flag, "grit_label": lambda: lettering("GRIT", BLACK, (240, 200, 30, 255)),
    "litter_label": lambda: lettering("LITTER", GOLD, (24, 24, 26, 255), 420, 110),
}


def _pub_faces():
    sys.path.insert(0, HERE)
    import pubs
    return {f"pub_{k}": (lambda k=k: pub_board(k)) for k in pubs.PUBS}


FACES.update(_pub_faces())


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, make in FACES.items():
        im = make()
        im.save(os.path.join(OUT, name + ".png"))
        print(name, im.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
