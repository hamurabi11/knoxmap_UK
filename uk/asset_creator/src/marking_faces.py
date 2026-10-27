"""The UK road markings of road_markings_uk_02, drawn flat from above, for
streetscape.py to lay on the road.

    python marking_faces.py          (the Knoxmap venv's Python: PIL)

Writes asset_creator/renders/marking_faces/<name>.png: RGBA, transparent
off the paint, PX pixels a tile (drawn at 2x and halved, so the edges are
soft), each picture the size of its marking in tiles - W across the lane, L
along it. A marking read by a driver is drawn as a driver heading north
reads it (its top the way they are going); streetscape.py turns it to each
heading. The rest (a yellow box, double white lines, the mini roundabout's
disc) are drawn as they lie on the map, north up.

Also the road gully's square grate (street_drains_uk_01), on the map as it lies.

The designs are TSRGD 1994's (diagrams 1023 give-way triangle, 1024 SLOW,
1038 lane arrows, 1014 deflection arrow, 1001.3 mini roundabout, 1043 box
junction, 1001 zebra and 1002.1 zig-zags, 1065 the speed roundel), at
vanilla's scale - a tile a metre or so, a lane three or four - so each is
shortened to one or two tiles (user, 2026-10-14: a 1 and 2 tile set): the
real ones are elongated several times over for a driver's shallow view,
these half as much again as they are wide. Lettering is DejaVu Sans Bold
(asset_creator/fonts/LICENSE_DEJAVU), standing in for Transport.
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "renders", "marking_faces")
FONT = os.path.join(os.path.dirname(HERE), "fonts", "DejaVuSans-Bold.ttf")
PX = 256                          # pixels a tile, as written
SS = 2                            # drawn at this many times that, then halved
PAINT = (236, 236, 230, 255)      # white road paint, a little worn
YELLOW = (238, 196, 40, 255)      # as road_markings_uk_01's
LINE = 0.09                       # tiles: a marking's line


def canvas(w, l):
    im = Image.new("RGBA", (round(w * PX * SS), round(l * PX * SS)), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def at(x, y):
    """Tiles -> the canvas's pixels."""
    return (x * PX * SS, y * PX * SS)


def px(t):
    return max(1, round(t * PX * SS))


def done(im):
    return im.resize((im.width // SS, im.height // SS), Image.LANCZOS)


def stretched_text(text, w, l, box_w, box_h, cx, cy):
    """`text` in white, squeezed into box_w x box_h tiles (stretched along the
    lane, as a road marking is), its middle at (cx, cy)."""
    f = ImageFont.truetype(FONT, 200)
    tmp = Image.new("L", (1, 1))
    x0, y0, x1, y1 = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=f)
    glyph = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(glyph).text((-x0, -y0), text, font=f, fill=255)
    glyph = glyph.resize((px(box_w), px(box_h)), Image.LANCZOS)
    im, _ = canvas(w, l)
    paint = Image.new("RGBA", glyph.size, PAINT)
    im.paste(paint, (round(at(cx, cy)[0] - glyph.width / 2), round(at(cx, cy)[1] - glyph.height / 2)), glyph)
    return im


# --- what a driver reads (heading north: the top of the picture ahead) --------------------------
def give_way_triangle():
    """Diagram 1023: an outlined triangle, its base nearest the give-way line (ahead)."""
    im, d = canvas(1, 2)
    d.polygon([at(0.10, 0.12), at(0.90, 0.12), at(0.5, 1.90)], outline=PAINT, width=px(LINE))
    return done(im)


def limit(n):
    """Diagram 1065: the limit in a ring, as the sign beside it."""
    im = stretched_text(str(n), 1, 2, 0.56, 1.05, 0.5, 1.0)
    d = ImageDraw.Draw(im)
    d.ellipse((*at(0.07, 0.10), *at(0.93, 1.90)), outline=PAINT, width=px(LINE))
    return done(im)


def slow():
    """Diagram 1024: SLOW across the lane, stretched along it."""
    return done(stretched_text("SLOW", 2, 2, 1.80, 1.60, 1.0, 1.0))


def _head(d, tip, back, half):
    """An arrowhead: its tip, the middle of its back, its half-width (tiles)."""
    tx, ty = tip
    bx, by = back
    nx, ny = by - ty, tx - bx
    n = (nx * nx + ny * ny) ** 0.5
    nx, ny = nx / n * half, ny / n * half
    d.polygon([at(tx, ty), at(bx + nx, by + ny), at(bx - nx, by - ny)], fill=PAINT)


def _shaft(d, pts, w=0.13):
    d.line([at(*p) for p in pts], fill=PAINT, width=px(w), joint="curve")


def arrow(kind):
    """Diagram 1038: ahead, left, right, ahead-or-left, ahead-or-right;
    1014: the deflection arrow (bent back towards the nearside - left)."""
    im, d = canvas(1, 2)
    if kind == "ahead":
        _shaft(d, [(0.5, 1.92), (0.5, 0.75)])
        _head(d, (0.5, 0.06), (0.5, 0.80), 0.30)
    elif kind in ("left", "right"):
        s = -1 if kind == "left" else 1
        _shaft(d, [(0.5, 1.92), (0.5, 0.95), (0.5 + s * 0.12, 0.70)])
        _head(d, (0.5 + s * 0.42, 0.18), (0.5 + s * 0.04, 0.82), 0.27)
    elif kind in ("ahead_left", "ahead_right"):
        s = -1 if kind == "ahead_left" else 1      # the turn's side; the ahead shaft off the other way
        _shaft(d, [(0.5 - s * 0.10, 1.92), (0.5 - s * 0.10, 0.80)], 0.11)
        _head(d, (0.5 - s * 0.10, 0.06), (0.5 - s * 0.10, 0.70), 0.22)
        _shaft(d, [(0.5 - s * 0.10, 1.35), (0.5 + s * 0.10, 1.05)], 0.10)
        _head(d, (0.5 + s * 0.44, 0.62), (0.5 + s * 0.10, 1.12), 0.20)
    elif kind == "deflect":
        _shaft(d, [(0.62, 1.92), (0.52, 0.80)], 0.11)
        _head(d, (0.38, 0.06), (0.53, 0.86), 0.26)
    return done(im)


def zebra():
    """Diagram 1001: across the crossing, a white band half a tile wide along the
    road and the road between (the road runs N-S here; the next square on carries
    the pattern on). White only (user, 2026-10-14: the painted black bands
    looked messy against the tarmac)."""
    im, d = canvas(1, 1)
    d.rectangle((*at(0.0, 0.0), *at(0.5, 1.0)), fill=PAINT)
    return done(im)


def zigzag():
    """Diagram 1002.1: a zebra's zig-zag, beside the kerb on the left (the W
    edge, heading north) - one zig a tile."""
    im, d = canvas(1, 1)
    d.line([at(0.12, -0.02), at(0.40, 0.5), at(0.12, 1.02)], fill=PAINT, width=px(0.08), joint="curve")
    return done(im)


# --- as they lie on the map, north up ----------------------------------------------------------
def box(edges):
    """Diagram 1043: a yellow box - each square crossed corner to corner (the
    hatching runs on square to square), its edges where the box ends."""
    im, d = canvas(1, 1)
    w = px(0.09)
    d.line([at(0, 0), at(1, 1)], fill=YELLOW, width=w)
    d.line([at(1, 0), at(0, 1)], fill=YELLOW, width=w)
    b = 0.11
    for e in edges:
        x0, y0, x1, y1 = {"W": (0.03, 0, 0.03 + b, 1), "E": (0.97 - b, 0, 0.97, 1),
                          "N": (0, 0.03, 1, 0.03 + b), "S": (0, 0.97 - b, 1, 0.97)}[e]
        d.rectangle((*at(x0, y0), *at(x1, y1)), fill=YELLOW)
    return done(im)


def double_white(edge):
    """Diagram 1013.1: two solid white lines, through a bend; along the W (a
    road running N-S) or N edge, where vanilla's centre line lies."""
    im, d = canvas(1, 1)
    for a, b in ((0.03, 0.12), (0.22, 0.31)):
        box_ = (a, 0, b, 1) if edge == "W" else (0, a, 1, b)
        d.rectangle((*at(box_[0], box_[1]), *at(box_[2], box_[3])), fill=PAINT)
    return done(im)


def mini_disc():
    """Diagram 1001.3: a mini roundabout's white disc, across 2 x 2 squares."""
    im, d = canvas(2, 2)
    d.ellipse((*at(0.18, 0.18), *at(1.82, 1.82)), fill=PAINT)
    return done(im)


IRON = (70, 70, 72, 255)          # a gully grate's ductile iron, worn
SLOT = (16, 16, 18, 255)          # the dark below its bars


def gully(edge):
    """A road gully's grate, square (the British 450 mm one; user, 2026-10-14:
    vanilla's drain is a long American kerb inlet), in the gutter by the kerb
    on the square's `edge`: an iron frame, bars across it to the kerb, the
    dark between."""
    im, d = canvas(1, 1)
    side, inset = 0.42, 0.06
    a0, a1 = 0.5 - side / 2, 0.5 + side / 2                  # along the kerb
    b0, b1 = inset, inset + side                              # out from it
    box_ = {"W": (b0, a0, b1, a1), "E": (1 - b1, a0, 1 - b0, a1),
            "N": (a0, b0, a1, b1), "S": (a0, 1 - b1, a1, 1 - b0)}[edge]
    x0, y0, x1, y1 = box_
    d.rectangle((*at(x0, y0), *at(x1, y1)), fill=IRON)
    f = 0.045                                                  # the frame
    d.rectangle((*at(x0 + f, y0 + f), *at(x1 - f, y1 - f)), fill=SLOT)
    n, bar = 6, 0.022
    across_x = edge in ("W", "E")                              # bars run out from the kerb
    for k in range(1, n):
        t = k / n
        if across_x:
            y = y0 + f + (y1 - y0 - 2 * f) * t
            d.rectangle((*at(x0 + f, y - bar), *at(x1 - f, y + bar)), fill=IRON)
        else:
            x = x0 + f + (x1 - x0 - 2 * f) * t
            d.rectangle((*at(x - bar, y0 + f), *at(x + bar, y1 - f)), fill=IRON)
    return done(im)


ARROWS = ("ahead", "left", "right", "ahead_left", "ahead_right", "deflect")
BOX = {"fill": "", "W": "W", "N": "N", "E": "E", "S": "S", "NW": "NW", "NE": "NE", "SE": "SE", "SW": "SW"}


def pictures():
    out = {"give_way_triangle": give_way_triangle(), "limit_30": limit(30), "limit_40": limit(40), "slow": slow(),
           "zebra": zebra(), "zigzag": zigzag(), "mini_disc": mini_disc(),
           "double_white_W": double_white("W"), "double_white_N": double_white("N")}
    out.update({f"arrow_{k}": arrow(k) for k in ARROWS})
    out.update({f"box_{k}": box(v) for k, v in BOX.items()})
    out.update({f"gully_{e}": gully(e) for e in "WNES"})
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    pics = pictures()
    for name, im in pics.items():
        im.save(os.path.join(OUT, name + ".png"))
    # One look at them all, on tarmac.
    sheet = Image.new("RGBA", (sum(im.width + 20 for im in pics.values()), 2 * PX + 20), (58, 58, 60, 255))
    x = 10
    for im in pics.values():
        sheet.alpha_composite(im, (x, 10))
        x += im.width + 20
    sheet.save(os.path.join(os.path.dirname(OUT), "marking_faces_preview.png"))
    print(len(pics), "markings ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
