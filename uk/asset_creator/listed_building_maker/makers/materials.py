"""Materials and patterns for recipe_kit.py: what a recipe's parts are made of.

A material colours a face: it is given each pixel's place on the face in
metres (u along it, v down it, world-anchored, so a texture runs on across
faces that meet) and returns its colour. A pattern paints a panel (a flat
picture on a face: an arch's spandrel, a painted board, a window's tracery)
and returns colour and alpha, given the pixel's place on the panel (s, t
from 0 to 1) and the panel's size in metres.

Everything is drawn here, procedurally: no photo is copied (the photos of a
landmark are for looking at; README "Pictures and text"). `kit:<material>`
samples one of wall_kit.py's own wall textures, so a piece can match the
walls it stands against.

To add one: write a function and put it in MATERIALS or PATTERNS. A recipe
names it with "mat" (a name, or {"name": ..., "rgb": [r, g, b], ...}) or with
"pattern" ({"name": ..., ...}). `python recipe_kit.py --materials` draws them
all on a sheet to look at.
"""
from __future__ import annotations

import math

import numpy as np

# ---- noise ----


def _hash(ix, iy, seed):
    h = (np.asarray(ix, np.int64).astype(np.uint64) * np.uint64(374761393)
         + np.asarray(iy, np.int64).astype(np.uint64) * np.uint64(668265263)
         + np.uint64(seed * 2654435761 % (1 << 32)))
    h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
    h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFF)).astype(float) / 65535.0


def noise(u, v, cell, seed=0, wrap=None):
    """Smooth value noise in 0..1, features about `cell` metres across;
    repeating every `wrap` metres along u and v if given (a multiple of cell)."""
    x, y = np.asarray(u) / cell, np.asarray(v) / cell
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    n = int(round(wrap / cell)) if wrap else 0
    w = (lambda i: i % n) if n else (lambda i: i)     # repeats every `wrap` metres, along and down
    a, b = _hash(w(ix), w(iy), seed), _hash(w(ix + 1), w(iy), seed)
    c, d = _hash(w(ix), w(iy + 1), seed), _hash(w(ix + 1), w(iy + 1), seed)
    return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy


def cells(u, v, size, seed=0, jitter=0.8, wrap=None):
    """Jittered-grid cells: (cell id, distance to its centre, to the nearest edge-ish) per point;
    repeating every `wrap` metres along u and v if given (a multiple of size)."""
    n = int(round(wrap / size)) if wrap else 0
    x, y = np.asarray(u) / size, np.asarray(v) / size
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    best = np.full(x.shape, 9.0)
    second = np.full(x.shape, 9.0)
    bid = np.zeros(x.shape, np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            hx = cx % n if n else cx
            hy = cy % n if n else cy
            px = cx + 0.5 + (_hash(hx, hy, seed) - 0.5) * jitter
            py = cy + 0.5 + (_hash(hx, hy, seed + 7) - 0.5) * jitter
            d = np.hypot(x - px, y - py)
            closer = d < best
            second = np.where(closer, best, np.minimum(second, d))
            bid = np.where(closer, hx * 92821 + hy, bid)
            best = np.where(closer, d, best)
    return bid, best * size, (second - best) * size


def _rgb(spec, default):
    return np.array(spec.get("rgb", default), float)


def _tone(base, amount, n):
    return base[None, :] * (1 + amount * (n[:, None] - 0.5) * 2)


# ---- materials: (u, v, normal, spec) -> rgb, per pixel ----
# u runs along the face, v down it (on a level face, u is east and v south).


def wood(u, v, nrm, spec, base=(104, 76, 52)):
    """Timber: grain up a standing face, along a lying one."""
    c = _rgb(spec, base)
    upright = abs(nrm[2]) < 0.5
    a, b = (u, v) if upright else (v, u)
    grain = noise(a * 1.0, b * 0.06, 0.02, 3) * 0.6 + noise(a, b, 0.15, 5) * 0.4
    streak = (np.sin((a + noise(a, b, 0.3, 9) * 0.05) * 2 * math.pi / 0.011) > 0.7) * 0.08
    return c[None, :] * (0.84 + 0.26 * grain[:, None] - streak[:, None])


def carved(u, v, nrm, spec):
    """Carved oak (a Jacobean pulpit, a chest): panels in moulded frames, a
    lozenge or a round arch in each, cut a shade darker, with light on the
    upper edges of the cuts. Motif "fielded": the Georgian raised panel, its
    bevelled border lit above and shaded below. Motif "linenfold": Tudor
    folds (`folds` across a panel); "tracery": a sunk cusped pointed arch and a
    quatrefoil (Pugin's blind tracery). `rgb` tints the oak; with
    `painted` it is that colour of paint."""
    c = plaster(u, v, nrm, spec) if spec.get("painted") else wood(u, v, nrm, spec, base=(86, 58, 38))
    pw, ph = spec.get("panel", [0.28, 0.36])
    a, b = np.mod(u, pw) / pw, np.mod(v, ph) / ph
    frame = (a < 0.12) | (a > 0.88) | (b < 0.10) | (b > 0.90)
    bevel = ((a > 0.10) & (a < 0.15)) | ((b > 0.08) & (b < 0.13))
    shade = ((a > 0.85) & (a < 0.90)) | ((b > 0.87) & (b < 0.92))
    motif = spec.get("motif", "lozenge")
    x, y = a - 0.5, b - 0.5
    if motif == "fielded":
        field = (np.abs(x) < 0.28) & (np.abs(y) < 0.30)
        lit = ~field & (np.abs(x) < 0.36) & (np.abs(y) < 0.38) & ((y < -np.abs(x) + 0.05) | (x < -0.3))
        k = np.ones(u.shape)
        k = np.where(frame, 1.04, k)
        k = np.where(shade, 0.78, k)
        k = np.where(~frame & ~field, 0.90, k)
        k = np.where(~frame & lit, 1.10, k)
        k = np.where(field, 1.02, k)
        return c * k[:, None]
    if motif == "linenfold":
        # Tudor linenfold: upright folds across the field, their ends cut in a wavy line
        fx = np.clip((a - 0.15) / 0.70, 0, 1)
        folds = spec.get("folds", 4)
        ridge = 0.5 + 0.5 * np.cos(fx * 2 * math.pi * folds)
        wave = 0.05 * np.cos(fx * 2 * math.pi * folds)
        field = ~frame & ~bevel & ~shade
        ends = field & ((b < 0.19 + wave) | (b > 0.81 + wave))
        k = np.ones(u.shape)
        k = np.where(frame, 1.06, k)
        k = np.where(bevel, 1.16, k)
        k = np.where(shade, 0.70, k)
        k = np.where(field, 0.80 + 0.34 * ridge, k)
        k = np.where(ends, 0.74, k)
        return c * k[:, None]
    if motif == "tracery":
        # blind Gothic tracery: a cusped pointed arch sunk in each panel, a quatrefoil over it
        px, py = x * pw, (0.5 - y) * ph                     # metres from the panel's middle, y up
        hole = arch_opening(px, py + ph * 0.38, pw * 0.60, ph * 0.42, "pointed")
        rim = arch_opening(px, py + ph * 0.38, pw * 0.74, ph * 0.42, "pointed") & ~hole
        mull = hole & (np.abs(px) < pw * 0.035)
        qy = py - ph * 0.28
        r4 = np.minimum.reduce([np.hypot(px - dx, qy - dy) for dx, dy in
                                ((pw * 0.09, 0), (-pw * 0.09, 0), (0, pw * 0.09), (0, -pw * 0.09))])
        quat = (r4 < pw * 0.075) & (py < ph * 0.48)
        k = np.ones(u.shape)
        k = np.where(frame, 1.06, k)
        k = np.where(~frame & rim, 1.16, k)
        k = np.where(~frame & hole, 0.70, k)
        k = np.where(~frame & mull, 1.06, k)
        k = np.where(~frame & quat & ~hole, 0.72, k)
        k = np.where(shade, 0.72, k)
        return c * k[:, None]
    if motif == "arch":
        inner = (np.abs(x) < 0.26) & ((y > -0.05) | (x * x + (y + 0.05) ** 2 < 0.26 ** 2)) & (y < 0.32)
        ring = inner & ~((np.abs(x) < 0.20) & ((y > -0.05) | (x * x + (y + 0.05) ** 2 < 0.20 ** 2)) & (y < 0.28))
    else:
        d = np.abs(x) / 0.30 + np.abs(y) / 0.36
        ring = (d > 0.78) & (d < 1.0)
        inner = d < 0.45
    k = np.ones(u.shape)
    k = np.where(frame, 1.08, k)
    k = np.where(bevel, 1.20, k)
    k = np.where(shade, 0.70, k)
    k = np.where(~frame & ring, 0.66, k)
    k = np.where(~frame & inner, 1.12, k)
    return c * k[:, None]


def ashlar(u, v, nrm, spec):
    """Dressed stone (clunch, Bath stone) in courses, joints a little darker;
    `wrap` (a wall kit's face) repeats it every metre (block must divide it)."""
    c = _rgb(spec, (210, 200, 176))
    course, block = spec.get("course", 0.30), spec.get("block", 0.45)
    wrap = spec.get("wrap")
    row = np.floor(v / course)
    off = (row % 2) * block / 2
    col = np.floor((u + off) / block)
    if wrap:
        col = np.mod(col, int(round(wrap / block)))
    tone = _hash(col.astype(np.int64), row.astype(np.int64), 11)
    jy = np.mod(v, course) < 0.012
    jx = np.mod(u + off, block) < 0.012
    k = 0.99 + spec.get("tone", 0.12) * (tone - 0.5) + 0.06 * (noise(u, v, 0.05, 2, wrap) - 0.5)
    k = np.where(jy | jx, 0.80, k)
    if abs(nrm[2]) > 0.5:                                  # a level top: no courses
        k = 0.95 + 0.08 * (noise(u, v, 0.08, 2) - 0.5)
    return c[None, :] * k[:, None]


ANSTON = (196, 163, 126)                                   # Anston magnesian limestone, pale honey-buff (was 200,156,106: read peach-orange in game, review 2026-10-24)


def _cusped(x, z, half, spring, ratio=0.75):
    """Inside a drop-pointed head over |x| <= half, springing at z = spring (z up)."""
    r = 2 * half * ratio
    c = r - half
    return (np.abs(x) <= half) & ((z <= spring) | ((np.abs(x) + c) ** 2 + (z - spring) ** 2 <= r * r))


def _head_rise(half, ratio=0.75):
    r = 2 * half * ratio
    return math.sqrt(max(r * r - (r - half) ** 2, 0.0))


def perp_panel(u, v, nrm, spec):
    """Ashlar with close blind Perpendicular panelling (Barry and Pugin's walls
    at Westminster): upright panels `panel` m wide, each sunk with a pointed
    head, in tiers up each storey (`tiers`: [[z0, z1], ...] m up from the
    storey's floor), the bands between them left plain - every floor's string
    course. `shields`: a small shield in each panel of the lowest tier. Repeats
    every storey (3 m) and, with `wrap`, every metre."""
    base = ashlar(u, v, nrm, dict(spec, rgb=spec.get("rgb", ANSTON), course=spec.get("course", 0.30),
                                  block=spec.get("block", 0.5), tone=spec.get("tone", 0.07)))
    if abs(nrm[2]) > 0.5:
        return base
    pw = spec.get("panel", 0.5)
    rib = spec.get("rib", 0.04)
    z = np.mod(-v, 3.0)                                    # up from the storey's floor
    x = np.mod(u, pw) - pw / 2
    half = pw / 2 - rib
    rise = _head_rise(half)
    k = np.ones(u.shape)
    tiers = spec.get("tiers", [[0.14, 0.84], [0.92, 2.86]])
    for n, (z0, z1) in enumerate(tiers):
        spring = z1 - rise
        inner = _cusped(x, z, half, spring) & (z >= z0) & (z <= z1)
        shadow = inner & ~_cusped(x - 0.018, z + 0.022, half, spring) | inner & (x < -half + 0.018)
        k = np.where(inner, 0.89, k)
        k = np.where(shadow, 0.78, k)
        if n == 0 and spec.get("shields"):
            sy = (z - (z0 + z1) / 2) / ((z1 - z0) * 0.32)
            sx = x / (half * 0.62)
            shield = (np.abs(sx) <= 1) & (sy <= 1) & ((sy >= 0) | (sx * sx + sy * sy * 0.8 <= 1)) & (sy >= -1.2)
            k = np.where(inner & shield, 0.98, k)
    return base * k[:, None]


def stone(u, v, nrm, spec):
    """Plain stone, lightly mottled."""
    c = _rgb(spec, (206, 198, 176))
    k = 0.92 + 0.12 * noise(u, v, 0.09, 4) + 0.05 * (noise(u, v, 0.015, 6) - 0.5)
    return c[None, :] * k[:, None]


def marble(u, v, nrm, spec):
    c = _rgb(spec, (226, 224, 218))
    vein = np.abs(np.sin((u * 0.7 + v + noise(u, v, 0.12, 8) * 0.25) * 40)) < 0.06
    k = 0.96 + 0.06 * noise(u, v, 0.1, 9) - vein * 0.18
    return c[None, :] * k[:, None]


FLINT_TONES = np.array([(52, 56, 62), (36, 37, 41), (84, 86, 92), (112, 110, 108), (68, 74, 84)], float)


def flint(u, v, nrm, spec):
    """Knapped and field flints in lime mortar; `clunch` scatters chalk-stone
    blocks through it and `lacing` lays courses of brick (as at All Saints,
    Odiham: looked at in photos only)."""
    mortar = _rgb(spec, (198, 190, 170))
    fade = spec.get("fade", 0.0)                       # weathered: the flints grey towards the mortar
    wrap = spec.get("wrap")                            # a wall kit's face: repeat every metre
    bid, d, edge = cells(u, v, 1 / 16, 21, jitter=0.9, wrap=wrap)
    size = 0.026 + 0.010 * _hash(bid, bid // 7, 3)
    tone = FLINT_TONES[(_hash(bid, 2, 9) * len(FLINT_TONES)).astype(int).clip(0, len(FLINT_TONES) - 1)]
    tone = tone * (1 - fade) + mortar[None, :] * 0.85 * fade
    rim = (d > size - 0.006) & (d < size)
    out = np.where((d < size)[:, None], tone, mortar[None, :] * (0.93 + 0.1 * noise(u, v, 1 / 50, 4, wrap)[:, None]))
    out = np.where(rim[:, None], np.array([196, 192, 182.0])[None, :], out)
    hi = (d < size * 0.5) & (_hash(bid, 1, 5) > 0.6)
    out = np.where(hi[:, None], out * 1.35, out)
    if spec.get("lacing"):
        band = np.mod(v + 0.4, spec["lacing"]) < 0.08                      # one course, here and there
        row = np.floor(v / 0.08)
        brick = band & (np.mod(v, 0.08) > 0.012) & (np.mod(u + (row % 2) * 0.125, 0.25) > 0.012) \
            & (noise(u, v, 0.25, 31, wrap) > 0.58)
        bcol = np.floor((u + (row % 2) * 0.125) / 0.25).astype(np.int64)
        out = np.where(brick[:, None], np.array([150, 82, 62.0])[None, :]
                       * (0.85 + 0.3 * _hash(bcol % 4 if wrap else bcol, row.astype(np.int64), 2))[:, None], out)
    if spec.get("clunch"):
        bw = 1 / 3
        bx, by = np.floor(u / bw).astype(np.int64), np.floor(v / 0.22).astype(np.int64)
        pick = _hash(bx % 3 if wrap else bx, by, 17) < spec["clunch"]    # (v: a wall kit's storey repeats as a whole)
        inner = (np.mod(u, bw) > 0.03) & (np.mod(u, bw) < bw - 0.03) & (np.mod(v, 0.22) > 0.03) & (np.mod(v, 0.22) < 0.19)
        block = pick & inner
        out = np.where(block[:, None], np.array([214, 204, 176.0])[None, :]
                       * (0.9 + 0.15 * noise(u, v, 1 / 16, 2, wrap))[:, None], out)
    return out


BRICK_REDS = np.array([(150, 72, 54), (138, 62, 48), (164, 86, 62), (124, 58, 46), (156, 96, 70)], float)


def brick(u, v, nrm, spec):
    """English bond red brick (C17): a course of stretchers, a course of headers."""
    mortar = np.array(spec.get("mortar", (186, 176, 158)), float)
    course = 0.075
    row = np.floor(v / course)
    unit = np.where(row % 2 == 0, 0.225, 0.1125)
    off = np.where(row % 2 == 0, 0.0, 0.056)
    col = np.floor((u + off) / unit)
    tone = BRICK_REDS[(_hash(col.astype(np.int64), row.astype(np.int64), 4) * len(BRICK_REDS)).astype(int)
                      .clip(0, len(BRICK_REDS) - 1)]
    if spec.get("rgb"):
        tone = tone * (np.array(spec["rgb"], float) / BRICK_REDS.mean(0))[None, :]
    joint = (np.mod(v, course) < 0.011) | (np.mod(u + off, unit) < 0.011)
    k = (0.92 + 0.14 * noise(u, v, 0.03, 8))[:, None]
    out = np.where(joint[:, None], mortar[None, :], tone * k)
    if abs(nrm[2]) > 0.5:
        out = tone * k
    return out


def plaster(u, v, nrm, spec):
    c = _rgb(spec, (238, 234, 224))
    k = 0.96 + 0.05 * noise(u, v, 0.3, 2) + 0.02 * (noise(u, v, 0.02, 3) - 0.5)
    return c[None, :] * k[:, None]


def flat(u, v, nrm, spec):
    """A plain colour (paint, lead, iron, gilding), with a little life in it."""
    c = _rgb(spec, (128, 128, 128))
    k = 1 + spec.get("mottle", 0.04) * (noise(u, v, spec.get("cell", 0.05), 2) - 0.5) * 2
    return c[None, :] * k[:, None]


def gilt(u, v, nrm, spec):
    c = _rgb(spec, (206, 164, 72))
    k = 0.85 + 0.35 * noise(u, v, 0.012, 3)
    return c[None, :] * k[:, None]


def leather(u, v, nrm, spec):
    """Buttoned leather upholstery (the Commons' green benches, the Lords' red):
    a diamond grid of buttons, each pulling a dimple, with a soft sheen."""
    c = _rgb(spec, (62, 104, 72))
    p = spec.get("button", 0.2)
    a, b = (u + v) / p, (u - v) / p
    da, db = np.abs(np.mod(a, 1) - 0.5), np.abs(np.mod(b, 1) - 0.5)
    pucker = np.minimum(da, db)                            # 0 on the creases between buttons
    button = (da > 0.40) & (db > 0.40)
    k = 0.90 + 0.16 * np.clip(pucker * 2.2, 0, 1) + 0.04 * (noise(u, v, 0.05, 6) - 0.5)
    k = np.where(button, 0.74, k)
    return c[None, :] * k[:, None]


def carpet(u, v, nrm, spec):
    """A fitted carpet (the chambers'): a ground colour and a small repeating
    motif in `motif` colour, 8 to the metre so it runs on square by square."""
    c = _rgb(spec, (70, 104, 70))
    m = np.array(spec.get("motif", (c * 0.78).tolist()), float)
    t = 1 / 8
    a, b = np.mod(u, t) / t - 0.5, np.mod(v, t) / t - 0.5
    i, j = np.floor(u / t).astype(np.int64), np.floor(v / t).astype(np.int64)
    dot = (np.abs(a) + np.abs(b) < 0.22) & ((i + j) % 2 == 0)
    pile = 0.94 + 0.08 * noise(u, v, 1 / 32, 4, 1.0)
    return np.where(dot[:, None], m[None, :] * pile[:, None], c[None, :] * pile[:, None])


def encaustic_lattice(u, v, nrm, spec):
    """Encaustic tiles in a great diagonal lattice (St Stephen's Hall, the Central
    Lobby): dark bands on the diagonals a metre apart, the diamonds between them
    filled with small red and buff tiles; repeats every metre."""
    t = 1 / 8
    i, j = np.floor(u / t).astype(np.int64), np.floor(v / t).astype(np.int64)
    pal = np.array([spec.get("red", (138, 108, 78)), spec.get("buff", (176, 150, 110))], float)
    small = pal[(i + j) % 2]
    p, q = u + v, u - v
    band = (np.abs(np.mod(p, 1.0) - 0.5) > 0.445) | (np.abs(np.mod(q, 1.0) - 0.5) > 0.445)
    dark = np.array(spec.get("band", (60, 46, 38)), float)
    joint = (np.mod(u, t) < 0.008) | (np.mod(v, t) < 0.008)
    out = np.where(band[:, None], dark[None, :], small)
    out = np.where((joint & ~band)[:, None], np.array([120, 110, 96.0])[None, :], out)
    return out * (0.94 + 0.1 * noise(u, v, 0.0625, 3, 1.0))[:, None]


def cloth(u, v, nrm, spec):
    """A hanging (an altar frontal, a cushion): folds down it."""
    c = _rgb(spec, (128, 28, 36))
    fold = 0.86 + 0.16 * np.cos(u * 2 * math.pi / spec.get("fold", 0.09))
    return c[None, :] * (fold * (0.96 + 0.06 * noise(u, v, 0.02, 1)))[:, None]


# Floors: one tile repeated square by square, so each repeats exactly every metre,
# along and down (every period divides 1 m; hashes and noise wrap at 1 m).

def flags(u, v, nrm, spec):
    """Stone flags (a church floor): two courses a metre, slabs 0.5 m, every other course half a slab on."""
    c = _rgb(spec, (178, 172, 158))
    row = np.floor(v / 0.5)
    off = np.mod(row, 2) * 0.25
    col = np.floor((u + off) / 0.5)
    tone = _hash(np.mod(col, 2).astype(np.int64), np.mod(row, 2).astype(np.int64), 15)
    joint = (np.mod(v, 0.5) < 0.012) | (np.mod(u + off, 0.5) < 0.012)
    k = 0.88 + 0.16 * tone + 0.06 * (noise(u, v, 0.0625, 2, 1.0) - 0.5)
    k = np.where(joint, 0.70, k)
    return c[None, :] * k[:, None]


def ledger(u, v, nrm, spec):
    """Flags with a dark ledger stone among them (a chancel floor): one slab in four."""
    out = flags(u, v, nrm, spec)
    a, b = np.mod(u + np.mod(np.floor(v / 0.5), 2) * 0.25, 1.0), np.mod(v, 1.0)
    slab = (a > 0.52) & (a < 0.98) & (b > 0.52) & (b < 0.98)
    dark = np.array([64, 64, 66.0]) * (0.9 + 0.2 * noise(u, v, 0.0625, 5, 1.0))[:, None]
    return np.where(slab[:, None], dark, out)


def parquet(u, v, nrm, spec):
    """Woodblock in basket weave (a nave's floor): squares of 0.25 m, each four
    blocks laid one way, the next square the other way; it repeats every half
    metre (a 45-degree herringbone cannot repeat within a square metre)."""
    c = _rgb(spec, (168, 120, 74))
    sq = 0.25
    i, j = np.floor(u / sq), np.floor(v / sq)
    along = np.mod(i + j, 2) == 0                         # blocks running along u, or along v
    a = np.where(along, np.mod(v, sq), np.mod(u, sq))     # across the blocks
    lane = np.floor(a / (sq / 4))
    tone = _hash((np.mod(i, 4) * 4 + lane).astype(np.int64), np.mod(j, 4).astype(np.int64), 18)
    joint = (np.mod(a, sq / 4) < 0.006) | (np.mod(u, sq) < 0.008) | (np.mod(v, sq) < 0.008)
    k = 0.84 + 0.24 * tone + 0.06 * (noise(u, v, 0.0625, 3, 1.0) - 0.5)
    k = np.where(joint, 0.68, k)
    return c[None, :] * k[:, None]


def encaustic(u, v, nrm, spec):
    """Victorian encaustic tiles (a chancel): red, black and buff, 125 mm (8 a metre)."""
    t = 1 / 8
    i, j = np.floor(u / t).astype(np.int64), np.floor(v / t).astype(np.int64)
    pal = np.array([(150, 62, 44), (46, 40, 38), (196, 170, 120), (150, 62, 44)], float)
    pick = pal[((i + j) % 2) * 2 + ((i // 2 + j // 2) % 2)]
    a, b = np.mod(u, t) / t - 0.5, np.mod(v, t) / t - 0.5
    inlay = (np.abs(a) + np.abs(b) < 0.28) & ((i + j) % 2 == 0)
    pick = np.where(inlay[:, None], np.array([206, 182, 128.0])[None, :], pick)
    joint = (np.mod(u, t) < 0.008) | (np.mod(v, t) < 0.008)
    return np.where(joint[:, None], np.array([120, 112, 100.0])[None, :],
                    pick * (0.94 + 0.1 * noise(u, v, 0.0625, 3, 1.0))[:, None])


def roof_tile(u, v, nrm, spec):
    """Weathered clay plain tiles (as roof_kit's)."""
    c = _rgb(spec, (118, 72, 56))
    row = np.floor(v / 0.1)
    col = np.floor((u + (row % 2) * 0.0825) / 0.165)
    tone = _hash(col.astype(np.int64), row.astype(np.int64), 6)
    k = 0.82 + 0.25 * tone - (np.mod(v, 0.1) > 0.085) * 0.25
    lichen = noise(u, v, 0.08, 9) > 0.78
    out = c[None, :] * k[:, None]
    return np.where(lichen[:, None], np.array([150, 150, 120.0])[None, :], out)


def shingle(u, v, nrm, spec):
    """Oak shingles weathered silver-grey (a broach spire): staggered courses, dark gaps."""
    c = _rgb(spec, (112, 108, 102))
    # `width`: each shingle's width (default 0.14); `wrap`: repeat every `wrap` m along u (a
    # roof piece repeated square by square: width must divide it), so the pieces meet unseen
    sw = spec.get("width", 0.14)
    wrap = spec.get("wrap")
    row = np.floor(v / 0.11)
    col = np.floor((u + (row % 2) * sw / 2) / sw)
    if wrap:
        col = np.mod(col, int(round(wrap / sw)))
    tone = _hash(col.astype(np.int64), row.astype(np.int64), 23)
    gap = (np.mod(v, 0.11) > 0.095) | (np.mod(u + (row % 2) * sw / 2, sw) < 0.012)
    k = 0.82 + 0.3 * tone + 0.06 * (noise(u, v, 0.03, 4, wrap) - 0.5)
    return np.where(gap[:, None], np.array([40, 38, 36.0])[None, :], c[None, :] * k[:, None])


def iron_plate(u, v, nrm, spec):
    """Cast-iron roof plates (Barry's roofs and spires at Westminster): panels
    `plate` m wide and high, a raised rib lit on one side and shaded on the
    other at each joint, painted dark grey."""
    c = _rgb(spec, (70, 74, 80))
    pw, ph = spec.get("plate", [0.6, 0.9])
    a, b = np.mod(u, pw), np.mod(v, ph)
    k = 0.94 + 0.08 * noise(u, v, 0.2, 7)
    k = np.where(a < 0.03, 1.12, k)
    k = np.where((a >= 0.03) & (a < 0.05), 0.78, k)
    k = np.where(b < 0.02, 0.84, k)
    return c[None, :] * k[:, None]


def leaded_glass(u, v, nrm, spec):
    """Clear glass in diamond quarries, a green-grey cast."""
    c = _rgb(spec, (120, 140, 134))
    q = spec.get("quarry", 0.07)
    a, b = (u + v) / q, (u - v) / q
    lead = (np.mod(a, 1) < 0.10) | (np.mod(b, 1) < 0.10)
    sheen = np.clip(1 - np.abs((u * 0.8 + v) % 0.6 - 0.3) * 6, 0, 1) * 0.25
    out = c[None, :] * (1 + sheen[:, None]) * (0.95 + 0.1 * noise(u, v, 0.02, 4))[:, None]
    return np.where(lead[:, None], np.array([54, 56, 58.0])[None, :], out)


STAINED = np.array([(40, 72, 150), (150, 30, 36), (206, 166, 60), (44, 110, 70), (210, 206, 190),
                    (40, 72, 150), (110, 40, 110), (52, 90, 170), (36, 60, 130), (150, 30, 36)], float)


def stained_glass(u, v, nrm, spec):
    """Stained glass: small pieces of colour in lead, mostly blue and ruby."""
    bid, d, edge = cells(u, v, spec.get("piece", 0.07), 33, jitter=0.9)
    col = STAINED[(_hash(bid, 4, 13) * len(STAINED)).astype(int).clip(0, len(STAINED) - 1)]
    lead = edge < 0.006
    return np.where(lead[:, None], np.array([34, 34, 36.0])[None, :], col * (0.9 + 0.2 * noise(u, v, 0.02, 2))[:, None])


def _louvre(u, v):
    """Louvre boards as a material (a bell opening's "glass")."""
    k = np.mod(v, 0.16) / 0.16
    return np.where((k < 0.65)[:, None], np.array((112, 104, 96.0))[None, :] * (0.8 + 0.3 * k)[:, None],
                    np.array((28, 26, 26.0))[None, :])


def kit_texture(name):
    """A wall_kit material's own face texture (192 x 64: a storey by a metre), sampled by u, v."""
    import wall_kit as WK
    make, _trim = WK.MATERIALS[name]
    tex = make(np.random.default_rng(sum(map(ord, name))))[..., :3].astype(float)

    def sample(u, v, nrm, spec):
        i = np.mod(np.floor(v * 64).astype(int), tex.shape[0])
        j = np.mod(np.floor(u * 64).astype(int), tex.shape[1])
        return tex[i, j]
    return sample


MATERIALS = {
    "oak": wood,
    "oak_dark": lambda u, v, n, s: wood(u, v, n, s, (70, 48, 32)),
    "oak_pale": lambda u, v, n, s: wood(u, v, n, s, (176, 138, 92)),
    "pine": lambda u, v, n, s: wood(u, v, n, s, (172, 128, 80)),
    "oak_carved": carved,
    "ashlar": ashlar,
    "stone": stone,
    "marble": marble,
    "flint": flint,
    "brick": brick,
    "plaster": plaster,
    "paint": flat,
    "lead": lambda u, v, n, s: flat(u, v, n, {"rgb": (108, 112, 118), **s}),
    "iron": lambda u, v, n, s: flat(u, v, n, {"rgb": (38, 38, 40), **s}),
    "brass": lambda u, v, n, s: gilt(u, v, n, {"rgb": (184, 152, 80), **s}),
    "gilt": gilt,
    "cloth": cloth,
    "leather": leather,
    "carpet": carpet,
    "encaustic_lattice": encaustic_lattice,
    "flags": flags,
    "ledger": ledger,
    "parquet": parquet,
    "encaustic": encaustic,
    "roof_tile": roof_tile,
    "shingle": shingle,
    "glass_leaded": leaded_glass,
    "glass_stained": stained_glass,
    "louvre": lambda u, v, n, s: _louvre(u, v),
    "anston": lambda u, v, n, s: ashlar(u, v, n, {"rgb": ANSTON, "block": 0.5, "tone": 0.07, **s}),
    "perp_panel": perp_panel,
    "iron_plate": iron_plate,
    "gilt_plain": lambda u, v, n, s: flat(u, v, n, {"rgb": (190, 150, 64), "mottle": 0.03, "cell": 0.1, **s}),
    "slate": lambda u, v, n, s: shingle(u, v, n, {"rgb": (84, 88, 96), **s}),
}
_KIT: dict = {}


def material(spec):
    """A recipe's "mat" (a name, or a dict with "name") -> (function, spec dict)."""
    if isinstance(spec, str):
        spec = {"name": spec}
    name = spec["name"]
    if name.startswith("kit:"):
        if name not in _KIT:
            _KIT[name] = kit_texture(name[4:])
        return _KIT[name], spec
    if name not in MATERIALS:
        raise KeyError(f"no material {name!r}: one of {sorted(MATERIALS)} or kit:<wall_kit material>")
    return MATERIALS[name], spec


# ---- patterns: (s, t, w, h, spec) -> rgba, per pixel; s across, t down, 0..1 ----


def _over(rgba, mask, rgb, a=255):
    rgba[mask, :3] = rgb
    rgba[mask, 3] = a


def arch_opening(x, y, span, spring, shape, rise=None):
    """Inside an arch whose opening is |x| <= span/2, from y = 0 (its foot)
    up: the jambs to `spring`, then the head (pointed, round, segmental or
    four-centred). x, y in metres, y up."""
    half = span / 2
    inside = (np.abs(x) <= half) & (y >= 0)
    above = y - spring
    if shape == "tudor":
        # a four-centred (Tudor) arch: steep haunches, a low pointed crown `rise` over the springing
        t = np.clip(np.abs(x) / half, 0, 1)
        R = rise if rise is not None else half * 0.5
        head = (above <= 0) | (above <= R * (0.65 * np.sqrt(1 - t * t) + 0.35 * (1 - t)))
    elif shape == "round":
        head = (above <= 0) | (x * x + above * above <= half * half)
    elif shape == "segmental":
        r = half * 1.6
        head = (above <= 0) | (x * x + (above + r - half * 0.45) ** 2 <= r * r)
    elif shape == "four_centred":
        r1, r2 = half * 0.42, half * 2.2
        small = ((np.abs(x) - (half - r1)) ** 2 + above ** 2 <= r1 * r1) | (np.abs(x) <= half - r1) & (above <= r1 * 0.6)
        big = (np.abs(x) + (r2 - half)) ** 2 + (above + r2 * 0.35 - r1 * 0.6) ** 2 <= r2 * r2
        head = (above <= 0) | small & big | (np.abs(x) <= half - r1) & big
    else:                                                        # two-centred (equilateral) pointed arch
        r = span * spec_ratio(shape)
        c = r - half
        head = (above <= 0) | ((np.abs(x) + c) ** 2 + above * above <= r * r)
    return inside & head


def spec_ratio(shape):
    return {"lancet": 1.4, "pointed": 1.0, "drop": 0.75}.get(shape, 1.0)


def arch(s, t, w, h, spec, rng):
    """A wall face with an arch through it (an arcade, a chancel arch): the
    `mat` face where solid, a moulded band round the opening in `trim`."""
    x = (s - 0.5) * w
    y = (1 - t) * h - spec.get("foot", 0.0)
    span, spring = spec.get("span", w * 0.8), spec.get("spring", h * 0.4)
    shape = spec.get("shape", "pointed")
    rise = spec.get("rise")
    bw = spec.get("band", 0.12)
    hole = arch_opening(x, y, span, spring, shape, rise)
    band = arch_opening(x, y, span + 2 * bw, spring, shape, None if rise is None else rise + bw) & ~hole
    out = np.zeros(s.shape + (4,))
    fn, ms = material(spec.get("mat", "plaster"))
    u, v = s * w, t * h
    nrm = np.array([0.0, 1.0, 0.0])
    out[:, :3] = fn(u, v, nrm, ms)
    out[:, 3] = 255
    tfn, ts = material(spec.get("trim", "ashlar"))
    trim = tfn(u, v, nrm, ts)
    inner = arch_opening(x, y, span + 0.05, spring, shape, None if rise is None else rise + 0.025) & ~hole
    out[band, :3] = trim[band]
    out[inner, :3] = trim[inner] * 0.78
    out[hole, 3] = 0
    return out


def board(s, t, w, h, spec, rng):
    """A painted board in a moulded frame (benefactions, the Commandments):
    gilt lettering as rows of strokes on black."""
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    out[:, :3] = spec.get("ground", (26, 26, 28))
    fr = spec.get("frame", 0.05)
    edge = (s * w < fr) | (s * w > w - fr) | (t * h < fr) | (t * h > h - fr)
    out[edge, :3] = np.array(spec.get("frame_rgb", (176, 140, 64)))
    rows = spec.get("rows", 8)
    ty = (t * h - fr * 2) / (h - fr * 4) * rows
    line = (np.mod(ty, 1) > 0.35) & (np.mod(ty, 1) < 0.65) & (ty > 0) & (ty < rows)
    xw = s * w
    word = noise(np.floor(xw / 0.035), np.floor(ty), 1.0, 5) > 0.3
    letters = line & word & (xw > fr * 2) & (xw < w - fr * 2) & (np.mod(xw, 0.012) < 0.008)
    out[letters, :3] = np.array(spec.get("ink", (214, 176, 86)))     # gilt; or painted letters on a wall
    return out


def royal_arms(s, t, w, h, spec, rng):
    """A Royal Coat of Arms on a board (Charles II, 1661: a quartered shield,
    the crown over it, lion and unicorn either side), drawn plainly, as the
    game's own paintings are."""
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    out[:, :3] = spec.get("ground", (30, 30, 34))           # the board (gilt or crimson on a canopy)
    x, y = (s - 0.5) * 2, (t - 0.5) * 2                     # -1..1, y down
    frame = (np.abs(x) > 0.94) | (np.abs(y) > 0.94)
    out[frame, :3] = (176, 140, 64)
    # the shield: square top, rounded foot
    sx, sy = x / 0.42, (y - 0.12) / 0.55
    shield = (np.abs(sx) <= 1) & (sy >= -1) & ((sy <= 0.3) | (sx * sx + (sy - 0.3) ** 2 / 0.49 <= 1))
    q = (sx >= 0).astype(int) + 2 * (sy >= -0.3).astype(int)
    colours = np.array([(150, 26, 30), (196, 156, 56), (36, 52, 130), (150, 26, 30)], float)
    out[shield, :3] = colours[q[shield]]
    charge = shield & (noise(sx * 3, sy * 3, 0.5, q.astype(int) + 3) > 0.62)
    gold = np.array((214, 176, 70.0))
    out[charge & (q != 1), :3] = gold
    out[charge & (q == 1), :3] = (150, 26, 30)
    rim = shield & ~((np.abs(sx) <= 0.93) & (sy >= -0.94) & ((sy <= 0.3) | (sx * sx + (sy - 0.3) ** 2 / 0.4 <= 0.86)))
    out[rim, :3] = gold
    # the crown
    crown = (np.abs(x) < 0.22) & (y > -0.86) & (y < -0.6) & ((y > -0.70) | (np.mod((x + 0.22) * 20, 2) < 1))
    out[crown, :3] = gold
    # the supporters: lion (gold) left, unicorn (white) right
    for side, rgb in ((-1, (200, 160, 60)), (1, (226, 224, 214))):
        bx, by = (x - side * 0.66) / 0.2, (y - 0.05) / 0.5
        body = (bx * bx + by * by <= 1) & (noise(bx * 2, by * 2, 0.6, 7 + side) > 0.25)
        out[body, :3] = rgb
    # the motto scroll
    scroll = (np.abs(x) < 0.7) & (y > 0.76) & (y < 0.86)
    out[scroll, :3] = (220, 214, 190)
    return out


def organ(s, t, w, h, spec, rng):
    """Organ pipes in a case: towers of zinc and gilt pipes, their mouths dark."""
    out = np.zeros(s.shape + (4,))
    x, y = s * w, (1 - t) * h
    n = spec.get("pipes", 21)
    i = np.floor(s * n)
    xc = (s * n - i - 0.5)
    mid = n / 2
    top = h * (0.55 + 0.35 * np.abs(np.cos((i - mid) / n * math.pi * spec.get("towers", 3))))
    pipe = (np.abs(xc) < 0.38) & (y < top) & (y > h * 0.06)
    out[pipe, 3] = 255
    shade = 0.75 + 0.4 * (0.5 - np.abs(xc))
    metal = np.array(spec.get("rgb", (186, 182, 170)), float)
    out[pipe, :3] = (metal[None, :] * shade[pipe, None])
    mouth = pipe & (y > h * 0.14) & (y < h * 0.19) & (np.abs(xc) < 0.2)
    out[mouth, :3] = (40, 38, 36)
    case = (y <= h * 0.06) | (s < 0.02) | (s > 0.98)
    fn, ms = material(spec.get("case", "oak_dark"))
    if case.any():
        out[case, :3] = fn(x[case], y[case], np.array([0.0, 1, 0]), ms)
        out[case, 3] = 255
    return out


def brass(s, t, w, h, spec, rng):
    """A brass set in a stone slab: a small figure and its inscription plate."""
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    fn, ms = material(spec.get("mat", {"name": "stone", "rgb": [96, 94, 92]}))
    out[:, :3] = fn(s * w, t * h, np.array([0.0, 0, 1]), ms)
    x, y = (s - 0.5) * 2, (t - 0.5) * 2
    head = (x * x + (y + 0.55) ** 2 < 0.02)
    gown = (y > -0.45) & (y < 0.45) & (np.abs(x) < 0.12 + 0.12 * (y + 0.45))
    plate = (np.abs(x) < 0.5) & (y > 0.6) & (y < 0.8)
    b = np.array((182, 150, 78.0))
    for m in (head, gown, plate):
        out[m, :3] = b * (0.9 + 0.2 * noise(s[m] * 3, t[m] * 3, 0.1, 2))[:, None]
    return out


def clock(s, t, w, h, spec, rng):
    """A church clock: a blue dial, gilt ring, numerals and hands, on its stone surround."""
    out = np.zeros(s.shape + (4,))
    x, y = (s - 0.5) * 2, (t - 0.5) * 2
    r = np.hypot(x, y)
    dial = r <= 0.92
    out[dial, 3] = 255
    out[dial, :3] = spec.get("rgb", (40, 64, 140))
    gold = np.array((214, 176, 70.0))
    ring = (r > 0.80) & (r <= 0.92)
    out[ring, :3] = gold
    ang = np.mod(np.arctan2(x, -y), 2 * math.pi)
    marks = (r > 0.58) & (r < 0.76) & (np.abs(np.mod(ang + math.pi / 12, math.pi / 6) - math.pi / 12) < 0.07)
    out[marks, :3] = gold
    for a, ln, wd in ((math.radians(300), 0.45, 0.07), (math.radians(60), 0.66, 0.05)):   # ten past ten
        hx, hy = math.sin(a), -math.cos(a)
        along = x * hx + y * hy
        across = np.abs(x * hy - y * hx)
        out[(along > 0) & (along < ln) & (across < wd), :3] = gold
    rim = (r > 0.92) & (r <= 1.0)
    out[rim, 3] = 255
    out[rim, :3] = (200, 192, 170)
    return out


def inn_sign(s, t, w, h, spec, rng):
    """An inn's hanging sign: a painted board in a gilt frame - St George on a white
    horse over the dragon (drawn plainly), the name on a scroll."""
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    out[:, :3] = spec.get("ground", (34, 52, 40))
    x, y = (s - 0.5) * 2, (t - 0.5) * 2
    frame = (np.abs(x) > 0.88) | (np.abs(y) > 0.9)
    out[frame, :3] = (196, 160, 72)
    sky = ~frame & (y < 0.55)
    out[sky, :3] = (150, 172, 186)
    ground = ~frame & (y >= 0.55)
    out[ground, :3] = (70, 96, 52)
    horse = ((x + 0.05) ** 2 / 0.16 + (y - 0.05) ** 2 / 0.03 < 1) | ((np.abs(x + 0.05) - 0.25) ** 2 < 0.002) & (y > 0.05) & (y < 0.45)         | ((x - 0.33) ** 2 / 0.01 + (y + 0.1) ** 2 / 0.02 < 1)
    out[horse & ~frame, :3] = (232, 230, 222)
    rider = ((x + 0.02) ** 2 / 0.006 + (y + 0.22) ** 2 / 0.04 < 1)
    out[rider & ~frame, :3] = (176, 178, 186)
    cross = rider & ((np.abs(x + 0.02) < 0.02) | (np.abs(y + 0.2) < 0.02))
    out[cross & ~frame, :3] = (170, 30, 34)
    lance = (np.abs((y + 0.3) - (x - 0.05) * 0.9) < 0.02) & (x > -0.3) & (x < 0.7)
    out[lance & ~frame, :3] = (90, 70, 50)
    dragon = ((x - 0.45) ** 2 / 0.04 + (y - 0.65) ** 2 / 0.012 < 1)
    out[dragon & ~frame, :3] = (40, 110, 60)
    scroll = (np.abs(x) < 0.7) & (y > 0.72) & (y < 0.86)
    out[scroll, :3] = (226, 214, 180)
    letters = scroll & (np.mod((x + 0.7) * 25, 1) < 0.55) & (np.abs(y - 0.79) < 0.035)
    out[letters, :3] = (40, 34, 30)
    return out


def frieze_doric(s, t, w, h, spec, rng):
    """A Doric frieze: triglyphs (three upright bars, grooves between) and
    square metopes, each with a trophy - a round shield over crossed arms -
    in stone colour on a painted ground."""
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    u, v = s * w, t * h
    ground = np.array(spec.get("rgb", (228, 222, 206)), float)
    out[:, :3] = ground
    tw = spec.get("triglyph", 0.14)
    mw = h                                              # the metope is square
    a = np.mod(u, tw + mw)
    tri = a < tw
    groove = tri & ((np.abs(a - tw / 3) < 0.012) | (np.abs(a - 2 * tw / 3) < 0.012))
    out[tri, :3] = ground * 1.06
    out[groove, :3] = ground * 0.62
    mx, my = a - tw - mw / 2, v - h / 2                 # from the metope's middle
    inm = ~tri
    r = np.hypot(mx, my)
    shield = inm & (r < mw * 0.22)
    rim = inm & (np.abs(r - mw * 0.22) < 0.012)
    arms = inm & ((np.abs(mx - my) < 0.01) | (np.abs(mx + my) < 0.01)) & (r < mw * 0.40) & ~shield
    out[shield, :3] = ground * 0.92
    out[rim | arms, :3] = ground * 0.66
    edge = (v < 0.02) | (v > h - 0.02)
    out[edge, :3] = ground * 0.80
    return out


def mural(s, t, w, h, spec, rng):
    """A C16 wall painting on plaster: red-ochre and black strapwork, faded."""
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    u, v = s * w, t * h
    out[:, :3] = (222, 208, 176)
    a, b = np.mod(u, 0.5) / 0.5 - 0.5, np.mod(v, 0.5) / 0.5 - 0.5
    ring = np.abs(np.hypot(a, b) - 0.32) < 0.05
    diag = (np.abs(np.abs(a) - np.abs(b)) < 0.05) & (np.hypot(a, b) > 0.36)
    leaf = (np.hypot(a, b) < 0.12)
    out[ring | diag, :3] = (150, 70, 46)
    out[leaf, :3] = (40, 36, 34)
    border = (v < 0.12) | (v > h - 0.12)
    out[border, :3] = (150, 70, 46)
    fade = 0.75 + 0.35 * noise(u, v, 0.25, 7)
    out[:, :3] = 222 - (222 - out[:, :3]) * np.clip(fade, 0, 1)[:, None]
    return out


def louvre(s, t, w, h, spec, rng):
    """Bell-chamber louvres: grey oak boards sloping out, dark between (`rgb`: the
    boards' colour, lead-grey on a roof lantern; `pitch`: board spacing, m)."""
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    pitch = spec.get("pitch", 0.16)
    k = np.mod(t * h, pitch) / pitch
    out[:, :3] = np.where((k < 0.65)[:, None], np.array(spec.get("rgb", (112, 104, 96)), float)[None, :]
                          * (0.8 + 0.3 * k)[:, None], np.array((28, 26, 26.0))[None, :])
    return out


def ruin(s, t, w, h, spec, rng):
    """A ruined wall's face: its material where the wall still stands - below its
    ragged top (`top`: [[s, z]], with a broken edge), outside its breaches
    (`arches`: [[s0, s1, spring, rise]], round-headed) and round windows
    (`rounds`: [[s, z, r]]) - with rows of putlog holes (`putlogs`: heights)."""
    # s along the wall's centre line (a mitred face runs past its ends: s0 .. s1)
    s = spec.get("s0", 0.0) + s * (spec.get("s1", 1.0) - spec.get("s0", 0.0))
    w = spec.get("length", w)
    x, z = s * w, (1 - t) * h                          # along the wall, up from its foot (metres)
    top = spec["top"]
    ts, tz = np.array([q[0] for q in top]), np.array([q[1] for q in top])
    # only a small broken edge on the profile: the broken top's faces across the
    # wall's thickness follow the profile itself
    height = np.interp(s, ts, tz) + (noise(x, 0 * x, 0.08, spec.get("seed", 0) + 3) - 0.5) * 0.25
    stands = z <= height
    for s0, s1, spring, rise in spec.get("arches", []):
        cx, half = (s0 + s1) / 2 * w, (s1 - s0) / 2 * w
        head = ((x - cx) / half) ** 2 + ((z - spring) / max(rise, 0.05)) ** 2 <= 1   # a head `rise` high
        stands &= ~((np.abs(x - cx) <= half) & (z >= 0) & ((z <= spring) | head))
    for sr, zr, rr in spec.get("rounds", []):
        stands &= np.hypot(x - sr * w, z - zr) > rr
    fn, ms = material(spec.get("mat", "flint"))
    out = np.zeros(s.shape + (4,))
    out[:, :3] = fn(x, (1 - t) * 0 + (h - z), np.array([0.0, 1, 0]), ms)
    out[:, 3] = np.where(stands, 255, 0)
    for zp in spec.get("putlogs", []):                 # small square holes in rows, dark
        hole = (np.abs(z - zp) < 0.09) & (np.mod(x + 0.3, 0.75) < 0.17) \
            & (_hash(np.floor((x + 0.3) / 0.75).astype(np.int64), int(zp * 10), spec.get("seed", 0)) < 0.65)
        out[hole & stands, :3] = (34, 32, 30)
    edge = stands & (z > height - 0.12)                 # the broken top, a little lighter
    out[edge, :3] = out[edge, :3] * 1.12
    return out


def glass_pane(s, t, w, h, spec, rng):
    """A pane of glass (a display case): see-through, drawn as the game draws
    glass - a pale rim and two diagonal highlight streaks, clear between."""
    out = np.zeros(s.shape + (4,))
    rgb = np.array(spec.get("rgb", (200, 220, 225)), float)
    x, y = s * w, t * h
    rim = (x < 0.012) | (x > w - 0.012) | (y < 0.012) | (y > h - 0.012)
    d = x + y * 0.8
    n = spec.get("streaks", 2)                             # 0 on a pane seen through another
    streak = (np.abs(d - 0.30 * (w + h)) < 0.022) & (n >= 1) | (np.abs(d - 0.42 * (w + h)) < 0.010) & (n >= 2)
    on = rim | streak
    out[on, :3] = np.where(streak[on, None], np.array([240, 246, 246.0])[None, :], rgb[None, :])
    out[on, 3] = 255
    return out


def union_flag(s, t, w, h, spec, rng):
    """The Union flag (2:1, the hoist at s = 0): a blue field, the white saltire with the red
    counterchanged (the broad white uppermost at the hoist), the white-edged red cross. `s0`, `s1`:
    the part of the flag's length this panel shows (a flag in folds is several panels)."""
    s0, s1 = spec.get("s0", 0.0), spec.get("s1", 1.0)
    x = (s0 + (s1 - s0) * s) * 60.0                     # the flag's own units, 60 x 30
    y = t * 30.0
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    blue, white, red = np.array((1, 33, 105.0)), np.array((236, 234, 228.0)), np.array((200, 16, 46.0))
    out[:, :3] = blue
    k = 1 / math.sqrt(5)                                # distance across a diagonal, per unit
    d1 = (y - x * 0.5) * 2 * k                          # top-left to bottom-right, + below it
    d2 = (y - (30 - x * 0.5)) * 2 * k                   # bottom-left to top-right, + below it
    hoist = x < 30
    out[(np.abs(d1) < 3) | (np.abs(d2) < 3), :3] = white
    r1 = np.where(hoist, (d1 > 0) & (d1 < 2), (d1 < 0) & (d1 > -2))
    r2 = np.where(hoist, (d2 > 0) & (d2 < 2), (d2 < 0) & (d2 > -2))
    out[r1 | r2, :3] = red
    out[(np.abs(x - 30) < 5) | (np.abs(y - 15) < 5), :3] = white
    out[(np.abs(x - 30) < 3) | (np.abs(y - 15) < 3), :3] = red
    shade = 1 - spec.get("fold", 0.0) * np.abs(s - 0.5)  # a fold: darker towards one edge
    out[:, :3] *= shade[:, None] if np.ndim(shade) else shade
    return out


def paint_pattern(s, t, w, h, spec, rng):
    """A panel all of one material (a hanging, a slab)."""
    fn, ms = material(spec.get("mat", "plaster"))
    out = np.zeros(s.shape + (4,))
    out[:, :3] = fn(s * w, t * h, np.array([0.0, 1, 0]), ms)
    out[:, 3] = 255
    return out


def tracery(s, t, w, h, spec, rng):
    """A row of Perpendicular panels across a face (a pierced parapet, a
    blind-traceried frieze): each `panel` m wide, a pointed head (`form`
    "arch") or a quatrefoil (`form` "quatrefoil") in a `rib`-wide frame of
    `mat` (`ratio`: how pointed the heads are); `open` cuts them through (alpha 0), else they are sunk darker
    (`shade`: how much of the face's light is left in them, 0.7)."""
    fn, ms = material(spec.get("mat", {"name": "ashlar", "rgb": list(ANSTON)}))
    out = np.zeros(s.shape + (4,))
    u, y = s * w, (1 - t) * h                              # along, and up from the foot
    out[:, :3] = fn(u, h - y, np.array([0.0, 1, 0]), ms)
    out[:, 3] = 255
    pw = spec.get("panel", 1 / 3)
    rib = spec.get("rib", 0.05)
    x = np.mod(u, pw) - pw / 2
    half = pw / 2 - rib
    foot, top = spec.get("foot", rib), h - spec.get("top", rib)
    if spec.get("form", "arch") == "quatrefoil":
        cz, r = (foot + top) / 2, min(half, (top - foot) / 2)
        a = r / 1.75                                       # four lobes 0.75 a round, a out: cusps between them
        hole = np.minimum.reduce([np.hypot(x - dx, y - cz - dz) for dx, dz in
                                  ((a, 0), (-a, 0), (0, a), (0, -a))]) < 0.75 * a
        hole |= np.hypot(x, y - cz) < 0.55 * a
    else:
        ratio = spec.get("ratio", 0.75)                    # the heads' pointedness: 0.75 drop, 1.2 lancet
        hole = _cusped(x, y, half, top - _head_rise(half, ratio), ratio) & (y >= foot) & (y <= top)
        if spec.get("mullion"):                            # a mullion down each panel, two lights
            hole &= np.abs(x) > rib * 0.35
    if spec.get("open"):
        out[hole, 3] = 0
    else:
        out[hole, :3] *= spec.get("shade", 0.70)
    return out


def _seg_dist(px, py, a, b):
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    L = dx * dx + dy * dy
    tt = np.clip(((px - x0) * dx + (py - y0) * dy) / L, 0, 1) if L > 0 else 0
    return np.hypot(px - (x0 + tt * dx), py - (y0 + tt * dy))


ROMAN = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]


def clock_gothic(s, t, w, h, spec, rng):
    """A great Gothic clock face (the Elizabeth Tower's, drawn plainly): a square
    black frame with a gilt-and-black chequered border and gilt fillets, gilt
    quatrefoils in the spandrels, a pale opal dial in segments, a dark iron
    chapter ring with minute marks and crosses, black Roman numerals, a gilt
    rosette of tracery at the centre, and black hands."""
    out = np.zeros(s.shape + (4,))
    out[:, 3] = 255
    x, y = (s - 0.5) * 2, (t - 0.5) * 2                    # -1..1 across, y down
    ax, ay = np.abs(x), np.abs(y)
    sq = np.maximum(ax, ay)
    r = np.hypot(x, y)
    black = np.array(spec.get("frame_rgb", (34, 32, 30)), float)
    gold = np.array(spec.get("gilt", (206, 166, 72)), float)
    pale = np.array(spec.get("dial", (232, 230, 220)), float)
    iron = np.array((52, 52, 54.0))
    out[:, :3] = black
    chk = spec.get("check", 0.04)
    border = sq > 0.93
    cells_ = ((np.floor(x / chk) + np.floor(y / chk)) % 2 == 0)
    out[border & cells_, :3] = gold * 0.92
    out[(sq > 0.875) & (sq < 0.90), :3] = gold
    # the spandrels: a gilt quatrefoil in each corner, on black
    for cx, cy in ((0.70, 0.70), (-0.70, 0.70), (0.70, -0.70), (-0.70, -0.70)):
        d = np.minimum.reduce([np.hypot(x - cx - dx, y - cy - dy) for dx, dy in
                               ((0.055, 0), (-0.055, 0), (0, 0.055), (0, -0.055))])
        out[(np.abs(d - 0.06) < 0.012) & (r > 0.86), :3] = gold
        out[(np.hypot(x - cx, y - cy) < 0.035) & (r > 0.86), :3] = gold
    out[(np.abs(r - 0.86) < 0.012) & (sq < 0.875), :3] = gold          # a gilt ring round the dial
    dial = r < 0.845
    ang = np.mod(np.arctan2(x, -y), 2 * math.pi)                       # clockwise from the top
    out[dial, :3] = pale
    seg = dial & (np.abs(np.mod(ang, math.pi / 30) - math.pi / 60) > math.pi / 60 - 0.006 / np.maximum(r, 0.05))
    out[seg & (r > 0.44), :3] = pale * 0.90                            # the opal glass's glazing bars
    # the chapter ring: two iron rings, minute ticks between, a cross at each hour
    ring = dial & ((np.abs(r - 0.79) < 0.012) | (np.abs(r - 0.69) < 0.012))
    tick = dial & (r > 0.69) & (r < 0.79) & (np.abs(np.mod(ang + math.pi / 60, math.pi / 30) - math.pi / 60) < 0.010 / 0.74)
    out[ring | tick, :3] = iron
    for k in range(12):
        a = k * math.pi / 6
        er = np.array([math.sin(a), -math.cos(a)])
        et = np.array([math.cos(a), math.sin(a)])
        c = er * 0.74
        pr = (x - c[0]) * er[0] + (y - c[1]) * er[1]
        pt = (x - c[0]) * et[0] + (y - c[1]) * et[1]
        cross = ((np.abs(pr) < 0.075) & (np.abs(pt) < 0.016)) | ((np.abs(pt) < 0.05) & (np.abs(pr) < 0.016))
        out[cross, :3] = iron
        # the numeral, upright to the centre
        c = er * 0.555
        pr = (x - c[0]) * er[0] + (y - c[1]) * er[1]       # outward (the letter's up)
        pt = (x - c[0]) * et[0] + (y - c[1]) * et[1]       # along the ring (its right)
        hh, st = 0.075, 0.018
        widths = {"I": 0.034, "V": 0.075, "X": 0.075}
        txt = ROMAN[k]
        total = sum(widths[ch] for ch in txt) + 0.012 * (len(txt) - 1)
        gx = -total / 2
        ink = np.zeros(x.shape, bool)
        for ch in txt:
            wd = widths[ch]
            mx = gx + wd / 2
            if ch == "I":
                ink |= (np.abs(pt - mx) < st / 2 + 0.002) & (np.abs(pr) < hh)
            elif ch == "V":
                ink |= (_seg_dist(pt, pr, (mx - wd / 2, hh), (mx, -hh)) < st / 2) \
                    | (_seg_dist(pt, pr, (mx + wd / 2, hh), (mx, -hh)) < st / 2 * 0.7)
            else:
                ink |= (_seg_dist(pt, pr, (mx - wd / 2, hh), (mx + wd / 2, -hh)) < st / 2) \
                    | (_seg_dist(pt, pr, (mx + wd / 2, hh), (mx - wd / 2, -hh)) < st / 2 * 0.7)
            gx += wd + 0.012
        ink |= (np.abs(np.abs(pr) - hh) < 0.008) & (np.abs(pt) < total / 2 + 0.01)    # the letters' serifs
        out[ink & dial, :3] = (22, 22, 22)
    out[dial & (np.abs(r - 0.44) < 0.012), :3] = iron
    # the centre: a rosette of gilt tracery on the pale glass
    rose = np.zeros(x.shape, bool)
    for k in range(12):
        a = k * math.pi / 6
        cx, cy = 0.21 * math.sin(a), -0.21 * math.cos(a)
        rose |= np.abs(np.hypot(x - cx, y - cy) - 0.21) < 0.009
    out[rose & (r < 0.43), :3] = gold
    out[(np.abs(r - 0.42) < 0.008), :3] = gold
    # the hands (black): hour to `hour`, minute to `minute`
    hr, mn = spec.get("time", [10, 10])
    for a, ln, wd in ((math.radians(30 * (hr % 12) + mn / 2), 0.52, 0.035), (math.radians(6 * mn), 0.80, 0.022)):
        hx, hy = math.sin(a), -math.cos(a)
        along = x * hx + y * hy
        across = np.abs(x * hy - y * hx)
        hand = (along > -0.08) & (along < ln) & (across < wd * (1 - 0.5 * np.clip(along / ln, 0, 1)))
        tip = (np.abs(along - ln * 0.86) + across * 1.3 < wd * 2.2)
        out[hand | tip & (along > 0), :3] = (18, 18, 18)
    out[r < 0.05, :3] = (18, 18, 18)
    return out


PATTERNS = {
    "tracery": tracery,
    "clock_gothic": clock_gothic,
    "arch": arch,
    "board": board,
    "royal_arms": royal_arms,
    "organ": organ,
    "brass_slab": brass,
    "louvre": louvre,
    "clock": clock,
    "inn_sign": inn_sign,
    "mural": mural,
    "frieze_doric": frieze_doric,
    "ruin": ruin,
    "fill": paint_pattern,
    "glass_pane": glass_pane,
    "union_flag": union_flag,
}


def pattern(spec):
    if spec["name"] not in PATTERNS:
        raise KeyError(f"no pattern {spec['name']!r}: one of {sorted(PATTERNS)}")
    return PATTERNS[spec["name"]]
