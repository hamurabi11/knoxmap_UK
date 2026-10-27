"""UK street furniture, 1990s, modelled in Blender and rendered at the game's camera.

    python sign_faces.py        (first: the sign faces and door faces, PIL)
    "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b -P street_objects.py -- [sheet ...] [slot ...]

Every sheet in SHEETS (or those named; and only the slots named, if any).
Writes asset_creator/renders/<sheet>/: for each slot of the
sheet, <slot>.png (the sprite, 128 x 256 RGBA, 2x) and <slot>.npy (each
pixel's map position, x y z in metres, for its depth picture), and
manifest.json (what each slot is, its facing, its collision shapes). Then
street_sheet.py (ordinary Python) makes the sheet, its depth picture and a
preview, and Tile_creator/src/pack.py packs it.

The camera (calibrated 2026-10-14 on a 1 m cube against walls.py: 0 pixels
of silhouette apart, positions within a quarter pixel): orthographic, 30
degrees down, looking north-west, 64 px a metre along the grid; the game
draws heights at 64 px a metre too, where a true camera at 30 degrees gives
78.4, so everything is built squashed in height by SQUASH. A tile's square
is map x 0..1, y 0..1 (x east, y south), its north corner at pixel (63, 192)
as walls.py draws.

Low-poly, as vanilla's props are drawn (the user, 2026-10-14): a handful of
primitives each, 12-16 sides to a round thing, details as flat colour on
the faces (glazing bars, lettering, stripes) rather than geometry. Each
model's triangles are printed. Lit the way walls.py lights its faces (no
lamps, no shadows): south faces 1.0, east 0.80, tops 1.12; lights that glow
(a lit beacon, a signal's aspect) are emission.

The objects, as on a British street in the 1990s:
- pillar box: the classic cylindrical Type A/C (most of them then), pillar
  box red, black plinth, beaded cap, slot, white collection plate, EIIR.
- K6 kiosk: Giles Gilbert Scott's red box (1935; tens of thousands still
  standing), crowned TELEPHONE headers, glazing in 3 x 8 panes.
- KX100 kiosk: BT's 1985 aluminium-and-glass box that replaced most K6s.
- Belisha beacon: amber globe on a black-and-white banded pole, unlit and lit.
- traffic cone: red-orange, white reflective sleeve, square black base.
- traffic signal: three-aspect black head with hoods, black backing board
  with a white border, on a grey pole with a WAIT box; red, red and amber,
  green, amber.
street_furniture_uk_02:
- TV aerial: a UHF Yagi lashed to a chimney stack (drawn on the level over
  the stack, its mast's foot reaching down to the stack's top), pointing
  S, E, N, W.
- bench: hardwood slats on black cast-iron ends, 1.8 m: two tiles a facing,
  as vanilla's park bench (SpriteGridPos).
- bollards: black cast iron with a white band; concrete; the internally lit
  white keep-left bollard of every traffic island.
- road works barrier: red and white plank and toe board on its feet, on a
  square's W or N edge as a fence, with and without its amber lamp.
road_signs_uk_01: TSRGD 1994 signs on a grey post (30, 40, national limit,
  no entry, give way, stop, road works, no through road), each facing S, E,
  N, W; then each town's entry sign, two tiles a facing.
doors_uk_01: front doors with their letter plates, as vanilla's doors lay
  them out: closed W, closed N, open W, open N.

Two-tile things (benches, place signs) are built whole, then each tile's
piece cut out of it at the square's edges (piece()); a sign's plate is a
cut-out picture (sign_faces.py) with a grey back of the same shape.
"""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RENDERS = os.path.join(ROOT, "renders")
FACES_DIR = os.path.join(RENDERS, "sign_faces")
TILE_W, TILE_H = 128, 256
SQUASH = 64 / (90.50966799 * math.cos(math.radians(30)))    # the game's heights against a true 30-degree view
SHIFT_X = 0.75 / 256                                         # calibration: the silhouette on walls.py's pixels
SAMPLES = 64
FACINGS = {"S": 0, "E": 90, "N": 180, "W": 270}              # the front faces map south, east...

# 1990s colours, sRGB.
POSTBOX_RED = (196, 30, 36)        # pillar box red (BS 381C 538)
K6_RED = (180, 28, 32)             # BT's red for the K6 in the 1990s
BLACK = (24, 24, 26)
WHITE = (230, 230, 226)
GOLD = (196, 160, 70)
SILVER = (168, 172, 176)
GLASS = (92, 112, 122)
DARK_GLASS = (40, 52, 60)
AMBER = (236, 128, 24)             # a Belisha globe: deep amber
CONE = (232, 78, 30)               # UK cones run red-orange
GREY = (128, 132, 132)             # galvanised signal pole
RED_LAMP, AMBER_LAMP, GREEN_LAMP = (255, 40, 30), (255, 170, 20), (40, 255, 150)
SIGN_BACK = (138, 142, 140)        # signs' backs: grey
POST = (150, 154, 154)             # galvanised sign post
IRON = (30, 32, 32)                # cast iron, painted black
TEAK = (146, 110, 74)              # weathered hardwood slats
CONCRETE = (172, 170, 162)
BARRIER_RED = (204, 30, 40)


def lin(c):
    """sRGB 0-255 -> linear 0-1 (what Blender's colours are)."""
    return tuple(((v / 255) / 12.92) if v / 255 <= 0.04045 else (((v / 255) + 0.055) / 1.055) ** 2.4 for v in c) + (1.0,)


# --- a tiny bitmap font for the lettering (5 x 7, 1 = ink) --------------------------------------
FONT = {
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    " ": ["00000"] * 7,
}


def text_image(name, text, ink, paper, pad=2):
    """A small image of `text` in the bitmap font: closest-sampled on a face,
    a couple of pixels a letter in the sprite - lettering, not legible type,
    as vanilla's signs are."""
    w = len(text) * 6 - 1 + 2 * pad
    h = 7 + 2 * pad
    px = np.zeros((h, w, 4), np.float32)
    px[:] = lin(paper)
    for k, ch in enumerate(text):
        for r, row in enumerate(FONT[ch]):
            for c, bit in enumerate(row):
                if bit == "1":
                    px[pad + r, pad + k * 6 + c] = lin(ink)
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.pixels = px[::-1].ravel()
    return img


# --- materials -------------------------------------------------------------------------------------
def _flat_light(nt, colour_socket):
    """walls.py's lighting on any face: tops 1.12, south 1.0, east 0.80, north
    and west 0.70, bottoms 0.5, weighted by the normal's squared components
    (so a cylinder shades smoothly round)."""
    n = nt.nodes
    geo = n.new("ShaderNodeNewGeometry")
    sep = n.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sep.inputs[0])
    total = None
    # Blender x = map east, Blender -y = map south.
    for axis, pos_w, neg_w in (("X", 0.80, 0.70), ("Y", 0.70, 1.00), ("Z", 1.12, 0.50)):
        comp = sep.outputs[axis]
        sq = n.new("ShaderNodeMath")
        sq.operation = "MULTIPLY"
        nt.links.new(comp, sq.inputs[0])
        nt.links.new(comp, sq.inputs[1])
        gt = n.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = 0.0
        nt.links.new(comp, gt.inputs[0])
        w = n.new("ShaderNodeMapRange")              # 0 -> neg_w, 1 -> pos_w
        w.inputs["To Min"].default_value = neg_w
        w.inputs["To Max"].default_value = pos_w
        nt.links.new(gt.outputs[0], w.inputs["Value"])
        term = n.new("ShaderNodeMath")
        term.operation = "MULTIPLY"
        nt.links.new(sq.outputs[0], term.inputs[0])
        nt.links.new(w.outputs[0], term.inputs[1])
        if total is None:
            total = term.outputs[0]
        else:
            add = n.new("ShaderNodeMath")
            add.operation = "ADD"
            nt.links.new(total, add.inputs[0])
            nt.links.new(term.outputs[0], add.inputs[1])
            total = add.outputs[0]
    # The factor applies to the sRGB value, as walls.py multiplies its colours: in linear, f ** 2.2.
    pw = n.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.2
    nt.links.new(total, pw.inputs[0])
    mul = n.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    nt.links.new(colour_socket, mul.inputs["Color1"])
    nt.links.new(pw.outputs[0], mul.inputs["Color2"])
    return mul.outputs["Color"]


def _grime(nt, colour_socket, amount=0.08):
    """A little 1990s weathering: the colour varied a few percent over the object."""
    n = nt.nodes
    tc = n.new("ShaderNodeTexCoord")
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    rng = n.new("ShaderNodeMapRange")
    rng.inputs["To Min"].default_value = 1.0 - amount
    rng.inputs["To Max"].default_value = 1.0 + amount / 2
    nt.links.new(noise.outputs["Fac"], rng.inputs["Value"])
    mul = n.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    nt.links.new(colour_socket, mul.inputs["Color1"])
    nt.links.new(rng.outputs[0], mul.inputs["Color2"])
    return mul.outputs["Color"]


def material(name, colour=None, image=None, glow=0.0, pattern=None, grime=0.08, interp="Closest", alpha=None):
    """A flat-lit material: a colour, an image on the face's UVs (closest, or
    `interp`), or a pattern (a function adding nodes that return a colour
    socket); glow > 0 is a lamp: emission at that strength, unlit. `alpha`
    (an image) cuts the face out to where that image is opaque, on the same
    UVs (a sign's shape; its back the same)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = interp
        uv = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
        col = tex.outputs["Color"]
    elif pattern is not None:
        col = pattern(nt)
    else:
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = lin(colour)
        col = rgb.outputs[0]
    if glow > 0:
        nt.links.new(col, em.inputs["Color"])
        em.inputs["Strength"].default_value = glow
    else:
        if grime:
            col = _grime(nt, col, grime)
        nt.links.new(_flat_light(nt, col), em.inputs["Color"])
    shader = em.outputs[0]
    if alpha is not None:
        shader = _cut_out(nt, shader, alpha)
        m["cutout"] = alpha.name
    nt.links.new(shader, out.inputs["Surface"])
    return m


def _cut_out(nt, shader, image):
    """The shader where `image` is opaque (on the UVs), nothing elsewhere."""
    n = nt.nodes
    tex = n.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"
    uv = n.new("ShaderNodeTexCoord")
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    clear = n.new("ShaderNodeBsdfTransparent")
    mix = n.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(shader, mix.inputs[2])
    return mix.outputs[0]


_FACES: dict = {}


def face(name):
    """A sign or door face from sign_faces.py (loaded once)."""
    if name not in _FACES:
        _FACES[name] = bpy.data.images.load(os.path.join(FACES_DIR, name + ".png"))
    return _FACES[name]


def stripes(axis, c1, c2, period):
    """A pattern: blocks of c1 and c2 along map `axis` ("x" or "y"), each period / 2 long (world position)."""
    def build(nt):
        n = nt.nodes
        geo = n.new("ShaderNodeNewGeometry")
        sep = n.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Position"], sep.inputs[0])
        div = n.new("ShaderNodeMath")
        div.operation = "DIVIDE"
        div.inputs[1].default_value = period
        nt.links.new(sep.outputs["X" if axis == "x" else "Y"], div.inputs[0])
        fr = n.new("ShaderNodeMath")
        fr.operation = "FRACT"
        nt.links.new(div.outputs[0], fr.inputs[0])
        gt = n.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = 0.5
        nt.links.new(fr.outputs[0], gt.inputs[0])
        mix = n.new("ShaderNodeMixRGB")
        mix.inputs["Color1"].default_value = lin(c1)
        mix.inputs["Color2"].default_value = lin(c2)
        nt.links.new(gt.outputs[0], mix.inputs["Fac"])
        return mix.outputs["Color"]
    return build


def bands(c1, c2, height):
    """A pattern: horizontal bands `height` metres deep (real metres, from the ground), c1 then c2 up."""
    def build(nt):
        n = nt.nodes
        geo = n.new("ShaderNodeNewGeometry")         # world position: heights from the ground
        sep = n.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Position"], sep.inputs[0])
        div = n.new("ShaderNodeMath")
        div.operation = "DIVIDE"
        div.inputs[1].default_value = height * SQUASH
        nt.links.new(sep.outputs["Z"], div.inputs[0])
        fr = n.new("ShaderNodeMath")
        fr.operation = "FRACT"
        nt.links.new(div.outputs[0], fr.inputs[0])
        gt = n.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = 0.5
        nt.links.new(fr.outputs[0], gt.inputs[0])
        mix = n.new("ShaderNodeMixRGB")
        mix.inputs["Color1"].default_value = lin(c1)
        mix.inputs["Color2"].default_value = lin(c2)
        nt.links.new(gt.outputs[0], mix.inputs["Fac"])
        return mix.outputs["Color"]
    return build


def band_between(base, band, z0, z1):
    """A pattern: `band` between heights z0..z1 above the ground (real metres), `base` elsewhere."""
    def build(nt):
        n = nt.nodes
        geo = n.new("ShaderNodeNewGeometry")
        sep = n.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Position"], sep.inputs[0])
        a = n.new("ShaderNodeMath")
        a.operation = "GREATER_THAN"
        a.inputs[1].default_value = z0 * SQUASH
        nt.links.new(sep.outputs["Z"], a.inputs[0])
        b = n.new("ShaderNodeMath")
        b.operation = "LESS_THAN"
        b.inputs[1].default_value = z1 * SQUASH
        nt.links.new(sep.outputs["Z"], b.inputs[0])
        both = n.new("ShaderNodeMath")
        both.operation = "MULTIPLY"
        nt.links.new(a.outputs[0], both.inputs[0])
        nt.links.new(b.outputs[0], both.inputs[1])
        mix = n.new("ShaderNodeMixRGB")
        mix.inputs["Color1"].default_value = lin(base)
        mix.inputs["Color2"].default_value = lin(band)
        nt.links.new(both.outputs[0], mix.inputs["Fac"])
        return mix.outputs["Color"]
    return build


def grid(frame, pane, cols, rows, bar=0.12, border=0.06):
    """A pattern on a face's UVs: panes in a frame (glazing bars), `bar` of a
    pane's width, `border` of the face round them."""
    def build(nt):
        n = nt.nodes
        tc = n.new("ShaderNodeTexCoord")
        sep = n.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["UV"], sep.inputs[0])
        ink = None
        for comp, count in (("X", cols), ("Y", rows)):
            # inside the border, then the bars between panes
            m1 = n.new("ShaderNodeMapRange")
            m1.inputs["From Min"].default_value = border
            m1.inputs["From Max"].default_value = 1 - border
            m1.inputs["To Max"].default_value = count
            m1.clamp = False
            nt.links.new(sep.outputs[comp], m1.inputs["Value"])
            fr = n.new("ShaderNodeMath")
            fr.operation = "FRACT"
            nt.links.new(m1.outputs[0], fr.inputs[0])
            lo = n.new("ShaderNodeMath")
            lo.operation = "LESS_THAN"
            lo.inputs[1].default_value = bar / 2
            nt.links.new(fr.outputs[0], lo.inputs[0])
            hi = n.new("ShaderNodeMath")
            hi.operation = "GREATER_THAN"
            hi.inputs[1].default_value = 1 - bar / 2
            nt.links.new(fr.outputs[0], hi.inputs[0])
            out_lo = n.new("ShaderNodeMath")
            out_lo.operation = "LESS_THAN"
            out_lo.inputs[1].default_value = border
            nt.links.new(sep.outputs[comp], out_lo.inputs[0])
            out_hi = n.new("ShaderNodeMath")
            out_hi.operation = "GREATER_THAN"
            out_hi.inputs[1].default_value = 1 - border
            nt.links.new(sep.outputs[comp], out_hi.inputs[0])
            for s in (lo, hi, out_lo, out_hi):
                if ink is None:
                    ink = s.outputs[0]
                else:
                    mx = n.new("ShaderNodeMath")
                    mx.operation = "MAXIMUM"
                    nt.links.new(ink, mx.inputs[0])
                    nt.links.new(s.outputs[0], mx.inputs[1])
                    ink = mx.outputs[0]
        mix = n.new("ShaderNodeMixRGB")
        mix.inputs["Color1"].default_value = lin(pane)
        mix.inputs["Color2"].default_value = lin(frame)
        nt.links.new(ink, mix.inputs["Fac"])
        return mix.outputs["Color"]
    return build


def board(edge, field, border=0.07):
    """A pattern on a face's UVs: a field with an edge `border` wide (a signal's backing board)."""
    return grid(edge, field, 1, 1, bar=0.0, border=border)


# --- primitives (real metres; the map frame: x east, y south; z up) -------------------------------
PARTS: list = []


def _place(obj, mat):
    obj.data.materials.append(mat)
    PARTS.append(obj)
    return obj


def box(cx, cy, z0, sx, sy, sz, mat):
    """A box centred on map (cx, cy), its foot at z0, sx east-west, sy north-south, sz high."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, -cy, (z0 + sz / 2) * SQUASH))
    o = bpy.context.object
    o.scale = (sx, sy, sz * SQUASH)
    return _place(o, mat)


def cylinder(cx, cy, z0, r, h, mat, sides=16, r2=None):
    """A cylinder (or with r2 a cone) standing on z0."""
    if r2 is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=sides, radius=r, depth=h, location=(cx, -cy, (z0 + h / 2) * SQUASH))
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=sides, radius1=r, radius2=r2, depth=h,
                                        location=(cx, -cy, (z0 + h / 2) * SQUASH))
    o = bpy.context.object
    o.scale = (1, 1, SQUASH)
    return _place(o, mat)


def sphere(cx, cy, cz, r, mat, segments=12, rings=8, squash_z=1.0):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=r, location=(cx, -cy, cz * SQUASH))
    o = bpy.context.object
    o.scale = (1, 1, SQUASH * squash_z)
    return _place(o, mat)


def _mesh(verts_map, faces, uvs, mat, name="mesh"):
    """A mesh from map-coordinate vertices (x east, y south, z up: squashed
    into Blender's frame), each face's loops given UVs."""
    verts = [(x, -y, z * SQUASH) for x, y, z in verts_map]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap")
    for poly, fuv in zip(me.polygons, uvs):
        for li, t in zip(poly.loop_indices, fuv):
            uv.data[li].uv = t
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return _place(o, mat)


def _outward(verts_map, f, inside):
    """Face f's vertex order turned so its normal (in Blender's frame) points away from `inside`."""
    b = [np.array((x, -y, z * SQUASH)) for x, y, z in (verts_map[i] for i in f)]
    c = np.array((inside[0], -inside[1], inside[2] * SQUASH))
    nrm = np.cross(b[1] - b[0], b[-1] - b[0])
    return f if np.dot(nrm, sum(b) / len(b) - c) >= 0 else f[::-1]


def quad(p0, u, v, mat, normal, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    """A flat face: corners p0, p0 + u, p0 + u + v, p0 + v (map metres), its
    front towards `normal` (map), UVs per corner in that order."""
    p0, u, v = np.array(p0, float), np.array(u, float), np.array(v, float)
    corners = [p0, p0 + u, p0 + u + v, p0 + v]
    inside = sum(corners) / 4 - np.array(normal, float)
    f = _outward(corners, [0, 1, 2, 3], inside)
    return _mesh([tuple(c) for c in corners], [f], [[uv[i] for i in f]], mat, "quad")


def slab(p0, u, v, w, mat):
    """A box with edges u, v, w (map vectors; need not be square to the grid) from corner p0."""
    p0, u, v, w = (np.array(a, float) for a in (p0, u, v, w))
    vs = [p0 + a * u + b * v + c * w for c in (0, 1) for b in (0, 1) for a in (0, 1)]
    centre = p0 + (u + v + w) / 2
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    faces = [_outward(vs, f, centre) for f in faces]
    uvs = [[(0, 0), (1, 0), (1, 1), (0, 1)] for _ in faces]
    return _mesh([tuple(x) for x in vs], faces, uvs, mat, "slab")


def sign_plate(cx, cy, zc, w, h, facing, picture, out=0.0, glow=0.0, back=True):
    """A sign: `picture` (sign_faces.py) cut out on a plate w x h centred at
    height zc, its front towards `facing`, `out` in front of (cx, cy); its
    back (grey, the same shape) just behind."""
    nx, ny = {"S": (0, 1), "E": (1, 0), "N": (0, -1), "W": (-1, 0)}[facing]
    rx, ry = ny, -nx                                         # the viewer's right, facing the sign
    img = face(picture)
    front = material(f"sign_{picture}", image=img, interp="Linear", alpha=img, grime=0.03, glow=glow)
    p0 = (cx + nx * out - rx * w / 2, cy + ny * out - ry * w / 2, zc - h / 2)
    quad(p0, (rx * w, ry * w, 0), (0, 0, h), front, (nx, ny, 0))
    if back:
        b = material(f"back_{picture}", SIGN_BACK, alpha=img, grime=0.05)
        pb = (p0[0] - nx * 0.006, p0[1] - ny * 0.006, p0[2])
        quad(pb, (rx * w, ry * w, 0), (0, 0, h), b, (-nx, -ny, 0))   # UVs as the front's: the same shape from behind


def panel(cx, cy, z0, w, h, facing, mat, out=0.004):
    """A flat face (a window, a header, a board) w wide and h high, standing
    on z0, its front towards `facing` (S, E, N, W), `out` proud of (cx, cy);
    UVs 0..1 left to right and bottom to top as seen from the front. Built as
    mesh data (the operators act on whatever is selected: a second panel came
    out at the origin)."""
    nx, ny = {"S": (0, 1), "E": (1, 0), "N": (0, -1), "W": (-1, 0)}[facing]
    rx, ry = ny, -nx                                  # the viewer's right, facing the panel
    px, py = cx + nx * out, cy + ny * out
    corners = [(px - rx * w / 2, py - ry * w / 2, z0), (px + rx * w / 2, py + ry * w / 2, z0),
               (px + rx * w / 2, py + ry * w / 2, z0 + h), (px - rx * w / 2, py - ry * w / 2, z0 + h)]
    verts = [(x, -y, z * SQUASH) for x, y, z in corners]  # map -> Blender
    face = [0, 1, 2, 3]
    e1 = np.subtract(verts[1], verts[0])
    e2 = np.subtract(verts[3], verts[0])
    if np.dot(np.cross(e1, e2), (nx, -ny, 0)) < 0:
        face = face[::-1]
    me = bpy.data.meshes.new("panel")
    me.from_pydata(verts, [], [face])
    uv = me.uv_layers.new(name="UVMap")
    uvs = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
    for loop in me.loops:
        uv.data[loop.index].uv = uvs[loop.vertex_index]
    me.update()
    o = bpy.data.objects.new("panel", me)
    bpy.context.scene.collection.objects.link(o)
    return _place(o, mat)


# --- the objects (built facing south at the square's middle; turned after) ------------------------
def pillar_box():
    red = material("pb_red", POSTBOX_RED)
    blk = material("pb_black", BLACK)
    c = 0.5
    cylinder(c, c, 0.0, 0.30, 0.10, blk)                     # plinth
    cylinder(c, c, 0.10, 0.265, 1.30, red)                    # body
    cylinder(c, c, 1.40, 0.30, 0.05, red)                     # beaded rim
    sphere(c, c, 1.45, 0.285, red, squash_z=0.42)             # cap
    box(c, c + 0.258, 1.18, 0.20, 0.03, 0.035, blk)           # the slot
    panel(c, c + 0.266, 0.96, 0.16, 0.13, "S", material("pb_plate", WHITE))   # collection plate
    panel(c, c + 0.266, 0.78, 0.07, 0.08, "S", material("pb_cypher", GOLD))   # EIIR
    # Collision as vanilla's mail box: one upright cylinder.
    return {"geometry": [("cylinder", 0.5, 0.5, 0.0, 0.29, 1.55)]}


def k6_kiosk():
    red = material("k6_red", K6_RED)
    s, h = 0.91, 2.05
    c = 0.5
    box(c, c, 0.0, s + 0.06, s + 0.06, 0.10, red)             # plinth
    box(c, c, 0.10, s, s, h, red)                             # body
    glazing = material("k6_glazing", pattern=grid(K6_RED, DARK_GLASS, 3, 8, bar=0.18, border=0.09), grime=0.04)
    header = material("k6_header", image=text_image("k6_tel", "TELEPHONE", BLACK, WHITE), grime=0.0)
    for f in FACINGS:
        dx, dy = {"S": (0, 1), "E": (1, 0), "N": (0, -1), "W": (-1, 0)}[f]
        px, py = c + dx * s / 2, c + dy * s / 2
        panel(px, py, 0.30, s * 0.72, 1.62, f, glazing)
        panel(px, py, 2.18 - 0.30, s * 0.80, 0.14, f, header)  # (on the body's top band)
    box(c, c, 0.10 + h, s + 0.04, s + 0.04, 0.10, red)        # cornice
    sphere(c, c, 0.20 + h, s / 2 * 0.98, red, segments=8, rings=4, squash_z=0.30)  # the shallow domed roof
    box(c + 0.30, c + s / 2 + 0.01, 0.95, 0.03, 0.02, 0.12, material("k6_handle", BLACK))  # door handle
    return {"geometry": [("box", 0.5 - s / 2, 0.5 - s / 2, 0.0, 0.5 + s / 2, 0.5 + s / 2, 2.4)]}


def kx100_kiosk():
    al = material("kx_alu", SILVER, grime=0.05)
    s, h = 0.90, 2.15
    c = 0.5
    box(c, c, 0.0, s, s, 0.08, material("kx_base", (90, 92, 94)))
    for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):        # corner posts
        box(c + ox * (s / 2 - 0.03), c + oy * (s / 2 - 0.03), 0.08, 0.06, 0.06, h - 0.08, al)
    glass = material("kx_glass", pattern=grid(SILVER, GLASS, 1, 2, bar=0.06, border=0.04), grime=0.03)
    header = material("kx_header", image=text_image("kx_tel", "TELEPHONE", WHITE, (40, 44, 48)), grime=0.0)
    for f in FACINGS:
        dx, dy = {"S": (0, 1), "E": (1, 0), "N": (0, -1), "W": (-1, 0)}[f]
        px, py = c + dx * (s / 2 - 0.01), c + dy * (s / 2 - 0.01)
        panel(px, py, 0.10, s - 0.12, h - 0.40, f, glass)
        panel(px, py, h - 0.26, s - 0.02, 0.20, f, header)
    box(c, c, h, s + 0.04, s + 0.04, 0.07, al)                # flat roof
    return {"geometry": [("box", 0.5 - s / 2, 0.5 - s / 2, 0.0, 0.5 + s / 2, 0.5 + s / 2, 2.25)]}


def belisha_beacon(lit: bool):
    c = 0.5
    cylinder(c, c, 0.0, 0.07, 0.06, material("bb_foot", BLACK))
    cylinder(c, c, 0.06, 0.05, 2.30, material("bb_pole", pattern=bands(BLACK, WHITE, 0.30)), sides=12)
    cylinder(c, c, 2.36, 0.06, 0.06, material("bb_collar", BLACK), sides=12)
    if lit:
        sphere(c, c, 2.58, 0.17, material("bb_globe_lit", AMBER, glow=1.15))
    else:
        sphere(c, c, 2.58, 0.17, material("bb_globe", (196, 110, 34)))
    return {"geometry": [("cylinder", 0.5, 0.5, 0.0, 0.07, 2.75)]}


def traffic_cone():
    c = 0.5
    box(c, c, 0.0, 0.38, 0.38, 0.035, material("tc_base", BLACK))
    cylinder(c, c, 0.035, 0.15, 0.68, material("tc_body", pattern=band_between(CONE, WHITE, 0.33, 0.50)),
             sides=12, r2=0.025)
    return {"geometry": [("box", 0.31, 0.31, 0.0, 0.69, 0.69, 0.72)]}


SIGNAL_STATES = {"red": ("red",), "red_amber": ("red", "amber"), "green": ("green",), "amber": ("amber",)}


def traffic_signal(state: str):
    c = 0.5
    grey = material("ts_pole", GREY)
    blk = material("ts_black", BLACK)
    cylinder(c, c, 0.0, 0.055, 3.25, grey, sides=12)                    # pole
    box(c, c + 0.08, 1.05, 0.10, 0.08, 0.16, material("ts_wait", (62, 64, 66)))     # push-button box (front)
    head_y = c + 0.16                                                    # the head in front of the pole
    panel(c, head_y - 0.06, 2.20, 0.56, 1.08, "S", material("ts_board", pattern=board(WHITE, BLACK, 0.06), grime=0.03))
    box(c, head_y, 2.30, 0.30, 0.22, 0.88, blk)                         # signal head
    lit = SIGNAL_STATES[state]
    for k, (name, col) in enumerate((("red", RED_LAMP), ("amber", AMBER_LAMP), ("green", GREEN_LAMP))):
        z = 2.30 + 0.88 - 0.16 - k * 0.28
        on = name in lit
        dim = tuple(int(v * 0.22) + 18 for v in col)
        lens = material(f"ts_{name}_{'on' if on else 'off'}", col if on else dim, glow=3.0 if on else 0.0, grime=0.0)
        panel(c, head_y + 0.11, z - 0.11, 0.20, 0.20, "S", lens)
        box(c, head_y + 0.18, z + 0.08, 0.24, 0.14, 0.03, blk)          # hood
    return {"geometry": [("cylinder", 0.5, 0.5, 0.0, 0.06, 3.25), ("box", 0.35, 0.55, 2.3, 0.65, 0.77, 3.18)]}


AERIAL_MAST = (0.80, 0.80)              # on a stack's top, at its front corner (its pots fill the middle)
AERIAL_FOOT = -0.45                     # the mast's foot below its own floor: on the level over a
                                        # chimney stack (2.7 m tall, of a 3 m storey) it meets the stack's top


def tv_aerial(facing):
    """A 1990s UHF Yagi lashed to a chimney stack, its boom pointing `facing`
    (towards the transmitter): reflector at the back, dipole box, seven
    directors. Not turned whole: the mast stays at the stack's corner."""
    alu = material("ae_alu", (178, 182, 186), grime=0.04)
    mx, my = AERIAL_MAST
    bx, by = {"S": (0, 1), "E": (1, 0), "N": (0, -1), "W": (-1, 0)}[facing]

    def rod(t, z, length, thick=0.012, along=False, h=None):
        """A rod t metres along the boom from the mast at height z: across it, or along it."""
        cx, cy = mx + bx * t, my + by * t
        if along:
            sx, sy = (abs(bx) * length or thick), (abs(by) * length or thick)
        else:
            sx, sy = (abs(by) * length or thick), (abs(bx) * length or thick)
        box(cx, cy, z, sx, sy, h or thick, alu)

    cylinder(mx, my, AERIAL_FOOT, 0.02, 1.75 - AERIAL_FOOT, alu, sides=8)               # mast
    box(mx, my, 1.58, 0.05, 0.05, 0.08, material("ae_clamp", (90, 92, 94)))             # clamp
    zb = 1.62
    rod(0.18, zb, 0.70, thick=0.018, along=True)                                        # boom, -0.17..0.53
    rod(-0.15, zb - 0.16, 0.014, thick=0.014, h=0.34)                                   # reflector upright
    for dz in (-0.14, -0.05, 0.05, 0.14):
        rod(-0.15, zb + dz, 0.34)                                                       # reflector rods
    cx, cy = mx - bx * 0.02, my - by * 0.02
    box(cx, cy, zb - 0.03, 0.06, 0.06, 0.04, material("ae_box", BLACK))                 # dipole box
    rod(-0.02, zb, 0.28, thick=0.014)                                                   # dipole
    for k in range(7):
        rod(0.08 + k * 0.065, zb, 0.24 - k * 0.006)                                     # directors
    return {"geometry": []}


def bench():
    """A 1.8 m bench, its seat facing south, from x 0.1 to 1.9 (two squares):
    three seat slats, three back slats leaning back, cast-iron ends."""
    wood = material("bn_wood", TEAK, grime=0.12)
    iron = material("bn_iron", IRON)
    x0, x1 = 0.10, 1.90
    for k in range(3):                                                      # seat, front (south) to back
        slab((x0, 0.60 - k * 0.125, 0.40), (x1 - x0, 0, 0), (0, 0.10, 0), (0, 0, 0.035), wood)
    for k in range(3):
        z = 0.52 + k * 0.13
        p0 = np.array((x0, 0.30 - (z - 0.48) * 0.24, z))
        slab(p0, (x1 - x0, 0, 0), (0, 0.03, 0), (0, -0.025, 0.10), wood)
    for x in (0.16, 1.84):                                                  # the cast-iron ends
        slab((x - 0.025, 0.66, 0.0), (0.05, 0, 0), (0, 0.05, 0), (0, 0, 0.44), iron)        # front leg
        slab((x - 0.025, 0.30, 0.0), (0.05, 0, 0), (0, 0.05, 0), (0, -0.07, 0.90), iron)    # back leg and back
        slab((x - 0.025, 0.28, 0.38), (0.05, 0, 0), (0, 0.44, 0), (0, 0, 0.04), iron)      # seat rail
        slab((x - 0.03, 0.30, 0.62), (0.06, 0, 0), (0, 0.42, 0), (0, 0, 0.04), iron)       # arm
        slab((x - 0.02, 0.68, 0.40), (0.04, 0, 0), (0, 0.03, 0), (0, 0, 0.24), iron)       # arm's front post
    return {"geometry": [("box", x0, 0.28, 0.0, x1, 0.72, 0.45), ("box", x0, 0.20, 0.0, x1, 0.34, 0.92)]}


def bollard(kind):
    c = 0.5
    if kind == "iron":                                                      # black cast iron, a white band
        iron = material("bo_iron", pattern=band_between(IRON, WHITE, 0.66, 0.74))
        cylinder(c, c, 0.0, 0.11, 0.06, material("bo_foot", IRON), sides=12)
        cylinder(c, c, 0.06, 0.085, 0.78, iron, sides=12, r2=0.072)
        cylinder(c, c, 0.84, 0.088, 0.04, material("bo_collar", IRON), sides=12)
        sphere(c, c, 0.88, 0.072, material("bo_cap", IRON), segments=12, rings=6, squash_z=0.7)
        return {"geometry": [("cylinder", 0.5, 0.5, 0.0, 0.10, 0.95)]}
    con = material("bo_concrete", CONCRETE, grime=0.14)                    # precast concrete, chamfered top
    cylinder(c, c, 0.0, 0.16, 0.66, con, sides=12)
    cylinder(c, c, 0.66, 0.16, 0.08, con, sides=12, r2=0.11)
    return {"geometry": [("cylinder", 0.5, 0.5, 0.0, 0.16, 0.74)]}


def keep_left():
    """The white, internally lit keep-left bollard of a traffic island; its sign faces south."""
    c = 0.5
    box(c, c, 0.0, 0.46, 0.38, 0.10, material("kl_base", (70, 72, 74)))
    box(c, c, 0.10, 0.40, 0.32, 0.92, material("kl_body", (232, 232, 226), grime=0.10))
    box(c, c, 1.02, 0.42, 0.34, 0.06, material("kl_cap", (210, 210, 204)))
    sign_plate(c, c + 0.16, 0.76, 0.30, 0.30, "S", "keep_left", out=0.005, glow=0.9, back=False)
    return {"geometry": [("box", 0.30, 0.31, 0.0, 0.70, 0.69, 1.08)]}


def barrier(edge, lamp):
    """A 1990s road works barrier on the square's W or N edge: a red and
    white plank at waist height and a toe board, on two posts with feet;
    `lamp`, an amber road lamp on the far post."""
    along = "y" if edge == "W" else "x"
    plank = material(f"rb_plank_{along}", pattern=stripes(along, BARRIER_RED, WHITE, 0.5), grime=0.06)
    post = material("rb_post", (60, 62, 62))
    feet = material("rb_foot", (34, 34, 34))
    if edge == "N":
        box(0.5, 0.03, 0.80, 0.98, 0.025, 0.18, plank)
        box(0.5, 0.03, 0.04, 0.98, 0.025, 0.14, plank)
        for x in (0.05, 0.95):
            box(x, 0.03, 0.0, 0.035, 0.035, 1.0, post)
            box(x, 0.10, 0.0, 0.06, 0.20, 0.04, feet)
        top = (0.95, 0.03)
    else:
        box(0.03, 0.5, 0.80, 0.025, 0.98, 0.18, plank)
        box(0.03, 0.5, 0.04, 0.025, 0.98, 0.14, plank)
        for y in (0.05, 0.95):
            box(0.03, y, 0.0, 0.035, 0.035, 1.0, post)
            box(0.10, y, 0.0, 0.20, 0.06, 0.04, feet)
        top = (0.03, 0.95)
    if lamp:
        box(top[0], top[1], 1.0, 0.07, 0.07, 0.09, material("rb_battery", (30, 30, 30)))
        cylinder(top[0], top[1], 1.09, 0.045, 0.07, material("rb_lamp", AMBER_LAMP, glow=1.6), sides=12)
    if edge == "N":
        return {"geometry": [("box", 0.0, 0.0, 0.0, 1.0, 0.06, 1.0)]}
    return {"geometry": [("box", 0.0, 0.0, 0.0, 0.06, 1.0, 1.0)]}


POLICE_BLUE = (24, 44, 86)


def police_box():
    """An easter egg: the Metropolitan Police box of 1929 (Mackenzie Trench),
    as the real ones stood - blue, panelled, POLICE PUBLIC CALL BOX on each
    side, a stepped roof and a lamp; its notice on the front (south) door.
    1.36 m square in life: drawn to fill the square, 2.6 m to the lamp."""
    c, w = 0.5, 0.94
    blue = material("pbx_blue", POLICE_BLUE, grime=0.06)
    box(c, c, 0.0, w + 0.04, w + 0.04, 0.06, material("pbx_plinth", (96, 98, 96)))
    box(c, c, 0.06, w - 0.10, w - 0.10, 1.96, blue)                      # the body behind the panels
    for ox in (-1, 1):
        for oy in (-1, 1):                                                # corner posts
            box(c + ox * (w / 2 - 0.05), c + oy * (w / 2 - 0.05), 0.06, 0.10, 0.10, 2.12, blue)
    header = material("pbx_header", image=face("police_header"), interp="Linear", grime=0.0)
    side = material("pbx_side", image=face("police_side"), interp="Linear", grime=0.04)
    front = material("pbx_front", image=face("police_front"), interp="Linear", grime=0.04)
    for f in FACINGS:
        dx, dy = {"S": (0, 1), "E": (1, 0), "N": (0, -1), "W": (-1, 0)}[f]
        px, py = c + dx * (w / 2 - 0.05), c + dy * (w / 2 - 0.05)
        panel(px, py, 0.10, w - 0.20, 1.80, f, front if f == "S" else side)
        box(c + dx * (w / 2 - 0.03), c + dy * (w / 2 - 0.03), 1.94,
            (w - 0.08) if dy else 0.06, (w - 0.08) if dx else 0.06, 0.20, material("pbx_black", BLACK))
        panel(c + dx * (w / 2), c + dy * (w / 2), 1.96, w - 0.16, 0.16, f, header)
    box(c, c, 2.14, w + 0.06, w + 0.06, 0.07, blue)                      # the stepped roof
    box(c, c, 2.21, w - 0.12, w - 0.12, 0.07, blue)
    box(c, c, 2.28, w - 0.32, w - 0.32, 0.06, blue)
    box(c, c, 2.34, 0.16, 0.16, 0.04, blue)
    cylinder(c, c, 2.38, 0.06, 0.14, material("pbx_lamp", (250, 250, 236), glow=1.3), sides=10)
    cylinder(c, c, 2.52, 0.075, 0.04, blue, sides=10)
    return {"geometry": [("box", 0.5 - w / 2, 0.5 - w / 2, 0.0, 0.5 + w / 2, 0.5 + w / 2, 2.4)]}


# Road signs: the face (sign_faces.py) and its plate's size in metres (w, h):
# a size up on TSRGD's smallest (600 mm roundels), so they read in the sprite.
ROAD_SIGNS = {"speed_30": (0.75, 0.75), "speed_40": (0.75, 0.75), "national": (0.75, 0.75),
              "no_entry": (0.75, 0.75), "give_way": (0.90, 0.79), "stop": (0.75, 0.75),
              "road_works": (0.90, 0.79), "no_through_road": (0.60, 0.75)}


def road_sign(name):
    """A sign on a grey post, its face south (turned to its facing)."""
    w, h = ROAD_SIGNS[name]
    c, zc = 0.5, 2.20
    top = zc + h / 2 + 0.03
    cylinder(c, c, 0.0, 0.035, top, material("rs_post", POST), sides=10)
    cylinder(c, c, top, 0.038, 0.02, material("rs_cap", (110, 112, 112)), sides=10)
    sign_plate(c, c, zc, w, h, "S", name, out=0.05)
    return {"geometry": [("cylinder", 0.5, 0.5, 0.0, 0.04, top)]}


def place_sign(town):
    """A town's entry sign, 1.6 x 0.75 m on two posts, centred on x = 1 (two
    squares), facing south: the board drawn as two halves, one a square,
    so the cut between the tiles falls on their seam."""
    post = material("ps_post", POST)
    for x in (0.28, 1.72):
        box(x, 0.5, 0.0, 0.06, 0.06, 1.42, post)
    img = face(f"place_{town}")
    front = material(f"place_{town}", image=img, interp="Linear", grime=0.03)
    back = material("ps_back", SIGN_BACK, grime=0.05)
    z0, h = 0.62, 0.75
    for x0, x1, u0, u1 in ((0.2, 1.0, 0.0, 0.5), (1.0, 1.8, 0.5, 1.0)):
        quad((x0, 0.54, z0), (x1 - x0, 0, 0), (0, 0, h), front, (0, 1, 0), uv=((u0, 0), (u1, 0), (u1, 1), (u0, 1)))
        quad((x0, 0.533, z0), (x1 - x0, 0, 0), (0, 0, h), back, (0, -1, 0))
        slab((x0, 0.533, z0 + h), (x1 - x0, 0, 0), (0, 0.007, 0), (0, 0, 0.012), back)      # the board's top edge
    return {"geometry": [("box", 0.25, 0.47, 0.0, 0.31, 0.53, 1.42), ("box", 1.69, 0.47, 0.0, 1.75, 0.53, 1.42),
                         ("box", 0.2, 0.52, 0.62, 1.8, 0.55, 1.37)]}


# Front doors: the leaf's paint (its edges) and furniture (sign_faces.py DOORS draws the faces).
DOOR_PAINT = {"black_victorian": (26, 26, 28), "red_victorian": (128, 22, 30), "blue_edwardian": (28, 44, 92),
              "green_victorian": (26, 66, 46), "cream_victorian": (222, 212, 176), "white_upvc": (236, 236, 232),
              "hardwood_fanlight": (112, 52, 30), "oak_cottage": (150, 108, 62)}
DOOR_METAL = {"white_upvc": (190, 194, 198), "oak_cottage": (30, 30, 30)}
DOOR_STATES = ("closed_W", "closed_N", "open_W", "open_N")
DOOR_W, DOOR_H, DOOR_T = 0.80, 2.35, 0.045


def front_door(style, state):
    """A door leaf as vanilla's lie (tileGeometry.txt, fixtures_doors_01_0-3):
    shut in the square's W or N edge, or swung open 90 degrees into the
    square about its hinge at the edge's north or west end. `o` is the
    hinge-side corner, `a` along the leaf (hinge to latch), `t` its thickness."""
    o, a, t = {"closed_W": ((0.0015, 0.10), (0, 1), (1, 0)), "closed_N": ((0.10, 0.0015), (1, 0), (0, 1)),
               "open_W": ((0.024, 0.10), (1, 0), (0, 1)), "open_N": ((0.10, 0.024), (0, 1), (1, 0))}[state]
    o, a, t = np.array(o + (0,), float), np.array(a + (0,), float), np.array(t + (0,), float)
    up = np.array((0, 0, 1.0))
    edge = material(f"dr_{style}_edge", DOOR_PAINT[style], grime=0.05)
    slab(o, a * DOOR_W, t * DOOR_T, up * DOOR_H, edge)
    pic = material(f"dr_{style}", image=face(f"door_{style}"), interp="Linear", grime=0.05)
    metal = material(f"dr_{style}_metal", DOOR_METAL.get(style, GOLD), grime=0.0)
    for side, off in ((-1, -0.002), (1, DOOR_T + 0.002)):                  # both faces, the same picture
        quad(o + t * off, a * DOOR_W, up * DOOR_H, pic, tuple(t * side))
        f0 = o + (t * off if side > 0 else t * (off - 0.010))
        slab(f0 + a * DOOR_W * 0.30 + up * (DOOR_H * 0.51 - 0.05), a * DOOR_W * 0.40, t * 0.010, up * 0.10, metal)  # letter plate
        slab(f0 + a * DOOR_W * 0.82 + up * (DOOR_H * 0.48 - 0.02), a * 0.04, t * 0.010, up * 0.04, metal)          # knob
    lo, hi = np.minimum(o, o + a * DOOR_W + t * DOOR_T), np.maximum(o, o + a * DOOR_W + t * DOOR_T)
    return {"geometry": [("box", lo[0], lo[1], 0.0, hi[0], hi[1], DOOR_H)]}


# The sheets: per slot (name, builder, args, facing or None for a round
# thing, grid: a two-tile thing's piece, its square in the whole). A sheet's
# slots only ever grow at the end (a sprite's id is its place).
def slot(name, builder=None, args=(), facing=None, grid=None):
    return (name, builder, args, facing, grid)


SHEETS = {}
s1 = []
for f in FACINGS:
    s1.append(slot(f"pillar_box_{f}", pillar_box, (), f))
for f in FACINGS:
    s1.append(slot(f"k6_kiosk_{f}", k6_kiosk, (), f))
for f in FACINGS:
    s1.append(slot(f"kx100_kiosk_{f}", kx100_kiosk, (), f))
s1.append(slot("belisha_beacon", belisha_beacon, (False,)))
s1.append(slot("belisha_beacon_lit", belisha_beacon, (True,)))
s1.append(slot("traffic_cone", traffic_cone))
s1.append(slot("empty"))
for state in SIGNAL_STATES:
    for f in FACINGS:
        s1.append(slot(f"signal_{state}_{f}", traffic_signal, (state,), f))
SHEETS["street_furniture_uk_01"] = s1

# Benches as vanilla's park bench (furniture_seating_outdoor_01_0-7): per
# facing two pieces, each at its SpriteGridPos.
BENCH_PIECES = [("S", (0, 0)), ("S", (1, 0)), ("E", (0, 1)), ("E", (0, 0)),
                ("N", (0, 0)), ("N", (1, 0)), ("W", (0, 1)), ("W", (0, 0))]
s2 = [slot(f"tv_aerial_{f}", tv_aerial, (f,)) for f in FACINGS]
s2 += [slot(f"bench_{f}_{gx}{gy}", bench, (), f, (gx, gy)) for f, (gx, gy) in BENCH_PIECES]
s2 += [slot("bollard_iron", bollard, ("iron",)), slot("bollard_concrete", bollard, ("concrete",))]
s2 += [slot(f"keep_left_{f}", keep_left, (), f) for f in FACINGS]
s2 += [slot("barrier_W", barrier, ("W", False)), slot("barrier_N", barrier, ("N", False)),
       slot("barrier_lamp_W", barrier, ("W", True)), slot("barrier_lamp_N", barrier, ("N", True))]
s2 += [slot("police_box_S", police_box, (), "S"), slot("police_box_E", police_box, (), "E")]   # (were empty)
SHEETS["street_furniture_uk_02"] = s2

TOWNS = ("odiham", "alton")                   # sign_faces.py TOWNS; a new town: add at the end
s3 = [slot(f"sign_{n}_{f}", road_sign, (n,), f) for n in ROAD_SIGNS for f in FACINGS]
for town in TOWNS:
    for f in FACINGS:
        for g in ((0, 0), (1, 0)) if f in "SN" else ((0, 0), (0, 1)):
            s3.append(slot(f"place_{town}_{f}_{g[0]}{g[1]}", place_sign, (town,), f, g))
SHEETS["road_signs_uk_01"] = s3

SHEETS["doors_uk_01"] = [slot(f"door_{st}_{state}", front_door, (st, state)) for st in DOOR_PAINT for state in DOOR_STATES]


# --- rendering ---------------------------------------------------------------------------------------
def scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = TILE_W, TILE_H, 100
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    target = bpy.data.objects.new("target", None)
    sc.collection.objects.link(target)
    target.location = (-1.0, 1.0, 0.0)                       # map (-1, -1): the square's north corner 64 px below the middle
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2 * math.sqrt(2)
    cam_data.shift_x = SHIFT_X
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    h = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)           # towards the viewer: map (+x, +y)
    d = math.cos(math.radians(30)) * h + np.array([0, 0, math.sin(math.radians(30))])
    cam.location = tuple(np.array(target.location) + 30 * d)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    sc.camera = cam
    return sc


def position_material(cutout=None):
    """Emission of each point's Blender position (+10, kept positive): the
    depth pass; with `cutout` (an image) only where it is opaque."""
    m = bpy.data.materials.new("position")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (10, 10, 10)
    em = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(geo.outputs["Position"], add.inputs[0])
    nt.links.new(add.outputs[0], em.inputs["Color"])
    shader = em.outputs[0]
    if cutout is not None:
        shader = _cut_out(nt, shader, bpy.data.images[cutout])
    nt.links.new(shader, out.inputs["Surface"])
    return m


def turn(facing):
    """Every part turned about the square's middle so the front faces `facing`
    (each part's world matrix rotated directly: parenting to a pivot not yet
    updated moved them by the pivot's place as well)."""
    if facing is None:
        return
    from mathutils import Matrix
    ang = math.radians(FACINGS[facing])
    m = Matrix.Translation((0.5, -0.5, 0)) @ Matrix.Rotation(ang, 4, "Z") @ Matrix.Translation((-0.5, 0.5, 0))
    bpy.context.view_layer.update()                  # (each part's matrix_world as its scale and place now are)
    for o in PARTS:
        o.matrix_world = m @ o.matrix_world
    bpy.context.view_layer.update()


def piece(grid, geom):
    """Cut a two-tile thing (already turned) down to one square's piece: the
    whole shifted so its squares start at 0 and `grid` is the square 0..1,
    each part cut there (bmesh bisect) and dropped when nothing is left; its
    collision shapes likewise. Returns the piece's shapes."""
    import bmesh
    from mathutils import Matrix, Vector
    bpy.context.view_layer.update()
    for o in PARTS:                                  # bake each part's transform into its mesh
        o.data.transform(o.matrix_world)
        o.matrix_world = Matrix.Identity(4)
    xs = [v.co.x for o in PARTS for v in o.data.vertices]
    ys = [-v.co.y for o in PARTS for v in o.data.vertices]
    fx, fy = math.floor(min(xs) + 0.05), math.floor(min(ys) + 0.05)
    dx, dy = -(fx + grid[0]), -(fy + grid[1])
    planes = [((0, 0, 0), (-1, 0, 0)), ((1, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, -1, 0), (0, -1, 0))]
    eps = 1e-4
    for o in list(PARTS):
        me = o.data
        me.transform(Matrix.Translation((dx, -dy, 0)))
        vx = [v.co.x for v in me.vertices]
        vy = [-v.co.y for v in me.vertices]
        if max(vx) <= eps or min(vx) >= 1 - eps or max(vy) <= eps or min(vy) >= 1 - eps:
            PARTS.remove(o)
            bpy.data.objects.remove(o)
            continue
        if min(vx) >= -eps and max(vx) <= 1 + eps and min(vy) >= -eps and max(vy) <= 1 + eps:
            continue
        bm = bmesh.new()
        bm.from_mesh(me)
        for co, no in planes:
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=Vector(co),
                                   plane_no=Vector(no), clear_outer=True, dist=eps)
        bm.to_mesh(me)
        bm.free()
        me.update()
    out = []
    for g in geom:
        if g[0] == "cylinder":
            _, x, y, z0, r, h = g
            if 0 <= x + dx <= 1 and 0 <= y + dy <= 1:
                out.append(("cylinder", x + dx, y + dy, z0, r, h))
            continue
        _, x0, y0, z0, x1, y1, z1 = g
        x0, x1 = sorted((x0 + dx, x1 + dx))
        y0, y1 = sorted((y0 + dy, y1 + dy))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, 1), min(y1, 1)
        if x1 - x0 > 0.01 and y1 - y0 > 0.01:
            out.append(("box", x0, y0, z0, x1, y1, z1))
    return out


def triangles():
    dg = bpy.context.evaluated_depsgraph_get()
    n = 0
    for o in PARTS:
        me = o.evaluated_get(dg).to_mesh()
        me.calc_loop_triangles()
        n += len(me.loop_triangles)
        o.evaluated_get(dg).to_mesh_clear()
    return n


def turn_geometry(geom, facing):
    """Collision shapes turned with the object (about the square's middle)."""
    if facing is None:
        return geom
    k = FACINGS[facing] // 90
    out = []
    for g in geom:
        if g[0] == "cylinder":
            _, x, y, z0, r, h = g
            for _ in range(k):                               # map (x, y) -> (y, 1 - x): a quarter turn, south to east
                x, y = y, 1 - x
            out.append(("cylinder", x, y, z0, r, h))
        else:
            _, x0, y0, z0, x1, y1, z1 = g
            for _ in range(k):                               # (as turn(): (1 - y, x) put an east signal's head west)
                x0, y0, x1, y1 = y0, 1 - x1, y1, 1 - x0
            out.append(("box", x0, y0, z0, x1, y1, z1))
    return out


def render(path, samples, filt, fmt):
    sc = bpy.context.scene
    sc.cycles.samples = samples
    sc.cycles.pixel_filter_type = "BOX" if filt < 0.1 else "BLACKMAN_HARRIS"
    sc.cycles.filter_width = filt
    sc.render.image_settings.file_format = fmt
    if fmt == "OPEN_EXR":
        sc.render.image_settings.color_depth = "32"
    else:
        sc.render.image_settings.color_mode = "RGBA"
        sc.render.image_settings.color_depth = "8"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sheets = [a for a in args if a in SHEETS] or list(SHEETS)
    only = {a for a in args if a not in SHEETS}
    for sheet in sheets:
        render_sheet(sheet, only)


def render_sheet(sheet, only):
    out_dir = os.path.join(RENDERS, sheet)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for k, (name, builder, a, facing, grid) in enumerate(SHEETS[sheet]):
        entry = {"slot": k, "name": name, "facing": facing}
        if builder is None:
            manifest.append(entry)
            continue
        if only and name not in only:
            continue
        PARTS.clear()
        _FACES.clear()
        scene()
        info = builder(*a)
        turn(facing)
        geom = turn_geometry(info["geometry"], facing)
        if grid is not None:
            entry["grid"] = list(grid)
            geom = piece(grid, geom)
        entry["triangles"] = triangles()
        entry["geometry"] = geom
        render(os.path.join(out_dir, f"{k:02d}_{name}.png"), SAMPLES, 1.0, "PNG")
        plain, cut = position_material(), {}
        for o in PARTS:
            c = o.data.materials[0].get("cutout") if o.data.materials else None
            if c and c not in cut:
                cut[c] = position_material(c)
            o.data.materials.clear()
            o.data.materials.append(cut[c] if c else plain)
        exr = os.path.join(out_dir, f"{k:02d}_{name}.exr")
        render(exr, 1, 0.01, "OPEN_EXR")
        img = bpy.data.images.load(exr)
        a_ = np.array(img.pixels[:], np.float32).reshape(TILE_H, TILE_W, 4)[::-1]
        bpy.data.images.remove(img)
        p = a_[..., :3] - 10.0
        mapxyz = np.stack([p[..., 0], -p[..., 1], p[..., 2] / SQUASH, a_[..., 3]], axis=-1)   # map x, y, z (m), coverage
        np.save(os.path.join(out_dir, f"{k:02d}_{name}.npy"), mapxyz.astype(np.float32))
        os.remove(exr)
        manifest.append(entry)
        print(f"{sheet} slot {k:2d} {name}: {entry['triangles']} triangles")
    old = {}
    mpath = os.path.join(out_dir, "manifest.json")
    if only and os.path.exists(mpath):
        old = {e["slot"]: e for e in json.load(open(mpath))["slots"]}
    for e in manifest:
        old[e["slot"]] = e
    json.dump({"sheet": sheet, "cols": 8, "slots": [old[k] for k in sorted(old)]}, open(mpath, "w"), indent=1)


if __name__ == "__main__":
    main()
