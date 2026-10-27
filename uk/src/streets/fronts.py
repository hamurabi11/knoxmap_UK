"""Garden fronts along a redrawn diagonal road: one line, not a staircase of pieces.

Called by kerbs.py after diagonals.py. plots.py squares every garden into at
most three rectangles, so where a row of gardens fronts a 45-degree road each
front boundary steps every few tiles: 4-tile runs of hedge with gaps between,
which in game read as a row of "^". Here, for each redrawn stretch:

1. The garden-front pieces just beyond its pavement are set aside: lines
   drawn by plots.py lying wholly within FRONT_REACH tiles of the pavement's
   outer edge, in the stretch, none longer than PIECE_TILES.
2. The front is drawn again along the back of the pavement, every edge
   between a pavement square and the garden square beside it, in the style
   of the nearest piece set aside - and only where one was (a drive or an
   open frontage stays open).
3. A fence or railing follows that edge as one staircase of 1-tile steps; a
   hedge becomes bushes on the garden squares along it, the way a hedge
   reads on the diagonal (a clipped hedge in 1-tile steps is a saw).

Returns (lines kept, lines set aside, new lines, bush squares).
"""
from __future__ import annotations

import math

import numpy as np

FRONT_REACH = 6          # tiles beyond the pavement a garden-front piece may lie
PIECE_TILES = 6          # longer lines are side or back boundaries, left alone
HEDGES = {"hedge_tall", "hedge_low"}
SOURCE = "kerbs_front"


def _uv(x, y, down):
    return (x - y, x + y) if down else (x + y, x - y)


def redraw(features: list, stretches: list, street: np.ndarray, blocked: np.ndarray, proj):
    H, W = street.shape
    kept, aside, new, bushes = [], [], [], set()
    pieces = []                     # (feature, points in tiles)
    for ft in features:
        props = ft.get("properties") or {}
        if props.get("source") == SOURCE:
            continue                # drawn on a previous run: drawn again below
        g = ft.get("geometry") or {}
        if props.get("source") == "plots" and g.get("type") == "LineString":
            pieces.append((ft, [proj.to_px(la, lo) for lo, la in g["coordinates"]]))
        else:
            kept.append(ft)
    taken = set()
    edges = []                      # (corner a, corner b, style, outside square)
    for st in stretches:
        if not st["side"]:
            continue
        down, u0, n, side, lo, hi = st["down"], st["u0"], st["n"], st["side"], st["lo"], st["hi"]
        # The two outer edges of the pavement, in u: below u0 - side, above u0 + n + side - 1.
        outer = {-1: u0 - side, +1: u0 + n + side - 1}
        found = {-1: [], +1: []}
        for k, (ft, pts) in enumerate(pieces):
            if k in taken:
                continue
            length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
            if length > PIECE_TILES:
                continue
            uv = [_uv(x, y, down) for x, y in pts]
            if not all(lo <= v <= hi for _, v in uv):
                continue
            for sgn, ue in outer.items():
                beyond = [(u - ue) * sgn for u, _ in uv]
                if all(-1 <= b <= FRONT_REACH * 2 for b in beyond):   # u moves 2 a diagonal step
                    found[sgn].append((ft, pts))
                    taken.add(k)
                    break
        for sgn, got in found.items():
            if not got:
                continue
            mids = np.array([np.mean(p, axis=0) for _, p in got])
            styles = [(f.get("properties") or {}).get("knox_style") for f, _ in got]
            # Every edge between a pavement square on this outer row and the
            # garden square beside it, in the stretch.
            ue = outer[sgn]
            for y in range(H):
                xs = []
                if down:
                    x = ue + y
                    xs = [x] if 0 <= x < W else []
                else:
                    x = ue - y
                    xs = [x] if 0 <= x < W else []
                for x in xs:
                    u, v = _uv(x, y, down)
                    if not (lo <= v <= hi) or not street[y, x]:
                        continue
                    for nx, ny, a, b in ((x - 1, y, (x, y), (x, y + 1)), (x + 1, y, (x + 1, y), (x + 1, y + 1)),
                                         (x, y - 1, (x, y), (x + 1, y)), (x, y + 1, (x, y + 1), (x + 1, y + 1))):
                        if not (0 <= nx < W and 0 <= ny < H) or street[ny, nx] or blocked[ny, nx]:
                            continue
                        if (_uv(nx, ny, down)[0] - u) * sgn <= 0:
                            continue            # the garden lies beyond the pavement
                        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                        d = np.hypot(mids[:, 0] - mid[0], mids[:, 1] - mid[1])
                        i = int(np.argmin(d))
                        if d[i] <= FRONT_REACH + 2:
                            edges.append((a, b, styles[i], (nx, ny)))
        aside += [ft for ft, _ in found[-1] + found[+1]]
    kept += [ft for k, (ft, _) in enumerate(pieces) if k not in taken]

    # Chained into lines, one per style; hedges as bushes.
    by_style: dict = {}
    for a, b, style, sq in edges:
        if style in HEDGES:
            bushes.add(sq)
        else:
            by_style.setdefault(style, set()).add((a, b) if a <= b else (b, a))
    for style, segs in by_style.items():
        nbr: dict = {}
        for a, b in segs:
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
        used = set()
        for start in sorted(nbr, key=lambda c: len(nbr[c])):
            for nxt in nbr[start]:
                if (start, nxt) in used or (nxt, start) in used:
                    continue
                line = [start, nxt]
                used.add((start, nxt))
                while True:
                    tail = line[-1]
                    step = [c for c in nbr[tail] if (tail, c) not in used and (c, tail) not in used]
                    if not step:
                        break
                    used.add((tail, step[0]))
                    line.append(step[0])
                coords = [list(reversed(proj.to_latlon(x, y))) for x, y in line]
                new.append({"type": "Feature", "properties": {"barrier": "fence", "knox_style": style,
                                                              "source": SOURCE},
                            "geometry": {"type": "LineString", "coordinates": coords}})
    return kept, aside, new, bushes
