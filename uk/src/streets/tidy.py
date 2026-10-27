"""Road surface tidied after the 45-degree redraw, before the kerbs are laid.

Called by kerbs.py. Two things left the town's tarmac looking broken in the
game (LOG 2026-10-10, the north of the town):

- squares of tarmac touching only at a corner: Generate draws a narrow lane
  on the diagonal one square a row, and a car park beside a lane often meets
  it at a corner. In the game that is a dotted staircase with grass between,
  not a road. The square between is made tarmac too (never a building's or
  water), so the two meet edge to edge;
- the ends of service lanes left leading nowhere: a building placed over the
  middle of a lane (blocks.py takes the lane off under it) left the far end
  standing alone in a garden. A piece of service-lane tarmac joined to no
  other road, smaller than STRAY_MAX and clear of the map's edge (a road
  leaving the map is joined beyond it), goes back to the ground beside it.
  Car parks and roads are never taken off, nor on a road-graph map a lane
  the graph drew (`drawn`: a dead-end lane to garages is real; taken off,
  its plots still fenced it as street - a grass alley, 2026-10-13);
- a piece of road, lane or car park stopping just short of the network
  (within JOIN_MAX tiles of the road or its pavement; OSM drew it so, or the
  straightening moved the road away) gets a straight lane two tiles wide to
  it, where no building or water is in the way.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

STRAY_MAX = 200          # a lane piece joined to nothing, smaller than this, is taken off
EDGE = 2                 # ...unless it comes this near the map's edge
JOIN_MAX = 8             # a road piece this near the network gets a lane to it


def roads(ground: np.ndarray, road_colours, lane_colour, pave_colour, is_colour,
          blocked: np.ndarray | None, water_colour, drawn: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """The ground with corner joins filled and stray lane ends taken off;
    and how many of each."""
    out = ground.copy()
    H, W = out.shape[:2]
    keep_off = is_colour(out, (water_colour,))
    if blocked is not None:
        keep_off |= blocked[:H, :W]

    # --- corner joins -------------------------------------------------------------------------
    filled = 0
    for _ in range(3):                      # a fill can leave a new corner beside it
        road = is_colour(out, road_colours)
        added = 0
        # Each pair of squares touching at a corner: (dy, dx) to the other one.
        for dy, dx in ((1, 1), (1, -1)):
            # Road here and at the corner (y + dy, x + dx), neither square
            # between them road.
            corner = road & _shift(road, dy, dx) & ~_shift(road, 0, dx) & ~_shift(road, dy, 0)
            if not corner.any():
                continue
            ys, xs = np.nonzero(corner)
            for y, x in zip(ys, xs):
                for ty, tx in ((y, x + dx), (y + dy, x)):
                    if 0 <= ty < H and 0 <= tx < W and not keep_off[ty, tx] and not road[ty, tx]:
                        out[ty, tx] = out[y, x]
                        road[ty, tx] = True
                        added += 1
                        break
        filled += added
        if not added:
            break

    # --- stray lane ends ----------------------------------------------------------------------
    road = is_colour(out, road_colours)
    lab, n = ndimage.label(road)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes)) if n else 0
    lane = is_colour(out, (lane_colour,))
    only_lane = np.bincount(lab[road & ~lane], minlength=n + 1) == 0
    edge = np.zeros((H, W), bool)
    edge[:EDGE, :] = edge[-EDGE:, :] = True
    edge[:, :EDGE] = edge[:, -EDGE:] = True
    at_edge = np.bincount(lab[edge & road], minlength=n + 1) > 0
    stray = np.zeros(n + 1, bool)
    stray[1:] = (sizes[1:] < STRAY_MAX) & only_lane[1:] & ~at_edge[1:]
    stray[main] = False
    if drawn is not None:                   # mostly the graph's own road: kept
        ours = np.bincount(lab[road & drawn[:H, :W]], minlength=n + 1)
        stray &= ours * 2 < np.maximum(sizes, 1)
    gone = stray[lab] & road
    removed = 0
    if gone.any():
        # Each square taken off takes the ground of the nearest square that is
        # not road.
        _d, (iy, ix) = ndimage.distance_transform_edt(road, return_indices=True)
        out[gone] = out[iy[gone], ix[gone]]
        removed = int(stray.sum())
    # --- pieces just short of the network -----------------------------------------------------
    # A lane or a car park stopping a few tiles short of the road (OSM drew
    # it so, or the road moved away from it in the straightening) gets a
    # straight lane to the nearest road, where nothing stands in the way.
    road = is_colour(out, road_colours)
    lab, n = ndimage.label(road)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes)) if n else 0
    at_edge = np.bincount(lab[edge & road], minlength=n + 1) > 0
    # The network a piece joins: its tarmac, and the pavement beside it (a
    # drive meets the road at the pavement, over a dropped kerb).
    network = lab == main
    network |= is_colour(out, (pave_colour,)) & ndimage.binary_dilation(network, iterations=4)
    dist, (iy, ix) = ndimage.distance_transform_edt(~network, return_indices=True)
    joined = 0
    for k in range(1, n + 1):
        if k == main or at_edge[k] or sizes[k] < 10:
            continue
        here = lab == k
        d = np.where(here, dist, np.inf)
        y0, x0 = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[y0, x0] > JOIN_MAX:
            continue
        y1, x1 = int(iy[y0, x0]), int(ix[y0, x0])
        steps = int(max(abs(y1 - y0), abs(x1 - x0)))
        line = {(int(round(y0 + (y1 - y0) * t / steps)), int(round(x0 + (x1 - x0) * t / steps)))
                for t in range(steps + 1)}
        # Two tiles wide so a diagonal link meets edge to edge.
        line |= {(y, x + 1) for y, x in line} if abs(y1 - y0) >= abs(x1 - x0) else {(y + 1, x) for y, x in line}
        line = {(y, x) for y, x in line if 0 <= y < H and 0 <= x < W}
        if any(keep_off[y, x] for y, x in line):
            continue
        for y, x in line:
            if not road[y, x]:
                out[y, x] = lane_colour
        joined += 1
    return out, {"corners joined": filled, "stray lane ends taken off": removed,
                 "tiles taken off": int(gone.sum()), "pieces joined to the road": joined}


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """a moved so out[y, x] = a[y + dy, x + dx] (False beyond the edge)."""
    out = np.zeros_like(a)
    H, W = a.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[ys, xs] = a[yd, xd]
    return out
