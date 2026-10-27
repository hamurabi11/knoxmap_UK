"""Plot lines (KnoxMap UK, Building_types/src/ground/plots2.py) and how Build lays
them (knoxbuild/fences.py): lattice runs keep to the side asked and to their band,
a stretch on the lattice is left as it is, nothing is laid across the street, and
an exact 45-degree or 2:1 segment handed to Build gets a picture on every square."""
import math
import os
import sys
import unittest

from shapely.geometry import LineString, box

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)
sys.path.insert(0, os.path.join(ukdirs.UK_SRC, "ground"))

from ground import plots2  # noqa: E402
from knoxbuild import fences as F  # noqa: E402


def side_offsets(a, b, pts):
    """Each point's offset from the line a-b along the world's minor axis."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    out = []
    for x, y in pts:
        if abs(dx) >= abs(dy):
            t = (x - a[0]) / dx
            out.append(y - (a[1] + dy * t))
        else:
            t = (y - a[1]) / dy
            out.append(x - (a[0] + dx * t))
    return out


class LatticeRunTests(unittest.TestCase):
    def test_side_is_the_world_side_whichever_way_the_line_runs(self):
        # The review found +1 drifting to -y for lines running the other way.
        for a, b in (((0, 0), (12, 4)), ((0, 0), (12, -4)), ((12, 4), (0, 0)), ((0, 0), (-12, 4))):
            for side in (1, -1):
                pts = plots2.lattice_runs(a, b, side)[1:-1]
                offs = side_offsets(a, b, pts)
                self.assertTrue(all(o * side >= -1e-6 for o in offs), (a, b, side, offs))

    def test_drift_stays_in_its_band(self):
        for side in (1, -1, 0):
            pts = plots2.lattice_runs((0, 0), (30, 12.4), side)[1:-1]
            offs = side_offsets((0, 0), (30, 12.4), pts)
            self.assertLessEqual(max(abs(o) for o in offs), plots2.LATTICE_DRIFT + 0.5)

    def test_runs_start_on_a_tile_corner(self):
        pts = plots2.lattice_runs((3.4, 7.6), (20.4, 13.1), 0)
        self.assertEqual(pts[0], (3, 8))
        for p, q in zip(pts[1:-2], pts[2:-1]):
            self.assertTrue(all(abs(v - round(v)) < 1e-9 for v in (*p, *q)))


class OnLatticeLineTests(unittest.TestCase):
    def test_a_2_to_1_stretch_is_left_as_it_is(self):
        coords = [(0.0, 0.0), (20.0, 10.0)]
        self.assertEqual(plots2.on_lattice_line(coords, True, box(0, 20, 30, 30)), [(0.0, 0.0), (20.0, 10.0)])

    def test_never_across_the_street(self):
        street = box(-5, 3, 40, 20)             # the garden is below y = 3
        coords = [(0.0, 0.0), (30.0, 2.9)]       # a front just inside it, off the lattice
        out = plots2.on_lattice_line(coords, True, street)
        self.assertFalse(LineString(out).intersects(street.buffer(-0.05)))


class SmoothPathTests(unittest.TestCase):
    def test_a_path_is_long_runs_on_the_16_directions_near_its_line(self):
        raw = [(315, 399), (327, 394), (366, 381), (407, 367)]
        out = plots2.smooth_path(raw)
        line = LineString(raw)
        dirs = {(v[0] / math.gcd(*v), v[1] / math.gcd(*v)) for v in plots2.DIRECTIONS16}
        for p, q in zip(out[:-2], out[1:-1]):
            d = (q[0] - p[0], q[1] - p[1])
            g = math.gcd(int(d[0]), int(d[1]))
            self.assertIn((d[0] / g, d[1] / g), dirs)
        self.assertLessEqual(len(out), 6)                       # a few long runs, not a staircase
        from shapely.geometry import Point
        self.assertLessEqual(max(line.distance(Point(p)) for p in out), plots2.PATH_DRIFT / 2 + 0.5)

    def test_runs_keep_out_of_houses_the_line_misses(self):
        raw = [(0, 0), (40, 13)]
        house = box(18, 2, 24, 5)                             # just below the line's middle
        self.assertEqual(LineString(raw).intersection(house).length, 0)
        out = plots2.smooth_path(raw, house)
        self.assertLess(LineString(out).intersection(house).length, 0.5)


class GivenLineTests(unittest.TestCase):
    def test_an_exact_45_degree_segment_gets_a_picture_on_every_square(self):
        a, b = (10, 30), (16, 24)
        walk = F._grid_path([a, b])
        edges, keys = {}, []
        for (x1, y1), (x2, y2) in zip(walk, walk[1:]):
            key = ((min(x1, x2), y1), "N") if y1 == y2 else ((x1, min(y1, y2)), "W")
            edges.setdefault(key[0], {})[key[1]] = "hedge_low"
            keys.append(key)
        given = [("hedge_low", a, b, keys, [], [(0, 0)])]
        pictures, covered = F._diagonals(edges, 60, 60, given=given)
        squares = {sq for sq, _p0, _p1 in F.line_squares(a, b)}
        self.assertEqual(set(pictures), squares)
        self.assertEqual(covered, set(keys))


if __name__ == "__main__":
    unittest.main()
