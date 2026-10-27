"""INSPIRE evidence and the V check (KnoxMap UK: Building_types/src/ground/inspire.py,
Building_types/src/checks/v_check.py): a title cut in two by a strip keeps both
parts; a road ending inside a title is its drive, a road split at a junction
there is not; the reach rays; V shapes found on any lattice line, square notches,
steps and a small plot's point not."""
import os
import sys
import unittest

from shapely.geometry import LineString, box

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)
sys.path.insert(0, os.path.join(ukdirs.UK_SRC, "checks"))
sys.path.insert(0, os.path.join(HERE, ".."))

from ground import inspire as I  # noqa: E402
import v_check  # noqa: E402


class Clean(unittest.TestCase):
    def test_a_strip_across_a_title_leaves_both_parts(self):
        polys = [box(0, 0, 100, 40), box(48, -1, 52, 41)]          # a field and an access strip across it
        I.clean(polys)
        self.assertAlmostEqual(polys[0].area + polys[1].area, 100 * 40 + 4 * 2, delta=1.0)


class Drives(unittest.TestCase):
    def _nodes(self, ways):
        from collections import Counter
        return Counter(I._key(c) for _t, ln in ways for c in ln.coords)

    def test_a_road_ending_inside_is_a_drive(self):
        title = box(0, 0, 20, 20)
        ways = [({"highway": "service"}, LineString([(10, -10), (10, 12)]))]
        self.assertEqual(len(I.drives_in(title, ways, self._nodes(ways))), 1)

    def test_a_road_split_at_a_junction_inside_is_not(self):
        title = box(0, 0, 20, 20)
        ways = [({"highway": "residential"}, LineString([(10, -10), (10, 10)])),
                ({"highway": "residential"}, LineString([(10, 10), (10, 30)])),
                ({"highway": "residential"}, LineString([(10, 10), (30, 10)]))]
        self.assertEqual(I.drives_in(title, ways, self._nodes(ways)), [])

    def test_ray(self):
        self.assertAlmostEqual(I._ray(box(0, 0, 10, 10), (2, 5), (1, 0)), 8.0)
        self.assertEqual(I._ray(box(0, 0, 10, 10), (20, 5), (1, 0)), 0.0)


class Vees(unittest.TestCase):
    def test_a_dip_in_a_straight_side_is_a_v(self):
        self.assertEqual(len(v_check.vees([(0, 0), (5, 0), (7.5, 2.5), (10, 0), (20, 0)])), 1)

    def test_on_a_45_degree_line_and_with_a_flat_bottom(self):
        self.assertEqual(len(v_check.vees([(-5, -5), (0, 0), (5, 5), (6, 8), (9, 9), (15, 15)])), 1)
        self.assertEqual(len(v_check.vees([(0, 0), (5, 0), (6, 2), (8, 2), (10, 0), (20, 0)])), 1)

    def test_not_a_step_a_square_notch_or_a_plots_point(self):
        self.assertEqual(v_check.vees([(0, 0), (5, 0), (5, 2), (10, 2), (20, 2)]), [])
        self.assertEqual(v_check.vees([(0, 0), (5, 0), (5, 2), (8, 2), (8, 0), (20, 0)]), [])
        self.assertEqual(v_check.vees([(0, 0), (5, 3), (10, 0)]), [])

    def test_depth_and_width_each(self):
        self.assertEqual(v_check.vees([(0, 0), (5, 0), (7, 5), (9, 0), (20, 0)]), [])        # too deep
        self.assertEqual(v_check.vees([(0, 0), (5, 0), (11, 2), (17, 0), (30, 0)]), [])     # too wide


if __name__ == "__main__":
    unittest.main()
