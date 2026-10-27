"""Garden fronts on the street's edge (KnoxMap UK, Building_types/src/ground/plots2.py):
a strip of one plot (or of land no plot got) along the street in front of
another plot goes to the plot behind (street_strips_to_behind). User, 2026-10-14."""
import os
import sys
import unittest

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)

from ground import plots2  # noqa: E402

STREET = box(0, 0, 40, 4)                      # a street along the north of the block


def run(shapes, land, houses):
    shapes = list(shapes)
    moved = plots2.street_strips_to_behind(shapes, land, STREET, unary_union(houses), plots2.Boxes(shapes))
    return moved, shapes


class StripsAlongTheStreet(unittest.TestCase):
    def test_a_plots_strip_in_front_of_another_goes_to_it(self):
        behind = box(0, 7, 10, 20)                                   # its house at 3-7, 10-14
        arm = unary_union([box(0, 4, 10, 7), box(10, 4, 20, 20)])    # the next plot, with a strip in front of it
        moved, shapes = run([arm, behind], box(0, 4, 40, 20), [box(3, 10, 7, 14), box(13, 10, 17, 14)])
        self.assertEqual(moved, 1)
        self.assertLess(shapes[1].distance(STREET), 1e-6)            # the plot behind now reaches the street
        self.assertLess(shapes[0].distance(STREET), 1e-6)            # and the arm's own plot still does
        self.assertLess(shapes[0].intersection(box(0, 4, 10, 7)).area, 1e-6)   # no lip left behind (the review)

    def test_land_no_plot_got_in_front_of_a_plot_goes_to_it(self):
        behind = box(0, 6, 10, 20)
        moved, shapes = run([behind], box(0, 4, 10, 20), [box(3, 10, 7, 14)])
        self.assertEqual(moved, 1)
        self.assertLess(shapes[0].distance(STREET), 1e-6)

    def test_a_wedge_of_land_no_plot_got_gives_its_band(self):
        # Open ground widening from 2 to 8 tiles between the street and a plot:
        # the band within STRIP_TILES of the street goes to the plot.
        behind = Polygon([(0, 6), (20, 12), (20, 20), (0, 20)])
        moved, shapes = run([behind], box(0, 4, 20, 20), [box(8, 14, 12, 18)])
        self.assertEqual(moved, 1)
        self.assertLess(shapes[0].distance(STREET), 1e-6)
        self.assertGreater(shapes[0].intersection(box(0, 4, 20, 8)).area, 60)

    def test_never_a_plots_only_frontage(self):
        behind = box(0, 7, 10, 20)
        arm = unary_union([box(0, 4, 10, 7), box(10, 7, 20, 20)])    # its only touch on the street is the strip
        moved, shapes = run([arm, behind], box(0, 4, 40, 20), [box(3, 10, 7, 14), box(13, 10, 17, 14)])
        self.assertEqual(moved, 0)

    def test_not_given_sideways(self):
        # A strip with open ground behind and a plot only at its end: stays.
        strip = box(0, 4, 10, 7)
        side = box(10, 4, 20, 20)
        moved, _shapes = run([strip.union(box(0, 7, 2, 9)), side], box(0, 4, 40, 20), [box(13, 10, 17, 14)])
        self.assertEqual(moved, 0)

    def test_a_short_lip_is_not_handed_sideways_and_back(self):
        # A lip 3 long and 3.5 deep off a plot's side, open ground behind it
        # and a plot beside it: the review found it given to the neighbour
        # (sharing 3.5 >= half of 3) and back, leaving a sliver.
        a = unary_union([box(0, 4, 3, 7.5), box(-20, 4, 0, 20)])
        b = box(3, 4, 20, 20)
        moved, shapes = run([a, b], box(-20, 4, 40, 20), [box(-15, 10, -11, 14), box(10, 10, 14, 14)])
        self.assertEqual(moved, 0)
        self.assertTrue(shapes[0].equals(a) and shapes[1].equals(b))

    def test_a_deep_front_garden_stays(self):
        # A front garden 6 deep is no strip.
        behind = box(0, 10, 10, 20)
        front = unary_union([box(0, 4, 10, 10), box(10, 4, 20, 20)])
        moved, _shapes = run([front, behind], box(0, 4, 40, 20), [box(3, 12, 7, 16), box(13, 12, 17, 16)])
        self.assertEqual(moved, 0)


if __name__ == "__main__":
    unittest.main()
