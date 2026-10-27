"""Front-anchored squaring and the stepped-front measure (KnoxMap UK:
Building_types/src/buildings/blocks.py FRONT_ANCHOR, roadgraph/mode_test.py;
LOG 2026-10-25 #3): a 45-degree front of houses squared one by one reads as
stepped, a staircase of single tiles as flush; the street-facing edge is found; a
house squared about it keeps its front on the building line where one squared
about its centre does not; a run's fronts go on one line, a house really set back
stays."""
import os
import sys
import unittest

import numpy as np
import shapely
import shapely.affinity
from shapely import affinity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)
sys.path.insert(0, os.path.join(HERE, ".."))

from buildings import blocks as B  # noqa: E402
from roadgraph import mode_test as T  # noqa: E402


class Stepped(unittest.TestCase):
    def test_a_staircase_is_flush_and_squared_houses_are_stepped(self):
        H = W = 300
        line = shapely.LineString([(50, 50), (250, 250)])              # a 45-degree street
        yy, xx = np.mgrid[0:H, 0:W] + 0.5
        off = (xx - yy) / np.sqrt(2)
        stair = (off >= 6) & (off <= 14) & (xx + yy > 100) & (xx + yy < 500)
        houses = np.zeros((H, W), bool)
        for k in range(10, 190, 7):                                      # 7 tiles apart, each upright
            houses[50 + k - 11:50 + k - 3, 50 + k + 4:50 + k + 11] = True
        a = T.side_score(T.distances(line, stair, -1))
        b = T.side_score(T.distances(line, houses, -1))
        self.assertGreater(a["flush"], 0.9)
        self.assertLess(a["stepped"], 0.05)
        self.assertGreater(b["stepped"], 0.6)
        self.assertLess(b["flush"], 0.2)


class Front(unittest.TestCase):
    def tearDown(self):
        B.ANCHOR_FRONT = False

    def test_the_street_facing_edge(self):
        street = shapely.LineString([(0, 0), (100, 0)])
        house = shapely.Polygon([(10, 4), (16, 4), (16, 14), (10, 14)])
        self.assertEqual(B.front_point(house, street), (13.0, 4.0))

    def _street(self):
        real = shapely.LineString([(0, 0), (200, 0)])
        return B.Street("Test Street", real, real, lambda s: 0.0)

    def test_squared_about_its_front_the_front_stays_on_the_line(self):
        st = self._street()
        # at 30 degrees, as a house on a 2:1 street is: squared about its centre its front
        # moves r (1 - cos 30) back, r the 6 tiles from its centre to its front
        house = affinity.rotate(shapely.Polygon([(50, 3), (56, 3), (56, 15), (50, 15)]), 30, origin=(53, 3))
        B.ANCHOR_FRONT = False
        centred = st.fit(house)[0]
        B.ANCHOR_FRONT = True
        front = st.fit(house)[0]
        self.assertGreaterEqual(B.SNAP_DEG, 30)                         # (the fit squares a 30-degree house)
        self.assertAlmostEqual(front.bounds[1], 3.0, places=1)          # its front still 3 tiles off
        self.assertGreater(abs(centred.bounds[1] - 3.0), 0.2)           # squared about its centre: moved

    def test_a_runs_fronts_go_on_one_line(self):
        st = self._street()
        B.ANCHOR_FRONT = True
        houses = [shapely.Polygon([(x, y), (x + 6, y), (x + 6, y + 10), (x, y + 10)])
                  for x, y in ((10, 3.0), (16, 4.0), (22, 3.0), (28, 9.0))]   # the last really set back
        fits = B.one_line([st.fit(h) for h in houses], [st] * len(houses))
        fronts = [f[0].bounds[1] for f in fits]
        self.assertAlmostEqual(fronts[0], fronts[1], places=6)        # one line (their median, 3.5)
        self.assertAlmostEqual(fronts[1], fronts[2], places=6)
        self.assertAlmostEqual(fronts[3], 9.0, places=6)              # set back 5.5: stays

    def test_a_runs_fronts_on_each_side_go_on_their_own_line(self):
        st = self._street()
        B.ANCHOR_FRONT = True
        north = [shapely.Polygon([(x, -y - 10), (x + 6, -y - 10), (x + 6, -y), (x, -y)])
                 for x, y in ((10, 4.0), (16, 5.0), (22, 4.0))]                 # fronts 4-5 tiles north
        south = [shapely.Polygon([(x, y), (x + 6, y), (x + 6, y + 10), (x, y + 10)])
                 for x, y in ((10, 3.0), (16, 3.0), (22, 2.0))]                 # fronts 2-3 tiles south
        fits = B.one_line([st.fit(h) for h in north + south], [st] * 6)
        n_fronts = {round(-f[0].bounds[3], 6) for f in fits[:3]}
        s_fronts = {round(f[0].bounds[1], 6) for f in fits[3:]}
        self.assertEqual((n_fronts, s_fronts), ({4.0}, {3.0}))                # each side its own median

    def test_off_the_road_given_makes_no_difference(self):
        block = [shapely.Polygon([(10, 5), (16, 5), (16, 15), (10, 15)]).buffer(0)]
        block = [shapely.affinity.rotate(block[0], 10, origin=(13, 10))]
        road = shapely.LineString([(0, 0), (40, 0)])
        B.ANCHOR_FRONT = False
        a = B.moved_rigidly(block, block, [], block[0], road)
        b = B.moved_rigidly(block, block, [], block[0], None)
        self.assertTrue(a[0][0].equals(b[0][0]))


if __name__ == "__main__":
    unittest.main()
