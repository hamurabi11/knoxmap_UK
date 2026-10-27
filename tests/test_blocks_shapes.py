"""Big buildings in blocks.py (KnoxMap UK, Building_types/src/buildings/blocks.py):
walls a few degrees off the grid are made square (squared_walls), a real
diagonal wing is kept, and a building round a courtyard is cut into lots that
leave the courtyard open (cut_into_lots) - Alton College and Amery Hill
School, 2026-10-14."""
import os
import sys
import unittest

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)

from buildings import blocks as B  # noqa: E402


def angles(poly):
    """Each wall's direction, 0-180 degrees."""
    import math
    pts = list(poly.exterior.coords)
    return [math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180 for (x0, y0), (x1, y1) in zip(pts, pts[1:])]


class SquaredWalls(unittest.TestCase):
    def test_walls_a_few_degrees_off_are_made_square(self):
        # A 60 x 20 block whose long walls run 3 and -2 degrees off the grid.
        poly = Polygon([(0, 0), (60, 3.1), (60.5, 23), (0, 21)])
        out = B.squared_walls(poly)
        for a in angles(out):
            self.assertTrue(min(a % 90, 90 - a % 90) < 1e-6, a)
        self.assertLess(abs(out.area - poly.area), 0.05 * poly.area)

    def test_a_diagonal_wing_is_kept(self):
        # An L whose wing runs at 45 degrees: its walls stay at 45.
        poly = Polygon([(0, 0), (40, 0), (40, 10), (20, 10), (40, 30), (33, 37), (10, 14), (10, 10), (0, 10)])
        out = B.squared_walls(poly)
        self.assertTrue(any(abs(a - 45) < 1e-6 or abs(a - 135) < 1e-6 for a in angles(out)))
        self.assertLess(abs(out.area - poly.area), 0.01 * poly.area)

    def test_courtyard_walls_squared_too(self):
        outer = affinity.rotate(Polygon([(0, 0), (50, 0), (50, 40), (0, 40)]), 2, origin=(0, 0))
        inner = affinity.rotate(Polygon([(15, 12), (35, 12), (35, 28), (15, 28)]), -3, origin=(25, 20))
        out = B.squared_walls(Polygon(outer.exterior.coords, [inner.exterior.coords]))
        self.assertEqual(len(out.interiors), 1)
        ring = Polygon(out.interiors[0].coords)
        for a in angles(ring):
            self.assertTrue(min(a % 90, 90 - a % 90) < 1e-6, a)

    def test_an_off_grid_courtyard_is_kept_as_drawn(self):
        # A courtyard whose walls all run at 45 degrees: not squared, not lost
        # (the review, 2026-10-14).
        outer = affinity.rotate(Polygon([(0, 0), (50, 0), (50, 46), (0, 46)]), 2, origin=(0, 0))
        diamond = [(25, 15), (33, 23), (25, 31), (17, 23)]
        out = B.squared_walls(Polygon(outer.exterior.coords, [diamond]))
        self.assertEqual(len(out.interiors), 1)
        self.assertAlmostEqual(Polygon(out.interiors[0].coords).area, Polygon(diamond).area, places=6)


class CourtyardLots(unittest.TestCase):
    def test_lots_leave_the_courtyard_open(self):
        owner = np.zeros((30, 30), np.int32)
        owner[2:28, 2:28] = 1
        owner[10:20, 10:20] = 0                      # a 10 x 10 courtyard
        pieces, final, lost, count = B.cut_into_lots(owner.copy(), 1)
        self.assertGreater(count[0], 1)              # cut round it
        for _i, _k, x0, y0, piece in pieces:
            h, w = piece.shape
            # No piece's rectangle takes in a courtyard square.
            self.assertFalse((owner[y0:y0 + h, x0:x0 + w] == 0).any())
        self.assertEqual(int((final == 1).sum()), int((owner == 1).sum()) - lost)

    def test_a_building_in_the_courtyard_is_kept(self):
        owner = np.zeros((30, 30), np.int32)
        owner[2:28, 2:28] = 1
        owner[8:22, 8:22] = 0                        # a courtyard...
        owner[12:18, 12:18] = 2                      # ...with a hall standing in it
        _pieces, final, _lost, _count = B.cut_into_lots(owner.copy(), 2)
        self.assertEqual(int((final == 2).sum()), 36)

    def test_a_small_hole_is_built_over(self):
        owner = np.zeros((20, 20), np.int32)
        owner[2:18, 2:18] = 1
        owner[9:11, 9:11] = 0                        # 4 squares: under COURT_MIN
        pieces, _final, _lost, count = B.cut_into_lots(owner.copy(), 1)
        self.assertEqual(count[0], 1)


if __name__ == "__main__":
    unittest.main()


class LongStraight(unittest.TestCase):
    """A long building at an angle to the grid (the Quaker meeting house,
    Alton, 2026-10-19): laid as one straight rectangle inside its block,
    not a staircase across a road; homes and terraces keep stepping."""

    @staticmethod
    def _range(deg=33.0):
        return affinity.rotate(Polygon([(0, 0), (24, 0), (24, 8.5), (0, 8.5)]), deg, origin=(12, 4))

    def test_a_range_is_long_and_an_l_is_not(self):
        self.assertTrue(B.is_long(self._range()))
        ell = Polygon([(0, 0), (24, 0), (24, 8), (8, 8), (8, 24), (0, 24)])
        self.assertFalse(B.is_long(ell))
        self.assertFalse(B.is_long(Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])))   # too short

    def test_homes_keep_stepping(self):
        ft = lambda **t: {"id": "1", "ft": {"properties": t}}                          # noqa: E731
        self.assertFalse(B.not_homes(ft(building="terrace")))
        self.assertFalse(B.not_homes(ft(building="yes")))                              # may be a terrace
        self.assertTrue(B.not_homes(ft(building="yes", amenity="place_of_worship")))
        self.assertFalse(B.not_homes({"id": str(B.NEW_ID + 5), "ft": {"properties": {"building": "yes", "name": "x"}}}))

    def test_side_of_a_road(self):
        from shapely.geometry import LineString, Point
        road = LineString([(0, 0), (10, 10)])
        self.assertEqual(B.side_of(road, Point(2, 6)), -B.side_of(road, Point(6, 2)))
        self.assertEqual(B.side_of(road, Point(20, 25)), 0)                            # past its end

    def test_laid_straight_off_the_road_and_apart(self):
        from shapely.geometry import LineString, box
        at = B.Placement(60, 60)
        at.owner[0:5, 0:60] = 99                                                        # a neighbour along the top
        real = affinity.translate(self._range(), 18, 20)
        placed = affinity.translate(self._range(40.0), 18, 12)                          # laid as a staircase
        road = {"name": "Church Street", "minor": False, "half": 3.0,
                "game": LineString([(0, 59), (59, 50)]), "real": LineString([(0, 59), (59, 50)])}
        got = B.straighten_long(at, real, placed, 30.0, 16.0, [(road, 4.0)])
        self.assertIsNotNone(got)
        mask, x0, y0, _d, turned = got
        self.assertEqual(sorted(mask.shape), [8, 24])
        self.assertFalse((at.owner[y0 - 1:y0 + mask.shape[0] + 1, x0 - 1:x0 + mask.shape[1] + 1] > 0).any())
        self.assertGreaterEqual(box(x0, y0, x0 + mask.shape[1], y0 + mask.shape[0]).distance(road["game"]), 4.0)


class OffTheStreet(unittest.TestCase):
    """A block left with a corner on the painted street (wider than the road
    step 5 keeps off) is moved back onto free ground; a row whose buildings
    stand apart moves only the ones on it (Church Street, Alton, 2026-10-20)."""

    @staticmethod
    def _at(blocks):
        at = B.Placement(30, 30)
        mine = []
        for bi, rects in enumerate(blocks):
            for y0, y1, x0, x1 in rects:
                mine.append({"id": str(len(mine)), "ft": {"properties": {}}, "frame": [0.0, 0.0, 0.0, 5.0, 5.0]})
                at.owner[y0:y1, x0:x1] = len(mine)
                at.block_owner[y0:y1, x0:x1] = bi + 1
        return at, mine

    def test_a_corner_on_the_pavement_is_moved_back(self):
        street = np.zeros((30, 30), bool)
        street[20:, :] = True                                                           # pavement along the bottom
        at, mine = self._at([[(12, 21, 5, 12)]])                                        # its bottom row on it
        B.street_pass(at, mine, street, np.zeros_like(street))
        self.assertFalse((street & (at.owner > 0)).any())
        self.assertEqual(int((at.owner == 1).sum()), 63)
        self.assertEqual(mine[0]["how"]["off_street"], [0, -1])
        self.assertEqual(mine[0]["frame"][2:], [-1.0, 5.0, 4.0])

    def test_not_onto_a_neighbour_nor_beside_one(self):
        street = np.zeros((30, 30), bool)
        street[20:, :] = True
        at, mine = self._at([[(12, 21, 5, 12)], [(5, 11, 5, 12)]])                      # a neighbour a tile behind
        B.street_pass(at, mine, street, np.zeros_like(street))
        self.assertTrue((street & (at.owner == 1)).any())                               # no room: left as it was
        self.assertNotIn("how", mine[0])

    def test_a_row_moves_piece_by_piece(self):
        street = np.zeros((30, 30), bool)
        street[20:, 0:6] = True                                                          # only under the first
        at, mine = self._at([[(14, 21, 2, 6), (14, 20, 8, 12), (14, 20, 14, 18)]])
        B.street_pass(at, mine, street, np.zeros_like(street))
        self.assertFalse((street & (at.owner > 0)).any())
        self.assertIn("off_street", mine[0]["how"])
        self.assertNotIn("how", mine[1])                                                # its frontage kept
        self.assertNotIn("how", mine[2])


class Rows(unittest.TestCase):
    """Homes OSM draws one by one (Normandy Street, Alton): a row near the grid
    is one building; a row at 45 degrees is left to step house by house."""

    @staticmethod
    def _house(k, deg=0.0, osm=None):
        from shapely.ops import transform as _t
        p = affinity.rotate(Polygon([(0, 5 * k), (8, 5 * k), (8, 5 * k + 5), (0, 5 * k + 5)]), deg, origin=(4, 10))
        return {"id": osm or str(100 + k), "ft": {"properties": {"building": "house"}}, "bng": p, "px": p, "moved": p}

    def test_a_row_near_the_grid_is_joined(self):
        mine, n, _h = B.join_rows([self._house(k) for k in range(4)], None)
        self.assertEqual((n, len(mine)), (1, 1))
        self.assertEqual(mine[0]["ft"]["properties"]["row_of"], ["100", "101", "102", "103"])
        self.assertAlmostEqual(mine[0]["px"].area, 160.0, delta=1.0)

    def test_a_row_at_45_degrees_keeps_stepping(self):
        mine, n, _h = B.join_rows([self._house(k, 45.0) for k in range(4)], None)
        self.assertEqual((n, len(mine)), (0, 4))

    def test_a_shop_beside_a_house_is_not_joined(self):
        shop = self._house(1)
        shop["ft"] = {"properties": {"building": "retail", "shop": "bakery"}}
        mine, n, _h = B.join_rows([self._house(0), shop], None)
        self.assertEqual((n, len(mine)), (0, 2))
