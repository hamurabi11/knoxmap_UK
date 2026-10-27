"""Minor lanes (KnoxMap UK: Building_types/src/roadgraph/lanes.py, snap.HARD; LOG
2026-10-25 #7): a lane with a building on its line goes round it when one is in the
way and there is room, and has no path when the building fills the band; a driveway
is one only when every way of it is; one run of roof with open ends is a passage,
two runs or a covered end are not; the network's parts."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)
sys.path.insert(0, os.path.join(HERE, ".."))

from roadgraph import lanes, snap  # noqa: E402

H = W = 200


def route(s, g, hard, width=3):
    return snap._route([s, g], s, g, 0, 0, 2, snap.min_run(width), {}, 0, road_width=width, hard=hard)


class Hard(unittest.TestCase):
    def test_a_lane_goes_round_a_building_on_its_line(self):
        s, g = (20, 100), (180, 100)
        b = np.zeros((H, W), bool)
        b[97:104, 90:110] = True                     # across the real line, room beside it
        path = route(s, g, b)
        self.assertIsNotNone(path)
        under = snap.occ_field(3, b)
        self.assertEqual(sum(under[y, x] > 0.5 / 3 for x, y in path[3:-3]), 0)

    def test_no_path_where_the_building_fills_the_band(self):
        s, g = (20, 100), (180, 100)
        b = np.zeros((H, W), bool)
        b[60:140, 90:110] = True                     # wider than the band either side
        self.assertIsNone(route(s, g, b))


class Kinds(unittest.TestCase):
    def test_a_driveway_only_when_every_way_is(self):
        L = type("L", (), {})
        a, b = L(), L()
        a.ways, b.ways = [1, 2], [1, 3]
        tags = {1: {"service": "driveway"}, 2: {"service": "driveway"}, 3: {"service": "alley"}}
        self.assertTrue(lanes.is_driveway(a, tags))
        self.assertFalse(lanes.is_driveway(b, tags))

    def test_one_roof_with_open_ends_is_a_passage(self):
        built = np.zeros((50, 100), bool)
        built[20:30, 40:50] = True
        runs, open_ends = lanes.roof_runs([(10, 25), (90, 25)], built)
        self.assertEqual((len(runs), open_ends), (1, True))
        built[20:30, 70:80] = True                   # a second building on the line
        self.assertEqual(len(lanes.roof_runs([(10, 25), (90, 25)], built)[0]), 2)
        built[20:30, 5:15] = True                    # and its start under a roof
        self.assertFalse(lanes.roof_runs([(10, 25), (90, 25)], built)[1])

    def test_a_passage_is_capped_by_its_building_and_the_cap(self):
        import shapely
        built = np.zeros((50, 100), bool)
        built[20:30, 40:50] = True
        house = shapely.box(40, 20, 50, 30)                      # 10 tiles deep along the lane
        ok, info = lanes.passage_check([(10, 25), (90, 25)], built, [house])
        self.assertTrue(ok, info)
        self.assertAlmostEqual(info["under roof"], 10.0, delta=1.0)
        self.assertFalse(lanes.passage_check([(10, 25), (90, 25)], built, [house], cap=8.0)[0])
        narrow = shapely.box(40, 20, 44, 30)                     # the outline shallower than the roof
        self.assertFalse(lanes.passage_check([(10, 25), (90, 25)], built, [narrow])[0])

    def test_a_driveways_paving_stops_at_the_first_building(self):
        import shapely
        garage = shapely.box(30, 8, 40, 14)
        paving = lanes.drive_paving([([(0, 10), (60, 10)], 0)], [garage], 20, 70)
        self.assertTrue(paving[10, 20])
        self.assertFalse(paving[10, 35])                         # inside the garage
        self.assertFalse(paving[10, 50])                         # beyond it
        back = lanes.drive_paving([([(0, 10), (60, 10)], -1)], [garage], 20, 70)
        self.assertTrue(back[10, 50] and not back[10, 20])       # from its other end

    def test_finish_takes_out_the_dropped_and_lets_a_passage_through(self):
        b = np.zeros((H, W), bool)
        res = {"dropped": [1], "passages": [2], "hard": {0: b, 2: b, 3: b}}
        keep, at, hard = lanes.finish(res, 4)
        self.assertEqual((keep, at), ([0, 2, 3], {0: 0, 2: 1, 3: 2}))
        self.assertEqual(sorted(hard), [0, 2])                   # lane 3 now 2; the passage (2 -> 1) free

    def test_a_passages_band_is_cleared_from_the_footprints(self):
        fp = np.ones((20, 40), bool)
        out = lanes.clear_footprints(fp, [[(0, 10), (39, 10)]], width=3)
        self.assertFalse(out[10, 20])
        self.assertTrue(out[2, 20] and fp[10, 20])               # the rest kept; the input untouched

    def test_the_networks_parts(self):
        L = type("L", (), {})
        links = []
        for u, v in ((1, 2), (2, 3), (7, 8)):
            l = L()
            l.u, l.v = u, v
            links.append(l)
        parts = sorted(map(sorted, lanes.components(links)))
        self.assertEqual(parts, [[0, 1], [2]])


if __name__ == "__main__":
    unittest.main()
