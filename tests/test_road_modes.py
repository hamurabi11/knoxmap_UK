"""The road graph's modes (KnoxMap UK: Building_types/src/roadgraph/modes.py, snap.py
COST and JOG; LOG 2026-10-25 #2): the frontage-density field; a sparse street keeps
its 2:1; a dense one is drawn on the grid; a terraced street's jog moves to the gap in
its frontage, or stays put at a plot boundary; a vanilla shape crossing a dense
frontage at 2:1 is refused; a mode's constants are put back after it."""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)
sys.path.insert(0, os.path.join(HERE, ".."))

from roadgraph import modes, raster, snap  # noqa: E402

H = W = 200


def route(s, g, pts, h=None, width=4):
    return snap._route(pts, s, g, h, h, 2, snap.min_run(width), {}, 0, road_width=width)


def share_2to1(path) -> float:
    vs = raster.vertices(path)
    seg = [(math.hypot(b[0] - a[0], b[1] - a[1]), a, b) for a, b in zip(vs, vs[1:])]
    two = sum(L for L, a, b in seg if a[0] != b[0] and a[1] != b[1] and abs(b[0] - a[0]) != abs(b[1] - a[1]))
    return two / sum(L for L, _a, _b in seg)


class Modes(unittest.TestCase):
    def tearDown(self):
        modes.reset()

    def test_density_is_high_by_a_frontage_and_nothing_far_off(self):
        edge = np.zeros((H, W), bool)
        edge[100, 20:180] = True                     # a long row of house fronts
        d = modes.density(edge)
        self.assertGreater(d[100, 100], 0.5)
        self.assertLess(d[100 - 40, 100], 0.01)
        self.assertLessEqual(d.max(), 1.0)

    def test_a_sparse_street_keeps_its_2to1(self):
        s, g = (20, 50), (140, 110)                  # a street at 2:1, nothing beside it
        snap.COST = np.zeros((H, W))
        path = route(s, g, [s, g], h=1)
        self.assertGreater(share_2to1(path), 0.9)

    def test_a_dense_street_is_drawn_on_the_grid(self):
        s, g = (20, 50), (140, 110)
        free = route(s, g, [s, g])
        snap.COST = np.ones((H, W))                  # lined with buildings all along
        dense = route(s, g, [s, g])
        self.assertLess(share_2to1(dense), share_2to1(free) - 0.5)

    def test_a_terraced_streets_jog_moves_to_the_gap(self):
        # The real street steps 10 tiles south at x 40; terraces both sides all along but
        # for a gap at x 80-105 (no building within JOG_GAP of the road's edge there).
        s, g = (10, 50), (130, 60)
        line = [s, (40, 50), (40, 60), g]
        snap.COST = np.ones((H, W))
        dist = np.full((H, W), 3.0)
        dist[:, 80:106] = 20.0
        snap.JOG = {"dist": dist, "plots": None}
        path = route(s, g, line, h=0)
        inner = raster.vertices(path)[1:-1]
        self.assertTrue(inner)
        for x, _y in inner:
            self.assertTrue(78 <= x <= 107, f"a turn at x {x}, not in the gap: {inner}")
        snap.JOG = None                              # without the rule it jogs near the real step
        free = raster.vertices(route(s, g, line, h=0))[1:-1]
        self.assertTrue(any(x < 70 for x, _y in free), free)

    def test_a_turn_at_a_plot_boundary_is_allowed_mid_frontage(self):
        snap.COST = np.ones((H, W))
        plots = np.zeros((H, W), np.int32)
        plots[:, :100] = 1                           # two titles, their boundary at x 100
        plots[:, 100:] = 2
        snap.JOG = {"dist": np.full((H, W), 3.0), "plots": plots}
        far = ((-500, -500), (500, 500))
        self.assertTrue(snap.turn_ok(100, 50, 0, 4, *far))
        self.assertFalse(snap.turn_ok(60, 50, 0, 4, *far))
        self.assertTrue(snap.turn_ok(60, 50, 0, 4, (55, 50), (500, 500)))     # at a junction

    def test_a_main_road_may_turn_at_its_inner_junctions(self):
        snap.COST = np.ones((H, W))
        snap.JOG = {"dist": np.full((H, W), 3.0), "plots": None}
        far = ((-500, -500), (500, 500))
        self.assertFalse(snap.turn_ok(60, 50, 0, 4, *far))
        self.assertTrue(snap.turn_ok(60, 50, 0, 4, *far, stops=[(62, 50)]))   # a side road joins here

    def test_a_vanilla_shape_at_2to1_through_a_dense_frontage_is_refused(self):
        shape = [(0, 0), (2, 1), (4, 2), (6, 3), (8, 4)]
        snap.COST = np.zeros((H, W))
        self.assertFalse(snap.jog_breaks(shape, 4))
        snap.COST = np.ones((H, W))
        self.assertTrue(snap.jog_breaks(shape, 4))

    def test_occupancy_steers_a_road_off_a_building_on_its_line(self):
        s, g = (20, 100), (180, 100)
        fp = np.zeros((H, W), bool)
        fp[97:106, 85:115] = True                    # a building standing across the real line
        under = lambda path: snap.occ_field(4)[[y for _x, y in path], [x for x, _y in path]].sum()
        snap.OCC = fp
        straight = route(s, g, [s, g], h=0)          # OCC_WEIGHT 0: as before, through it
        snap.OCC_WEIGHT = 4.0
        try:
            around = route(s, g, [s, g], h=0)
        finally:
            snap.OCC_WEIGHT = 0.0
        self.assertGreater(under(straight), 0)
        self.assertLess(under(around), under(straight) / 4)

    def test_cost_front_routes_as_cost_low_occ1_and_anchors_the_fronts(self):
        f = {"density": np.zeros((H, W)), "building_dist": np.full((H, W), 50.0), "plots": None,
             "footprints": np.zeros((H, W), bool)}
        info = modes.prepare("cost-front", "", "", H, W, f)
        self.assertEqual((info["mode"], info["preset"], info.get("front_anchor")), ("cost", "cost-front", True))
        self.assertEqual(snap.OCC_WEIGHT, 1.0)
        self.assertEqual(snap.DIR_PENALTY, modes.LOW)
        modes.reset()
        self.assertNotIn("front_anchor", modes.prepare("cost-low-occ1", "", "", H, W, f))

    def test_a_stale_road_graph_file_is_no_passages(self):
        import json
        import tempfile
        from regional.passages import made_passages
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "m_roadgraph.json"), "w") as f:
                json.dump({"passages": [5, 6]}, f)
            self.assertEqual(made_passages(d, {}), set())                      # no road graph named
            self.assertEqual(made_passages(d, {"road_graph": "m_roadgraph.json"}), {5, 6})

    def test_a_modes_constants_are_put_back(self):
        before = snap.JOG_DENSITY
        edge = np.zeros((H, W), bool)
        f = {"density": modes.density(edge), "building_dist": np.full((H, W), 50.0), "plots": None}
        info = modes.prepare("cost", "", "", H, W, f, {"JOG_DENSITY": 0.9})
        self.assertEqual(info["mode"], "cost")
        self.assertEqual(snap.JOG_DENSITY, 0.9)
        self.assertIsNotNone(snap.COST)
        modes.reset()
        self.assertEqual(snap.JOG_DENSITY, before)
        self.assertIsNone(snap.COST)
        self.assertIsNone(snap.JOG)


if __name__ == "__main__":
    unittest.main()
