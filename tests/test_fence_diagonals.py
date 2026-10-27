"""Angled boundaries (KnoxMap UK): staircases of fence edges found and drawn
as straight lines, each picture on its own line."""
import unittest

from shapely.geometry import LineString, Point

from knoxbuild import fences as F

STYLE = "hedge_low"


def staircase(start, moves, skip=()):
    """Fence edges walking `moves` ((1, 0) east, (0, -1) north...) from
    `start`; steps whose index is in `skip` left out (a path's gap)."""
    edges, v = {}, start
    for k, (dx, dy) in enumerate(moves):
        w = (v[0] + dx, v[1] + dy)
        if k not in skip:
            if dy == 0:
                edges.setdefault((min(v[0], w[0]), v[1]), {})["N"] = STYLE
            else:
                edges.setdefault((v[0], min(v[1], w[1])), {})["W"] = STYLE
        v = w
    return edges


def ratio(line):
    _style, a, b = line[:3]
    dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
    return max(dx, dy) / min(dx, dy)


class StaircaseRunTests(unittest.TestCase):
    def test_45_degree_and_2_to_1_runs(self):
        self.assertEqual(F._staircase_runs([(1, 0), (0, -1)] * 4), [(0, 8)])
        self.assertEqual(F._staircase_runs([(1, 0), (1, 0), (0, -1)] * 3), [(0, 9)])

    def test_straight_run_is_no_staircase(self):
        self.assertEqual(F._staircase_runs([(1, 0)] * 8), [])

    def test_run_trimmed_to_whole_repeats(self):
        # A run ending part-way through its pattern gave a line off the
        # 45-degree and 2:1 angles.
        (i, j), = F._staircase_runs([(1, 0), (1, 0), (0, -1)] * 3 + [(1, 0)])
        self.assertEqual((j - i) % 3, 0)


class DiagonalLineTests(unittest.TestCase):
    def test_lines_are_45_degree_or_2_to_1(self):
        for moves, want in (([(1, 0), (0, -1)] * 5, 1.0), ([(1, 0), (1, 0), (0, -1)] * 4, 2.0)):
            lines = F.diagonal_lines(staircase((10, 20), moves))
            self.assertEqual(len(lines), 1)
            self.assertAlmostEqual(ratio(lines[0]), want)

    def test_gap_in_an_angled_front_is_bridged_and_left_open(self):
        edges = staircase((10, 20), [(1, 0), (0, -1)] * 6, skip={4, 5})
        (line,) = F.diagonal_lines(edges)
        _style, _a, _b, keys, gaps, real = line
        self.assertTrue(gaps)                                  # the gap is part of the line...
        self.assertEqual(len(keys), sum(len(s) for s in edges.values()))   # ...but no wall

    def test_gap_in_a_straight_front_is_not_bridged(self):
        # Two halves of a straight front either side of a drive were joined by
        # a zig-zag, drawn as an angled wall across the drive.
        edges = staircase((10, 20), [(1, 0)] * 6)
        edges.update(staircase((18, 20), [(1, 0)] * 6))
        self.assertEqual(F.diagonal_lines(edges), [])

    def test_ring_of_one_style_is_found(self):
        # A plot fenced all round: every vertex has two links.
        edges = {}
        path = [(0, 0), (10, 0), (10, 10), (6, 10)]
        corner = [(6, 10)]
        for k in range(6):                                     # a 45-degree corner cut
            x, y = corner[-1]
            corner.append((x, y - 1) if k % 2 == 0 else (x - 1, y))
        path += corner[1:] + [(0, 0)]
        v = path[0]
        for w in path[1:]:
            while v != w:
                n = (v[0] + (w[0] > v[0]) - (w[0] < v[0]), v[1]) if v[0] != w[0] else (v[0], v[1] + (w[1] > v[1]) - (w[1] < v[1]))
                if v[1] == n[1]:
                    edges.setdefault((min(v[0], n[0]), v[1]), {})["N"] = STYLE
                else:
                    edges.setdefault((v[0], min(v[1], n[1])), {})["W"] = STYLE
                v = n
        self.assertEqual(len(F.diagonal_lines(edges)), 1)


class LatticeTests(unittest.TestCase):
    def test_every_square_of_a_line_has_a_picture(self):
        for moves in ([(1, 0), (0, -1)] * 6, [(1, 0), (1, 0), (0, -1)] * 4, [(1, 0), (0, -1), (0, -1)] * 4):
            pictures, covered = F._diagonals(staircase((10, 30), moves), 60, 60)
            (line,) = F.diagonal_lines(staircase((10, 30), moves))
            self.assertTrue(pictures)
            self.assertEqual(covered, set(line[3]))
            for tile in pictures.values():
                self.assertLess(int(tile.rsplit("_", 1)[1]) % F.DIAG_SLOTS, len(F.DIAG_CUTS))

    def test_cut_slot_either_way_round(self):
        for i, (a, b) in enumerate(F.DIAG_CUTS):
            self.assertEqual(F._cut_slot(a, b), i)
            self.assertEqual(F._cut_slot(b, a), i)


class PicturesOnTheLineTests(unittest.TestCase):
    def test_pictures_stay_on_their_line(self):
        # Each direction of staircase: every picture on the line down the
        # staircase's middle - never moved towards the street (user,
        # 2026-10-14: fences on the road or pavement's edge).
        for moves in ([(1, 0), (0, -1)] * 6, [(1, 0), (0, 1)] * 6,
                      [(1, 0), (1, 0), (0, -1)] * 4, [(1, 0), (1, 0), (0, 1)] * 4):
            edges = staircase((20, 30), moves)
            (line,) = F.diagonal_lines(edges)
            _s, a, b = line[:3]
            pictures, _covered = F._diagonals(edges, 60, 60)
            self.assertTrue(pictures, moves)
            seg = LineString([a, b])
            for (x, y) in pictures:
                self.assertLess(seg.distance(Point(x + 0.5, y + 0.5)), 0.8, (moves, x, y))

if __name__ == "__main__":
    unittest.main()
