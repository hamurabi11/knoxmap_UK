"""KnoxMap UK: a landmark built to its own description (knoxbuild/landmark_uk.py).

A small two-storey "church" from a description: its rooms, the open edges
between them, its own floors, doors, windows (one tile entry a square) and
painted pieces (an edge piece and a two-storey object), and what tbx.py
writes from them.
"""
import unittest
import xml.etree.ElementTree as ET

from knoxbuild import landmark_uk as LM
from knoxbuild.buildtiles import parse_tbx
from knoxbuild.layout import Building, Plan, Room
from knoxbuild.tbx import render_tbx

W, H = 8, 10
ITEMS = {
    "floor_flags": {"kind": "floor", "tile": "church_furnishings_uk_01_90"},
    "pier": {"kind": "edge", "tiles": {"0": {"W": {"0,0": "church_furnishings_uk_01_40"},
                                             "N": {"0,0": "church_furnishings_uk_01_41"}}}},
    "organ": {"kind": "object", "tiles": {"0": {"S": {"0,0": "church_furnishings_uk_01_10", "1,0": "church_furnishings_uk_01_11"}},
                                          "1": {"S": {"0,0": "church_furnishings_uk_01_12"}}}},
    "perp3": {"kind": "window", "squares": [{"W": "church_windows_uk_01_0", "N": "church_windows_uk_01_1"},
                                            {"W": "church_windows_uk_01_2", "N": "church_windows_uk_01_3"}]},
}
DESC = {
    "entry": 1, "origin": [100, 200], "size": [W, H], "door": "oak_cottage",
    "rooms": [
        {"name": "nave", "kind": "church", "levels": [0], "rects": [[0, 3, 7, 9]], "floor": "floor_flags"},
        {"name": "chancel", "kind": "church", "levels": [0], "rects": [[0, 0, 7, 2]]},
        {"name": "gallery", "kind": "church", "levels": [1], "rects": [[0, 7, 7, 9]], "keep_floor": True},
        {"name": "roof space", "kind": "church", "levels": [1], "rects": [[0, 0, 7, 9]], "void": True},
    ],
    "open_levels": [0],
    "doors": [[0, 3, 10, "N"]],
    "windows": [[0, "perp3", 0, 4, "W"]],
    "pieces": [[0, "pier", 4, 3, "W"], [0, "pier", 4, 3, "N"]],
    "furniture": [[0, "organ", 2, 8, "N"]],
    "walls": {"interior": "walls_interior_house_03_020"},
}


def shell(width=W, height=H, levels=2):
    storeys = []
    for _ in range(levels):
        grid = [[1] * width for _ in range(height)]
        storeys.append(Plan(width=width, height=height, rooms=[Room(0, 0, width - 1, height - 1, "church")],
                            grid=grid))
    return Building(width=width, height=height, storeys=storeys, stairs=[(1, 1, "N")])


def dressed(at=(100, 200)):
    style = {"landmark": {"desc": DESC, "items": ITEMS, "at": list(at)}}
    return LM.dress(shell(), {"void": {}}, style)


class Rooms(unittest.TestCase):
    def test_rooms_from_rectangles_first_claim_wins(self):
        b, _roof, _style = dressed()
        g0, g1 = b.storeys[0].grid, b.storeys[1].grid
        self.assertEqual([r.landmark_name for r in b.storeys[0].rooms], ["nave", "chancel"])
        self.assertEqual((g0[0][0], g0[3][0]), (2, 1))
        self.assertEqual((g1[8][0], g1[0][0]), (1, 2))          # the gallery before the roof space

    def test_open_level_has_no_walls_between_rooms(self):
        b, _roof, _style = dressed()
        self.assertEqual(b.storeys[0].open_edge, {(x, 3, "N") for x in range(W)})
        self.assertEqual(b.storeys[1].open_edge, set())

    def test_void_keeps_the_gallery_floor(self):
        _b, roof, _style = dressed()
        self.assertNotIn((0, 8), roof["void"][1])
        self.assertIn((0, 0), roof["void"][1])

    def test_room_floor_is_the_recipes(self):
        b, _roof, _style = dressed()
        self.assertEqual(b.storeys[0].rooms[0].landmark_floor, "church_furnishings_uk_01_90")
        self.assertIsNone(b.storeys[0].rooms[1].landmark_floor)


class Openings(unittest.TestCase):
    def test_window_square_by_square(self):
        b, _roof, _style = dressed()
        st = b.storeys[0]
        self.assertEqual(st.windows, [(0, 4, "W"), (0, 5, "W")])
        self.assertEqual(st.window_entries[(0, 5, "W")]["tiles"], {"West": "church_windows_uk_01_2",
                                                                  "North": "church_windows_uk_01_3"})

    def test_moved_box_moves_everything(self):
        b, _roof, _style = dressed(at=(99, 200))                # Build's box one square further west
        self.assertEqual(b.storeys[0].doors, [(4, 10, "N")])
        self.assertEqual(b.storeys[0].windows[0], (1, 4, "W"))

    def test_doors_are_the_descriptions(self):
        _b, _roof, style = dressed()
        self.assertTrue(style["doors_all_front"])
        self.assertEqual(style["front_door"]["category"], "doors")


class Painted(unittest.TestCase):
    def test_two_pieces_on_one_square_take_two_layers(self):
        b, _roof, _style = dressed()
        layers = b.painted[0]
        self.assertEqual(layers["Furniture"][(4, 3)], "church_furnishings_uk_01_40")
        self.assertEqual(layers["Furniture2"][(4, 3)], "church_furnishings_uk_01_41")

    def test_an_object_facing_north_and_its_storey_above(self):
        b, _roof, _style = dressed()
        self.assertEqual(b.painted[0]["Furniture"][(3, 8)], "church_furnishings_uk_01_11")
        self.assertEqual(b.painted[1]["Furniture"][(2, 8)], "church_furnishings_uk_01_12")

    def test_one_plaster_inside(self):
        _b, _roof, style = dressed()
        self.assertTrue(style["one_interior"])
        self.assertEqual(style["interior"]["tiles"]["West"], "walls_interior_house_03_020")


class Review(unittest.TestCase):
    """What the QA review of 2026-10-19 found."""

    def test_unknown_room_kind_is_named(self):
        desc = dict(DESC, rooms=[dict(DESC["rooms"][0], kind="nave")])
        style = {"landmark": {"desc": desc, "items": ITEMS, "at": [100, 200]}}
        with self.assertRaisesRegex(ValueError, "nave"):
            LM.dress(shell(), {"void": {}}, style)

    def test_no_rooms_keeps_builds_void(self):
        desc = {k: v for k, v in DESC.items() if k != "rooms"}
        style = {"landmark": {"desc": desc, "items": ITEMS, "at": [100, 200]}}
        _b, roof, _style = LM.dress(shell(), {"void": {1: {(0, 0)}}}, style)
        self.assertEqual(roof["void"], {1: {(0, 0)}})

    def test_no_switch_on_a_window_nor_furniture_on_the_stairs(self):
        b, _roof, _style = dressed()
        st = b.storeys[0]
        windows = set(st.windows)
        self.assertFalse([f for f in st.furniture if (f[1], f[2], f[3]) in windows])
        sx, sy, sdir = b.stairs[0]
        run = {(sx, sy + k) for k in range(5)} if sdir == "N" else {(sx + k, sy) for k in range(5)}
        self.assertFalse([f for f in st.furniture if (f[1], f[2]) in run])


class Joined(unittest.TestCase):
    """regional_uk.join_units: a landmark's parts as one building."""

    @staticmethod
    def _geo(gap_m):
        d = gap_m / 111_320                                   # metres north, in degrees
        box = lambda y0, y1: {"type": "Polygon", "coordinates": [[[0, y0], [1e-4, y0], [1e-4, y1], [0, y1], [0, y0]]]}  # noqa: E731
        return {"features": [{"properties": {"osm_id": "a"}, "geometry": box(51.0, 51.0001)},
                             {"properties": {"osm_id": "b"}, "geometry": box(51.0001 + d, 51.0002 + d)}]}

    def test_footprint_replaces_the_outline_and_its_joins(self):
        from knoxbuild import regional_uk as R

        class Proj:                                     # a tile is 1e-5 degrees each way
            def to_latlon(self, x, y):
                return (51.0 + y * 1e-5, x * 1e-5)
        geo = self._geo(9.0)
        out = R.extend_outlines(geo, {"buildings": {"a": {"join": ["b"], "footprint": [[0, 0, 4, 3], [4, 2, 3, 3]]}}},
                                Proj())
        self.assertEqual([f["properties"]["osm_id"] for f in out["features"]], ["a"])
        xs = [p[0] for p in out["features"][0]["geometry"]["coordinates"][0]]
        self.assertAlmostEqual(max(xs), 7e-5)

    def test_parts_far_apart_stay_apart(self):
        from knoxbuild import regional_uk as R
        out = R.join_units(self._geo(25.0), {"buildings": {"a": {"join": ["b"]}}})
        self.assertEqual(len(out["features"]), 2)


class Written(unittest.TestCase):
    def setUp(self):
        b, roof, style = dressed()
        style.pop("landmark")
        style.setdefault("exterior", {"category": "exterior_walls", "tiles": {"West": "walls_exterior_house_01_000",
                                                                              "North": "walls_exterior_house_01_001"}})
        self.text = render_tbx(b, "test church", style)
        self.root = ET.fromstring(self.text)

    def test_the_file_parses(self):
        parsed = parse_tbx(self.text)
        self.assertEqual(len(parsed.floors), 3)

    def test_painted_layers_written(self):
        f0 = self.root.findall("floor")[0]
        self.assertEqual(sorted(t.get("layer") for t in f0.findall("tiles")), ["Furniture", "Furniture2"])

    def test_windows_take_their_own_entries(self):
        entries = self.root.findall("tile_entry")
        f0 = self.root.findall("floor")[0]
        tiles = [entries[int(o.get("Tile")) - 1].find("tile").get("tile") for o in f0.findall("object")
                 if o.get("type") == "window"]
        self.assertEqual(tiles, ["church_windows_uk_01_0", "church_windows_uk_01_2"])

    def test_the_nave_has_its_floor(self):
        entries = self.root.findall("tile_entry")
        nave = self.root.findall("room")[0]
        self.assertEqual(entries[int(nave.get("Floor")) - 1].find("tile").get("tile"), "church_furnishings_uk_01_90")

    def test_open_edges_are_bare_walls(self):
        f0 = self.root.findall("floor")[0]
        bare = [o for o in f0.findall("object") if o.get("type") == "wall" and o.get("Tile") == "0"]
        self.assertEqual(len(bare), W)

    def test_one_interior_wall(self):
        self.assertEqual(len({r.get("InteriorWall") for r in self.root.findall("room")}), 1)


if __name__ == "__main__":
    unittest.main()
