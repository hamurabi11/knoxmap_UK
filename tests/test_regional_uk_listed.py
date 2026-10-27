"""KnoxMap UK: the listed buildings' hooks in regional_uk (listed_building_maker).

A listed building's style may carry its own roof covering, no grime, no
shutters; its entry may name a front door, a doorcase or porch, wall trims
and a hipped roof. Nothing of it may reach a building that does not ask.
"""
import unittest

from knoxbuild import catalog as C
from knoxbuild import regional_uk as R

KIT = "walls_house_uk_01_flint"
SPEC = {"name": "uk_flint__thatch__casement", "like": "brick", "wall": f"{KIT}_000", "shaped": f"{KIT}_016",
        "caps": f"{KIT}_048", "caps_shallow": f"{KIT}_064", "caps_30": f"{KIT}_072", "roof_from": "brick",
        "trim_from": "timber", "roof_sheet": "roofs_uk_01_thatch", "roof_sheet_30": "roofs_30_uk_01_thatch",
        "window": "fixtures_windows_uk_01_004", "grime": False, "shutters": False, "by_id_only": True}


def styles():
    return {s["name"]: s for s in C.HOUSE_STYLES}


class ListedStyle(unittest.TestCase):
    def test_roof_sheets_replace_the_donors_slopes_and_tops(self):
        s = R.make_style(SPEC, styles())
        sheets = {v.rsplit("_", 1)[0] for g in ("slopes", "tops") for v in s["roof"][g]["tiles"].values()}
        self.assertEqual(sheets, {"roofs_uk_01_thatch", "roofs_30_uk_01_thatch"})

    def test_slots_are_kept(self):
        s = R.make_style(SPEC, styles())
        donor = styles()["brick"]["roof"]["slopes"]["tiles"]
        for k, v in s["roof"]["slopes"]["tiles"].items():
            self.assertEqual(int(v.rsplit("_", 1)[1]), int(donor[k].rsplit("_", 1)[1]), k)

    def test_caps_move_onto_the_kit(self):
        s = R.make_style(SPEC, styles())
        caps = s["roof"]["caps"]["tiles"]
        self.assertTrue(all(v.startswith(KIT) for k, v in caps.items()), caps)
        self.assertEqual(caps["CapRiseE1"], f"{KIT}_048")

    def test_no_grime_no_shutters_no_trim(self):
        s = R.make_style(SPEC, styles())
        self.assertIsNone(s["grime"])
        self.assertIsNone(s["shutters"])
        self.assertFalse(s.get("trim"))

    def test_a_plain_spec_keeps_grime_and_shutters(self):
        s = R.make_style({"name": "red_brick", "like": "brick"}, styles())
        self.assertEqual(s["grime"], styles()["brick"]["grime"])
        self.assertEqual(s["shutters"], styles()["brick"]["shutters"])

    def test_window_is_the_kits(self):
        s = R.make_style(SPEC, styles())
        self.assertEqual(s["window"]["tiles"], {"West": "fixtures_windows_uk_01_004", "North": "fixtures_windows_uk_01_005"})


class ListedPalette(unittest.TestCase):
    def setUp(self):
        self.saved = list(C.HOUSE_STYLES)

    def tearDown(self):
        C.HOUSE_STYLES[:] = self.saved

    def test_by_id_only_styles_stay_out_of_the_random_pool(self):
        overlay = {"palette": {"add": [SPEC, {"name": "red_brick", "like": "brick"}]}}
        R.apply_palette(overlay)
        names = {s["name"] for s in C.HOUSE_STYLES}
        self.assertIn("red_brick", names)
        self.assertNotIn(SPEC["name"], names)
        self.assertIn(SPEC["name"], overlay["_styles"])


class ListedEntry(unittest.TestCase):
    overlay = {"buildings": {"1": {"door": "oak_cottage", "door_surround": "porch",
                                   "wall_trim": {"plinth": "plinth_black", "cornice": "cornice_stone",
                                                 "band": "no_such_trim"}},
                             "2": {"door": "not_a_door"}}}

    def test_door_from_the_entry(self):
        self.assertEqual(R.door_of(self.overlay, 1), "oak_cottage")
        self.assertIsNone(R.door_of(self.overlay, 2))
        self.assertIsNone(R.door_of(self.overlay, 3))
        door = R.front_door("seed", "oak_cottage")["tiles"]["West"]
        self.assertEqual(door, f"doors_uk_01_{4 * R.DOOR_STYLES.index('oak_cottage'):03d}")

    def test_surround_is_a_west_and_north_pair(self):
        self.assertEqual(R.surround_of(self.overlay, 1), ("doorways_uk_01_004", "doorways_uk_01_005"))
        self.assertIsNone(R.surround_of(self.overlay, 2))
        style = R.with_front_door({"name": "x"}, None, "s", "oak_cottage", R.surround_of(self.overlay, 1))
        self.assertEqual(style["door_surround"], ("doorways_uk_01_004", "doorways_uk_01_005"))

    def test_no_surround_for_a_shop(self):
        style = R.with_front_door({"name": "x"}, "shop", "s", "oak_cottage", ("a", "b"))
        self.assertNotIn("door_surround", style)

    def test_trims_onto_the_buildings_style_only(self):
        base = {"name": "brick"}
        got = R.style_for(self.overlay, 1, None, base)
        self.assertEqual(got["wall_trim"], {"plinth": ("wall_trim_uk_01_000", "wall_trim_uk_01_001"),
                                            "cornice": ("wall_trim_uk_01_010", "wall_trim_uk_01_011")})
        self.assertNotIn("wall_trim", base)
        self.assertNotIn("wall_trim", R.style_for(self.overlay, 2, None, base))


class Drains(unittest.TestCase):
    def test_drains_only_where_asked(self):
        overlay = {"buildings": {"1": {"drains": True}, "2": {}}}
        got = R.style_for(overlay, 1, None, {"name": "brick"})
        self.assertEqual(got["drains"][0][0], ["drains_uk_01_000", "drains_uk_01_001", "drains_uk_01_002",
                                               "drains_uk_01_003"])
        self.assertEqual(got["drains"][1][1][3], "drains_uk_01_015")
        self.assertNotIn("drains", R.style_for(overlay, 2, None, {"name": "brick"}))

    def test_a_pipe_at_each_end_of_a_seen_wall_run(self):
        from knoxbuild.tbx import _lay_drains

        class Storey:
            def __init__(self, grid):
                self.grid = grid
        two = [[1, 1, 1], [1, 1, 1]]               # 3 x 2, two storeys
        drains = {}
        _lay_drains([Storey(two), Storey(two)], 3, 2, R.DRAINS, drains)
        # the east wall (x = 3, along y 0-1) and the south wall (y = 2, along x 0-2): their ends
        self.assertEqual(set(drains[0]), {(3, 0), (3, 1), (0, 2), (2, 2)})
        self.assertEqual(drains[0][(3, 0)], "drains_uk_01_000")    # west-wall foot, run start
        self.assertEqual(drains[1][(2, 2)], "drains_uk_01_014")    # north-wall top, run end


class Hip(unittest.TestCase):
    def gable(self, box, hip, open_ends=True):
        pieces = {}
        R._gable(pieces, "WE", box, 2, lambda cells: open_ends, True, hip=hip)
        return [p[0] for ps in pieces.values() for p in ps]

    def test_hip_where_the_game_can_build_one(self):
        self.assertEqual(self.gable((0, 0, 9, 5), True), ["Peak30Quad"])

    def test_no_hip_unasked(self):
        self.assertNotIn("Peak30Quad", self.gable((0, 0, 9, 5), False))

    def test_no_hip_on_an_even_length(self):
        self.assertNotIn("Peak30Quad", self.gable((0, 0, 8, 5), True))

    def test_no_hip_against_a_neighbour(self):
        self.assertNotIn("Peak30Quad", self.gable((0, 0, 9, 5), True, open_ends=False))

    def test_an_even_width_hips_its_odd_part_and_flattens_the_strip(self):
        kinds = self.gable((0, 0, 9, 6), True)
        self.assertIn("Peak30Quad", kinds)
        self.assertIn("FlatTop", kinds)

    def test_too_wide_for_a_hip(self):
        self.assertNotIn("Peak30Quad", self.gable((0, 0, 15, 9), True))


if __name__ == "__main__":
    unittest.main()
