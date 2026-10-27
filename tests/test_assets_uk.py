"""KnoxMap UK with its UK assets off (knoxbuild/assets_uk.py; Building_types'
streets/vanilla_assets.py and uk_pipeline.py): the regional data comes with vanilla
stand-ins and no UK part; the loose tiles' kerbs go back to vanilla's pieces of the same
name and the rest of ours are left out; the ground is compiled by the vanilla rules; the
pipeline swaps, checks and skips the listed builds only when they are off."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)
sys.path.insert(0, os.path.join(HERE, ".."))

from knoxbuild import assets_uk  # noqa: E402
from streets import vanilla_assets  # noqa: E402

REGIONAL = {
    "palette": {"drop": ["timber", "clapboard"],
                "add": [{"name": "red_brick", "wall": "walls_exterior_house_01_004"},
                        {"name": "uk_edwardian", "wall": "walls_house_uk_02_brick_red_edw_000"},
                        {"name": "uk_barn_black_iron", "wall": "walls_farm_uk_01_000"}],
                "shop_signs": {"x": "fascias_uk_01_0"},
                "uk_stand_ins": {"uk_edwardian": "red_brick", "uk_barn_black_iron": "timber"}},
    "buildings": {
        "1": {"style": "uk_edwardian", "levels": 2, "bay": "iw", "drains": True, "door": "white_upvc",
              "roof": [[1, 2, 3, 4, "WE", 2]], "upper_wall": ["walls_house_uk_02_x_000", "walls_house_uk_02_x_016"]},
        "2": {"style": "uk_uk_edwardian__slate__casement", "fascia": {"sign": "a"}},
        "3": {"style": "uk_barn_black_iron", "listed": {"entry": 1}},
        "4": {"style": "red_brick", "shopfront": "walls_commercial_erika_04"},
    },
    "fuel": {"canopy": "commercial_fuel_uk_01_0"},
    "pub_props": [1],
}


class Regional(unittest.TestCase):
    def test_vanilla(self):
        v = assets_uk.vanilla(REGIONAL)
        b = v["buildings"]
        self.assertEqual(b["1"], {"style": "red_brick", "levels": 2, "roof": [[1, 2, 3, 4, "WE", 2]]})
        self.assertEqual(b["2"], {"style": "red_brick"})                 # a variant of a UK style
        self.assertEqual(b["3"], {"style": "timber"})                    # dropped by the palette: back
        self.assertEqual(b["4"], {"style": "red_brick", "shopfront": "walls_commercial_erika_04"})   # Erika's stay
        self.assertEqual([a["name"] for a in v["palette"]["add"]], ["red_brick"])
        self.assertNotIn("timber", v["palette"]["drop"])
        self.assertNotIn("shop_signs", v["palette"])
        self.assertNotIn("pub_props", v)
        self.assertFalse(assets_uk.has_uk(v))

    def test_the_switch(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(assets_uk.off(d))
            open(os.path.join(d, assets_uk.OFF_FILE), "w").close()
            self.assertTrue(assets_uk.off(d))


class LooseTiles(unittest.TestCase):
    def test_names(self):
        f = vanilla_assets.vanilla_name
        self.assertEqual(f("street_curbs_uk_01_diag_2_14"), "street_curbs_01_diag_2_14")
        self.assertEqual(f("street_curbs_uk_01_blend_street_40"), "street_curbs_01_blend_street_40")
        self.assertEqual(f("street_curbs_uk_01_blend_pavement_3"), "street_curbs_01_blend_gravel_3")
        self.assertIsNone(f("road_markings_uk_01_4"))
        self.assertIsNone(f("street_furniture_uk_02_12"))
        self.assertEqual(f("fencing_01_002"), "fencing_01_002")


class Rules(unittest.TestCase):
    def test_the_vanilla_rules_for_a_map_with_its_assets_off(self):
        from tools.terrain_source import rules_path, VANILLA_CONFIG
        try:
            rules_path()
        except FileNotFoundError:
            self.skipTest("no map tools here (Setup.bat installs them)")
        with tempfile.TemporaryDirectory() as d:
            self.assertNotIn(VANILLA_CONFIG, rules_path(d))
            open(os.path.join(d, assets_uk.OFF_FILE), "w").close()
            try:
                self.assertIn(VANILLA_CONFIG, rules_path(d))
            except FileNotFoundError as e:                 # no vanilla rules made on this PC yet
                self.assertIn("UK pipeline", str(e))


class Pipeline(unittest.TestCase):
    def plan(self, **s):
        import uk_pipeline as U
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(os.path.join(t, "bt", "data", "town"))
            os.makedirs(os.path.join(t, "knox", "output", "T"))
            json.dump({"area": "town", "source": "S", **s}, open(os.path.join(t, "knox", "output", "T", "uk.json"), "w"))
            was = U.KNOX, U.BT
            U.KNOX, U.BT = os.path.join(t, "knox"), os.path.join(t, "bt")
            try:
                sys.argv = ["uk_pipeline.py", "T", "--list"]
                out = io.StringIO()
                with redirect_stdout(out):
                    U.main()
            finally:
                U.KNOX, U.BT = was
        return out.getvalue().split()

    def test_steps_by_the_settings(self):
        on = self.plan()
        self.assertIn("listed", on)
        self.assertFalse({"vanilla_assets", "uk_check", "vanilla_rules"} & set(on))
        off = self.plan(uk_assets=False)
        self.assertNotIn("listed", off)
        self.assertLess(off.index("kerbs"), off.index("vanilla_assets"))
        self.assertLess(off.index("validate"), off.index("uk_check"))
        fit = self.plan(buildings="knoxmap")
        self.assertFalse({"blocks", "lidar_outlines", "split_units", "shops"} & set(fit))


if __name__ == "__main__":
    unittest.main()
