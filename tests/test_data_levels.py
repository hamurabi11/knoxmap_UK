"""The UK chain's data levels (KnoxMap UK: Building_types/src/area/area_conf.py; PLAN U
stage 1): at osm_lidar no extra file is read even where it is there, and an area with no
council reads no EPC (not Hart's); a home count from OSM's house numbers; a home with
no era takes a look by its built form; with no OS Open Roads every link is OSM's alone."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ukdirs  # noqa: E402 - KnoxMap UK: where Building_types is, in either layout
sys.path.insert(0, ukdirs.UK_SRC)
sys.path.insert(0, os.path.join(HERE, ".."))

from area import area_conf, subtypes  # noqa: E402
from buildings.addresses import house_numbers  # noqa: E402
from roadgraph import osroads  # noqa: E402


def area(tmp, name, conf=None, files=(), councils=()):
    a = os.path.join(tmp, name)
    os.makedirs(a)
    if conf is not None:
        json.dump(conf, open(os.path.join(a, "area.json"), "w"))
    for f in files:
        open(os.path.join(a, f), "w").close()
    for c in councils:
        os.makedirs(os.path.join(tmp, c), exist_ok=True)
    return a


class Levels(unittest.TestCase):
    def test_full_reads_what_is_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = area(tmp, "town_core", {"council": "hart"}, ["uprn.csv"], ["hart"])
            self.assertEqual(area_conf.extra(a, "uprn.csv"), os.path.join(a, "uprn.csv"))
            self.assertIsNone(area_conf.extra(a, "os_roads.geojson"))
            self.assertEqual(area_conf.epc_dir(a), os.path.normpath(os.path.join(tmp, "hart")))

    def test_osm_lidar_reads_no_extra_file_even_when_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = area(tmp, "town_osm", {"data": "osm_lidar", "council": "hart"},
                     ["uprn.csv", "os_roads.geojson", "inspire_town.geojson"], ["hart"])
            self.assertIsNone(area_conf.extra(a, "uprn.csv"))
            self.assertIsNone(area_conf.extra(a, "os_roads.geojson"))
            self.assertIsNone(area_conf.epc_dir(a))
            self.assertTrue(area_conf.is_extra(os.path.join(a, "inspire_town.geojson")))

    def test_no_council_no_epc(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = area(tmp, "elsewhere_core", councils=["hart"])
            self.assertEqual(area_conf.level(a), "full")
            self.assertIsNone(area_conf.epc_dir(a))

    def test_town_from_settings_else_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(area_conf.town(area(tmp, "alton_osm")), "alton")
            self.assertEqual(area_conf.town(area(tmp, "x_core", {"town": "Odiham"})), "odiham")

    def test_a_bad_level_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                area_conf.level(area(tmp, "t", {"data": "osm"}))


class HouseNumbers(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(house_numbers("12"), 1)
        self.assertEqual(house_numbers("4a"), 1)
        self.assertEqual(house_numbers("1;3;5"), 3)
        self.assertEqual(house_numbers("2-8"), 4)        # one side of the street
        self.assertEqual(house_numbers("1-4"), 4)        # both sides
        self.assertEqual(house_numbers("1-999"), 30)     # capped


class NoEra(unittest.TestCase):
    def b(self, form, **kw):
        return dict({"form": form, "btag": "yes", "era": "", "area_m2": 80.0}, **kw)

    def test_by_built_form(self):
        self.assertEqual(subtypes.subtype_of(self.b("mid_terrace"))[0], "unknown_terrace")
        self.assertEqual(subtypes.subtype_of(self.b("semi"))[0], "unknown_semi")
        self.assertEqual(subtypes.subtype_of(self.b("detached"))[0], "unknown_detached")
        self.assertEqual(subtypes.subtype_of(self.b("multi_unit"))[0], "unknown_flats")
        self.assertEqual(subtypes.subtype_of(self.b("detached"), roof={"storeys_lidar": "1"})[0], "unknown_bungalow")

    def test_an_era_still_wins(self):
        self.assertEqual(subtypes.subtype_of(self.b("mid_terrace", era="pre1900"))[0], "vic_terrace")

    def test_every_unknown_look_is_styled(self):
        pal = json.load(open(os.path.join(ukdirs.UK, "palettes", "english.json"),
                             encoding="utf-8"))
        for k in subtypes.SUBTYPES:
            if k.startswith("unknown_"):
                self.assertTrue(pal["subtype_styles"].get(k), k)


class FormFromTags(unittest.TestCase):
    def test_a_single_outline_by_its_tag_without_uprns(self):
        from area.features import refine_with_tags
        blds = [{"btag": t, "form": "detached"} for t in ("terrace", "semidetached_house", "apartments", "house")]
        blds.append({"btag": "terrace", "form": "detached", "uprn_count": 1})   # UPRNs counted: theirs stands
        refine_with_tags(blds)
        self.assertEqual([b["form"] for b in blds],
                         ["terrace_row", "semi_pair", "multi_unit", "detached", "detached"])


class SetupArea(unittest.TestCase):
    def test_the_councils_inspire_zip_by_name(self):
        from area import setup_area
        with tempfile.TemporaryDirectory() as tmp:
            for f in ("East_Hampshire_District_Council.zip", "City_of_Westminster.zip", "Hart_District_Council.zip"):
                open(os.path.join(tmp, f), "w").close()
            was, setup_area.INSPIRE_DIR = setup_area.INSPIRE_DIR, tmp
            try:
                self.assertTrue(setup_area.inspire_zip("East Hampshire").endswith("East_Hampshire_District_Council.zip"))
                self.assertTrue(setup_area.inspire_zip("Westminster").endswith("City_of_Westminster.zip"))
                self.assertIsNone(setup_area.inspire_zip("Rushmoor"))
            finally:
                setup_area.INSPIRE_DIR = was
        self.assertEqual(setup_area.slug("East Hampshire"), "east_hampshire")


class NoOsRoads(unittest.TestCase):
    def test_every_link_is_osms_alone(self):
        links = [SimpleNamespace(cls="service", tags={"name": "Back Lane"}, ways=[1]),
                 SimpleNamespace(cls="residential", tags={}, ways=[2, 3])]
        rep = osroads.conflate(links, None, [], log=lambda *a: None)
        self.assertEqual([e["link"] for e in rep["osm only"]], [0, 1])
        self.assertEqual(rep["matched"], 0)
        self.assertEqual([l.cls for l in links], ["service", "residential"])


if __name__ == "__main__":
    unittest.main()
