"""knoxbuild/checks.py: Build's hard checks on the .tbx it wrote (a storey with no rooms)."""
import os
import tempfile
import unittest

from knoxbuild import checks

W, H = 3, 2


def _grid(filled: bool) -> str:
    v = "1" if filled else "0"
    return "\n".join(",".join([v] * W) + "," for _ in range(H))


def _floor(rooms: bool, objects: str = "", tiles: str = "") -> str:
    return f" <floor>\n{objects}  <rooms>\n{_grid(rooms)}\n</rooms>\n{tiles} </floor>\n"


def _tbx(*floors: str) -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<building version="4" width="{W}" height="{H}">\n'
            ' <room Name="bar" InternalName="bar" Color="1 2 3"/>\n' + "".join(floors) + "</building>\n")


ROOF = '  <object type="roof" width="3" height="2" RoofType="PeakWE" Depth="Two" x="0" y="0"/>\n'
DOOR = '  <object type="door" x="1" y="2" dir="N" Tile="1" FrameTile="1"/>\n'
PLANT = '  <object type="furniture" FurnitureTiles="0" orient="N" x="1" y="1"/>\n'
FLOOR_TILE = '  <tiles layer="Floor">\n' + "\n".join(",".join(["1"] * (W + 1)) + "," for _ in range(H + 1)) + "\n</tiles>\n"


class EmptyStoreys(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_a_house_with_rooms_on_both_storeys_and_a_roof_floor_passes(self):
        p = self.write("a.tbx", _tbx(_floor(True), _floor(True), _floor(False, ROOF), _floor(False)))
        self.assertEqual(checks.empty_storeys(p, 2), [])
        self.assertEqual(checks.empty_storeys(p), [])

    def test_a_storey_of_the_plan_with_no_rooms_fails(self):
        # two storeys built, the upper one roomless: only its roof above it
        p = self.write("b.tbx", _tbx(_floor(True), _floor(False), _floor(False, ROOF)))
        errs = checks.empty_storeys(p, 2)
        self.assertEqual(len(errs), 1)
        self.assertIn("storey 1 has no rooms", errs[0])

    def test_a_storey_missing_out_of_the_middle_fails(self):
        p = self.write("c.tbx", _tbx(_floor(True), _floor(False), _floor(True), _floor(False, ROOF)))
        self.assertIn("storey 1 has no rooms (rooms on a floor above it)", checks.empty_storeys(p))

    def test_a_roomless_floor_with_a_door_or_a_floor_tile_fails(self):
        p = self.write("d.tbx", _tbx(_floor(True), _floor(False, DOOR)))
        self.assertEqual(checks.empty_storeys(p), ["storey 1 has no rooms (it holds door)"])
        p = self.write("e.tbx", _tbx(_floor(True), _floor(False, tiles=FLOOR_TILE)))
        self.assertEqual(checks.empty_storeys(p), ["storey 1 has no rooms (it holds Floor tiles)"])

    def test_rooftop_plant_on_the_roof_floor_passes(self):
        p = self.write("f.tbx", _tbx(_floor(True), _floor(False, ROOF + PLANT)))
        self.assertEqual(checks.empty_storeys(p, 1), [])

    def test_a_pack_of_loose_tiles_is_not_a_building(self):
        # Build's fences / structures / chimneys lots: no room anywhere, floor tiles on the ground
        p = self.write("pack.tbx", _tbx(_floor(False, tiles=FLOOR_TILE), _floor(False)))
        self.assertEqual(checks.empty_storeys(p), [])
        self.assertEqual(len(checks.empty_storeys(p, 1)), 1)     # ...but a building of one storey with none fails

    def test_check_buildings_names_each_building(self):
        self.write("g.tbx", _tbx(_floor(True), _floor(False), _floor(False, ROOF)))
        self.write("h.tbx", _tbx(_floor(True), _floor(True), _floor(False, ROOF)))
        rows = [{"file": "g.tbx", "name": "The Bad House", "levels": 2},
                {"file": "h.tbx", "name": "The Good House", "levels": 2}]
        self.assertEqual(checks.check_buildings(self.dir, rows),
                         ["g.tbx (The Bad House): storey 1 has no rooms (one of its 2 storeys)"])


if __name__ == "__main__":
    unittest.main()
