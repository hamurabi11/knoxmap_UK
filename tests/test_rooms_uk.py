"""rooms_uk: English house rooms by CP-SAT (ROOMS.md, step 1: made-up houses)."""
import unittest

from knoxbuild import rooms_uk as R

try:
    import ortools  # noqa: F401
    HAVE_ORTOOLS = True
except ImportError:
    HAVE_ORTOOLS = False

HOUSES = {name: (house, prog) for name, house, prog in R.demo_houses()}


@unittest.skipUnless(HAVE_ORTOOLS, "OR-Tools not installed")
class DemoHouses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lays = {name: R.solve(house, prog, seconds=3.0, workers=4)
                    for name, (house, prog) in HOUSES.items()
                    if not prog.startswith("shop") and prog != "flats_block"}

    def test_every_house_solves_and_keeps_the_rules(self):
        for name, lay in self.lays.items():
            with self.subTest(name=name):
                self.assertIsNotNone(lay)
                self.assertEqual(R.check(lay), [])

    def test_bedrooms_upstairs_except_in_a_bungalow(self):
        for name, lay in self.lays.items():
            ground = {r["kind"] for r in lay.storeys[0]["rooms"]}
            with self.subTest(name=name):
                if len(lay.storeys) > 1:
                    self.assertFalse(ground & set(R.RULES["bedroom_kinds"]))
                else:
                    self.assertTrue(ground & set(R.RULES["bedroom_kinds"]))
                    self.assertIsNone(lay.stairs)

    def test_rooms_with_stairs_are_small(self):
        # the hall and landing: the 5 stair squares and a little passage
        for name, lay in self.lays.items():
            for st in lay.storeys:
                for room in st["rooms"]:
                    if room["kind"] in R.ROOT_KINDS:
                        with self.subTest(name=name, kind=room["kind"]):
                            self.assertLessEqual(len(room["squares"]), R.STAIR_RUN + 8)

    def test_stairs_run_along_a_wall(self):
        for name, lay in self.lays.items():
            if lay.stairs:
                x, y, d = lay.stairs
                with self.subTest(name=name):
                    self.assertIn((x, y, 1 if d == "N" else 0), R.stair_places(lay.house))

    def test_kitchen_at_the_back_with_the_back_door(self):
        for name, lay in self.lays.items():
            with self.subTest(name=name):
                self.assertIsNotNone(lay.back_door)
                st = lay.storeys[0]
                a, b = R._door_squares(lay.back_door)
                inside = a if R._room_at(st["grid"], *a) else b
                room = st["rooms"][st["grid"][inside[1]][inside[0]] - 1]
                self.assertIn(room["kind"], ("kitchen", "kitchen_diner", "kitchen_breakfast", "utility"))

    def test_windows_are_the_rooms_own(self):
        # every window: in its room's wall, the room within its cap (the solver's own choice)
        for name, lay in self.lays.items():
            for st in lay.storeys:
                for room in st["rooms"]:
                    spots = room.get("windows", [])
                    with self.subTest(name=name, kind=room["kind"]):
                        self.assertLessEqual(len(spots), R.KINDS[room["kind"]]["cap"])
                        for (x, y, _) in spots:
                            self.assertIn((x, y), room["squares"])

    def test_bathrooms_are_at_least_two_by_two(self):
        # the game's bathroom fittings need 2 x 2 (user, 2026-10-20)
        for name, lay in self.lays.items():
            for st in lay.storeys:
                for room in st["rooms"]:
                    if room["build"] == "bathroom":
                        with self.subTest(name=name, kind=room["kind"]):
                            for x0, y0, x1, y1 in room["rects"]:
                                self.assertGreaterEqual(min(x1 - x0, y1 - y0) + 1, 2)
                            if room["kind"] == "bathroom":
                                # a bath, toilet, sink and cupboard: 3 x 2 at least
                                x0, y0, x1, y1 = room["rects"][0]
                                self.assertGreaterEqual(max(x1 - x0, y1 - y0) + 1, 3)

    def test_windows_stack(self):
        # most first-floor windows sit over a ground-floor one
        for name, lay in self.lays.items():
            if len(lay.storeys) < 2:
                continue
            down = set(lay.storeys[0]["windows"])
            up = set(lay.storeys[1]["windows"])
            with self.subTest(name=name):
                self.assertGreaterEqual(len(up & down), len(up) // 2)


@unittest.skipUnless(HAVE_ORTOOLS, "OR-Tools not installed")
class Cache(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()
        self.house, self.prog = HOUSES["iw_semi"]

    def test_saved_layouts_come_back_the_same(self):
        cache = R.LayoutCache(self.dir)
        first = R.layout_for(self.house, self.prog, "a", cache)
        cache.save()
        again = R.layout_for(self.house, self.prog, "a", R.LayoutCache(self.dir))
        self.assertEqual([st["grid"] for st in first.storeys], [st["grid"] for st in again.storeys])
        self.assertEqual(first.stairs, again.stairs)
        self.assertEqual(R.check(again), [])

    def test_the_pair_of_a_semi_comes_from_its_twin(self):
        cache = R.LayoutCache(self.dir)
        R.layout_for(self.house, self.prog, "a", cache)
        twin = R.mirror_house(self.house)
        self.assertTrue(cache.known(twin, self.prog))
        options = cache.options(twin, self.prog)
        self.assertTrue(options)
        party_edges = {R._edge(*e) for e in twin.party}
        for lay in options:
            self.assertEqual(R.check(lay), [])
            # the party wall moved to the other side: no window or door in it
            for st in lay.storeys:
                self.assertFalse(party_edges & set(st["windows"]))
                self.assertFalse(party_edges & set(st["doors"]))

    def test_a_footprint_with_no_layout_is_remembered(self):
        tiny = R.House(3, 3, [[[True] * 3 for _ in range(3)]] * 2, "S")
        cache = R.LayoutCache(self.dir)
        self.assertIsNone(R.layout_for(tiny, "semi_interwar", "a", cache))
        self.assertTrue(cache.known(tiny, "semi_interwar"))

    def test_editing_one_kind_keeps_other_programmes_cached(self):
        import copy
        before = {n: R.programme_hash(n) for n in ("semi_interwar", "period_large")}
        saved = copy.deepcopy(R.KINDS["drawing_room"])
        try:
            R.KINDS["drawing_room"]["target"] += 1
            R._PROG_HASH.clear()
            self.assertEqual(R.programme_hash("semi_interwar"), before["semi_interwar"])
            self.assertNotEqual(R.programme_hash("period_large"), before["period_large"])
        finally:
            R.KINDS["drawing_room"] = saved
            R._PROG_HASH.clear()

    def test_mirror_twice_is_the_same_layout(self):
        lay = R.solve(self.house, self.prog)
        twin = R.mirror_house(self.house)
        back = R.mirror(R.mirror(lay, twin), self.house)
        self.assertEqual([st["grid"] for st in back.storeys], [st["grid"] for st in lay.storeys])
        self.assertEqual(sorted(back.storeys[0]["windows"]), sorted(lay.storeys[0]["windows"]))
        self.assertEqual(sorted(back.storeys[0]["doors"]), sorted(lay.storeys[0]["doors"]))


@unittest.skipUnless(HAVE_ORTOOLS, "OR-Tools not installed")
class IntoBuild(unittest.TestCase):
    def test_a_layout_becomes_a_furnished_building_and_a_tbx(self):
        from knoxbuild.tbx import render_tbx
        house, prog = HOUSES["iw_semi"]
        lay = R.solve(house, prog)
        # Build's party walls: the neighbour is one storey (a bungalow beside it)
        party = {R._edge(*e): 1 for e in house.party}
        b = R.to_building(lay, seed=3, party=party)
        self.assertEqual(len(b.storeys), 2)
        self.assertEqual(b.stairs, [lay.stairs])
        self.assertTrue(b.storeys[0].party)
        self.assertFalse(b.storeys[1].party)      # above the neighbour: an outside wall
        self.assertTrue(all(s.furniture for s in b.storeys))
        self.assertIn("<room", render_tbx(b, "test", None))

    def test_on_by_default_and_a_map_may_switch_it_off(self):
        import os
        import tempfile
        saved = os.environ.pop("KNOXMAP_UK_ROOMS", None)
        try:
            folder = tempfile.mkdtemp()
            self.assertTrue(R.switch_for(folder))                       # the default
            open(os.path.join(folder, R.OFF_FILE), "w").close()
            self.assertFalse(R.switch_for(folder))
            os.remove(os.path.join(folder, R.OFF_FILE))
            open(os.path.join(folder, R.SWITCH_FILE), "w").close()
            self.assertTrue(R.switch_for(folder))                       # on again, over the 0 left behind
        finally:
            os.environ.pop("KNOXMAP_UK_ROOMS", None)
            if saved is not None:
                os.environ["KNOXMAP_UK_ROOMS"] = saved


@unittest.skipUnless(HAVE_ORTOOLS, "OR-Tools not installed")
class HomesOverShops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lays = {n: R.solve(*HOUSES[n]) for n in ("shop_home", "shop_two_flats")}

    def test_they_solve(self):
        for name, lay in self.lays.items():
            with self.subTest(name=name):
                self.assertIsNotNone(lay)
                self.assertEqual(R.check(lay), [])

    def test_no_door_between_two_flats(self):
        lay = self.lays["shop_two_flats"]
        for st in lay.storeys:
            for room in st["rooms"]:
                if room["parent"] is not None:
                    p = st["rooms"][room["parent"]]
                    if room["kind"] != "flat_hall":
                        self.assertEqual(room["unit"], p["unit"])

    def test_the_shop_and_the_flats_each_have_a_street_door(self):
        for name, lay in self.lays.items():
            st = lay.storeys[0]
            street = [e for e in st["doors"] if e[2] == "N" and e[1] == lay.house.height]   # street S
            with self.subTest(name=name):
                self.assertGreaterEqual(len(street), 2)
                self.assertTrue(st.get("shop_front"))

    def test_build_names_the_shop_and_keeps_the_lobby(self):
        lay = self.lays["shop_home"]
        b = R.to_building(lay, seed=1, uses=[("grocery", "grocerystorage")])
        ground = b.storeys[0]
        kinds = {r.kind for r in ground.rooms}
        self.assertIn("grocery", kinds)
        self.assertIn("hall", kinds)
        self.assertEqual(ground.kind, "shop")
        self.assertTrue(ground.shop_front)


BUSINESSES = ("shop_one_storey", "shop_stores_over", "takeaway_home", "bank", "surgery", "estate_agent",
              "pub_two_bars", "pub_function_room")


@unittest.skipUnless(HAVE_ORTOOLS, "OR-Tools not installed")
class Businesses(unittest.TestCase):
    """Commercial ground floors by class, store rooms over them, pubs (ROOMS.md section 17)."""
    @classmethod
    def setUpClass(cls):
        cls.lays = {n: R.solve(*HOUSES[n]) for n in BUSINESSES}

    def test_they_solve_and_keep_the_rules(self):
        for name, lay in self.lays.items():
            with self.subTest(name=name):
                self.assertIsNotNone(lay)
                self.assertEqual(R.check(lay), [])        # a pub's own checks too (pub_uk.check)

    def test_rooms_are_named_after_the_business(self):
        b = R.to_building(self.lays["takeaway_home"], seed=1, uses=[("restaurantdining", "pizzakitchen")])
        kinds = {r.kind for r in b.storeys[0].rooms}
        self.assertTrue({"restaurantdining", "pizzakitchen", "hall"} <= kinds)
        self.assertEqual(b.storeys[1].kind, "apartment")
        b = R.to_building(self.lays["surgery"], seed=1, uses=[("dentist", "medicaloffice")])
        self.assertIn("dentist", {r.kind for r in b.storeys[0].rooms})

    def test_a_pubs_kitchen_stays_a_kitchen(self):
        lay = self.lays["pub_two_bars"]
        b = R.to_building(lay, seed=1, uses=[("restaurantdining", "restaurantkitchen"), ("bar", "barstorage")])
        kinds = [r.kind for r in b.storeys[0].rooms]
        self.assertGreaterEqual(kinds.count("bar"), 1)       # the bars (the lounge and a snug are optional)
        self.assertIn("barstorage", kinds)
        if any(r["kind"] == "shop_kitchen" for r in lay.storeys[0]["rooms"]):
            self.assertIn("restaurantkitchen", kinds)

    def test_a_bar_is_fitted_with_its_counter_back_bar_and_tables(self):
        # a hand-made bar (solved layouts differ run to run): 8 x 7, the street to
        # the south, the cellar behind it to the north
        import collections
        from knoxbuild import layout as L, pub_uk
        w, h = 8, 9
        grid = [[2] * w for _ in range(2)] + [[1] * w for _ in range(7)]
        plan = L.Plan(width=w, height=h, rooms=[L.Room(0, 2, 7, 8, kind="bar"), L.Room(0, 0, 7, 1, kind="barstorage")],
                      grid=grid, drawn=True)
        plan.doors = [(3, 9, "N")]                  # the street door
        self.assertEqual(pub_uk.fit_bars(plan, "S"), 1)
        got = collections.Counter(role for role, x, y, o in plan.furniture)
        self.assertGreaterEqual(got["counter_4"], 3)
        self.assertGreaterEqual(got["shop_fridge"], 1)
        self.assertGreaterEqual(got["round_table"], 2)
        counter = [(x, y) for role, x, y, o in plan.furniture if role == "counter_4"]
        self.assertTrue(all(y == 4 for _, y in counter))        # one aisle in from the cellar wall
        self.assertTrue(all((x, y) not in {(3, 8), (3, 7)} for _, x, y, _ in plan.furniture))   # the doorway

    def test_no_homes_store_rooms_upstairs_and_one_street_door(self):
        lay = self.lays["shop_stores_over"]
        b = R.to_building(lay, seed=1, uses=[("generalstore", "storage")])
        self.assertEqual(b.storeys[1].kind, "shop")
        self.assertTrue({r.kind for r in b.storeys[1].rooms} <= {"storage", "breakroom", "bathroom", "closet", "hall"})
        st = lay.storeys[0]
        street = [e for e in st["doors"] if e[2] == "N" and e[1] == lay.house.height]     # street S
        self.assertEqual(street, [lay.front_door])        # the shop's: the stair hall has none

    def test_the_waiting_room_has_its_own_street_door(self):
        lay = self.lays["surgery"]
        waiting = next(r for r in lay.storeys[0]["rooms"] if r["kind"] == "waiting")
        mine = {R._edge(x, y, "S") for (x, y) in waiting["squares"] if y == lay.house.height - 1}
        self.assertIn(lay.front_door, mine)

    def test_programmes_by_homes_and_class(self):
        self.assertEqual(R.shop_programme(1, 0, "retail"), "shop_only@retail")
        self.assertEqual(R.shop_programme(2, 1, "retail"), "shop_house")
        self.assertEqual(R.shop_programme(2, 1, "food"), "shop_house@food")
        self.assertEqual(R.shop_programme(3, 3, "pub"), "shop_flats2@pub")
        self.assertIsNone(R.shop_programme(6, 0, "retail"))
        self.assertEqual(R.use_class(("bar", "barstorage")), "pub")
        self.assertEqual(R.use_class(("butcher", "storage")), "retail")
        self.assertEqual(R.shop_use_of([("restaurantdining", "restaurantkitchen"), ("bar", "barstorage")]),
                         ("bar", "barstorage"))
        self.assertIsNone(R.shop_use_of([("butcher", "storage"), ("bank", "office")]))
        self.assertTrue(R.is_programme("shop_only@bank") and R.is_programme("shop_flats@food"))
        self.assertFalse(R.is_programme("shop_only@nothing"))
        self.assertNotEqual(R.programme_hash("shop_only@bank"), R.programme_hash("shop_only@retail"))


@unittest.skipUnless(HAVE_ORTOOLS, "OR-Tools not installed")
class ConvertedHouses(unittest.TestCase):
    """Option C: an office, surgery or guest house in a house (ROOMS.md section 17, step 2)."""
    @classmethod
    def setUpClass(cls):
        cls.lay = R.solve(*HOUSES["iw_semi"])

    def test_which_businesses_are_converted_houses(self):
        self.assertEqual(R.conversion({"look": "georgian_townhouse"}, ("office", "office")), "office")
        self.assertEqual(R.conversion({"look": "flats_period"}, ("dentist", "medicaloffice")), "surgery")
        self.assertEqual(R.conversion({"look": "vic_villa"}, ("lobby", "office")), "guest")
        self.assertIsNone(R.conversion({"look": "non_domestic"}, ("office", "office")))
        self.assertIsNone(R.conversion({"look": "vic_terrace"}, ("butcher", "storage")))
        self.assertIsNone(R.shop_programme(2, 0, "guest"))          # a guest house not in a house: Build's

    def test_a_surgery_in_a_house(self):
        b = R.to_building(self.lay, seed=1, uses=[("dentist", "medicaloffice")], converted="surgery")
        kinds = {r.kind for s in b.storeys for r in s.rooms}
        self.assertIn("lobby", kinds)                       # the front room is the waiting room
        self.assertIn("dentist", kinds)
        self.assertFalse(kinds & {"bedroom", "livingroom"})
        self.assertTrue(all(s.kind == "shop" for s in b.storeys))
        self.assertFalse(b.storeys[0].shop_front)           # house windows, no shop front

    def test_an_office_with_a_flat_over_it(self):
        b = R.to_building(self.lay, seed=1, uses=[("office", "office")], converted="office", homes=True)
        self.assertIn("lobby", {r.kind for r in b.storeys[0].rooms})
        self.assertIn("bedroom", {r.kind for r in b.storeys[1].rooms})       # the flat upstairs

    def test_a_guest_house_lets_its_bedrooms(self):
        b = R.to_building(self.lay, seed=1, uses=[("lobby", "office")], converted="guest", homes=True)
        upstairs = [r.kind for r in b.storeys[1].rooms]
        self.assertIn("motelroom", upstairs)
        self.assertIn("livingroom", {r.kind for r in b.storeys[0].rooms})   # the guests' lounge


class ShopTrades(unittest.TestCase):
    def test_a_vague_use_becomes_the_fascias_trade(self):
        from knoxbuild import regional_uk as U
        from knoxbuild.uses import uses_of
        overlay = {"buildings": {"7": {"trade": "butcher"}}}
        epc = {"shop": "retail", "knox_source": "epc"}
        osm = {"shop": "florist", "knox_source": "osm"}
        doctor = {"amenity": "doctors", "knox_source": "epc"}
        self.assertEqual(uses_of(U.uses_with(overlay, 7, [{}, epc])), [("butcher", "storage")])
        self.assertEqual(uses_of(U.uses_with(overlay, 7, [{}, osm, doctor])),
                         uses_of([{}, osm, doctor]))      # named uses and a surgery's are kept
        self.assertEqual(U.uses_with(None, 7, [epc]), [epc])
        # the building's own tags say what it is: its business rates' "retail" is it, not a butcher
        society = {"amenity": "bank", "name": "Newbury Building Society"}
        self.assertEqual(uses_of(U.uses_with(overlay, 7, [society, epc])), [("bank", "office")])


@unittest.skipUnless(HAVE_ORTOOLS, "OR-Tools not installed")
class BlocksOfFlats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lays = {n: R.solve(h, p) for n, (h, p) in HOUSES.items() if p == "flats_block"}

    def test_they_solve(self):
        for name, lay in self.lays.items():
            with self.subTest(name=name):
                self.assertIsNotNone(lay)
                self.assertEqual(R.check(lay), [])

    def test_each_flat_has_one_front_door_off_the_shared_hall(self):
        for name, lay in self.lays.items():
            for st in lay.storeys:
                flats = {r["unit"] for r in st["rooms"] if r.get("unit")}
                for u in flats:
                    halls = [r for r in st["rooms"] if r.get("unit") == u and r["kind"] == "flat_hall"]
                    with self.subTest(name=name, flat=u):
                        self.assertEqual(len(halls), 1)
                        self.assertEqual(st["rooms"][halls[0]["parent"]].get("unit", 0), 0)
                for r in st["rooms"]:
                    if r["parent"] is not None and r["kind"] != "flat_hall":
                        self.assertEqual(r.get("unit", 0), st["rooms"][r["parent"]].get("unit", 0))

    def test_each_flat_has_a_bathroom_and_a_living_room(self):
        for name, lay in self.lays.items():
            for st in lay.storeys:
                for u in {r["unit"] for r in st["rooms"] if r.get("unit")}:
                    kinds = {r["kind"] for r in st["rooms"] if r.get("unit") == u}
                    with self.subTest(name=name, flat=u):
                        self.assertIn("bathroom", kinds)
                        self.assertTrue(kinds & {"living", "openplan"})

    def test_floors_of_one_outline_have_the_same_flats(self):
        lay = self.lays["flats_conversion"]
        shapes = [sorted((r["kind"], sorted(r["squares"])) for r in st["rooms"] if r.get("unit"))
                  for st in lay.storeys[1:]]
        self.assertEqual(shapes[0], shapes[1])


@unittest.skipUnless(HAVE_ORTOOLS, "OR-Tools not installed")
class ReviewFixes(unittest.TestCase):
    """The faults the 2026-10-20 code review found."""

    def test_a_shop_front_survives_the_cache_and_the_mirror(self):
        import tempfile
        house, prog = HOUSES["shop_home"]
        lay = R.solve(house, prog)
        self.assertTrue(lay.storeys[0]["shop_front"])
        back = R._from_json(R._to_json(lay), house)
        self.assertEqual(sorted(back.storeys[0]["shop_front"]), sorted(lay.storeys[0]["shop_front"]))
        cache = R.LayoutCache(tempfile.mkdtemp())
        cache.put(house, prog, [lay])
        b = R.to_building(cache.options(house, prog)[0], uses=[("grocery", "grocerystorage")])
        self.assertTrue(b.storeys[0].shop_front)
        twin = R.mirror_house(house)
        flipped = R.mirror(lay, twin)
        if flipped:
            self.assertEqual(len(flipped.storeys[0]["shop_front"]), len(lay.storeys[0]["shop_front"]))

    def test_a_middle_floor_landing_joins_the_stairs(self):
        m = [[True] * 6 for _ in range(9)]
        house = R.House(6, 9, [m, m, m], "S", R._party(6, 9, "EW"))
        for prog in ("townhouse_period", "townhouse_modern"):
            lay = R.solve(house, prog)
            with self.subTest(prog=prog):
                self.assertIsNotNone(lay)
                self.assertEqual(R.check(lay), [])

    def test_a_house_whose_twin_will_not_mirror_is_solved_itself(self):
        import tempfile
        m = [[True] * 12 for _ in range(5)]
        house = R.House(12, 5, [m, m], "S", R._party(12, 5, "E"))
        prog = "starter_80s"
        cache = R.LayoutCache(tempfile.mkdtemp())
        lay = R.layout_for(house, prog, "a", cache)
        twin = R.mirror_house(house)
        if lay and all(R.mirror(x, twin) is None for x in cache.options(house, prog)):
            self.assertFalse(cache.known(twin, prog))
        self.assertIsNotNone(R.layout_for(twin, prog, "b", cache))

    def test_a_solve_out_of_time_is_not_cached_as_no_layout(self):
        import tempfile
        house, prog = HOUSES["iw_semi"]
        cache = R.LayoutCache(tempfile.mkdtemp())
        real = R.solve_varied

        def timed_out(*a, **k):
            R.TIMED_OUT[0] = True
            return []
        R.solve_varied = timed_out
        try:
            self.assertIsNone(R.layout_for(house, prog, "a", cache))
        finally:
            R.solve_varied = real
        self.assertFalse(cache.known(house, prog))

    def test_after_prepare_a_worker_does_not_solve_again(self):
        # what prepare could not solve keeps Build's own layout, the same every
        # run - a worker solving it again at the same budget was a coin toss
        # (the White Horse, Alton, LOG 2026-10-27 #2)
        import tempfile
        house, prog = HOUSES["iw_semi"]
        cache = R.LayoutCache(tempfile.mkdtemp())
        real, calls = R.solve_varied, []
        R.solve_varied = lambda *a, **k: calls.append(1) or []
        try:
            self.assertIsNone(R.layout_for(house, prog, "a", cache, solve_missing=False))
        finally:
            R.solve_varied = real
        self.assertEqual(calls, [])

    def test_a_pub_has_cupboards_to_fill_a_stepped_outline(self):
        fill = [r for r in R.SHOPS["grounds"]["pub"] if r["kind"] == "cupboard" and r.get("value") == 0]
        self.assertEqual(len(fill), 3)
        self.assertTrue(all(r.get("optional") for r in fill))

    def test_window_stacking_is_on_for_houses(self):
        self.assertNotIn("window_stack_ground", R.WEIGHTS)
        self.assertEqual(R.programme_tables("shop_house")[1]["window_stack_ground"], 0)


class WindowColumns(unittest.TestCase):
    def test_spacing_on_a_plain_front(self):
        mask = [[True] * 9 for _ in range(6)]
        house = R.House(9, 6, [mask], "S")
        front = [c for c in R.window_columns(house, 0) if c[2] == "S"]
        self.assertEqual([c[0] for c in front], [1, 4, 7])

    def test_short_wall_by_a_back_addition_gets_a_spot(self):
        house, _ = HOUSES["vic_mid_terrace"]
        cols = R.window_columns(house, 0)
        self.assertIn((3, 4, "N"), cols)


if __name__ == "__main__":
    unittest.main()
