"""superstore_uk and pillars_uk: a superstore's inside from the game's Gigamarts, and
pillars in big open rooms (ROOMS.md section 18)."""
import collections
import unittest

from knoxbuild import layout as L
from knoxbuild import pillars_uk as P
from knoxbuild import superstore_uk as S


def store(w, h, front="S", uses=(("grocery", "grocerystorage"),)):
    mask = [[True] * w for _ in range(h)]
    roof = S.roof(0, 0, mask, w, h, front, None)
    return roof, S.layout(w, h, mask, front, roof["level_masks"], list(uses), seed=3)


def reachable(plan):
    """Rooms reached from room 1 through doors and open edges (pillars aside)."""
    g = plan.grid
    links = collections.defaultdict(set)
    for x, y, d in list(plan.doors) + list(plan.open_edge):
        a, b = ((x - 1, y), (x, y)) if d == "W" else ((x, y - 1), (x, y))
        if all(0 <= q[1] < len(g) and 0 <= q[0] < len(g[0]) for q in (a, b)):
            ra, rb = g[a[1]][a[0]], g[b[1]][b[0]]
            links[ra].add(rb)
            links[rb].add(ra)
    start = next(i for i, r in enumerate(plan.rooms, 1) if r.kind == "gigamart")
    seen, todo = set(), [start]
    while todo:
        r = todo.pop()
        if r in seen:
            continue
        seen.add(r)
        todo += links[r]
    return seen


class Superstore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roof, cls.big = store(60, 45)

    def test_one_storey_and_a_staff_floor_over_the_back(self):
        self.assertEqual(self.roof["levels"], 2)
        upper = sum(map(sum, self.roof["level_masks"][1]))
        self.assertTrue(0 < upper < 60 * 45 / 3)
        kinds = {r.kind for r in self.big.storeys[1].rooms}
        self.assertTrue({"breakroom", "office", "lockerroom", "bathroom"} <= kinds)
        self.assertEqual(len(self.big.stairs), 1)

    def test_a_superstore_has_its_departments(self):
        kinds = collections.Counter(r.kind for r in self.big.storeys[0].rooms)
        for k in ("gigamart", "butcher", "bakery", "gigamartkitchen", "grocerystorage",
                  "clothingstore", "changeroom", "liquorstore", "housewarestore", "pharmacy", "cafe", "bathroom"):
            self.assertIn(k, kinds, k)

    def test_its_fittings(self):
        got = collections.Counter(f[0] for f in self.big.storeys[0].furniture)
        # the game's checkout units (raw sprites): belts and tills
        got["belt"] = sum(n for r, n in got.items() if r.startswith("uk_t:industry_02_4"))
        got["till"] = sum(n for r, n in got.items() if r.startswith("uk_t:location_shop_accessories_01"))
        for role, least in (("belt", 6), ("till", 3), ("uk_trolley_rail", 4), ("shop_shelf", 20),
                            ("uk_atm", 1), ("uk_bin", 4), ("uk_coffee", 1),
                            ("shop_bin", 4), ("uk_produce_stand", 1), ("shop_freezer", 1), ("clothes_rack", 1),
                            ("metal_rack", 4)):
            self.assertGreaterEqual(got[role], least, role)
        self.assertTrue(self.big.storeys[0].shop_front)

    def test_every_room_can_be_reached(self):
        ground = self.big.storeys[0]
        rooms = {i for i, r in enumerate(ground.rooms, 1) if r.kind != P.KIND}
        self.assertEqual(rooms - reachable(ground), set())

    def test_no_pillars_on_the_sales_floor(self):
        # (a pillar's walls flicker under B42's cutaway as you walk round it)
        self.assertEqual(sum(r.kind == P.KIND for r in self.big.storeys[0].rooms), 0)

    def test_a_smaller_store_is_a_supermarket(self):
        _roof, small = store(36, 30, "E")
        kinds = {r.kind for r in small.storeys[0].rooms}
        self.assertIn("butcher", kinds)
        self.assertNotIn("clothingstore", kinds)

    def test_a_booze_aisle(self):
        kinds = collections.Counter(r.kind for r in self.big.storeys[0].rooms)
        self.assertGreaterEqual(kinds["liquorstore"], 1)

    def test_an_outcrop_is_a_concession(self):
        # a 70 x 60 store, a 5 x 4 bump on its front well away from its doors
        w, h = 70, 64
        mask = [[y < 60 or 60 <= x < 65 for x in range(w)] for y in range(h)]
        roof = S.roof(0, 0, mask, w, h, "S", None)
        b = S.layout(w, h, mask, "S", roof["level_masks"], [], seed=0)
        g = b.storeys[0].grid
        kinds = {b.storeys[0].rooms[g[y][x] - 1].kind for y in range(60, 64) for x in range(60, 65)}
        self.assertEqual(len(kinds), 1)
        self.assertIn(kinds.pop(), {k for _t, k in S.CONCESSIONS})


class Reach(unittest.TestCase):
    def test_every_fitting_can_be_reached(self):
        for w, h, front in ((60, 45, "S"), (80, 60, "E"), (36, 30, "N")):
            _roof, b = store(w, h, front)
            self.assertEqual(S.unreachable(b.storeys[0]), [], (w, h, front))

    def test_no_light_switch_in_a_pillar_and_none_covered(self):
        from knoxbuild import catalog as C
        _roof, b = store(80, 60)
        g = b.storeys[0]
        pillars = {i for i, r in enumerate(g.rooms, 1) if r.kind == P.KIND}
        switches = [(x, y) for r, x, y, o in g.furniture if r == L.SWITCH]
        self.assertTrue(switches)
        self.assertFalse([c for c in switches if g.grid[c[1]][c[0]] in pillars])
        solid = {c for r, x, y, o in g.furniture if C.FURNITURE_LAYERS.get(r, "Furniture") == "Furniture"
                 and r in C.FURNITURE and o in C.FURNITURE[r] for c in L._cells_for(r, x, y, o)}
        self.assertFalse(set(switches) & solid)


class Front(unittest.TestCase):
    def test_glazed_bays_piers_and_double_doors(self):
        mask = [[True] * 70 for _ in range(50)]
        sf = S.store_front(mask, 70, 50, "S")
        glass, doors = S.front_glass(sf["cells"], sf["ad"], "S", sf["door_a"], sf["d_check"])
        a = sorted(sf["ad"][c][0] for c in glass)
        self.assertTrue(a and all(x % S.BAY for x in a))                          # a pier every BAY
        self.assertTrue(all(abs(x - sf["door_a"]) > S.SHOPFRONT_HALF for x in a))  # the entrance's own front
        self.assertTrue(doors and set(doors) <= set(glass))

    def test_the_cafe_has_a_servery_and_a_walkway(self):
        _roof, b = store(60, 45)
        g = b.storeys[0]
        kinds = {r.kind for r in g.rooms}
        self.assertIn("cafekitchen", kinds)
        cafe = next(i for i, r in enumerate(g.rooms, 1) if r.kind == "cafe")
        self.assertEqual(g.rooms[cafe - 1].landmark_floor, S.CAFE_FLOOR)
        kitchen = next(i for i, r in enumerate(g.rooms, 1) if r.kind == "cafekitchen")
        # the kitchen's door opens into the café
        sides = set()
        for x, y, d in g.doors:
            a, b2 = ((x - 1, y), (x, y)) if d == "W" else ((x, y - 1), (x, y))
            rooms = {g.grid[q[1]][q[0]] for q in (a, b2) if 0 <= q[1] < len(g.grid) and 0 <= q[0] < len(g.grid[0])}
            if kitchen in rooms:
                sides |= rooms - {kitchen}
        self.assertEqual(sides, {cafe})


class Pillars(unittest.TestCase):
    def plan(self, kind, w=30, h=30):
        grid = [[1] * w for _ in range(h)]
        return L.Plan(width=w, height=h, rooms=[L.Room(0, 0, w - 1, h - 1, kind=kind)], grid=grid, drawn=True)

    def test_a_warehouse_floor_gets_pillars_on_a_grid(self):
        plan = self.plan("warehouse")
        self.assertGreaterEqual(P.add(plan), 4)
        spots = [(r.x0, r.y0) for r in plan.rooms if r.kind == P.KIND]
        xs = sorted({x for x, _ in spots})
        self.assertTrue(all(b - a == P.PITCH for a, b in zip(xs, xs[1:])))

    def test_none_in_a_sports_hall_or_a_small_room(self):
        self.assertEqual(P.add(self.plan("gym")), 0)
        self.assertEqual(P.add(self.plan("gigamart")), 0)
        self.assertEqual(P.add(self.plan("warehouse", 10, 10)), 0)

    def test_none_on_furniture_or_in_a_doorway(self):
        plan = self.plan("warehouse")
        plan.furniture = [("crate", x, y, "N") for x in range(30) for y in range(30)]
        self.assertEqual(P.add(plan), 0)


if __name__ == "__main__":
    unittest.main()
