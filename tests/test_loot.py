"""Every fitting Build stands in a room gets loot (tools/loot_check.py, knoxbuild/loot_uk.py):
its container type is one the game's loot table for that room fills, or the `all`
fallback does. Needs the game installed (skipped otherwise)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import loot_check as LC  # noqa: E402

from knoxbuild import catalog as C  # noqa: E402
from knoxbuild import interiors as I  # noqa: E402
from knoxbuild import layout as L  # noqa: E402
from knoxbuild import loot_uk  # noqa: E402


def kinds_of(role):
    return {c for tiles in C.FURNITURE.get(role, {}).values() for t in tiles.values() if (c := LC.container_of(t))}


@unittest.skipUnless(LC.available(), "the game is not installed")
class Loot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loot_uk.apply()

    def unfilled(self, room, roles):
        return sorted((r, c) for r in roles if r for c in kinds_of(r) if LC.filled(room, c) == "none")

    def test_shop_fittings(self):
        for kind, spec in I.STORES.items():
            roles = {spec.get("rows"), spec.get("counter")} | {r for key in ("wall", "back", "front", "aisles")
                                                               for r in spec.get(key, [])}
            with self.subTest(kind=kind):
                self.assertEqual(self.unfilled(kind, roles), [])

    def test_a_cafes_counter(self):
        self.assertEqual(self.unfilled("cafe", I.BACK_COUNTERS["cafe"]), [])

    def test_room_wishlists(self):
        for kind, (_floor, _label, wish) in L.ROOM_STYLE.items():
            with self.subTest(kind=kind):
                self.assertEqual(self.unfilled(L.ROOM_NAME.get(kind, kind), wish), [])

    def test_a_superstores_fittings(self):
        from knoxbuild import superstore_uk as S
        w, h = 60, 45
        mask = [[True] * w for _ in range(h)]
        roof = S.roof(0, 0, mask, w, h, "S", None)
        b = S.layout(w, h, mask, "S", roof["level_masks"], [("grocery", "grocerystorage")], seed=3)
        for st in b.storeys:
            for role, x, y, _o in st.furniture:
                room = st.rooms[st.grid[y][x] - 1].kind
                with self.subTest(room=room, role=role):
                    self.assertEqual(self.unfilled(L.ROOM_NAME.get(room, room), [role]), [])

    def test_shop_freezers_are_filled(self):
        # (their tables fill only the freezers their forceForTiles names: knoxbuild/lua/freezers.lua
        # adds the game's upright shop freezers, which our superstores stand)
        for room in ("gigamart", "grocery", "grocers"):
            for tile in C.FURNITURE["shop_freezer"].values():
                for sprite in tile.values():
                    with self.subTest(room=room, sprite=sprite):
                        self.assertNotEqual(LC.filled_tile(room, sprite), "miss")

    def test_force_for_tiles_is_read(self):
        # the game's own Gigamart freezer list knows its chest freezers only
        self.assertIn("appliances_refrigeration_01_20", LC.forced()[("gigamart", "freezer")])
        self.assertEqual(LC.container_of("appliances_refrigeration_01_49"), "freezer")


if __name__ == "__main__":
    unittest.main()
