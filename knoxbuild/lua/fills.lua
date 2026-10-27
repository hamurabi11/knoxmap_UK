-- Written into every map KnoxMap installs. Loot for fittings the game's tables
-- name only by tile.
--
-- A room's container is filled only for the tiles its loot lists name when every
-- list carries a forceForTiles. The game's "dining" table (home dining rooms use it:
-- SuburbsDistributions.diningroom = .dining) fills only bar counters and sinks on a
-- counter and only named shelf tiles; restaurant dining rooms and Chinese
-- restaurants the same on a counter. KnoxMap's dining-room shelves and counters and
-- its restaurant counters are other tiles, so they stayed empty. Each line below
-- adds one list with no forceForTiles: the container's ordinary loot wherever no
-- named tile matches (a named sink or bar still gets its own list).
--
-- tools/loot_check.py reads FILLS from this file: keep one plain {room, container,
-- list} per line. Every map's copy is the same file; running it twice changes nothing.

local FILLS = {
    {"dining", "shelves", "DishCabinetGeneric"},
    {"dining", "counter", "KitchenDishes"},
    {"restaurantdining", "counter", "StoreKitchenDishes"},
    {"chineserestaurant", "counter", "StoreKitchenDishes"},
}

local function addFills()
    if not SuburbsDistributions then return end
    for _, f in ipairs(FILLS) do
        local room = SuburbsDistributions[f[1]]
        local cont = room and room[f[2]]
        local lists = cont and cont.procList
        if lists then
            local open = false
            for _, entry in ipairs(lists) do
                if not entry.forceForTiles or entry.forceForTiles == "" then open = true end
            end
            if not open then
                table.insert(lists, {name = f[3], min = 0, max = 99})
            end
        end
    end
end

Events.OnPostDistributionMerge.Add(addFills)
