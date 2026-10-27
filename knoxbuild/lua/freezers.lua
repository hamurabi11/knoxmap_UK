-- Written into every map KnoxMap installs. Frozen food in the shop freezers.
--
-- A shop's freezer is filled only for the tiles its loot table names: every
-- list in SuburbsDistributions.gigamart.freezer (and grocery's) carries a
-- forceForTiles with the game's chest freezers (appliances_refrigeration_01_20,
-- 21, 38, 39). KnoxMap's supermarkets stand the game's upright shop freezers
-- (01_48 to 51), which only grocerystorage's freezer knows, so they stayed
-- empty. This adds those four tiles to the freezer lists of the rooms below,
-- once the game has merged its distribution tables and before it reads them.
--
-- tools/loot_check.py reads ROOMS and TILES from this file: keep them plain lists.
-- Every map's copy is the same file; running it twice changes nothing.

local ROOMS = {"gigamart", "grocery"}
local TILES = {"appliances_refrigeration_01_48", "appliances_refrigeration_01_49",
               "appliances_refrigeration_01_50", "appliances_refrigeration_01_51"}

local function hasTile(list, tile)
    for t in string.gmatch(list, "[^;]+") do
        if t == tile then return true end
    end
    return false
end

local function addFreezerTiles()
    if not SuburbsDistributions then return end
    for _, room in ipairs(ROOMS) do
        local freezer = SuburbsDistributions[room] and SuburbsDistributions[room].freezer
        local lists = freezer and freezer.procList
        if lists then
            for _, entry in ipairs(lists) do
                if type(entry.forceForTiles) == "string" then
                    local s = entry.forceForTiles
                    for _, tile in ipairs(TILES) do
                        if not hasTile(s, tile) then
                            if s ~= "" and string.sub(s, -1) ~= ";" then s = s .. ";" end
                            s = s .. tile
                        end
                    end
                    entry.forceForTiles = s
                end
            end
        end
    end
end

Events.OnPostDistributionMerge.Add(addFreezerTiles)
