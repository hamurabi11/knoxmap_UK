-- KnoxMap UK: the storeys of a tall building above the sixteenth drawn from the street.
--
-- Build 42's native line of sight (Lighting64.dll, LosUtil.lineClear) only
-- traces squares from 16 levels below the player to 15 above; past that a
-- square is Blocked, never seen, and draws black - a 31-storey tower stands as
-- a silhouette from z 16 up until the player has been up there (vanilla's
-- Louisville towers too). A square already marked seen is sent to the native
-- side as seen (LightingJNI.updateChunk), so it draws as remembered ground
-- does: dimmed, not black. This marks every square loaded at z 16 or above
-- seen for each player. Copied into a map's mod by tools/make_map_mod.py only
-- when the map has a building that tall (Westminster: Parliament's towers).

local HIGH_Z = 16

local function markSeen(square)
    if square and square:getZ() >= HIGH_Z then
        for p = 0, 3 do
            square:setIsSeen(p, true)
        end
    end
end

Events.LoadGridsquare.Add(markSeen)
