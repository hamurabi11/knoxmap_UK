"""What each listing feature needs to be built, and whether we have it.

A lookup table, feature -> Asset. Each asset is in one of these states:
  uk        ours, in the pack and laid by Build (asset_creator README "Done")
  uk_sheet  ours, a sheet drawn (Tile_creator/build/sheets/<group>/) but not wired into
            the palette or Build yet
  vanilla   a vanilla tile stands in (fine, or until ours is made)
  build     Build already makes it (geometry, not art)
  missing   nothing: the building is made without it

`check()` confirms each named sheet is on disk (ours in Tile_creator/build/
sheets, vanilla's in the map tools' Tiles/2x), so the table cannot claim a
sheet that is not there. `recipe` is the command that draws an asset where a
generator exists (`survey.py --make` runs it when its sheet is missing); every
other missing asset gets a brief (briefs/<asset>.md) for the asset work.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "Tile_creator", "src")))
from sheet_paths import sheet_file  # noqa: E402  (our sheets: Tile_creator/build/sheets/<group>/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # asset_creator: acpaths
import acpaths  # noqa: E402
VANILLA = acpaths.TOOLS_TILES


@dataclass
class Asset:
    id: str
    what: str
    state: str
    sheets: tuple = ()          # sheets that hold it now (ours or vanilla's)
    stand_in: str = ""          # what Build lays meanwhile
    recipe: str = ""            # the generator, where one exists
    plan: str = ""              # where the work is planned


A = Asset
MK = "python listed_building_maker/makers/{}   (from asset_creator/, KnoxMap's Python)"
WALLS, ROOFS, WINDOWS = MK.format("wall_kit.py"), MK.format("roof_kit.py"), MK.format("window_kit.py")
DOORS, OBJECTS, TRIMS = MK.format("doorway_kit.py"), MK.format("structure_kit.py"), MK.format("trim_kit.py")
KIT = "walls_house_uk_01_"
TABLE = [
    # walls: makers/wall_kit.py, a whole palette style per material (wall set,
    # 16 window-shaped pieces, gable caps), chosen by design.py
    A("wall.red_brick", "red brick wall set, Flemish bond, rubbed flat arches, stone cills", "uk",
      (KIT + "brick_flemish",), "kit brick_flemish (after 1840: palette victorian)", WALLS),
    A("wall.painted_brick", "painted brick wall set", "uk", (KIT + "brick_painted",), "kit brick_painted", WALLS),
    A("wall.render", "lime render / stucco (white, cream), painted cills", "uk", (KIT + "render_white", KIT + "render_cream"),
      "kit render_white / render_cream", WALLS),
    A("wall.flint", "flint with brick arches and cills", "uk", (KIT + "flint",), "kit flint", WALLS),
    A("wall.tile_hung", "clay tile-hanging", "uk", (KIT + "tile_hung",), "kit tile_hung", WALLS),
    A("wall.timber_frame", "exposed close-studded frame, lime render or brick-nogged infill", "uk",
      (KIT + "timber_white", KIT + "timber_brick"), "kit timber_white / timber_brick", WALLS),
    A("wall.weatherboard", "black (barns) and white weatherboarding", "uk",
      (KIT + "weatherboard_black", KIT + "weatherboard_white"), "kit weatherboard_*", WALLS),
    A("wall.stone", "stone and chalk / clunch walling", "vanilla", ("walls_exterior_house_02",), "palette stone"),
    A("wall.blue_brick", "chequer brick: blue headers in red", "uk", (KIT + "brick_chequer",), "kit brick_chequer", WALLS),
    # roofs: makers/roof_kit.py, every vanilla roof piece in our covering
    A("roof.slate", "Welsh slate roof", "uk", ("roofs_uk_01_slate", "roofs_30_uk_01_slate"), "roof_kit slate", ROOFS),
    A("roof.plain_tile", "clay plain tile roof", "uk", ("roofs_uk_01_tile", "roofs_30_uk_01_tile"), "roof_kit tile", ROOFS),
    A("roof.thatch", "long-straw / reed thatch", "uk", ("roofs_uk_01_thatch", "roofs_30_uk_01_thatch"), "roof_kit thatch", ROOFS),
    A("roof.stone_slate", "stone slate roof", "uk", ("roofs_uk_01_slate",), "roof_kit slate", ROOFS),
    A("roof.hipped", "hipped roof ends", "build", (), "a 30-degree hip (Peak30Quad) where the game can build one "
      "(both ends free, odd width up to 7, odd length: regional_uk._gable hip), else gables",
      plan="45-degree and wider hips: knoxlots / BuildingEd have no piece"),
    A("roof.half_hipped", "half-hipped (jerkin-head) ends", "missing", (), "gables"),
    A("roof.mansard", "mansard roof", "missing", (), "a pitched roof"),
    A("roof.catslide", "catslide over a rear outshot", "build", (), "roof_plan lean-tos"),
    A("roof.dormer", "dormers: flat-roofed, gabled, hipped", "build", ("walls_exterior_roofs_03",),
      "box dormers on attics (roof_plan add_attics)", plan="PLAN C2: gabled dormers"),
    A("roof.chimney", "brick chimney stacks and pots", "vanilla", ("fixtures_fireplaces_01",), "fixtures_fireplaces_01 64-73"),
    A("roof.parapet", "parapet with stone coping in front of the roof", "uk", ("wall_trim_uk_01",),
      "trim_kit parapet, on the roof's floor along the top storey's outside walls", TRIMS),
    A("roof.lead", "lead flat or dome", "missing", (), "flat roof"),
    # windows: makers/window_kit.py (closed, open, smashed, glass gone)
    A("window.sash", "white sashes: Georgian 6-over-6, Victorian 2-over-2", "uk", ("fixtures_windows_uk_01",),
      "window_kit sash_georgian / sash_victorian", WINDOWS),
    A("window.casement", "small-pane timber casements", "uk", ("fixtures_windows_uk_01",), "window_kit casement", WINDOWS),
    A("window.leaded", "leaded lights in oak casements", "uk", ("fixtures_windows_uk_01",), "window_kit casement_leaded", WINDOWS),
    A("window.mullion", "stone mullion-and-transom windows", "uk", ("fixtures_windows_uk_01",), "window_kit mullion", WINDOWS),
    A("window.round_headed", "round-headed sashes", "uk", ("fixtures_windows_uk_01",), "window_kit sash_arched", WINDOWS),
    A("window.bay", "canted / square / bow bay windows", "missing", (), "flat windows (Build makes no bays)"),
    A("window.oriel", "oriel windows", "missing", (), "flat windows"),
    A("window.venetian", "Venetian window", "missing", (), "flat windows"),
    A("window.gothic", "pointed (Gothic) windows with tracery", "missing", (), "flat windows", plan="PLAN C2: church windows"),
    A("window.french", "French windows", "vanilla", ("fixtures_doors_01",), "glazed doors"),
    # doors: doors_uk_01 (the street furniture work); doorways: makers/doorway_kit.py
    A("door.six_panel", "six-panelled front door", "uk", ("doors_uk_01",), "doors_uk_01 hardwood_fanlight"),
    A("door.fanlight", "door with fanlight", "uk", ("doors_uk_01",), "doors_uk_01 hardwood_fanlight"),
    A("door.four_panel", "four-panelled door", "uk", ("doors_uk_01",), "doors_uk_01 Victorian four-panel"),
    A("door.panelled", "panelled door", "uk", ("doors_uk_01",), "doors_uk_01 Victorian four-panel"),
    A("door.boarded", "boarded and ledged door", "uk", ("doors_uk_01",), "doors_uk_01 oak_cottage"),
    A("door.doorcase", "doorcase: pilasters, entablature, pediment", "uk", ("doorways_uk_01",), "doorway_kit doorcase", DOORS),
    A("door.pediment", "pediment over a door", "uk", ("doorways_uk_01",), "doorway_kit doorcase", DOORS),
    A("door.porch", "porch: columns and a flat roof, or oak posts and a tiled gable", "uk", ("doorways_uk_01",),
      "doorway_kit porch / porch_timber", DOORS),
    A("door.hood", "door hood on brackets", "uk", ("doorways_uk_01",), "doorway_kit hood", DOORS),
    # trim
    A("trim.cornice", "eaves cornice", "uk", ("wall_trim_uk_01",), "trim_kit cornice, along the top storey's outside walls", TRIMS),
    A("trim.band", "first-floor band / string course", "uk", ("wall_trim_uk_01",), "trim_kit band, along the first floor's foot", TRIMS),
    A("trim.plinth", "plinth", "uk", ("wall_trim_uk_01",), "trim_kit plinth, along the ground floor's foot", TRIMS),
    A("trim.quoins", "quoins", "missing", (), "plain wall"),
    A("trim.pilasters", "pilasters and columns", "missing", (), "plain wall (a doorcase has its own: door.doorcase)"),
    A("trim.flat_arches", "rubbed / cambered brick arches over windows", "uk", (KIT + "brick_flemish",),
      "drawn into the brick and flint kits' openings", WALLS),
    A("trim.cills", "stone cills", "uk", (KIT + "brick_flemish",), "drawn into the kits' window openings", WALLS),
    A("trim.bargeboards", "decorative bargeboards", "missing", (), "plain gable"),
    A("trim.jetty", "jettied upper floor", "missing", (), "a flush upper floor (Build: storeys share a footprint)"),
    A("trim.balcony", "wrought-iron balconies", "missing", (), "none"),
    A("trim.railings", "wrought / cast-iron railings and guard rails", "uk", ("walls_garden_uk_01",), "fences.py railings"),
    A("trim.sign_bracket", "inn sign on a wall bracket", "uk", ("pub_signs_uk_01",), "pub sign on a post (pubs.py)"),
    # ground floor
    A("ground.shopfront", "Georgian / Victorian timber shopfront (fascia, pilasters, stall riser)", "vanilla",
      (), "Erika's storefronts in heritage colours", plan="PLAN E: shop fronts"),
    A("ground.carriage_entry", "carriage entry through the building", "build", (), "regional passage (arch)"),
    # churches
    A("church.tower", "west tower", "build", (), "lidar_outlines tower, hall roof", plan="PLAN C2: church details"),
    A("church.spire", "spire", "missing", (), "tower top"),
    A("church.buttress", "buttresses", "missing", (), "plain wall"),
    A("church.battlements", "battlements", "missing", (), "plain parapet", plan="PLAN C2"),
    A("church.bellcote", "bellcote / lantern / clock", "missing", (), "none"),
]
# Structures (not buildings): what an entry of this kind is made as.
STRUCTURES = [
    A("structure.wall", "listed boundary wall, gate piers, gates", "uk", ("walls_garden_uk_01",),
      "fences.py brick walls where OSM maps the wall (a list entry has only a point)"),
    A("structure.kiosk", "K6 telephone kiosk", "uk", ("street_furniture_uk_01",), "furniture.py kiosks"),
    A("structure.memorial", "war memorial: wheel-head cross on a stepped base", "uk", ("listed_structures_uk_01",),
      "structure_kit memorial (apply.py stands it at its point)", OBJECTS),
    A("structure.tomb", "chest and table tombs, headstones", "missing", ("location_community_cemetary_01",),
      "vanilla gravestones"),
    A("structure.milestone", "milestone", "uk", ("listed_structures_uk_01",), "structure_kit milestone", OBJECTS),
    A("structure.pump", "village pump", "missing", (), "none"),
    A("structure.stocks", "village stocks and whipping post", "uk", ("listed_structures_uk_01",), "structure_kit stocks", OBJECTS),
    A("structure.bridge", "brick canal / river bridge", "build", (), "Generate's bridges"),
    A("structure.lock", "canal lock", "missing", (), "none"),
]
BY_ID = {a.id: a for a in TABLE + STRUCTURES}


def sheet_on_disk(sheet: str) -> str | None:
    """'ours' or 'vanilla' where the sheet is found, else None."""
    if os.path.exists(sheet_file(sheet + ".png")):
        return "ours"
    if os.path.isdir(VANILLA) and any(n.startswith(sheet) for n in os.listdir(VANILLA)):
        return "vanilla"
    return None


def check() -> list[str]:
    """The table's claims against the disk: a list of problems (empty if none)."""
    bad = []
    for a in BY_ID.values():
        for s in a.sheets:
            if sheet_on_disk(s) is None:
                bad.append(f"{a.id}: sheet {s} not found")
    return bad


if __name__ == "__main__":
    probs = check()
    print("\n".join(probs) or f"all {sum(len(a.sheets) for a in BY_ID.values())} sheets found")
