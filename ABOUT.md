# KnoxMap UK

A fork of [KnoxMap](https://github.com/spytheeuclidean-a11y/knoxmap) for English
towns. KnoxMap turns real places into Project Zomboid maps. This fork adds a **UK
mode**: real road widths, buildings carried in from LiDAR, house types and ages from
open data, English garden boundaries, kerbs, road markings and street furniture.

Everything KnoxMap does is unchanged. The UK parts only come into play when you use
UK mode.

## Using it

1. Set up and start KnoxMap as usual (`Setup.bat`, then `KnoxMap.bat`).
2. Click **UK mode** in the top bar. The map moves to London and the UK panel opens.
3. Under **Data**, see which optional datasets you have. Each has a download link and
   says what it adds.
4. Draw a box on the map, give the area a name, choose **Full** or **OSM and LiDAR
   only**, then click **Set up the area**. This downloads OpenStreetMap, the
   Environment Agency's LiDAR and anything else needed.
5. Choose your options and click **Run UK pipeline**. The panel explains each option
   and how long it adds.
6. When it finishes, the map opens in the usual Compile and Install cards.

`KnoxMap.bat --uk` starts with UK mode on.

From the command line, the same two steps are `uk/src/area/setup_area.py` and
`uk/src/uk_pipeline.py`. Both are run with KnoxMap's environment, from this folder.

## What is where

| Folder | What |
|---|---|
| the root | KnoxMap, with UK mode wired in: `web/uk.py`, `static/js/uk.js` and `knoxbuild/*_uk.py`, plus small hooks in Build, fences and the compile tools |
| `uk/asset_creator/` | the UK assets' code and tables: listed buildings, shop signs and fronts, and the tile pack's registration with the map tools (the art itself is the release download below) |
| `uk/` | the UK pipeline: `src/` (area data, roads, buildings, gardens, regional styles), `palettes/english.json`, and `README.md`: every step, its script and what it makes |
| `uk/data/` | each area's data, made by the setup (not committed), and `raw/` for the national downloads |

## Data

With **OSM and LiDAR only**, nothing has to be downloaded by hand: OpenStreetMap
and the Environment Agency's LiDAR are fetched for each area. Each of these adds
more, saved where the Data section says:

| Dataset | Adds | Where to get it |
|---|---|---|
| OS Open Roads | road classes, names and numbers | <https://osdatahub.os.uk/downloads/open/OpenRoads> |
| OS Open UPRN | every address: homes per building, garden plots | <https://osdatahub.os.uk/downloads/open/OpenUPRN> |
| EPC certificates, domestic and non-domestic (free account) | house ages and types; shops and offices | <https://epc.opendatacommunities.org/> |
| HM Land Registry INSPIRE Index Polygons | garden plots from land titles | <https://use-land-property-data.service.gov.uk/datasets/inspire/download> |

Historic England's listed buildings, Food Standards Agency ratings and the council
lookup (postcodes.io) are fetched automatically at the Full level. The land-title
zips go in `uk/data/raw/inspire` (or set `KNOXMAP_INSPIRE_DIR`).
`uk/SOURCES.md` lists every source, its licence and the attribution it needs. Maps
made from these sources carry those obligations, so credit them when you share a map:
OS, HM Land Registry, Historic England, the Environment Agency, MHCLG (EPC) and
OpenStreetMap contributors.

## UK assets

The UK tile pack (the KnoxMapUK_Tiles mod) holds the English walls, windows, doors,
shop fronts, kerbs, road markings, signs and street furniture. It is a download on
this repository's [releases](https://github.com/hamurabi11/knoxmap_UK/releases), not
in the code: in the UK panel, under **Data**, click **Download and install**. It is
fetched, checked against its published checksum and put in the game's mods folder,
and its ground tiles are registered with the map tools.

- **With it**, keep **UK assets** ticked. Maps made that way need the KnoxMapUK_Tiles
  mod (installed by the button) in the game.
- **Without it**, untick **UK assets**. Maps then use vanilla and Erika's tiles only,
  with each English style replaced by its nearest vanilla look, and need no extra
  mod.

The pack is original art: drawn by script, from the authors' own photos and public
domain (CC0) texture scans, with nothing taken or traced from the game's art. It is
licensed **CC BY 4.0**: credit "KnoxMap UK Tiles, CC BY 4.0" with a link here.

## Licence

KnoxMap's code is under its own licence (`LICENSE-MIT.txt`, `LICENSES.md`), and so are
this fork's changes to it. The UK tile pack (the release download) is CC BY 4.0.
The data sources carry their own licences (see above).
Project Zomboid and its art belong to The Indie Stone. None of it is in this
repository.
