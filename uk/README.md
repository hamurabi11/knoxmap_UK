# Building types

Works out what type each building is (Tier 1 in [../PLAN.md](../PLAN.md)),
from open data, so the right kit can be rolled out on the right streets.
Everything is keyed by **OSM way id** and measured in British National Grid
metres.

## Code layout

The scripts in `src/` are grouped by the stage of the pipeline they belong to.
Each folder is a package (its `__init__.py` says what it holds); scripts import
each other by folder, e.g. `from buildings import blocks as B`, and every one
is still run on its own as `src/<folder>/<script>.py`.

| Folder | What is in it | Scripts |
|---|---|---|
| `src/area/` | **Area data**, made once per area before any map: listings, property points, EPC, FSA ratings, INSPIRE plots, the point cloud, per-building evidence and the LiDAR roofs. `roofs.py` is also the LiDAR reader (`mosaic`) the map steps use. | `fetch_nhle`, `filter_uprn`, `features`, `extract_epc`, `assign_types`, `roofs`, `nlp_crop`, `fetch_inspire`, `fetch_fsa`, `fetch_road_nodes`, `render_map` |
| `src/streets/` | **Streets**: real widths written into the OSM download before Generate; after it, kerbs that follow the road, 45° roads redrawn, and the garden fronts along them. `uk_road_rules.py` (once, on the map tools) draws carriageways in vanilla's dark asphalt, street2. | `street_widths`, `kerbs`, `diagonals`, `fronts`, `measure_kerbs`, `topdown`, `uk_road_rules` |
| `src/buildings/` | **Buildings into the map**: outlines carried in whole from the LiDAR, landmark outlines, units cut from the address points, and what each building is (shops, homes, which way it faces). | `blocks`, `lidar_outlines`, `split_units`, `addresses`, `shops`, `tidy_outlines` |
| `src/ground/` | **The ground round the buildings**: square garden plots and their boundaries, trees from the point cloud, sports pitches and playgrounds. | `plots2`, `plotlib`, `inspire` (HM Land Registry titles as evidence for the plots, `plots2.py --inspire-attrs`), `trees`, `uk_tree_rules`, `sports` |
| `src/regional/` | **What Build reads** from the map's `regional.json`: styles and storeys per building, and roofs planned as ranges (with attics). | `export_regional`, `roof_plan` |
| `src/checks/` | **Checks and pictures** on a built map (see Checks, below). | `roof_float_check`, `roof_cover_check`, `flat_audit`, `roof_stats`, `frontage_measure`, `road_audit`, `road_links`, `road_width_check`, `road_overlap`, `lidar_check`, `blocks_check`, `boundary_score`, `fence_check`, `v_check`, `tile_defs`, `multitile_check` |
| `src/roadgraph/` | **Roads as a graph** (branch `road-graph`; see The road graph, below): OSM's roads routed on vanilla's sixteen directions, rasterised, handed over to KnoxMap as a new map; then markings, signs and street furniture. | `network`, `osroads`, `widths`, `roadside`, `markings`, `decals` (UK markings of one and two tiles), `furniture`, `town_centre` (bus stops, bins, cabinets, pub signs, railing), `snap`, `templates`, `raster`, `style`, `run`, `handover` |
| `src/demos/` | **Demos and prototypes** kept for reference; not part of the pipeline. | `demo_roof_examples`, `demo_roof_parts`, `kerb_proto` |

Shared helpers live where they started: `buildings/blocks.py` (`Ground`,
`Street`, `load_streets`, `rect_cover`, the BNG transforms) and
`ground/plotlib.py` (the plot rules' constants, `Heights`, `raster`, the OSM
features where Generate put them, footpaths) is what is left of the old raster
plot steps (`plots.py`, `boundaries.py`, deleted 2026-10-13). `tidy_outlines.py`
was the old step 3, kept for its helpers.

## The road graph

`src/roadgraph/` (branch `road-graph`; the design is `../new_road_approach.md`, which says what changed in the build). The maps in use (`Odiham_town_graph`, `Alton_town_graph`) are made by it.

- **The graph** (`network.py`): OSM's roads as junction-to-junction links, generalised (turning loops, dual carriageways, lanes hugging a bigger road); classes checked against OS Open Roads (`osroads.py` extracts the area's links from the national GeoPackage into `data/<area>/os_roads.geojson`); widths measured from the LiDAR (`widths.py`).
- **Routing** (`snap.py`, `templates.py`): on vanilla B42's sixteen directions on tile corners, each link one of vanilla's own street shapes where one fits (from the game's `streets.xml`), else built from vanilla's turns. `style.py` measures how a town's roads are drawn against vanilla's and scores the gap.
- **Rasterising** (`raster.py`): every edge - the carriageway's and each roadside band's - on the lattice through tile corners in its direction (45° through corners; 2:1 two tiles along to one across, `raster.lattice_k`), so angled roads are kerbed straight.
- **Roadside** (`roadside.py`): each side of each road its parking bays, grass verge and pavement (OSM's sidewalk and parking tags, the LiDAR's verge-then-pavement pattern, a pavement by default on town streets), each band an exact outline offset from the centreline (`roadside.outlines`, `raster.outline(..., extra=)`).
- **Markings and furniture** (`markings.py`, `decals.py`, `furniture.py`, `town_centre.py`): UK markings and signals from the graph and `road_nodes.geojson` (centre lines, give-ways, double yellows, zebras, signal crossings; the one- and two-tile markings), signs, lamps, post boxes, kiosks, bus stops and the rest, handed to `kerbs.py` as `<map>_markings.json`.
- **Hand-over** (`handover.py`): a whole map from the graph - a new map generated with the graph's lines in place of straightening, its roads painted from the graph's tiles with their verges and pavements (`<map>_roadside.npz`, read by `kerbs.py` and `plots2.py`), everything else moved with the roads by the graph file's displacement samples. Each later step that straightens roads reads the map's own graph from its info.json (`straighten_roads(..., map_dir=)`). `run.py` runs the graph on a square of a built map and draws it beside the map as it is.

## Pipeline

In the app: the top bar's **UK mode** button (or `Knoxmap/KnoxMap.bat --uk`) shows the UK panel (PLAN U): set up an area from the box drawn on the map (setup_area.py, with the data level), then run the UK chain on its map with every option (uk_pipeline.py), with progress and Stop; the map then opens in the usual Build, Compile and Install cards. The command lines below do the same.

Run with the KnoxMap environment: `Knoxmap/.venv/Scripts/python.exe`, from `Knoxmap/`.
`<map>` is the KnoxMap map folder (`Knoxmap/output/<name>`); `data/<area>` is
the area's data folder here (e.g. `data/odiham_core`).

### Area data (once per area)

Each area has a settings file, `data/<area>/area.json` (`src/area/area_conf.py`): `{"name", "town", "council", "data"}`. `council` is the folder of the council's EPC extracts (`data/<council>`; none, no EPC). `town` names the place signs (else the folder's name up to its first `_`). `data` is the data level:
- `full` (the default): every source below that is there;
- `osm_lidar`: the OSM download, rail land, road nodes, the DSM/DTM and the point cloud only. OS Open Roads, UPRN, NHLE, EPC, FSA and INSPIRE are not read even where their files are. Each step falls back: OSM's road classes, OSM's house numbers for homes, no listed builds, and homes with no era styled by built form (`unknown_*` looks, one style a development).

All of it in one command (PLAN U stage 2): `src/area/setup_area.py <map name> S W N E [--area <folder>] [--data full|osm_lidar] [--rotate DEG]`. It finds the council (postcodes.io), downloads OSM, runs the steps below in order, renders the source map with the streets' widths (step 2 below), downloads the point cloud tiles from the EA catalogue, and writes `area.json` with what it found. A step whose output is there is skipped (`--force` redoes all); a failed extra source is noted and the rest go on, and a rerun tries it again.

| Step | Script | Output |
|---|---|---|
| Listed buildings for a box | `src/area/fetch_nhle.py S W N E data/<area>` | `nhle_points/polygons/monuments.geojson` |
| Property points for a box | `src/area/filter_uprn.py data/raw/osopenuprn_*.zip S W N E data/<area>/uprn.csv` | `uprn.csv` |
| Per-building evidence | `src/area/features.py <map> data/<area> S W N E` | `buildings.csv`, `buildings.geojson` |
| EPC for one council | `src/area/extract_epc.py <domestic.zip> <non-domestic.zip> E07000089 data/hart` | `epc_domestic.csv`, `epc_nondomestic.csv` |
| EPC join, era, types | `src/area/assign_types.py data/<area> [data/<council>] [plots.parquet]` (the EPC folder defaults to `area.json`'s council) | `types.csv`; era/type added to `buildings.geojson` |
| LiDAR heights and roofs (EA WCS) | `src/area/roofs.py data/<area> --fetch` | `roofs.csv`; DSM/DTM tiles cached in `lidar/` |
| Point cloud for a box | `src/area/nlp_crop.py <tile.laz> data/<area>/nlp/core.npz E0 N0 E1 N1` | `nlp/core.npz` |
| The railway's land for a box (stations, platforms, `landuse=railway`; for `buildings/rail_kit.py`) | `src/area/fetch_rail_land.py data/<area> S W N E` | `rail_land.geojson` |
| INSPIRE plots for a box (the whole town's into `inspire_town.geojson` for `plots2.py --inspire-attrs`: cut to a scratch folder, then copied) | `src/area/fetch_inspire.py <council .gml> data/<area> E0 N0 E1 N1` | `inspire.geojson` |
| Map of any property | `src/area/render_map.py data/<area> <property> out/<file>.png` | PNG |
| Crossings, signals, give-ways, post boxes, bus stops, benches... (OSM points KnoxMap's download leaves out) | `src/area/fetch_road_nodes.py data/<area> S W N E` | `road_nodes.geojson` |
| OS Open Roads for the area | `src/roadgraph/osroads.py <map> data/<area> [<oproad_gb.gpkg>]` (from `data/raw/oproad`, once) | `os_roads.geojson` |

### Into a KnoxMap map (in this order)

All of it in one command (PLAN U stage 3): `src/uk_pipeline.py <map>_graph [--source <source map>] [--area <area folder>] [--from/--to/--only STEP] [--force]`. Its settings are `output/<map>/uk.json` (roads - the road graph's or KnoxMap's own straightening; buildings - from the LiDAR or KnoxMap's own fitting; UK assets on or off; lanes, front anchoring, rows at any angle, rooms, dormers, INSPIRE as evidence, streets, world cell, compile and backend: its docstring). With UK assets off the map needs no KnoxMapUK_Tiles: vanilla stand-ins (palette `uk_stand_ins`), `streets/vanilla_assets.py` on the loose tiles, the ground by vanilla rules, and a check that no UK tile is left; it writes the switch files from them. A step whose code, data and options are as when it last finished is skipped; one that runs makes every step after it run.

| Step | Script | Output |
|---|---|---|
| 1. Real street widths, before Generate | `src/streets/street_widths.py <osm cache .json.gz> data/<area> [--streets "High Street"]` | `width=`, `sidewalk:both:width=`, `verge=no` in the OSM download (original kept as `.orig`) |
| 2. Generate the source map | `src/area/new_map.py <name> S W N E [--rotate DEG] [--widths data/<area>]` (step 1 included with `--widths`; or KnoxMap's app: street alignment on, main street up-right); then delete `*_buildings.orig.geojson` and `*_fences.orig.geojson` | terrain + `<map>_buildings.geojson` |
| 2b. Hand over to the road graph | `src/roadgraph/handover.py <source map> data/<area> <name>_graph [--roads graph\|cost-front] [--lanes]` (`--markings` redoes only the markings) | the `_graph` map: roads, verges, pavements, markings; the steps below run on it. `--roads` (`roadgraph/modes.py`): `graph`, the default - vanilla's sixteen directions everywhere; `cost-front` - built-up streets kept off 2:1 and 45 where buildings line them, turning only at junctions, frontage gaps and plot boundaries, kept off the buildings, and the map's buildings squared about their street-facing edge (it writes `front_anchor.on` for blocks.py; LOG 2026-10-25 #2-#4: on Alton the High Street 63% -> 70% at its building line, buildings on the street -17%). The others (`grid`, `cost-low`, `cost-low-occ1` ...) are the experiments it was chosen from; `roadgraph/mode_test.py` scores any of them on a square. `--lanes` (`roadgraph/lanes.py`, off by default; LOG 2026-10-25 #7-#8, 2026-10-26 #1): driveways out of the road network, painted as drive paving from the street to the first building; OSM-only service lanes (no OS Open Roads match) routed with the buildings as hard obstacles - stuck, through a gap the LiDAR shows, else a passage under their building (no deeper than it, 12 tiles at most: blocks.py keeps the building, Build leaves its ground floor out), else dropped and listed |
| 3. Blocks from the LiDAR, carried in whole | `src/buildings/blocks.py <map> data/<area> [--streets ...] [--rect N] [--rect-count N]` | `placed` outlines with `lidar_frame`; LiDAR-only buildings added (none on the railway); long non-home buildings laid straight and whole inside their block (`laid_straight`); on a road-graph map, blocks with squares on the painted street (pavement, verge, bays) moved up to 3 tiles back onto free ground, piece by piece (`off_street`); homes drawn one by one in a row near the grid joined into one building (`row_of`); buildings kept off roads and railways, the railway's own (station, signal box, lift towers: `rail_kit.py`) let stand on its land, footbridges left out; `lidar_only.geojson` (original kept as `.orig`) |
| 4. Landmark outlines from the LiDAR | `src/buildings/lidar_outlines.py <map> data/<area> [--ids ...]` | landmark outlines replaced |
| 5. Units in long outlines | `src/buildings/split_units.py <map> data/<area>` | `<id>u<k>` units; `units_report.csv` |
| 5b. Shops, pubs, offices (needs `src/area/fetch_fsa.py data/<area> LAT LON [miles]` once → `fsa.json`) | `src/buildings/shops.py <map> data/<area> [--epc <epc_nondomestic.csv>]` | `knox_uses` (and `knox_flats_over`) on each placed building, from OSM points, non-domestic EPCs and FSA ratings, carried with the building |
| 5b. Railway track laid out (Alton; a map with no railway gets an empty layout) | `src/streets/rails.py <map> [--rebuild-table]` | each OSM railway on the grid and the diagonals with vanilla's pieces (bends learnt from vanilla's lots into `data/vanilla_rail.json`), its bed, the railway's land between its lines, crossing signals, the railway's own lifted bridges brought to the ground: `<map>_rails.json`, which plots2.py takes as the railway |
| 6. Garden plots and boundaries | `src/ground/plots2.py <map> data/<area> --inspire-attrs data/<area>/inspire_town.geojson [--picture out.png] [--box x,y,w,h]` (the normal run since 2026-10-16; without the flag, plots2 alone. `--inspire-attrs`: HM Land Registry's titles as evidence - garages joined to their home, gardens trimmed to the title's reach, road access, drives; `ground/inspire.py`, PLAN K) | plots as shapes from the road graph's street (fronts on its lattice lines, house by house), footpaths from OSM as separators; boundaries in `<map>_fences.geojson` (original kept as `.orig`; each plot line with its exact `tiles`); `<map>_plots.npz` (plot, street, drive, path...), `<map>_street.wkb`. Checks: `checks/boundary_score.py`, `checks/fence_check.py` |
| 6b. Trees (needs `src/ground/uk_tree_rules.py` run once on the map tools) | `src/ground/trees.py <map> data/<area> [--picture out.png]` | real trees from the point cloud, as big as their crowns, British look-alike species, on boundaries where they stand, in `<map>_veg_base.bmp`; plots with ≥ 2 trees/shrubs turned from paving to garden in `<map>_ground_base.bmp` (originals kept as `.pretrees.bmp`) |
| 6c. Sports pitches, courts, playgrounds | `src/ground/sports.py <map> [--picture out.png] [--box x,y,w,h]` | OSM `leisure=pitch` / `playground` squared to the grid and built with vanilla kit (tennis courts and nets, football lines and goals, hoops, cricket nets, playground kit) as loose tiles in `<map>_structures.json` (original kept as `.orig.json`); chain-link and railings in `<map>_fences.geojson` (`source=sports`); trees cleared off them; a playground's ground sand inside its railings only |
| 6c2. Railway track into the map | `src/streets/rails.py <map> --apply` | the track and signals as loose tiles in `<map>_structures.json` ahead of kerbs.py's (`rails_added`), the railway's bridges' deck, ramps and railings out, the bed painted dirt in `<map>_ground_base.bmp` and cleared in `<map>_veg_base.bmp` (both kept as `.prerails.bmp`) |
| 6d. Kerbs that follow the road (needs `src/streets/measure_kerbs.py` once: `data/kerbs/`) | `src/streets/kerbs.py <map> [--sigma 1.5] [--picture out.png] [--box x,y,w,h]`; check with `src/checks/road_width_check.py <map> [--before]` | 45° roads redrawn one width all along with a 2-tile pavement and no kerbs (`diagonals.py`); road corners, 45° dog-legs and cul-de-sac ends rounded within the road and pavement; B42 kerbs in 16 directions and road overlays laid by name into `<map>_structures.json` (`kerbs_added`) instead of the 8 rule kerbs (cleared from `<map>_veg_base.bmp`); `<map>_ground_base.bmp` takes the new road edge (both kept as `.prekerbs.bmp`). On a road-graph map nothing is rounded: each square an angled (45° or 2:1) edge of tarmac crosses - the carriageway and its bays - is laid as vanilla lays its own: tarmac, vanilla's kerb, overlay and stub pieces for that cut (`ANGLED_TABLE`, 20 rows), the ground beside painted over the part beyond (`SIDE_SHEETS`: pale concrete beside the grey slabs, a stand-in, PLAN E); the back of an angled pavement the same, unkerbed, with the garden's ground |
| 7. Styles + storeys for Build | `src/regional/export_regional.py data/<area> palettes/english.json <map>` | `regional.json`, read by `knoxbuild/regional_uk.py`: style, storeys and each house's parts (`house_parts`: upper-floor wall, porch, bays, plinth, downpipes) |
| 7b. Listed buildings and landmarks | `../../asset_creator/listed_building_maker/apply.py <map> data/<area>` (after roof_plan, before Build) | their own styles, roofs and landmark builds in `regional.json`; **run it after every export_regional**, or they lose their builds |
| 8. Roofs as ranges | `src/regional/roof_plan.py <map> data/<area> [--dormers]` | `roof` lists in `regional.json`; a house's front gable where the LiDAR shows one (`front_gables`: an end bay of the main range sloping across it both ways, a cross-wing of its own; a hipped end, sloping one way, is left) |
| 9. Build, validate, compile | `python -m knoxbuild <map>`, `tools/validate_tbx.py`, `tools/compile_map.py`, `tools/make_pictures.py` (in `Knoxmap/`) | the map; pictures go in `out/<topic>/` |

`tidy_outlines.py` was the old step 3; `blocks.py` replaces it.

### The palette (`palettes/english.json`)

- `palette.drop`: KnoxMap styles not to use (the American sidings, logs...).
- `palette.add`: new styles, each made `like` an existing one, with:
  - `wall`: the West tile of a full wall set;
  - `shaped`: its window-shaped pieces;
  - `window`;
  - `roof_from`: whose roof tiles to borrow;
  - `caps` / `caps_shallow` / `caps_30`: the first tile of the gable-cap sets that match the wall;
  - `chimney`: the stack tile, `fixtures_fireplaces_01` 64–70.
- `type_styles`, `listed_styles`, `high_street_unlisted`, `name_styles`, `unit_styles`: which styles each kind of building may take (`export_regional.py`).
- `subtype_styles`: the styles for each housing look (below), before `type_styles`. A unit split from a terrace keeps a housing-kit style (`by_id_only`); only others draw on `unit_styles`.
- A style's house parts (`asset_creator/housing_kit`, 2026-10-19): `upper_wall` (a wall kit from the first floor up; `upper_sides: "long"` only on the walls along the ridge, brick piers at the ends), `porch`, `bay` (where the house stands 2 m or more back from the road), `plinth`, `garage` (2026-10-19, to-do K09.24: `{"kit": <wall kit>, "integral": bool}`; Build makes the house's one-storey flat or lean-to wing beside it on the street side its garage, or, with `integral` on a detached house, builds one into a front corner - an up-and-over door (`housing_kit/garages.py`) and a drive (`knoxbuild/yards.py`)). `palette.shutters: false` takes the outside shutters off every style.

`knoxbuild/regional_uk.py` turns these into KnoxMap house styles at Build (its docstring has the details).

### Checks (after Build)

| Check | Script |
|---|---|
| Roof tiles over nothing | `src/checks/roof_float_check.py <map>` |
| Rooms open to the sky, open lean-tos (a taller building next door counts as backing) | `src/checks/roof_cover_check.py <map>` |
| Where flat roof comes from | `src/checks/flat_audit.py <map>` |
| Off-grid angles, roof pieces, flat share | `src/checks/roof_stats.py <map>` |
| Street frontages against the real ones | `src/checks/frontage_measure.py <map> data/<area>` |
| Building tiles on roads | `src/checks/road_audit.py <map>` |
| Road links lost in straightening | `src/checks/road_links.py <map>` |
| Buildings on the road graph's street, by layer and cause (width or position) | `src/checks/road_overlap.py <map> data/<area>` |
| Garden boundaries: messy fronts, by area | `src/checks/boundary_score.py <map> [--pictures]` |
| Fences: walled path mouths, edges across tarmac, dashed angled fronts | `src/checks/fence_check.py <map>` |
| V shapes in the plot boundaries (sides and backs; fronts apart) | `src/checks/v_check.py <map> [--list N]` |
| One building's roof against its LiDAR | `src/checks/lidar_check.py <map> data/<area> <osm_id> <out.png>` |
| Blocks, game against real (pictures) | `src/checks/blocks_check.py <map> data/<area> <out prefix>` |
| Tiles the game does not define (after compile) | `src/checks/tile_defs.py <map>` |
| Multi-tile objects with a piece missing or out of place (after compile; the game hung on them; exit 1 if any) | `src/checks/multitile_check.py <map> [--rebuild-table]` (vanilla's own placements as the reference: `data/vanilla_multitile.json`) |

## What `features.py` works out

- **Shape:** area, rectangularity, long/short sides, aspect, orientation, vertex count.
- **Built form:** detached, semi, end or mid terrace from shared walls. Then, for single outlines holding several homes, from the property points inside: `semi_pair` (2), `terrace_row` (3–12, house-sized footprint per home), `multi_unit` (flats, or shops with flats over).
- **Street:** nearest named street, setback, and whether the building faces the street or stands end-on.
- **Listing:** grade, list entry number and name (Historic England polygons, then points).
- **Homes inside:** UPRN count and ids (for the EPC join).

## Results so far: `odiham_core`

885 buildings, 107 of them listed, 2,782 property points in the area.

**Built form** checked against the forms OSM mappers tagged:

| OSM tag | Agree | Notes |
|---|---|---|
| `semidetached_house` | 46/50 (92%) | 23/50 before counting homes per outline |
| `detached` | 6/7 | |
| `terrace` | 10/14 | the rest are courtyards/flats, for EPC to settle |
| `bungalow` | 10/17 detached | 7 came out as semi pairs; semi-detached bungalows are common, so probably right |

**Known gaps**
- Non-residential buildings (e.g. the school) get a form from shared walls; ignore form for them.
- 157 non-outbuildings have no property point inside (median 61 m²): garages and sheds over 30 m², or homes whose point falls just outside the outline.

## Era and type (`assign_types.py`)

How a building gets its era: listing (→ pre-1900), then its own EPC, then
its development (alike neighbours on the same street), then the nearest
buildings with a known era, then its street.

On `odiham_core`:
- **58%** of homes have an EPC.
- **Hold-out era prediction: 60% exact, 83% within one era.** The ceiling is about 75% / 95%: homes built together in one outline only agree that often in EPC data.

**Sub-types** (`subtypes.py`, 2026-10-19; `subtype` and `subtype_why` beside `type` in `types.csv` and `buildings.geojson`): about 35, each a tile set of its own (to-do K09.22). A lookup, in order of trust: the listing's date and features (asset_creator's listed_building_maker lexicon); the EPC majority (finer age band, built form, property type, walls, social tenancies, floor height); the development's share of social tenancies (council housing, Right to Buy notwithstanding) and median year; the LiDAR's storeys and roof shape (`roofs.csv`); the footprint. The 1900-1929 band is split by built form and footprint (Edwardian terrace / semi / villa or early inter-war semi / detached); council housing has its own sub-type in each era (`iw_council`, `pw_council`, `e70_council`, `e80_social`). The older `type` is unchanged. Each building's `look` (`subtypes.look_of`) is its sub-type as it stood in 1993, the game's year: anything built since looks 1983-1995 (`e80_house`, `e80_terrace`, `e80_social`); `export_regional` styles houses by look through the palette's `subtype_styles`.

## Data needed from you (done)

**EPC (domestic and non-domestic), Hart District** — free, needs an account:
1. Register at https://epc.opendatacommunities.org/ (email sign-in link).
2. Download **Domestic** certificates by local authority: **Hart** (code E07000089).
3. Download **Non-domestic** certificates for **Hart** too.
4. Put both zips in `Building_types/data/raw/`.
